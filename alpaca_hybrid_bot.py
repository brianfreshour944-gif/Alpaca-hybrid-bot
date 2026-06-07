#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Robust Mean Reversion
Bot name: alpaca_hybrid_bot (or set via BOT_NAME env var)
Logs to terminal AND database, never crashes silently.
"""

import asyncio
import os
import logging
import psycopg2
import sys
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv()

# ---------- LOGGING (terminal first) ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------- DATABASE HELPERS (fail‑safe) ----------
def get_db_connection():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.warning("DATABASE_URL not set – database features disabled")
        return None
    try:
        return psycopg2.connect(db_url, connect_timeout=5)
    except Exception as e:
        logger.error(f"Cannot connect to database: {e}")
        return None

def ensure_db_tables():
    """Create missing tables/columns so the bot never fails on first run."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # trades table with fee column
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    bot_name TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    quantity REAL,
                    value REAL,
                    fee REAL DEFAULT 0,
                    order_id TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            # bot_orders table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_orders (
                    order_id TEXT PRIMARY KEY,
                    bot_name TEXT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # bot_errors table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_errors (
                    id SERIAL PRIMARY KEY,
                    bot_name TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            # bot_status table (kill switch)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    bot_name TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'RUNNING',
                    last_update TIMESTAMP DEFAULT NOW(),
                    daily_loss REAL DEFAULT 0,
                    daily_loss_limit REAL DEFAULT 100,
                    config TEXT DEFAULT '{}'
                )
            """)
            # Add fee column if missing (for existing tables)
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='trades' AND column_name='fee') THEN
                        ALTER TABLE trades ADD COLUMN fee REAL DEFAULT 0;
                    END IF;
                END $$;
            """)
            conn.commit()
            logger.info("Database tables verified/created.")
    except Exception as e:
        logger.error(f"DB table check failed: {e}")
    finally:
        conn.close()

def log_error_to_db(bot_name, error_msg):
    logger.error(f"[{bot_name}] {error_msg}")  # always log to terminal
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)", (bot_name, str(error_msg)))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log error to DB: {e}")
    finally:
        conn.close()

def write_trade_to_db(bot_name, symbol, side, price, qty, order_id, fee=0.0):
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, fee, order_id, timestamp)
                VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (bot_name, symbol, side, float(price), float(qty), float(price * qty), float(fee), str(order_id)))
            conn.commit()
            logger.info(f"Trade logged: {side} {qty} {symbol} @ ${price:.4f} (fee: ${fee:.4f})")
        conn.close()
    except Exception as e:
        logger.error(f"Failed to write trade: {e}")

def register_order_in_db(bot_name, order_id, symbol, side, price):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_orders (order_id, bot_name, symbol, side, price, status)
                VALUES (%s, %s, %s, %s, %s, 'OPEN')
                ON CONFLICT (order_id) DO NOTHING
            """, (str(order_id), bot_name, symbol, side, float(price)))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to register order: {e}")
    finally:
        conn.close()

def check_status(bot_name):
    """Check kill switch. Stop if status = 'STOP'."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Update heartbeat
            cur.execute("""
                INSERT INTO bot_status (bot_name, status, last_update)
                VALUES (%s, 'RUNNING', NOW())
                ON CONFLICT (bot_name) DO UPDATE
                SET last_update = NOW(), status = EXCLUDED.status
            """, (bot_name,))
            conn.commit()
            # Check kill switch
            cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
            row = cur.fetchone()
            if row and row[0] == 'STOP':
                logger.info(f"Kill switch activated for {bot_name}. Exiting.")
                sys.exit(0)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
    finally:
        conn.close()

# ---------- BOT CLASS ----------
class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'alpaca_hybrid_bot')
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        if not api_key or not api_secret:
            logger.critical("Missing Alpaca API credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            sys.exit(1)
        self.trading = TradingClient(api_key, api_secret, paper=True)
        logger.info(f"Bot '{self.bot_name}' initialized (paper trading)")
        
        # Initial health check
        check_status(self.bot_name)

    def place_order_tracked(self, symbol, side, qty):
        """Place a market order and track it in DB."""
        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            )
            register_order_in_db(self.bot_name, order.id, symbol, side.value, 0.0)
            logger.info(f"Placed {side.value} order for {qty} {symbol} (ID: {order.id})")
            return order
        except Exception as e:
            log_error_to_db(self.bot_name, f"Order placement failed: {e}")
            return None

    async def sync_orders(self):
        """Check for filled orders and log trades."""
        conn = get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'",
                    (self.bot_name,)
                )
                open_orders = cur.fetchall()
                for order_id, symbol in open_orders:
                    try:
                        alpaca_order = self.trading.get_order_by_id(order_id)
                        if alpaca_order.status == 'filled':
                            cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (order_id,))
                            # Alpaca does not return fee in the order object easily; set to 0 for now.
                            write_trade_to_db(
                                self.bot_name, symbol,
                                alpaca_order.side.value,
                                float(alpaca_order.filled_avg_price),
                                float(alpaca_order.filled_qty),
                                order_id,
                                fee=0.0
                            )
                            logger.info(f"Order {order_id} filled and logged")
                    except Exception as e:
                        logger.error(f"Error processing order {order_id}: {e}")
                conn.commit()
        except Exception as e:
            log_error_to_db(self.bot_name, f"Sync error: {e}")
        finally:
            conn.close()

    # ----- YOUR MEAN REVERSION LOGIC GOES HERE -----
    async def check_for_signals(self):
        """
        Replace with your actual strategy.
        Return (symbol, side, qty) or None.
        For now, it does nothing (no orders placed).
        """
        return None

    async def run(self):
        logger.info(f"{self.bot_name} started. Waiting 10 seconds...")
        await asyncio.sleep(10)

        while True:
            try:
                # 1. Kill switch check
                check_status(self.bot_name)

                # 2. Sync filled orders
                await self.sync_orders()

                # 3. Strategy signal (placeholder)
                signal = await self.check_for_signals()
                if signal:
                    symbol, side, qty = signal
                    self.place_order_tracked(symbol, side, qty)

                # 4. Loop interval
                await asyncio.sleep(60)  # adjust as needed

            except Exception as e:
                log_error_to_db(self.bot_name, f"Main loop error: {e}")
                await asyncio.sleep(30)

# ---------- MAIN ----------
if __name__ == "__main__":
    # Ensure DB tables exist before starting
    ensure_db_tables()
    bot = MeanReversionBot()
    asyncio.run(bot.run())
