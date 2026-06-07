#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Mean Reversion Strategy
Bot name: alpaca_hybrid_bot (overridable via BOT_NAME env var)
Works with the Trading Command Center dashboard.
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

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------- DATABASE HELPERS ----------
def get_db_connection():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        return None
    return psycopg2.connect(db_url)

def ensure_tables():
    """Create all necessary tables if they don't exist."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Bot orders table
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
            # Bot errors table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_errors (
                    id SERIAL PRIMARY KEY,
                    bot_name TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            # Trades table (matches dashboard expectations)
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
                    order_id TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            # Bot status table (used for kill switch)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    bot_name TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'RUNNING',
                    last_update TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            logger.info("Database tables verified/created.")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
    finally:
        conn.close()

def log_error_to_db(bot_name, error_msg):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)",
                (bot_name, str(error_msg))
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Could not log error to DB: {e}")
    finally:
        conn.close()

def write_trade_to_db(bot_name, symbol, side, price, qty, order_id):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (bot_name, "Alpaca", symbol, side, price, qty, price * qty, order_id))
            conn.commit()
            logger.info(f"Trade logged: {side} {qty} {symbol} @ ${price:.2f}")
    except Exception as e:
        logger.error(f"Failed to write trade: {e}")
    finally:
        conn.close()

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
    """Stop the bot if database status == 'STOP'."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Update last_seen and ensure status is RUNNING
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
                logger.info(f"Kill switch activated for {bot_name}. Shutting down.")
                sys.exit(0)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
    finally:
        conn.close()

# ---------- BOT CLASS ----------
class MeanReversionBot:
    def __init__(self):
        # Bot name: default 'alpaca_hybrid_bot' – can be overridden via BOT_NAME env var
        self.bot_name = os.getenv('BOT_NAME', 'alpaca_hybrid_bot')
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        if not api_key or not api_secret:
            raise ValueError("Missing Alpaca API credentials (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
        self.trading = TradingClient(api_key, api_secret, paper=True)
        logger.info(f"Bot '{self.bot_name}' initialized (Paper trading)")
        check_status(self.bot_name)

    def place_order_tracked(self, symbol, side, qty):
        """Place a market order and track it in the database."""
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
            logger.error(f"Failed to place order: {e}")
            log_error_to_db(self.bot_name, f"Order placement failed: {e}")
            return None

    async def sync_orders(self):
        """Check for filled orders and update DB + log trades."""
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
                            cur.execute(
                                "UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s",
                                (order_id,)
                            )
                            write_trade_to_db(
                                self.bot_name,
                                symbol,
                                alpaca_order.side.value,
                                float(alpaca_order.filled_avg_price),
                                float(alpaca_order.filled_qty),
                                order_id
                            )
                            logger.info(f"Order {order_id} filled and logged")
                    except Exception as e:
                        logger.error(f"Error processing order {order_id}: {e}")
                conn.commit()
        except Exception as e:
            logger.error(f"Sync orders failed: {e}")
            log_error_to_db(self.bot_name, f"Sync error: {e}")
        finally:
            conn.close()

    # ========== YOUR MEAN REVERSION LOGIC GOES HERE ==========
    async def check_for_signals(self):
        """
        Implement your mean reversion strategy.
        This is called every main loop iteration.
        Return (symbol, side, qty) or None.
        """
        # Example placeholder logic (replace with real indicators)
        # For now, just return None to avoid trading.
        # You can fetch current price, compute z-score, etc.
        return None

    async def run(self):
        logger.info(f"{self.bot_name} started. Waiting 10 seconds to stabilize...")
        await asyncio.sleep(10)

        while True:
            try:
                # 1. Check kill switch
                check_status(self.bot_name)

                # 2. Sync any filled orders
                await self.sync_orders()

                # 3. Run your mean reversion signal
                signal = await self.check_for_signals()
                if signal:
                    symbol, side, qty = signal
                    self.place_order_tracked(symbol, side, qty)

                # 4. Wait before next cycle (adjust as needed)
                await asyncio.sleep(60)  # 1 minute – change to 300 for 5 min

            except Exception as e:
                error_msg = f"Main loop error: {str(e)}"
                logger.error(error_msg)
                log_error_to_db(self.bot_name, error_msg)
                await asyncio.sleep(30)

# ---------- MAIN ENTRY POINT ----------
if __name__ == "__main__":
    try:
        ensure_tables()
        bot = MeanReversionBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}")
        try:
            log_error_to_db("startup", f"Bot crashed before starting: {e}")
        except:
            pass
        sys.exit(1)
