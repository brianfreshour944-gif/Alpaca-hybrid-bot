#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Mean Reversion with Bollinger Bands
Bot name: alpaca_hybrid_bot (or set via BOT_NAME env var)
Logs to terminal AND database, never crashes silently.
"""

import asyncio
import os
import logging
import psycopg2
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_errors (
                    id SERIAL PRIMARY KEY,
                    bot_name TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
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
            cur.execute("""
                INSERT INTO bot_status (bot_name, status, last_update)
                VALUES (%s, 'RUNNING', NOW())
                ON CONFLICT (bot_name) DO UPDATE
                SET last_update = NOW(), status = EXCLUDED.status
            """, (bot_name,))
            conn.commit()
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
        self.symbol = "BTC/USD"               # Change to any crypto pair
        self.trade_size_usd = 50.0            # $50 per trade
        self.in_position = False
        self.entry_price = 0.0
        self.stop_loss_pct = 0.95             # 5% stop loss
        self.cooldown_until = 0.0

        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        if not api_key or not api_secret:
            logger.critical("Missing Alpaca API credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            sys.exit(1)
        self.trading = TradingClient(api_key, api_secret, paper=True)
        self.data_client = CryptoHistoricalDataClient()
        logger.info(f"Bot '{self.bot_name}' initialized (paper trading) – trading {self.symbol}")

        # Initial health check
        check_status(self.bot_name)

    def place_order_tracked(self, symbol, side, qty):
        """Place a market order and track it in DB."""
        try:
            # FIX: Use GTC for crypto orders (DAY is not allowed)
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC   # <--- CRITICAL FIX
                )
            )
            register_order_in_db(self.bot_name, order.id, symbol, side.value, 0.0)
            logger.info(f"Placed {side.value} order for {qty:.8f} {symbol} (ID: {order.id})")
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
                            # Update position flag if it was a sell order
                            if alpaca_order.side == OrderSide.SELL:
                                self.in_position = False
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

    async def get_bollinger_bands(self):
        """
        Fetch recent 5-minute candles, compute SMA20 and ±2 standard deviation bands.
        Returns (current_price, lower_band, middle_band, upper_band) or (None, None, None, None)
        """
        end = datetime.now()
        start = end - timedelta(hours=6)   # enough for ~72 5-min bars
        request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=500
        )
        bars = self.data_client.get_crypto_bars(request).data.get(self.symbol, [])
        if len(bars) < 30:
            logger.warning(f"Insufficient minute bars: {len(bars)}")
            return None, None, None, None

        # Convert to DataFrame, resample to 5 minutes
        df = pd.DataFrame([{
            'timestamp': b.timestamp,
            'close': float(b.close)
        } for b in bars])
        df.sort_values('timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        ohlc_5 = df.resample('5min').agg({'close': 'last'}).dropna()
        closes = ohlc_5['close'].values
        if len(closes) < 20:
            logger.warning(f"Not enough 5-min bars: {len(closes)}")
            return None, None, None, None

        # Calculate Bollinger Bands (20 period, 2 std)
        sma = pd.Series(closes).rolling(20).mean().iloc[-1]
        std = pd.Series(closes).rolling(20).std().iloc[-1]
        lower = sma - 2 * std
        upper = sma + 2 * std
        current_price = closes[-1]
        return current_price, lower, sma, upper

    async def check_for_signals(self):
        """
        Mean reversion logic using Bollinger Bands.
        Returns (symbol, side, qty) or None.
        """
        # Cooldown after a sell
        if time.time() < self.cooldown_until:
            logger.debug("Cooldown active")
            return None

        current_price, lower, middle, upper = await self.get_bollinger_bands()
        if current_price is None:
            return None

        logger.info(f"Price: {current_price:.2f} | Lower: {lower:.2f} | Middle: {middle:.2f} | Upper: {upper:.2f} | In position: {self.in_position}")

        if not self.in_position:
            # Buy signal: price below lower band (oversold)
            # --- IMPROVED CODE ---
if current_price < lower:
    # Ensure the order value is at least $10.01 to satisfy Alpaca's minimum
    min_order_value = 10.01
    actual_order_value = max(self.trade_size_usd, min_order_value)
    
    qty = actual_order_value / current_price
    logger.info(f"*** BUY SIGNAL – price {current_price:.2f} below lower band {lower:.2f} ***")
    logger.info(f"Submitting buy order for ${actual_order_value:.2f} worth of {self.symbol}")
    return (self.symbol, OrderSide.BUY, qty)
        else:
            # Exit conditions: price returned above middle band OR stop loss hit
            if current_price > middle:
                logger.info(f"*** SELL SIGNAL – price reverted above middle band {middle:.2f} ***")
                return (self.symbol, OrderSide.SELL, None)   # qty will be fetched from position
            elif current_price <= self.entry_price * self.stop_loss_pct:
                logger.info(f"*** STOP LOSS TRIGGERED at {current_price:.2f} (entry {self.entry_price:.2f}) ***")
                return (self.symbol, OrderSide.SELL, None)
        return None

    async def run(self):
        logger.info(f"{self.bot_name} started. Waiting 10 seconds...")
        await asyncio.sleep(10)

        while True:
            try:
                # 1. Kill switch check
                check_status(self.bot_name)

                # 2. Sync filled orders (update in_position flag)
                await self.sync_orders()

                # 3. Get trading signal
                signal = await self.check_for_signals()
                if signal:
                    symbol, side, qty = signal
                    if side == OrderSide.SELL:
                        # Need to fetch current position size
                        try:
                            # Note: Alpaca requires symbol without slash for get_position
                            pos_symbol = symbol.replace("/", "")
                            position = self.trading.get_position(pos_symbol)
                            qty = float(position.qty)
                        except Exception:
                            logger.warning(f"No position to sell for {symbol}")
                            self.in_position = False
                            qty = 0.0
                        if qty > 0:
                            order = self.place_order_tracked(symbol, side, qty)
                            if order:
                                self.in_position = False
                                self.cooldown_until = time.time() + 900   # 15 min cooldown
                    else:  # BUY
                        order = self.place_order_tracked(symbol, side, qty)
                        if order:
                            self.in_position = True
                            # Wait a moment for the order to fill, then get the actual fill price
                            await asyncio.sleep(2)
                            try:
                                # Fetch the order again to get filled price
                                filled_order = self.trading.get_order_by_id(order.id)
                                if filled_order.filled_avg_price:
                                    self.entry_price = float(filled_order.filled_avg_price)
                                else:
                                    # Fallback to current price
                                    self.entry_price = (await self.get_bollinger_bands())[0]
                            except:
                                self.entry_price = (await self.get_bollinger_bands())[0]
                            logger.info(f"Position opened, entry price ${self.entry_price:.2f}")

                # 4. Loop interval
                await asyncio.sleep(60)  # check every minute

            except Exception as e:
                log_error_to_db(self.bot_name, f"Main loop error: {e}")
                await asyncio.sleep(30)

# ---------- MAIN ----------
if __name__ == "__main__":
    import time  # for cooldown
    ensure_db_tables()
    bot = MeanReversionBot()
    asyncio.run(bot.run())
