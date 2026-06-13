
#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Mean Reversion with Bollinger Bands
Bot name: alpaca_hybrid_bot
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
        return None
    try:
        return psycopg2.connect(db_url, connect_timeout=5)
    except Exception as e:
        logger.error(f"Cannot connect to database: {e}")
        return None

def ensure_db_tables():
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY, bot_name TEXT, exchange TEXT,
                    symbol TEXT, side TEXT, price REAL, quantity REAL,
                    value REAL, fee REAL DEFAULT 0, order_id TEXT,
                    timestamp TIMESTAMP DEFAULT NOW())""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_orders (
                    order_id TEXT PRIMARY KEY, bot_name TEXT, symbol TEXT,
                    side TEXT, price REAL, status TEXT,
                    created_at TIMESTAMP DEFAULT NOW())""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_errors (
                    id SERIAL PRIMARY KEY, bot_name TEXT,
                    error_message TEXT, timestamp TIMESTAMP DEFAULT NOW())""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    bot_name TEXT PRIMARY KEY, status TEXT DEFAULT 'RUNNING',
                    last_update TIMESTAMP DEFAULT NOW(),
                    daily_loss REAL DEFAULT 0, daily_loss_limit REAL DEFAULT 100,
                    config TEXT DEFAULT '{}')""")
            # Safely add new columns to existing bot_status table
            cur.execute("""
                ALTER TABLE bot_status
                    ADD COLUMN IF NOT EXISTS in_position BOOLEAN DEFAULT FALSE""")
            cur.execute("""
                ALTER TABLE bot_status
                    ADD COLUMN IF NOT EXISTS entry_price REAL DEFAULT 0""")
            conn.commit()
    finally:
        conn.close()

def write_trade_to_db(bot_name, symbol, side, price, qty, order_id, fee=0.0):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                    (bot_name, exchange, symbol, side, price, quantity,
                     value, fee, order_id, timestamp)
                VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, NOW())""",
                (bot_name, symbol, side, float(price), float(qty),
                 float(price * qty), float(fee), str(order_id)))
            conn.commit()
    finally:
        conn.close()

def register_order_in_db(bot_name, order_id, symbol, side, price):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_orders
                    (order_id, bot_name, symbol, side, price, status)
                VALUES (%s, %s, %s, %s, %s, 'OPEN')
                ON CONFLICT (order_id) DO NOTHING""",
                (str(order_id), bot_name, symbol, side, float(price)))
            conn.commit()
    finally:
        conn.close()

def save_position_state(bot_name, in_position, entry_price):
    """Persist position state to DB so it survives restarts."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_status (bot_name, in_position, entry_price, last_update)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (bot_name) DO UPDATE
                    SET in_position = EXCLUDED.in_position,
                        entry_price = EXCLUDED.entry_price,
                        last_update = NOW()""",
                (bot_name, in_position, float(entry_price)))
            conn.commit()
    finally:
        conn.close()

def load_position_state(bot_name):
    """Restore position state from DB on restart.
    Also ensures the required columns exist — safe to call before ensure_db_tables."""
    conn = get_db_connection()
    if not conn: return False, 0.0
    try:
        with conn.cursor() as cur:
            # Guarantee columns exist regardless of call order
            cur.execute("""
                ALTER TABLE bot_status
                    ADD COLUMN IF NOT EXISTS in_position BOOLEAN DEFAULT FALSE""")
            cur.execute("""
                ALTER TABLE bot_status
                    ADD COLUMN IF NOT EXISTS entry_price REAL DEFAULT 0""")
            conn.commit()
            cur.execute(
                "SELECT in_position, entry_price FROM bot_status WHERE bot_name = %s",
                (bot_name,))
            row = cur.fetchone()
            if row:
                return bool(row[0]), float(row[1] or 0.0)
            return False, 0.0
    except Exception as e:
        logger.error(f"load_position_state error: {e}")
        return False, 0.0
    finally:
        conn.close()

def check_status(bot_name):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_status (bot_name, status, last_update)
                VALUES (%s, 'RUNNING', NOW())
                ON CONFLICT (bot_name) DO UPDATE
                    SET last_update = NOW(),
                        status = CASE WHEN bot_status.status = 'STOP'
                                      THEN 'STOP' ELSE 'RUNNING' END""",
                (bot_name,))
            conn.commit()
            cur.execute(
                "SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
            row = cur.fetchone()
            if row and row[0] == 'STOP':
                logger.info("🛑 STOP signal received. Exiting.")
                sys.exit(0)
    finally:
        conn.close()


# ---------- BOT CLASS ----------
class MeanReversionBot:
    def __init__(self):
        self.bot_name       = os.getenv('BOT_NAME', 'alpaca_hybrid_bot')
        self.symbol         = "BTC/USD"
        self.trade_size_usd = 50.0
        self.stop_loss_pct  = 0.95   # exit if price drops 5% below entry
        self.cooldown_until = 0.0

        api_key    = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.trading     = TradingClient(api_key, api_secret, paper=True)
        self.data_client = CryptoHistoricalDataClient()

        # Restore position state from DB on startup
        self.in_position, self.entry_price = load_position_state(self.bot_name)
        if self.in_position:
            logger.info(
                f"♻️  Restored position state from DB — "
                f"in_position=True, entry_price={self.entry_price:.2f}"
            )
            # Verify the position actually exists on Alpaca
            actual_qty = self._get_position_qty()
            if actual_qty == 0.0:
                logger.warning(
                    "DB says in_position=True but no BTC found on Alpaca. "
                    "Resetting position state."
                )
                self.in_position = False
                self.entry_price = 0.0
                save_position_state(self.bot_name, False, 0.0)
        else:
            logger.info("No saved position — starting fresh.")

    def _get_position_qty(self) -> float:
        """
        Robustly fetch BTC position qty from Alpaca.
        Tries multiple symbol formats, falls back to scanning all positions.
        Returns 0.0 if no position found.
        """
        for sym in ["BTCUSD", "BTC/USD", "BTC"]:
            try:
                position = self.trading.get_position(sym)
                qty = float(position.qty)
                if qty > 0:
                    logger.debug(f"Position found under symbol '{sym}': {qty} BTC")
                    return qty
            except Exception:
                continue

        # Last resort: scan all positions
        try:
            positions = self.trading.get_all_positions()
            for p in positions:
                if 'BTC' in p.symbol.upper():
                    qty = float(p.qty)
                    logger.debug(f"Position found via scan: {p.symbol} qty={qty}")
                    return qty
        except Exception as e:
            logger.error(f"get_all_positions failed: {e}")

        return 0.0

    def place_order_tracked(self, symbol, side, qty):
        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol, qty=qty,
                    side=side, time_in_force=TimeInForce.GTC
                )
            )
            register_order_in_db(
                self.bot_name, order.id, symbol, side.value, 0.0)
            return order
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    async def sync_orders(self):
        conn = get_db_connection()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT order_id, symbol FROM bot_orders "
                    "WHERE bot_name = %s AND status = 'OPEN'",
                    (self.bot_name,))
                rows = cur.fetchall()
                for order_id, symbol in rows:
                    try:
                        alpaca_order = self.trading.get_order_by_id(order_id)
                        if alpaca_order.status.value == 'filled':
                            cur.execute(
                                "UPDATE bot_orders SET status = 'CLOSED' "
                                "WHERE order_id = %s", (order_id,))
                            side_val  = alpaca_order.side.value
                            avg_price = float(alpaca_order.filled_avg_price or 0)
                            filled_qty = float(alpaca_order.filled_qty or 0)
                            write_trade_to_db(
                                self.bot_name, symbol, side_val,
                                avg_price, filled_qty, order_id)
                            if alpaca_order.side == OrderSide.SELL:
                                self.in_position = False
                                self.entry_price = 0.0
                                save_position_state(self.bot_name, False, 0.0)
                                logger.info(
                                    f"✅ Sell order {order_id} confirmed filled "
                                    f"@ {avg_price:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not sync order {order_id}: {e}")
                conn.commit()
        finally:
            conn.close()

    async def get_bollinger_bands(self):
        end   = datetime.now()
        start = end - timedelta(hours=6)
        request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start, end=end, limit=500)
        bars = self.data_client.get_crypto_bars(request).data.get(
            self.symbol, [])
        if len(bars) < 30:
            return None, None, None, None

        df = pd.DataFrame([
            {'timestamp': b.timestamp, 'close': float(b.close)}
            for b in bars
        ])
        df.set_index('timestamp', inplace=True)
        ohlc_5 = df.resample('5min').agg({'close': 'last'}).dropna()
        closes = ohlc_5['close'].values

        sma = pd.Series(closes).rolling(20).mean().iloc[-1]
        std = pd.Series(closes).rolling(20).std().iloc[-1]
        return closes[-1], sma - 2 * std, sma, sma + 2 * std

    async def check_for_signals(self):
        if time.time() < self.cooldown_until:
            remaining = self.cooldown_until - time.time()
            logger.info(f"⏳ Cooldown active — {remaining:.0f}s remaining")
            return None

        price, lower, middle, upper = await self.get_bollinger_bands()
        if price is None:
            logger.warning("Not enough data for Bollinger Bands")
            return None

        logger.info(
            f"Price: {price:.2f} | Lower: {lower:.2f} | "
            f"Middle: {middle:.2f} | Upper: {upper:.2f} | "
            f"In pos: {self.in_position}"
        )

        if not self.in_position:
            if price < lower:
                qty = max(self.trade_size_usd, 10.01) / price
                logger.info(
                    f"📈 BUY signal — price {price:.2f} below lower band {lower:.2f}")
                return (self.symbol, OrderSide.BUY, qty)

        else:
            stop_hit    = price <= self.entry_price * self.stop_loss_pct
            target_hit  = price >= upper
            manual_exit = price >= middle and (price - middle) > (upper - middle) * 0.5

            if stop_hit:
                logger.info(
                    f"🛑 Stop loss hit — price {price:.2f} <= "
                    f"stop {self.entry_price * self.stop_loss_pct:.2f}")
                return (self.symbol, OrderSide.SELL, None)
            if target_hit:
                logger.info(
                    f"🎯 Target hit — price {price:.2f} >= upper band {upper:.2f}")
                return (self.symbol, OrderSide.SELL, None)
            if manual_exit:
                logger.info(
                    f"📉 Mid-upper exit — price {price:.2f} past midpoint of bands")
                return (self.symbol, OrderSide.SELL, None)

        return None

    async def run(self):
        logger.info(f"🤖 {self.bot_name} starting...")
        while True:
            try:
                check_status(self.bot_name)
                await self.sync_orders()
                signal = await self.check_for_signals()

                if signal:
                    symbol, side, qty = signal

                    if side == OrderSide.SELL:
                        qty_to_sell = self._get_position_qty()
                        if qty_to_sell == 0.0:
                            logger.warning(
                                "Exit signal fired but no BTC position found on Alpaca. "
                                "Resetting in_position to False."
                            )
                            self.in_position = False
                            self.entry_price = 0.0
                            save_position_state(self.bot_name, False, 0.0)
                        else:
                            order = self.place_order_tracked(symbol, side, qty_to_sell)
                            if order:
                                logger.info(
                                    f"✅ SELL order placed | "
                                    f"qty={qty_to_sell:.6f} | order_id={order.id}"
                                )
                                self.in_position = False
                                self.entry_price = 0.0
                                self.cooldown_until = time.time() + 900
                                save_position_state(self.bot_name, False, 0.0)

                    else:  # BUY
                        order = self.place_order_tracked(symbol, side, qty)
                        if order:
                            price_now = (await self.get_bollinger_bands())[0]
                            self.in_position = True
                            self.entry_price = price_now or 0.0
                            save_position_state(self.bot_name, True, self.entry_price)
                            logger.info(
                                f"✅ BUY order placed | qty={qty:.6f} | "
                                f"entry={self.entry_price:.2f} | "
                                f"order_id={order.id}"
                            )

            except Exception as e:
                logger.error(f"Main loop error: {e}")

            await asyncio.sleep(60)


if __name__ == "__main__":
    ensure_db_tables()
    bot = MeanReversionBot()
    asyncio.run(bot.run())
