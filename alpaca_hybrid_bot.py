#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Robust Version
Logs errors to terminal first, then tries the database.
"""

import asyncio
import os
import logging
import psycopg2
import sys
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

# ---------- FAIL-SAFE DATABASE HELPERS ----------
def get_db_connection():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return None
    # Added connect_timeout to prevent the bot from hanging during startup
    return psycopg2.connect(db_url, connect_timeout=5)

def log_error_to_db(bot_name, error_msg):
    # Log to terminal immediately
    logger.error(f"BOT ERROR [{bot_name}]: {error_msg}")
    
    # Try logging to DB, but don't crash if it fails
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)", (bot_name, str(error_msg)))
                conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Could not log error to database: {e}")

def write_trade_to_db(bot_name, symbol, side, price, qty, order_id, fee=0.0):
    try:
        conn = get_db_connection()
        if not conn: return
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, fee, order_id, timestamp)
                VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (bot_name, symbol, side, float(price), float(qty), float(price * qty), float(fee), str(order_id)))
            conn.commit()
            conn.close()
            logger.info(f"Trade logged: {side} {qty} {symbol} @ ${price:.2f} (Fee: ${fee:.4f})")
    except psycopg2.errors.UndefinedColumn:
        logger.error("FATAL: Database table 'trades' is missing the 'fee' column!")
    except Exception as e:
        logger.error(f"Failed to write trade: {e}")

# ---------- BOT CLASS ----------
class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'alpaca_hybrid_bot')
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            logger.critical("API credentials missing!")
            sys.exit(1)
            
        self.trading = TradingClient(api_key, api_secret, paper=True)
        # Attempt status check, but handle failure gracefully
        try:
            from __main__ import check_status
            check_status(self.bot_name)
        except Exception as e:
            logger.warning(f"Status check skipped or failed: {e}")

    async def sync_orders(self):
        try:
            conn = get_db_connection()
            if not conn: return
            with conn.cursor() as cur:
                cur.execute("SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (self.bot_name,))
                open_orders = cur.fetchall()
                for order_id, symbol in open_orders:
                    alpaca_order = self.trading.get_order_by_id(order_id)
                    if alpaca_order.status == 'filled':
                        cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (order_id,))
                        write_trade_to_db(self.bot_name, symbol, alpaca_order.side.value, 
                                          alpaca_order.filled_avg_price, alpaca_order.filled_qty, order_id, fee=0.0)
                conn.commit()
            conn.close()
        except Exception as e:
            log_error_to_db(self.bot_name, f"Sync error: {e}")

    async def run(self):
        logger.info(f"{self.bot_name} running...")
        while True:
            await self.sync_orders()
            await asyncio.sleep(60)

if __name__ == "__main__":
    bot = MeanReversionBot()
    asyncio.run(bot.run())
