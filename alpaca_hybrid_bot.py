#!/usr/bin/env python3
"""
Alpaca Hybrid Bot – Mean Reversion Strategy (Updated with Fee Tracking)
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
        return None
    return psycopg2.connect(db_url)

def write_trade_to_db(bot_name, symbol, side, price, qty, order_id, fee=0.0):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            # Note: Ensure you ran 'ALTER TABLE trades ADD COLUMN fee REAL DEFAULT 0;' in SQL
            cur.execute("""
                INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, fee, order_id, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (bot_name, "Alpaca", symbol, side, price, qty, price * qty, fee, order_id))
            conn.commit()
            logger.info(f"Trade logged: {side} {qty} {symbol} @ ${price:.2f} (Fee: ${fee:.4f})")
    except Exception as e:
        logger.error(f"Failed to write trade: {e}")
    finally:
        conn.close()

# ... (Keep ensure_tables, log_error_to_db, register_order_in_db, and check_status as they were) ...

# ---------- BOT CLASS ----------
class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'alpaca_hybrid_bot')
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.trading = TradingClient(api_key, api_secret, paper=True)
        check_status(self.bot_name)

    async def sync_orders(self):
        conn = get_db_connection()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (self.bot_name,))
                for order_id, symbol in cur.fetchall():
                    alpaca_order = self.trading.get_order_by_id(order_id)
                    if alpaca_order.status == 'filled':
                        cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (order_id,))
                        
                        # Calculate Fee: Alpaca doesn't return this in the object, 
                        # so set your estimated fee here (e.g., 0.0 or a custom calc)
                        fee = 0.0 
                        
                        write_trade_to_db(
                            self.bot_name, symbol, alpaca_order.side.value,
                            float(alpaca_order.filled_avg_price),
                            float(alpaca_order.filled_qty),
                            order_id,
                            fee=fee
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Sync orders failed: {e}")
        finally:
            conn.close()

# ... (Keep run loop and entry point) ...
