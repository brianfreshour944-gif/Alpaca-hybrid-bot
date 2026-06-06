
#!/usr/bin/env python3
import asyncio
import os
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone, timedelta
import numpy as np
import psycopg2 

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- DATABASE ENGINE ---
def write_trade_to_db(bot_name, symbol, side, price, qty=0.0, value=0.0, order_id='N/A'):
    """Writes a trade event to the unified trades table."""
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ''', (bot_name, 'Alpaca', symbol, side, float(price), float(qty), float(value), str(order_id)))
                conn.commit()
    except Exception as e:
        logger.error(f"Database write error: {e}")

def log_bot_startup(bot_name):
    """Signals to database that this specific bot is live."""
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ''', (bot_name, 'Alpaca', 'N/A', 'SYSTEM', 0.0, 0.0, 0.0, 'STARTUP_SIGNAL'))
                conn.commit()
        logger.info(f"[{bot_name}] Heartbeat: Logged to DB.")
    except Exception as e:
        logger.error(f"Startup log failed: {e}")

# ... [Keep your existing calc_rsi, calc_ema, etc. functions here]

class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'Unnamed_Bot')
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data = CryptoHistoricalDataClient()
        
        # Trigger heartbeat on init
        log_bot_startup(self.bot_name)
        
        logger.info(f'Bot {self.bot_name} initialized.')

    async def manage_position(self):
        # ... (Keep existing logic)
        if exit_reason:
            success, fill_qty, fill_price = await self.submit_order(sym, 'sell', qty)
            if success:
                # Update DB: Using fill_price * fill_qty for the 'value' column
                write_trade_to_db(self.bot_name, sym, 'SELL', fill_price, fill_qty, (fill_price * fill_qty), 'MANUAL_EXIT')
                self.position = {}

    async def look_for_entry(self, portfolio_value, cash, positions_cache):
        # ... (Keep existing logic)
        success, fill_qty, fill_price = await self.submit_order(best_sym, 'buy', qty)
        if success:
            # Update DB:
            write_trade_to_db(self.bot_name, best_sym, 'BUY', fill_price, fill_qty, (fill_price * fill_qty), 'ENTRY_SIGNAL')

    async def run(self):
        while True:
            # ... (Keep existing loop)
            await asyncio.sleep(300)

if __name__ == '__main__':
    bot = MeanReversionBot()
    asyncio.run(bot.run())
