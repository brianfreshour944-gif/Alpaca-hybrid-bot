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

def check_status(bot_name):
    """Heartbeat and Kill Switch check for the bot_status table."""
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # 1. Update Heartbeat (Upsert)
                cur.execute('''
                    INSERT INTO bot_status (bot_name, last_update, status)
                    VALUES (%s, NOW(), 'RUNNING')
                    ON CONFLICT (bot_name) 
                    DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
                ''', (bot_name,))
                
                # 2. Check for Kill Switch
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                row = cur.fetchone()
                if row and row[0] == 'STOP':
                    logger.warning(f"🛑 Kill switch activated for {bot_name}. Shutting down.")
                    exit(0)
                conn.commit()
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'Unnamed_Bot')
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data = CryptoHistoricalDataClient()
        
        # Initial status check
        check_status(self.bot_name)
        logger.info(f'Bot {self.bot_name} initialized.')

    async def manage_position(self):
        # ... (Existing logic)
        if exit_reason:
            success, fill_qty, fill_price = await self.submit_order(sym, 'sell', qty)
            if success:
                write_trade_to_db(self.bot_name, sym, 'SELL', fill_price, fill_qty, (fill_price * fill_qty), 'MANUAL_EXIT')
                self.position = {}

    async def look_for_entry(self, portfolio_value, cash, positions_cache):
        # ... (Existing logic)
        success, fill_qty, fill_price = await self.submit_order(best_sym, 'buy', qty)
        if success:
            write_trade_to_db(self.bot_name, best_sym, 'BUY', fill_price, fill_qty, (fill_price * fill_qty), 'ENTRY_SIGNAL')

    async def run(self):
        while True:
            # Heartbeat & Kill Switch check at the start of every loop
            check_status(self.bot_name)
            
            # ... (Existing loop logic)
            await asyncio.sleep(300)

if __name__ == '__main__':
    bot = MeanReversionBot()
    asyncio.run(bot.run())
