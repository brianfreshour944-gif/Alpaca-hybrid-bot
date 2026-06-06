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

def log_error_to_db(bot_name, error_msg):
    """Logs errors to the bot_errors table."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)",
                    (bot_name, str(error_msg))
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to log error to DB: {e}")

def write_trade_to_db(bot_name, symbol, side, price, qty=0.0, value=0.0, order_id='N/A'):
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
        error_msg = f"Database write error: {e}"
        logger.error(error_msg)
        log_error_to_db(bot_name, error_msg)

def check_status(bot_name):
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO bot_status (bot_name, last_update, status)
                    VALUES (%s, NOW(), 'RUNNING')
                    ON CONFLICT (bot_name) 
                    DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
                ''', (bot_name,))
                
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                row = cur.fetchone()
                if row and row[0] == 'STOP':
                    logger.warning(f"🛑 Kill switch activated for {bot_name}. Shutting down.")
                    exit(0)
                conn.commit()
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

# --- BOT CLASS ---
class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'Unnamed_Bot')
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data = CryptoHistoricalDataClient()
        
        check_status(self.bot_name)
        logger.info(f'Bot {self.bot_name} initialized.')

    async def run(self):
        while True:
            try:
                check_status(self.bot_name)
                
                # ... (Existing logic: look_for_entry, manage_position)
                
                await asyncio.sleep(300)
            except Exception as e:
                error_msg = f"Main loop error: {str(e)}"
                logger.error(error_msg)
                log_error_to_db(self.bot_name, error_msg)
                await asyncio.sleep(60)

if __name__ == '__main__':
    try:
        bot = MeanReversionBot()
        asyncio.run(bot.run())
    except Exception as e:
        error_msg = f"FATAL CRASH: {str(e)}"
        logger.critical(error_msg)
        log_error_to_db(os.getenv('BOT_NAME', 'Unnamed_Bot'), error_msg)
        sys.exit(1)
