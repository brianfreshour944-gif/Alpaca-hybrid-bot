#!/usr/bin/env python3
import asyncio
import os
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone, timedelta
import numpy as np
import psycopg2 # Ensure this is in requirements.txt

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
BUY_THRESHOLD = 0.25
MIN_HOLD_BARS = 24
MAX_HOLD_BARS = 168
ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 8.0
DAILY_LOSS_LIMIT = -5_000.0
POSITION_PCT = 0.40
COOLDOWN_LOSS_H = 48
COOLDOWN_WIN_H = 4
MIN_ATR_PCT = 0.015
BASELINE_PCT = 0.30
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
BTC_SYMBOL = 'BTC/USD'
WARMUP_BARS = 60
CYCLE_SECS = 300

# --- DATABASE ENGINE ---
def write_trade_to_db(bot_name, symbol, side, price, qty=0.0, pnl=0.0, total_pnl=0.0, score=0.0, exit_reason=''):
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO trades (bot_name, symbol, side, price, qty, pnl, total_pnl, score, exit_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (bot_name, symbol, side, price, qty, pnl, total_pnl, score, exit_reason))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database write error: {e}")

# [Keep your calc_rsi, calc_ema, calc_sma, calc_atr, compute_score functions here as they are]

class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'Unnamed_Bot')
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data = CryptoHistoricalDataClient()
        self.closes = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.highs = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.lows = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.volumes = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.position = {}
        self.global_cooldown = datetime.min.replace(tzinfo=timezone.utc)
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.current_day = datetime.now(timezone.utc).date()
        logger.info(f'Bot {self.bot_name} initialized.')

    # [Keep fetch_bars, _norm, get_account, get_all_positions, submit_order, maintain_baseline here]

    async def manage_position(self):
        # ... (keep existing logic)
        if exit_reason:
            success, fill_qty, fill_price = await self.submit_order(sym, 'sell', qty)
            if success:
                pnl = (fill_price - entry_price) * fill_qty
                self.total_pnl += pnl
                # CALL THE NEW DATABASE FUNCTION:
                write_trade_to_db(self.bot_name, sym, 'SELL', fill_price, fill_qty, pnl, self.total_pnl, exit_reason=exit_reason)
                self.position = {}

    async def look_for_entry(self, portfolio_value, cash, positions_cache):
        # ... (keep entry logic)
        success, fill_qty, fill_price = await self.submit_order(best_sym, 'buy', qty)
        if success:
            # ... (update self.position dictionary)
            # CALL THE NEW DATABASE FUNCTION:
            write_trade_to_db(self.bot_name, best_sym, 'BUY', fill_price, fill_qty, score=best_score)

    async def run(self):
        while True:
            # ... (keep loop logic)
            await asyncio.sleep(CYCLE_SECS)

if __name__ == '__main__':
    bot = MeanReversionBot()
    asyncio.run(bot.run())
