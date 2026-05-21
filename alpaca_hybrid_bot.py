#!/usr/bin/env python3
"""
Hybrid Alpaca Trading Bot - Mean Reversion + Trend Filter
Backtest Win Rate: 74.6% | Return: 7.19%

Environment variables:
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- PAPER_MODE (default true)
- INTERVAL_MINUTES (default 5)
- BUY_THRESHOLD (default 0.62)
- SELL_THRESHOLD (default 0.38)
- SYMBOLS (comma-separated, default "SPY,QQQ,IWM")
"""

import asyncio
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment, DataFeed  # ✅ Fixed import
from alpaca.data.timeframe import TimeFrame

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class HybridPredictor:
    """Your proven hybrid strategy: Mean Reversion + RSI + Trend Filter."""
    def __init__(self):
        self.score_history = []

    def predict(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 50:
            return 0.5

        close = df['close'].values
        volume = df['volume'].values

        # Mean reversion (Z-score)
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0

        # RSI
        rsi = self._calculate_rsi(close)

        # Trend (EMA9/EMA21)
        ema9 = self._calculate_ema(close, 9)
        ema21 = self._calculate_ema(close, 21)
        is_uptrend = ema9[-1] > ema21[-1] and close[-1] > ema9[-1]
        is_downtrend = ema9[-1] < ema21[-1] and close[-1] < ema9[-1]

        # Volume surge
        vol_avg = np.mean(volume[-10:])
        vol_surge = volume[-1] / vol_avg if vol_avg > 0 else 1

        score = 0.5
        # Z-score contribution
        if z_score < -1.2:
            score += 0.35
        elif z_score < -0.8:
            score += 0.25
        elif z_score < -0.4:
            score += 0.15
        elif z_score > 1.2:
            score -= 0.35
        elif z_score > 0.8:
            score -= 0.25
        elif z_score > 0.4:
            score -= 0.15

        # RSI contribution
        if rsi < 35:
            score += 0.10
        elif rsi < 45:
            score += 0.05
        elif rsi > 65:
            score -= 0.10
        elif rsi > 55:
            score -= 0.05

        # Trend filter
        if is_uptrend and score > 0.5:
            score += 0.08
        elif is_downtrend and score < 0.5:
            score -= 0.08

        # Volume confirmation
        if vol_surge > 1.3:
            score += 0.05 if score > 0.5 else -0.05

        score = max(0.0, min(1.0, score))

        self.score_history.append(score)
        if len(self.score_history) > 5:
            self.score_history.pop(0)

        return np.mean(self.score_history) if self.score_history else score

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-period-1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        return ema


class AlpacaHybridBot:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", "5"))

        # Strategy thresholds
        self.buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.62"))
        self.sell_threshold = float(os.getenv("SELL_THRESHOLD", "0.38"))

        # API keys from environment
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

        # Clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_mode)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        # Symbols: parse comma-separated, strip spaces, ignore extra text
        symbols_raw = os.getenv("SYMBOLS", "SPY,QQQ,IWM")
        self.symbols = []
        for s in symbols_raw.split(','):
            s = s.strip().split()[0]  # e.g., "IWM (or add" becomes "IWM"
            if s and s.isalpha():
                self.symbols.append(s)
        if not self.symbols:
            self.symbols = ['SPY', 'QQQ', 'IWM']

        self.ml = HybridPredictor()
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.running = True

        self.load_state()

    def load_state(self):
        if os.path.exists("state.json"):
            try:
                with open("state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def save_state(self):
        try:
            with open("state.json", "w") as f:
                json.dump({
                    "positions": self.positions,
                    "trades": self.trades[-100:]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_historical_bars(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=2)
            timeframe = TimeFrame(self.interval_minutes, TimeFrame.Minute)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX   # ✅ Free tier IEX data
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol in bars.data and bars.data[symbol]:
                df = pd.DataFrame([{
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                } for bar in bars.data[symbol]])
                return df
            else:
                logger.warning(f"No data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def submit_order(self, symbol: str, side: OrderSide, qty: int = 1):
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            resp = self.trading_client.submit_order(order)
            logger.info(f"Order {side} {qty} {symbol} (ID: {resp.id})")
            return resp
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None

    async def run(self):
        logger.info("="*60)
        logger.info(f"🚀 Alpaca Hybrid Bot")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Interval: {self.interval_minutes} min")
        logger.info(f"   Buy > {self.buy_threshold} | Sell < {self.sell_threshold}")
        logger.info("="*60)

        try:
            acc = self.trading_client.get_account()
            logger.info(f"✅ Account ID: {acc.id}")
            logger.info(f"   Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Account error: {e}")
            return

        while self.running:
            cycle_start = datetime.now()
            logger.info(f"--- Cycle {cycle_start} ---")

            for symbol in self.symbols:
                try:
                    df = self.get_historical_bars(symbol, limit=100)
                    if df is None or len(df) < 50:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue

                    current_price = df['close'].iloc[-1]
                    score = self.ml.predict(df)
                    logger.info(f"{symbol} ${current_price:.2f} | Score: {score:.3f}")

                    if score > self.buy_threshold and symbol not in self.positions:
                        logger.info(f"🟢 BUY {symbol} @ ${current_price:.2f}")
                        order = self.submit_order(symbol, OrderSide.BUY)
                        if order:
                            self.positions[symbol] = {
                                'price': current_price,
                                'entry_time': datetime.now().isoformat(),
                                'order_id': order.id
                            }
                            self.save_state()

                    elif score < self.sell_threshold and symbol in self.positions:
                        entry = self.positions[symbol]['price']
                        pnl = (current_price - entry) / entry * 100
                        logger.info(f"🔴 SELL {symbol} @ ${current_price:.2f} (PnL: {pnl:.2f}%)")
                        order = self.submit_order(symbol, OrderSide.SELL)
                        if order:
                            self.trades.append({
                                'symbol': symbol,
                                'entry_price': entry,
                                'exit_price': current_price,
                                'pnl_pct': pnl,
                                'exit_time': datetime.now().isoformat()
                            })
                            del self.positions[symbol]
                            self.save_state()

                except Exception as e:
                    logger.error(f"Error on {symbol}: {e}")

            elapsed = (datetime.now() - cycle_start).total_seconds()
            wait = max(0, self.interval_minutes * 60 - elapsed)
            logger.info(f"Wait {wait:.0f}s...")
            await asyncio.sleep(wait)

        logger.info("Bot stopped.")

    def stop(self):
        self.running = False
        self.save_state()


if __name__ == "__main__":
    bot = AlpacaHybridBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Shutdown complete.")
