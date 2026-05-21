
#!/usr/bin/env python3
"""
Hybrid Alpaca Trading Bot - Crypto Version with Weekly Self‑Tuning
Phase 1: Weekly backtest on 10k-15k candles + adaptive thresholds.

Environment variables:
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- PAPER_MODE (default true)
- INTERVAL_MINUTES (default 5)
- INIT_BUY_THRESHOLD (default 0.62)
- INIT_SELL_THRESHOLD (default 0.38)
- SYMBOLS (comma-separated, default "BTC/USD,ETH/USD,SOL/USD")
- ORDER_SIZE_USD (default 10)
- OPTIMIZATION_INTERVAL_DAYS (default 7)
- BACKTEST_BARS (default 10000)
"""

import asyncio
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
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

        if rsi < 35:
            score += 0.10
        elif rsi < 45:
            score += 0.05
        elif rsi > 65:
            score -= 0.10
        elif rsi > 55:
            score -= 0.05

        if is_uptrend and score > 0.5:
            score += 0.08
        elif is_downtrend and score < 0.5:
            score -= 0.08

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

        # Initial thresholds (can be overridden by state)
        self.default_buy = float(os.getenv("INIT_BUY_THRESHOLD", "0.62"))
        self.default_sell = float(os.getenv("INIT_SELL_THRESHOLD", "0.38"))

        self.order_size_usd = float(os.getenv("ORDER_SIZE_USD", "10"))

        # Optimization settings
        self.optimization_interval_days = int(os.getenv("OPTIMIZATION_INTERVAL_DAYS", "7"))
        self.backtest_bars = int(os.getenv("BACKTEST_BARS", "10000"))

        # API keys
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_mode)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)

        symbols_raw = os.getenv("SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
        self.symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

        self.ml = HybridPredictor()
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.symbol_thresholds: Dict[str, Tuple[float, float]] = {}  # (buy, sell)
        self.running = True
        self.last_optimization: Optional[datetime] = None

        self.load_state()

    def load_state(self):
        if os.path.exists("state.json"):
            try:
                with open("state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                    # Load per‑symbol thresholds if present
                    saved_thresholds = data.get("symbol_thresholds", {})
                    for sym in self.symbols:
                        if sym in saved_thresholds:
                            self.symbol_thresholds[sym] = tuple(saved_thresholds[sym])
                        else:
                            self.symbol_thresholds[sym] = (self.default_buy, self.default_sell)
                    # Load last optimization timestamp
                    last_opt_str = data.get("last_optimization")
                    if last_opt_str:
                        self.last_optimization = datetime.fromisoformat(last_opt_str)
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        else:
            # No state file, set default thresholds
            for sym in self.symbols:
                self.symbol_thresholds[sym] = (self.default_buy, self.default_sell)

    def save_state(self):
        try:
            # Convert tuple thresholds to list for JSON
            thresholds_serializable = {k: list(v) for k, v in self.symbol_thresholds.items()}
            state = {
                "positions": self.positions,
                "trades": self.trades[-100:],
                "symbol_thresholds": thresholds_serializable,
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None
            }
            with open("state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    # ---------- Data fetching with date shifting (no pagination errors) ----------
    async def fetch_many_bars(self, symbol: str, target_bars: int) -> pd.DataFrame:
        """Fetch up to target_bars 5‑minute bars (uses multiple date‑shifted requests)."""
        all_bars = []
        max_per_request = 10000
        remaining = target_bars
        end = datetime.now()

        while remaining > 0:
            # Estimate how many days back to cover the remaining bars (each day ≈ 288 bars for 5-min)
            days_back = max(1, (remaining // 200) + 2)
            start = end - timedelta(days=days_back)

            try:
                timeframe = TimeFrame(self.interval_minutes, TimeFrame.Minute)
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=min(max_per_request, remaining)
                )
                bars = self.data_client.get_crypto_bars(request)

                if symbol in bars.data and bars.data[symbol]:
                    batch = bars.data[symbol]
                    all_bars.extend(batch)
                    remaining -= len(batch)

                    # If we got fewer than requested, we've reached the earliest available data
                    if len(batch) < min(max_per_request, target_bars):
                        break

                    # Move the end date to the earliest bar's timestamp to avoid overlap
                    end = batch[0].timestamp - timedelta(minutes=1)
                else:
                    break

                await asyncio.sleep(0.2)  # be kind to rate limits
            except Exception as e:
                logger.error(f"Error fetching bars for {symbol}: {e}")
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'timestamp': bar.timestamp
        } for bar in all_bars])
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} bars for {symbol} (target {target_bars})")
        return df

    # ---------- Backtesting for optimization ----------
    def backtest_strategy(self, df: pd.DataFrame, buy_thr: float, sell_thr: float, order_size_usd: float) -> float:
        """
        Simulate trades on a single symbol's DataFrame.
        Returns Sharpe ratio (annualized) of the equity curve.
        """
        if df is None or len(df) < 50:
            return -np.inf

        predictor = HybridPredictor()  # fresh instance
        scores = []
        for i in range(len(df)):
            sub_df = df.iloc[:i+1]
            score = predictor.predict(sub_df)
            scores.append(score)
        scores = np.array(scores)

        # Simulate trading
        cash = 10000.0  # starting capital
        holdings = 0.0
        equity = []
        in_position = False
        entry_price = 0.0

        for i, (idx, row) in enumerate(df.iterrows()):
            price = row['close']
            score = scores[i]

            if not in_position and score > buy_thr:
                qty = order_size_usd / price
                cash -= order_size_usd
                holdings = qty
                entry_price = price
                in_position = True
            elif in_position and score < sell_thr:
                proceeds = holdings * price
                cash += proceeds
                holdings = 0.0
                in_position = False

            current_equity = cash + holdings * price
            equity.append(current_equity)

        if len(equity) < 2:
            return -np.inf

        equity_series = pd.Series(equity)
        df_back = df.copy()
        df_back['equity'] = equity_series
        df_back.set_index('timestamp', inplace=True)
        daily_equity = df_back['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        return sharpe

    async def run_optimization(self):
        """Fetch large historical data, find best thresholds per symbol, update bot."""
        logger.info("=== Starting weekly optimization ===")
        new_thresholds = {}

        for symbol in self.symbols:
            logger.info(f"Fetching {self.backtest_bars} bars for {symbol}...")
            df = await self.fetch_many_bars(symbol, self.backtest_bars)
            if df.empty or len(df) < 500:
                logger.warning(f"Not enough data for {symbol}, keeping current thresholds")
                new_thresholds[symbol] = self.symbol_thresholds.get(symbol, (self.default_buy, self.default_sell))
                continue

            # Grid search
            best_sharpe = -np.inf
            best_buy = self.default_buy
            best_sell = self.default_sell

            buy_grid = np.arange(0.50, 0.76, 0.05)   # 0.50,0.55,0.60,0.65,0.70,0.75
            sell_grid = np.arange(0.25, 0.51, 0.05)  # 0.25,0.30,0.35,0.40,0.45,0.50

            for buy_thr in buy_grid:
                for sell_thr in sell_grid:
                    if buy_thr <= sell_thr:
                        continue
                    sharpe = self.backtest_strategy(df, buy_thr, sell_thr, self.order_size_usd)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_buy = buy_thr
                        best_sell = sell_thr

            logger.info(f"Optimized {symbol}: buy={best_buy:.2f}, sell={best_sell:.2f}, Sharpe={best_sharpe:.3f}")
            new_thresholds[symbol] = (best_buy, best_sell)

        self.symbol_thresholds = new_thresholds
        self.last_optimization = datetime.now()
        self.save_state()
        logger.info(f"Optimization complete. New thresholds: {self.symbol_thresholds}")

    async def weekly_optimizer(self):
        """Background task that runs optimization every N days."""
        while self.running:
            now = datetime.now()
            if self.last_optimization is None:
                await self.run_optimization()
            else:
                days_since = (now - self.last_optimization).days
                if days_since >= self.optimization_interval_days:
                    await self.run_optimization()
            await asyncio.sleep(6 * 3600)  # check every 6 hours

    # ---------- Normal trading methods (same as before) ----------
    def get_historical_bars(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=2)
            timeframe = TimeFrame(self.interval_minutes, TimeFrame.Minute)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit
            )
            bars = self.data_client.get_crypto_bars(request)
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
                logger.warning(f"No crypto data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        df = self.get_historical_bars(symbol, limit=2)
        if df is not None and not df.empty:
            return df['close'].iloc[-1]
        return None

    def submit_order(self, symbol: str, side: OrderSide):
        try:
            price = self.get_current_price(symbol)
            if price is None:
                logger.error(f"Cannot get price for {symbol}, order aborted.")
                return None
            qty = self.order_size_usd / price
            qty = round(qty, 6)
            order = MarketOrderRequest(
                symbol=symbol,
                notional=self.order_size_usd,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            resp = self.trading_client.submit_order(order)
            logger.info(f"Order {side} ${self.order_size_usd} of {symbol} (≈{qty} units) | ID: {resp.id}")
            return resp
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None

    async def run(self):
        logger.info("="*60)
        logger.info(f"🚀 Alpaca Crypto Hybrid Bot with Weekly Self‑Tuning")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Interval: {self.interval_minutes} min")
        logger.info(f"   Order size: ${self.order_size_usd}")
        logger.info(f"   Optimization every {self.optimization_interval_days} days")
        logger.info("="*60)

        try:
            acc = self.trading_client.get_account()
            logger.info(f"✅ Account ID: {acc.id}")
            logger.info(f"   Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Account error: {e}")
            return

        # Start background optimizer
        asyncio.create_task(self.weekly_optimizer())

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
                    buy_thr, sell_thr = self.symbol_thresholds.get(symbol, (self.default_buy, self.default_sell))
                    logger.info(f"{symbol} ${current_price:.2f} | Score: {score:.3f} | Thr: buy>{buy_thr} sell<{sell_thr}")

                    if score > buy_thr and symbol not in self.positions:
                        logger.info(f"🟢 BUY signal {symbol} @ ${current_price:.2f}")
                        order = self.submit_order(symbol, OrderSide.BUY)
                        if order:
                            self.positions[symbol] = {
                                'price': current_price,
                                'entry_time': datetime.now().isoformat(),
                                'order_id': str(order.id)
                            }
                            self.save_state()

                    elif score < sell_thr and symbol in self.positions:
                        entry_price = self.positions[symbol]['price']
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"🔴 SELL signal {symbol} @ ${current_price:.2f} (PnL: {pnl_pct:.2f}%)")
                        order = self.submit_order(symbol, OrderSide.SELL)
                        if order:
                            trade_record = {
                                'symbol': symbol,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl_pct': pnl_pct,
                                'exit_time': datetime.now().isoformat()
                            }
                            self.trades.append(trade_record)
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
