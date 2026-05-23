
#!/usr/bin/env python3
"""
Hybrid Alpaca Trading Bot - Phase 3: Deep Reinforcement Learning
- PPO agent replaces rule-based score
- HMM regime detection as state input
- Weekly training on 15k candles

Environment variables:
- ALPACA_API_KEY, ALPACA_SECRET_KEY
- PAPER_MODE (default true)
- INTERVAL_MINUTES (default 5)
- SYMBOLS (default "BTC/USD,ETH/USD,SOL/USD")
- ORDER_SIZE_USD (default 10)
- TRAINING_EPISODES (default 5)
- BACKTEST_BARS (default 15000)
"""

import asyncio
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# RL imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# HMM imports
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HMM REGIME DETECTOR
# ============================================================================
class HMMRegimeDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.hmm_observation_window = 500
        self.regime_labels = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) < 20:
            return np.array([])
        close_series = df['close']
        returns = close_series.pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(0)
        X = np.column_stack([returns.values, volatility.values])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, df: pd.DataFrame):
        if df is None or len(df) < self.hmm_observation_window:
            logger.warning("Insufficient data for HMM training.")
            return
        X = self._prepare_features(df)
        if len(X) < self.hmm_observation_window:
            logger.warning("Not enough feature rows for HMM training.")
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
        try:
            self.model.fit(X_scaled)
            self.is_fitted = True
            logger.info("HMM regime detection model trained successfully.")
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            self.is_fitted = False

    def predict_current_regime(self, df: pd.DataFrame) -> int:
        if not self.is_fitted or self.model is None or df is None or len(df) < 20:
            return 1
        X = self._prepare_features(df)
        if len(X) == 0:
            return 1
        X_recent = X[-1:].reshape(1, -1)
        X_recent_scaled = self.scaler.transform(X_recent)
        return self.model.predict(X_recent_scaled)[0]

    def get_regime_name(self, regime_id: int) -> str:
        return self.regime_labels.get(regime_id, "UNKNOWN")


# ============================================================================
# CUSTOM GYM ENVIRONMENT FOR TRADING
# ============================================================================
class TradingEnv(gym.Env):
    """Custom Gym environment for crypto trading with PPO."""
    
    def __init__(self, df: pd.DataFrame, order_size_usd: float = 10, initial_balance: float = 10000, window_size: int = 20):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.order_size_usd = order_size_usd
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 5,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.in_position = False
        self.entry_price = 0.0
        self.trades = []
        self.done = False
        return self._get_observation(), {}
    
    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return min(100.0, max(0.0, rsi))
    
    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step + 1
        prices = self.df['close'].iloc[start:end].values
        
        returns = self._calculate_returns(prices)[-self.window_size:]
        if len(returns) < self.window_size:
            returns = np.pad(returns, (self.window_size - len(returns), 0), 'constant')
        
        rsi = self._calculate_rsi(prices)
        
        # FIXED: Use pandas Series for volatility calculation
        close_series = self.df['close']
        returns_series = close_series.pct_change()
        volatility_series = returns_series.rolling(20).std()
        volatility = volatility_series.iloc[self.current_step] if self.current_step < len(volatility_series) else 0.0
        volatility = 0.0 if pd.isna(volatility) else volatility
        
        regime = self.df.get('regime', 1).iloc[self.current_step] if 'regime' in self.df.columns else 1
        in_position = float(self.in_position)
        balance_ratio = self.balance / self.initial_balance
        
        features = np.array([rsi, volatility, regime, in_position, balance_ratio], dtype=np.float32)
        obs = np.concatenate([returns.astype(np.float32), features])
        return obs
    
    def step(self, action):
        if self.done or self.current_step >= len(self.df) - 1:
            self.done = True
            return self._get_observation(), 0.0, True, False, {}
        
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0.0
        
        if action == 1 and not self.in_position:
            self.holdings = self.order_size_usd / current_price
            self.balance -= self.order_size_usd
            self.entry_price = current_price
            self.in_position = True
        elif action == 2 and self.in_position:
            proceeds = self.holdings * current_price
            self.balance += proceeds
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl * 100
            self.trades.append({'entry': self.entry_price, 'exit': current_price, 'pnl_pct': pnl * 100})
            self.holdings = 0.0
            self.in_position = False
        
        if self.in_position:
            reward -= 0.01
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            self.done = True
            if self.in_position:
                final_price = self.df['close'].iloc[-1]
                proceeds = self.holdings * final_price
                self.balance += proceeds
                pnl = (final_price - self.entry_price) / self.entry_price
                reward += pnl * 100
        
        obs = self._get_observation()
        return obs, reward, self.done, False, {}
    
    def render(self):
        pass


# ============================================================================
# PPO TRAINER & CALLBACK
# ============================================================================
class SaveModelCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
    
    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            self.model.save(f"{self.save_path}_step_{self.n_calls}")
        return True


class PPOTrainer:
    def __init__(self, symbol: str, order_size_usd: float = 10, training_episodes: int = 5):
        self.symbol = symbol
        self.order_size_usd = order_size_usd
        self.training_episodes = training_episodes
        self.model = None
    
    def prepare_data_with_regime(self, df: pd.DataFrame, regime_detector: HMMRegimeDetector) -> pd.DataFrame:
        df_copy = df.copy()
        regimes = []
        for i in range(len(df_copy)):
            sub_df = df_copy.iloc[:i+1]
            if len(sub_df) >= 20:
                regime = regime_detector.predict_current_regime(sub_df)
            else:
                regime = 1
            regimes.append(regime)
        df_copy['regime'] = regimes
        return df_copy
    
    def train(self, df: pd.DataFrame, regime_detector: HMMRegimeDetector):
        logger.info(f"Training PPO agent for {self.symbol} on {len(df)} bars...")
        df_with_regime = self.prepare_data_with_regime(df, regime_detector)
        env = TradingEnv(df_with_regime, order_size_usd=self.order_size_usd)
        env = DummyVecEnv([lambda: env])
        
        self.model = PPO('MlpPolicy', env, verbose=0, learning_rate=3e-4, n_steps=2048,
                         batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                         clip_range=0.2, ent_coef=0.01, tensorboard_log=None)
        
        logger.info("Starting PPO training...")
        self.model.learn(total_timesteps=self.training_episodes * len(df))
        logger.info("PPO training complete.")
        return self.model
    
    def save_model(self, path: str):
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        if os.path.exists(path):
            self.model = PPO.load(path)
            logger.info(f"Model loaded from {path}")
            return True
        return False
    
    def predict(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        if self.model is None:
            return 0, np.array([0])
        action, _ = self.model.predict(observation, deterministic=True)
        return action, _


# ============================================================================
# MAIN BOT WITH DRL
# ============================================================================
class AlpacaHybridBot:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", "5"))
        self.order_size_usd = float(os.getenv("ORDER_SIZE_USD", "10"))
        self.training_episodes = int(os.getenv("TRAINING_EPISODES", "5"))
        self.optimization_interval_days = int(os.getenv("OPTIMIZATION_INTERVAL_DAYS", "7"))
        self.backtest_bars = int(os.getenv("BACKTEST_BARS", "15000"))

        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_mode)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)

        symbols_raw = os.getenv("SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
        self.symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

        self.regime_detector = HMMRegimeDetector()
        self.trainers: Dict[str, PPOTrainer] = {}
        self.last_optimization: Optional[datetime] = None
        self.running = True
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        
        self.load_state()
    
    def load_state(self):
        if os.path.exists("state.json"):
            try:
                with open("state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                    last_opt_str = data.get("last_optimization")
                    if last_opt_str:
                        self.last_optimization = datetime.fromisoformat(last_opt_str)
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def save_state(self):
        try:
            state = {
                "positions": self.positions,
                "trades": self.trades[-100:],
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None
            }
            with open("state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def fetch_many_bars(self, symbol: str, target_bars: int) -> pd.DataFrame:
        all_bars = []
        max_per_request = 10000
        remaining = target_bars
        end = datetime.now()
        
        while remaining > 0:
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
                    if len(batch) < min(max_per_request, target_bars):
                        break
                    end = batch[0].timestamp - timedelta(minutes=1)
                else:
                    break
                await asyncio.sleep(0.2)
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
        return df
    
    async def train_models(self):
        logger.info("=== Starting weekly DRL training ===")
        
        combined_df = None
        for symbol in self.symbols:
            df = await self.fetch_many_bars(symbol, self.backtest_bars)
            if df.empty or len(df) < 500:
                logger.warning(f"Insufficient data for {symbol}, skipping HMM training")
                continue
            if combined_df is None:
                combined_df = df
                break
        
        if combined_df is not None and len(combined_df) >= self.regime_detector.hmm_observation_window:
            logger.info("Training HMM regime detector...")
            self.regime_detector.fit(combined_df)
        else:
            logger.warning("Not enough data for HMM training")
        
        for symbol in self.symbols:
            logger.info(f"Fetching {self.backtest_bars} bars for {symbol}...")
            df = await self.fetch_many_bars(symbol, self.backtest_bars)
            if df.empty or len(df) < 500:
                logger.warning(f"Not enough data for {symbol}, skipping PPO training")
                continue
            
            trainer = PPOTrainer(symbol, self.order_size_usd, self.training_episodes)
            trainer.train(df, self.regime_detector)
            trainer.save_model(f"ppo_model_{symbol.replace('/', '_')}.zip")
            self.trainers[symbol] = trainer
        
        self.last_optimization = datetime.now()
        self.save_state()
        logger.info("DRL training complete.")
    
    async def weekly_trainer(self):
        while self.running:
            now = datetime.now()
            if self.last_optimization is None:
                await self.train_models()
            else:
                days_since = (now - self.last_optimization).days
                if days_since >= self.optimization_interval_days:
                    await self.train_models()
            await asyncio.sleep(6 * 3600)
    
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
    
    def build_observation(self, symbol: str) -> Optional[np.ndarray]:
        """Build observation vector for the current state."""
        df = self.get_historical_bars(symbol, limit=50)
        if df is None or len(df) < 21:
            return None
        
        closes = df['close'].values
        returns = np.diff(closes) / closes[:-1]
        returns = np.insert(returns, 0, 0)
        returns_window = returns[-20:] if len(returns) >= 20 else np.pad(returns, (20 - len(returns), 0), 'constant')
        
        rsi = self._calculate_rsi(closes)
        
        # FIXED: Use pandas Series for rolling, not numpy array
        close_series = df['close']
        returns_series = close_series.pct_change()
        volatility_series = returns_series.rolling(20).std()
        volatility = volatility_series.iloc[-1] if len(volatility_series) > 0 else 0.0
        volatility = 0.0 if pd.isna(volatility) else volatility
        
        regime = self.regime_detector.predict_current_regime(df)
        in_position = float(symbol in self.positions)
        balance_ratio = 1.0
        
        features = np.array([rsi, volatility, float(regime), in_position, balance_ratio], dtype=np.float32)
        obs = np.concatenate([returns_window.astype(np.float32), features])
        return obs
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return min(100.0, max(0.0, rsi))
    
    def submit_order(self, symbol: str, side: OrderSide):
        try:
            df = self.get_historical_bars(symbol, limit=2)
            if df is None or df.empty:
                logger.error(f"Cannot get price for {symbol}, order aborted.")
                return None
            
            current_price = df['close'].iloc[-1]
            qty = self.order_size_usd / current_price
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
        logger.info(f"🚀 Alpaca Crypto DRL Bot with HMM + PPO")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Interval: {self.interval_minutes} min")
        logger.info(f"   Order size: ${self.order_size_usd}")
        logger.info(f"   Training episodes: {self.training_episodes}")
        logger.info(f"   Training every {self.optimization_interval_days} days")
        logger.info("="*60)
        
        try:
            acc = self.trading_client.get_account()
            logger.info(f"✅ Account ID: {acc.id}")
            logger.info(f"   Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Account error: {e}")
            return
        
        for symbol in self.symbols:
            trainer = PPOTrainer(symbol, self.order_size_usd, self.training_episodes)
            model_path = f"ppo_model_{symbol.replace('/', '_')}.zip"
            if trainer.load_model(model_path):
                self.trainers[symbol] = trainer
                logger.info(f"Loaded existing model for {symbol}")
            else:
                logger.info(f"No existing model for {symbol}, will train on first cycle")
        
        asyncio.create_task(self.weekly_trainer())
        
        try:
            init_df = await self.fetch_many_bars(self.symbols[0], 600)
            if init_df is not None and len(init_df) >= 500:
                self.regime_detector.fit(init_df)
                logger.info("Initial HMM training complete.")
        except Exception as e:
            logger.warning(f"Initial HMM training failed: {e}")
        
        while self.running:
            cycle_start = datetime.now()
            logger.info(f"--- Cycle {cycle_start} ---")
            
            df_regime = self.get_historical_bars(self.symbols[0], limit=100)
            if df_regime is not None and len(df_regime) >= 20:
                regime_id = self.regime_detector.predict_current_regime(df_regime)
                regime_name = self.regime_detector.get_regime_name(regime_id)
                logger.info(f"Market regime: {regime_name}")
            else:
                regime_name = "SIDEWAYS (default)"
            
            for symbol in self.symbols:
                try:
                    trainer = self.trainers.get(symbol)
                    if trainer is None or trainer.model is None:
                        logger.warning(f"No trained model for {symbol}, skipping")
                        continue
                    
                    obs = self.build_observation(symbol)
                    if obs is None:
                        logger.warning(f"Cannot build observation for {symbol}")
                        continue
                    
                    action, _ = trainer.predict(obs.reshape(1, -1))
                    
                    df = self.get_historical_bars(symbol, limit=5)
                    if df is None or df.empty:
                        continue
                    current_price = df['close'].iloc[-1]
                    
                    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                    logger.info(f"{symbol} ${current_price:.2f} | Action: {action_map[action]} | Regime: {regime_name}")
                    
                    if action == 1 and symbol not in self.positions:
                        logger.info(f"🟢 RL BUY decision for {symbol} @ ${current_price:.2f}")
                        order = self.submit_order(symbol, OrderSide.BUY)
                        if order:
                            self.positions[symbol] = {
                                'price': current_price,
                                'entry_time': datetime.now().isoformat(),
                                'order_id': str(order.id)
                            }
                            self.save_state()
                    
                    elif action == 2 and symbol in self.positions:
                        entry_price = self.positions[symbol]['price']
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"🔴 RL SELL decision for {symbol} @ ${current_price:.2f} (PnL: {pnl_pct:.2f}%)")
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
