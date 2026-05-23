
#!/usr/bin/env python3
"""
Hybrid Alpaca Trading Bot - Phase 3: Deep Reinforcement Learning
- PPO agent replaces rule-based score
- HMM regime detection as state input
- Weekly training on 15k candles
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
# SIMPLE TRADING ENVIRONMENT FOR PPO
# ============================================================================
class SimpleTradingEnv:
    """Simplified trading environment for PPO training."""
    
    def __init__(self, df: pd.DataFrame, order_size_usd: float = 10, initial_balance: float = 10000):
        self.df = df.reset_index(drop=True)
        self.order_size_usd = order_size_usd
        self.initial_balance = initial_balance
        self.current_step = 20
        self.balance = initial_balance
        self.holdings = 0.0
        self.in_position = False
        self.entry_price = 0.0
        
        # Define observation and action spaces for stable-baselines3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
    
    def reset(self):
        self.current_step = 20
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.in_position = False
        self.entry_price = 0.0
        return self._get_observation()
    
    def _get_observation(self):
        """Return observation as a flat numpy array."""
        start = max(0, self.current_step - 20)
        end = self.current_step + 1
        prices = self.df['close'].iloc[start:end].values
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
        returns = np.insert(returns, 0, 0)
        if len(returns) < 20:
            returns = np.pad(returns, (20 - len(returns), 0), 'constant')
        else:
            returns = returns[-20:]
        
        # Current features
        current_price = prices[-1]
        rsi = self._calculate_rsi(prices)
        
        # Volatility
        close_series = self.df['close']
        returns_series = close_series.pct_change()
        volatility_series = returns_series.rolling(20).std()
        volatility = volatility_series.iloc[self.current_step] if self.current_step < len(volatility_series) else 0.0
        volatility = 0.0 if pd.isna(volatility) else volatility
        
        # Get regime if available
        regime = self.df.get('regime', 1).iloc[self.current_step] if 'regime' in self.df.columns and self.current_step < len(self.df) else 1
        
        in_position = float(self.in_position)
        balance_ratio = self.balance / self.initial_balance
        
        # Combine into flat array
        obs = np.concatenate([
            returns.astype(np.float32),
            np.array([rsi, volatility, regime, in_position, balance_ratio], dtype=np.float32)
        ])
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
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0.0
        
        if action == 1 and not self.in_position:  # BUY
            self.holdings = self.order_size_usd / current_price
            self.balance -= self.order_size_usd
            self.entry_price = current_price
            self.in_position = True
        elif action == 2 and self.in_position:  # SELL
            proceeds = self.holdings * current_price
            self.balance += proceeds
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl * 100
            self.holdings = 0.0
            self.in_position = False
        
        if self.in_position:
            reward -= 0.005  # Small holding penalty
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            if self.in_position:
                final_price = self.df['close'].iloc[-1]
                proceeds = self.holdings * final_price
                self.balance += proceeds
                pnl = (final_price - self.entry_price) / self.entry_price
                reward += pnl * 100
        
        obs = self._get_observation()
        return obs, reward, False, False, {}
    
    def render(self):
        pass


# Import gym after defining the environment
import gymnasium as gym
gym.register('SimpleTradingEnv-v0', entry_point=lambda: None)  # Dummy registration


# ============================================================================
# PPO TRAINER
# ============================================================================
class PPOTrainer:
    def __init__(self, symbol: str, order_size_usd: float = 10, training_episodes: int = 3):
        self.symbol = symbol
        self.order_size_usd = order_size_usd
        self.training_episodes = training_episodes
        self.model = None
    
    def train(self, df: pd.DataFrame, regime_detector: HMMRegimeDetector):
        """Train PPO agent on historical data."""
        logger.info(f"Training PPO agent for {self.symbol} on {len(df)} bars...")
        
        # Add regime predictions to dataframe
        df_copy = df.copy()
        regimes = []
        for i in range(len(df_copy)):
            sub_df = df_copy.iloc[:i+1]
            regime = regime_detector.predict_current_regime(sub_df) if len(sub_df) >= 20 else 1
            regimes.append(regime)
        df_copy['regime'] = regimes
        
        # Create environment
        env = SimpleTradingEnv(df_copy, order_size_usd=self.order_size_usd)
        env = DummyVecEnv([lambda: env])
        
        # Initialize PPO
        self.model = PPO(
            'MlpPolicy',
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )
        
        logger.info(f"Starting PPO training for {self.symbol}...")
        self.model.learn(total_timesteps=self.training_episodes * len(df))
        logger.info(f"PPO training complete for {self.symbol}")
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
    
    def predict(self, observation: np.ndarray) -> int:
        """Predict action from observation. Returns 0=HOLD, 1=BUY, 2=SELL."""
        if self.model is None:
            return 0
        # Ensure observation is 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action[0])


# ============================================================================
# MAIN BOT WITH DRL
# ============================================================================
class AlpacaHybridBot:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", "5"))
        self.order_size_usd = float(os.getenv("ORDER_SIZE_USD", "10"))
        self.training_episodes = int(os.getenv("TRAINING_EPISODES", "3"))
        self.optimization_interval_days = int(os.getenv("OPTIMIZATION_INTERVAL_DAYS", "7"))
        self.backtest_bars = int(os.getenv("BACKTEST_BARS", "10000"))

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
        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df
    
    async def train_models(self):
        logger.info("=== Starting weekly DRL training ===")
        
        # Train HMM first
        for symbol in self.symbols:
            df = await self.fetch_many_bars(symbol, 1000)
            if not df.empty and len(df) >= self.regime_detector.hmm_observation_window:
                logger.info(f"Training HMM regime detector on {symbol}...")
                self.regime_detector.fit(df)
                break
        
        # Train PPO for each symbol
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
        await asyncio.sleep(5)  # Wait for bot to start
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
        
        # RSI
        rsi = self._calculate_rsi(closes)
        
        # Volatility
        close_series = df['close']
        returns_series = close_series.pct_change()
        volatility_series = returns_series.rolling(20).std()
        volatility = volatility_series.iloc[-1] if len(volatility_series) > 0 else 0.0
        volatility = 0.0 if pd.isna(volatility) else volatility
        
        # Regime
        regime = self.regime_detector.predict_current_regime(df)
        in_position = float(symbol in self.positions)
        balance_ratio = 1.0
        
        # Combine into flat array
        obs = np.concatenate([
            returns_window.astype(np.float32),
            np.array([rsi, volatility, float(regime), in_position, balance_ratio], dtype=np.float32)
        ])
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
        
        # Load existing models
        for symbol in self.symbols:
            trainer = PPOTrainer(symbol, self.order_size_usd, self.training_episodes)
            model_path = f"ppo_model_{symbol.replace('/', '_')}.zip"
            if trainer.load_model(model_path):
                self.trainers[symbol] = trainer
                logger.info(f"Loaded existing model for {symbol}")
        
        # Start weekly training
        asyncio.create_task(self.weekly_trainer())
        
        # Initial HMM training
        try:
            init_df = await self.fetch_many_bars(self.symbols[0], 600)
            if init_df is not None and len(init_df) >= 500:
                self.regime_detector.fit(init_df)
                logger.info("Initial HMM training complete.")
        except Exception as e:
            logger.warning(f"Initial HMM training failed: {e}")
        
        # Main trading loop
        while self.running:
            cycle_start = datetime.now()
            logger.info(f"--- Cycle {cycle_start} ---")
            
            # Get market regime
            df_regime = self.get_historical_bars(self.symbols[0], limit=100)
            if df_regime is not None and len(df_regime) >= 20:
                regime_id = self.regime_detector.predict_current_regime(df_regime)
                regime_name = self.regime_detector.get_regime_name(regime_id)
                logger.info(f"Market regime: {regime_name}")
            else:
                regime_name = "SIDEWAYS"
            
            for symbol in self.symbols:
                try:
                    trainer = self.trainers.get(symbol)
                    if trainer is None or trainer.model is None:
                        logger.debug(f"No trained model for {symbol}, skipping")
                        continue
                    
                    obs = self.build_observation(symbol)
                    if obs is None:
                        continue
                    
                    # Get action from PPO
                    action = trainer.predict(obs)
                    
                    # Get current price
                    df = self.get_historical_bars(symbol, limit=5)
                    if df is None or df.empty:
                        continue
                    current_price = df['close'].iloc[-1]
                    
                    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                    logger.info(f"{symbol} ${current_price:.2f} | Action: {action_map[action]} | Regime: {regime_name}")
                    
                    # Execute action
                    if action == 1 and symbol not in self.positions:
                        logger.info(f"🟢 RL BUY for {symbol} @ ${current_price:.2f}")
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
                        logger.info(f"🔴 RL SELL for {symbol} @ ${current_price:.2f} (PnL: {pnl_pct:.2f}%)")
                        order = self.submit_order(symbol, OrderSide.SELL)
                        if order:
                            self.trades.append({
                                'symbol': symbol,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl_pct': pnl_pct,
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
