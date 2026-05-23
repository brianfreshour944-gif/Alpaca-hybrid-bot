
#!/usr/bin/env python3
"""
Hybrid Alpaca Trading Bot - Phase 3: Synchronous DRL
- PPO agent with simple observation space
- Synchronous operations (no async complexity)
"""

import time
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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
import gymnasium as gym
from gymnasium import spaces

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE TRADING ENVIRONMENT
# ============================================================================
class TradingEnv(gym.Env):
    """Simple trading environment for PPO."""
    
    def __init__(self, data: pd.DataFrame, order_size: float = 10):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.order_size = order_size
        self.current_idx = 50
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.current_idx = 50
        self.balance = 10000.0
        self.holdings = 0.0
        self.in_position = False
        self.entry_price = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.current_idx < 1:
            return np.array([0.0, 50.0, 0.0, 0.0], dtype=np.float32)
        
        current_price = self.data.iloc[self.current_idx]['close']
        prev_price = self.data.iloc[self.current_idx - 1]['close']
        price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        rsi = self._calc_rsi()
        
        current_vol = self.data.iloc[self.current_idx]['volume']
        avg_vol = self.data.iloc[max(0, self.current_idx-20):self.current_idx]['volume'].mean()
        vol_change = current_vol / avg_vol if avg_vol > 0 else 1
        
        in_pos = 1.0 if self.in_position else 0.0
        
        return np.array([float(price_change), float(rsi), float(vol_change), float(in_pos)], dtype=np.float32)
    
    def _calc_rsi(self, period: int = 14) -> float:
        if self.current_idx < period:
            return 50.0
        
        closes = self.data.iloc[max(0, self.current_idx-period):self.current_idx+1]['close'].values
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return min(100.0, max(0.0, rsi))
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
            return self._get_obs(), 0.0, True, False, {}
        
        current_price = self.data.iloc[self.current_idx]['close']
        reward = 0.0
        
        if action == 1 and not self.in_position:
            self.holdings = self.order_size / current_price
            self.balance -= self.order_size
            self.entry_price = current_price
            self.in_position = True
            reward = -0.01
        
        elif action == 2 and self.in_position:
            proceeds = self.holdings * current_price
            self.balance += proceeds
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl * 100
            self.holdings = 0.0
            self.in_position = False
        
        if self.in_position:
            reward -= 0.001
        
        self.current_idx += 1
        
        if self.current_idx >= len(self.data) - 1:
            if self.in_position:
                final_price = self.data.iloc[-1]['close']
                proceeds = self.holdings * final_price
                self.balance += proceeds
                pnl = (final_price - self.entry_price) / self.entry_price
                reward += pnl * 100
            return self._get_obs(), reward, True, False, {}
        
        return self._get_obs(), reward, False, False, {}


# ============================================================================
# MAIN BOT (SYNCHRONOUS)
# ============================================================================
class AlpacaHybridBot:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", "5"))
        self.order_size_usd = float(os.getenv("ORDER_SIZE_USD", "10"))
        
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
        
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_mode)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        
        symbols_raw = os.getenv("SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
        self.symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        
        self.models: Dict[str, PPO] = {}
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
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def save_state(self):
        try:
            state = {
                "positions": self.positions,
                "trades": self.trades[-100:]
            }
            with open("state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def fetch_bars(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical bars (synchronous)."""
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
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
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                } for bar in bars.data[symbol]])
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def train_model(self, symbol: str):
        """Train PPO model for a symbol."""
        logger.info(f"Training model for {symbol}...")
        
        df = self.fetch_bars(symbol, limit=5000)
        if len(df) < 200:
            logger.warning(f"Not enough data for {symbol}")
            return
        
        env = TradingEnv(df, order_size=self.order_size_usd)
        env = DummyVecEnv([lambda: env])
        
        model = PPO(
            'MlpPolicy',
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )
        
        logger.info(f"Starting training for {symbol}...")
        model.learn(total_timesteps=10000)
        model.save(f"model_{symbol.replace('/', '_')}.zip")
        self.models[symbol] = model
        logger.info(f"Training complete for {symbol}")
    
    def ensure_model(self, symbol: str):
        """Ensure model exists."""
        model_path = f"model_{symbol.replace('/', '_')}.zip"
        
        if symbol not in self.models:
            if os.path.exists(model_path):
                logger.info(f"Loading existing model for {symbol}")
                self.models[symbol] = PPO.load(model_path)
            else:
                logger.info(f"No model found for {symbol}, training...")
                self.train_model(symbol)
    
    def get_signal(self, symbol: str) -> str:
        """Get trading signal from model."""
        model = self.models.get(symbol)
        if model is None:
            return "HOLD"
        
        # Fetch recent data
        df = self.fetch_bars(symbol, limit=100)
        if len(df) < 50:
            return "HOLD"
        
        if len(df) < 2:
            return "HOLD"
        
        current_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close']
        price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        # Simple RSI
        closes = df['close'].values[-20:]
        if len(closes) >= 14:
            deltas = np.diff(closes)
            gains = deltas[deltas > 0]
            losses = -deltas[deltas < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0
        
        # Volume change
        current_vol = df.iloc[-1]['volume']
        avg_vol = df.iloc[-20:]['volume'].mean()
        vol_change = current_vol / avg_vol if avg_vol > 0 else 1
        
        # In position?
        in_position = 1.0 if symbol in self.positions else 0.0
        
        # Create observation
        obs = np.array([[float(price_change), float(rsi), float(vol_change), float(in_position)]], dtype=np.float32)
        
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        return ["HOLD", "BUY", "SELL"][action[0]]
    
    def submit_order(self, symbol: str, side: OrderSide):
        """Submit market order."""
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=self.order_size_usd,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            resp = self.trading_client.submit_order(order)
            logger.info(f"Order {side} ${self.order_size_usd} of {symbol} | ID: {resp.id}")
            return resp
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None
    
    def run(self):
        """Main trading loop (synchronous)."""
        logger.info("="*60)
        logger.info(f"🚀 Alpaca Crypto DRL Bot (Synchronous)")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Interval: {self.interval_minutes} min")
        logger.info(f"   Order size: ${self.order_size_usd}")
        logger.info("="*60)
        
        try:
            acc = self.trading_client.get_account()
            logger.info(f"✅ Account ID: {acc.id}")
            logger.info(f"   Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Account error: {e}")
            return
        
        # Ensure models are loaded/trained
        for symbol in self.symbols:
            self.ensure_model(symbol)
        
        # Main trading loop
        while True:
            try:
                cycle_start = datetime.now()
                logger.info(f"--- Cycle {cycle_start} ---")
                
                for symbol in self.symbols:
                    try:
                        signal = self.get_signal(symbol)
                        logger.info(f"{symbol} | Signal: {signal}")
                        
                        if signal == "BUY" and symbol not in self.positions:
                            logger.info(f"🟢 BUY {symbol}")
                            order = self.submit_order(symbol, OrderSide.BUY)
                            if order:
                                self.positions[symbol] = {
                                    'entry_time': datetime.now().isoformat(),
                                    'order_id': str(order.id)
                                }
                                self.save_state()
                        
                        elif signal == "SELL" and symbol in self.positions:
                            logger.info(f"🔴 SELL {symbol}")
                            order = self.submit_order(symbol, OrderSide.SELL)
                            if order:
                                self.trades.append({
                                    'symbol': symbol,
                                    'exit_time': datetime.now().isoformat()
                                })
                                del self.positions[symbol]
                                self.save_state()
                        
                    except Exception as e:
                        logger.error(f"Error on {symbol}: {e}")
                
                # Wait for next interval
                elapsed = (datetime.now() - cycle_start).total_seconds()
                wait = max(0, self.interval_minutes * 60 - elapsed)
                logger.info(f"Wait {wait:.0f}s...")
                time.sleep(wait)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(60)
        
        logger.info("Bot stopped.")
        self.save_state()


if __name__ == "__main__":
    bot = AlpacaHybridBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")
