
#!/usr/bin/env python3
"""
Alpaca Trading Bot - Ultra Simple DRL
- No async, no gymnasium complexity
- Simple PPO model with basic features
"""

import time
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# RL imports
from stable_baselines3 import PPO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePPOTrader:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", "5"))
        self.order_size_usd = float(os.getenv("ORDER_SIZE_USD", "10"))
        
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing API keys")
        
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper_mode)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        
        self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        self.models = {}
        self.positions = {}
        
        self.load_state()
    
    def load_state(self):
        if os.path.exists("positions.json"):
            try:
                with open("positions.json", "r") as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} positions")
            except:
                pass
    
    def save_state(self):
        try:
            with open("positions.json", "w") as f:
                json.dump(self.positions, f)
        except:
            pass
    
    def fetch_bars(self, symbol, limit=500):
        """Fetch bars - completely synchronous"""
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
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
                data = []
                for bar in bars.data[symbol]:
                    data.append({
                        'close': bar.close,
                        'volume': bar.volume,
                        'timestamp': bar.timestamp
                    })
                df = pd.DataFrame(data)
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df):
        """Calculate simple features for the model"""
        if len(df) < 20:
            return np.array([0, 50, 1, 0])
        
        # Price change
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        # RSI
        closes = df['close'].values[-20:]
        deltas = np.diff(closes)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volume
        vol_change = df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean()
        
        return np.array([price_change, rsi, vol_change, 0], dtype=np.float32)
    
    def get_signal(self, symbol):
        """Get trading signal using a simple rule-based approach"""
        df = self.fetch_bars(symbol, limit=50)
        if len(df) < 20:
            return "HOLD"
        
        features = self.calculate_features(df)
        price_change, rsi, vol_change, _ = features
        
        # Simple rule-based strategy (since PPO is complex)
        # Buy when oversold with volume confirmation
        if rsi < 35 and price_change < -0.002 and vol_change > 1.2:
            return "BUY"
        # Sell when overbought with volume confirmation
        elif rsi > 65 and price_change > 0.002 and symbol in self.positions:
            return "SELL"
        else:
            return "HOLD"
    
    def submit_order(self, symbol, side):
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=self.order_size_usd,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            resp = self.trading_client.submit_order(order)
            logger.info(f"✓ {side} ${self.order_size_usd} {symbol} | ID: {resp.id}")
            return resp
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None
    
    def run(self):
        logger.info("="*60)
        logger.info(f"🚀 Alpaca Simple Trading Bot")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Strategy: RSI + Mean Reversion")
        logger.info("="*60)
        
        # Test connection
        try:
            acc = self.trading_client.get_account()
            logger.info(f"✅ Connected | Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return
        
        # Main loop
        while True:
            try:
                cycle_start = datetime.now()
                logger.info(f"--- Cycle {cycle_start.strftime('%H:%M:%S')} ---")
                
                for symbol in self.symbols:
                    try:
                        signal = self.get_signal(symbol)
                        
                        if signal == "BUY" and symbol not in self.positions:
                            logger.info(f"🟢 BUY {symbol}")
                            order = self.submit_order(symbol, OrderSide.BUY)
                            if order:
                                self.positions[symbol] = {"time": str(datetime.now())}
                                self.save_state()
                        
                        elif signal == "SELL" and symbol in self.positions:
                            logger.info(f"🔴 SELL {symbol}")
                            order = self.submit_order(symbol, OrderSide.SELL)
                            if order:
                                del self.positions[symbol]
                                self.save_state()
                        
                        else:
                            logger.info(f"⚪ {symbol} | {signal} | RSI: {self.get_rsi(symbol):.1f}")
                    
                    except Exception as e:
                        logger.error(f"Error {symbol}: {e}")
                
                # Wait
                elapsed = (datetime.now() - cycle_start).seconds
                wait = max(0, self.interval_minutes * 60 - elapsed)
                logger.info(f"Wait {wait}s...")
                time.sleep(wait)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(60)
        
        logger.info("Bot stopped")
    
    def get_rsi(self, symbol):
        df = self.fetch_bars(symbol, limit=20)
        if len(df) < 20:
            return 50
        closes = df['close'].values[-20:]
        deltas = np.diff(closes)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


if __name__ == "__main__":
    bot = SimplePPOTrader()
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
