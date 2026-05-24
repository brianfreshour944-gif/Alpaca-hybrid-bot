
#!/usr/bin/env python3
"""
Alpaca Trading Bot - Ultra Simple DRL with Prometheus Metrics
- No async, no gymnasium complexity
- Simple PPO model with basic features
- Full Grafana/Prometheus integration
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

# Prometheus imports
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PROMETHEUS METRICS DEFINITIONS
# ==============================================================================

# Trade counters (buy/sell)
TRADES_TOTAL = Counter('alpaca_trades_total', 'Total trades executed', ['symbol', 'side'])

# Trading volume in USD
TRADING_VOLUME_USD = Counter('alpaca_volume_usd_total', 'Total trading volume in USD', ['symbol'])

# Current P&L tracking
RUNNING_PNL_USD = Gauge('alpaca_running_pnl_usd', 'Running P&L in USD', ['symbol'])

# Current price per symbol
CURRENT_PRICE = Gauge('alpaca_current_price_usd', 'Current market price in USD', ['symbol'])

# RSI value per symbol (for strategy monitoring)
RSI_VALUE = Gauge('alpaca_rsi_value', 'Current RSI value', ['symbol'])

# Signal generated (1=BUY, -1=SELL, 0=HOLD)
TRADING_SIGNAL = Gauge('alpaca_trading_signal', 'Trading signal (1=BUY, -1=SELL, 0=HOLD)', ['symbol'])

# Bot status (1=Running, 0=Stopped)
BOT_STATUS = Gauge('alpaca_bot_status', 'Bot health status', ['bot_name'])

# Open positions count
OPEN_POSITIONS_COUNT = Gauge('alpaca_open_positions_count', 'Number of currently open positions')

# Total P&L across all symbols
TOTAL_PNL_USD = Gauge('alpaca_total_pnl_usd', 'Total P&L across all symbols')

# Order execution latency
ORDER_LATENCY = Histogram('alpaca_order_latency_seconds', 'Order execution latency', ['symbol', 'side'])


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
        self.positions = {}
        
        # Track P&L per symbol
        self.symbol_pnl = {symbol: 0.0 for symbol in self.symbols}
        self.total_trades = 0
        self.winning_trades = 0
        
        # Track entry prices for P&L calculation
        self.entry_prices = {}
        
        # Start Prometheus metrics server on port 9092 (different from OKX bot's 3000)
        logger.info("Starting Prometheus metrics server on port 9092...")
        start_http_server(9092)
        
        # Set bot status to running
        BOT_STATUS.labels(bot_name="alpaca-simple-bot").set(1)
        
        self.load_state()
    
    def load_state(self):
        if os.path.exists("alpaca_positions.json"):
            try:
                with open("alpaca_positions.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.symbol_pnl = data.get("symbol_pnl", self.symbol_pnl)
                    self.total_trades = data.get("total_trades", 0)
                    self.winning_trades = data.get("winning_trades", 0)
                    self.entry_prices = data.get("entry_prices", {})
                
                # Restore position status in Prometheus
                for symbol in self.symbols:
                    if symbol in self.positions:
                        RUNNING_PNL_USD.labels(symbol=symbol).set(self.symbol_pnl.get(symbol, 0.0))
                
                OPEN_POSITIONS_COUNT.set(len(self.positions))
                TOTAL_PNL_USD.set(sum(self.symbol_pnl.values()))
                
                logger.info(f"Loaded state: {len(self.positions)} positions, {self.total_trades} total trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def save_state(self):
        try:
            with open("alpaca_positions.json", "w") as f:
                json.dump({
                    "positions": self.positions,
                    "symbol_pnl": self.symbol_pnl,
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "entry_prices": self.entry_prices
                }, f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def update_aggregate_metrics(self):
        """Update aggregated metrics"""
        TOTAL_PNL_USD.set(sum(self.symbol_pnl.values()))
        OPEN_POSITIONS_COUNT.set(len(self.positions))
    
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
                
                # Update current price metric
                if len(df) > 0:
                    CURRENT_PRICE.labels(symbol=symbol).set(df['close'].iloc[-1])
                
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
        
        # Update RSI metric
        symbol = df.get('symbol', 'unknown')
        # Note: Can't easily get symbol here, will update in get_signal
        
        # Volume
        vol_change = df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean()
        
        return np.array([price_change, rsi, vol_change, 0], dtype=np.float32), rsi
    
    def get_signal(self, symbol):
        """Get trading signal using a simple rule-based approach"""
        df = self.fetch_bars(symbol, limit=50)
        if len(df) < 20:
            TRADING_SIGNAL.labels(symbol=symbol).set(0)
            return "HOLD", 50
        
        features, rsi = self.calculate_features(df)
        price_change, _, vol_change, _ = features
        
        # Update RSI metric
        RSI_VALUE.labels(symbol=symbol).set(rsi)
        
        # Simple rule-based strategy
        # Buy when oversold with volume confirmation
        if rsi < 35 and price_change < -0.002 and vol_change > 1.2:
            TRADING_SIGNAL.labels(symbol=symbol).set(1)  # BUY signal
            return "BUY", rsi
        # Sell when overbought with volume confirmation
        elif rsi > 65 and price_change > 0.002 and symbol in self.positions:
            TRADING_SIGNAL.labels(symbol=symbol).set(-1)  # SELL signal
            return "SELL", rsi
        else:
            TRADING_SIGNAL.labels(symbol=symbol).set(0)  # HOLD
            return "HOLD", rsi
    
    def submit_order(self, symbol, side):
        start_time = time.time()
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=self.order_size_usd,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            resp = self.trading_client.submit_order(order)
            
            # Record latency
            latency = time.time() - start_time
            ORDER_LATENCY.labels(symbol=symbol, side=side.value).observe(latency)
            
            # Record trade metric
            TRADES_TOTAL.labels(symbol=symbol, side=side.value.lower()).inc()
            TRADING_VOLUME_USD.labels(symbol=symbol).inc(self.order_size_usd)
            
            logger.info(f"✓ {side} ${self.order_size_usd} {symbol} | Latency: {latency:.2f}s | ID: {resp.id}")
            return resp
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None
    
    def calculate_pnl(self, symbol, current_price):
        """Calculate P&L for a position"""
        if symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            pnl_usd = (current_price - entry_price) / entry_price * self.order_size_usd
            return pnl_usd
        return 0.0
    
    def run(self):
        logger.info("="*60)
        logger.info(f"🚀 Alpaca Simple Trading Bot - WITH GRAFANA METRICS")
        logger.info(f"   Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Strategy: RSI + Mean Reversion")
        logger.info(f"   Metrics Port: 9092 (Prometheus)")
        logger.info("="*60)
        
        # Test connection
        try:
            acc = self.trading_client.get_account()
            buying_power = float(acc.buying_power)
            logger.info(f"✅ Connected | Buying Power: ${buying_power:,.2f}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            BOT_STATUS.labels(bot_name="alpaca-simple-bot").set(0)
            return
        
        # Main loop
        while True:
            try:
                cycle_start = datetime.now()
                logger.info(f"--- Cycle {cycle_start.strftime('%H:%M:%S')} ---")
                
                for symbol in self.symbols:
                    try:
                        signal, rsi = self.get_signal(symbol)
                        
                        # Get current price for P&L calculation
                        df = self.fetch_bars(symbol, limit=5)
                        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                        
                        if signal == "BUY" and symbol not in self.positions:
                            logger.info(f"🟢 BUY SIGNAL: {symbol} (RSI: {rsi:.1f})")
                            order = self.submit_order(symbol, OrderSide.BUY)
                            if order:
                                self.positions[symbol] = {"time": str(datetime.now())}
                                self.entry_prices[symbol] = current_price
                                self.save_state()
                                self.update_aggregate_metrics()
                        
                        elif signal == "SELL" and symbol in self.positions:
                            # Calculate P&L for this trade
                            pnl_usd = self.calculate_pnl(symbol, current_price)
                            self.symbol_pnl[symbol] = self.symbol_pnl.get(symbol, 0.0) + pnl_usd
                            
                            # Update win/loss tracking
                            self.total_trades += 1
                            if pnl_usd > 0:
                                self.winning_trades += 1
                            
                            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                            
                            logger.info(f"🔴 SELL SIGNAL: {symbol} (RSI: {rsi:.1f}) | PnL: ${pnl_usd:.2f} | Win Rate: {win_rate:.1f}%")
                            
                            order = self.submit_order(symbol, OrderSide.SELL)
                            if order:
                                # Update P&L metric in Prometheus
                                RUNNING_PNL_USD.labels(symbol=symbol).set(self.symbol_pnl[symbol])
                                del self.positions[symbol]
                                del self.entry_prices[symbol]
                                self.save_state()
                                self.update_aggregate_metrics()
                        
                        else:
                            # Update current price and P&L for open positions
                            if symbol in self.positions:
                                pnl_usd = self.calculate_pnl(symbol, current_price)
                                RUNNING_PNL_USD.labels(symbol=symbol).set(self.symbol_pnl.get(symbol, 0.0) + pnl_usd)
                            
                            logger.info(f"⚪ {symbol} | {signal} | RSI: {rsi:.1f} | Price: ${current_price:,.2f}")
                    
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
        BOT_STATUS.labels(bot_name="alpaca-simple-bot").set(0)


if __name__ == "__main__":
    bot = SimplePPOTrader()
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
