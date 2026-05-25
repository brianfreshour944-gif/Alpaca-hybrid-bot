# Alpaca Hybrid Trading Bot

import asyncio
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import logging
import json
import os
import time
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CSV TRADE LOGGING
# ==============================================================================

def init_csv():
    """Initialize CSV file with headers"""
    if not os.path.exists('trades.csv'):
        with open('trades.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Symbol', 'Side', 'Price', 'PnL_USD', 'Total_PnL_USD', 'Score'])

def write_trade_to_csv(symbol, side, price, pnl_usd=None, total_pnl=None, score=None):
    """Write a trade to CSV file"""
    with open('trades.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol,
            side,
            price,
            pnl_usd if pnl_usd is not None else '',
            total_pnl if total_pnl is not None else '',
            score if score is not None else ''
        ])

# ==============================================================================
# HYBRID PREDICTOR
# ==============================================================================

class HybridPredictor:
    def __init__(self):
        self.score_history = []
        
    def predict(self, df):
        if df is None or len(df) < 50:
            return 0.5
        
        close = df['close'].values
        volume = df['volume'].values
        
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0
        
        rsi = self._calculate_rsi(close)
        
        ema9 = self._calculate_ema(close, 9)
        ema21 = self._calculate_ema(close, 21)
        is_uptrend = ema9[-1] > ema21[-1] and close[-1] > ema9[-1]
        is_downtrend = ema9[-1] < ema21[-1] and close[-1] < ema9[-1]
        
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
            if score > 0.5:
                score += 0.05
            else:
                score -= 0.05
        
        score = max(0.0, min(1.0, score))
        
        self.score_history.append(score)
        if len(self.score_history) > 5:
            self.score_history.pop(0)
        
        return np.mean(self.score_history) if self.score_history else score
    
    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-period-1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        return ema

# ==============================================================================
# MAIN ALPACA BOT CLASS
# ==============================================================================

class AlpacaHybridBot:
    def __init__(self, paper_mode: bool = True, interval_minutes: int = 5):
        self.paper_mode = paper_mode
        self.interval_minutes = interval_minutes
        
        self.buy_threshold = 0.51
        self.sell_threshold = 0.49
        self.position_size = 100  # Dollar amount per trade
        
        # Alpaca API keys from environment
        self.api_key = os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY", "")
        
        if not self.api_key or not self.secret_key:
            logger.warning("⚠️ Alpaca API keys not set. Bot will run in analysis-only mode.")
        
        # Alpaca endpoints
        if paper_mode:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        # Stock symbols to trade
        self.symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
        
        self.ml = HybridPredictor()
        self.positions = {}
        self.trades = []
        self.running = True
        self.total_pnl = 0.0
        
        # Initialize CSV
        init_csv()
        logger.info("📊 CSV logging initialized: trades.csv")
        
        self.load_state()

    def load_state(self):
        if os.path.exists("alpaca_state.json"):
            try:
                with open("alpaca_state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                    self.total_pnl = data.get("total_pnl", 0.0)
            except Exception as e:
                logger.warning(f"Load state failed: {e}")

    def save_state(self):
        try:
            with open("alpaca_state.json", "w") as f:
                json.dump({
                    "positions": self.positions,
                    "trades": self.trades[-100:],
                    "total_pnl": self.total_pnl
                }, f)
        except Exception as e:
            logger.warning(f"Save state failed: {e}")

    def get_api(self):
        """Get Alpaca API client"""
        if not self.api_key or not self.secret_key:
            return None
        return tradeapi.REST(self.api_key, self.secret_key, base_url=self.base_url)

    def fetch_historical_data(self, symbol):
        """Fetch historical data from Alpaca"""
        try:
            api = self.get_api()
            if api is None:
                return None, None
            
            # Get 5-minute bars
            bars = api.get_bars(
                symbol,
                timeframe='5Min',
                limit=100
            ).df
            
            if bars.empty:
                return None, None
            
            df = pd.DataFrame({
                'close': bars['close'],
                'volume': bars['volume']
            })
            
            current_price = bars['close'].iloc[-1]
            
            return current_price, df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None, None

    def get_position_qty(self, symbol):
        """Get current position quantity"""
        try:
            api = self.get_api()
            if api is None:
                return 0
            position = api.get_position(symbol)
            return float(position.qty)
        except:
            return 0

    async def run(self):
        logger.info("=" * 60)
        logger.info("🚀 ALPACA PAPER TRADING MODE - HYBRID STRATEGY")
        logger.info(f"🎯 Buy Threshold: {self.buy_threshold} | Sell: {self.sell_threshold}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)
        
        while self.running:
            try:
                # Wait for next interval
                now = time.time()
                interval_seconds = self.interval_minutes * 60
                time_to_sleep = interval_seconds - (now % interval_seconds)
                await asyncio.sleep(max(1, time_to_sleep))
                
                for symbol in self.symbols:
                    try:
                        # Get data
                        price, df = self.fetch_historical_data(symbol)
                        
                        if price is None or df is None:
                            continue
                        
                        score = self.ml.predict(df)
                        current_position = self.get_position_qty(symbol)
                        
                        logger.info(f"📊 {symbol} | ${price:.2f} | Score: {score:.3f} | Position: {current_position}")
                        
                        # ========== BUY SIGNAL ==========
                        if score > self.buy_threshold and current_position == 0:
                            logger.info(f"🟢 BUY SIGNAL: {symbol} @ ${price:.2f} (Score: {score:.3f})")
                            
                            write_trade_to_csv(symbol, 'BUY', price, score=score)
                            
                            if not self.paper_mode and self.api_key:
                                api = self.get_api()
                                shares = int(self.position_size / price)
                                if shares > 0:
                                    api.submit_order(
                                        symbol=symbol,
                                        qty=shares,
                                        side='buy',
                                        type='market',
                                        time_in_force='day'
                                    )
                            
                            self.positions[symbol] = {
                                'price': price,
                                'entry_time': datetime.now().isoformat(),
                                'entry_score': score
                            }
                            self.save_state()
                        
                        # ========== SELL SIGNAL ==========
                        elif score < self.sell_threshold and current_position > 0:
                            entry_price = self.positions[symbol]['price'] if symbol in self.positions else price
                            pnl_pct = ((price - entry_price) / entry_price) * 100
                            pnl_usd = (price - entry_price) / entry_price * self.position_size
                            self.total_pnl += pnl_usd
                            
                            logger.info(f"🔴 SELL SIGNAL: {symbol} @ ${price:.2f} | PnL: {pnl_pct:.2f}% (${pnl_usd:.2f}) | Total: ${self.total_pnl:.2f}")
                            
                            write_trade_to_csv(symbol, 'SELL', price, pnl_usd, self.total_pnl, score)
                            
                            if not self.paper_mode and self.api_key:
                                api = self.get_api()
                                api.submit_order(
                                    symbol=symbol,
                                    qty=current_position,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            
                            if symbol in self.positions:
                                del self.positions[symbol]
                            self.save_state()
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
        
    def stop(self):
        self.running = False
        self.save_state()
        logger.info("=" * 60)
        logger.info(f"🛑 Bot stopped | Total PnL: ${self.total_pnl:.2f}")
        logger.info("=" * 60)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
    bot = AlpacaHybridBot(paper_mode=paper_mode, interval_minutes=5)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Shutdown complete")
