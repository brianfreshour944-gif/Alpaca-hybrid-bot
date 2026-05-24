
#!/usr/bin/env python3
# GROK_OKX_APEX_V8 - HYBRID STRATEGY (WITH POSTGRESQL TRADING DASHBOARD)

import asyncio
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import logging
import json
import os
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# POSTGRESQL DATABASE SETUP
# ==============================================================================

def init_database():
    """Initialize PostgreSQL database and create trades table"""
    try:
        conn = psycopg2.connect(
            host="postgresql",
            database="grafana",
            user="grafana",
            password="grafana"
        )
        cur = conn.cursor()
        
        # Create trades table (similar to TradeLab's structure)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20),
                side VARCHAR(10),
                entry_price DECIMAL(20, 8),
                exit_price DECIMAL(20, 8),
                quantity DECIMAL(20, 8),
                pnl_pct DECIMAL(10, 4),
                pnl_usdt DECIMAL(20, 4),
                score DECIMAL(10, 4),
                status VARCHAR(20) DEFAULT 'OPEN'
            )
        """)
        
        # Create strategy_scores table for tracking predictions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS strategy_scores (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20),
                score DECIMAL(10, 4),
                price DECIMAL(20, 8)
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ PostgreSQL database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.warning("⚠️  Continuing without database - trades will not be saved")
        return False

def save_trade(symbol, side, price, score, quantity=0.01):
    """Save a trade to PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host="postgresql",
            database="grafana",
            user="grafana",
            password="grafana"
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (symbol, side, entry_price, score, quantity, status)
            VALUES (%s, %s, %s, %s, %s, 'OPEN')
        """, (symbol, side, price, score, quantity))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"💾 Trade saved to database: {side} {symbol} @ ${price:.2f}")
        return True
    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        return False

def update_trade_exit(symbol, exit_price, pnl_pct, pnl_usdt):
    """Update an open trade with exit information"""
    try:
        conn = psycopg2.connect(
            host="postgresql",
            database="grafana",
            user="grafana",
            password="grafana"
        )
        cur = conn.cursor()
        cur.execute("""
            UPDATE trades 
            SET exit_price = %s, pnl_pct = %s, pnl_usdt = %s, status = 'CLOSED'
            WHERE symbol = %s AND side = 'BUY' AND status = 'OPEN'
            ORDER BY timestamp DESC LIMIT 1
        """, (exit_price, pnl_pct, pnl_usdt, symbol))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"💾 Trade exit saved: {symbol} PnL: {pnl_pct:.2f}%")
        return True
    except Exception as e:
        logger.error(f"Failed to update trade exit: {e}")
        return False

def save_strategy_score(symbol, score, price):
    """Save strategy score for historical analysis"""
    try:
        conn = psycopg2.connect(
            host="postgresql",
            database="grafana",
            user="grafana",
            password="grafana"
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO strategy_scores (symbol, score, price)
            VALUES (%s, %s, %s)
        """, (symbol, score, price))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        pass  # Silent fail for scores - not critical

# ==============================================================================
# PROMETHEUS METRICS (Keep for compatibility)
# ==============================================================================

from prometheus_client import start_http_server, Gauge, Counter

HYBRID_STRATEGY_SCORE = Gauge('hybrid_strategy_score', 'Latest strategy prediction score', ['symbol'])
HYBRID_POSITION_STATUS = Gauge('hybrid_position_status', 'Active position status', ['symbol'])
HYBRID_LAST_PNL_PCT = Gauge('hybrid_last_pnl_pct', 'PnL % of last closed trade', ['symbol'])
TRADES_TOTAL = Counter('trading_trades_total', 'Total number of trades executed', ['symbol', 'side'])

# ==============================================================================
# HYBRID PREDICTOR (Your existing strategy)
# ==============================================================================

class HybridPredictor:
    def __init__(self):
        self.regime = "neutral"
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
# MAIN BOT CLASS
# ==============================================================================

class GrokApexIroncladBot:
    def __init__(self, paper_mode: bool = True, interval_minutes: int = 5):
        self.paper_mode = paper_mode
        self.interval_minutes = interval_minutes
        
        # Lowered thresholds for more trading activity
        self.buy_threshold = 0.51
        self.sell_threshold = 0.49
        self.position_size = 0.01
        
        self.api_key = os.getenv("OKX_API_KEY", "")
        self.secret = os.getenv("OKX_SECRET_KEY", "")
        self.passphrase = os.getenv("OKX_PASSPHRASE", "")
        
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        self.ml = HybridPredictor()
        self.positions = {}
        self.trades = []
        self.running = True
        self.total_pnl = 0.0
        
        # Initialize database
        self.db_enabled = init_database()
        
        # Start Prometheus metrics (optional, on port 3000)
        try:
            start_http_server(3000)
            logger.info("📊 Prometheus metrics available on port 3000")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")
        
        self.load_state()

    def load_state(self):
        if os.path.exists("grok_apex_state.json"):
            try:
                with open("grok_apex_state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
            except:
                pass

    def save_state(self):
        try:
            with open("grok_apex_state.json", "w") as f:
                json.dump({"positions": self.positions, "trades": self.trades[-100:]}, f)
        except:
            pass

    async def fetch_symbol_data(self, exchange, symbol):
        try:
            ticker = await exchange.watch_ticker(symbol)
            price = ticker['last']
            ohlcv = await exchange.fetch_ohlcv(symbol, '5m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            return symbol, price, df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return symbol, None, None

    async def wait_for_next_even_interval(self):
        now = time.time()
        interval_seconds = self.interval_minutes * 60
        time_to_sleep = interval_seconds - (now % interval_seconds)
        await asyncio.sleep(time_to_sleep)

    async def run(self):
        exchange = ccxtpro.okx({
            'apiKey': self.api_key,
            'secret': self.secret,
            'password': self.passphrase,
            'hostname': os.getenv('OKX_DOMAIN', 'www.okx.com'),
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        if self.paper_mode:
            exchange.set_sandbox_mode(True)
            logger.info("=" * 60)
            logger.info("🚀 PAPER TRADING MODE - HYBRID STRATEGY")
            logger.info(f"🎯 Buy Threshold: {self.buy_threshold} | Sell: {self.sell_threshold}")
            logger.info(f"💾 Database enabled: {self.db_enabled}")
            logger.info("=" * 60)
        
        await exchange.load_markets()
        logger.info(f"📈 Tracking {len(self.symbols)} symbols")
        
        while self.running:
            await self.wait_for_next_even_interval()
            
            tasks = [self.fetch_symbol_data(exchange, symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks)
            
            for symbol, price, df in results:
                if price is None or df is None:
                    continue
                
                try:
                    score = self.ml.predict(df)
                    
                    logger.info(f"📊 {symbol} | ${price:.2f} | Score: {score:.3f}")
                    
                    # Save score to database
                    if self.db_enabled:
                        save_strategy_score(symbol, score, price)
                    
                    # Update Prometheus
                    HYBRID_STRATEGY_SCORE.labels(symbol=symbol).set(score)
                    
                    # ========== BUY SIGNAL ==========
                    if score > self.buy_threshold and symbol not in self.positions:
                        logger.info(f"🟢 BUY: {symbol} @ ${price:.2f} (Score: {score:.3f})")
                        
                        # Save to database
                        if self.db_enabled:
                            save_trade(symbol, 'BUY', price, score, self.position_size)
                        
                        # Update Prometheus
                        TRADES_TOTAL.labels(symbol=symbol, side='buy').inc()
                        
                        if not self.paper_mode:
                            await exchange.create_order(symbol, 'market', 'buy', self.position_size)
                        
                        self.positions[symbol] = {
                            'price': price, 
                            'entry_time': datetime.now().isoformat(),
                            'entry_score': score
                        }
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(1)
                        self.save_state()
                    
                    # ========== SELL SIGNAL ==========
                    elif score < self.sell_threshold and symbol in self.positions:
                        entry_price = self.positions[symbol]['price']
                        pnl_pct = ((price - entry_price) / entry_price) * 100
                        pnl_usdt = (price - entry_price) / entry_price * 10.0
                        self.total_pnl += pnl_usdt
                        
                        logger.info(f"🔴 SELL: {symbol} @ ${price:.2f} | PnL: {pnl_pct:.2f}% (${pnl_usdt:.2f}) | Total PnL: ${self.total_pnl:.2f}")
                        
                        # Update database with exit info
                        if self.db_enabled:
                            update_trade_exit(symbol, price, pnl_pct, pnl_usdt)
                        
                        # Update Prometheus
                        TRADES_TOTAL.labels(symbol=symbol, side='sell').inc()
                        HYBRID_LAST_PNL_PCT.labels(symbol=symbol).set(pnl_pct)
                        
                        if not self.paper_mode:
                            await exchange.create_order(symbol, 'market', 'sell', self.position_size)
                        
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(0)
                        del self.positions[symbol]
                        self.save_state()
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            await asyncio.sleep(1)
        
        await exchange.close()

    def stop(self):
        self.running = False
        self.save_state()
        logger.info("=" * 60)
        logger.info(f"🛑 Bot stopped | Total PnL: ${self.total_pnl:.2f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
    bot = GrokApexIroncladBot(paper_mode=paper_mode, interval_minutes=5)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Shutdown complete")
