
#!/usr/bin/env python3
# GROK_OKX_APEX_V8 - HYBRID STRATEGY (FULL METRICS VERSION)

import asyncio
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import logging
import json
import os
import time
from datetime import datetime

# Import Prometheus tools
from prometheus_client import start_http_server, Gauge, Counter, Histogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# PROMETHEUS METRICS - COMPLETE TRADING METRICS
# ==============================================================================

# Existing metrics
HYBRID_STRATEGY_SCORE = Gauge('hybrid_strategy_score', 'Latest strategy prediction score', ['symbol'])
HYBRID_POSITION_STATUS = Gauge('hybrid_position_status', 'Active position status', ['symbol'])
HYBRID_LAST_PNL_PCT = Gauge('hybrid_last_pnl_pct', 'PnL % of last closed trade', ['symbol'])

# NEW TRADING METRICS FOR BUY/SELL LOGS
TRADES_TOTAL = Counter('trading_trades_total', 'Total number of trades executed', ['symbol', 'side'])
TRADING_VOLUME_USDT = Counter('trading_volume_usdt_total', 'Total trading volume in USDT', ['symbol'])
RUNNING_PNL_USDT = Gauge('trading_running_pnl_usdt', 'Accumulated P&L in USDT', ['symbol'])
CURRENT_PRICE = Gauge('trading_current_price_usdt', 'Current market price', ['symbol'])
TOTAL_PNL_USDT = Gauge('trading_total_pnl_usdt', 'Total P&L across all symbols')
OPEN_POSITIONS_COUNT = Gauge('trading_open_positions_count', 'Number of currently open positions')
PREDICTIONS_TOTAL = Counter('strategy_predictions_total', 'Total number of strategy predictions', ['symbol'])


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


class GrokApexIroncladBot:
    def __init__(self, paper_mode: bool = True, interval_minutes: int = 5):
        self.paper_mode = paper_mode
        self.interval_minutes = interval_minutes
        
        # THRESHOLDS - TEMPORARILY LOWERED FOR TESTING
        # Change back to 0.62/0.38 after verifying metrics work
        self.buy_threshold = 0.51   # Temporarily lowered from 0.62
        self.sell_threshold = 0.49  # Temporarily raised from 0.38
        self.position_size = 0.01
        
        # Track P&L per symbol
        self.symbol_pnl = {symbol: 0.0 for symbol in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']}
        self.total_trades_count = 0
        self.winning_trades_count = 0
        
        self.api_key = os.getenv("OKX_API_KEY", "")
        self.secret = os.getenv("OKX_SECRET_KEY", "")
        self.passphrase = os.getenv("OKX_PASSPHRASE", "")
        
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        self.ml = HybridPredictor()
        self.positions = {}
        self.trades = []
        self.running = True
        
        # START METRICS ON PORT 3000 (KEEP AS WORKING CONFIGURATION)
        logger.info("=" * 60)
        logger.info("Starting Prometheus metrics server on port 3000...")
        logger.info("Metrics available at: http://localhost:3000/metrics")
        logger.info("=" * 60)
        start_http_server(3000)
        
        self.load_state()

    def load_state(self):
        if os.path.exists("grok_apex_state.json"):
            try:
                with open("grok_apex_state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                    self.symbol_pnl = data.get("symbol_pnl", self.symbol_pnl)
                    self.total_trades_count = data.get("total_trades_count", 0)
                    self.winning_trades_count = data.get("winning_trades_count", 0)
                    
                for symbol in self.symbols:
                    if symbol in self.positions:
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(1)
                    else:
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(0)
                    RUNNING_PNL_USDT.labels(symbol=symbol).set(self.symbol_pnl.get(symbol, 0.0))
                    
                TOTAL_PNL_USDT.set(sum(self.symbol_pnl.values()))
                OPEN_POSITIONS_COUNT.set(len(self.positions))
                logger.info("✅ State loaded successfully")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def save_state(self):
        try:
            with open("grok_apex_state.json", "w") as f:
                json.dump({
                    "positions": self.positions, 
                    "trades": self.trades[-100:],
                    "symbol_pnl": self.symbol_pnl,
                    "total_trades_count": self.total_trades_count,
                    "winning_trades_count": self.winning_trades_count
                }, f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

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
            logger.info(f"📊 Backtest Results: 74.6% Win Rate | 7.19% Return")
            logger.info(f"🎯 Buy Threshold: {self.buy_threshold} | Sell: {self.sell_threshold}")
            logger.info(f"📈 Prometheus metrics: http://localhost:3000/metrics")
            logger.info("=" * 60)
        
        await exchange.load_markets()
        logger.info(f"📈 Tracking {len(self.symbols)} symbols: {', '.join(self.symbols)}")
        
        while self.running:
            await self.wait_for_next_even_interval()
            
            tasks = [self.fetch_symbol_data(exchange, symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks)
            
            for symbol, price, df in results:
                if price is None or df is None:
                    continue
                
                try:
                    # Update current price metric
                    CURRENT_PRICE.labels(symbol=symbol).set(price)
                    
                    score = self.ml.predict(df)
                    
                    # Record prediction
                    PREDICTIONS_TOTAL.labels(symbol=symbol).inc()
                    
                    logger.info(f"📊 {symbol} | ${price:.2f} | Score: {score:.3f} | Thresh: {self.buy_threshold}/{self.sell_threshold}")
                    
                    # Send live model scores to Prometheus
                    HYBRID_STRATEGY_SCORE.labels(symbol=symbol).set(score)
                    
                    # ========== BUY SIGNAL ==========
                    if score > self.buy_threshold and symbol not in self.positions:
                        logger.info(f"🟢 BUY: {symbol} @ ${price:.2f} (Score: {score:.3f})")
                        
                        # Record buy trade metric
                        TRADES_TOTAL.labels(symbol=symbol, side='buy').inc()
                        
                        # Record volume (approx $10 per trade)
                        TRADING_VOLUME_USDT.labels(symbol=symbol).inc(10.0)
                        
                        if not self.paper_mode:
                            await exchange.create_order(symbol, 'market', 'buy', self.position_size)
                        
                        self.positions[symbol] = {
                            'price': price, 
                            'entry_time': datetime.now().isoformat(),
                            'entry_score': score
                        }
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(1)
                        OPEN_POSITIONS_COUNT.set(len(self.positions))
                        self.save_state()
                    
                    # ========== SELL SIGNAL ==========
                    elif score < self.sell_threshold and symbol in self.positions:
                        entry_price = self.positions[symbol]['price']
                        entry_score = self.positions[symbol].get('entry_score', 0)
                        pnl_pct = ((price - entry_price) / entry_price) * 100
                        
                        # Calculate P&L in USDT (approx $10 position)
                        pnl_usdt = (price - entry_price) / entry_price * 10.0
                        
                        # Update running P&L
                        self.symbol_pnl[symbol] = self.symbol_pnl.get(symbol, 0.0) + pnl_usdt
                        self.total_trades_count += 1
                        if pnl_usdt > 0:
                            self.winning_trades_count += 1
                        
                        win_rate = (self.winning_trades_count / self.total_trades_count * 100) if self.total_trades_count > 0 else 0
                        
                        logger.info(f"🔴 SELL: {symbol} @ ${price:.2f} | PnL: {pnl_pct:.2f}% (${pnl_usdt:.2f}) | Win Rate: {win_rate:.1f}%")
                        
                        # Record sell trade metric
                        TRADES_TOTAL.labels(symbol=symbol, side='sell').inc()
                        TRADING_VOLUME_USDT.labels(symbol=symbol).inc(10.0)
                        
                        # Update metrics
                        HYBRID_LAST_PNL_PCT.labels(symbol=symbol).set(pnl_pct)
                        RUNNING_PNL_USDT.labels(symbol=symbol).set(self.symbol_pnl[symbol])
                        HYBRID_POSITION_STATUS.labels(symbol=symbol).set(0)
                        TOTAL_PNL_USDT.set(sum(self.symbol_pnl.values()))
                        OPEN_POSITIONS_COUNT.set(len(self.positions) - 1)
                        
                        if not self.paper_mode:
                            await exchange.create_order(symbol, 'market', 'sell', self.position_size)
                        
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
        logger.info("🛑 Bot stopped")
        logger.info(f"📊 Final Stats - Total Trades: {self.total_trades_count}, Winning: {self.winning_trades_count}")
        logger.info(f"💰 Final P&L: ${sum(self.symbol_pnl.values()):.2f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
    bot = GrokApexIroncladBot(paper_mode=paper_mode, interval_minutes=5)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Shutdown complete")
