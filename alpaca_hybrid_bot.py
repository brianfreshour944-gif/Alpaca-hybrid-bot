#!/usr/bin/env python3
# Alpaca Crypto Hybrid Trading Bot - FINAL with aggressive duplicate-buy prevention

import asyncio
import pandas as pd
import numpy as np
import logging
import json
import os
import time
import csv
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CSV TRADE LOGGING
# ==============================================================================

def init_csv():
    if not os.path.exists('trades.csv'):
        with open('trades.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Symbol', 'Side', 'Price',
                'PnL_USD', 'Total_PnL_USD', 'Score'
            ])

def write_trade_to_csv(symbol, side, price, pnl_usd=None, total_pnl=None, score=None):
    with open('trades.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol, side, price,
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
# MAIN BOT CLASS
# ==============================================================================

class AlpacaCryptoBot:
    def __init__(self, paper_mode: bool = True, interval_minutes: int = 5):
        self.paper_mode = paper_mode
        self.interval_minutes = interval_minutes

        # STRATEGY SETTINGS
        self.buy_threshold = 0.51
        self.sell_threshold = 0.49
        self.position_usd = 15            
        self.min_buying_power_reserve = 20

        # API KEYS
        self.api_key = os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY", "")

        # CLIENTS
        self.trading_client = None
        self.data_client = CryptoHistoricalDataClient()

        if self.api_key and self.secret_key:
            self.trading_client = TradingClient(
                self.api_key, self.secret_key, paper=paper_mode
            )
            logger.info("✅ Alpaca trading client initialized")
            try:
                account = self.trading_client.get_account()
                logger.info(f"✅ Connected to Alpaca Account: {account.id}")
                logger.info(f"💵 Buying Power: ${account.buying_power}")
            except Exception as e:
                logger.error(f"❌ Alpaca connection failed: {e}")
        else:
            logger.warning("⚠️ Alpaca API keys not set. Bot will run in analysis-only mode.")

        # SYMBOLS
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'LTC/USD']
        self.ml = HybridPredictor()
        self.positions = {}        
        self.trades = []
        self.cooldowns = {}         
        self.global_cooldown_until = None
        self.running = True
        self.total_pnl = 0.0

        init_csv()
        logger.info("📊 CSV logging initialized: trades.csv")
        self.load_state()

    # ==========================================================================
    # STATE MANAGEMENT
    # ==========================================================================
    def load_state(self):
        if os.path.exists("alpaca_crypto_state.json"):
            try:
                with open("alpaca_crypto_state.json", "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trades = data.get("trades", [])
                    self.total_pnl = data.get("total_pnl", 0.0)
                    logger.info(f"📂 Loaded state: {len(self.positions)} positions")
            except Exception as e:
                logger.warning(f"Load state failed: {e}")

    def save_state(self):
        try:
            with open("alpaca_crypto_state.json", "w") as f:
                json.dump({
                    "positions": self.positions,
                    "trades": self.trades[-100:],
                    "total_pnl": self.total_pnl
                }, f)
        except Exception as e:
            logger.warning(f"Save state failed: {e}")

    # ==========================================================================
    # ACCOUNT & POSITION HELPERS
    # ==========================================================================
    def get_buying_power(self):
        try:
            if self.trading_client:
                account = self.trading_client.get_account()
                return float(account.buying_power)
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
        return 0.0

    def get_position_qty(self, symbol):
        try:
            if self.trading_client:
                positions = self.trading_client.get_all_positions()
                for pos in positions:
                    if pos.symbol == symbol:
                        return float(pos.qty)
        except Exception as e:
            logger.debug(f"Position check failed: {e}")
        return 0

    def verify_position_exists(self, symbol):
        for attempt in range(3):
            time.sleep(2)
            qty = self.get_position_qty(symbol)
            if qty > 0:
                return True
            logger.info(f"Waiting for position to appear (attempt {attempt+1}/3)...")
        return False

    # ==========================================================================
    # ORDER EXECUTION
    # ==========================================================================
    def submit_crypto_order(self, symbol, usd_amount, side):
        try:
            if not self.trading_client:
                logger.error("Trading client not initialized")
                return False

            if side == 'buy':
                buying_power = self.get_buying_power()
                if buying_power < self.min_buying_power_reserve:
                    logger.error(f"Insufficient buying power: ${buying_power:.2f} (need ${self.min_buying_power_reserve})")
                    self.global_cooldown_until = datetime.now() + timedelta(minutes=30)
                    return False
                if buying_power < usd_amount:
                    logger.error(f"Not enough funds: need ${usd_amount}, have ${buying_power:.2f}")
                    self.global_cooldown_until = datetime.now() + timedelta(minutes=30)
                    return False

            price = self.get_current_price(symbol)
            if not price:
                logger.error(f"Cannot get price for {symbol}")
                return False

            qty = round(usd_amount / price, 6)
            order_value = qty * price
            if order_value < 10:
                logger.warning(f"Order too small: ${order_value:.2f}")
                return False

            logger.info(f"📤 Submitting order: {side.upper()} {qty} {symbol} @ ${price:.2f}")

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            order = self.trading_client.submit_order(order_data)
            logger.info(f"✅ ORDER EXECUTED | {side.upper()} {qty} {symbol} | Order ID: {order.id}")

            if side == 'buy':
                if not self.verify_position_exists(symbol):
                    logger.error(f"⚠️ Position for {symbol} did not appear after buy – marking cooldown")
                    self.cooldowns[symbol] = datetime.now() + timedelta(hours=1)
                    return False
            return True

        except Exception as e:
            logger.error(f"❌ Order failed for {symbol}: {e}")
            self.cooldowns[symbol] = datetime.now() + timedelta(hours=1)
            return False

    # ==========================================================================
    # MARKET DATA
    # ==========================================================================
    def get_current_price(self, symbol):
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            bars = self.data_client.get_crypto_bars(request)
            if bars and bars.data and symbol in bars.data:
                return bars.data[symbol][-1].close
        except Exception as e:
            logger.debug(f"Price fetch failed: {e}")
        return None

    async def fetch_crypto_data(self, symbol):
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(days=1),
                limit=100
            )
            bars = self.data_client.get_crypto_bars(request)

            if not bars or not bars.data or symbol not in bars.data:
                return None, None

            bars_list = bars.data[symbol]
            df = pd.DataFrame({
                'close': [bar.close for bar in bars_list],
                'volume': [bar.volume for bar in bars_list]
            })
            current_price = bars_list[-1].close
            return current_price, df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None, None

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================
    async def run(self):
        logger.info("=" * 60)
        logger.info("🚀 ALPACA CRYPTO HYBRID STRATEGY (with duplicate-buy protection)")
        logger.info(f"🎯 Buy: {self.buy_threshold} | Sell: {self.sell_threshold}")
        logger.info(f"💰 Position Size: ${self.position_usd} per trade")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        last_cycle = 0
        interval_seconds = self.interval_minutes * 60

        while self.running:
            try:
                now = time.time()
                if now - last_cycle >= interval_seconds:
                    last_cycle = now

                    if self.global_cooldown_until and datetime.now() < self.global_cooldown_until:
                        remaining = (self.global_cooldown_until - datetime.now()).total_seconds()
                        logger.info(f"🌍 Global cooldown: {remaining:.0f}s – skipping")
                        continue
                    elif self.global_cooldown_until:
                        self.global_cooldown_until = None
                        logger.info("🌍 Global cooldown expired")

                    for symbol in self.symbols:
                        try:
                            if symbol in self.cooldowns:
                                if datetime.now() < self.cooldowns[symbol]:
                                    remaining = (self.cooldowns[symbol] - datetime.now()).total_seconds()
                                    logger.info(f"⏳ {symbol} cooldown: {remaining:.0f}s – skipping")
                                    continue
                                else:
                                    del self.cooldowns[symbol]

                            price, df = await self.fetch_crypto_data(symbol)
                            if price is None or df is None:
                                continue

                            score = self.ml.predict(df)
                            live_qty = self.get_position_qty(symbol)

                            logger.info(f"📊 {symbol} | ${price:.2f} | Score: {score:.3f} | Live Qty: {live_qty:.8f} | Tracked: {symbol in self.positions}")

                            # --- SELL ---
                            if score < self.sell_threshold and live_qty > 0:
                                logger.info(f"🔴 SELL SIGNAL: {symbol} @ ${price:.2f}")
                                entry_price = self.positions[symbol]['price'] if symbol in self.positions else price
                                pnl_pct = ((price - entry_price) / entry_price) * 100
                                pnl_usd = ((price - entry_price) / entry_price) * self.position_usd
                                self.total_pnl += pnl_usd

                                logger.info(f"   PnL: {pnl_pct:.2f}% (${pnl_usd:.2f}) | Total: ${self.total_pnl:.2f}")
                                write_trade_to_csv(symbol, 'SELL', price, pnl_usd, self.total_pnl, score)

                                if self.trading_client:
                                    success = self.submit_crypto_order(symbol, self.position_usd, 'sell')
                                    if success:
                                        if symbol in self.positions:
                                            del self.positions[symbol]
                                        self.save_state()

                            # --- BUY ---
                            elif score > self.buy_threshold and live_qty == 0 and symbol not in self.positions:
                                logger.info(f"🟢 BUY SIGNAL: {symbol} @ ${price:.2f} (Score: {score:.3f})")
                                if self.trading_client:
                                    success = self.submit_crypto_order(symbol, self.position_usd, 'buy')
                                    if success:
                                        self.cooldowns[symbol] = datetime.now() + timedelta(hours=1)
                                        self.positions[symbol] = {
                                            'price': price,
                                            'entry_time': datetime.now().isoformat(),
                                            'entry_score': score
                                        }
                                        write_trade_to_csv(symbol, 'BUY', price, score=score)
                                        self.save_state()
                                    else:
                                        pass

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


if __name__ == "__main__":
    paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
    bot = AlpacaCryptoBot(paper_mode=paper_mode, interval_minutes=5)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Shutdown complete")
