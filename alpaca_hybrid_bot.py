import asyncio
import pandas as pd
import numpy as np
import logging
import json
import os
import csv
import time

from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CSV LOGGING
# ==============================================================================

def init_csv():
    if not os.path.exists("trades.csv"):
        with open("trades.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Symbol",
                "Side",
                "Price",
                "Qty",
                "PnL_USD",
                "Total_PnL_USD",
                "Score",
                "StopLoss",
                "TakeProfit"
            ])


def write_trade_to_csv(
    symbol,
    side,
    price,
    qty,
    pnl_usd=None,
    total_pnl=None,
    score=None,
    stop_loss=False,
    take_profit=False
):
    with open("trades.csv", "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            side,
            price,
            qty,
            pnl_usd if pnl_usd is not None else "",
            total_pnl if total_pnl is not None else "",
            score if score is not None else "",
            "YES" if stop_loss else "",
            "YES" if take_profit else ""
        ])

# ==============================================================================
# HYBRID PREDICTOR
# ==============================================================================

class HybridPredictor:

    def __init__(self):
        self.score_history = {}

    def predict(self, symbol, df):

        if df is None or len(df) < 50:
            return 0.5

        close = df["close"].values
        volume = df["volume"].values

        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])

        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0

        rsi = self.calculate_rsi(close)

        ema9 = self.calculate_ema(close, 9)
        ema21 = self.calculate_ema(close, 21)

        is_uptrend = ema9[-1] > ema21[-1]
        is_downtrend = ema9[-1] < ema21[-1]

        vol_avg = np.mean(volume[-10:])
        vol_surge = volume[-1] / vol_avg if vol_avg > 0 else 1

        score = 0.5

        # Mean reversion
        if z_score < -1.2:
            score += 0.35
        elif z_score < -0.8:
            score += 0.20
        elif z_score > 1.2:
            score -= 0.35
        elif z_score > 0.8:
            score -= 0.20

        # RSI
        if rsi < 35:
            score += 0.10
        elif rsi > 65:
            score -= 0.10

        # Trend confirmation
        if is_uptrend and score > 0.5:
            score += 0.05

        if is_downtrend and score < 0.5:
            score -= 0.05

        # Volume
        if vol_surge > 1.5:
            if score > 0.5:
                score += 0.05
            else:
                score -= 0.05

        score = max(0.0, min(1.0, score))

        if symbol not in self.score_history:
            self.score_history[symbol] = []

        self.score_history[symbol].append(score)

        if len(self.score_history[symbol]) > 5:
            self.score_history[symbol].pop(0)

        return np.mean(self.score_history[symbol])

    def calculate_rsi(self, prices, period=14):

        if len(prices) < period + 1:
            return 50

        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss

        return 100 - (100 / (1 + rs))

    def calculate_ema(self, prices, period):

        return (
            pd.Series(prices)
            .ewm(span=period, adjust=False)
            .mean()
            .values
        )

# ==============================================================================
# MAIN BOT
# ==============================================================================

class AlpacaCryptoBot:

    def __init__(self):

        self.paper_mode = True

        self.interval_minutes = 5

        # STRATEGY

        self.buy_threshold = 0.58
        self.sell_threshold = 0.42

        self.position_usd = 15

        # RISK

        self.stop_loss_pct = -0.05
        self.take_profit_pct = 0.08

        self.daily_loss_limit = -30.0

        self.max_daily_trades = 20

        self.max_unrealized_drawdown = -50.0

        self.min_buying_power_reserve = 20

        # API

        self.api_key = os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY", "")

        self.trading_client = None

        self.data_client = CryptoHistoricalDataClient()

        if self.api_key and self.secret_key:

            self.trading_client = TradingClient(
                self.api_key,
                self.secret_key,
                paper=self.paper_mode
            )

            logger.info("Connected to Alpaca")

        else:
            logger.warning("No API keys found")

        # SYMBOLS

        self.symbols = [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
            "LTC/USD"
        ]

        self.ml = HybridPredictor()

        self.positions = {}

        self.cooldowns = {}

        self.total_pnl = 0.0

        self.daily_pnl = 0.0

        self.daily_trade_count = 0

        self.current_day = datetime.now().date()

        self.running = True

        init_csv()

        self.load_state()

    # ==========================================================================
    # STATE
    # ==========================================================================

    def save_state(self):

        cooldowns = {
            s: dt.isoformat()
            for s, dt in self.cooldowns.items()
        }

        with open("alpaca_crypto_state.json", "w") as f:

            json.dump({
                "positions": self.positions,
                "cooldowns": cooldowns,
                "total_pnl": self.total_pnl,
                "daily_pnl": self.daily_pnl,
                "daily_trade_count": self.daily_trade_count,
                "current_day": self.current_day.isoformat()
            }, f)

    def load_state(self):

        if not os.path.exists("alpaca_crypto_state.json"):
            return

        try:

            with open("alpaca_crypto_state.json", "r") as f:

                data = json.load(f)

            self.positions = data.get("positions", {})

            self.total_pnl = data.get("total_pnl", 0.0)

            self.daily_pnl = data.get("daily_pnl", 0.0)

            self.daily_trade_count = data.get("daily_trade_count", 0)

            self.current_day = datetime.fromisoformat(
                data.get("current_day")
            ).date()

            raw = data.get("cooldowns", {})

            now = datetime.now()

            self.cooldowns = {
                s: datetime.fromisoformat(v)
                for s, v in raw.items()
                if datetime.fromisoformat(v) > now
            }

        except Exception as e:
            logger.error(f"Load state failed: {e}")

    # ==========================================================================
    # ACCOUNT
    # ==========================================================================

    async def get_positions_cache(self):

        try:
            positions = self.trading_client.get_all_positions()

            cache = {}

            for p in positions:

                cache[p.symbol.replace("/", "")] = {
                    "qty": float(p.qty),
                    "avg_price": float(p.avg_entry_price)
                }

            return cache

        except Exception as e:

            logger.error(f"Position cache error: {e}")

            return {}

    def normalize_symbol(self, symbol):

        return symbol.replace("/", "").replace("-", "")

    # ==========================================================================
    # DATA
    # ==========================================================================

    async def fetch_crypto_data(self, symbol):

        try:

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=100
            )

            bars = self.data_client.get_crypto_bars(request)

            if symbol not in bars.data:
                return None, None

            bars_list = bars.data[symbol]

            df = pd.DataFrame({
                "close": [b.close for b in bars_list],
                "volume": [b.volume for b in bars_list]
            })

            return bars_list[-1].close, df

        except Exception as e:

            logger.error(f"{symbol} fetch failed: {e}")

            return None, None

    # ==========================================================================
    # ORDERS
    # ==========================================================================

    async def submit_order(self, symbol, side, usd_amount=None):

        try:
            price, _ = await self.fetch_crypto_data(symbol)

            if not price:
                return False, 0, 0

            if side == "buy":
                # Down-rounding buffer prevents standard round() from buying slightly over budget
                qty = round((usd_amount / price) - 0.0000005, 6)
            else:
                positions = await self.get_positions_cache()
                norm = self.normalize_symbol(symbol)

                if norm not in positions:
                    logger.error(f"No position to sell: {symbol}")
                    return False, 0, 0

                # CRITICAL PRECISION FIX: Force downward truncation down to 6 decimals.
                # Subtracting a sub-satoshi fraction forces standard round() to floor rather than ceil.
                raw_qty = positions[norm]["qty"]
                qty = round(raw_qty - 0.0000005, 6)
                
                # Backup safety: if down-rounding zeros it out entirely, preserve raw balance
                if qty <= 0:
                    qty = raw_qty

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )

            self.trading_client.submit_order(order)
            logger.info(f"{side.upper()} order successfully sent for {qty} {symbol}")

            return True, qty, price

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return False, 0, 0

    # ==========================================================================
    # RISK
    # ==========================================================================

    async def calculate_unrealized_pnl(self):

        total = 0.0

        positions = await self.get_positions_cache()

        for symbol in self.symbols:

            norm = self.normalize_symbol(symbol)

            if norm not in positions:
                continue

            current_price, _ = await self.fetch_crypto_data(symbol)

            if not current_price:
                continue

            entry = positions[norm]["avg_price"]

            qty = positions[norm]["qty"]

            pnl = (current_price - entry) * qty

            total += pnl

        return total

    async def check_risk_limits(self):

        unrealized = await self.calculate_unrealized_pnl()

        combined = self.daily_pnl + unrealized

        if combined <= self.max_unrealized_drawdown:

            logger.error(
                f"MAX DRAWDOWN HIT: ${combined:.2f}"
            )

            self.running = False

            return False

        if self.daily_pnl <= self.daily_loss_limit:

            logger.error(
                f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}"
            )

            self.running = False

            return False

        return True

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    async def run(self):

        logger.info("Bot started")

        last_cycle = 0

        interval_seconds = self.interval_minutes * 60

        while self.running:

            try:

                if not await self.check_risk_limits():
                    break

                now = time.time()

                if now - last_cycle >= interval_seconds:

                    last_cycle = now

                    positions_cache = await self.get_positions_cache()

                    for symbol in self.symbols:

                        try:

                            # cooldown

                            if symbol in self.cooldowns:

                                if datetime.now() < self.cooldowns[symbol]:
                                    continue

                                del self.cooldowns[symbol]

                            # market data

                            price, df = await self.fetch_crypto_data(symbol)

                            if price is None:
                                continue

                            # ML score

                            score = self.ml.predict(symbol, df)

                            norm = self.normalize_symbol(symbol)

                            has_position = norm in positions_cache

                            # =========================
                            # SELL LOGIC
                            # =========================

                            if has_position:

                                qty = positions_cache[norm]["qty"]

                                entry_price = positions_cache[norm]["avg_price"]

                                pnl_pct = (
                                    (price - entry_price)
                                    / entry_price
                                )

                                pnl_usd = (
                                    (price - entry_price)
                                    * qty
                                )

                                stop_loss_hit = (
                                    pnl_pct <= self.stop_loss_pct
                                )

                                take_profit_hit = (
                                    pnl_pct >= self.take_profit_pct
                                )

                                sell_signal = (
                                    score < self.sell_threshold
                                )

                                if (
                                    stop_loss_hit
                                    or take_profit_hit
                                    or sell_signal
                                ):

                                    success, fill_qty, fill_price = (
                                        await self.submit_order(
                                            symbol,
                                            "sell"
                                        )
                                    )

                                    if success:

                                        self.total_pnl += pnl_usd

                                        self.daily_pnl += pnl_usd

                                        self.daily_trade_count += 1

                                        logger.info(
                                            f"SELL {symbol} | "
                                            f"PnL ${pnl_usd:.2f}"
                                        )

                                        write_trade_to_csv(
                                            symbol,
                                            "SELL",
                                            fill_price,
                                            fill_qty,
                                            pnl_usd,
                                            self.total_pnl,
                                            score,
                                            stop_loss_hit,
                                            take_profit_hit
                                        )

                                        # Clear state memory tracking to avoid bloating tracking dictionary
                                        if symbol in self.positions:
                                            del self.positions[symbol]

                                        self.cooldowns[symbol] = (
                                            datetime.now()
                                            + timedelta(hours=1)
                                        )

                            # =========================
                            # BUY LOGIC
                            # =========================

                            else:

                                if (
                                    score > self.buy_threshold
                                    and self.daily_trade_count
                                    < self.max_daily_trades
                                ):

                                    success, fill_qty, fill_price = (
                                        await self.submit_order(
                                            symbol,
                                            "buy",
                                            self.position_usd
                                        )
                                    )

                                    if success:

                                        logger.info(
                                            f"BUY {symbol} "
                                            f"{fill_qty} @ ${fill_price:.2f}"
                                        )

                                        write_trade_to_csv(
                                            symbol,
                                            "BUY",
                                            fill_price,
                                            fill_qty,
                                            score=score
                                        )

                                        self.daily_trade_count += 1

                                        self.cooldowns[symbol] = (
                                            datetime.now()
                                            + timedelta(hours=1)
                                        )

                                        self.positions[symbol] = {
                                            "entry_time":
                                            datetime.now().isoformat(),

                                            "reconcile_after":
                                            (
                                                datetime.now()
                                                + timedelta(minutes=2)
                                            ).isoformat()
                                        }

                            self.save_state()

                        except Exception as e:

                            logger.error(
                                f"{symbol} processing failed: {e}"
                            )

                await asyncio.sleep(1)

            except Exception as e:

                logger.error(f"Main loop error: {e}")

                await asyncio.sleep(5)

    # ==========================================================================
    # STOP
    # ==========================================================================

    def stop(self):

        self.running = False

        self.save_state()

        logger.info(
            f"Stopped | Total PnL: ${self.total_pnl:.2f}"
        )

# ==============================================================================
# ENTRY
# ==============================================================================

if __name__ == "__main__":

    bot = AlpacaCryptoBot()

    try:
        asyncio.run(bot.run())

    except KeyboardInterrupt:
        bot.stop()


