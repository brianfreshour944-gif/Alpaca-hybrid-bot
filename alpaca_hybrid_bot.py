#!/usr/bin/env python3
# crypto_mean_reversion_alpaca.py

import asyncio
import os
import csv
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG  — Mirror of QC Tunables (Corrected for 5-Minute Timeframe Consistency)
# ==============================================================================

BUY_THRESHOLD    = 0.80
MIN_HOLD_BARS    = 24    # 2 hours on a 5-min chart
MAX_HOLD_BARS    = 168   # 14 hours on a 5-min chart
ATR_STOP_MULT    = 1.5
ATR_TARGET_MULT  = 8.0
DAILY_LOSS_LIMIT = -5_000.0
POSITION_PCT     = 0.40
COOLDOWN_LOSS_H  = 48
COOLDOWN_WIN_H   = 4
MIN_ATR_PCT      = 0.015
BASELINE_PCT     = 0.30

SYMBOLS     = ['BTC/USD', 'ETH/USD', 'SOL/USD']
BTC_SYMBOL  = 'BTC/USD'
WARMUP_BARS = 60
CYCLE_SECS  = 300  # 5-minute execution loop

# ==============================================================================
# CSV ENGINE
# ==============================================================================

def init_csv():
    if not os.path.exists('trades.csv'):
        with open('trades.csv', 'w', newline='') as f:
            csv.writer(f).writerow([
                'Timestamp', 'Symbol', 'Side', 'FillPrice', 'Qty',
                'PnL_USD', 'Total_PnL_USD', 'Score', 'ExitReason',
                'StopPrice', 'TargetPrice'
            ])

def write_trade(symbol, side, fill_price, qty=None, pnl=None, total_pnl=None,
                score=None, exit_reason=None, stop_price=None, target_price=None):
    with open('trades.csv', 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol, side, f'{fill_price:.4f}',
            qty          if qty          is not None else '',
            f'{pnl:.4f}'       if pnl          is not None else '',
            f'{total_pnl:.4f}' if total_pnl    is not None else '',
            f'{score:.4f}'     if score         is not None else '',
            exit_reason  if exit_reason  is not None else '',
            f'{stop_price:.4f}'   if stop_price   is not None else '',
            f'{target_price:.4f}' if target_price is not None else '',
        ])

# ==============================================================================
# INDICATORS
# ==============================================================================

def calc_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas   = np.diff(prices[-(period + 1):])
    gains    = deltas[deltas > 0]
    losses   = -deltas[deltas < 0]
    avg_gain = float(np.mean(gains))  if len(gains)  > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

def calc_ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    alpha = 2.0 / (period + 1)
    ema   = prices[0]
    for p in prices[1:]:
        ema = p * alpha + ema * (1 - alpha)
    return ema

def calc_sma(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    return float(np.mean(prices[-period:]))

def calc_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    if len(highs) < period + 1:
        return closes[-1] * 0.01 if closes else 1.0
    h = np.array(highs[-(period + 1):])
    l = np.array(lows[-(period + 1):])
    c = np.array(closes[-(period + 1):])
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr))

def compute_score(prices: list, volumes: list) -> float:
    if len(prices) < 50 or len(volumes) < 10:
        return 0.0

    close   = np.array(prices)
    ma20    = np.mean(close[-20:])
    std20   = np.std(close[-20:])
    z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0.0

    rsi_val    = calc_rsi(prices)
    ema9       = calc_ema(prices, 9)
    ema21      = calc_ema(prices, 21)
    is_uptrend = ema9 > ema21 and close[-1] > ema9

    vol_arr   = np.array(volumes[-10:])
    vol_avg   = float(np.mean(vol_arr))
    vol_surge = float(volumes[-1]) / vol_avg if vol_avg > 0 else 1.0

    score = 0.5

    if is_uptrend:
        if z_score < -2.0:   score += 0.30
        elif z_score < -1.5: score += 0.22
        elif z_score < -1.0: score += 0.15
        elif z_score < -0.5: score += 0.08
        elif z_score > 1.5:  score -= 0.25
        elif z_score > 1.0:  score -= 0.15
        elif z_score > 0.5:  score -= 0.08
        score += 0.08
    else:
        if z_score < -2.0:   score += 0.20
        elif z_score < -1.5: score += 0.12
        elif z_score > 1.0:  score -= 0.25
        elif z_score > 0.5:  score -= 0.15

    if rsi_val < 30:   score += 0.12
    elif rsi_val < 40: score += 0.06
    elif rsi_val > 70: score -= 0.12
    elif rsi_val > 60: score -= 0.06

    if vol_surge > 1.5:
        score += 0.03

    return max(0.0, min(1.0, score))

# ==============================================================================
# BOT ENGINE
# ==============================================================================

class MeanReversionBot:

    def __init__(self):
        self.api_key    = os.getenv('APCA_API_KEY_ID', '')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY', '')
        if not self.api_key or not self.secret_key:
            raise RuntimeError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.')

        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data    = CryptoHistoricalDataClient()

        self.closes  = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.highs   = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.lows    = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.volumes = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}

        self.position: dict         = {}
        self.global_cooldown        = datetime.min.replace(tzinfo=timezone.utc)
        self.daily_pnl              = 0.0
        self.total_pnl              = 0.0
        self.current_day            = datetime.now(timezone.utc).date()
        self.baseline_last_rebal    = None 

        init_csv()
        self.load_state()
        logger.info('MeanReversionBot ready (Fixed Timeframe Edition | paper=True)')

    def save_state(self):
        with open('bot_state.json', 'w') as f:
            json.dump({
                'position':       self.position,
                'global_cooldown': self.global_cooldown.isoformat(),
                'daily_pnl':      self.daily_pnl,
                'total_pnl':      self.total_pnl,
                'current_day':    self.current_day.isoformat(),
            }, f, indent=2)

    def load_state(self):
        if not os.path.exists('bot_state.json'):
            return
        try:
            with open('bot_state.json') as f:
                d = json.load(f)
            self.position      = d.get('position', {})
            self.daily_pnl     = d.get('daily_pnl', 0.0)
            self.total_pnl     = d.get('total_pnl', 0.0)
            self.current_day   = datetime.fromisoformat(d['current_day']).date()
            self.global_cooldown = datetime.fromisoformat(
                d.get('global_cooldown', datetime.min.replace(tzinfo=timezone.utc).isoformat())
            )
            if self.global_cooldown.tzinfo is None:
                self.global_cooldown = self.global_cooldown.replace(tzinfo=timezone.utc)
            logger.info(f'State loaded | total P&L ${self.total_pnl:.2f}')
        except Exception as e:
            logger.error(f'State load error: {e}')

    async def fetch_bars(self, symbol: str) -> bool:
        """Fetch last WARMUP_BARS 5-minute bars and update rolling windows."""
        try:
            loop = asyncio.get_running_loop()
            req  = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),  # FIXED: Unified timeframe tracking
                limit=WARMUP_BARS,
            )
            bars = await loop.run_in_executor(None, self.data.get_crypto_bars, req)
            if symbol not in bars.data:
                return False
            rows = bars.data[symbol]
            
            self.closes[symbol].clear()
            self.highs[symbol].clear()
            self.lows[symbol].clear()
            self.volumes[symbol].clear()
            for b in rows:
                self.closes[symbol].append(b.close)
                self.highs[symbol].append(b.high)
                self.lows[symbol].append(b.low)
                self.volumes[symbol].append(b.volume)
            return True
        except Exception as e:
            logger.error(f'fetch_bars {symbol}: {e}')
            return False

    def _norm(self, symbol: str) -> str:
        return symbol.replace('/', '').replace('-', '')

    async def get_account(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.trading.get_account)

    async def get_all_positions(self) -> dict:
        try:
            loop = asyncio.get_running_loop()
            pos  = await loop.run_in_executor(None, self.trading.get_all_positions)
            return {p.symbol: {'qty': float(p.qty), 'avg_price': float(p.avg_entry_price)}
                    for p in pos}
        except Exception as e:
            logger.error(f'get_all_positions: {e}')
            return {}

    async def submit_order(self, symbol: str, side: str, qty: float):
        try:
            loop = asyncio.get_running_loop()
            norm = self._norm(symbol)
            qty  = round(qty, 6)
            if qty <= 0:
                return False, 0.0, 0.0

            req = MarketOrderRequest(
                symbol=norm,
                qty=qty,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            order = await loop.run_in_executor(None, self.trading.submit_order, req)

            await asyncio.sleep(2)
            try:
                loop2    = asyncio.get_running_loop()
                filled   = await loop2.run_in_executor(
                    None, self.trading.get_order_by_id, str(order.id)
                )
                fill_price = float(filled.filled_avg_price) if filled.filled_avg_price else 0.0
                fill_qty   = float(filled.filled_qty)       if filled.filled_qty       else qty
            except Exception:
                fill_price = float(self.closes[symbol][-1]) if self.closes[symbol] else 0.0
                fill_qty   = qty

            logger.info(f'FILLED {side.upper()} {fill_qty} {symbol} @ ${fill_price:.4f}')
            return True, fill_qty, fill_price

        except Exception as e:
            logger.error(f'submit_order {symbol} {side}: {e}')
            return False, 0.0, 0.0

    async def maintain_baseline(self, portfolio_value: float, btc_price: float):
        today = datetime.now(timezone.utc).date()
        if self.baseline_last_rebal == today:
            return
        self.baseline_last_rebal = today

        if btc_price == 0:
            return

        target_notional  = portfolio_value * BASELINE_PCT
        positions_cache  = await self.get_all_positions()
        btc_norm         = self._norm(BTC_SYMBOL)
        current_value    = (positions_cache.get(btc_norm, {}).get('qty', 0.0) * btc_price)
        diff_notional    = target_notional - current_value

        if abs(diff_notional) < portfolio_value * 0.02:
            logger.info(f'Baseline BTC OK (${current_value:.0f} vs target ${target_notional:.0f})')
            return

        qty  = diff_notional / btc_price
        side = 'buy' if qty > 0 else 'sell'
        logger.info(f'BASELINE {side.upper()} BTC {abs(qty):.6f} @ ${btc_price:.2f}')
        await self.submit_order(BTC_SYMBOL, side, abs(qty))

    async def manage_position(self):
        if not self.position:
            return

        sym          = self.position['symbol']
        entry_price  = self.position['entry_price']
        stop_price   = self.position['stop_price']
        target_price = self.position['target_price']
        qty          = self.position['qty']
        bars_held    = self.position.get('bars_held', 0)

        price = float(self.closes[sym][-1]) if self.closes[sym] else 0.0
        if price == 0:
            return

        self.position['bars_held'] = bars_held + 1

        exit_reason = None
        if price <= stop_price:
            exit_reason = 'STOP_LOSS'
        elif price >= target_price:
            exit_reason = 'TARGET_HIT'
        elif bars_held >= MAX_HOLD_BARS:
            exit_reason = 'MAX_HOLD'

        pnl_pct = (price - entry_price) / entry_price * 100
        logger.info(
            f'Tracking {sym} | ${price:.4f} | P&L {pnl_pct:+.2f}% | '
            f'SL ${stop_price:.4f} | TP ${target_price:.4f} | Bars Held: {bars_held}'
        )

        if exit_reason:
            logger.info(f'EXIT {sym} — {exit_reason} @ ${price:.4f}')
            success, fill_qty, fill_price = await self.submit_order(sym, 'sell', qty)
            if success:
                pnl = (fill_price - entry_price) * fill_qty
                self.daily_pnl += pnl
                self.total_pnl += pnl
                write_trade(
                    sym, 'SELL', fill_price, fill_qty,
                    pnl, self.total_pnl,
                    exit_reason=exit_reason,
                    stop_price=stop_price,
                    target_price=target_price,
                )
                self.position = {}
                now = datetime.now(timezone.utc)
                if exit_reason == 'STOP_LOSS':
                    self.global_cooldown = now + timedelta(hours=COOLDOWN_LOSS_H)
                    logger.info(f'Cooldown {COOLDOWN_LOSS_H}h set (loss)')
                else:
                    self.global_cooldown = now + timedelta(hours=COOLDOWN_WIN_H)
                    logger.info(f'Short {COOLDOWN_WIN_H}h cooldown ({exit_reason})')

    async def look_for_entry(self, portfolio_value: float, cash: float, positions_cache: dict):
        now = datetime.now(timezone.utc)
        if now < self.global_cooldown:
            remaining = (self.global_cooldown - now).seconds // 60
            logger.info(f'Global cooldown active — {remaining}m remaining')
            return

        btc_norm        = self._norm(BTC_SYMBOL)
        btc_price       = float(self.closes[BTC_SYMBOL][-1]) if self.closes[BTC_SYMBOL] else 0
        baseline_val    = positions_cache.get(btc_norm, {}).get('qty', 0.0) * btc_price
        baseline_target = portfolio_value * BASELINE_PCT
        cash_for_baseline = max(0.0, baseline_target - baseline_val)
        trade_reserve   = portfolio_value * POSITION_PCT * 1.05
        if cash < cash_for_baseline + trade_reserve:
            logger.info(f'Insufficient cash for trade (need ${cash_for_baseline + trade_reserve:.0f}, have ${cash:.0f})')
            return

        candidates = []
        for sym in SYMBOLS:
            if not self.closes[sym] or len(self.closes[sym]) < 50:
                continue
            price  = float(self.closes[sym][-1])
            score  = compute_score(list(self.closes[sym]), list(self.volumes[sym]))
            candidates.append((sym, score, price))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sym, best_score, best_price = candidates[0]

        logger.info(f'Best candidate: {best_sym} score={best_score:.3f} price=${best_price:.4f}')

        if best_score <= BUY_THRESHOLD:
            logger.info(f'No entry — best score {best_score:.3f} below threshold {BUY_THRESHOLD}')
            return

        prices = list(self.closes[best_sym])
        sma50  = calc_sma(prices, 50)
        if best_price <= sma50:
            logger.info(f'Reject {best_sym} — below 50-bar SMA (${best_price:.2f} <= ${sma50:.2f})')
            return

        highs  = list(self.highs[best_sym])
        lows   = list(self.lows[best_sym])
        atr    = calc_atr(highs, lows, prices, 14)

        if atr < best_price * MIN_ATR_PCT:
            logger.info(f'Reject {best_sym} — ATR too low ({atr:.4f})')
            return

        stop_price   = best_price - atr * ATR_STOP_MULT
        target_price = best_price + atr * ATR_TARGET_MULT

        notional = portfolio_value * POSITION_PCT
        qty      = notional / best_price

        logger.info(
            f'BUY {best_sym} @ ${best_price:.4f} | score={best_score:.3f} | '
            f'SL=${stop_price:.4f} TP=${target_price:.4f} | qty={qty:.6f}'
        )

        success, fill_qty, fill_price = await self.submit_order(best_sym, 'buy', qty)
        if success:
            self.position = {
                'symbol':       best_sym,
                'entry_time':   datetime.now(timezone.utc).isoformat(),
                'entry_price':  fill_price,
                'stop_price':   stop_price,
                'target_price': target_price,
                'qty':          fill_qty,
                'bars_held':    0,
            }
            write_trade(
                best_sym, 'BUY', fill_price, fill_qty,
                score=best_score,
                stop_price=stop_price,
                target_price=target_price,
            )

    async def run(self):
        logger.info('=' * 60)
        logger.info('ALPACA MEAN REVERSION BOT — STARTED')
        logger.info(f'  Execution Mode : 5-Minute Bars Unified')
        logger.info(f'  Buy threshold  : {BUY_THRESHOLD}')
        logger.info(f'  Stop / Target  : ATR×{ATR_STOP_MULT} / ATR×{ATR_TARGET_MULT}')
        logger.info(f'  Position size  : {POSITION_PCT*100:.0f}% of portfolio')
        logger.info(f'  Baseline BTC   : {BASELINE_PCT*100:.0f}% of portfolio')
        logger.info('=' * 60)

        last_heartbeat = 0

        while True:
            try:
                now_t = time.time()
                if now_t - last_heartbeat >= 60:
                    logger.info(
                        f'[Heartbeat] Position: {self.position.get("symbol", "none")} | '
                        f'Daily P&L: ${self.daily_pnl:.2f} | Total P&L: ${self.total_pnl:.2f}'
                    )
                    last_heartbeat = now_t

                today = datetime.now(timezone.utc).date()
                if today != self.current_day:
                    logger.info(f'New day — resetting daily P&L (was ${self.daily_pnl:.2f})')
                    self.daily_pnl   = 0.0
                    self.current_day = today

                if self.daily_pnl <= DAILY_LOSS_LIMIT:
                    logger.error(f'Daily loss limit hit (${self.daily_pnl:.2f}) — pausing 1h')
                    await asyncio.sleep(3600)
                    continue

                # FIXED: Staggered sequential execution to prevent data request pooling timeouts
                results = []
                for s in SYMBOLS:
                    res = await self.fetch_bars(s)
                    results.append(res)
                    await asyncio.sleep(1.2)  # Stagger rate-limit window

                if not any(results):
                    logger.warning('All data fetches failed — skipping cycle')
                    await asyncio.sleep(10)
                    continue

                account = await self.get_account()
                portfolio_value = float(account.portfolio_value)
                cash            = float(account.cash)
                positions_cache = await self.get_all_positions()

                btc_price = float(self.closes[BTC_SYMBOL][-1]) if self.closes[BTC_SYMBOL] else 0.0

                await self.maintain_baseline(portfolio_value, btc_price)

                if self.position:
                    await self.manage_position()
                else:
                    await self.look_for_entry(portfolio_value, cash, positions_cache)

                self.save_state()
                await asyncio.sleep(CYCLE_SECS)

            except Exception as e:
                logger.error(f'Top-level error: {e}', exc_info=True)
                await asyncio.sleep(30)

    def stop(self):
        self.save_state()
        logger.info(f'Shutdown. Total P&L: ${self.total_pnl:.2f}')


if __name__ == '__main__':
    bot = MeanReversionBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
