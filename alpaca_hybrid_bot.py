# region imports
from collections import defaultdict
from typing import Dict, List, Any, Tuple
# endregion


Crypto mean-reversion bot — v3: extreme selectivity.
    Only trades panic dips in strong uptrends.

    New: 20% baseline BTC hold to capture market beta during cash periods.
    """

    def initialize(self) -> None:
        self.set_start_date(2023, 1, 1)
        self.set_account_currency("USD", 100_000)
        self.set_brokerage_model(BrokerageName.BITFINEX, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.02

        # Tunables
        self._buy_threshold      = 0.80
        self._min_hold_bars      = 24
        self._max_hold_bars      = 168
        self._atr_stop_mult      = 1.5
        self._atr_target_mult    = 8.0
        self._daily_loss_limit   = -5_000.0
        self._position_pct       = 0.40
        self._cooldown_hours     = 48
        self._min_atr_pct        = 0.015
        self._baseline_pct       = 0.30

        # Assets
        self._symbols_raw = ["BTCUSD", "ETHUSD", "SOLUSD"]
        self._symbols: List[Symbol] = []
        for raw in self._symbols_raw:
            sec = self.add_crypto(raw, Resolution.HOUR)
            self._symbols.append(sec.symbol)

        self._btc_sym = self._symbols[0]
        self.set_benchmark(self._btc_sym)

        # Warm-up windows
        self._warmup_period = 60
        self._prices:  Dict[Symbol, RollingWindow] = {}
        self._highs:   Dict[Symbol, RollingWindow] = {}
        self._lows:    Dict[Symbol, RollingWindow] = {}
        self._volumes: Dict[Symbol, RollingWindow] = {}
        for sym in self._symbols:
            self._prices[sym]  = RollingWindow(self._warmup_period)
            self._highs[sym]   = RollingWindow(self._warmup_period)
            self._lows[sym]    = RollingWindow(self._warmup_period)
            self._volumes[sym] = RollingWindow(self._warmup_period)

        # State — only one mean-reversion position at a time
        self._position: Dict[str, Any] = {}
        self._global_cooldown: datetime = datetime.min
        self._daily_pnl = 0.0
        self._current_day = self.time.date()

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(TimeSpan.from_hours(1)),
            self._rebalance
        )

        self.set_warm_up(self._warmup_period, Resolution.HOUR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_daily_counters(self) -> None:
        today = self.time.date()
        if today != self._current_day:
            self._daily_pnl = 0.0
            self._current_day = today
            self.log(f"New trading day: {today}")

    def _in_global_cooldown(self) -> bool:
        return self.time < self._global_cooldown

    def _calc_rsi(self, prices: list, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calc_ema(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * alpha + ema * (1 - alpha)
        return ema

    def _calc_sma(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return float(np.mean(prices[-period:]))

    def _calc_atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return closes[-1] * 0.01 if closes else 1.0
        high_arr  = np.array(highs[-(period + 1):])
        low_arr   = np.array(lows[-(period + 1):])
        close_arr = np.array(closes[-(period + 1):])
        tr = np.maximum(
            high_arr[1:] - low_arr[1:],
            np.maximum(
                np.abs(high_arr[1:] - close_arr[:-1]),
                np.abs(low_arr[1:] - close_arr[:-1])
            )
        )
        return float(np.mean(tr))

    def _compute_score(self, sym: Symbol) -> float:
        prices  = list(self._prices[sym])
        volumes = list(self._volumes[sym])
        if len(prices) < 50 or len(volumes) < 10:
            return 0.0

        close = np.array(prices)
        ma20  = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0.0

        rsi_val = self._calc_rsi(prices)
        ema9    = self._calc_ema(prices, 9)
        ema21   = self._calc_ema(prices, 21)
        is_uptrend = ema9 > ema21 and close[-1] > ema9

        vol_arr = np.array(volumes)
        vol_avg = np.mean(vol_arr[-10:]) if len(vol_arr) >= 10 else 1.0
        vol_surge = vol_arr[-1] / vol_avg if vol_avg > 0 else 1.0

        score = 0.5

        if is_uptrend:
            if z_score < -2.0:
                score += 0.30
            elif z_score < -1.5:
                score += 0.22
            elif z_score < -1.0:
                score += 0.15
            elif z_score < -0.5:
                score += 0.08
            elif z_score > 1.5:
                score -= 0.25
            elif z_score > 1.0:
                score -= 0.15
            elif z_score > 0.5:
                score -= 0.08
            score += 0.08
        else:
            if z_score < -2.0:
                score += 0.20
            elif z_score < -1.5:
                score += 0.12
            elif z_score > 1.0:
                score -= 0.25
            elif z_score > 0.5:
                score -= 0.15

        if rsi_val < 30:
            score += 0.12
        elif rsi_val < 40:
            score += 0.06
        elif rsi_val > 70:
            score -= 0.12
        elif rsi_val > 60:
            score -= 0.06

        if vol_surge > 1.5:
            score += 0.03

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def on_data(self, data: Slice) -> None:
        for sym in self._symbols:
            bar = data.bars.get(sym)
            if bar is None:
                continue
            self._prices[sym].add(bar.close)
            self._highs[sym].add(bar.high)
            self._lows[sym].add(bar.low)
            self._volumes[sym].add(bar.volume)

    # ------------------------------------------------------------------
    # Rebalance logic (called every hour)
    # ------------------------------------------------------------------

    def _rebalance(self) -> None:
        self._reset_daily_counters()

        # Maintain baseline BTC hold
        self._maintain_baseline()

        if self._daily_pnl <= self._daily_loss_limit:
            self.log("Daily loss limit hit.")
            return

        # If in a mean-reversion position, manage it
        if self._position:
            self._manage_position()
            return

        # Not in a mean-reversion position — look for best entry
        if self._in_global_cooldown():
            return

        # Reserve baseline cash + buffer for the trade
        available_cash = self.portfolio.cash
        baseline_notional = self.portfolio.total_portfolio_value * self._baseline_pct
        baseline_value = self.portfolio[self._btc_sym].holdings_value
        cash_needed_for_baseline = max(0, baseline_notional - baseline_value)
        trade_reserve = self.portfolio.total_portfolio_value * self._position_pct * 1.05
        if available_cash < cash_needed_for_baseline + trade_reserve:
            return

        # Score all symbols and pick the best
        candidates: List[Tuple[Symbol, float, float]] = []
        for sym in self._symbols:
            price = self.securities[sym].price
            if price == 0:
                continue
            score = self._compute_score(sym)
            candidates.append((sym, score, price))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sym, best_score, best_price = candidates[0]

        if best_score <= self._buy_threshold:
            return

        # Regime filter: price must be above 50-bar SMA
        prices = list(self._prices[best_sym])
        if len(prices) < 50:
            return
        sma50 = self._calc_sma(prices, 50)
        if best_price <= sma50:
            return

        highs  = list(self._highs[best_sym])
        lows   = list(self._lows[best_sym])
        closes = list(self._prices[best_sym])
        atr = self._calc_atr(highs, lows, closes, 14)

        # Volatility filter
        if atr < best_price * self._min_atr_pct:
            self.log(f"Reject {best_sym.value} — ATR too low")
            return

        stop_price   = best_price - atr * self._atr_stop_mult
        target_price = best_price + atr * self._atr_target_mult

        notional = self.portfolio.total_portfolio_value * self._position_pct
        qty = notional / best_price
        qty = round(qty, 6)
        if qty <= 0:
            return

        self.log(
            f"BUY {best_sym.value} @ {best_price:.2f} | score={best_score:.3f} "
            f"SL={stop_price:.2f} TP={target_price:.2f}"
        )

        ticket = self.market_order(best_sym, qty)
        if ticket.status == OrderStatus.FILLED or ticket.status == OrderStatus.PARTIALLY_FILLED:
            fill_price = ticket.average_fill_price if ticket.average_fill_price != 0 else best_price
            self._position = {
                "symbol":       best_sym,
                "entry_time":   self.time,
                "entry_price":  fill_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "qty":          qty,
                "bars_held":    0,
            }

    def _maintain_baseline(self) -> None:
        """Keep a 20% BTC baseline position, rebalanced once per day."""
        # Only check baseline at midnight UTC to avoid hourly churn
        if self.time.hour != 0:
            return

        target_notional = self.portfolio.total_portfolio_value * self._baseline_pct
        current_value = self.portfolio[self._btc_sym].holdings_value
        price = self.securities[self._btc_sym].price
        if price == 0:
            return

        diff_notional = target_notional - current_value
        # Only act if deviation is > 2% of portfolio
        if abs(diff_notional) < self.portfolio.total_portfolio_value * 0.02:
            return

        qty = diff_notional / price
        qty = round(qty, 6)
        if qty == 0:
            return

        action = "BUY" if qty > 0 else "SELL"
        self.log(f"BASELINE {action} BTC {abs(qty):.6f} @ {price:.2f}")
        self.market_order(self._btc_sym, qty)

    def _manage_position(self) -> None:
        pos = self._position
        sym = pos["symbol"]
        price = self.securities[sym].price
        if price == 0:
            return

        pos["bars_held"] += 1
        entry_price  = pos["entry_price"]
        stop_price   = pos["stop_price"]
        target_price = pos["target_price"]
        bars_held    = pos["bars_held"]
        qty          = pos["qty"]

        exit_reason = None

        if price <= stop_price:
            exit_reason = "STOP_LOSS"
        elif price >= target_price:
            exit_reason = "TARGET_HIT"
        elif bars_held >= self._max_hold_bars:
            exit_reason = "MAX_HOLD"

        if exit_reason:
            self.log(f"EXIT {sym.value} — {exit_reason} @ {price:.2f}")
            self.liquidate(sym)
            pnl = (price - entry_price) * qty
            self._daily_pnl += pnl
            self._position = {}
            self._global_cooldown = self.time + timedelta(hours=self._cooldown_hours)
            self.log(f"Global cooldown {self._cooldown_hours}h ({exit_reason})")

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status == OrderStatus.FILLED:
            self.log(
                f"FILL {order_event.symbol.value} "
                f"{order_event.direction} {order_event.fill_quantity} @ {order_event.fill_price}"
            )
