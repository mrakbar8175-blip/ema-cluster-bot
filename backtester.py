#!/usr/bin/env python3
"""
Swing Sentinel – Historical Backtester (No Look‑Ahead Bias)
Dynamic end date – defaults to today.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math, sys, os, json, time

# Import bot’s configuration and logic
from swing_sentinel import (
    CONFIG, score_pair, to_yahoo, get_hybrid_klines,
    ema, atr, rsi, macd, adx, support_resistance_levels,
    FRACTIONS, TP_MULTIPLIERS, MIN_SCORE_ENTER, MAX_RISKY_TRADES,
    RISK_PER_TRADE_PCT, STOP_BOUNDS, ATR_MULT, DAILY_LOSS_LIMIT,
    BLACKLIST, COIN_RANK, get_current_stop
)

# ==================== BACKTEST CONFIGURATION ====================
BACKTEST_CONFIG = {
    "start_date": "2023-01-01",        # start of backtest window
    "end_date": "today",               # "today" = current date/time, or YYYY‑MM‑DD
    "initial_balance": 1000.0,
    "fee_pct": 0.1,                    # % per fill (e.g. 0.1% = 0.001)
    "slippage_pct": {                  # per order (decimal: 0.02 = 2%)
        "top_10": 0.02,
        "other": 0.05
    },
    "run_interval_hours": 4,           # how often the bot would check (4h)
    "output": {
        "equity_curve": "backtest_equity.csv",
        "trade_log": "backtest_trades.csv",
        "summary": "backtest_summary.txt"
    }
}
# ==============================================================

class BacktestEngine:
    def __init__(self, coins):
        self.coins = coins
        self.start = datetime.strptime(BACKTEST_CONFIG["start_date"], "%Y-%m-%d")
        # Handle dynamic end date
        end_str = BACKTEST_CONFIG["end_date"]
        if end_str.lower() == "today":
            self.end = datetime.now()
        else:
            self.end = datetime.strptime(end_str, "%Y-%m-%d")
        self.balance = BACKTEST_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []          # (datetime, balance)
        self.fee = BACKTEST_CONFIG["fee_pct"] / 100.0
        self.slippage = BACKTEST_CONFIG["slippage_pct"]
        self.interval_hours = BACKTEST_CONFIG["run_interval_hours"]
        # Pre‑compute date range
        self.dates = pd.date_range(self.start, self.end, freq=f'{self.interval_hours}h')

    def get_coin_rank(self, sym):
        return COIN_RANK.get(sym, 99)

    def get_slippage(self, sym):
        rank = self.get_coin_rank(sym)
        if rank <= 10:
            return self.slippage["top_10"] / 100.0
        else:
            return self.slippage["other"] / 100.0

    def apply_fee_and_slippage(self, price, side, sym):
        """Returns (filled_price, fee_cost_per_unit)."""
        slip = self.get_slippage(sym)
        if side == 'buy':
            filled_price = price * (1 + slip)
        else:
            filled_price = price * (1 - slip)
        fee_cost = filled_price * self.fee
        return filled_price, fee_cost

    # -----------------------------------------------------------------
    # Core historical data fetcher (no look‑ahead)
    # -----------------------------------------------------------------
    def get_historical_data(self, sym, interval, days, end_date):
        """Get data exactly as available up to end_date (exclusive of later)."""
        try:
            start_date = end_date - timedelta(days=days)
            df = yf.download(sym, start=start_date, end=end_date,
                             interval=interval, progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except:
            return None

    # -----------------------------------------------------------------
    # Scoring adapted for backtest (uses pre‑fetched data, no look‑ahead)
    # -----------------------------------------------------------------
    def score_pair_backtest(self, pair, end_date):
        """
        Replicates score_pair but using only data up to end_date.
        Returns the same tuple as score_pair: (score, direction, price, atr, swing_level, layers)
        """
        layers = {}
        # Daily data: need at least 200 days for EMA200
        df_d = self.get_historical_data(pair, '1d', 200, end_date)
        if df_d is None or len(df_d) < 50:
            return 0, None, None, None, None, {"Daily data": (0,0,"FAIL")}
        df_4h = self.get_historical_data(pair, '4h', 14, end_date)
        if df_4h is None or len(df_4h) < 50:
            return 0, None, None, None, None, {"4h data": (0,0,"FAIL")}
        df_1h = self.get_historical_data(pair, '1h', 3, end_date)
        if df_1h is None or len(df_1h) < 10:
            return 0, None, None, None, None, {"1h data": (0,0,"FAIL")}

        price = df_4h['Close'].iloc[-1]

        # Trend detection
        ema50_d = ema(df_d['Close'], 50)
        ema200_d = ema(df_d['Close'], 200)
        trend_daily = 0
        if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
            trend_daily = 1
        elif price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
            trend_daily = -1

        if trend_daily == 0:
            if len(df_4h) >= 200:
                ema50_4h = ema(df_4h['Close'], 50)
                ema200_4h = ema(df_4h['Close'], 200)
                if price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]:
                    trend_daily = 1
                elif price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]:
                    trend_daily = -1
                else:
                    return 0, None, None, None, None, {"Trend": (0,0,"FAIL")}
            else:
                ema20_4h = ema(df_4h['Close'], 20)
                ema50_4h = ema(df_4h['Close'], 50)
                if ema20_4h.iloc[-1] > ema50_4h.iloc[-1]:
                    trend_daily = 1
                elif ema20_4h.iloc[-1] < ema50_4h.iloc[-1]:
                    trend_daily = -1
                else:
                    return 0, None, None, None, None, {"Trend": (0,0,"FAIL")}

        direction = "LONG" if trend_daily == 1 else "SHORT"

        ema50_4h = ema(df_4h['Close'], 50)
        ema200_4h = ema(df_4h['Close'], 200) if len(df_4h) >= 200 else None
        adx_val, di_plus, di_minus = adx(df_4h)
        rsi_val = rsi(df_4h)
        macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
        atr_val = atr(df_4h)
        res, sup = support_resistance_levels(df_4h, 20)

        rsi_1h_val = rsi(df_1h, 14)
        last_candle = df_1h.iloc[-1]
        prev_candle = df_1h.iloc[-2]
        candle_range = last_candle['High'] - last_candle['Low']
        bullish_momentum = (last_candle['Close'] - last_candle['Open']) / candle_range if candle_range > 0 else 0

        vol_last = df_4h['Volume'].iloc[-1]
        vol_avg = df_4h['Volume'].iloc[-6:-1].mean() if len(df_4h) >= 6 else vol_last
        vol_surge = vol_last > vol_avg * 1.2 if vol_avg > 0 else False

        # BTC context (use cached data from the same end_date)
        btc_df = self.get_historical_data("BTC-USD", '4h', 14, end_date)
        market_aligned = False
        if btc_df is not None and len(btc_df) >= 50:
            btc_ema50 = ema(btc_df['Close'], 50)
            btc_trend_up = btc_df['Close'].iloc[-1] > btc_ema50.iloc[-1]
            if trend_daily == 1 and btc_trend_up:
                market_aligned = True
            elif trend_daily == -1 and not btc_trend_up:
                market_aligned = True
        else:
            layers["Market"] = (0, 0.5, "FAIL: BTC data unavailable")

        def bool_score(cond):
            return 1 if cond else 0

        # 11 layers (same as live bot)
        if ema200_4h is not None:
            if direction == "LONG":
                ema_align = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
            else:
                ema_align = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
        else:
            ema20_4h = ema(df_4h['Close'], 20)
            if direction == "LONG":
                ema_align = price > ema20_4h.iloc[-1] and ema20_4h.iloc[-1] > ema50_4h.iloc[-1]
            else:
                ema_align = price < ema20_4h.iloc[-1] and ema20_4h.iloc[-1] < ema50_4h.iloc[-1]
        layers["EMA Align"] = (bool_score(ema_align) * 1.5, 1.5, "OK")

        adx_trending = adx_val > 20
        adx_dir = (di_plus > di_minus) if direction == "LONG" else (di_minus > di_plus)
        layers["ADX"] = (bool_score(adx_trending and adx_dir) * 1.0, 1.0, "OK")

        if rsi_val is not None:
            layers["RSI"] = (bool_score((direction=="LONG" and rsi_val>50) or
                                        (direction=="SHORT" and rsi_val<50)) * 1.5, 1.5, "OK")
        else:
            layers["RSI"] = (0, 1.5, "FAIL: RSI NaN")

        macd_expanding = (direction=="LONG" and macd_hist>0 and macd_hist>macd_hist_prev) or \
                         (direction=="SHORT" and macd_hist<0 and macd_hist<macd_hist_prev)
        layers["MACD"] = (bool_score(macd_expanding) * 1.0, 1.0, "OK")

        if atr_val and atr_val>0:
            if direction=="LONG":
                sr_score = bool_score((price-sup) < atr_val*0.5)
            else:
                sr_score = bool_score((res-price) < atr_val*0.5)
            layers["S/R"] = (sr_score*1.0, 1.0, "OK")
        else:
            layers["S/R"] = (0, 1.0, "FAIL: ATR missing")

        layers["Volume"] = (bool_score(vol_surge)*0.5, 0.5, "OK")

        if "Market" not in layers:
            layers["Market"] = (bool_score(market_aligned)*0.5, 0.5, "OK")

        candle_ok = (bullish_momentum > 0.5) if direction=="LONG" else (bullish_momentum < -0.5)
        layers["Candle Mom"] = (bool_score(candle_ok)*2.0, 2.0, "OK")

        if rsi_1h_val is not None:
            rsi_1h_ok = (rsi_1h_val < 63) if direction=="LONG" else (rsi_1h_val > 37)
            layers["RSI 1h"] = (bool_score(rsi_1h_ok)*1.5, 1.5, "OK")
        else:
            layers["RSI 1h"] = (0, 1.5, "FAIL: RSI 1h NaN")

        if atr_val and price>0:
            layers["ATR"] = (bool_score(atr_val > price*0.005)*1.0, 1.0, "OK")
        else:
            layers["ATR"] = (0, 1.0, "FAIL: ATR missing")

        if direction=="LONG":
            micro_ok = last_candle['Close'] > last_candle['Open'] and prev_candle['Close'] > prev_candle['Open']
        else:
            micro_ok = last_candle['Close'] < last_candle['Open'] and prev_candle['Close'] < prev_candle['Open']
        layers["Micro Trend"] = (bool_score(micro_ok)*2.0, 2.0, "OK")

        total = sum(score for score,_,_ in layers.values() if isinstance(score,(int,float)))
        return total, direction, price, atr_val, (sup if direction=="LONG" else res), layers

    # -----------------------------------------------------------------
    # Signal generation at a given point in time
    # -----------------------------------------------------------------
    def generate_signal_at(self, date):
        """Generates a trade signal exactly as the live bot would, but at historical date."""
        open_symbols_risky = set()
        for t in self.open_trades:
            if not t.get("breakeven", False):
                open_symbols_risky.add(t["symbol"])

        if len(open_symbols_risky) >= MAX_RISKY_TRADES:
            return None

        all_scored = []
        for pair in self.coins:
            if pair in open_symbols_risky:
                continue
            score, direction, price, atr_val, swing_level, layers = self.score_pair_backtest(pair, date)
            if direction is None:
                continue
            all_scored.append((pair, score, direction, price, atr_val, swing_level, layers))

        if not all_scored:
            return None

        candidates = [x for x in all_scored if x[1] >= MIN_SCORE_ENTER]
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        pair, score, direction, price, atr_val, swing_level, layers = candidates[0]

        rank = self.get_coin_rank(pair)
        if rank <= 10:
            min_stop_pct = STOP_BOUNDS["top_10"]["min"]
            max_stop_pct = STOP_BOUNDS["top_10"]["max"]
        else:
            min_stop_pct = STOP_BOUNDS["other"]["min"]
            max_stop_pct = STOP_BOUNDS["other"]["max"]

        raw_stop = (atr_val * ATR_MULT) if (atr_val is not None and not math.isnan(atr_val)) else price * 0.02
        stop_distance = np.clip(raw_stop, price*min_stop_pct, price*max_stop_pct)
        if direction == "LONG":
            stop = price - stop_distance
            if swing_level and swing_level > price - stop_distance*1.2:
                stop = min(stop, swing_level - 0.05*(atr_val if atr_val else price*0.01))
        else:
            stop = price + stop_distance
            if swing_level and swing_level < price + stop_distance*1.2:
                stop = max(stop, swing_level + 0.05*(atr_val if atr_val else price*0.01))

        # Re‑clip after swing adjustment
        if direction == "LONG":
            stop = max(stop, price*(1 - max_stop_pct))
            stop = min(stop, price*(1 - min_stop_pct))
        else:
            stop = min(stop, price*(1 + max_stop_pct))
            stop = max(stop, price*(1 + min_stop_pct))

        risk = abs(price - stop)
        tps = [round(price + m*risk, 6) if direction=="LONG" else round(price - m*risk, 6) for m in TP_MULTIPLIERS]
        quantity = round((self.balance * RISK_PER_TRADE_PCT) / risk, 8)

        # Apply entry slippage
        filled_entry, fee_per_unit = self.apply_fee_and_slippage(price, 'buy' if direction=="LONG" else 'sell', pair)
        entry_fee = fee_per_unit * quantity
        self.balance -= entry_fee

        trade = {
            "action": direction,
            "symbol": pair,
            "quantity": quantity,
            "limit_price": filled_entry,
            "stop_loss": stop,
            "take_profits": tps,
            "score": score,
            "atr": atr_val,
            "layers": layers,
            "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_fee": entry_fee,
            "original_qty": quantity,
            "remaining_qty": quantity,
            "highest_tp": -1,
            "breakeven": False
        }
        # Deduct cost of the position from balance (cash account)
        # We assume we use the full balance to buy the asset.
        # However, for simplicity we'll just deduct the entry fee; the position value is not subtracted.
        # This is a simplification: in a real cash account, you'd subtract quantity * filled_entry.
        # For backtesting the 1% risk model, this doesn't affect the risk logic because we only risk 1%.
        # We'll keep this note and proceed.
        # (A more accurate simulation would track position value and margin.)
        return trade

    # -----------------------------------------------------------------
    # Simulate open trades between two dates using historical 1h bars
    # -----------------------------------------------------------------
    def simulate_trade_life(self, trade, current_date, next_date):
        sym = trade["symbol"]
        direction = trade["action"]
        entry = trade["limit_price"]
        stop_orig = trade["stop_loss"]
        tps = trade["take_profits"]
        remaining_qty = trade["remaining_qty"]
        original_qty = trade["original_qty"]
        highest_tp_idx = trade["highest_tp"]
        breakeven = trade["breakeven"]
        current_stop = get_current_stop(trade)

        # Get 1h data from current_date to next_date
        days_needed = max((next_date - current_date).days + 2, 5)  # enough bars
        df_1h = self.get_historical_data(sym, '1h', days_needed, next_date)
        if df_1h is None or df_1h.empty:
            return trade, False

        # Filter to only candles that start after the trade entry time
        df_1h = df_1h[df_1h.index >= current_date]
        if df_1h.empty:
            return trade, False

        closed_parts = []
        trade_closed = False

        for candle_time, candle in df_1h.iterrows():
            high = candle['High']; low = candle['Low']

            # Always check stop before TP in the same candle (as per live bot logic)
            sl_hit = (direction == "LONG" and low <= current_stop) or \
                     (direction == "SHORT" and high >= current_stop)

            if sl_hit:
                exit_price = current_stop
                slip, fee_per_unit = self.apply_fee_and_slippage(exit_price, 'sell' if direction=="LONG" else 'buy', sym)
                filled_exit = slip
                fee_cost = fee_per_unit * remaining_qty
                if direction == "LONG":
                    pnl = (filled_exit - entry) * remaining_qty - fee_cost
                else:
                    pnl = (entry - filled_exit) * remaining_qty - fee_cost
                closed_parts.append({
                    "exit_price": filled_exit,
                    "quantity": remaining_qty,
                    "pnl": pnl,
                    "hit_level": "STOP LOSS" if highest_tp_idx==-1 else f"STOP after TP{highest_tp_idx+1}"
                })
                remaining_qty = 0
                trade_closed = True
                break

            # Process TP levels (only if stop not hit)
            if direction == "LONG":
                new_tp_idx = None
                for i in range(len(tps)-1, -1, -1):
                    if high >= tps[i] and i > highest_tp_idx:
                        new_tp_idx = i
                        break
            else:
                new_tp_idx = None
                for i in range(len(tps)-1, -1, -1):
                    if low <= tps[i] and i > highest_tp_idx:
                        new_tp_idx = i
                        break

            if new_tp_idx is not None:
                for i in range(highest_tp_idx+1, new_tp_idx+1):
                    if remaining_qty <= 0:
                        break
                    fraction = FRACTIONS[i]
                    exit_qty = original_qty * fraction
                    if exit_qty > remaining_qty:
                        exit_qty = remaining_qty
                    if exit_qty > 0:
                        exit_price_tp = tps[i]
                        slip, fee_per_unit = self.apply_fee_and_slippage(exit_price_tp, 'sell' if direction=="LONG" else 'buy', sym)
                        filled_exit_tp = slip
                        fee_cost = fee_per_unit * exit_qty
                        if direction == "LONG":
                            pnl = (filled_exit_tp - entry) * exit_qty - fee_cost
                        else:
                            pnl = (entry - filled_exit_tp) * exit_qty - fee_cost
                        closed_parts.append({
                            "exit_price": filled_exit_tp,
                            "quantity": exit_qty,
                            "pnl": pnl,
                            "hit_level": f"TP{i+1}"
                        })
                        remaining_qty -= exit_qty
                        highest_tp_idx = i
                        if i == 0:
                            breakeven = True
                        # Update current_stop after each partial fill
                        # We can recalc from trade dict:
                        trade["highest_tp"] = highest_tp_idx
                        trade["breakeven"] = breakeven
                        current_stop = get_current_stop(trade)
                    if remaining_qty <= 0:
                        trade_closed = True
                        break
                # If trade closed, break the candle loop
                if trade_closed:
                    break

        # After processing all candles
        if remaining_qty > 0:
            # Trade still open, update its state
            trade["remaining_qty"] = remaining_qty
            trade["highest_tp"] = highest_tp_idx
            trade["breakeven"] = breakeven
            return trade, False
        else:
            # Trade fully closed, add P&L to balance
            total_pnl = sum(cp["pnl"] for cp in closed_parts)
            self.balance += total_pnl
            # Log the closed parts
            for cp in closed_parts:
                self.closed_trades.append({
                    "open_time": trade["timestamp"],
                    "close_time": candle_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
                    "action": direction,
                    "entry": entry,
                    "stop": stop_orig,
                    "take_profits": str(tps),
                    "exit_price": cp["exit_price"],
                    "quantity": cp["quantity"],
                    "pnl": cp["pnl"],
                    "hit_level": cp["hit_level"]
                })
            return None, True  # signal to remove trade

    # -----------------------------------------------------------------
    # Main backtest loop
    # -----------------------------------------------------------------
    def run(self):
        print(f"Backtesting {len(self.coins)} coins from {self.start.date()} to {self.end.date()}...")
        for i, current_date in enumerate(self.dates):
            # 1. Process open trades
            next_date = self.dates[i+1] if i+1 < len(self.dates) else current_date + timedelta(hours=self.interval_hours)
            new_open = []
            for trade in self.open_trades:
                updated_trade, closed = self.simulate_trade_life(trade, current_date, next_date)
                if not closed:
                    new_open.append(updated_trade)
            self.open_trades = new_open

            # 2. Record equity
            self.equity_curve.append((current_date, self.balance))

            # 3. Try to open a new trade
            signal = self.generate_signal_at(current_date)
            if signal is not None:
                # Deduct the cost of the position (entry fee already deducted; position value not deducted)
                # For a proper cash simulation, you'd deduct the asset cost here.
                # We'll assume the balance can go negative if needed, or we can simply ignore.
                # Since we only risk 1% of balance, we don't need to simulate the full position cost.
                self.open_trades.append(signal)

        # Save results
        self.save_results()

    def save_results(self):
        # Equity curve
        eq_df = pd.DataFrame(self.equity_curve, columns=["date", "balance"])
        eq_df.to_csv(BACKTEST_CONFIG["output"]["equity_curve"], index=False)
        # Closed trades
        if self.closed_trades:
            pd.DataFrame(self.closed_trades).to_csv(BACKTEST_CONFIG["output"]["trade_log"], index=False)
        # Summary
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        sharpe = None
        if len(eq_df) > 1:
            returns = eq_df["balance"].pct_change().dropna()
            if returns.std() != 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(365*24/self.interval_hours)
        summary = f"""Backtest completed!
Period: {self.start.date()} to {self.end.date()}
Initial balance: ${self.initial_balance:.2f}
Final balance:   ${self.balance:.2f}
Total return: {total_return:.2f}%
Sharpe ratio: {sharpe:.2f} (approx, assuming 24h markets)
Number of closed trades: {len(self.closed_trades)}
"""
        with open(BACKTEST_CONFIG["output"]["summary"], "w") as f:
            f.write(summary)
        print(summary)


# ==================== MAIN ====================
if __name__ == "__main__":
    # Use the live coin list from swing_sentinel
    from swing_sentinel import CRYPTO_PAIRS
    bt = BacktestEngine(CRYPTO_PAIRS)
    bt.run()