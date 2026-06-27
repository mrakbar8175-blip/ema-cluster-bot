#!/usr/bin/env python3
"""
KuCoin Historical Backtester – 4H & 1H data, full bot logic.
Usage: python backtest_ku.py FETCH    (download data)
       python backtest_ku.py BACKTEST (run simulation)
"""

import ccxt
import pandas as pd
import numpy as np
import os, sys, time, math, traceback
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION (adjust as needed)
# ============================================================
DAYS_HISTORY = 730        # 2 years of 4H data (3 years possible, just increase)
BACKTEST_START = "2025-01-01"   # start date of backtest
INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.01
MAX_RISKY_TRADES = 5
DATA_FOLDER = "kucoin_data"

# Top‑50 coins (same as live bot, minus blacklisted ones)
CRYPTO_PAIRS = [
    "BTC-USDT","ETH-USDT","BNB-USDT","SOL-USDT","XRP-USDT",
    "ADA-USDT","DOGE-USDT","DOT-USDT","MATIC-USDT","LINK-USDT",
    "UNI-USDT","AVAX-USDT","LTC-USDT","FIL-USDT","TRX-USDT",
    "ATOM-USDT","XLM-USDT","ETC-USDT","BCH-USDT","NEAR-USDT",
    "VET-USDT","ICP-USDT","HBAR-USDT","APT-USDT","ARB-USDT",
    "OP-USDT","GRT-USDT","THETA-USDT","ALGO-USDT","FTM-USDT",
    "EGLD-USDT","IMX-USDT","SAND-USDT","AXS-USDT","MANA-USDT",
    "AAVE-USDT","MKR-USDT","SNX-USDT","CRV-USDT","COMP-USDT",
    "ZEC-USDT","BAT-USDT","ENJ-USDT","CHZ-USDT","HOT-USDT",
    "KSM-USDT","DASH-USDT","CELO-USDT","QTUM-USDT","IOST-USDT"
]

# ============================================================
# TECHNICAL INDICATORS (exact copies from live bot)
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean().iloc[-1]
    return atr_val if not pd.isna(atr_val) else None

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
    return rsi_val if not pd.isna(rsi_val) else None

def macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return (macd_line.iloc[-1], signal.iloc[-1], histogram.iloc[-1],
            histogram.iloc[-2] if len(histogram) > 1 else 0)

def adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    dm_plus = h.diff()
    dm_minus = -l.diff()
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

def support_resistance_levels(df, lookback=20):
    recent = df.tail(lookback)
    return recent['High'].max(), recent['Low'].min()

# ============================================================
# SCORING (identical to live bot – 4H + daily + 1H)
# ============================================================
def score_pair(df_4h, df_d, df_1h, btc_df_4h=None):
    """
    Returns: (total_score, direction, price, atr_val, swing_level, layers_dict)
    layers_dict contains (earned_score, max_score, status)
    """
    layers = {}
    if df_d.empty or len(df_d) < 50:
        return 0, None, None, None, None, {"Daily data": (0, 0, "FAIL: insufficient daily candles")}
    if df_4h.empty or len(df_4h) < 50:
        return 0, None, None, None, None, {"4h data": (0, 0, "FAIL: insufficient 4h candles")}
    if df_1h.empty or len(df_1h) < 10:
        return 0, None, None, None, None, {"1h data": (0, 0, "FAIL: insufficient 1h candles")}

    price = df_4h['Close'].iloc[-1]

    # Daily trend
    ema50_d = ema(df_d['Close'], 50)
    ema200_d = ema(df_d['Close'], 200)
    trend_daily = 0
    if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
        trend_daily = 1
    elif price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
        trend_daily = -1

    # Fallback to 4h trend
    if trend_daily == 0:
        ema50_4h = ema(df_4h['Close'], 50)
        ema200_4h = ema(df_4h['Close'], 200)
        if price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]:
            trend_daily = 1
        elif price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]:
            trend_daily = -1
        else:
            return 0, None, None, None, None, {"Daily trend": (0, 0, "FAIL: no clear trend")}

    direction = "LONG" if trend_daily == 1 else "SHORT"

    # 4H indicators
    ema50_4h = ema(df_4h['Close'], 50)
    ema200_4h = ema(df_4h['Close'], 200)
    adx_val, di_plus, di_minus = adx(df_4h)
    rsi_val = rsi(df_4h)
    macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
    atr_val = atr(df_4h)
    res, sup = support_resistance_levels(df_4h, 20)

    # 1H momentum
    rsi_1h_val = rsi(df_1h, 14)
    last_candle_1h = df_1h.iloc[-1]
    prev_candle_1h = df_1h.iloc[-2]
    candle_range = last_candle_1h['High'] - last_candle_1h['Low']
    bullish_momentum = (last_candle_1h['Close'] - last_candle_1h['Open']) / candle_range if candle_range > 0 else 0

    vol_last = df_4h['Volume'].iloc[-1]
    vol_avg = df_4h['Volume'].iloc[-6:-1].mean() if len(df_4h) >= 6 else vol_last
    vol_surge = vol_last > vol_avg * 1.2 if vol_avg > 0 else False

    # BTC context (use provided btc_df or fallback to self)
    market_aligned = False
    if btc_df_4h is not None and len(btc_df_4h) >= 50:
        btc_ema50 = ema(btc_df_4h['Close'], 50)
        btc_trend_up = btc_df_4h['Close'].iloc[-1] > btc_ema50.iloc[-1]
        if trend_daily == 1 and btc_trend_up:
            market_aligned = True
        elif trend_daily == -1 and not btc_trend_up:
            market_aligned = True
    else:
        layers["Market"] = (0, 0.5, "FAIL: BTC data unavailable")

    def bool_score(cond):
        return 1 if cond else 0

    # Build layers
    if direction == "LONG":
        ema_align = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
    else:
        ema_align = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
    layers["EMA Align"] = (bool_score(ema_align) * 1.5, 1.5, "OK")
    adx_trending = adx_val > 20
    adx_dir = (di_plus > di_minus) if direction == "LONG" else (di_minus > di_plus)
    layers["ADX"] = (bool_score(adx_trending and adx_dir) * 1.0, 1.0, "OK")
    if rsi_val is not None:
        layers["RSI"] = (bool_score((direction == "LONG" and rsi_val > 50) or (direction == "SHORT" and rsi_val < 50)) * 1.5, 1.5, "OK")
    else:
        layers["RSI"] = (0, 1.5, "FAIL: RSI NaN")
    macd_expanding = (direction == "LONG" and macd_hist > 0 and macd_hist > macd_hist_prev) or \
                     (direction == "SHORT" and macd_hist < 0 and macd_hist < macd_hist_prev)
    layers["MACD"] = (bool_score(macd_expanding) * 1.0, 1.0, "OK")
    if atr_val and atr_val > 0:
        if direction == "LONG":
            sr_score = bool_score((price - sup) < atr_val * 0.5)
        else:
            sr_score = bool_score((res - price) < atr_val * 0.5)
        layers["S/R"] = (sr_score * 1.0, 1.0, "OK")
    else:
        layers["S/R"] = (0, 1.0, "FAIL: ATR missing")
    layers["Volume"] = (bool_score(vol_surge) * 0.5, 0.5, "OK")
    if "Market" not in layers:
        layers["Market"] = (bool_score(market_aligned) * 0.5, 0.5, "OK")
    candle_ok = (bullish_momentum > 0.5) if direction == "LONG" else (bullish_momentum < -0.5)
    layers["Candle Mom"] = (bool_score(candle_ok) * 2.0, 2.0, "OK")
    if rsi_1h_val is not None:
        rsi_1h_ok = (rsi_1h_val < 63) if direction == "LONG" else (rsi_1h_val > 37)
        layers["RSI 1h"] = (bool_score(rsi_1h_ok) * 1.5, 1.5, "OK")
    else:
        layers["RSI 1h"] = (0, 1.5, "FAIL: RSI 1h NaN")
    if atr_val and price > 0:
        layers["ATR"] = (bool_score(atr_val > price * 0.005) * 1.0, 1.0, "OK")
    else:
        layers["ATR"] = (0, 1.0, "FAIL: ATR missing")
    if direction == "LONG":
        micro_ok = last_candle_1h['Close'] > last_candle_1h['Open'] and prev_candle_1h['Close'] > prev_candle_1h['Open']
    else:
        micro_ok = last_candle_1h['Close'] < last_candle_1h['Open'] and prev_candle_1h['Close'] < prev_candle_1h['Open']
    layers["Micro Trend"] = (bool_score(micro_ok) * 2.0, 2.0, "OK")
    total = sum(score for score, _, _ in layers.values() if isinstance(score, (int, float)))
    return total, direction, price, atr_val, (sup if direction == "LONG" else res), layers

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_and_save(symbol_ccxt, timeframe, days_back=DAYS_HISTORY):
    """
    Fetch OHLCV from KuCoin, save as parquet.
    Returns the DataFrame.
    """
    exchange = ccxt.kucoin({'enableRateLimit': True})
    since = exchange.parse8601((datetime.now() - timedelta(days=days_back)).isoformat())
    all_candles = []
    print(f"Fetching {symbol_ccxt} {timeframe}...")
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol_ccxt, timeframe, since=since, limit=1500)
            if not candles:
                break
            all_candles += candles
            since = candles[-1][0] + 1
            if len(candles) < 1500:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"Error: {e}")
            break
    if not all_candles:
        print(f"  No data for {symbol_ccxt}")
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    file_path = os.path.join(DATA_FOLDER, f"{symbol_ccxt.replace('/', '_')}_{timeframe}.parquet")
    df.to_parquet(file_path)
    print(f"  Saved {len(df)} rows to {file_path}")
    return df

def fetch_all_data():
    """
    Fetch 4H and 1H data for all pairs. Skips if file already exists (from cache).
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for pair in CRYPTO_PAIRS:
        ccxt_symbol = pair.replace("-USDT", "/USDT")
        # 4H
        fname_4h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_4h.parquet")
        if not os.path.exists(fname_4h):
            fetch_and_save(ccxt_symbol, '4h')
        else:
            print(f"4H data exists for {pair}, skipping fetch.")
        # 1H (needed for momentum/Rsi in scoring)
        fname_1h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_1h.parquet")
        if not os.path.exists(fname_1h):
            fetch_and_save(ccxt_symbol, '1h')
        else:
            print(f"1H data exists for {pair}, skipping fetch.")

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest():
    print("Loading data into memory...")
    data = {}
    for pair in CRYPTO_PAIRS:
        ccxt_symbol = pair.replace("-USDT", "/USDT")
        fname_4h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_4h.parquet")
        fname_1h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_1h.parquet")
        if not os.path.exists(fname_4h) or not os.path.exists(fname_1h):
            print(f"Missing data for {pair}, skipping.")
            continue
        df_4h = pd.read_parquet(fname_4h)
        df_1h = pd.read_parquet(fname_1h)
        yahoo_symbol = pair.replace("-USDT", "-USD")
        # Build daily from 4h
        df_d = df_4h.resample('1d').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        data[yahoo_symbol] = {'4h': df_4h, '1h': df_1h, '1d': df_d}

    if "BTC-USD" not in data:
        print("BTC data missing, cannot continue.")
        return

    btc_4h = data["BTC-USD"]['4h']
    start_date = pd.Timestamp(BACKTEST_START)
    end_date = pd.Timestamp.now()
    timeline = btc_4h.index[(btc_4h.index >= start_date) & (btc_4h.index <= end_date)]

    balance = INITIAL_BALANCE
    open_trades = []       # each trade dict like live bot
    trade_log = []         # record of every partial close
    equity_curve = []

    print(f"Running backtest from {start_date.date()} to {end_date.date()}...")
    print(f"Timeline length: {len(timeline)} 4H candles")

    for current_time in timeline:
        # ---- 1. Check existing trades ----
        closed_indices = []
        for idx, trade in enumerate(open_trades):
            sym = trade['symbol']
            if sym not in data:
                continue
            df_sym_4h = data[sym]['4h']
            if current_time not in df_sym_4h.index:
                continue
            candle = df_sym_4h.loc[current_time]
            high, low = candle['High'], candle['Low']
            entry, stop, tps = trade['entry'], trade['stop'], trade['take_profits']
            direction = trade['direction']
            highest_tp = trade.get('highest_tp', -1)
            breakeven = trade.get('breakeven', False)
            remaining_qty = trade['quantity']
            fractions = [0.30, 0.10, 0.10, 0.10, 0.40]

            # Check TP
            new_tp_idx = None
            if direction == "LONG":
                for i in range(len(tps)-1, -1, -1):
                    if high >= tps[i] and i > highest_tp:
                        new_tp_idx = i
                        break
            else:
                for i in range(len(tps)-1, -1, -1):
                    if low <= tps[i] and i > highest_tp:
                        new_tp_idx = i
                        break

            if new_tp_idx is not None:
                for i in range(highest_tp+1, new_tp_idx+1):
                    if remaining_qty <= 0:
                        break
                    fraction = fractions[i]
                    exit_qty = trade['original_qty'] * fraction
                    if exit_qty > remaining_qty:
                        exit_qty = remaining_qty
                    if exit_qty > 0:
                        exit_price = tps[i]
                        pnl = (exit_price - entry) * exit_qty if direction == "LONG" else (entry - exit_price) * exit_qty
                        trade_log.append({
                            'timestamp': current_time,
                            'symbol': sym,
                            'action': direction,
                            'hit_level': f"TP{i+1}",
                            'exit_price': exit_price,
                            'quantity': exit_qty,
                            'pnl': round(pnl, 4)
                        })
                        balance += pnl
                        remaining_qty -= exit_qty
                        highest_tp = i
                        if i == 0:
                            breakeven = True
                if remaining_qty <= 0:
                    closed_indices.append(idx)
                    continue

            # Check stop loss
            if remaining_qty > 0:
                current_stop = entry if breakeven else stop
                sl_hit = (low <= current_stop) if direction == "LONG" else (high >= current_stop)
                if sl_hit:
                    exit_price = current_stop
                    pnl = (exit_price - entry) * remaining_qty if direction == "LONG" else (entry - exit_price) * remaining_qty
                    trade_log.append({
                        'timestamp': current_time,
                        'symbol': sym,
                        'action': direction,
                        'hit_level': "STOP LOSS" if not breakeven else "BREAKEVEN",
                        'exit_price': exit_price,
                        'quantity': remaining_qty,
                        'pnl': round(pnl, 4)
                    })
                    balance += pnl
                    closed_indices.append(idx)
                    continue

            # Update trade
            trade['highest_tp'] = highest_tp
            trade['breakeven'] = breakeven
            trade['quantity'] = remaining_qty

        # Remove closed trades (in reverse order)
        for idx in sorted(closed_indices, reverse=True):
            open_trades.pop(idx)

        # ---- 2. Generate new signals (if under limit) ----
        risky_count = len(open_trades)
        if risky_count < MAX_RISKY_TRADES:
            open_symbols = {t['symbol'] for t in open_trades}
            candidates = []
            for sym_yahoo, sym_data in data.items():
                if sym_yahoo in open_symbols:
                    continue
                # Slice data up to current_time
                df_4h = sym_data['4h'].loc[:current_time]
                df_1h = sym_data['1h'].loc[:current_time]
                df_d = sym_data['1d'].loc[:current_time]
                if len(df_4h) < 50:
                    continue
                # BTC context for scoring: use btc_4h up to current_time
                btc_ctx = btc_4h.loc[:current_time] if btc_4h is not None else None
                score, direction, price, atr_val, swing_level, layers = score_pair(df_4h, df_d, df_1h, btc_ctx)
                if direction is None or score < 6.0:
                    continue

                # Compute stop & TPs (exact same logic as live bot)
                rank = 99   # default, since we don't have rank in backtest
                min_stop_pct = 0.02
                max_stop_pct = 0.06
                raw_stop = (atr_val * 2.5) if (atr_val is not None and not math.isnan(atr_val)) else price * 0.02
                stop_distance = np.clip(raw_stop, price * min_stop_pct, price * max_stop_pct)
                if direction == "LONG":
                    stop = price - stop_distance
                    if swing_level and swing_level > price - stop_distance * 1.2:
                        stop = min(stop, swing_level - 0.05 * (atr_val if atr_val else price * 0.01))
                else:
                    stop = price + stop_distance
                    if swing_level and swing_level < price + stop_distance * 1.2:
                        stop = max(stop, swing_level + 0.05 * (atr_val if atr_val else price * 0.01))
                stop = round(stop, 6)
                risk = abs(price - stop)
                tp_multipliers = [0.4, 0.8, 1.2, 1.6, 2.0]
                tps = [round(price + m * risk, 6) if direction == "LONG" else round(price - m * risk, 6) for m in tp_multipliers]
                quantity = round((balance * RISK_PER_TRADE) / risk, 8)

                candidates.append({
                    'symbol': sym_yahoo,
                    'score': score,
                    'direction': direction,
                    'entry': price,
                    'stop': stop,
                    'take_profits': tps,
                    'quantity': quantity,
                    'original_qty': quantity,
                    'highest_tp': -1,
                    'breakeven': False
                })
            if candidates:
                best = max(candidates, key=lambda x: x['score'])
                if risky_count < MAX_RISKY_TRADES:
                    open_trades.append(best)

        # Record equity
        equity_curve.append((current_time, balance))

    # ---- After loop: close any remaining trades at last available price (optional) ----
    for trade in open_trades:
        sym = trade['symbol']
        if sym in data:
            df_sym = data[sym]['4h']
            last_price = df_sym['Close'].iloc[-1]
            entry = trade['entry']
            remaining_qty = trade['quantity']
            direction = trade['direction']
            pnl = (last_price - entry) * remaining_qty if direction == "LONG" else (entry - last_price) * remaining_qty
            trade_log.append({
                'timestamp': df_sym.index[-1],
                'symbol': sym,
                'action': direction,
                'hit_level': 'MARKET CLOSE',
                'exit_price': last_price,
                'quantity': remaining_qty,
                'pnl': round(pnl, 4)
            })
            balance += pnl

    # ---- Compute performance metrics ----
    if not trade_log:
        print("No trades were generated.")
        return

    trades_df = pd.DataFrame(trade_log)
    # Group by (timestamp, symbol) to get full trade PnL
    trade_groups = trades_df.groupby(['timestamp', 'symbol'])
    full_trades = []
    for (ts, sym), group in trade_groups:
        total_pnl = group['pnl'].sum()
        full_trades.append({
            'entry_time': ts,
            'symbol': sym,
            'total_pnl': total_pnl,
            'action': group['action'].iloc[0]
        })
    full_df = pd.DataFrame(full_trades)
    wins = full_df[full_df['total_pnl'] > 0]
    losses = full_df[full_df['total_pnl'] < 0]
    total_trades = len(full_df)
    total_pnl = full_df['total_pnl'].sum()
    winrate = (len(wins) / max(total_trades, 1)) * 100
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if len(losses) > 0 else float('inf')
    final_balance = INITIAL_BALANCE + total_pnl

    # Drawdown calculation from equity curve
    equity_df = pd.DataFrame(equity_curve, columns=['time', 'balance'])
    equity_df['peak'] = equity_df['balance'].cummax()
    equity_df['drawdown'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].max() * 100

    # Print summary
    summary = (
        f"\n{'='*50}\n"
        f"BACKTEST RESULTS\n"
        f"{'='*50}\n"
        f"Period: {BACKTEST_START} → {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Initial Balance: ${INITIAL_BALANCE:.2f}\n"
        f"Final Balance: ${final_balance:.2f}\n"
        f"Total Trades: {total_trades}\n"
        f"Winrate: {winrate:.1f}% ({len(wins)}W / {len(losses)}L)\n"
        f"Total P&L: ${total_pnl:.2f}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Average R per trade: {total_pnl/(total_trades*INITIAL_BALANCE*RISK_PER_TRADE):.2f}\n"
        f"{'='*50}"
    )
    print(summary)

    # Save to files
    with open("backtest_summary.txt", "w") as f:
        f.write(summary)
    full_df.to_csv("backtest_trades.csv", index=False)
    print("\nResults saved to backtest_summary.txt and backtest_trades.csv")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "FETCH"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if mode == "FETCH":
        fetch_all_data()
    elif mode == "BACKTEST":
        run_backtest()
    else:
        print("Usage: python backtest_ku.py FETCH|BACKTEST")