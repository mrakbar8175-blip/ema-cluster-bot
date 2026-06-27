#!/usr/bin/env python3
"""
KuCoin Historical Backtester – Enhanced EMA Pullback, 1:2 RR, Micro‑Stop
Filters: ADX>20, RSI zone, volume confirmation, stronger break.
Tight stop (0.5‑0.7 ATR), target = 1‑1.4 ATR (2R).
Usage: python backtest_ku.py FETCH    (download data)
       python backtest_ku.py BACKTEST (run simulation)
"""

import ccxt
import pandas as pd
import numpy as np
import os, sys, time, math
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
DAYS_HISTORY = 730
BACKTEST_START = "2025-01-01"
INITIAL_BALANCE = 20.0       # instead of 1000.0
RISK_PER_TRADE = 0.025       # instead of 0.01
MAX_RISKY_TRADES = 5
DATA_FOLDER = "kucoin_data"

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
# TECHNICAL INDICATORS (enhanced)
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()  # make positive when price moves down

    # DM must be positive only when one direction exceeds the other
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)

    # True Range
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val, plus_di, minus_di

# ============================================================
# ENHANCED PULLBACK ENTRY (higher winrate filters)
# ============================================================
def detect_entry(df_4h, btc_df_4h=None):
    """
    Returns (direction, entry_price, stop_loss, tp) or (None,None,None,None)
    Now with ADX, RSI, volume & stronger trend-resumption confirmation.
    """
    if df_4h.empty or len(df_4h) < 70:
        return None, None, None, None

    price = df_4h['Close'].iloc[-1]
    ema50 = ema(df_4h['Close'], 50)
    ema200 = ema(df_4h['Close'], 200)
    atr_val = atr(df_4h)

    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        return None, None, None, None

    # ADX & DI
    adx_series, plus_di, minus_di = adx(df_4h, 14)
    adx_val = adx_series.iloc[-1]
    plus = plus_di.iloc[-1]
    minus = minus_di.iloc[-1]

    # RSI
    rsi_series = rsi(df_4h['Close'], 14)
    rsi_val = rsi_series.iloc[-1]

    # Volume
    vol = df_4h['Volume']
    vol_current = vol.iloc[-1]
    vol_previous = vol.iloc[-2] if len(vol) >= 2 else 0

    # Trend check
    uptrend = price > ema50.iloc[-1] and ema50.iloc[-1] > ema200.iloc[-1]
    downtrend = price < ema50.iloc[-1] and ema50.iloc[-1] < ema200.iloc[-1]
    if not uptrend and not downtrend:
        return None, None, None, None

    # BTC filter
    if btc_df_4h is not None and len(btc_df_4h) >= 50:
        btc_ema50 = ema(btc_df_4h['Close'], 50)
        btc_trend_up = btc_df_4h['Close'].iloc[-1] > btc_ema50.iloc[-1]
    else:
        return None, None, None, None

    # Pullback condition: price touched 50-EMA recently (within 1.5 ATR)
    ema50_val = ema50.iloc[-1]
    low_curr, low_prev = df_4h['Low'].iloc[-1], df_4h['Low'].iloc[-2]
    high_curr, high_prev = df_4h['High'].iloc[-1], df_4h['High'].iloc[-2]
    open_curr, close_curr = df_4h['Open'].iloc[-1], price
    close_prev = df_4h['Close'].iloc[-2]

    # ----- LONG SETUP -----
    if uptrend and btc_trend_up:
        pullback = (abs(low_curr - ema50_val) < 1.5 * atr_val) or (abs(low_prev - ema50_val) < 1.5 * atr_val)
        # Strong resumption: close > previous high, green candle, volume expansion
        resumption = (close_curr > high_prev) and (close_curr > open_curr) and (vol_current > vol_previous)
        rsi_ok = 40 < rsi_val < 65          # not overbought, not too weak
        adx_ok = adx_val > 20 and plus > minus

        if pullback and resumption and rsi_ok and adx_ok:
            entry = price
            recent_low = df_4h['Low'].iloc[-5:].min()
            stop = min(recent_low, ema50_val - 0.5 * atr_val)
            # Enforce micro-stop boundaries (0.5 – 0.7 ATR)
            if entry - stop > 0.7 * atr_val:
                stop = entry - 0.6 * atr_val
            if entry - stop < 0.2 * atr_val:
                stop = entry - 0.3 * atr_val
            risk = entry - stop
            if risk <= 0:
                return None, None, None, None
            tp = entry + 2 * risk
            return "LONG", entry, stop, tp

    # ----- SHORT SETUP -----
    if downtrend and not btc_trend_up:
        pullback = (abs(high_curr - ema50_val) < 1.5 * atr_val) or (abs(high_prev - ema50_val) < 1.5 * atr_val)
        resumption = (close_curr < low_prev) and (close_curr < open_curr) and (vol_current > vol_previous)
        rsi_ok = 35 < rsi_val < 60
        adx_ok = adx_val > 20 and minus > plus

        if pullback and resumption and rsi_ok and adx_ok:
            entry = price
            recent_high = df_4h['High'].iloc[-5:].max()
            stop = max(recent_high, ema50_val + 0.5 * atr_val)
            if stop - entry > 0.7 * atr_val:
                stop = entry + 0.6 * atr_val
            if stop - entry < 0.2 * atr_val:
                stop = entry + 0.3 * atr_val
            risk = stop - entry
            if risk <= 0:
                return None, None, None, None
            tp = entry - 2 * risk
            return "SHORT", entry, stop, tp

    return None, None, None, None

# ============================================================
# DATA FETCHING (unchanged)
# ============================================================
def fetch_and_save(symbol_ccxt, timeframe, days_back=DAYS_HISTORY):
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
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    file_path = os.path.join(DATA_FOLDER, f"{symbol_ccxt.replace('/', '_')}_{timeframe}.parquet")
    df.to_parquet(file_path)
    print(f"  Saved {len(df)} rows to {file_path}")
    return df

def fetch_all_data():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for pair in CRYPTO_PAIRS:
        ccxt_symbol = pair.replace("-USDT", "/USDT")
        fname_4h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_4h.parquet")
        if not os.path.exists(fname_4h):
            fetch_and_save(ccxt_symbol, '4h')
        else:
            print(f"4H data exists for {pair}, skipping fetch.")

# ============================================================
# BACKTEST ENGINE (unchanged)
# ============================================================
def run_backtest():
    print("Loading data into memory...")
    data = {}
    for pair in CRYPTO_PAIRS:
        ccxt_symbol = pair.replace("-USDT", "/USDT")
        fname_4h = os.path.join(DATA_FOLDER, f"{ccxt_symbol.replace('/', '_')}_4h.parquet")
        if not os.path.exists(fname_4h):
            continue
        df_4h = pd.read_parquet(fname_4h)
        yahoo_symbol = pair.replace("-USDT", "-USD")
        data[yahoo_symbol] = {'4h': df_4h}

    if "BTC-USD" not in data:
        print("BTC data missing, cannot continue.")
        return

    btc_4h = data["BTC-USD"]['4h']
    start_date = pd.Timestamp(BACKTEST_START)
    end_date = pd.Timestamp.now()
    timeline = btc_4h.index[(btc_4h.index >= start_date) & (btc_4h.index <= end_date)]

    balance = INITIAL_BALANCE
    open_trades = []
    trade_log = []
    equity_curve = []

    print(f"Running ENHANCED PULLBACK (1:2 RR, micro‑stop) from {start_date.date()} to {end_date.date()}...")
    print(f"Timeline: {len(timeline)} 4H candles")

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
            entry, stop, tp = trade['entry'], trade['stop'], trade['tp']
            direction = trade['direction']
            remaining_qty = trade['quantity']

            hit_tp = False
            hit_sl = False
            exit_price = None

            if direction == "LONG":
                if high >= tp:
                    hit_tp = True
                    exit_price = tp
                elif low <= stop:
                    hit_sl = True
                    exit_price = stop
            else:
                if low <= tp:
                    hit_tp = True
                    exit_price = tp
                elif high >= stop:
                    hit_sl = True
                    exit_price = stop

            if hit_tp or hit_sl:
                pnl = (exit_price - entry) * remaining_qty if direction == "LONG" else (entry - exit_price) * remaining_qty
                trade_log.append({
                    'timestamp': current_time,
                    'symbol': sym,
                    'action': direction,
                    'hit_level': "TP" if hit_tp else "STOP LOSS",
                    'exit_price': exit_price,
                    'quantity': remaining_qty,
                    'pnl': round(pnl, 4)
                })
                balance += pnl
                closed_indices.append(idx)

        for idx in sorted(closed_indices, reverse=True):
            open_trades.pop(idx)

        # ---- 2. Generate new signals ----
        risky_count = len(open_trades)
        if risky_count < MAX_RISKY_TRADES:
            open_symbols = {t['symbol'] for t in open_trades}
            candidates = []
            for sym_yahoo, sym_data in data.items():
                if sym_yahoo in open_symbols:
                    continue
                df_4h = sym_data['4h'].loc[:current_time]
                btc_ctx = btc_4h.loc[:current_time]
                direction, entry, stop, tp = detect_entry(df_4h, btc_ctx)
                if direction is None:
                    continue
                risk = abs(entry - stop)
                if risk <= 0:
                    continue
                quantity = round((balance * RISK_PER_TRADE) / risk, 8)
                candidates.append({
                    'symbol': sym_yahoo,
                    'direction': direction,
                    'entry': entry,
                    'stop': stop,
                    'tp': tp,
                    'quantity': quantity,
                    'original_qty': quantity
                })
            for trade in candidates:
                if risky_count >= MAX_RISKY_TRADES:
                    break
                open_trades.append(trade)
                risky_count += 1

        equity_curve.append((current_time, balance))

    # Close remaining trades at last price
    for trade in open_trades:
        sym = trade['symbol']
        if sym in data:
            last_price = data[sym]['4h']['Close'].iloc[-1]
            entry = trade['entry']
            remaining_qty = trade['quantity']
            direction = trade['direction']
            pnl = (last_price - entry) * remaining_qty if direction == "LONG" else (entry - last_price) * remaining_qty
            trade_log.append({
                'timestamp': data[sym]['4h'].index[-1],
                'symbol': sym,
                'action': direction,
                'hit_level': 'MARKET CLOSE',
                'exit_price': last_price,
                'quantity': remaining_qty,
                'pnl': round(pnl, 4)
            })
            balance += pnl

    if not trade_log:
        print("No trades were generated.")
        return

    trades_df = pd.DataFrame(trade_log)
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
    full_df = pd.DataFrame(full_trades).sort_values('entry_time')
    full_df['is_win'] = full_df['total_pnl'] > 0
    full_df['is_loss'] = full_df['total_pnl'] < 0

    curr_win_streak = 0
    curr_loss_streak = 0
    longest_win_streak = 0
    longest_loss_streak = 0
    win_buf = 0
    loss_buf = 0
    for _, row in full_df.iterrows():
        if row['is_win']:
            win_buf += 1
            loss_buf = 0
            longest_win_streak = max(longest_win_streak, win_buf)
        elif row['is_loss']:
            loss_buf += 1
            win_buf = 0
            longest_loss_streak = max(longest_loss_streak, loss_buf)
        else:
            win_buf = 0
            loss_buf = 0
    for _, row in full_df.iloc[::-1].iterrows():
        if row['is_win']:
            if curr_loss_streak == 0:
                curr_win_streak += 1
            else:
                break
        elif row['is_loss']:
            if curr_win_streak == 0:
                curr_loss_streak += 1
            else:
                break
        else:
            break

    wins = full_df[full_df['is_win']]
    losses = full_df[full_df['is_loss']]
    total_trades = len(full_df)
    total_pnl = full_df['total_pnl'].sum()
    winrate = (len(wins) / max(total_trades, 1)) * 100
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if len(losses) > 0 else float('inf')
    final_balance = INITIAL_BALANCE + total_pnl

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'balance'])
    equity_df['peak'] = equity_df['balance'].cummax()
    equity_df['drawdown'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].max() * 100

    summary = (
        f"\n{'='*50}\n"
        f"BACKTEST RESULTS (ENHANCED PULLBACK, 1:2 RR, MICRO‑STOP)\n"
        f"{'='*50}\n"
        f"Period: {BACKTEST_START} → {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Initial Balance: ${INITIAL_BALANCE:.2f}\n"
        f"Final Balance: ${final_balance:.2f}\n"
        f"Total Trades: {total_trades}\n"
        f"Winrate: {winrate:.1f}% ({len(wins)}W / {len(losses)}L)\n"
        f"Total P&L: ${total_pnl:.2f}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Current Win Streak: {curr_win_streak} 🔥\n"
        f"Current Loss Streak: {curr_loss_streak} 😞\n"
        f"Longest Win Streak: {longest_win_streak}\n"
        f"Longest Loss Streak: {longest_loss_streak}\n"
        f"Average R per trade: {total_pnl/(total_trades*INITIAL_BALANCE*RISK_PER_TRADE):.2f}\n"
        f"{'='*50}"
    )
    print(summary)

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