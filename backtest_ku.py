#!/usr/bin/env python3
"""
KuCoin Historical Backtester – ADX+RSI+Bollinger Pullback, 1:2 RR, Micro‑Stop
Tight stop (0.5‑0.7 ATR), target = 1‑1.4 ATR (2R).
Strong trend (ADX>25), RSI filter, Bollinger Band confirmation, BTC aligned.
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
INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.01
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
# TECHNICAL INDICATORS
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).iloc[-1] if not rs.isna().iloc[-1] else None

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
    return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

def bollinger_bands(df, period=20, std=2):
    sma20 = sma(df['Close'], period)
    rolling_std = df['Close'].rolling(period).std()
    upper = sma20 + std * rolling_std
    lower = sma20 - std * rolling_std
    return upper.iloc[-1], sma20.iloc[-1], lower.iloc[-1]

# ============================================================
# REFINED PULLBACK ENTRY (ADX+RSI+Bollinger)
# ============================================================
def detect_entry(df_4h, btc_df_4h=None):
    if df_4h.empty or len(df_4h) < 80:
        return None, None, None, None

    price = df_4h['Close'].iloc[-1]
    ema50 = ema(df_4h['Close'], 50).iloc[-1]
    ema200 = ema(df_4h['Close'], 200).iloc[-1]
    atr_val = atr(df_4h)
    rsi_val = rsi(df_4h)
    adx_val = adx(df_4h)

    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        return None, None, None, None

    # Minimum ATR (0.8% of price)
    if atr_val < price * 0.008:
        return None, None, None, None

    # Trend check
    uptrend = price > ema50 and ema50 > ema200
    downtrend = price < ema50 and ema50 < ema200
    if not uptrend and not downtrend:
        return None, None, None, None

    # ADX must be strong (>25)
    if adx_val is None or pd.isna(adx_val) or adx_val < 25:
        return None, None, None, None

    # BTC filter
    if btc_df_4h is not None and len(btc_df_4h) >= 50:
        btc_ema50 = ema(btc_df_4h['Close'], 50).iloc[-1]
        btc_trend_up = btc_df_4h['Close'].iloc[-1] > btc_ema50
    else:
        return None, None, None, None

    # RSI filter
    if rsi_val is None or pd.isna(rsi_val):
        return None, None, None, None

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(df_4h)

    # Volume check (last bar volume > 1.2 x average of last 20 bars excluding current)
    if len(df_4h) >= 21:
        vol_avg = df_4h['Volume'].iloc[-21:-1].mean()
        if df_4h['Volume'].iloc[-1] < vol_avg * 1.2:
            return None, None, None, None
    else:
        return None, None, None, None

    last = df_4h.iloc[-1]

    # ----- LONG SETUP -----
    if uptrend and btc_trend_up:
        # RSI condition: not overbought
        if rsi_val >= 65:
            return None, None, None, None
        # Price must be near or below the middle Bollinger Band (pullback to value)
        if price > bb_mid * 1.01:   # only allow entry if price is at or below the middle band
            return None, None, None, None
        # Pullback to EMA50 (within 1.5 ATR)
        if abs(price - ema50) > 1.5 * atr_val:
            return None, None, None, None
        # Bullish candle: close > open
        if last['Close'] <= last['Open']:
            return None, None, None, None

        entry = price
        stop = max(ema50 - 0.5 * atr_val, df_4h['Low'].iloc[-5:].min() - 0.1 * atr_val)
        if entry - stop > 0.8 * atr_val:
            stop = entry - 0.7 * atr_val
        if entry - stop < 0.2 * atr_val:
            stop = entry - 0.3 * atr_val
        risk = entry - stop
        if risk <= 0:
            return None, None, None, None
        tp = entry + 2 * risk
        return "LONG", entry, stop, tp

    # ----- SHORT SETUP -----
    if downtrend and not btc_trend_up:
        if rsi_val <= 35:
            return None, None, None, None
        if price < bb_mid * 0.99:
            return None, None, None, None
        if abs(price - ema50) > 1.5 * atr_val:
            return None, None, None, None
        if last['Close'] >= last['Open']:
            return None, None, None, None

        entry = price
        stop = min(ema50 + 0.5 * atr_val, df_4h['High'].iloc[-5:].max() + 0.1 * atr_val)
        if stop - entry > 0.8 * atr_val:
            stop = entry + 0.7 * atr_val
        if stop - entry < 0.2 * atr_val:
            stop = entry + 0.3 * atr_val
        risk = stop - entry
        if risk <= 0:
            return None, None, None, None
        tp = entry - 2 * risk
        return "SHORT", entry, stop, tp

    return None, None, None, None

# ============================================================
# DATA FETCHING
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
# BACKTEST ENGINE
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

    print(f"Running REFINED PULLBACK (ADX+RSI+BB) 1:2 RR from {start_date.date()} to {end_date.date()}...")
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
        f"BACKTEST RESULTS (REFINED PULLBACK, 1:2 RR)\n"
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