#!/usr/bin/env python3
"""
KuCoin Historical Backtester – MACD Zero‑Line Crossover + 200 EMA, 1:2 RR
Tight stop: min(5-bar swing low, 1.5 ATR) for longs, max(5-bar swing high, 1.5 ATR) for shorts
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
BTC_FILTER = True          # set to False to disable BTC trend filter
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

def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ============================================================
# MACD ZERO‑LINE CROSSOVER + 200 EMA ENTRY
# ============================================================
def detect_entry(df_4h, btc_df_4h=None):
    """
    Returns (direction, entry_price, stop_loss, tp) or (None,None,None,None)
    """
    if df_4h.empty or len(df_4h) < 80:
        return None, None, None, None

    price = df_4h['Close'].iloc[-1]
    ema200 = ema(df_4h['Close'], 200).iloc[-1]
    atr_val = atr(df_4h)

    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        return None, None, None, None

    # MACD calculation
    macd_line, macd_signal = macd(df_4h['Close'])
    macd_curr = macd_line.iloc[-1]
    macd_prev = macd_line.iloc[-2]
    sig_curr = macd_signal.iloc[-1]
    sig_prev = macd_signal.iloc[-2]

    # BTC filter (optional)
    if BTC_FILTER and btc_df_4h is not None and len(btc_df_4h) >= 50:
        btc_ema50 = ema(btc_df_4h['Close'], 50).iloc[-1]
        btc_trend_up = btc_df_4h['Close'].iloc[-1] > btc_ema50
    else:
        btc_trend_up = True
        btc_trend_down = True

    # ----- LONG SIGNAL -----
    if price > ema200 and macd_curr < 0 and sig_curr < 0:
        if macd_prev < sig_prev and macd_curr > sig_curr:   # crossover up
            if BTC_FILTER and not btc_trend_up:
                return None, None, None, None
            # Stop = min(5-bar swing low, entry - 1.5*ATR)
            swing_low = df_4h['Low'].iloc[-6:-1].min()   # last 5 bars excluding current
            stop_atr = price - 1.5 * atr_val
            stop = max(swing_low, stop_atr)   # choose the higher (tighter) stop
            if price - stop <= 0:
                return None, None, None, None
            tp = price + 2 * (price - stop)
            return "LONG", price, stop, tp

    # ----- SHORT SIGNAL -----
    if price < ema200 and macd_curr > 0 and sig_curr > 0:
        if macd_prev > sig_prev and macd_curr < sig_curr:   # crossover down
            if BTC_FILTER and btc_trend_up:
                return None, None, None, None
            # Stop = max(5-bar swing high, entry + 1.5*ATR)
            swing_high = df_4h['High'].iloc[-6:-1].max()
            stop_atr = price + 1.5 * atr_val
            stop = min(swing_high, stop_atr)   # choose the lower (tighter) stop
            if stop - price <= 0:
                return None, None, None, None
            tp = price - 2 * (stop - price)
            return "SHORT", price, stop, tp

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

    print(f"Running MACD Zero‑Line Crossover + 200 EMA (Tight Stop) from {start_date.date()} to {end_date.date()}...")
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
                btc_ctx = btc_4h.loc[:current_time] if BTC_FILTER else None
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

    # Streaks
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
        f"BACKTEST RESULTS (MACD Zero‑Line Crossover + 200 EMA, Tight Stop)\n"
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