#!/usr/bin/env python3
"""
KuCoin Historical Backtester – EMA+Bollinger Pullback, 1:2 RR
Long: price>200EMA, touches lower BB, green candle, volume>1.5x avg, stop=signal low, target=2R
Short: price<200EMA, touches upper BB, red candle, volume surge, stop=signal high, target=2R
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

def bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    return upper, lower

# ============================================================
# ENTRY LOGIC
# ============================================================
def detect_entry(df_4h):
    """
    Returns (direction, entry_price, stop_loss, tp) or None
    """
    if df_4h.empty or len(df_4h) < 50:
        return None

    price = df_4h['Close'].iloc[-1]
    ema200 = ema(df_4h['Close'], 200).iloc[-1]
    bb_upper, bb_lower = bollinger_bands(df_4h['Close'])
    bb_upper = bb_upper.iloc[-1]
    bb_lower = bb_lower.iloc[-1]

    vol_curr = df_4h['Volume'].iloc[-1]
    if len(df_4h) >= 21:
        vol_avg = df_4h['Volume'].iloc[-21:-1].mean()
    else:
        vol_avg = vol_curr

    # Candle info
    open_curr = df_4h['Open'].iloc[-1]
    close_curr = df_4h['Close'].iloc[-1]
    low_curr = df_4h['Low'].iloc[-1]
    high_curr = df_4h['High'].iloc[-1]

    # ----- LONG -----
    if price > ema200 and low_curr <= bb_lower:
        if close_curr > open_curr and vol_curr > vol_avg * 1.5:
            entry = close_curr
            stop = low_curr
            if entry - stop <= 0:
                return None
            tp = entry + 2 * (entry - stop)
            return "LONG", entry, stop, tp

    # ----- SHORT -----
    if price < ema200 and high_curr >= bb_upper:
        if close_curr < open_curr and vol_curr > vol_avg * 1.5:
            entry = close_curr
            stop = high_curr
            if stop - entry <= 0:
                return None
            tp = entry - 2 * (stop - entry)
            return "SHORT", entry, stop, tp

    return None

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

    if "BTC-USD" in data:
        btc_4h = data["BTC-USD"]['4h']
    else:
        btc_4h = next(iter(data.values()))['4h']

    start_date = pd.Timestamp(BACKTEST_START)
    end_date = pd.Timestamp.now()
    timeline = btc_4h.index[(btc_4h.index >= start_date) & (btc_4h.index <= end_date)]

    balance = INITIAL_BALANCE
    open_trades = []
    trade_log = []
    equity_curve = []

    print(f"Running EMA+BOLLINGER PULLBACK (1:2 RR) from {start_date.date()} to {end_date.date()}...")
    print(f"Timeline: {len(timeline)} 4H candles")

    for current_time in timeline:
        # Check existing trades
        closed_indices = []
        for idx, trade in enumerate(open_trades):
            sym = trade['symbol']
            if sym not in data:
                continue
            df_sym = data[sym]['4h']
            if current_time not in df_sym.index:
                continue
            bar = df_sym.loc[current_time]
            high, low = bar['High'], bar['Low']
            entry, stop, tp = trade['entry'], trade['stop'], trade['tp']
            direction = trade['direction']
            qty = trade['quantity']

            hit = False
            if direction == "LONG":
                if high >= tp:
                    exit_price = tp
                    hit = True
                elif low <= stop:
                    exit_price = stop
                    hit = True
            else:
                if low <= tp:
                    exit_price = tp
                    hit = True
                elif high >= stop:
                    exit_price = stop
                    hit = True

            if hit:
                pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
                trade_log.append({
                    'timestamp': current_time, 'symbol': sym, 'action': direction,
                    'hit_level': "TP" if exit_price == tp else "STOP LOSS",
                    'exit_price': exit_price, 'quantity': qty, 'pnl': round(pnl, 4)
                })
                balance += pnl
                closed_indices.append(idx)

        for idx in sorted(closed_indices, reverse=True):
            open_trades.pop(idx)

        # New signals
        if len(open_trades) < MAX_RISKY_TRADES:
            open_symbols = {t['symbol'] for t in open_trades}
            candidates = []
            for sym, sym_data in data.items():
                if sym in open_symbols:
                    continue
                df_4h = sym_data['4h'].loc[:current_time]
                res = detect_entry(df_4h)
                if res is None:
                    continue
                direction, entry, stop, tp = res
                risk = abs(entry - stop)
                if risk <= 0:
                    continue
                qty = round((balance * RISK_PER_TRADE) / risk, 8)
                candidates.append({
                    'symbol': sym, 'direction': direction, 'entry': entry,
                    'stop': stop, 'tp': tp, 'quantity': qty, 'original_qty': qty
                })
            for t in candidates:
                if len(open_trades) >= MAX_RISKY_TRADES:
                    break
                open_trades.append(t)

        equity_curve.append((current_time, balance))

    # Close remaining trades at last price
    for trade in open_trades:
        sym = trade['symbol']
        if sym in data:
            last_price = data[sym]['4h']['Close'].iloc[-1]
            entry, qty, direction = trade['entry'], trade['quantity'], trade['direction']
            pnl = (last_price - entry) * qty if direction == "LONG" else (entry - last_price) * qty
            trade_log.append({
                'timestamp': data[sym]['4h'].index[-1], 'symbol': sym, 'action': direction,
                'hit_level': 'MARKET CLOSE', 'exit_price': last_price, 'quantity': qty, 'pnl': round(pnl, 4)
            })
            balance += pnl

    if not trade_log:
        print("No trades were generated.")
        return

    trades_df = pd.DataFrame(trade_log)
    groups = trades_df.groupby(['timestamp', 'symbol'])
    full_trades = []
    for (ts, sym), grp in groups:
        total_pnl = grp['pnl'].sum()
        full_trades.append({'entry_time': ts, 'symbol': sym, 'total_pnl': total_pnl, 'action': grp['action'].iloc[0]})
    full_df = pd.DataFrame(full_trades).sort_values('entry_time')
    full_df['is_win'] = full_df['total_pnl'] > 0
    full_df['is_loss'] = full_df['total_pnl'] < 0

    wins = full_df[full_df['is_win']]
    losses = full_df[full_df['is_loss']]
    total_trades = len(full_df)
    total_pnl = full_df['total_pnl'].sum()
    winrate = len(wins) / max(total_trades, 1) * 100
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if len(losses) > 0 else float('inf')
    final_balance = INITIAL_BALANCE + total_pnl

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'balance'])
    equity_df['peak'] = equity_df['balance'].cummax()
    equity_df['dd'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak']
    max_dd = equity_df['dd'].max() * 100

    # Streaks
    curr_win, curr_loss = 0, 0
    win_buf, loss_buf = 0, 0
    longest_win, longest_loss = 0, 0
    for _, r in full_df.iterrows():
        if r['is_win']:
            win_buf += 1; loss_buf = 0
            longest_win = max(longest_win, win_buf)
        elif r['is_loss']:
            loss_buf += 1; win_buf = 0
            longest_loss = max(longest_loss, loss_buf)
        else:
            win_buf = loss_buf = 0
    for _, r in full_df.iloc[::-1].iterrows():
        if r['is_win']:
            if curr_loss == 0: curr_win += 1
            else: break
        elif r['is_loss']:
            if curr_win == 0: curr_loss += 1
            else: break
        else: break

    summary = (
        f"\n{'='*50}\n"
        f"BACKTEST RESULTS (EMA+BOLLINGER PULLBACK, 1:2 RR)\n"
        f"{'='*50}\n"
        f"Period: {BACKTEST_START} → {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Initial Balance: ${INITIAL_BALANCE:.2f}\n"
        f"Final Balance: ${final_balance:.2f}\n"
        f"Total Trades: {total_trades}\n"
        f"Winrate: {winrate:.1f}% ({len(wins)}W / {len(losses)}L)\n"
        f"Total P&L: ${total_pnl:.2f}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Current Win Streak: {curr_win} 🔥\n"
        f"Current Loss Streak: {curr_loss} 😞\n"
        f"Longest Win Streak: {longest_win}\n"
        f"Longest Loss Streak: {longest_loss}\n"
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