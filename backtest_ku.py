#!/usr/bin/env python3
"""
Fetches raw indicator CSV if needed, then prints a condensed statistical report.
"""

import os, sys, time, json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
YEARS_BACK = 2
CACHE_DIR = "cache"
OUTPUT_CSV = "raw_indicator_data_2y.csv"
MAX_COINS = 20

# ==================== DATA FETCHING ====================
def extraction_needed():
    if not os.path.exists(OUTPUT_CSV):
        return True
    file_age = time.time() - os.path.getmtime(OUTPUT_CSV)
    return file_age > 86400

def fetch_current_momentum_coins(limit=100, momentum_top=MAX_COINS):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "14d"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        candidates = []
        for coin in data:
            symbol = coin.get("symbol", "").upper()
            if symbol in {"QUQ","USDT","USDC","DAI","BUSD","TUSD","USDP","FDUSD","LEO","WBT"}:
                continue
            roc_14 = coin.get("price_change_percentage_14d_in_currency")
            momentum = abs(roc_14) if roc_14 is not None else 0.0
            candidates.append((symbol, momentum))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in candidates[:momentum_top]]
    except:
        return ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","DOT","MATIC","LINK",
                "AVAX","SHIB","UNI","LTC","ATOM","NEAR","FIL","APT","ARB","OP"]

def fetch_kucoin_usdt_symbols():
    try:
        resp = requests.get("https://api.kucoin.com/api/v1/symbols", timeout=10)
        data = resp.json()
        if data.get("code") != "200000":
            return set()
        return {item["symbol"] for item in data["data"]
                if item["quoteCurrency"] == "USDT" and item["enableTrading"]}
    except:
        return set()

def build_valid_coin_list():
    momentum = fetch_current_momentum_coins(limit=100, momentum_top=MAX_COINS)
    kucoin = fetch_kucoin_usdt_symbols()
    valid = [f"{s}-USDT" for s in momentum if f"{s}-USDT" in kucoin]
    if len(valid) < 5:
        fallback = ["BTC-USDT","ETH-USDT","BNB-USDT","SOL-USDT","XRP-USDT",
                    "ADA-USDT","DOGE-USDT","DOT-USDT","MATIC-USDT","LINK-USDT",
                    "AVAX-USDT","SHIB-USDT","UNI-USDT","LTC-USDT","ATOM-USDT",
                    "NEAR-USDT","FIL-USDT","APT-USDT","ARB-USDT","OP-USDT"]
        valid = [f for f in fallback if f in kucoin][:MAX_COINS]
    return valid

def get_kucoin_klines(symbol, interval, start_time, end_time, limit=1000):
    interval_map = {"1h": "1hour", "4h": "4hour", "1d": "1day"}
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
        "type": interval_map.get(interval, interval),
        "symbol": symbol,
        "startAt": int(start_time.timestamp()),
        "endAt": int(end_time.timestamp()),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data["data"] if data.get("code") == "200000" else []
    except:
        return []

def fetch_ohlcv_kucoin(symbol, interval, start_date, end_date):
    all_candles = []
    chunk_end = end_date
    while chunk_end > start_date:
        chunk_start = chunk_end - timedelta(hours=1000)
        if chunk_start < start_date:
            chunk_start = start_date
        candles = get_kucoin_klines(symbol, interval, chunk_start, chunk_end, limit=1000)
        if not candles:
            chunk_end = chunk_start
            continue
        for c in candles:
            ts = datetime.utcfromtimestamp(int(c[0]))
            all_candles.append({
                "open_time": ts,
                "Open": float(c[1]), "Close": float(c[2]),
                "High": float(c[3]), "Low": float(c[4]), "Volume": float(c[5])
            })
        earliest = min(int(c[0]) for c in candles)
        chunk_end = datetime.utcfromtimestamp(earliest) - timedelta(hours=1)
        time.sleep(0.1)
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles).drop_duplicates("open_time").sort_values("open_time")
    df.set_index("open_time", inplace=True)
    return df

def cached_fetch(symbol, interval, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname, index_col=0, parse_dates=True)
    df = fetch_ohlcv_kucoin(symbol, interval, start_date, end_date)
    if not df.empty:
        df.to_csv(fname)
    return df

def compute_indicators(df_4h, df_daily_raw, df_1h, df_btc_4h):
    d = df_4h[['Open','High','Low','Close','Volume']].copy()
    d.index.name = 'timestamp'

    d['EMA50_4h'] = d['Close'].ewm(span=50, adjust=False).mean()
    d['EMA200_4h'] = d['Close'].ewm(span=200, adjust=False).mean()

    h, l, c = d['High'], d['Low'], d['Close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR_4h'] = tr.rolling(14).mean()

    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    d['RSI_4h'] = 100 - (100 / (1 + rs))

    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    d['MACD_line'] = macd_line
    d['MACD_signal'] = macd_signal
    d['MACD_hist'] = macd_line - macd_signal

    dm_plus = h.diff().clip(lower=0)
    dm_minus = -l.diff().clip(lower=0)
    atr_ewm = tr.ewm(alpha=1/14, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr_ewm)
    di_minus = 100 * (dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr_ewm)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    d['ADX_4h'] = dx.ewm(alpha=1/14, adjust=False).mean()
    d['DI_plus'] = di_plus
    d['DI_minus'] = di_minus

    d['Volume_surge'] = d['Volume'] > (d['Volume'].shift(1).rolling(5).mean() * 1.2)
    d['SR_High_20'] = d['High'].rolling(20).max()
    d['SR_Low_20'] = d['Low'].rolling(20).min()

    if not df_daily_raw.empty:
        daily_ema50 = df_daily_raw['Close'].ewm(span=50, adjust=False).mean()
        daily_ema200 = df_daily_raw['Close'].ewm(span=200, adjust=False).mean()
        daily_ema50_aligned = daily_ema50.reindex(d.index, method='ffill')
        daily_ema200_aligned = daily_ema200.reindex(d.index, method='ffill')
    else:
        df_daily_fallback = d['Close'].resample('D').last().dropna()
        daily_ema50 = df_daily_fallback.ewm(span=50, adjust=False).mean()
        daily_ema200 = df_daily_fallback.ewm(span=200, adjust=False).mean()
        daily_ema50_aligned = daily_ema50.reindex(d.index, method='ffill')
        daily_ema200_aligned = daily_ema200.reindex(d.index, method='ffill')
    d['daily_EMA50'] = daily_ema50_aligned
    d['daily_EMA200'] = daily_ema200_aligned

    if not df_btc_4h.empty:
        d['BTC_close_4h'] = df_btc_4h['Close'].reindex(d.index, method='ffill')
        d['BTC_EMA50_4h'] = df_btc_4h['Close'].ewm(span=50, adjust=False).mean().reindex(d.index, method='ffill')
    else:
        d['BTC_close_4h'] = np.nan
        d['BTC_EMA50_4h'] = np.nan

    if not df_1h.empty:
        delta_1h = df_1h['Close'].diff()
        gain_1h = delta_1h.clip(lower=0)
        loss_1h = -delta_1h.clip(upper=0)
        avg_gain_1h = gain_1h.ewm(alpha=1/14, adjust=False).mean()
        avg_loss_1h = loss_1h.ewm(alpha=1/14, adjust=False).mean()
        rs_1h = avg_gain_1h / avg_loss_1h
        rsi_1h = 100 - (100 / (1 + rs_1h))
        rsi_1h_aligned = rsi_1h.resample('4h').last().reindex(d.index, method='ffill')
        d['1h_RSI'] = rsi_1h_aligned

        df_1h_copy = df_1h.copy()
        df_1h_copy['bull_mom'] = (df_1h_copy['Close'] - df_1h_copy['Open']) / (df_1h_copy['High'] - df_1h_copy['Low'])
        d['1h_bullish_momentum'] = df_1h_copy['bull_mom'].resample('4h').last().reindex(d.index, method='ffill')
    else:
        d['1h_RSI'] = np.nan
        d['1h_bullish_momentum'] = np.nan

    d['fwd_return_1d'] = d['Close'].shift(-6) / d['Close'] - 1
    d['fwd_return_1w'] = d['Close'].shift(-42) / d['Close'] - 1
    d['fwd_return_2w'] = d['Close'].shift(-84) / d['Close'] - 1
    return d

def run_extraction():
    print("📦 PHASE 1: DATA EXTRACTION")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_BACK)
    print(f"Window: {start_date.date()} → {end_date.date()}")

    print("Fetching BTC...")
    btc_1h = cached_fetch("BTC-USDT", "1h", start_date, end_date)
    if btc_1h.empty:
        print("❌ BTC 1h empty."); sys.exit(1)
    btc_4h = btc_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    coins = build_valid_coin_list()
    print(f"Coin list ({len(coins)}): {coins}")
    all_dfs = []
    for sym in coins:
        print(f"Processing {sym}...")
        df_1h = cached_fetch(sym, "1h", start_date, end_date)
        if df_1h.empty: continue
        df_4h = df_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        df_daily = cached_fetch(sym, "1d", start_date, end_date)
        res = compute_indicators(df_4h, df_daily, df_1h, btc_4h)
        res['symbol'] = sym
        all_dfs.append(res)
        time.sleep(0.05)

    if not all_dfs:
        print("❌ No data."); sys.exit(1)
    final = pd.concat(all_dfs).reset_index().sort_values(['symbol','timestamp'])
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {OUTPUT_CSV} ({len(final)} rows).\n")

# ==================== CONDENSED ANALYSIS ====================
def condensed_analysis():
    df = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
    print(f"Data: {len(df)} rows, {df['symbol'].nunique()} coins")
    print(f"Period: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

    targets = ["fwd_return_1d", "fwd_return_1w", "fwd_return_2w"]
    features = [c for c in df.columns if c not in ["timestamp","symbol","Open","High","Low","Close","Volume"]+targets]

    for target in targets:
        if target not in df.columns: continue
        print(f"=== Correlation with {target} (top 10) ===")
        corrs = {}
        for f in features:
            valid = df[[f, target]].dropna()
            if len(valid) < 100: continue
            corrs[f] = valid.corr().iloc[0,1]
        for f, r in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:10]:
            print(f"  {f:30s} {r:+.4f}")
        print()

    for regime_name, cond_col, cond_val in [
        ("BTC above 50EMA", "BTC_close_4h", "BTC_EMA50_4h"),
        ("Price above 200EMA", "Close", "EMA200_4h"),
    ]:
        print(f"=== Regime: {regime_name} ===")
        if cond_col in df.columns and cond_val in df.columns:
            d = df.dropna(subset=[cond_col, cond_val, "fwd_return_1w"])
            mask = d[cond_col] > d[cond_val]
            print(f"  Above: {d.loc[mask,'fwd_return_1w'].mean():+.4%}  (n={mask.sum()})")
            print(f"  Below: {d.loc[~mask,'fwd_return_1w'].mean():+.4%}  (n={(~mask).sum()})")
        print()

    for col in ["RSI_4h","ADX_4h","MACD_hist","1h_RSI"]:
        if col not in df.columns: continue
        valid = df[[col, "fwd_return_1w"]].dropna()
        if len(valid) < 200: continue
        valid["decile"] = pd.qcut(valid[col], 10, duplicates="drop")
        avg = valid.groupby("decile")["fwd_return_1w"].mean()
        print(f"=== {col} deciles → 1w return ===")
        for interval, ret in avg.items():
            print(f"  {str(interval):>20s} {ret:+.4%}")
        print()

    print("=== Max Adverse Excursion (next 1w) ===")
    if "Low" in df.columns and "Close" in df.columns:
        df2 = df.sort_values(["symbol","timestamp"]).copy()
        df2["future_low"] = df2.groupby("symbol")["Low"].transform(
            lambda x: x.shift(-42).rolling(42, min_periods=1).min()
        )
        valid = df2.dropna(subset=["future_low","Close","fwd_return_1w"])
        valid["mae"] = (valid["Close"] - valid["future_low"]) / valid["Close"]
        wins = valid[valid["fwd_return_1w"] > 0]
        losses = valid[valid["fwd_return_1w"] <= 0]
        print(f"Total valid: {len(valid)}, wins: {len(wins)}, losses: {len(losses)}")
        for name, sub in [("All", valid), ("Wins", wins), ("Losses", losses)]:
            if len(sub) == 0: continue
            print(f"\n{name} MAE percentiles:")
            for p in [50, 75, 90, 95]:
                print(f"  {p}%: {sub['mae'].quantile(p/100)*100:.2f}%")
        print()

    print("=== Best Single Split (1w return) ===")
    for col in ["RSI_4h","ADX_4h","MACD_hist"]:
        if col not in df.columns: continue
        valid = df[[col, "fwd_return_1w"]].dropna()
        if len(valid) < 500: continue
        vals = valid[col].values
        rets = valid["fwd_return_1w"].values
        thresholds = np.quantile(vals, np.linspace(0.1,0.9,9))
        best, best_t = 0, None
        for t in thresholds:
            mask = vals > t
            if mask.sum()<20 or (~mask).sum()<20: continue
            diff = abs(rets[mask].mean() - rets[~mask].mean())
            if diff > best:
                best, best_t = diff, t
        if best_t:
            print(f"{col}: threshold={best_t:.4f}  above={rets[vals>best_t].mean():+.4%}  below={rets[vals<=best_t].mean():+.4%}")

# ==================== MAIN ====================
if __name__ == "__main__":
    if extraction_needed():
        run_extraction()
    else:
        print(f"Found existing {OUTPUT_CSV}, skipping extraction.\n")
    condensed_analysis()