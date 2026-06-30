#!/usr/bin/env python3
"""
Raw Indicator Extractor – KuCoin API, 2‑year history
Exports every 4H candle with all indicators for liquid momentum coins.
"""

import sys, os, time, json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIG ==========
# 2 years of history (KuCoin can go further, but 2 years is a solid sample)
YEARS_BACK = 2
OUTPUT_FILE = "raw_indicator_data_2y.csv"
CACHE_DIR = "cache"

# ========== BLACKLIST ==========
BLACKLIST = {
    "QUQ", "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD",
    "LEO", "WBT"
}

# ========== UNIVERSE ==========
def fetch_current_momentum_coins(limit=100, momentum_top=20):
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
            if not symbol or symbol in BLACKLIST:
                continue
            roc_14 = coin.get("price_change_percentage_14d_in_currency")
            momentum = abs(roc_14) if roc_14 is not None else 0.0
            candidates.append((symbol, momentum))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in candidates[:momentum_top]]
        return [f"{sym}-USDT" for sym in top_symbols]   # KuCoin format
    except Exception as e:
        print(f"CoinGecko error: {e}. Using fallback list.")
        fallback = [
            "BTC-USDT","ETH-USDT","BNB-USDT","SOL-USDT","XRP-USDT",
            "ADA-USDT","DOGE-USDT","DOT-USDT","MATIC-USDT","LINK-USDT",
            "AVAX-USDT","SHIB-USDT","UNI-USDT","LTC-USDT","ATOM-USDT"
        ]
        return fallback[:momentum_top]

print("🌍 Fetching current momentum universe...")
COINS = fetch_current_momentum_coins(limit=100, momentum_top=20)
print(f"Coins: {COINS}")

# ========== Kucoin API helpers ==========
KUCOIN_BASE = "https://api.kucoin.com"
def get_kucoin_klines(symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch klines from KuCoin. Returns list of candles (list of lists).
    Timestamps in seconds, start_time/end_time as datetimes.
    """
    # Map interval to KuCoin format
    interval_map = {
        "1h": "1hour",
        "4h": "4hour",
    }
    kucoin_interval = interval_map.get(interval, interval)
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {
        "type": kucoin_interval,
        "symbol": symbol,
        "startAt": int(start_time.timestamp()),
        "endAt": int(end_time.timestamp()),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("code") != "200000":
            return []
        return data["data"]
    except Exception as e:
        print(f"   KuCoin API error: {e}")
        return []

def fetch_ohlcv_kucoin(symbol, interval, start_date, end_date):
    """
    Fetch all candles between start_date and end_date using pagination.
    Returns DataFrame with columns: open_time, Open, High, Low, Close, Volume.
    """
    all_candles = []
    chunk_end = end_date
    while chunk_end > start_date:
        chunk_start = chunk_end - timedelta(hours=1000 * 1)  # ~41 days at 1h, but API max 1000 candles
        if chunk_start < start_date:
            chunk_start = start_date
        candles = get_kucoin_klines(symbol, interval, chunk_start, chunk_end, limit=1000)
        if not candles:
            # If no data, move back and continue (maybe early coin)
            chunk_end = chunk_start
            continue
        # Convert to DataFrame rows
        for c in candles:
            ts = datetime.utcfromtimestamp(int(c[0]))
            row = {
                "open_time": ts,
                "Open": float(c[1]),
                "Close": float(c[2]),
                "High": float(c[3]),
                "Low": float(c[4]),
                "Volume": float(c[5]),
            }
            all_candles.append(row)
        # Next chunk: use the earliest timestamp in this batch minus 1 hour to avoid overlap
        earliest_ts = min(int(c[0]) for c in candles)
        chunk_end = datetime.utcfromtimestamp(earliest_ts) - timedelta(hours=1)
        time.sleep(0.2)   # be gentle
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df.set_index("open_time", inplace=True)
    return df

# ========== CACHE SYSTEM ==========
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_kucoin_fetch(symbol, interval, start_date, end_date):
    fname = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    if os.path.exists(fname):
        print(f"   Loading cached {fname}")
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        return df
    print(f"   Fetching KuCoin {symbol} {interval} from {start_date.date()} to {end_date.date()}...")
    df = fetch_ohlcv_kucoin(symbol, interval, start_date, end_date)
    if not df.empty:
        df.to_csv(fname)
    return df

# ========== INDICATOR COMPUTATION (same as before) ==========
def compute_all_indicators(df_4h, df_daily, df_1h, df_btc_4h):
    """Returns a DataFrame indexed by 4H timestamp with all indicators."""
    d = df_4h[['Open','High','Low','Close','Volume']].copy()
    d.index.name = 'timestamp'

    # 4H EMAs
    d['EMA50_4h'] = d['Close'].ewm(span=50, adjust=False).mean()
    d['EMA200_4h'] = d['Close'].ewm(span=200, adjust=False).mean()

    # ATR
    h, l, c = d['High'], d['Low'], d['Close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR_4h'] = tr.rolling(14).mean()

    # RSI 4H
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    d['RSI_4h'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    d['MACD_line'] = macd_line
    d['MACD_signal'] = macd_signal
    d['MACD_hist'] = macd_line - macd_signal

    # ADX
    dm_plus = h.diff()
    dm_minus = -l.diff()
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    atr_ewm = tr.ewm(alpha=1/14, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr_ewm)
    di_minus = 100 * (dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr_ewm)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    d['ADX_4h'] = dx.ewm(alpha=1/14, adjust=False).mean()
    d['DI_plus'] = di_plus
    d['DI_minus'] = di_minus

    # Volume surge
    d['Volume_surge'] = d['Volume'] > (d['Volume'].shift(1).rolling(5).mean() * 1.2)

    # Support/Resistance
    d['SR_High_20'] = d['High'].rolling(20).max()
    d['SR_Low_20'] = d['Low'].rolling(20).min()

    # Daily context aligned to 4H
    if not df_daily.empty:
        daily_ema50 = df_daily['Close'].ewm(span=50, adjust=False).mean()
        daily_ema200 = df_daily['Close'].ewm(span=200, adjust=False).mean()
        daily_ema50_aligned = daily_ema50.reindex(d.index, method='ffill')
        daily_ema200_aligned = daily_ema200.reindex(d.index, method='ffill')
        d['daily_EMA50'] = daily_ema50_aligned
        d['daily_EMA200'] = daily_ema200_aligned
    else:
        d['daily_EMA50'] = np.nan
        d['daily_EMA200'] = np.nan

    # BTC 4H context
    if not df_btc_4h.empty:
        btc_close = df_btc_4h['Close'].reindex(d.index, method='ffill')
        btc_ema50 = df_btc_4h['Close'].ewm(span=50, adjust=False).mean().reindex(d.index, method='ffill')
        d['BTC_close_4h'] = btc_close
        d['BTC_EMA50_4h'] = btc_ema50
    else:
        d['BTC_close_4h'] = np.nan
        d['BTC_EMA50_4h'] = np.nan

    # 1H indicators (resampled to 4H timestamps)
    if not df_1h.empty:
        # RSI 1H
        delta_1h = df_1h['Close'].diff()
        gain_1h = delta_1h.where(delta_1h > 0, 0.0)
        loss_1h = -delta_1h.where(delta_1h < 0, 0.0)
        avg_gain_1h = gain_1h.ewm(alpha=1/14, adjust=False).mean()
        avg_loss_1h = loss_1h.ewm(alpha=1/14, adjust=False).mean()
        rs_1h = avg_gain_1h / avg_loss_1h
        rsi_1h = 100 - (100 / (1 + rs_1h))
        rsi_1h_aligned = rsi_1h.resample('4h').last().reindex(d.index, method='ffill')
        d['1h_RSI'] = rsi_1h_aligned

        df_1h_copy = df_1h.copy()
        df_1h_copy['bull_mom'] = (df_1h_copy['Close'] - df_1h_copy['Open']) / (df_1h_copy['High'] - df_1h_copy['Low'])
        bull_mom_aligned = df_1h_copy['bull_mom'].resample('4h').last().reindex(d.index, method='ffill')
        d['1h_bullish_momentum'] = bull_mom_aligned
    else:
        d['1h_RSI'] = np.nan
        d['1h_bullish_momentum'] = np.nan

    # Future returns
    d['fwd_return_1d'] = d['Close'].shift(-6) / d['Close'] - 1   # 6 * 4h = 1 day
    d['fwd_return_1w'] = d['Close'].shift(-42) / d['Close'] - 1  # 7 days
    d['fwd_return_2w'] = d['Close'].shift(-84) / d['Close'] - 1  # 14 days

    return d

# ========== MAIN ==========
def main():
    # time window: last YEARS_BACK years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_BACK)
    print(f"⏱️ Data window: {start_date.date()} → {end_date.date()}")

    # --- BTC data (market context) ---
    print("📦 Fetching BTC 1h data (for resampling to 4h)...")
    btc_1h = cached_kucoin_fetch("BTC-USDT", "1h", start_date, end_date)
    if btc_1h.empty:
        print("❌ BTC data unavailable. Exiting.")
        sys.exit(1)
    btc_4h = btc_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # --- BTC daily data for daily indicators of BTC (not strictly needed but helpful) ---
    print("📦 Fetching BTC daily data...")
    btc_daily = cached_kucoin_fetch("BTC-USDT", "1d", start_date, end_date)

    all_coin_data = []

    for i, sym in enumerate(COINS):
        print(f"\n⏳ Processing {sym} ({i+1}/{len(COINS)})")
        try:
            # 1H data
            df_1h = cached_kucoin_fetch(sym, "1h", start_date, end_date)
            if df_1h.empty:
                print(f"   No 1H data for {sym}, skipping.")
                continue

            # Resample to 4H
            df_4h = df_1h.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            # Daily data
            df_daily = cached_kucoin_fetch(sym, "1d", start_date, end_date)
            if df_daily.empty:
                print(f"   No daily data for {sym}, skipping.")
                continue

            # Compute indicators
            result = compute_all_indicators(df_4h, df_daily, df_1h, btc_4h)
            result['symbol'] = sym
            all_coin_data.append(result)
            print(f"   {len(result)} rows added.")
            time.sleep(0.2)
        except Exception as e:
            print(f"   Error on {sym}: {e}")

    if not all_coin_data:
        print("❌ No data collected. Exiting.")
        sys.exit(1)

    final_df = pd.concat(all_coin_data)
    final_df = final_df.reset_index().sort_values(['symbol', 'timestamp'])
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Done! File saved to {OUTPUT_FILE} with {len(final_df)} rows.")
    print("   Upload this artifact and send the CSV for quant analysis.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].upper() == "FETCH":
        main()
    else:
        print("Usage: python backtest_ku.py FETCH")