#!/usr/bin/env python3
"""
Raw Indicator Extractor – 2‑year historical dataset
Exports every 4H candle with all indicators for liquid momentum coins.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time, os, warnings
warnings.filterwarnings("ignore")

# ========== CONFIG ==========
START_DATE = "2021-01-01"        # fetch from this date onward
OUTPUT_FILE = "raw_indicator_data_2y.csv"
CACHE_DIR = "cache"              # local storage to avoid re-downloading

# ========== BLACKLIST ==========
BLACKLIST = {
    "QUQ", "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD",
    "LEO", "WBT"
}

# ========== UNIVERSE (current top momentum coins) ==========
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
        return [f"{sym}-USD" for sym in top_symbols]
    except Exception as e:
        print(f"CoinGecko error: {e}. Using fallback list.")
        fallback = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
                    "ADA-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD",
                    "AVAX-USD","SHIB-USD","UNI-USD","LTC-USD","ATOM-USD"]
        return fallback[:momentum_top]

print("🌍 Fetching current momentum universe...")
COINS = fetch_current_momentum_coins(limit=100, momentum_top=20)
print(f"Coins: {COINS}")

# ========== CACHE SYSTEM ==========
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_download(symbol, interval, start, end):
    """Download from Yahoo or load from cache."""
    fname = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start}_{end}.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        return df
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            df.to_csv(fname)
        return df
    except Exception as e:
        print(f"Error downloading {symbol} {interval}: {e}")
        return pd.DataFrame()

# ========== INDICATOR COMPUTATION ==========
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
        # Map daily values to 4H timestamps (forward fill)
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

        # 1H bullish momentum (of the last 1H candle within the 4H bar)
        df_1h_copy = df_1h.copy()
        df_1h_copy['bull_mom'] = (df_1h_copy['Close'] - df_1h_copy['Open']) / (df_1h_copy['High'] - df_1h_copy['Low'])
        # For each 4H bar, take the bull_mom of the most recent 1H candle
        bull_mom_aligned = df_1h_copy['bull_mom'].resample('4h').last().reindex(d.index, method='ffill')
        d['1h_bullish_momentum'] = bull_mom_aligned
    else:
        d['1h_RSI'] = np.nan
        d['1h_bullish_momentum'] = np.nan

    # Future returns for target variables
    d['fwd_return_1d'] = d['Close'].shift(-6) / d['Close'] - 1   # 6 * 4h = 1 day
    d['fwd_return_1w'] = d['Close'].shift(-42) / d['Close'] - 1  # 7 days
    d['fwd_return_2w'] = d['Close'].shift(-84) / d['Close'] - 1  # 14 days

    return d

# ========== MAIN EXTRACTION ==========
end_date = datetime.now().strftime("%Y-%m-%d")
all_data = []

print("📦 Downloading BTC 4H data for market context...")
btc_4h = cached_download("BTC-USD", "1h", START_DATE, end_date)  # we need 4h, but yf 4h not available so use 1h and resample
btc_4h = btc_4h.resample('4h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()

for i, sym in enumerate(COINS):
    print(f"\n⏳ Processing {sym} ({i+1}/{len(COINS)})")
    try:
        # Download 1H data for both 4H resampling and 1H indicators
        df_1h = cached_download(sym, "1h", START_DATE, end_date)
        if df_1h.empty:
            print(f"   No data for {sym}, skipping.")
            continue

        # Resample to 4H
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        # Daily data
        df_daily = cached_download(sym, "1d", START_DATE, end_date)
        if df_daily.empty:
            print(f"   No daily data for {sym}, skipping.")
            continue

        # Compute indicators
        result = compute_all_indicators(df_4h, df_daily, df_1h, btc_4h)
        result['symbol'] = sym
        all_data.append(result)
        print(f"   {len(result)} rows added.")
        time.sleep(0.5)  # be gentle to Yahoo API
    except Exception as e:
        print(f"   Error on {sym}: {e}")

if not all_data:
    print("❌ No data collected. Exiting.")
    exit(1)

final_df = pd.concat(all_data)
final_df = final_df.reset_index().sort_values(['symbol', 'timestamp'])
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Done! File saved to {OUTPUT_FILE} with {len(final_df)} rows.")
print("   Send this file to your AI quant for Nobel‑worthy model building.")