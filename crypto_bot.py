#!/usr/bin/env python3
"""
Crypto Swing Bot – 4H, 4 TP levels (0.5/1.0/1.5/2.0R)
Full position closed by trailing stop or final TP (2R) – no partials.
Trailing stop moves: 0.5R → breakeven, 1.0R → 0.5R, 1.5R → 1.0R
"""

import requests, json, os, traceback, math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ========== ENVIRONMENT ==========
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set – AI filtering disabled.")

# ========== BLACKLIST ==========
BLACKLIST = {
    "QUQ", "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD"
}

# ========== CHALLENGE PARAMETERS ==========
RISK_PERCENT = 0.10
DAILY_LOSS_PCT = 0.20
MAX_RISKY_TRADES = 2
MIN_SCORE = 8.0
TP_MULTIPLIER = 2.0           # final TP is 2R (TP4)
MIN_NOTIONAL = 1.0

# ========== DYNAMIC COIN LIST ==========
def fetch_top_liquid_coins(limit=50):
    global COIN_RANK
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        yahoo_symbols = []
        COIN_RANK = {}
        rank = 1
        for coin in data:
            symbol = coin.get("symbol", "").upper()
            if symbol and symbol not in BLACKLIST:
                ys = f"{symbol}-USD"
                if ys not in yahoo_symbols:
                    yahoo_symbols.append(ys)
                    COIN_RANK[ys] = rank
                    rank += 1
        print(f"Fetched {len(yahoo_symbols)} coins (blacklist filtered)")
        return yahoo_symbols[:limit]
    except Exception as e:
        print(f"CoinGecko API failed: {e}. Using fallback.")
        fallback = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
                    "ADA-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD"]
        COIN_RANK = {sym: i+1 for i, sym in enumerate(fallback)}
        return fallback[:limit]

COIN_RANK = {}
CRYPTO_PAIRS = fetch_top_liquid_coins(50)

# ========== PORTFOLIO ==========
PORTFOLIO_FILE = "crypto_portfolio.json"
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f: data = json.load(f)
            return {
                "balance": data.get("balance", 20.0),
                "realized_pnl": data.get("realized_pnl", 0.0),
                "open_positions": data.get("open_positions", 0)
            }
        except: pass
    return {"balance": 20.0, "realized_pnl": 0.0, "open_positions": 0}

def save_portfolio(p):
    try:
        with open(PORTFOLIO_FILE, "w") as f: json.dump(p, f, indent=2)
    except: pass

portfolio = load_portfolio()

# ========== CSV LOGGING ==========
TRADE_LOG_CSV = "crypto_trade_log.csv"
OPEN_TRADES_CSV = "crypto_open_trades.csv"
TRADE_RESULTS_CSV = "crypto_trade_results.csv"
PERF_COUNTER_FILE = "perf_counter.txt"

def init_csv(f, cols):
    if not os.path.exists(f): pd.DataFrame(columns=cols).to_csv(f, index=False)

def append_csv(f, df_new):
    try:
        existing = pd.read_csv(f)
        updated = pd.concat([existing, df_new], ignore_index=True)
    except: updated = df_new
    updated.to_csv(f, index=False)

def save_csv(f, df): df.to_csv(f, index=False)

def initialize_trade_files():
    init_csv(TRADE_LOG_CSV, ["timestamp","symbol","action","entry","stop",
                             "TP","score","ai_approved"])
    init_csv(OPEN_TRADES_CSV, ["timestamp","symbol","action","entry","stop",
                               "TP","status","quantity","original_qty",
                               "highest_tp","breakeven"])
    init_csv(TRADE_RESULTS_CSV, ["timestamp","symbol","action","entry","stop",
                                 "TP","status","hit_level","close_time",
                                 "exit_price","quantity","pnl"])

def log_signal(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP": sig["take_profit"], "score": sig["score"],
           "ai_approved": sig.get("ai_approved", False)}
    append_csv(TRADE_LOG_CSV, pd.DataFrame([row]))

def add_open_trade(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP": sig["take_profit"], "status": "open",
           "quantity": sig["quantity"], "original_qty": sig["quantity"],
           "highest_tp": -1, "breakeven": False}
    append_csv(OPEN_TRADES_CSV, pd.DataFrame([row]))

# ========== PORTFOLIO HELPERS ==========
def daily_pnl():
    try:
        df = pd.read_csv(TRADE_RESULTS_CSV)
        if df.empty: return 0.0
        today = datetime.now().strftime("%Y-%m-%d")
        df['close_time'] = pd.to_datetime(df['close_time'])
        daily = df[df['close_time'].dt.strftime("%Y-%m-%d") == today]
        return daily['pnl'].sum() if not daily.empty else 0.0
    except: return 0.0

def update_portfolio(trade_result):
    portfolio['balance'] += trade_result['pnl']
    portfolio['realized_pnl'] += trade_result['pnl']
    save_portfolio(portfolio)

# ========== DAILY LOSS LIMIT ==========
DAILY_BALANCE_FILE = "daily_start_balance.txt"

def get_daily_start_balance():
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(DAILY_BALANCE_FILE, 'r') as f:
            lines = f.read().strip().split(',')
            if lines[0] == today_str:
                return float(lines[1])
    except:
        pass
    balance = portfolio['balance']
    with open(DAILY_BALANCE_FILE, 'w') as f:
        f.write(f"{today_str},{balance}")
    return balance

# ========== SYMBOL CONVERTER ==========
def to_yahoo(sym):
    clean = sym.replace("-USD", "").replace("USDT", "").replace("-USDT", "")
    clean = clean.strip("-")
    return f"{clean}-USD"

def yahoo_to_kucoin(sym_yahoo):
    base = sym_yahoo.replace("-USD", "")
    return f"{base}-USDT"

# ========== KUCOIN DATA FETCH ==========
def get_kucoin_klines(sym_kucoin, interval, limit=100, start_time=None, end_time=None):
    interval_map = {'1h': '1hour', '4h': '4hour', '1d': '1day'}
    kucoin_interval = interval_map.get(interval, interval)
    base_url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": kucoin_interval, "symbol": sym_kucoin}
    if start_time: params["startAt"] = int(start_time.timestamp())
    if end_time: params["endAt"] = int(end_time.timestamp())
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        data = resp.json()
        if data.get("code") != "200000": return pd.DataFrame()
        candles = data["data"]
        if not candles: return pd.DataFrame()
        rows = []
        for c in candles:
            ts = datetime.utcfromtimestamp(int(c[0]))
            rows.append({'open_time': ts,
                         'Open': float(c[1]), 'Close': float(c[2]),
                         'High': float(c[3]), 'Low': float(c[4]), 'Volume': float(c[5])})
        df = pd.DataFrame(rows).set_index('open_time').sort_index()
        df = df[['Open','High','Low','Close','Volume']]
        if len(df) > limit: df = df.tail(limit)
        return df
    except Exception as e:
        print(f"KuCoin error for {sym_kucoin}: {e}")
        return pd.DataFrame()

# ========== YAHOO FALLBACK ==========
def get_yahoo_klines(sym_yahoo, interval, days=14, start=None, end=None):
    if start is None:
        end = datetime.now()
        start = end - timedelta(days=days)
    else:
        end = end if end else datetime.now()
    try:
        df = yf.download(sym_yahoo, start=start, end=end, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

def get_hybrid_klines(sym_yahoo, interval, days=14, start=None, end=None):
    kucoin_sym = yahoo_to_kucoin(sym_yahoo)
    df = get_kucoin_klines(kucoin_sym, interval, limit=500 if interval == '1h' else 100,
                           start_time=start, end_time=end)
    if not df.empty:
        print(f"Using KuCoin data for {sym_yahoo}")
        return df
    print(f"KuCoin failed/unavailable for {sym_yahoo}, falling back to Yahoo")
    df = get_yahoo_klines(sym_yahoo, interval, days=days, start=start, end=end)
    if not df.empty: return df
    if interval == '1h':
        print(f"Yahoo 1h empty for {sym_yahoo}, trying 4h")
        df = get_yahoo_klines(sym_yahoo, '4h', days=days, start=start, end=end)
    return df

# ========== TECHNICAL INDICATORS ==========
def ema(series, period): return series.ewm(span=period, adjust=False).mean()

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
    dm_plus = h.diff(); dm_minus = -l.diff()
    dm_plus[dm_plus < 0] = 0; dm_minus[dm_minus < 0] = 0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

# ========== ENHANCED S/R DETECTION ==========
def strong_support_resistance(df, direction, lookback=50, min_touches=2, tolerance=0.005):
    price = df['Close'].iloc[-1]
    if direction == "LONG":
        levels = df['Low'].tail(lookback)
        rounded = (levels / price).round(3)
        vc = rounded.value_counts()
        valid = vc[vc >= min_touches].index.tolist()
        if not valid:
            return False, None
        supports = [level * price for level in valid if level * price < price]
        if not supports:
            return False, None
        support = max(supports)
        return True, support
    else:
        levels = df['High'].tail(lookback)
        rounded = (levels / price).round(3)
        vc = rounded.value_counts()
        valid = vc[vc >= min_touches].index.tolist()
        if not valid:
            return False, None
        resistances = [level * price for level in valid if level * price > price]
        if not resistances:
            return False, None
        resistance = min(resistances)
        return True, resistance

# ========== SCORING (same robust filters) ==========
def score_pair_2R(pair):
    layers = {}
    df_d = get_yahoo_klines(pair, '1d', days=90)
    if df_d.empty or len(df_d) < 50:
        return 0, None, None, None, None, {"Daily data": (0,0,"FAIL: insufficient daily candles")}
    df_4h = get_hybrid_klines(pair, '4h', days=14)
    if df_4h.empty or len(df_4h) < 50:
        return 0, None, None, None, None, {"4h data": (0,0,"FAIL: insufficient 4h candles")}
    df_1h = get_hybrid_klines(pair, '1h', days=3)
    if df_1h.empty or len(df_1h) < 10:
        return 0, None, None, None, None, {"1h data": (0,0,"FAIL: insufficient 1h candles")}

    price = df_4h['Close'].iloc[-1]
    atr_val = atr(df_4h)
    if atr_val is None or atr_val <= 0:
        return 0, None, None, None, None, {"ATR": (0,0,"FAIL: ATR invalid")}

    ema50_d = ema(df_d['Close'], 50)
    ema200_d = ema(df_d['Close'], 200)
    if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
        trend_daily = 1
    elif price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
        trend_daily = -1
    else:
        ema50_4h = ema(df_4h['Close'], 50)
        ema200_4h = ema(df_4h['Close'], 200)
        if price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]:
            trend_daily = 1
        elif price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]:
            trend_daily = -1
        else:
            return 0, None, None, None, None, {"Daily trend": (0,0,"FAIL: no clear trend")}

    direction = "LONG" if trend_daily == 1 else "SHORT"

    raw_stop = atr_val * 2.5
    if direction == "LONG":
        stop = price - raw_stop
    else:
        stop = price + raw_stop
    risk = abs(price - stop)
    tp_final = price + TP_MULTIPLIER * risk if direction == "LONG" else price - TP_MULTIPLIER * risk
    if abs(tp_final - price) > 5.5 * atr_val:
        return 0, direction, price, atr_val, None, {"2R Feasibility": (0,0,"FAIL: TP > 5.5x ATR")}

    # Soft exhaustion penalty
    exhaustion_penalty = 0
    dist_ema50 = 0
    if direction == "LONG":
        dist_ema50 = (price - ema50_d.iloc[-1]) / ema50_d.iloc[-1] * 100
        if dist_ema50 > 6.0:
            return 0, direction, price, atr_val, None, {"Trend Exhaustion": (0,0,"FAIL: extended >6% above EMA50")}
        elif dist_ema50 > 3.5:
            exhaustion_penalty = 1.5
    else:
        dist_ema50 = (ema50_d.iloc[-1] - price) / ema50_d.iloc[-1] * 100
        if dist_ema50 > 6.0:
            return 0, direction, price, atr_val, None, {"Trend Exhaustion": (0,0,"FAIL: extended >6% below EMA50")}
        elif dist_ema50 > 3.5:
            exhaustion_penalty = 1.5

    # ---- Layers ----
    ema50_4h = ema(df_4h['Close'], 50)
    ema200_4h = ema(df_4h['Close'], 200)
    if direction == "LONG":
        ema_ok = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
    else:
        ema_ok = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
    layers["EMA Align"] = (2.5 if ema_ok else 0, 2.5, "OK" if ema_ok else "FAIL")

    adx_val, di_plus, di_minus = adx(df_4h)
    adx_dir = (di_plus > di_minus) if direction == "LONG" else (di_minus > di_plus)
    adx_ok = adx_val > 22 and adx_dir
    layers["ADX"] = (2.0 if adx_ok else 0, 2.0, "OK" if adx_ok else "FAIL")

    rsi_val = rsi(df_4h)
    if rsi_val is not None:
        rsi_ok = rsi_val > 50 if direction == "LONG" else rsi_val < 50
        layers["RSI 4h"] = (1.5 if rsi_ok else 0, 1.5, "OK" if rsi_ok else "FAIL")
    else:
        layers["RSI 4h"] = (0, 1.5, "FAIL: NaN")

    macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
    macd_expanding = (direction=="LONG" and macd_hist>0 and macd_hist>macd_hist_prev) or \
                     (direction=="SHORT" and macd_hist<0 and macd_hist<macd_hist_prev)
    layers["MACD"] = (1.5 if macd_expanding else 0, 1.5, "OK" if macd_expanding else "FAIL")

    sr_ok, sr_level = strong_support_resistance(df_4h, direction, lookback=50, min_touches=2)
    if sr_ok and sr_level:
        if direction == "LONG":
            valid = price - sr_level <= atr_val * 1.2
        else:
            valid = sr_level - price <= atr_val * 1.2
        layers["S/R"] = (1.5 if valid else 0, 1.5, "OK" if valid else "FAIL")
    else:
        layers["S/R"] = (0, 1.5, "FAIL: no strong level")

    last_candle_1h = df_1h.iloc[-1]
    candle_range = last_candle_1h['High'] - last_candle_1h['Low']
    bullish_mom = (last_candle_1h['Close'] - last_candle_1h['Open']) / candle_range if candle_range > 0 else 0
    mom_ok = (bullish_mom > 0.6) if direction=="LONG" else (bullish_mom < -0.6)
    layers["Candle Mom"] = (1.0 if mom_ok else 0, 1.0, "OK" if mom_ok else "FAIL")

    prev_candle_1h = df_1h.iloc[-2]
    micro_up = last_candle_1h['Close'] > last_candle_1h['Open'] and prev_candle_1h['Close'] > prev_candle_1h['Open']
    micro_down = last_candle_1h['Close'] < last_candle_1h['Open'] and prev_candle_1h['Close'] < prev_candle_1h['Open']
    micro_ok = (micro_up if direction=="LONG" else micro_down)
    layers["Micro Trend"] = (1.0 if micro_ok else 0, 1.0, "OK" if micro_ok else "FAIL")

    vol_last = df_4h['Volume'].iloc[-1]
    vol_avg20 = df_4h['Volume'].iloc[-21:-1].mean() if len(df_4h) >= 21 else df_4h['Volume'].mean()
    vol_ok = vol_last > vol_avg20 * 1.5 if vol_avg20 > 0 else False
    layers["Volume"] = (1.0 if vol_ok else 0, 1.0, "OK" if vol_ok else "FAIL")

    btc_df = get_hybrid_klines("BTC-USD", '4h', days=14)
    market_ok = False
    if not btc_df.empty and len(btc_df) >= 50:
        btc_ema50 = ema(btc_df['Close'], 50)
        btc_trend_up = btc_df['Close'].iloc[-1] > btc_ema50.iloc[-1]
        if direction == "LONG" and btc_trend_up:
            market_ok = True
        elif direction == "SHORT" and not btc_trend_up:
            market_ok = True
    layers["Market"] = (0.5 if market_ok else 0, 0.5, "OK" if market_ok else "FAIL")

    atr_pct = atr_val / price
    atr_ok = atr_pct > 0.008
    layers["ATR"] = (1.0 if atr_ok else 0, 1.0, "OK" if atr_ok else "FAIL")

    rsi_d = rsi(df_d, 14)
    macd_d_line, macd_d_signal, macd_d_hist, macd_d_hist_prev = macd(df_d)
    daily_points = 0
    if direction == "LONG":
        if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
            daily_points += 1.0
        if rsi_d is not None and rsi_d > 50:
            daily_points += 0.5
        if macd_d_hist > 0 and macd_d_hist > macd_d_hist_prev:
            daily_points += 0.5
    else:
        if price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
            daily_points += 1.0
        if rsi_d is not None and rsi_d < 50:
            daily_points += 0.5
        if macd_d_hist < 0 and macd_d_hist < macd_d_hist_prev:
            daily_points += 0.5
    layers["Daily Conf."] = (daily_points, 2.0, "OK")

    if exhaustion_penalty > 0:
        layers["Trend Exhaustion"] = (-exhaustion_penalty, 0, f"WARN: {dist_ema50:.1f}% extended, penalty -{exhaustion_penalty}")
    else:
        layers["Trend Exhaustion"] = (0, 0, "OK")

    total = sum(score for score,_,_ in layers.values())
    total -= exhaustion_penalty
    total = max(0, total)

    return total, direction, price, atr_val, (sr_level if sr_ok else None), layers

# ========== AI CONFIRMATION ==========
def ai_confirm_trade(signal_dict):
    if not GROQ_API_KEY: return True
    prompt = (
        f"Crypto trade setup:\nPair: {signal_dict['symbol']}\nDirection: {signal_dict['action']}\n"
        f"Entry: {signal_dict['limit_price']:.5f}\nStop: {signal_dict['stop_loss']:.5f}\n"
        f"TP (2R): {signal_dict['take_profit']:.5f}\nScore: {signal_dict['score']:.1f}/{11.5}\n"
        f"Will this trade likely hit 2R before hitting the stop? Answer PASS or FAIL."
    )
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [
                {"role":"system","content":"You are a professional crypto analyst. Respond with only PASS or FAIL."},
                {"role":"user","content": prompt}], "temperature":0.1, "max_tokens":5}, timeout=15)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return "FAIL" not in text
    except: pass
    return True

# ========== SIGNAL GENERATION ==========
def generate_signal():
    risky_count = 0
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
        if not open_df.empty:
            if "breakeven" in open_df.columns:
                risky = open_df[open_df["breakeven"] == False]
            else:
                risky = open_df
            risky_count = len(risky)
    except: pass

    if risky_count >= MAX_RISKY_TRADES:
        print(f"Max {MAX_RISKY_TRADES} risky trades limit reached ({risky_count}). No new signals.")
        return None, [], {}, 0, risky_count

    open_symbols_risky = set()
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
        if not open_df.empty:
            if "symbol" in open_df.columns:
                open_df["symbol"] = open_df["symbol"].apply(to_yahoo)
            if "breakeven" in open_df.columns:
                risky = open_df[open_df["breakeven"] == False]
            else:
                risky = open_df
            open_symbols_risky = set(risky["symbol"].values)
    except: pass

    all_scored = []
    data_failures = 0
    for pair in CRYPTO_PAIRS:
        if pair in open_symbols_risky: continue
        score, direction, price, atr_val, swing_level, layers = score_pair_2R(pair)
        if direction is None:
            data_failures += 1
            continue
        all_scored.append((pair, score, direction, price, atr_val, swing_level, layers))

    all_scored.sort(key=lambda x: x[1], reverse=True)
    top5 = all_scored[:5]
    top_overall = top5[0] if top5 else None

    print(f"Scored pairs: {len(all_scored)}, data failures: {data_failures}")

    candidates = [item for item in all_scored if item[1] >= MIN_SCORE]
    if not candidates:
        return None, top5, top_overall[6] if top_overall else {}, data_failures, risky_count

    pair, score, direction, price, atr_val, swing_level, layers = candidates[0]

    # Stop placement
    raw_stop = atr_val * 2.5
    if direction == "LONG":
        stop = price - raw_stop
        if swing_level and swing_level > price - raw_stop * 1.2:
            stop = min(stop, swing_level - 0.05 * atr_val)
    else:
        stop = price + raw_stop
        if swing_level and swing_level < price + raw_stop * 1.2:
            stop = max(stop, swing_level + 0.05 * atr_val)
    risk = abs(price - stop)

    # ---- 4 TP levels: 0.5 / 1.0 / 1.5 / 2.0R ----
    tp_multipliers = [0.5, 1.0, 1.5, 2.0]
    take_profits = [price + m * risk if direction == "LONG" else price - m * risk for m in tp_multipliers]
    final_tp = take_profits[-1]   # 2R

    quantity = round((portfolio['balance'] * RISK_PERCENT) / risk, 8)
    if quantity * price < MIN_NOTIONAL:
        print(f"Skipping {pair}: trade value {quantity*price:.2f} < ${MIN_NOTIONAL}")
        return None, top5, layers, data_failures, risky_count

    signal = {
        "action": direction,
        "symbol": pair,
        "quantity": quantity,
        "limit_price": price,
        "stop_loss": stop,
        "take_profits": take_profits,    # list of 4 prices
        "take_profit": final_tp,         # for CSV log (2R)
        "score": score,
        "atr": atr_val,
        "layers": layers
    }
    if not ai_confirm_trade(signal):
        print(f"AI rejected {pair} {direction} (score {score:.1f})")
        return None, top5, layers, data_failures, risky_count
    signal["ai_approved"] = True
    return signal, top5, layers, data_failures, risky_count

# ========== DISCORD HELPERS ==========
def send_discord_message(text):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": text[:2000]}, timeout=10)
    except Exception as e: print("Discord text error:", e)

def send_discord_image(image_path, caption=""):
    if not os.path.exists(image_path): return
    try:
        with open(image_path, 'rb') as img:
            files = {'file': img}
            payload = {'content': caption[:2000]} if caption else {}
            resp = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=15)
            print(f"Image sent, status: {resp.status_code}")
    except Exception as e: print("Discord image error:", e)

# ========== STEPPED TRAILING STOP (4 TP) ==========
def get_trailing_stop(entry, risk, direction, highest_price, lowest_price):
    """
    0.5R move → breakeven
    1.0R move → stop at 0.5R
    1.5R move → stop at 1.0R
    """
    if direction == "LONG":
        move = highest_price - entry
        if move >= 1.5 * risk:
            return entry + 1.0 * risk
        elif move >= 1.0 * risk:
            return entry + 0.5 * risk
        elif move >= 0.5 * risk:
            return entry   # breakeven
        else:
            return entry - risk   # original stop
    else:
        move = entry - lowest_price
        if move >= 1.5 * risk:
            return entry - 1.0 * risk
        elif move >= 1.0 * risk:
            return entry - 0.5 * risk
        elif move >= 0.5 * risk:
            return entry
        else:
            return entry + risk

# ========== TRADE MANAGEMENT ==========
def check_open_trades():
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
    except: return
    if open_df.empty: return

    open_df["symbol"] = open_df["symbol"].apply(to_yahoo)
    save_csv(OPEN_TRADES_CSV, open_df)

    for col in ["highest_tp","quantity","original_qty","breakeven"]:
        if col not in open_df.columns:
            open_df[col] = -1 if col=="highest_tp" else (False if col=="breakeven" else 0.0)

    results = []; still_open = []; alerts = []
    now = datetime.now()

    for idx, trade in open_df.iterrows():
        try:
            sym = trade["symbol"]; direction = trade["action"]
            entry = float(trade["entry"]); original_stop = float(trade["stop"])
            final_tp = float(trade["TP"])
            qty = float(trade["quantity"])
            try: entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
            except: still_open.append(trade); continue

            df_1h = get_hybrid_klines(sym, '1h', start=entry_time, end=now)
            if df_1h.empty:
                still_open.append(trade)
                continue

            risk = abs(entry - original_stop)
            highest_seen = entry if direction == "LONG" else -entry
            lowest_seen = entry if direction == "SHORT" else float('inf')
            trade_closed = False
            exit_price = None
            hit_level = ""

            for candle_time, candle in df_1h.iterrows():
                high = candle['High']; low = candle['Low']

                # Hard exit at final TP (2R)
                if direction == "LONG" and high >= final_tp:
                    exit_price = final_tp; hit_level = "TP4 (2R)"; trade_closed = True; break
                if direction == "SHORT" and low <= final_tp:
                    exit_price = final_tp; hit_level = "TP4 (2R)"; trade_closed = True; break

                if direction == "LONG":
                    highest_seen = max(highest_seen, high)
                else:
                    lowest_seen = min(lowest_seen, low)

                current_stop = get_trailing_stop(entry, risk, direction, highest_seen, lowest_seen)

                if direction == "LONG" and low <= current_stop:
                    exit_price = current_stop
                    hit_level = "TRAILING STOP" if current_stop != original_stop else "STOP LOSS"
                    trade_closed = True
                    break
                if direction == "SHORT" and high >= current_stop:
                    exit_price = current_stop
                    hit_level = "TRAILING STOP" if current_stop != original_stop else "STOP LOSS"
                    trade_closed = True
                    break

            if not trade_closed:
                trade["quantity"] = qty
                still_open.append(trade)
                continue

            pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
            pnl = round(pnl, 4)

            result_row = trade.to_dict()
            result_row["hit_level"] = hit_level
            result_row["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            result_row["exit_price"] = exit_price
            result_row["quantity"] = qty
            result_row["pnl"] = pnl
            results.append(result_row)
            update_portfolio({'pnl': pnl})

            if "TP4" in hit_level:
                msg = f"🎯 **TP4 (2R) Hit!** {sym} {direction} closed at {exit_price:.6f}. P&L: {pnl:.2f} USDT"
            elif "TRAILING" in hit_level:
                msg = f"🛑 **Trailing Stop Hit** {sym} {direction} closed at {exit_price:.6f}. P&L: {pnl:.2f} USDT"
            else:
                msg = f"🔴 **Stop Loss Hit** {sym} {direction} closed at {exit_price:.6f}. P&L: {pnl:.2f} USDT"
            alerts.append(msg)
            print("ALERT:", msg)
            send_discord_message(msg)
            send_trade_close_chart(trade, hit_level, exit_price, pnl)

        except Exception as e:
            print(f"Error processing trade {trade.get('symbol','?')}: {e}")
            still_open.append(trade)

    if results:
        append_csv(TRADE_RESULTS_CSV, pd.DataFrame(results))
    if still_open:
        save_csv(OPEN_TRADES_CSV, pd.DataFrame(still_open))
        portfolio['open_positions'] = len(still_open)
    else:
        save_csv(OPEN_TRADES_CSV, pd.DataFrame())
        portfolio['open_positions'] = 0
    save_portfolio(portfolio)

    check_and_send_perf_report()

    risky_count = sum(1 for t in still_open if t.get("breakeven", False) == False)
    be_count = len(still_open) - risky_count
    summary = f"🔍 Open trades: {risky_count} risky, {be_count} breakeven (total {len(still_open)})"
    print(summary); send_discord_message(summary)

# ---------- PERFORMANCE REPORT ----------
def get_completed_trades():
    try:
        df = pd.read_csv(TRADE_RESULTS_CSV)
    except: return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    trade_groups = df.groupby(['timestamp', 'symbol'])
    trades = []
    for (ts, sym), group in trade_groups:
        total_pnl = group['pnl'].sum()
        trades.append({'timestamp': ts, 'symbol': sym, 'total_pnl': total_pnl,
                       'action': group['action'].iloc[0]})
    trade_df = pd.DataFrame(trades)
    trade_df['is_win'] = trade_df['total_pnl'] > 0
    trade_df['is_loss'] = trade_df['total_pnl'] < 0
    trade_df['is_breakeven'] = trade_df['total_pnl'] == 0
    return trade_df

def check_and_send_perf_report():
    trade_df = get_completed_trades()
    if trade_df.empty: return
    total_trades = len(trade_df)
    last_reported = 0
    if os.path.exists(PERF_COUNTER_FILE):
        try:
            with open(PERF_COUNTER_FILE, 'r') as f:
                last_reported = int(f.read().strip())
        except: pass
    current_milestone = (total_trades // 10) * 10
    if current_milestone <= last_reported:
        return

    wins = trade_df[trade_df['is_win']]
    losses = trade_df[trade_df['is_loss']]
    total_wins = len(wins); total_losses = len(losses)
    winrate = (total_wins / max(total_wins + total_losses, 1)) * 100
    total_pnl = trade_df['total_pnl'].sum()
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if total_losses > 0 else float('inf')

    current_win_streak = 0; current_loss_streak = 0
    for _, row in trade_df.iloc[::-1].iterrows():
        if row['is_win']:
            if current_loss_streak == 0: current_win_streak += 1
            else: break
        elif row['is_loss']:
            if current_win_streak == 0: current_loss_streak += 1
            else: break
        else: break

    best_trade = trade_df.loc[trade_df['total_pnl'].idxmax()]
    worst_trade = trade_df.loc[trade_df['total_pnl'].idxmin()]

    report = (
        "📊 **Performance Report** – All Time ({} closed trades)\n\n"
        "**Total P&L:** {:.2f} USDT\n"
        "**Winrate:** {:.1f}% ({}W / {}L)\n"
        "**Profit Factor:** {:.2f}\n"
        "**Current Win Streak:** {} 🔥\n"
        "**Current Loss Streak:** {} 😞\n"
        "**Best Trade:** {} {} {:.2f} USDT\n"
        "**Worst Trade:** {} {} {:.2f} USDT\n"
    ).format(
        total_trades, total_pnl, winrate, total_wins, total_losses,
        profit_factor, current_win_streak, current_loss_streak,
        best_trade['symbol'], best_trade['action'], best_trade['total_pnl'],
        worst_trade['symbol'], worst_trade['action'], worst_trade['total_pnl']
    )
    send_discord_message(report)
    with open(PERF_COUNTER_FILE, 'w') as f:
        f.write(str(current_milestone))

def send_trade_close_chart(trade, hit_level, exit_price, pnl):
    sym = trade["symbol"]
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import mplfinance as mpf
        entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        df = get_hybrid_klines(sym, '1h', start=entry_time, end=datetime.now())
        if df.empty: return
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        fig, ax = mpf.plot(df, type='candle', style=mpf_style,
                           title=f"{sym} {trade['action']} – {hit_level} (PnL: {pnl:.2f}$)", ylabel='Price',
                           returnfig=True, figsize=(8,6))
        entry = float(trade["entry"]); stop = float(trade["stop"]); tp = float(trade["TP"])
        ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
        ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Initial Stop')
        ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1.5, label='TP (2R)')
        ax.axhline(y=exit_price, color='#e67e22', linewidth=2, label=f'Exit ({hit_level})')
        ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')
        chart_path = f"{sym}_close_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(chart_path, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        send_discord_image(chart_path, caption=f"{sym} – {trade['action']} {hit_level}")
        os.remove(chart_path)
    except Exception as e: print(f"Close chart error: {e}")

# ========== SIGNAL FORMATTING ==========
def fmt_price(price, reference_price=None):
    if reference_price is None: reference_price = abs(price)
    if reference_price < 1: return f"{price:.5f}"
    elif reference_price < 1000: return f"{price:.4f}"
    else: return f"{price:.2f}"

def format_signal(sig):
    sym = sig["symbol"].replace("-USD","")
    direction = sig["action"]
    entry = sig["limit_price"]; stop = sig["stop_loss"]
    tps = sig["take_profits"]   # 4 prices
    risk = abs(entry - stop); stop_pct = risk / entry * 100
    icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    score = sig["score"]
    layers = sig.get("layers", {})
    fail_warning = ""
    if layers:
        failed = [name for name, (_,_,status) in layers.items() if "FAIL" in status]
        if failed: fail_warning = f" ⚠️ Data: {', '.join(failed)}"
    tp_str = " / ".join([fmt_price(tp, entry) for tp in tps])
    return (f"${sym} – {icon} Setup (4H) | Score: {score:.1f}/{11.5}\n"
            f"Entry: {fmt_price(entry, entry)} | Stop: {fmt_price(stop, entry)} (-{stop_pct:.2f}%)\n"
            f"TPs (0.5/1.0/1.5/2.0R): {tp_str}{fail_warning}")

# ========== HOLD MESSAGE ==========
def format_hold_message(top5, top_layers, skipped=0, risky_limit=False):
    if risky_limit:
        return "HOLD – Maximum risky trades reached. No new signals until a trade hits breakeven."
    if not top5:
        return f"HOLD – No valid setups (all {skipped} coins failed data fetch)."
    lines = []
    top_score = top5[0][1]
    if top_score < MIN_SCORE:
        lines.append("HOLD – No high‑conviction setup found.")
    else:
        lines.append("HOLD – No further trades allowed (max risky / daily limit).")

    lines.append(f"\n📊 **Top Coin Scores** (of {len(top5)})")
    for idx, (pair, score, direction, _, _, _, _) in enumerate(top5, 1):
        short = pair.replace("-USD","")
        dir_str = direction if direction else "N/A"
        lines.append(f"{idx}. {short} → {dir_str} ({score:.1f}/11.5)")

    if top_layers:
        top_pair = top5[0][0].replace("-USD","")
        top_score_val = top5[0][1]
        top_dir = top5[0][2] if top5[0][2] else "N/A"
        lines.append(f"\n🔎 **Top Coin Layer Breakdown:** {top_pair} ({top_dir}, {top_score_val:.1f})")
        for name, (earned, max_, status) in top_layers.items():
            if "FAIL" in status or "WARN" in status:
                lines.append(f"• {name} ({max_}): ⚠️ {status}")
            elif earned > 0:
                lines.append(f"• {name} ({max_}): ✅")
            else:
                lines.append(f"• {name} ({max_}): ❌")
    else:
        lines.append("\nNo layer data available.")

    if skipped > 0:
        lines.append(f"\n({skipped} coins skipped due to data failure.)")
    return "\n".join(lines)

# ========== CHART ON SIGNAL ==========
def send_trade_chart(signal):
    sym = signal['symbol']
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import mplfinance as mpf
        df = get_hybrid_klines(sym, '4h', days=21)
        if df.empty or len(df) < 20: raise ValueError("not enough candles")
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        ema50 = df['Close'].ewm(span=min(50,len(df)), adjust=False).mean()
        addplots = [mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50')]
        fig, axes = mpf.plot(df, type='candle', style=mpf_style, title=f"{sym} 4h", ylabel='Price',
                             addplot=addplots, returnfig=True, figsize=(8,6))
        ax = axes[0]
        entry = signal.get('limit_price'); stop = signal.get('stop_loss')
        tps = signal.get('take_profits', [])
        if entry:
            ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
            ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
            if tps:
                for i, tp in enumerate(tps):
                    ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.6,
                               label=f'TP{i+1}' if i==0 else None)
            ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')
        chart_path = f"{sym}_chart.png"
        fig.savefig(chart_path, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        send_discord_image(chart_path, caption=f"{sym} – {signal['action']} Setup (4H)")
        os.remove(chart_path)
        send_discord_message(format_signal(signal))
    except Exception as e:
        print(f"Chart error: {e}")
        send_discord_message(format_signal(signal))

# ========== MAIN ==========
def main():
    try:
        initialize_trade_files()
        check_open_trades()

        start_balance = get_daily_start_balance()
        if daily_pnl() <= -start_balance * DAILY_LOSS_PCT:
            send_discord_message(
                f"🚨 Daily loss limit reached ({-start_balance*DAILY_LOSS_PCT:.2f} USDT). "
                f"No new trades today.")
            return

        sig, top5, top_layers, skipped, risky_count = generate_signal()
        if sig:
            log_signal(sig); add_open_trade(sig)
            portfolio['open_positions'] += 1; save_portfolio(portfolio)
            send_trade_chart(sig)
        else:
            if risky_count >= MAX_RISKY_TRADES:
                send_discord_message(format_hold_message(top5, top_layers, risky_limit=True))
            else:
                send_discord_message(format_hold_message(top5, top_layers, skipped))
    except Exception as e:
        err = f"Bot crashed: {traceback.format_exc()[:500]}"
        print(err); send_discord_message(err)

if __name__ == "__main__":
    main()