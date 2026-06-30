#!/usr/bin/env python3
"""
Swing Sentinel – Crypto Swing Bot (4H, KuCoin+Yahoo)
Hardened & fixed version. All configuration in the CONFIG dict below.
"""

import requests, json, os, traceback, time, atexit, sys, math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ==================== CONFIGURATION (EDIT HERE) ====================
CONFIG = {
    "trading": {
        "max_risky_trades": 5,              # max open trades without TP1 hit
        "risk_per_trade_pct": 1.0,          # % of balance risked per trade
        "min_score_to_enter": 6.0,          # minimum overall score to take trade
        "stop_bounds": {
            "top_10": {"min": 0.01, "max": 0.04},
            "other":  {"min": 0.02, "max": 0.06}
        },
        "atr_stop_multiplier": 2.5,
        "tp_multipliers": [0.4, 0.8, 1.2, 1.6, 2.0],
        "fractions": [0.30, 0.10, 0.10, 0.10, 0.40],
        "daily_loss_limit": -100            # negative value (USDT)
    },
    "universe": {
        "limit": 50,
        "blacklist": ["QUQ","USDT","USDC","DAI","BUSD","TUSD","USDP","FDUSD","LEO","WBT"]
    },
    "ai": {
        "enabled": True,
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1
    },
    "files": {
        "portfolio_file": "crypto_portfolio.json",
        "trade_log": "crypto_trade_log.csv",
        "open_trades": "crypto_open_trades.csv",
        "trade_results": "crypto_trade_results.csv",
        "perf_counter": "perf_counter.txt"
    }
}
# ==================== END CONFIGURATION ====================

# Derived constants
MAX_RISKY_TRADES = CONFIG["trading"]["max_risky_trades"]
RISK_PER_TRADE_PCT = CONFIG["trading"]["risk_per_trade_pct"] / 100.0
MIN_SCORE_ENTER = CONFIG["trading"]["min_score_to_enter"]
STOP_BOUNDS = CONFIG["trading"]["stop_bounds"]
ATR_MULT = CONFIG["trading"]["atr_stop_multiplier"]
TP_MULTIPLIERS = CONFIG["trading"]["tp_multipliers"]
FRACTIONS = CONFIG["trading"]["fractions"]
DAILY_LOSS_LIMIT = CONFIG["trading"]["daily_loss_limit"]
BLACKLIST = set(CONFIG["universe"]["blacklist"])
UNIVERSE_LIMIT = CONFIG["universe"]["limit"]
AI_ENABLED = CONFIG["ai"]["enabled"]
AI_MODEL = CONFIG["ai"]["model"]
AI_TEMP = CONFIG["ai"]["temperature"]
FILES = CONFIG["files"]

# Environment secrets
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    AI_ENABLED = False
    print("WARNING: GROQ_API_KEY not set – AI filtering disabled.")

LOCK_FILE = "crypto_bot.lock"
DATA_CACHE = {}

# ========== INSTANCE LOCK (avoid concurrent runs) ==========
def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            mtime = os.path.getmtime(LOCK_FILE)
            if time.time() - mtime < 600:
                print("Another instance is running. Exiting.")
                sys.exit(0)
        except: pass
    with open(LOCK_FILE, 'w') as f:
        f.write(str(datetime.now()))

def release_lock():
    try: os.remove(LOCK_FILE)
    except: pass
atexit.register(release_lock)

# ========== DYNAMIC COIN LIST (Top‑50 Market Cap) ==========
def fetch_top_liquid_coins(limit=UNIVERSE_LIMIT):
    global COIN_RANK
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"market_cap_desc","per_page":limit,"page":1,
              "sparkline":False,"price_change_percentage":"24h"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        yahoo_symbols = []
        COIN_RANK = {}
        rank = 1
        for coin in data:
            symbol = coin.get("symbol","").upper()
            if symbol and symbol not in BLACKLIST:
                ys = f"{symbol}-USD"
                if ys not in yahoo_symbols:
                    yahoo_symbols.append(ys)
                    COIN_RANK[ys] = rank
                    rank += 1
        print(f"Fetched {len(yahoo_symbols)} coins (blacklist filtered)")
        return yahoo_symbols[:limit]
    except Exception as e:
        print(f"CoinGecko API failed: {e}. Using fallback 10 coins.")
        fallback = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
                    "ADA-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD"]
        COIN_RANK = {sym: i+1 for i, sym in enumerate(fallback)}
        return fallback[:limit]

COIN_RANK = {}
CRYPTO_PAIRS = fetch_top_liquid_coins(UNIVERSE_LIMIT)

# ========== PORTFOLIO ==========
def load_portfolio():
    pf = FILES["portfolio_file"]
    if os.path.exists(pf):
        try:
            with open(pf) as f: data = json.load(f)
            return {
                "balance": data.get("balance",1000.0),
                "realized_pnl": data.get("realized_pnl",0.0),
                "open_positions": data.get("open_positions",0),
                "daily_loss_limit": data.get("daily_loss_limit",DAILY_LOSS_LIMIT)
            }
        except: pass
    return {"balance":1000.0, "realized_pnl":0.0, "open_positions":0, "daily_loss_limit":DAILY_LOSS_LIMIT}

def save_portfolio(p):
    with open(FILES["portfolio_file"],"w") as f: json.dump(p,f,indent=2)

portfolio = load_portfolio()

# ========== CSV HELPERS ==========
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
    init_csv(FILES["trade_log"], ["timestamp","symbol","action","entry","stop",
                                  "TP1","TP2","TP3","TP4","TP5","score","ai_approved"])
    init_csv(FILES["open_trades"], ["timestamp","symbol","action","entry","stop",
                                    "TP1","TP2","TP3","TP4","TP5","status",
                                    "quantity","original_qty","highest_tp","breakeven"])
    init_csv(FILES["trade_results"], ["timestamp","symbol","action","entry","stop",
                                      "TP1","TP2","TP3","TP4","TP5","status",
                                      "hit_level","close_time","exit_price","quantity","pnl"])

def log_signal(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP1": sig["take_profits"][0], "TP2": sig["take_profits"][1],
           "TP3": sig["take_profits"][2], "TP4": sig["take_profits"][3],
           "TP5": sig["take_profits"][4], "score": sig["score"],
           "ai_approved": sig.get("ai_approved", False)}
    append_csv(FILES["trade_log"], pd.DataFrame([row]))

def add_open_trade(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP1": sig["take_profits"][0], "TP2": sig["take_profits"][1],
           "TP3": sig["take_profits"][2], "TP4": sig["take_profits"][3],
           "TP5": sig["take_profits"][4], "status": "open",
           "quantity": sig["quantity"], "original_qty": sig["quantity"],
           "highest_tp": -1, "breakeven": False}
    append_csv(FILES["open_trades"], pd.DataFrame([row]))

# ========== SYMBOL CONVERTERS ==========
def to_yahoo(sym):
    clean = sym.replace("-USD","").replace("USDT","").replace("-USDT","").strip("-")
    return f"{clean}-USD"

def yahoo_to_kucoin(sym_yahoo):
    base = sym_yahoo.replace("-USD","")
    return f"{base}-USDT"

# ========== DATA FETCH (with caching & rate limiting) ==========
def get_kucoin_klines(sym_kucoin, interval, limit=100, start_time=None, end_time=None):
    interval_map = {'1h':'1hour','4h':'4hour','1d':'1day'}
    params = {"type": interval_map.get(interval, interval), "symbol": sym_kucoin}
    if start_time: params["startAt"] = int(start_time.timestamp())
    if end_time: params["endAt"] = int(end_time.timestamp())
    try:
        time.sleep(0.2)  # rate limit throttle
        resp = requests.get("https://api.kucoin.com/api/v1/market/candles",
                            params=params, timeout=10)
        data = resp.json()
        if data.get("code") != "200000": return pd.DataFrame()
        candles = data["data"]
        if not candles: return pd.DataFrame()
        rows = []
        for c in candles:
            ts = datetime.utcfromtimestamp(int(c[0]))
            rows.append({'open_time':ts, 'Open':float(c[1]), 'Close':float(c[2]),
                         'High':float(c[3]), 'Low':float(c[4]), 'Volume':float(c[5])})
        df = pd.DataFrame(rows).set_index('open_time').sort_index()
        df = df[['Open','High','Low','Close','Volume']]
        return df.tail(limit) if len(df)>limit else df
    except Exception as e:
        print(f"KuCoin error for {sym_kucoin}: {e}")
        return pd.DataFrame()

def get_yahoo_klines(sym_yahoo, interval, days=14, start=None, end=None):
    if start is None:
        end = datetime.now()
        start = end - timedelta(days=days)
    else:
        end = end if end else datetime.now()
    try:
        time.sleep(0.2)
        df = yf.download(sym_yahoo, start=start, end=end, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

def get_hybrid_klines(sym_yahoo, interval, days=14, start=None, end=None):
    cache_key = (sym_yahoo, interval, days, str(start), str(end))
    if cache_key in DATA_CACHE: return DATA_CACHE[cache_key]
    kucoin_sym = yahoo_to_kucoin(sym_yahoo)
    df = get_kucoin_klines(kucoin_sym, interval,
                           limit=500 if interval=='1h' else 100,
                           start_time=start, end_time=end)
    if not df.empty:
        DATA_CACHE[cache_key] = df
        return df
    print(f"KuCoin failed/unavailable for {sym_yahoo}, falling back to Yahoo")
    df = get_yahoo_klines(sym_yahoo, interval, days=days, start=start, end=end)
    if not df.empty:
        DATA_CACHE[cache_key] = df
        return df
    if interval == '1h':
        df = get_yahoo_klines(sym_yahoo, '4h', days=days, start=start, end=end)
        if not df.empty:
            DATA_CACHE[(sym_yahoo,'4h',days,str(start),str(end))] = df
    return df

# ========== TECHNICAL INDICATORS ==========
def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return val if not pd.isna(val) else None

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0.0)
    loss = -delta.where(delta<0,0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else None

def macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return (macd_line.iloc[-1], signal.iloc[-1], hist.iloc[-1],
            hist.iloc[-2] if len(hist)>1 else 0)

def adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    dm_plus = h.diff(); dm_minus = -l.diff()
    dm_plus[dm_plus<0] = 0; dm_minus[dm_minus<0] = 0
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

def support_resistance_levels(df, lookback=20):
    recent = df.tail(lookback)
    return recent['High'].max(), recent['Low'].min()

# ========== SCORING (fixed EMAs, fallback trend) ==========
def score_pair(pair):
    layers = {}
    df_d = get_yahoo_klines(pair, '1d', days=200)
    if df_d.empty or len(df_d) < 50:
        return 0, None, None, None, None, {"Daily data": (0,0,"FAIL: insufficient daily candles")}
    df_4h = get_hybrid_klines(pair, '4h', days=14)
    if df_4h.empty or len(df_4h) < 50:
        return 0, None, None, None, None, {"4h data": (0,0,"FAIL: insufficient 4h candles")}
    df_1h = get_hybrid_klines(pair, '1h', days=3)
    if df_1h.empty or len(df_1h) < 10:
        return 0, None, None, None, None, {"1h data": (0,0,"FAIL: insufficient 1h candles")}

    price = df_4h['Close'].iloc[-1]

    # Daily trend (now with sufficient data for EMA200)
    ema50_d = ema(df_d['Close'], 50); ema200_d = ema(df_d['Close'], 200)
    trend_daily = 0
    if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
        trend_daily = 1
    elif price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
        trend_daily = -1

    # Fallback to 4h trend if daily still fails
    if trend_daily == 0:
        if len(df_4h) >= 200:
            ema50_4h = ema(df_4h['Close'], 50)
            ema200_4h = ema(df_4h['Close'], 200)
            if price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]:
                trend_daily = 1
            elif price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]:
                trend_daily = -1
            else:
                return 0, None, None, None, None, {"Trend": (0,0,"FAIL: no clear trend on 4h")}
        else:
            # Use 20/50 EMA cross as last resort
            ema20_4h = ema(df_4h['Close'], 20)
            ema50_4h = ema(df_4h['Close'], 50)
            if ema20_4h.iloc[-1] > ema50_4h.iloc[-1]:
                trend_daily = 1
            elif ema20_4h.iloc[-1] < ema50_4h.iloc[-1]:
                trend_daily = -1
            else:
                return 0, None, None, None, None, {"Trend": (0,0,"FAIL: no short‑term trend")}

    direction = "LONG" if trend_daily == 1 else "SHORT"

    ema50_4h = ema(df_4h['Close'], 50)
    ema200_4h = ema(df_4h['Close'], 200) if len(df_4h) >= 200 else None
    adx_val, di_plus, di_minus = adx(df_4h)
    rsi_val = rsi(df_4h)
    macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
    atr_val = atr(df_4h)
    res, sup = support_resistance_levels(df_4h, 20)

    rsi_1h_val = rsi(df_1h, 14)
    last_candle = df_1h.iloc[-1]; prev_candle = df_1h.iloc[-2]
    candle_range = last_candle['High'] - last_candle['Low']
    bullish_momentum = (last_candle['Close'] - last_candle['Open']) / candle_range if candle_range > 0 else 0

    vol_last = df_4h['Volume'].iloc[-1]
    vol_avg = df_4h['Volume'].iloc[-6:-1].mean() if len(df_4h) >= 6 else vol_last
    vol_surge = vol_last > vol_avg * 1.2 if vol_avg > 0 else False

    # BTC context (cached)
    btc_df = get_hybrid_klines("BTC-USD", '4h', days=14)
    market_aligned = False
    if not btc_df.empty and len(btc_df) >= 50:
        btc_ema50 = ema(btc_df['Close'], 50)
        btc_trend_up = btc_df['Close'].iloc[-1] > btc_ema50.iloc[-1]
        if trend_daily == 1 and btc_trend_up: market_aligned = True
        elif trend_daily == -1 and not btc_trend_up: market_aligned = True
    else:
        layers["Market"] = (0, 0.5, "FAIL: BTC data unavailable")

    def bool_score(cond): return 1 if cond else 0

    # 11 scoring layers
    if ema200_4h is not None:
        if direction == "LONG": ema_align = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
        else: ema_align = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
    else:
        ema20_4h = ema(df_4h['Close'], 20)
        if direction == "LONG": ema_align = price > ema20_4h.iloc[-1] and ema20_4h.iloc[-1] > ema50_4h.iloc[-1]
        else: ema_align = price < ema20_4h.iloc[-1] and ema20_4h.iloc[-1] < ema50_4h.iloc[-1]
    layers["EMA Align"] = (bool_score(ema_align) * 1.5, 1.5, "OK")

    adx_trending = adx_val > 20
    adx_dir = (di_plus > di_minus) if direction == "LONG" else (di_minus > di_plus)
    layers["ADX"] = (bool_score(adx_trending and adx_dir) * 1.0, 1.0, "OK")

    if rsi_val is not None:
        layers["RSI"] = (bool_score((direction=="LONG" and rsi_val>50) or
                                    (direction=="SHORT" and rsi_val<50)) * 1.5, 1.5, "OK")
    else: layers["RSI"] = (0, 1.5, "FAIL: RSI NaN")

    macd_expanding = (direction=="LONG" and macd_hist>0 and macd_hist>macd_hist_prev) or \
                     (direction=="SHORT" and macd_hist<0 and macd_hist<macd_hist_prev)
    layers["MACD"] = (bool_score(macd_expanding) * 1.0, 1.0, "OK")

    if atr_val and atr_val>0:
        if direction=="LONG": sr_score = bool_score((price-sup) < atr_val*0.5)
        else: sr_score = bool_score((res-price) < atr_val*0.5)
        layers["S/R"] = (sr_score*1.0, 1.0, "OK")
    else: layers["S/R"] = (0, 1.0, "FAIL: ATR missing")

    layers["Volume"] = (bool_score(vol_surge)*0.5, 0.5, "OK")

    if "Market" not in layers:
        layers["Market"] = (bool_score(market_aligned)*0.5, 0.5, "OK")

    candle_ok = (bullish_momentum > 0.5) if direction=="LONG" else (bullish_momentum < -0.5)
    layers["Candle Mom"] = (bool_score(candle_ok)*2.0, 2.0, "OK")

    if rsi_1h_val is not None:
        rsi_1h_ok = (rsi_1h_val < 63) if direction=="LONG" else (rsi_1h_val > 37)
        layers["RSI 1h"] = (bool_score(rsi_1h_ok)*1.5, 1.5, "OK")
    else: layers["RSI 1h"] = (0, 1.5, "FAIL: RSI 1h NaN")

    if atr_val and price>0:
        layers["ATR"] = (bool_score(atr_val > price*0.005)*1.0, 1.0, "OK")
    else: layers["ATR"] = (0, 1.0, "FAIL: ATR missing")

    if direction=="LONG": micro_ok = last_candle['Close'] > last_candle['Open'] and prev_candle['Close'] > prev_candle['Open']
    else: micro_ok = last_candle['Close'] < last_candle['Open'] and prev_candle['Close'] < prev_candle['Open']
    layers["Micro Trend"] = (bool_score(micro_ok)*2.0, 2.0, "OK")

    total = sum(score for score,_,_ in layers.values() if isinstance(score,(int,float)))
    return total, direction, price, atr_val, (sup if direction=="LONG" else res), layers

# ========== AI GATE ==========
def ai_confirm_trade(signal_dict):
    if not AI_ENABLED: return True
    prompt = (f"Crypto trade setup:\nPair: {signal_dict['symbol']}\nDirection: {signal_dict['action']}\n"
              f"Entry: {signal_dict['limit_price']:.5f}\nStop: {signal_dict['stop_loss']:.5f}\n"
              f"Score: {signal_dict['score']:.1f}/13.5\n"
              f"Will this trade likely hit TP1 (0.4x the stop distance) before hitting the stop? Answer PASS or FAIL.")
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": AI_MODEL, "messages": [
                {"role":"system","content":"You are a professional crypto analyst. Respond with only PASS or FAIL."},
                {"role":"user","content": prompt}], "temperature":AI_TEMP, "max_tokens":5}, timeout=15)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return "FAIL" not in text
    except: pass
    return True

# ========== SIGNAL GENERATION ==========
def generate_signal():
    open_symbols_risky = set()
    try:
        open_df = pd.read_csv(FILES["open_trades"])
        if not open_df.empty:
            if "symbol" in open_df.columns: open_df["symbol"] = open_df["symbol"].apply(to_yahoo)
            if "breakeven" in open_df.columns: risky = open_df[open_df["breakeven"] == False]
            else: risky = open_df
            open_symbols_risky = set(risky["symbol"].values)
    except: pass

    if len(open_symbols_risky) >= MAX_RISKY_TRADES:
        print(f"Max {MAX_RISKY_TRADES} risky trades limit reached. No new signals.")
        return None, [], {}, 0, 0, MAX_RISKY_TRADES

    all_scored = []; top_overall = None; skipped_no_trend = 0; skipped_data = 0
    for pair in CRYPTO_PAIRS:
        if pair in open_symbols_risky: continue
        score, direction, price, atr_val, swing_level, layers = score_pair(pair)
        if direction is None:
            if "Daily trend" in layers: skipped_no_trend += 1
            else: skipped_data += 1
            continue
        all_scored.append((pair, score, direction, price, atr_val, swing_level, layers))
        if top_overall is None or score > top_overall[1]:
            top_overall = (pair, score, direction, price, atr_val, swing_level, layers)

    print(f"Scored pairs: {len(all_scored)} | No trend: {skipped_no_trend} | Data fail: {skipped_data}")
    top5 = sorted(all_scored, key=lambda x: x[1], reverse=True)[:5]
    top_layers = top_overall[6] if top_overall else {}
    candidates = [item for item in all_scored if item[1] >= MIN_SCORE_ENTER]
    if not candidates:
        return None, top5, top_layers, skipped_no_trend, skipped_data, len(open_symbols_risky)

    candidates.sort(key=lambda x: x[1], reverse=True)
    pair, score, direction, price, atr_val, swing_level, layers = candidates[0]
    rank = COIN_RANK.get(pair, 99)
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

    stop = round(stop, 6); risk = abs(price - stop)
    tps = [round(price + m*risk, 6) if direction=="LONG" else round(price - m*risk, 6) for m in TP_MULTIPLIERS]
    quantity = round((portfolio['balance'] * RISK_PER_TRADE_PCT) / risk, 8)

    signal = {"action": direction, "symbol": pair, "quantity": quantity,
              "limit_price": price, "stop_loss": stop, "take_profits": tps,
              "score": score, "atr": atr_val, "layers": layers}
    if not ai_confirm_trade(signal):
        print(f"AI rejected {pair} {direction} (score {score:.1f})")
        return None, top5, top_layers, skipped_no_trend, skipped_data, len(open_symbols_risky)
    signal["ai_approved"] = True
    return signal, top5, top_layers, skipped_no_trend, skipped_data, len(open_symbols_risky)

# ========== DISCORD HELPERS ==========
def send_discord_message(text):
    try: requests.post(DISCORD_WEBHOOK_URL, json={"content": text[:2000]}, timeout=10)
    except Exception as e: print("Discord text error:", e)

def send_discord_image(image_path, caption=""):
    if not os.path.exists(image_path): return
    try:
        with open(image_path, 'rb') as img:
            files = {'file': img}
            payload = {'content': caption[:2000]} if caption else {}
            requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=15)
    except Exception as e: print("Discord image error:", e)

# ========== STOP MANAGEMENT ==========
def get_current_stop(trade):
    entry = float(trade["entry"]); stop_orig = float(trade["stop"])
    tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
    highest_tp_idx = int(trade.get("highest_tp", -1))
    breakeven = trade.get("breakeven", False)
    if not breakeven and highest_tp_idx == -1: return stop_orig
    if highest_tp_idx >= 0:
        if highest_tp_idx == 0: return entry
        elif highest_tp_idx == 1: return tps[0]
        elif highest_tp_idx == 2: return tps[1]
        elif highest_tp_idx >= 3: return tps[2]
    return stop_orig

# ========== TRADE MANAGEMENT (stop-first candle logic) ==========
def check_open_trades():
    try: open_df = pd.read_csv(FILES["open_trades"])
    except: return
    if open_df.empty: return

    open_df["symbol"] = open_df["symbol"].apply(to_yahoo)
    save_csv(FILES["open_trades"], open_df)

    for col in ["highest_tp","quantity","original_qty","breakeven"]:
        if col not in open_df.columns:
            open_df[col] = -1 if col=="highest_tp" else (False if col=="breakeven" else 0.0)

    results = []; still_open = []; alerts = []
    now = datetime.now()

    for idx, trade in open_df.iterrows():
        try:
            sym = trade["symbol"]; direction = trade["action"]
            entry = float(trade["entry"]); stop_orig = float(trade["stop"])
            original_qty = float(trade.get("original_qty", trade.get("quantity",0)))
            remaining_qty = float(trade.get("quantity", original_qty))
            breakeven = trade.get("breakeven", False)
            tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
            try: entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
            except: still_open.append(trade); continue

            df_1h = get_hybrid_klines(sym, '1h', start=entry_time, end=now)
            if df_1h.empty:
                still_open.append(trade); continue
            # Only candles that start on or after entry_time
            df_1h = df_1h[df_1h.index >= entry_time]
            if df_1h.empty:
                still_open.append(trade); continue

            highest_tp_idx = int(trade.get("highest_tp", -1))
            current_stop = get_current_stop(trade)
            trade_closed = False

            for candle_time, candle in df_1h.iterrows():
                high = candle['High']; low = candle['Low']

                # Always check stop before TP in the same candle
                sl_hit = (direction == "LONG" and low <= current_stop) or \
                         (direction == "SHORT" and high >= current_stop)
                if sl_hit:
                    exit_price = current_stop
                    pnl = (exit_price - entry) * remaining_qty if direction=="LONG" else (entry - exit_price) * remaining_qty
                    final = trade.to_dict()
                    if breakeven: desc = "BREAKEVEN STOP"; pnl = 0.0
                    else: desc = "STOP LOSS" if highest_tp_idx==-1 else f"STOP LOSS after TP{highest_tp_idx+1}"
                    final["hit_level"] = desc
                    final["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    final["exit_price"] = exit_price
                    final["quantity"] = remaining_qty
                    final["pnl"] = round(pnl,4)
                    results.append(final); update_portfolio({'pnl':pnl})
                    remaining_qty = 0; trade_closed = True
                    alert_line = f"**{sym} {direction}**\n{'🔴' if 'STOP' in desc else '🛑'} {desc}\nP&L: {pnl:.2f} USDT"
                    alerts.append(alert_line)
                    send_discord_message(alert_line)
                    send_trade_close_chart(trade, desc, exit_price, pnl)
                    break

                # No stop hit – process TP levels
                if direction == "LONG":
                    new_tp_idx = None
                    for i in range(len(tps)-1, -1, -1):
                        if high >= tps[i] and i > highest_tp_idx: new_tp_idx = i; break
                else:
                    new_tp_idx = None
                    for i in range(len(tps)-1, -1, -1):
                        if low <= tps[i] and i > highest_tp_idx: new_tp_idx = i; break

                if new_tp_idx is not None:
                    for i in range(highest_tp_idx+1, new_tp_idx+1):
                        if remaining_qty <= 0: break
                        fraction = FRACTIONS[i]; exit_qty = original_qty * fraction
                        if exit_qty > remaining_qty: exit_qty = remaining_qty
                        if exit_qty > 0:
                            exit_price = tps[i]
                            pnl = (exit_price - entry) * exit_qty if direction=="LONG" else (entry - exit_price) * exit_qty
                            partial = trade.to_dict()
                            partial["hit_level"] = f"TP{i+1}"
                            partial["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                            partial["exit_price"] = exit_price
                            partial["quantity"] = exit_qty
                            partial["pnl"] = round(pnl,4)
                            results.append(partial); update_portfolio({'pnl':pnl})
                            remaining_qty -= exit_qty; highest_tp_idx = i
                            trade["highest_tp"] = highest_tp_idx
                            trade["quantity"] = remaining_qty
                            if i == 0: trade["breakeven"] = True
                            tp_emoji = "🎯"
                            if i == 0: msg = f"{tp_emoji} **TP1 Hit!** 30% closed. SL moved to Breakeven. 🛡️"
                            elif i == 1: msg = f"{tp_emoji} **TP2 Hit!** 10% closed. SL moved to TP1 (1R locked). 🔒"
                            elif i == 2: msg = f"{tp_emoji} **TP3 Hit!** 10% closed. SL moved to TP2 (2R locked). 🔒"
                            elif i == 3: msg = f"{tp_emoji} **TP4 Hit!** 10% closed. SL moved to TP3 (3R locked). 🔒"
                            elif i == 4: msg = f"{tp_emoji} **TP5 Hit!** Final 40% closed – Home run! 🏆💰"
                            alert_line = f"**{sym} {direction}**\n{msg}\nP&L: {pnl:.2f} USDT | Remaining: {remaining_qty:.6f} units"
                            alerts.append(alert_line); send_discord_message(alert_line)
                            if remaining_qty <= 0: trade_closed = True; break
                    current_stop = get_current_stop(trade)

            if remaining_qty > 0 and not trade_closed:
                trade["quantity"] = remaining_qty; trade["highest_tp"] = highest_tp_idx
                still_open.append(trade)
        except Exception as e:
            print(f"Error processing trade {trade.get('symbol','?')}: {e}")
            still_open.append(trade)

    if results: append_csv(FILES["trade_results"], pd.DataFrame(results))
    if still_open:
        save_csv(FILES["open_trades"], pd.DataFrame(still_open))
        portfolio['open_positions'] = len(still_open)
    else:
        save_csv(FILES["open_trades"], pd.DataFrame())
        portfolio['open_positions'] = 0
    save_portfolio(portfolio)

    check_and_send_perf_report()

    risky_count = sum(1 for t in still_open if t.get("breakeven", False) == False)
    be_count = len(still_open) - risky_count
    summary = f"🔍 Open trades status: {risky_count} risky, {be_count} breakeven. Total: {len(still_open)}"
    print(summary); send_discord_message(summary)

# ========== PERFORMANCE REPORT ==========
def get_completed_trades():
    try: df = pd.read_csv(FILES["trade_results"])
    except: return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    trade_groups = df.groupby(['timestamp', 'symbol'])
    trades = []
    for (ts, sym), group in trade_groups:
        total_pnl = group['pnl'].sum()
        trades.append({'timestamp': ts, 'symbol': sym, 'total_pnl': total_pnl, 'action': group['action'].iloc[0]})
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty: trade_df = trade_df.sort_values('timestamp')
    trade_df['is_win'] = trade_df['total_pnl'] > 0
    trade_df['is_loss'] = trade_df['total_pnl'] < 0
    trade_df['is_breakeven'] = trade_df['total_pnl'] == 0
    return trade_df

def check_and_send_perf_report():
    trade_df = get_completed_trades()
    if trade_df.empty: return
    total_trades = len(trade_df)
    last_reported = 0
    if os.path.exists(FILES["perf_counter"]):
        try:
            with open(FILES["perf_counter"],'r') as f: last_reported = int(f.read().strip())
        except: pass
    current_milestone = (total_trades // 10) * 10
    if current_milestone <= last_reported: return

    wins = trade_df[trade_df['is_win']]; losses = trade_df[trade_df['is_loss']]
    total_wins = len(wins); total_losses = len(losses)
    winrate = (total_wins / max(total_wins+total_losses,1)) * 100
    total_pnl = trade_df['total_pnl'].sum()
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if total_losses>0 else float('inf')
    current_win_streak = 0; current_loss_streak = 0
    for _, row in trade_df.iloc[::-1].iterrows():
        if row['is_win']:
            if current_loss_streak == 0: current_win_streak += 1
            else: break
        elif row['is_loss']:
            if current_win_streak == 0: current_loss_streak += 1
            else: break
        else: break
    best = trade_df.loc[trade_df['total_pnl'].idxmax()]
    worst = trade_df.loc[trade_df['total_pnl'].idxmin()]
    report = (f"📊 **Performance Report** – All Time ({total_trades} closed trades)\n\n"
              f"**Total P&L:** {total_pnl:.2f} USDT\n"
              f"**Winrate:** {winrate:.1f}% ({total_wins}W / {total_losses}L)\n"
              f"**Profit Factor:** {profit_factor:.2f}\n"
              f"**Current Win Streak:** {current_win_streak} 🔥\n"
              f"**Current Loss Streak:** {current_loss_streak} 😞\n"
              f"**Best Trade:** {best['symbol']} {best['action']} {best['total_pnl']:.2f} USDT\n"
              f"**Worst Trade:** {worst['symbol']} {worst['action']} {worst['total_pnl']:.2f} USDT")
    send_discord_message(report)
    with open(FILES["perf_counter"],'w') as f: f.write(str(current_milestone))

def send_trade_close_chart(trade, hit_level, exit_price, pnl):
    sym = trade["symbol"]; direction = trade["action"]
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import mplfinance as mpf
        entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        df = get_hybrid_klines(sym, '1h', start=entry_time, end=datetime.now())
        if df.empty: return
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        fig, ax = mpf.plot(df, type='candle', style=mpf_style,
                           title=f"{sym} {direction} – {hit_level} (PnL: {pnl:.2f}$)", ylabel='Price',
                           returnfig=True, figsize=(8,6))
        entry = float(trade["entry"]); stop = float(trade["stop"])
        tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
        ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
        ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
        for i, tp in enumerate(tps):
            ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.6, label=f'TP{i+1}' if i==0 else None)
        ax.axhline(y=exit_price, color='#e67e22', linewidth=2, label=f'Exit ({hit_level})')
        ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')
        chart_path = f"{sym}_close_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(chart_path, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        send_discord_image(chart_path, caption=f"{sym} {direction} – {hit_level}")
        os.remove(chart_path)
    except Exception as e: print(f"Close chart error: {e}")

# ========== UNREALISED P&L ==========
def get_current_price(sym):
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty: return hist['Close'].iloc[-1]
    except: pass
    return None

def total_daily_pnl_including_unrealised():
    realised = 0.0
    try:
        df = pd.read_csv(FILES["trade_results"])
        if not df.empty:
            today = datetime.now().strftime("%Y-%m-%d")
            df['close_time'] = pd.to_datetime(df['close_time'])
            daily = df[df['close_time'].dt.strftime("%Y-%m-%d") == today]
            realised = daily['pnl'].sum()
    except: pass
    unrealised = 0.0
    try:
        open_df = pd.read_csv(FILES["open_trades"])
        if not open_df.empty:
            for _, t in open_df.iterrows():
                sym = t["symbol"]; direction = t["action"]; entry = float(t["entry"])
                qty = float(t.get("quantity", t.get("original_qty",0)))
                price = get_current_price(sym)
                if price is not None:
                    if direction == "LONG": unrealised += (price - entry) * qty
                    else: unrealised += (entry - price) * qty
    except: pass
    return realised + unrealised

# ========== SMART FORMATTING ==========
def fmt_price(price, reference_price=None):
    if reference_price is None: reference_price = abs(price)
    if reference_price < 1: return f"{price:.5f}"
    elif reference_price < 1000: return f"{price:.4f}"
    else: return f"{price:.2f}"

def format_signal(sig):
    sym = sig["symbol"].replace("-USD","")
    direction = sig["action"]; entry = sig["limit_price"]; stop = sig["stop_loss"]
    tps = sig["take_profits"]; risk = abs(entry - stop); stop_pct = risk / entry * 100
    direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    score = sig["score"]; layers = sig.get("layers",{})
    tp_str = " / ".join([fmt_price(tp, entry) for tp in tps])
    entry_str = fmt_price(entry, entry); stop_str = fmt_price(stop, entry)
    fail_warning = ""
    if layers:
        failed = [name for name, (_,_,status) in layers.items() if "FAIL" in status]
        if failed: fail_warning = f" ⚠️ Data: {', '.join(failed)}"
    return (f"${sym} – {direction_icon} Setup (4H) | Score: {score:.1f}/13.5\n"
            f"Entry: {entry_str} | Stop: {stop_str} (-{stop_pct:.2f}%)\n"
            f"TPs: {tp_str}{fail_warning}")

def format_hold_message(top5, top_layers, skipped_no_trend=0, skipped_data=0, risky_limit=False):
    if risky_limit:
        return f"HOLD – Maximum {MAX_RISKY_TRADES} risky trades limit reached. No new signals until a TP1 is hit."
    if not top5:
        msg = "HOLD – No valid trade setups found."
        if skipped_no_trend>0 or skipped_data>0:
            msg += f"\n({skipped_no_trend} coins lacked clear trend, {skipped_data} failed data)"
        else: msg += "\n(Market is fully trendless.)"
        return msg
    lines = [f"HOLD – No high‑conviction crypto setup found.\n📊 **Top Coin Scores** (of {len(top5)})"]
    for idx, (pair, score, direction, _, _, _, _) in enumerate(top5,1):
        short = pair.replace("-USD","")
        lines.append(f"{idx}. {short} → {direction} ({score:.1f}/13.5)")
    if top_layers:
        top_pair = top5[0][0].replace("-USD",""); top_score = top5[0][1]; top_dir = top5[0][2]
        lines.append(f"\n🔎 **Top Coin Layer Breakdown:** {top_pair} ({top_dir}, {top_score:.1f})")
        for name, (earned, max_, status) in top_layers.items():
            if "FAIL" in status: lines.append(f"• {name} ({max_}): ⚠️ {status}")
            else: lines.append(f"• {name} ({max_}): {'✅' if earned > 0 else '❌'}")
    else: lines.append("\nNo layer data available.")
    if skipped_no_trend>0 or skipped_data>0:
        lines.append(f"\n({skipped_no_trend} skipped – no trend, {skipped_data} skipped – data failure)")
    lines.append("\n💬 Are you stalking any setups? Drop your watchlist below! 👇")
    return "\n".join(lines)

# ========== CHART ON SIGNAL ==========
def send_trade_chart(signal):
    sym = signal['symbol']
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import mplfinance as mpf
        df = get_hybrid_klines(sym, '4h', days=21)
        if df.empty or len(df) < 20: raise ValueError("not enough candles")
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        ema50 = df['Close'].ewm(span=min(50,len(df)), adjust=False).mean()
        addplots = [mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50')]
        if df['Volume'].sum() > 0:
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
            addplots.append(mpf.make_addplot(vwap, color='#3498db', width=1, linestyle='--', label='VWAP'))
        fig, axes = mpf.plot(df, type='candle', style=mpf_style,
                             title=f"{sym} 4h", ylabel='Price', addplot=addplots,
                             returnfig=True, figsize=(8,6))
        ax = axes[0]
        entry = signal.get('limit_price'); stop = signal.get('stop_loss'); tps = signal.get('take_profits')
        if entry:
            ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
            ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
            if tps:
                for i, tp in enumerate(tps):
                    ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.8, label=f'TP{i+1}' if i==0 else None)
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
        acquire_lock()
        initialize_trade_files()
        check_open_trades()
        try:
            open_df = pd.read_csv(FILES["open_trades"])
            print(f"Currently {len(open_df)} open trade(s).")
        except: print("No open trades file.")

        # Daily loss limit including mark‑to‑market
        current_total_pnl = total_daily_pnl_including_unrealised()
        if current_total_pnl <= portfolio['daily_loss_limit']:
            send_discord_message(f"Daily loss limit reached (current P&L: {current_total_pnl:.2f} USDT). No new trades today.")
            return

        sig, top5, top_layers, skipped_no_trend, skipped_data, risky_count = generate_signal()
        if sig:
            log_signal(sig); add_open_trade(sig)
            portfolio['open_positions'] += 1; save_portfolio(portfolio)
            send_trade_chart(sig)
        else:
            if risky_count >= MAX_RISKY_TRADES:
                send_discord_message(format_hold_message(top5, top_layers, risky_limit=True))
            else:
                send_discord_message(format_hold_message(top5, top_layers, skipped_no_trend, skipped_data))
    except Exception as e:
        err = f"Bot crashed: {traceback.format_exc()[:500]}"
        print(err); send_discord_message(err)
    finally:
        release_lock()

if __name__ == "__main__":
    main()