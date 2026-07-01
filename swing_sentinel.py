#!/usr/bin/env python3
"""
Swing Sentinel – Quant‑Grade Crypto Swing Bot (4H)
5‑layer scoring + KuCoin data + integrated backtester.
Run live:    python swing_sentinel.py
Backtest:    python swing_sentinel.py --backtest
"""

import os, json, time, atexit, sys, math, traceback, re
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ==================== CONFIGURATION ====================
CONFIG = {
    "trading": {
        "max_risky_trades": 5,
        "risk_per_trade_pct": 1.0,
        "min_score_to_enter": 1.49,
        "atr_stop_multiplier": 2.5,
        "trailing_atr_multiplier": 2.0,
        "tp_multipliers": [0.4, 0.8, 1.2, 1.6, 2.0],
        "fractions": [0.30, 0.10, 0.10, 0.10, 0.40],
        "daily_loss_limit": -100
    },
    "scoring": {
        "weights": {
            "tech": 0.20,
            "buying_pressure": 0.45,
            "volatility": 0.05,
            "intermarket": 0.25,
            "volume_trend": 0.05
        }
    },
    "universe": {
        "limit": 50,
        "blacklist": [
            "QUQ","USDT","USDC","DAI","BUSD","TUSD","USDP","FDUSD",
            "LEO","WBT","USD1","USDS","USDE","USDG","USDY","PYUSD",
            "BUIDL","USYC","FIGR_HELOC","LAB","GRAM","CC","RAIN",
            "ASTER","HTX","WLFI","HYPE"
        ]
    },
    "ai": {
        "enabled": True,
        "model": "llama-3.3-70b-versatile",   # primary model
        "fallback_model": "qwen-2.5-32b",     # fallback if primary fails
        "temperature": 0.3
    },
    "files": {
        "portfolio_file": "portfolio.json",
        "trade_log": "trade_log.csv",
        "open_trades": "open_trades.csv",
        "trade_results": "trade_results.csv",
        "perf_counter": "perf_counter.txt"
    },
    "backtest": {
        "start_date": "2024-08-01",           # within KuCoin/Yahoo range
        "end_date": "today",
        "initial_balance": 1000.0,
        "fee_pct": 0.1,
        "slippage_pct": {"top_10": 0.02, "other": 0.05},
        "run_interval_hours": 4
    }
}
# =======================================================

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI_ENABLED = CONFIG["ai"]["enabled"] and GROQ_API_KEY is not None

LOCK_FILE = "bot.lock"
DATA_CACHE = {}
COIN_RANK = {}

# ========== INSTANCE LOCK ==========
def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            if time.time() - os.path.getmtime(LOCK_FILE) < 600:
                print("Another instance running. Exiting.")
                sys.exit(0)
        except: pass
    with open(LOCK_FILE, 'w') as f:
        f.write(str(datetime.now()))

def release_lock():
    try: os.remove(LOCK_FILE)
    except: pass
atexit.register(release_lock)

# ========== DYNAMIC COIN LIST ==========
def fetch_top_liquid_coins(limit=CONFIG["universe"]["limit"]):
    global COIN_RANK
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"market_cap_desc","per_page":limit,"page":1,"sparkline":False,"price_change_percentage":"24h"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        symbols = []
        COIN_RANK = {}
        rank = 1
        blacklist = set(CONFIG["universe"]["blacklist"])
        for coin in data:
            sym = coin.get("symbol","").upper()
            if sym and sym not in blacklist:
                ys = f"{sym}-USD"
                if ys not in symbols:
                    symbols.append(ys)
                    COIN_RANK[ys] = rank
                    rank += 1
        print(f"Fetched {len(symbols)} coins")
        return symbols[:limit]
    except Exception as e:
        print(f"CoinGecko failed: {e}. Using fallback.")
        fallback = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD"]
        COIN_RANK = {sym: i+1 for i, sym in enumerate(fallback)}
        return fallback[:limit]

CRYPTO_PAIRS = fetch_top_liquid_coins()

# ========== PORTFOLIO ==========
def load_portfolio():
    pf = CONFIG["files"]["portfolio_file"]
    if os.path.exists(pf):
        with open(pf) as f:
            data = json.load(f)
        return {"balance": data.get("balance",1000.0),
                "realized_pnl": data.get("realized_pnl",0.0),
                "open_positions": data.get("open_positions",0)}
    return {"balance":1000.0, "realized_pnl":0.0, "open_positions":0}

def save_portfolio(p):
    with open(CONFIG["files"]["portfolio_file"],"w") as f:
        json.dump(p, f, indent=2)

portfolio = load_portfolio()

# ========== SAFE CSV ==========
def safe_append_csv(filepath, df_new):
    tmp = filepath + ".tmp"
    try:
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath)
            updated = pd.concat([existing, df_new], ignore_index=True)
        else:
            updated = df_new
        updated.to_csv(tmp, index=False)
        os.replace(tmp, filepath)
    except:
        df_new.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

def safe_save_csv(filepath, df):
    tmp = filepath + ".tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, filepath)
    except:
        df.to_csv(filepath, index=False)

def init_csv(f, cols):
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

def initialize_files():
    init_csv(CONFIG["files"]["trade_log"], ["timestamp","symbol","action","entry","stop","TP1","TP2","TP3","TP4","TP5","score","ai_confidence"])
    init_csv(CONFIG["files"]["open_trades"], ["timestamp","symbol","action","entry","stop","TP1","TP2","TP3","TP4","TP5","status","quantity","original_qty","highest_tp","breakeven"])
    init_csv(CONFIG["files"]["trade_results"], ["timestamp","symbol","action","entry","stop","TP1","TP2","TP3","TP4","TP5","status","hit_level","close_time","exit_price","quantity","pnl"])

def log_signal(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP1": sig["take_profits"][0], "TP2": sig["take_profits"][1],
           "TP3": sig["take_profits"][2], "TP4": sig["take_profits"][3],
           "TP5": sig["take_profits"][4], "score": sig["conviction_score"], "ai_confidence": sig["confidence_score"]}
    safe_append_csv(CONFIG["files"]["trade_log"], pd.DataFrame([row]))

def add_open_trade(sig):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": sig["symbol"], "action": sig["action"],
           "entry": sig["limit_price"], "stop": sig["stop_loss"],
           "TP1": sig["take_profits"][0], "TP2": sig["take_profits"][1],
           "TP3": sig["take_profits"][2], "TP4": sig["take_profits"][3],
           "TP5": sig["take_profits"][4], "status": "open",
           "quantity": sig["quantity"], "original_qty": sig["quantity"],
           "highest_tp": -1, "breakeven": False}
    safe_append_csv(CONFIG["files"]["open_trades"], pd.DataFrame([row]))

# ========== KUCOIN DATA FETCH (primary) ==========
def get_kucoin_klines(sym_kucoin, interval, limit=100, start_time=None, end_time=None):
    interval_map = {'1h': '1hour', '4h': '4hour', '1d': '1day'}
    kucoin_interval = interval_map.get(interval, interval)
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": kucoin_interval, "symbol": sym_kucoin}
    if start_time: params["startAt"] = int(start_time.timestamp())
    if end_time: params["endAt"] = int(end_time.timestamp())
    try:
        time.sleep(0.2)
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("code") != "200000": return pd.DataFrame()
        candles = data["data"]
        if not candles: return pd.DataFrame()
        rows = []
        for c in candles:
            ts = datetime.utcfromtimestamp(int(c[0]))
            rows.append({'open_time': ts, 'Open': float(c[1]), 'Close': float(c[2]),
                         'High': float(c[3]), 'Low': float(c[4]), 'Volume': float(c[5])})
        df = pd.DataFrame(rows).set_index('open_time').sort_index()
        df = df[['Open','High','Low','Close','Volume']]
        return df.tail(limit) if len(df) > limit else df
    except Exception as e:
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
    base = sym_yahoo.replace("-USD","")
    kucoin_sym = f"{base}-USDT"
    df = get_kucoin_klines(kucoin_sym, interval, limit=500 if interval=='1h' else 100,
                           start_time=start, end_time=end)
    if not df.empty:
        DATA_CACHE[cache_key] = df
        return df
    df = get_yahoo_klines(sym_yahoo, interval, days=days, start=start, end=end)
    if not df.empty:
        DATA_CACHE[cache_key] = df
        return df
    return pd.DataFrame()

# ========== INDICATORS ==========
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return val if not pd.isna(val) else None
def adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    dm_plus = h.diff(); dm_minus = -l.diff()
    dm_plus[dm_plus<0] = 0; dm_minus[dm_minus<0] = 0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

# ========== 5‑LAYER SCORING ==========
def get_technicals(sym_yahoo, df=None):
    if df is None:
        df = get_hybrid_klines(sym_yahoo, '4h', days=14)
    if df.empty or len(df) < 50:
        return {"trend":0, "adx":0, "structure":0, "combined":0, "trend_dir":"up", "ema50_distance":1.0, "adx_value":0, "error":"insufficient 4h data"}
    closes = df['Close']; highs = df['High']; lows = df['Low']
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200) if len(closes) >= 200 else ema50
    current = closes.iloc[-1]
    trend = 0
    if current > ema50.iloc[-1]: trend += 1.5
    else: trend -= 1.5
    if ema50.iloc[-1] > ema200.iloc[-1]: trend += 1.5
    else: trend -= 1.5
    trend = max(-3, min(3, trend))

    adx_val, di_plus, di_minus = adx(df)
    adx_score = 0
    if adx_val > 25:
        if di_plus > di_minus: adx_score = 2.5
        else: adx_score = -2.5
    elif adx_val > 20:
        if di_plus > di_minus: adx_score = 1.0
        else: adx_score = -1.0

    window = 7
    lookback = min(50, len(highs))
    h = highs.iloc[-lookback:]; l = lows.iloc[-lookback:]
    swing_highs = []; swing_lows = []
    for i in range(window, len(h)-window):
        if h.iloc[i] >= h.iloc[i-window:i+window+1].max():
            swing_highs.append(h.iloc[i])
        if l.iloc[i] <= l.iloc[i-window:i+window+1].min():
            swing_lows.append(l.iloc[i])
    structure_score = 0
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]; hl = swing_lows[-1] > swing_lows[-2]
        if hh and hl:
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                if swing_highs[-2] > swing_highs[-3] and swing_lows[-2] > swing_lows[-3]:
                    structure_score = 3.0
                else: structure_score = 2.0
            else: structure_score = 2.0
        elif (not hh) and (not hl):
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                if swing_highs[-2] < swing_highs[-3] and swing_lows[-2] < swing_lows[-3]:
                    structure_score = -3.0
                else: structure_score = -2.0
            else: structure_score = -2.0
    structure_score = max(-3, min(3, structure_score))

    combined = trend * 0.30 + adx_score * 0.25 + structure_score * 0.45
    trend_dir = "up" if current > ema50.iloc[-1] else "down"
    ema50_distance = abs(current - ema50.iloc[-1]) / current
    return {"trend": trend, "adx": adx_score, "structure": structure_score, "combined": combined,
            "trend_dir": trend_dir, "ema50_distance": ema50_distance, "adx_value": adx_val, "error": None}

def get_buying_pressure(sym_yahoo, df=None):
    if df is None:
        df = get_hybrid_klines(sym_yahoo, '4h', days=10)
    if df.empty or len(df) < 48: return 0.0
    df = df.tail(48)
    buy_vol = df.loc[df['Close'] > df['Open'], 'Volume'].sum()
    sell_vol = df.loc[df['Close'] <= df['Open'], 'Volume'].sum()
    total = buy_vol + sell_vol
    if total == 0: return 0.0
    return (buy_vol - sell_vol) / total

def get_volatility_score(sym_yahoo, current_price, df=None):
    if df is None:
        df = get_hybrid_klines(sym_yahoo, '4h', days=14)
    if df.empty or len(df) < 14: return -1
    a = atr(df)
    if a is None: return -1
    pct = a / current_price * 100
    return 1 if 2 <= pct <= 7 else -1

def btc_trend_score(df=None):
    if df is None:
        df = get_hybrid_klines("BTC-USD", '4h', days=14)
    if df.empty or len(df) < 50: return 0
    ema50 = ema(df['Close'], 50)
    current = df['Close'].iloc[-1]
    return 2 if current > ema50.iloc[-1] else -2

def volume_trend_score(sym_yahoo, direction=None, df=None):
    if df is None:
        df = get_hybrid_klines(sym_yahoo, '4h', days=5)
    if df.empty or len(df) < 12: return 0
    recent = df['Volume'].tail(6)
    first = recent[:3].mean(); second = recent[3:].mean()
    if second > first * 1.05:
        return -2 if direction == "down" else 2
    elif second < first * 0.95:
        return -2 if direction == "up" else -2
    return 0

def momentum_alignment_score(sym_yahoo, direction, layers, df=None):
    if df is None:
        df = get_hybrid_klines(sym_yahoo, '4h', days=2)
    if df.empty or len(df) < 2: return 0.0
    last = df.iloc[-1]
    candle_ok = (direction=="LONG" and last['Close']>last['Open']) or (direction=="SHORT" and last['Close']<last['Open'])
    if not candle_ok: return 0.0
    supporting = 0
    if direction == "LONG":
        if layers.get("buying_press",0) > 0.5: supporting += 1
        if layers.get("intermarket",0) > 0.5: supporting += 1
        if layers.get("volume_trend",0) > 0.5: supporting += 1
    else:
        if layers.get("buying_press",0) < -0.5: supporting += 1
        if layers.get("intermarket",0) < -0.5: supporting += 1
        if layers.get("volume_trend",0) < -0.5: supporting += 1
    if supporting >= 2: return 0.20 if direction=="LONG" else -0.20
    return 0.0

def trend_strength_bonus(adx_value, base_score):
    if adx_value > 35 and abs(base_score) > 0.5:
        return 0.30 if base_score > 0 else -0.30
    elif adx_value > 30 and abs(base_score) > 0.5:
        return 0.20 if base_score > 0 else -0.20
    return 0.0

def score_coin(sym_yahoo, current_price, btc_score, dfs=None):
    layers = {}
    if dfs:
        tech = get_technicals(sym_yahoo, df=dfs.get('4h'))
        buying = get_buying_pressure(sym_yahoo, df=dfs.get('4h'))
        vol_score = get_volatility_score(sym_yahoo, current_price, df=dfs.get('4h'))
        vol_trend = volume_trend_score(sym_yahoo, direction=tech.get('trend_dir'), df=dfs.get('4h'))
    else:
        tech = get_technicals(sym_yahoo)
        buying = get_buying_pressure(sym_yahoo)
        vol_score = get_volatility_score(sym_yahoo, current_price)
        vol_trend = volume_trend_score(sym_yahoo, direction=tech.get('trend_dir'))
    if tech.get("error"): return 0, layers, 1.0, 0, "up", [tech["error"]]
    intermarket = btc_score
    total = (CONFIG["scoring"]["weights"]["tech"] * tech["combined"] +
             CONFIG["scoring"]["weights"]["buying_pressure"] * buying * 3 +
             CONFIG["scoring"]["weights"]["volatility"] * vol_score +
             CONFIG["scoring"]["weights"]["intermarket"] * intermarket +
             CONFIG["scoring"]["weights"]["volume_trend"] * vol_trend)
    total = max(-3, min(3, total))
    layers = {"tech": tech["combined"], "buying_press": buying*3, "volatility": vol_score,
              "intermarket": intermarket, "volume_trend": vol_trend}
    return total, layers, tech["ema50_distance"], tech["adx_value"], tech["trend_dir"], []

# ========== AI REASONING (with fallback) ==========
def call_ai(model, prompt):
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role":"user","content":prompt}],
                  "temperature":CONFIG["ai"]["temperature"], "max_tokens":150}, timeout=20)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
            reason_match = re.search(r'REASONING:\s*(.*)', text)
            conf = int(conf_match.group(1)) if conf_match else 5
            conf = max(4, min(7, conf))
            reason = reason_match.group(1).strip() if reason_match else "Automated signal."
            return conf, reason
    except: pass
    return None, None

def ai_reasoning(sym, entry, atr_val, layers, errors):
    if not AI_ENABLED: return 6, "AI disabled"
    layer_str = "; ".join(f"{k}={v:.2f}" for k,v in layers.items())
    prompt = (f"Crypto trade signal for {sym} at {entry:.5f}. 4h ATR: {atr_val:.4f}. "
              f"Layer scores: {layer_str}. "
              f"Provide concise reasoning (1-2 sentences) and a confidence 4-7. "
              f"Format: CONFIDENCE: 6 | REASONING: ...")
    # try primary model
    conf, reason = call_ai(CONFIG["ai"]["model"], prompt)
    if conf is not None:
        return conf, reason
    # try fallback
    conf, reason = call_ai(CONFIG["ai"].get("fallback_model", ""), prompt)
    if conf is not None:
        return conf, reason
    # ultimate fallback
    return 6, "AI unavailable, using default confidence"

# ========== SIGNAL GENERATION (live) ==========
def generate_signal():
    open_risky = set()
    try:
        open_df = pd.read_csv(CONFIG["files"]["open_trades"])
        if not open_df.empty and "breakeven" in open_df.columns:
            risky = open_df[open_df["breakeven"] == False]
            open_risky = set(risky["symbol"])
    except: pass

    if len(open_risky) >= CONFIG["trading"]["max_risky_trades"]:
        return {"action": "HOLD", "reasoning": f"Max {CONFIG['trading']['max_risky_trades']} risky trades open."}

    btc_score = btc_trend_score()
    candidates = []
    for yahoo_sym in CRYPTO_PAIRS:
        if yahoo_sym in open_risky: continue
        price_df = get_hybrid_klines(yahoo_sym, '1h', days=1)
        if price_df.empty: continue
        price = price_df['Close'].iloc[-1]
        score, layers, ema_dist, adx_val, trend_dir, errors = score_coin(yahoo_sym, price, btc_score)
        atr_val = atr(get_hybrid_klines(yahoo_sym, '4h', days=14)) or price*0.02
        if atr_val/price > 0.07: score = 0.0
        candidates.append({"symbol": yahoo_sym, "price": price, "score": score, "layers": layers,
                           "adx": adx_val, "trend_dir": trend_dir, "atr": atr_val, "errors": errors})
    if not candidates:
        return {"action": "HOLD", "reasoning": "No valid candidates."}
    best = max(candidates, key=lambda x: abs(x["score"]))
    if abs(best["score"]) < CONFIG["trading"]["min_score_to_enter"]:
        return {"action": "HOLD", "reasoning": f"No strong conviction. Best: {best['symbol']} ({best['score']:.2f})"}

    direction = "LONG" if best["score"] > 0 else "SHORT"
    if (direction == "LONG" and best["trend_dir"] == "down") or (direction == "SHORT" and best["trend_dir"] == "up"):
        return {"action": "HOLD", "reasoning": "Signal rejected by 4h trend filter."}

    best["score"] += trend_strength_bonus(best["adx"], best["score"])
    best["score"] += momentum_alignment_score(best["symbol"], direction, best["layers"])
    if abs(best["score"]) < CONFIG["trading"]["min_score_to_enter"]:
        return {"action": "HOLD", "reasoning": "Confidence below threshold after bonuses."}

    entry = best["price"] * (0.999 if direction=="LONG" else 1.001)
    atr_val = best["atr"]
    stop_distance = max(2.5 * atr_val, best["price"] * 0.01)
    stop = entry - stop_distance if direction=="LONG" else entry + stop_distance
    risk = abs(entry - stop)
    qty = round((portfolio["balance"] * CONFIG["trading"]["risk_per_trade_pct"] / 100) / risk, 6)
    tps = [round(entry + m*risk if direction=="LONG" else entry - m*risk, 6) for m in CONFIG["trading"]["tp_multipliers"]]
    conf, reason = ai_reasoning(best["symbol"], entry, atr_val, best["layers"], best["errors"])
    if conf < 5:
        return {"action": "HOLD", "reasoning": f"AI confidence too low ({conf}/10)."}

    return {
        "action": direction,
        "symbol": best["symbol"],
        "quantity": qty,
        "limit_price": entry,
        "stop_loss": stop,
        "take_profits": tps,
        "conviction_score": round(best["score"], 2),
        "confidence_score": conf,
        "reasoning": reason,
        "layers": best["layers"],
        "errors": best["errors"],
        "atr": atr_val
    }

# ========== STOP MANAGEMENT ==========
def get_current_stop(trade, current_price=None, atr_val=None):
    entry = float(trade["entry"])
    stop_orig = float(trade["stop"])
    # Support both list (backtest) and individual keys (CSV)
    if "take_profits" in trade and isinstance(trade["take_profits"], list):
        tps = [float(tp) for tp in trade["take_profits"]]
    else:
        tps = [float(trade.get(f"TP{i+1}", 0)) for i in range(5)]
    highest_tp_idx = int(trade.get("highest_tp", -1))
    breakeven = trade.get("breakeven", False)

    if not breakeven and highest_tp_idx == -1:
        return stop_orig
    if highest_tp_idx == 0:
        return entry
    if highest_tp_idx == 1:
        return tps[0]
    if highest_tp_idx == 2:
        return tps[1]
    if highest_tp_idx >= 3 and current_price is not None and atr_val is not None:
        trail_mult = CONFIG["trading"]["trailing_atr_multiplier"]
        if trade["action"] == "LONG":
            trail_stop = current_price - trail_mult * atr_val
            return max(trail_stop, tps[1])
        else:
            trail_stop = current_price + trail_mult * atr_val
            return min(trail_stop, tps[1])
    return stop_orig

# ========== LIVE TRADE MANAGEMENT ==========
def check_open_trades():
    try: open_df = pd.read_csv(CONFIG["files"]["open_trades"])
    except: return
    if open_df.empty: return
    for col in ["highest_tp","quantity","original_qty","breakeven"]:
        if col not in open_df.columns:
            open_df[col] = -1 if col=="highest_tp" else (False if col=="breakeven" else 0.0)
    results = []; still_open = []; alerts = []
    now = datetime.now()
    for _, trade in open_df.iterrows():
        sym = trade["symbol"]; direction = trade["action"]
        entry = float(trade["entry"]); stop_orig = float(trade["stop"])
        original_qty = float(trade.get("original_qty", trade.get("quantity",0)))
        remaining_qty = float(trade.get("quantity", original_qty))
        breakeven = trade.get("breakeven", False)
        # For CSV, TP columns already exist
        tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
        try: entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        except: still_open.append(trade); continue

        df_1h = get_hybrid_klines(sym, '1h', start=entry_time, end=now)
        if df_1h.empty:
            still_open.append(trade); continue
        df_1h = df_1h[df_1h.index >= entry_time]
        if df_1h.empty:
            still_open.append(trade); continue

        highest_tp_idx = int(trade.get("highest_tp", -1))
        current_stop = get_current_stop(trade)
        trade_closed = False
        closed_parts = []
        atr_val_trade = float(trade.get("atr", 0)) or (atr(get_hybrid_klines(sym, '4h', days=14)) or 0)

        for candle_time, candle in df_1h.iterrows():
            high = candle['High']; low = candle['Low']
            sl_hit = (direction == "LONG" and low <= current_stop) or (direction == "SHORT" and high >= current_stop)
            if sl_hit:
                exit_price = current_stop
                pnl = (exit_price - entry) * remaining_qty if direction=="LONG" else (entry - exit_price) * remaining_qty
                hit_desc = "BREAKEVEN STOP" if breakeven else ("STOP LOSS" if highest_tp_idx==-1 else f"STOP after TP{highest_tp_idx+1}")
                closed_parts.append({"exit_price": exit_price, "quantity": remaining_qty, "pnl": pnl, "hit_level": hit_desc})
                remaining_qty = 0; trade_closed = True
                alerts.append(f"**{sym.replace('-USD','')} {direction}**\n{'🔴' if 'STOP' in hit_desc else '🛑'} {hit_desc}\nP&L: {pnl:.2f} USDT")
                break

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
                    fraction = CONFIG["trading"]["fractions"][i]
                    exit_qty = original_qty * fraction
                    if exit_qty > remaining_qty: exit_qty = remaining_qty
                    if exit_qty > 0:
                        exit_price_tp = tps[i]
                        pnl = (exit_price_tp - entry) * exit_qty if direction=="LONG" else (entry - exit_price_tp) * exit_qty
                        closed_parts.append({"exit_price": exit_price_tp, "quantity": exit_qty, "pnl": pnl, "hit_level": f"TP{i+1}"})
                        remaining_qty -= exit_qty
                        highest_tp_idx = i
                        if i == 0: breakeven = True
                        trade["highest_tp"] = highest_tp_idx; trade["breakeven"] = breakeven
                        current_stop = get_current_stop(trade, current_price=high if direction=="LONG" else low, atr_val=atr_val_trade)
                        alerts.append(f"🎯 **{sym.replace('-USD','')} {direction}** TP{i+1} Hit! PnL: {pnl:.2f} USDT | Remaining: {remaining_qty:.6f} units")
                        if remaining_qty <= 0: trade_closed = True; break
                if trade_closed: break

        if remaining_qty > 0 and not trade_closed:
            trade["quantity"] = remaining_qty; trade["highest_tp"] = highest_tp_idx; trade["breakeven"] = breakeven
            still_open.append(trade)
        else:
            total_pnl = sum(p["pnl"] for p in closed_parts)
            portfolio["balance"] += total_pnl
            portfolio["realized_pnl"] += total_pnl
            for cp in closed_parts:
                results.append({**trade.to_dict(), "hit_level": cp["hit_level"],
                                "close_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                                "exit_price": cp["exit_price"], "quantity": cp["quantity"], "pnl": cp["pnl"]})
            if closed_parts:
                send_trade_close_chart(trade, closed_parts[-1]["hit_level"], closed_parts[-1]["exit_price"], total_pnl)

    if results:
        safe_append_csv(CONFIG["files"]["trade_results"], pd.DataFrame(results))
    if still_open:
        safe_save_csv(CONFIG["files"]["open_trades"], pd.DataFrame(still_open))
        portfolio["open_positions"] = len(still_open)
    else:
        safe_save_csv(CONFIG["files"]["open_trades"], pd.DataFrame())
        portfolio["open_positions"] = 0
    save_portfolio(portfolio)

    for alert in alerts:
        send_discord(alert)
    check_performance_report()

# ========== PERFORMANCE REPORT ==========
def check_performance_report():
    try:
        df = pd.read_csv(CONFIG["files"]["trade_results"])
    except: return
    if df.empty: return
    trade_groups = df.groupby(['timestamp', 'symbol'])
    closed_trades = trade_groups.agg(total_pnl=('pnl','sum'), action=('action','first')).reset_index()
    total_trades = len(closed_trades)
    last_reported = 0
    if os.path.exists(CONFIG["files"]["perf_counter"]):
        with open(CONFIG["files"]["perf_counter"]) as f:
            try: last_reported = int(f.read().strip())
            except: pass
    milestone = (total_trades // 10) * 10
    if milestone <= last_reported: return
    wins = closed_trades[closed_trades['total_pnl'] > 0]
    losses = closed_trades[closed_trades['total_pnl'] < 0]
    total_wins = len(wins); total_losses = len(losses)
    winrate = (total_wins / (total_wins+total_losses)) * 100 if (total_wins+total_losses) > 0 else 0
    total_pnl = closed_trades['total_pnl'].sum()
    profit_factor = wins['total_pnl'].sum() / abs(losses['total_pnl'].sum()) if total_losses > 0 else float('inf')
    closed_trades = closed_trades.sort_values('timestamp')
    win_streak = loss_streak = 0
    for _, row in closed_trades.iloc[::-1].iterrows():
        if row['total_pnl'] > 0:
            if loss_streak == 0: win_streak += 1
            else: break
        elif row['total_pnl'] < 0:
            if win_streak == 0: loss_streak += 1
            else: break
        else: break
    best = closed_trades.loc[closed_trades['total_pnl'].idxmax()]
    worst = closed_trades.loc[closed_trades['total_pnl'].idxmin()]
    report = (
        f"📊 **Performance Report** – All Time ({total_trades} trades)\n"
        f"Total P&L: {total_pnl:.2f} USDT\n"
        f"Winrate: {winrate:.1f}% ({total_wins}W/{total_losses}L)\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Win Streak: {win_streak} 🔥 | Loss Streak: {loss_streak} 😞\n"
        f"Best: {best['symbol']} {best['action']} {best['total_pnl']:.2f} USDT\n"
        f"Worst: {worst['symbol']} {worst['action']} {worst['total_pnl']:.2f} USDT"
    )
    send_discord(report)
    with open(CONFIG["files"]["perf_counter"], 'w') as f:
        f.write(str(milestone))

# ========== DISCORD HELPERS (chart‑ready) ==========
def send_discord(text):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json={"content": text[:2000]}, timeout=10)
        except: pass

def send_discord_image(image_path, caption=""):
    if not DISCORD_WEBHOOK_URL or not os.path.exists(image_path): return
    try:
        with open(image_path, 'rb') as img:
            requests.post(DISCORD_WEBHOOK_URL, data={'content': caption[:2000]}, files={'file': img}, timeout=15)
    except: pass

def send_trade_chart(signal):
    sym = signal['symbol']
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import mplfinance as mpf
        df = get_hybrid_klines(sym, '4h', days=21)
        if df.empty or len(df) < 20:
            raise ValueError(f"not enough candles ({len(df)})")
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        ema50 = df['Close'].ewm(span=min(50,len(df)), adjust=False).mean()
        addplots = [mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50')]
        fig, axes = mpf.plot(df, type='candle', style=mpf_style, title=f"{sym} 4h", ylabel='Price',
                             addplot=addplots, returnfig=True, figsize=(8,6))
        ax = axes[0]
        entry = signal.get('limit_price'); stop = signal.get('stop_loss'); tps = signal.get('take_profits')
        if entry:
            ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
            ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
            for i, tp in enumerate(tps):
                ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.8, label=f'TP{i+1}' if i==0 else None)
            ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')
        path = f"chart_{sym}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        send_discord_image(path, caption=f"{sym} – {signal['action']} Setup")
        os.remove(path)
    except Exception as e:
        err = f"⚠️ Chart error for {sym}: {str(e)[:200]}"
        print(err)
        send_discord(err)

def send_trade_close_chart(trade, hit_level, exit_price, pnl):
    sym = trade["symbol"]; direction = trade["action"]
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import mplfinance as mpf
        entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        df = get_hybrid_klines(sym, '1h', start=entry_time, end=datetime.now())
        if df.empty: raise ValueError("no 1h data")
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', facecolor='#000000', gridcolor='#2a2e39',
                                       rc={'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.titlecolor':'white'})
        fig, axes = mpf.plot(df, type='candle', style=mpf_style,
                             title=f"{sym} {direction} – {hit_level} (PnL: {pnl:.2f}$)", ylabel='Price',
                             returnfig=True, figsize=(8,6))
        ax = axes[0]
        entry = float(trade["entry"]); stop = float(trade["stop"])
        tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
        ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
        ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
        for i, tp in enumerate(tps):
            ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.6, label=f'TP{i+1}' if i==0 else None)
        ax.axhline(y=exit_price, color='#e67e22', linewidth=2, label=f'Exit ({hit_level})')
        ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')
        path = f"close_{sym}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        send_discord_image(path, caption=f"{sym} {direction} – {hit_level}")
        os.remove(path)
    except Exception as e:
        err = f"⚠️ Close chart error for {sym}: {str(e)[:200]}"
        print(err)
        send_discord(err)

# ========== DAILY PNL ==========
def daily_pnl():
    try:
        df = pd.read_csv(CONFIG["files"]["trade_results"])
        if df.empty: return 0.0
        today = datetime.now().strftime("%Y-%m-%d")
        df['close_time'] = pd.to_datetime(df['close_time'])
        daily = df[df['close_time'].dt.strftime("%Y-%m-%d") == today]
        return daily['pnl'].sum() if not daily.empty else 0.0
    except: return 0.0

# ========== LIVE MAIN ==========
def live_main():
    acquire_lock()
    initialize_files()
    check_open_trades()
    if daily_pnl() <= CONFIG["trading"]["daily_loss_limit"]:
        send_discord(f"Daily loss limit reached ({daily_pnl():.2f} USDT). No new trades.")
        release_lock()
        return

    sig = generate_signal()
    if sig["action"] in ("LONG", "SHORT"):
        log_signal(sig)
        add_open_trade(sig)
        portfolio["open_positions"] += 1
        save_portfolio(portfolio)
        sym = sig["symbol"].replace("-USD","")
        entry = sig["limit_price"]; stop = sig["stop_loss"]; tps = sig["take_profits"]
        risk = abs(entry - stop); stop_pct = risk/entry*100
        tp_str = " / ".join([f"{tp:.4f}" if tp<1000 else f"{tp:.2f}" for tp in tps])
        msg = (f"🟢 LONG {sym}" if sig["action"]=="LONG" else f"🔴 SHORT {sym}") + \
              f" | Entry: {entry:.4f} | Stop: {stop:.4f} ({stop_pct:.2f}%)\nTPs: {tp_str}\n" + \
              f"Conviction: {sig['conviction_score']:.2f}/3 | AI: {sig['confidence_score']}/10\n{sig['reasoning']}"
        send_discord(msg)
        send_trade_chart(sig)
    else:
        send_discord(f"HOLD – {sig['reasoning']}")
    release_lock()

# ========== BACKTEST ENGINE ==========
class BacktestEngine:
    def __init__(self, coins):
        self.coins = coins
        self.start = datetime.strptime(CONFIG["backtest"]["start_date"], "%Y-%m-%d")
        end_str = CONFIG["backtest"]["end_date"]
        self.end = datetime.now() if end_str.lower() == "today" else datetime.strptime(end_str, "%Y-%m-%d")
        self.balance = CONFIG["backtest"]["initial_balance"]
        self.initial_balance = self.balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []
        self.fee = CONFIG["backtest"]["fee_pct"] / 100.0
        self.slippage = CONFIG["backtest"]["slippage_pct"]
        self.interval_hours = CONFIG["backtest"]["run_interval_hours"]
        self.dates = pd.date_range(self.start, self.end, freq=f'{self.interval_hours}h')

    def get_coin_rank(self, sym):
        return COIN_RANK.get(sym, 99)

    def get_slippage(self, sym):
        rank = self.get_coin_rank(sym)
        return (self.slippage["top_10"] if rank <= 10 else self.slippage["other"]) / 100.0

    def apply_fee_and_slippage(self, price, side, sym):
        slip = self.get_slippage(sym)
        filled_price = price * (1 + slip) if side == 'buy' else price * (1 - slip)
        fee_cost = filled_price * self.fee
        return filled_price, fee_cost

    def get_historical_klines(self, sym_yahoo, interval, end_date, days=14):
        base = sym_yahoo.replace("-USD","")
        kucoin_sym = f"{base}-USDT"
        start_date = end_date - timedelta(days=days)
        df = get_kucoin_klines(kucoin_sym, interval, limit=500 if interval=='1h' else 100,
                               start_time=start_date, end_time=end_date)
        if not df.empty:
            return df
        return get_yahoo_klines(sym_yahoo, interval, days=days, start=start_date, end=end_date)

    def score_at(self, sym_yahoo, current_price, btc_score, date):
        df_4h = self.get_historical_klines(sym_yahoo, '4h', date, days=14)
        if df_4h.empty: return None
        tech = get_technicals(sym_yahoo, df=df_4h)
        if tech.get("error"): return None
        buying = get_buying_pressure(sym_yahoo, df=df_4h)
        vol_score = get_volatility_score(sym_yahoo, current_price, df=df_4h)
        vol_trend = volume_trend_score(sym_yahoo, direction=tech['trend_dir'], df=df_4h)
        intermarket = btc_score
        total = (CONFIG["scoring"]["weights"]["tech"] * tech["combined"] +
                 CONFIG["scoring"]["weights"]["buying_pressure"] * buying * 3 +
                 CONFIG["scoring"]["weights"]["volatility"] * vol_score +
                 CONFIG["scoring"]["weights"]["intermarket"] * intermarket +
                 CONFIG["scoring"]["weights"]["volume_trend"] * vol_trend)
        total = max(-3, min(3, total))
        layers = {"tech": tech["combined"], "buying_press": buying*3, "volatility": vol_score,
                  "intermarket": intermarket, "volume_trend": vol_trend}
        return {"score": total, "layers": layers, "adx": tech["adx_value"], "trend_dir": tech["trend_dir"],
                "atr": atr(df_4h) or current_price*0.02, "errors": []}

    def generate_signal_at(self, date):
        open_risky = set()
        for t in self.open_trades:
            if not t.get("breakeven", False):
                open_risky.add(t["symbol"])
        if len(open_risky) >= CONFIG["trading"]["max_risky_trades"]:
            return None

        btc_df = self.get_historical_klines("BTC-USD", '4h', date, days=14)
        btc_score = btc_trend_score(df=btc_df) if not btc_df.empty else 0

        candidates = []
        for yahoo_sym in self.coins:
            if yahoo_sym in open_risky: continue
            price_df = self.get_historical_klines(yahoo_sym, '1h', date, days=1)
            if price_df.empty: continue
            price = price_df['Close'].iloc[-1]
            res = self.score_at(yahoo_sym, price, btc_score, date)
            if res is None: continue
            atr_val = res['atr']
            if atr_val/price > 0.07: res['score'] = 0.0
            candidates.append({"symbol": yahoo_sym, "price": price, "score": res['score'],
                               "layers": res['layers'], "adx": res['adx'], "trend_dir": res['trend_dir'],
                               "atr": atr_val, "errors": res['errors']})
        if not candidates: return None
        best = max(candidates, key=lambda x: abs(x["score"]))
        if abs(best["score"]) < CONFIG["trading"]["min_score_to_enter"]: return None

        direction = "LONG" if best["score"] > 0 else "SHORT"
        if (direction == "LONG" and best["trend_dir"] == "down") or (direction == "SHORT" and best["trend_dir"] == "up"):
            return None

        best["score"] += trend_strength_bonus(best["adx"], best["score"])
        if abs(best["score"]) < CONFIG["trading"]["min_score_to_enter"]: return None

        entry = best["price"] * (0.999 if direction=="LONG" else 1.001)
        atr_val = best["atr"]
        stop_distance = max(2.5 * atr_val, best["price"] * 0.01)
        stop = entry - stop_distance if direction=="LONG" else entry + stop_distance
        risk = abs(entry - stop)
        qty = round((self.balance * CONFIG["trading"]["risk_per_trade_pct"] / 100) / risk, 6)
        tps = [round(entry + m*risk if direction=="LONG" else entry - m*risk, 6) for m in CONFIG["trading"]["tp_multipliers"]]

        filled_entry, fee_per_unit = self.apply_fee_and_slippage(entry, 'buy' if direction=="LONG" else 'sell', best["symbol"])
        entry_fee = fee_per_unit * qty
        self.balance -= entry_fee

        trade = {
            "action": direction, "symbol": best["symbol"], "quantity": qty,
            "limit_price": filled_entry, "stop_loss": stop, "take_profits": tps,
            "score": best["score"], "atr": atr_val, "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
            "original_qty": qty, "remaining_qty": qty, "highest_tp": -1, "breakeven": False,
            "entry": filled_entry, "stop": stop
        }
        # Add individual TP keys for compatibility with get_current_stop
        for i, tp in enumerate(tps):
            trade[f"TP{i+1}"] = tp
        return trade

    def simulate_trade_life(self, trade, current_date, next_date):
        sym = trade["symbol"]; direction = trade["action"]
        entry = trade["entry"]; stop_orig = trade["stop"]
        tps = trade["take_profits"]
        remaining_qty = trade["remaining_qty"]; original_qty = trade["original_qty"]
        highest_tp_idx = trade["highest_tp"]; breakeven = trade["breakeven"]
        current_stop = get_current_stop(trade)

        days_needed = max((next_date - current_date).days + 2, 5)
        df_1h = self.get_historical_klines(sym, '1h', next_date, days=days_needed)
        if df_1h.empty: return trade, False
        df_1h = df_1h[df_1h.index >= current_date]
        if df_1h.empty: return trade, False

        closed_parts = []; trade_closed = False
        atr_val_trade = trade.get("atr", 0) or (atr(df_1h) or 0)

        for candle_time, candle in df_1h.iterrows():
            high = candle['High']; low = candle['Low']
            sl_hit = (direction == "LONG" and low <= current_stop) or (direction == "SHORT" and high >= current_stop)
            if sl_hit:
                exit_price = current_stop
                slip, fee_per_unit = self.apply_fee_and_slippage(exit_price, 'sell' if direction=="LONG" else 'buy', sym)
                filled_exit = slip; fee_cost = fee_per_unit * remaining_qty
                pnl = (filled_exit - entry) * remaining_qty if direction=="LONG" else (entry - filled_exit) * remaining_qty
                pnl -= fee_cost
                closed_parts.append({"exit_price": filled_exit, "quantity": remaining_qty, "pnl": pnl,
                                     "hit_level": "STOP LOSS" if highest_tp_idx==-1 else f"STOP after TP{highest_tp_idx+1}"})
                remaining_qty = 0; trade_closed = True; break

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
                    fraction = CONFIG["trading"]["fractions"][i]; exit_qty = original_qty * fraction
                    if exit_qty > remaining_qty: exit_qty = remaining_qty
                    if exit_qty > 0:
                        exit_price_tp = tps[i]
                        slip, fee_per_unit = self.apply_fee_and_slippage(exit_price_tp, 'sell' if direction=="LONG" else 'buy', sym)
                        filled_exit_tp = slip; fee_cost = fee_per_unit * exit_qty
                        pnl = (filled_exit_tp - entry) * exit_qty if direction=="LONG" else (entry - filled_exit_tp) * exit_qty
                        pnl -= fee_cost
                        closed_parts.append({"exit_price": filled_exit_tp, "quantity": exit_qty, "pnl": pnl, "hit_level": f"TP{i+1}"})
                        remaining_qty -= exit_qty
                        highest_tp_idx = i
                        if i == 0: breakeven = True
                        trade["highest_tp"] = highest_tp_idx; trade["breakeven"] = breakeven
                        current_stop = get_current_stop(trade, current_price=high if direction=="LONG" else low, atr_val=atr_val_trade)
                    if remaining_qty <= 0: trade_closed = True; break
                if trade_closed: break

        if remaining_qty > 0 and not trade_closed:
            trade["remaining_qty"] = remaining_qty; trade["highest_tp"] = highest_tp_idx; trade["breakeven"] = breakeven
            return trade, False
        else:
            total_pnl = sum(p["pnl"] for p in closed_parts)
            self.balance += total_pnl
            for cp in closed_parts:
                self.closed_trades.append({
                    "open_time": trade["timestamp"], "close_time": candle_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym, "action": direction, "entry": entry, "stop": stop_orig,
                    "take_profits": str(tps), "exit_price": cp["exit_price"],
                    "quantity": cp["quantity"], "pnl": cp["pnl"], "hit_level": cp["hit_level"]
                })
            return None, True

    def run(self):
        print(f"Backtesting {len(self.coins)} coins from {self.start.date()} to {self.end.date()}...")
        for i, current_date in enumerate(self.dates):
            next_date = self.dates[i+1] if i+1 < len(self.dates) else current_date + timedelta(hours=self.interval_hours)
            new_open = []
            for trade in self.open_trades:
                updated_trade, closed = self.simulate_trade_life(trade, current_date, next_date)
                if not closed: new_open.append(updated_trade)
            self.open_trades = new_open
            self.equity_curve.append((current_date, self.balance))
            signal = self.generate_signal_at(current_date)
            if signal is not None:
                self.open_trades.append(signal)
        self.save_results()

    def save_results(self):
        eq_df = pd.DataFrame(self.equity_curve, columns=["date", "balance"])
        eq_df.to_csv("backtest_equity.csv", index=False)
        if self.closed_trades:
            pd.DataFrame(self.closed_trades).to_csv("backtest_trades.csv", index=False)
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        sharpe = None
        if len(eq_df) > 1:
            returns = eq_df["balance"].pct_change().dropna()
            if returns.std() != 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(365*24/self.interval_hours)
        summary = f"""Backtest completed!
Period: {self.start.date()} to {self.end.date()}
Initial balance: ${self.initial_balance:.2f}
Final balance:   ${self.balance:.2f}
Total return: {total_return:.2f}%
Sharpe ratio: {sharpe:.2f} (approx)
Number of closed trades: {len(self.closed_trades)}
"""
        with open("backtest_summary.txt", "w") as f:
            f.write(summary)
        print(summary)

def backtest_main():
    bt = BacktestEngine(CRYPTO_PAIRS)
    bt.run()

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest_main()
    else:
        live_main()