#!/usr/bin/env python3
"""
Forex Bot – Multi‑factor scoring, partial take‑profits, trailing stop, Telegram alerts.
Adapted from crypto version for major/minor forex pairs.
Uses yfinance for data (educational only, not for live trading).
Environment variables required:
  TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY
"""

import requests, json, os, traceback, re, time, random
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ========== ENVIRONMENT ==========
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in secrets.")

# ========== FOREX CONFIG ==========
FOREX_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF",
    "USDCHF", "AUDJPY", "NZDJPY", "GBPCHF", "CADJPY"
]

def get_pip_scale(symbol):
    """Return pip size for a 6‑char forex pair (e.g., EURUSD -> 0.0001, USDJPY -> 0.01)."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001

# ========== PERSISTENT PORTFOLIO ==========
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            return {
                "balance_usdt": data.get("balance_usdt", 1000.0),  # account balance (e.g. USD)
                "realized_pnl": data.get("realized_pnl", 0.0),
                "open_positions": data.get("open_positions", 0),
                "daily_loss_limit": data.get("daily_loss_limit", -20)
            }
        except:
            pass
    return {
        "balance_usdt": 1000.0,
        "realized_pnl": 0.0,
        "open_positions": 0,
        "daily_loss_limit": -20
    }

def save_portfolio(p):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(p, f, indent=2)
    except:
        print("Warning: Could not save portfolio.json")

portfolio = load_portfolio()

# ========== CSV FILE PATHS ==========
TRADE_LOG_CSV = "trade_log.csv"
OPEN_TRADES_CSV = "open_trades.csv"
TRADE_RESULTS_CSV = "trade_results.csv"

# ========== DATA HELPERS ==========
def get_forex_data(pair, interval='4h', days=60, start=None, end=None):
    """
    Fetch forex OHLCV from Yahoo Finance.
    pair: e.g. 'EURUSD' -> 'EURUSD=X'
    """
    yahoo_symbol = f"{pair}=X"
    if start is None:
        end = datetime.now()
        start = end - timedelta(days=days)
    else:
        if end is None:
            end = datetime.now()
    try:
        df = yf.download(yahoo_symbol, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# Same for DXY (use futures)
def get_dxy_data(interval='4h', days=14):
    """Fetch US Dollar Index (DXY) via DX-Y.NYB"""
    return get_forex_data("DX-Y.NYB", interval, days)

# ========== CSV LOGGING ==========
def init_csv(filepath, columns):
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=columns)
        df.to_csv(filepath, index=False)

def append_csv(filepath, df_new):
    try:
        existing = pd.read_csv(filepath)
        updated = pd.concat([existing, df_new], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        updated = df_new
    updated.to_csv(filepath, index=False)

def save_csv(filepath, df):
    df.to_csv(filepath, index=False)

def initialize_trade_files():
    init_csv(TRADE_LOG_CSV, ["timestamp", "symbol", "action", "entry", "stop",
                             "TP1", "TP2", "TP3", "TP4", "TP5", "conviction", "ai_confidence"])
    # Add lot_size to open trades
    init_csv(OPEN_TRADES_CSV, ["timestamp", "symbol", "action", "entry", "stop",
                               "TP1", "TP2", "TP3", "TP4", "TP5", "status",
                               "quantity", "original_qty", "highest_tp", "lot_size"])
    init_csv(TRADE_RESULTS_CSV, ["timestamp", "symbol", "action", "entry", "stop",
                                 "TP1", "TP2", "TP3", "TP4", "TP5", "status", "hit_level",
                                 "close_time", "exit_price", "quantity", "pnl_usdt"])

def log_signal(signal):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": signal["symbol"],
        "action": signal["action"],
        "entry": signal["limit_price"],
        "stop": signal["stop_loss"],
        "TP1": signal["take_profits"][0],
        "TP2": signal["take_profits"][1],
        "TP3": signal["take_profits"][2],
        "TP4": signal["take_profits"][3],
        "TP5": signal["take_profits"][4],
        "conviction": signal["conviction_score"],
        "ai_confidence": signal["confidence_score"],
    }
    df = pd.DataFrame([row])
    append_csv(TRADE_LOG_CSV, df)

def add_open_trade(signal):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": signal["symbol"],
        "action": signal["action"],
        "entry": signal["limit_price"],
        "stop": signal["stop_loss"],
        "TP1": signal["take_profits"][0],
        "TP2": signal["take_profits"][1],
        "TP3": signal["take_profits"][2],
        "TP4": signal["take_profits"][3],
        "TP5": signal["take_profits"][4],
        "status": "open",
        "quantity": signal["quantity"],      # actual units of base currency
        "original_qty": signal["quantity"],
        "highest_tp": -1,
        "lot_size": signal.get("lot_size", 0.0)
    }
    df = pd.DataFrame([row])
    append_csv(OPEN_TRADES_CSV, df)

# ========== PORTFOLIO HELPERS ==========
def get_daily_pnl():
    try:
        df = pd.read_csv(TRADE_RESULTS_CSV)
        if df.empty:
            return 0.0
        today = datetime.now().strftime("%Y-%m-%d")
        df['close_time'] = pd.to_datetime(df['close_time'])
        daily = df[df['close_time'].dt.strftime("%Y-%m-%d") == today]
        if daily.empty:
            return 0.0
        return daily['pnl_usdt'].sum()
    except:
        return 0.0

def update_portfolio(trade_result):
    portfolio['balance_usdt'] += trade_result['pnl_usdt']
    portfolio['realized_pnl'] += trade_result['pnl_usdt']
    save_portfolio(portfolio)

# ========== INSTITUTIONAL IMPROVEMENTS (Forex Adaptation) ==========
def dxy_trend_score():
    """Score US Dollar strength based on DXY (4h chart)."""
    df = get_dxy_data(interval='4h', days=14)
    if df.empty or len(df) < 50:
        return 0, "DXY data unavailable"
    closes = df['Close']
    ema50 = closes.ewm(span=50, adjust=False).mean()
    current = closes.iloc[-1]
    ema_now = ema50.iloc[-1]
    if len(ema50) >= 7:
        ema_prev = ema50.iloc[-7]
        slope_up = ema_now > ema_prev
    else:
        slope_up = True
    price_above = current > ema_now
    if price_above and slope_up:
        return 2, None   # Dollar strong
    elif not price_above and not slope_up:
        return -2, None  # Dollar weak
    return 0, None

def institutional_macro_filter():
    """Use DXY trend as macro filter; returns score -2..2."""
    dxy_score, _ = dxy_trend_score()
    return dxy_score

# ========== TECHNICAL INDICATORS ==========
def anchored_vwap_score(df, current_price):
    if len(df) < 50:
        return 0
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    df = df.copy()
    df['vpv'] = typical * df['Volume']
    total_vol = df['Volume'].sum()
    if total_vol == 0:
        return 0
    vwap = df['vpv'].sum() / total_vol
    deviation = (current_price - vwap) / vwap * 100
    if deviation > 1:
        return 1
    elif deviation < -1:
        return -1
    else:
        return 0

def refined_buying_pressure(pair):
    df = get_forex_data(pair, interval='4h', days=10)
    if df.empty or len(df) < 48:
        return 0, 0
    short = df.tail(12)
    buy_vol_s = short.loc[short['Close'] > short['Open'], 'Volume'].sum()
    sell_vol_s = short.loc[short['Close'] <= short['Open'], 'Volume'].sum()
    total_s = buy_vol_s + sell_vol_s
    short_press = (buy_vol_s - sell_vol_s) / total_s if total_s > 0 else 0

    long_df = df.tail(48)
    buy_vol_l = long_df.loc[long_df['Close'] > long_df['Open'], 'Volume'].sum()
    sell_vol_l = long_df.loc[long_df['Close'] <= long_df['Open'], 'Volume'].sum()
    total_l = buy_vol_l + sell_vol_l
    long_press = (buy_vol_l - sell_vol_l) / total_l if total_l > 0 else 0

    return short_press, long_press

# ========== 4‑HOUR ANALYSIS (unchanged logic, adapted inputs) ==========
def get_technicals(pair):
    df = get_forex_data(pair, interval='4h', days=14)
    error = None
    if df.empty or len(df) < 50:
        error = f"insufficient 4h data ({len(df)} candles)"
        return {
            "trend": 0, "adx": 0, "structure": 0,
            "combined": 0, "ema50_distance": 1.0, "error": error
        }
    closes = df['Close']
    highs = df['High']
    lows = df['Low']
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema200 = closes.ewm(span=200, adjust=False).mean() if len(closes) >= 200 else ema50
    current = closes.iloc[-1]
    trend = 0
    if current > ema50.iloc[-1]:
        trend += 1.5
    else:
        trend -= 1.5
    if ema50.iloc[-1] > ema200.iloc[-1]:
        trend += 1.5
    else:
        trend -= 1.5
    trend = max(-3, min(3, trend))

    def calc_adx(high, low, close, period=14):
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr)
        di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx, di_plus, di_minus

    adx_series, di_plus, di_minus = calc_adx(highs, lows, closes, 14)
    adx_now = adx_series.iloc[-1]
    di_plus_now = di_plus.iloc[-1]
    di_minus_now = di_minus.iloc[-1]

    adx_score = 0
    if adx_now > 25:
        if di_plus_now > di_minus_now:
            adx_score = 2.5
        else:
            adx_score = -2.5
    elif adx_now > 20:
        if di_plus_now > di_minus_now:
            adx_score = 1.0
        else:
            adx_score = -1.0

    # structure detection (unchanged)
    window = 7
    lookback = min(50, len(highs))
    recent_highs = highs.iloc[-lookback:]
    recent_lows  = lows.iloc[-lookback:]
    swing_highs = []
    swing_lows  = []
    for i in range(window, len(recent_highs) - window):
        if all(recent_highs.iloc[i] >= recent_highs.iloc[i-window:i+window+1]):
            swing_highs.append((i, recent_highs.iloc[i]))
        if all(recent_lows.iloc[i] <= recent_lows.iloc[i-window:i+window+1]):
            swing_lows.append((i, recent_lows.iloc[i]))

    structure_score = 0
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_hh = swing_highs[-1][1] > swing_highs[-2][1]
        last_hl = swing_lows[-1][1] > swing_lows[-2][1]
        if last_hh and last_hl:
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                prev_hh = swing_highs[-2][1] > swing_highs[-3][1]
                prev_hl = swing_lows[-2][1] > swing_lows[-3][1]
                if prev_hh and prev_hl:
                    structure_score = 3.0
                else:
                    structure_score = 2.0
            else:
                structure_score = 2.0
        elif (not last_hh) and (not last_hl):
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                prev_lh = swing_highs[-2][1] < swing_highs[-3][1]
                prev_ll = swing_lows[-2][1] < swing_lows[-3][1]
                if prev_lh and prev_ll:
                    structure_score = -3.0
                else:
                    structure_score = -2.0
            else:
                structure_score = -2.0
    structure_score = max(-3, min(3, structure_score))

    combined = (
        trend * 0.30 +
        adx_score * 0.25 +
        structure_score * 0.45
    )
    ema50_val = ema50.iloc[-1]
    distance_pct = abs(current - ema50_val) / current
    trend_dir = "up" if current > ema50.iloc[-1] else "down"
    return {
        "trend": trend, "adx": adx_score, "structure": structure_score,
        "combined": combined, "ema50_distance": distance_pct,
        "adx_value": adx_now, "trend_dir": trend_dir, "error": None
    }

def get_4h_atr(pair, current_price):
    df = get_forex_data(pair, interval='4h', days=14)
    if df.empty or len(df) < 14:
        return current_price * 0.02, "ATR data insufficient, using 2% fallback"
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if pd.isna(atr):
        return current_price * 0.02, "ATR calculation failed, using 2% fallback"
    return atr, None

def get_buying_pressure(pair):
    short_p, long_p = refined_buying_pressure(pair)
    if short_p * long_p > 0:
        score = (short_p + long_p) / 2 * 3
    else:
        score = (short_p + long_p) / 2 * 3 * 0.3
    return score, None

def get_volatility_score(pair, current_price):
    atr, atr_err = get_4h_atr(pair, current_price)
    atr_pct = atr / current_price * 100
    # Forex typical range: 0.1% to 1% per 4h. Adjust limits accordingly.
    if atr_pct < 0.05 or atr_pct > 1.0:
        return -1, f"Volatility ATR%={atr_pct:.2f} out of normal range"
    return 1, None

def volume_trend_score(pair, direction=None):
    df = get_forex_data(pair, interval='4h', days=5)
    if df.empty or len(df) < 12:
        return 0, f"volume data insufficient ({len(df)} candles)"
    recent = df['Volume'].tail(6)
    first_half = recent[:3].mean()
    second_half = recent[3:].mean()
    if second_half > first_half * 1.05:
        if direction == "down":
            return -2, None
        return 2, None
    elif second_half < first_half * 0.95:
        if direction == "up":
            return -2, None
        return -2, None
    return 0, None

def momentum_alignment_score(pair, direction, layers):
    df = get_forex_data(pair, interval='4h', days=2)
    if df.empty or len(df) < 2:
        return 0.0
    last = df.iloc[-1]
    candle_agrees = (direction == "LONG" and last['Close'] > last['Open']) or \
                    (direction == "SHORT" and last['Close'] < last['Open'])
    if not candle_agrees:
        return 0.0
    supporting = 0
    if direction == "LONG":
        if layers.get("buying_press", 0) > 0.5: supporting += 1
        if layers.get("intermarket", 0) > 0.5: supporting += 1
        if layers.get("volume_trend", 0) > 0.5: supporting += 1
    else:
        if layers.get("buying_press", 0) < -0.5: supporting += 1
        if layers.get("intermarket", 0) < -0.5: supporting += 1
        if layers.get("volume_trend", 0) < -0.5: supporting += 1
    if supporting >= 2:
        return 0.20 if direction == "LONG" else -0.20
    return 0.0

def trend_strength_bonus(adx_value, base_score):
    if adx_value > 35 and abs(base_score) > 0.5:
        return 0.30 * (1 if base_score > 0 else -1)
    elif adx_value > 30 and abs(base_score) > 0.5:
        return 0.20 * (1 if base_score > 0 else -1)
    return 0.0

# ========== SCORING ENGINE (Forex) ==========
def score_forex_pair(pair, price, volume, dxy_score, dxy_error, macro_score):
    errors = []
    tech = get_technicals(pair)
    if tech.get("error"):
        errors.append(f"tech({pair}): {tech['error']}")
    tech_combined = tech["combined"]
    ema50_distance = tech["ema50_distance"]
    adx_value = tech.get("adx_value", 0)
    trend_dir = tech.get("trend_dir", "up")

    buying_score, buy_err = get_buying_pressure(pair)
    if buy_err:
        errors.append(f"buying_press({pair}): {buy_err}")

    vol_score, vol_err = get_volatility_score(pair, price)
    if vol_err:
        errors.append(f"volatility({pair}): {vol_err}")

    intermarket_s = dxy_score
    if dxy_error:
        errors.append(f"intermarket: {dxy_error}")

    vol_trend_s, vt_err = volume_trend_score(pair, direction=trend_dir)
    if vt_err:
        errors.append(f"volume_trend({pair}): {vt_err}")

    df_vwap = get_forex_data(pair, interval='4h', days=14)
    vwap_score = anchored_vwap_score(df_vwap, price)

    total = (
        0.20 * tech_combined +
        0.45 * buying_score +
        0.05 * vol_score +
        0.25 * intermarket_s +
        0.05 * vol_trend_s
    )
    macro_multiplier = 1 + 0.15 * macro_score
    total *= macro_multiplier
    total += vwap_score * 0.1

    layers = {
        "tech": tech_combined,
        "buying_press": buying_score,
        "volatility": vol_score,
        "intermarket": intermarket_s,
        "volume_trend": vol_trend_s,
    }
    return max(-3, min(3, total)), layers, ema50_distance, adx_value, trend_dir, errors

# ========== AI REASONING (Forex prompt) ==========
def call_groq_reasoning(pair, entry, atr, layers, errors=None):
    layer_str = "; ".join([f"{k}={v:.2f}" for k,v in layers.items()])
    err_str = ""
    if errors:
        err_str = " | Data issues: " + "; ".join(errors)

    directional_scores = [layers["tech"], layers["buying_press"], layers["intermarket"], layers["volume_trend"]]
    bearish_count = sum(1 for s in directional_scores if s < -0.5)
    bullish_count = sum(1 for s in directional_scores if s > 0.5)
    alignment_strength = max(bearish_count, bullish_count)

    system_msg = (
        "You are a professional forex market analyst writing a short post for social media. "
        "The chart shows 4h candlesticks with the 50-period EMA (orange) and Anchored VWAP (blue dotted). "
        "Write exactly 2 sentences explaining the technical reasoning behind the trade. Explicitly mention the EMA50 and VWAP, "
        "and refer to support/resistance levels or breakouts visible on the chart. Use natural, non‑hype language. "
        "After that, write one short, open‑ended question to encourage community engagement (avoid engagement bait). "
        "Also output a confidence score between 4 and 7 (5 if 2 metrics align, 6 if 3, 7 if 4). "
        "Output format:\n"
        "CONFIDENCE: 7 | HOOK: [your 2 sentences] | QUESTION: [your question]"
    )

    user_prompt = (
        f"Trade signal for {pair} at {entry}. 4h ATR: {atr:.5f}. "
        f"Internal metrics: {layer_str}{err_str}. "
        f"Alignment: {alignment_strength}/4 directional metrics are strongly aligned. "
        "Generate the compliant hook and question as instructed."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
            hook_match = re.search(r'HOOK:\s*(.*?)(?:\||$)', text)
            q_match = re.search(r'QUESTION:\s*(.*?)(?:\||$)', text)
            conf = int(conf_match.group(1)) if conf_match else 5
            conf = max(4, min(7, conf))
            hook = hook_match.group(1).strip() if hook_match else "Price is holding above the 50‑EMA with VWAP support, signalling bullish continuation."
            question = q_match.group(1).strip() if q_match else "How are you positioning for this move?"
            return conf, hook, question
    except:
        pass
    return 5, "The 4‑hour chart shows price above the 50‑EMA and VWAP, suggesting a potential continuation.", "Do you see this as a strong entry?"

# ========== MARKET HOURS ==========
def is_forex_market_open():
    """Very rough check: closed from Friday 22:00 UTC to Sunday 22:00 UTC."""
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour
    if weekday == 5 or (weekday == 6 and hour < 22) or (weekday == 4 and hour >= 22):
        return False
    return True

# ========== OPEN TRADE HELPERS ==========
def get_open_trade_symbols():
    try:
        df = pd.read_csv(OPEN_TRADES_CSV)
        if df.empty:
            return []
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").drop_duplicates(subset="symbol", keep="last")
        else:
            df = df.drop_duplicates(subset="symbol", keep="last")
        return list(df["symbol"].values)
    except:
        return []

def get_risky_open_count():
    try:
        df = pd.read_csv(OPEN_TRADES_CSV)
        if df.empty:
            return 0
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").drop_duplicates(subset="symbol", keep="last")
        else:
            df = df.drop_duplicates(subset="symbol", keep="last")
        if "highest_tp" in df.columns:
            return (df["highest_tp"] == -1).sum()
        else:
            return len(df)
    except:
        return 0

# ========== TRADE MANAGEMENT (unchanged logic) ==========
def check_open_trades():
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return

    if open_df.empty:
        return

    if "timestamp" in open_df.columns:
        open_df = open_df.sort_values("timestamp").drop_duplicates(subset="symbol", keep="last")
    else:
        open_df = open_df.drop_duplicates(subset="symbol", keep="last")

    for col in ["highest_tp", "quantity", "original_qty"]:
        if col not in open_df.columns:
            open_df[col] = 0.0 if col != "highest_tp" else -1

    results = []
    still_open = []
    alerts = []
    now = datetime.now()
    mults = [0.4, 0.8, 1.2, 1.6, 2.0]
    fractions = [0.20, 0.20, 0.30]

    for idx, trade in open_df.iterrows():
        sym = trade["symbol"]
        direction = trade["action"]
        entry = float(trade["entry"])
        stop_orig = float(trade["stop"])
        original_qty = float(trade.get("original_qty", trade.get("quantity", 0)))
        remaining_qty = float(trade.get("quantity", original_qty))
        risk = abs(entry - stop_orig)

        tps = []
        for m in mults:
            if direction == "LONG":
                tps.append(entry + m * risk)
            else:
                tps.append(entry - m * risk)

        try:
            entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        except:
            still_open.append(trade)
            continue

        df_1h = get_forex_data(sym, interval='1h', start=entry_time, end=now)
        if df_1h.empty:
            still_open.append(trade)
            continue

        highest_tp_idx = int(trade.get("highest_tp", -1))
        current_stop = entry if highest_tp_idx >= 0 else stop_orig

        for candle_time, candle in df_1h.iterrows():
            high = candle['High']
            low = candle['Low']

            new_tp_idx = None
            if direction == "LONG":
                for i in range(len(tps)-1, -1, -1):
                    if high >= tps[i] and i > highest_tp_idx:
                        new_tp_idx = i
                        break
            else:
                for i in range(len(tps)-1, -1, -1):
                    if low <= tps[i] and i > highest_tp_idx:
                        new_tp_idx = i
                        break

            if new_tp_idx is not None:
                for i in range(highest_tp_idx+1, new_tp_idx+1):
                    if remaining_qty <= 0:
                        break

                    if i <= 2:
                        fraction = fractions[i]
                        exit_qty = original_qty * fraction
                        if exit_qty > remaining_qty:
                            exit_qty = remaining_qty
                        if exit_qty > 0:
                            exit_price = tps[i]
                            pnl = (exit_price - entry) * exit_qty if direction == "LONG" else (entry - exit_price) * exit_qty
                            partial = trade.to_dict()
                            partial["hit_level"] = f"TP{i+1} (partial)"
                            partial["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                            partial["exit_price"] = exit_price
                            partial["quantity"] = exit_qty
                            partial["pnl_usdt"] = round(pnl, 4)
                            results.append(partial)
                            update_portfolio({'pnl_usdt': pnl})
                            remaining_qty -= exit_qty
                            highest_tp_idx = i
                            if i == 0:
                                current_stop = entry
                        alerts.append(f"🚀 {sym} {direction} TP{i+1} hit — {fraction*100:.0f}% closed, SL now {'BE' if i==0 else 'at entry'}")

                    elif i == 4:
                        if remaining_qty > 0:
                            exit_price = tps[4]
                            pnl = (exit_price - entry) * remaining_qty if direction == "LONG" else (entry - exit_price) * remaining_qty
                            final = trade.to_dict()
                            final["hit_level"] = "TP5 (final)"
                            final["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                            final["exit_price"] = exit_price
                            final["quantity"] = remaining_qty
                            final["pnl_usdt"] = round(pnl, 4)
                            results.append(final)
                            update_portfolio({'pnl_usdt': pnl})
                            remaining_qty = 0
                            highest_tp_idx = 4
                            alerts.append(f"🔔 {sym} {direction} TP5 hit — remaining closed")
                        break
                    else:
                        highest_tp_idx = 3

                if remaining_qty <= 0:
                    break

            if remaining_qty > 0:
                sl_hit = (low <= current_stop) if direction == "LONG" else (high >= current_stop)
                if sl_hit:
                    exit_price = current_stop
                    pnl = (exit_price - entry) * remaining_qty if direction == "LONG" else (entry - exit_price) * remaining_qty
                    final = trade.to_dict()
                    desc = "STOP LOSS" if highest_tp_idx == -1 else f"STOP LOSS (after TP{highest_tp_idx+1})"
                    final["hit_level"] = desc
                    final["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    final["exit_price"] = exit_price
                    final["quantity"] = remaining_qty
                    final["pnl_usdt"] = round(pnl, 4)
                    results.append(final)
                    update_portfolio({'pnl_usdt': pnl})
                    remaining_qty = 0
                    alerts.append(f"🔴 {sym} {direction} → {desc} (remaining closed)")
                    break

        if remaining_qty > 0:
            trade["highest_tp"] = highest_tp_idx
            trade["quantity"] = remaining_qty
            still_open.append(trade)

    if results:
        df_results = pd.DataFrame(results)
        append_csv(TRADE_RESULTS_CSV, df_results)

    if still_open:
        df_still = pd.DataFrame(still_open)
        for col in ["original_qty", "quantity", "highest_tp"]:
            if col not in df_still.columns:
                df_still[col] = 0 if col != "highest_tp" else -1
        portfolio['open_positions'] = len(df_still)
        save_csv(OPEN_TRADES_CSV, df_still)
    else:
        portfolio['open_positions'] = 0
        save_csv(OPEN_TRADES_CSV, pd.DataFrame())
    save_portfolio(portfolio)

    if alerts:
        msg = "Trade updates:\n" + "\n".join(alerts)
        send_telegram(msg)

# ========== SIGNAL GENERATION ==========
def generate_signal(balance_usdt):
    if not is_forex_market_open():
        return {"action": "HOLD", "reasoning": "Forex market closed. No new trades until open."}

    open_symbols = get_open_trade_symbols()
    risky_count = get_risky_open_count()
    if risky_count >= 3:
        return {"action": "HOLD", "reasoning": f"Max 3 active risky trades reached ({risky_count}). Waiting for TP1."}

    dxy_score, dxy_error = dxy_trend_score()
    macro_score = institutional_macro_filter()

    candidates = []
    for pair in FOREX_PAIRS:
        if pair in open_symbols:
            continue
        df = get_forex_data(pair, interval='4h', days=5)
        if df.empty:
            continue
        price = df['Close'].iloc[-1]
        volume = df['Volume'].iloc[-1]
        candidates.append({"symbol": pair, "price": price, "volume": volume})

    if not candidates:
        return {"action": "HOLD", "reasoning": "No candidates available (all pairs have open trades or no data)."}

    all_scored = []
    best = None
    best_score = 0

    for coin in candidates:
        pair = coin["symbol"]
        price = coin["price"]
        volume = coin["volume"]

        total_score, layers, ema_dist, adx_val, trend_dir, errors = score_forex_pair(
            pair, price, volume, dxy_score, dxy_error, macro_score
        )
        atr, _ = get_4h_atr(pair, price)
        atr_pct = atr / price * 100
        if atr_pct > 1.0:   # cap for forex
            total_score = 0.0
            errors.append("volatility cap triggered (ATR>1%)")
        coin["score"] = total_score
        coin["atr"] = atr
        coin["layers"] = layers
        coin["ema_distance"] = ema_dist
        coin["adx_value"] = adx_val
        coin["trend_dir"] = trend_dir
        coin["errors"] = errors

        all_scored.append(coin)

        if best is None or abs(total_score) > abs(best_score):
            best = coin
            best_score = total_score

    if dxy_error:
        best["errors"].append(f"intermarket: {dxy_error}")

    best_layers = best["layers"]
    best_ema_distance = best["ema_distance"]
    best_adx = best["adx_value"]
    best_trend_dir = best["trend_dir"]
    best_errors = best["errors"]

    all_scored_sorted = sorted(all_scored, key=lambda x: abs(x["score"]), reverse=True)
    coin_summary = " | ".join([f"{c['symbol']}: {c['score']:.2f}" for c in all_scored_sorted])

    if best is None or abs(best_score) < 1.49:
        best_sym = best["symbol"] if best else "none"
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        reason = (f"No strong conviction. Best score: {best_score:+.2f}/3 for {best_sym}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All pairs: {coin_summary}")
        return {"action": "HOLD", "reasoning": reason, "best_candidate": best}

    direction = "LONG" if best_score >= 0 else "SHORT"

    if (direction == "LONG" and best_trend_dir == "down") or \
       (direction == "SHORT" and best_trend_dir == "up"):
        best_sym = best["symbol"]
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        reason = (f"Signal {direction} rejected due to 4h trend filter ({best_trend_dir}). "
                  f"Best score: {best_score:+.2f}/3 for {best_sym}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All pairs: {coin_summary}")
        return {"action": "HOLD", "reasoning": reason, "best_candidate": best}

    best_score += trend_strength_bonus(best_adx, best_score)
    momentum_bonus = momentum_alignment_score(best["symbol"], direction, best_layers)
    best_score += momentum_bonus

    if abs(best_score) < 1.49:
        best_sym = best["symbol"]
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        reason = (f"No strong conviction after bonuses. Best score: {best_score:+.2f}/3 for {best_sym}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All pairs: {coin_summary}")
        return {"action": "HOLD", "reasoning": reason, "best_candidate": best}

    # Determine entry, stop, lot size
    pair = best["symbol"]
    price = best["price"]
    atr = best["atr"]
    pip_scale = get_pip_scale(pair)
    min_stop_distance = max(1.5 * atr, 10 * pip_scale)  # min 10 pips
    if direction == "LONG":
        entry = price  # simple market entry
        stop = entry - min_stop_distance
    else:
        entry = price
        stop = entry + min_stop_distance
    stop = round(stop, 6)
    risk_per_share = abs(entry - stop)

    risk_percent = 0.01
    risk_amount = balance_usdt * risk_percent
    quantity_base = risk_amount / risk_per_share   # units of base currency
    lot_size = round(quantity_base / 1000, 2)       # micro lots (0.01 lot = 1000 units)
    lot_size = max(0.01, lot_size)
    actual_units = lot_size * 1000

    mults = [0.4, 0.8, 1.2, 1.6, 2.0]
    tps = []
    for mult in mults:
        if direction == "LONG":
            tps.append(round(entry + mult * risk_per_share, 6))
        else:
            tps.append(round(entry - mult * risk_per_share, 6))

    conf, hook, question = call_groq_reasoning(pair, entry, atr, best_layers, best_errors)
    if conf < 5:
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        reason = (f"AI confidence too low ({conf}/10). Best score: {best_score:+.2f}/3 for {pair}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All pairs: {coin_summary}\n{hook}")
        return {"action": "HOLD", "reasoning": reason, "best_candidate": best}

    conviction10 = round(best_score * 10 / 3)
    conviction_str = f"+{conviction10}/10" if conviction10 >= 0 else f"{conviction10}/10"

    return {
        "action": direction,
        "symbol": pair,
        "quantity": actual_units,
        "lot_size": lot_size,
        "limit_price": entry,
        "stop_loss": stop,
        "take_profits": tps,
        "confidence_score": conf,
        "hook": hook,
        "question": question,
        "conviction_score": round(best_score, 2),
        "conviction10_str": conviction_str,
        "layers": best_layers,
        "errors": best_errors,
        "best_candidate": best
    }

# ========== CHART & TELEGRAM ==========
def send_trade_chart(signal, title_suffix=""):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        sym = signal['symbol']
        df = get_forex_data(sym, interval='4h', days=10)
        if df.empty or len(df) < 20:
            return

        mpf_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            facecolor='#000000',
            gridcolor='#2a2e39',
            rc={'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'axes.titlecolor': 'white'}
        )

        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()

        apds = [
            mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50'),
            mpf.make_addplot(vwap, color='#3498db', width=1, linestyle='--', label='VWAP')
        ]

        title = f"{sym} 4h"
        if title_suffix:
            title += title_suffix

        fig, axes = mpf.plot(df, type='candle', style=mpf_style,
                             title=title, ylabel='Price', addplot=apds,
                             returnfig=True, figsize=(8,6))
        ax = axes[0]

        entry = signal.get('limit_price')
        stop = signal.get('stop_loss')
        tps = signal.get('take_profits')
        if entry is not None and stop is not None:
            ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
            ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
            if tps:
                for i, tp in enumerate(tps):
                    ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.8,
                               label=f'TP{i+1}' if i==0 else None)
            ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')

        chart_path = f"{sym}_chart.png"
        fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(chart_path, 'rb') as img:
            requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': img})
        os.remove(chart_path)
    except ImportError:
        sym = signal['symbol']
        studies = "&studies[]=STD%3BEMA%3B50&studies[]=STD%3BVWAP"
        url = f"https://www.tradingview.com/chart/?symbol=FX:{sym}&interval=240{studies}"
        send_telegram(f"📈 Chart with EMA & VWAP: {url}")
    except Exception as e:
        print(f"Chart error: {e}")
        sym = signal['symbol']
        studies = "&studies[]=STD%3BEMA%3B50&studies[]=STD%3BVWAP"
        url = f"https://www.tradingview.com/chart/?symbol=FX:{sym}&interval=240{studies}"
        send_telegram(f"📈 Chart with EMA & VWAP: {url}")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

# ========== MAIN ==========
def main():
    try:
        initialize_trade_files()
        print("Checking open trades...")
        check_open_trades()

        daily_pnl = get_daily_pnl()
        if daily_pnl <= portfolio['daily_loss_limit']:
            msg = f"Daily loss limit reached (PnL: {daily_pnl:.2f} USD). No new trades today."
            send_telegram(msg)
            return

        balance = portfolio['balance_usdt']
        dec = generate_signal(balance)
        action = dec.get('action', 'HOLD')
        if action in ["LONG", "SHORT"]:
            log_signal(dec)
            add_open_trade(dec)
            portfolio['open_positions'] += 1
            save_portfolio(portfolio)

            sym = dec['symbol']
            direction_icon = "🟢" if action == "LONG" else "🔴"
            entry_price = dec['limit_price']
            stop_price = dec['stop_loss']
            tps = dec['take_profits']
            lot_size = dec.get('lot_size', 0)
            hook = dec.get('hook', 'Price is holding above the 50‑EMA with VWAP support.')
            question = dec.get('question', 'How are you positioning for this move?')
            pip_scale = get_pip_scale(sym)
            stop_pips = round(abs(entry_price - stop_price) / pip_scale, 1)

            tp_pips = [round(abs(tp - entry_price) / pip_scale, 1) for tp in tps]
            tp_str = " / ".join([f"{tp:,.5f} ({p} pips)" for tp, p in zip(tps, tp_pips)])

            title = f"{direction_icon} ${sym} 4h {'Buy' if action=='LONG' else 'Sell'} Setup"

            msg = (
                f"{title}\n"
                f"{hook}\n\n"
                f"Entry: {entry_price:.5f}\n"
                f"Stop Loss: {stop_price:.5f} ({stop_pips} pips)\n"
                f"Take Profit Targets: {tp_str}\n"
                f"Lot Size: {lot_size:.2f} (Risk: 1% of account)\n\n"
                f"{question}\n\n"
                f"#{sym} #Forex #TechnicalAnalysis #DYOR\n"
                f"*Disclaimer: This visual analysis is based on technical indicators for educational purposes only "
                f"and does not constitute financial advice. Please manage your risk and do your own research.*"
            )
            send_telegram(msg)
            send_trade_chart(dec)
        else:
            msg = f"HOLD\n{dec.get('reasoning', 'No signal')}"
            send_telegram(msg)
            best = dec.get('best_candidate')
            if best:
                send_trade_chart({
                    'symbol': best['symbol'],
                    'limit_price': None,
                    'stop_loss': None,
                    'take_profits': None
                })
    except Exception as e:
        err_msg = f"Bot crashed: {traceback.format_exc()}"
        print(err_msg)
        send_telegram(err_msg[:500])

if __name__ == "__main__":
    main()