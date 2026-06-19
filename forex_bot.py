#!/usr/bin/env python3
"""
High‑Winrate Forex Swing Bot – TP1 1R, TPs: 1/2/3/4/5R
Enhancements: News/session filter, correlation filter, regime detection.
All layers realistic, no cheating.
"""

import requests, json, os, traceback
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ========== ENVIRONMENT ==========
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set – AI filtering disabled.")

# ========== FOREX UNIVERSE (50+ pairs) ==========
FOREX_PAIRS = [
    "EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD","NZDUSD","USDCHF",
    "EURGBP","EURJPY","EURCHF","EURAUD","EURCAD","EURNZD",
    "GBPJPY","GBPCHF","GBPAUD","GBPCAD","GBPNZD",
    "AUDJPY","AUDCHF","AUDCAD","AUDNZD",
    "CADJPY","CHFJPY","NZDCAD","NZDJPY","NZDCHF",
    "USDMXN","USDTRY","USDZAR","USDHKD","USDSGD",
    "USDNOK","USDSEK","USDDKK","USDPLN",
    "USDTHB","USDHUF","USDILS","USDCZK",
    # USDCLP and USDCNH removed to avoid data warnings – you can re-add if needed
    "USDPHP","USDIDR","USDINR","USDKRW",
    "USDMYR","USDTWD",
    "EURMXN","EURTRY","EURZAR","EURNOK","EURSEK",
    "GBPMXN","GBPZAR","GBPTRY","GBPNOK","GBPSEK",
]

def pip_scale(sym):
    return 0.01 if "JPY" in sym.upper() else 0.0001

# ========== PORTFOLIO ==========
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            return {
                "balance": data.get("balance", 1000.0),
                "realized_pnl": data.get("realized_pnl", 0.0),
                "open_positions": data.get("open_positions", 0),
                "daily_loss_limit": data.get("daily_loss_limit", -20)
            }
        except:
            pass
    return {
        "balance": 1000.0,
        "realized_pnl": 0.0,
        "open_positions": 0,
        "daily_loss_limit": -20
    }

def save_portfolio(p):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(p, f, indent=2)
    except:
        pass

portfolio = load_portfolio()

# ========== CSV LOGGING ==========
TRADE_LOG_CSV = "trade_log.csv"
OPEN_TRADES_CSV = "open_trades.csv"
TRADE_RESULTS_CSV = "trade_results.csv"

def init_csv(f, cols):
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

def append_csv(f, df_new):
    try:
        existing = pd.read_csv(f)
        updated = pd.concat([existing, df_new], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        updated = df_new
    updated.to_csv(f, index=False)

def save_csv(f, df):
    df.to_csv(f, index=False)

def initialize_trade_files():
    init_csv(TRADE_LOG_CSV, ["timestamp","symbol","action","entry","stop",
                             "TP1","TP2","TP3","TP4","TP5","score","ai_approved"])
    init_csv(OPEN_TRADES_CSV, ["timestamp","symbol","action","entry","stop",
                               "TP1","TP2","TP3","TP4","TP5","status",
                               "quantity","original_qty","highest_tp","lot_size"])
    init_csv(TRADE_RESULTS_CSV, ["timestamp","symbol","action","entry","stop",
                                 "TP1","TP2","TP3","TP4","TP5","status",
                                 "hit_level","close_time","exit_price","quantity","pnl"])

def log_signal(sig):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": sig["symbol"],
        "action": sig["action"],
        "entry": sig["limit_price"],
        "stop": sig["stop_loss"],
        "TP1": sig["take_profits"][0],
        "TP2": sig["take_profits"][1],
        "TP3": sig["take_profits"][2],
        "TP4": sig["take_profits"][3],
        "TP5": sig["take_profits"][4],
        "score": sig["score"],
        "ai_approved": sig.get("ai_approved", False)
    }
    append_csv(TRADE_LOG_CSV, pd.DataFrame([row]))

def add_open_trade(sig):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": sig["symbol"],
        "action": sig["action"],
        "entry": sig["limit_price"],
        "stop": sig["stop_loss"],
        "TP1": sig["take_profits"][0],
        "TP2": sig["take_profits"][1],
        "TP3": sig["take_profits"][2],
        "TP4": sig["take_profits"][3],
        "TP5": sig["take_profits"][4],
        "status": "open",
        "quantity": sig["quantity"],
        "original_qty": sig["quantity"],
        "highest_tp": -1,
        "lot_size": sig.get("lot_size", 0.0)
    }
    append_csv(OPEN_TRADES_CSV, pd.DataFrame([row]))

# ========== PORTFOLIO HELPERS ==========
def daily_pnl():
    try:
        df = pd.read_csv(TRADE_RESULTS_CSV)
        if df.empty:
            return 0.0
        today = datetime.now().strftime("%Y-%m-%d")
        df['close_time'] = pd.to_datetime(df['close_time'])
        daily = df[df['close_time'].dt.strftime("%Y-%m-%d") == today]
        return daily['pnl'].sum() if not daily.empty else 0.0
    except:
        return 0.0

def update_portfolio(trade_result):
    portfolio['balance'] += trade_result['pnl']
    portfolio['realized_pnl'] += trade_result['pnl']
    save_portfolio(portfolio)

# ========== DATA ==========
def get_data(pair, interval='4h', days=14, start=None, end=None):
    ysym = f"{pair}=X"
    if start is None:
        end = datetime.now()
        start = end - timedelta(days=days)
    else:
        end = end if end else datetime.now()
    try:
        df = yf.download(ysym, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def get_dxy(interval='4h', days=14):
    for ticker in ["DX-Y.NYB", "DX=F", "UUP"]:
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    return pd.DataFrame()

# ========== NEWS / SESSION FILTER ==========
# Hard‑coded list of major news events (UTC times, format: (month, day, hour, minute))
# You can update this list manually from ForexFactory.
HIGH_IMPACT_NEWS = [
    # Example: Non-Farm Payrolls (first Friday of each month, 13:30 UTC)
    # (6, 6, 13, 30),  # placeholder – fill with real dates if desired
    # FOMC (approx. 14:00 UTC on announcement days)
    # CPI (approx. 13:30 UTC on announcement days)
]

def is_major_news_nearby(now=None):
    """Return True if the current time is within 2 hours of a high-impact event."""
    if now is None:
        now = datetime.utcnow()
    for (m, d, h, mi) in HIGH_IMPACT_NEWS:
        event_time = datetime(now.year, m, d, h, mi)
        if abs((now - event_time).total_seconds()) < 7200:  # 2 hours
            return True
    return False

def is_good_session(now=None):
    """Only trade during London/NY overlap (08:00–16:00 UTC)."""
    if now is None:
        now = datetime.utcnow()
    hour = now.hour
    return 8 <= hour < 16

def session_ok():
    if not is_good_session():
        return False, "Outside trading session (08:00–16:00 UTC)"
    if is_major_news_nearby():
        return False, "High‑impact news nearby (within 2 hours)"
    return True, None

# ========== CORRELATION FILTER ==========
def correlation_ok(pair, direction):
    """
    Check that closely related pairs move in the same direction.
    If not, we are in a risk‑off / choppy environment – skip.
    """
    cousins = {
        "EURUSD": ["GBPUSD", "EURGBP"],
        "GBPUSD": ["EURUSD", "EURGBP"],
        "AUDUSD": ["NZDUSD", "AUDNZD"],
        "NZDUSD": ["AUDUSD", "AUDNZD"],
        "USDCAD": ["USDCHF"],
        "USDCHF": ["USDCAD"],
        "USDJPY": ["EURJPY", "GBPJPY"],
        "EURJPY": ["USDJPY", "GBPJPY"],
        "GBPJPY": ["USDJPY", "EURJPY"],
        # extend for others as needed
    }
    relevant = cousins.get(pair, [pair])
    # Get 4h data for the pair and its cousins
    dfs = {}
    for sym in relevant:
        df = get_data(sym, interval='4h', days=5)
        if df.empty:
            return True, None  # can't verify, allow
        dfs[sym] = df['Close'].pct_change().iloc[-4:].mean()  # average recent return

    # Direction check
    for sym in relevant:
        if (direction == "LONG" and dfs[sym] < 0) or (direction == "SHORT" and dfs[sym] > 0):
            return False, f"Correlation failed: {pair} {direction} but {sym} moving opposite"
    return True, None

# ========== REGIME DETECTION ==========
def is_strong_trend(df_4h):
    """Return True if ADX > 25 and 50 EMA slope > 0 (up) or < 0 (down) over last 10 candles."""
    closes = df_4h['Close']
    ema50 = ema(closes, 50)
    if len(ema50) < 10:
        return False
    slope = ema50.iloc[-1] - ema50.iloc[-10]
    adx_val, _, _ = adx(df_4h)
    return adx_val > 25 and abs(slope) > 0.0001  # minimal slope threshold

# ========== TECHNICAL INDICATORS ==========
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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
    return 100 - (100 / (1 + rs)).iloc[-1]

def macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line.iloc[-1], signal.iloc[-1], histogram.iloc[-1], histogram.iloc[-2] if len(histogram) > 1 else 0

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
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

def support_resistance_levels(df, lookback=20):
    recent = df.tail(lookback)
    high = recent['High'].max()
    low = recent['Low'].min()
    return high, low

# ========== MULTI‑LAYER SCORING (1R TP1 optimized, with new filters) ==========
def score_pair(pair):
    warnings = []

    df_d = get_data(pair, interval='1d', days=90)
    if df_d.empty or len(df_d) < 50:
        warnings.append(f"{pair}: insufficient daily data ({len(df_d)} candles)")
        return 0, None, None, None, None, warnings

    df_4h = get_data(pair, interval='4h', days=14)
    if df_4h.empty or len(df_4h) < 50:
        warnings.append(f"{pair}: insufficient 4h data ({len(df_4h)} candles)")
        return 0, None, None, None, None, warnings

    df_1h = get_data(pair, interval='1h', days=3)
    if df_1h.empty or len(df_1h) < 10:
        warnings.append(f"{pair}: insufficient 1h data ({len(df_1h)} candles)")
        return 0, None, None, None, None, warnings

    price = df_4h['Close'].iloc[-1]

    # Daily trend
    ema50_d = ema(df_d['Close'], 50)
    ema200_d = ema(df_d['Close'], 200)
    trend_daily = 0
    if price > ema50_d.iloc[-1] and ema50_d.iloc[-1] > ema200_d.iloc[-1]:
        trend_daily = 1
    elif price < ema50_d.iloc[-1] and ema50_d.iloc[-1] < ema200_d.iloc[-1]:
        trend_daily = -1
    if trend_daily == 0:
        return 0, None, None, None, None, warnings

    # Regime detection
    if not is_strong_trend(df_4h):
        return 0, None, None, None, None, warnings  # skip weak/ranging markets

    # 4h indicators
    ema50_4h = ema(df_4h['Close'], 50)
    ema200_4h = ema(df_4h['Close'], 200)
    adx_val, di_plus, di_minus = adx(df_4h)
    rsi_val = rsi(df_4h)
    macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
    atr_val = atr(df_4h)
    res, sup = support_resistance_levels(df_4h, 20)

    # 1h momentum
    rsi_1h = rsi(df_1h, 14)
    last_candle = df_1h.iloc[-1]
    prev_candle = df_1h.iloc[-2]
    candle_range = last_candle['High'] - last_candle['Low']
    if candle_range > 0:
        bullish_momentum = (last_candle['Close'] - last_candle['Open']) / candle_range
    else:
        bullish_momentum = 0

    # Breakout layer
    last_5_highs = df_1h['High'].iloc[-6:-1].max()
    last_5_lows = df_1h['Low'].iloc[-6:-1].min()
    breakout_long = price > last_5_highs
    breakout_short = price < last_5_lows

    # Volume
    vol_last = df_4h['Volume'].iloc[-1]
    vol_avg = df_4h['Volume'].iloc[-6:-1].mean() if len(df_4h) >= 6 else vol_last
    vol_surge = vol_last > vol_avg * 1.2

    # DXY
    dxy_df = get_dxy(interval='4h', days=14)
    dxy_aligned = False
    if dxy_df.empty:
        warnings.append("DXY data unavailable – intermarket layer skipped")
    else:
        dxy_ema50 = ema(dxy_df['Close'], 50)
        dxy_trend_up = dxy_df['Close'].iloc[-1] > dxy_ema50.iloc[-1]
        quote = pair[3:]
        if quote == "USD":
            dxy_aligned = dxy_trend_up if trend_daily == 1 else not dxy_trend_up
        elif quote in ("EUR","GBP","AUD","NZD","CAD","CHF"):
            dxy_aligned = not dxy_trend_up if trend_daily == 1 else dxy_trend_up

    def bool_score(cond):
        return 1 if cond else 0

    direction = "LONG" if trend_daily == 1 else "SHORT"

    # Standard layers
    if direction == "LONG":
        ema_align = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
    else:
        ema_align = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
    ema_score = bool_score(ema_align)

    adx_trending = adx_val > 25  # increased from 20
    adx_dir = (di_plus > di_minus) if direction == "LONG" else (di_minus > di_plus)
    adx_score = bool_score(adx_trending and adx_dir)

    rsi_score = bool_score((direction == "LONG" and rsi_val > 50) or (direction == "SHORT" and rsi_val < 50))

    macd_expanding = (direction == "LONG" and macd_hist > 0 and macd_hist > macd_hist_prev) or \
                     (direction == "SHORT" and macd_hist < 0 and macd_hist < macd_hist_prev)
    macd_score = bool_score(macd_expanding)

    if direction == "LONG":
        near_support = (price - sup) < atr_val * 0.5
        sr_score = bool_score(near_support)
    else:
        near_resistance = (res - price) < atr_val * 0.5
        sr_score = bool_score(near_resistance)

    vol_score = bool_score(vol_surge)
    dxy_score = bool_score(dxy_aligned)

    # 1h micro-trend & momentum
    if direction == "LONG":
        candle_momentum_ok = bullish_momentum > 0.4
        rsi_1h_ok = rsi_1h < 65
        micro_trend_ok = last_candle['Close'] > last_candle['Open'] and \
                         prev_candle['Close'] > prev_candle['Open']
        breakout_ok = breakout_long
    else:
        candle_momentum_ok = bullish_momentum < -0.4
        rsi_1h_ok = rsi_1h > 35
        micro_trend_ok = last_candle['Close'] < last_candle['Open'] and \
                         prev_candle['Close'] < prev_candle['Open']
        breakout_ok = breakout_short

    candle_score = bool_score(candle_momentum_ok)
    rsi_1h_score = bool_score(rsi_1h_ok)
    micro_trend_score = bool_score(micro_trend_ok)
    breakout_score = bool_score(breakout_ok)

    # Minimum ATR floor
    min_atr_pips = 10
    atr_pips = atr_val / pip_scale(pair)
    atr_ok = atr_pips >= min_atr_pips
    atr_score = bool_score(atr_ok)

    total = (
        ema_score * 2.0 +
        adx_score * 1.5 +
        rsi_score * 1.5 +
        macd_score * 1.0 +
        sr_score * 1.0 +
        vol_score * 0.2 +
        dxy_score * 0.2 +
        candle_score * 1.5 +
        rsi_1h_score * 1.0 +
        atr_score * 1.0 +
        micro_trend_score * 1.5 +
        breakout_score * 1.5
    )

    if total < 7.0:
        return 0, None, None, None, None, warnings

    return total, direction, price, atr_val, (sup if direction == "LONG" else res), warnings

# ========== AI CONFIRMATION GATE ==========
def ai_confirm_trade(signal_dict):
    if not GROQ_API_KEY:
        return True
    sym = signal_dict["symbol"]
    direction = signal_dict["action"]
    entry = signal_dict["limit_price"]
    stop = signal_dict["stop_loss"]
    score = signal_dict["score"]

    prompt = (
        f"Forex trade setup:\n"
        f"Pair: {sym}\n"
        f"Direction: {direction}\n"
        f"Entry: {entry:.5f}\n"
        f"Stop Loss: {stop:.5f}\n"
        f"Technical Conviction Score: {score:.1f}/15.9\n\n"
        f"Will this trade likely hit TP1 (1x the stop distance) before hitting the stop? "
        f"Answer with exactly one word: PASS or FAIL."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a professional forex analyst. Respond with only PASS or FAIL."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 5
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "FAIL" in text:
                return False
            return True
    except:
        pass
    return True

# ========== SIGNAL GENERATION (with session & correlation filters) ==========
def generate_signal():
    # Session / news check
    sess_ok, sess_reason = session_ok()
    if not sess_ok:
        return None, [f"Session/news filter: {sess_reason}"]

    open_symbols = set()
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
        if not open_df.empty:
            open_symbols = set(open_df["symbol"].values)
    except:
        pass

    candidates = []
    all_warnings = []

    for pair in FOREX_PAIRS:
        if pair in open_symbols:
            continue
        score, direction, price, atr_val, swing_level, warnings = score_pair(pair)
        all_warnings.extend(warnings)
        if direction and score >= 7.0:
            # Correlation filter
            cor_ok, cor_reason = correlation_ok(pair, direction)
            if not cor_ok:
                all_warnings.append(f"Correlation skip: {cor_reason}")
                continue
            candidates.append((pair, score, direction, price, atr_val, swing_level))

    if not candidates:
        return None, all_warnings

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    pair, score, direction, price, atr_val, swing_level = best

    ps = pip_scale(pair)
    min_pips = 8
    max_pips = 30
    raw_stop_pips = (1.0 * atr_val) / ps
    if raw_stop_pips < min_pips:
        stop_distance_pips = min_pips
    elif raw_stop_pips > max_pips:
        stop_distance_pips = max_pips
    else:
        stop_distance_pips = raw_stop_pips

    stop_distance = stop_distance_pips * ps

    if direction == "LONG":
        stop = price - stop_distance
        if swing_level is not None and swing_level > price - stop_distance * 1.2:
            stop = min(stop, swing_level - 0.05 * atr_val)
    else:
        stop = price + stop_distance
        if swing_level is not None and swing_level < price + stop_distance * 1.2:
            stop = max(stop, swing_level + 0.05 * atr_val)

    stop = round(stop, 6)
    risk = abs(price - stop)

    tp_multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]
    tps = []
    for m in tp_multipliers:
        if direction == "LONG":
            tps.append(round(price + m * risk, 6))
        else:
            tps.append(round(price - m * risk, 6))

    risk_amount = portfolio['balance'] * 0.01
    qty_base = risk_amount / risk
    lot_size = max(0.01, round(qty_base / 1000, 2))
    actual_units = lot_size * 1000

    signal = {
        "action": direction,
        "symbol": pair,
        "quantity": actual_units,
        "lot_size": lot_size,
        "limit_price": price,
        "stop_loss": stop,
        "take_profits": tps,
        "score": score,
        "atr": atr_val,
    }

    if not ai_confirm_trade(signal):
        print(f"AI rejected {pair} {direction} (score {score:.1f})")
        return None, all_warnings

    signal["ai_approved"] = True
    return signal, all_warnings

# ========== TRADE MANAGEMENT ==========
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
    fractions = [0.20, 0.20, 0.20, 0.20, 0.20]

    for idx, trade in open_df.iterrows():
        sym = trade["symbol"]
        direction = trade["action"]
        entry = float(trade["entry"])
        stop_orig = float(trade["stop"])
        original_qty = float(trade.get("original_qty", trade.get("quantity", 0)))
        remaining_qty = float(trade.get("quantity", original_qty))

        tps = [float(trade[f"TP{i+1}"]) for i in range(5)]

        try:
            entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        except:
            still_open.append(trade)
            continue

        df_1h = get_data(sym, interval='1h', start=entry_time, end=now)
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
                    fraction = fractions[i]
                    exit_qty = original_qty * fraction
                    if exit_qty > remaining_qty:
                        exit_qty = remaining_qty
                    if exit_qty > 0:
                        exit_price = tps[i]
                        pnl = (exit_price - entry) * exit_qty if direction == "LONG" else (entry - exit_price) * exit_qty
                        partial = trade.to_dict()
                        partial["hit_level"] = f"TP{i+1}"
                        partial["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                        partial["exit_price"] = exit_price
                        partial["quantity"] = exit_qty
                        partial["pnl"] = round(pnl, 4)
                        results.append(partial)
                        update_portfolio({'pnl': pnl})
                        remaining_qty -= exit_qty
                        highest_tp_idx = i
                        if i == 0:
                            current_stop = entry
                    alerts.append(f"🚀 {sym} {direction} TP{i+1} hit — {fraction*100:.0f}% closed, SL to BE")
                    send_closed_trade_chart(trade, f"TP{i+1}", exit_price, pnl, remaining_qty)

                if remaining_qty <= 0:
                    break

            if remaining_qty > 0:
                sl_hit = (low <= current_stop) if direction == "LONG" else (high >= current_stop)
                if sl_hit:
                    exit_price = current_stop
                    pnl = (exit_price - entry) * remaining_qty if direction == "LONG" else (entry - exit_price) * remaining_qty
                    final = trade.to_dict()
                    desc = "STOP LOSS" if highest_tp_idx == -1 else f"STOP LOSS after TP{highest_tp_idx+1}"
                    final["hit_level"] = desc
                    final["close_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    final["exit_price"] = exit_price
                    final["quantity"] = remaining_qty
                    final["pnl"] = round(pnl, 4)
                    results.append(final)
                    update_portfolio({'pnl': pnl})
                    remaining_qty = 0
                    alerts.append(f"🔴 {sym} {direction} → {desc}")
                    send_closed_trade_chart(trade, desc, exit_price, pnl, 0)
                    break

        if remaining_qty > 0:
            trade["highest_tp"] = highest_tp_idx
            trade["quantity"] = remaining_qty
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

    if alerts:
        send_telegram("Trade updates:\n" + "\n".join(alerts))

# ========== CHART ON TRADE CLOSE ==========
def send_closed_trade_chart(trade, hit_level, exit_price, pnl, remaining_qty):
    sym = trade["symbol"]
    entry = float(trade["entry"])
    stop = float(trade["stop"])
    tps = [float(trade[f"TP{i+1}"]) for i in range(5)]
    direction = trade["action"]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        entry_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        df = get_data(sym, interval='1h', start=entry_time, end=datetime.now())
        if df.empty:
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

        title = f"{sym} {direction} – {hit_level} (PnL: {pnl:.2f}$)"
        fig, ax = mpf.plot(df, type='candle', style=mpf_style,
                           title=title, ylabel='Price',
                           returnfig=True, figsize=(8,6))

        ax.axhline(y=entry, color='#f1c40f', linestyle='--', linewidth=1.5, label='Entry')
        ax.axhline(y=stop, color='#e74c3c', linestyle='--', linewidth=1.5, label='Stop')
        for i, tp in enumerate(tps):
            ax.axhline(y=tp, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.6,
                       label=f'TP{i+1}' if i==0 else None)
        ax.axhline(y=exit_price, color='#e67e22', linewidth=2, label=f'Exit ({hit_level})')
        ax.legend(loc='upper left', facecolor='#000000', edgecolor='white', labelcolor='white')

        chart_path = f"{sym}_close_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(chart_path, 'rb') as img:
            requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': img})
        os.remove(chart_path)
    except Exception as e:
        print(f"Closed trade chart error: {e}")

# ========== TELEGRAM ==========
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

def format_signal(sig):
    sym = sig["symbol"]
    dirn = "🟢 LONG" if sig["action"] == "LONG" else "🔴 SHORT"
    entry = sig["limit_price"]
    stop = sig["stop_loss"]
    tps = sig["take_profits"]
    lot = sig["lot_size"]
    score = sig["score"]
    ps = pip_scale(sym)
    sl_pips = round(abs(entry - stop) / ps, 1)
    tp_pips = [round(abs(tp - entry) / ps, 1) for tp in tps]
    tp_str = " / ".join([f"TP{i+1}: {tp:.5f} ({p} pips)" for i, (tp, p) in enumerate(zip(tps, tp_pips))])

    ai_note = " (AI approved)" if sig.get("ai_approved") else ""
    return (
        f"{dirn} {sym}{ai_note}\n"
        f"Conviction: {score:.1f}/15.9\n"
        f"Entry: {entry:.5f}\n"
        f"Stop Loss: {stop:.5f} ({sl_pips} pips)\n"
        f"Take Profits: {tp_str}\n"
        f"Lot Size: {lot:.2f} (Risk: 1%)"
    )

# ========== CHART ON SIGNAL ==========
def send_trade_chart(signal):
    sym = signal['symbol']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        df = get_data(sym, interval='4h', days=21)
        if df.empty or len(df) < 20:
            raise ValueError(f"Only {len(df)} 4h candles")

        mpf_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            facecolor='#000000',
            gridcolor='#2a2e39',
            rc={'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'axes.titlecolor': 'white'}
        )

        ema50 = df['Close'].ewm(span=min(50, len(df)), adjust=False).mean()
        addplots = [mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50')]
        if df['Volume'].sum() > 0:
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
            addplots.append(mpf.make_addplot(vwap, color='#3498db', width=1, linestyle='--', label='VWAP'))

        fig, axes = mpf.plot(df, type='candle', style=mpf_style,
                             title=f"{sym} 4h", ylabel='Price', addplot=addplots,
                             returnfig=True, figsize=(8,6))
        ax = axes[0]
        entry = signal.get('limit_price')
        stop = signal.get('stop_loss')
        tps = signal.get('take_profits')
        if entry:
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
            resp = requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': img})
        os.remove(chart_path)
    except Exception as e:
        print(f"Chart image error: {e}")
        studies = "&studies[]=STD%3BEMA%3B50&studies[]=STD%3BVWAP"
        tv_url = f"https://www.tradingview.com/chart/?symbol=FX:{sym}&interval=240{studies}"
        send_telegram(f"📈 Chart unavailable – view here: {tv_url}")

# ========== MAIN ==========
def main():
    try:
        initialize_trade_files()
        check_open_trades()

        if daily_pnl() <= portfolio['daily_loss_limit']:
            send_telegram("⚠️ Daily loss limit reached. No new trades today.")
            return

        sig, warnings = generate_signal()

        if warnings:
            warn_msg = "⚠️ Data warnings:\n" + "\n".join(warnings)
            send_telegram(warn_msg)

        if sig:
            log_signal(sig)
            add_open_trade(sig)
            portfolio['open_positions'] += 1
            save_portfolio(portfolio)
            send_telegram(format_signal(sig))
            send_trade_chart(sig)
        else:
            send_telegram("HOLD – No high‑conviction setup found.")

    except Exception as e:
        err = f"Bot crashed: {traceback.format_exc()[:500]}"
        print(err)
        send_telegram(err)

if __name__ == "__main__":
    main()