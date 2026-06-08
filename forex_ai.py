import requests, json, os, traceback, re
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

# ========== PAPER PORTFOLIO ==========
portfolio = {
    "balance_usd": 1000.0,
    "positions": [],
    "realized_pnl": 0.0,
    "daily_loss_limit": -20
}

# ========== UNIVERSE (Forex + Commodities + Indices) ==========
INSTRUMENTS = [
    # Forex Majors & Crosses
    {"pair": "EURUSD",   "ticker": "EURUSD=X"},
    {"pair": "GBPUSD",   "ticker": "GBPUSD=X"},
    {"pair": "USDJPY",   "ticker": "USDJPY=X"},
    {"pair": "AUDUSD",   "ticker": "AUDUSD=X"},
    {"pair": "NZDUSD",   "ticker": "NZDUSD=X"},
    {"pair": "USDCAD",   "ticker": "USDCAD=X"},
    {"pair": "USDCHF",   "ticker": "USDCHF=X"},
    {"pair": "EURGBP",   "ticker": "EURGBP=X"},
    {"pair": "EURJPY",   "ticker": "EURJPY=X"},
    {"pair": "GBPJPY",   "ticker": "GBPJPY=X"},
    {"pair": "AUDJPY",   "ticker": "AUDJPY=X"},
    {"pair": "NZDJPY",   "ticker": "NZDJPY=X"},
    {"pair": "EURAUD",   "ticker": "EURAUD=X"},
    {"pair": "GBPAUD",   "ticker": "GBPAUD=X"},
    {"pair": "EURCHF",   "ticker": "EURCHF=X"},
    {"pair": "GBPCHF",   "ticker": "GBPCHF=X"},
    {"pair": "AUDCAD",   "ticker": "AUDCAD=X"},
    {"pair": "NZDCAD",   "ticker": "NZDCAD=X"},
    {"pair": "CADJPY",   "ticker": "CADJPY=X"},
    {"pair": "CHFJPY",   "ticker": "CHFJPY=X"},
    # Additional forex
    {"pair": "USDSEK",   "ticker": "USDSEK=X"},
    {"pair": "USDSGD",   "ticker": "USDSGD=X"},
    {"pair": "USDNOK",   "ticker": "USDNOK=X"},
    {"pair": "USDHKD",   "ticker": "USDHKD=X"},
    {"pair": "USDMXN",   "ticker": "USDMXN=X"},
    {"pair": "USDZAR",   "ticker": "USDZAR=X"},
    {"pair": "USDTRY",   "ticker": "USDTRY=X"},
    # Commodities (futures)
    {"pair": "XAUUSD",   "ticker": "GC=F"},       # Gold
    {"pair": "XAGUSD",   "ticker": "SI=F"},       # Silver
    {"pair": "USOIL",    "ticker": "CL=F"},       # Crude Oil
    # Stock Indices (futures)
    {"pair": "SPX500",   "ticker": "ES=F"},       # S&P 500
    {"pair": "NAS100",   "ticker": "NQ=F"},       # Nasdaq 100
    {"pair": "US30",     "ticker": "YM=F"},       # Dow Jones
]

# ========== CSV FILES ==========
TRADE_LOG_CSV   = "multi_trade_log.csv"
OPEN_TRADES_CSV = "multi_open_trades.csv"
TRADE_RESULTS_CSV = "multi_trade_results.csv"

# ========== DATA HELPERS ==========
def get_yahoo_klines(ticker, interval='4h', days=60):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# ========== CSV LOGGING (unchanged) ==========
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
    init_csv(TRADE_LOG_CSV, ["timestamp", "pair", "action", "entry", "stop",
                             "TP1", "TP2", "TP3", "TP4", "TP5", "conviction", "ai_confidence"])
    init_csv(OPEN_TRADES_CSV, ["timestamp", "pair", "action", "entry", "stop",
                               "TP1", "TP2", "TP3", "TP4", "TP5", "status"])
    init_csv(TRADE_RESULTS_CSV, ["timestamp", "pair", "action", "entry", "stop",
                                 "TP1", "TP2", "TP3", "TP4", "TP5", "status", "hit_level", "close_time"])

def log_signal(signal):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": signal["pair"],
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
        "pair": signal["pair"],
        "action": signal["action"],
        "entry": signal["limit_price"],
        "stop": signal["stop_loss"],
        "TP1": signal["take_profits"][0],
        "TP2": signal["take_profits"][1],
        "TP3": signal["take_profits"][2],
        "TP4": signal["take_profits"][3],
        "TP5": signal["take_profits"][4],
        "status": "open"
    }
    df = pd.DataFrame([row])
    append_csv(OPEN_TRADES_CSV, df)

def check_open_trades(current_prices):
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

    results = []
    still_open = []
    for _, trade in open_df.iterrows():
        pair = trade["pair"]
        if pair not in current_prices:
            still_open.append(trade)
            continue
        price = current_prices[pair]
        direction = trade["action"]
        entry = trade["entry"]
        stop = trade["stop"]
        tps = [trade[f"TP{i}"] for i in range(1,6)]

        hit = None
        if direction == "LONG":
            if price <= stop:
                hit = "STOP LOSS"
            else:
                for i, tp in enumerate(tps, 1):
                    if price >= tp:
                        hit = f"TP{i}"
                        break
        else:  # SHORT
            if price >= stop:
                hit = "STOP LOSS"
            else:
                for i, tp in enumerate(tps, 1):
                    if price <= tp:
                        hit = f"TP{i}"
                        break
        if hit:
            result = trade.to_dict()
            result["hit_level"] = hit
            result["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(result)
        else:
            still_open.append(trade)

    if results:
        df_results = pd.DataFrame(results)
        append_csv(TRADE_RESULTS_CSV, df_results)

    if still_open:
        df_still_open = pd.DataFrame(still_open)
        save_csv(OPEN_TRADES_CSV, df_still_open)
        return df_still_open
    else:
        save_csv(OPEN_TRADES_CSV, pd.DataFrame())
        return None

def fetch_current_prices():
    prices = {}
    for instr in INSTRUMENTS:
        df = get_yahoo_klines(instr["ticker"], interval='4h', days=2)
        if not df.empty and len(df) >= 1:
            prices[instr["pair"]] = df['Close'].iloc[-1]
    return prices

# ========== LAYERS & SCORING (unchanged) ==========
def get_technicals(ticker):
    df = get_yahoo_klines(ticker, interval='4h', days=14)
    error = None
    if df.empty or len(df) < 50:
        error = f"insufficient 4h data ({len(df)} candles)"
        return {
            "trend": 0, "adx": 0, "structure": 0,
            "combined": 0, "ema50_distance": 1.0, "error": error
        }

    closes = df['Close']
    highs  = df['High']
    lows   = df['Low']

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

    return {
        "trend": trend, "adx": adx_score, "structure": structure_score,
        "combined": combined, "ema50_distance": distance_pct,
        "adx_value": adx_now, "error": None
    }

def get_4h_atr(ticker, current_price):
    df = get_yahoo_klines(ticker, interval='4h', days=14)
    if df.empty or len(df) < 14:
        return current_price * 0.002, "ATR data insufficient, using 0.2% fallback"
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if pd.isna(atr):
        return current_price * 0.002, "ATR calculation failed, using 0.2% fallback"
    return atr, None

def get_buying_pressure(ticker):
    df = get_yahoo_klines(ticker, interval='4h', days=10)
    if df.empty or len(df) < 48:
        return 0.0, f"insufficient volume data ({len(df)} candles)"
    df = df.tail(48)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        buy_vol = df.loc[df['Close'] > df['Open'], 'Volume'].sum()
        sell_vol = df.loc[df['Close'] <= df['Open'], 'Volume'].sum()
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total
    else:
        up_bars = df['Close'].pct_change().clip(lower=0).sum()
        down_bars = -df['Close'].pct_change().clip(upper=0).sum()
        total = up_bars + down_bars
        if total == 0:
            return 0.0
        return (up_bars - down_bars) / total

def get_volatility_score(ticker, current_price):
    atr, atr_err = get_4h_atr(ticker, current_price)
    atr_pct = atr / current_price * 100
    if atr_pct < 0.2 or atr_pct > 2.0:
        return -1, atr_err
    return 1, None

def dxy_trend_score():
    df = get_yahoo_klines("DX-Y.NYB", interval='4h', days=14)
    if df.empty or len(df) < 50:
        return 0, "DXY data unavailable"
    closes = df['Close']
    ema50 = closes.ewm(span=50, adjust=False).mean()
    current = closes.iloc[-1]
    if current > ema50.iloc[-1]:
        return 2, None
    else:
        return -2, None

def volume_trend_score(ticker):
    df = get_yahoo_klines(ticker, interval='4h', days=5)
    if df.empty or len(df) < 12:
        return 0, "volume data insufficient"
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        recent = df['Volume'].tail(6)
        first_half = recent[:3].mean()
        second_half = recent[3:].mean()
        if second_half > first_half * 1.05:
            return 2, None
        elif second_half < first_half * 0.95:
            return -2, None
        return 0, None
    else:
        recent = (df['High'] - df['Low']).tail(6)
        first_half = recent[:3].mean()
        second_half = recent[3:].mean()
        if second_half > first_half * 1.05:
            return 2, None
        elif second_half < first_half * 0.95:
            return -2, None
        return 0, None

def momentum_alignment_score(ticker, direction):
    df = get_yahoo_klines(ticker, interval='4h', days=2)
    if df.empty or len(df) < 2:
        return 0.0
    last = df.iloc[-1]
    if direction == "LONG" and last['Close'] > last['Open']:
        return 0.20
    elif direction == "SHORT" and last['Close'] < last['Open']:
        return 0.20
    return 0.0

def trend_strength_bonus(adx_value, base_score):
    if adx_value > 30 and abs(base_score) > 0.5:
        return 0.20 * (1 if base_score > 0 else -1)
    return 0.0

def score_instrument(instr, price, dxy_score, dxy_error):
    errors = []
    tech = get_technicals(instr["ticker"])
    if tech.get("error"):
        errors.append(f"tech({instr['pair']}): {tech['error']}")
    tech_combined = tech["combined"]
    ema50_distance = tech["ema50_distance"]
    adx_value = tech.get("adx_value", 0)

    buying, buy_err = get_buying_pressure(instr["ticker"])
    if buy_err:
        errors.append(f"buying_press({instr['pair']}): {buy_err}")
    buying_score = buying * 3

    vol_score, vol_err = get_volatility_score(instr["ticker"], price)
    if vol_err:
        errors.append(f"volatility({instr['pair']}): {vol_err}")

    intermarket_s = dxy_score
    if dxy_error:
        errors.append(f"intermarket: {dxy_error}")

    vol_trend_s, vt_err = volume_trend_score(instr["ticker"])
    if vt_err:
        errors.append(f"volume_trend({instr['pair']}): {vt_err}")

    total = (
        0.20 * tech_combined +
        0.45 * buying_score +
        0.05 * vol_score +
        0.25 * intermarket_s +
        0.05 * vol_trend_s
    )

    layers = {
        "tech": tech_combined,
        "buying_press": buying_score,
        "volatility": vol_score,
        "intermarket": intermarket_s,
        "volume_trend": vol_trend_s,
    }
    return max(-3, min(3, total)), layers, ema50_distance, adx_value, errors

def call_groq_reasoning(pair, entry, atr, layers, errors=None):
    layer_str = "; ".join([f"{k}={v:.2f}" for k,v in layers.items()])
    err_str = ""
    if errors:
        err_str = " | Data issues: " + "; ".join(errors)

    directional_scores = [layers["tech"], layers["buying_press"], layers["intermarket"], layers["volume_trend"]]
    bearish_count = sum(1 for s in directional_scores if s < -0.5)
    bullish_count = sum(1 for s in directional_scores if s > 0.5)
    alignment_strength = max(bearish_count, bullish_count)

    prompt = (
        f"Trade signal for {pair} at {entry:.5f}. 4h ATR: {atr:.5f}. "
        f"Layer scores: {layer_str}{err_str}. "
        f"All {alignment_strength} out of 4 directional layers are strongly aligned (bearish/bullish). "
        "Provide a concise, punchy reasoning (max 2 sentences) capturing why this trade sets up well. "
        "Also give a confidence score between 4 and 7 (never higher than 7, never lower than 4). "
        "Confidence must reflect the number of aligned layers: at least 5 if 2 layers agree, 6 if 3 layers agree, 7 only if all 4 layers agree. "
        "Low ATR is a positive sign for risk management, not a caution. "
        "Format: CONFIDENCE: 7 | REASONING: [text]"
    )
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
            reason_match = re.search(r'REASONING:\s*(.*)', text)
            conf = int(conf_match.group(1)) if conf_match else 5
            conf = max(4, min(7, conf))
            reason = reason_match.group(1).strip() if reason_match else "Automated signal."
            return conf, reason
    except:
        pass
    return 5, "Multi-factor model (AI unavailable)."

# ========== SIGNAL GENERATION (skips open trades) ==========
def generate_signal():
    # Get currently open trades
    open_pairs = set()
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
        if not open_df.empty:
            open_pairs = set(open_df["pair"].values)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    candidates = []
    for instr in INSTRUMENTS:
        if instr["pair"] in open_pairs:
            continue
        df = get_yahoo_klines(instr["ticker"], interval='4h', days=2)
        if df.empty or len(df) < 2:
            continue
        price = df['Close'].iloc[-1]
        if price > 0:
            candidates.append({"instr": instr, "price": price})

    if not candidates:
        if open_pairs:
            return {"action": "HOLD", "reasoning": "All instruments with valid data already have open trades."}
        else:
            return {"action": "HOLD", "reasoning": "No valid market data (weekend or holiday)."}

    dxy_score, dxy_error = dxy_trend_score()

    all_scored = []
    best = None
    best_score = 0
    best_layers = None
    best_ema_distance = 0.0
    best_adx = 0
    best_errors = []

    for item in candidates:
        instr = item["instr"]
        price = item["price"]

        total_score, layers, ema_dist, adx_val, errors = score_instrument(
            instr, price, dxy_score, dxy_error
        )
        atr, _ = get_4h_atr(instr["ticker"], price)
        if atr / price > 0.02:
            total_score = 0.0
            errors.append("volatility cap triggered (ATR>2%)")
        item["score"] = total_score
        item["atr"] = atr
        item["layers"] = layers
        item["ema_distance"] = ema_dist
        item["adx_value"] = adx_val
        item["errors"] = errors

        all_scored.append(item)

        if best is None or abs(total_score) > abs(best_score):
            best = item
            best_score = total_score
            best_layers = layers
            best_ema_distance = ema_dist
            best_adx = adx_val
            best_errors = errors

    if dxy_error:
        best_errors.append(f"intermarket: {dxy_error}")

    all_scored_sorted = sorted(all_scored, key=lambda x: abs(x["score"]), reverse=True)
    pair_summary_list = []
    for c in all_scored_sorted:
        pair_summary_list.append(f"{c['instr']['pair']}: {c['score']:.2f}")
    pair_summary = " | ".join(pair_summary_list)

    if best is None or abs(best_score) < 1.49:
        best_pair = best["instr"]["pair"] if best else "none"
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"No strong conviction. Best score: {display_score:+.2f}/3 for {best_pair}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All instruments: {pair_summary}")
        return {"action": "HOLD", "reasoning": reason}

    direction = "LONG" if best_score >= 0 else "SHORT"

    best_score += trend_strength_bonus(best_adx, best_score)
    momentum_bonus = momentum_alignment_score(best["instr"]["ticker"], direction)
    best_score += momentum_bonus

    if abs(best_score) < 1.49:
        best_pair = best["instr"]["pair"]
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"No strong conviction after bonuses. Best score: {display_score:+.2f}/3 for {best_pair}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All instruments: {pair_summary}")
        return {"action": "HOLD", "reasoning": reason}

    entry = best["price"]
    atr = best["atr"]
    min_stop = max(1.5 * atr, entry * 0.002)
    stop = entry - min_stop if direction == "LONG" else entry + min_stop
    stop = round(stop, 5)
    risk = abs(entry - stop)
    qty = round(10 / risk, 4)

    mults = [0.4, 0.8, 1.2, 1.6, 2.0]
    tps = []
    for mult in mults:
        if direction == "LONG":
            tps.append(round(entry + mult * risk, 5))
        else:
            tps.append(round(entry - mult * risk, 5))

    conf, reason = call_groq_reasoning(best["instr"]["pair"], entry, atr, best_layers, best_errors)
    if conf < 5:
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"AI confidence too low ({conf}/10). Best score: {display_score:+.2f}/3 for {best['instr']['pair']}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All instruments: {pair_summary}\n{reason}")
        return {"action": "HOLD", "reasoning": reason}

    conviction_display = round(best_score, 2)

    return {
        "action": direction,
        "pair": best["instr"]["pair"],
        "quantity": qty,
        "limit_price": entry,
        "stop_loss": stop,
        "take_profits": tps,
        "confidence_score": conf,
        "reasoning": reason,
        "conviction_score": conviction_display,
        "layers": best_layers,
        "errors": best_errors
    }

# ========== TELEGRAM ==========
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

def main():
    try:
        initialize_trade_files()

        # -------- Check open trades and send status updates --------
        print("Checking open multi‑asset trades...")
        current_prices = fetch_current_prices()
        open_trades_df = check_open_trades(current_prices)

        if open_trades_df is not None and not open_trades_df.empty:
            # Send a summary of each open trade
            for _, t in open_trades_df.iterrows():
                pair = t["pair"]
                entry = float(t["entry"])
                stop = float(t["stop"])
                current = current_prices.get(pair)
                if current is None:
                    continue
                direction = t["action"]
                pnl = (current - entry) / abs(entry - stop) if direction == "LONG" else (entry - current) / abs(entry - stop)
                msg = (f"📊 Open Trade: {pair} {direction}\n"
                       f"Entry: {entry:.5f}  |  Current: {current:.5f}\n"
                       f"P&L: {pnl:+.2f}R  |  Stop: {stop:.5f}")
                send_telegram(msg)

        # -------- Generate new signal --------
        dec = generate_signal()
        action = dec.get('action', 'HOLD')
        if action in ["LONG", "SHORT"]:
            log_signal(dec)
            add_open_trade(dec)
            pair = dec.get('pair', '')
            direction_icon = "🟢" if action == "LONG" else "🔴"
            entry_price = dec.get('limit_price', 0)
            stop_price = dec.get('stop_loss', 0)
            confidence = dec.get('confidence_score', 0)
            conviction = dec.get('conviction_score', 0)
            tps = dec.get('take_profits', [])

            sl_pct = -abs(stop_price - entry_price) / entry_price * 100
            tp_lines = ""
            for i, tp in enumerate(tps, start=1):
                tp_lines += f"TP{i}: {tp:,.5f}\n"
            tp_lines = tp_lines.strip()

            msg = (
                f"${pair}\n"
                f"{action} {direction_icon}\n"
                f"⛔ Entry: {entry_price:,.5f}\n"
                f"🛑 Stop: {stop_price:,.5f} ({sl_pct:+.2f}%)\n"
                f"💰 Targets:\n"
                f"{tp_lines}\n"
                f"Conviction: {conviction:+.2f}/3  |  AI: {confidence}/10"
            )
            send_telegram(msg)
        else:
            send_telegram(f"📊 HOLD\n{dec.get('reasoning', 'No signal')}")

    except Exception as e:
        err_msg = f"Bot crashed: {traceback.format_exc()}"
        print(err_msg)
        send_telegram(err_msg[:500])

if __name__ == "__main__":
    main()