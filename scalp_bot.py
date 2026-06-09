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

# ========== UNIVERSE (top 20 by 24h volume) ==========
STATIC_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT"
]

# ========== CSV LOGGING ==========
TRADE_LOG_CSV = "scalp_trade_log.csv"

def init_csv():
    if not os.path.exists(TRADE_LOG_CSV):
        df = pd.DataFrame(columns=[
            "timestamp", "symbol", "action", "entry", "stop",
            "TP1", "TP2", "conviction", "ai_confidence"
        ])
        df.to_csv(TRADE_LOG_CSV, index=False)

def log_signal(signal):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": signal["symbol"],
        "action": signal["action"],
        "entry": signal["limit_price"],
        "stop": signal["stop_loss"],
        "TP1": signal["take_profits"][0],
        "TP2": signal["take_profits"][1],
        "conviction": signal["conviction_score"],
        "ai_confidence": signal["confidence_score"],
    }
    df = pd.DataFrame([row])
    try:
        existing = pd.read_csv(TRADE_LOG_CSV)
        updated = pd.concat([existing, df], ignore_index=True)
    except:
        updated = df
    updated.to_csv(TRADE_LOG_CSV, index=False)

# ========== DATA HELPERS ==========
def fetch_coingecko(url, retries=2):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except:
            pass
    return None

def get_yahoo_klines(symbol_usdt, interval='5m', days=3):
    yahoo_symbol = symbol_usdt.replace("USDT", "-USD")
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(yahoo_symbol, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def get_live_price(symbol_usdt):
    coin_id = symbol_usdt.replace("USDT", "").lower()
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    data = fetch_coingecko(url)
    if data and coin_id in data:
        return data[coin_id]["usd"]
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=1)
    if not df.empty:
        return df['Close'].iloc[-1]
    return 0

# ========== LAYER 1: TECHNICALS (5m) – weight 20% ==========
def get_technicals(symbol_usdt):
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=2)
    error = None
    if df.empty or len(df) < 100:
        error = f"insufficient 5m data ({len(df)} candles)"
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

    window = 5
    lookback = min(100, len(highs))
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
            structure_score = 2.0
        elif (not last_hh) and (not last_hl):
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

def get_5m_atr(symbol_usdt, current_price):
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=1)
    if df.empty or len(df) < 14:
        return current_price * 0.002, "ATR data insufficient, using 0.2% fallback"
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if pd.isna(atr):
        return current_price * 0.002, "ATR calculation failed, using 0.2% fallback"
    return atr, None

# ========== LAYER 2: BUYING PRESSURE (5m, 24 candles) – weight 45% ==========
def get_buying_pressure(symbol_usdt):
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=1)
    if df.empty or len(df) < 24:
        return 0.0, f"insufficient data ({len(df)} candles)"
    df = df.tail(24)
    buy_vol = df.loc[df['Close'] > df['Open'], 'Volume'].sum()
    sell_vol = df.loc[df['Close'] <= df['Open'], 'Volume'].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0, "zero total volume"
    return (buy_vol - sell_vol) / total, None

# ========== LAYER 3: VOLATILITY (5m) – weight 5% ==========
def get_volatility_score(symbol_usdt, current_price):
    atr, atr_err = get_5m_atr(symbol_usdt, current_price)
    atr_pct = atr / current_price * 100
    if atr_pct < 0.2 or atr_pct > 1.5:
        return -1, atr_err
    return 1, None

# ========== LAYER 4: INTERMARKET (BTC 5m trend) – weight 25% ==========
def btc_trend_score():
    df = get_yahoo_klines("BTCUSDT", interval='5m', days=2)
    if df.empty or len(df) < 100:
        return 0, "BTC data unavailable"
    closes = df['Close']
    ema50 = closes.ewm(span=50, adjust=False).mean()
    current = closes.iloc[-1]
    if current > ema50.iloc[-1]:
        return 2, None
    else:
        return -2, None

# ========== LAYER 5: VOLUME TREND (5m, 6 candles) – weight 5% ==========
def volume_trend_score(symbol_usdt, direction=None):
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=1)
    if df.empty or len(df) < 12:
        return 0, "volume data insufficient"
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

# ========== MOMENTUM ALIGNMENT (directional) ==========
def momentum_alignment_score(symbol_usdt, direction, layers):
    df = get_yahoo_klines(symbol_usdt, interval='5m', days=1)
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
        if direction == "LONG":
            return 0.20
        else:
            return -0.20
    return 0.0

# ========== TREND STRENGTH BONUS ==========
def trend_strength_bonus(adx_value, base_score):
    if adx_value > 30 and abs(base_score) > 0.5:
        return 0.30 * (1 if base_score > 0 else -1)
    elif adx_value > 25 and abs(base_score) > 0.5:
        return 0.20 * (1 if base_score > 0 else -1)
    return 0.0

# ========== SCORING ENGINE ==========
def score_coin(symbol, price, volume_24h, btc_score, btc_error):
    errors = []
    tech = get_technicals(symbol)
    if tech.get("error"):
        errors.append(f"tech({symbol}): {tech['error']}")
    tech_combined = tech["combined"]
    adx_value = tech.get("adx_value", 0)
    trend_dir = tech.get("trend_dir", "up")

    buying, buy_err = get_buying_pressure(symbol)
    if buy_err:
        errors.append(f"buying_press({symbol}): {buy_err}")
    buying_score = buying * 3

    vol_score, vol_err = get_volatility_score(symbol, price)
    if vol_err:
        errors.append(f"volatility({symbol}): {vol_err}")

    intermarket_s = btc_score
    if btc_error:
        errors.append(f"intermarket: {btc_error}")

    vol_trend_s, vt_err = volume_trend_score(symbol, direction=trend_dir)
    if vt_err:
        errors.append(f"volume_trend({symbol}): {vt_err}")

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
    return max(-3, min(3, total)), layers, adx_value, trend_dir, errors

# ========== AI REASONING (confidence 4‑7) ==========
def call_groq_reasoning(symbol, entry, atr, layers, errors=None):
    layer_str = "; ".join([f"{k}={v:.2f}" for k,v in layers.items()])
    err_str = ""
    if errors:
        err_str = " | Data issues: " + "; ".join(errors)

    directional_scores = [layers["tech"], layers["buying_press"], layers["intermarket"], layers["volume_trend"]]
    bearish_count = sum(1 for s in directional_scores if s < -0.5)
    bullish_count = sum(1 for s in directional_scores if s > 0.5)
    alignment_strength = max(bearish_count, bullish_count)

    prompt = (
        f"Scalp signal for {symbol} at {entry}. 5m ATR: {atr:.4f}. "
        f"Layer scores: {layer_str}{err_str}. "
        f"All {alignment_strength} out of 4 directional layers are strongly aligned (bearish/bullish). "
        "Provide a concise, punchy reasoning (max 2 sentences) capturing why this scalp sets up well. "
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

# ========== SIGNAL GENERATION (top 20 coins, positive RR) ==========
def generate_signal():
    cg_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=volume_desc&per_page=50&page=1"
    coins_data = fetch_coingecko(cg_url)
    if not coins_data:
        return {"action": "HOLD", "reasoning": "CoinGecko market data unavailable."}

    cg_map = {}
    for coin in coins_data:
        sym = coin.get("symbol", "").upper() + "USDT"
        if coin.get("current_price", 0) > 0:
            cg_map[sym] = {"price": coin["current_price"], "volume": coin.get("total_volume", 0)}

    candidates = []
for sym in cg_map:
    if "QUQ" in sym.upper():          # ← skip QUQ and anything similar
        continue
    candidates.append({"symbol": sym, "price": cg_map[sym]["price"], "volume": cg_map[sym]["volume"]})
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    candidates = candidates[:20]

    if not candidates:
        return {"action": "HOLD", "reasoning": "No liquid coins available."}

    btc_score, btc_error = btc_trend_score()

    all_scored = []
    best = None
    best_score = 0
    best_layers = None
    best_adx = 0
    best_trend_dir = None
    best_errors = []

    for coin in candidates:
        sym = coin["symbol"]
        price = coin["price"]
        volume = coin["volume"]

        total_score, layers, adx_val, trend_dir, errors = score_coin(
            sym, price, volume, btc_score, btc_error
        )
        atr, _ = get_5m_atr(sym, price)
        if atr / price > 0.015:
            total_score = 0.0
            errors.append("volatility cap triggered (ATR>1.5%)")
        coin["score"] = total_score
        coin["atr"] = atr
        coin["layers"] = layers
        coin["adx_value"] = adx_val
        coin["trend_dir"] = trend_dir
        coin["errors"] = errors

        all_scored.append(coin)

        if best is None or abs(total_score) > abs(best_score):
            best = coin
            best_score = total_score
            best_layers = layers
            best_adx = adx_val
            best_trend_dir = trend_dir
            best_errors = errors

    if btc_error:
        best_errors.append(f"intermarket: {btc_error}")

    all_scored_sorted = sorted(all_scored, key=lambda x: abs(x["score"]), reverse=True)
    coin_summary_list = []
    for c in all_scored_sorted:
        coin_summary_list.append(f"{c['symbol'].replace('USDT','')}: {c['score']:.2f}")
    coin_summary = " | ".join(coin_summary_list)

    if best is None or abs(best_score) < 0.99:
        best_sym = best["symbol"] if best else "none"
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"No strong conviction. Best score: {display_score:+.2f}/3 for {best_sym}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All coins: {coin_summary}")
        return {"action": "HOLD", "reasoning": reason}

    direction = "LONG" if best_score >= 0 else "SHORT"

    if best_trend_dir:
        if (direction == "LONG" and best_trend_dir == "down") or \
           (direction == "SHORT" and best_trend_dir == "up"):
            best_sym = best["symbol"]
            layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
            err_str = ""
            if best_errors:
                err_str = " | Errors: " + "; ".join(best_errors)
            display_score = round(best_score, 2)
            reason = (f"Signal {direction} rejected due to 5m trend filter ({best_trend_dir}). "
                      f"Best score: {display_score:+.2f}/3 for {best_sym}.\n"
                      f"Layers: {layer_str}{err_str}\n"
                      f"All coins: {coin_summary}")
            return {"action": "HOLD", "reasoning": reason}

    best_score += trend_strength_bonus(best_adx, best_score)
    momentum_bonus = momentum_alignment_score(best["symbol"], direction, best_layers)
    best_score += momentum_bonus

    if abs(best_score) < 0.99:
        best_sym = best["symbol"]
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"No strong conviction after bonuses. Best score: {display_score:+.2f}/3 for {best_sym}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All coins: {coin_summary}")
        return {"action": "HOLD", "reasoning": reason}

    live_entry = get_live_price(best["symbol"])
    if live_entry <= 0:
        return {"action": "HOLD", "reasoning": "Could not fetch live price for entry."}

    atr = best["atr"]
    min_stop = max(1.5 * atr, live_entry * 0.002)
    stop = live_entry - min_stop if direction == "LONG" else live_entry + min_stop
    stop = round(stop, 6)
    risk = abs(live_entry - stop)
    qty = round(10 / risk, 4)

    mults = [0.8, 1.2]
    tps = []
    for mult in mults:
        if direction == "LONG":
            tps.append(round(live_entry + mult * risk, 6))
        else:
            tps.append(round(live_entry - mult * risk, 6))

    conf, reason = call_groq_reasoning(best["symbol"], live_entry, atr, best_layers, best_errors)
    if conf < 4:
        layer_str = "; ".join([f"{k}={v:.2f}" for k,v in best_layers.items()])
        err_str = ""
        if best_errors:
            err_str = " | Errors: " + "; ".join(best_errors)
        display_score = round(best_score, 2)
        reason = (f"AI confidence too low ({conf}/10). Best score: {display_score:+.2f}/3 for {best['symbol']}.\n"
                  f"Layers: {layer_str}{err_str}\n"
                  f"All coins: {coin_summary}\n{reason}")
        return {"action": "HOLD", "reasoning": reason}

    conviction_display = round(best_score, 2)

    return {
        "action": direction,
        "symbol": best["symbol"],
        "quantity": qty,
        "limit_price": live_entry,
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
        init_csv()
        dec = generate_signal()
        action = dec.get('action', 'HOLD')
        if action in ["LONG", "SHORT"]:
            log_signal(dec)
            raw_symbol = dec.get('symbol', '')
            symbol = raw_symbol.replace("USDT", "/USDT") if raw_symbol else ""
            direction_icon = "🟢" if action == "LONG" else "🔴"
            entry_price = dec.get('limit_price', 0)
            stop_price = dec.get('stop_loss', 0)
            confidence = dec.get('confidence_score', 0)
            conviction = dec.get('conviction_score', 0)
            tps = dec.get('take_profits', [])

            sl_pct = -abs(stop_price - entry_price) / entry_price * 100
            tp_lines = ""
            for i, tp in enumerate(tps, start=1):
                tp_lines += f"TP{i}: {tp:,.6f}\n"
            tp_lines = tp_lines.strip()

            msg = (
                f"⚡ SCALP ${symbol}\n"
                f"{action} {direction_icon}\n"
                f"⛔ Entry: {entry_price:,.6f}\n"
                f"🛑 Stop: {stop_price:,.6f} ({sl_pct:+.2f}%)\n"
                f"💰 Targets:\n"
                f"{tp_lines}\n"
                f"Conviction: {conviction:+.2f}/3  |  AI: {confidence}/10"
            )
            send_telegram(msg)
        else:
            msg = f"📊 SCALP HOLD\n{dec.get('reasoning', 'No signal')}"
            send_telegram(msg)
    except Exception as e:
        err_msg = f"Scalp bot crashed: {traceback.format_exc()}"
        print(err_msg)
        send_telegram(err_msg[:500])

if __name__ == "__main__":
    main()