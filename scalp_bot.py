import requests, json, os, traceback, re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

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
    cols = [
        "timestamp", "symbol", "action", "entry", "stop",
        "TP1", "TP2", "conviction", "ai_confidence",
        "outcome", "outcome_time"          # new columns
    ]
    if not os.path.exists(TRADE_LOG_CSV):
        df = pd.DataFrame(columns=cols)
        df.to_csv(TRADE_LOG_CSV, index=False)
    else:
        # ensure new columns exist (for existing log files)
        df = pd.read_csv(TRADE_LOG_CSV)
        for c in ["outcome", "outcome_time"]:
            if c not in df.columns:
                df[c] = ""
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
        "outcome": "",               # empty for now
        "outcome_time": ""
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
            time.sleep(1)
    return None

def get_yahoo_klines(symbol_usdt, interval='5m', days=None, start=None, end=None):
    """
    Fetch klines. If start/end are given, use them; otherwise use days from now.
    """
    yahoo_symbol = symbol_usdt.replace("USDT", "-USD")
    if start is None:
        end = datetime.now()
        start = end - timedelta(days=days if days else 3)
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

# ========== LAYER 1‑5 (unchanged) ==========
def get_technicals(symbol_usdt):
    # ... (same code as original)
    pass

def get_5m_atr(symbol_usdt, current_price):
    # ... (same code)
    pass

def get_buying_pressure(symbol_usdt):
    # ... (same code)
    pass

def get_volatility_score(symbol_usdt, current_price):
    # ... (same code)
    pass

def btc_trend_score():
    # ... (same code)
    pass

def volume_trend_score(symbol_usdt, direction=None):
    # ... (same code)
    pass

def momentum_alignment_score(symbol_usdt, direction, layers):
    # ... (same code)
    pass

def trend_strength_bonus(adx_value, base_score):
    # ... (same code)
    pass

def score_coin(symbol, price, volume_24h, btc_score, btc_error):
    # ... (same code)
    pass

def call_groq_reasoning(symbol, entry, atr, layers, errors=None):
    # ... (same code)
    pass

def generate_signal():
    # ... (same code, unchanged)
    pass

# ========== TRACKING SYSTEM ==========
def check_open_trades():
    """Check all unresolved trades and update their outcome if SL/TP touched."""
    try:
        existing = pd.read_csv(TRADE_LOG_CSV)
    except:
        return  # no file yet

    open_trades = existing[existing["outcome"].isna() | (existing["outcome"] == "")]
    if open_trades.empty:
        return

    now = datetime.now()
    # We'll collect outcomes to send one summary telegram
    outcomes = []

    for idx, row in open_trades.iterrows():
        symbol = row["symbol"]
        action = row["action"]  # "LONG" or "SHORT"
        entry = float(row["entry"])
        stop = float(row["stop"])
        tp1 = float(row["TP1"])
        tp2 = float(row["TP2"])
        entry_time_str = row["timestamp"]
        try:
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        except:
            # If parsing fails, skip this trade
            continue

        # Fetch 5m candles from entry_time to now
        df = get_yahoo_klines(symbol, interval='5m', start=entry_time, end=now)
        if df.empty:
            # If no data, try one more time with a small delay
            time.sleep(1)
            df = get_yahoo_klines(symbol, interval='5m', start=entry_time, end=now)
        if df.empty:
            continue  # can't check, leave for next run

        # For touch detection we need highest high and lowest low
        period_high = df['High'].max()
        period_low = df['Low'].min()

        outcome = None
        exit_price = None
        outcome_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if action == "LONG":
            # STOP-LOSS TOUCHED (check first)
            if period_low <= stop:
                outcome = "SL"
                exit_price = stop
            elif period_high >= tp2:
                outcome = "TP2"
                exit_price = tp2
            elif period_high >= tp1:
                outcome = "TP1"
                exit_price = tp1

        else:  # SHORT
            if period_high >= stop:
                outcome = "SL"
                exit_price = stop
            elif period_low <= tp2:
                outcome = "TP2"
                exit_price = tp2
            elif period_low <= tp1:
                outcome = "TP1"
                exit_price = tp1

        if outcome:
            # Update the CSV row
            existing.at[idx, "outcome"] = outcome
            existing.at[idx, "outcome_time"] = outcome_time
            # Optionally calculate PnL
            pnl_pct = 0
            if action == "LONG":
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100
            outcomes.append(f"{symbol.replace('USDT','')} {action} → {outcome} ({pnl_pct:+.2f}%)")

    if outcomes:
        existing.to_csv(TRADE_LOG_CSV, index=False)
        msg = "🔄 Trade outcomes:\n" + "\n".join(outcomes)
        send_telegram(msg)

# ========== TELEGRAM ==========
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

# ========== MAIN ==========
def main():
    try:
        init_csv()

        # --- STEP 1: check existing open trades ---
        check_open_trades()

        # --- STEP 2: generate new signal ---
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