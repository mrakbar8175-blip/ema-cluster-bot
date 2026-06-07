import requests, json, os, traceback
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ========== ENVIRONMENT ==========
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# ========== UNIVERSE (top liquid coins, same as before) ==========
COIN_LIST = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT",
    "STXUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "TIAUSDT", "SEIUSDT", "RUNEUSDT", "GRTUSDT", "AAVEUSDT",
    "ALGOUSDT", "SANDUSDT", "MANAUSDT", "THETAUSDT", "FTMUSDT",
    "EOSUSDT", "MKRUSDT", "LDOUSDT", "IMXUSDT", "FLOWUSDT",
    "XTZUSDT", "NEOUSDT", "KSMUSDT", "ZECUSDT", "DASHUSDT",
    "EGLDUSDT", "MINAUSDT", "GALAUSDT", "HNTUSDT", "CFXUSDT",
    "ARUSDT", "FETUSDT", "AGIXUSDT", "OCEANUSDT", "1INCHUSDT",
    "CRVUSDT", "AXSUSDT", "CHZUSDT", "ENJUSDT", "BATUSDT",
    "SNXUSDT", "COMPUSDT", "YFIUSDT", "SUSHIUSDT", "ZRXUSDT",
    "RENUSDT", "CELOUSDT", "LRCUSDT", "ANKRUSDT", "STORJUSDT",
    "COTIUSDT", "KAVAUSDT", "ICXUSDT", "ONTUSDT", "ZILUSDT",
    "WAVESUSDT", "QTUMUSDT", "OMGUSDT", "BANDUSDT", "DENTUSDT",
    "HOTUSDT", "IOSTUSDT", "RVNUSDT", "SCUSDT", "ZENUSDT",
    "CKBUSDT", "SKLUSDT", "CTSIUSDT", "CTKUSDT", "LINAUSDT",
    "TRBUSDT", "BALUSDT", "PERPUSDT", "BNTUSDT", "RSRUSDT",
    "TOMOUSDT", "DGBUSDT", "DUSKUSDT", "REEFUSDT", "ALPHAUSDT",
    "FORTHUSDT", "POLSUSDT", "C98USDT", "RAREUSDT", "ATAUSDT",
    "IDEXUSDT", "MLNUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT",
    "FLOKIUSDT", "SHIBUSDT", "APTUSDT", "SUIUSDT"
]
COIN_LIST = list(set(COIN_LIST))   # deduplicate

# ========== CSV FILES ==========
TRADE_LOG_CSV = "trade_log.csv"
OPEN_TRADE_CSV = "open_trade.csv"
TRADE_RESULTS_CSV = "trade_results.csv"

# ========== HELPERS ==========
def get_yahoo_klines(symbol_usdt, interval='1d', days=200):
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

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

# ========== CSV LOGGING ==========
def init_csv(filepath, columns):
    if not os.path.exists(filepath):
        pd.DataFrame(columns=columns).to_csv(filepath, index=False)

def initialize_files():
    init_csv(TRADE_LOG_CSV, ["timestamp", "symbol", "direction", "entry", "stop", "risk", "target_10R"])
    init_csv(OPEN_TRADE_CSV, ["symbol", "direction", "entry", "initial_stop", "current_stop", "breakeven_triggered", "target_10R"])
    init_csv(TRADE_RESULTS_CSV, ["symbol", "direction", "entry", "exit_price", "exit_reason", "pnl_R", "close_time"])

def save_open_trade(data):
    df = pd.DataFrame([data])
    df.to_csv(OPEN_TRADE_CSV, index=False)

def load_open_trade():
    try:
        df = pd.read_csv(OPEN_TRADE_CSV)
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except:
        return None

def clear_open_trade():
    pd.DataFrame().to_csv(OPEN_TRADE_CSV, index=False)

def log_signal(entry, direction, symbol, stop, risk, target_10r):
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "target_10R": target_10r
    }
    df = pd.DataFrame([row])
    try:
        existing = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass
    df.to_csv(TRADE_LOG_CSV, index=False)

def log_result(symbol, direction, entry, exit_price, exit_reason, pnl_R):
    row = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_R": round(pnl_R, 2),
        "close_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.DataFrame([row])
    try:
        existing = pd.read_csv(TRADE_RESULTS_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass
    df.to_csv(TRADE_RESULTS_CSV, index=False)

# ========== EMA CLUSTER LOGIC ==========
def ema_cluster_aligned(df):
    """Check if 9>21>50>200 and all sloping up (long) or opposite for short."""
    if len(df) < 200:
        return None
    closes = df['Close']
    ema9 = closes.ewm(span=9, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema200 = closes.ewm(span=200, adjust=False).mean()

    # current values
    e9 = ema9.iloc[-1]
    e21 = ema21.iloc[-1]
    e50 = ema50.iloc[-1]
    e200 = ema200.iloc[-1]

    # slopes (simple difference from previous day)
    def sloping_up(series):
        return series.iloc[-1] > series.iloc[-2]
    def sloping_down(series):
        return series.iloc[-1] < series.iloc[-2]

    # long condition
    if e9 > e21 > e50 > e200 and sloping_up(ema9) and sloping_up(ema50):
        return "LONG"
    # short condition
    if e9 < e21 < e50 < e200 and sloping_down(ema9) and sloping_down(ema50):
        return "SHORT"
    return None

def check_pullback_entry(df, direction):
    """Return True if yesterday's candle touched the 9 EMA and today's candle closed beyond the 9 EMA in the direction."""
    closes = df['Close']
    highs = df['High']
    lows = df['Low']
    ema9 = closes.ewm(span=9, adjust=False).mean()

    # We need at least 2 daily candles
    if len(df) < 10:
        return None, None

    # Use yesterday as the pullback candle, today as the confirmation candle
    yesterday = {
        'open': df['Open'].iloc[-2],
        'high': highs.iloc[-2],
        'low': lows.iloc[-2],
        'close': closes.iloc[-2]
    }
    today = {
        'open': df['Open'].iloc[-1],
        'high': highs.iloc[-1],
        'low': lows.iloc[-1],
        'close': closes.iloc[-1]
    }
    ema9_yesterday = ema9.iloc[-2]
    ema9_today = ema9.iloc[-1]

    if direction == "LONG":
        # Pullback touch: yesterday's low <= ema9_yesterday (within ~0.5%? We'll just use <=)
        touch = yesterday['low'] <= ema9_yesterday
        # Confirmation: today's close > ema9_today and today's close > yesterday's close
        confirm = today['close'] > ema9_today and today['close'] > yesterday['close']
        if touch and confirm:
            return today['close'], today['low']   # entry = today's close, stop = today's low
    else:  # SHORT
        touch = yesterday['high'] >= ema9_yesterday
        confirm = today['close'] < ema9_today and today['close'] < yesterday['close']
        if touch and confirm:
            return today['close'], today['high']  # entry = today's close, stop = today's high
    return None, None

# ========== TRADE MANAGEMENT ==========
def manage_open_trade():
    """Check the single open trade and exit if conditions met. Return True if trade is still open."""
    trade = load_open_trade()
    if not trade:
        return False   # no trade

    sym = trade['symbol']
    direction = trade['direction']
    entry = float(trade['entry'])
    current_stop = float(trade['current_stop'])
    initial_stop = float(trade['initial_stop'])
    breakeven_done = trade['breakeven_triggered']
    target_10r = float(trade['target_10R'])

    # Fetch recent daily data (enough to check 9 EMA)
    df = get_yahoo_klines(sym, interval='1d', days=30)
    if df.empty:
        send_telegram(f"⚠️ Unable to manage {sym} trade – no data.")
        return True   # keep trade open for now

    current_price = df['Close'].iloc[-1]
    daily_high = df['High'].iloc[-1]
    daily_low = df['Low'].iloc[-1]
    ema9 = df['Close'].ewm(span=9, adjust=False).mean().iloc[-1]

    exit_reason = None
    exit_price = None

    # Check stop loss first
    if direction == "LONG":
        if daily_low <= current_stop:
            exit_reason = "STOP LOSS"
            exit_price = current_stop
        elif current_price >= target_10r:
            exit_reason = "TARGET 10R"
            exit_price = target_10r
        elif current_price < ema9:   # daily close below 9 EMA
            exit_reason = "EMA CLOSE"
            exit_price = current_price   # exit at close
    else:  # SHORT
        if daily_high >= current_stop:
            exit_reason = "STOP LOSS"
            exit_price = current_stop
        elif current_price <= target_10r:
            exit_reason = "TARGET 10R"
            exit_price = target_10r
        elif current_price > ema9:
            exit_reason = "EMA CLOSE"
            exit_price = current_price

    if exit_reason:
        # Calculate P&L in R
        risk = abs(entry - initial_stop)
        if direction == "LONG":
            pnl = (exit_price - entry) / risk
        else:
            pnl = (entry - exit_price) / risk
        log_result(sym, direction, entry, exit_price, exit_reason, pnl)
        clear_open_trade()
        send_telegram(f"🚪 Trade closed: {sym} {direction}\nExit: {exit_reason} @ {exit_price:.5f}\nP&L: {pnl:.2f}R")
        return False   # trade closed

    # Breakeven adjustment
    if not breakeven_done:
        risk = abs(entry - initial_stop)
        if direction == "LONG" and current_price >= entry + risk:
            # move stop to entry
            trade['current_stop'] = entry
            trade['breakeven_triggered'] = True
            save_open_trade(trade)
            send_telegram(f"🔄 {sym} {direction} – Stop moved to breakeven.")
        elif direction == "SHORT" and current_price <= entry - risk:
            trade['current_stop'] = entry
            trade['breakeven_triggered'] = True
            save_open_trade(trade)
            send_telegram(f"🔄 {sym} {direction} – Stop moved to breakeven.")

    return True   # trade still open

# ========== NEW SIGNAL SCANNING ==========
def find_new_signal():
    """Scan universe for a valid EMA cluster + pullback entry. Return signal dict or None."""
    candidates = []
    for sym in COIN_LIST:
        df = get_yahoo_klines(sym, interval='1d', days=200)
        if df.empty or len(df) < 200:
            continue
        direction = ema_cluster_aligned(df)
        if not direction:
            continue
        entry, stop = check_pullback_entry(df, direction)
        if entry is None:
            continue
        candidates.append({
            "symbol": sym,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "risk": abs(entry - stop)
        })
    if not candidates:
        return None
    # pick the one with the smallest risk (tighter stop) – proxy for best setup
    candidates.sort(key=lambda x: x['risk'])
    return candidates[0]

# ========== MAIN ==========
def main():
    initialize_files()

    # Step 1: manage existing trade
    trade_active = manage_open_trade()
    if trade_active:
        print("Trade still open – no new signals.")
        return

    # Step 2: find new trade
    signal = find_new_signal()
    if not signal:
        send_telegram("📊 HOLD\nNo EMA cluster + pullback setup found.")
        return

    # Step 3: enter new trade
    entry = signal['entry']
    stop = signal['stop']
    direction = signal['direction']
    risk = signal['risk']
    target_10r = entry + 10 * risk if direction == "LONG" else entry - 10 * risk

    # Save open trade
    trade_data = {
        "symbol": signal['symbol'],
        "direction": direction,
        "entry": entry,
        "initial_stop": stop,
        "current_stop": stop,
        "breakeven_triggered": False,
        "target_10R": target_10r
    }
    save_open_trade(trade_data)

    # Log signal
    log_signal(entry, direction, signal['symbol'], stop, risk, target_10r)

    # Telegram message
    emoji = "🟢" if direction == "LONG" else "🔴"
    msg = (
        f"🚨 NEW DAILY SIGNAL 🚨\n"
        f"${signal['symbol'].replace('USDT','/USDT')}\n"
        f"{direction} {emoji}\n"
        f"⛔ Entry: {entry:.5f}\n"
        f"🛑 Stop: {stop:.5f}\n"
        f"🎯 1:10 Target: {target_10r:.5f}\n"
        f"Risk: {risk:.5f}\n"
        f"Exit rules: daily close beyond 9 EMA or hit target"
    )
    send_telegram(msg)

if __name__ == "__main__":
    main()
