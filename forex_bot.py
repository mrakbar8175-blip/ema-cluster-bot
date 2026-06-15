#!/usr/bin/env python3
"""
High‑Winrate Forex Swing Bot + Chart + Alerts
Multi‑timeframe confirmation with conviction score.
"""

import requests, json, os, traceback
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ========== ENVIRONMENT ==========
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

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
    "USDCLP","USDCOP","USDPHP","USDIDR","USDINR","USDKRW",
    "USDMYR","USDTWD","USDCNH",
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
                             "TP1","TP2","TP3","score"])
    init_csv(OPEN_TRADES_CSV, ["timestamp","symbol","action","entry","stop",
                               "TP1","TP2","TP3","status","quantity",
                               "original_qty","highest_tp","lot_size"])
    init_csv(TRADE_RESULTS_CSV, ["timestamp","symbol","action","entry","stop",
                                 "TP1","TP2","TP3","status","hit_level",
                                 "close_time","exit_price","quantity","pnl"])

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
        "score": sig["score"],
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

# ========== MULTI‑LAYER SCORING ==========
def score_pair(pair):
    df_d = get_data(pair, interval='1d', days=90)
    if df_d.empty or len(df_d) < 50:
        return 0, None, None, None, None

    df_4h = get_data(pair, interval='4h', days=14)
    if df_4h.empty or len(df_4h) < 50:
        return 0, None, None, None, None

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
        return 0, None, None, None, None

    # 4h indicators
    ema50_4h = ema(df_4h['Close'], 50)
    ema200_4h = ema(df_4h['Close'], 200)
    adx_val, di_plus, di_minus = adx(df_4h)
    rsi_val = rsi(df_4h)
    macd_line, macd_signal, macd_hist, macd_hist_prev = macd(df_4h)
    atr_val = atr(df_4h)
    res, sup = support_resistance_levels(df_4h, 20)

    vol_last = df_4h['Volume'].iloc[-1]
    vol_avg = df_4h['Volume'].iloc[-6:-1].mean() if len(df_4h) >= 6 else vol_last
    vol_surge = vol_last > vol_avg * 1.2

    dxy_df = get_dxy(interval='4h', days=14)
    dxy_aligned = False
    if not dxy_df.empty:
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

    if direction == "LONG":
        ema_align = price > ema50_4h.iloc[-1] and ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
    else:
        ema_align = price < ema50_4h.iloc[-1] and ema50_4h.iloc[-1] < ema200_4h.iloc[-1]
    ema_score = bool_score(ema_align)

    adx_trending = adx_val > 20
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

    total = (
        ema_score * 2.0 +
        adx_score * 1.5 +
        rsi_score * 1.5 +
        macd_score * 1.0 +
        sr_score * 1.0 +
        vol_score * 0.5 +
        dxy_score * 0.5
    )

    if total < 5:
        return 0, None, None, None, None

    return total, direction, price, atr_val, (sup if direction == "LONG" else res)

# ========== SIGNAL GENERATION ==========
def generate_signal():
    open_symbols = set()
    try:
        open_df = pd.read_csv(OPEN_TRADES_CSV)
        if not open_df.empty:
            open_symbols = set(open_df["symbol"].values)
    except:
        pass

    candidates = []
    for pair in FOREX_PAIRS:
        if pair in open_symbols:
            continue
        score, direction, price, atr_val, swing_level = score_pair(pair)
        if direction and score >= 5:
            candidates.append((pair, score, direction, price, atr_val, swing_level))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    pair, score, direction, price, atr_val, swing_level = best

    min_stop_dist = 1.5 * atr_val
    if direction == "LONG":
        stop = min(price - min_stop_dist, swing_level - 0.1 * atr_val)
    else:
        stop = max(price + min_stop_dist, swing_level + 0.1 * atr_val)
    stop = round(stop, 6)

    risk_per_share = abs(price - stop)
    risk_amount = portfolio['balance'] * 0.01
    qty_base = risk_amount / risk_per_share
    lot_size = max(0.01, round(qty_base / 1000, 2))
    actual_units = lot_size * 1000

    mults = [0.5, 1.0, 1.5]
    tps = []
    for m in mults:
        if direction == "LONG":
            tps.append(round(price + m * risk_per_share, 6))
        else:
            tps.append(round(price - m * risk_per_share, 6))

    return {
        "action": direction,
        "symbol": pair,
        "quantity": actual_units,
        "lot_size": lot_size,
        "limit_price": price,
        "stop_loss": stop,
        "take_profits": tps,
        "score": score,
    }

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
    mults = [0.5, 1.0, 1.5]
    fractions = [0.50, 0.30, 0.20]

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
                    alerts.append(f"🚀 {sym} {direction} TP{i+1} hit — partial close, SL to BE")

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

# ========== CHART (FIXED – NO VWAP DIVISION BY ZERO) ==========
def send_trade_chart(signal):
    """
    Generate dark 4h chart with EMA50, VWAP, entry, SL, TPs.
    If total volume is zero, skip VWAP. If any error, send TradingView link.
    """
    sym = signal['symbol']

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        # Fetch 21 days of 4h data for enough candles
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

        # EMA50 – use available length if less than 50
        ema50 = df['Close'].ewm(span=min(50, len(df)), adjust=False).mean()
        addplots = [
            mpf.make_addplot(ema50, color='#f39c12', width=1.5, label='EMA50')
        ]

        # VWAP only if we have non‑zero total volume
        total_vol = df['Volume'].sum()
        if total_vol > 0:
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
            addplots.append(
                mpf.make_addplot(vwap, color='#3498db', width=1, linestyle='--', label='VWAP')
            )

        title = f"{sym} 4h"

        fig, axes = mpf.plot(df, type='candle', style=mpf_style,
                             title=title, ylabel='Price', addplot=addplots,
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
            resp = requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': img})
            if resp.status_code != 200:
                print(f"Telegram photo send failed: {resp.text}")
        os.remove(chart_path)
        return  # success

    except Exception as e:
        print(f"Chart image error: {e}")
        send_telegram(f"⚠️ Chart image unavailable ({str(e)[:80]}).")

    # --- Fallback: TradingView link ---
    studies = "&studies[]=STD%3BEMA%3B50&studies[]=STD%3BVWAP"
    tv_url = f"https://www.tradingview.com/chart/?symbol=FX:{sym}&interval=240{studies}"
    send_telegram(f"📈 {sym} 4h chart (EMA50 + VWAP): {tv_url}")

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
    tp_str = " / ".join([f"{tp:.5f} ({p} pips)" for tp, p in zip(tps, tp_pips)])

    return (
        f"{dirn} {sym}\n"
        f"Conviction: {score:.1f}/7.0\n"
        f"Entry: {entry:.5f}\n"
        f"Stop Loss: {stop:.5f} ({sl_pips} pips)\n"
        f"Take Profits: {tp_str}\n"
        f"Lot Size: {lot:.2f} (Risk: 1%)"
    )

# ========== MAIN ==========
def main():
    try:
        initialize_trade_files()
        check_open_trades()

        if daily_pnl() <= portfolio['daily_loss_limit']:
            send_telegram("Daily loss limit reached. No new trades today.")
            return

        sig = generate_signal()
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