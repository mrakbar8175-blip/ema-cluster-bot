#!/usr/bin/env python3
"""
AlphaSwing FX v1.0 – Professional Forex Quant Signal Generator
================================================================
30-minute automated cycle: Monitor -> Route Alerts -> Generate Signals -> Auto-Add Trades

Key Features:
- 29 MAJOR FX PAIRS + GOLD + SILVER universe
- DXY-BASED REGIME DETECTION (institutional approach)
- SESSION MOMENTUM FACTOR (replaces CLV for FX)
- 3-BRAIN ADAPTIVE LEARNING (Bull/Bear/Sideways)
- OUT-OF-SAMPLE VALIDATION (prevents overfitting)
- 5-CHANNEL DISCORD ROUTING (Main/Trade Updates/Wins-Losses/Public/VIP)
- 50x LEVERAGE DISPLAY
- AUTO-ADD ENABLED BY DEFAULT
- WEEKEND-AWARE (skips signals during market closure)

Architecture mirrors AlphaSwing Crypto v8.5.1 for consistency.
"""

import os
import json
import time
import sys
import atexit
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "trading": {
        "max_signals": 1,
        "max_concurrent_risky_trades": 3,
        "risk_per_trade_pct": 1.0,
        "min_score_to_enter": 1.5,
        "atr_stop_multiplier": 2.0,
        "tp_multipliers": [0.5, 1.0, 1.5, 2.0, 3.0],
        "fractions": [0.40, 0.20, 0.15, 0.15, 0.10],
        "trailing_atr_multiplier": 1.5,
    },
    "universe": {
        # 29 pairs: 7 majors + 19 crosses + 2 commodities + 1 regime indicator
        "pairs": [
            # Majors (7)
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
            "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
            # Crosses (19)
            "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X",
            "EURAUD=X", "EURCAD=X", "EURNZD=X", "GBPAUD=X",
            "GBPCAD=X", "GBPNZD=X", "GBPCHF=X", "AUDJPY=X",
            "AUDNZD=X", "AUDCAD=X", "AUDCHF=X", "CADJPY=X",
            "CADCHF=X", "NZDJPY=X", "CHFJPY=X",
            # Commodities (2)
            "GC=F",  # Gold
            "SI=F",  # Silver
        ],
        "regime_indicator": "DX-Y.NYB",  # DXY - US Dollar Index
        "blacklist": [],  # User-defined blacklist
    },
    "files": {
        "portfolio_file": "portfolio.json",
        "signal_log": "signal_log.csv",
        "open_trades_file": "open_trades.json",
        "trade_results_file": "trade_results.csv",
        "perf_counter_file": "perf_counter.txt",
        "banned_pairs_file": "banned_pairs.json",
        "weights_bull_file": "learned_weights_bull.json",
        "weights_bear_file": "learned_weights_bear.json",
        "weights_sideways_file": "learned_weights_sideways.json",
        "regime_state_file": "regime_state.json",
    },
    "loop_interval_hours": 4,
    "report_every_n_trades": 10,
    "learning": {
        "min_trades_to_learn": 10,
        "rolling_window": 30,
        "learning_rate": 0.7,
        "min_weight": 0.10,
        "max_weight": 0.70,
        "min_penalty": -0.60,
        "max_penalty": -0.05,
        "penalty_step": 0.05,
        "validation_split": 0.30,
    },
    "regime": {
        "ema_period": 50,
        "adx_period": 14,
        "adx_trend_threshold": 20,
        "price_ema_threshold_pct": 0.005,  # 0.5% for DXY (more sensitive than BTC)
        "confirmation_candles": 3,
        "hysteresis_multiplier": 1.5,
    },
    "automation": {
        "auto_add_signals": True,  # ENABLED BY DEFAULT
    },
    "leverage": {
        "display_multiplier": 50,
    },
    "session_momentum": {
        "lookback_hours": 24,  # 24 hours = 24 candles on 1H chart
        "normalization_factor": 0.0015,  # 0.15% per hour = strong trend
    },
}

# ==============================================================================
# MULTI-CHANNEL WEBHOOK CONFIGURATION
# ==============================================================================
WEBHOOKS = {
    "MAIN": os.environ.get("FX_WEBHOOK_MAIN"),
    "TRADE_UPDATES": os.environ.get("FX_WEBHOOK_TRADE_UPDATES"),
    "WINS_LOSSES": os.environ.get("FX_WEBHOOK_WINS_LOSSES"),
    "PUBLIC": os.environ.get("FX_WEBHOOK_PUBLIC"),
    "VIP": os.environ.get("FX_WEBHOOK_VIP"),
}

if os.environ.get("FX_AUTO_ADD_SIGNALS", "").lower() == "false":
    CONFIG["automation"]["auto_add_signals"] = False
    print("[*] Auto-add signals DISABLED via environment variable")

LOCK_FILE = "bot.lock"

DEFAULT_WEIGHTS = {
    "momentum": 0.60,
    "session_momentum": 0.30,  # Replaces CLV
    "volatility": 0.10,
    "regime_penalty": -0.30,
    "updated_at": "never",
    "based_on_trades": 0,
    "validation_score": 0.0,
}

# ==============================================================================
# PID-BASED FILE LOCKING
# ==============================================================================
def acquire_lock():
    """Acquires a PID-based file lock to prevent concurrent execution."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                data = json.load(f)
            old_pid = data.get("pid")
            if old_pid:
                try:
                    os.kill(old_pid, 0)
                    print(f"[!] Another instance is running (PID {old_pid}). Exiting.")
                    sys.exit(0)
                except (OSError, ProcessLookupError):
                    print(f"[*] Taking over stale lock (PID {old_pid} is dead).")
        except Exception as e:
            print(f"[!] Error reading lock file: {e}")
    
    with open(LOCK_FILE, 'w') as f:
        json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)

def release_lock():
    """Releases the file lock only if it belongs to the current process."""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                data = json.load(f)
            if data.get("pid") == os.getpid():
                os.remove(LOCK_FILE)
    except Exception as e:
        print(f"[!] Error releasing lock: {e}")

atexit.register(release_lock)

# ==============================================================================
# AUTO FILE INITIALIZATION
# ==============================================================================
def initialize_files():
    """Creates all necessary JSON and CSV files with default headers."""
    print("[*] Initializing data files...")
    
    files_to_create = [
        (CONFIG["files"]["portfolio_file"], {"balance": 1000.0}, "json"),
        (CONFIG["files"]["open_trades_file"], [], "json"),
        (CONFIG["files"]["banned_pairs_file"], [], "json"),
        (CONFIG["files"]["perf_counter_file"], "0", "txt"),
        (CONFIG["files"]["weights_bull_file"], DEFAULT_WEIGHTS.copy(), "json"),
        (CONFIG["files"]["weights_bear_file"], DEFAULT_WEIGHTS.copy(), "json"),
        (CONFIG["files"]["weights_sideways_file"], DEFAULT_WEIGHTS.copy(), "json"),
        (CONFIG["files"]["regime_state_file"], {
            "current_regime": "NEUTRAL", 
            "confirmed_at": None, 
            "candles_in_regime": 0
        }, "json"),
    ]
    
    for filepath, default_content, ftype in files_to_create:
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                if ftype == "json":
                    json.dump(default_content, f, indent=2)
                else:
                    f.write(str(default_content))
            print(f"  ✓ Created {filepath}")
    
    csv_files = [
        (CONFIG["files"]["signal_log"], [
            "timestamp", "symbol", "direction", "entry", "stop", "tp1", "tp2", "tp3", 
            "score", "mom_z", "session_mom", "qty", "notional", "regime", "regime_confidence"
        ]),
        (CONFIG["files"]["trade_results_file"], [
            "open_time", "close_time", "symbol", "direction", "entry", "stop",
            "tp1", "tp2", "tp3", "tp4", "tp5", "exit_price", "qty", "pnl_pct",
            "pnl_dollars", "r_multiple", "hit_level", "score", "mom_z", "session_mom", 
            "vol_regime_score", "regime", "regime_confidence"
        ]),
    ]
    
    for filepath, cols in csv_files:
        if not os.path.exists(filepath):
            pd.DataFrame(columns=cols).to_csv(filepath, index=False)
            print(f"  ✓ Created {filepath}")
            
    print("[✓] All data files initialized!\n")

# ==============================================================================
# PORTFOLIO & FILE MANAGEMENT
# ==============================================================================
def load_portfolio():
    """Loads the portfolio balance from JSON."""
    pf = CONFIG["files"]["portfolio_file"]
    if os.path.exists(pf):
        try:
            with open(pf, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"balance": 1000.0}

def save_portfolio(p):
    """Atomically saves the portfolio balance to JSON."""
    tmp = CONFIG["files"]["portfolio_file"] + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, CONFIG["files"]["portfolio_file"])

portfolio = load_portfolio()

def load_open_trades():
    """Loads the list of currently monitored trades."""
    filepath = CONFIG["files"]["open_trades_file"]
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_open_trades(trades):
    """Atomically saves the list of monitored trades."""
    filepath = CONFIG["files"]["open_trades_file"]
    tmp = filepath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(trades, f, indent=2)
    os.replace(tmp, filepath)

def load_banned_pairs():
    """Loads the user-defined blacklist of pairs."""
    filepath = CONFIG["files"]["banned_pairs_file"]
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_banned_pairs(banned):
    """Atomically saves the blacklist."""
    filepath = CONFIG["files"]["banned_pairs_file"]
    tmp = filepath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(banned, f, indent=2)
    os.replace(tmp, filepath)

def load_regime_state():
    """Loads the persistent regime state to prevent whipsaws."""
    filepath = CONFIG["files"]["regime_state_file"]
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_regime": "NEUTRAL", "confirmed_at": None, "candles_in_regime": 0}

def save_regime_state(state):
    """Atomically saves the regime state."""
    filepath = CONFIG["files"]["regime_state_file"]
    tmp = filepath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, filepath)

def get_weights_file(brain_name):
    """Returns the correct JSON file path for a specific brain."""
    brain_name = brain_name.lower()
    if brain_name in ["bullish", "bull"]:
        return CONFIG["files"]["weights_bull_file"]
    elif brain_name in ["bearish", "bear"]:
        return CONFIG["files"]["weights_bear_file"]
    else:
        return CONFIG["files"]["weights_sideways_file"]

def load_brain_weights(brain_name):
    """Loads the learned weights for a specific brain."""
    filepath = get_weights_file(brain_name)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for key in DEFAULT_WEIGHTS:
                    if key not in data:
                        data[key] = DEFAULT_WEIGHTS[key]
                return data
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()

def save_brain_weights(brain_name, weights):
    """Atomically saves the learned weights for a specific brain."""
    filepath = get_weights_file(brain_name)
    tmp = filepath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(weights, f, indent=2)
    os.replace(tmp, filepath)

def safe_append_csv(filepath, df_new):
    """Atomically appends a DataFrame to a CSV file with backward compatibility."""
    tmp = filepath + ".tmp"
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            existing = pd.read_csv(filepath)
            for col in df_new.columns:
                if col not in existing.columns:
                    existing[col] = None
            updated = pd.concat([existing, df_new], ignore_index=True)
        else:
            updated = df_new
            
        updated.to_csv(tmp, index=False)
        os.replace(tmp, filepath)
    except Exception as e:
        print(f"[!] CSV append failed: {e}")
        header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        df_new.to_csv(filepath, mode='a', header=header, index=False)

# ==============================================================================
# MULTI-CHANNEL DISCORD ROUTER
# ==============================================================================
def send_discord_message(channel, text, image_path=None):
    """Routes messages to the correct Discord channel."""
    webhook_url = WEBHOOKS.get(channel)
    if not webhook_url:
        print(f"[!] Webhook for {channel} not configured. Skipping.")
        return
    
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                resp = requests.post(
                    webhook_url, 
                    data={'content': text[:2000]}, 
                    files={'file': img}, 
                    timeout=15
                )
        else:
            resp = requests.post(
                webhook_url, 
                json={"content": text[:2000]}, 
                timeout=10
            )
        
        if resp.status_code not in [200, 204]:
            print(f"[!] Discord send to {channel} failed: {resp.status_code}")
        else:
            print(f"[✓] Message sent to {channel}")
    except Exception as e:
        print(f"[!] Discord error for {channel}: {e}")

# ==============================================================================
# BANNED PAIRS & TRADE MANAGEMENT
# ==============================================================================
def clean_pair_name(symbol):
    """Converts yfinance symbol to clean display name (EURUSD=X -> EURUSD)."""
    return symbol.replace("=X", "").replace("=F", "")

def ban_pair(symbol):
    """Adds a pair to the permanent blacklist."""
    symbol = symbol.upper()
    if not symbol.endswith("=X") and not symbol.endswith("=F"):
        # Try to guess the format
        if symbol in ["GC", "SI"]:
            symbol = f"{symbol}=F"
        else:
            symbol = f"{symbol}=X"
        
    banned = load_banned_pairs()
    if symbol in banned:
        print(f"[!] {symbol} is already banned.")
        return
        
    banned.append(symbol)
    save_banned_pairs(banned)
    
    display_name = clean_pair_name(symbol)
    print(f"\n{'='*55}\n  🚫 PAIR BANNED: {display_name}\n  Total banned: {len(banned)}\n{'='*55}\n")
    send_discord_message("MAIN", f"🚫 **Pair Banned**\n`{display_name}` added to blacklist.\nTotal banned: {len(banned)}")

def unban_pair(symbol):
    """Removes a pair from the permanent blacklist."""
    symbol = symbol.upper()
    if not symbol.endswith("=X") and not symbol.endswith("=F"):
        if symbol in ["GC", "SI"]:
            symbol = f"{symbol}=F"
        else:
            symbol = f"{symbol}=X"
        
    banned = load_banned_pairs()
    if symbol not in banned:
        print(f"[!] {symbol} is not in the blacklist.")
        return
        
    banned.remove(symbol)
    save_banned_pairs(banned)
    
    display_name = clean_pair_name(symbol)
    print(f"\n✅ {display_name} has been unbanned.")
    send_discord_message("MAIN", f"✅ **Pair Unbanned**\n`{display_name}` removed from blacklist.")

def list_banned_pairs():
    """Displays and broadcasts the current blacklist."""
    banned = load_banned_pairs()
    print(f"\n{'='*55}\n  🚫 BANNED PAIRS ({len(banned)} total)\n{'='*55}")
    
    if not banned:
        print("  (No pairs banned yet)")
        msg = "🚫 **Banned Pairs:** None"
    else:
        lines = [f"🚫 **BANNED PAIRS ({len(banned)} total)**", "━━━━━━━━━━━━━━━━━━━━━━━━"]
        for i, pair in enumerate(banned):
            display_name = clean_pair_name(pair)
            print(f"  {i+1}. {display_name}")
            lines.append(f"  {i+1}. `{display_name}`")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        msg = "\n".join(lines)
        
    print(f"{'='*55}\n")
    send_discord_message("MAIN", msg)

def add_last_signal_as_trade(send_alerts=True):
    """Takes the most recent signal from the log and adds it to monitoring."""
    filepath = CONFIG["files"]["signal_log"]
    if not os.path.exists(filepath):
        print("[!] No signal log found. Run signals first.")
        return False
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print("[!] Signal log is empty.")
            return False
            
        last = df.iloc[-1]
        symbol = last["symbol"]
        direction = last["direction"]
        entry = float(last["entry"])
        stop = float(last["stop"])
        tp1, tp2, tp3 = float(last["tp1"]), float(last["tp2"]), float(last["tp3"])
        qty = float(last["qty"])
        score = float(last["score"])
        mom_z = float(last["mom_z"])
        session_mom = float(last["session_mom"])
        notional = float(last["notional"])
        regime = last.get("regime", "NEUTRAL")
        regime_conf = last.get("regime_confidence", 0)
        
        risk = abs(entry - stop)
        tp4 = entry + 2.0 * risk if direction == "LONG" else entry - 2.0 * risk
        tp5 = entry + 3.0 * risk if direction == "LONG" else entry - 3.0 * risk
        atr = risk / CONFIG["trading"]["atr_stop_multiplier"]
        
        trades = load_open_trades()
        if any(t["symbol"] == symbol for t in trades):
            print(f"[!] {symbol} is already being monitored.")
            return False
            
        trade = {
            "symbol": symbol, "direction": direction, "entry": entry, "stop": stop,
            "tps": [tp1, tp2, tp3, tp4, tp5], "qty": qty, "atr": atr,
            "score": score, "mom_z": mom_z, "session_mom": session_mom, "notional": notional,
            "highest_tp_hit": -1, "current_stop": stop,
            "opened_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "regime": regime, "regime_confidence": regime_conf,
        }
        
        trades.append(trade)
        save_open_trades(trades)
        
        display_name = clean_pair_name(symbol)
        print(f"\n{'='*55}\n  ✅ TRADE ADDED: {direction} {display_name} @ {entry:.5f}\n{'='*55}\n")
        
        if send_alerts:
            alert = (
                f"📝 **Trade Added to Monitor**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{'🟢' if direction == 'LONG' else '🔴'} {direction} {display_name}\n"
                f"📍 Entry: `{entry:.5f}` | 🛑 Stop: `{stop:.5f}`\n"
                f"🧠 Regime: `{regime}` (conf: {regime_conf:.0%})\n"
                f"🎯 TPs: `{tp1:.5f}` → `{tp5:.5f}`"
            )
            send_discord_message("MAIN", alert)
        
        return True
        
    except Exception as e:
        print(f"[!] Error adding trade: {e}")
        return False

def add_manual_trade(symbol, direction, entry, stop):
    """Manually adds a trade with custom parameters."""
    try:
        entry = float(entry)
        stop = float(stop)
    except ValueError:
        print("[!] Entry and stop must be numbers.")
        return
        
    if direction.upper() not in ["LONG", "SHORT"]:
        print("[!] Direction must be LONG or SHORT.")
        return
        
    direction = direction.upper()
    symbol = symbol.upper()
    if not symbol.endswith("=X") and not symbol.endswith("=F"):
        if symbol in ["GC", "SI"]:
            symbol = f"{symbol}=F"
        else:
            symbol = f"{symbol}=X"
        
    dxy_df = get_yfinance_klines(CONFIG["universe"]["regime_indicator"], '1h', days=21)
    regime, conf = get_dxy_regime_robust(dxy_df)
    
    risk = abs(entry - stop)
    tps = [round(entry + m * risk if direction == "LONG" else entry - m * risk, 6) for m in CONFIG["trading"]["tp_multipliers"]]
    atr = risk / CONFIG["trading"]["atr_stop_multiplier"]
    risk_dollars = portfolio["balance"] * CONFIG["trading"]["risk_per_trade_pct"] / 100
    qty = round(risk_dollars / risk, 6)
    notional = round(qty * entry, 2)
    
    trades = load_open_trades()
    if any(t["symbol"] == symbol for t in trades):
        print(f"[!] {symbol} is already being monitored.")
        return
        
    trade = {
        "symbol": symbol, "direction": direction, "entry": entry, "stop": stop,
        "tps": tps, "qty": qty, "atr": atr, "score": 0, "mom_z": 0, "session_mom": 0,
        "notional": notional, "highest_tp_hit": -1, "current_stop": stop,
        "opened_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "regime": regime, "regime_confidence": conf,
    }
    
    trades.append(trade)
    save_open_trades(trades)
    
    display_name = clean_pair_name(symbol)
    alert = (
        f"📝 **Trade Added to Monitor**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{'🟢' if direction == 'LONG' else '🔴'} {direction} {display_name}\n"
        f"📍 Entry: `{entry:.5f}` | 🛑 Stop: `{stop:.5f}`\n"
        f"🧠 Regime: `{regime}` (conf: {conf:.0%})\n"
        f"🎯 TPs: `{tps[0]:.5f}` → `{tps[-1]:.5f}`"
    )
    send_discord_message("MAIN", alert)
    
    print(f"\n✅ Added {direction} {display_name} @ {entry:.5f} | Regime: {regime}")

def generate_list_trades_message(trades, include_brain_info=True):
    """Generates the formatted string for List Trades."""
    if not trades:
        return "📋 **No open trades being monitored.**"
        
    lines = [f"📋 **OPEN TRADES ({len(trades)} active) - 50x Leverage**", "━━━━━━━━━━━━━━━━━━━━━━━━"]
    
    for i, t in enumerate(trades):
        display_name = clean_pair_name(t["symbol"])
        icon = "🟢" if t["direction"] == "LONG" else "🔴"
        tp_status = "None" if t["highest_tp_hit"] == -1 else f"TP{t['highest_tp_hit']+1}"
        risk_icon = "🔴" if t["highest_tp_hit"] == -1 else "🟢"
        
        try:
            current_price = get_current_fx_price(t["symbol"])
            if current_price:
                if t["direction"] == "LONG":
                    pnl_pct = (current_price - t["entry"]) / t["entry"] * 100
                else:
                    pnl_pct = (t["entry"] - current_price) / t["entry"] * 100
                pnl_50x = pnl_pct * CONFIG["leverage"]["display_multiplier"]
                pnl_str = f"**50x PnL: {pnl_50x:+.1f}%**"
            else:
                pnl_str = "N/A"
        except Exception:
            pnl_str = "N/A"
        
        trade_info = (
            f"{icon} **{display_name}** {t['direction']} {risk_icon}\n"
            f"  Entry: `{t['entry']:.5f}`\n"
            f"  TP Hit: {tp_status} | {pnl_str}"
        )
        
        if include_brain_info:
            brain = get_active_brain(t.get("regime", "NEUTRAL"))
            trade_info += f"\n  Brain: `{brain}`"
        
        lines.append(trade_info)
        
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

def list_open_trades():
    """Displays and broadcasts the current list of monitored trades."""
    trades = load_open_trades()
    
    main_msg = generate_list_trades_message(trades, include_brain_info=True)
    print(main_msg)
    send_discord_message("MAIN", main_msg)
    
    updates_msg = generate_list_trades_message(trades, include_brain_info=False)
    send_discord_message("TRADE_UPDATES", updates_msg)

def remove_trade(symbol):
    """Removes a trade from the monitoring list."""
    symbol = symbol.upper()
    if not symbol.endswith("=X") and not symbol.endswith("=F"):
        if symbol in ["GC", "SI"]:
            symbol = f"{symbol}=F"
        else:
            symbol = f"{symbol}=X"
        
    trades = load_open_trades()
    new_trades = [t for t in trades if t["symbol"] != symbol]
    
    if len(new_trades) == len(trades):
        print(f"[!] {symbol} not found.")
        return
        
    save_open_trades(new_trades)
    display_name = clean_pair_name(symbol)
    print(f"✅ Removed {display_name} from monitoring.")
    send_discord_message("MAIN", f"✅ **Trade Removed**: `{display_name}`")
    send_discord_message("TRADE_UPDATES", f"✅ **Trade Removed**: `{display_name}`")

# ==============================================================================
# DATA FETCHING (YFINANCE)
# ==============================================================================
def is_market_open():
    """Checks if FX market is currently open (not weekend)."""
    now = datetime.now(timezone.utc)
    # FX market closed from Friday 22:00 UTC to Sunday 22:00 UTC
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    if weekday == 5 or weekday == 6:  # Saturday or Sunday
        return False
    return True

def get_yfinance_klines(symbol, interval, days=60):
    """Fetches OHLCV data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        # Normalize column names
        df.columns = [col.title() for col in df.columns]
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                if col == 'Volume':
                    df['Volume'] = 1  # FX has no real volume, use placeholder
                else:
                    return pd.DataFrame()
        
        # Remove timezone info for mplfinance compatibility
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        print(f"[!] yfinance error for {symbol}: {e}")
        return pd.DataFrame()

def get_current_fx_price(symbol):
    """Gets current price for a FX pair."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1h")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    
    # Fallback: try fast_info
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price:
            return float(price)
    except Exception:
        pass
    
    return None

def fetch_all_fx_data(pairs):
    """Fetches 1H data for all pairs in parallel."""
    results = {}
    
    def fetch_one(symbol):
        df = get_yfinance_klines(symbol, '1h', days=60)
        return symbol, df
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in pairs}
        
        for future in as_completed(futures):
            try:
                sym, df = future.result()
                if not df.empty and len(df) >= 50:
                    results[sym] = df
                else:
                    print(f"[!] Insufficient data for {sym}")
            except Exception as e:
                print(f"[!] Error fetching {sym}: {e}")
    
    return results

# ==============================================================================
# QUANT FACTORS
# ==============================================================================
def factor_session_momentum(df, lookback=None):
    """
    Calculates session-weighted momentum.
    Higher weight given to London and NY sessions (highest volume).
    Returns normalized score from -1 to +1.
    """
    if lookback is None:
        lookback = CONFIG["session_momentum"]["lookback_hours"]
    
    if len(df) < lookback:
        return 0.0
    
    recent = df.tail(lookback)
    hours = recent.index.hour
    
    # Session importance weights
    session_weights = []
    for h in hours:
        if 13 <= h < 16:  # London+NY overlap (highest volume)
            session_weights.append(3.0)
        elif 8 <= h < 13 or 16 <= h < 21:  # London or NY
            session_weights.append(2.0)
        elif 0 <= h < 8:  # Asian
            session_weights.append(1.0)
        else:  # 21:00-00:00 (dead zone)
            session_weights.append(0.5)
    
    weights = np.array(session_weights)
    
    # Hourly returns
    returns = recent['Close'].pct_change().fillna(0)
    
    # Weighted sum of returns
    weighted_sum = (returns * weights).sum()
    total_weight = weights.sum()
    
    if total_weight == 0:
        return 0.0
    
    # Average weighted return
    avg_weighted_return = weighted_sum / total_weight
    
    # Normalize
    norm_factor = CONFIG["session_momentum"]["normalization_factor"]
    normalized = avg_weighted_return / norm_factor
    
    return float(np.clip(normalized, -1, 1))

def factor_volatility_score(df, lookback=50):
    """Continuous volatility score based on ATR percentile."""
    if len(df) < lookback:
        return 0.0
        
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean().dropna()
    if len(atr) < 20:
        return 0.0
        
    current_atr = atr.iloc[-1]
    percentile = (atr < current_atr).sum() / len(atr) * 100
    
    distance_from_center = abs(percentile - 50) / 50
    return float(1.0 - 2.0 * distance_from_center)

def factor_cross_sectional_momentum(all_returns, target_sym):
    """Calculates Z-score for an asset's return relative to universe average."""
    if target_sym not in all_returns or len(all_returns) < 10:
        return 0.0
        
    values = list(all_returns.values())
    mean_ret = np.mean(values)
    std_ret = np.std(values)
    
    if std_ret < 1e-9:
        return 0.0
        
    return float((all_returns[target_sym] - mean_ret) / std_ret)

def calculate_atr(df, period=14):
    """Calculates Average True Range."""
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr_val = tr.rolling(period).mean().iloc[-1]
    return float(atr_val) if not pd.isna(atr_val) else None

def calculate_adx(df, period=14):
    """Calculates Average Directional Index for trend strength."""
    if len(df) < period * 2:
        return 0.0
        
    h, l, c = df['High'], df['Low'], df['Close']
    
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0

# ==============================================================================
# DXY-BASED REGIME DETECTION
# ==============================================================================
def detect_raw_regime(dxy_df):
    """Multi-signal regime detection using DXY."""
    if dxy_df.empty or len(dxy_df) < 60:
        return "NEUTRAL", 0.0
        
    cfg = CONFIG["regime"]
    closes = dxy_df['Close']
    
    ema = closes.ewm(span=cfg["ema_period"], adjust=False).mean()
    current_price = closes.iloc[-1]
    ema_val = ema.iloc[-1]
    distance_pct = (current_price - ema_val) / ema_val
    
    ema_slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5] if len(ema) >= 5 else 0
    
    adx = calculate_adx(dxy_df, cfg["adx_period"])
    
    bull_signals = 0
    bear_signals = 0
    total_signals = 0
    
    if distance_pct > cfg["price_ema_threshold_pct"]:
        bull_signals += 1
        total_signals += 1
    elif distance_pct < -cfg["price_ema_threshold_pct"]:
        bear_signals += 1
        total_signals += 1
        
    if ema_slope > 0.0005:  # DXY moves less than BTC
        bull_signals += 1
        total_signals += 1
    elif ema_slope < -0.0005:
        bear_signals += 1
        total_signals += 1
        
    if adx > cfg["adx_trend_threshold"]:
        if distance_pct > 0:
            bull_signals += 1
        else:
            bear_signals += 1
        total_signals += 1
        
    if total_signals == 0:
        return "NEUTRAL", 0.0
        
    bull_ratio = bull_signals / total_signals
    bear_ratio = bear_signals / total_signals
    
    if bull_ratio >= 0.67:
        raw_regime = "BULLISH"
        confidence = bull_ratio
    elif bear_ratio >= 0.67:
        raw_regime = "BEARISH"
        confidence = bear_ratio
    else:
        raw_regime = "NEUTRAL"
        confidence = 1.0 - max(bull_ratio, bear_ratio)
        
    if adx > 30:
        confidence = min(1.0, confidence * 1.2)
        
    return raw_regime, float(confidence)

def get_dxy_regime_robust(dxy_df):
    """Regime detection with confirmation and hysteresis."""
    raw_regime, raw_confidence = detect_raw_regime(dxy_df)
    
    state = load_regime_state()
    current_regime = state.get("current_regime", "NEUTRAL")
    candles_in_regime = state.get("candles_in_regime", 0)
    
    confirmation_needed = CONFIG["regime"]["confirmation_candles"]
    hysteresis = CONFIG["regime"]["hysteresis_multiplier"]
    
    if raw_regime == current_regime:
        candles_in_regime += 1
        final_regime = current_regime
        final_confidence = min(1.0, raw_confidence * (1 + 0.1 * min(candles_in_regime, 5)))
    else:
        if raw_regime != state.get("last_raw_regime"):
            candles_in_regime = 1
            final_regime = current_regime
            final_confidence = raw_confidence
        else:
            candles_in_regime += 1
            
            required_candles = confirmation_needed
            if candles_in_regime > 5:
                required_candles = int(confirmation_needed * hysteresis)
                
            if candles_in_regime >= required_candles and raw_confidence > 0.6:
                final_regime = raw_regime
                final_confidence = raw_confidence
                candles_in_regime = 1
                print(f"[*] Regime flip confirmed: {current_regime} → {raw_regime}")
            else:
                final_regime = current_regime
                final_confidence = raw_confidence * 0.7
                
    new_state = {
        "current_regime": final_regime,
        "confirmed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if final_regime != current_regime else state.get("confirmed_at"),
        "candles_in_regime": candles_in_regime,
        "last_raw_regime": raw_regime,
        "raw_confidence": raw_confidence,
    }
    save_regime_state(new_state)
    
    return final_regime, final_confidence

def get_active_brain(regime):
    """Maps the regime to the correct brain name."""
    regime = regime.upper()
    if regime == "BULLISH":
        return "bull"
    elif regime == "BEARISH":
        return "bear"
    else:
        return "sideways"

# ==============================================================================
# 3-BRAIN LEARNING
# ==============================================================================
def adapt_brain_weights(brain_name):
    """Out-of-sample learning with train/validation split."""
    filepath = CONFIG["files"]["trade_results_file"]
    if not os.path.exists(filepath):
        return None
        
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None
        
    if df.empty:
        return None
        
    regime_map = {"bull": "BULLISH", "bear": "BEARISH", "sideways": "NEUTRAL"}
    target_regime = regime_map.get(brain_name, "NEUTRAL")
    
    if "regime" not in df.columns:
        print(f"[*] {brain_name} brain: No regime data yet.")
        return None
        
    brain_trades = df[df["regime"] == target_regime].copy()
    total_in_brain = len(brain_trades)
    min_trades = CONFIG["learning"]["min_trades_to_learn"]
    
    if total_in_brain < min_trades:
        print(f"[*] {brain_name} brain: Need {min_trades}, have {total_in_brain}. Waiting...")
        return None
        
    window = min(CONFIG["learning"]["rolling_window"], total_in_brain)
    recent = brain_trades.tail(window).copy()
    recent = recent[recent["pnl_dollars"] != 0]
    
    if len(recent) < 8:
        print(f"[*] {brain_name} brain: Not enough non-breakeven trades ({len(recent)}).")
        return None
        
    split_ratio = 1.0 - CONFIG["learning"]["validation_split"]
    split_idx = int(len(recent) * split_ratio)
    train_data = recent.iloc[:split_idx]
    val_data = recent.iloc[split_idx:]
    
    if len(train_data) < 5 or len(val_data) < 2:
        print(f"[*] {brain_name} brain: Split too small.")
        return None
        
    train_data = train_data.copy()
    train_data["effective_session_mom"] = train_data["session_mom"] * np.sign(train_data["mom_z"])
    
    try:
        mom_corr = train_data["mom_z"].corr(train_data["pnl_dollars"])
        sess_corr = train_data["effective_session_mom"].corr(train_data["pnl_dollars"])
        vol_corr = train_data["vol_regime_score"].corr(train_data["pnl_dollars"]) if "vol_regime_score" in train_data.columns else 0
    except Exception:
        print(f"[!] {brain_name} brain: Correlation failed.")
        return None
        
    if pd.isna(mom_corr): mom_corr = 0
    if pd.isna(sess_corr): sess_corr = 0
    if pd.isna(vol_corr): vol_corr = 0
    
    raw_mom = max(0.01, mom_corr + 1)
    raw_sess = max(0.01, sess_corr + 1)
    raw_vol = max(0.01, vol_corr + 1)
    
    total_raw = raw_mom + raw_sess + raw_vol
    new_mom = raw_mom / total_raw
    new_sess = raw_sess / total_raw
    new_vol = raw_vol / total_raw
    
    min_w = CONFIG["learning"]["min_weight"]
    max_w = CONFIG["learning"]["max_weight"]
    new_mom = max(min_w, min(max_w, new_mom))
    new_sess = max(min_w, min(max_w, new_sess))
    new_vol = max(min_w, min(max_w, new_vol))
    
    total_clamped = new_mom + new_sess + new_vol
    new_mom /= total_clamped
    new_sess /= total_clamped
    new_vol /= total_clamped
    
    current = load_brain_weights(brain_name)
    val_score_new = score_weights_on_data(new_mom, new_sess, new_vol, val_data)
    val_score_old = score_weights_on_data(
        current["momentum"], current["session_momentum"], current["volatility"], val_data
    )
    
    if val_score_new < val_score_old * 0.9 and current.get("based_on_trades", 0) >= 20:
        print(f"[*] {brain_name} brain: New weights worse on validation. Keeping old.")
        return None
        
    lr = CONFIG["learning"]["learning_rate"]
    final_mom = lr * new_mom + (1 - lr) * current["momentum"]
    final_sess = lr * new_sess + (1 - lr) * current["session_momentum"]
    final_vol = lr * new_vol + (1 - lr) * current["volatility"]
    
    total_final = final_mom + final_sess + final_vol
    final_mom /= total_final
    final_sess /= total_final
    final_vol /= total_final
    
    new_penalty = adapt_regime_penalty(brain_name, train_data, current.get("regime_penalty", -0.30))
    
    new_weights = {
        "momentum": round(final_mom, 3),
        "session_momentum": round(final_sess, 3),
        "volatility": round(final_vol, 3),
        "regime_penalty": round(new_penalty, 3),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "based_on_trades": total_in_brain,
        "validation_score": round(val_score_new, 4),
        "correlations": {
            "momentum": round(mom_corr, 3),
            "session_momentum": round(sess_corr, 3),
            "volatility": round(vol_corr, 3)
        },
        "window_size": window,
    }
    
    save_brain_weights(brain_name, new_weights)
    
    print(f"\n{'='*60}\n  🧠 {brain_name.upper()} BRAIN LEARNED (out-of-sample)\n{'='*60}")
    
    msg = (
        f"🧠 **{brain_name.upper()} BRAIN LEARNED** (out-of-sample)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Train: {len(train_data)} | Validate: {len(val_data)}\n"
        f"Validation: `{val_score_new:.3f}` (was `{val_score_old:.3f}`)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Mom: **{final_mom*100:.1f}%** | Session: **{final_sess*100:.1f}%** | Vol: **{final_vol*100:.1f}%**\n"
        f"Penalty: **{new_penalty:+.3f}**"
    )
    send_discord_message("MAIN", msg)
    
    return new_weights

def score_weights_on_data(w_mom, w_sess, w_vol, data_df):
    """Scores a set of weights on data using correlation."""
    if data_df.empty:
        return 0.0
        
    data_df = data_df.copy()
    data_df["effective_session_mom"] = data_df["session_mom"] * np.sign(data_df["mom_z"])
    
    data_df["reconstructed_score"] = (
        w_mom * data_df["mom_z"] + 
        w_sess * data_df["effective_session_mom"] + 
        w_vol * (data_df.get("vol_regime_score", pd.Series([0]*len(data_df))) if "vol_regime_score" in data_df.columns else 0)
    )
    
    try:
        score = data_df["reconstructed_score"].corr(data_df["pnl_dollars"])
        return float(score) if not pd.isna(score) else 0.0
    except Exception:
        return 0.0

def adapt_regime_penalty(brain_name, trades_df, current_penalty):
    """Adapts the regime penalty based on counter-trend performance."""
    if brain_name == "bull":
        counter_trend = trades_df[trades_df["direction"] == "SHORT"]
    elif brain_name == "bear":
        counter_trend = trades_df[trades_df["direction"] == "LONG"]
    else:
        counter_trend = trades_df
        
    if len(counter_trend) < 3:
        return current_penalty
        
    avg_pnl = counter_trend["pnl_dollars"].mean()
    step = CONFIG["learning"]["penalty_step"]
    min_p = CONFIG["learning"]["min_penalty"]
    max_p = CONFIG["learning"]["max_penalty"]
    
    if brain_name == "sideways":
        if avg_pnl > 0:
            new_penalty = min(max_p, current_penalty + step)
        elif avg_pnl < 0:
            new_penalty = max(-0.20, current_penalty - step)
        else:
            new_penalty = current_penalty
    else:
        if avg_pnl < 0:
            new_penalty = max(min_p, current_penalty - step)
        elif avg_pnl > 0:
            new_penalty = min(max_p, current_penalty + step)
        else:
            new_penalty = current_penalty
            
    return new_penalty

def show_learning():
    """Displays and broadcasts the current learned weights for all 3 brains."""
    brains = ["bull", "bear", "sideways"]
    brain_names = {"bull": "🟢 BULL", "bear": "🔴 BEAR", "sideways": "⚪ SIDEWAYS"}
    
    print(f"\n{'='*70}\n  🧠 ALL 3 BRAINS — LEARNED WEIGHTS\n{'='*70}")
    discord_lines = ["🧠 **All 3 Brains — Learned Weights**", "━━━━━━━━━━━━━━━━━━━━━━━━"]
    
    for brain in brains:
        weights = load_brain_weights(brain)
        print(f"\n  {brain_names[brain]} BRAIN:\n    Momentum: {weights['momentum']*100:.1f}% | Session: {weights['session_momentum']*100:.1f}% | Vol: {weights['volatility']*100:.1f}%\n    Penalty: {weights.get('regime_penalty', -0.30):+.3f} | Trades: {weights.get('based_on_trades', 0)} | Val: {weights.get('validation_score', 0):.3f}")
        
        discord_lines.append(
            f"\n**{brain_names[brain]}**\n"
            f"  Mom: {weights['momentum']*100:.1f}% | Session: {weights['session_momentum']*100:.1f}% | Vol: {weights['volatility']*100:.1f}%\n"
            f"  Penalty: `{weights.get('regime_penalty', -0.30):+.3f}` | Val: `{weights.get('validation_score', 0):.3f}` | Trades: {weights.get('based_on_trades', 0)}"
        )
        
    state = load_regime_state()
    print(f"\n  📊 REGIME STATE:\n    Current: {state.get('current_regime', 'NEUTRAL')} | Candles: {state.get('candles_in_regime', 0)}")
    print(f"{'='*70}\n")
    
    send_discord_message("MAIN", "\n".join(discord_lines))

def reset_learning(brain=None):
    """Resets learned weights to defaults."""
    if brain is None:
        for b in ["bull", "bear", "sideways"]:
            save_brain_weights(b, DEFAULT_WEIGHTS.copy())
        print("\n✅ All 3 brains reset.")
        send_discord_message("MAIN", "🔄 **All Brains Reset**")
    else:
        save_brain_weights(brain, DEFAULT_WEIGHTS.copy())
        print(f"\n✅ {brain.upper()} brain reset.")
        send_discord_message("MAIN", f"🔄 **{brain.upper()} Brain Reset**")

# ==============================================================================
# SIGNAL GENERATION & ROUTING
# ==============================================================================
def generate_signals():
    """Generates trading signals using the active brain's weights."""
    # Check if market is open
    if not is_market_open():
        print("[*] FX market closed (weekend). Skipping signal generation.")
        return [], [], "NEUTRAL", 0.0
    
    dxy_df = get_yfinance_klines(CONFIG["universe"]["regime_indicator"], '1h', days=60)
    regime, regime_confidence = get_dxy_regime_robust(dxy_df)
    active_brain = get_active_brain(regime)
    
    weights = load_brain_weights(active_brain)
    w_mom = weights["momentum"]
    w_sess = weights["session_momentum"]
    w_vol = weights["volatility"]
    regime_penalty = weights.get("regime_penalty", -0.30)
    
    bt = weights.get("based_on_trades", 0)
    confidence = "HIGH" if bt >= 60 else "MEDIUM" if bt >= 30 else "LOW" if bt >= 10 else "DEFAULT"
    
    print(f"[*] Regime: {regime} (conf: {regime_confidence:.0%}) | Brain: {active_brain.upper()} ({confidence})")
    
    trades = load_open_trades()
    
    open_symbols = {t["symbol"] for t in trades}
    risky_symbols = {t["symbol"] for t in trades if t.get("highest_tp_hit", -1) == -1}
    
    if len(risky_symbols) >= CONFIG["trading"]["max_concurrent_risky_trades"]:
        print(f"\n[*] Max risky trades open. Skipping signals.")
        return [], [], regime, regime_confidence
    
    # Get universe
    pairs = CONFIG["universe"]["pairs"]
    banned = load_banned_pairs()
    pairs = [p for p in pairs if p not in banned]
    
    print(f"\n[*] Scanning {len(pairs)} pairs...")
    
    if open_symbols:
        display_names = [clean_pair_name(s) for s in open_symbols]
        print(f"[*] Blocking signals for {len(open_symbols)} pairs with open trades: {', '.join(display_names)}")
    
    all_data = fetch_all_fx_data(pairs)
    print(f"[*] Got valid data for {len(all_data)} pairs")
    
    if len(all_data) < 10:
        print("[!] Not enough data.")
        return [], [], regime, regime_confidence
        
    all_returns = {}
    for sym, df in all_data.items():
        if len(df) >= 24:
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-24]) - 1
            all_returns[sym] = ret
    
    scored = []
    for sym, df in all_data.items():
        if sym in open_symbols:
            continue
            
        price = float(df['Close'].iloc[-1])
        session_mom = factor_session_momentum(df)
        vol = factor_volatility_score(df)
        mom_z = factor_cross_sectional_momentum(all_returns, sym)
        
        direction = "LONG" if mom_z > 0 else "SHORT"
        
        p_regime = 0.0
        if regime == "BULLISH" and direction == "SHORT":
            p_regime = regime_penalty * regime_confidence
        if regime == "BEARISH" and direction == "LONG":
            p_regime = regime_penalty * regime_confidence
            
        score = (w_mom * mom_z) + (w_sess * session_mom * np.sign(mom_z)) + (w_vol * vol) + p_regime
        
        atr_val = calculate_atr(df) or price * 0.005
        
        scored.append({
            "symbol": sym, "price": price, "score": round(score, 3),
            "direction": direction, "atr": atr_val,
            "mom_z": round(mom_z, 2), "session_mom": round(session_mom, 2),
            "vol_regime_score": round(vol, 2), "regime": regime,
            "regime_confidence": regime_confidence
        })
        
    scored.sort(key=lambda x: abs(x["score"]), reverse=True)
    
    signals = []
    for s in scored:
        if abs(s["score"]) >= CONFIG["trading"]["min_score_to_enter"]:
            signals.append(build_signal(s))
        if len(signals) >= CONFIG["trading"]["max_signals"]:
            break
            
    return signals, scored[:10], regime, regime_confidence

def build_signal(s):
    """Builds a complete signal object with position sizing."""
    direction = s["direction"]
    price = s["price"]
    atr_val = s["atr"]
    
    entry = price * (0.9995 if direction == "LONG" else 1.0005)
    stop_dist = max(CONFIG["trading"]["atr_stop_multiplier"] * atr_val, entry * 0.005)
    stop = entry - stop_dist if direction == "LONG" else entry + stop_dist
    risk = abs(entry - stop)
    
    tps = [round(entry + m * risk if direction == "LONG" else entry - m * risk, 6) for m in CONFIG["trading"]["tp_multipliers"]]
    
    risk_dollars = portfolio["balance"] * CONFIG["trading"]["risk_per_trade_pct"] / 100
    qty = risk_dollars / risk
    notional = qty * entry
    
    abs_score = abs(s["score"])
    strength = "🟢 STRONG" if abs_score >= 2.5 else "🟡 GOOD" if abs_score >= 2.0 else "🟠 MODERATE" if abs_score >= 1.5 else "⚪ WEAK"
    
    return {
        "symbol": s["symbol"], "direction": direction, "strength": strength,
        "entry": round(entry, 6), "stop": round(stop, 6), "tps": tps,
        "qty": round(qty, 6), "notional": round(notional, 2),
        "risk_dollars": round(risk_dollars, 2), "risk_pct": round(risk / entry * 100, 3),
        "score": s["score"], "mom_z": s["mom_z"], "session_mom": s["session_mom"],
        "vol_regime_score": s["vol_regime_score"], "regime": s["regime"],
        "regime_confidence": s["regime_confidence"], "atr": atr_val
    }

def format_discord_alert(sig, include_technicals=True):
    """Formats the Discord alert message for a signal."""
    regime = sig.get("regime", "NEUTRAL")
    regime_conf = sig.get("regime_confidence", 0)
    brain = get_active_brain(regime)
    weights = load_brain_weights(brain)
    
    bt = weights.get("based_on_trades", 0)
    confidence = "HIGH" if bt >= 60 else "MEDIUM" if bt >= 30 else "LOW" if bt >= 10 else "DEFAULT"
    
    display_name = clean_pair_name(sig['symbol'])
    icon = "🟢" if sig['direction'] == "LONG" else "🔴"
    
    tp_lines = "".join([
        f"  TP{i+1}: `{tp:.5f}` ({CONFIG['trading']['tp_multipliers'][i]}R → close {CONFIG['trading']['fractions'][i]*100:.0f}%)\n" 
        for i, tp in enumerate(sig['tps'])
    ])
    
    if include_technicals:
        weights_info = (
            f"{brain.upper()} ({confidence}) | "
            f"Mom:{weights['momentum']*100:.0f}% Session:{weights['session_momentum']*100:.0f}% "
            f"Vol:{weights['volatility']*100:.0f}% Pen:{weights.get('regime_penalty',-0.3):+.2f}"
        )
        
        return (
            f"{icon} **{sig['direction']} {display_name}** {sig['strength']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Score: `{sig['score']}` | MomZ: `{sig['mom_z']}` | Session: `{sig['session_mom']}`\n"
            f"🌊 DXY Regime: `{regime}` (conf: {regime_conf:.0%})\n"
            f"🧠 `{weights_info}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry: `{sig['entry']:.5f}`\n"
            f"🛑 Stop: `{sig['stop']:.5f}` (-{sig['risk_pct']:.3f}%)\n"
            f"{tp_lines}"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ **Take with 50x leverage or lower**\n"
            f"💡 Auto-adding to monitoring..."
        )
    else:
        return (
            f"{icon} **{sig['direction']} {display_name}** {sig['strength']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry: `{sig['entry']:.5f}`\n"
            f"🛑 Stop: `{sig['stop']:.5f}` (-{sig['risk_pct']:.3f}%)\n"
            f"{tp_lines}"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ **Take with 50x leverage or lower**"
        )

def run_signals():
    """Generates signals and routes them to the correct channels."""
    acquire_lock()
    try:
        result = generate_signals()
        if not result or len(result) < 4:
            print("[!] Signal generation failed.")
            return
            
        signals, top_candidates, regime, regime_conf = result
        print_console_report(signals, top_candidates, regime, regime_conf)
        
        for sig in signals:
            log_signal(sig)
            chart_path = generate_chart(sig)
            strength = sig['strength']
            
            if "STRONG" in strength:
                alert_text_full = format_discord_alert(sig, include_technicals=True)
                send_discord_message("MAIN", alert_text_full, chart_path)
                alert_text_clean = format_discord_alert(sig, include_technicals=False)
                send_discord_message("PUBLIC", alert_text_clean, chart_path)
            else:
                alert_text_full = format_discord_alert(sig, include_technicals=True)
                send_discord_message("MAIN", alert_text_full, chart_path)
                alert_text_clean = format_discord_alert(sig, include_technicals=False)
                send_discord_message("VIP", alert_text_clean, chart_path)
                
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                except Exception:
                    pass
        
        if CONFIG["automation"]["auto_add_signals"] and signals:
            print("[*] Auto-adding last signal to monitoring...")
            add_last_signal_as_trade(send_alerts=True)
            
    finally:
        release_lock()

# ==============================================================================
# TRADE MONITORING & ROUTING
# ==============================================================================
def safe_calculate_r_multiple(pnl_pct, entry, stop):
    """Safely calculates R-multiple, handling entry == stop."""
    risk_pct = abs(entry - stop) / entry * 100 if entry != 0 else 0
    if risk_pct == 0:
        return 0.0
    return pnl_pct / risk_pct

def monitor_open_trades():
    """Monitors open trades and routes alerts."""
    trades = load_open_trades()
    if not trades:
        msg = "📋 **No open trades being monitored.**"
        send_discord_message("TRADE_UPDATES", msg)
        return

    print(f"\n[*] Monitoring {len(trades)} open trades...")
    
    tp_sl_alerts = []
    closed_trades = []
    update_occurred = False

    for trade in trades:
        sym = trade["symbol"]
        display_name = clean_pair_name(sym)
        direction = trade["direction"]
        entry = trade["entry"]
        current_stop = trade["current_stop"]
        tps = trade["tps"]
        highest_tp_hit = trade["highest_tp_hit"]
        atr = trade["atr"]
        qty = trade["qty"]
        regime = trade.get("regime", "NEUTRAL")
        
        current_price = get_current_fx_price(sym)
        if current_price is None:
            print(f"[!] Could not fetch price for {display_name}")
            continue

        for i in range(highest_tp_hit + 1, len(tps)):
            tp = tps[i]
            tp_hit = (direction == "LONG" and current_price >= tp) or (direction == "SHORT" and current_price <= tp)
            
            if tp_hit:
                trade["highest_tp_hit"] = i
                update_occurred = True
                
                if i == 0:
                    trade["current_stop"] = entry
                    stop_msg = "🔒 Move stop to BREAKEVEN"
                elif i == 1:
                    trade["current_stop"] = tps[0]
                    stop_msg = "🔒 Move stop to TP1"
                elif i >= 2:
                    trail = current_price - (CONFIG["trading"]["trailing_atr_multiplier"] * atr) if direction == "LONG" else current_price + (CONFIG["trading"]["trailing_atr_multiplier"] * atr)
                    trade["current_stop"] = max(trail, tps[i-1]) if direction == "LONG" else min(trail, tps[i-1])
                    stop_msg = f"📈 Trail stop to `{trade['current_stop']:.5f}`"
                    
                pnl_pct = (current_price - entry) / entry * 100 if direction == "LONG" else (entry - current_price) / entry * 100
                pnl_50x = pnl_pct * CONFIG["leverage"]["display_multiplier"]
                
                alert = (
                    f"🎯 **{display_name} {direction}** - TP{i+1} HIT! (50x Leverage)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 Current: `{current_price:.5f}`\n"
                    f"📊 **50x PnL: {pnl_50x:+.1f}%**\n"
                    f"{stop_msg}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📋 Close {CONFIG['trading']['fractions'][i]*100:.0f}% of position | {len(tps)-i-1} TPs left"
                )
                tp_sl_alerts.append(alert)
                print(f"[✓] {display_name} hit TP{i+1}")
                break

        stop_hit = (direction == "LONG" and current_price <= current_stop) or (direction == "SHORT" and current_price >= current_stop)
        
        if stop_hit:
            update_occurred = True
            exit_price = current_stop
            pnl_pct = (exit_price - entry) / entry * 100 if direction == "LONG" else (entry - exit_price) / entry * 100
            pnl_50x = pnl_pct * CONFIG["leverage"]["display_multiplier"]
            hit_level = "STOP" if highest_tp_hit == -1 else f"STOP after TP{highest_tp_hit+1}"
            
            r_multiple = safe_calculate_r_multiple(pnl_pct, entry, trade["stop"])
            
            closed_trades.append({
                "open_time": trade["opened_at"],
                "close_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "symbol": sym, "direction": direction, "entry": entry, "stop": trade["stop"],
                "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "tp4": tps[3], "tp5": tps[4],
                "exit_price": exit_price, "qty": qty, "pnl_pct": round(pnl_pct, 2),
                "pnl_dollars": trade["notional"] * (pnl_pct / 100), "r_multiple": round(r_multiple, 2),
                "hit_level": hit_level, "score": trade.get("score", 0),
                "mom_z": trade.get("mom_z", 0), "session_mom": trade.get("session_mom", 0),
                "vol_regime_score": trade.get("vol_regime_score", 0),
                "regime": regime, "regime_confidence": trade.get("regime_confidence", 0)
            })
            
            alert = (
                f"🛑 **{display_name} {direction}** - {hit_level} (50x Leverage)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Exit: `{exit_price:.5f}`\n"
                f"📊 **50x PnL: {pnl_50x:+.1f}%**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ Trade closed. Use isolated margin next time."
            )
            tp_sl_alerts.append(alert)
            print(f"[✗] {display_name} stopped out")

    list_msg_main = generate_list_trades_message(trades, include_brain_info=True)
    list_msg_updates = generate_list_trades_message(trades, include_brain_info=False)
    
    if update_occurred:
        for alert in tp_sl_alerts:
            send_discord_message("MAIN", alert)
            send_discord_message("TRADE_UPDATES", alert)
            send_discord_message("WINS_LOSSES", alert)
            
        send_discord_message("MAIN", list_msg_main)
        send_discord_message("TRADE_UPDATES", list_msg_updates)
    else:
        send_discord_message("TRADE_UPDATES", list_msg_updates)

    if closed_trades:
        safe_append_csv(CONFIG["files"]["trade_results_file"], pd.DataFrame(closed_trades))
        closed_symbols = {t["symbol"] for t in closed_trades}
        trades = [t for t in trades if t["symbol"] not in closed_symbols]
        save_open_trades(trades)
        check_performance_report()
    else:
        save_open_trades(trades)

def run_monitor():
    """Acquires the file lock and runs the trade monitor."""
    acquire_lock()
    try:
        monitor_open_trades()
    finally:
        release_lock()

def check_performance_report():
    """Generates a performance report every 10 trades and triggers learning."""
    filepath = CONFIG["files"]["trade_results_file"]
    if not os.path.exists(filepath):
        return
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return
            
        total_trades = len(df)
        counter_file = CONFIG["files"]["perf_counter_file"]
        last_reported = 0
        
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                try:
                    last_reported = int(f.read().strip())
                except Exception:
                    pass
                    
        milestone = (total_trades // CONFIG["report_every_n_trades"]) * CONFIG["report_every_n_trades"]
        if milestone <= last_reported or milestone == 0:
            return
            
        wins = df[df["pnl_dollars"] > 0]
        losses = df[df["pnl_dollars"] < 0]
        winrate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_r = df["r_multiple"].mean()
        profit_factor = wins["pnl_dollars"].sum() / abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else float('inf')
        
        tp1_hits = len(df[df["hit_level"].str.contains("TP1", na=False)])
        tp2_hits = len(df[df["hit_level"].str.contains("TP2|TP3|TP4|TP5", na=False)])
        
        report = (
            f"📊 **Performance Report** – {total_trades} Trades (50x Leverage)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Win Rate: `{winrate:.1f}%` ({len(wins)}W / {len(losses)}L)\n"
            f"📊 Profit Factor: `{profit_factor:.2f}`\n"
            f"🎯 Avg R: `{avg_r:+.2f}R`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 TP1: `{tp1_hits}/{total_trades}` ({tp1_hits/total_trades*100:.1f}%)\n"
            f"🎯 TP2+: `{tp2_hits}/{total_trades}` ({tp2_hits/total_trades*100:.1f}%)"
        )
        
        send_discord_message("MAIN", report)
        send_discord_message("WINS_LOSSES", report)
        print(f"\n{report}")
        
        with open(counter_file, 'w') as f:
            f.write(str(milestone))
            
        dxy_df = get_yfinance_klines(CONFIG["universe"]["regime_indicator"], '1h', days=60)
        regime, _ = get_dxy_regime_robust(dxy_df)
        active_brain = get_active_brain(regime)
        
        print(f"\n[*] Triggering learning for {active_brain.upper()} brain...")
        adapt_brain_weights(active_brain)
        
    except Exception as e:
        print(f"[!] Report failed: {e}")

def show_performance_report():
    """Displays the performance report to the console."""
    filepath = CONFIG["files"]["trade_results_file"]
    if not os.path.exists(filepath):
        print("[!] No trade results yet.")
        return
        
    df = pd.read_csv(filepath)
    if df.empty:
        print("[!] No trades closed yet.")
        return
        
    total_trades = len(df)
    wins = df[df["pnl_dollars"] > 0]
    losses = df[df["pnl_dollars"] < 0]
    winrate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_r = df["r_multiple"].mean()
    profit_factor = wins["pnl_dollars"].sum() / abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else float('inf')
    
    print(f"\n{'='*55}\n  PERFORMANCE REPORT – {total_trades} Trades (50x Leverage)\n{'='*55}\n  Win Rate: {winrate:.1f}%\n  Profit Factor: {profit_factor:.2f}\n  Avg R: {avg_r:+.2f}R\n{'='*55}\n")

# ==============================================================================
# CHARTING & CONSOLE
# ==============================================================================
def generate_chart(sig):
    """Generates a candlestick chart with entry/stop/TP overlays."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf
    except ImportError as e:
        print(f"[!] Chart libraries not installed: {e}")
        return None
        
    sym = sig["symbol"]
    display_name = clean_pair_name(sym)
    df = get_yfinance_klines(sym, '1h', days=30)
    
    if df.empty or len(df) < 20:
        print(f"[!] Not enough data for chart")
        return None
    
    # Handle volume (FX has no real volume)
    if df['Volume'].sum() == 0:
        df['Volume'] = 1
        
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds', facecolor='#0d1117',
        gridcolor='#1c2333',
        rc={'axes.labelcolor': '#c9d1d9', 'xtick.color': '#8b949e',
            'ytick.color': '#8b949e', 'axes.titlecolor': '#f0f6fc'}
    )
    
    ema50 = df['Close'].ewm(span=min(50, len(df)), adjust=False).mean()
    ema20 = df['Close'].ewm(span=min(20, len(df)), adjust=False).mean()
    
    addplots = [
        mpf.make_addplot(ema50, color='#f39c12', width=1.2, label='EMA50'),
        mpf.make_addplot(ema20, color='#3498db', width=1.0, label='EMA20')
    ]
    
    title = f"{display_name} 1H | {sig['direction']} | Score: {sig['score']}"
    
    fig, axes = mpf.plot(
        df, type='candle', style=style, volume=True,
        title=title, ylabel='Price', ylabel_lower='Vol',
        addplot=addplots, returnfig=True, figsize=(10, 7)
    )
    
    ax = axes[0]
    
    ax.axhline(y=sig['entry'], color='#f1c40f', linestyle='-', linewidth=2, label=f"Entry: {sig['entry']:.5f}")
    ax.axhline(y=sig['stop'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Stop: {sig['stop']:.5f}")
    
    colors = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#0e8c72']
    for i, tp in enumerate(sig['tps']):
        frac = CONFIG['trading']['fractions'][i] * 100
        label = f"TP{i+1}: {tp:.5f} ({frac}%)" if i < 2 else None
        ax.axhline(y=tp, color=colors[i], linestyle='--', linewidth=1, alpha=0.8, label=label)
        
    ax.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)
    
    path = f"chart_{display_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    
    print(f"[✓] Chart generated: {path}")
    return path

def log_signal(sig):
    """Logs a signal to the CSV file."""
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbol": sig["symbol"], "direction": sig["direction"],
        "entry": sig["entry"], "stop": sig["stop"],
        "tp1": sig["tps"][0], "tp2": sig["tps"][1], "tp3": sig["tps"][2],
        "score": sig["score"], "mom_z": sig["mom_z"], "session_mom": sig["session_mom"],
        "qty": sig["qty"], "notional": sig["notional"],
        "regime": sig.get("regime", "NEUTRAL"),
        "regime_confidence": sig.get("regime_confidence", 0)
    }
    safe_append_csv(CONFIG["files"]["signal_log"], pd.DataFrame([row]))

def print_console_report(signals, top_candidates, regime, regime_conf):
    """Prints the signal report to the console."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    weights = load_brain_weights(get_active_brain(regime))
    
    print(f"\n{'='*60}\n  ALPHASWING FX v1.0 – SIGNAL REPORT\n  {now}\n  DXY Regime: {regime} (conf: {regime_conf:.0%}) | Balance: ${portfolio['balance']:.2f}")
    print(f"  🧠 Weights: Mom={weights['momentum']:.2f} | Session={weights['session_momentum']:.2f} | Vol={weights['volatility']:.2f}\n  🧠 Penalty: {weights.get('regime_penalty', -0.30):+.2f} | Trades: {weights.get('based_on_trades', 0)}\n{'='*60}")
    
    print(f"\n  TOP 10 CANDIDATES:\n  {'#':<3} {'Pair':<12} {'Dir':<6} {'Score':<7} {'MomZ':<7} {'Sess':<7} {'Vol':<5}")
    print("  " + "-" * 50)
    
    for i, c in enumerate(top_candidates):
        display_name = clean_pair_name(c['symbol'])
        print(f"  {i+1:<3} {display_name:<12} {c['direction']:<6} {c['score']:<7} {c['mom_z']:<7} {c['session_mom']:<7} {c['vol_regime_score']:<5.2f}")
        
    if not signals:
        print(f"\n  ⚪ NO SIGNALS")
    else:
        for sig in signals:
            icon = "🟢" if sig['direction'] == "LONG" else "🔴"
            display_name = clean_pair_name(sig['symbol'])
            print(f"\n  {icon} {sig['direction']} {display_name} {sig['strength']}")
            print(f"  Entry: {sig['entry']:.5f} | Stop: {sig['stop']:.5f} (-{sig['risk_pct']:.3f}%)")
            for i, tp in enumerate(sig['tps']):
                print(f"  TP{i+1}: {tp:.5f} ({CONFIG['trading']['tp_multipliers'][i]}R → {CONFIG['trading']['fractions'][i]*100:.0f}%)")
            print(f"  Size: {sig['qty']} units (${sig['notional']:.2f}) | Risk: ${sig['risk_dollars']:.2f}")
            
    print(f"\n{'='*60}\n")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_full_cycle():
    """Runs the complete 30-minute cycle: Monitor -> Signals -> Auto-Add."""
    print("[*] Starting Full 30-Minute Cycle...")
    run_monitor()
    run_signals()
    print("[*] Cycle Complete.")

def main():
    """Main entry point for the script."""
    initialize_files()
    args = sys.argv[1:]
    
    if "--full-cycle" in args:
        run_full_cycle()
    elif "--signals-only" in args:
        run_signals()
    elif "--monitor-only" in args:
        run_monitor()
    elif "--add-last-signal" in args:
        add_last_signal_as_trade(send_alerts=True)
    elif "--add-trade" in args:
        idx = args.index("--add-trade")
        if len(args) >= idx + 5:
            add_manual_trade(args[idx + 1], args[idx + 2], args[idx + 3], args[idx + 4])
        else:
            print("Usage: python alphaswing_fx.py --add-trade SYMBOL DIRECTION ENTRY STOP")
    elif "--list-trades" in args:
        list_open_trades()
    elif "--remove-trade" in args:
        idx = args.index("--remove-trade")
        if len(args) >= idx + 2:
            remove_trade(args[idx + 1])
        else:
            print("Usage: python alphaswing_fx.py --remove-trade SYMBOL")
    elif "--ban-pair" in args:
        idx = args.index("--ban-pair")
        if len(args) >= idx + 2:
            ban_pair(args[idx + 1])
        else:
            print("Usage: python alphaswing_fx.py --ban-pair SYMBOL")
    elif "--unban-pair" in args:
        idx = args.index("--unban-pair")
        if len(args) >= idx + 2:
            unban_pair(args[idx + 1])
        else:
            print("Usage: python alphaswing_fx.py --unban-pair SYMBOL")
    elif "--list-banned" in args:
        list_banned_pairs()
    elif "--show-learning" in args:
        show_learning()
    elif "--reset-learning" in args:
        if len(args) >= 2 and args[1] in ["bull", "bear", "sideways"]:
            reset_learning(args[1])
        else:
            reset_learning(None)
    elif "--learn-now" in args:
        print("[*] Forcing learning cycle...")
        dxy_df = get_yfinance_klines(CONFIG["universe"]["regime_indicator"], '1h', days=60)
        regime, _ = get_dxy_regime_robust(dxy_df)
        active_brain = get_active_brain(regime)
        result = adapt_brain_weights(active_brain)
        if result is None:
            print("[!] Not enough data to learn yet.")
    elif "--report" in args:
        show_performance_report()
    elif "--loop" in args:
        print(f"[*] AlphaSwing FX v1.0 – Loop mode")
        while True:
            try:
                run_full_cycle()
            except Exception as e:
                print(f"[!] Error: {e}")
            print(f"[*] Sleeping {CONFIG['loop_interval_hours']} hours...")
            time.sleep(CONFIG['loop_interval_hours'] * 3600)
    else:
        run_full_cycle()

if __name__ == "__main__":
    main()