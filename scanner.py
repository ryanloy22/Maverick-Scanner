"""
Maverick Scanner — volatility-breakout momentum engine.

Philosophy (deliberately different from TradeSignal, not an iteration on it):
TradeSignal is a confirmation-heavy trend-follower — 8 indicators have to
agree before it acts, which means it's late to every move but rarely wrong
about the regime. This scanner is a momentum-breakout system: it trades the
first real thrust out of a range, on a leaner 3-condition gate, sized bigger,
levered harder, on a universe picked for amplitude instead of stability. It
will be wrong more often. When it's right, it should be right by a lot more.

Two systems, same market, opposite bets on what makes money: precision or
convexity. That's the comparison.
"""

import os
import json
import time
import datetime
import warnings
import collections

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from ta.volatility import AverageTrueRange

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# UNIVERSE — picked for amplitude, not stability
# ─────────────────────────────────────────────────────────────────────────

# 3x leveraged ETFs — same setups as the underlying index, 3x the move,
# 3x the whipsaw. Bull side; direction is set by which way it breaks.
LEVERAGED_ETFS = [
    "TQQQ", "SOXL", "SPXL", "UPRO", "TECL", "TNA", "FAS",
    "LABU", "NAIL", "WEBL", "BOIL", "YINN",
]

# High-beta single names — story stocks, retail-driven, prone to
# multi-day momentum runs and equally violent reversals.
MOMENTUM_STOCKS = [
    "MSTR", "COIN", "PLTR", "SMCI", "IONQ", "RIVN", "CVNA", "SOFI",
    "MARA", "RIOT", "AFRM", "UPST", "ARM", "AI", "RGTI", "CLSK",
    "APLD", "HOOD", "DKNG", "RKLB",
]

# Wider altcoin basket — beyond majors, where the real convexity lives
# and also where the real air-pockets live.
CRYPTO_ALTS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
    "ARB11841-USD", "OP-USD", "INJ-USD", "SUI20947-USD", "DOGE-USD",
    "WIF-USD", "SEI-USD", "ONDO-USD", "AAVE-USD", "BNB-USD",
]

CONFIG = {
    "DONCHIAN_LOOKBACK":       20,     # bars to define the breakout range
    "EMA_TREND_SPAN":          50,     # single trend filter, not multi-timeframe
    "ATR_PERIOD":              14,
    "VOL_SURGE_RATIO":         1.3,    # current bar vol vs 20-bar avg
    "VOL_EXPANSION_LOOKBACK":  10,     # bars to judge "is ATR expanding"
    "STOP_ATR_MULT":           2.5,    # wide — breakouts need room to breathe
    "TARGET1_ATR_MULT":        3.0,
    "TARGET2_ATR_MULT":        5.0,
    "TARGET3_ATR_MULT":        8.0,    # let winners run
    "MIN_SCORE":               6,      # lean gate on purpose — see analyze_ticker
    "REQUIRE_TREND_ALIGN":     True,   # hard gate, not just a score bonus —
                                        # backtest.py showed non-aligned breakouts
                                        # drag expectancy down on both sides
    "ALLOW_SHORTS":            False,  # SHORT backtested negative on every filter
                                        # combination tried (trend-aligned, strong
                                        # ATR expansion, high score) over a 2yr
                                        # daily backtest — LONG is where the
                                        # demonstrated edge lives. Re-enable only
                                        # after re-validating with backtest.py.
    "RISK_PCT_PER_TRADE":      8.0,    # aggressive; TradeSignal risks 5%
    "CRYPTO_LEVERAGE":         10,
    "STOCK_LEVERAGE":          4,
    "CRYPTO_ACCOUNT":          20000,  # same notional base as TradeSignal —
    "STOCK_ACCOUNT":           1200,   # apples-to-apples on capital, not risk dial
    "TELEGRAM_BOT_TOKEN":      os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID":        os.getenv("TELEGRAM_CHAT_ID", ""),
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_PATH = os.path.join(OUTPUT_DIR, "alerts.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.json")


# ─────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────

def get_col(df, name):
    c = df[name]
    if isinstance(df.columns, pd.MultiIndex):
        c = c.iloc[:, 0]
    return c.squeeze()


def fetch(ticker, is_crypto):
    # Daily bars, not intraday — backtest.py showed 15m/1h breakout signals
    # run negative expectancy (too much false-breakout noise); the edge only
    # shows up on daily-bar Donchian breakouts, same timeframe the original
    # Turtle-style trend systems were validated on.
    interval, period = "1d", "1y"
    try:
        df = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True, threads=False)
        if df is None or len(df) < CONFIG["EMA_TREND_SPAN"] + 5:
            return None
        return df
    except Exception:
        return None


def calc_atr_series(df, period=14):
    hi, lo, cl = get_col(df, "High"), get_col(df, "Low"), get_col(df, "Close")
    return AverageTrueRange(hi, lo, cl, window=period).average_true_range()


# ─────────────────────────────────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────────────────

def analyze_ticker(ticker, is_crypto, btc_trend=None):
    df = fetch(ticker, is_crypto)
    if df is None:
        return None

    close = get_col(df, "Close")
    high = get_col(df, "High")
    low = get_col(df, "Low")
    vol = get_col(df, "Volume")

    lookback = CONFIG["DONCHIAN_LOOKBACK"]
    if len(df) < lookback + CONFIG["EMA_TREND_SPAN"]:
        return None

    price = float(close.iloc[-1])

    # Donchian range EXCLUDES the current bar — breaking your own high doesn't count
    prior_high = float(high.iloc[-lookback - 1:-1].max())
    prior_low = float(low.iloc[-lookback - 1:-1].min())

    breakout_long = price > prior_high
    breakout_short = price < prior_low
    if not (breakout_long or breakout_short):
        return None
    direction = "LONG" if breakout_long else "SHORT"
    if direction == "SHORT" and not CONFIG["ALLOW_SHORTS"]:
        return None

    # ── Volatility expansion — breakouts on dead volume/range are traps ──
    atr_series = calc_atr_series(df, CONFIG["ATR_PERIOD"])
    atr_val = float(atr_series.iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0:
        return None
    prior_atr_avg = float(atr_series.iloc[-CONFIG["VOL_EXPANSION_LOOKBACK"] - 1:-1].mean())
    if pd.isna(prior_atr_avg) or prior_atr_avg <= 0:
        return None
    atr_expanding = atr_val >= prior_atr_avg

    # ── Volume surge ──────────────────────────────────────────────────
    avg_vol20 = float(vol.iloc[-21:-1].mean())
    cur_vol = float(vol.iloc[-1])
    vol_ratio = (cur_vol / avg_vol20) if avg_vol20 > 0 else 0
    vol_surge = vol_ratio >= CONFIG["VOL_SURGE_RATIO"]

    # ── Hard gate: breakout needs both fuel (volume) and room (expanding ATR) ──
    if not (vol_surge and atr_expanding):
        return None

    # ── Trend alignment — single lean filter, not a multi-indicator stack ──
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=CONFIG["EMA_TREND_SPAN"], adjust=False).mean()
    trend_bullish = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
    trend_aligned = (direction == "LONG" and trend_bullish) or \
                    (direction == "SHORT" and not trend_bullish)
    if CONFIG["REQUIRE_TREND_ALIGN"] and not trend_aligned:
        return None

    # ── Score — additive, lean, no penalty stack ──────────────────────
    score = 3  # base: cleared the hard gate (breakout + volume + volatility)
    notes = ["Breakout+3"]
    if vol_ratio >= 2.0:
        score += 3
        notes.append(f"VolSurge x{vol_ratio:.1f}+3")
    else:
        score += 2
        notes.append(f"VolSurge x{vol_ratio:.1f}+2")
    if trend_aligned:
        score += 2
        notes.append("TrendAlign+2")
    if atr_val >= prior_atr_avg * 1.3:
        score += 1
        notes.append("VolExpand+1")
    if is_crypto and btc_trend:
        if (direction == "LONG" and btc_trend == "bullish") or \
           (direction == "SHORT" and btc_trend == "bearish"):
            score += 1
            notes.append(f"BTC{btc_trend}+1")

    if score < CONFIG["MIN_SCORE"]:
        return None

    # ── Levels ─────────────────────────────────────────────────────────
    stop_dist = atr_val * CONFIG["STOP_ATR_MULT"]
    t1_dist = atr_val * CONFIG["TARGET1_ATR_MULT"]
    t2_dist = atr_val * CONFIG["TARGET2_ATR_MULT"]
    t3_dist = atr_val * CONFIG["TARGET3_ATR_MULT"]

    if direction == "LONG":
        stop = price - stop_dist
        t1, t2, t3 = price + t1_dist, price + t2_dist, price + t3_dist
    else:
        stop = price + stop_dist
        t1, t2, t3 = price - t1_dist, price - t2_dist, price - t3_dist

    rr = t1_dist / stop_dist if stop_dist else 0

    account = CONFIG["CRYPTO_ACCOUNT"] if is_crypto else CONFIG["STOCK_ACCOUNT"]
    leverage = CONFIG["CRYPTO_LEVERAGE"] if is_crypto else CONFIG["STOCK_LEVERAGE"]
    risk_dollars = account * CONFIG["RISK_PCT_PER_TRADE"] / 100
    units = risk_dollars / stop_dist
    notional = units * price
    max_notional = account * leverage
    if notional > max_notional:
        units = max_notional / price
        notional = max_notional

    return {
        "alerted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ticker": ticker,
        "type": "crypto" if is_crypto else "stock/etf",
        "direction": direction,
        "entry": round(price, 6),
        "stop_loss": round(stop, 6),
        "target1": round(t1, 6),
        "target2": round(t2, 6),
        "target3": round(t3, 6),
        "score": score,
        "rr_ratio": round(rr, 2),
        "atr": round(atr_val, 6),
        "vol_ratio": round(vol_ratio, 2),
        "trend_aligned": trend_aligned,
        "notes": notes,
        "position": {
            "units": round(units, 6),
            "dollar_risk": round(risk_dollars, 2),
            "notional": round(notional, 2),
            "account": "crypto" if is_crypto else "stocks",
            "leverage": leverage,
        },
    }


def get_btc_trend():
    df = fetch("BTC-USD", is_crypto=True)
    if df is None:
        return None
    close = get_col(df, "Close")
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    return "bullish" if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else "bearish"


# ─────────────────────────────────────────────────────────────────────────
# ALERTING
# ─────────────────────────────────────────────────────────────────────────

def already_alerted_today(ticker, direction):
    if not os.path.exists(ALERTS_PATH):
        return False
    with open(ALERTS_PATH) as f:
        alerts = json.load(f)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    for a in alerts:
        if a["ticker"] == ticker and a["direction"] == direction:
            a_date = datetime.datetime.fromisoformat(a["alerted_at"]).date()
            if a_date == today:
                return True
    return False


def save_alert(signal):
    alerts = []
    if os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH) as f:
            alerts = json.load(f)
    alerts.append(signal)
    with open(ALERTS_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


def update_open_positions():
    """Mark existing alerts as stopped/target-hit based on current price,
    so the dashboard doesn't keep showing invalidated setups as live."""
    if not os.path.exists(ALERTS_PATH):
        return
    with open(ALERTS_PATH) as f:
        alerts = json.load(f)

    changed = False
    for a in alerts:
        if a.get("status", "open") != "open":
            continue
        is_crypto = a.get("type") == "crypto"
        df = fetch(a["ticker"], is_crypto)
        if df is None:
            continue
        price = float(get_col(df, "Close").iloc[-1])
        long_dir = a["direction"] == "LONG"

        stopped = price <= a["stop_loss"] if long_dir else price >= a["stop_loss"]
        if stopped:
            a["status"] = "stopped"
            a["closed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            a["close_price"] = round(price, 6)
            changed = True
            continue

        hit = None
        for label in ("target3", "target2", "target1"):
            level = a.get(label)
            if level is None:
                continue
            if (long_dir and price >= level) or (not long_dir and price <= level):
                hit = label
                break
        if hit:
            a["status"] = f"{hit}_hit"
            a["closed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            a["close_price"] = round(price, 6)
            changed = True

    if changed:
        with open(ALERTS_PATH, "w") as f:
            json.dump(alerts, f, indent=2)


def format_telegram(signal):
    arr = "▲" if signal["direction"] == "LONG" else "▼"
    pos = signal["position"]
    lines = [
        f"\U0001f406 <b>MAVERICK BREAKOUT</b>",
        f"<b>{signal['ticker']}</b>  {arr}{signal['direction']}  |  Score: {signal['score']}  |  R/R: {signal['rr_ratio']}:1",
        "",
        f"Entry:  {signal['entry']}",
        f"Stop:   {signal['stop_loss']}",
        f"T1:     {signal['target1']}",
        f"T2:     {signal['target2']}",
        f"T3:     {signal['target3']}",
        "",
        f"Size: {pos['units']}  (≈${pos['notional']:,.0f} notional, {pos['leverage']}x)",
        f"Risk: ${pos['dollar_risk']:.0f}  ({CONFIG['RISK_PCT_PER_TRADE']}% of acct)",
        f"Vol: {signal['vol_ratio']}x avg  |  TrendAligned: {signal['trend_aligned']}",
    ]
    return "\n".join(lines)


def send_telegram(message):
    token, chat_id = CONFIG["TELEGRAM_BOT_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"]
    if not token or not chat_id:
        print("  [telegram disabled — no token/chat_id set]")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  ⚠ telegram send failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def run_scanner():
    print("\n" + "━" * 60)
    print("  Maverick Scanner — momentum breakout, riskier by design")
    print("━" * 60)

    btc_trend = get_btc_trend()
    print(f"  BTC trend: {btc_trend}")

    update_open_positions()

    universe = (
        [(t, False) for t in LEVERAGED_ETFS] +
        [(t, False) for t in MOMENTUM_STOCKS] +
        [(t, True) for t in CRYPTO_ALTS]
    )

    found = 0
    for ticker, is_crypto in universe:
        try:
            sig = analyze_ticker(ticker, is_crypto, btc_trend)
        except Exception as e:
            print(f"  ⚠ {ticker}: {e}")
            continue
        if not sig:
            continue
        if already_alerted_today(ticker, sig["direction"]):
            continue
        found += 1
        arr = "▲" if sig["direction"] == "LONG" else "▼"
        print(f"  {arr} {ticker:14s} {sig['direction']:5s} score={sig['score']} "
              f"entry={sig['entry']} stop={sig['stop_loss']} rr={sig['rr_ratio']}")
        save_alert(sig)
        send_telegram(format_telegram(sig))
        time.sleep(0.3)

    print(f"\n  Scanned {len(universe)} tickers, {found} new alert(s).")
    print("━" * 60 + "\n")

    with open(STATUS_PATH, "w") as f:
        json.dump({
            "last_run": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "btc_trend": btc_trend,
            "scanned": len(universe),
            "found": found,
        }, f, indent=2)


if __name__ == "__main__":
    run_scanner()
