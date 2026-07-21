"""
Walk-forward validation for the Maverick breakout logic. Same discipline the
TradeSignal checkpoint review demanded: don't trust a strategy that hasn't
been checked against real price history, and grade outcomes sequentially
(stop-first, in time order) so a stopped-out trade can't score as a win.

Precomputes every rolling indicator (Donchian range, ATR, volume average,
EMAs) over the whole series first, then walks bar-by-bar — each bar only
ever sees data at or before its own index, so there's no lookahead.
"""

import warnings
import collections

import pandas as pd
import yfinance as yf
from ta.volatility import AverageTrueRange

from scanner import LEVERAGED_ETFS, MOMENTUM_STOCKS, CRYPTO_ALTS, CONFIG, get_col

warnings.filterwarnings("ignore")


def precompute(df):
    close, high, low, vol = get_col(df, "Close"), get_col(df, "High"), get_col(df, "Low"), get_col(df, "Volume")
    lookback = CONFIG["DONCHIAN_LOOKBACK"]

    donch_high = high.rolling(lookback).max().shift(1)   # excludes current bar
    donch_low = low.rolling(lookback).min().shift(1)

    atr = AverageTrueRange(high, low, close, window=CONFIG["ATR_PERIOD"]).average_true_range()
    atr_avg_prior = atr.rolling(CONFIG["VOL_EXPANSION_LOOKBACK"]).mean().shift(1)

    vol_avg20 = vol.rolling(20).mean().shift(1)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=CONFIG["EMA_TREND_SPAN"], adjust=False).mean()

    return dict(close=close, high=high, low=low, vol=vol,
                donch_high=donch_high, donch_low=donch_low,
                atr=atr, atr_avg_prior=atr_avg_prior,
                vol_avg20=vol_avg20, ema20=ema20, ema50=ema50)


def find_signals(df, ticker, is_crypto):
    p = precompute(df)
    signals = []
    start = max(CONFIG["DONCHIAN_LOOKBACK"] + 1, CONFIG["EMA_TREND_SPAN"] + 1, 21)

    for i in range(start, len(df)):
        price = float(p["close"].iloc[i])
        dh, dl = p["donch_high"].iloc[i], p["donch_low"].iloc[i]
        if pd.isna(dh) or pd.isna(dl):
            continue
        breakout_long = price > float(dh)
        breakout_short = price < float(dl)
        if not (breakout_long or breakout_short):
            continue
        direction = "LONG" if breakout_long else "SHORT"
        if direction == "SHORT" and not CONFIG["ALLOW_SHORTS"]:
            continue

        atr_val, atr_prior = p["atr"].iloc[i], p["atr_avg_prior"].iloc[i]
        if pd.isna(atr_val) or pd.isna(atr_prior) or atr_prior <= 0:
            continue
        atr_expanding = float(atr_val) >= float(atr_prior)

        avg_vol, cur_vol = p["vol_avg20"].iloc[i], p["vol"].iloc[i]
        if pd.isna(avg_vol) or avg_vol <= 0:
            continue
        vol_ratio = float(cur_vol) / float(avg_vol)
        if not (vol_ratio >= CONFIG["VOL_SURGE_RATIO"] and atr_expanding):
            continue

        trend_bullish = float(p["ema20"].iloc[i]) > float(p["ema50"].iloc[i])
        trend_aligned = (direction == "LONG" and trend_bullish) or (direction == "SHORT" and not trend_bullish)
        if CONFIG["REQUIRE_TREND_ALIGN"] and not trend_aligned:
            continue

        score = 3
        score += 3 if vol_ratio >= 2.0 else 2
        if trend_aligned:
            score += 2
        if float(atr_val) >= float(atr_prior) * 1.3:
            score += 1
        if score < CONFIG["MIN_SCORE"]:
            continue

        atr_v = float(atr_val)
        stop_dist = atr_v * CONFIG["STOP_ATR_MULT"]
        t1_dist, t2_dist, t3_dist = (atr_v * CONFIG["TARGET1_ATR_MULT"],
                                      atr_v * CONFIG["TARGET2_ATR_MULT"],
                                      atr_v * CONFIG["TARGET3_ATR_MULT"])
        if direction == "LONG":
            stop, t1, t2, t3 = price - stop_dist, price + t1_dist, price + t2_dist, price + t3_dist
        else:
            stop, t1, t2, t3 = price + stop_dist, price - t1_dist, price - t2_dist, price - t3_dist

        signals.append({
            "ticker": ticker, "type": "crypto" if is_crypto else "stock/etf",
            "direction": direction, "idx": i, "alerted_at": df.index[i].isoformat(),
            "entry": price, "stop_loss": stop, "target1": t1, "target2": t2, "target3": t3,
            "score": score, "vol_ratio": round(vol_ratio, 2),
            "trend_aligned": trend_aligned,
            "atr_ratio": round(float(atr_val) / float(atr_prior), 2),
        })
    return signals


def grade_signal(df, sig):
    """All-or-nothing grade: first level touched (stop or a target) closes
    the whole position. Useful for a single-take-profit read, but it can't
    tell you whether 'let winners run' actually works — see grade_scaled."""
    high, low = get_col(df, "High"), get_col(df, "Low")
    direction = sig["direction"]
    stop, t1, t2, t3 = sig["stop_loss"], sig["target1"], sig["target2"], sig["target3"]
    for i in range(sig["idx"] + 1, len(df)):
        hi, lo = float(high.iloc[i]), float(low.iloc[i])
        if direction == "LONG":
            stop_hit = lo <= stop
            t3_hit, t2_hit, t1_hit = hi >= t3, hi >= t2, hi >= t1
        else:
            stop_hit = hi >= stop
            t3_hit, t2_hit, t1_hit = lo <= t3, lo <= t2, lo <= t1
        if stop_hit:
            return "LOSS"
        if t3_hit:
            return "WIN_T3"
        if t2_hit:
            return "WIN_T2"
        if t1_hit:
            return "WIN_T1"
    return "ACTIVE"


# Position scaled out 40/30/30 across T1/T2/T3, stop trailed to breakeven
# after T1 and to T1 after T2 — the actual mechanic "let winners run"
# requires. Without this, grade_signal() closes 100% at first touch, which
# is why the first backtest pass showed 92% of wins clipping at T1 (1.2R)
# and never getting a chance at the bigger T2/T3 payouts the design is
# supposed to be selling.
SCALE_FRACTIONS = (0.4, 0.3, 0.3)


def grade_scaled(df, sig):
    high, low = get_col(df, "High"), get_col(df, "Low")
    direction = sig["direction"]
    entry, orig_stop = sig["entry"], sig["stop_loss"]
    t1, t2, t3 = sig["target1"], sig["target2"], sig["target3"]
    r1 = CONFIG["TARGET1_ATR_MULT"] / CONFIG["STOP_ATR_MULT"]
    r2 = CONFIG["TARGET2_ATR_MULT"] / CONFIG["STOP_ATR_MULT"]
    r3 = CONFIG["TARGET3_ATR_MULT"] / CONFIG["STOP_ATR_MULT"]

    stage = 0          # 0 = pre-T1, 1 = post-T1 (stop @ breakeven), 2 = post-T2 (stop @ T1)
    cur_stop = orig_stop
    realized_r = 0.0

    for i in range(sig["idx"] + 1, len(df)):
        hi, lo = float(high.iloc[i]), float(low.iloc[i])
        if direction == "LONG":
            stop_hit = lo <= cur_stop
            t1_hit, t2_hit, t3_hit = hi >= t1, hi >= t2, hi >= t3
        else:
            stop_hit = hi >= cur_stop
            t1_hit, t2_hit, t3_hit = lo <= t1, lo <= t2, lo <= t3

        # at most one stage transition per bar — a simplification, slightly
        # conservative since a single huge bar could in theory clear two
        # levels at once
        if stage == 0:
            if stop_hit and not t1_hit:
                return -1.0
            if t1_hit:
                realized_r += SCALE_FRACTIONS[0] * r1
                cur_stop, stage = entry, 1
                continue
        elif stage == 1:
            if stop_hit:
                return realized_r  # remaining exits flat at breakeven
            if t2_hit:
                realized_r += SCALE_FRACTIONS[1] * r2
                cur_stop, stage = t1, 2
                continue
        elif stage == 2:
            if stop_hit:
                return realized_r + SCALE_FRACTIONS[2] * r1  # remainder exits at T1 (trailed stop)
            if t3_hit:
                return realized_r + SCALE_FRACTIONS[2] * r3
    return None  # still open at end of data — excluded from expectancy


def run_backtest():
    universe = ([(t, False) for t in LEVERAGED_ETFS] +
                [(t, False) for t in MOMENTUM_STOCKS] +
                [(t, True) for t in CRYPTO_ALTS])

    all_results = []
    for ticker, is_crypto in universe:
        interval, period = "1d", "2y"
        try:
            df = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True, threads=False)
        except Exception as e:
            print(f"  ⚠ {ticker}: download failed ({e})")
            continue
        if df is None or len(df) < 80:
            continue
        df.index = pd.to_datetime(df.index, utc=True)
        sigs = find_signals(df, ticker, is_crypto)
        for s in sigs:
            s["outcome"] = grade_signal(df, s)
            s["r"] = grade_scaled(df, s)
        all_results.extend(sigs)
        if sigs:
            print(f"  {ticker}: {len(sigs)} signal(s)")

    resolved = [r for r in all_results if r["outcome"] in ("LOSS", "WIN_T1", "WIN_T2", "WIN_T3")]
    wins = [r for r in resolved if r["outcome"].startswith("WIN")]
    closed = [r for r in all_results if r["r"] is not None]

    print("\n" + "=" * 60)
    print(f"TOTAL SIGNALS: {len(all_results)}  |  RESOLVED: {len(resolved)}  |  ACTIVE: {len(all_results) - len(resolved)}")
    if resolved:
        print(f"ALL-OR-NOTHING WIN RATE (single exit at first level touched): "
              f"{len(wins)}/{len(resolved)} = {100 * len(wins) / len(resolved):.1f}%")

    if closed:
        total_r = sum(r["r"] for r in closed)
        avg_r = total_r / len(closed)
        print(f"\nSCALED EXIT (40/30/30 across T1/T2/T3, stop trailed to BE then T1):")
        print(f"  Trades closed: {len(closed)}  |  Sum R: {total_r:.1f}  |  Expectancy: {avg_r:.3f}R/trade")

    for label, key in [("asset type", "type"), ("direction", "direction")]:
        grouped = collections.defaultdict(list)
        for r in closed:
            grouped[r[key]].append(r)
        print(f"\n-- expectancy by {label} (scaled exit) --")
        for k, rs in grouped.items():
            avg = sum(r["r"] for r in rs) / len(rs)
            print(f"  {k}: n={len(rs)}  expectancy={avg:.3f}R")

    return all_results


if __name__ == "__main__":
    run_backtest()
