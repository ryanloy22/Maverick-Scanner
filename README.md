# Maverick Scanner

A comparison scanner, built from scratch against [TradeSignal](../Top-signals-),
on the opposite side of a real design tradeoff: **precision vs. convexity.**

## Backtested result (2yr daily bars, full universe, 560 trades)

**+0.117R expectancy per trade** — but that number only holds after two
corrections the first backtest pass forced:

1. **First pass ran on 15m/1h intraday bars and was flat-to-negative
   (-0.06R to -0.09R)** across every filter slice tried (score, volume
   ratio, ATR strength, trend alignment). Intraday breakouts on this
   universe are mostly noise. Moved to **daily bars** — the timeframe the
   original Turtle-style Donchian systems were actually validated on — and
   the edge showed up: +0.06R on 2mo-of-daily-equivalent history.
2. **SHORT backtested negative on every slice** (trend-aligned: -0.03R,
   strong ATR expansion: still negative, high score: still negative) —
   the same "ADX gates strength, not direction" trap the TradeSignal
   review found, except here trend-alignment didn't rescue it either.
   Shorts are now hard-disabled (`ALLOW_SHORTS = False`) until they
   requalify against fresh data.
3. **Stocks/ETFs carry the edge (+0.195R, n=307); crypto is close to
   breakeven (+0.022R, n=253).** Both are live in the default universe —
   worth watching separately once real alerts accumulate, not assuming
   crypto contributes equally.

Re-run `python3 backtest.py` before trusting any further parameter change —
that's not a formality, it's what caught both of the above.

TradeSignal is a confirmation-heavy trend-follower — 8 indicators, sector
rotation, news sentiment, and a multi-timeframe check all have to agree
before it acts. That buys you fewer bad trades and costs you speed: it's
reliably late to every move.

Maverick is a momentum-breakout system. It trades the first real thrust out
of a range on a lean 3-condition gate — Donchian breakout, volume surge,
expanding ATR — sized bigger, levered harder, on a universe picked for
amplitude (leveraged ETFs, high-beta momentum names, a wider altcoin
basket) instead of stability. It will be wrong more often. When it's right,
it's built to be right by a lot more.

Same market, opposite bet on what makes money. That's the comparison.

## Why this design

- **Donchian breakout, not multi-indicator scoring.** TradeSignal's 8-gate
  system is precisely what makes it slow. A 20-bar high/low breakout plus
  volume plus expanding volatility is the leanest signal that still filters
  out pure noise — three conditions, not eight.
- **Trend alignment (EMA20 vs EMA50) is a hard gate, not a score bonus.**
  Originally this was going to be scored, not required — the thesis being
  that counter-trend breakouts are often the biggest reversals and worth
  the swing. The backtest didn't support that: non-aligned breakouts drag
  expectancy down on both sides, so alignment is now required to fire at
  all. This is the same lesson from the TradeSignal review — ADX gates
  trend *strength*, not direction — landing differently here: alignment
  turned out to matter enough to be a wall, not a nudge.
- **Wide stops, room to run.** Breakout entries get stopped out by normal
  pullback noise if the stop is tight. Stop = 2.5x ATR; targets run to 3x,
  5x, 8x ATR. Bigger loser, much bigger winner — the R/R is the edge, not
  the win rate.
- **Universe picked for amplitude.** 3x leveraged ETFs, high-beta single
  names (MSTR, COIN, PLTR, SMCI...), and altcoins beyond BTC/ETH. These
  move harder in both directions than TradeSignal's blue-chip/majors
  universe — that's the point, not a side effect.
- **Risk dial turned up.** 8% risk per trade (TradeSignal risks 5%), up to
  10x crypto leverage (vs. 5x), 4x on stocks/ETFs via margin.

## What it is NOT (yet)

- **Alerts only.** No order placement, no Alpaca wiring. It logs to
  `alerts.json` and pushes to Telegram (same env vars as TradeSignal:
  `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — messages are branded
  🐆 MAVERICK so they're easy to tell apart in the same chat feed).
- **Not validated on live P&L yet.** Only backtested on recent history
  (see below). Treat every number here as a starting hypothesis, not a
  track record — that's exactly the lesson the TradeSignal checkpoint
  review just taught: verify before you trust a system with money.

## Real risk, said plainly

This is a genuinely more aggressive system than TradeSignal by every axis
that matters — more leverage, wider stops (bigger dollar loss per stop-out),
higher risk per trade, and a universe of names built to swing hard. That
means bigger drawdowns are the expected cost of the design, not a bug. If
you eventually wire this to real capital, expect a rougher equity curve
than TradeSignal's, win or lose. Don't put real money behind it before it's
been checked against at least a few weeks of live signal grading, the same
way TradeSignal's May 23 rebuild was validated before being trusted.

## Running it

```bash
pip install -r requirements.txt

# validate the logic against recent history first
python3 backtest.py

# then run a live scan (alerts only)
python3 scanner.py
```

## Files

- `scanner.py` — live scanning engine + Telegram alerting
- `backtest.py` — walk-forward validation (Donchian/ATR/volume computed
  once per ticker, then walked bar-by-bar with no lookahead; outcomes
  graded sequentially — stop-first — so a stopped-out trade can never
  score as a win)
- `alerts.json` — generated at runtime, one row per alert fired
