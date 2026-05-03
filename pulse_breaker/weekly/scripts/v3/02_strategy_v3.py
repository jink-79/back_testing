"""
=============================================================
  PULSE BREAKER — Weekly — v3
  Location : pulse_breaker/weekly/scripts/v3/02_strategy_v3.py
  Results  : pulse_breaker/weekly/results/v3/

  Changes from v1:
    - BREAKOUT_LOOKBACK : 1  → 2   (close > 2-week high)
    - EXIT_LOOKBACK     : 3  → 5   (exit below 5-week low)
    - TRAIL_TRIGGER_PCT : 3% → 5%  (trail activates later)
    - TRAIL_OFFSET_PCT  : 2% → 2%  (unchanged)
    - RS filter         : REMOVED entirely
    - RS slope filter   : REMOVED entirely

  Entry : Close > highest high of previous 2 weeks
  Exit 1: Weekly close < lowest low of last 5 candles
  Exit 2: Trailing stop — activates at +5% above entry,
           trails at (highest close since entry) - 2%
  Capital: Rs 40,000 per trade, simultaneous trades OK
  Re-entry: Allowed on fresh valid signal

  To run:
    cd pulse_breaker/weekly/scripts/v3
    python 02_strategy_v3.py
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parents[3] / "common" / "data" / "weekly"
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# STRATEGY PARAMETERS — v3
# ──────────────────────────────────────────────
VERSION             = "v3"
BREAKOUT_LOOKBACK   = 2       # close > highest high of last 2 weeks
EXIT_LOOKBACK       = 5       # exit if close < lowest low of last 5 candles
TRAIL_TRIGGER_PCT   = 5.0     # trail activates after +5% from entry
TRAIL_OFFSET_PCT    = 2.0     # trail = highest close since entry - 2%
CAPITAL_PER_TRADE   = 40_000
NIFTY_FILE          = "NIFTY_weekly.csv"

# RS filter and slope filter: OFF in this version
RS_FILTER_ON        = False
RS_SLOPE_ON         = False

print("=" * 65)
print(f"  PULSE BREAKER — Weekly — {VERSION}")
print(f"  BREAKOUT_LOOKBACK : {BREAKOUT_LOOKBACK} weeks")
print(f"  EXIT_LOOKBACK     : {EXIT_LOOKBACK} weeks")
print(f"  TRAIL_TRIGGER_PCT : {TRAIL_TRIGGER_PCT}%")
print(f"  TRAIL_OFFSET_PCT  : {TRAIL_OFFSET_PCT}%")
print(f"  RS filter         : OFF")
print(f"  RS slope filter   : OFF")
print(f"  Data    : {DATA_DIR}")
print(f"  Results : {RESULTS_DIR}")
print("=" * 65)


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════

def load_csv(filepath):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def load_all():
    print("\n--- Loading Data ---")

    nifty = load_csv(DATA_DIR / NIFTY_FILE)
    if nifty is None:
        raise FileNotFoundError(
            f"NIFTY not found at {DATA_DIR / NIFTY_FILE}\n"
            "Run common/fetch_data.py first."
        )

    stocks = {}
    skipped = 0
    min_bars = BREAKOUT_LOOKBACK + EXIT_LOOKBACK + 10

    for f in sorted(DATA_DIR.glob("*_weekly.csv")):
        if f.name.upper() == NIFTY_FILE.upper():
            continue
        df = load_csv(f)
        if df is not None and len(df) >= min_bars:
            stocks[f.stem.replace("_weekly", "")] = df
        else:
            skipped += 1

    print(f"  NIFTY loaded  : {len(nifty)} bars")
    print(f"  Stocks loaded : {len(stocks)}")
    if skipped:
        print(f"  Skipped       : {skipped} (too few bars)")
    return nifty, stocks


# ══════════════════════════════════════════════
# MANSFIELD RS — computed for logging only
# ══════════════════════════════════════════════

def calc_mansfield_rs(stock_df, nifty_df, period=55):
    combined    = pd.DataFrame({'s': stock_df['close'], 'n': nifty_df['close']}).dropna()
    raw_rs      = combined['s'] / combined['n']
    raw_rs_prev = raw_rs.shift(period)
    return ((raw_rs / raw_rs_prev) - 1) * 100


# ══════════════════════════════════════════════
# STRATEGY SIMULATION
# ══════════════════════════════════════════════

def run_strategy(name, stock_df, nifty_df):
    df = stock_df.copy()

    # Mansfield RS — logged as observation only, NOT used as filter
    mrs = calc_mansfield_rs(df, nifty_df)
    df  = df.join(mrs.rename('mansfield_rs'), how='left')

    # Entry: close > highest high of previous BREAKOUT_LOOKBACK weeks
    df['prev_high'] = df['high'].rolling(BREAKOUT_LOOKBACK).max().shift(1)

    # Exit: close < lowest low of previous EXIT_LOOKBACK candles
    df['exit_low']  = df['low'].rolling(EXIT_LOOKBACK).min().shift(1)

    # Volume ratio — observation only
    df['vol_avg_10'] = df['volume'].rolling(10).mean().shift(1)
    df['vol_ratio']  = (df['volume'] / df['vol_avg_10']).round(2)

    df.dropna(inplace=True)

    trades        = []
    in_trade      = False
    entry_price   = None
    entry_date    = None
    trail_active  = False
    trail_stop    = None
    highest_close = None
    shares        = None
    index_list    = list(df.index)

    for i, (date, row) in enumerate(df.iterrows()):

        # ── EXIT ──────────────────────────────
        if in_trade:
            cur = row['close']

            if cur > highest_close:
                highest_close = cur

            if not trail_active and cur >= entry_price * (1 + TRAIL_TRIGGER_PCT / 100):
                trail_active = True

            if trail_active:
                trail_stop = highest_close * (1 - TRAIL_OFFSET_PCT / 100)

            structure_exit = cur < row['exit_low']
            trail_exit     = trail_active and (cur < trail_stop)

            if structure_exit or trail_exit:
                exit_reason = "Structure Break" if structure_exit else "Trailing Stop"
                pnl_total   = (cur - entry_price) * shares
                pnl_pct     = (cur / entry_price - 1) * 100
                entry_idx   = index_list.index(entry_date)

                trades.append({
                    "version":               VERSION,
                    "stock":                 name,
                    "entry_date":            entry_date.date(),
                    "exit_date":             date.date(),
                    "entry_price":           round(entry_price, 2),
                    "exit_price":            round(cur, 2),
                    "shares":                shares,
                    "capital":               CAPITAL_PER_TRADE,
                    "pnl_rs":                round(pnl_total, 2),
                    "pnl_pct":               round(pnl_pct, 2),
                    "exit_reason":           exit_reason,
                    "trail_activated":       trail_active,
                    "weeks_held":            i - entry_idx,
                    "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2)
                                             if not pd.isna(df.loc[entry_date, 'mansfield_rs']) else None,
                    "vol_ratio_at_entry":    df.loc[entry_date, 'vol_ratio'],
                })

                in_trade      = False
                entry_price   = None
                entry_date    = None
                trail_active  = False
                trail_stop    = None
                highest_close = None
                shares        = None

        # ── ENTRY ─────────────────────────────
        # No RS filter, no slope filter — pure price action only
        if not in_trade:
            if row['close'] > row['prev_high']:
                entry_price   = row['close']
                entry_date    = date
                shares        = max(1, int(CAPITAL_PER_TRADE / entry_price))
                in_trade      = True
                trail_active  = False
                trail_stop    = None
                highest_close = entry_price

    # Close any open trade at last bar
    if in_trade:
        last      = df.iloc[-1]
        pnl_total = (last['close'] - entry_price) * shares
        pnl_pct   = (last['close'] / entry_price - 1) * 100
        entry_idx = index_list.index(entry_date)

        trades.append({
            "version":               VERSION,
            "stock":                 name,
            "entry_date":            entry_date.date(),
            "exit_date":             df.index[-1].date(),
            "entry_price":           round(entry_price, 2),
            "exit_price":            round(last['close'], 2),
            "shares":                shares,
            "capital":               CAPITAL_PER_TRADE,
            "pnl_rs":                round(pnl_total, 2),
            "pnl_pct":               round(pnl_pct, 2),
            "exit_reason":           "Open at End",
            "trail_activated":       trail_active,
            "weeks_held":            len(df) - 1 - entry_idx,
            "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2)
                                     if not pd.isna(df.loc[entry_date, 'mansfield_rs']) else None,
            "vol_ratio_at_entry":    df.loc[entry_date, 'vol_ratio'],
        })

    return trades


# ══════════════════════════════════════════════
# PERFORMANCE SUMMARY
# ══════════════════════════════════════════════

def summarise(trades_df):
    print("\n" + "=" * 65)
    print(f"  PERFORMANCE SUMMARY — PULSE BREAKER {VERSION}")
    print("=" * 65)

    summary_rows = []

    for stock in sorted(trades_df['stock'].unique()):
        t       = trades_df[trades_df['stock'] == stock]
        winners = t[t['pnl_rs'] > 0]
        losers  = t[t['pnl_rs'] <= 0]
        wr      = len(winners) / len(t) * 100 if len(t) else 0

        summary_rows.append({
            "version":        VERSION,
            "stock":          stock,
            "total_trades":   len(t),
            "winners":        len(winners),
            "losers":         len(losers),
            "win_rate_pct":   round(wr, 1),
            "total_pnl_rs":   round(t['pnl_rs'].sum(), 2),
            "avg_win_pct":    round(winners['pnl_pct'].mean(), 2) if len(winners) else 0,
            "avg_loss_pct":   round(losers['pnl_pct'].mean(), 2)  if len(losers)  else 0,
            "avg_weeks_held": round(t['weeks_held'].mean(), 1),
        })

    summary_df   = pd.DataFrame(summary_rows)
    total        = len(trades_df)
    total_pnl    = trades_df['pnl_rs'].sum()
    winners_all  = trades_df[trades_df['pnl_rs'] > 0]
    losers_all   = trades_df[trades_df['pnl_rs'] <= 0]
    wr_overall   = len(winners_all) / total * 100 if total else 0
    avg_win      = winners_all['pnl_pct'].mean() if len(winners_all) else 0
    avg_loss     = losers_all['pnl_pct'].mean()  if len(losers_all)  else 0
    rr           = abs(avg_win / avg_loss) if avg_loss else 0
    capital      = CAPITAL_PER_TRADE * total
    roi          = total_pnl / capital * 100 if capital else 0
    exits        = trades_df['exit_reason'].value_counts()
    neg_stocks   = (summary_df['total_pnl_rs'] < 0).sum()

    print(f"\n  OVERALL — {VERSION}")
    print(f"    Total Trades   : {total:,}")
    print(f"    Win Rate       : {wr_overall:.1f}%")
    print(f"    Total P&L      : Rs {total_pnl:,.0f}")
    print(f"    Capital Used   : Rs {capital:,.0f}")
    print(f"    Overall ROI    : {roi:.2f}%")
    print(f"    Avg Win        : {avg_win:.2f}%")
    print(f"    Avg Loss       : {avg_loss:.2f}%")
    print(f"    Risk/Reward    : {rr:.2f}x")
    print(f"    Negative Stocks: {neg_stocks}")
    print(f"    Exit Reasons   : " + "  ".join(f"{k}={v}" for k, v in exits.items()))
    print("=" * 65)

    return summary_df


# ══════════════════════════════════════════════
# COMPARE v1 / v2 / v3
# ══════════════════════════════════════════════

def compare_versions(v3_summary):
    print("\n" + "=" * 65)
    print("  VERSION COMPARISON — v1 vs v2 vs v3")
    print("=" * 65)

    results_root = SCRIPT_DIR.parents[1] / "results"

    def read_summary(version):
        p = results_root / version / "summary.csv"
        if not p.exists():
            return None
        return pd.read_csv(p)

    versions = {"v1": read_summary("v1"),
                "v2": read_summary("v2"),
                "v3": v3_summary}

    rows = []
    for vname, df in versions.items():
        if df is None:
            print(f"  {vname}: summary.csv not found — skipping")
            continue
        total_t = df['total_trades'].sum()
        total_p = df['total_pnl_rs'].sum()
        wr      = (df['winners'].sum() / total_t * 100) if total_t else 0
        cap     = CAPITAL_PER_TRADE * total_t
        roi     = total_p / cap * 100 if cap else 0
        avg_w   = df['avg_win_pct'].mean()
        avg_l   = df['avg_loss_pct'].mean()
        rr      = abs(avg_w / avg_l) if avg_l else 0
        neg     = (df['total_pnl_rs'] < 0).sum()
        rows.append((vname, total_t, wr, total_p, roi, avg_w, avg_l, rr, neg))

    print(f"\n  {'Version':<8} {'Trades':>8} {'WR%':>7} {'Total P&L':>14} "
          f"{'ROI%':>7} {'Avg Win':>8} {'Avg Loss':>9} {'R/R':>6} {'Neg Stks':>9}")
    print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*9}")
    for r in rows:
        vn, tt, wr, tp, roi, aw, al, rr, neg = r
        print(f"  {vn:<8} {tt:>8,} {wr:>6.1f}% {tp:>14,.0f} "
              f"{roi:>6.2f}% {aw:>7.2f}% {al:>8.2f}% {rr:>6.2f}x {neg:>9}")
    print("=" * 65)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    nifty, stocks = load_all()

    if not stocks:
        print("\nERROR: No stock data loaded. Check data folder path.")
        return

    print(f"\n--- Running Strategy ({VERSION}) ---")
    all_trades = []

    for i, (name, df) in enumerate(stocks.items(), 1):
        trades = run_strategy(name, df, nifty)
        if trades:
            all_trades.extend(trades)
        if i % 25 == 0:
            print(f"  [{i:>3}/{len(stocks)}] stocks processed...")

    print(f"  Done. Total trades: {len(all_trades):,}")

    if not all_trades:
        print("\nNo trades found. Check data.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values('entry_date', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)

    # Save
    tl_path = RESULTS_DIR / "trade_log.csv"
    trades_df.to_csv(tl_path, index=False)
    print(f"\n  Trade log : {tl_path}  ({len(trades_df):,} rows)")

    summary_df = summarise(trades_df)

    sm_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(sm_path, index=False)
    print(f"  Summary   : {sm_path}")

    compare_versions(summary_df)
    print("\nDone. Check pulse_breaker/weekly/results/v3/")


if __name__ == "__main__":
    main()