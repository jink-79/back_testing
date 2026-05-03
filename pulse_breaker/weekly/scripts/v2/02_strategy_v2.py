"""
=============================================================
  PULSE BREAKER — Weekly — v2
  Location : pulse_breaker/weekly/scripts/v2/02_strategy_v2.py
  Results  : pulse_breaker/weekly/results/v2/

  Changes from v1:
    1. TRAIL_OFFSET_PCT  : 2% → 3%  (let winners run longer)
    2. RS Slope Filter   : Mansfield RS must be rising for 3
                           consecutive weeks, not just RS > 0
                           (eliminates flat/weak RS entries)

  All other rules unchanged from v1:
    Entry  : Close > prev week high + RS > 0 + RS rising 3 weeks
    Exit 1 : Weekly close < lowest low of last 3 candles
    Exit 2 : Trailing — activates at +3% above entry
              trails at (highest close since entry) - 3%  ← changed
    Capital: Rs 40,000 per trade, simultaneous trades OK
    Re-entry: Allowed on fresh valid signal

  To run:
    cd pulse_breaker/weekly/scripts/v2
    python 02_strategy_v2.py
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# Script  : pulse_breaker/weekly/scripts/v2/
# Data    : common/data/weekly/   (4 levels up → common)
# Results : pulse_breaker/weekly/results/v2/
# ──────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parents[3] / "common" / "data" / "weekly"
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# STRATEGY PARAMETERS
# ──────────────────────────────────────────────
VERSION             = "v2"
RS_PERIOD           = 55      # Mansfield RS lookback (weeks)
RS_SLOPE_WEEKS      = 3       # NEW: RS must be rising for this many consecutive weeks
BREAKOUT_LOOKBACK   = 1       # Close must be > high of previous N weeks
EXIT_LOOKBACK       = 3       # Exit if close < lowest low of last N candles
TRAIL_TRIGGER_PCT   = 3.0     # Trailing activates after +3% from entry (unchanged)
TRAIL_OFFSET_PCT    = 3.0     # CHANGED: 2% → 3% (wider trail, let winners run)
CAPITAL_PER_TRADE   = 40_000  # Rs per trade
NIFTY_FILE          = "NIFTY_weekly.csv"

print("=" * 65)
print(f"  PULSE BREAKER — Weekly — {VERSION}")
print(f"  Changes vs v1:")
print(f"    Trail offset     : 2% → 3%")
print(f"    RS slope filter  : RS must rise for {RS_SLOPE_WEEKS} consecutive weeks")
print(f"  Data    : {DATA_DIR}")
print(f"  Results : {RESULTS_DIR}")
print("=" * 65)


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════

def load_csv(filepath, name):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def load_all():
    print("\n--- Loading Data ---")

    nifty = load_csv(DATA_DIR / NIFTY_FILE, "NIFTY")
    if nifty is None:
        raise FileNotFoundError(
            f"NIFTY file not found at {DATA_DIR / NIFTY_FILE}\n"
            "Run common/fetch_data.py first."
        )

    stocks = {}
    skipped = 0
    for file in sorted(DATA_DIR.glob("*_weekly.csv")):
        if file.name.upper() == NIFTY_FILE.upper():
            continue
        name = file.stem.replace("_weekly", "")
        df   = load_csv(file, name)
        # Need extra bars for RS slope warmup on top of RS period
        if df is not None and len(df) >= (RS_PERIOD + RS_SLOPE_WEEKS + 10):
            stocks[name] = df
        else:
            skipped += 1

    print(f"  NIFTY loaded    : {len(nifty)} bars")
    print(f"  Stocks loaded   : {len(stocks)}")
    print(f"  Stocks skipped  : {skipped}  (insufficient bars)")
    return nifty, stocks


# ══════════════════════════════════════════════
# MANSFIELD RS
# ══════════════════════════════════════════════

def calc_mansfield_rs(stock_df, nifty_df, period=55):
    """
    Mansfield RS = ((Stock/Nifty) / (Stock/Nifty [period weeks ago]) - 1) * 100
    """
    combined = pd.DataFrame({
        'stock': stock_df['close'],
        'nifty': nifty_df['close']
    }).dropna()

    raw_rs      = combined['stock'] / combined['nifty']
    raw_rs_prev = raw_rs.shift(period)
    return ((raw_rs / raw_rs_prev) - 1) * 100


# ══════════════════════════════════════════════
# RS SLOPE FILTER  (NEW in v2)
# ══════════════════════════════════════════════

def calc_rs_rising(mrs_series, slope_weeks=3):
    """
    Returns a boolean Series: True if RS has been strictly rising
    for `slope_weeks` consecutive weeks.

    Rising = each week's RS > the previous week's RS
    We need slope_weeks consecutive increases, so we check:
      RS[t] > RS[t-1] > RS[t-2] > ... > RS[t-slope_weeks]

    Implementation: compute week-over-week diff, then rolling min
    over slope_weeks periods. If rolling min > 0 → all diffs positive
    → RS has been rising every single week in the window.
    """
    rs_diff         = mrs_series.diff()                         # weekly change in RS
    rolling_min     = rs_diff.rolling(slope_weeks).min()        # smallest change in window
    rs_rising       = rolling_min > 0                           # True only if ALL diffs > 0
    return rs_rising


# ══════════════════════════════════════════════
# STRATEGY + TRADE SIMULATION
# ══════════════════════════════════════════════

def run_strategy(name, stock_df, nifty_df):
    df = stock_df.copy()

    # ── Indicators ────────────────────────────
    mrs          = calc_mansfield_rs(df, nifty_df, RS_PERIOD)
    rs_rising    = calc_rs_rising(mrs, RS_SLOPE_WEEKS)

    df = df.join(mrs.rename('mansfield_rs'),    how='left')
    df = df.join(rs_rising.rename('rs_rising'), how='left')

    df['prev_high']  = df['high'].shift(BREAKOUT_LOOKBACK)
    df['exit_low']   = df['low'].rolling(EXIT_LOOKBACK).min().shift(1)
    df['vol_avg_10'] = df['volume'].rolling(10).mean().shift(1)
    df['vol_ratio']  = df['volume'] / df['vol_avg_10']
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
                exit_reason   = "Structure Break" if structure_exit else "Trailing Stop"
                pnl_total     = (cur - entry_price) * shares
                pnl_pct       = (cur / entry_price - 1) * 100
                entry_idx     = index_list.index(entry_date)

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
                    "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2),
                    "rs_rising_at_entry":    bool(df.loc[entry_date, 'rs_rising']),
                    "vol_ratio_at_entry":    round(df.loc[entry_date, 'vol_ratio'], 2),
                })

                in_trade = False
                entry_price = entry_date = trail_stop = highest_close = shares = None
                trail_active = False

        # ── ENTRY ─────────────────────────────
        if not in_trade:
            breakout = row['close'] > row['prev_high']
            rs_ok    = row['mansfield_rs'] > 0           # v1 condition: RS > 0
            slope_ok = bool(row['rs_rising'])            # v2 addition: RS rising 3 weeks

            if breakout and rs_ok and slope_ok:
                entry_price   = row['close']
                entry_date    = date
                shares        = int(CAPITAL_PER_TRADE / entry_price)
                in_trade      = True
                trail_active  = False
                trail_stop    = None
                highest_close = entry_price

    # Close open trade at last bar
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
            "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2),
            "rs_rising_at_entry":    bool(df.loc[entry_date, 'rs_rising']),
            "vol_ratio_at_entry":    round(df.loc[entry_date, 'vol_ratio'], 2),
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
            "avg_loss_pct":   round(losers['pnl_pct'].mean(), 2) if len(losers) else 0,
            "avg_weeks_held": round(t['weeks_held'].mean(), 1),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Overall
    total      = len(trades_df)
    total_pnl  = trades_df['pnl_rs'].sum()
    wr_overall = (trades_df['pnl_rs'] > 0).mean() * 100
    capital    = CAPITAL_PER_TRADE * total
    roi        = total_pnl / capital * 100 if capital else 0
    winners_all = trades_df[trades_df['pnl_rs'] > 0]
    losers_all  = trades_df[trades_df['pnl_rs'] <= 0]
    avg_win    = winners_all['pnl_pct'].mean()
    avg_loss   = losers_all['pnl_pct'].mean()
    rr         = abs(avg_win / avg_loss) if avg_loss else 0
    exits      = trades_df['exit_reason'].value_counts()

    print(f"\n  OVERALL — {VERSION}")
    print(f"    Total Trades  : {total}")
    print(f"    Win Rate      : {wr_overall:.1f}%")
    print(f"    Total P&L     : Rs {total_pnl:,.0f}")
    print(f"    Total Capital : Rs {capital:,.0f}")
    print(f"    Overall ROI   : {roi:.1f}%")
    print(f"    Avg Win       : {avg_win:.2f}%")
    print(f"    Avg Loss      : {avg_loss:.2f}%")
    print(f"    Risk/Reward   : {rr:.2f}x")
    print(f"    Exit Reasons  : " + "  ".join(f"{k}={v}" for k, v in exits.items()))
    print("=" * 65)

    return summary_df


# ══════════════════════════════════════════════
# COMPARE WITH V1 (if v1 summary exists)
# ══════════════════════════════════════════════

def compare_with_v1(v2_trades_df, v2_summary_df):
    v1_path = SCRIPT_DIR.parents[1] / "results" / "v1" / "summary.csv"
    if not v1_path.exists():
        print("\n  (v1 summary.csv not found — skipping comparison)")
        return

    v1 = pd.read_csv(v1_path)
    v2 = v2_summary_df.copy()

    # Overall v1
    v1_total_trades = v1['total_trades'].sum()
    v1_total_pnl    = v1['total_pnl_rs'].sum()
    v1_wr           = (v1['winners'].sum() / v1_total_trades) * 100
    v1_capital      = CAPITAL_PER_TRADE * v1_total_trades
    v1_roi          = v1_total_pnl / v1_capital * 100

    # Overall v2
    v2_total_trades = v2['total_trades'].sum()
    v2_total_pnl    = v2['total_pnl_rs'].sum()
    v2_wr           = (v2['winners'].sum() / v2_total_trades) * 100
    v2_capital      = CAPITAL_PER_TRADE * v2_total_trades
    v2_roi          = v2_total_pnl / v2_capital * 100

    # Winners/losers count
    v1_neg = (v1['total_pnl_rs'] < 0).sum()
    v2_neg = (v2['total_pnl_rs'] < 0).sum()

    # Avg win / avg loss
    v1_avg_win  = v1['avg_win_pct'].mean()
    v1_avg_loss = v1['avg_loss_pct'].mean()
    v2_avg_win  = v2['avg_win_pct'].mean()
    v2_avg_loss = v2['avg_loss_pct'].mean()
    v1_rr = abs(v1_avg_win / v1_avg_loss)
    v2_rr = abs(v2_avg_win / v2_avg_loss)

    def delta(v2_val, v1_val, higher_better=True):
        d = v2_val - v1_val
        sign = "+" if d > 0 else ""
        good = (d > 0) == higher_better
        flag = "BETTER" if good else "WORSE"
        return f"{sign}{d:.1f}  [{flag}]"

    print("\n" + "=" * 65)
    print("  V1 vs V2 COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<22}  {'v1':>12}  {'v2':>12}  {'Change':>20}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*20}")
    print(f"  {'Total Trades':<22}  {v1_total_trades:>12,}  {v2_total_trades:>12,}  {delta(v2_total_trades, v1_total_trades, higher_better=False)}")
    print(f"  {'Win Rate %':<22}  {v1_wr:>11.1f}%  {v2_wr:>11.1f}%  {delta(v2_wr, v1_wr)}")
    print(f"  {'Total P&L (Rs)':<22}  {v1_total_pnl:>12,.0f}  {v2_total_pnl:>12,.0f}  {delta(v2_total_pnl, v1_total_pnl)}")
    print(f"  {'ROI %':<22}  {v1_roi:>11.1f}%  {v2_roi:>11.1f}%  {delta(v2_roi, v1_roi)}")
    print(f"  {'Avg Win %':<22}  {v1_avg_win:>11.2f}%  {v2_avg_win:>11.2f}%  {delta(v2_avg_win, v1_avg_win)}")
    print(f"  {'Avg Loss %':<22}  {v1_avg_loss:>11.2f}%  {v2_avg_loss:>11.2f}%  {delta(v2_avg_loss, v1_avg_loss, higher_better=False)}")
    print(f"  {'Risk/Reward':<22}  {v1_rr:>12.2f}  {v2_rr:>12.2f}  {delta(v2_rr, v1_rr)}")
    print(f"  {'Stocks -ve PnL':<22}  {v1_neg:>12}  {v2_neg:>12}  {delta(v2_neg, v1_neg, higher_better=False)}")
    print("=" * 65)

    # Per-stock comparison for stocks in both versions
    merged = pd.merge(
        v1[['stock','total_trades','win_rate_pct','total_pnl_rs','avg_win_pct','avg_loss_pct']],
        v2[['stock','total_trades','win_rate_pct','total_pnl_rs','avg_win_pct','avg_loss_pct']],
        on='stock', suffixes=('_v1','_v2'), how='inner'
    )
    merged['pnl_delta']    = merged['total_pnl_rs_v2'] - merged['total_pnl_rs_v1']
    merged['wr_delta']     = merged['win_rate_pct_v2'] - merged['win_rate_pct_v1']
    merged['avgwin_delta'] = merged['avg_win_pct_v2']  - merged['avg_win_pct_v1']

    print(f"\n  TOP 10 MOST IMPROVED STOCKS (by P&L delta):")
    top_improved = merged.nlargest(10, 'pnl_delta')[
        ['stock','total_pnl_rs_v1','total_pnl_rs_v2','pnl_delta','avg_win_pct_v1','avg_win_pct_v2']
    ]
    print(top_improved.to_string(index=False))

    print(f"\n  TOP 10 MOST HURT STOCKS (by P&L delta):")
    top_hurt = merged.nsmallest(10, 'pnl_delta')[
        ['stock','total_pnl_rs_v1','total_pnl_rs_v2','pnl_delta','wr_delta']
    ]
    print(top_hurt.to_string(index=False))

    # Save comparison
    comp_path = RESULTS_DIR / "v1_vs_v2_comparison.csv"
    merged.to_csv(comp_path, index=False)
    print(f"\n  Full comparison saved: {comp_path}")


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
        if i % 20 == 0:
            print(f"  Processed {i}/{len(stocks)} stocks...")

    print(f"  Done. Total trades found: {len(all_trades)}")

    if not all_trades:
        print("\nNo trades found. Check data has enough bars.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values('entry_date', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)

    # Save trade log
    trade_log_path = RESULTS_DIR / "trade_log.csv"
    trades_df.to_csv(trade_log_path, index=False)
    print(f"\n  Trade log : {trade_log_path}  ({len(trades_df)} rows)")

    # Summary
    summary_df   = summarise(trades_df)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary   : {summary_path}")

    # Compare with v1
    compare_with_v1(trades_df, summary_df)

    print("\nDone. Check pulse_breaker/weekly/results/v2/")


if __name__ == "__main__":
    main()