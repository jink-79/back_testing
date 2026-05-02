"""
=============================================================
  SCRIPT 02 — PULSE BREAKER v3 STRATEGY
  
  What this script does:
    1. Loads weekly OHLCV CSVs from common/data/weekly/
    2. Calculates Mansfield RS (55-week) vs Nifty 50
    3. Detects breakout signals (weekly close > prev week high)
    4. Applies RS filter (RS must be strictly > 0)
    5. Simulates all trades with entry, exit, trailing stop
    6. Saves trade log to results/trade_log.csv
    7. Prints performance summary per stock + overall

  Entry  : Close of breakout candle (RS > 0, close > prev high)
  Exit 1 : Weekly close < lowest low of last 3 candles
  Exit 2 : Trailing stop — activates at +3% above entry,
            then trails at (highest close since entry) - 2%
  Capital: Rs 40,000 per trade, simultaneous trades allowed
  Re-entry: Allowed on any fresh valid signal

  Data path  : D:/code/swing_trading/backtesting/common/data/weekly/
  Output path: results/trade_log.csv
              results/summary.csv
=============================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

# Path to common weekly data folder (relative to this script's location)
# Script is at: pulse_breaker_strategy_backtest/02_strategy.py
# Data is at  : backtesting/common/data/weekly/
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "common" / "data" / "weekly"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# STRATEGY PARAMETERS
# ──────────────────────────────────────────────
RS_PERIOD           = 55        # Mansfield RS lookback (weeks)
BREAKOUT_LOOKBACK   = 1         # Close must be > high of previous N weeks
EXIT_LOOKBACK       = 3         # Exit if close < lowest low of last N candles
TRAIL_TRIGGER_PCT   = 3.0       # Trailing stop activates after +3% from entry
TRAIL_OFFSET_PCT    = 2.0       # Trail = highest close since entry - 2%
CAPITAL_PER_TRADE   = 40_000    # Rs per trade

# ──────────────────────────────────────────────
# STOCKS
# ──────────────────────────────────────────────
STOCKS = {
    "RELIANCE":  "RELIANCE_weekly.csv",
    "HDFCBANK":  "HDFCBANK_weekly.csv",
    "TCS":       "TCS_weekly.csv",
    "MM":        "MM_weekly.csv",
    "TATASTEEL": "TATASTEEL_weekly.csv",
}
NIFTY_FILE = "NIFTY_weekly.csv"


# ══════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════

def load_csv(filepath, name):
    """Load a weekly CSV and return a cleaned DataFrame."""
    if not filepath.exists():
        print(f"  ERROR: File not found — {filepath}")
        return None
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print(f"  Loaded {name}: {len(df)} weekly bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


def load_all():
    """Load Nifty and all stock CSVs."""
    print("\n--- Loading Data ---")

    nifty_path = DATA_DIR / NIFTY_FILE
    nifty = load_csv(nifty_path, "NIFTY")
    if nifty is None:
        raise FileNotFoundError(
            f"NIFTY file not found at {nifty_path}\n"
            "Run 01_fetch_data.py first."
        )

    stocks = {}
    for name, fname in STOCKS.items():
        path = DATA_DIR / fname
        df   = load_csv(path, name)
        if df is not None:
            stocks[name] = df

    print(f"\n  Stocks loaded: {list(stocks.keys())}")
    return nifty, stocks


# ══════════════════════════════════════════════
# STEP 2 — MANSFIELD RELATIVE STRENGTH
# ══════════════════════════════════════════════

def calc_mansfield_rs(stock_df, nifty_df, period=55):
    """
    Mansfield RS formula:
      Raw RS        = Stock Close / Nifty Close
      RS Ratio      = Raw RS / (RS 52-week ago)    [we use `period` weeks ago]
      Mansfield RS  = (RS Ratio - 1) * 100

    Returns a Series aligned to stock_df index.
    """
    # Align both series on common dates
    combined = pd.DataFrame({
        'stock': stock_df['close'],
        'nifty': nifty_df['close']
    }).dropna()

    raw_rs       = combined['stock'] / combined['nifty']
    raw_rs_prev  = raw_rs.shift(period)
    mansfield_rs = ((raw_rs / raw_rs_prev) - 1) * 100

    return mansfield_rs


# ══════════════════════════════════════════════
# STEP 3 — SIGNAL DETECTION + TRADE SIMULATION
# ══════════════════════════════════════════════

def run_strategy(name, stock_df, nifty_df):
    """
    Run Pulse Breaker v3 on one stock.
    Returns a list of trade dicts.
    """

    # ── Calculate indicators ──────────────────
    df = stock_df.copy()

    # Mansfield RS
    mrs = calc_mansfield_rs(df, nifty_df, RS_PERIOD)
    df  = df.join(mrs.rename('mansfield_rs'), how='left')

    # Previous week high (for breakout detection)
    df['prev_high'] = df['high'].shift(BREAKOUT_LOOKBACK)

    # Lowest low of last EXIT_LOOKBACK candles (rolling, not including current)
    # We use shift(1) so we compare against already-closed candles
    df['exit_low'] = df['low'].rolling(EXIT_LOOKBACK).min().shift(1)

    # Volume ratio vs 10-week average (observation only, not a filter)
    df['vol_avg_10']  = df['volume'].rolling(10).mean().shift(1)
    df['vol_ratio']   = df['volume'] / df['vol_avg_10']

    df.dropna(inplace=True)

    trades     = []
    in_trade   = False
    entry_price      = None
    entry_date       = None
    trail_active     = False
    trail_stop       = None
    highest_close    = None
    shares           = None

    for i in range(len(df)):
        row  = df.iloc[i]
        date = df.index[i]

        # ── CHECK EXIT (if in trade) ──────────
        if in_trade:
            current_close = row['close']

            # Update highest close for trailing stop
            if current_close > highest_close:
                highest_close = current_close

            # Check if trailing stop activates
            if not trail_active:
                if current_close >= entry_price * (1 + TRAIL_TRIGGER_PCT / 100):
                    trail_active = True

            # Update trailing stop level
            if trail_active:
                trail_stop = highest_close * (1 - TRAIL_OFFSET_PCT / 100)

            # Exit condition 1: Close below 3-week low (structure break)
            structure_exit = current_close < row['exit_low']

            # Exit condition 2: Trailing stop triggered
            trail_exit = trail_active and (current_close < trail_stop)

            if structure_exit or trail_exit:
                exit_price  = current_close
                exit_reason = "Structure Break" if structure_exit else "Trailing Stop"

                pnl_per_share = exit_price - entry_price
                pnl_total     = pnl_per_share * shares
                pnl_pct       = (exit_price / entry_price - 1) * 100

                trades.append({
                    "stock":          name,
                    "entry_date":     entry_date.date(),
                    "exit_date":      date.date(),
                    "entry_price":    round(entry_price, 2),
                    "exit_price":     round(exit_price, 2),
                    "shares":         shares,
                    "capital":        CAPITAL_PER_TRADE,
                    "pnl_rs":         round(pnl_total, 2),
                    "pnl_pct":        round(pnl_pct, 2),
                    "exit_reason":    exit_reason,
                    "trail_activated":trail_active,
                    "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2),
                    "vol_ratio_at_entry":    round(df.loc[entry_date, 'vol_ratio'], 2),
                    "weeks_held":     i - list(df.index).index(entry_date),
                })

                in_trade      = False
                entry_price   = None
                entry_date    = None
                trail_active  = False
                trail_stop    = None
                highest_close = None
                shares        = None

        # ── CHECK ENTRY (only if not in trade) ──
        if not in_trade:
            breakout_signal = row['close'] > row['prev_high']
            rs_signal       = row['mansfield_rs'] > 0

            if breakout_signal and rs_signal:
                entry_price   = row['close']
                entry_date    = date
                shares        = int(CAPITAL_PER_TRADE / entry_price)
                in_trade      = True
                trail_active  = False
                trail_stop    = None
                highest_close = entry_price

    # ── Close any open trade at last bar ─────
    if in_trade:
        last_row   = df.iloc[-1]
        exit_price = last_row['close']
        pnl_total  = (exit_price - entry_price) * shares
        pnl_pct    = (exit_price / entry_price - 1) * 100

        trades.append({
            "stock":          name,
            "entry_date":     entry_date.date(),
            "exit_date":      df.index[-1].date(),
            "entry_price":    round(entry_price, 2),
            "exit_price":     round(exit_price, 2),
            "shares":         shares,
            "capital":        CAPITAL_PER_TRADE,
            "pnl_rs":         round(pnl_total, 2),
            "pnl_pct":        round(pnl_pct, 2),
            "exit_reason":    "Open at End",
            "trail_activated":trail_active,
            "mansfield_rs_at_entry": round(df.loc[entry_date, 'mansfield_rs'], 2),
            "vol_ratio_at_entry":    round(df.loc[entry_date, 'vol_ratio'], 2),
            "weeks_held":     len(df) - 1 - list(df.index).index(entry_date),
        })

    return trades


# ══════════════════════════════════════════════
# STEP 4 — PERFORMANCE SUMMARY
# ══════════════════════════════════════════════

def summarise(trades_df):
    """Print per-stock and overall performance."""

    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY — PULSE BREAKER v3")
    print("=" * 65)

    summary_rows = []

    for stock in trades_df['stock'].unique():
        t = trades_df[trades_df['stock'] == stock]

        total_trades  = len(t)
        winners       = t[t['pnl_rs'] > 0]
        losers        = t[t['pnl_rs'] <= 0]
        win_rate      = len(winners) / total_trades * 100 if total_trades else 0
        total_pnl     = t['pnl_rs'].sum()
        avg_win       = winners['pnl_pct'].mean() if len(winners) else 0
        avg_loss      = losers['pnl_pct'].mean() if len(losers) else 0
        avg_weeks     = t['weeks_held'].mean()

        print(f"\n  {stock}")
        print(f"    Trades     : {total_trades}")
        print(f"    Win Rate   : {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
        print(f"    Total P&L  : Rs {total_pnl:,.0f}")
        print(f"    Avg Win    : {avg_win:.1f}%")
        print(f"    Avg Loss   : {avg_loss:.1f}%")
        print(f"    Avg Hold   : {avg_weeks:.1f} weeks")

        summary_rows.append({
            "stock":       stock,
            "total_trades":total_trades,
            "winners":     len(winners),
            "losers":      len(losers),
            "win_rate_pct":round(win_rate, 1),
            "total_pnl_rs":round(total_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct":round(avg_loss, 2),
            "avg_weeks_held": round(avg_weeks, 1),
        })

    # Overall
    print("\n" + "-" * 65)
    total_trades  = len(trades_df)
    total_pnl     = trades_df['pnl_rs'].sum()
    win_rate      = (trades_df['pnl_rs'] > 0).mean() * 100
    total_capital = CAPITAL_PER_TRADE * total_trades
    roi           = total_pnl / total_capital * 100 if total_capital else 0

    struct_exits  = (trades_df['exit_reason'] == 'Structure Break').sum()
    trail_exits   = (trades_df['exit_reason'] == 'Trailing Stop').sum()
    open_exits    = (trades_df['exit_reason'] == 'Open at End').sum()

    print(f"  OVERALL")
    print(f"    Total Trades : {total_trades}")
    print(f"    Win Rate     : {win_rate:.1f}%")
    print(f"    Total P&L    : Rs {total_pnl:,.0f}")
    print(f"    Total Capital: Rs {total_capital:,.0f}")
    print(f"    Overall ROI  : {roi:.1f}%")
    print(f"    Exit Reasons : Structure={struct_exits}  Trailing={trail_exits}  Open={open_exits}")
    print("=" * 65)

    return pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SCRIPT 02 — PULSE BREAKER v3 STRATEGY BACKTEST")
    print(f"  Data   : {DATA_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 65)

    # Load
    nifty, stocks = load_all()

    if not stocks:
        print("\nERROR: No stock data loaded. Check your data folder path.")
        return

    # Run strategy on each stock
    print("\n--- Running Strategy ---")
    all_trades = []

    for name, df in stocks.items():
        print(f"\n  Processing {name}...")
        trades = run_strategy(name, df, nifty)
        print(f"  Found {len(trades)} trades")
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades found. Check if data has enough bars (need >55 weeks).")
        return

    # Build trade log DataFrame
    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values('entry_date', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)

    # Save trade log
    trade_log_path = RESULTS_DIR / "trade_log.csv"
    trades_df.to_csv(trade_log_path, index=False)
    print(f"\n  Trade log saved: {trade_log_path}")

    # Print full trade log
    print("\n--- Trade Log ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(trades_df[[
        'stock', 'entry_date', 'exit_date', 'entry_price',
        'exit_price', 'pnl_rs', 'pnl_pct', 'exit_reason',
        'weeks_held', 'mansfield_rs_at_entry', 'vol_ratio_at_entry'
    ]].to_string(index=True))

    # Summary
    summary_df = summarise(trades_df)

    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved: {summary_path}")
    print("\nDone. Check results/ folder for CSV outputs.")


if __name__ == "__main__":
    main()