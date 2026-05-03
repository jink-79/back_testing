"""
=============================================================
  GRID SEARCH OPTIMIZER — PULSE BREAKER v3 (WEEKLY)

  Runs parameter grid search across all weekly F&O stocks.

  Ranking priority:
    1) Profit Factor (higher better)
    2) Expectancy per trade (higher better)
    3) Max Drawdown (lower better)

  Output:
    combinations/results/all_combinations.csv
    combinations/results/top_combinations.csv
    combinations/results/summary_report.csv
    combinations/results/report.pdf

  Notes:
    - Unlimited simultaneous trades allowed (fast pass)
    - RS filter applied only at entry
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from datetime import datetime

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# backtesting folder is 4 levels up from your script location
# ...pulse_breaker/weekly/combinations/scripts/
BACKTESTING_ROOT = Path(r"D:\code\swing_trading\backtesting")
DATA_DIR = BACKTESTING_ROOT / "common" / "data" / "weekly"

PROJECT_ROOT = BACKTESTING_ROOT / "pulse_breaker" / "weekly"
RESULTS_DIR = PROJECT_ROOT / "combinations" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NIFTY_FILE = "NIFTY_weekly.csv"


# ──────────────────────────────────────────────
# STRATEGY CONSTANTS
# ──────────────────────────────────────────────
RS_PERIOD = 55
CAPITAL_PER_TRADE = 40_000


# ──────────────────────────────────────────────
# GRID SEARCH PARAMS
# ──────────────────────────────────────────────
ENTRY_BREAKOUT_LOOKBACK_LIST = [1, 2, 3]
EXIT_LOOKBACK_LIST           = [1, 2, 3, 4, 5]
RS_THRESHOLD_LIST            = [-999, 0, 5, 10]     # -999 = RS OFF
TRAIL_TRIGGER_LIST           = [2.0, 3.0, 5.0]
TRAIL_OFFSET_LIST            = [1.5, 2.0, 2.5, 3.0, 4.0]
RS_SLOPE_WEEKS_LIST          = [0, 2, 3]            # 0 = slope filter OFF


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def load_csv(filepath: Path, name: str):
    if not filepath.exists():
        print(f"  ERROR: File not found — {filepath}")
        return None

    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    return df


def load_all_weekly_data():
    print("\n--- Loading Weekly Data ---")

    nifty_path = DATA_DIR / NIFTY_FILE
    nifty = load_csv(nifty_path, "NIFTY")

    if nifty is None:
        raise FileNotFoundError(f"NIFTY file missing: {nifty_path}")

    stock_files = list(DATA_DIR.glob("*_weekly.csv"))
    stocks = {}

    for file in stock_files:
        if file.name.upper() == NIFTY_FILE.upper():
            continue

        stock_name = file.stem.replace("_weekly", "")
        df = load_csv(file, stock_name)

        if df is None:
            continue

        if len(df) < 60:
            continue

        stocks[stock_name] = df

    print(f"Loaded NIFTY: {len(nifty)} bars")
    print(f"Loaded Stocks: {len(stocks)}")

    return nifty, stocks


def calc_mansfield_rs(stock_df, nifty_df, period=55):
    combined = pd.DataFrame({
        "stock": stock_df["close"],
        "nifty": nifty_df["close"]
    }).dropna()

    raw_rs = combined["stock"] / combined["nifty"]
    raw_rs_prev = raw_rs.shift(period)
    mansfield_rs = ((raw_rs / raw_rs_prev) - 1) * 100

    return mansfield_rs


def run_strategy_one_stock(
    name,
    stock_df,
    nifty_df,
    breakout_lookback,
    exit_lookback,
    rs_threshold,
    trail_trigger_pct,
    trail_offset_pct,
    rs_slope_weeks
):
    """
    Returns list of trades (dicts).
    """

    df = stock_df.copy()

    # Mansfield RS
    mrs = calc_mansfield_rs(df, nifty_df, RS_PERIOD)
    df = df.join(mrs.rename("mansfield_rs"), how="left")

    # breakout prev high based on lookback
    df["prev_high"] = df["high"].rolling(breakout_lookback).max().shift(1)

    # exit structure break low
    df["exit_low"] = df["low"].rolling(exit_lookback).min().shift(1)

    # RS slope filter
    if rs_slope_weeks > 0:
        df["rs_slope"] = df["mansfield_rs"] - df["mansfield_rs"].shift(rs_slope_weeks)
    else:
        df["rs_slope"] = np.nan

    df.dropna(inplace=True)

    trades = []
    in_trade = False

    entry_price = None
    entry_date = None
    trail_active = False
    trail_stop = None
    highest_close = None
    shares = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]

        # EXIT LOGIC
        if in_trade:
            current_close = row["close"]

            if current_close > highest_close:
                highest_close = current_close

            if not trail_active:
                if current_close >= entry_price * (1 + trail_trigger_pct / 100):
                    trail_active = True

            if trail_active:
                trail_stop = highest_close * (1 - trail_offset_pct / 100)

            structure_exit = current_close < row["exit_low"]
            trail_exit = trail_active and (current_close < trail_stop)

            if structure_exit or trail_exit:
                exit_price = current_close
                exit_reason = "Structure Break" if structure_exit else "Trailing Stop"

                pnl_total = (exit_price - entry_price) * shares
                pnl_pct = (exit_price / entry_price - 1) * 100

                trades.append({
                    "stock": name,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl_rs": pnl_total,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "weeks_held": (date - entry_date).days // 7
                })

                in_trade = False
                entry_price = None
                entry_date = None
                trail_active = False
                trail_stop = None
                highest_close = None
                shares = None

        # ENTRY LOGIC
        if not in_trade:
            breakout_signal = row["close"] > row["prev_high"]

            # RS threshold filter
            if rs_threshold <= -999:
                rs_ok = True
            else:
                rs_ok = row["mansfield_rs"] > rs_threshold

            # RS slope filter
            if rs_slope_weeks == 0:
                slope_ok = True
            else:
                slope_ok = row["rs_slope"] > 0

            if breakout_signal and rs_ok and slope_ok:
                entry_price = row["close"]
                entry_date = date
                shares = int(CAPITAL_PER_TRADE / entry_price)
                if shares <= 0:
                    continue

                in_trade = True
                trail_active = False
                trail_stop = None
                highest_close = entry_price

    # Close open trade at end
    if in_trade:
        exit_price = df.iloc[-1]["close"]
        exit_date = df.index[-1]
        pnl_total = (exit_price - entry_price) * shares
        pnl_pct = (exit_price / entry_price - 1) * 100

        trades.append({
            "stock": name,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl_rs": pnl_total,
            "pnl_pct": pnl_pct,
            "exit_reason": "Open at End",
            "weeks_held": (exit_date - entry_date).days // 7
        })

    return trades


def compute_equity_curve_drawdown(trades_df):
    """
    Approx equity curve from sequential trades ordered by exit_date.
    Unlimited capital means we just sum PnL over time.
    """
    if trades_df.empty:
        return 0.0

    t = trades_df.sort_values("exit_date").copy()
    t["equity"] = t["pnl_rs"].cumsum()

    peak = t["equity"].cummax()
    dd = (t["equity"] - peak)

    max_dd = dd.min()  # negative value
    return float(max_dd)


def compute_metrics(trades_df):
    if trades_df.empty:
        return None

    total_trades = len(trades_df)
    winners = trades_df[trades_df["pnl_rs"] > 0]
    losers = trades_df[trades_df["pnl_rs"] <= 0]

    gross_profit = winners["pnl_rs"].sum()
    gross_loss = abs(losers["pnl_rs"].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    win_rate = len(winners) / total_trades * 100

    expectancy = trades_df["pnl_rs"].mean()

    avg_win = winners["pnl_rs"].mean() if len(winners) else 0
    avg_loss = losers["pnl_rs"].mean() if len(losers) else 0

    avg_win_pct = winners["pnl_pct"].mean() if len(winners) else 0
    avg_loss_pct = losers["pnl_pct"].mean() if len(losers) else 0

    total_pnl = trades_df["pnl_rs"].sum()

    max_dd = compute_equity_curve_drawdown(trades_df)

    avg_weeks = trades_df["weeks_held"].mean()

    return {
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy_rs": expectancy,
        "total_pnl_rs": total_pnl,
        "avg_win_rs": avg_win,
        "avg_loss_rs": avg_loss,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "max_drawdown_rs": max_dd,
        "avg_weeks_held": avg_weeks
    }


def generate_pdf_report(all_df, top_df, output_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    elements = []

    elements.append(Paragraph("Pulse Breaker v3 — Grid Search Optimization Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Ranking Priority:", styles["Heading2"]))
    elements.append(Paragraph("1) Profit Factor (higher better)", styles["Normal"]))
    elements.append(Paragraph("2) Expectancy per Trade (higher better)", styles["Normal"]))
    elements.append(Paragraph("3) Max Drawdown (lower better)", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Total combinations tested: {len(all_df)}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Top 20 Combinations", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    display_cols = [
        "breakout_lookback",
        "exit_lookback",
        "rs_threshold",
        "trail_trigger_pct",
        "trail_offset_pct",
        "rs_slope_weeks",
        "profit_factor",
        "expectancy_rs",
        "max_drawdown_rs",
        "win_rate_pct",
        "total_trades",
        "total_pnl_rs"
    ]

    top20 = top_df.head(20)[display_cols].copy()

    table_data = [display_cols] + top20.round(2).values.tolist()

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    elements.append(table)
    elements.append(PageBreak())

    elements.append(Paragraph("Interpretation Notes (For Beginners)", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "<b>Profit Factor</b>: Above 1.5 is good. Above 2 is excellent.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        "<b>Expectancy</b>: Average rupees earned per trade. Positive is required.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        "<b>Max Drawdown</b>: Worst peak-to-valley equity drop. Smaller magnitude is safer.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        "<b>RS Threshold</b>: Higher threshold means stricter outperformance filter.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        "<b>Trail Trigger</b>: How much profit required before trailing stop activates.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        "<b>Trail Offset</b>: Tight trail exits faster, loose trail holds longer.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    doc.build(elements)


# ──────────────────────────────────────────────
# MAIN GRID SEARCH
# ──────────────────────────────────────────────
def main():
    print("=" * 80)
    print("GRID SEARCH — PULSE BREAKER v3 (WEEKLY)")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"RESULTS_DIR : {RESULTS_DIR}")
    print("=" * 80)

    nifty, stocks = load_all_weekly_data()

    if len(stocks) == 0:
        print("ERROR: No stocks loaded. Check weekly data folder.")
        return

    all_combos = list(itertools.product(
        ENTRY_BREAKOUT_LOOKBACK_LIST,
        EXIT_LOOKBACK_LIST,
        RS_THRESHOLD_LIST,
        TRAIL_TRIGGER_LIST,
        TRAIL_OFFSET_LIST,
        RS_SLOPE_WEEKS_LIST
    ))

    print(f"\nTotal combinations to test: {len(all_combos)}")

    results = []
    combo_count = 0

    for (
        breakout_lookback,
        exit_lookback,
        rs_threshold,
        trail_trigger_pct,
        trail_offset_pct,
        rs_slope_weeks
    ) in all_combos:

        combo_count += 1
        print(f"Running combo {combo_count}/{len(all_combos)}...")

        all_trades = []

        for stock_name, stock_df in stocks.items():
            trades = run_strategy_one_stock(
                stock_name,
                stock_df,
                nifty,
                breakout_lookback,
                exit_lookback,
                rs_threshold,
                trail_trigger_pct,
                trail_offset_pct,
                rs_slope_weeks
            )
            all_trades.extend(trades)

        trades_df = pd.DataFrame(all_trades)

        metrics = compute_metrics(trades_df)
        if metrics is None:
            continue

        results.append({
            "breakout_lookback": breakout_lookback,
            "exit_lookback": exit_lookback,
            "rs_threshold": rs_threshold,
            "trail_trigger_pct": trail_trigger_pct,
            "trail_offset_pct": trail_offset_pct,
            "rs_slope_weeks": rs_slope_weeks,
            **metrics
        })

    results_df = pd.DataFrame(results)

    # Ranking: Profit Factor desc, Expectancy desc, Max Drawdown desc (less negative better)
    results_df.sort_values(
        by=["profit_factor", "expectancy_rs", "max_drawdown_rs"],
        ascending=[False, False, False],
        inplace=True
    )

    # Save all
    all_path = RESULTS_DIR / "all_combinations.csv"
    results_df.to_csv(all_path, index=False)

    # Top combos
    top_df = results_df.head(100).copy()
    top_path = RESULTS_DIR / "top_combinations.csv"
    top_df.to_csv(top_path, index=False)

    # Summary report (top 50)
    summary_df = results_df.head(50).copy()
    summary_path = RESULTS_DIR / "summary_report.csv"
    summary_df.to_csv(summary_path, index=False)

    # PDF Report
    pdf_path = RESULTS_DIR / "report.pdf"
    generate_pdf_report(results_df, top_df, pdf_path)

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"Saved: {all_path}")
    print(f"Saved: {top_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pdf_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()