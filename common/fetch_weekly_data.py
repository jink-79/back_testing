"""
=============================================================
  SCRIPT 01 — FETCH DATA (WEEKLY)
  Fetches WEEKLY OHLCV data directly from TradingView for:
    - RELIANCE  (NSE)
    - HDFCBANK  (NSE)
    - TCS       (NSE)
    - M_M       (NSE)  <- M&M on TradingView
    - TATASTEEL (NSE)
    - NIFTY 50  (NSE) — benchmark for Mansfield RS (55-week)

  Output -> data/ folder:
    RELIANCE_weekly.csv
    HDFCBANK_weekly.csv
    TCS_weekly.csv
    MM_weekly.csv
    TATASTEEL_weekly.csv
    NIFTY_weekly.csv

  Why weekly directly?
    Strategy is weekly timeframe. Fetching weekly from TradingView
    is more accurate than resampling daily data ourselves.
    Each row = 1 complete trading week (Mon-Fri).

  Requirements:
    pip install tradingview-datafeed pandas
    OR
    pip install git+https://github.com/rongardF/tvdatafeed.git
=============================================================
"""

import pandas as pd
import os
from tvDatafeed import TvDatafeed, Interval

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Add your TradingView credentials for longer history (recommended)
# Guest access gives only ~100 bars — not enough for backtesting!
# 100 weekly bars = ~2 years only. With login you get 500+ weeks (~10 years)
TV_USERNAME = ""
TV_PASSWORD = ""

# Number of weekly bars to fetch
# 500 weeks = ~10 years of data — good for backtesting
BARS = 500

# Output folder
OUTPUT_DIR = "data/weekly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# STOCKS TO FETCH
# ──────────────────────────────────────────────
STOCKS = [
    {"symbol": "RELIANCE",  "exchange": "NSE", "filename": "RELIANCE_weekly.csv",  "label": "Reliance Industries"},
    {"symbol": "HDFCBANK",  "exchange": "NSE", "filename": "HDFCBANK_weekly.csv",  "label": "HDFC Bank"},
    {"symbol": "TCS",       "exchange": "NSE", "filename": "TCS_weekly.csv",       "label": "Tata Consultancy Services"},
    {"symbol": "M_M",       "exchange": "NSE", "filename": "MM_weekly.csv",        "label": "Mahindra & Mahindra"},
    {"symbol": "TATASTEEL", "exchange": "NSE", "filename": "TATASTEEL_weekly.csv", "label": "Tata Steel"},
    # Benchmark — MUST be fetched for Mansfield RS calculation
    {"symbol": "NIFTY",     "exchange": "NSE", "filename": "NIFTY_weekly.csv",     "label": "Nifty 50 (Benchmark)"},
]


# ──────────────────────────────────────────────
# CONNECT
# ──────────────────────────────────────────────
def connect():
    if TV_USERNAME and TV_PASSWORD:
        print(f"Logging in as {TV_USERNAME}...")
        tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
    else:
        print("Using guest access (no login)")
        print("  WARNING: Guest mode gives only ~100 weekly bars (~2 years).")
        print("  TIP: Add TV_USERNAME + TV_PASSWORD for 500 bars (~10 years).\n")
        tv = TvDatafeed()
    return tv


# ──────────────────────────────────────────────
# FETCH ONE STOCK
# ──────────────────────────────────────────────
def fetch_and_save(tv, symbol, exchange, filename, label):
    print(f"Fetching {label} ({symbol}) — Weekly...")

    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.in_weekly,   # WEEKLY directly from TradingView
            n_bars=BARS
        )

        if df is None or df.empty:
            print(f"  ERROR: No data returned for {symbol}.")
            print(f"         Double-check symbol name on TradingView.")
            return None

        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Clean index
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Save
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath)

        print(f"  OK    : {len(df)} weekly bars fetched")
        print(f"  Range : {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Saved : {filepath}\n")

        return df

    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}\n")
        return None


# ──────────────────────────────────────────────
# VALIDATE DATA QUALITY
# ──────────────────────────────────────────────
def validate(df, symbol):
    issues = []

    # Need at least 60 weeks for Mansfield RS (55) + warmup buffer
    if len(df) < 60:
        issues.append(f"Only {len(df)} bars — need at least 60 for Mansfield RS (55-week)")

    # Check for large gaps between weeks
    days_diff = df.index.to_series().diff().dt.days
    big_gaps = (days_diff > 10).sum()
    if big_gaps > 0:
        issues.append(f"{big_gaps} large gaps found (corporate actions / exchange halts)")

    # Zero volume rows
    zero_vol = (df['volume'] == 0).sum()
    if zero_vol > 0:
        issues.append(f"{zero_vol} weeks with zero volume")

    if issues:
        print(f"  Data quality issues for {symbol}:")
        for i in issues:
            print(f"    WARNING: {i}")
    else:
        print(f"  Data quality: OK for {symbol}")


# ──────────────────────────────────────────────
# PREVIEW
# ──────────────────────────────────────────────
def preview(df, label):
    print(f"\n--- {label} — Last 3 weekly bars ---")
    print(df.tail(3).to_string())
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  SCRIPT 01 — FETCH WEEKLY DATA")
    print("  Stocks : RELIANCE, HDFCBANK, TCS, M&M, TATASTEEL")
    print("  Index  : NIFTY 50 (Mansfield RS Benchmark)")
    print("  TF     : Weekly (1W) — directly from TradingView")
    print("=" * 58 + "\n")

    tv = connect()

    results = {}

    for stock in STOCKS:
        df = fetch_and_save(
            tv,
            symbol   = stock["symbol"],
            exchange = stock["exchange"],
            filename = stock["filename"],
            label    = stock["label"],
        )
        if df is not None:
            validate(df, stock["symbol"])
            print()
        results[stock["symbol"]] = df

    # ── Summary ──────────────────────────────────────
    print("\n" + "=" * 58)
    print("  FETCH SUMMARY")
    print("=" * 58)

    success, failed = [], []
    for stock in STOCKS:
        sym = stock["symbol"]
        df  = results[sym]
        if df is not None:
            success.append(sym)
            preview(df, stock["label"])
        else:
            failed.append(sym)

    print(f"Fetched OK : {', '.join(success) if success else 'None'}")

    if failed:
        print(f"Failed     : {', '.join(failed)}")
        print("  -> Check symbol names on TradingView search bar")
        print("  -> Common issue: M&M is M_M on TradingView")
    else:
        print("All 6 weekly files saved in data/ folder.")

    print("\nFiles saved:")
    for stock in STOCKS:
        if results[stock["symbol"]] is not None:
            bars = len(results[stock["symbol"]])
            print(f"  data/{stock['filename']}  ({bars} bars)")

    print("\nNext step: Run 02_strategy.py")
    print("=" * 58)


if __name__ == "__main__":
    main()