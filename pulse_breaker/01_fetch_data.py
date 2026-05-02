"""
=============================================================
  SCRIPT 01 — FETCH DATA
  Fetches daily OHLCV data for:
    - RELIANCE  (NSE)
    - HDFCBANK  (NSE)
    - TCS       (NSE)
    - M&M       (NSE)
    - TATASTEEL (NSE)
    - NIFTY 50  (NSE) — benchmark for Comparative Relative Strength

  Output → data/ folder:
    RELIANCE_daily.csv
    HDFCBANK_daily.csv
    TCS_daily.csv
    MM_daily.csv
    TATASTEEL_daily.csv
    NIFTY_daily.csv

  Requirements:
    pip install tvdatafeed pandas
=============================================================
"""

import pandas as pd
import os
from tvDatafeed import TvDatafeed, Interval

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Optional: Add your TradingView credentials for longer history
# Leave blank to use guest access (~100 bars only)
TV_USERNAME = ""
TV_PASSWORD = ""

# Number of daily bars to fetch (500 ≈ 2 years)
BARS = 500

# Output folder
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# STOCKS TO FETCH
# symbol        → TradingView symbol name
# exchange      → Exchange (NSE for Indian stocks)
# filename      → Output CSV filename
# ──────────────────────────────────────────────
STOCKS = [
    {"symbol": "RELIANCE",  "exchange": "NSE", "filename": "RELIANCE_daily.csv"},
    {"symbol": "HDFCBANK",  "exchange": "NSE", "filename": "HDFCBANK_daily.csv"},
    {"symbol": "TCS",       "exchange": "NSE", "filename": "TCS_daily.csv"},
    {"symbol": "M_M",       "exchange": "NSE", "filename": "MM_daily.csv"},       # M&M on TradingView
    {"symbol": "TATASTEEL", "exchange": "NSE", "filename": "TATASTEEL_daily.csv"},
    # Benchmark — required for Comparative Relative Strength
    {"symbol": "NIFTY",     "exchange": "NSE", "filename": "NIFTY_daily.csv"},
]


# ──────────────────────────────────────────────
# CONNECT TO TRADINGVIEW
# ──────────────────────────────────────────────
def connect():
    if TV_USERNAME and TV_PASSWORD:
        print(f"🔐 Logging in as {TV_USERNAME}...")
        tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
    else:
        print("👤 Using guest access...")
        print("   ⚠️  Guest gives ~100 bars only.")
        print("   💡 Add TV_USERNAME + TV_PASSWORD for full history.\n")
        tv = TvDatafeed()
    return tv


# ──────────────────────────────────────────────
# FETCH FUNCTION
# ──────────────────────────────────────────────
def fetch_and_save(tv, symbol, exchange, filename):
    print(f"📥 Fetching {symbol} from {exchange}...")

    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.in_daily,
            n_bars=BARS
        )

        if df is None or df.empty:
            print(f"  ❌ No data returned for {symbol}. Check symbol/exchange name.")
            return None

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Clean up index
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Save to CSV
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath)

        print(f"  ✅ {len(df)} bars fetched")
        print(f"  📅 Range  : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  💾 Saved  : {filepath}\n")

        return df

    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}\n")
        return None


# ──────────────────────────────────────────────
# QUICK SANITY CHECK
# ──────────────────────────────────────────────
def preview(df, name):
    print(f"--- {name} — Last 3 rows ---")
    print(df.tail(3).to_string())
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SCRIPT 01 — FETCH DATA")
    print("  Stocks : RELIANCE, HDFCBANK, TCS, M&M, TATASTEEL")
    print("  Index  : NIFTY 50 (CRS Benchmark)")
    print("=" * 55 + "\n")

    tv = connect()

    results = {}

    for stock in STOCKS:
        df = fetch_and_save(
            tv,
            symbol=stock["symbol"],
            exchange=stock["exchange"],
            filename=stock["filename"],
        )
        results[stock["symbol"]] = df

    # ── Summary ──────────────────────────────
    print("\n" + "=" * 55)
    print("  FETCH SUMMARY")
    print("=" * 55)

    success, failed = [], []

    for stock in STOCKS:
        sym = stock["symbol"]
        df  = results[sym]
        if df is not None:
            success.append(sym)
            preview(df, sym)
        else:
            failed.append(sym)

    print(f"✅ Fetched successfully : {', '.join(success) if success else 'None'}")
    if failed:
        print(f"❌ Failed              : {', '.join(failed)}")
        print("   → Double-check symbol names on TradingView")
    else:
        print("🎉 All 6 files saved in data/ folder.")
        print("   Next step: Run 02_strategy.py")

    print("=" * 55)


if __name__ == "__main__":
    main()