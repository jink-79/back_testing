"""
=============================================================
  SCRIPT — FETCH DAILY DATA (ALL F&O STOCKS)

  Fetches DAILY OHLCV data directly from TradingView using tvDatafeed
  for all F&O symbols (NSE).

  Output -> data/daily/
    SYMBOL_daily.csv

  Also generates:
    data/daily/failed_symbols.txt

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
TV_USERNAME = ""
TV_PASSWORD = ""

BARS = 2000   # ~8 years of daily data (recommended for backtesting)

OUTPUT_DIR = "data/daily"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAILED_LOG_FILE = os.path.join(OUTPUT_DIR, "failed_symbols.txt")

# ──────────────────────────────────────────────
# F&O SYMBOL LIST
# ──────────────────────────────────────────────
FNO_SYMBOLS = [
    "MPHASIS","BAJAJFINSV","ONGC","BAJFINANCE","OIL","LODHA","COALINDIA","SHREECEM","NMDC",
    "360ONE","SAIL","CIPLA","WIPRO","NESTLEIND","MANAPPURAM","GMRAIRPORT","LUPIN","CAMS",
    "SAMMAANCAP","GLENMARK","COLPAL","LTM","HYUNDAI","LICHSGFIN","CGPOWER","ABB","GODREJCP",
    "SIEMENS","HINDUNILVR","ALKEM","ANGELONE","HEROMOTOCO","APOLLOHOSP","APLAPOLLO",
    "OBEROIRLTY","ICICIPRULI","INDHOTEL","FORTIS","AXISBANK","ICICIGI","IRFC","DMART",
    "INDUSTOWER","RECLTD","PAGEIND","DABUR","DIVISLAB","MANKIND","DRREDDY","DLF","VMM",
    "DALBHARAT","PATANJALI","PERSISTENT","PIDILITIND","POLICYBZR","KPITTECH","PHOENIXLTD",
    "BAJAJ-AUTO","PNBHOUSING","PIIND","POLYCAB","BOSCHLTD","BAJAJHLDNG","BLUESTARCO",
    "KALYANKJIL","BRITANNIA","NUVAMA","NBCC","JUBLFOOD","TATACONSUM","NAM-INDIA","NHPC",
    "NYKAA","TIINDIA","TATAELXSI","TATASTEEL","TITAN","UNOMINDA","TORNTPHARM","SUPREMEIND",
    "UPL","MARICO","MFSL","MARUTI","MOTHERSON","YESBANK","VBL","VOLTAS","VEDL","ZYDUSLIFE",
    "RBLBANK","MAZDOCK","BHEL","HDFCAMC","LICI","ITC","EXIDEIND","HAVELLS","IEX","CHOLAFIN",
    "JSWENERGY","TRENT","CUMMINSIND","POWERINDIA","BANKBARODA","SONACOMS","PFC","MCX",
    "HCLTECH","TCS","IDEA","SBICARD","INFY","AMBER","COFORGE","LT","BDL","BANDHANBNK",
    "HDFCLIFE","TECHM","HAL","JSWSTEEL","TMPV","AUBANK","BEL","IDFCFIRSTB","GAIL","IREDA",
    "FEDERALBNK","INOXWIND","SUNPHARMA","PRESTIGE","SBIN","PETRONET","GODREJPROP",
    "GODFRYPHLP","AUROPHARMA","PNB","JIOFIN","KEI","CONCOR","POWERGRID","SWIGGY",
    "MUTHOOTFIN","INDUSINDBK","DIXON","BANKINDIA","COCHINSHIP","RELIANCE","CANBK",
    "NATIONALUM","EICHERMOT","SBILIFE","SRF","OFSS","CDSL","NTPC","PAYTM","BIOCON",
    "KAYNES","ADANIPORTS","BHARATFORG","HINDALCO","BSE","ASIANPAINT","TATAPOWER",
    "ULTRACEMCO","JINDALSTEL","LAURUSLABS","ABCAPITAL","SUZLON","HINDZINC","MOTILALOFS",
    "LTF","IOC","ASTRAL","RVNL","MAXHEALTH","GRASIM","BPCL","ASHOKLEY","BHARTIARTL","M&M",
    "NAUKRI","SOLARINDS","ADANIPOWER","ADANIGREEN","ADANIENT","AMBUJACEM","PREMIERENE",
    "HDFCBANK","UNITDSPR","DELHIVERY","CROMPTON","INDIANB","ETERNAL","UNIONBANK",
    "SHRIRAMFIN","KOTAKBANK","ICICIBANK","PGEL","TVSMOTOR","HINDPETRO","INDIGO","FORCEMOT",
    "KFINTECH","WAAREEENER","ADANIENSOL"
]

# ──────────────────────────────────────────────
# TRADINGVIEW SYMBOL OVERRIDES
# ──────────────────────────────────────────────
TV_SYMBOL_OVERRIDES = {
    "M&M": "M_M",
}

# ──────────────────────────────────────────────
# FORMAT SYMBOL FOR TRADINGVIEW
# ──────────────────────────────────────────────
def format_tv_symbol(symbol: str) -> str:
    if symbol in TV_SYMBOL_OVERRIDES:
        return TV_SYMBOL_OVERRIDES[symbol]

    s = symbol.strip().upper()
    s = s.replace("-", "_")
    s = s.replace("&", "_")
    return s


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
# FETCH + SAVE
# ──────────────────────────────────────────────
def fetch_and_save(tv, original_symbol):
    tv_symbol = format_tv_symbol(original_symbol)

    print(f"📥 Fetching {original_symbol} (TV: {tv_symbol}) — Daily...")

    try:
        df = tv.get_hist(
            symbol=tv_symbol,
            exchange="NSE",
            interval=Interval.in_daily,
            n_bars=BARS
        )

        if df is None or df.empty:
            print(f"  ❌ No data returned for {original_symbol} (TV: {tv_symbol})\n")
            return False

        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Clean index
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Save
        filename = f"{original_symbol.replace('&','_').replace('-','_')}_daily.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        df.to_csv(filepath)

        print(f"  ✅ {len(df)} daily bars fetched")
        print(f"  📅 Range : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  💾 Saved : {filepath}\n")

        return True

    except Exception as e:
        print(f"  ❌ Error fetching {original_symbol} (TV: {tv_symbol}): {e}\n")
        return False


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SCRIPT — FETCH DAILY DATA (ALL F&O STOCKS)")
    print("  Exchange : NSE")
    print("  TF       : Daily (1D)")
    print(f"  Bars     : {BARS}")
    print("=" * 70 + "\n")

    tv = connect()

    failed = []
    success_count = 0

    for sym in FNO_SYMBOLS:
        ok = fetch_and_save(tv, sym)
        if ok:
            success_count += 1
        else:
            failed.append(sym)

    # Save failed symbols report
    if failed:
        with open(FAILED_LOG_FILE, "w") as f:
            for s in failed:
                f.write(s + "\n")

    # Summary
    print("\n" + "=" * 70)
    print("  FETCH SUMMARY")
    print("=" * 70)
    print(f"Total Symbols : {len(FNO_SYMBOLS)}")
    print(f"Fetched OK    : {success_count}")
    print(f"Failed        : {len(failed)}")

    if failed:
        print("\nFailed symbols saved in:")
        print(f"  {FAILED_LOG_FILE}")
        print("\nMost common reasons:")
        print("  - TradingView symbol mismatch")
        print("  - Newly listed stocks")
        print("  - Stock renamed / delisted / merged")
        print("  - Symbol requires override like M&M -> M_M")

    print("\nAll fetched CSV files saved in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()