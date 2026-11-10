"""
Download free 1-minute candles of other markets from Dukascopy (Swiss broker).
Works with dukascopy-python 4.x (fetch) and 5.x (fetch_candles). 5.0.0 is broken on import,
so install 4.0.1:   pip install "dukascopy-python==4.0.1"

    python download_markets.py --find    # list matching instrument names in the package
    python download_markets.py           # download everything in MARKETS

Saves <ML>/data/markets/<NAME>_1m.parquet (open, high, low, close, volume; UTC minute
index = candle start, like the Binance files). Works from any subfolder of the project.
Requirements: pip install "dukascopy-python==4.0.1" pandas pyarrow
"""

import argparse
import time
from datetime import timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "tokenizer.py").exists())
OUT = ROOT / "data" / "markets"

# our name -> Dukascopy instrument id (run --find if one is rejected)
MARKETS = {
    "SPX": "USA500.IDX/USD",       # S&P 500 index (CFD)
    "NDX": "USATECH.IDX/USD",      # Nasdaq 100 index (CFD)
    "XAU": "XAU/USD",              # gold
    "EURUSD": "EUR/USD",           # euro vs dollar (up = weaker dollar)
}
START, END = "2024-10-01", "2026-11-01"

try:
    import dukascopy_python as duka
except Exception as e:                                   # show the real reason
    raise SystemExit(f"Could not import dukascopy_python: {type(e).__name__}: {e}")


def find():
    import dukascopy_python.instruments as ins
    keys = ("USA500", "SNP", "USATECH", "NAS", "XAU", "EUR_USD")
    for k, v in vars(ins).items():
        if k.startswith(("I_", "INSTRUMENT")) and any(s in k for s in keys):
            print(f"{k} = {v!r}")


def fetch_month(instrument, a, b):
    """One month of 1-minute bid candles as a DataFrame, for either package version."""
    if hasattr(duka, "fetch_candles"):                                   # version 5
        candles = list(duka.fetch_candles(
            instrument=instrument, interval=1, time_unit=duka.TimeUnit.MINUTE,
            offer_side=duka.OfferSide.BID, start=a, end=b, validate_instrument_name=False))
        return duka.to_dataframe(candles) if candles else pd.DataFrame()
    return duka.fetch(instrument, duka.INTERVAL_MIN_1, duka.OFFER_SIDE_BID, a, b)   # version 4


def download(instrument):
    frames = []
    for month in pd.date_range(START, END, freq="MS", tz="UTC"):
        a = month.to_pydatetime()
        b = min(month + pd.offsets.MonthBegin(1), pd.Timestamp(END, tz="UTC")).to_pydatetime()
        for attempt in range(3):
            try:
                df = fetch_month(instrument, a, b)
                break
            except Exception as e:                       # network hiccup: retry twice
                if attempt == 2:
                    raise
                print(f"    retry {month:%Y-%m} ({type(e).__name__}: {e})")
                time.sleep(5)
        print(f"    {month:%Y-%m}: {len(df):,} candles")
        if len(df):
            frames.append(df)
    df = pd.concat(frames)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df.columns:                        # time may come as a column or the index
        df = df.set_index("timestamp")
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.tz_convert(timezone.utc) if idx.tz is not None else idx
    df.index.name = "open_time"
    df = df[~df.index.duplicated()].sort_index()
    return df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--find", action="store_true", help="list matching instrument names and exit")
    if p.parse_args().find:
        find()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, instrument in MARKETS.items():
        print(f"{name} ({instrument})")
        df = download(instrument)
        df.to_parquet(OUT / f"{name}_1m.parquet")
        print(f"  saved {len(df):,} candles, {df.index[0]} .. {df.index[-1]} -> {OUT}")


if __name__ == "__main__":
    main()