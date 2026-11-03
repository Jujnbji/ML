"""
Download historical candle (kline) data from Binance's public data archive
(data.binance.vision) and store one clean Parquet file per symbol.

No account or API key needed: these are public monthly ZIP archives.

Usage examples
--------------
    # 1-minute candles for the default universe, Jan 2021 to last full month
    python download_klines.py

    # Custom symbols, interval and range
    python download_klines.py --symbols SOLUSDT AVAXUSDT LINKUSDT \
        --interval 1m --start 2022-01 --end 2026-09

Requirements:  pip install pandas pyarrow requests tqdm
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Starter universe: liquid mid/large caps. BTC and ETH are included as
# market-wide context features, not necessarily as trading targets.
# NOTE: a fixed list chosen today has survivorship bias (coins that died
# are missing). Fine for prototyping; fix later with point-in-time universes.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT",                       # market context
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(m) for m in months]


def last_full_month() -> str:
    today = date.today()
    return str(pd.Period(today, freq="M") - 1)


def to_datetime_utc(ts: pd.Series) -> pd.Series:
    """Binance switched spot timestamps from milliseconds to microseconds
    starting 2025-01-01. Detect the unit per value by magnitude."""
    ts = ts.astype("int64")
    is_us = ts > 10**14  # ms timestamps are ~1.7e12, us are ~1.7e15
    ms = ts.where(~is_us, ts // 1000)
    return pd.to_datetime(ms, unit="ms", utc=True)


def fetch_month(session: requests.Session, symbol: str, interval: str,
                month: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    r = session.get(url, timeout=60)
    if r.status_code == 404:          # symbol not listed yet / delisted
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    # Some archives include a header row; drop it if present.
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():
        df = df.iloc[1:]
    df.columns = COLUMNS
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["ignore", "close_time"])
    df["open_time"] = to_datetime_utc(df["open_time"])
    num_cols = [c for c in df.columns if c != "open_time"]
    df[num_cols] = df[num_cols].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int64")
    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .set_index("open_time"))
    return df


def download_symbol(symbol: str, interval: str, months: list[str],
                    out_dir: Path, workers: int = 8) -> tuple[str, int]:
    out_path = out_dir / f"{symbol}_{interval}.parquet"
    frames = []
    with requests.Session() as session, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_month, session, symbol, interval, m): m
                   for m in months}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None:
                frames.append(df)
    if not frames:
        return symbol, 0
    data = clean(pd.concat(frames, ignore_index=True))
    data.to_parquet(out_path)
    return symbol, len(data)


def report_gaps(path: Path, interval: str) -> None:
    """Print how many candles are missing (exchange outages, listing date)."""
    df = pd.read_parquet(path)
    step = pd.Timedelta(interval.replace("m", "min"))
    expected = int((df.index[-1] - df.index[0]) / step) + 1
    missing = expected - len(df)
    print(f"  {path.stem}: {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}, "
          f"{len(df):,} candles, {missing:,} missing "
          f"({missing / expected:.2%})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--interval", default="1m",
                   help="1m, 3m, 5m, 15m, 30m, 1h, ... (default 1m)")
    p.add_argument("--start", default="2021-01", help="YYYY-MM")
    p.add_argument("--end", default=last_full_month(), help="YYYY-MM")
    p.add_argument("--out", default="data/raw", help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start, args.end)
    print(f"Downloading {len(args.symbols)} symbols, {args.interval}, "
          f"{months[0]} to {months[-1]} ({len(months)} months)")

    for symbol in tqdm(args.symbols, desc="symbols"):
        sym, n = download_symbol(symbol, args.interval, months, out_dir)
        if n == 0:
            tqdm.write(f"  {sym}: no data found (check the symbol name)")

    print("\nData quality summary:")
    for path in sorted(out_dir.glob(f"*_{args.interval}.parquet")):
        report_gaps(path, args.interval)


if __name__ == "__main__":
    main()