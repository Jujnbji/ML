"""
Plot candlesticks + volume for one coin from the downloaded Parquet files.
Opens an interactive chart in your browser (zoom, pan, range slider).

Usage examples
--------------
    python plot_coin.py SOLUSDT --start 2024-03-15 --end 2024-03-16
    python plot_coin.py ETHUSDT --interval 1m --start "2025-06-01 12:00" --end "2025-06-01 18:00"
    python plot_coin.py BTCUSDT --start 2024-01-01 --end 2024-02-01 --resample 1h

Requirements:  pip install pandas pyarrow plotly
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

MAX_CANDLES = 5000  



SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
START    = "2026-10-29 01:00"            # UTC, e.g. "2024-03-15" or "2024-03-15 14:00"; None = first candle
END      = "2026-11-01 22:00"            # None = last candle
RESAMPLE = "auto"                   # e.g. "15min", "1h", "1D"; None = keep original candles
DATA_DIR = Path(__file__).parent / "raw"   # the 'raw' folder next to this script
# ------------------------------------------------------


LADDER = [
    ("5min", "5min"), ("10min", "10min"), ("15min", "15min"), ("30min", "30min"),
    ("1h", "1h"), ("2h", "2h"), ("4h", "4h"), ("6h", "6h"), ("12h", "12h"),
    ("1D", "1D"), ("3D", "3D"), ("W-MON", "7D"),
]
# ------------------------------------------------------


def normalize_rule(rule):
    """'10m' -> '10min' (pandas reads 'm' as months). Leaves '10min', '1h', '1D' alone."""
    if rule and rule.endswith("m") and not rule.endswith("min"):
        return rule[:-1] + "min"
    return rule


def load(symbol, interval, data_dir, start, end) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}_{interval}.parquet"
    df = pd.read_parquet(path)
    start_ts = pd.Timestamp(start, tz="UTC") if start else df.index[0]
    end_ts = pd.Timestamp(end, tz="UTC") if end else df.index[-1]
    return df.loc[start_ts:end_ts]


def auto_rule(df, interval):
    """Smallest ladder rule that gives at most MAX_CANDLES bars. None = keep original."""
    base = pd.Timedelta(normalize_rule(interval))
    span = df.index[-1] - df.index[0] + base
    if span / base <= MAX_CANDLES:
        return None                                    # already small enough
    for rule, length in LADDER:
        step = pd.Timedelta(length)
        if step <= base or step % base:                # must be coarser and divide evenly
            continue
        if math.ceil(span / step) <= MAX_CANDLES:
            return rule
    raise SystemExit("Range too long even for weekly candles. Narrow START/END.")


def resample(df, rule, interval) -> pd.DataFrame:
    base = pd.Timedelta(normalize_rule(interval))
    if not rule.startswith("W"):
        step = pd.Timedelta(rule)
        if step % base:
            raise SystemExit(f"{rule} is not a multiple of the {interval} file interval.")
    # Index = candle OPEN time, so bins are [start, end) and labelled by their start.
    return df.resample(rule, closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])                         # drop empty bins (data gaps)


def plot(df, title) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="price"), row=1, col=1)
    colors = ["#26a69a" if c >= o else "#ef5350"
              for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors,
                         name="volume"), row=2, col=1)
    fig.update_layout(
        title=title, height=750, showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.06)),
    )
    fig.update_yaxes(title_text="price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="volume", row=2, col=1)
    fig.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default=SYMBOL)
    p.add_argument("--interval", default=INTERVAL, help="file interval (default 5m)")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--resample", default=RESAMPLE,
                   help='"auto" (default), or e.g. 10m, 1h, 1D, W-MON')
    p.add_argument("--data", default=DATA_DIR)
    args = p.parse_args()

    df = load(args.symbol, args.interval, args.data, args.start, args.end)
    if df.empty:
        raise SystemExit("No candles in that range. Check --start/--end.")

    rule = normalize_rule(args.resample)
    if rule == "auto":
        rule = auto_rule(df, args.interval)
    if rule:
        df = resample(df, rule, args.interval)

    if len(df) > MAX_CANDLES:
        raise SystemExit(f"{len(df):,} candles is too many to plot smoothly. "
                         f'Narrow the range or use --resample auto.')

    bar = rule or args.interval
    title = (f"{args.symbol}  |  {bar} candles  |  "
             f"{df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC  "
             f"({len(df):,} candles)")
    print(title)
    plot(df, title)


if __name__ == "__main__":
    main()