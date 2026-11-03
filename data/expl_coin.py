"""
Exploratory plots for one coin.

Figure 1 - Market chart (time on the x-axis):
    row 1: candles + daily VWAP, with a VOLUME PROFILE on the right
           (how much was traded at each price level, split into
           aggressive buys / aggressive sells)
    row 2: volume per candle
    row 3: order-flow imbalance  = taker-buy share - 0.5
           (> 0: buyers were pushing, < 0: sellers were pushing)
    row 4: rolling volatility (std of log returns, in %)

Figure 2 - Statistics ("stylized facts"), computed on the raw candles:
    1. return distribution vs. a Gaussian (fat tails)
    2. |return| vs. volume (do big moves come with big volume?)
    3. autocorrelation of returns and |returns| (predictability / vol clustering)
    4. heatmap of average volume by hour of day x weekday (seasonality)

Requirements:  pip install pandas pyarrow plotly numpy
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- SETTINGS: edit these ----------------
SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
START    = "2026-10-01"            # UTC; None = first candle
END      = "2026-11-01"            # None = last candle
RESAMPLE = "auto"                  # "auto", None, or e.g. "15min", "1h", "1D"
DATA_DIR = Path(__file__).parent / "raw"

SHOW_CHART   = True
SHOW_STATS   = True                # most meaningful over weeks/months of data
PROFILE_BINS = 80                  # price levels in the volume profile
VOL_WINDOW   = 20                  # candles in the rolling volatility window
ACF_LAGS     = 120                 # lags (in raw candles) for autocorrelation

SHOW_FLOW    = True                # figure 3: direction vs. volume and order flow
FLOW_HORIZON = 5                   # raw candles ahead for the predictive test
FLOW_BINS    = 10                  # buy-share bins (deciles)
FEE_BPS      = 20                  # round-trip taker fees in basis points (0.2 %)
# ------------------------------------------------------

MAX_CANDLES = 5000
LADDER = [
    ("5min", "5min"), ("10min", "10min"), ("15min", "15min"), ("30min", "30min"),
    ("1h", "1h"), ("2h", "2h"), ("4h", "4h"), ("6h", "6h"), ("12h", "12h"),
    ("1D", "1D"), ("3D", "3D"), ("W-MON", "7D"),
]
AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
    "taker_buy_base_volume": "sum", "taker_buy_quote_volume": "sum",
}
GREEN, RED, ORANGE, BLUE = "#26a69a", "#ef5350", "#ff9800", "#42a5f5"


# ---------------- loading & resampling ----------------
def normalize_rule(rule):
    if rule and rule.endswith("m") and not rule.endswith("min"):
        return rule[:-1] + "min"
    return rule


def load(symbol, interval, data_dir, start, end) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_dir) / f"{symbol}_{interval}.parquet")
    start_ts = pd.Timestamp(start, tz="UTC") if start else df.index[0]
    end_ts = pd.Timestamp(end, tz="UTC") if end else df.index[-1]
    return df.loc[start_ts:end_ts]


def auto_rule(df, interval):
    base = pd.Timedelta(normalize_rule(interval))
    span = df.index[-1] - df.index[0] + base
    if span / base <= MAX_CANDLES:
        return None
    for rule, length in LADDER:
        step = pd.Timedelta(length)
        if step <= base or step % base:
            continue
        if math.ceil(span / step) <= MAX_CANDLES:
            return rule
    raise SystemExit("Range too long even for weekly candles. Narrow START/END.")


def resample(df, rule, interval) -> pd.DataFrame:
    base = pd.Timedelta(normalize_rule(interval))
    if not rule.startswith("W") and pd.Timedelta(rule) % base:
        raise SystemExit(f"{rule} is not a multiple of the {interval} file interval.")
    return (df.resample(rule, closed="left", label="left")
              .agg(AGG).dropna(subset=["open"]))


# ---------------- derived metrics ----------------
def add_features(df) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["volatility_pct"] = df["log_ret"].rolling(VOL_WINDOW).std() * 100
    df["buy_share"] = df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)

    # VWAP anchored at each UTC midnight (the reference price many traders watch)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("1D")
    df["vwap"] = ((typical * df["volume"]).groupby(day).cumsum()
                  / df["volume"].groupby(day).cumsum())
    return df


def volume_profile(df, bins):
    """Spread each candle's volume evenly over its [low, high] range and sum
    per price bin. Returns bin centers, buy volume, sell volume (in USDT)."""
    edges = np.linspace(df["low"].min(), df["high"].max(), bins + 1)
    lo = df["low"].to_numpy()[:, None]
    hi = df["high"].to_numpy()[:, None]
    overlap = np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0, None)
    width = hi - lo
    weights = np.divide(overlap, width, out=np.zeros_like(overlap), where=width > 0)

    flat = np.flatnonzero(width[:, 0] == 0)        # candles with high == low
    if flat.size:
        idx = np.searchsorted(edges, df["close"].to_numpy()[flat], side="right") - 1
        weights[flat, np.clip(idx, 0, bins - 1)] = 1.0

    buy = weights.T @ df["taker_buy_quote_volume"].to_numpy()
    total = weights.T @ df["quote_volume"].to_numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, buy, total - buy


# ---------------- figure 1: market chart ----------------
def plot_chart(df, title) -> None:
    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, shared_yaxes=True,
        column_widths=[0.86, 0.14], row_heights=[0.55, 0.15, 0.15, 0.15],
        horizontal_spacing=0.01, vertical_spacing=0.025,
        specs=[[{}, {}], [{}, None], [{}, None], [{}, None]],
    )

    # row 1: candles + VWAP
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"],
        close=df["close"], name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], name="daily VWAP",
                             line=dict(color=ORANGE, width=1.2)), row=1, col=1)

    # row 1, right: volume profile
    centers, buy, sell = volume_profile(df, PROFILE_BINS)
    bar_h = centers[1] - centers[0]
    fig.add_trace(go.Bar(x=buy, y=centers, orientation="h", width=bar_h,
                         marker_color=GREEN, name="taker buys"), row=1, col=2)
    fig.add_trace(go.Bar(x=sell, y=centers, orientation="h", width=bar_h,
                         marker_color=RED, name="taker sells"), row=1, col=2)
    poc = centers[np.argmax(buy + sell)]           # "point of control"
    fig.add_hline(y=poc, line=dict(color=ORANGE, dash="dot", width=1),
                  annotation_text="POC", annotation_position="top left",
                  row=1, col=1)

    # row 2: volume
    up = df["close"] >= df["open"]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="volume",
                         marker_color=np.where(up, GREEN, RED)), row=2, col=1)

    # row 3: order-flow imbalance
    imb = df["buy_share"] - 0.5
    fig.add_trace(go.Bar(x=df.index, y=imb, name="buy share - 0.5",
                         marker_color=np.where(imb >= 0, GREEN, RED)), row=3, col=1)

    # row 4: volatility
    fig.add_trace(go.Scatter(x=df.index, y=df["volatility_pct"], name="volatility",
                             line=dict(color=BLUE, width=1.2)), row=4, col=1)

    fig.update_layout(title=title, height=1000, showlegend=False,
                      barmode="stack", bargap=0)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="price (USDT)", row=1, col=1)
    fig.update_xaxes(title_text="USDT traded", row=1, col=2)
    fig.update_yaxes(title_text="volume", row=2, col=1)
    fig.update_yaxes(title_text="buy share − 0.5", row=3, col=1)
    fig.update_yaxes(title_text=f"vol % ({VOL_WINDOW} bars)", row=4, col=1)
    fig.show()


# ---------------- figure 2: statistics ----------------
def plot_stats(raw, title) -> None:
    r = np.log(raw["close"]).diff().dropna()
    r = r[np.isfinite(r)]
    z = (r - r.mean()) / r.std()

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, subplot_titles=[
        f"Return distribution (excess kurtosis = {r.kurt():.1f}; Gaussian = 0)",
        "|return| vs. volume",
        "Autocorrelation",
        "Mean volume by hour (UTC) × weekday, relative to average",
    ])

    # 1) return histogram vs Gaussian, log scale shows the tails
    dens, edges = np.histogram(z, bins=200, range=(-12, 12), density=True)
    x = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Scatter(x=x, y=np.where(dens > 0, dens, np.nan), mode="markers",
                             marker=dict(size=4, color=BLUE), name="data"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
                             line=dict(color=ORANGE), name="Gaussian"), row=1, col=1)
    fig.update_yaxes(type="log", range=[-6, 0], title_text="density", row=1, col=1)
    fig.update_xaxes(title_text="standardized return (σ)", row=1, col=1)

    # 2) |return| vs volume, log-log
    s = pd.DataFrame({"absret": r.abs(), "qv": raw["quote_volume"].reindex(r.index)})
    s = s[(s["absret"] > 0) & (s["qv"] > 0)]
    if len(s) > 20000:
        s = s.sample(20000, random_state=0)
    corr = np.corrcoef(np.log(s["absret"]), np.log(s["qv"]))[0, 1]
    fig.add_trace(go.Scattergl(x=s["qv"], y=s["absret"] * 100, mode="markers",
                               marker=dict(size=3, color=BLUE, opacity=0.3),
                               name=f"corr(log,log) = {corr:.2f}"), row=1, col=2)
    fig.update_xaxes(type="log", title_text="USDT volume per candle", row=1, col=2)
    fig.update_yaxes(type="log", title_text=f"|return| %   (log-log corr = {corr:.2f})",
                     row=1, col=2)

    # 3) autocorrelation of returns and |returns|
    lags = np.arange(1, ACF_LAGS + 1)
    band = 1.96 / np.sqrt(len(r))
    fig.add_trace(go.Scatter(x=lags, y=[r.autocorr(l) for l in lags], name="returns",
                             line=dict(color=BLUE)), row=2, col=1)
    fig.add_trace(go.Scatter(x=lags, y=[r.abs().autocorr(l) for l in lags],
                             name="|returns|", line=dict(color=ORANGE)), row=2, col=1)
    for y in (band, -band):
        fig.add_hline(y=y, line=dict(color="gray", dash="dash", width=1), row=2, col=1)
    fig.update_xaxes(title_text=f"lag (× {INTERVAL})", row=2, col=1)
    fig.update_yaxes(title_text="autocorrelation (blue: returns, orange: |returns|)",
                     row=2, col=1)

    # 4) intraday / weekly seasonality
    qv = raw["quote_volume"]
    grid = (qv.groupby([qv.index.dayofweek, qv.index.hour]).mean()
              .unstack() / qv.mean())
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.add_trace(go.Heatmap(z=grid.values, x=grid.columns,
                             y=[days[d] for d in grid.index], colorscale="Viridis",
                             colorbar=dict(title="× avg", len=0.4, y=0.2)), row=2, col=2)
    fig.update_xaxes(title_text="hour of day (UTC)", row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)

    span_days = (raw.index[-1] - raw.index[0]).days
    note = "  ⚠ short range, statistics are noisy" if span_days < 14 else ""
    fig.update_layout(title=title + note, height=900, showlegend=False)
    fig.show()


# ---------------- figure 3: direction & order flow ----------------
def plot_flow(raw, title) -> None:
    """Signed returns vs. volume, coloured by who was aggressive, plus the
    key test: does the buy share of a candle predict the NEXT returns?
    Returns are in basis points (1 bp = 0.01 %)."""
    lc = np.log(raw["close"])
    df = pd.DataFrame({
        "ret": lc.diff() * 1e4,                                # this candle
        "fwd": (lc.shift(-FLOW_HORIZON) - lc) * 1e4,            # next H candles
        "qv": raw["quote_volume"],
        "buy": raw["taker_buy_base_volume"] / raw["volume"].replace(0, np.nan),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    fig = make_subplots(rows=1, cols=3, column_widths=[0.42, 0.29, 0.29],
                        horizontal_spacing=0.08, subplot_titles=[
        "Signed return vs. volume (colour = taker-buy share)",
        "Same candle: return vs. buy share",
        f"Next {FLOW_HORIZON} candles: return vs. buy share",
    ])

    # 1) signed return vs volume
    s = df.sample(min(len(df), 20000), random_state=0)
    lo, hi = df["ret"].quantile([0.001, 0.999])
    fig.add_trace(go.Scattergl(
        x=s["qv"], y=s["ret"], mode="markers",
        marker=dict(size=3, opacity=0.5, color=s["buy"], colorscale="RdYlGn",
                    cmin=0.2, cmax=0.8,
                    colorbar=dict(title="buy<br>share", x=0.4, len=0.9))),
        row=1, col=1)
    fig.update_xaxes(type="log", title_text="USDT volume per candle", row=1, col=1)
    fig.update_yaxes(range=[lo, hi], title_text="return (bp)", row=1, col=1)

    # 2) + 3) binned means with 95 % error bars
    df["bin"] = pd.qcut(df["buy"], FLOW_BINS, duplicates="drop")
    panels = [
        (2, df, "ret"),
        # non-overlapping windows, otherwise the error bars are too small
        (3, df.iloc[::FLOW_HORIZON], "fwd"),
    ]
    for col, data, target in panels:
        g = data.groupby("bin", observed=True)
        fig.add_trace(go.Scatter(
            x=g["buy"].mean(), y=g[target].mean(), mode="markers+lines",
            error_y=dict(type="data", array=1.96 * g[target].sem()),
            line=dict(color=BLUE)), row=1, col=col)
        fig.add_hline(y=0, line=dict(color="gray", width=1), row=1, col=col)
        fig.update_xaxes(title_text="taker-buy share of candle", row=1, col=col)
        fig.update_yaxes(title_text="mean return (bp)", row=1, col=col)

    for y in (FEE_BPS, -FEE_BPS):
        fig.add_hline(y=y, line=dict(color=RED, dash="dash", width=1),
                      annotation_text="round-trip fees", row=1, col=3)

    fig.update_layout(title=title, height=550, showlegend=False)
    fig.show()


# ---------------- main ----------------
def main() -> None:
    raw = load(SYMBOL, INTERVAL, DATA_DIR, START, END)
    if raw.empty:
        raise SystemExit("No candles in that range. Check START/END.")
    period = f"{raw.index[0]:%Y-%m-%d %H:%M} → {raw.index[-1]:%Y-%m-%d %H:%M} UTC"

    if SHOW_STATS:
        plot_stats(raw, f"{SYMBOL}  |  statistics on {len(raw):,} {INTERVAL} candles  |  {period}")

    if SHOW_FLOW:
        plot_flow(raw, f"{SYMBOL}  |  direction & order flow on {INTERVAL} candles  |  {period}")

    if SHOW_CHART:
        rule = normalize_rule(RESAMPLE)
        if rule == "auto":
            rule = auto_rule(raw, INTERVAL)
        df = resample(raw, rule, INTERVAL) if rule else raw
        if len(df) > MAX_CANDLES:
            raise SystemExit(f"{len(df):,} candles is too many. Narrow the range "
                             f'or set RESAMPLE = "auto".')
        df = add_features(df)
        plot_chart(df, f"{SYMBOL}  |  {rule or INTERVAL} candles  |  {period}  ({len(df):,} candles)")


if __name__ == "__main__":
    main()