"""
Visualize a trained grid tokenizer (a run folder from train_simple_tokenizer.py).

Figure 1  Partition per feature: histogram of the feature with the grid values
          (solid) and bin edges (dashed); next to it, the mean forward return
          per bin on TRAIN vs VAL.
Figure 2  2D partition of two features: grid cells coloured by mean forward
          return, on TRAIN and on VAL (numbers = candles per cell).
Figure 3  Token usage (rank-frequency) and the most bullish / bearish tokens
          chosen on TRAIN, checked on VAL (does the effect survive?).
Figure 4  A short candle chart with the token id of every candle.

Requirements:  pip install pandas pyarrow numpy plotly
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tokenizer import GridTokenizer, compute_features

# ===================== SETTINGS: edit these =====================
HERE = Path(__file__).parent
RUN_DIR  = None                      # None = newest folder in tokenizer_runs/
DATA_DIR = HERE / "raw"

TRAIN_START = "2024-10-01"           # must match the training run
TRAIN_END   = "2026-06-30"
VAL_END     = "2026-08-31"

PAIR      = ("ret", "buy")           # two features for the 2D partition plot
MIN_COUNT = 500                      # min. train candles for a token in figure 3
TOP_N     = 10                       # bullish and bearish tokens to show

CHART_SYMBOL  = "BTCUSDT"            # figure 4
CHART_START   = "2026-07-15 14:00"   # UTC
CHART_CANDLES = 60
# ================================================================

AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
       "taker_buy_base_volume": "sum", "taker_buy_quote_volume": "sum"}
BLUE, ORANGE = "#42a5f5", "#ff9800"


# ---------------- loading ----------------
def find_run() -> Path:
    if RUN_DIR:
        return Path(RUN_DIR)
    runs = sorted((HERE / "tokenizer_runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Run train_simple_tokenizer.py first.")
    return runs[-1]


def load_coin(symbol, interval, resample) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_{interval}.parquet")
    df = df.loc[pd.Timestamp(TRAIN_START, tz="UTC"):]
    if resample:
        df = (df.resample(resample, closed="left", label="left")
                .agg({k: v for k, v in AGG.items() if k in df})
                .dropna(subset=["open"]))
    return df


def build_table(tok, symbols, interval, resample, horizon):
    """One row per candle: features, per-feature bin index, token, forward return, split."""
    t_end, v_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")
    rows, raw = [], {}
    for sym in symbols:
        df = load_coin(sym, interval, resample)
        raw[sym] = df
        feats = compute_features(df, tok.window)
        idx = tok.feature_indices(feats)
        d = feats[tok.features].copy()
        for j, f in enumerate(tok.features):
            d[f"i_{f}"] = idx[:, j]
        d["token"] = tok.encode(df).to_numpy()
        lc = np.log(df["close"])
        d["fwd"] = (lc.shift(-horizon) - lc) * 1e4
        d["split"] = np.where(d.index < t_end, "train", np.where(d.index < v_end, "val", "test"))
        d["coin"] = sym
        rows.append(d)
    d = pd.concat(rows)
    return d[(d["token"] >= 0) & d["fwd"].notna()], raw


def describe(tok, token) -> str:
    vals = tok.decode([token]).iloc[0]
    return ", ".join(f"{f} {vals[f]:+.2g}" for f in tok.features)


def mean_se(x: pd.Series):
    return x.mean(), x.std() / np.sqrt(max(len(x), 1))


# ---------------- figure 1 ----------------
def fig_partition(tok, d, title):
    n = len(tok.features)
    fig = make_subplots(rows=n, cols=2, column_widths=[0.6, 0.4],
                        horizontal_spacing=0.08, vertical_spacing=0.06,
                        subplot_titles=[s for f in tok.features for s in
                                        (f"{f}: distribution and grid",
                                         f"{f}: mean forward return per bin")])
    tr = d[d["split"] == "train"]
    for r, f in enumerate(tok.features, start=1):
        x = tr[f]
        lo, hi = x.quantile([0.005, 0.995])
        x = x[(x >= lo) & (x <= hi)]
        fig.add_trace(go.Histogram(x=x, nbinsx=150, histnorm="probability density",
                                   marker_color=BLUE, opacity=0.7), row=r, col=1)
        v = tok.grid[f]
        for val in v:
            fig.add_vline(x=val, line=dict(color=ORANGE, width=2), row=r, col=1)
        for e in (v[:-1] + v[1:]) / 2:
            fig.add_vline(x=e, line=dict(color="gray", dash="dash", width=1), row=r, col=1)

        labels = [f"{val:+.2g}" for val in v]
        for split, color in (("train", BLUE), ("val", ORANGE)):
            g = d[d["split"] == split].groupby(f"i_{f}")["fwd"]
            stats = g.agg(["mean", "std", "count"]).reindex(range(len(v)))
            fig.add_trace(go.Bar(
                x=labels, y=stats["mean"], name=split, marker_color=color,
                error_y=dict(type="data", array=1.96 * stats["std"] / np.sqrt(stats["count"])),
                showlegend=(r == 1)), row=r, col=2)
        fig.update_yaxes(title_text="bp", row=r, col=2)
    fig.update_layout(title=title, height=260 * n, barmode="group", bargap=0.2,
                      legend=dict(orientation="h", y=1.02, x=0.75))
    fig.show()


# ---------------- figure 2 ----------------
def fig_pair(tok, d, title):
    a, b = PAIR
    va, vb = tok.grid[a], tok.grid[b]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["TRAIN: mean forward return (bp)",
                                        "VAL: mean forward return (bp)"])
    for col, split in ((1, "train"), (2, "val")):
        g = d[d["split"] == split].groupby([f"i_{b}", f"i_{a}"])["fwd"].agg(["mean", "count"])
        z = np.full((len(vb), len(va)), np.nan)
        cnt = np.zeros((len(vb), len(va)), dtype=int)
        for (ib, ia), row in g.iterrows():
            z[ib, ia], cnt[ib, ia] = row["mean"], row["count"]
        lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
        fig.add_trace(go.Heatmap(
            z=z, x=[f"{v:+.2g}" for v in va], y=[f"{v:+.2g}" for v in vb],
            text=cnt, texttemplate="%{text}", colorscale="RdYlGn", zmid=0,
            zmin=-lim, zmax=lim, colorbar=dict(x=0.44 if col == 1 else 1.0, len=0.9)),
            row=1, col=col)
        fig.update_xaxes(title_text=f"{a} (grid value)", row=1, col=col)
        fig.update_yaxes(title_text=f"{b} (grid value)", row=1, col=col)
    fig.update_layout(title=title, height=520)
    fig.show()


# ---------------- figure 3 ----------------
def fig_tokens(tok, d, title):
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    counts = tr["token"].value_counts()
    st_tr = tr.groupby("token")["fwd"].agg(["mean", "std", "count"])
    st_va = va.groupby("token")["fwd"].agg(["mean", "std", "count"])
    eligible = st_tr[st_tr["count"] >= MIN_COUNT].sort_values("mean")
    chosen = pd.concat([eligible.head(TOP_N), eligible.tail(TOP_N)]).index

    fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65], horizontal_spacing=0.32,
                        subplot_titles=["Token usage (train)",
                                        f"Most bearish / bullish tokens chosen on train "
                                        f"(n ≥ {MIN_COUNT})"])
    fig.add_trace(go.Scatter(x=np.arange(1, len(counts) + 1), y=counts.values,
                             mode="lines", line=dict(color=BLUE)), row=1, col=1)
    fig.update_xaxes(type="log", title_text="token rank", row=1, col=1)
    fig.update_yaxes(type="log", title_text="candles", row=1, col=1)

    labels = [f"#{t}: {describe(tok, t)}" for t in chosen]
    for st, name, color in ((st_tr, "train", BLUE), (st_va, "val", ORANGE)):
        s = st.reindex(chosen)
        fig.add_trace(go.Bar(
            y=labels, x=s["mean"], orientation="h", name=name, marker_color=color,
            error_x=dict(type="data", array=1.96 * s["std"] / np.sqrt(s["count"]))),
            row=1, col=2)
    fig.update_xaxes(title_text="mean forward return (bp)", row=1, col=2)
    fig.update_layout(title=title, height=max(500, 45 * len(chosen)), barmode="group",
                      legend=dict(orientation="h", y=1.04, x=0.8))
    fig.show()


# ---------------- figure 4 ----------------
def fig_chart(tok, raw, title):
    df = raw[CHART_SYMBOL]
    tokens = tok.encode(df)
    start = pd.Timestamp(CHART_START, tz="UTC")
    w = df.loc[start:].iloc[:CHART_CANDLES]
    t = tokens.loc[w.index]
    hover = [describe(tok, x) if x >= 0 else "warm-up" for x in t]
    fig = go.Figure(go.Candlestick(x=w.index, open=w["open"], high=w["high"],
                                   low=w["low"], close=w["close"], name="price",
                                   text=hover, hoverinfo="x+text"))
    fig.add_trace(go.Scatter(x=w.index, y=w["high"] * 1.0008, mode="text",
                             text=[str(x) for x in t], textfont=dict(size=10),
                             hovertext=hover, hoverinfo="text"))
    fig.update_layout(title=title, height=550, showlegend=False,
                      xaxis_rangeslider_visible=False)
    fig.show()


def main():
    run = find_run()
    tok = GridTokenizer.load(run / "tokenizer.json")
    s = json.loads((run / "results.json").read_text())["settings"]
    print(f"Run {run.name}: features {tok.features}, vocabulary {tok.vocab_size}")
    d, raw = build_table(tok, s["symbols"], s["interval"], s["resample"], s["horizon"])
    bars = s["resample"] or s["interval"]
    head = f"{run.name}  |  {bars} candles  |  forward return over {s['horizon']} candles"
    fig_partition(tok, d, f"Figure 1: partition per feature  |  {head}")
    fig_pair(tok, d, f"Figure 2: {PAIR[0]} × {PAIR[1]} partition  |  {head}")
    fig_tokens(tok, d, f"Figure 3: tokens  |  {head}")
    if CHART_SYMBOL in raw:
        fig_chart(tok, raw, f"Figure 4: {CHART_SYMBOL} with token ids  |  {bars} candles")


if __name__ == "__main__":
    main()