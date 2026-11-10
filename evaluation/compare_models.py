"""
Compare several trained models on exactly the same validation windows.

Metrics (one-step-ahead: each prediction sees the real history):
  d_bits     paired difference in bits/token vs. the baseline (mbit, below 0 = better)
  bits@1/64  bits/token with 1 and with 64 candles of context
  AUC, d_AUC next-candle direction; d_AUC is paired against the baseline
  IC, d_IC   rank correlation of expected and real return; d_IC paired
  d_QLIKE    one-step variance forecasts summed over 15-candle blocks vs. EWMA (below 0 = better)
Curves (figure 2): ROC with log x-axis, and hit rate vs. fraction of the list traded.

Tokens are decoded with their EMPIRICAL meaning, measured on training data before
VAL_START (mean return, mean squared return, share of up candles per token), not with
the grid values, so the grid's rounding doesn't bias the forecasts.
95 % confidence intervals from a block bootstrap over coin-days.

Run:  python evaluation/compare_models.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "tokenizer.py").exists())
sys.path[:0] = [str(ROOT / "data"), str(next(ROOT.rglob("GPT1_1_more_input.py")).parent)]
from tokenizer import GridTokenizer, compute_features
from GPT1_1_more_input import ContextGPT

# ============================== SETTINGS ==============================
RUN_PATTERNS = ["ctx_*.pt"]          # which checkpoints to compare (searched in **/runs/)
BASELINE     = None                  # file name; None = a ctx_none_* run, else newest gpt_*.pt
INTERVAL     = "1m"
VAL_START, VAL_END = "2026-06-30", "2026-08-31"
STATS_DAYS   = 120                   # training days before VAL_START used for token statistics
VOL_BLOCK    = 15
EWMA_LAMBDA  = 0.94
N_BOOT       = 100
SEED         = 0
# ======================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
LN2 = np.log(2)
COVERAGE = np.logspace(-3, 0, 25)    # fractions of the list for the hit-rate curves


# ---------------- models ----------------
def find_runs():
    runs = sorted({p for pat in RUN_PATTERNS for p in ROOT.glob(f"**/runs/{pat}")})
    if BASELINE:
        base = next(ROOT.glob(f"**/runs/{BASELINE}"))
    else:
        none_runs = [p for p in runs if p.name.startswith("ctx_none")]
        base = none_runs[0] if none_runs else sorted(ROOT.glob("**/runs/gpt_*.pt"))[-1]
    return [base] + [p for p in runs if p != base]


def load(path):
    ckpt = torch.load(path, map_location=device)
    s = ckpt["settings"]
    sw = s.get("switches", {})
    run_name = re.split(r"[\\/]", ckpt["tokenizer_run"].rstrip("\\/"))[-1]
    tok = GridTokenizer.load(ROOT / "data" / "tokenizer_runs" / run_name / "tokenizer.json")
    model = ContextGPT(tok.vocab_size, s["embed_dim"], s["context"], s["n_heads"], s["n_layers"],
                       n_coins=len(s["symbols"]), use_time=sw.get("time", False),
                       use_weekday=sw.get("weekday", False), use_coin=sw.get("coin", False),
                       use_dailyvol=sw.get("daily_vol", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, s, ckpt, run_name


# ---------------- what each token really means ----------------
def token_stats(tok, symbols, min_count=50, n_samples=256):
    """On training data before VAL_START, per token: mean z, mean z^2, share z > 0,
    and a sample of real z values (z = return / sigma, the unrounded 'ret' feature)."""
    v0 = pd.Timestamp(VAL_START, tz="UTC")
    zs, ts = [], []
    for sym in symbols:
        df = pd.read_parquet(ROOT / "data" / "raw" / f"{sym}_{INTERVAL}.parquet")
        df = df.loc[v0 - pd.Timedelta(days=STATS_DAYS): v0 - pd.Timedelta(minutes=1)]
        z = compute_features(df, tok.window)["ret"].to_numpy()
        t = tok.encode(df).to_numpy()
        ok = (t >= 0) & np.isfinite(z)
        zs.append(z[ok])
        ts.append(t[ok])
    z, t = np.concatenate(zs), np.concatenate(ts)
    V = tok.vocab_size
    grid = tok.decode(np.arange(V))["ret"].to_numpy()
    n = np.bincount(t, minlength=V)
    enough = n >= min_count
    safe_n = np.maximum(n, 1)
    stats = {
        "m1": np.where(enough, np.bincount(t, z, V) / safe_n, grid),
        "m2": np.where(enough, np.bincount(t, z ** 2, V) / safe_n, grid ** 2),
        "pu": np.where(enough, np.bincount(t, (z > 0).astype(float), V) / safe_n, (grid > 0) * 1.0),
    }
    order = np.argsort(t, kind="stable")
    groups = np.split(z[order], np.cumsum(n)[:-1])
    stats["samples"] = np.stack([rng.choice(g, n_samples) if enough[j] else np.full(n_samples, grid[j])
                                 for j, g in enumerate(groups)])          # (V, n_samples)
    return stats


# ---------------- validation data (built once) ----------------
def build_val(tok, symbols, T):
    W = tok.window
    one = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
    v0, v1 = pd.Timestamp(VAL_START, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")
    keys = ("tok", "slot", "day", "coin", "dvol", "sigma", "ewma", "ret", "daykey")
    d = {k: [] for k in keys}
    starts, n = [], 0
    for c, sym in enumerate(symbols):
        df = pd.read_parquet(ROOT / "data" / "raw" / f"{sym}_{INTERVAL}.parquet")
        df = df.loc[v0 - pd.Timedelta(days=3): v1]
        lr = np.log(df["close"]).diff()
        col = {
            "tok": tok.encode(df).to_numpy(),
            "slot": ((df.index.hour * 60 + df.index.minute) // 15).to_numpy(),
            "day": df.index.dayofweek.to_numpy(),
            "coin": np.full(len(df), c),
            "dvol": np.log(lr.rolling(1440, min_periods=720).std())
                      .replace([np.inf, -np.inf], np.nan).to_numpy(),
            "sigma": lr.rolling(W).std().shift(1).to_numpy(),
            "ewma": (lr ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().shift(1).to_numpy(),
            "ret": lr.to_numpy(),
            "daykey": c * 100000 + df.index.values.astype("datetime64[D]").astype(np.int64),
        }
        gap = np.r_[True, np.diff(df.index.values) != one]
        idx = np.flatnonzero((df.index >= v0) & (df.index < v1) & (col["tok"] >= 0))
        breaks = np.flatnonzero((np.diff(idx) != 1) | gap[idx[1:]])
        for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + 1):
            if b - a > T:
                starts.append(n + np.arange(b - a - T))
            for k in keys:
                d[k].append(col[k][a:b])
            n += b - a
    d = {k: np.concatenate(v) for k, v in d.items()}
    return d, np.concatenate(starts)[::T]                          # non-overlapping windows


@torch.no_grad()
def run_model(model, s, stats, d, starts, T):
    """Per target candle: bits, P(up), expected z, expected z^2 (empirical token meaning)."""
    dvol = np.nan_to_num((d["dvol"] - s.get("dvol_mean", 0.0)) / s.get("dvol_std", 1.0))
    tensors = {k: torch.tensor(d[k], device=device) for k in ("tok", "slot", "day", "coin")}
    tensors["dvol"] = torch.tensor(dvol, dtype=torch.float32, device=device)
    tstats = {k: torch.tensor(stats[k], dtype=torch.float32, device=device) for k in ("m1", "m2", "pu")}
    offsets = torch.arange(T + 1, device=device)
    out = {k: [] for k in ("bits", "p_up", "exp_ret", "m2")}
    st = torch.tensor(starts, device=device)
    for i in range(0, len(st), 512):
        pos = st[i:i + 512, None] + offsets
        x, y = tensors["tok"][pos[:, :-1]], tensors["tok"][pos[:, 1:]]
        ctx = {k: tensors[k][pos[:, :-1]] for k in ("slot", "day", "coin", "dvol")}
        logp = F.log_softmax(model(x, ctx), dim=-1)
        probs = logp.exp()
        out["bits"].append((-logp.gather(-1, y[..., None])[..., 0] / LN2).cpu())
        out["p_up"].append((probs @ tstats["pu"]).cpu())
        out["exp_ret"].append((probs @ tstats["m1"]).cpu())
        out["m2"].append((probs @ tstats["m2"]).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}       # each (N_windows, T)


# ---------------- statistics ----------------
def ranks(a):
    return pd.Series(a).rank().to_numpy()


def auc(score, up):
    rk = ranks(score)
    return (rk[up].sum() - up.sum() * (up.sum() + 1) / 2) / (up.sum() * (~up).sum())


def spearman(a, b):
    return np.corrcoef(ranks(a), ranks(b))[0, 1]


def group_index(groups):
    return np.unique(groups, return_inverse=True)[1]


def mean_ci(values, groups):
    """Mean with block-bootstrap CI (resampling group sums)."""
    inv = group_index(groups)
    S, N, G = np.bincount(inv, weights=values), np.bincount(inv), inv.max() + 1
    W = np.stack([np.bincount(rng.integers(G, size=G), minlength=G) for _ in range(N_BOOT)])
    return [float(values.mean()), *np.percentile(W @ S / (W @ N), [2.5, 97.5]).tolist()]


def stat_ci(fn, arrays, groups):
    """Any statistic with block-bootstrap CI (resampling whole groups)."""
    inv = group_index(groups)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
        boots.append(fn(*(a[idx] for a in arrays)))
    return [float(fn(*arrays)), *np.percentile(boots, [2.5, 97.5]).tolist()]


def qlike(forecast, real, floor=1e-12):
    ratio = np.maximum(real, floor) / np.maximum(forecast, floor)
    return ratio - np.log(ratio) - 1


def roc_curve(score, y, n_points=400):
    """FPR, TPR when going down the list sorted by score (log-spaced points for plotting)."""
    order = np.argsort(-score, kind="stable")
    tpr = np.cumsum(y[order]) / y.sum()
    fpr = np.cumsum(~y[order]) / (~y).sum()
    keep = np.unique(np.searchsorted(fpr, np.logspace(-4, 0, n_points)).clip(0, len(fpr) - 1))
    return np.r_[0, fpr[keep]], np.r_[0, tpr[keep]]


def precision_curve(score, y, groups):
    """Hit rate among the top q of the list, for every q in COVERAGE, with bootstrap band."""
    inv = group_index(groups)
    G = inv.max() + 1
    S, N = np.zeros((G, len(COVERAGE))), np.zeros((G, len(COVERAGE)))
    for j, q in enumerate(COVERAGE):
        sel = score >= np.quantile(score, 1 - q)
        S[:, j] = np.bincount(inv, weights=(sel & y).astype(float), minlength=G)
        N[:, j] = np.bincount(inv, weights=sel.astype(float), minlength=G)
    W = np.stack([np.bincount(rng.integers(G, size=G), minlength=G) for _ in range(N_BOOT)])
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (W @ S) / (W @ N)
    return S.sum(0) / N.sum(0), np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def plot_curves(ax_roc, ax_prec, curves, base_rate, title_roc, title_prec):
    for c in curves:
        line, = ax_roc.plot(c["fpr"], c["tpr"], label=c["label"])
        ax_prec.plot(COVERAGE, c["prec"][0], color=line.get_color(), label=c["label"])
        ax_prec.fill_between(COVERAGE, c["prec"][1], c["prec"][2], color=line.get_color(), alpha=0.12)
    diag = np.logspace(-4, 0, 200)                                 # TPR = FPR, drawn on a log axis
    ax_roc.plot(diag, diag, "k--", lw=1, label="random")
    ax_roc.set(xscale="log", xlim=(1e-3, 1), ylim=(0, 1), title=title_roc,
               xlabel="FPR: share of losers selected (log)", ylabel="TPR: share of winners caught")
    ax_prec.axhline(base_rate, color="k", ls="--", lw=1, label=f"base rate {base_rate:.3f}")
    ax_prec.set(xscale="log", title=title_prec, xlabel="fraction of the list traded (top by score, log)",
                ylabel="hit rate (precision)")
    for a in (ax_roc, ax_prec):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)


# ---------------- main ----------------
def main():
    runs = find_runs()
    print("Baseline:", runs[0].name)
    base_model, tok, s0, ck0, tok_name = load(runs[0])
    T = s0["context"]
    print("Measuring what each token means on training data ...")
    stats = token_stats(tok, s0["symbols"])
    d, starts = build_val(tok, s0["symbols"], T)
    tgt = starts[:, None] + 1 + np.arange(T)
    real, sigma, ewma, daykey = d["ret"][tgt], d["sigma"][tgt], d["ewma"][tgt], d["daykey"][tgt]
    up, days = (real > 0).ravel(), daykey.ravel()
    n_blocks = T // VOL_BLOCK
    blk = slice(T - n_blocks * VOL_BLOCK, T)
    block_sum = lambda a: a[:, blk].reshape(len(a), n_blocks, VOL_BLOCK).sum(2).ravel()
    real_var, ewma_var = block_sum(real ** 2), block_sum(ewma)
    block_day = daykey[:, blk][:, ::VOL_BLOCK].ravel()
    print(f"{len(starts):,} validation windows, {real.size:,} predictions per model")

    rows, curves, base = [], [], None
    for path in runs:
        model, tok_m, s, ck, name = (base_model, tok, s0, ck0, tok_name) if path == runs[0] \
            else load(path)
        if name != tok_name or s["context"] != T:
            print(f"  skipping {path.name}: different tokenizer or context")
            continue
        r = run_model(model, s, stats, d, starts, T)
        p_up, exp_ret = r["p_up"].ravel(), r["exp_ret"].ravel()
        if base is None:
            base = {"bits": r["bits"], "p_up": p_up, "exp_ret": exp_ret}
        label = path.stem.split("_2026")[0]
        model_var = block_sum(r["m2"] * sigma ** 2)
        row = {
            "model": label,
            "d_bits_mbit": [1000 * v for v in mean_ci((r["bits"] - base["bits"]).ravel(), days)],
            "below_unigram_mbit": 1000 * (ck.get("unigram_bits", np.nan) - r["bits"].mean()),
            "bits_ctx1": float(r["bits"][:, 0].mean()),
            "bits_ctx64": float(r["bits"][:, -1].mean()),
            "auc": float(auc(p_up, up)),
            "d_auc": stat_ci(lambda a, b, u: auc(a, u) - auc(b, u), (p_up, base["p_up"], up), days),
            "ic": float(spearman(exp_ret, real.ravel())),
            "d_ic": stat_ci(lambda a, b, x: spearman(a, x) - spearman(b, x),
                            (exp_ret, base["exp_ret"], real.ravel()), days),
            "d_qlike_vs_ewma": mean_ci(qlike(model_var, real_var) - qlike(ewma_var, real_var), block_day),
        }
        fpr, tpr = roc_curve(p_up, up)
        curves.append({"label": label, "fpr": fpr, "tpr": tpr, "prec": precision_curve(p_up, up, days)})
        rows.append(row)
        print(f"  {label:18s} d_bits {row['d_bits_mbit'][0]:+7.1f} | AUC {row['auc']:.4f} "
              f"(d {row['d_auc'][0]:+.4f}) | IC {row['ic']:+.4f} | dQLIKE {row['d_qlike_vs_ewma'][0]:+.4f}")

    # ---------------- table ----------------
    fmt = lambda v, p: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"
    table = pd.DataFrame({
        "model": [r["model"] for r in rows],
        "d_bits (mbit)": [fmt(r["d_bits_mbit"], 1) for r in rows],
        "below unigram": [f"{r['below_unigram_mbit']:.1f}" for r in rows],
        "bits ctx1": [f"{r['bits_ctx1']:.4f}" for r in rows],
        "bits ctx64": [f"{r['bits_ctx64']:.4f}" for r in rows],
        "AUC": [f"{r['auc']:.4f}" for r in rows],
        "d_AUC vs base": [fmt(r["d_auc"], 4) for r in rows],
        "IC": [f"{r['ic']:+.4f}" for r in rows],
        "d_IC vs base": [fmt(r["d_ic"], 4) for r in rows],
        "dQLIKE vs EWMA": [fmt(r["d_qlike_vs_ewma"], 4) for r in rows],
    })
    print("\n" + table.to_string(index=False))
    out_dir = HERE / "results" / f"compare_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    table.to_csv(out_dir / "comparison.csv", index=False)

    # ---------------- figure 1: summary with intervals ----------------
    labels, y = [r["model"] for r in rows], np.arange(len(rows))
    fig, ax = plt.subplots(1, 4, figsize=(20, 0.5 * len(rows) + 2.5), sharey=True)
    for a, key, title in ((ax[0], "d_bits_mbit", "bits vs. baseline (mbit, left = better)"),
                          (ax[1], "d_auc", "next-candle AUC vs. baseline (right = better)"),
                          (ax[2], "d_ic", "next-candle IC vs. baseline (right = better)"),
                          (ax[3], "d_qlike_vs_ewma", "QLIKE vs. EWMA (left = better)")):
        v = np.array([r[key] for r in rows])
        a.errorbar(v[:, 0], y, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", capsize=3)
        a.axvline(0, color="k", ls="--", lw=1)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_yticks(y, labels)
    ax[0].invert_yaxis()
    fig.suptitle(f"Model comparison on {VAL_START} .. {VAL_END} (baseline: {labels[0]})")
    fig.tight_layout()
    fig.savefig(out_dir / "1_summary.png", dpi=150)

    # ---------------- figure 2: ROC and hit-rate curves ----------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_curves(ax[0], ax[1], curves, up.mean(), "ROC of next-candle P(up)",
                "Hit rate when trading only the most confident calls")
    fig.tight_layout()
    fig.savefig(out_dir / "2_roc_hit_rate.png", dpi=150)
    print(f"\nSaved comparison.csv, 1_summary.png, 2_roc_hit_rate.png to {out_dir}")


if __name__ == "__main__":
    main()