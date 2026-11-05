"""
Rigorous evaluation of a trained candle GPT.

For many random, non-overlapping start points in the evaluation period, the model
samples N_PATHS futures (free run, temperature 1). From the paths we forecast, for
each horizon h:
    direction   P(price higher after h candles), expected return
    volatility  realized volatility over the next h candles
    calibration where the real outcome falls inside the predicted distribution
and compare with what really happened and with simple baselines:
    direction   base rate (always the majority), momentum (sign of the last h candles)
    volatility  EWMA of squared returns (RiskMetrics), rolling std (as in the tokenizer)
Every metric gets a 95 % confidence interval from a block bootstrap (blocks = coin-days).

Run:           python evaluation/evaluate_gpt.py
Requirements:  pip install torch numpy pandas pyarrow matplotlib
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                          # write image files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "tokenizer.py").exists())
sys.path[:0] = [str(ROOT / "data"), str(next(ROOT.rglob("GPT1_0.py")).parent)]
from tokenizer import GridTokenizer
from GPT1_0 import GPT

# ============================== SETTINGS ==============================
CHECKPOINT      = None                         # None = newest **/runs/gpt_*.pt
SYMBOLS         = None                         # None = the coins the model was trained on
INTERVAL        = "1m"
EVAL_START, EVAL_END = "2026-07-01", "2026-08-31"   # validation period (keep test for the end!)
HORIZONS        = [1, 5, 10, 15, 30, 60]               # candles ahead
STARTS_PER_COIN = 400
N_PATHS         = 100
TEMPERATURE     = 1.0
GEN_BATCH       = 8192                         # sequences per forward pass on the GPU
EWMA_LAMBDA     = 0.94
N_BOOT          = 500
SEED            = 0
# ======================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
H_MAX = max(HORIZONS)


# ---------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------
def load_model():
    path = Path(CHECKPOINT) if CHECKPOINT else sorted(ROOT.glob("**/runs/gpt_*.pt"))[-1]
    ckpt = torch.load(path, map_location=device)
    s = ckpt["settings"]
    run_name = re.split(r"[\\/]", ckpt["tokenizer_run"].rstrip("\\/"))[-1]
    tok = GridTokenizer.load(ROOT / "data" / "tokenizer_runs" / run_name / "tokenizer.json")
    model = GPT(tok.vocab_size, s["embed_dim"], s["context"], s["n_heads"], s["n_layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return path, model, tok, s


# ---------------------------------------------------------------------
# 2. Start points and everything we need to know about them
# ---------------------------------------------------------------------
def collect_starts(tok, T, symbols, rng):
    """Per start k: context tokens, past returns (for sigma), real future returns,
    baselines. Starts are >= H_MAX apart, so their futures never overlap."""
    W = tok.window
    one = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
    e0, e1 = pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC")
    out = {k: [] for k in ("ctx", "past", "future", "mom", "ewma", "roll", "group", "coin")}
    for c, sym in enumerate(symbols):
        df = pd.read_parquet(ROOT / "data" / "raw" / f"{sym}_{INTERVAL}.parquet")
        df = df.loc[e0 - pd.Timedelta(days=2): e1]
        tokens = tok.encode(df).to_numpy()
        logc = np.log(df["close"].to_numpy())
        r = np.r_[np.nan, np.diff(logc)]                       # r[j] = return of candle j
        ewma = pd.Series(r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().to_numpy()
        gap = np.r_[True, np.diff(df.index.values) != one].astype(int)
        cum_gap = np.r_[0, np.cumsum(gap)]                     # gaps in [a, b) = cum_gap[b]-cum_gap[a]
        bad_tok = np.r_[0, np.cumsum(tokens < 0)]

        n = len(df)
        k = np.arange(T + W + 1, n - H_MAX)                    # candidate first predicted candles
        ok = ((df.index[k] >= e0) & (df.index[k + H_MAX - 1] < e1)
              & (bad_tok[k] - bad_tok[k - T] == 0)             # valid context tokens
              & (cum_gap[k + H_MAX] - cum_gap[k - T - W] == 0))  # no gaps in history or future
        k = k[ok][::H_MAX]                                     # non-overlapping futures
        k = np.sort(rng.choice(k, size=min(STARTS_PER_COIN, len(k)), replace=False))

        out["ctx"].append(tokens[k[:, None] - T + np.arange(T)])
        out["past"].append(r[k[:, None] - W + np.arange(W)])   # returns of candles k-W .. k-1
        out["future"].append(r[k[:, None] + np.arange(H_MAX)]) # returns of candles k .. k+H-1
        out["mom"].append(np.stack([logc[k - 1] - logc[k - 1 - h] for h in HORIZONS], 1))
        out["ewma"].append(ewma[k - 1])                        # per-candle variance known at k-1
        out["roll"].append(np.var(r[k[:, None] - W + np.arange(W)], axis=1, ddof=1))
        out["group"].append(c * 100000 + df.index[k].normalize().asi8 // 86_400_000_000_000)
        out["coin"].append(np.full(len(k), c))
        print(f"  {sym}: {len(k)} start points")
    return {key: np.concatenate(v) for key, v in out.items()}


# ---------------------------------------------------------------------
# 3. Sample futures with the model, turn tokens into returns
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_tokens(model, T, ctx):
    """ctx (S, T) -> generated tokens (S, P, H_MAX), free run."""
    seqs = torch.tensor(ctx, device=device).repeat_interleave(N_PATHS, dim=0)
    out = []
    for i in range(0, len(seqs), GEN_BATCH):
        x = seqs[i:i + GEN_BATCH]
        for _ in range(H_MAX):
            probs = F.softmax(model(x[:, -T:])[:, -1, :] / TEMPERATURE, dim=-1)
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
        out.append(x[:, T:].cpu().numpy())
        print(f"\r  sampled {min(i + GEN_BATCH, len(seqs)):,}/{len(seqs):,} paths", end="")
    print()
    return np.concatenate(out).reshape(len(ctx), N_PATHS, H_MAX)


def tokens_to_returns(tok, gen, past):
    """Generated tokens -> log returns, with sigma updated from each path's own history."""
    S, P, H = gen.shape
    W = tok.window
    ret_value = tok.decode(np.arange(tok.vocab_size))["ret"].to_numpy()
    z = ret_value[gen].reshape(S * P, H)                       # returns in sigma units
    buf = np.repeat(past, P, axis=0).astype(np.float64)        # last W returns per path
    r = np.zeros((S * P, H))
    for j in range(H):
        sigma = buf.std(axis=1, ddof=1)
        r[:, j] = z[:, j] * sigma
        buf[:, j % W] = r[:, j]                                # replace the oldest return
    return r.reshape(S, P, H)


# ---------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------
def ranks(a):
    return pd.Series(a).rank().to_numpy()


def spearman(a, b):
    return np.corrcoef(ranks(a), ranks(b))[0, 1]


def auc(score, y):
    pos = y.astype(bool)
    if pos.all() or not pos.any():
        return np.nan
    rk = ranks(score)
    return (rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def qlike(forecast_var, real_var, floor=1e-10):
    ratio = np.maximum(real_var, floor) / np.maximum(forecast_var, floor)
    return ratio - np.log(ratio) - 1                           # 0 if perfect, lower = better


def evaluate(st, paths):
    groups = np.unique(st["group"])
    members = [np.flatnonzero(st["group"] == g) for g in groups]
    rng = np.random.default_rng(SEED + 1)
    boot_idx = [np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
                for _ in range(N_BOOT)]

    def with_ci(fn):
        vals = np.array([fn(idx) for idx in boot_idx])
        return [float(fn(np.arange(len(st["group"])))), *np.nanpercentile(vals, [2.5, 97.5]).tolist()]

    cum_paths, cum_sq_paths = paths.cumsum(2), (paths ** 2).cumsum(2)
    real_cum, real_sq = st["future"].cumsum(1), (st["future"] ** 2).cumsum(1)
    results, arrays = {}, {}
    for hi, h in enumerate(HORIZONS):
        R = cum_paths[:, :, h - 1]                             # (S, P) path returns
        real = real_cum[:, h - 1]
        up = (real > 0).astype(float)
        p_up = (R > 0).mean(1) + 0.5 * (R == 0).mean(1)
        exp_r = R.mean(1)
        mom = st["mom"][:, hi]
        var_model = cum_sq_paths[:, :, h - 1].mean(1)
        var_real = real_sq[:, h - 1]
        var_ewma, var_roll = h * st["ewma"], h * st["roll"]
        pit = (R < real[:, None]).mean(1) + 0.5 * (R == real[:, None]).mean(1)
        lo, hi_ = np.percentile(R, [10, 90], axis=1)
        covered = ((real >= lo) & (real <= hi_)).astype(float)
        base = up.mean()

        results[h] = {
            "n": int(len(real)),
            "up_rate": float(base),
            "acc_model": with_ci(lambda i: np.mean((p_up[i] > 0.5) == up[i])),
            "acc_momentum": with_ci(lambda i: np.mean((mom[i] > 0) == up[i])),
            "acc_base": float(max(base, 1 - base)),
            "auc_model": with_ci(lambda i: auc(p_up[i], up[i])),
            "auc_momentum": with_ci(lambda i: auc(mom[i], up[i])),
            "brier_skill": with_ci(lambda i: 1 - np.mean((p_up[i] - up[i]) ** 2)
                                   / np.mean((up[i].mean() - up[i]) ** 2)),
            "ic_model": with_ci(lambda i: spearman(exp_r[i], real[i])),
            "ic_momentum": with_ci(lambda i: spearman(mom[i], real[i])),
            "vol_rank_model": with_ci(lambda i: spearman(var_model[i], var_real[i])),
            "vol_rank_ewma": with_ci(lambda i: spearman(var_ewma[i], var_real[i])),
            "vol_rank_roll": with_ci(lambda i: spearman(var_roll[i], var_real[i])),
            "qlike_model": float(qlike(var_model, var_real).mean()),
            "qlike_ewma": float(qlike(var_ewma, var_real).mean()),
            "qlike_diff_vs_ewma": with_ci(lambda i: np.mean(qlike(var_model[i], var_real[i])
                                                            - qlike(var_ewma[i], var_real[i]))),
            "coverage_80": with_ci(lambda i: covered[i].mean()),
        }
        arrays[h] = {"p_up": p_up, "up": up, "pit": pit, "var_model": var_model,
                     "var_ewma": var_ewma, "var_real": var_real}
    return results, arrays


# ---------------------------------------------------------------------
# 5. Plots (matplotlib, saved as PNG)
# ---------------------------------------------------------------------
def errbar(ax, x, triples, label, **kw):
    v = np.array(triples)
    ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                marker="o", capsize=3, label=label, **kw)


def plot_all(res, arr, out_dir, title):
    H = HORIZONS
    get = lambda key: [res[h][key] for h in H]

    # --- direction ---
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    errbar(ax[0], H, get("acc_model"), "model")
    errbar(ax[0], H, get("acc_momentum"), "momentum")
    ax[0].plot(H, get("acc_base"), "k--", label="base rate (always majority)")
    ax[0].set(title="Direction accuracy", xlabel="horizon (candles)", ylabel="accuracy")
    errbar(ax[1], H, get("auc_model"), "model")
    errbar(ax[1], H, get("auc_momentum"), "momentum")
    ax[1].axhline(0.5, color="k", ls="--", label="no skill")
    ax[1].set(title="AUC of P(up)", xlabel="horizon (candles)")
    errbar(ax[2], H, get("ic_model"), "model: expected return")
    errbar(ax[2], H, get("ic_momentum"), "momentum")
    ax[2].axhline(0, color="k", ls="--")
    ax[2].set(title="Rank IC with the real return", xlabel="horizon (candles)")
    h = H[1] if len(H) > 1 else H[0]
    p, y = arr[h]["p_up"], arr[h]["up"]
    bins = np.quantile(p, np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
    ax[3].plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax[3].plot([p[which == b].mean() for b in range(10)],
               [y[which == b].mean() for b in range(10)], "o-", label="model")
    lo, hi = np.quantile(p, [0.01, 0.99])
    ax[3].set(title=f"Reliability of P(up), h = {h}", xlabel="predicted P(up)",
              ylabel="observed frequency", xlim=(lo - 0.02, hi + 0.02), ylim=(lo - 0.05, hi + 0.05))
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"Direction  |  {title}")
    fig.tight_layout()
    fig.savefig(out_dir / "1_direction.png", dpi=150)

    # --- volatility ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    errbar(ax[0], H, get("vol_rank_model"), "model")
    errbar(ax[0], H, get("vol_rank_ewma"), "EWMA")
    errbar(ax[0], H, get("vol_rank_roll"), "rolling std")
    ax[0].set(title="Rank correlation with realized volatility", xlabel="horizon (candles)")
    errbar(ax[1], H, get("qlike_diff_vs_ewma"), "QLIKE(model) - QLIKE(EWMA)", color="C3")
    ax[1].axhline(0, color="k", ls="--")
    ax[1].set(title="Loss difference (below 0: model better)", xlabel="horizon (candles)")
    h = H[2] if len(H) > 2 else H[-1]
    a = arr[h]
    for name, key, col in (("model", "var_model", "C0"), ("EWMA", "var_ewma", "C1")):
        ax[2].scatter(np.sqrt(a[key]) * 1e4, np.sqrt(np.maximum(a["var_real"], 1e-12)) * 1e4,
                      s=4, alpha=0.3, color=col, label=name)
    lim = np.percentile(np.sqrt(a["var_real"][a["var_real"] > 0]) * 1e4, [1, 99.5])
    ax[2].plot(lim, lim, "k--")
    ax[2].set(xscale="log", yscale="log", xlim=lim, ylim=lim, title=f"Forecast vs. realized, h = {h}",
              xlabel="forecast volatility (bp)", ylabel="realized volatility (bp)")
    for a_ in ax:
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    fig.suptitle(f"Volatility  |  {title}")
    fig.tight_layout()
    fig.savefig(out_dir / "2_volatility.png", dpi=150)

    # --- calibration ---
    fig, ax = plt.subplots(1, len(H), figsize=(4 * len(H), 3.8), squeeze=False)
    for a_, h in zip(ax[0], H):
        a_.hist(arr[h]["pit"], bins=20, range=(0, 1), density=True, color="C0", alpha=0.8)
        a_.axhline(1, color="k", ls="--")
        cov = res[h]["coverage_80"]
        a_.set(title=f"h = {h}: 80 % band holds {cov[0]:.0%} [{cov[1]:.0%}, {cov[2]:.0%}]",
               xlabel="rank of real outcome among paths", ylabel="density")
    fig.suptitle(f"Calibration (flat = honest uncertainty; U-shape = too confident, "
                 f"hump = too cautious)  |  {title}")
    fig.tight_layout()
    fig.savefig(out_dir / "3_calibration.png", dpi=150)
    plt.close("all")


# ---------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    path, model, tok, s = load_model()
    symbols = SYMBOLS or s["symbols"]
    print(f"Model {path.name} on {device}, evaluating {EVAL_START} .. {EVAL_END}")

    st = collect_starts(tok, s["context"], symbols, rng)
    print(f"{len(st['ctx'])} start points, {N_PATHS} paths each, horizons {HORIZONS}")
    gen = sample_tokens(model, s["context"], st["ctx"])
    paths = tokens_to_returns(tok, gen, st["past"])
    res, arr = evaluate(st, paths)

    out_dir = HERE / "results" / f"{path.stem}__{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    fmt = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if isinstance(v, list) else f"{v:.3f}"
    for h in HORIZONS:
        print(f"\n=== horizon {h} candles  (n = {res[h]['n']}, up rate {res[h]['up_rate']:.3f}) ===")
        for key, val in res[h].items():
            if key not in ("n", "up_rate"):
                print(f"  {key:20s} {fmt(val)}")
    (out_dir / "results.json").write_text(json.dumps(
        {"checkpoint": path.name, "settings": {"eval": [EVAL_START, EVAL_END], "horizons": HORIZONS,
         "starts_per_coin": STARTS_PER_COIN, "paths": N_PATHS, "temperature": TEMPERATURE,
         "symbols": symbols}, "results": res}, indent=2))
    plot_all(res, arr, out_dir, f"{path.stem}, {EVAL_START} .. {EVAL_END}")
    print(f"\nSaved results.json and 3 figures to {out_dir}")


if __name__ == "__main__":
    main()