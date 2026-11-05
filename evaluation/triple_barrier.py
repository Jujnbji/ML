"""
Triple-barrier evaluation: from a start point, which comes first within HORIZON candles,
price +BARRIER_PCT %, price -BARRIER_PCT %, or neither?

Two questions, each against a baseline:
  swing      P(any barrier is hit)            vs. a Gaussian random walk with EWMA volatility
  direction  P(upper first | a barrier is hit) vs. a coin flip (the random walk is symmetric)
For each: AUC (with paired difference to the random walk), Brier skill, ROC and hit-rate
curves with bootstrap bands, and a reliability diagram for the swing probability.

The model's futures are sampled (free run, temperature 1). Tokens are turned into returns
with real z values drawn from training candles that had the same token, so the paths
keep the true spread inside each token. Barriers are checked on close prices.

Run:  python evaluation/triple_barrier.py
"""

import json
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
sys.path.insert(0, str(HERE))
import compare_models as cm
from compare_models import (ROOT, device, load, token_stats, auc, stat_ci, roc_curve,
                            precision_curve, plot_curves)

# ============================== SETTINGS ==============================
MODELS          = None               # list of file names; None = newest ctx_none_s0 + newest ctx_twcd
INTERVAL        = "1m"
BARRIER_PCT     = 1.0                # +/- this many percent
HORIZON         = 60                 # candles
STARTS_PER_COIN = 1000
N_PATHS         = 100
GEN_BATCH       = 8192
SEED            = 0
# ======================================================================

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
H = HORIZON


def find_models():
    if MODELS:
        return [next(ROOT.glob(f"**/runs/{m}")) for m in MODELS]
    runs = sorted(ROOT.glob("**/runs/ctx_*.pt"))
    picks = ([p for p in runs if p.name.startswith("ctx_none_s0")][-1:]
             + [p for p in runs if p.name.startswith("ctx_twcd")][-1:])
    return picks or runs[-1:]


# ---------------- start points ----------------
def collect_starts(tok, symbols, T):
    W = tok.window
    one = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
    v0, v1 = pd.Timestamp(cm.VAL_START, tz="UTC"), pd.Timestamp(cm.VAL_END, tz="UTC")
    keys = ("ctx_tok", "ctx_slot", "ctx_day", "ctx_dvol", "fut_slot", "fut_day",
            "coin", "past", "real", "ewma", "group")
    out = {k: [] for k in keys}
    for c, sym in enumerate(symbols):
        df = pd.read_parquet(ROOT / "data" / "raw" / f"{sym}_{INTERVAL}.parquet")
        df = df.loc[v0 - pd.Timedelta(days=3): v1]
        tokens = tok.encode(df).to_numpy()
        logc = np.log(df["close"].to_numpy())
        r = np.r_[np.nan, np.diff(logc)]
        ewma = pd.Series(r ** 2).ewm(alpha=1 - cm.EWMA_LAMBDA, adjust=False).mean().shift(1).to_numpy()
        dvol = np.log(pd.Series(r).rolling(1440, min_periods=720).std()) \
                 .replace([np.inf, -np.inf], np.nan).to_numpy()
        slot = ((df.index.hour * 60 + df.index.minute) // 15).to_numpy()
        day = df.index.dayofweek.to_numpy()
        cum_gap = np.r_[0, np.cumsum(np.r_[True, np.diff(df.index.values) != one])]
        bad_tok = np.r_[0, np.cumsum(tokens < 0)]

        k = np.arange(T + W + 1, len(df) - H)
        ok = ((df.index[k] >= v0) & (df.index[k + H - 1] < v1)
              & (bad_tok[k] - bad_tok[k - T] == 0)
              & (cum_gap[k + H] - cum_gap[k - T - W] == 0))
        k = k[ok][::H]                                            # futures never overlap
        k = np.sort(rng.choice(k, size=min(STARTS_PER_COIN, len(k)), replace=False))
        ctx, fut = k[:, None] - T + np.arange(T), k[:, None] + np.arange(H)
        out["ctx_tok"].append(tokens[ctx])
        out["ctx_slot"].append(slot[ctx])
        out["ctx_day"].append(day[ctx])
        out["ctx_dvol"].append(dvol[ctx])
        out["fut_slot"].append(slot[fut])
        out["fut_day"].append(day[fut])
        out["coin"].append(np.full(len(k), c))
        out["past"].append(r[k[:, None] - W + np.arange(W)])      # returns of candles k-W .. k-1
        out["real"].append(logc[fut] - logc[k - 1][:, None])      # cumulative real return
        out["ewma"].append(ewma[k])                               # one-step variance known at k-1
        out["group"].append(c * 100000 + df.index[k].normalize().asi8 // 86_400_000_000_000)
        print(f"  {sym}: {len(k)} start points")
    return {key: np.concatenate(v) for key, v in out.items()}


# ---------------- futures ----------------
@torch.no_grad()
def sample_tokens(model, s, st, T):
    """Free-run token futures (S, P, H). Context for generated candles: their real clock
    time and the coin; the daily volatility is frozen at its value before the start."""
    S = len(st["ctx_tok"])
    dv = np.nan_to_num((st["ctx_dvol"] - s.get("dvol_mean", 0.0)) / s.get("dvol_std", 1.0))
    full = {
        "slot": np.concatenate([st["ctx_slot"], st["fut_slot"]], 1),
        "day": np.concatenate([st["ctx_day"], st["fut_day"]], 1),
        "coin": np.repeat(st["coin"][:, None], T + H, 1),
        "dvol": np.concatenate([dv, np.repeat(dv[:, -1:], H, 1)], 1),
    }
    full = {k: torch.tensor(v, dtype=torch.float32 if k == "dvol" else torch.long, device=device)
            for k, v in full.items()}
    ctx_tok = torch.tensor(st["ctx_tok"], device=device)
    out = []
    for i in range(0, S * N_PATHS, GEN_BATCH):
        which = torch.arange(i, min(i + GEN_BATCH, S * N_PATHS), device=device) // N_PATHS
        x = ctx_tok[which]
        ctx_all = {k: v[which] for k, v in full.items()}
        for n in range(H):
            ctx = {k: v[:, n:n + T] for k, v in ctx_all.items()}  # context of the current window
            probs = F.softmax(model(x[:, -T:], ctx)[:, -1, :], dim=-1)
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
        out.append(x[:, T:].cpu().numpy())
        print(f"\r    sampled {min(i + GEN_BATCH, S * N_PATHS):,}/{S * N_PATHS:,} paths", end="")
    print()
    return np.concatenate(out).reshape(S, N_PATHS, H)


def tokens_to_paths(gen, past, stats):
    """Tokens -> cumulative log returns (S, P, H), sigma updated along each path."""
    S, P, _ = gen.shape
    W = past.shape[1]
    samples = stats["samples"].astype(np.float32)
    z = samples[gen.reshape(S * P, H), rng.integers(samples.shape[1], size=(S * P, H))]
    buf = np.repeat(past, P, axis=0).astype(np.float32)
    r = np.zeros((S * P, H), dtype=np.float32)
    for j in range(H):
        r[:, j] = z[:, j] * buf.std(axis=1, ddof=1)
        buf[:, j % W] = r[:, j]                                   # replace the oldest return
    return r.reshape(S, P, H).cumsum(2)


def random_walk_paths(ewma_var):
    steps = rng.standard_normal((len(ewma_var), N_PATHS, H)).astype(np.float32)
    return (steps * np.sqrt(ewma_var)[:, None, None].astype(np.float32)).cumsum(2)


def first_hit(R, up, down):
    """+1 if the upper barrier is hit first, -1 if the lower, 0 if neither within H."""
    hu, hd = R >= up, R <= down
    iu = np.where(hu.any(-1), hu.argmax(-1), H)
    idn = np.where(hd.any(-1), hd.argmax(-1), H)
    return np.where(iu < idn, 1, np.where(idn < iu, -1, 0))


def forecasts(paths, up, down):
    o = first_hit(paths, up, down)                                # (S, P)
    p_up, p_dn = (o == 1).mean(1), (o == -1).mean(1)
    p_swing = p_up + p_dn
    p_dir = np.divide(p_up, p_swing, out=np.full_like(p_up, 0.5), where=p_swing > 0)
    return p_swing, p_dir


def brier_skill(p, y):
    return 1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2)


def reliability(p, y, n_bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    used = np.unique(b)
    return np.array([p[b == i].mean() for i in used]), np.array([y[b == i].mean() for i in used])


# ---------------- main ----------------
def main():
    paths_ = find_models()
    models = [load(p) for p in paths_]
    _, tok, s0, _, _ = models[0]
    T = s0["context"]
    print("Models:", [p.name for p in paths_])
    print("Measuring what each token means on training data ...")
    stats = token_stats(tok, s0["symbols"])
    st = collect_starts(tok, s0["symbols"], T)
    up, down = np.log(1 + BARRIER_PCT / 100), np.log(1 - BARRIER_PCT / 100)

    real = first_hit(st["real"], up, down)
    y_swing = real != 0
    has_dir = y_swing
    y_dir = real[has_dir] == 1
    print(f"{len(real)} start points | swing of +/-{BARRIER_PCT}% within {H} candles in "
          f"{y_swing.mean():.1%} | of those, upper first {y_dir.mean():.1%}")

    fc = {"random walk (EWMA)": forecasts(random_walk_paths(st["ewma"]), up, down)}
    for path, (model, _, s, _, _) in zip(paths_, models):
        print(f"  sampling {path.name} ...")
        gen = sample_tokens(model, s, st, T)
        fc[path.stem.split("_2026")[0]] = forecasts(tokens_to_paths(gen, st["past"], stats), up, down)

    g, g_dir = st["group"], st["group"][has_dir]
    rw_swing = fc["random walk (EWMA)"][0]
    results, swing_curves, dir_curves, rel = {}, [], [], {}
    for name, (p_swing, p_dir) in fc.items():
        pd_ = p_dir[has_dir]
        results[name] = {
            "swing_auc": stat_ci(auc, (p_swing, y_swing), g),
            "swing_d_auc_vs_rw": stat_ci(lambda a, b, y: auc(a, y) - auc(b, y), (p_swing, rw_swing, y_swing), g),
            "swing_brier_skill": stat_ci(brier_skill, (p_swing, y_swing.astype(float)), g),
            "dir_auc": stat_ci(auc, (pd_, y_dir), g_dir),
            "dir_accuracy": stat_ci(lambda p, y: np.mean((p > 0.5) == y), (pd_, y_dir), g_dir),
        }
        fpr, tpr = roc_curve(p_swing, y_swing)
        swing_curves.append({"label": name, "fpr": fpr, "tpr": tpr,
                             "prec": precision_curve(p_swing, y_swing, g)})
        fpr, tpr = roc_curve(pd_, y_dir)
        dir_curves.append({"label": name, "fpr": fpr, "tpr": tpr, "prec": precision_curve(pd_, y_dir, g_dir)})
        rel[name] = reliability(p_swing, y_swing.astype(float))

    fmt = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    print(f"\nSwing base rate {y_swing.mean():.3f} (n = {len(y_swing)}) | "
          f"direction base rate {y_dir.mean():.3f} (n = {len(y_dir)})")
    for name, r in results.items():
        print(f"\n{name}")
        for k, v in r.items():
            print(f"  {k:20s} {fmt(v)}")

    out_dir = HERE / "results" / f"barrier_{BARRIER_PCT}pct_{H}c_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"models": [p.name for p in paths_], "barrier_pct": BARRIER_PCT, "horizon": H,
         "n_starts": int(len(real)), "swing_rate": float(y_swing.mean()),
         "dir_up_rate": float(y_dir.mean()), "results": results}, indent=2))

    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    plot_curves(ax[0, 0], ax[0, 1], swing_curves, y_swing.mean(),
                f"Swing (+/-{BARRIER_PCT}% within {H}): ROC", "Swing: hit rate of the most confident calls")
    ax[0, 2].plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, (px, py) in rel.items():
        ax[0, 2].plot(px, py, "o-", label=name)
    ax[0, 2].set(title="Swing: predicted vs. observed frequency", xlabel="predicted P(swing)",
                 ylabel="observed frequency")
    ax[0, 2].legend(fontsize=7)
    ax[0, 2].grid(alpha=0.3)
    plot_curves(ax[1, 0], ax[1, 1], dir_curves, y_dir.mean(),
                "Direction given a swing: ROC", "Direction: hit rate of the most confident long calls")
    names = list(results)
    for j, (key, title) in enumerate((("swing_auc", "swing AUC"), ("dir_auc", "direction AUC"))):
        v = np.array([results[n][key] for n in names])
        ax[1, 2].errorbar(v[:, 0], np.arange(len(names)) + 0.15 * j,
                          xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", capsize=3, label=title)
    ax[1, 2].axvline(0.5, color="k", ls="--", lw=1)
    ax[1, 2].set_yticks(np.arange(len(names)), names)
    ax[1, 2].invert_yaxis()
    ax[1, 2].set(title="AUC with 95 % intervals", xlabel="AUC")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3)
    fig.suptitle(f"Triple barrier +/-{BARRIER_PCT}% within {H} candles | {cm.VAL_START} .. {cm.VAL_END} "
                 f"| {len(real)} starts, swing rate {y_swing.mean():.1%}")
    fig.tight_layout()
    fig.savefig(out_dir / "triple_barrier.png", dpi=150)
    print(f"\nSaved results.json and triple_barrier.png to {out_dir}")


if __name__ == "__main__":
    main()