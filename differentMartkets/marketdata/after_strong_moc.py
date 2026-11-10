"""
What does an asset do in the hour AFTER its own strong move?

Event: the price moves at least K sigma (sigma = normal volatility measured before the move)
within DELTA_T minutes; one event per episode. From the end of the move (t = 0) we follow
the price for POST minutes, separately for up-moves and down-moves.
    continues in the same direction  -> momentum
    turns back                       -> mean reversion
Figure 1: average price path (bp) after up- and down-moves, per asset, with 95 % bands.
Figure 2: continuation after 60 min vs. the size of the move (K), against the trading cost.
Uses the training period only (settings shared with marketlinks.py in the same folder).

Run:  python analysis/after_strong_moves.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import marketlinks as ml

# ============================== SETTINGS ==============================
CRYPTO   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
MARKETS  = ["SPX", "NDX", "XAU", "EURUSD"]          # remove any you have not downloaded
DELTA_T  = 5                                        # minutes
K_MAIN   = 4.0                                      # threshold for figure 1 (sigma)
K_LIST   = [3, 4, 6, 8]                             # thresholds for figure 2
PRE, POST = 30, 60                                  # minutes before / after
HORIZONS = [5, 15, 30, 60]
COST_BP  = 20                                       # round-trip cost, for reference
# ======================================================================

ml.CRYPTO, ml.MARKETS = CRYPTO, MARKETS
ASSETS = CRYPTO + MARKETS


def find_events(logp, scale, a, k):
    move = (logp[a] - logp[a].shift(DELTA_T)).to_numpy()
    thr = k * scale[a].shift(DELTA_T).to_numpy() * np.sqrt(DELTA_T)       # known before the move
    cand = np.flatnonzero(np.abs(move) >= thr)
    events, last = [], -10 ** 9
    for i in cand:
        if i - last > DELTA_T + POST:
            events.append(i)
            last = i
    events = np.array(events, dtype=int)
    return events, np.sign(move[events])


def price_paths(logp, a, events):
    """Log price around each event in bp, zero at the end of the move. Rows with gaps dropped."""
    u = np.arange(-(DELTA_T + PRE), POST + 1)
    lp = logp[a].to_numpy()
    events = events[(events + u[0] >= 0) & (events + u[-1] < len(lp))]
    p = lp[events[:, None] + u]
    p = (p - p[:, [np.flatnonzero(u == 0)[0]]]) * 1e4
    keep = np.isfinite(p).all(1)
    return u, p[keep], events[keep]


def main():
    logp, z, scale, us, day = ml.build()
    out_dir = Path(__file__).resolve().parent / "results" / f"after_moves_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)

    # unconditional drift: average 60-min return from any minute (the "nothing happened" baseline)
    drift = {a: float(np.nanmean((logp[a].shift(-POST) - logp[a]).to_numpy()) * 1e4) for a in ASSETS}

    rows, curves = [], {}
    for a in ASSETS:
        for k in K_LIST:
            ev, sign = find_events(logp, scale, a, k)
            for name, s in (("up", 1), ("down", -1)):
                u, p, e = price_paths(logp, a, ev[sign == s])
                if len(e) < 10:
                    continue
                g = day[e]
                row = {"asset": a, "K": k, "move": name, "events": len(e)}
                for h in HORIZONS:
                    m, lo, hi = ml.mean_ci(p[:, np.flatnonzero(u == h)[0]], g)
                    row[f"after {h} min (bp)"] = f"{m[0]:+.1f} [{lo[0]:+.1f}, {hi[0]:+.1f}]"
                    if h == POST:
                        # continuation: positive = same direction as the move
                        row["_cont"] = (s * m[0], *sorted((s * lo[0], s * hi[0])))
                rows.append(row)
                if k == K_MAIN:
                    curves[(a, name)] = (u, ml.mean_ci(p, g), len(e))

    table = pd.DataFrame(rows)
    print(table.drop(columns="_cont").to_string(index=False))
    print("\nUnconditional 60-min drift (bp):", {a: round(v, 2) for a, v in drift.items()})
    table.drop(columns="_cont").to_csv(out_dir / "after_moves.csv", index=False)

    # ---------- figure 1: paths after up- and down-moves ----------
    cols = 5
    rows_n = int(np.ceil(len(ASSETS) / cols))
    fig, ax = plt.subplots(rows_n, cols, figsize=(4.4 * cols, 3.8 * rows_n), squeeze=False)
    for k_, a in enumerate(ASSETS):
        a_ = ax.flat[k_]
        for name, color in (("up", "C2"), ("down", "C3")):
            if (a, name) not in curves:
                continue
            u, (m, lo, hi), n = curves[(a, name)]
            a_.plot(u, m, color=color, label=f"after up-moves ({n})" if name == "up" else f"after down-moves ({n})")
            a_.fill_between(u, lo, hi, color=color, alpha=0.2)
        a_.axhline(0, color="k", lw=0.8)
        a_.axvline(0, color="gray", ls="--", lw=0.8)
        a_.axvline(-DELTA_T, color="gray", ls=":", lw=0.8)
        a_.set(title=a, xlabel="minutes from end of the move", ylabel="price change vs. t = 0 (bp)",
               xlim=(-10, POST))
        a_.grid(alpha=0.3)
        a_.legend(fontsize=7)
    for k_ in range(len(ASSETS), ax.size):
        ax.flat[k_].axis("off")
    fig.suptitle(f"Price after strong moves (> {K_MAIN} sigma within {DELTA_T} min), training period. "
                 f"Green rising / red falling after 0 = momentum; turning back = reversal")
    fig.tight_layout()
    fig.savefig(out_dir / "1_paths_after_moves.png", dpi=150)

    # ---------- figure 2: continuation after 60 min vs. move size ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for a_, name in zip(ax, ("up", "down")):
        for i, a in enumerate(ASSETS):
            sub = [r for r in rows if r["asset"] == a and r["move"] == name]
            if not sub:
                continue
            ks = np.array([r["K"] for r in sub]) + (i - len(ASSETS) / 2) * 0.04
            c = np.array([r["_cont"] for r in sub])
            a_.errorbar(ks, c[:, 0], yerr=[c[:, 0] - c[:, 1], c[:, 2] - c[:, 0]], marker="o",
                        capsize=2, label=a, alpha=0.85)
        a_.axhline(0, color="k", lw=0.8)
        a_.axhspan(-COST_BP, COST_BP, color="gray", alpha=0.12, label=f"within +/- {COST_BP} bp costs")
        a_.set(title=f"After {name}-moves: continuation after {POST} min (positive = momentum)",
               xlabel="size of the move K (sigma)", ylabel="bp in the direction of the move")
        a_.grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "2_continuation_vs_size.png", dpi=150)
    print(f"\nSaved after_moves.csv and 2 figures to {out_dir}")


if __name__ == "__main__":
    main()