"""
How are crypto and other markets linked, and does one move FIRST?

Part 1  Cross-correlation C(tau) = corr(z_market(t), z_crypto(t + tau)) of 1-minute returns,
        for tau = -MAX_LAG .. +MAX_LAG minutes, all hours and US cash hours.
        tau > 0 with C > 0: the market leads crypto. Peak only at tau = 0: they move together.
Part 2  Event study of STRONG moves: whenever market A moves more than K_SIGMA expected
        standard deviations within DELTA_T minutes, follow market B from PRE minutes before
        to POST minutes after. B's path is multiplied by the sign of A's move, so "B follows A"
        always shows up as a positive number. Both directions: market -> crypto, crypto -> market.

1-minute returns are scaled by their recent volatility (z = r / rolling std of the past hour),
so calm and turbulent periods count equally. 95 % intervals: block bootstrap over days.
Uses the TRAINING period only, so validation and test stay untouched.

Place in ML/analysis/. Run:  python analysis/market_links.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "tokenizer.py").exists())

# ============================== SETTINGS ==============================
CRYPTO     = ["BTCUSDT", "ETHUSDT"]
MARKETS    = ["SPX", "NDX", "XAU", "EURUSD"]
START, END = "2024-10-01", "2026-06-30"      # training period only
MAX_LAG    = 30                              # minutes, cross-correlation
VOL_WINDOW = 60                              # minutes for the rolling volatility scale
DELTA_T    = 5                               # a "strong move" happens within this many minutes
K_SIGMA    = 4.0                             # ... and is at least this many expected std devs
PCT        = None                            # or a fixed threshold in percent (overrides K_SIGMA)
PRE, POST  = 30, 60                          # minutes shown before / after the event
HORIZONS   = [5, 15, 30, 60]                 # follow-through measured after these minutes
N_BOOT     = 500
SEED       = 0
# ======================================================================

rng = np.random.default_rng(SEED)


# ---------------- data ----------------
def load_close(name):
    path = (ROOT / "data" / "raw" / f"{name}_1m.parquet" if name in CRYPTO
            else ROOT / "data" / "markets" / f"{name}_1m.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    close = df["close"].astype(float)
    if name not in CRYPTO and "volume" in df:
        close[df["volume"] <= 0] = np.nan                 # no ticks = no real price
    return close


def build():
    grid = pd.date_range(START, END, freq="1min", tz="UTC", inclusive="left")
    names = CRYPTO + MARKETS
    logp = pd.DataFrame({n: np.log(load_close(n)).reindex(grid) for n in names})
    r = logp.diff()                                       # NaN wherever a minute is missing
    scale = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().shift(1)   # past only
    z = (r / scale).clip(-10, 10)
    ny = grid.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    us = np.asarray((minutes >= 9 * 60 + 30) & (minutes < 16 * 60) & (ny.day_of_week < 5))
    day = grid.values.astype("datetime64[D]").astype(np.int64)
    return logp, z, scale, us, day


# ---------------- statistics ----------------
def mean_ci(values, groups):
    """Column means of values (n, k) with a block-bootstrap 95 % interval over groups."""
    values = values.reshape(len(values), -1)
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, values.shape[1]))
    np.add.at(S, inv, values)
    N = np.bincount(inv, minlength=G)
    W = np.stack([np.bincount(rng.integers(G, size=G), minlength=G) for _ in range(N_BOOT)])
    boots = (W @ S) / (W @ N)[:, None]
    return values.mean(0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# ---------------- part 1: cross-correlation ----------------
def cross_corr(z, a, b, mask, day):
    """C(tau) = corr(z_a(t), z_b(t+tau)) with CI, for tau = -MAX_LAG .. MAX_LAG."""
    za = z[a].to_numpy()
    out = np.zeros((2 * MAX_LAG + 1, 3))
    for j, tau in enumerate(range(-MAX_LAG, MAX_LAG + 1)):
        zb = z[b].shift(-tau).to_numpy()                  # value of b at t + tau
        ok = mask & np.isfinite(za) & np.isfinite(zb)
        prod = za[ok] * zb[ok]
        norm = np.sqrt(np.mean(za[ok] ** 2) * np.mean(zb[ok] ** 2))
        m, lo, hi = mean_ci(prod, day[ok])
        out[j] = m[0] / norm, lo[0] / norm, hi[0] / norm
    return out


# ---------------- part 2: event study ----------------
def find_events(logp, scale, a):
    """Indices t where a moved strongly over (t - DELTA_T, t], declustered."""
    move = (logp[a] - logp[a].shift(DELTA_T)).to_numpy()
    if PCT is not None:
        thr = np.log(1 + PCT / 100)
    else:
        thr = K_SIGMA * scale[a].shift(DELTA_T).to_numpy() * np.sqrt(DELTA_T)   # known before the move
    cand = np.flatnonzero(np.abs(move) >= thr)
    events, last = [], -10 ** 9
    for i in cand:
        if i - last > DELTA_T + POST:                     # one event per episode
            events.append(i)
            last = i
    events = np.array(events, dtype=int)
    return events, np.sign(move[events])


def response(logp, b, events, sign):
    """b's log price from -(DELTA_T+PRE) to +POST around each event, aligned with a's move (bp)."""
    u = np.arange(-(DELTA_T + PRE), POST + 1)
    lp = logp[b].to_numpy()
    inside = (events + u[0] >= 0) & (events + u[-1] < len(lp))
    events, sign = events[inside], sign[inside]
    path = lp[events[:, None] + u]
    path = (path - path[:, [np.flatnonzero(u == 0)[0]]]) * sign[:, None] * 1e4
    keep = np.isfinite(path).all(1)
    return u, path[keep], events[keep]


def analyse_pair(logp, scale, day, a, b):
    events, sign = find_events(logp, scale, a)
    u, pb, ev = response(logp, b, events, sign)
    if len(ev) < 10:
        return None
    _, pa, _ = response(logp, a, ev, sign[np.isin(events, ev)])
    g = day[ev]
    at = lambda h: pb[:, np.flatnonzero(u == h)[0]]
    res = {"pair": f"{a} -> {b}", "n_events": len(ev), "u": u,
           "path_b": mean_ci(pb, g), "path_a": pa.mean(0),
           "during": mean_ci(-at(-DELTA_T), g)}              # b's move while a was moving
    for h in HORIZONS:
        res[f"follow_{h}"] = mean_ci(at(h), g)                # b's move AFTER a's move ended
        res[f"hit_{h}"] = mean_ci((at(h) > 0).astype(float), g)
    return res


# ---------------- main ----------------
def main():
    logp, z, scale, us, day = build()
    print("Overlapping minutes with crypto:", {m: int(np.isfinite(z[m]).sum()) for m in MARKETS})
    out_dir = HERE / "results" / f"market_links_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)

    # part 1
    fig, ax = plt.subplots(len(CRYPTO), len(MARKETS), figsize=(5 * len(MARKETS), 4 * len(CRYPTO)),
                           squeeze=False, sharex=True)
    for i, c in enumerate(CRYPTO):
        for j, m in enumerate(MARKETS):
            for mask, label in ((np.ones(len(us), bool), "all hours"), (us, "US cash hours")):
                cc = cross_corr(z, m, c, mask, day)
                line, = ax[i, j].plot(lags, cc[:, 0], marker=".", label=label)
                ax[i, j].fill_between(lags, cc[:, 1], cc[:, 2], color=line.get_color(), alpha=0.2)
            ax[i, j].axhline(0, color="k", lw=0.8)
            ax[i, j].axvline(0, color="gray", ls="--", lw=0.8)
            ax[i, j].set(title=f"{m} vs {c}", xlabel="lag tau (min); tau > 0: market leads")
            ax[i, j].grid(alpha=0.3)
            ax[i, j].legend(fontsize=8)
    fig.suptitle(f"Cross-correlation of volatility-scaled 1-min returns, {START} .. {END}")
    fig.tight_layout()
    fig.savefig(out_dir / "1_cross_correlation.png", dpi=150)
    print("Saved cross-correlation figure")

    # part 2
    pairs = [(m, c) for m in MARKETS for c in CRYPTO] + [(c, m) for c in CRYPTO for m in MARKETS]
    results = [r for r in (analyse_pair(logp, scale, day, a, b) for a, b in pairs) if r]
    rows = []
    for r in results:
        row = {"pair": r["pair"], "events": r["n_events"],
               "during (bp)": "{:+.1f} [{:+.1f}, {:+.1f}]".format(*(x[0] for x in r["during"]))}
        for h in HORIZONS:
            row[f"after {h} min (bp)"] = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*(x[0] for x in r[f"follow_{h}"]))
        h = HORIZONS[1]
        row[f"same direction after {h} min"] = "{:.2f} [{:.2f}, {:.2f}]".format(*(x[0] for x in r[f"hit_{h}"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    table.to_csv(out_dir / "events.csv", index=False)

    n = len(results)
    cols = 4
    fig, ax = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5 * cols, 3.8 * np.ceil(n / cols)),
                           squeeze=False)
    for k, r in enumerate(results):
        a_ = ax.flat[k]
        m, lo, hi = r["path_b"]
        a_.plot(r["u"], m, color="C0", label=f"{r['pair'].split(' -> ')[1]} (B)")
        a_.fill_between(r["u"], lo, hi, color="C0", alpha=0.2)
        twin = a_.twinx()
        twin.plot(r["u"], r["path_a"], color="C1", lw=1, alpha=0.7)
        twin.set_ylabel(f"{r['pair'].split(' -> ')[0]} (A), bp", color="C1", fontsize=8)
        a_.axhline(0, color="k", lw=0.8)
        a_.axvline(0, color="gray", ls="--", lw=0.8)
        a_.axvline(-DELTA_T, color="gray", ls=":", lw=0.8)
        a_.set(title=f"{r['pair']}  ({r['n_events']} events)", xlabel="minutes from end of A's move",
               ylabel="B, aligned with A's move (bp)")
        a_.grid(alpha=0.3)
    for k in range(n, ax.size):
        ax.flat[k].axis("off")
    thr = f"{PCT}%" if PCT is not None else f"{K_SIGMA} sigma"
    fig.suptitle(f"Strong moves (> {thr} within {DELTA_T} min) in A, and what B did around them. "
                 f"Positive after 0 = B follows A")
    fig.tight_layout()
    fig.savefig(out_dir / "2_event_study.png", dpi=150)
    print(f"\nSaved figures and events.csv to {out_dir}")


if __name__ == "__main__":
    main()