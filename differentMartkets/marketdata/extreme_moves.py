"""
Hypothesis test (stated in advance, tested on data where it was NOT found):

    "After a move of at least K_MIN sigma within DELTA_T minutes, altcoins rise
     over the next HOLD minutes."  (up-moves continue, down-moves bounce back)

Simulated trade per event: buy at the OPEN of the candle after the move, sell at the
close HOLD minutes later, minus COST_BP. Compared with trades entered at random times
(same coins, same period). Events that happen at nearly the same time on several coins
form one cluster; the bootstrap resamples clusters, so one crash day cannot count 20 times.

Reports mean, MEDIAN and hit rate (share of profitable trades) with 95 % intervals,
for up-moves, down-moves and both together.

Run:  python test_extreme_moves.py          (place anywhere inside the ML project)
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
SOURCE      = "1m"                   # "1m": data/raw/*_1m (6 coins); "5m": data5min/raw/*_5m (20 coins)
EXCLUDE     = ["BTCUSDT", "ETHUSDT"] # the hypothesis is about altcoins
PERIOD      = "val"                  # "train" (where it was found), "val" (the test), "test" (only once, at the end!)
PERIODS     = {"train": ("2024-10-01", "2026-06-30"), "val": ("2026-06-30", "2026-08-31"),
               "test": ("2026-08-31", "2026-11-01")}
K_MIN       = 6.0                    # strong move: at least this many sigma ...
DELTA_T     = 5                      # ... within this many minutes
HOLD        = 60                     # minutes the trade is held
VOL_WINDOW  = 60                     # minutes of history for sigma
COST_BP     = 25                     # round-trip cost (fees + spread + slippage)
CLUSTER_MIN = 60                     # events closer than this (any coin) form one cluster
RANDOM_PER_EVENT = 50                # random comparison trades per event
N_BOOT      = 2000
SEED        = 0
# ======================================================================

rng = np.random.default_rng(SEED)
if PERIOD == "test":
    print("WARNING: using the test period. Do this once, at the very end.")


def load(path, step):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    start, end = PERIODS[PERIOD]
    lead = pd.Timedelta(minutes=VOL_WINDOW + DELTA_T + 10)
    return df.loc[pd.Timestamp(start, tz="UTC") - lead: pd.Timestamp(end, tz="UTC") + pd.Timedelta(minutes=HOLD + 10)]


def trades_for_coin(df, step):
    """Event trades and random trades for one coin. step = candle length in minutes."""
    dt, hold, volw = max(1, DELTA_T // step), max(1, HOLD // step), max(2, VOL_WINDOW // step)
    t = df.index
    o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
    lc = np.log(c)
    lr = np.r_[np.nan, np.diff(lc)]
    sigma = pd.Series(lr).rolling(volw).std().shift(dt).to_numpy()   # known BEFORE the move
    move = lc - np.r_[np.full(dt, np.nan), lc[:-dt]]
    minutes = (t.values.astype("datetime64[m]").astype(np.int64))
    start, end = (pd.Timestamp(x, tz="UTC") for x in PERIODS[PERIOD])

    i = np.arange(volw + dt + 1, len(df) - hold - 1)
    span_ok = (minutes[i + hold] - minutes[i - dt]) == (hold + dt) * step   # no gaps around the trade
    in_period = (t[i] >= start) & (t[i] < end)
    i = i[span_ok & in_period & np.isfinite(sigma[i])]
    trade = lambda j: (np.log(c[j + hold]) - np.log(o[j + 1])) * 1e4      # enter next open, exit later

    strong = np.abs(move[i]) >= K_MIN * sigma[i] * np.sqrt(dt)
    events, last = [], -10 ** 9
    for j in i[strong]:
        if j - last > dt + hold:                                          # one event per episode
            events.append(j)
            last = j
    events = np.array(events, dtype=int)
    rand = rng.choice(i, size=len(events) * RANDOM_PER_EVENT) if len(events) else np.array([], int)
    ev = pd.DataFrame({"time": t[events], "direction": np.where(move[events] > 0, "up", "down"),
                       "move_bp": move[events] * 1e4, "gross_bp": trade(events)})
    return ev, pd.DataFrame({"time": t[rand], "gross_bp": trade(rand)})


def boot_ci(values, clusters, fn):
    """Statistic with a 95 % interval from resampling whole clusters."""
    _, inv = np.unique(clusters, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    boots = [fn(values[np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])])
             for _ in range(N_BOOT)]
    return fn(values), *np.percentile(boots, [2.5, 97.5])


def describe(net, clusters):
    f = lambda v: "{:+.1f} [{:+.1f}, {:+.1f}]".format(*v)
    p = lambda v: "{:.0%} [{:.0%}, {:.0%}]".format(*v)
    return {"trades": len(net), "clusters": len(np.unique(clusters)),
            "mean net (bp)": f(boot_ci(net, clusters, np.mean)),
            "median net (bp)": f(boot_ci(net, clusters, np.median)),
            "profitable": p(boot_ci(net, clusters, lambda v: np.mean(v > 0)))}


def main():
    folder, step = (ROOT / "data" / "raw", 1) if SOURCE == "1m" else (ROOT / "data5min" / "raw", 5)
    files = sorted(folder.glob(f"*_{SOURCE}.parquet"))
    events, randoms = [], []
    for f in files:
        sym = f.name.rsplit("_", 1)[0]
        if sym in EXCLUDE:
            continue
        ev, rd = trades_for_coin(load(f, step), step)
        ev["coin"], rd["coin"] = sym, sym
        events.append(ev)
        randoms.append(rd)
        print(f"  {sym}: {len(ev)} events")
    ev = pd.concat(events).sort_values("time").reset_index(drop=True)
    rd = pd.concat(randoms).reset_index(drop=True)
    if ev.empty:
        raise SystemExit("No events found.")

    # clusters: events closer than CLUSTER_MIN minutes (on any coin) belong together
    gap = ev["time"].diff().dt.total_seconds().fillna(1e9) / 60
    ev["cluster"] = np.cumsum(gap > CLUSTER_MIN)
    rd["cluster"] = (rd["time"].values.astype("datetime64[h]").astype(np.int64))   # hours as clusters
    ev["net_bp"], rd["net_bp"] = ev["gross_bp"] - COST_BP, rd["gross_bp"] - COST_BP

    rows = []
    for name, sub in (("after up-moves", ev[ev.direction == "up"]),
                      ("after down-moves", ev[ev.direction == "down"]),
                      ("after all extreme moves", ev),
                      ("random entries", rd)):
        if len(sub):
            rows.append({"trades on": name, **describe(sub["net_bp"].to_numpy(), sub["cluster"].to_numpy())})
    table = pd.DataFrame(rows)
    start, end = PERIODS[PERIOD]
    print(f"\nHypothesis: long for {HOLD} min after >= {K_MIN} sigma moves in {DELTA_T} min | "
          f"{PERIOD} period {start} .. {end} | cost {COST_BP} bp")
    print(table.to_string(index=False))
    biggest = ev.reindex(ev["net_bp"].abs().sort_values(ascending=False).index).head(5)
    print("\nLargest single trades (do they dominate?):")
    print(biggest[["time", "coin", "direction", "move_bp", "net_bp"]].to_string(index=False))

    out = HERE / "results" / f"extreme_moves_{PERIOD}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    ev.to_csv(out / "trades.csv", index=False)
    table.to_csv(out / "summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    lim = np.percentile(np.abs(np.r_[ev["net_bp"], rd["net_bp"]]), 99)
    bins = np.linspace(-lim, lim, 60)
    ax[0].hist(rd["net_bp"], bins=bins, density=True, alpha=0.5, label="random entries")
    ax[0].hist(ev["net_bp"], bins=bins, density=True, alpha=0.6, label="after extreme moves")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set(title="Distribution of net trade results", xlabel="net result (bp)", ylabel="density")
    ax[0].legend()
    for d, color in (("up", "C2"), ("down", "C3")):
        s = ev[ev.direction == d]
        ax[1].plot(s["time"], s["net_bp"].cumsum(), color=color, label=f"after {d}-moves")
    ax[1].plot(ev["time"], ev["net_bp"].cumsum(), color="k", label="all")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set(title="Cumulative net result over time (a few jumps = a few days carry it)",
              xlabel="time", ylabel="cumulative bp")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Long {HOLD} min after >= {K_MIN} sigma moves | {PERIOD} | cost {COST_BP} bp")
    fig.tight_layout()
    fig.savefig(out / "extreme_moves.png", dpi=150)
    print(f"\nSaved trades.csv, summary.csv, extreme_moves.png to {out}")


if __name__ == "__main__":
    main()