"""
Fit the grid tokenizer and measure how PREDICTIVE its tokens are.

Pipeline
--------
1. Load candles for several coins, optionally resample (e.g. 1m -> 5m).
2. Split by time:  train | val | test   (test is never touched unless you ask).
3. Fit the tokenizer on TRAIN (fixed values, or Lloyd-refined).
4. Encode every coin into token sequences (causal).
5. "Judge": an n-gram model (lookup table) predicts a FIXED target from the
   last m tokens. The target does not depend on the tokenizer, so different
   tokenizers are comparable:
       target = forward log return over H candles, bucketed into quantile
                bins whose edges are computed once on train.
   Score = information gain on VAL in bits per prediction
         = H(target) - cross-entropy(target | token history).
   > 0: the tokens carry information about the future.  ~0: they don't.
6. Also reports a rank IC and a decile spread (in basis points) for the
   predicted mean forward return, to compare against trading costs.

Requirements:  pip install pandas pyarrow numpy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tokenizer import GridTokenizer, compute_features

# ===================== SETTINGS: edit these =====================
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"                     # file interval
RESAMPLE = None                  # None, or coarser bars ("5min", "15min", "1h")
DATA_DIR = Path(__file__).parent / "raw"

TRAIN_START = "2024-10-01"
TRAIN_END   = "2026-06-30"          # train:  TRAIN_START .. TRAIN_END
VAL_END     = "2026-08-31"          # val:    TRAIN_END   .. VAL_END
EVALUATE_TEST = False               # test:   VAL_END     .. end   (only at the very end!)

# Your token values, per feature (in sigma units, except relvol = log-ratio, buy = share - 0.5).
# Vocabulary size = product of list lengths. Keep it in the hundreds to low thousands.
GRID = {
    "ret":    [-2.5, -1.2, -0.4, 0.4, 1.2, 2.5],
    "upper":  [0.1, 0.6, 1.5],
    "lower":  [0.1, 0.6, 1.5],
    "relvol": [-0.7, 0.0, 0.8],
    "buy":    [-0.15, 0.0, 0.15],
}
VOL_WINDOW = 60                     # past candles for sigma and mean volume
FIT_METHOD = "fixed"                # "fixed" = your values as given, "lloyd" = refine them

# Judge
HORIZON         = 3                 # predict the return over the next H candles
TARGET_BINS     = 5                 # fixed quantile bins of that return
CONTEXT_LENGTHS = [1, 2, 3]         # n-gram context: last m tokens
ALPHA           = 500.0             # smoothing (pseudo-counts). Market signals are tiny,
                                    # so strong smoothing is needed or noise dominates
FEE_BPS         = 20                # round-trip fees, for comparison
OUT_DIR         = Path(__file__).parent / "tokenizer_runs"
# ================================================================

AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
       "taker_buy_base_volume": "sum", "taker_buy_quote_volume": "sum"}


# ---------------- data ----------------
def load_coin(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet")
    df = df.loc[pd.Timestamp(TRAIN_START, tz="UTC"):]
    if RESAMPLE:
        df = (df.resample(RESAMPLE, closed="left", label="left")
                .agg({k: v for k, v in AGG.items() if k in df})
                .dropna(subset=["open"]))
    return df


def split_of(index: pd.DatetimeIndex) -> np.ndarray:
    t_end, v_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")
    return np.where(index < t_end, "train", np.where(index < v_end, "val", "test"))


# ---------------- n-gram judge ----------------
def contexts(tokens: np.ndarray, m: int, vocab: int) -> np.ndarray:
    """Context id of the last m tokens ending at t (int64), -1 if incomplete."""
    ctx = np.zeros(len(tokens), dtype=np.int64)
    valid = np.ones(len(tokens), dtype=bool)
    for j in range(m):
        shifted = np.full(len(tokens), -1, dtype=np.int64)
        shifted[j:] = tokens[:len(tokens) - j]
        valid &= shifted >= 0
        ctx = ctx * vocab + np.maximum(shifted, 0)
    ctx[~valid] = -1
    return ctx


def judge(table: pd.DataFrame, m: int, vocab: int, eval_split: str) -> dict:
    """table: columns coin, split, token, fwd (bps), y (target bin)."""
    parts = []
    for _, g in table.groupby("coin", sort=False):       # contexts never cross coins
        g = g.copy()
        g["ctx"] = contexts(g["token"].to_numpy(), m, vocab)
        parts.append(g)
    t = pd.concat(parts)
    t = t[(t["ctx"] >= 0) & t["y"].notna() & t["fwd"].notna()]
    tr, ev = t[t["split"] == "train"], t[t["split"] == eval_split]
    K = TARGET_BINS
    y_tr, y_ev = tr["y"].to_numpy(int), ev["y"].to_numpy(int)

    # unconditional target distribution (baseline)
    p0 = np.bincount(y_tr, minlength=K) / len(y_tr)
    mu0 = tr["fwd"].mean()

    # counts n(ctx), n(ctx, y), sum of fwd per ctx on train
    keys, inv = np.unique(tr["ctx"].to_numpy(), return_inverse=True)
    n_ctx = np.bincount(inv)
    n_ctx_y = np.zeros((len(keys), K))
    np.add.at(n_ctx_y, (inv, y_tr), 1)
    sum_fwd = np.bincount(inv, weights=tr["fwd"].to_numpy())

    # look up eval contexts (unseen contexts fall back to the baseline)
    pos = np.searchsorted(keys, ev["ctx"].to_numpy())
    pos = np.clip(pos, 0, len(keys) - 1)
    seen = keys[pos] == ev["ctx"].to_numpy()
    n = np.where(seen, n_ctx[pos], 0)
    n_y = np.where(seen, n_ctx_y[pos, y_ev], 0)
    s = np.where(seen, sum_fwd[pos], 0)

    # Dirichlet-smoothed probabilities and shrunk mean forecast
    p = (n_y + ALPHA * p0[y_ev]) / (n + ALPHA)
    pred = (s + ALPHA * mu0) / (n + ALPHA)

    ce = -np.mean(np.log2(p))
    h0 = -np.mean(np.log2(p0[y_ev]))
    fwd_ev = ev["fwd"].to_numpy()
    ic = pd.Series(pred).corr(pd.Series(fwd_ev), method="spearman")
    dec = pd.qcut(pred, 10, labels=False, duplicates="drop")
    by_dec = pd.Series(fwd_ev).groupby(dec).mean()
    return {
        "m": m,
        "info_gain_bits": float(h0 - ce),
        "baseline_entropy_bits": float(h0),
        "eval_contexts_seen_in_train": float(seen.mean()),
        "rank_ic": float(ic),
        "decile_spread_bps": float(by_dec.iloc[-1] - by_dec.iloc[0]) if len(by_dec) > 1 else 0.0,
        "n_eval": int(len(ev)),
    }


# ---------------- main ----------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tok = GridTokenizer(GRID, window=VOL_WINDOW)
    print(f"Tokenizer: features {tok.features}, values per feature {tok.radix}, "
          f"vocabulary {tok.vocab_size}")

    coins = {s: load_coin(s) for s in SYMBOLS}

    # 1) fit on TRAIN features only (pooled over coins)
    train_feats = pd.concat([
        compute_features(df, VOL_WINDOW)[split_of(df.index) == "train"]
        for df in coins.values()])
    stats = tok.fit(train_feats, method=FIT_METHOD)
    print(f"\nFit ({FIT_METHOD}) on {len(train_feats):,} train candles:")
    for f in tok.features:
        print(f"  {f:7s} values = {np.round(tok.grid[f], 3).tolist()}"
              f"   quantization MSE = {stats['mse_per_feature'][f]:.3f}")
    print(f"  tokens used: {stats['tokens_used']}/{stats['vocab_size']}, "
          f"entropy {stats['entropy_bits']:.2f} of max {stats['max_entropy_bits']:.2f} bits, "
          f"perplexity {stats['perplexity']:.0f}")

    # 2) tokens + fixed target per coin
    rows = []
    for sym, df in coins.items():
        lc = np.log(df["close"])
        fwd = (lc.shift(-HORIZON) - lc) * 1e4                     # bps
        split = split_of(df.index)
        # purge: train rows whose target window reaches into val
        purge = (split == "train") & (np.roll(split, -HORIZON) != "train")
        fwd[purge] = np.nan
        rows.append(pd.DataFrame({"coin": sym, "split": split,
                                  "token": tok.encode(df).to_numpy(),
                                  "fwd": fwd.to_numpy()}, index=df.index))
    table = pd.concat(rows)
    train_fwd = table.loc[table["split"] == "train", "fwd"].dropna()
    edges = np.quantile(train_fwd, np.linspace(0, 1, TARGET_BINS + 1)[1:-1])
    table["y"] = np.where(table["fwd"].notna(), np.searchsorted(edges, table["fwd"]), np.nan)

    # 3) judge
    eval_split = "test" if EVALUATE_TEST else "val"
    print(f"\nJudge on {eval_split.upper()}: target = {HORIZON}-candle forward return "
          f"in {TARGET_BINS} bins (edges in bps: {np.round(edges, 1).tolist()})")
    print(f"{'m':>3} {'info gain (mbit)':>17} {'seen ctx':>9} {'rank IC':>8} "
          f"{'decile spread (bp)':>19}")
    results = []
    for m in CONTEXT_LENGTHS:
        r = judge(table, m, tok.vocab_size, eval_split)
        results.append(r)
        print(f"{m:>3} {1000 * r['info_gain_bits']:>17.3f} {r['eval_contexts_seen_in_train']:>9.1%} "
              f"{r['rank_ic']:>8.4f} {r['decile_spread_bps']:>19.2f}")
    print(f"\nFor comparison: round-trip fees = {FEE_BPS} bp. "
          f"A decile spread far below that is not tradeable on its own.")

    # 4) save
    run = OUT_DIR / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir()
    tok.save(run / "tokenizer.json")
    (run / "results.json").write_text(json.dumps({
        "settings": {"symbols": SYMBOLS, "interval": INTERVAL, "resample": RESAMPLE,
                     "fit_method": FIT_METHOD, "grid_given": GRID, "vol_window": VOL_WINDOW,
                     "horizon": HORIZON, "target_bins": TARGET_BINS, "alpha": ALPHA,
                     "eval_split": eval_split},
        "fit_stats": stats, "judge": results}, indent=2))
    print(f"Saved tokenizer and results to {run}")


if __name__ == "__main__":
    main()