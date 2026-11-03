"""
Grid tokenizer for candles: every candle -> one integer token.

How it works
------------
1. From each candle, compute scale-free features using ONLY the past
   (strictly causal, no look-ahead):
       ret    log(close_t / close_{t-1}) / sigma_t
       upper  upper wick  log(high / max(open, close)) / sigma_t
       lower  lower wick  log(min(open, close) / low) / sigma_t
       body   log(close / open) / sigma_t
       relvol log(volume_t / mean volume of the previous W candles)
       buy    taker-buy share - 0.5   (> 0: aggressive buyers dominated)
   sigma_t is the std of log returns over the W candles BEFORE t, so
   "ret = 2" means "a 2-sigma move relative to recent conditions".

2. For each feature you choose, you specify a list of values (the grid).
   Every feature value is snapped to the nearest grid value.

3. The per-feature indices are combined into one token id
   (mixed radix, like digits of a number). Vocabulary size = product of
   the number of values per feature.

Optional "training" (fit): refine your grid values with Lloyd's
algorithm (1D k-means per feature), starting from the values you gave.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("ret", "upper", "lower", "body", "relvol", "buy")


def compute_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Causal, scale-free features for one coin. First `window` rows are NaN."""
    lc = np.log(df["close"])
    r = lc.diff()
    sigma = r.rolling(window).std().shift(1)          # past only, excludes t
    sigma = sigma.where(sigma > 0)

    top = np.maximum(df["open"], df["close"])
    bot = np.minimum(df["open"], df["close"])
    vol = df["volume"]
    past_vol = vol.rolling(window).mean().shift(1)

    f = pd.DataFrame(index=df.index)
    f["ret"] = r / sigma
    f["upper"] = np.log(df["high"] / top) / sigma
    f["lower"] = np.log(bot / df["low"]) / sigma
    f["body"] = np.log(df["close"] / df["open"]) / sigma
    f["relvol"] = np.log(vol.clip(lower=1e-12) / past_vol)
    if "taker_buy_base_volume" in df:
        f["buy"] = (df["taker_buy_base_volume"] / vol.replace(0, np.nan) - 0.5).fillna(0.0)
    f = f.replace([np.inf, -np.inf], np.nan)
    f.iloc[:window + 1] = np.nan                       # warm-up
    return f


class GridTokenizer:
    def __init__(self, grid: dict[str, list[float]], window: int = 60):
        unknown = set(grid) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown features {unknown}. Choose from {FEATURES}.")
        self.features = list(grid)
        self.grid = {k: np.sort(np.asarray(v, dtype=float)) for k, v in grid.items()}
        self.window = window

    # ---------- basic properties ----------
    @property
    def radix(self) -> list[int]:
        return [len(self.grid[f]) for f in self.features]

    @property
    def vocab_size(self) -> int:
        return int(np.prod(self.radix))

    def _edges(self, f):
        v = self.grid[f]
        return (v[:-1] + v[1:]) / 2                    # bin edges = midpoints

    # ---------- training ----------
    def fit(self, feats: pd.DataFrame, method: str = "fixed", n_iter: int = 30) -> dict:
        """method = "fixed": keep your values.
           method = "lloyd": refine values by Lloyd's algorithm (min. squared error),
                             starting from your values. Their number stays the same."""
        if method == "lloyd":
            for f in self.features:
                x = feats[f].dropna().to_numpy()
                v = self.grid[f].copy()
                for _ in range(n_iter):
                    idx = np.searchsorted((v[:-1] + v[1:]) / 2, x)
                    sums = np.bincount(idx, weights=x, minlength=len(v))
                    cnts = np.bincount(idx, minlength=len(v))
                    new = np.where(cnts > 0, sums / np.maximum(cnts, 1), v)
                    if np.allclose(new, v):
                        break
                    v = np.sort(new)
                self.grid[f] = v
        elif method != "fixed":
            raise ValueError('method must be "fixed" or "lloyd"')
        return self.stats(feats)

    # ---------- encoding / decoding ----------
    def feature_indices(self, feats: pd.DataFrame) -> np.ndarray:
        """(n, n_features) array of grid indices; -1 where a feature is missing."""
        out = np.full((len(feats), len(self.features)), -1, dtype=np.int64)
        for j, f in enumerate(self.features):
            x = feats[f].to_numpy()
            ok = np.isfinite(x)
            out[ok, j] = np.searchsorted(self._edges(f), x[ok])
        return out

    def encode(self, df: pd.DataFrame) -> pd.Series:
        """Candles of ONE coin -> token ids (int). -1 during warm-up / missing data."""
        idx = self.feature_indices(compute_features(df, self.window))
        valid = (idx >= 0).all(axis=1)
        tokens = np.zeros(len(df), dtype=np.int64)
        for j, base in enumerate(self.radix):
            tokens = tokens * base + idx[:, j]
        tokens[~valid] = -1
        return pd.Series(tokens, index=df.index, name="token")

    def decode(self, tokens) -> pd.DataFrame:
        """Token ids -> the grid value of each feature (the 'reconstruction')."""
        t = np.asarray(tokens, dtype=np.int64).copy()
        cols = {}
        for f, base in reversed(list(zip(self.features, self.radix))):
            cols[f] = self.grid[f][t % base]
            t //= base
        return pd.DataFrame({f: cols[f] for f in self.features})

    # ---------- diagnostics ----------
    def stats(self, feats: pd.DataFrame) -> dict:
        idx = self.feature_indices(feats)
        valid = (idx >= 0).all(axis=1)
        tokens = np.zeros(valid.sum(), dtype=np.int64)
        for j, base in enumerate(self.radix):
            tokens = tokens * base + idx[valid, j]
        counts = np.bincount(tokens, minlength=self.vocab_size)
        p = counts[counts > 0] / counts.sum()
        entropy = float(-(p * np.log2(p)).sum())

        mse = {}
        for j, f in enumerate(self.features):
            x = feats[f].to_numpy()[valid]
            mse[f] = float(np.mean((x - self.grid[f][idx[valid, j]]) ** 2))
        return {
            "vocab_size": self.vocab_size,
            "tokens_used": int((counts > 0).sum()),
            "entropy_bits": entropy,
            "perplexity": 2 ** entropy,                 # effective number of tokens
            "max_entropy_bits": float(np.log2(self.vocab_size)),
            "mse_per_feature": mse,                    # quantization error (sigma units)
        }

    # ---------- save / load ----------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "window": self.window,
            "grid": {f: self.grid[f].tolist() for f in self.features},
        }, indent=2))

    @classmethod
    def load(cls, path) -> "GridTokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["grid"], d["window"])