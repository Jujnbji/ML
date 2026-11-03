"""
Train a tiny GPT on candle tokens from a saved grid tokenizer.

Steps
-----
1. Load candles, turn them into tokens with the saved tokenizer (causal).
2. Cut each coin's token stream into contiguous segments per split
   (train / val / test by time). Windows never cross coins, splits or gaps.
3. Training loop: random windows -> predict next token at every position ->
   cross-entropy -> backpropagation -> AdamW update.
4. Evaluation on VAL: next-token cross-entropy vs. unigram and bigram
   baselines (all in bits per token), plus rank IC and decile spread of the
   predicted next-candle return.

Requirements:  pip install torch pandas pyarrow numpy plotly
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


import sys

ROOT = Path(__file__).resolve().parent.parent       # the ML folder
sys.path.insert(0, str(ROOT / "data"))              # lets Python find data/tokenizer.py

from simple_model import CandleGPT                  # your file name for the model
from tokenizer import GridTokenizer
import torch
# ===================== SETTINGS: edit these =====================
HERE = Path(__file__).parent
TOKENIZER_RUN = None                 # None = newest folder in tokenizer_runs/
DATA_DIR =  ROOT / "data" / "raw"   
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"
RESAMPLE = "5min"                    # None = keep 1m candles

DATA_START = "2024-10-01"
TRAIN_END  = "2026-06-30"
VAL_END    = "2026-08-31"            # test (after VAL_END) is not used here

PRESET = "laptop"                    # "laptop" (CPU, minutes) or "workstation" (GPU)
PRESETS = {
    "laptop":      dict(max_len=64,  d_model=64,  n_layers=2, n_heads=4, batch_size=32,
                        max_steps=3000,  eval_every=200, lr=1e-3),
    "workstation": dict(max_len=128, d_model=128, n_layers=4, n_heads=4, batch_size=128,
                        max_steps=20000, eval_every=500, lr=6e-4),
}
FACTORIZED   = True                  # per-feature embeddings (True) or one table (False)
DROPOUT      = 0.1
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 200
EVAL_BATCHES = 50                    # batches for the quick validation loss
PATIENCE     = 8                     # stop after this many evals without improvement
SEED         = 0
OUT_DIR      = HERE / "gpt_runs"
# ================================================================

AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
       "taker_buy_base_volume": "sum", "taker_buy_quote_volume": "sum"}
SPLITS = ("train", "val")
LN2 = math.log(2)


# ---------------- data ----------------
def find_tokenizer_run() -> Path:
    if TOKENIZER_RUN:
        return Path(TOKENIZER_RUN)
    runs = sorted((ROOT / "data" / "tokenizer_runs").iterdir())   # was: HERE / "tokenizer_runs"
    if not runs:
        raise SystemExit("No tokenizer runs found. Run train_simple_tokenizer.py first.")
    return runs[-1]


def load_coin(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet")
    df = df.loc[pd.Timestamp(DATA_START, tz="UTC"):]
    if RESAMPLE:
        df = (df.resample(RESAMPLE, closed="left", label="left")
                .agg({k: v for k, v in AGG.items() if k in df})
                .dropna(subset=["open"]))
    return df


def build_data(tok: GridTokenizer, window: int) -> dict:
    """Per split: one long token array, the matching next-candle returns,
    the valid window start positions, and the segment boundaries."""
    t_end, v_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")
    data = {s: {"tok": [], "ret": [], "starts": [], "segments": [], "n": 0} for s in SPLITS}
    expected_step = None
    for sym in SYMBOLS:
        df = load_coin(sym)
        tokens = tok.encode(df).to_numpy()
        ret_bp = (np.log(df["close"]).diff() * 1e4).to_numpy()     # return of each candle
        split = np.where(df.index < t_end, "train", np.where(df.index < v_end, "val", "test"))
        # a gap in the data (missing candles) also breaks a segment
        step = df.index.to_series().diff()
        expected_step = expected_step or step.mode().iloc[0]
        gap = (step != expected_step).to_numpy()
        for s in SPLITS:
            ok = (split == s) & (tokens >= 0) & np.isfinite(ret_bp)
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            breaks = np.flatnonzero((np.diff(idx) != 1) | gap[idx[1:]])
            seg_a = np.r_[idx[0], idx[breaks + 1]]
            seg_b = np.r_[idx[breaks], idx[-1]] + 1
            d = data[s]
            for a, b in zip(seg_a, seg_b):
                n = b - a
                d["tok"].append(tokens[a:b])
                d["ret"].append(ret_bp[a:b])
                d["segments"].append((d["n"], d["n"] + n))
                if n >= window + 1:                 # room for input + shifted target
                    d["starts"].append(d["n"] + np.arange(n - window))
                d["n"] += n
    for s in SPLITS:
        d = data[s]
        d["tok"] = np.concatenate(d["tok"]).astype(np.int64)
        d["ret"] = np.concatenate(d["ret"])
        d["starts"] = np.concatenate(d["starts"])
    return data


def get_batch(d: dict, batch_size: int, window: int, device, rng: np.random.Generator):
    """x = tokens[i : i+L],  y = tokens[i+1 : i+L+1]  (the same window shifted by one)."""
    starts = rng.choice(d["starts"], size=batch_size)
    offsets = np.arange(window)
    x = d["tok"][starts[:, None] + offsets]                       # (B, L)
    y = d["tok"][starts[:, None] + offsets + 1]                   # (B, L)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---------------- baselines ----------------
def ngram_baselines(data: dict, vocab: int, alpha: float = 100.0) -> dict:
    """Unigram and smoothed bigram cross-entropy on VAL, in bits per token."""
    tr, va = data["train"], data["val"]

    def pairs(d):
        prev = np.concatenate([d["tok"][a:b - 1] for a, b in d["segments"] if b - a > 1])
        nxt = np.concatenate([d["tok"][a + 1:b] for a, b in d["segments"] if b - a > 1])
        return prev, nxt

    uni = (np.bincount(tr["tok"], minlength=vocab) + 1.0)
    uni /= uni.sum()
    p_tr, n_tr = pairs(tr)
    big = np.zeros((vocab, vocab))
    np.add.at(big, (p_tr, n_tr), 1)
    big = (big + alpha * uni[None, :]) / (big.sum(1, keepdims=True) + alpha)
    p_va, n_va = pairs(va)
    return {"unigram_bits": float(-np.mean(np.log2(uni[n_va]))),
            "bigram_bits": float(-np.mean(np.log2(big[p_va, n_va])))}


# ---------------- evaluation ----------------
@torch.no_grad()
def quick_val_loss(model, d, cfg, device, n_batches) -> float:
    """Mean val loss (nats) on a FIXED set of random windows, so evals are comparable."""
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(d, cfg["batch_size"], cfg["max_len"], device, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def full_evaluation(model, d, cfg, device, ret_values: np.ndarray | None) -> dict:
    """Non-overlapping val windows: every val token is predicted about once."""
    model.eval()
    L = cfg["max_len"]
    starts = d["starts"][::L]
    nll, exp_ret, act_ret = [], [], []
    for i in range(0, len(starts), 256):
        s = starts[i:i + 256]
        offsets = np.arange(L)
        x = torch.from_numpy(d["tok"][s[:, None] + offsets]).to(device)
        y = d["tok"][s[:, None] + offsets + 1]
        logits, _ = model(x)
        logp = torch.log_softmax(logits, dim=-1).cpu().numpy()           # (B, L, V)
        nll.append(-np.take_along_axis(logp, y[..., None], axis=-1)[..., 0])
        if ret_values is not None:
            exp_ret.append(np.exp(logp) @ ret_values)                    # E[ret of next candle]
            act_ret.append(d["ret"][s[:, None] + offsets + 1])
    model.train()
    nll = np.concatenate(nll)                                            # (N, L)
    out = {"gpt_bits": float(nll.mean() / LN2),
           "gpt_bits_by_position": (nll.mean(0) / LN2).tolist()}
    if ret_values is not None:
        e, a = np.concatenate(exp_ret).ravel(), np.concatenate(act_ret).ravel()
        out["rank_ic"] = float(pd.Series(e).corr(pd.Series(a), method="spearman"))
        dec = pd.qcut(e, 10, labels=False, duplicates="drop")
        by = pd.Series(a).groupby(dec).mean()
        out["decile_spread_bps"] = float(by.iloc[-1] - by.iloc[0])
    return out


def lr_at(step: int, base_lr: float, max_steps: int) -> float:
    """Linear warmup, then cosine decay to 10 % of the base learning rate."""
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------- main ----------------
def main():
    cfg = PRESETS[PRESET]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  preset: {PRESET}  |  {cfg}")

    tok_run = find_tokenizer_run()
    tok = GridTokenizer.load(tok_run / "tokenizer.json")
    V = tok.vocab_size
    print(f"Tokenizer {tok_run.name}: features {tok.features}, vocabulary {V}")

    t0 = time.time()
    data = build_data(tok, cfg["max_len"])
    for s in SPLITS:
        print(f"  {s}: {len(data[s]['tok']):,} tokens, {len(data[s]['segments'])} segments, "
              f"{len(data[s]['starts']):,} possible windows")
    print(f"  data ready in {time.time() - t0:.0f} s")

    base = ngram_baselines(data, V)
    print(f"Baselines on VAL: unigram {base['unigram_bits']:.4f} bits, "
          f"bigram {base['bigram_bits']:.4f} bits per token")

    model = CandleGPT(V, tok.radix, max_len=cfg["max_len"], d_model=cfg["d_model"],
                      n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                      dropout=DROPOUT, factorized=FACTORIZED).to(device)
    print(f"Model: {model.n_params():,} parameters "
          f"({'factorized' if FACTORIZED else 'plain'} embedding)")

    # weight decay on matrices (linear layers, embedding tables), not on biases / LayerNorm
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95))

    run = OUT_DIR / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    history, best, bad_evals = [], float("inf"), 0
    t0 = time.time()

    # ======================= the training loop =======================
    for step in range(cfg["max_steps"]):
        lr = lr_at(step, cfg["lr"], cfg["max_steps"])
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = get_batch(data["train"], cfg["batch_size"], cfg["max_len"], device, rng)
        _, loss = model(x, y)                 # forward pass: predictions and loss
        opt.zero_grad(set_to_none=True)       # forget the previous step's gradients
        loss.backward()                       # backpropagation: d loss / d every weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                            # move every weight a little downhill

        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"] - 1:
            val = quick_val_loss(model, data["val"], cfg, device, EVAL_BATCHES)
            history.append({"step": step, "train_bits": loss.item() / LN2,
                            "val_bits": val / LN2, "lr": lr})
            print(f"step {step:6d} | lr {lr:.2e} | train {loss.item() / LN2:.4f} "
                  f"| val {val / LN2:.4f} bits | {time.time() - t0:.0f} s")
            if val < best:
                best, bad_evals = val, 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "factorized": FACTORIZED,
                            "dropout": DROPOUT, "tokenizer_run": str(tok_run)},
                           run / "best_model.pt")
            else:
                bad_evals += 1
                if bad_evals >= PATIENCE:
                    print("Early stopping: validation loss stopped improving.")
                    break
    # ==================================================================

    ckpt = torch.load(run / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    ret_values = tok.decode(np.arange(V))["ret"].to_numpy() if "ret" in tok.features else None
    ev = full_evaluation(model, data["val"], cfg, device, ret_values)

    print("\n=== Final evaluation on VAL (best checkpoint) ===")
    print(f"  unigram baseline  {base['unigram_bits']:.4f} bits/token")
    print(f"  bigram baseline   {base['bigram_bits']:.4f} bits/token")
    print(f"  GPT               {ev['gpt_bits']:.4f} bits/token "
          f"(gain over unigram {1000 * (base['unigram_bits'] - ev['gpt_bits']):.1f} mbit, "
          f"over bigram {1000 * (base['bigram_bits'] - ev['gpt_bits']):.1f} mbit)")
    pos = ev["gpt_bits_by_position"]
    print(f"  GPT by context length: 1 candle {pos[0]:.4f}, 8 {pos[7]:.4f}, "
          f"{len(pos)} {pos[-1]:.4f} bits")
    if "rank_ic" in ev:
        print(f"  next-candle return: rank IC {ev['rank_ic']:.4f}, "
              f"decile spread {ev['decile_spread_bps']:.2f} bp")

    (run / "results.json").write_text(json.dumps({
        "settings": {"symbols": SYMBOLS, "interval": INTERVAL, "resample": RESAMPLE,
                     "preset": PRESET, "cfg": cfg, "factorized": FACTORIZED,
                     "tokenizer_run": str(tok_run)},
        "baselines": base, "evaluation": ev, "history": history}, indent=2))
    print(f"Saved model and results to {run}")


if __name__ == "__main__":
    main()