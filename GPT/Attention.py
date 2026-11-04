"""
Step 3: let every candle look at the candles before it.

    window of tokens b_1..b_T
      -> h_t = E[b_t] + P[t]                       token vector + position vector
      -> one causal attention head:                 h_t <- h_t + sum_{s<=t} a_ts v_s
      -> output layer -> p(b_{t+1} | b_1..b_t)     at every position t

Everything (E, P, attention, output layer) is trained together.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "data"))
from tokenizer import GridTokenizer

TOKENIZER_RUN = sorted((ROOT / "data" / "tokenizer_runs").iterdir())[-1]
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"
DATA_START, TRAIN_END, VAL_END = "2024-10-01", "2026-06-30", "2026-08-31"

CONTEXT    = 64                                        # T: candles per window
EMBED_DIM  = 32                                        # C
BATCH_SIZE = 256
STEPS      = 5000
LR         = 3e-3
EVAL_EVERY = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(0)
LN2 = math.log(2)
(HERE / "runs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. Data: one long token array per split + the start positions of all
#    windows that lie inside one unbroken stretch of candles
# ---------------------------------------------------------------------
tok = GridTokenizer.load(TOKENIZER_RUN / "tokenizer.json")
V = tok.vocab_size
one_candle = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
train_end, val_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")

data = {s: {"tok": [], "starts": [], "n": 0} for s in ("train", "val")}
for symbol in SYMBOLS:
    df = pd.read_parquet(ROOT / "data" / "raw" / f"{symbol}_{INTERVAL}.parquet")
    df = df.loc[pd.Timestamp(DATA_START, tz="UTC"):]
    t = tok.encode(df).to_numpy()
    gap = np.r_[True, np.diff(df.index.values) != one_candle]   # candle not 1 step after previous
    for split, in_split in (("train", df.index < train_end),
                            ("val", (df.index >= train_end) & (df.index < val_end))):
        idx = np.flatnonzero(in_split & (t >= 0))
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero((np.diff(idx) != 1) | gap[idx[1:]])
        seg_start = np.r_[idx[0], idx[breaks + 1]]
        seg_end = np.r_[idx[breaks], idx[-1]] + 1
        d = data[split]
        for a, b in zip(seg_start, seg_end):
            if b - a > CONTEXT:                                   # room for T inputs + 1 target
                d["starts"].append(d["n"] + np.arange(b - a - CONTEXT))
            d["tok"].append(t[a:b])
            d["n"] += b - a

for d in data.values():
    d["tok"] = torch.tensor(np.concatenate(d["tok"]), device=device)
    d["starts"] = torch.tensor(np.concatenate(d["starts"]), device=device)
offsets = torch.arange(CONTEXT + 1, device=device)


def get_batch(split, starts=None):
    d = data[split]
    if starts is None:                                            # random windows
        starts = d["starts"][torch.randint(len(d["starts"]), (BATCH_SIZE,), device=device)]
    w = d["tok"][starts[:, None] + offsets]                       # (B, T+1)
    return w[:, :-1], w[:, 1:]                                    # input, target = shifted by 1


# ---------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------
class Step3Model(nn.Module):
    def __init__(self, V, C, T):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)                         # E: V x C
        self.pos_emb = nn.Embedding(T, C)                         # P: T x C
        self.query = nn.Linear(C, C, bias=False)                  # W_Q
        self.key = nn.Linear(C, C, bias=False)                    # W_K
        self.value = nn.Linear(C, C, bias=False)                  # W_V
        self.out = nn.Linear(C, V)                                # W, c
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))  # (B, T, C)
        q, k, v = self.query(h), self.key(h), self.value(h)                       # (B, T, C)
        scores = q @ k.transpose(1, 2) / math.sqrt(h.shape[-1])                   # (B, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))            # no future
        att = F.softmax(scores, dim=-1)                                           # rows sum to 1
        h = h + att @ v                                   # own vector + what it gathered
        return self.out(h), att                                                   # (B, T, V)


model = Step3Model(V, EMBED_DIM, CONTEXT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def loss_fn(x, y, reduction="mean"):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction=reduction)


@torch.no_grad()
def evaluate():
    """Val loss in bits, overall and by position in the window (= amount of context)."""
    starts = data["val"]["starts"][::CONTEXT]                     # non-overlapping windows
    total = torch.zeros(CONTEXT, device=device)
    for i in range(0, len(starts), 512):
        x, y = get_batch("val", starts[i:i + 512])
        total += loss_fn(x, y, "none").view(len(x), CONTEXT).sum(0)
    per_position = (total / len(starts) / LN2).cpu().numpy()
    return per_position.mean(), per_position


freq = torch.bincount(data["train"]["tok"], minlength=V).float() + 1
unigram_bits = -torch.log2(freq / freq.sum())[data["val"]["tok"]].mean().item()
print(f"{len(data['train']['starts']):,} train windows, {len(data['val']['starts']):,} val windows "
      f"| {sum(p.numel() for p in model.parameters()):,} parameters | unigram {unigram_bits:.3f} bits")

# ---------------------------------------------------------------------
# 3. Training loop (same five lines as step 2)
# ---------------------------------------------------------------------
for step in range(STEPS + 1):
    x, y = get_batch("train")
    loss = loss_fn(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % EVAL_EVERY == 0:
        val, _ = evaluate()
        print(f"step {step:5d} | train {loss.item() / LN2:.4f} | val {val:.4f} bits "
              f"| below unigram {1000 * (unigram_bits - val):+.1f} mbit")

val, per_position = evaluate()
print("\nVal bits by amount of context (position in the window):")
for p in (1, 2, 4, 8, 16, 32, CONTEXT):
    print(f"  {p:3d} candles seen: {per_position[p - 1]:.4f} bits")

name = f"step3_C{EMBED_DIM}_T{CONTEXT}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pt"
torch.save({"model": model.state_dict(), "tokenizer_run": str(TOKENIZER_RUN),
            "settings": {"embed_dim": EMBED_DIM, "context": CONTEXT, "steps": STEPS, "lr": LR,
                         "batch_size": BATCH_SIZE, "symbols": SYMBOLS},
            "val_bits": float(val), "per_position": per_position.tolist(),
            "unigram_bits": unigram_bits}, HERE / "runs" / name)
print(f"saved runs/{name}")