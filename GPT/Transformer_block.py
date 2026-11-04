"""
Step 4: transformer blocks = attention + MLP, with LayerNorm and residuals, stacked N times.

    H = E[b_t] + P[t]
    repeat N times:
        H <- H + Attention(LN1(H))      communicate between positions
        H <- H + MLP(LN2(H))            compute within each position
    logits = LN_f(H) W_O + c
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

CONTEXT    = 64                                        # T
EMBED_DIM  = 32                                        # C
N_LAYERS   = 4                                         # N: number of blocks
BATCH_SIZE = 256
STEPS      = 6000
LR         = 1e-3
EVAL_EVERY = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(0)
LN2 = math.log(2)
(HERE / "runs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. Data (identical to step 3)
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
    gap = np.r_[True, np.diff(df.index.values) != one_candle]
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
            if b - a > CONTEXT:
                d["starts"].append(d["n"] + np.arange(b - a - CONTEXT))
            d["tok"].append(t[a:b])
            d["n"] += b - a

for d in data.values():
    d["tok"] = torch.tensor(np.concatenate(d["tok"]), device=device)
    d["starts"] = torch.tensor(np.concatenate(d["starts"]), device=device)
offsets = torch.arange(CONTEXT + 1, device=device)


def get_batch(split, starts=None):
    d = data[split]
    if starts is None:
        starts = d["starts"][torch.randint(len(d["starts"]), (BATCH_SIZE,), device=device)]
    w = d["tok"][starts[:, None] + offsets]
    return w[:, :-1], w[:, 1:]


# ---------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------
class Attention(nn.Module):
    """One causal attention head (the same as in step 3), returns A @ U."""

    def __init__(self, C, T):
        super().__init__()
        self.query = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))
        self.att = None                                           # kept for visualization

    def forward(self, h):                                         # h: (B, T, C)
        T = h.shape[1]
        q, k, v = self.query(h), self.key(h), self.value(h)
        scores = q @ k.transpose(1, 2) / math.sqrt(h.shape[-1])  # (B, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(scores, dim=-1)
        self.att = att.detach()
        return att @ v                                            # (B, T, C)


class MLP(nn.Module):
    """Per position: expand to 4C, GELU, compress back to C."""

    def __init__(self, C):
        super().__init__()
        self.fc1 = nn.Linear(C, 4 * C)                            # W_1, b_1
        self.fc2 = nn.Linear(4 * C, C)                            # W_2, b_2

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


class Block(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = Attention(C, T)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP(C)

    def forward(self, h):
        h = h + self.attn(self.ln1(h))                            # communicate
        h = h + self.mlp(self.ln2(h))                             # compute
        return h


class Step4Model(nn.Module):
    def __init__(self, V, C, T, n_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(T, C)
        self.blocks = nn.ModuleList(Block(C, T) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(C)
        self.out = nn.Linear(C, V)
        for m in self.modules():                                  # LayerNorm keeps its 1 / 0 start
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        h = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln_f(h))                             # (B, T, V)


model = Step4Model(V, EMBED_DIM, CONTEXT, N_LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def loss_fn(x, y, reduction="mean"):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction=reduction)


@torch.no_grad()
def evaluate():
    starts = data["val"]["starts"][::CONTEXT]
    total = torch.zeros(CONTEXT, device=device)
    for i in range(0, len(starts), 512):
        x, y = get_batch("val", starts[i:i + 512])
        total += loss_fn(x, y, "none").view(len(x), CONTEXT).sum(0)
    per_position = (total / len(starts) / LN2).cpu().numpy()
    return per_position.mean(), per_position


freq = torch.bincount(data["train"]["tok"], minlength=V).float() + 1
unigram_bits = -torch.log2(freq / freq.sum())[data["val"]["tok"]].mean().item()
print(f"{len(data['train']['starts']):,} train windows | {N_LAYERS} blocks, "
      f"{sum(p.numel() for p in model.parameters()):,} parameters | unigram {unigram_bits:.3f} bits")

# ---------------------------------------------------------------------
# 3. Training loop (unchanged)
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
print("\nVal bits by amount of context:")
for p in (1, 2, 4, 8, 16, 32, CONTEXT):
    print(f"  {p:3d} candles seen: {per_position[p - 1]:.4f} bits")

name = f"step4_C{EMBED_DIM}_T{CONTEXT}_L{N_LAYERS}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pt"
torch.save({"model": model.state_dict(), "tokenizer_run": str(TOKENIZER_RUN),
            "settings": {"embed_dim": EMBED_DIM, "context": CONTEXT, "n_layers": N_LAYERS,
                         "steps": STEPS, "lr": LR, "batch_size": BATCH_SIZE, "symbols": SYMBOLS},
            "val_bits": float(val), "per_position": per_position.tolist(),
            "unigram_bits": unigram_bits}, HERE / "runs" / name)
print(f"saved runs/{name}")