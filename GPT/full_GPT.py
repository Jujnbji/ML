"""
Step 5: a complete, compact GPT on candle tokens.

    tokens b_1..b_T
    H = E[b_t] + P[t]                                   [A] embedding + position
    repeat N times:                                     [D] block
        H <- H + MultiHeadAttention(LN1(H))             [B] h heads + W_O
        H <- H + MLP(LN2(H))                            [C] 4C, GELU, back to C
    logits = LN_f(H) W_out + c  ->  cross-entropy       [E] output + loss
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

# ============================== SETTINGS ==============================
TOKENIZER_RUN = sorted((ROOT / "data" / "tokenizer_runs").iterdir())[-1]
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"
DATA_START, TRAIN_END, VAL_END = "2024-10-01", "2026-06-30", "2026-08-31"

CONTEXT, EMBED_DIM, N_HEADS, N_LAYERS = 64, 64, 4, 4      # T, C, h, N
BATCH_SIZE, STEPS, LR, WARMUP, EVAL_EVERY = 256, 10000, 1e-3, 300, 500
# ======================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
LN2 = math.log(2)
(HERE / "runs").mkdir(exist_ok=True)

# ---------------- [0] data: token windows that never cross a gap ----------------
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
        d = data[split]
        for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + 1):
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
    w = d["tok"][starts[:, None] + offsets]                       # (B, T+1)
    return w[:, :-1], w[:, 1:]                                    # inputs, targets


# ---------------- [B] multi-head causal attention ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, C, T, n_heads):
        super().__init__()
        self.h, self.d = n_heads, C // n_heads                    # d = d_k = d_v = C / h
        self.qkv = nn.Linear(C, 3 * C, bias=False)                # all W_Q^i, W_K^i, W_V^i
        self.proj = nn.Linear(C, C)                               # W_O
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))
        self.att = None

    def forward(self, x):                                         # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                     # 3 x (B, T, C)
        q, k, v = (z.view(B, T, self.h, self.d).transpose(1, 2) for z in (q, k, v))  # (B, h, T, d)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)      # (B, h, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(scores, dim=-1)                           # one A per head
        self.att = att.detach()
        y = (att @ v).transpose(1, 2).reshape(B, T, C)            # concat heads -> (B, T, C)
        return self.proj(y)                                       # mix heads with W_O


# ---------------- [C] MLP ----------------
class MLP(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(C, 4 * C), nn.Linear(4 * C, C)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------- [D] transformer block ----------------
class Block(nn.Module):
    def __init__(self, C, T, n_heads):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(C), MultiHeadAttention(C, T, n_heads)
        self.ln2, self.mlp = nn.LayerNorm(C), MLP(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                            # communicate
        return x + self.mlp(self.ln2(x))                          # compute


# ---------------- [A] + [E] the full GPT ----------------
class GPT(nn.Module):
    def __init__(self, V, C, T, n_heads, n_layers):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(V, C), nn.Embedding(T, C)    # [A] E, P
        self.blocks = nn.ModuleList(Block(C, T, n_heads) for _ in range(n_layers))
        self.ln_f, self.out = nn.LayerNorm(C), nn.Linear(C, V)                 # [E]
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):                                       # idx: (B, T)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.out(self.ln_f(x))                             # logits (B, T, V)


model = GPT(V, EMBED_DIM, CONTEXT, N_HEADS, N_LAYERS).to(device)


def loss_fn(x, y, reduction="mean"):                              # [E] cross-entropy
    return F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction=reduction)


# ---------------- [F] training ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def lr_at(step):                                                  # warmup, then cosine decay
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP))))


@torch.no_grad()
def evaluate():                                                   # val bits per position
    model.eval()
    starts = data["val"]["starts"][::CONTEXT]
    total = torch.zeros(CONTEXT, device=device)
    for i in range(0, len(starts), 512):
        x, y = get_batch("val", starts[i:i + 512])
        total += loss_fn(x, y, "none").view(len(x), CONTEXT).sum(0)
    model.train()
    per_position = (total / len(starts) / LN2).cpu().numpy()
    return per_position.mean(), per_position


freq = torch.bincount(data["train"]["tok"], minlength=V).float() + 1
unigram_bits = -torch.log2(freq / freq.sum())[data["val"]["tok"]].mean().item()
print(f"device {device} | {len(data['train']['starts']):,} train windows | "
      f"{sum(p.numel() for p in model.parameters()):,} parameters | unigram {unigram_bits:.3f} bits")

for step in range(STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch("train")
    loss = loss_fn(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # cap rare huge gradients
    opt.step()
    if step % EVAL_EVERY == 0:
        val, _ = evaluate()
        print(f"step {step:5d} | lr {lr_at(step):.1e} | train {loss.item() / LN2:.4f} | "
              f"val {val:.4f} bits | below unigram {1000 * (unigram_bits - val):+.1f} mbit")

# ---------------- [G] final evaluation + save ----------------
val, per_position = evaluate()
print("\nVal bits by amount of context:")
for p in (1, 2, 4, 8, 16, 32, CONTEXT):
    print(f"  {p:3d} candles seen: {per_position[p - 1]:.4f} bits")

name = (f"gpt_C{EMBED_DIM}_H{N_HEADS}_L{N_LAYERS}_T{CONTEXT}_"
        f"{pd.Timestamp.now():%Y%m%d_%H%M%S}.pt")
torch.save({"model": model.state_dict(), "tokenizer_run": str(TOKENIZER_RUN),
            "settings": {"context": CONTEXT, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
                         "n_layers": N_LAYERS, "steps": STEPS, "lr": LR, "symbols": SYMBOLS},
            "val_bits": float(val), "per_position": per_position.tolist(),
            "unigram_bits": unigram_bits}, HERE / "runs" / name)
print(f"saved runs/{name}")