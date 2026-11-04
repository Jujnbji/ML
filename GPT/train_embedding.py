"""
Step 2: train the embedding by predicting the next token.

    b_t --embedding E--> e_t --linear layer (W, c)--> logits z_t --softmax--> p(b_{t+1} | b_t)

Loss: cross-entropy  -log p(true next token), reported in bits.
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
ROOT = HERE.parent                                     # the ML folder
sys.path.insert(0, str(ROOT / "data"))
from tokenizer import GridTokenizer

TOKENIZER_RUN = sorted((ROOT / "data" / "tokenizer_runs").iterdir())[-1]
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"
DATA_START, TRAIN_END, VAL_END = "2024-10-01", "2026-06-30", "2026-08-31"

EMBED_DIM  = 8                                         # C
BATCH_SIZE = 1024
STEPS      = 3000
LR         = 1e-2
EVAL_EVERY = 250

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "")
torch.manual_seed(0)
LN2 = math.log(2)

# ---------------------------------------------------------------------
# 1. Data: pairs (current token, next token), split by time
# ---------------------------------------------------------------------
tok = GridTokenizer.load(TOKENIZER_RUN / "tokenizer.json")
V = tok.vocab_size
one_candle = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
train_end, val_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")

pairs = {"train": ([], []), "val": ([], [])}
for symbol in SYMBOLS:
    df = pd.read_parquet(ROOT / "data" / "raw" / f"{symbol}_{INTERVAL}.parquet")
    df = df.loc[pd.Timestamp(DATA_START, tz="UTC"):]
    t = tok.encode(df).to_numpy()
    current, nxt = t[:-1], t[1:]
    ok = (current >= 0) & (nxt >= 0) & (np.diff(df.index.values) == one_candle)
    time = df.index[1:]                                # time of the candle we predict
    for split, in_split in (("train", time < train_end),
                            ("val", (time >= train_end) & (time < val_end))):
        keep = ok & in_split
        pairs[split][0].append(current[keep])
        pairs[split][1].append(nxt[keep])

X_train, Y_train, X_val, Y_val = (
    torch.tensor(np.concatenate(pairs[s][k]), device=device)
    for s in ("train", "val") for k in (0, 1))

# ---------------------------------------------------------------------
# 2. Model: embedding table E (V x C) and output layer W (V x C), c (V)
# ---------------------------------------------------------------------
emb = nn.Embedding(V, EMBED_DIM).to(device)
out = nn.Linear(EMBED_DIM, V).to(device)
nn.init.normal_(emb.weight, mean=0.0, std=0.02)
nn.init.normal_(out.weight, mean=0.0, std=0.02)
nn.init.zeros_(out.bias)

opt = torch.optim.Adam(list(emb.parameters()) + list(out.parameters()), lr=LR)


def loss_fn(x, y, reduction="mean"):
    logits = out(emb(x))                               # (N, C) -> (N, V)
    return F.cross_entropy(logits, y, reduction=reduction)


@torch.no_grad()
def val_bits():
    total = sum(loss_fn(X_val[i:i + 65536], Y_val[i:i + 65536], "sum").item()
                for i in range(0, len(X_val), 65536))
    return total / len(X_val) / LN2


# baseline: predict the train token frequencies, ignoring the current candle
freq = torch.bincount(Y_train, minlength=V).float() + 1
unigram_bits = -torch.log2(freq / freq.sum())[Y_val].mean().item()
print(f"{len(X_train):,} train pairs, {len(X_val):,} val pairs | "
      f"uniform {math.log2(V):.3f} bits, unigram {unigram_bits:.3f} bits")

# ---------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------
for step in range(STEPS + 1):
    i = torch.randint(len(X_train), (BATCH_SIZE,), device=device)   # random batch
    loss = loss_fn(X_train[i], Y_train[i])            # forward pass
    opt.zero_grad()                                    # clear old gradients
    loss.backward()                                    # gradients of every parameter
    opt.step()                                         # update E, W, c

    if step % EVAL_EVERY == 0:
        v = val_bits()
        print(f"step {step:5d} | train {loss.item() / LN2:.4f} | val {v:.4f} bits "
              f"| below unigram {1000 * (unigram_bits - v):+.1f} mbit")

torch.save({"emb": emb.state_dict(), "out": out.state_dict(),
            "tokenizer_run": str(TOKENIZER_RUN)}, HERE / "step2_embedding.pt")