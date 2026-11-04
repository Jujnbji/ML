"""
Step 1: give every token a vector (the embedding table).

    candles x_t  --tokenizer-->  ids b_t  --embedding-->  vectors e_t = E[b_t]

E is a V x C matrix: one row (a C-dimensional vector) per token.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent          # the ML folder
sys.path.insert(0, str(ROOT / "data"))
from tokenizer import GridTokenizer

TOKENIZER_RUN = sorted((ROOT / "data" / "tokenizer_runs").iterdir())[-1]
SYMBOL, INTERVAL = "BTCUSDT", "1m"
START, N_CANDLES = "2026-07-15 14:00", 3
EMBED_DIM = 8                                          # C

torch.manual_seed(0)

# 1. candles -> token ids
tok = GridTokenizer.load(TOKENIZER_RUN / "tokenizer.json")
df = pd.read_parquet(ROOT / "data" / "raw" / f"{SYMBOL}_{INTERVAL}.parquet")
i = df.index.searchsorted(pd.Timestamp(START, tz="UTC"))
tokens = tok.encode(df.iloc[i - 200 : i + N_CANDLES]).iloc[-N_CANDLES:]   # 200 candles of history
ids = torch.tensor(tokens.to_numpy())      
print(ids)            # shape (N,)

# 2. the embedding table E: V x C, every entry ~ N(0, 0.02^2)
emb = nn.Embedding(tok.vocab_size, EMBED_DIM)
nn.init.normal_(emb.weight, mean=0.0, std=0.02)

# 3. lookup: row b_t of E for every id
vectors = emb(ids)         

print(emb.weight.shape, emb.weight)                            # shape (N, C)