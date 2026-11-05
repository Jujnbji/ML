"""
GPT with extra context added to the input:

    H = XE + P  [+ D_time[slot]]  [+ D_day[weekday]]  [+ D_coin[coin]]  [+ dvol * W_dvol]

Each extra term is switched on or off. The extra tables start at exactly zero, so a
fresh model computes the same as the plain GPT and only uses the context once
training finds it useful. Place next to candle_gpt.py.
"""

import torch
import torch.nn as nn

from GPT1_0 import Block


class ContextGPT(nn.Module):
    def __init__(self, V, C, T, n_heads, n_layers, n_coins=1, n_slots=96,
                 use_time=False, use_weekday=False, use_coin=False, use_dailyvol=False):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)                                  # E
        self.pos_emb = nn.Embedding(T, C)                                  # P
        self.time_emb = nn.Embedding(n_slots, C) if use_time else None     # D_time
        self.day_emb = nn.Embedding(7, C) if use_weekday else None         # D_day
        self.coin_emb = nn.Embedding(n_coins, C) if use_coin else None     # D_coin
        self.dvol_proj = nn.Linear(1, C, bias=False) if use_dailyvol else None   # W_dvol
        self.blocks = nn.ModuleList(Block(C, T, n_heads) for _ in range(n_layers))
        self.ln_f, self.out = nn.LayerNorm(C), nn.Linear(C, V)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for extra in (self.time_emb, self.day_emb, self.coin_emb, self.dvol_proj):
            if extra is not None:
                nn.init.zeros_(extra.weight)                               # start = plain GPT

    def forward(self, idx, ctx):
        """idx: (B, T) token ids. ctx: dict of (B, T) tensors: slot, day, coin, dvol."""
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        if self.time_emb is not None:
            x = x + self.time_emb(ctx["slot"])
        if self.day_emb is not None:
            x = x + self.day_emb(ctx["day"])
        if self.coin_emb is not None:
            x = x + self.coin_emb(ctx["coin"])
        if self.dvol_proj is not None:
            x = x + self.dvol_proj(ctx["dvol"].unsqueeze(-1))
        for block in self.blocks:
            x = block(x)
        return self.out(self.ln_f(x))