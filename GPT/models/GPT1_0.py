"""
The GPT model classes (identical to step5_gpt.py), in their own file so that
other scripts can load trained weights without retraining.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, C, T, n_heads):
        super().__init__()
        self.h, self.d = n_heads, C // n_heads
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))
        self.att = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (z.view(B, T, self.h, self.d).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(scores, dim=-1)
        self.att = att.detach()
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(C, 4 * C), nn.Linear(4 * C, C)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, C, T, n_heads):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(C), MultiHeadAttention(C, T, n_heads)
        self.ln2, self.mlp = nn.LayerNorm(C), MLP(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, C, T, n_heads, n_layers):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(V, C), nn.Embedding(T, C)
        self.blocks = nn.ModuleList(Block(C, T, n_heads) for _ in range(n_layers))
        self.ln_f, self.out = nn.LayerNorm(C), nn.Linear(C, V)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.out(self.ln_f(x))