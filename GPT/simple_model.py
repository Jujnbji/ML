"""
A tiny GPT for candle tokens, written out explicitly for learning.

Shapes used in the comments:
    B = batch size (number of windows processed in parallel)
    T = sequence length (number of candles in a window)
    C = d_model (width of every hidden vector)
    H = number of attention heads,  D = C // H  (width per head)
    V = vocabulary size (number of tokens)

Requirements:  pip install torch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Token embedding: token id -> vector of length C
# ----------------------------------------------------------------------------
class FactorizedEmbedding(nn.Module):
    """e(token) = E_ret[i1] + E_upper[i2] + ... (one small table per feature).

    A token id is a mixed-radix number, e.g. bases (6, 3, 3, 3, 3). We recover
    the per-feature indices like reading the digits of a number, look up one
    vector per feature and add them. Rare tokens are built from common parts.
    """

    def __init__(self, radix: list[int], d_model: int):
        super().__init__()
        self.radix = radix
        self.tables = nn.ModuleList(nn.Embedding(base, d_model) for base in radix)

    def digits(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """(B, T) token ids -> list of (B, T) feature indices, in feature order."""
        out, rest = [], idx
        for base in reversed(self.radix):          # last feature = last digit
            out.append(rest % base)
            rest = rest // base
        return out[::-1]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return sum(table(d) for table, d in zip(self.tables, self.digits(idx)))


# ----------------------------------------------------------------------------
# 2. Causal self-attention: every candle looks at itself and earlier candles
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V in one matrix
        self.proj = nn.Linear(d_model, d_model)      # W_O, mixes the heads again
        self.drop = nn.Dropout(dropout)
        # Lower-triangular matrix of ones: mask[i, j] = 1 if j <= i (past or present).
        # register_buffer: part of the module (moves to GPU with it), but not trained.
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).bool())
        self.last_attention = None                   # kept for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                       # 3 x (B, T, C)
        # split C into H heads of width D and move the head axis forward
        q = q.view(B, T, self.h, self.d).transpose(1, 2)            # (B, H, T, D)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)        # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # no future
        att = F.softmax(scores, dim=-1)                             # rows sum to 1
        self.last_attention = att.detach()
        att = self.drop(att)

        y = att @ v                                                 # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # heads side by side
        return self.drop(self.proj(y))


# ----------------------------------------------------------------------------
# 3. Feed-forward network: processes each candle position on its own
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.out = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.out(F.gelu(self.fc(x))))


# ----------------------------------------------------------------------------
# 4. Transformer block = attention + MLP, each with LayerNorm and a residual
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # communicate between candles
        x = x + self.mlp(self.ln2(x))     # think about each candle
        return x


# ----------------------------------------------------------------------------
# 5. The full model
# ----------------------------------------------------------------------------
class CandleGPT(nn.Module):
    def __init__(self, vocab_size: int, radix: list[int], max_len: int = 64,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, factorized: bool = True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = (FactorizedEmbedding(radix, d_model) if factorized
                        else nn.Embedding(vocab_size, d_model))
        self.pos_emb = nn.Embedding(max_len, d_model)    # one vector per position
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, max_len, dropout)
                                    for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)       # vector -> one score per token
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: (B, T) the NEXT token at each position."""
        B, T = idx.shape
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        pos = torch.arange(T, device=idx.device)                 # 0, 1, ..., T-1
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                         # (B, T, V)

        loss = None
        if targets is not None:
            # cross-entropy averaged over all B*T positions (in nats)
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())