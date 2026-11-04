"""
Step 2b: look at what the embedding learned.

Figure 1  All token vectors projected to 2D (PCA), one panel per feature,
          coloured by that feature's value. Point size = how often the token occurs.
Figure 2  Per feature: cosine similarity between the average vectors of its values
          (e.g. is the "ret = +2.5" direction similar to the "ret = -2.5" direction?).
Printed   Nearest neighbours of the most common tokens.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "data"))
from tokenizer import GridTokenizer

CHECKPOINT = HERE / "step2_embedding.pt"
COUNT_SYMBOL, INTERVAL = "BTCUSDT", "1m"         # used only to count token frequencies
COUNT_START, COUNT_END = "2024-10-01", "2026-06-30"
N_NEIGHBOURS = 5

# ---------------- load ----------------
ckpt = torch.load(CHECKPOINT, map_location="cpu")
tok = GridTokenizer.load(Path(ckpt["tokenizer_run"]) / "tokenizer.json")
E = ckpt["emb"]["weight"].numpy()                 # (V, C)
V, C = E.shape
meaning = tok.decode(np.arange(V))                # (V, n_features): grid value per token

df = pd.read_parquet(ROOT / "data" / "raw" / f"{COUNT_SYMBOL}_{INTERVAL}.parquet")
df = df.loc[pd.Timestamp(COUNT_START, tz="UTC"):pd.Timestamp(COUNT_END, tz="UTC")]
ids = tok.encode(df).to_numpy()
counts = np.bincount(ids[ids >= 0], minlength=V)

# ---------------- PCA to 2D ----------------
# Centre the vectors, then find the two directions of largest spread (SVD).
Ec = E - E.mean(axis=0)
U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
xy = Ec @ Vt[:2].T                                # (V, 2) coordinates
explained = S ** 2 / (S ** 2).sum()

# ---------------- figure 1: scatter per feature ----------------
feats = tok.features
fig = make_subplots(rows=1, cols=len(feats), subplot_titles=feats, horizontal_spacing=0.03)
size = 3 + 12 * np.log1p(counts) / np.log1p(counts.max())
hover = [", ".join(f"{f} {meaning[f][j]:+.2g}" for f in feats) + f"  (n={counts[j]:,})"
         for j in range(V)]
for col, f in enumerate(feats, start=1):
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=size, color=meaning[f], colorscale="RdBu", showscale=False,
                    line=dict(width=0.3, color="black"))), row=1, col=col)
fig.update_layout(height=450, showlegend=False,
                  title=f"Token vectors in 2D (PCA of C={C}; PC1 {explained[0]:.0%}, "
                        f"PC2 {explained[1]:.0%} of variance). Colour = feature value "
                        f"(red low, blue high), size = frequency")
fig.show()

# ---------------- figure 2: similarity between feature values ----------------
def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

fig = make_subplots(rows=1, cols=len(feats), subplot_titles=feats, horizontal_spacing=0.05)
for col, f in enumerate(feats, start=1):
    values = tok.grid[f]
    centroids = np.stack([
        np.average(Ec[meaning[f] == val], axis=0, weights=counts[meaning[f] == val] + 1)
        for val in values])                       # frequency-weighted mean vector per value
    sim = unit(centroids) @ unit(centroids).T     # cosine similarity matrix (K, K)
    labels = [f"{v:+.2g}" for v in values]
    fig.add_trace(go.Heatmap(z=sim, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu",
                             text=np.round(sim, 2), texttemplate="%{text}",
                             showscale=(col == len(feats))), row=1, col=col)
    fig.update_yaxes(autorange="reversed", row=1, col=col)
fig.update_layout(height=420, title="Cosine similarity between the average vectors of each "
                                    "feature value (1 = same direction, -1 = opposite)")
fig.show()

# ---------------- nearest neighbours ----------------
cos = unit(E) @ unit(E).T
for j in np.argsort(-counts)[:5]:
    # only tokens seen often enough: never-seen tokens keep their random start vectors
    near = [k for k in np.argsort(-cos[j]) if k != j and counts[k] >= 200][:N_NEIGHBOURS]
    print(f"\nToken {j} ({hover[j]})")
    for k in near:
        print(f"   cos {cos[j, k]:+.2f}  token {k:4d}: {hover[k]}")