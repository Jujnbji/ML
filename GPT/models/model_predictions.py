"""
Let a trained GPT continue a real chart for N candles and plot it against reality.

1. Take the last T real tokens before START as context.
2. Sample the next token, append it, repeat N times (for N_PATHS independent paths).
3. Turn tokens back into candles: decode grid values, undo the sigma / volume scaling.
4. Plot: real candles, the median and 10-90 % band of all paths, and one example path.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "data"))
from tokenizer import GridTokenizer
from GPT1_0 import GPT

# ============================== SETTINGS ==============================
CHECKPOINT  = None                   # None = newest runs/gpt_*.pt, or e.g. HERE / "runs" / "gpt_....pt"
SYMBOL      = "BTCUSDT"
INTERVAL    = "1m"                   # must be the interval the model was trained on
START       = "2026-08-10 14:00"     # UTC; first predicted candle. Use the val/test period!
N_FUTURE    = 30                     # candles to predict
N_PATHS     = 100                    # sampled futures
TEMPERATURE = 1.0                    # <1: more cautious, >1: more random
SHOW_HISTORY = 60                    # real candles shown before START
SEED        = 0
# ======================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------- load model and tokenizer ----------------
path = Path(CHECKPOINT) if CHECKPOINT else sorted((HERE.parent / "runs").glob("gpt_*.pt"))[-1]
ckpt = torch.load(path, map_location=device)
s = ckpt["settings"]
tok = GridTokenizer.load(Path(ckpt["tokenizer_run"]) / "tokenizer.json")
T, W = s["context"], tok.window
model = GPT(tok.vocab_size, s["embed_dim"], T, s["n_heads"], s["n_layers"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"model {path.name}: val {ckpt['val_bits']:.4f} bits")

# ---------------- real data around START ----------------
df = pd.read_parquet(ROOT / "data" / "raw" / f"{SYMBOL}_{INTERVAL}.parquet")
i = df.index.searchsorted(pd.Timestamp(START, tz="UTC"))
hist = df.iloc[i - (T + W + 10): i]                      # enough history for sigma + context
real_future = df.iloc[i: i + N_FUTURE]
context = tok.encode(hist).to_numpy()[-T:]
if (context < 0).any():
    raise SystemExit("Context contains missing data. Pick another START.")

# ---------------- 1 + 2: sample N_PATHS token futures ----------------
x = torch.tensor(context, device=device).repeat(N_PATHS, 1)     # (P, T)
with torch.no_grad():
    for n in range(N_FUTURE):
        logits = model(x[:, -T:])[:, -1, :] / TEMPERATURE       # prediction at the last position
        probs = F.softmax(logits, dim=-1)                       # (P, V)
        if n == 0:
            first_probs = probs[0].cpu().numpy()
        x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)  # draw one token per path
generated = x[:, T:].cpu().numpy()                              # (P, N)

actual_first = tok.encode(df.iloc[i - (T + W + 10): i + 1]).iloc[-1]
print("\nModel's top 5 guesses for the first candle:")
for j in np.argsort(-first_probs)[:5]:
    vals = tok.decode([j]).iloc[0]
    print(f"  p={first_probs[j]:.3f}  token {j:4d}: " +
          ", ".join(f"{f} {vals[f]:+.2g}" for f in tok.features))
print(f"Actual first token: {actual_first} (model gave it p={first_probs[actual_first]:.3f})"
      if actual_first >= 0 else "Actual first token: missing data")

# ---------------- 3: tokens -> candles ----------------
dec = tok.decode(generated.ravel())
feat = {f: (dec[f].to_numpy().reshape(N_PATHS, N_FUTURE) if f in dec else
            np.zeros((N_PATHS, N_FUTURE))) for f in ("ret", "upper", "lower", "relvol")}
past_ret = np.diff(np.log(hist["close"].to_numpy()))[-W:]
past_vol = hist["volume"].to_numpy()[-W:]
ohlcv = np.zeros((N_PATHS, N_FUTURE, 5))
for p in range(N_PATHS):
    rets, vols, close = list(past_ret), list(past_vol), hist["close"].iloc[-1]
    for n in range(N_FUTURE):
        sigma = np.std(rets[-W:], ddof=1)                       # same sigma as the tokenizer
        o, c = close, close * np.exp(feat["ret"][p, n] * sigma)
        h = max(o, c) * np.exp(feat["upper"][p, n] * sigma)
        lo = min(o, c) * np.exp(-feat["lower"][p, n] * sigma)
        v = np.mean(vols[-W:]) * np.exp(feat["relvol"][p, n])
        ohlcv[p, n] = (o, h, lo, c, v)
        rets.append(np.log(c / o))
        vols.append(v)
        close = c

# ---------------- 4: plot ----------------
step = pd.Timedelta(INTERVAL.replace("m", "min"))
times = pd.Timestamp(START, tz="UTC") + step * np.arange(N_FUTURE)
shown = pd.concat([hist.iloc[-SHOW_HISTORY:], real_future])
q10, q50, q90 = np.percentile(ohlcv[:, :, 3], [10, 50, 90], axis=0)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=shown.index, open=shown["open"], high=shown["high"],
                             low=shown["low"], close=shown["close"], name="real"))
fig.add_trace(go.Scatter(x=times, y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=times, y=q10, fill="tonexty", fillcolor="rgba(66,165,245,0.25)",
                         line=dict(width=0), name="10-90 % of paths"))
fig.add_trace(go.Scatter(x=times, y=q50, line=dict(color="#1565c0", width=2),
                         name="median predicted close"))
ex = ohlcv[0]
fig.add_trace(go.Candlestick(x=times, open=ex[:, 0], high=ex[:, 1], low=ex[:, 2], close=ex[:, 3],
                             name="one sampled path", opacity=0.6,
                             increasing_line_color="#7e57c2", decreasing_line_color="#7e57c2"))
fig.add_vline(x=times[0], line=dict(color="gray", dash="dash"))
fig.update_layout(title=f"{SYMBOL} {INTERVAL}: real vs. {N_PATHS} sampled futures from "
                        f"{path.name} (temperature {TEMPERATURE})",
                  height=650, xaxis_rangeslider_visible=False)
fig.show()