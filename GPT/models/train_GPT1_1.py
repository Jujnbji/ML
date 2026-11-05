"""
Train the GPT with switchable extra context (time of day, weekday, coin, daily volatility).

Run it several times with different switches (and seeds) and compare the final scorecard:
    val bits/token              next-token loss (lower = better), also by amount of context
    next-candle rank IC, AUC    quick direction check from the predicted distribution
Place next to candle_gpt.py and context_gpt.py.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "tokenizer.py").exists())
sys.path[:0] = [str(ROOT / "data"), str(HERE)]
from tokenizer import GridTokenizer
from GPT1_1_more_input import ContextGPT

# ============================== SWITCHES ==============================
USE_TIME      = False                 # D_time: 15-minute slot of the day (96 rows)
USE_WEEKDAY   = False                # D_day: Monday .. Sunday (7 rows)
USE_COIN      = False                # D_coin: which coin (one row per coin)
USE_DAILY_VOL = False                # standardized log volatility of the last 24 hours
SEED          = 2                    # change to measure run-to-run spread
# ============================== SETTINGS ==============================
TOKENIZER_RUN = sorted((ROOT / "data" / "tokenizer_runs").iterdir())[-1]
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
INTERVAL = "1m"
DATA_START, TRAIN_END, VAL_END = "2024-10-01", "2026-06-30", "2026-08-31"
CONTEXT, EMBED_DIM, N_HEADS, N_LAYERS = 64, 64, 4, 4
BATCH_SIZE, STEPS, LR, WARMUP, EVAL_EVERY = 256, 10000, 1e-3, 300, 1000
# ======================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)
LN2 = math.log(2)
(HERE.parent / "runs").mkdir(exist_ok=True)
CTX_KEYS = ("slot", "day", "coin", "dvol")

# ---------------- data: tokens + context per candle ----------------
tok = GridTokenizer.load(TOKENIZER_RUN / "tokenizer.json")
V = tok.vocab_size
one_candle = pd.Timedelta(INTERVAL.replace("m", "min")).to_timedelta64()
train_end, val_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")

fields = ("tok", "ret", *CTX_KEYS)
data = {s: {**{f: [] for f in fields}, "starts": [], "n": 0} for s in ("train", "val")}
for c, symbol in enumerate(SYMBOLS):
    df = pd.read_parquet(ROOT / "data" / "raw" / f"{symbol}_{INTERVAL}.parquet")
    df = df.loc[pd.Timestamp(DATA_START, tz="UTC"):]
    per_candle = {
        "tok": tok.encode(df).to_numpy(),
        "ret": (np.log(df["close"]).diff() * 1e4).to_numpy(),                 # return in bp
        "slot": ((df.index.hour * 60 + df.index.minute) // 15).to_numpy(),
        "day": df.index.dayofweek.to_numpy(),
        "coin": np.full(len(df), c),
        "dvol": np.log(np.log(df["close"]).diff().rolling(1440, min_periods=720).std())
                 .replace([np.inf, -np.inf], np.nan).to_numpy(),                 # log 24h volatility
    }
    gap = np.r_[True, np.diff(df.index.values) != one_candle]
    for split, in_split in (("train", df.index < train_end),
                            ("val", (df.index >= train_end) & (df.index < val_end))):
        idx = np.flatnonzero(in_split & (per_candle["tok"] >= 0))
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero((np.diff(idx) != 1) | gap[idx[1:]])
        d = data[split]
        for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + 1):
            if b - a > CONTEXT:
                d["starts"].append(d["n"] + np.arange(b - a - CONTEXT))
            for f in fields:
                d[f].append(per_candle[f][a:b])
            d["n"] += b - a
    print(f"  {symbol} done")

for d in data.values():
    for f in fields:
        d[f] = np.concatenate(d[f])
dv_mean = np.nanmean(data["train"]["dvol"])                  # standardize with TRAIN statistics
dv_std = np.nanstd(data["train"]["dvol"])
for d in data.values():
    d["dvol"] = np.nan_to_num((d["dvol"] - dv_mean) / dv_std)   # missing (first day) -> average
    d["starts"] = torch.tensor(np.concatenate(d["starts"]), device=device)
    for f, dtype in (("tok", torch.long), ("slot", torch.long), ("day", torch.long),
                     ("coin", torch.long), ("dvol", torch.float32), ("ret", torch.float32)):
        d[f] = torch.tensor(d[f], dtype=dtype, device=device)
offsets = torch.arange(CONTEXT + 1, device=device)


def get_batch(split, starts=None):
    d = data[split]
    if starts is None:
        starts = d["starts"][torch.randint(len(d["starts"]), (BATCH_SIZE,), device=device)]
    pos = starts[:, None] + offsets                               # (B, T+1) positions
    x, y = d["tok"][pos[:, :-1]], d["tok"][pos[:, 1:]]
    ctx = {k: d[k][pos[:, :-1]] for k in CTX_KEYS}                # context of the input candles
    return x, y, ctx, d["ret"][pos[:, 1:]]                        # + real return of the targets


# ---------------- model ----------------
model = ContextGPT(V, EMBED_DIM, CONTEXT, N_HEADS, N_LAYERS, n_coins=len(SYMBOLS),
                   use_time=USE_TIME, use_weekday=USE_WEEKDAY, use_coin=USE_COIN,
                   use_dailyvol=USE_DAILY_VOL).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
ret_value = torch.tensor(tok.decode(np.arange(V))["ret"].to_numpy(), dtype=torch.float32,
                         device=device) if "ret" in tok.features else None


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP))))


def ranks(a):
    return pd.Series(a).rank().to_numpy()


@torch.no_grad()
def evaluate(full=False):
    """Val bits per position; with full=True also the quick direction check."""
    model.eval()
    starts = data["val"]["starts"][::CONTEXT]
    total = torch.zeros(CONTEXT, device=device)
    exp_ret, p_up, actual = [], [], []
    for i in range(0, len(starts), 512):
        x, y, ctx, real = get_batch("val", starts[i:i + 512])
        logits = model(x, ctx)
        nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
        total += nll.view(len(x), CONTEXT).sum(0)
        if full and ret_value is not None:
            probs = F.softmax(logits, dim=-1)
            exp_ret.append((probs @ ret_value).flatten().cpu())                 # E[ret of next candle]
            p_up.append((probs @ (ret_value > 0).float()).flatten().cpu())      # P(next candle up)
            actual.append(real.flatten().cpu())
    model.train()
    per_position = (total / len(starts) / LN2).cpu().numpy()
    out = {"val_bits": float(per_position.mean()), "per_position": per_position.tolist()}
    if exp_ret:
        e, p, a = (torch.cat(z).numpy() for z in (exp_ret, p_up, actual))
        up = a > 0
        out["next_candle_ic"] = float(np.corrcoef(ranks(e), ranks(a))[0, 1])
        rk = ranks(p)
        out["next_candle_auc"] = float((rk[up].sum() - up.sum() * (up.sum() + 1) / 2)
                                       / (up.sum() * (~up).sum()))
    return out


freq = torch.bincount(data["train"]["tok"], minlength=V).float() + 1
unigram_bits = -torch.log2(freq / freq.sum())[data["val"]["tok"]].mean().item()
switches = {"time": USE_TIME, "weekday": USE_WEEKDAY, "coin": USE_COIN, "daily_vol": USE_DAILY_VOL}
print(f"device {device} | switches {switches} | seed {SEED} | "
      f"{sum(p.numel() for p in model.parameters()):,} parameters | unigram {unigram_bits:.3f} bits")

# ---------------- training ----------------
for step in range(STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y, ctx, _ = get_batch("train")
    loss = F.cross_entropy(model(x, ctx).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % EVAL_EVERY == 0:
        v = evaluate()["val_bits"]
        print(f"step {step:5d} | train {loss.item() / LN2:.4f} | val {v:.4f} bits "
              f"| below unigram {1000 * (unigram_bits - v):+.1f} mbit")

# ---------------- scorecard + save ----------------
res = evaluate(full=True)
pp = res["per_position"]
print(f"\n=== Scorecard {switches}, seed {SEED} ===")
print(f"  val bits/token        {res['val_bits']:.4f}   ({1000 * (unigram_bits - res['val_bits']):+.1f} mbit below unigram)")
print(f"  bits with 1 / 8 / {CONTEXT} candles   {pp[0]:.4f} / {pp[7]:.4f} / {pp[-1]:.4f}")
if "next_candle_ic" in res:
    print(f"  next-candle rank IC   {res['next_candle_ic']:+.4f}")
    print(f"  next-candle AUC       {res['next_candle_auc']:.4f}")

tag = "".join(k[0] for k, on in switches.items() if on) or "none"
name = f"ctx_{tag}_s{SEED}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pt"
torch.save({"model": model.state_dict(), "tokenizer_run": str(TOKENIZER_RUN),
            "settings": {"context": CONTEXT, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
                         "n_layers": N_LAYERS, "steps": STEPS, "lr": LR, "symbols": SYMBOLS,
                         "switches": switches, "seed": SEED,
                         "dvol_mean": float(dv_mean), "dvol_std": float(dv_std)},
            "unigram_bits": unigram_bits, **res}, HERE.parent / "runs" / name)
print(f"saved runs/{name}")