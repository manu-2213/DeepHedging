"""Timing of the encoder work in one training iteration, for the LSTM and transformer variants.

One v2.3/v2.4 iteration (BSM call, 100 paths x 20 envs = 2000 envs, N = 250 steps, window 10) does
  - rollout: encoder features for every (env, step) without gradients  -> 500,000 windows
  - update : 4 gradient steps (ppo_epochs 1, minibatch 125,000 samples)
  - every 10th iteration a validation rollout (500 paths -> 10,000 envs = 2.5M windows), i.e. 250,000 per iteration
This script times exactly that encoder work (forward + backward), not the per-step environment loop,
which is the same for every encoder.

Variants
  lstm            window LSTM (the v2.3 encoder)
  tf              window transformer as trained in v2.4 (fp32)
  tf_amp          same, under autocast (bf16 where supported, else fp16)
  tf_amp_compile  same, torch.compile'd
  path_tf         one causal pass over the whole path with a sliding 10-step attention band and a learned
                  relative-position bias: every observation is encoded once instead of 10 times. NOT the same
                  model (2 layers see up to 19 steps back); timed here to see whether it is worth building.
  path_tf_amp     path_tf under autocast

Usage (from deep_hedging_v0/, on a GPU):  python -m fast_hedging.tests.bench_transformer
"""
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 16 GB V100: avoid fragmentation OOM

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_hedging.networks import Encoder

ENVS, N, W, FEAT, H = 2000, 250, 10, 11, 64
MINIBATCH, STEPS = 125_000, 4
VAL_WINDOWS = 250_000          # validation work per iteration (one 2.5M-window rollout every 10 iterations)
CHUNK = 131_072                # rollout chunk, as ppo._encode_all


class PathTransformer(nn.Module):
    """Causal transformer over a whole path; token t attends to t-W+1..t (relative position bias per head)."""

    def __init__(self, in_dim, hidden, window, layers=2, heads=4):
        super().__init__()
        self.window, self.heads = window, heads
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "n1": nn.LayerNorm(hidden), "qkv": nn.Linear(hidden, 3 * hidden), "out": nn.Linear(hidden, hidden),
            "n2": nn.LayerNorm(hidden), "ff": nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                                           nn.Linear(2 * hidden, hidden))}) for _ in range(layers))
        self.rel = nn.Parameter(torch.zeros(layers, heads, window))
        self.norm = nn.LayerNorm(hidden)

    def bias(self, i, T, device, dtype):
        d = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]   # t - s
        inside = (d >= 0) & (d < self.window)
        b = torch.full((self.heads, T, T), float("-inf"), device=device, dtype=dtype)
        b[:, inside] = self.rel[i][:, d[inside]].to(dtype)
        return b

    def forward(self, x):                        # (B, T, F) -> (B, T, h)
        B, T, _ = x.shape
        h = self.embed(x)
        for i, L in enumerate(self.layers):
            q, k, v = L["qkv"](L["n1"](h)).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias(i, T, x.device, q.dtype))
            h = h + L["out"](a.transpose(1, 2).reshape(B, T, -1))
            h = h + L["ff"](L["n2"](h))
        return self.norm(h)


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=3, warm=2):
    for _ in range(warm):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps


def window_work(model, amp_dtype):
    windows = torch.randn(CHUNK, W, FEAT, device="cuda")
    mb = torch.randn(MINIBATCH, W, FEAT, device="cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else torch.enable_grad

    def iteration():
        with torch.no_grad(), ctx():
            for _ in range(round((ENVS * N + VAL_WINDOWS) / CHUNK)):
                model(windows)
        for _ in range(STEPS):
            with ctx():
                loss = model(mb).float().square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return iteration


def path_work(model, amp_dtype):
    paths = torch.randn(ENVS + VAL_WINDOWS // N, N, FEAT, device="cuda")      # rollout + validation share
    mb = torch.randn(MINIBATCH // N, N, FEAT, device="cuda")                  # whole paths, as recurrent mode
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else torch.enable_grad
    chunk = max(1, CHUNK // N)

    def iteration():
        with torch.no_grad(), ctx():
            for i in range(0, paths.shape[0], chunk):
                model(paths[i:i + chunk])
        for _ in range(STEPS):
            with ctx():
                loss = model(mb).float().square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return iteration


def main():
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0)
    # bf16 only on cards that have it in hardware (A100/H100); torch reports emulated bf16 as supported on V100
    amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"GPU {name}, torch {torch.__version__}, autocast dtype {amp}", flush=True)
    variants = [
        ("lstm", lambda: window_work(Encoder("window", FEAT, H).cuda(), None)),
        ("tf", lambda: window_work(Encoder("transformer", FEAT, H, window=W).cuda(), None)),
        ("tf_amp", lambda: window_work(Encoder("transformer", FEAT, H, window=W).cuda(), amp)),
        ("tf_amp_compile", lambda: window_work(torch.compile(Encoder("transformer", FEAT, H, window=W).cuda()), amp)),
        ("path_tf", lambda: path_work(PathTransformer(FEAT, H, W).cuda(), None)),
        ("path_tf_amp", lambda: path_work(PathTransformer(FEAT, H, W).cuda(), amp)),
    ]
    base = None
    for label, build in variants:
        try:
            t0 = time.perf_counter()
            fn = build()
            sec = timed(fn)
            mem = torch.cuda.max_memory_allocated() / 2**30
            if label == "lstm":
                base = sec
            print(f"{label:16s} {sec:7.3f} s per iteration  ({(base / sec if base else float('nan')):4.2f}x the LSTM)  peak {mem:5.1f} GB"
                  f"  [setup+warmup {time.perf_counter() - t0 - 3 * sec:5.1f} s]", flush=True)
        except Exception as e:                      # one failing variant must not hide the others
            print(f"{label:16s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
