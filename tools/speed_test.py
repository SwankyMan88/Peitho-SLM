"""Measure training throughput so speed changes can be judged instead of guessed.

Reports characters of training text processed per second, which is the number that
decides how long a run takes. Steps per second alone is misleading: a bigger batch
does fewer steps but more work per step.
"""

import argparse
import time

import torch

from model import GPT, GPTConfig, build_vocab, encode
from train import get_batch, make_optimizer, PRESETS


def measure(text, preset, block, batch, steps, compile_model, device):
    stoi, _ = build_vocab(text)
    arch = dict(PRESETS[preset])
    arch["block_size"] = block
    config = GPTConfig(vocab_size=len(stoi), dropout=0.0, **arch)
    model = GPT(config).to(device)
    if compile_model:
        model = torch.compile(model)

    data = torch.tensor(encode(text, stoi), dtype=torch.int16).to(device)
    optimizer = make_optimizer(model, 1e-3, 0.01)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    for warm in range(5):  # let cuDNN pick kernels and any compile happen
        x, y = get_batch(data, block, batch, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        x, y = get_batch(data, block, batch, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    chars = steps * batch * block
    return elapsed / steps * 1000, chars / elapsed


def main():
    p = argparse.ArgumentParser(description="Time training throughput.")
    p.add_argument("--data", default="training.txt")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--preset", default="medium")
    p.add_argument("--compile", action="store_true", help="Also try torch.compile.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with open(args.data, encoding="utf-8") as f:
        text = f.read()[:2_000_000]

    print(f"device {device} | preset {args.preset} | {args.steps} timed steps each\n")
    print(f"{'block':>6} {'batch':>6} {'compile':>8} {'ms/step':>9} {'chars/sec':>12}  relative")
    baseline = None
    for block, batch, comp in [(384, 64, False), (384, 128, False), (384, 256, False),
                               (256, 128, False), (256, 256, False), (192, 256, False)]:
        try:
            ms, cps = measure(text, args.preset, block, batch, args.steps, comp, device)
        except torch.cuda.OutOfMemoryError:
            print(f"{block:>6} {batch:>6} {str(comp):>8}   out of memory")
            torch.cuda.empty_cache()
            continue
        baseline = baseline or cps
        print(f"{block:>6} {batch:>6} {str(comp):>8} {ms:>9.1f} {cps:>12,.0f}  {cps/baseline:>6.2f}x")
        torch.cuda.empty_cache()

    if args.compile:
        try:
            ms, cps = measure(text, args.preset, 384, 128, args.steps, True, device)
            print(f"{384:>6} {128:>6} {'True':>8} {ms:>9.1f} {cps:>12,.0f}  {cps/baseline:>6.2f}x")
        except Exception as e:
            print(f"torch.compile unavailable here: {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    main()
