"""Build the conformance kit: a micro model and the numbers a port must reproduce.

A port of the decoder is easy to get subtly wrong - a transposed matmul, the wrong
layernorm epsilon, nibbles read low-half-first instead of high - and every one of
those still produces fluent-looking text. The only way to know a port is correct is
to check it against fixed numbers.

This writes:

    conformance/micro_1.0.txt   a real export, small enough to read by eye
    conformance/vectors.json    prompts -> logits, and the greedy continuation

The model is randomly initialised on purpose. Conformance is about arithmetic, not
about quality, and random weights exercise the same code paths while keeping the
file a few kilobytes rather than half a megabyte.

    py tools/make_conformance.py
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from export import export_compressed, import_compressed
from model import GPT, GPTConfig

# Deliberately tiny and deliberately awkward: an odd vocab size and a width that is
# not a multiple of the group size, so padding and group boundaries are exercised.
VOCAB = "\nABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ."
CONFIG = dict(block_size=16, n_layer=2, n_head=2, n_embd=20, dropout=0.0)
SEED = 1234

# Chosen to cover: one token, a full-length context, and a repeat that a copying
# circuit would treat differently from a first occurrence.
PROMPTS = ["A", "ABC", "HELLO WORLD.", "AB AB AB AB AB AB", "z"]
DIGITS = 6          # decimal places stored, comfortably inside float32 agreement
GREEDY_STEPS = 8


def build():
    torch.manual_seed(SEED)
    stoi = {c: i for i, c in enumerate(VOCAB)}
    itos = {i: c for c, i in stoi.items()}
    config = GPTConfig(vocab_size=len(VOCAB), **CONFIG)
    model = GPT(config).eval()
    return model, config, stoi, itos


@torch.no_grad()
def logits_for(model, ids):
    out, _ = model(torch.tensor([ids], dtype=torch.long))
    return out[0, -1].tolist()


@torch.no_grad()
def greedy(model, ids, steps, block_size):
    ids = list(ids)
    for _ in range(steps):
        window = ids[-block_size:]
        nxt = int(torch.argmax(torch.tensor(logits_for(model, window))))
        ids.append(nxt)
    return ids


def main():
    p = argparse.ArgumentParser(description="Write the conformance kit.")
    p.add_argument("--out_dir", default=os.path.join(paths.ROOT, "conformance"))
    p.add_argument("--bits", type=int, choices=(4, 8), default=8)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, config, stoi, itos = build()
    export_path = os.path.join(args.out_dir, "micro_1.0.txt")
    export_compressed(model, config, stoi, itos, export_path,
                      bits=args.bits, group_size=32)

    # Vectors come from the quantized model, not the full-precision one: a port
    # reads the export, so the export is what it has to agree with.
    loaded, lcfg, lstoi, litos = import_compressed(export_path, device="cpu")
    loaded.eval()

    cases = []
    for prompt in PROMPTS:
        ids = [lstoi[c] for c in prompt][-lcfg.block_size:]
        continuation = greedy(loaded, ids, GREEDY_STEPS, lcfg.block_size)
        cases.append({
            "prompt": prompt,
            "ids": ids,
            "logits": [round(v, DIGITS) for v in logits_for(loaded, ids)],
            "greedy_ids": continuation,
            "greedy_text": "".join(litos[i] for i in continuation),
        })

    vectors = {
        "about": "Expected outputs for conformance/micro_1.0.txt. See docs/spec.md.",
        "tolerance": 1e-4,
        "greedy_steps": GREEDY_STEPS,
        "cases": cases,
    }
    vectors_path = os.path.join(args.out_dir, "vectors.json")
    with open(vectors_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(vectors, f, indent=1)
        f.write("\n")

    print(f"{paths.short(export_path)}: {os.path.getsize(export_path):,} bytes, "
          f"{sum(q.numel() for q in model.parameters()):,} params")
    print(f"{paths.short(vectors_path)}: {len(cases)} cases, "
          f"{len(cases[0]['logits'])} logits each")
    for case in cases:
        print(f"  {case['prompt']!r:20} -> {case['greedy_text'][len(case['prompt']):]!r}")


if __name__ == "__main__":
    main()
