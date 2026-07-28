"""Measure what the model is actually good and bad at.

Reported aspects:
  language  - bits per character on held-out conversations (lower is better)
  format    - does it close its turn with the end marker instead of rambling?
  spelling  - are the words it writes real words?
  variety   - is it repeating itself?
  novelty   - how much of a reply is NOT found verbatim in the training text
  copying   - is it reciting the training file verbatim?
  arith     - does it land the right number on sums it has never seen?
  export    - what quantization costs in quality and bytes

Held-out conversations never appear in training at all, so the language score
measures writing rather than recitation. Recall of facts is handled by the page
and is not a property of the model, so it is not measured here.
"""

import argparse
import json
import math
import os
import random
import re
import sys

import torch

from model import GPT, START_MARK, USER_MARK, BOT_MARK, END_MARK, encode, decode
from export import import_compressed
import versions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_model(path, compressed, device):
    if compressed:
        return import_compressed(path, device=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"], ckpt["stoi"], ckpt["itos"]


@torch.no_grad()
def heldout_loss(model, text, stoi, block_size, device, max_windows=400):
    """Deterministic sweep of consecutive windows, so the number is repeatable."""
    ids = [stoi[c] for c in text if c in stoi]
    data = torch.tensor(ids, dtype=torch.long)
    stride = block_size
    starts = list(range(0, len(data) - block_size - 1, stride))[:max_windows]
    if not starts:
        return float("nan")
    model.eval()
    total = 0.0
    for s in starts:
        x = data[s:s + block_size].unsqueeze(0).to(device)
        y = data[s + 1:s + 1 + block_size].unsqueeze(0).to(device)
        _, loss = model(x, y)
        total += loss.item()
    return total / len(starts)


@torch.no_grad()
def reply_to(model, prompt, stoi, itos, config, device, max_new=220, temperature=0.7, top_k=40):
    prompt = "".join(c for c in prompt if c in stoi)[-config.block_size:]
    ids = encode(prompt, stoi)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_id = stoi.get(END_MARK)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, stop_id=stop_id)
    new_ids = out[0, len(ids):].tolist()
    stopped = bool(new_ids) and new_ids[-1] == stop_id
    text = decode(new_ids, itos)
    for mark in (END_MARK, USER_MARK, BOT_MARK, START_MARK, "\n"):
        i = text.find(mark)
        if i != -1:
            text = text[:i]
    return text.strip(), stopped


def bench_generation(model, stoi, itos, config, device, prompts, train_text, word_set):
    stopped_count = 0
    lengths = []
    in_vocab = total_words = 0
    trigram_ratios = []
    exact_copies = 0
    prefix_matches = []
    empties = 0

    for prompt_text in prompts:
        prompt = f"{START_MARK}\n{USER_MARK}{prompt_text}{END_MARK}\n{BOT_MARK}"
        text, stopped = reply_to(model, prompt, stoi, itos, config, device)
        stopped_count += stopped
        lengths.append(len(text))
        if not text:
            empties += 1
            continue

        words = re.findall(r"[A-Za-z]+", text.lower())
        total_words += len(words)
        in_vocab += sum(w in word_set for w in words)

        trigrams = [text[i:i + 3] for i in range(max(0, len(text) - 2))]
        if trigrams:
            trigram_ratios.append(len(set(trigrams)) / len(trigrams))

        if len(text) >= 20 and text in train_text:
            exact_copies += 1
        lo, hi = 0, len(text)
        while lo < hi:  # longest prefix of the reply that appears verbatim in training
            mid = (lo + hi + 1) // 2
            if text[:mid] in train_text:
                lo = mid
            else:
                hi = mid - 1
        prefix_matches.append(lo)

    n = len(prompts)
    return {
        "format_ok": stopped_count / n,
        "novelty": 1 - (sum(prefix_matches) / max(1, sum(lengths))),
        "mean_len": sum(lengths) / n,
        "empty": empties / n,
        "spelling": in_vocab / total_words if total_words else float("nan"),
        "variety": sum(trigram_ratios) / len(trigram_ratios) if trigram_ratios else float("nan"),
        "exact_copy": exact_copies / n,
        "mean_copied_prefix": sum(prefix_matches) / len(prefix_matches) if prefix_matches else 0,
    }


def bench_arithmetic(model, stoi, itos, config, device, rng, trials=15):
    """Accuracy on sums the model has never seen, by operand size.

    Scored on the last number in the reply, which is where every method in
    arith.py puts its answer. Working arithmetic is a copying task before it is a
    counting one - the operands have to be read out of the context first - so this
    stays at zero until the copying circuit forms, long after the loss curve
    flattens."""
    rows = []
    for label, lo, hi in (("1 digit", 1, 9), ("2 digit", 10, 99), ("3 digit", 100, 999)):
        for op in "+-*":
            right = 0
            for _ in range(trials):
                lhs, rhs = rng.randint(lo, hi), rng.randint(lo, hi)
                if op == "-":
                    lhs, rhs = max(lhs, rhs), min(lhs, rhs)
                if op == "*":
                    rhs = rng.randint(2, 12)
                truth = lhs + rhs if op == "+" else lhs - rhs if op == "-" else lhs * rhs
                prompt = (f"{START_MARK}\n{USER_MARK}What is {lhs} {op} {rhs}?"
                          f"{END_MARK}\n{BOT_MARK}")
                text, _ = reply_to(model, prompt, stoi, itos, config, device,
                                   max_new=300, temperature=0.6, top_k=20)
                numbers = [int(n) for n in re.findall(r"-?\d+", text)]
                right += bool(numbers) and numbers[-1] == truth
            rows.append((f"{label} {op}", right / trials))
    return rows


def bar(fraction, width=20):
    if fraction != fraction:  # NaN
        return "?" * width
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark the trained SLM.",
        epilog="Examples:  py benchmark.py small_1.2   |   py benchmark.py small   |   "
               "py benchmark.py models/large_1.0.txt   |   py benchmark.py (the checkpoint)")
    p.add_argument("model", nargs="?", default="",
                   help="An export to benchmark: a name such as \"small_1.2\", a base name "
                        "such as \"small\" for its highest version, or a path. Omitted, the "
                        "full-precision checkpoint is used instead.")
    p.add_argument("--checkpoint", default="model_full.pt")
    p.add_argument("--compressed_path", default="",
                   help="Same as passing the model as the first argument.")
    p.add_argument("--train_data", default="training.txt")
    p.add_argument("--heldout", default="heldout.txt")
    p.add_argument("--gen_samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--from_compressed", action="store_true",
                   help="Benchmark the most recent export rather than the checkpoint. "
                        "Implied by naming a model.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # Naming a model means benchmarking that export; the flag is only needed to ask
    # for the most recent one without naming it.
    wanted = args.model or args.compressed_path
    compressed = bool(wanted) or args.from_compressed
    src = versions.resolve(wanted) if compressed else args.checkpoint
    model, config, stoi, itos = load_model(src, compressed, device)
    model.eval()
    if compressed:
        with open(src, encoding="utf-8") as f:
            head = json.loads(f.readline())
        print(f"model source: {src} ({head['bits']}-bit export, group {head['group']})")
    else:
        print(f"model source: {src} (full precision)")
    n_params = sum(q.numel() for q in model.parameters())

    train_text = open(args.train_data, encoding="utf-8").read()
    word_set = set(re.findall(r"[a-z]+", train_text.lower()))

    print(f"device {device} | params {n_params:,} | block_size {config.block_size} "
          f"| vocab {config.vocab_size}")
    print(f"training corpus {len(train_text):,} chars "
          f"({len(train_text)/n_params:.2f} chars per parameter)\n")

    # ---- language modelling
    print("LANGUAGE")
    held_text = open(args.heldout, encoding="utf-8").read()
    hl = heldout_loss(model, held_text, stoi, config.block_size, device)
    tl = heldout_loss(model, train_text[:len(held_text)], stoi, config.block_size, device)
    print(f"  heldout loss        {hl:.4f} nats = {hl/math.log(2):.2f} bits/char")
    print(f"  training loss       {tl:.4f} nats = {tl/math.log(2):.2f} bits/char")
    print(f"  generalization gap  {hl - tl:+.4f}       (large = memorizing)\n")

    # ---- generation quality
    print(f"GENERATION  ({args.gen_samples} prompts)")
    user_lines = re.findall(f"{USER_MARK}(.*?){END_MARK}", held_text)
    prompts = rng.sample(user_lines, min(args.gen_samples, len(user_lines)))
    g = bench_generation(model, stoi, itos, config, device, prompts, train_text, word_set)
    print(f"  format (ends turn)   {bar(g['format_ok'])} {g['format_ok']:.0%}")
    print(f"  spelling (real words) {bar(g['spelling'])} {g['spelling']:.0%}")
    print(f"  variety (distinct 3g) {bar(g['variety'])} {g['variety']:.0%}")
    print(f"  novelty (not recited) {bar(g['novelty'])} {g['novelty']:.0%}")
    print(f"  mean reply length   {g['mean_len']:.0f} chars | empty replies {g['empty']:.0%}")
    print(f"  verbatim copies     {g['exact_copy']:.0%} of replies | "
          f"mean copied prefix {g['mean_copied_prefix']:.0f} chars\n")

    # ---- arithmetic
    print("ARITHMETIC  (unseen sums, scored on the final number)")
    rows = bench_arithmetic(model, stoi, itos, config, device, rng)
    for label, score in rows:
        print(f"  {label:12} {bar(score)} {score:.0%}")
    print(f"  overall            {sum(s for _, s in rows) / len(rows):.0%}\n")

    # ---- export cost, when a checkpoint was measured and an export exists to compare
    if not args.skip_export and not compressed:
        try:
            export = versions.resolve("")
        except SystemExit:
            export = ""
        if not export:
            return
        print("EXPORT")
        cmodel, ccfg, cstoi, _ = load_model(export, True, device)
        cmodel.eval()
        chl = heldout_loss(cmodel, held_text, cstoi, ccfg.block_size, device)
        size = os.path.getsize(export)
        print(f"  compressed loss   {chl:.4f} nats = {chl/math.log(2):.2f} bits/char "
              f"({chl - hl:+.4f} vs full)")
        print(f"  file size         {size:,} bytes = {8*size/n_params:.2f} bits/param, "
              f"{sum(1 for _ in open(export, encoding='utf-8'))} lines")


if __name__ == "__main__":
    main()
