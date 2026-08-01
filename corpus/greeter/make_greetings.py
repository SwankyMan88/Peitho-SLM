"""Write the greetings corpus: one bot turn per conversation, nothing else.

The greeter answers no questions, so it needs no user turns at all. Each
conversation is a start marker and a single model turn:

    ◈
    ◀Hello. I am a small language model. Ask me something.■

Prompted with `◈\\n◀` it will produce a greeting and stop at the end marker.

    py corpus/greeter/make_greetings.py
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import paths
from model import START_MARK, BOT_MARK, END_MARK

sys.path.insert(0, paths.GREETER)
import greetings


def render(rng, target_chars):
    parts, total = [], 0
    while total < target_chars:
        line = START_MARK + "\n" + BOT_MARK + greetings.greeting(rng) + END_MARK
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts) + "\n"


def main():
    p = argparse.ArgumentParser(description="Write the greetings corpus.")
    p.add_argument("--target_chars", type=int, default=3_000_000)
    p.add_argument("--heldout_chars", type=int, default=200_000)
    p.add_argument("--train_out", default=paths.GREETINGS)
    p.add_argument("--heldout_out", default=paths.GREETINGS_HELD)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    paths.ensure_build()
    train = render(random.Random(args.seed), args.target_chars)
    # A different seed, so the held-out greetings are drawn independently. They will
    # still overlap: the point is that the phrasing is not memorised, and with 24,000
    # distinct greetings in 40,000 draws some repetition is unavoidable and honest.
    held = render(random.Random(args.seed + 1), args.heldout_chars)

    for path, text, label in ((args.train_out, train, "train"),
                              (args.heldout_out, held, "heldout")):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        lines = [l for l in text.split("\n") if l.startswith(BOT_MARK)]
        print(f"{label:8} -> {paths.short(path)}: {len(text):,} chars, "
              f"{len(lines):,} greetings, {len(set(lines)):,} distinct "
              f"({len(set(lines))/len(lines):.0%}), {len(set(text))} characters")


if __name__ == "__main__":
    main()
