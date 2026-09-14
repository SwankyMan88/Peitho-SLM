"""Build training.txt and heldout.txt from alpaca_data.json.

Source:
  alpaca_data.json  a list of {"instruction", "input", "output"} records

Each record becomes one conversation in the same format make_corpus.py writes:

  ◈
  ▶instruction (and input, when there is one)■
  ◀a short plan◇the output■

Turns are one line each, so newlines inside a record are folded into spaces. The
model's vocabulary is ASCII plus the markers, so typographic punctuation is mapped
to its plain form and a record that still needs anything else - a translation into
Chinese, say - is skipped rather than mangled.

Alpaca has no working to think through, so the thought is a short plan built from
the instruction. It is there so the corpus uses one convention throughout - a share
of turns with the marker and a share without is what makes a model leak its notes
as speech (see docs/thinking.md). Pass --no_thinking to write the output alone.
"""

import argparse
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK, BASE_CHARS, MARKERS

ALPACA = os.path.join(paths.ROOT, "alpaca_data.json")

# Characters outside the vocabulary that have an obvious plain form.
PLAIN = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": " - ", "‒": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", "•": "-", "·": "-", "×": "x",
    "÷": "/", "≤": "<=", "≥": ">=", "≠": "!=", "°": " degrees",
}
ALLOWED = set(BASE_CHARS) - set(MARKERS) - {"\t", "\n"}


def clean(text):
    """One line of vocabulary-only text, or None if it cannot be made so honestly."""
    text = "".join(PLAIN.get(c, c) for c in text)
    # Accents come off (cafe, naive); anything without an ASCII base survives
    # this unchanged and is caught below.
    text = "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))
    text = " ".join(text.split())       # newlines and tabs fold into single spaces
    if not text or any(c not in ALLOWED for c in text):
        return None
    return text


PLANS_QUESTION = [
    "They are asking: {task}. Answer it plainly.",
    "A question - {task}. Say what I know, and keep it clear.",
    "Question is {task}. Answer directly.",
]
PLANS_TASK = [
    "They want me to {task}. Do that, and nothing extra.",
    "Asked to {task}. Keep to what was asked.",
    "The job: {task}. Do it plainly.",
]
WITH_INPUT = [" Work from what they gave me.", " Use the text they supplied.", ""]


def plan(rng, instruction, has_input, limit=70):
    """A short, honest note of what is being asked - the instruction restated."""
    question = instruction.rstrip().endswith("?")
    task = instruction.rstrip(" ?.!:")
    task = task[0].lower() + task[1:]
    if len(task) > limit:
        # Cut at a word; the template supplies the full stop.
        task = task[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-.")
    note = rng.choice(PLANS_QUESTION if question else PLANS_TASK).format(task=task)
    if has_input:
        note += rng.choice(WITH_INPUT)
    return note


def conversation(rng, record, thinking_on):
    instruction = clean(record.get("instruction", ""))
    given = record.get("input", "") or ""
    given = clean(given) if given.strip() else ""
    output = clean(record.get("output", ""))
    if instruction is None or given is None or output is None:
        return None

    asked = instruction
    if given:
        asked += (" " if instruction[-1] in ".?!:" else ": ") + given
    reply = (plan(rng, instruction, bool(given)) + THINK_MARK + output) if thinking_on else output
    return "\n".join([START_MARK, USER_MARK + asked + END_MARK, BOT_MARK + reply + END_MARK])


def main():
    p = argparse.ArgumentParser(description="Turn alpaca_data.json into the conversation corpus.")
    p.add_argument("--alpaca", default=ALPACA)
    p.add_argument("--train_out", default=paths.TRAINING)
    p.add_argument("--heldout_out", default=paths.HELDOUT)
    p.add_argument("--heldout_frac", type=float, default=0.05,
                   help="Fraction of records withheld from training entirely.")
    p.add_argument("--max_turn_chars", type=int, default=0,
                   help="Skip records whose reply is longer than this (0 = keep all). "
                        "A reply longer than --block_size never fits in context whole.")
    p.add_argument("--no_thinking", action="store_true",
                   help="Write the output alone, with no plan and no think marker.")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    with open(args.alpaca, "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    thinking_on = not args.no_thinking

    chunks, skipped_chars, skipped_long = [], 0, 0
    for record in records:
        chunk = conversation(rng, record, thinking_on)
        if chunk is None:
            skipped_chars += 1
            continue
        if args.max_turn_chars and len(chunk.rsplit("\n", 1)[-1]) > args.max_turn_chars:
            skipped_long += 1
            continue
        chunks.append(chunk)

    n_held = max(1, int(len(chunks) * args.heldout_frac))
    held, train = chunks[:n_held], chunks[n_held:]

    paths.ensure_build()
    for path, part, label in ((args.train_out, train, "train"),
                              (args.heldout_out, held, "heldout")):
        text = "\n".join(part) + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"{label:8} -> {paths.short(path)}: {len(text):,} chars, "
              f"{len(part):,} conversations, {len(set(text))} distinct characters")

    print(f"\n{len(records):,} records read, {len(chunks):,} kept")
    print(f"{skipped_chars:,} skipped for characters outside the vocabulary")
    if args.max_turn_chars:
        print(f"{skipped_long:,} skipped for replies over {args.max_turn_chars} chars")
    lengths = sorted(len(c) for c in chunks)
    print(f"conversation length: median {lengths[len(lengths) // 2]:,} chars, "
          f"90th percentile {lengths[int(len(lengths) * 0.9)]:,}, longest {lengths[-1]:,}")


if __name__ == "__main__":
    main()
