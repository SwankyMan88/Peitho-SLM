"""Build a corpus from human-written text only - no generators, no templates.

Every sentence in the result was written by a person: the hand-written
conversations in corpus/chat/conversations.txt, and seven outside datasets whose
licenses allow any use, downloaded as repository zips from GitHub into
build/datasets/:

  source           repository                                            license
  Schema-Guided    google-research-datasets/dstc8-schema-guided-dialogue CC BY-SA 4.0
  MultiWOZ 2.2     budzianowski/multiwoz                                 MIT
  Dolly-15k        databricks (copy: Dongdong0704/databricks-dolly-15k-project)  CC BY-SA 3.0
  QReCC            apple/ml-qrecc                                        CC BY-SA 3.0
  CCPE-M           google-research-datasets/ccpe                         CC BY 4.0
  SQuAD 1.1        rajpurkar/SQuAD-explorer                              CC BY-SA 4.0
  GSM8K            openai/grade-school-math                              MIT

Thinking, the corpus-wide format, is kept honest rather than invented: a grade-school
maths solution is real working, so its lines become the thought and its final line
the reply. Nothing else in these datasets has working, so every other turn thinks
nothing - "◀◇reply■" - which keeps one convention throughout without writing a plan
nobody wrote.

    py corpus/human/make_human.py     -> build/training_human.txt, build/heldout_human.txt
"""

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK

sys.path.insert(0, os.path.join(paths.CORPUS, "alpaca"))
sys.path.insert(0, os.path.join(paths.CORPUS, "dialogue"))
from make_alpaca import clean, plan
import make_dialogue

DATASETS = os.path.join(paths.BUILD, "datasets")

# --plans: give every turn a short plan before the think marker, as the generated
# corpus does, so the two can be mixed without two conventions for the same kind of
# prompt (see docs/thinking.md). Off, turns with no real working think nothing.
PLANS = False
PLAN_RNG = random.Random(7)


def plan_for(question, has_input=False):
    return plan(PLAN_RNG, clean(question) or "Answer it", has_input) if PLANS else None


def conversation(turns):
    """[(speaker, text, thought)] -> a finished conversation, or None if any turn
    cannot be written in the model's alphabet. Consecutive turns by one speaker are
    joined, the person speaks first, and the model has the last word."""
    merged = []
    for speaker, text, thought in turns:
        text = clean(text or "")
        if text is None:
            return None
        if not text:
            continue
        if merged and merged[-1][0] == speaker:
            merged[-1][1] += " " + text
        else:
            merged.append([speaker, text, thought])
    while merged and merged[0][0] != "user":
        merged.pop(0)
    while merged and merged[-1][0] != "bot":
        merged.pop()
    if len(merged) < 2:
        return None
    lines = [START_MARK]
    for speaker, text, thought in merged:
        if speaker == "user":
            lines.append(USER_MARK + text + END_MARK)
        else:
            thought = clean(thought or "") or ""
            lines.append(BOT_MARK + thought + THINK_MARK + text + END_MARK)
    return "\n".join(lines)


def without_plans(chunk):
    """make_dialogue writes a plan built from the dataset's labels; that is generated
    text, so it goes - the reply, which a person wrote, stays."""
    return re.sub(f"{re.escape(BOT_MARK)}[^{THINK_MARK}\n]*{THINK_MARK}",
                  BOT_MARK + THINK_MARK, chunk)


def load_handwritten():
    with open(paths.CONVERSATIONS, encoding="utf-8") as f:
        blocks = [b.strip().split("\n") for b in f.read().split("\n\n") if b.strip()]
    out = []
    for block in blocks:
        lines = [START_MARK]
        for line in block:
            if line.startswith(BOT_MARK) and THINK_MARK not in line:
                line = BOT_MARK + THINK_MARK + line[1:]
            lines.append(line)
        out.append("\n".join(lines))
    return out


def load_gsm8k(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            # "48/2 = <<48/2=24>>24" -> "48/2 = 24": the brackets are calculator notes.
            steps = [re.sub(r"<<[^>]*>>", "", s).strip()
                     for s in r["answer"].split("\n") if s.strip()]
            steps = [s for s in steps if not s.startswith("####")]
            if not steps:
                continue
            thought, reply = " ".join(steps[:-1]), steps[-1]
            out.append(conversation([("user", r["question"], None), ("bot", reply, thought)]))
    return out


def load_dolly(root):
    out = []
    for line in open(os.path.join(root, "databricks-dolly-15k.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        asked = r["instruction"].strip()
        has_context = bool(r.get("context", "").strip())
        if has_context:
            asked += " " + r["context"].strip()
        out.append(conversation([("user", asked, None),
                                 ("bot", r["response"], plan_for(r["instruction"], has_context))]))
    return out


def load_qrecc(root):
    by_conv = {}
    for path in sorted(glob.glob(os.path.join(root, "qrecc_*.json"))):
        for t in json.load(open(path, encoding="utf-8")):
            key = (os.path.basename(path), t["Conversation_no"])
            by_conv.setdefault(key, []).append(t)
    out = []
    for turns in by_conv.values():
        turns.sort(key=lambda t: t["Turn_no"])
        seq = []
        for t in turns:
            seq += [("user", t["Question"], None), ("bot", t["Answer"], plan_for(t["Question"]))]
        out.append(conversation(seq))
    return out


def load_ccpe(root):
    out = []
    for d in json.load(open(os.path.join(root, "data.json"), encoding="utf-8")):
        seq = [("bot" if u["speaker"] == "ASSISTANT" else "user", u["text"], None)
               for u in d["utterances"]]
        out.append(conversation(seq))
    return out


# Said back when a passage arrives with no question yet. Short and varied, so the
# model does not learn one stock sentence for every statement it is given.
READ_IT = ["Read it. Ask away.", "Got it. What do you want to know?",
           "I have read that. Go ahead.", "Right, I have it. Ask.",
           "Read. What would you like to know about it?"]
# Put the passage in a turn of its own, so answering means reading back through the
# conversation rather than the same turn - which is what remembering looks like here.
SPLIT_PASSAGE = False


def load_squad(root):
    """One passage per conversation: the passage and a first question, then the rest
    of that passage's questions as follow-ups. With SPLIT_PASSAGE, the passage gets a
    turn of its own first and every question comes later."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*-v1.1.json"))):
        for article in json.load(open(path, encoding="utf-8"))["data"]:
            for p in article["paragraphs"]:
                seq = []
                if SPLIT_PASSAGE:
                    seq += [("user", p["context"], None),
                            ("bot", PLAN_RNG.choice(READ_IT),
                             "A passage to read, with no question yet. Say I have it.")]
                # Six questions on one passage: answering them means reading back
                # through the context, the nearest thing here to remembering.
                for i, q in enumerate(p["qas"][:6]):
                    asked = q["question"] if (SPLIT_PASSAGE or i) else p["context"] + " " + q["question"]
                    seq += [("user", asked, None),
                            ("bot", q["answers"][0]["text"], plan_for(q["question"], i == 0))]
                out.append(conversation(seq))
    return out


def load_dialogues(root_sgd, root_woz):
    both = make_dialogue.load_sgd(root_sgd) + make_dialogue.load_multiwoz(root_woz)
    return [c if PLANS else without_plans(c) for c in both if c]


# name: (loader, default character budget for training, most passes allowed)
SOURCES = {
    "handwritten": (lambda a: load_handwritten(), 150_000, 3),
    "gsm8k": (lambda a: load_gsm8k(os.path.join(a.datasets, "gsm8k")), 9_000_000, 2),
    "dolly": (lambda a: load_dolly(os.path.join(a.datasets, "dolly")), 13_000_000, 1),
    "qrecc": (lambda a: load_qrecc(os.path.join(a.datasets, "qrecc")), 12_000_000, 1),
    "ccpe": (lambda a: load_ccpe(os.path.join(a.datasets, "ccpe")), 2_500_000, 1),
    "squad": (lambda a: load_squad(os.path.join(a.datasets, "squad")), 7_000_000, 1),
    "dialogue": (lambda a: load_dialogues(os.path.join(a.datasets, "sgd"),
                                          os.path.join(a.datasets, "multiwoz")), 10_000_000, 1),
}


def fill(rng, chunks, budget, max_passes):
    """Up to budget characters from chunks, taking each once before any repeats."""
    picked, total, passes = [], 0, 0
    while total < budget and passes < max_passes:
        order = list(chunks)
        rng.shuffle(order)
        for c in order:
            if total >= budget:
                break
            picked.append(c)
            total += len(c) + 1
        passes += 1
    return picked, total


def main():
    p = argparse.ArgumentParser(description="Build a corpus from human-written text only.")
    p.add_argument("--datasets", default=DATASETS)
    p.add_argument("--train_out", default=os.path.join(paths.BUILD, "training_human.txt"))
    p.add_argument("--heldout_out", default=os.path.join(paths.BUILD, "heldout_human.txt"))
    p.add_argument("--heldout_frac", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply every source's character budget by this.")
    p.add_argument("--set", action="append", default=[], metavar="NAME=CHARS[:PASSES]",
                   help="Override one source, e.g. --set dolly=20000000:2 "
                        "--set dialogue=4000000. Repeatable.")
    p.add_argument("--plans", action="store_true",
                   help="Give every turn a short plan before the think marker, so this text "
                        "can be mixed with the generated corpus.")
    p.add_argument("--split_passage", action="store_true",
                   help="Give each SQuAD passage a turn of its own, so the questions "
                        "that follow are answered from earlier in the conversation.")
    p.add_argument("--json_out", default="",
                   help="Instead of training and held-out files, write the chosen "
                        "conversations as a JSON list for make_corpus.py --dialogue, which "
                        "does its own held-out split and adds the hand-written ones itself.")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    global PLANS, SPLIT_PASSAGE
    PLANS = args.plans
    SPLIT_PASSAGE = args.split_passage
    if args.json_out:
        args.heldout_frac = 0.0

    sources = dict(SOURCES)
    if args.json_out:
        sources.pop("handwritten")
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in sources or not value:
            raise SystemExit(f"--set {item!r}: expected NAME=CHARS[:PASSES] with NAME one of "
                             f"{', '.join(sources)}")
        chars, _, passes = value.partition(":")
        loader, budget, max_passes = sources[name]
        sources[name] = (loader, int(float(chars)), int(passes) if passes else max_passes)

    rng = random.Random(args.seed)
    train, held = [], []
    print(f"{'source':12} {'available':>10} {'unique chars':>13} {'held out':>9} "
          f"{'written':>12} {'passes':>7}")
    for name, (loader, budget, max_passes) in sources.items():
        chunks = [c for c in loader(args) if c]
        rng.shuffle(chunks)
        n_held = max(1, int(len(chunks) * args.heldout_frac)) if name != "handwritten" else 0
        held += chunks[:n_held]
        pool = chunks[n_held:]
        unique = sum(len(c) + 1 for c in pool)
        picked, written = fill(rng, pool, int(budget * args.scale), max_passes)
        train += picked
        print(f"{name:12} {len(chunks):10,} {unique:13,} {n_held:9,} {written:12,} "
              f"{written / max(1, unique):7.2f}")

    rng.shuffle(train)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(train + held, f)
        print(f"{len(train) + len(held):,} conversations -> {paths.short(args.json_out)} "
              f"({sum(len(c) for c in train + held):,} chars)")
        return
    rng.shuffle(held)
    for path, part, label in ((args.train_out, train, "train"), (args.heldout_out, held, "heldout")):
        text = "\n".join(part) + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"{label:8} -> {paths.short(path)}: {len(text):,} chars, "
              f"{len(part):,} conversations, {len(set(text))} distinct characters")


if __name__ == "__main__":
    main()
