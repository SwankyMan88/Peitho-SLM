"""Copy the working corpus into data/ as the published one, with its recipe.

build/ holds whatever was last generated and is not in git - it changes every time
anyone experiments. data/ holds the corpus a released model was actually trained on,
so somebody can clone the repository and train without running the generators at
all. This copies one to the other and records how it was made.

    py tools/publish_corpus.py --command "py corpus/chat/make_corpus.py --think 1.0 ..."
"""

import argparse
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slm"))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK

DATA = os.path.join(paths.ROOT, "data")
DEFAULT_COMMAND = ("py corpus/chat/make_corpus.py --target_chars 20000000 "
                   "--composed 0.95 --think 1.0")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def describe(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    bots = [l for l in lines if l.startswith(BOT_MARK)]
    return {
        "chars": len(text),
        "conversations": text.count(START_MARK),
        "turns": sum(l.startswith((USER_MARK, BOT_MARK)) for l in lines),
        "thinks": sum(THINK_MARK in l for l in bots),
        "bots": len(bots),
        "vocab": len(set(text)),
        "sha256": digest(path),
        "bytes": os.path.getsize(path),
    }


def main():
    p = argparse.ArgumentParser(description="Publish the corpus into data/.")
    p.add_argument("--train", default=paths.TRAINING)
    p.add_argument("--heldout", default=paths.HELDOUT)
    p.add_argument("--command", default=DEFAULT_COMMAND,
                   help="The command that generated these, recorded in data/README.md.")
    p.add_argument("--models", default="",
                   help="Which exports were trained on this, for the record.")
    args = p.parse_args()

    os.makedirs(DATA, exist_ok=True)
    published = {}
    for source, name in ((args.train, "training.txt"), (args.heldout, "heldout.txt")):
        if not os.path.exists(source):
            raise SystemExit(f"{source} does not exist. Generate a corpus first.")
        target = os.path.join(DATA, name)
        shutil.copyfile(source, target)
        published[name] = describe(target)
        print(f"  {paths.short(source)} -> {paths.short(target)} "
              f"({published[name]['bytes']:,} bytes)")

    train, held = published["training.txt"], published["heldout.txt"]
    readme = f"""# The corpus

The text the released models were trained on, so this repository can be cloned and
trained from without running the generators. It is generated, not hand-written -
`corpus/` holds the programs that produce it, and `corpus/chat/conversations.txt` holds
the only part a person wrote by hand.

| | training.txt | heldout.txt |
|---|---|---|
| characters | {train['chars']:,} | {held['chars']:,} |
| conversations | {train['conversations']:,} | {held['conversations']:,} |
| turns | {train['turns']:,} | {held['turns']:,} |
| model turns that think first | {train['thinks']:,} of {train['bots']:,} | {held['thinks']:,} of {held['bots']:,} |
| distinct characters | {train['vocab']} | {held['vocab']} |

```
training.txt  sha256 {train['sha256']}
heldout.txt   sha256 {held['sha256']}
```

## Train on it

```bash
py train.py --data data/training.txt --val_data data/heldout.txt \\
    --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

Held-out conversations never appear in training, so the validation loss means
something. Both files use the turn markers described in
[../docs/spec.md](../docs/spec.md).

## Regenerate it

```bash
{args.command}
py tools/publish_corpus.py
```

The generators are seeded, so the same command on the same commit reproduces these
files byte for byte - the hashes above are worth checking if you change anything in
`corpus/` and want to know whether you changed the corpus too.
"""
    if args.models:
        readme += f"\nModels trained on this corpus: {args.models}.\n"

    with open(os.path.join(DATA, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(readme)
    print(f"  wrote {paths.short(os.path.join(DATA, 'README.md'))}")
    print(f"\n  training.txt sha256 {train['sha256'][:16]}...")


if __name__ == "__main__":
    main()
