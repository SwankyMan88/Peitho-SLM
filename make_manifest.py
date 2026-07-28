"""List the exports in models/ into models/index.json.

The browser page reads this to decide which models to offer. HTTP gives no way to
list a directory - GitHub Pages will not, and neither will a plain file server -
so the folder has to describe itself in a file.

Run it after training, or after adding or removing an export by hand:

    py make_manifest.py
"""

import argparse
import json
import math
import os

import versions


def describe(path):
    """Everything the page needs about one export, from its header line alone."""
    with open(path, encoding="utf-8") as f:
        head = json.loads(f.readline())
    config = head["config"]
    weights = sum(math.prod(tensor["s"]) for tensor in head["tensors"])
    parsed = versions.parse(path)
    return {
        "file": os.path.basename(path),
        "base": parsed[0] if parsed else os.path.basename(path),
        "version": versions.format_version(parsed[1]) if parsed else "",
        "params": weights,
        "n_layer": config["n_layer"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "bits": head["bits"],
        "bytes": os.path.getsize(path),
    }


def build(folder=versions.MODELS_DIR):
    found = versions.existing(None, folder)
    return [describe(versions.path_for(base, version, folder)) for base, version in found]


def main():
    p = argparse.ArgumentParser(description="Write models/index.json.")
    p.add_argument("--models_dir", default=versions.MODELS_DIR)
    p.add_argument("--out", default="")
    args = p.parse_args()

    models = build(args.models_dir)
    out = args.out or os.path.join(args.models_dir, "index.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"models": models}, f, indent=1)
        f.write("\n")

    print(f"{out}: {len(models)} model(s)")
    for m in models:
        print(f"  {m['file']:22} {m['params']:>9,} params  {m['bytes']:>9,} bytes  "
              f"{m['n_layer']}x{m['n_embd']}")


if __name__ == "__main__":
    main()
