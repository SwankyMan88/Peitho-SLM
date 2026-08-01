"""Bake an export into a page template that has a `var MODEL = {...};` block.

peitho.html does not: it loads every model from models/ at run time, so it needs
no baking and no build step. This is for pages that have to carry their weights
because they cannot fetch anything - a sandbox with no networking, say.

    py make_html.py --template mypage.html --out built.html --model small
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slm"))
import paths
import versions


def main():
    p = argparse.ArgumentParser(description="Embed the model into the HTML page.")
    p.add_argument("--model", default="",
                   help="Path to an export, or a base name such as \"small\" for its "
                        "highest version, or omitted for the most recent export.")
    p.add_argument("--template", required=True,
                   help="Page with a `var MODEL = {...};` block to replace.")
    p.add_argument("--out", default="")
    p.add_argument("--inplace", action="store_true")
    args = p.parse_args()

    model_path = versions.resolve(args.model)
    with open(model_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) != 3:
        raise SystemExit(f"{model_path}: expected 3 lines, found {len(lines)}")

    head_line, weights, scales = lines
    head = json.loads(head_line)

    # The payload goes inside single-quoted JS strings; the base85 alphabet
    # guarantees no quote or backslash, but a corrupt file would break the page
    # silently, so check. The header is emitted unquoted, as a JS object literal.
    for name, blob in (("weights", weights), ("scales", scales)):
        for bad in ("'", '"', "\\"):
            if bad in blob:
                raise SystemExit(f"{name} line contains {bad!r} and would break the JS string")

    with open(args.template, "r", encoding="utf-8") as f:
        html = f.read()

    # `file` lets the page tell which manifest entry it already carries, so the
    # built-in model is not offered twice.
    block = ("var MODEL = {\n"
             f"    file:    '{os.path.basename(model_path)}',\n"
             f"    header:  {head_line},\n"
             f"    weights: '{weights}',\n"
             f"    scales:  '{scales}'\n"
             "};")
    pattern = re.compile(r"var MODEL = \{.*?\};", re.DOTALL)
    if not pattern.search(html):
        raise SystemExit(
            f"no `var MODEL = {{...}};` block in {args.template}. peitho.html has none "
            "by design - it loads models/ at run time. Bake into a template that "
            "cannot fetch instead.")
    html = pattern.sub(lambda _: block, html, count=1)

    out = args.template if args.inplace else (args.out or "built.html")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    cfg = head["config"]
    print(f"Embedded {paths.short(model_path)} ({head['bits']}-bit, group {head['group']}) into {paths.short(out)}")
    print(f"  {cfg['n_layer']} layers, {cfg['n_embd']} dim, context {cfg['block_size']}, "
          f"vocab {cfg['vocab_size']}")
    print(f"  page size: {os.path.getsize(out):,} bytes")


if __name__ == "__main__":
    main()
