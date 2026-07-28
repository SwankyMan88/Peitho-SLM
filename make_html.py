"""Bake model_compressed.txt into peitho.html as a single self-contained page.

    py make_html.py             -> peitho_model.html
    py make_html.py --inplace   -> overwrite peitho.html
"""

import argparse
import json
import os
import re

import versions


def main():
    p = argparse.ArgumentParser(description="Embed the model into the HTML page.")
    p.add_argument("--model", default="",
                   help="Path to an export, or a base name such as \"small\" for its "
                        "highest version, or omitted for the most recent export.")
    p.add_argument("--template", default="peitho.html")
    p.add_argument("--out", default="peitho_model.html")
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

    block = ("var MODEL = {\n"
             f"    header:  {head_line},\n"
             f"    weights: '{weights}',\n"
             f"    scales:  '{scales}'\n"
             "};")
    pattern = re.compile(r"var MODEL = \{.*?\};", re.DOTALL)
    if not pattern.search(html):
        raise SystemExit(f"no MODEL block found in {args.template}")
    html = pattern.sub(lambda _: block, html, count=1)

    out = args.template if args.inplace else args.out
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    cfg = head["config"]
    print(f"Embedded {model_path} ({head['bits']}-bit, group {head['group']}) into {out}")
    print(f"  {cfg['n_layer']} layers, {cfg['n_embd']} dim, context {cfg['block_size']}, "
          f"vocab {cfg['vocab_size']}")
    print(f"  page size: {os.path.getsize(out):,} bytes")


if __name__ == "__main__":
    main()
