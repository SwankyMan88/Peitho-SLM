"""Write each export a second time as models/<name>.js.

Some sandboxes forbid fetch and XHR to another host while still allowing a script
tag - a Content-Security-Policy can block connect-src and allow script-src - so an
export that is also a .js file can be loaded where the .txt cannot be read.

The .js form registers itself on a global and calls a hook if one is waiting:

    (window.PEITHO_MODELS = window.PEITHO_MODELS || {})["small_1.2"] = {...};
    if (window.PEITHO_ARRIVED) { window.PEITHO_ARRIVED("small_1.2"); }

    py make_js_models.py
"""

import argparse
import os

import versions


def wrap(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != 3:
        raise SystemExit(f"{path}: expected 3 lines, found {len(lines)}")
    header, weights, scales = lines

    # base85 excludes quotes and backslashes, so single quotes are always safe -
    # but a corrupt file would break the script silently, so check.
    for name, blob in (("weights", weights), ("scales", scales)):
        for bad in ("'", '"', "\\"):
            if bad in blob:
                raise SystemExit(f"{path}: {name} contains {bad!r} and would break the JS")

    key = os.path.basename(path)[:-len(".txt")]
    return (f"(window.PEITHO_MODELS = window.PEITHO_MODELS || {{}})['{key}'] = {{\n"
            f"    header: {header},\n"
            f"    weights: '{weights}',\n"
            f"    scales: '{scales}'\n"
            "};\n"
            f"if (window.PEITHO_ARRIVED) {{ window.PEITHO_ARRIVED('{key}'); }}\n")


def main():
    p = argparse.ArgumentParser(description="Mirror models/*.txt as loadable .js files.")
    p.add_argument("--models_dir", default=versions.MODELS_DIR)
    p.add_argument("--only", default="",
                   help="Base name to wrap, for example \"medium\". Default is all of them.")
    args = p.parse_args()

    found = versions.existing(args.only or None, args.models_dir)
    if not found:
        raise SystemExit(f"No exports in {args.models_dir}/")

    for base, version in found:
        source = versions.path_for(base, version, args.models_dir)
        out = source[:-len(".txt")] + ".js"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(wrap(source))
        print(f"  {source} -> {out} ({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
