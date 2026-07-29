"""Check every implementation against conformance/vectors.json.

Three decoders read the same export: PyTorch, the pure-Python one in
standalone.py, and the JavaScript inside peitho.html. A port is only correct if it
agrees with the stored numbers, so this runs all three against them.

    py tests/test_conformance.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

KIT = os.path.join(paths.ROOT, "conformance")
EXPORT = os.path.join(KIT, "micro_1.0.txt")
VECTORS = os.path.join(KIT, "vectors.json")

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def worst(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def load_vectors():
    with open(VECTORS, encoding="utf-8") as f:
        return json.load(f)


def check_pytorch(vectors):
    import torch
    from export import import_compressed

    model, config, stoi, itos = import_compressed(EXPORT, device="cpu")
    model.eval()
    for case in vectors["cases"]:
        with torch.no_grad():
            out, _ = model(torch.tensor([case["ids"]], dtype=torch.long))
        gap = worst(out[0, -1].tolist(), case["logits"])
        check(f"pytorch logits for {case['prompt']!r}", gap <= vectors["tolerance"],
              f"worst logit differs by {gap:.2e}")


def check_standalone(vectors):
    sys.path.insert(0, paths.ROOT)
    import standalone

    model = standalone.Model(EXPORT)
    for case in vectors["cases"]:
        model.reset()
        logits = None
        for token in case["ids"]:
            logits = model.step(token)
            model.pos += 1
        gap = worst(logits, case["logits"])
        check(f"pure python logits for {case['prompt']!r}", gap <= vectors["tolerance"],
              f"worst logit differs by {gap:.2e}")

        # The greedy continuation catches an error the logits alone can miss: a
        # key-value cache that is right on the first token and wrong afterwards.
        ids = list(case["ids"])
        for _ in range(vectors["greedy_steps"]):
            model.reset()
            step_logits = None
            for token in ids[-model.block_size:]:
                step_logits = model.step(token)
                model.pos += 1
            ids.append(max(range(len(step_logits)), key=lambda i: step_logits[i]))
        check(f"pure python greedy for {case['prompt']!r}", ids == case["greedy_ids"],
              f"got {ids}, expected {case['greedy_ids']}")


def check_javascript(vectors):
    """Run the decoder that ships inside peitho.html, under node."""
    node = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if node.returncode != 0:
        print("  skip  javascript: node is not installed")
        return

    runner = os.path.join(KIT, "check.mjs")
    result = subprocess.run(["node", runner], capture_output=True, text=True,
                            cwd=paths.ROOT, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        check("javascript agrees with the vectors", False,
              (result.stdout + result.stderr).strip()[-400:])
        return
    for line in result.stdout.splitlines():
        if line.startswith("ok "):
            check("javascript " + line[3:], True)
        elif line.startswith("fail "):
            check("javascript " + line[5:], False, "see conformance/check.mjs output")


def main():
    if not os.path.exists(VECTORS):
        raise SystemExit("No conformance kit. Run py tools/make_conformance.py first.")
    vectors = load_vectors()
    print(f"Checking {len(vectors['cases'])} cases against "
          f"{paths.short(EXPORT)} (tolerance {vectors['tolerance']:g})\n")
    check_pytorch(vectors)
    check_standalone(vectors)
    check_javascript(vectors)
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
