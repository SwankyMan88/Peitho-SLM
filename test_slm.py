"""End-to-end check: corpus -> training -> export -> pure-Python inference.

Runs on CPU in about a minute with a deliberately tiny model, so it can gate a
commit. It is not a quality test - the model it trains is far too small to say
anything - it tests that the pieces still fit together, which is what breaks.

    py test_slm.py
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import arith
import compose
import versions
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, build_vocab, encode, decode

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def run(*args, cwd=HERE):
    """A script run the way a user would run it."""
    result = subprocess.run([PY, "-X", "utf8", *args], cwd=cwd, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def test_markers():
    """A literal end marker inside a turn would teach the model to stop mid-sentence."""
    for path in ("conversations.txt",):
        with open(os.path.join(HERE, path), encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        stray = [l for l in lines if END_MARK in l[:-1]]
        check(f"{path}: no end marker inside a turn", not stray,
              f"{len(stray)} lines, first: {stray[0][:60] if stray else ''}")
        wrong = [l for l in lines if not l.startswith((USER_MARK, BOT_MARK))]
        check(f"{path}: every line is a marked turn", not wrong,
              f"{len(wrong)} lines, first: {wrong[0][:60] if wrong else ''}")


def test_vocab_round_trip():
    text = "Hello 123 " + START_MARK + USER_MARK + BOT_MARK + END_MARK
    stoi, itos = build_vocab(text)
    check("encode/decode round trip", decode(encode(text, stoi), itos) == text)


def test_arithmetic_is_correct():
    """The corpus teaches a method, so the worked answers in it must be right."""
    rng = random.Random(0)
    wrong = []
    for _ in range(4000):
        op = rng.choice("+-*/")
        lhs, rhs = arith.operands(rng, op)
        truth = (lhs + rhs if op == "+" else lhs - rhs if op == "-"
                 else lhs * rhs if op == "*" else lhs // rhs)
        if op == "/" and not truth:
            continue  # "does not fit even once" states the answer in words
        for method in arith.METHODS[op]:
            text = method(rng, lhs, rhs)
            if text and str(truth) not in text:
                wrong.append((op, lhs, rhs, truth, method.__name__, text))
    check("every worked sum states the true answer", not wrong,
          f"{len(wrong)} wrong, first: {wrong[0] if wrong else ''}")


def test_composition_is_varied():
    """Repeated text is what the model recites back, so the generators must not
    return the same sentences over and over."""
    rng = random.Random(0)
    drawn = [compose.statement(rng, "kettle") for _ in range(600)]
    check("composed sentences are mostly distinct", len(set(drawn)) > 570,
          f"{len(set(drawn))} distinct of 600")
    greetings = {compose.social_exchange(rng)[1] for _ in range(400)}
    check("greetings have many replies", len(greetings) > 60, f"{len(greetings)} distinct")


def test_versioning():
    with tempfile.TemporaryDirectory() as folder:
        first = versions.next_path("small", folder)
        check("first export is _1.0", first.endswith("small_1.0.txt"), first)
        open(first, "w").close()
        second = versions.next_path("small", folder)
        check("second export is _1.1", second.endswith("small_1.1.txt"), second)
        open(second, "w").close()
        open(os.path.join(folder, "small_1.4.txt"), "w").close()  # renamed by hand
        check("naming continues past the highest version, never overwriting",
              versions.next_path("small", folder).endswith("small_1.5.txt"))
        check("resolve finds the highest version",
              versions.resolve("small", folder).endswith("small_1.4.txt"))
        check("a different base name starts its own series",
              versions.next_path("medium", folder).endswith("medium_1.0.txt"))


def test_pipeline():
    """The real thing, end to end, at a size that finishes quickly."""
    work = tempfile.mkdtemp(prefix="slm_test_")
    try:
        code, out = run("make_corpus.py", "--target_chars", "60000",
                        "--train_out", os.path.join(work, "t.txt"),
                        "--heldout_out", os.path.join(work, "h.txt"))
        check("make_corpus runs", code == 0, out[-400:])
        corpus = os.path.join(work, "t.txt")
        check("corpus was written", os.path.exists(corpus) and os.path.getsize(corpus) > 50000)

        code, out = run("train.py", "--data", corpus,
                        "--val_data", os.path.join(work, "h.txt"),
                        "--fresh", "--steps", "20", "--eval_interval", "20",
                        "--preset", "tiny", "--block_size", "64", "--batch_size", "8",
                        "--out_dir", work, "--models_dir", work, "--name", "test")
        check("train runs and exports", code == 0, out[-600:])

        export = os.path.join(work, "test_1.0.txt")
        check("export exists", os.path.exists(export))
        with open(export, encoding="utf-8") as f:
            lines = f.read().splitlines()
        check("export is exactly 3 lines", len(lines) == 3, f"{len(lines)} lines")
        check("line 1 is JSON", lines[0].startswith("{"))
        for bad in ("'", '"', "\\"):
            check(f"payload contains no {bad!r} (so it pastes into a JS literal)",
                  bad not in lines[1] and bad not in lines[2])

        code, out = run("standalone.py", "--model", export, "--prompt", "hello",
                        "--max_new", "20")
        check("standalone.py runs without torch or numpy", code == 0, out[-400:])
        check("standalone.py produced a reply", "Bot:" in out, out[-200:])

        code, out = run("make_manifest.py", "--models_dir", work,
                        "--out", os.path.join(work, "index.json"))
        check("make_manifest runs", code == 0, out[-300:])
        with open(os.path.join(work, "index.json"), encoding="utf-8") as f:
            listing = json.load(f)
        entry = listing["models"][0]
        check("manifest lists the export", entry["file"] == "test_1.0.txt", str(entry))
        check("manifest counts parameters", entry["params"] > 0, str(entry))

        # peitho.html itself has no MODEL block - it fetches models/ - so baking is
        # checked against a stub standing in for a page that cannot fetch.
        template = os.path.join(work, "template.html")
        with open(template, "w", encoding="utf-8", newline="\n") as f:
            f.write("<!DOCTYPE html>\n<html><body><script>\nvar MODEL = {\n"
                    "    header:  null,\n    weights: '',\n    scales:  ''\n};\n"
                    "</script></body></html>\n")
        code, out = run("make_html.py", "--model", export, "--template", template,
                        "--out", os.path.join(work, "page.html"))
        check("make_html bakes a page", code == 0, out[-400:])
        page = os.path.join(work, "page.html")
        if os.path.exists(page):
            with open(page, encoding="utf-8") as f:
                html = f.read()
            # The page may fetch models/ beside itself, but must never reach off
            # the host: it has to keep working with no network at all.
            # The page may fetch models/ beside itself, but must never reach off
            # the host: it has to keep working with no network at all.
            offsite = ("src=\"http", "href=\"http", "fetch(\"http", "fetch('http",
                       "<link", "<script src")
            check("page requests nothing from another host",
                  not any(mark in html for mark in offsite))
            check("page carries the weights", len(html) > os.path.getsize(export) * 0.9)
            check("page records which export it carries",
                  "file:    'test_1.0.txt'" in html)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    print("Checking the corpus sources")
    test_markers()
    test_vocab_round_trip()
    test_arithmetic_is_correct()
    test_composition_is_varied()
    print("\nChecking export naming")
    test_versioning()
    print("\nChecking the whole pipeline")
    test_pipeline()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
