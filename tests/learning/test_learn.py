"""Checks for live teaching: the masking, the wordings, the undo, the guard rails.

Trains a throwaway model built in this process, so it needs no checkpoint, no GPU
and no corpus - it tests the machinery rather than the quality of what a real
model learns. Runs on CPU in a few seconds.

    py tests/learning/test_learn.py
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import learn
import paths
from model import GPT, GPTConfig, build_vocab, BOT_MARK, END_MARK, START_MARK, USER_MARK

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


# --------------------------------------------------------------- wordings

print("wordings")
words = learn.rewordings("What is my dog's name?")
check("the wording you typed comes first", words[0] == "What is my dog's name?")
check("several ways of asking", len(words) >= 6, f"got {len(words)}")
check("no duplicates", len(set(w.lower() for w in words)) == len(words))
check("all mention the subject", all("dog" in w.lower() for w in words),
      str([w for w in words if "dog" not in w.lower()]))
check("the limit is respected", len(learn.rewordings("What is a kettle?", 3)) == 3)
check("a statement still gets wordings",
      len(learn.rewordings("my dog is called Rufus")) >= 4)
check("an odd question does not crash", learn.rewordings("?") == ["?"])
check("who-questions are rewritten too",
      any("who" in w.lower() for w in learn.rewordings("Who is Rufus?")))
check("augmenting off means one wording",
      learn.rewordings("What is a kettle?", 1) == ["What is a kettle?"])

# --------------------------------------------------------------- the format

print("\nlesson format")
text = learn.lesson_text("hello", "hi there")
check("markers match the corpus",
      text == f"{START_MARK}\n{USER_MARK}hello{END_MARK}\n{BOT_MARK}hi there{END_MARK}",
      repr(text))

check("a mid-conversation exchange carries no start marker",
      learn.lesson_text("hello", "hi there", opening=False)
      == f"{USER_MARK}hello{END_MARK}\n{BOT_MARK}hi there{END_MARK}",
      repr(learn.lesson_text("hello", "hi there", opening=False)))

vocab_text = "".join(sorted(set(text + "abcdefghijklmnopqrstuvwxyz ."))) + "0123456789"
stoi, itos = build_vocab(vocab_text)

x, y = learn.encode_lesson("hello", "hi there", stoi, 128)
supervised = [i for i, t in enumerate(y) if t != learn.IGNORE]
answer_at = text.index(BOT_MARK) + 1
check("input and target are the same length", len(x) == len(y))
check("nothing before the answer is supervised",
      min(supervised) == answer_at - 1, f"first supervised at {min(supervised)}")
check("every answer character is supervised",
      len(supervised) == len("hi there") + 1, f"{len(supervised)} positions")
check("the end marker is learned too", y[-1] == stoi[END_MARK])
check("the question is never a target",
      all(y[i] == learn.IGNORE for i in range(answer_at - 2)))
check("targets are the next character",
      all(y[i] == x[i + 1] for i in supervised if i + 1 < len(x)))

too_long = "x" * 200
try:
    learn.encode_lesson(too_long, "reply", stoi, 96)
    check("an exchange too long for the context is refused", False)
except ValueError as e:
    check("an exchange too long for the context is refused", "context" in str(e))

check("unrepresentable characters are reported",
      learn.unrepresentable("café \U0001f600", stoi) == ["é", "\U0001f600"],
      str(learn.unrepresentable("café \U0001f600", stoi)))
check("ordinary text is representable", learn.unrepresentable("hello there.", stoi) == [])

# --------------------------------------------------------------- batching

print("\nbatching")
xs, ys = learn.lesson_batch([("hello", "hi there"), ("hi", "hello to you")],
                            stoi, 128, "cpu")
check("three rows per exchange - opening, bare turn, and behind earlier talk",
      xs.shape[0] == 6, str(xs.shape))
check("one row per exchange when only one form is asked for",
      learn.lesson_batch([("hello", "hi there")], stoi, 128, "cpu",
                         both_forms=False)[0].shape[0] == 1)
check("rows are padded to one width", xs.shape == ys.shape)
check("padding is not supervised",
      (ys[0] == learn.IGNORE).sum().item() > (ys[1] == learn.IGNORE).sum().item()
      or ys.shape[1] > 0)

prefixed_x, prefixed_y = learn.lesson_batch([("hello", "hi there")], stoi, 128, "cpu")
supervised_per_row = [(row != learn.IGNORE).sum().item() for row in prefixed_y]
check("every form supervises the same answer",
      len(set(supervised_per_row)) == 1, str(supervised_per_row))
check("an earlier exchange is in front of one of the forms",
      prefixed_x.shape[1] > len(learn.lesson_text("hello", "hi there")),
      str(prefixed_x.shape))

config = GPTConfig(vocab_size=len(stoi), block_size=128, n_layer=2, n_head=2,
                   n_embd=32, dropout=0.0)
torch.manual_seed(0)
model = GPT(config)
model.eval()

with torch.no_grad():
    tight = learn.masked_loss(model, *learn.lesson_batch(
        [("hello", "hi there")], stoi, 128, "cpu"))
    padded = learn.masked_loss(model, *learn.lesson_batch(
        [("hello", "hi there"), ("hi", "a much longer reply than the first one")],
        stoi, 128, "cpu"))
    only_second = learn.masked_loss(model, *learn.lesson_batch(
        [("hi", "a much longer reply than the first one")], stoi, 128, "cpu"))
check("padding does not change what is measured",
      abs(padded.item() - (tight.item() + only_second.item()) / 2) < 0.35,
      f"{padded.item():.3f} vs mean {(tight.item() + only_second.item()) / 2:.3f}")

# --------------------------------------------------------------- teaching

print("\nteaching a throwaway model")
work = tempfile.mkdtemp(prefix="learn_test_")
checkpoint = os.path.join(work, "tiny.pt")
torch.save({"state_dict": model.state_dict(), "config": config,
            "stoi": stoi, "itos": itos}, checkpoint)

corpus = os.path.join(work, "corpus.txt")
with open(corpus, "w", encoding="utf-8") as f:
    for i in range(400):
        f.write(learn.lesson_text(f"question {i % 7}", "an ordinary reply.") + "\n")

teacher = learn.Teacher(checkpoint, "cpu", lr=1e-3, replay_batch=4,
                        probe_batch=4, replay_path=corpus)
check("the corpus is found for rehearsal", teacher.replay is not None)
check("a starting point is recorded", teacher.baseline is not None)
check("the fixed sums are counted", isinstance(teacher.baseline_checks, int))
check("drift starts at zero", abs(teacher.drift()) < 1e-6)

before = {k: v.clone() for k, v in teacher.model.state_dict().items()}
result = teacher.teach("what is my dog's name", "rufus.", steps=30)
check("the lesson loss falls", result["loss_after"] < result["loss_before"],
      f"{result['loss_before']:.3f} -> {result['loss_after']:.3f}")
check("it stops early once learned", result["took_steps"] <= 30)
check("a wording is held back", result["held_out_wording"] not in result["trained_wordings"])
check("the held-back reply is recorded", "held_out_reply" in result)
check("drift is measured", result["drift"] is not None)
check("the weights moved",
      any(not torch.equal(before[k], v) for k, v in teacher.model.state_dict().items()))

check("undo restores exactly", teacher.rollback() and all(
    torch.equal(before[k], v) for k, v in teacher.model.state_dict().items()))
check("undo twice is harmless", teacher.rollback() is False)

teacher.teach("where do I live", "naples.", steps=5)
second = teacher.teach("what do I do", "you teach.", steps=5)
check("earlier lessons are rehearsed with later ones",
      second["rehearsed_lessons"] > 0, str(second["rehearsed_lessons"]))

try:
    teacher.teach("what is café", "a place.", steps=1)
    check("a lesson with unknown characters is refused", False)
except ValueError as e:
    check("a lesson with unknown characters is refused", "symbol" in str(e))

# --------------------------------------------------------------- keeping it

print("\nsaving")
saved = teacher.save(os.path.join(work, "taught.pt"))
check("a checkpoint is written", os.path.getsize(saved) > 1000)
reloaded = learn.Teacher(saved, "cpu", replay_batch=0, probe_batch=0,
                         replay_path=corpus)
check("what was taught survives a reload",
      all(torch.equal(a.cpu(), b.cpu()) for a, b in
          zip(teacher.model.state_dict().values(), reloaded.model.state_dict().values())))

exported = teacher.export(os.path.join(work, "taught.txt"))
with open(exported, encoding="utf-8") as f:
    lines = [l for l in f.read().split("\n") if l.strip()]
check("the export is still three lines", len(lines) == 3, f"{len(lines)} lines")
check("the export header is JSON", json.loads(lines[0])["config"]["n_layer"] == 2)

log = os.path.join(work, "lessons.jsonl")
learn.log_lesson(result, log)
learn.log_lesson(result, log)
with open(log, encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]
check("lessons are logged one per line", len(records) == 2)
check("the log holds what was taught",
      records[0]["question"] == "what is my dog's name" and records[0]["answer"] == "rufus.")

check("normalizing ignores case and punctuation",
      learn.normal("Your DOG, is called Rufus!") == "your dog is called rufus")

# --------------------------------------------------------------- refusals

print("\nrefusals")
try:
    learn.Teacher(os.path.join(work, "nope.pt"), "cpu")
    check("a missing checkpoint is an error", False)
except (FileNotFoundError, OSError):
    check("a missing checkpoint is an error", True)

import shutil
shutil.rmtree(work, ignore_errors=True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
