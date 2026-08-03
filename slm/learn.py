"""Teach a trained model something new, live, one example at a time.

Two modes and one switch between them:

    teach   you give the question and the answer you wanted. It learns the pair.
    chat    an ordinary conversation. When a reply is wrong, /teach corrects it.

A lesson is not one gradient step on one string. Four things happen together, and
all four matter:

    the question is reworded several ways      so the answer attaches to the
                                              question's meaning, not its spelling
    one wording is held back                  so "did it learn?" can be answered
                                              with a phrasing it never trained on
    the loss covers only the answer            so it learns to reply, not to
                                              write questions like yours
    slices of the original corpus ride along   so the rest of the model is
                                              rehearsed in the same step

Without that last one a handful of steps on a single sentence produces a model
that answers one question perfectly and has forgotten how to form a sentence.

Everything is per-example: no training run, no retraining. Loading is from the
full-precision checkpoint, because gradients need the precision a quantized
export has thrown away.

    py slm/learn.py --checkpoint build/medium_v15.pt
"""

import argparse
import difflib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

import torch
from torch.nn import functional as F

import paths
import versions
from model import GPT, START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK
from model import encode, decode
from export import export_compressed
from train import load_ids, make_optimizer
from chat import build_prompt, clean_response, load_full_checkpoint, show, DIM, PLAIN

IGNORE = -100          # cross_entropy skips these positions
LESSON_LOG = os.path.join(paths.BUILD, "lessons.jsonl")

# Sums the model could do before the session started, checked again after every
# lesson. Corpus loss is not enough on its own: three lessons once moved the probe
# by +0.037 - nothing, apparently - while quietly taking arithmetic from 4/4 to
# 0/4. Behaviour has to be watched directly, so this watches it.
CHECK_SUMS = [(700, 933), (148, 267), (512, 205), (88, 45), (309, 87), (26, 418)]

# Corpus to rehearse from. build/ first, since that is what a local training run
# produced; the copy shipped in data/ is the fallback for a fresh clone.
REPLAY_FILES = [paths.TRAINING, os.path.join(paths.ROOT, "data", "training.txt")]


# --------------------------------------------------------------------- wordings

# Rewrites for the question forms that actually come up, tried in order. The
# capture is the subject; the rewrites are ways of asking the same thing.
FORMS = [
    (r"^what(?:'s|s| is| are)\s+(.+)$", [
        "what is {x}", "what's {x}", "whats {x}", "do you know what {x} is",
        "tell me what {x} is", "{x} - what is that", "remind me what {x} is",
        "can you tell me what {x} is", "any idea what {x} is",
    ]),
    (r"^who(?:'s|s| is| are)\s+(.+)$", [
        "who is {x}", "who's {x}", "whos {x}", "do you know who {x} is",
        "tell me who {x} is", "{x} - who is that", "remind me who {x} is",
        "can you tell me who {x} is",
    ]),
    (r"^where(?:'s|s| is| are)\s+(.+)$", [
        "where is {x}", "where's {x}", "wheres {x}", "do you know where {x} is",
        "tell me where {x} is", "remind me where {x} is",
        "can you tell me where {x} is",
    ]),
    (r"^when(?:'s|s| is| are| was)\s+(.+)$", [
        "when is {x}", "when's {x}", "do you know when {x} is",
        "tell me when {x} is", "remind me when {x} is",
    ]),
    (r"^how (?:do|does|did) (.+)$", [
        "how do {x}", "how does {x}", "do you know how {x}",
        "tell me how {x}", "any idea how {x}",
    ]),
    (r"^why (?:do|does|is|are) (.+)$", [
        "why is {x}", "why does {x}", "do you know why {x}",
        "tell me why {x}", "any idea why {x}",
    ]),
]

# Applied to anything the forms above do not match - a statement, an imperative,
# an odd phrasing. Deliberately mild: they must not change what was asked.
WRAPPERS = [
    "{q}", "{q}?", "hey - {q}", "quick one: {q}", "so {q}",
    "{q}, if you know", "do you remember {q}",
]


def agree(text):
    """Fix the verb after substituting a subject into a template.

    "Who is Rufus?" rewrites cleanly, but the same template with "you" as the
    subject produces "remind me who you is" - and a wording the model trains on
    should not teach broken grammar."""
    text = re.sub(r"\b(you|we|they|You|We|They) is\b", r"\1 are", text)
    text = re.sub(r"\bI is\b", "I am", text)
    # The question-word templates put the verb in front of the subject, where the
    # same disagreement reads as "who is you" and "who's you".
    text = re.sub(r"\b(who|what|where|when)(?:'s|s| is) (you|we|they)\b",
                  r"\1 are \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(who|what|where|when)(?:'s|s| is) I\b", r"\1 am I", text,
                  flags=re.IGNORECASE)
    return text


def rewordings(question, limit=8):
    """Several ways of asking the same thing, the original first.

    One example teaches a string; the same answer against several phrasings
    teaches a mapping. This is the whole difference between a model that only
    responds to your exact keystrokes and one that recognizes the question."""
    core = question.strip().rstrip("?").strip()
    if not core:
        return [question.strip()]

    out = [question.strip()]
    for pattern, rewrites in FORMS:
        match = re.match(pattern, core, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            out += [r.format(x=subject) for r in rewrites]
            break
    else:
        # Lowercased when something precedes it, so "hey - What kind of food..."
        # does not train a capital in the middle of a sentence.
        lowered = core[0].lower() + core[1:]
        out += [w.format(q=core if w.startswith("{q}") else lowered)
                for w in WRAPPERS]

    # Punctuation and capitalization are two more things the answer should not
    # depend on, and they cost nothing to vary.
    for text in list(out):
        out.append(text[0].upper() + text[1:] if text else text)
        if not text.endswith("?"):
            out.append(text + "?")

    # Case-sensitive deduplication on purpose. To a character-level model "Who"
    # and "who" share no characters at all, so folding them together drops a
    # wording that genuinely needs teaching - and people type the lowercase one.
    seen, unique = set(), []
    for text in out:
        key = agree(text.strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique[:limit]


# ---------------------------------------------------------------------- lessons

# Plausible earlier turns to sit a lesson behind. Their content does not matter -
# the loss never covers them - but their length does: positional embeddings are
# learned, so an exchange at offset 0 and the same exchange at offset 90 are
# genuinely different inputs to the model. A lesson taught only at offset 0 fires
# only at the start of a conversation.
CONTEXTS = [
    f"{USER_MARK}Hi, got a minute?{END_MARK}\n{BOT_MARK}Morning. Ask away.{END_MARK}\n",
    f"{USER_MARK}What is a kettle?{END_MARK}\n{BOT_MARK}It boils water.{END_MARK}\n",
    f"{USER_MARK}700+933{END_MARK}\n{BOT_MARK}That comes to 1633.{END_MARK}\n",
    f"{USER_MARK}What is a spring?{END_MARK}\n{BOT_MARK}A spring stores force by "
    f"bending.{END_MARK}\n{USER_MARK}Does it wear out?{END_MARK}\n"
    f"{BOT_MARK}Slowly, if you look after it.{END_MARK}\n",
]


def lesson_text(question, answer, opening=True):
    """One taught exchange, in the marker format the corpus uses.

    With `opening`, the exchange begins a conversation and carries the start
    marker; without, it is a turn in the middle of one. Both forms occur in real
    prompts - chat.py drops the start marker once older turns scroll out, and
    standalone.py never sends one - and a lesson taught in only one form does not
    reliably fire in the other."""
    head = f"{START_MARK}\n" if opening else ""
    return (f"{head}{USER_MARK}{question}{END_MARK}\n"
            f"{BOT_MARK}{answer}{END_MARK}")


def unrepresentable(text, stoi):
    """Characters this model has no symbol for. The vocabulary is fixed at
    training time, so a curly quote or an emoji cannot be learned - and silently
    dropping it would teach a lesson subtly different from the one typed."""
    return sorted(set(text) - set(stoi))


def encode_lesson(question, answer, stoi, block_size, opening=True, context=""):
    """Input and target ids for one exchange, with the loss masked to the answer.

    Everything up to and including the bot marker is context: the model is not
    being taught to write the question. The final end marker IS included, so it
    learns where the reply stops."""
    text = lesson_text(question, answer, opening)
    if len(text) > block_size + 1:
        raise ValueError(f"the exchange is {len(text)} characters and only "
                         f"{block_size + 1} fit in this model's context")
    answer_at = len(context) + text.index(BOT_MARK) + 1
    if context and len(context) + len(text) <= block_size + 1:
        text = context + text
    else:
        answer_at = text.index(BOT_MARK) + 1       # it did not fit; teach it alone
    ids = encode(text, stoi)

    x = ids[:-1]
    y = [ids[i + 1] if i + 1 >= answer_at else IGNORE for i in range(len(x))]
    return x, y


def lesson_batch(pairs, stoi, block_size, device, both_forms=True):
    """Pad a set of exchanges into one batch.

    Each exchange goes in twice by default, as a conversation opening and as a
    turn in the middle of one, because a prompt may be either.

    Padding goes on the right and its targets are ignored, so it cannot affect
    the loss - and attention is causal, so it cannot affect earlier positions
    either."""
    if both_forms:
        # Three rows: opening a conversation, a bare turn, and a turn behind some
        # earlier talk - the three shapes a real prompt arrives in.
        rows = []
        for q, a in pairs:
            rows.append(encode_lesson(q, a, stoi, block_size, True))
            rows.append(encode_lesson(q, a, stoi, block_size, False))
            # One or two earlier exchanges, so the lesson is seen at a spread of
            # offsets rather than one. A real conversation is rarely one turn deep.
            prefix = "".join(random.sample(CONTEXTS, random.choice((1, 2))))
            rows.append(encode_lesson(q, a, stoi, block_size, False, prefix))
    else:
        rows = [encode_lesson(q, a, stoi, block_size) for q, a in pairs]
    width = max(len(x) for x, _ in rows)
    xs, ys = [], []
    for x, y in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [IGNORE] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


def masked_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                           ignore_index=IGNORE)


# ---------------------------------------------------------------------- teacher

class Teacher:
    """A loaded model that can be taught, tested, and put back the way it was."""

    def __init__(self, checkpoint, device, lr=1e-4, replay_batch=64,
                 replay_weight=2.0, probe_batch=32, replay_path=None):
        self.device = device
        self.model, self.config, self.stoi, self.itos = load_full_checkpoint(
            checkpoint, device)
        self.model.eval()          # no dropout: one example, deterministic lessons
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.stop_id = self.stoi.get(END_MARK)
        self.replay_weight = replay_weight
        self.replay_batch = replay_batch
        self.optimizer = make_optimizer(self.model, lr, weight_decay=0.0,
                                        device=device)
        self.lr = lr

        self.replay = None
        self.replay_from = None
        for path in ([replay_path] if replay_path else REPLAY_FILES):
            if path and os.path.exists(path):
                missing = None
                with open(path, "r", encoding="utf-8") as f:
                    head = f.read(4096)
                missing = unrepresentable(head, self.stoi)
                if missing:
                    continue           # a corpus this model cannot even read
                self.replay = load_ids(path, self.stoi, device)
                self.replay_from = path
                break

        self.probe_rows = self._probe_rows(probe_batch)
        self.baseline = self.probe() if self.probe_rows is not None else None
        self.saved_state = None
        self.lessons = []

        # Every pair taught this session. New lessons rehearse a sample of the old
        # ones, for the same reason they rehearse the corpus: without it, lesson
        # three overwrites lesson one.
        self.taught = []
        self.baseline_checks = self.checks()

    # -- rehearsal ---------------------------------------------------------

    def _slices(self, count, generator=None):
        """`count` uniformly random windows of the corpus, as one batch.

        Aiming half of them at the arithmetic - findable by searching for "+" - was
        the obvious improvement, on the theory that random slices of 30M characters
        rarely land on a sum. Measured over three runs of eight lessons it was
        worse, not better: 3, 2 and 0 of six sums kept against 3, 4 and 4 for
        uniform sampling. Rehearsing a skill narrowly is not the same as keeping
        it. Uniform stays."""
        # count <= 0 means rehearsal is switched off. Returning an empty batch would
        # put a zero-row tensor through the model, which is a shape some torch
        # versions accept and some refuse.
        if self.replay is None or count < 1:
            return None
        block = self.config.block_size
        high = self.replay.numel() - block - 1
        ix = torch.randint(high, (count,), generator=generator)
        window = torch.arange(block)
        rows = (ix[:, None] + window[None, :]).to(self.device)
        return self.replay[rows].long(), self.replay[rows + 1].long()

    def _probe_rows(self, count):
        """A fixed sample, drawn once from a fixed seed, so that "has the model
        changed?" is asked of the same text every time.

        Measuring the model's general ability, which is why it is a plain uniform
        sample and why CHECK_SUMS exists alongside it."""
        return self._slices(count, generator=torch.Generator().manual_seed(0))

    @torch.no_grad()
    def probe(self):
        """Loss on the fixed sample: the model's general ability in one number."""
        if self.probe_rows is None:
            return None
        was_training = self.model.training
        self.model.eval()
        x, y = self.probe_rows
        _, loss = self.model(x, y)
        self.model.train(was_training)
        return loss.item()

    def drift(self):
        """How far the fixed sample has moved since the session began. Positive is
        worse. Necessary but not sufficient - see checks()."""
        if self.baseline is None:
            return None
        return self.probe() - self.baseline

    def checks(self):
        """How many of the fixed sums it still gets right.

        Greedy sampling, so this is deterministic: a change here is the model
        changing, not the dice. Arithmetic is the most fragile thing the model
        does, which makes it the best early warning that a lesson went too far."""
        right = 0
        for a, b in CHECK_SUMS:
            reply = self.ask(f"{a}+{b}", [], temperature=0.2, top_k=1)
            right += str(a + b) in reply
        return right

    # -- undo --------------------------------------------------------------

    def snapshot(self):
        self.saved_state = {k: v.detach().clone().cpu()
                            for k, v in self.model.state_dict().items()}

    def rollback(self):
        if self.saved_state is None:
            return False
        self.model.load_state_dict(self.saved_state)
        self.model.to(self.device)
        self.saved_state = None
        return True

    # -- learning ----------------------------------------------------------

    def steps_on(self, pairs, steps, lr=None, grad_clip=1.0, target=0.05):
        """Take up to `steps` gradient steps on these exchanges, rehearsing the
        corpus alongside each one, and stop as soon as they are learned.

        Lesson and rehearsal are separate forward passes whose gradients
        accumulate before the step, which is one mixed batch in every way that
        matters and avoids padding corpus slices to the lesson's length.

        The early stop matters more than it looks: the loss reaches zero after a
        few steps, and every step after that is pressure on the rest of the model
        for nothing in return."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr or self.lr

        self.model.eval()
        x, y = lesson_batch(pairs, self.stoi, self.config.block_size, self.device)
        history = []
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(self.model, x, y)
            loss.backward()

            replay = self._slices(self.replay_batch)
            if replay is not None and self.replay_weight > 0:
                _, rloss = self.model(*replay)
                (rloss * self.replay_weight).backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            history.append(loss.item())
            if target and loss.item() < target:
                break

        with torch.no_grad():
            history.append(masked_loss(self.model, x, y).item())
        return history

    def teach(self, question, answer, steps=20, lr=None, limit=8, augment=True,
              holdout=True):
        """One lesson, then an honest test of what it bought.

        The held-back wording is the point: answering a phrasing that was never
        trained on is the difference between learning the answer and memorizing
        the keystrokes."""
        bad = unrepresentable(question + answer, self.stoi)
        if bad:
            raise ValueError("this model has no symbol for: " + " ".join(repr(c) for c in bad))

        # Teaching a worked sum competes with the arithmetic the model already
        # carries, and it loses badly: the lesson does not carry over to a new
        # wording, and it leaves the reply distribution ragged enough that sampling
        # above about 0.5 produces nonsense on unrelated questions.
        self.warning = None
        if re.search(r"\d\s*[-+*/]\s*\d", question) and re.search(r"\d", answer):
            self.warning = ("this teaches a specific sum, which fights the "
                            "arithmetic already in the model - it usually does not "
                            "carry over, and it can make replies ragged")

        wordings = rewordings(question, limit) if augment else [question.strip()]
        held = wordings[-1] if (holdout and len(wordings) > 2) else None
        trained = [w for w in wordings if w != held]

        # Earlier lessons ride along, so learning a third fact does not overwrite
        # the first. Only what fits: a batch is bounded by the context length.
        older = random.sample(self.taught, min(len(self.taught), 8))
        pairs = [(w, answer) for w in trained] + older

        self.snapshot()

        # Train in short rounds and stop when the model actually says the new
        # answer, rather than when the loss looks small.
        #
        # Mean loss is misleading precisely where teaching is hardest. Replacing an
        # answer the model already had, the first character is the whole fight - the
        # rest of the reply is easy to continue - so an average over sixty
        # characters reads 0.03 while the old reply still wins the only position
        # that decides which reply comes out.
        losses, done = [], 0
        while done < steps:
            round_steps = min(4, steps - done)
            losses += self.steps_on(pairs, round_steps, lr, target=0)
            done += round_steps
            said = self.ask(question, [], temperature=0.2, top_k=1)
            if difflib.SequenceMatcher(None, normal(said), normal(answer)).ratio() >= 0.9:
                break

        self.taught += [(w, answer) for w in trained]

        result = {
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "trained_wordings": trained,
            "held_out_wording": held,
            "steps": steps,
            "lr": lr or self.lr,
            "loss_before": losses[0],
            "loss_after": losses[-1],
            "took_steps": done,
            "reply": said,
            "learned": difflib.SequenceMatcher(
                None, normal(said), normal(answer)).ratio() >= 0.9,
            "rehearsed_lessons": len(older),
            "drift": self.drift(),
            "checks": self.checks(),
            "checks_baseline": self.baseline_checks,
            "warning": self.warning,
        }
        # Mid-conversation, where the start marker is gone and two exchanges are
        # already in view. Cold recall was never the hard case.
        elsewhere = [{"role": "user", "text": "What is a kettle?"},
                     {"role": "model", "text": "It boils water."}]
        result["mid_chat_reply"] = self.ask(question, elsewhere, temperature=0.2,
                                            top_k=1)
        result["holds_mid_chat"] = difflib.SequenceMatcher(
            None, normal(result["mid_chat_reply"]), normal(answer)).ratio() >= 0.9

        if held:
            reply = self.ask(held, [], temperature=0.2, top_k=1)
            result["held_out_reply"] = reply
            result["match"] = difflib.SequenceMatcher(
                None, normal(reply), normal(answer)).ratio()
            result["generalized"] = result["match"] >= 0.75
        self.lessons.append(result)
        return result

    # -- speaking ----------------------------------------------------------

    def ask(self, text, history, temperature=0.8, top_k=40, max_new_tokens=400):
        prompt = build_prompt(history, text, self.config.block_size)
        prompt = "".join(c for c in prompt if c in self.stoi)
        ids = encode(prompt, self.stoi)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k,
                                  stop_id=self.stop_id)
        self.model.eval()          # generate() leaves the model in train mode
        return clean_response(decode(out[0, len(ids):].tolist(), self.itos))

    # -- keeping it ---------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "stoi": self.stoi,
            "itos": self.itos,
        }, path)
        return path

    def export(self, path, bits=8, group_size=32):
        """Write the 3-line quantized export - the same format train.py produces, so
        a taught model is interchangeable with the released ones."""
        export_compressed(self.model, self.config, self.stoi, self.itos, path,
                          bits=bits, group_size=group_size)
        return path


def normal(text):
    """Lowercase, unpunctuated, single-spaced - for comparing two answers by what
    they say rather than how they are typed."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def past_lessons(path):
    """Every kept lesson from an earlier session, as (wording, answer) pairs.

    Rehearsal is what stops a new lesson overwriting an older one, and it works
    from a list held in memory - which a reloaded checkpoint does not have. Without
    this, picking a taught model up tomorrow means teaching it fact eleven while it
    quietly loses facts one through ten. The log is the only record of what was
    taught, so the log is what restores it."""
    if not path or not os.path.exists(path):
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue           # a truncated final line, from an interrupted run
            if record.get("rolled_back"):
                continue
            for wording in record.get("trained_wordings", []):
                pairs.append((wording, record["answer"]))
    return pairs


def log_lesson(record, path=LESSON_LOG):
    """Append the lesson to a plain log.

    Weights record what was learned but not what was taught. With this, a session
    is replayable from the original checkpoint, and a run that went wrong can be
    rebuilt without the bad lessons instead of being lost."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------- CLI

HELP = """
  /mode            switch between teach and chat
  /teach <reply>   in chat: what it should have said to your last message
  /good            in chat: that reply was right - learn it
  /polish [steps]  train every lesson together, so the newest stops winning ties
  /undo            put the weights back the way they were before the last lesson
  /probe           how far the model has drifted from where it started
  /ask <text>      one question without adding to the conversation
  /context         the prompt the model actually sees
  /lessons         what has been taught this session
  /save [path]     write the full-precision checkpoint
  /export [path]   write the quantized 3-line export, versioned into models/
                   exactly like train.py does - the web page will find it
  /forget          clear the conversation (the weights keep what they learned)
  /help            this
  /exit            leave
"""


def report(result):
    print(f"  taught {len(result['trained_wordings'])} wordings in "
          f"{result['took_steps']} steps, "
          f"loss {result['loss_before']:.3f} -> {result['loss_after']:.3f}")
    if not result["learned"]:
        print(f"  it still answers \"{result['reply'][:70]}\" - the old answer is "
              "winning. More --steps, or say it closer to how the model already talks.")
    if result.get("held_out_wording"):
        verdict = "generalized" if result["generalized"] else "did not carry over"
        print(f"  held back \"{result['held_out_wording']}\" -> "
              f"\"{result['held_out_reply']}\"")
        print(f"  {verdict} ({result['match']:.0%} match to what you taught)")
    if not result.get("holds_mid_chat"):
        print(f"  mid-conversation it says \"{result['mid_chat_reply'][:60]}\" instead "
              "- ask it early in a chat, or teach it again")
    if result.get("drift") is not None:
        print(f"  drift from the original model: {result['drift']:+.4f}")
    lost = result["checks_baseline"] - result["checks"]
    print(f"  sums it can still do: {result['checks']}/{len(CHECK_SUMS)}"
          + (f"  <- {lost} fewer than when you started" if lost > 0 else ""))
    if result.get("warning"):
        print(f"  note: {result['warning']}. /undo takes it back.")


def export_now(teacher, args, target=""):
    """Write the export where train.py would write it, and say the same things.

    Named and versioned like every other export - models/taught_1.0.txt, then
    taught_1.1.txt - so peitho.html finds it by the same probe it uses for the
    released models, benchmark.py and standalone.py accept it by base name, and
    nothing about a taught model is a special case."""
    path = target or versions.next_path(args.name, args.models_dir)
    teacher.export(path, bits=args.bits, group_size=args.group_size)

    size = os.path.getsize(path)
    print(f"  exported {paths.short(path)} ({args.bits}-bit, group "
          f"{args.group_size}, 3 lines)")
    print(f"  {size:,} bytes for {teacher.n_params:,} parameters "
          f"({8 * size / teacher.n_params:.2f} bits/param on disk)")
    if os.path.dirname(os.path.abspath(path)) == os.path.abspath(args.models_dir):
        base = os.path.basename(path)[:-len(".txt")]
        print(f"  peitho.html will offer it as \"{base.replace('_', ' ')}\". "
              f"For a sandbox that cannot fetch: "
              f"py tools/make_js_models.py --only {args.name}")
    print()
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Teach the SLM live, one example at a time.")
    p.add_argument("--checkpoint", default=paths.CHECKPOINT,
                   help="Full-precision checkpoint. Quantized exports cannot be trained.")
    p.add_argument("--mode", choices=("teach", "chat"), default="teach")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Small on purpose: a lesson should nudge the model, not brand "
                        "itself into it.")
    p.add_argument("--steps", type=int, default=20,
                   help="Most gradient steps a lesson may take. It stops early once "
                        "the exchange is learned, which is usually 10 to 15.")
    p.add_argument("--wordings", type=int, default=8,
                   help="How many ways to ask the same question. 1 disables augmenting, "
                        "which means it memorizes the phrasing you typed.")
    p.add_argument("--no_holdout", action="store_true",
                   help="Train on every wording. Learns marginally more and loses the "
                        "only honest test of whether it generalized.")
    p.add_argument("--replay_batch", type=int, default=64,
                   help="Corpus slices rehearsed alongside each lesson. 0 disables "
                        "rehearsal, and the model will start forgetting. Measured: 16 "
                        "slices cost two of six fixed sums over five lessons; 64 cost "
                        "none.")
    p.add_argument("--replay_weight", type=float, default=2.0,
                   help="How much the rehearsal counts against the lesson.")
    p.add_argument("--replay_data", default="",
                   help="Corpus to rehearse from. Defaults to build/ then data/.")
    p.add_argument("--max_drift", type=float, default=0.15,
                   help="Undo a lesson automatically if it moves the fixed probe more "
                        "than this. 0 disables that half of the guard.")
    p.add_argument("--allow_forgetting", type=int, default=1,
                   help="How many of the fixed sums a lesson may cost before it is "
                        "undone. One is noise at this size; two is damage.")
    p.add_argument("--no_guard", action="store_true",
                   help="Keep every lesson, however much it costs.")
    p.add_argument("--style", action="store_true",
                   help="In chat mode, also learn from your messages as text. Picks up "
                        "your vocabulary; does not make its answers better.")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="Chat sampling, far below chat.py's 0.8 on purpose. A lesson "
                        "makes the taught reply a narrow path. Measured over four "
                        "lessons, five samples each: 0.2 returned the taught answer "
                        "20/20 times, 0.35 managed 18, 0.7 only 14 - while untaught "
                        "questions still vary at 0.2.")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--autosave", default="",
                   help="Checkpoint to write after every lesson.")
    p.add_argument("--name", default="taught",
                   help="Base name for /export, versioned the way train.py versions "
                        "its exports: models/taught_1.0.txt, then taught_1.1.txt.")
    p.add_argument("--models_dir", default=versions.MODELS_DIR)
    p.add_argument("--bits", type=int, choices=(4, 8), default=8,
                   help="Export width. 8 is effectively lossless; 4 halves the file "
                        "and wrecks in-context recall.")
    p.add_argument("--group_size", type=int, default=32,
                   help="Weights per quantization scale.")
    p.add_argument("--log", default=LESSON_LOG,
                   help="Where lessons are recorded. Give a taught model its own file "
                        "and you can hand that file back with --remember.")
    p.add_argument("--remember", default="",
                   help="A lesson log from earlier sessions with this model. Its "
                        "lessons are rehearsed alongside new ones, so continuing "
                        "tomorrow does not cost what you taught today.")
    p.add_argument("--export_on_exit", action="store_true",
                   help="Write the export automatically when you leave, if anything "
                        "was taught and kept.")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No checkpoint at {paths.short(args.checkpoint)}. "
                         "Train one first, or pass --checkpoint.")

    print(f"Using device: {device}")
    print(f"Loading {paths.short(args.checkpoint)}...")
    teacher = Teacher(args.checkpoint, device, lr=args.lr,
                      replay_batch=args.replay_batch,
                      replay_weight=args.replay_weight,
                      replay_path=args.replay_data or None)

    print(f"{teacher.n_params:,} parameters, {teacher.config.block_size}-character context")
    if teacher.replay is None:
        print("  WARNING: no corpus to rehearse from, so every lesson eats into what "
              "the model already knew. Generate one, or pass --replay_data.")
    else:
        print(f"Rehearsing from {paths.short(teacher.replay_from)} "
              f"({teacher.replay.numel():,} characters)")
    print(f"Starting point: {teacher.baseline_checks}/{len(CHECK_SUMS)} fixed sums right, "
          f"probe loss {teacher.baseline:.4f}" if teacher.baseline is not None else
          f"Starting point: {teacher.baseline_checks}/{len(CHECK_SUMS)} fixed sums right")
    if args.remember:
        teacher.taught = past_lessons(args.remember)
        if teacher.taught:
            print(f"Remembering {len(teacher.taught)} wordings from "
                  f"{paths.short(args.remember)}, to rehearse with anything new")
        else:
            print(f"  WARNING: no lessons found in {paths.short(args.remember)}")
    print(f"Mode: {args.mode}. /help for commands, /exit to leave.\n")

    mode = args.mode
    history = []
    last_user = None
    last_reply = None

    def do_lesson(question, answer):
        try:
            result = teacher.teach(question, answer, steps=args.steps,
                                   limit=max(1, args.wordings),
                                   augment=args.wordings > 1,
                                   holdout=not args.no_holdout)
        except ValueError as e:
            print(f"  cannot teach that: {e}\n")
            return
        report(result)

        # Two ways a lesson can be judged too expensive: the corpus probe moved, or
        # something it used to get right it no longer does. The second catches what
        # the first misses.
        too_far = (args.max_drift and result.get("drift") is not None
                   and result["drift"] > args.max_drift)
        broke = (result["checks_baseline"] - result["checks"]) > args.allow_forgetting
        if not args.no_guard and (too_far or broke):
            teacher.rollback()
            teacher.taught = teacher.taught[:-len(result["trained_wordings"])]
            reason = ("it drifted too far" if too_far else
                      "it cost sums the model could do before")
            # Not "fewer steps": measured, --steps 6 dropped recall to 1 lesson in 5
            # while costing just as much. The model is the lever, not the schedule.
            print(f"  undone: {reason}. A larger checkpoint absorbs lessons far "
                  f"better - large took the same five lessons for nothing. Or pass "
                  f"--allow_forgetting 2 if the sums matter less than the lesson.")
            result["rolled_back"] = True
        result["checkpoint"] = paths.short(args.checkpoint)
        log_lesson(result, args.log)
        if args.autosave:
            teacher.save(args.autosave)
            print(f"  saved to {paths.short(args.autosave)}")
        print()

    pending = None
    while True:
        if pending is not None:
            line, pending = pending, None
        else:
            try:
                line = input(f"[{mode}] "
                             + ("Ask: " if mode == "teach" else "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            command, rest = command.lower(), rest.strip()

            if command in ("exit", "quit"):
                break
            elif command == "help":
                print(HELP)
            elif command == "mode":
                mode = "chat" if mode == "teach" else "teach"
                print(f"  mode: {mode}\n")
            elif command == "probe":
                drift = teacher.drift()
                print("  no corpus to probe against\n" if drift is None else
                      f"  drift from the original model: {drift:+.4f} "
                      f"(probe loss {teacher.probe():.4f})\n")
            elif command == "polish":
                # Lessons are taught one at a time, so the last one is the freshest
                # and wins ties against its neighbours - "Who are you?" starts
                # answering with whatever was taught most recently. Training every
                # lesson together for a few steps removes the recency: none of them
                # is the most recent any more.
                pairs = list(dict.fromkeys(teacher.taught))
                if not pairs:
                    print("  nothing taught yet to polish\n")
                    continue
                steps = int(rest) if rest.isdigit() else 8
                sample = random.sample(pairs, min(len(pairs), 40))
                before_checks = teacher.checks()
                teacher.snapshot()
                losses = teacher.steps_on(sample, steps, target=0)
                after = teacher.checks()
                print(f"  polished {len(sample)} wordings over {steps} steps, "
                      f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
                print(f"  sums {after}/{len(CHECK_SUMS)} (was {before_checks}"
                      f"/{len(CHECK_SUMS)}), drift {teacher.drift():+.4f}\n")
            elif command == "undo":
                if teacher.rollback():
                    # Mark it, so /lessons and the closing summary do not go on
                    # claiming a lesson the model no longer holds.
                    for lesson in reversed(teacher.lessons):
                        if not lesson.get("rolled_back"):
                            lesson["rolled_back"] = True
                            teacher.taught = teacher.taught[
                                :-len(lesson["trained_wordings"])]
                            break
                    print("  weights restored\n")
                else:
                    print("  nothing to undo\n")
            elif command == "ask":
                if rest:
                    show(teacher.ask(rest, history, temperature=args.temperature,
                                     top_k=args.top_k))
            elif command == "context":
                prompt = build_prompt(history, last_user or "...", teacher.config.block_size)
                print(f"  {len(prompt)}/{teacher.config.block_size} characters in view:")
                print(f"{DIM}{prompt}{PLAIN}\n")
            elif command == "lessons":
                if not teacher.lessons:
                    print("  nothing taught yet\n")
                for i, lesson in enumerate(teacher.lessons, 1):
                    if lesson.get("rolled_back"):
                        mark = "undone "
                    elif lesson.get("generalized"):
                        mark = "ok     "
                    else:
                        mark = "       "
                    print(f"  {i}. {mark}{lesson['question']} -> {lesson['answer']}")
                print()
            elif command == "save":
                print(f"  saved to {paths.short(teacher.save(rest or paths.CHECKPOINT))}\n")
            elif command == "export":
                export_now(teacher, args, rest)
            elif command == "forget":
                history = []
                print("  conversation cleared; the weights keep what they learned\n")
            elif command == "teach":
                if not last_user:
                    print("  say something first, then correct the reply\n")
                elif not rest:
                    print("  /teach needs the reply you wanted\n")
                else:
                    do_lesson(last_user, rest)
            elif command == "good":
                if not (last_user and last_reply):
                    print("  nothing to approve yet\n")
                else:
                    do_lesson(last_user, last_reply)
            else:
                print(f"  no such command: /{command}. /help lists them.\n")
            continue

        if mode == "teach":
            question = line
            try:
                answer = input("      Say: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not answer:
                print("  nothing taught\n")
                continue
            if answer.startswith("/"):
                # Someone typing /exit here means leave, not "learn to say /exit".
                # Half a lesson is worth abandoning to avoid teaching that.
                print(f"  lesson abandoned - {answer.split()[0]} is a command\n")
                pending = answer
                continue
            do_lesson(question, answer)
            continue

        # chat: an ordinary turn, with the option to correct it afterwards
        reply = teacher.ask(line, history, temperature=args.temperature,
                            top_k=args.top_k)
        show(reply or "(no reply)")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        history.append({"role": "user", "text": line, "timestamp": now})
        history.append({"role": "model", "text": reply, "timestamp": now})
        last_user, last_reply = line, reply.partition(THINK_MARK)[2].strip() or reply

        if args.style:
            # The user's own words as plain text. There is no target reply here, so
            # this teaches phrasing and vocabulary - not better answers.
            try:
                teacher.steps_on([(line, reply or " ")], steps=1, lr=args.lr / 4,
                                 target=0)
            except ValueError:
                pass       # too long for the context; nothing to learn from it

    if teacher.lessons:
        kept = [l for l in teacher.lessons if not l.get("rolled_back")]
        taught = len(kept)
        carried = sum(1 for l in kept if l.get("generalized"))
        undone = len(teacher.lessons) - taught
        if undone:
            print(f"{undone} lesson(s) were undone, by you or by the guard.")
        print(f"{taught} lesson(s) this session, {carried} carried over to a wording "
              f"it never trained on.")
        drift = teacher.drift()
        if drift is not None:
            print(f"Drift from the model you started with: {drift:+.4f}")
        print(f"Fixed sums: {teacher.checks()}/{len(CHECK_SUMS)} right, "
              f"against {teacher.baseline_checks}/{len(CHECK_SUMS)} at the start.")
        print(f"Log: {paths.short(args.log)}"
              + ("" if args.remember else
                 " - pass it back with --remember when you continue this model"))
        if args.export_on_exit and taught:
            export_now(teacher, args)
        else:
            print("Nothing is written to disk unless you /save or /export - the "
                  "weights are in memory.")


if __name__ == "__main__":
    main()
