# Releases

## 1.6.0 — Teach it something while it runs

`slm/learn.py` teaches a trained model a new answer in about a second. No training
run, no GPU hours, no retraining — it loads the full-precision checkpoint, takes a
dozen gradient steps, and tells you honestly whether they worked.

```
[teach] Ask: Who made you?
        Say: SwankyMan built me from scratch! His profile is on GitHub under SwankyMan88.
  taught 7 wordings in 16 steps, loss 3.153 -> 0.021
  held back "Hey - who made you" -> "SwankyMan built me from scratch! His profile is on GitHub under SwankyMan88."
  generalized (100% match to what you taught)
  drift from the original model: +0.0013
  sums it can still do: 4/6
```

Two modes and one switch. In **teach** mode you give the question and the reply you
wanted. In **chat** mode you have an ordinary conversation, and when a reply is
wrong, `/teach <the reply you wanted>` corrects it on the spot.

### A lesson is four things at once

One example taught one way only memorises the keystrokes. Each of these was
measured against leaving it out:

| | with | without |
|---|---|---|
| the question reworded seven ways | **4/4** unseen phrasings recognised | 1/4 |
| every wording taught at three positions | **10/10** recalled mid-conversation | 4/10 |
| slices of the corpus rehearsed alongside | drift +0.05 | **+4.15** — it answers *everything* with the taught line |
| earlier lessons rehearsed too | **4/5** lessons remembered | 1/5, and the replies turn to word salad |

The third row is the one that matters most: without rehearsal, pushed hard, the
model stops being a model. The second was a surprise — positional embeddings are
learned, so the same question at the start of a chat and four exchanges in are
genuinely different inputs, and a lesson taught only at the start fires only there.

One wording is **held back from every lesson**, so "did it learn?" is answered with
a phrasing it never trained on rather than with a loss number.

### It watches what a lesson costs

Two measurements, because one is not enough. **Drift** is the loss on a fixed sample
of the corpus. **Six fixed sums** are re-checked after every lesson with greedy
sampling, so the result is deterministic.

The sums exist because drift missed real damage: three lessons once moved the probe
by +0.037 — nothing, apparently — while taking arithmetic from 4/6 to 0/6.
Arithmetic is the most fragile thing the model does, which makes it the best early
warning. A lesson costing more than one sum is undone automatically and reported.

### Carrying on tomorrow

* `--remember <lessons.jsonl>` replays an earlier session's lessons so continuing
  the next day does not cost what was taught the day before. Rehearsal works from a
  list in memory, which a reloaded checkpoint does not have — without this, fact
  eleven quietly costs facts one to ten.
* `/polish` trains every lesson together, so the newest stops winning ties. After
  ten lessons "Who are you?" had started answering with whatever came last; this
  took rephrasings from 5/6 to 6/6 at no cost.
* `/export` writes the same versioned 3-line export `train.py` writes. A taught
  model is not a special case: `peitho.html` finds `models/taught_1.0.txt` by the
  same probe it uses for the released models and offers it as "Taught 1.0",
  `standalone.py --model taught` runs it with no PyTorch at all, and
  `versions.resolve("taught")` takes the bare base name. Taught exports are
  gitignored — they hold your facts, not the project's.
* `tools/release.py` runs every suite, refuses to push if one fails, then pushes and
  tags. It adds no identity of its own and handles no tokens.

### Does teaching make it memorise?

Worth answering precisely, because the honest answer is not "no".

**The model already recites, and always has.** Roughly half of all replies open with
a string copied verbatim from `training.txt`. That is structural: every fact in the
corpus has exactly one phrasing, so a *correct* answer about a kettle is necessarily
a training string. What it does *not* do is overfit — the generalization gap is
+0.023, so it predicts unseen text nearly as well as text it trained on.

**A lesson memorises one fixed answer on purpose.** That is what teaching is. What it
does not memorise is your exact wording: six of six phrasings never trained on still
work.

**Teaching did not make recitation worse.** Same 60 prompts, same sums, taught model
against the model it came from:

| | `large_1.2` | after ten lessons |
|---|---|---|
| novelty — not recited | 7% | **17%** |
| verbatim copies | 55% | **48%** |
| variety — distinct 3-grams | 96% | 96% |
| held-out loss | 0.22 bits/char | 0.22 |
| arithmetic overall | 81% | **82%** |

It recites *less* than before, and untaught questions still vary run to run — a
kettle came back as "In practice…", "Well…" and "Simple enough…" on three
consecutive samples. Taught answers, by design, come back identically every time.

### Measured limits

Nothing here is hidden, because a tool that overstates what it did is worse than no
tool.

* **`small` is too small to teach.** 2 of 5 lessons held, and it paid for them
  elsewhere. `medium` and `large` hold all five; `large` learns in 8 steps rather
  than 12 and drifts +0.0007.
* **Ten lessons is past comfortable.** Two of ten needed a second attempt, and one
  was undone by the guard first.
* **Similar questions interfere.** "What is my dog's name?" and "What is my brother
  called?" are the same shape, and the later one can capture the earlier.
* **A question that resembles a taught one can pull the taught answer in.** After ten
  lessons, "What do you think about rain?" sometimes answers with "I can hold a
  short conversation…", where the untaught model talked about rainbows. A
  heavy-rehearsal pass fixed that and cost a taught lesson, so the trade was
  declined.
* **Do not teach arithmetic.** A worked sum fights the arithmetic already in the
  model: it rarely carries over and it leaves replies ragged. A lesson that looks
  like a sum now says so.
* **Sampling has to be colder than ordinary chat.** A lesson makes the taught reply
  a narrow path. Over four lessons, five samples each: temperature 0.2 returned the
  taught answer 20/20 times, 0.35 managed 18, 0.7 only 14. `learn.py` chats at 0.2.
* **It has no memory of you.** Say "my name is Guy" and it will still say it cannot
  know your name — the corpus taught that refusal, and it fires even with the name
  in view. Per-person memory needs corpus turns that demonstrate copying a name
  across hundreds of examples, which is a retrain, not a lesson.

### Verified

`tests/learning/test_learn.py` — 49 checks, added to CI — builds a throwaway model in
the test process, so it needs no GPU, no checkpoint and no corpus. All suites green:
49 + 74 + 25 + 20.

A ten-lesson model was taught by hand and checked four ways: 10/10 cold, 9/10 two
exchanges into a conversation, 6/6 on phrasings never trained on, 10/10 through the
8-bit export, 5/5 through the standard-library decoder, with arithmetic unchanged at
4/6 and drift +0.0000.

## 1.5.0 — Longer conversations, a fifth model, and a folder for everything

### Conversations that hold together

The generated corpus used to be one to three exchanges stapled together. It now runs
to nine on average and up to twenty-one, with a subject that persists across turns and
gets referred to by pronoun:

```
you:    Hi, got a minute?
peitho: Morning. Ask away.
you:    what do you know about springs?
peitho: A spring stores force by bending - that is the whole of it.
you:    Does it wear out?
peitho: Slowly, if you look after it.
you:    Anyway. Round 200 to the nearest ten.
peitho: 200.
you:    Got it, thanks
peitho: Take care.
```

Following up with "it" and "that" is the only thing that teaches a model to read *back*
through its context rather than answer the last line in isolation. Openings and
closings bracket the conversation; pivots mark a change of subject.

| | before | now |
|---|---|---|
| user messages per conversation | mean 1.6, range 1–3 | **mean 9.2, up to 21** |
| corpus | 20M characters | **30M** |
| facts it can talk about | 150 | **207** |
| honest "I do not know" | 7% | 12% |

### Four models retrained, and the ordering holds

| | small_1.4 | medium_1.3 | medium_think_1.3 | large_1.2 |
|---|---|---|---|---|
| held-out loss | 0.23 bits/char | 0.23 | 0.23 | **0.22** |
| generalization gap | **+0.018** | +0.026 | +0.025 | +0.023 |
| format — ends turn | 96% | **100%** | 96% | **100%** |
| spelling | **100%** | 99% | **100%** | **100%** |
| arithmetic overall | 26% | 68% | 68% | **81%** |
| 3-digit addition | 8% | 84% | 92% | **100%** |
| 3-digit subtraction | 0% | 16% | 12% | **56%** |

Held-out loss fell by a third against 1.4.0 and the gaps stayed small, so the extra
data went into generalizing. `large` gets three-digit addition right every time.

**Novelty is the one number that got worse** — 4-19%, against 66% at 1.4.0 — and it
deserves the caveat rather than a quiet omission. Measured against text the model never
saw it is 30% rather than 19%: the generators produce the same strings in the training
and held-out files, so roughly half the drop is the metric measuring corpus redundancy.
The real cause is structural. Every fact has exactly one predicate, so a *correct*
answer about a kettle is necessarily a string that appears in training. Raising it means
paraphrasing the facts three or four ways, not training differently.

### A fifth model that does one thing

`greeter_1.1` — **48,720 parameters, 65 KB** — is trained on greetings and nothing else.
It writes the line an empty page opens with, refreshed every 30 seconds.

```
Look who it is - I am not connected to anything. Your turn.
Good afternoon - I can work out small sums if you show me one. Ask me something.
Hello. Half a megabyte of text, pretending to hold a conversation.
```

It has never seen a question and cannot answer one, so it is excluded from the model
picker. Trained **on CPU in seven minutes**, which is the other half of the point: a
narrow task at this size does not need a GPU.

Sampling temperature matters more than size for it. Over 300 greetings: 98.3% contain
only real words at temperature 0.9, 99.7% at 0.7. The page uses 0.7 — at 0.9 about one
in sixty produces "I am ha few hundred thousand numbers", which is not worth 5% more
variety.

### CPU training, measured

`--device auto|cpu|cuda` on `slm/train.py` and `tools/speed_test.py`, defaulting to
CUDA when it exists. Asking for CUDA without a GPU now fails clearly instead of
falling back silently.

| preset | CPU (12 threads) | GPU (RTX 3060) | ratio |
|---|---|---|---|
| greeter | 72,314 chars/sec | 744,735 | **10x** |
| small | 42,885 | 1,209,559 | **28x** |
| medium | 22,937 | 916,991 | **40x** |

The ratio widens with size because the GPU is nowhere near saturated on the small
model. That is why the greeter trains on CPU in minutes while `medium` would take about
nine hours.

### A folder for everything

The root held eight loose Python files and two stray generated ones. It now holds two
pages, three files and eight folders:

| | |
|---|---|
| `slm/` | the engine: model, export, train, benchmark, chat, standalone, versions, paths |
| `corpus/chat/` | the conversational corpus and its generators |
| `corpus/greeter/` | greetings, for the tiny opening model |
| `tests/pipeline/`, `tests/conformance/`, `tests/pages/` | one folder per suite, kit included |

`train.py` stays in `slm/` rather than under either corpus, because it trains both.
Commands gain one directory: `py slm/train.py`, `py slm/chat.py`. `chat_history.json`
now writes to `build/` instead of the repository root.

### Also

* **The terminal chat understands thinking.** `slm/chat.py` printed a raw `◇`; it now
  prints the working dimmed and the reply plainly.
* **The Khan build asks for a greeter and shrugs it off.** Baking one in cost 67 KB on
  a page that is already 560 KB, which is a poor trade for one line of decoration. It
  tries to fetch the greeter instead and, where that is forbidden, shows the fixed
  placeholder with no message, no note and nothing in the console — a page that cannot
  have an opening line looks exactly like a page that was never going to.
* **`docs/models.md`** describes all five models: what each is for, what it was trained
  on, and what none of them can do.
* **A hidden working could still be visible.** `.thought` set `display: block`, and
  `hidden` works by a user-agent rule that any author rule beats — so a reply that
  streamed and then turned out to have no marker appeared twice, which read as the model
  repeating itself. `tests/pages/check_pages.mjs` now refuses a page that sets `display`
  on a class it hides by attribute; a property check cannot see this, because
  `element.hidden` was `true` the whole time.

### Verified

`tests/pipeline/test_slm.py` 73 checks, `tests/conformance/test_conformance.py` 25, and
`tests/pages/check_pages.mjs` 20 — all green from the new layout. Every entry point runs
from the root. The Khan build was exercised with `fetch` blocked, the way the sandbox
actually behaves: the built-in model answering, the remote models greying out with a
reason, and the greeter failing without saying so.

## 1.4.0 — It says something, and admits what it does not know

The generated conversation used to come from lyrical clause shapes over adjective and
place pools, which produced grammatical English with nothing in it:

```
What makes the weightless terrace interesting is that salt works into everything
eventually. Long before anyone noticed, an unlikely terrace waits over the rooftops.
```

Three clauses, nothing said - and the model wrote like that because that is what it
was shown. `corpus/talk.py` replaces it.

### Chat with content in it

A short, plainly true predicate for ~150 everyday things, wrapped many ways:

```
you: whats a barometer
bot: A barometer measures air pressure - that is the whole of it.

you: Tell me about a gutter.
bot: Mostly it takes rain off the roof.
```

The facts are deliberately dull - what a kettle does, what a hinge is for. Dull and
true beats interesting and invented, and a model this size has no business asserting
anything harder. Real knowledge still comes from `corpus/conversations.txt`, which is
hand-written.

### Two kinds of not knowing

```
you: What is a duck?
bot: No idea, honestly. It is not a word I have seen.

you: What is the date today?
bot: I cannot see anything beyond what you type, so I do not know.

you: What is my name?
bot: I would be making that up. Better to say I do not know.
```

Invented words cover "never taught it". The second kind covers what actually comes
up - the date, the news, your name, anything outside the conversation. Together they
are 7% of replies; at 2.5% the model did not reliably learn the behaviour at all.

### The thinking phase thinks about the question

It used to fill a template from a pool the corpus had stopped using, so it collapsed
to "They want the question. Keep it short." Now the thought is written where the turn
type is known:

```
you:      What is the date today?
thinking: They asked me what the date is. I do not have that built into me, so I
          will be honest.
bot:      No - I cannot see anything beyond what you type, so I do not know.
```

### Five more kinds of arithmetic

Percentages, squares, three-term addition, rounding and word problems, each worked
through rather than stated:

| | |
|---|---|
| `What is 25% of 160?` | 25% is a quarter, so divide by 4: 160 / 4 = 40 |
| `What is 14 squared?` | 14 * 10 = 140, 14 * 4 = 56. Add those: 196 |
| `Round 168 to the nearest ten.` | 168 sits between 160 and 170. It is 8 past 160, and half of 10 is 5, so it rounds up to 170 |
| `I had 240 and spent 85.` | First the 80: 240 - 80 = 160. Then the 5: 155 |

They are 30% of arithmetic turns, so `+ - * /` coverage barely moves: the shipped
model answers all six kinds correctly and still scores 86% on the fixed 225-sum
benchmark. Every generator was checked against Python over 30,000 problems rather
than trusted.

### The corpus is in the repository

`data/training.txt` and `data/heldout.txt` - 20M characters, 4.7 MB packed - with
`data/README.md` recording the exact command and both SHA-256 hashes. A repository
about training your own model should hand you data rather than a program that makes
data.

```bash
py train.py --data data/training.txt --val_data data/heldout.txt     --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

`build/` stays out of git, since it changes whenever anyone experiments;
`tools/publish_corpus.py` promotes a corpus to `data/` and rewrites the hashes.

### Known limits

It is a 855K-parameter character model, and it gets things wrong. In particular it
will confidently describe a **real** word it was never taught - asked about a
xylophone it borrowed the moon's predicate - because every unknown in training was an
invented word, so it learned to judge by spelling rather than familiarity. The fix is
to hold a slice of the facts out of training; it is not in this release.

### Verified

`tests/test_slm.py` 64 checks and `tests/test_conformance.py` 25 checks. Behaviour
probed directly: 0 of 20 known things wrongly refused across plain, polite and casual
wordings, all six arithmetic kinds correct, honest refusals for the date, a name and
an invented word.

Also verified from a **fresh download** rather than a working copy, which caught a
real fault: git does not track empty directories, so a clone had no `build/` and
training died saving its checkpoint - after doing the work. `train.py` now creates its
output directory, and the encoded-corpus cache moved into `build/` so that training
from `data/training.txt` no longer drops a 40 MB file into a tracked folder. In a
fresh unzip: training runs and exports, `standalone.py` answers with no PyTorch
installed, `corpus/make_corpus.py` builds a corpus, and both suites pass.

## 1.3.0 — It shows its working

A fourth model, `medium_think_1.0`, which works a problem out before answering:

```
you:      What is 148 + 267?
thinking: 267 is 3 short of 270. 148 + 270 = 418, then give back the 3: 415.
peitho:   That gives 415.
```

A new marker, `◇` (U+25C7), ends the working; what follows is the reply. Nothing about
the network changed — no extra pass, no second model, no special handling at
inference. The working is still generated one character at a time and still conditions
everything after it. The marker only tells a reader where to fold.

**Every other model is unaffected.** One trained before `◇` existed cannot emit it, so
a reply with no marker renders exactly as it always has. That is why all four sit in
the same folder and the same page with no flag anywhere.

### What it buys, measured properly

Two `medium` models trained on corpora holding the *same* conversations, differing
only in whether the working is thought or spoken:

| | medium_think_1.0 | the same corpus, no thinking |
|---|---|---|
| arithmetic (225 fixed sums) | 88% | 86% |
| mean reply length | **54 chars** | 98 chars |
| verbatim copies | **16%** | 36% |
| novelty | 66% | 65% |
| spelling / variety | 100% / 95% | 99% / 86% |

**Arithmetic is a wash** — 88 against 86 is four or five sums out of 225. Thinking
reorganises information the old format already carried; it does not add any. What it
buys is replies at half the length and less than half the verbatim copying.

An earlier reading of this was wrong and is worth recording. Thinking-medium scoring
88% where `medium_1.2` scores 65% looked like a large win for the format; it is not.
That comparison changed the corpus at the same time — the new one is roughly 70% sums
by construction against 28% arithmetic in the released one. More arithmetic exposure
is what moved the number, and a no-thinking model on the same corpus reaches 86%. The
same applies to a claim that it beat `large_1.1`. Change one thing at a time.

### The share has to be consistent

`--think` sets the fraction of generated conversations that think, and the middle is
worse than either end:

| `--think` | marker emitted on 40 sums | behaviour |
|---|---|---|
| 0.0 | 0/40 | the old format |
| 0.30 | 32/40 | **leaks** — sometimes writes a plan with no marker, so an internal note prints as speech |
| 1.0 | 40/40 | one convention, no leaks in 24 chat prompts |

At 0.30 the model saw both conventions for the same kind of prompt and blended them.

### Turning it off

```bash
py corpus/make_corpus.py --target_chars 20000000 --think 1.0                 # with
py corpus/make_corpus.py --target_chars 20000000 --think 1.0 --no_thinking   # without
```

Both forms of every turn are built from one draw of the generator, so the two corpora
hold the same sentences and differ only in format. Which half survives depends on
which half answers: for a sum the working stays, because it ends on the number; for a
chat turn the reply stays, because a plan is not an answer. Keeping the wrong side
would either bake internal notes in as speech or leave a bare total with nothing to
copy it from — the state that once scored 1 correct out of 180.

### Also

* `peitho.html` renders the working as a dim, foldable block and labels it *thinking*;
  `benchmark.py` grades only what follows the marker and reports how often the model
  thought first; `chat.py` prints the working dimmed.
* [docs/thinking.md](thinking.md) documents the format, the measurements and the
  confounds. [docs/spec.md](spec.md) records `◇` as optional and **not** a stop
  character — a decoder can tell whether an export uses it by looking for `◇` in the
  header's `vocab` string.
* Model buttons read "Medium think 1.0" rather than "Medium_think 1.0".

### Verified

`tests/test_slm.py` 44 checks and `tests/test_conformance.py` 25 checks, both green.
In the browser: the thinking model renders its working and answers `148 + 267` as 415,
and a released model in the same session renders with no thinking block at all.

## 1.2.1 — Bigger is finally better

Released together with 1.2.0, whose tag carries the same models. Three new exports,
a specification for the format, a demo you can click, and a composer that works on a
phone.

### The models improve with size now

Trained identically, on the same corpus, differing only in preset:

| | small_1.3 | medium_1.2 | large_1.1 |
|---|---|---|---|
| parameters | 382K | 855K | 2.7M |
| arithmetic on unseen sums | 25% | 65% | **83%** |
| 3-digit addition | 16% | 80% | **100%** |
| 3-digit subtraction | 0% | 28% | **76%** |
| held-out loss | 0.37 bits/char | 0.36 | 0.36 |
| generalization gap | +0.018 | +0.029 | +0.042 |
| verbatim copies | 12% | 16% | **8%** |

Before this, the same three presets scored 29%, 25% and 32% — flat, with the largest
model the *worst* on held-out text and reciting 38% of its replies word for word.

Nothing about the architecture changed. The corpus went from 2M characters to 20M,
which took large from 0.72 characters per parameter to 7.2. At 0.72, memorizing the
corpus was a cheaper way to cut loss than learning the method, so it took that route.
At 7.2 that is unaffordable: its generalization gap fell from +0.89 to +0.042, and it
now recites *less* than the small model.

Asked `148 + 267`, the difference is one carry:

```
Small 1.3:  Hundreds: 300. Tens: 40 + 60 = 90.  Ones: 15. -> 305   wrong
Large 1.1:  Hundreds: 300. Tens: 40 + 60 = 100. Ones: 15. -> 415   right
```

Small dropped 29% -> 25%, which is not a regression. Operands are now weighted
towards two and three digits, because one digit has only ~81 combinations per
operator and gives capacity nothing to buy. 382K parameters cannot hold the carry
procedure, and that is the honest shape of "bigger is better".

### The format is specified, with vectors that prove a port correct

[docs/spec.md](spec.md) describes the export completely enough to implement a decoder
without reading any Python: the three lines, every header field, nibble order and
sign extension at 4 bits, per-tensor group padding, float16 scales, the forward pass
with its exact layernorm and GELU, PyTorch's weight layout, and the seven ways a port
is usually wrong.

`conformance/` backs it with a purpose-built micro model — 2 layers, 20 dimensions,
17 KB — and the numbers it must produce: token ids, the logits after the final token,
and eight greedy steps. Three implementations are checked against them in CI:
PyTorch, the pure-Python decoder, and the JavaScript inside `peitho.html`, which is
read out of the page rather than copied, so what is tested is what ships.

The greedy vectors earn their place. A key-value cache that is correct on the first
token and wrong afterwards passes a logit comparison and fails this — which is
exactly the mistake the first conformance runner made.

### A demo you can click

`index.html` serves the bare Pages URL, so
[swankyman88.github.io/Peitho-SLM](https://swankyman88.github.io/Peitho-SLM/) opens
the chat instead of returning a 404.

### Usable on a phone

The composer put four controls in one row, which at 375px left the text box about 100
pixels wide — barely wider than its placeholder. Send stays beside the input; Clear
and Settings moved to a quiet second line, right-aligned, at a 34px minimum height so
they are still comfortable to tap.

| at 375 x 812 | before | after |
|---|---|---|
| text box width | ~100px | **220px** |
| text box font | 15px | **16px** |

The font is half the fix: iOS Safari zooms the whole page in when a focused field's
text is smaller than 16px, so typing would shove the layout sideways.

### Two bugs found while verifying

* **The picker offered a model that had been deleted, and loaded it.** A `HEAD` probe
  is a folder listing in disguise, and the browser cached both the probe and the body:
  the same request answered `200` from cache and `404` with `cache: no-store`. The
  probe now bypasses the cache.
* **`tools/make_html.py` called `paths.short()` without importing `paths`.** It only
  fails on the last line of a successful build, so it passed locally and broke CI. The
  suite now walks every source file and refuses one that uses a module it never
  imported.

### Also

* **Superseded exports removed**, so the page offers three models rather than five.
* **Training measured rather than guessed at.** Concurrent runs are *slower* than
  sequential on a saturated GPU, larger batches buy 7-10%, and `torch.compile` needs
  Triton, which has no Windows build. The corpus cache is int16 rather than float64,
  which matters at 20M characters: 40 MB instead of 160 MB.

### Verified

`tests/test_slm.py` 43 checks and `tests/test_conformance.py` 25 checks, both green,
plus the browser page loading each of the three models and answering correctly at
375x812 and 1280x800.

## 1.0.1 — Tidying up

No change to any model. The exports in `models/` are byte for byte what they were, and
every published URL keeps working: `v1.4` and earlier are git tags, so they are frozen
snapshots and nothing here can disturb them. This release is about the shape of the
repository.

**Everything has a folder now.** The root held twenty-odd files; it now holds the seven
that make up the model and the things you run.

| | |
|---|---|
| `corpus/` | `conversations.txt` and the generators — `compose.py`, `arith.py`, `make_corpus.py` |
| `tools/` | `make_html.py`, `make_js_models.py`, `speed_test.py` |
| `tests/` | `test_slm.py` |
| `docs/` | training, format and hosting guides |
| `build/` | the corpus, its caches and checkpoints — regenerable, not in git |
| `models/` | **unmoved**, so every jsDelivr URL still resolves |

**Scripts work from anywhere.** A new `paths.py` derives every default from its own
location, so `py train.py`, `py corpus/make_corpus.py` and `py ../benchmark.py` all
find the same corpus, checkpoint and models folder no matter which directory you are
standing in. Generated files land in `build/` instead of beside the source.

**The README is a front page again.** It was three hundred lines; the detail moved into
`docs/`, which is also where the comparison of the three model sizes now lives.

### Also in this release

Things that landed since the models were published:

* **Arithmetic is scored reproducibly.** The operand generator and torch's sampling
  generator were both shared with the rest of the benchmark, so `--gen_samples`
  changed the arithmetic score — the same file measured 30% and 39% on two runs. Both
  are now seeded from a constant for that section, and trials went from 15 to 25 per
  cell. The corrected comparison: small 29%, medium 25%, large 32%.
* **Models can be named the obvious way.** `py benchmark.py small_1.2` works, as do
  `small_1.2.txt`, `models/small_1.2.txt` and `small`. Naming a model implies
  benchmarking that export, so `--from_compressed --compressed_path …` is no longer
  needed. A name that matches nothing lists what is actually there.
* **Exports are mirrored as `.js`.** Khan Academy permits a script tag to another host
  while refusing fetch and XHR to one, so `tools/make_js_models.py` writes each export
  a second time as JavaScript that registers itself on `window.PEITHO_MODELS`. Pages
  that cannot fetch can now load the larger models.
* **The browser page finds its own models.** It used to read a `models/index.json`
  that had to be regenerated by hand — and did not survive being forgotten. It now
  asks the folder for every plausible export name at once. Removing the manifest
  exposed a bug it had been hiding: the previous version walked each version series
  and stopped after two gaps, so with `small_1.0` and `small_1.1` deleted it never
  reached `small_1.2` and silently fell back to a model that declines arithmetic.
* **A page opened from the disk says why it cannot work.** Browsers forbid a `file://`
  page from reading any sibling file, so the message names that rather than blaming a
  missing manifest.

### Verified

`tests/test_slm.py` — 41 checks, corpus through training, export, pure-Python
inference and page baking — passes from the new layout, as do `benchmark.py`,
`chat.py`, `standalone.py`, both `tools/` scripts, and the browser page with all three
models offered and loading.

The suite also now refuses a file that uses one of our own modules without
importing it. That mistake only fails when the offending line runs, so a print at
the end of a script can pass locally and break in CI - which is exactly how
`tools/make_html.py` shipped calling `paths.short()` with no `import paths`.
