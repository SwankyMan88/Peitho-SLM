# Releases

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
