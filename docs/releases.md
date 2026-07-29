# Releases

## 1.2.0 — Bigger is finally better

Three new models, and the point of the release is that they now improve with size.
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

Before this release the same three presets scored 29%, 25% and 32% — flat, with the
largest model the *worst* on held-out text and reciting 38% of its replies. Nothing
about the architecture changed. The corpus went from 2M characters to 20M.

At 2M characters, large had 0.72 characters per parameter, so memorizing the corpus
was a cheaper way to cut loss than learning the method, and it took that route. At
20M it has 7.2, memorizing is no longer affordable, and the capacity goes into
carries instead: its generalization gap fell from +0.89 to +0.042 and it now recites
*less* than the small model.

Small dropped 29% -> 25%, which is not a regression: the operand mix moved towards
two and three digits, so there are fewer trivial one-digit sums to get right for
free. 382K parameters cannot hold the carry procedure, and that is the honest shape
of "bigger is better".

### The format is now specified

[docs/spec.md](spec.md) describes the export completely enough to implement a decoder
without reading any Python: the three lines, every header field, nibble order and
sign extension at 4 bits, per-tensor group padding, float16 scales, the forward pass
with its exact layernorm and GELU, PyTorch's weight layout, and the seven ways a port
is usually wrong.

`conformance/` backs it with a purpose-built micro model — 2 layers, 20 dimensions,
17 KB — and the numbers it must produce: token ids, the logits after the final token,
and eight greedy steps. Three implementations are checked against them in CI:
PyTorch, the pure-Python decoder, and the JavaScript inside `peitho.html`, which is
read out of the page rather than copied so what is tested is what ships.

The greedy vectors earn their place. A key-value cache that is correct on the first
token and wrong afterwards passes a logit comparison and fails this — which is
exactly the mistake the first conformance runner made.

### Also in this release

* **A demo link.** `index.html` serves the bare Pages URL, so
  [swankyman88.github.io/Peitho-SLM](https://swankyman88.github.io/Peitho-SLM/) opens
  the chat instead of a 404.
* **Superseded exports removed.** `models/` holds the latest of each preset and
  nothing else, so the page offers three buttons rather than five.
* **Training measured rather than guessed at.** Concurrent runs are *slower* than
  sequential on a saturated GPU, larger batches buy 7-10%, and `torch.compile` needs
  Triton, which has no Windows build. The corpus cache is int16 rather than float64,
  which matters at 20M characters: 40 MB instead of 160 MB.

### Verified

`tests/test_slm.py` 43 checks and `tests/test_conformance.py` 25 checks, both green,
plus the browser page loading each of the three models and answering correctly.

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
