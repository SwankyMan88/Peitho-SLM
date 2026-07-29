# Peitho

**[Talk to it in your browser](https://swankyman88.github.io/Peitho-SLM/)** — no
install, nothing sent anywhere, the whole model arrives as half a megabyte of text.

A small language model written from scratch in PyTorch — a real character-level
transformer, not a wrapper around anything. You train it on your own text, talk to it
in the terminal, and export its weights to a **single 509 KB text file** that runs with
nothing but the standard library, or in a browser page with no server and no network.

```
you: What is 24 + 15?
peitho: Sure. Start at 24, add 10 to get 34, then 5 more. That is 39.

you: What are you bad at?
peitho: Anything that happened recently, long chains of reasoning, and arithmetic
        once the numbers get big. I will work a sum out rather than refuse, but
        check it.
```

382,272 parameters, 3 layers, 96 dimensions wide, a 384-character context — roughly a
millionth the size of the models people usually mean. It holds a short conversation,
composes sentences it has never seen, and works arithmetic out step by step. It also
gets things wrong confidently, which is the part to watch.

| | |
|---|---|
| held-out loss | 0.59 bits/char |
| spelling (real words) | 98% |
| variety (distinct trigrams) | 87% |
| novelty (not recited from training) | 83% |
| arithmetic on unseen sums | 29% overall, 67% on single digits |
| export size | 509,552 bytes, lossless against the checkpoint |

## Quick start

```bash
pip install -r requirements.txt
py corpus/make_corpus.py
py train.py --preset small --block_size 384 --fresh --dropout 0.0 --steps 30000 --select_by train
```

Then talk to it, measure it, or run it with no PyTorch at all:

```bash
py chat.py                     # terminal chat, remembers the conversation
py benchmark.py small          # what it is good and bad at
py standalone.py               # the same model, standard library only
py tests/test_slm.py           # the whole pipeline, about a minute
```

For the browser, serve the folder and open `peitho.html`:

```bash
py -m http.server
```

## Layout

| | |
|---|---|
| `model.py` | The network: embeddings, causal self-attention, MLP blocks, weight-tied head. Also the turn markers. |
| `train.py` | Trains on a text file, checkpoints to `build/`, exports to `models/`. |
| `export.py` | Quantizes weights and writes/reads the 3-line text export. |
| `standalone.py` | Runs an export with **no PyTorch and no numpy** — the reference decoder, and the blueprint for a JS port. |
| `chat.py` | Terminal chat. |
| `benchmark.py` | Language, format, spelling, variety, novelty, copying, arithmetic, export cost. |
| `versions.py`, `paths.py` | Where exports are named and where everything lives. |
| `peitho.html` | The browser page. Carries no weights: it finds the exports in `models/` itself. |
| `corpus/` | `conversations.txt` (hand-written — **edit this to change what it knows**), plus the generators: `compose.py` for varied English, `arith.py` for worked sums, `make_corpus.py` to assemble them. |
| `tools/` | `make_html.py` bakes an export into a page that cannot fetch; `make_js_models.py` mirrors exports as `.js`; `speed_test.py` measures training throughput. |
| `models/` | The exports. `small_1.2` is the one to use. |
| `build/` | Everything regenerable — corpus, caches, checkpoints. Not in git. |
| `docs/` | The detail. |

## Arithmetic

Nothing is hard-coded. The page contains no arithmetic — no `eval`, no digit parsing,
no lookup table — and there is nothing to retrieve from the weights either, because
every problem in a 2M-character corpus is a different one. What the model learned is a
*method*, from examples like these:

```
Tens: 20 + 10 = 30. Ones: 4 + 5 = 9. Add those up: 30 + 9 = 39.
Ones: 9 + 3 = 12, so write 2 and carry 1. Tens: ...
62 is 8 short of 70. 14 + 70 = 84, then give back the 8: 76.
9 is one less than 10. 7 * 10 = 70, then take off one 7: 63.
6 * 10 = 60, leaving 36. 6 * 6 = 36. Together that is 10 + 6 = 16.
```

`corpus/arith.py` solves each problem in Python — so the lesson is true — and offers
several methods per problem, so no wording maps to one fixed reply and a follow-up
("show me another way") can rework the same problem differently to the same number.

Working arithmetic is a **copying task before it is a counting one**: the operands have
to be read out of the context before anything can be done to them. That circuit forms
late, so accuracy sits at zero for a long time and then appears. Expect small sums to
land and large ones to drop a carry. The visible working is what lets you catch it.

## Documentation

* [docs/training.md](docs/training.md) — how to train it, the five settings that
  silently ruin it, why the bigger presets are worse, and how to read the benchmark.
* [docs/spec.md](docs/spec.md) — **the format specification**: everything needed to
  implement a decoder in any language, and a conformance kit to prove it correct.
* [docs/format.md](docs/format.md) — the practical tour of the export, why 8 bits and
  not 4, and the conversation markers.
* [docs/hosting.md](docs/hosting.md) — the browser page, serving exports over jsDelivr,
  and getting a model into a sandbox that cannot fetch.
* [docs/releases.md](docs/releases.md) — what changed, release by release.

## License

MIT. See [LICENSE](LICENSE).
