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

Three sizes, from 382K parameters to 2.7M — roughly a millionth to a hundred
thousandth the size of the models people usually mean. They hold a short
conversation, compose sentences they have never seen, and work arithmetic out step by
step. They also get things wrong confidently, which is the part to watch.

| | small_1.3 | medium_1.2 | large_1.1 | medium_think_1.0 |
|---|---|---|---|---|
| parameters | 382K | 855K | 2.7M | 855K |
| export size | **509 KB** | 1.1 MB | 3.6 MB | 1.1 MB |
| spelling (real words) | 99% | 98% | 99% | 100% |
| arithmetic on unseen sums | 25% | 65% | 83% | **88%** |
| mean reply length | 141 | 129 | 149 | **54** |
| verbatim copies | 12% | 16% | 8% | 16% |
| works out loud first | no | no | no | **yes** |

Use **large** if you can spare the download, **small** if the weights have to be
pasted somewhere by hand — 509 KB is the size that fits in a text box — and
**medium_think** for short answers with the working shown separately.

The arithmetic column is not a like-for-like comparison of the presets: `medium_think`
was trained on a corpus with far more arithmetic in it, and a no-thinking model on that
same corpus scores 86%. See [docs/thinking.md](docs/thinking.md).

## Quick start

```bash
pip install -r requirements.txt
py corpus/make_corpus.py --target_chars 20000000 --composed 0.95 --math 0.30
py train.py --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

Then talk to it, measure it, or run it with no PyTorch at all:

```bash
py chat.py                     # terminal chat, remembers the conversation
py benchmark.py large          # what it is good and bad at
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
| `corpus/` | `conversations.txt` (hand-written — **edit this to change what it knows**), plus the generators: `compose.py` for varied English, `arith.py` for worked sums, `thinking.py` for turns that work something out before answering, `make_corpus.py` to assemble them. |
| `tools/` | `make_html.py` bakes an export into a page that cannot fetch; `make_js_models.py` mirrors exports as `.js`; `speed_test.py` measures training throughput. |
| `models/` | The exports. `large_1.1` writes best, `small_1.3` fits in a text box, `medium_think_1.0` shows its working. |
| `build/` | Everything regenerable — corpus, caches, checkpoints. Not in git. |
| `docs/` | The detail. |

## Arithmetic

Nothing is hard-coded. The page contains no arithmetic — no `eval`, no digit parsing,
no lookup table — and there is nothing to retrieve from the weights either, because
almost every problem in a 20M-character corpus is a different one. What the model learned is a
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

* [docs/training.md](docs/training.md) — how to train your own custom models from scratch
* [docs/spec.md](docs/spec.md) — **the format specification**: everything needed to
  implement a decoder in any language, and a conformance kit to prove it correct.
* [docs/format.md](docs/format.md) — the practical tour of the export, why 8 bits and
  not 4, and the conversation markers.
* [docs/hosting.md](docs/hosting.md) — the browser page, serving exports over jsDelivr,
  and getting a model into a sandbox that cannot fetch.
* [docs/thinking.md](docs/thinking.md) — the working-then-answer format, what it
  measured, and what it costs. **Experimental, on the `thinking` branch.**
* [docs/releases.md](docs/releases.md) — what changed, release by release.

## License

MIT. See [LICENSE](LICENSE).
