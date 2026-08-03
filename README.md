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

| | small_1.5 | medium_1.4 | large_1.3 |
|---|---|---|---|
| parameters | 382K | 855K | 2.7M |
| export size | **509 KB** | 1.1 MB | 3.6 MB |
| held-out loss | 0.23 bits/char | 0.21 | **0.21** |
| spelling (real words) | 99% | **100%** | **100%** |
| new sentences — replies found nowhere in the corpus | **57%** | 52% | **57%** |
| arithmetic on unseen sums | 35% | 83% | **91%** |
| 3-digit addition | 32% | 88% | **100%** |
| 2-digit multiplication | 68% | 88% | **92%** |
| works out loud first | yes | yes | yes |

Use **large** if you can spare the download and **small** if the weights have to be
pasted somewhere by hand — 509 KB is the size that fits in a text box. **greeter** is
65 KB and does nothing but open a conversation.

"New sentences" is the honest measure of whether it composes: replies that appear
nowhere in the 30 million characters it trained on. What *is* memorized is the
substance of a fact — a correct answer about a compass has to say it points north.
The sentence around it is the model's own.

Every model in detail, including what none of them can do:
[docs/models.md](docs/models.md).

## Quick start

The corpus is in the repository, so training needs nothing generated first:

```bash
pip install -r requirements.txt
py slm/train.py --data data/training.txt --val_data data/heldout.txt --preset large --block_size 384 --fresh --dropout 0.0 --steps 32000 --select_by train
```

To build a corpus of your own instead — different size, different mix, your own
hand-written conversations in `corpus/chat/conversations.txt`:

```bash
py corpus/chat/make_corpus.py --target_chars 30000000 --composed 0.97 --think 1.0
```

Then talk to it, measure it, or run it with no PyTorch at all:

```bash
py slm/chat.py                     # terminal chat, remembers the conversation
py slm/learn.py                    # teach it something new, live, in about a second
py slm/benchmark.py large          # what it is good and bad at
py slm/standalone.py               # the same model, standard library only
py tests/pipeline/test_slm.py           # the whole pipeline, about a minute
```

`learn.py` is worth a look even if you only want to watch: you type a question and
the answer you wanted, and it takes a dozen gradient steps on the spot. It rewords
your question several ways so the answer attaches to the meaning rather than the
spelling, holds one wording back to test with, and tells you whether the model
answered a phrasing it never trained on. See [docs/learning.md](docs/learning.md).

For the browser, serve the folder and open `peitho.html`:

```bash
py -m http.server
```

## Layout

| | |
|---|---|
| `slm/model.py` | The network: embeddings, causal self-attention, MLP blocks, weight-tied head. Also the turn markers. |
| `slm/train.py` | Trains on a text file, checkpoints to `build/`, exports to `models/`. |
| `slm/export.py` | Quantizes weights and writes/reads the 3-line text export. |
| `slm/standalone.py` | Runs an export with **no PyTorch and no numpy** — the reference decoder, and the blueprint for a JS port. |
| `slm/chat.py` | Terminal chat. |
| `slm/learn.py` | Teaches a trained model new answers live, with rehearsal so it does not forget, and an undo when a lesson costs too much. |
| `slm/benchmark.py` | Language, format, spelling, variety, novelty, copying, arithmetic, export cost. |
| `slm/versions.py`, `slm/paths.py` | Where exports are named and where everything lives. |
| `peitho.html` | The browser page. Carries no weights: it finds the exports in `models/` itself. |
| `models/` | The exports, described one by one in [docs/models.md](docs/models.md). |
| `data/` | **The corpus the released models were trained on**, 30M characters, plus its recipe and hashes. Train from this directly. |
| `corpus/` | `conversations.txt` (hand-written — **edit this to change what it knows**), plus the generators: `talk.py` for conversation, `arith.py` for worked sums, `thinking.py` for turns that work something out first, `make_corpus.py` to assemble them. |
| `tools/` | `make_html.py` bakes an export into a page that cannot fetch; `make_js_models.py` mirrors exports as `.js`; `speed_test.py` measures training throughput. |
| `build/` | Everything regenerable — corpus, caches, checkpoints. Not in git. |
| `docs/` | The detail. |

## Arithmetic

Nothing is hard-coded. The page contains no arithmetic — no `eval`, no digit parsing,
no lookup table — and there is nothing to retrieve from the weights either, because
almost every problem in a 30M-character corpus is a different one. What the model learned is a
*method*, from examples like these:

```
Tens: 20 + 10 = 30. Ones: 4 + 5 = 9. Add those up: 30 + 9 = 39.
Ones: 9 + 3 = 12, so write 2 and carry 1. Tens: ...
62 is 8 short of 70. 14 + 70 = 84, then give back the 8: 76.
Split by place value: 200 * 7 = 1400, 60 * 7 = 420, 6 * 7 = 42. Add those: 1400 + 420 + 42 = 1862.
7 is 3 less than 10. 550 * 10 = 5500, then take off 3 lots of 550 (1650): 5500 - 1650 = 3850.
5 is half of 10. 807 * 10 = 8070, and half of that is 4035.
```

`corpus/chat/arith.py` solves each problem in Python — so the lesson is true — and offers
several methods per problem, so no wording maps to one fixed reply and a follow-up
("show me another way") can rework the same problem differently to the same number.

Every method has to work for **every** pair of numbers, and this is easy to get
wrong. Multiplication once had only special cases - "9 is one less than 10", "4 is
doubling twice" - so the model saw those phrasings hundreds of times and never saw
their conditions fail. It learned them as *the* way to multiply and applied them
anywhere: `550 * 7` came back as 4950, which is 550 * 9, with the working confidently
explaining that 7 is one less than 10. Adding one method that always applies -
splitting by place value, the way addition does - took three-digit multiplication from
48% to 72% and the overall score from 84% to 91%.

Working arithmetic is a **copying task before it is a counting one**: the operands have
to be read out of the context before anything can be done to them. That circuit forms
late, so accuracy sits at zero for a long time and then appears. Expect small sums to
land and large ones to drop a carry. The visible working is what lets you catch it.

## Documentation

* [docs/training.md](docs/training.md) — how to train your own custom models from scratch
* [docs/models.md](docs/models.md) — **every model in detail**: what each one is for,
  what it was trained on, and what none of them can do.
* [docs/spec.md](docs/spec.md) — **the format specification**: everything needed to
  implement a decoder in any language, and a conformance kit to prove it correct.
* [docs/format.md](docs/format.md) — the practical tour of the export, why 8 bits and
  not 4, and the conversation markers.
* [docs/hosting.md](docs/hosting.md) — the browser page, serving exports over jsDelivr,
  and getting a model into a sandbox that cannot fetch.
* [docs/learning.md](docs/learning.md) — **teaching a model while it runs**: how a
  lesson works, what keeps it from forgetting, and the measured limits
* [docs/thinking.md](docs/thinking.md) — the working-then-answer format, what it
  measured, and what it costs. **Experimental, on the `thinking` branch.**
* [docs/releases.md](docs/releases.md) — what changed, release by release.

## License

MIT. See [LICENSE](LICENSE).
