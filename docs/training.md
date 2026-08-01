# Training

The corpus that the released models learned from is committed, so you can train
without generating anything:

```bash
py slm/train.py --data data/training.txt --val_data data/heldout.txt     --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

To generate your own instead:

```bash
py corpus/chat/make_corpus.py --target_chars 20000000 --composed 0.95 --think 1.0
py slm/train.py --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

A generated corpus lands in `build/`, which is not in git because it changes every
time anyone experiments. `py tools/publish_corpus.py` copies it to `data/` as the
published one and records its hashes. The checkpoint goes to `build/`, and the export
to `models/` under the next free version.

## Five settings that silently ruin it

Each of these leaves loss, spelling and formatting looking healthy while the model is
quietly broken.

**1. Validation loss is a bad stopping signal.** A representative run:

| step | train | val | arithmetic |
|---|---|---|---|
| 2000 | 0.4350 | 0.4519 | ~0% |
| 10000 | 0.2544 | 0.3621 | — |
| 20000 | 0.1928 | 0.4854 | 29% |

Val loss bottoms early and then rises while the thing you actually want keeps
improving — the copied digits are a few characters out of hundreds, so they barely
move the loss. Selecting on val loss ends training with a half-formed copying
circuit: that run scored **1 correct out of 180**. Use `--select_by train` and give
it steps.

**2. Learning rate too low silently disables copying.** At `lr 3e-4`: 95% spelling,
100% format, healthy loss, and **0% recall**. It never formed the induction circuit.
At `lr 1e-3` the circuit forms within ~1000 steps. Do not lower `--lr` much.

**3. Dropout blocks the copying circuit.** At identical validation loss, `dropout 0.0`
scored 31% recall and `dropout 0.1` scored 14%. Regularize with **more data** instead —
the corpus is generated, so turn `--target_chars` up.

**4. Small value pools teach guessing instead of reading.** With only ~20 colours in
the corpus, guessing is a cheaper way to cut loss than copying from context. Every
pool that matters needs to be large; `corpus/chat/compose.py` builds thousands of values.

**5. Repetition is what gets recited.** The hand-written text is repeated to fill the
corpus, and how often decides whether the model quotes it. At 7 passes the
generalization gap was +0.31; at 3 passes it was +0.11. `--composed` is the dial, and
`make_corpus.py` prints the pass count and warns past 60.

## Presets

| preset | params | 8-bit export | corpus for 5+ chars/param |
|---|---|---|---|
| tiny | 110K | ~145 KB | 0.6M |
| small | 382K | ~509 KB | 2M |
| medium | 855K | ~1.1 MB | 4.3M |
| large | 2.7M | ~3.6 MB | 14M |

The last column is the one that decides whether a preset is worth using. Train a
preset on less than that and it will memorize instead of generalizing.

### Size only helps if the corpus grows with it

The same three presets, trained identically, on two different corpus sizes. On 2M
characters, bigger was *worse*:

| on 2M characters | small | medium | large |
|---|---|---|---|
| chars per parameter | 5.2 | 2.3 | **0.72** |
| held-out loss | **0.59 bits/char** | 1.12 | 1.35 |
| generalization gap | **+0.10** | +0.69 | +0.89 |
| verbatim copies | 12% | 8% | **38%**, 72 chars at a stretch |
| arithmetic overall | 29% | 25% | 32% |

At 0.72 characters per parameter, memorizing the corpus is a cheaper way for large
to cut loss than learning anything, so it did. On 20M characters the same three
runs invert completely:

| on 30M characters | small_1.4 | medium_1.3 | large_1.2 |
|---|---|---|---|
| chars per parameter | 78 | 35 | 11 |
| held-out loss | 0.23 bits/char | 0.23 | **0.22** |
| generalization gap | **+0.018** | +0.026 | +0.023 |
| arithmetic overall | 26% | 68% | **81%** |
| 3-digit addition | 8% | 84% | **100%** |
| 3-digit subtraction | 0% | 16% | **56%** |
| novelty - not recited | 7% | **19%** | 4% |

Held-out loss fell by a third against the 20M corpus and the gap stayed small, so the
extra data went into generalizing rather than memorizing. Novelty is the one number
that got worse, and it needs care: measured against text the model never saw it is
30% rather than 19%, because the generators produce the same strings in both files.
Every fact has exactly one predicate, so a correct answer is necessarily a string that
appears in training. Raising it means paraphrasing the facts, not training
differently.

Arithmetic goes 25 -> 65 -> 83, and large now recites *less* than small. Held-out
loss improved by a third across the board, because none of the three can afford to
memorize 20M characters.

Small got slightly worse at arithmetic (29% -> 25%) and that is not a regression in
training: the operand mix moved towards two and three digits, so there are fewer
trivial one-digit sums to get right for free. 382K parameters cannot hold the carry
procedure. That is what "bigger is better" actually means here - the capacity buys
carries, and only if there is enough unique text that memorizing is not the cheaper
option.

## Reading the benchmark

```bash
py slm/benchmark.py large_1.1      # that exact export
py slm/benchmark.py large          # the highest large_*
py slm/benchmark.py                # the full-precision checkpoint
```

Watch `generalization gap` (large = memorizing) and `verbatim copies` (is it reciting
the training text?). Short replies like "You are welcome." legitimately appear in
training, so a nonzero copy rate is expected.

Arithmetic is scored on 25 sums per operand size and operator — 225 generations — so
treat a few points as noise. The sums and the sampling are both seeded from a
constant, so the score is comparable between models and between runs, but it is
still a sample.

## Teaching it new things

Edit `corpus/chat/conversations.txt` — strict `▶…■` / `◀…■`, one turn per line, blank line
between conversations — then rebuild and retrain.

The dials that matter are `--composed` (share generated fresh rather than repeated),
`--math` (share of that which is worked arithmetic) and `--think` (share that works
something out before answering — see [thinking.md](thinking.md)). Hand-written text is
what the model *knows*; generated text is what teaches it to *compose*. They compete
for corpus share, and raising one costs the others.

`--block_size` sets how much conversation fits in context; keep it larger than your
longest exchange. Training resumes from `build/model_full.pt` by default — pass
`--fresh` after changing the architecture or the character vocabulary.
