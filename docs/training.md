# Training

The corpus that the released models learned from is committed, so you can train
without generating anything:

```bash
py slm/train.py --data data/training.txt --val_data data/heldout.txt     --preset large --block_size 384 --fresh --dropout 0.0 --steps 32000 --select_by train
```

To generate your own instead:

```bash
py corpus/chat/make_corpus.py --target_chars 20000000 --composed 0.95 --think 1.0
py slm/train.py --preset large --block_size 384 --fresh --dropout 0.0 --steps 32000 --select_by train
```

A generated corpus lands in `build/`, which is not in git because it changes every
time anyone experiments. `py tools/publish_corpus.py` copies it to `data/` as the
published one and records its hashes. The checkpoint goes to `build/`, and the export
to `models/` under the next free version.

## Real text alongside the generated corpus

Generated text is fresh conversation by conversation but not sentence by sentence:
measured on 1.5M characters of replies each, 92% of generated chat sentences occur
more than once, against 28-35% for human-written dialogue and 7% for Alpaca. That
recycling is what the model recites. Two outside sources dilute it:

| source | where from | converter | license |
|---|---|---|---|
| Schema-Guided Dialogue | `google-research-datasets/dstc8-schema-guided-dialogue` | `corpus/dialogue/make_dialogue.py` | CC BY-SA 4.0 |
| MultiWOZ 2.2 | `budzianowski/multiwoz` | `corpus/dialogue/make_dialogue.py` | MIT |
| Alpaca, GPT-4 answers | `Instruction-Tuning-with-GPT-4/GPT-4-LLM` | `corpus/alpaca/make_alpaca.py` | CC BY-NC 4.0, research use only |

Download each repository as a zip (Code, Download ZIP) into `build/datasets/`, then:

```bash
py corpus/dialogue/make_dialogue.py
py corpus/chat/make_corpus.py --target_chars 50000000 --composed 0.964 --think 1.0 \
    --alpaca build/datasets/gpt4/alpaca_gpt4_data.json --alpaca_share 0.075 \
    --dialogue build/dialogues.json --dialogue_share 0.495 \
    --train_out build/training_v2.txt --heldout_out build/heldout_v2.txt
```

The dialogue datasets label what every system turn does, so each thought is built
from those labels ("Ask for the destination and origin.") rather than a fixed
sentence.

### Human-written only

`corpus/human/make_human.py` builds a corpus with no generated text at all and no
non-commercial data - the hand-written conversations plus seven freely licensed
datasets (Schema-Guided, MultiWOZ, Dolly-15k, QReCC, CCPE-M, SQuAD 1.1, GSM8K; the
sources and licenses are in its docstring):

```bash
py corpus/human/make_human.py          # -> build/training_human.txt, heldout_human.txt
py slm/train.py --data build/training_human.txt --val_data build/heldout_human.txt \
    --preset xl --block_size 384 --fresh --dropout 0.0 --steps 40000 \
    --select_by val --patience 15 --seed 1 --checkpoint xl_human_full.pt --name xl_human
```

**It does not work at this size.** Two xl runs (4.1M parameters) on it, measured
against xl_1.0 on the generated corpus:

| | xl_human_1.0 | xl_human_1.1 (rebalanced) | xl_1.0 (generated) |
|---|---|---|---|
| verbatim copies | 15% | **10%** | 38% |
| new sentences | 85% | **90%** | 62% |
| bare sums | 0% | 2% | **86%** |
| held-out GSM8K word problems | 3% | 0% | 0% |
| held-out SQuAD reading | 0% | 2% | 3% |
| "What is my name?" | "It is a museum." | "The movie that I can release your name..." | "No - I do not know that yet." |

Recitation falls away, as intended - and so does everything else. The model learns
the most repetitive style in the mix, the booking dialogues, and answers
"Good morning!" with "What city should I search for?" even when they are 9% of the
corpus. Twelve passes over the hand-written conversations did not bring greetings or
honesty back, and nothing human-written teaches a bare sum. General knowledge is
spread across far more topics than 4M parameters can hold. Keep the generators for
what they teach and add real text for variety, rather than replacing them.

Only GSM8K has real working, so only its turns think; every other turn is
`◀◇reply■`, which keeps the format without inventing plans. `--select_by val` rather
than `train`: real text read twenty times over is memorised word for word, which
generated text - new every conversation - never gave the chance to be. Keep each outside source under one pass - `make_corpus.py` warns when one
repeats. The Alpaca data, original and GPT-4 alike, is licensed for non-commercial
research only, and a model trained on it inherits that.

## 10M parameters and a 768-character context

The `xxl` preset (8 layers, 320 wide, block 768, 10.1M parameters, 13.5 MB export) on
120M characters - 55% generated, 44% free human text, hand-written at 23 passes:

| | xxl_1.0 | xxl_1.1 (passage in its own turn) | xl_1.0 (4.1M) |
|---|---|---|---|
| held-out loss | 0.53 bits/char | 0.56 | 0.34 (own corpus) |
| arithmetic | 87% | 84% | 86% |
| ends its turn | 95% | 88% | 98% |
| verbatim copies | 52% | **35%** | 38% |
| held-out reading (SQuAD, unseen) | 7% | **10%** | 0% |
| word problems (unseen, hand-written) | 4% | **8%** | 0% |
| memory probes | 14% | 0% | 0% |

Doubling the context and the parameters bought reading and word problems, which were
zero at 4.1M, and a four-turn booking that actually completes. It did not buy facts
("the capital of France is 13 km long") or memory: asked for a date from a passage
given one turn earlier, both answer 1903 or 1905 for 1904. Quoting a number out of
context is the copying circuit, which matures late - and both runs selected the
checkpoint on validation loss, which stops while that circuit is half-formed. The
next run keeps `--select_by train` and adds `corpus/recall/make_recall.py`, which
generates the one shape the corpus never had: something said early, asked about
later, including a third where it was never said and the honest answer is to say so.

### What recall drills bought

`xxl_1.2` adds `corpus/recall/make_recall.py` (4% of the corpus) and goes back to
`--select_by train` for 26,000 steps:

| | xxl_1.0 | xxl_1.1 | **xxl_1.2** |
|---|---|---|---|
| memory probes (suite) | 14% | 0% | **57%** |
| arithmetic | 87% | 84% | **88%** |
| verbatim copies | 52% | **35%** | 50% |
| held-out reading | 7% | **10%** | 7% |
| word problems | 4% | **8%** | 0% |
| ends its turn | **95%** | 88% | 85% |

"You said your dog is called Pepper", "You said you work as a teacher", "The Wilson
Bridge was built in 1904" - the first exact date quoted back out of context by any of
these models. Measured a second way, with different phrasings and a different seed,
the same export scored 1 of 6, so read 57% as "sometimes now" rather than a rate.
The drills cost instruction-following: tips and poems degrade into repetition, and
word problems went to zero. Facts remain wrong at every size tried.

## 20M parameters, and what the corpus needs at that size

The `xxxl` preset (11 layers, 384 wide, block 768) is 19.85M parameters and a 26 MB
export - noticeably slow to load in a page. On a 3060 it runs at 531 ms/step at batch
64 (7.6 GB), so a day is about 149,000 steps and 7.3B characters.

`build/training_20m.txt` is 330M characters: 68% generated, 32% real text from the
free sources at ~1.4 passes, hand-written at 26. Two generators join the mix:

| generator | share | what it teaches |
|---|---|---|
| `corpus/recall/make_recall.py` | ~2% | a fact said earlier, asked later; a third are corrections ("no, it was Colin"), a third were never said at all |
| `corpus/identity/make_identity.py` | ~1% | it is called Peitho, what it is and is not, and whose name is whose |

Booking dialogue is down to 3% of the corpus from 9%: at the higher share it hijacked
every prompt, answering "Good morning!" with "What city should I search for?" and
turning a chat about a dog's name into an appointment with a phone number.

Two datasets were downloaded and then not used, and the reasons are worth keeping:
**BIG-bench** carries a canary line in every task file saying benchmark data must
never appear in a training corpus, and **natural-instructions** licenses each task's
instances under whatever the original dataset used, so it is not uniformly free.

## Pausing and resuming

Every progress line ends with an ETA, and every eval also writes
`build/<checkpoint>.resume`: weights, optimizer state, step, schedule and random
state. Pause with **Ctrl+C**, or - for a run in the background - by creating
`build/<checkpoint>.pause`; it finishes the step, saves, and exits. Continue with the
same command plus `--resume`:

```bash
py slm/train.py --preset large ... --checkpoint mixed_full.pt              # Ctrl+C
py slm/train.py --preset large ... --checkpoint mixed_full.pt --resume
```

A resumed run is the same run, not a restart: with `--seed`, pausing at step 50 and
resuming ends on exactly the losses an unpaused run does. The resume file is written
atomically, so a crash - a GPU driver reset, say - leaves the last good one in place,
and it is removed once the run has exported.

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

| on the 30M composing corpus | small_1.5 | medium_1.4 | large_1.3 |
|---|---|---|---|
| chars per parameter | 78 | 35 | 11 |
| held-out loss | 0.23 bits/char | 0.21 | **0.21** |
| generalization gap | **+0.0100** | +0.0114 | +0.0154 |
| new sentences - found nowhere in the corpus | **57%** | 52% | **57%** |
| arithmetic overall | 35% | 83% | **91%** |
| 3-digit addition | 32% | 88% | **100%** |
| 3-digit subtraction | 0% | 44% | **72%** |

Two things changed together here and both are worth separating. The corpus now says
each fact four ways rather than one, which took whole replies found nowhere in
training from 69% to 79% on `large`. And multiplication gained a method that works for
every pair of numbers, which took the overall arithmetic score from 81% to 91% - see
[releases.md](releases.md) for why teaching only special cases went wrong.

Steps went from 20,000 to 32,000, and that matters more than it looks: on the varied
corpus, `medium` scored 55% arithmetic at 20,000 steps and 75% at 32,000. More ways of
saying the same thing means more passes before the copying circuit matures.

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
py slm/benchmark.py large_2m8_1.3  # that exact export
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
