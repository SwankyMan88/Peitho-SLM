# Peitho

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

## What it is and is not

382,272 parameters, 3 layers, 96 dimensions wide, a 384-character context. That is
roughly a millionth the size of the models people usually mean. It holds a short
conversation, composes sentences it has never seen, and works arithmetic out step by
step — and it gets things wrong confidently, which is the part to watch.

Measured on the shipped export (`py benchmark.py --from_compressed`):

| | |
|---|---|
| held-out loss | 0.67 bits/char |
| spelling (real words) | 99% |
| variety (distinct trigrams) | 88% |
| novelty (not recited from training) | 78% |
| arithmetic, 1-digit | 67% `+`, 67% `−`, 47% `×` |
| arithmetic, 2-digit | 33% `+`, 7% `−`, 7% `×` |
| export size | 509,552 bytes, lossless vs the checkpoint |

## Quick start

```bash
pip install -r requirements.txt
```

```bash
py make_corpus.py
```

```bash
py train.py --preset small --block_size 384 --fresh --dropout 0.0 --steps 30000 --select_by train
```

```bash
py make_html.py
```

That last step writes `peitho_model.html` — one self-contained file with the model
hard-coded. Double-click it.

```bash
py chat.py           # talk to it in the terminal
py benchmark.py      # see what it is good and bad at
py standalone.py     # the same model with no PyTorch and no numpy
py test_slm.py       # corpus -> training -> export -> inference, about a minute
```

## Files

| File | What it is |
|---|---|
| `model.py` | The network: token/positional embeddings, causal self-attention, MLP blocks, weight-tied output head. Also the turn markers. |
| `train.py` | Trains on a text file, saves `model_full.pt`, exports to `models/`. |
| `export.py` | Quantizes weights and writes/reads the 3-line text export. |
| `standalone.py` | Runs an export with **no PyTorch and no numpy**. The reference decoder and the blueprint for a JS port. |
| `chat.py` | Terminal chat. Remembers the conversation in `chat_history.json`. |
| `benchmark.py` | Language, format, spelling, variety, novelty, copying, arithmetic, export cost. |
| `make_corpus.py` | Builds `training.txt` + `heldout.txt`. **The mix of sources is decided here.** |
| `conversations.txt` | Hand-written multi-turn conversations. **Edit this to change what it knows.** |
| `compose.py` | Generates varied English from word pools and clause shapes — most of the corpus. |
| `arith.py` | Generates worked arithmetic. Every sum is solved correctly here. |
| `versions.py` | Names and finds exports in `models/`. |
| `peitho.html` | The browser page. `MODEL` near the top is where an export goes. |
| `make_html.py` | Bakes an export into `peitho.html` → `peitho_model.html`. |
| `test_slm.py` | End-to-end check of the whole pipeline. |
| `speed_test.py` | Training throughput, so speed changes can be measured instead of guessed. |
| `models/` | Exported models, `<base>_<version>.txt`. `small_1.2` is the current one and the one baked into the page; `medium_1.0` predates the arithmetic work and is over the browser size budget. |
| `seed_pairs.txt` | Legacy single-turn pairs, unused by default. See `--pairs`. |

## The conversation format

Every turn is a marker, the text, then the end marker — one turn per line:

```
▶Hello AI!■
◀Hey there, how are you today?■
```

| Marker | Char | Meaning |
|---|---|---|
| `START_MARK` | `◈` U+25C8 | start of a conversation |
| `USER_MARK` | `▶` U+25B6 | a line you said |
| `BOT_MARK` | `◀` U+25C0 | a line the model said |
| `END_MARK` | `■` U+25A0 | end of turn — **stop generating** |

These are deliberately not on any keyboard, so they can never collide with typed text.
The end marker is why replies stop when they are finished rather than at a character
limit: `generate()` takes a `stop_id` and breaks as soon as it samples it.

If you write your own turns, never put a literal `■` inside the text of a turn — the
model would learn to stop mid-sentence. `test_slm.py` checks this.

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

`arith.py` solves each problem in Python — so the lesson is true — and offers several
methods per problem, so no wording maps to one fixed reply and a follow-up ("show me
another way") can rework the same problem differently and reach the same number.

Working arithmetic is a **copying task before it is a counting one**: the operands
have to be read out of the context before anything can be done to them. That circuit
forms late (see below), so accuracy sits at zero for a long time and then appears.
Expect small sums to land and large ones to drop a carry. The visible working is what
lets you catch it.

`--math` sets arithmetic's share of the generated stream. Higher is more accurate and
less conversational: at `0.4` the model gets 29% of unseen sums right but answers "are
you good at math?" with word salad. Around `0.25` is the balance point.

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
move the loss. Selecting on val loss ends training with a half-formed copying circuit:
that run scored **1 correct out of 180**. Use `--select_by train` and give it steps.

**2. Learning rate too low silently disables copying.** At `lr 3e-4`: 95% spelling,
100% format, healthy loss, and **0% recall**. It never formed the induction circuit.
At `lr 1e-3` the circuit forms within ~1000 steps. Do not lower `--lr` much.

**3. Dropout blocks the copying circuit.** At identical validation loss, `dropout 0.0`
scored 31% recall and `dropout 0.1` scored 14%. Regularize with **more data** instead —
the corpus is generated, so turn `--target_chars` up.

**4. Small value pools teach guessing instead of reading.** With only ~20 colours in
the corpus, guessing is a cheaper way to cut loss than copying from context. Every
pool that matters needs to be large; `compose.py` builds thousands of values.

**5. Repetition is what gets recited.** The hand-written text is repeated to fill the
corpus, and how often decides whether the model quotes it. At 7 passes the
generalization gap was +0.31; at 3 passes it was +0.11. `--composed` is the dial, and
`make_corpus.py` prints the pass count and warns past 60.

## The export format

`models/<name>_<version>.txt` is exactly 3 lines:

```
line 1  JSON header: config, vocab string, group size, tensor names + shapes
line 2  base85 of the packed quantized weights
line 3  base85 of the float16 scale factors, one per group of weights
```

No zlib, no pickle, so a port needs only base85, some bit math and a matmul. base85
rather than base64 for 25% overhead instead of 33%, and its alphabet contains no
quotes or backslashes, so a line pastes straight into a JS string literal.

### Use 8 bits, not 4

`--bits 8` is effectively lossless (+0.0001 nats). `--bits 4` halves the file but
**destroys memory** — recall drops from ~72% to ~6%, spelling from 97% to 64% —
because the copying circuits need more precision than 4 bits gives them. Scales are
per group of 32 weights for the same reason: per-tensor 4-bit cost 0.40 nats.

### Exports are versioned

Training writes to the next free version, so nothing is ever overwritten:

```
(empty)                  -> small_1.0
small_1.0                -> small_1.1
small_1.0, small_1.1     -> small_1.2
```

The base name defaults to the preset, so `--preset small` writes `small_*` and
`--name peitho` writes `peitho_*`. Anything that loads an export (`make_html.py`,
`chat.py --compressed`, `benchmark.py --from_compressed`, `standalone.py`) takes a
path, a base name for its highest version, or nothing for the most recent:

```bash
py make_html.py --model small       # highest small_*
py make_html.py                     # most recent of any name
```

| preset | params | 8-bit export |
|---|---|---|
| tiny | 110K | ~145 KB |
| small | 382K | ~509 KB |
| medium | 820K | ~1.1 MB |
| large | 2.7M | ~3.6 MB |

## The browser page

`make_html.py` writes `peitho_model.html`: one file, no server, no network, no
dependencies. Generation runs at roughly 500–2000 characters/sec on a laptop.

The model must be embedded — there is no loader UI. To hard-code one by hand, edit
`MODEL` near the top of the script in `peitho.html`:

* **line 1** (the JSON header) goes after `header:` with **no quotes at all**. JSON is
  already a valid JS object literal, and the header contains an apostrophe.
* **lines 2 and 3** go between the **single quotes**. Never backticks: the base85
  alphabet contains `` ` `` and `$`, so a template literal breaks. It contains no
  `'`, `"` or `\`, so single quotes are always safe.

Three things to know if you adapt the page:

* **Do not name a global `history`.** `window.history` is a getter-only accessor, so
  `var history = []` throws under `"use strict"` and silently kills the rest of the
  script — while hoisted functions still look defined, so the page appears loaded.
* **Avoid `requestAnimationFrame` for the generation loop.** It is throttled to a
  crawl whenever the tab is not visible. `setTimeout` keeps generating.
* **The scrollbar is drawn by hand** (`.rail` / `.thumb`), because a native
  scrollbar's width cannot be animated. It idles as a 2px line and grows to 8px while
  scrolling. `layout()` measures the height of the turns rather than of the log,
  because the log grows into space the wordmark gives up — judging by the container
  makes hiding the wordmark change the thing that decided to hide it.

## Hosting it

The export is a plain text file, so a GitHub repo is a CDN. Tag a release and pin to
it — branch URLs are cached for 12 hours, tagged URLs are permanent:

```bash
git tag v1.2 && git push origin v1.2
```

```
https://cdn.jsdelivr.net/gh/<user>/<repo>@v1.2/models/small_1.2.txt
```

jsDelivr answers with `access-control-allow-origin: *`, so:

```javascript
const [header, weights, scales] = (await fetch(URL).then(r => r.text())).split("\n");
const MODEL = { header: JSON.parse(header), weights, scales };
```

Sandboxes without networking (Khan Academy's, for one) have no `fetch`, so there the
model still has to be pasted in as a literal.

## Porting to JavaScript

`standalone.py` is the reference implementation, written to be transliterated. It runs
an export with only the standard library and is validated against PyTorch to ~1e-6
relative error. In order:

1. read line 1 as JSON — `JSON.parse`
2. base85-decode lines 2 and 3 — see `b85_decode`, about 20 lines
3. unpack weights — 8-bit is one signed byte each; 4-bit is two nibbles per byte
4. decode float16 scales — see `fp16`, pure bit math
5. `weight = value * scale_of_its_group`, groups being `group_size` consecutive weights
6. forward pass — layernorm, matmul, gelu, softmax, attention

Use a key-value cache, as `standalone.py` does, or each new character reprocesses the
whole context and generation crawls.

## Teaching it new things

Edit `conversations.txt` — strict `▶…■` / `◀…■`, one turn per line, blank line between
conversations — then rebuild and retrain:

```bash
py make_corpus.py && py train.py --preset small --block_size 384 --fresh --dropout 0.0 --steps 30000 --select_by train
```

`make_corpus.py` prints the mix and warns when it is lopsided. The two dials that
matter are `--composed` (share generated fresh rather than repeated) and `--math`
(share of that which is arithmetic). Hand-written text is what the model *knows*;
generated text is what teaches it to *compose*. They compete for corpus share, and
raising either costs the other.

`--block_size` sets how much conversation fits in context; keep it larger than your
longest exchange. Training resumes from `model_full.pt` by default — pass `--fresh`
after changing the architecture or the character vocabulary.

## License

MIT. See [LICENSE](LICENSE).
