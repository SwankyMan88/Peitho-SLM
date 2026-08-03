# The models

Everything in `models/` is a finished export: three lines of text holding a trained
network, described by [spec.md](spec.md). This page says what each one is, what it was
trained on, and where it falls over.

## Choosing one

| | for | size | when not to |
|---|---|---|---|
| `small_1.5` | pasting the weights somewhere by hand | **509 KB** | when you can spare a bigger download |
| `medium_1.4` | a reasonable default | 1.1 MB | when 509 KB is the budget |
| `large_1.3` | the best writing and arithmetic | 3.6 MB | on a slow connection, or in a text box |
| `medium_think_1.3` | kept from 1.5.0; every model thinks now, so this is no longer a separate kind | 1.1 MB | anything - prefer `medium_1.4` |
| `greeter_1.1` | one job: opening a conversation | **65 KB** | anything else - it cannot answer |

Measured on the 30M-character composing corpus, same 225 sums for each:

| | small_1.5 | medium_1.4 | large_1.3 |
|---|---|---|---|
| held-out loss | 0.23 bits/char | 0.21 | **0.21** |
| generalization gap | +0.0100 | +0.0114 | +0.0154 |
| format - ends turn | 98% | 98% | **100%** |
| spelling | 99% | **100%** | **100%** |
| new sentences - found nowhere in the corpus | **57%** | 52% | **57%** |
| arithmetic overall | 35% | 83% | **91%** |
| 3-digit addition | 32% | 88% | **100%** |
| 3-digit subtraction | 0% | 44% | **72%** |
| 2-digit multiplication | 68% | 88% | **92%** |
| 3-digit multiplication | 12% | 56% | **72%** |

`peitho.html` picks the smallest `small_*` it finds, because that is the quickest to
fetch and unpack. Change `START_WITH` near the top of its script to open with another.

## What they share

All of them are the same architecture, differing only in size: a decoder-only
character-level transformer, pre-norm, GELU, learned positional embeddings, output
head tied to the token embedding. 8-bit symmetric quantization with one float16 scale
per 32 weights, which costs about 0.0001 nats against the full-precision checkpoint —
effectively lossless. See [format.md](format.md) for why not 4-bit.

Every one reads and writes the same turn markers, and none of them has a tokenizer:
the alphabet is roughly 80 characters and one character is one token. That is why
arithmetic is possible at all — the digits are visible to it — and why the context
window, measured in characters, is short.

## small

**382,272 parameters. 3 layers, 96 wide, 384-character context. 509,552 bytes.**

The one that fits in a text box. 509 KB of text with no quotes or backslashes in it,
so it pastes into a JavaScript string literal — which is how it ends up in sandboxes
that cannot fetch anything, Khan Academy among them.

It holds a short conversation and works small sums out. It cannot hold the carrying
procedure for larger arithmetic: 382K parameters is not enough, and that is the
clearest single demonstration in this repository that capacity buys something specific
rather than general polish.

## medium

**855,424 parameters. 4 layers, 128 wide, 384-character context. 1,138,493 bytes.**

Twice the parameters for twice the download. Noticeably better at multi-digit
arithmetic than `small` and about the same at conversation.

## medium_think

**855,424 parameters — the same shape as `medium`, trained differently.**

Puts its working first, ends it with `◇`, then says the short thing a person wants:

```
◀Hundreds: 100 + 200 = 300. Tens: 40 + 60 = 100. Ones: 8 + 7 = 15.◇That gives 415.■
```

Measured against a model trained on corpora holding the *same conversations* with the
working spoken instead of thought: arithmetic is a wash (88% against 86%), but replies
are **54 characters instead of 98** and verbatim copying is **16% instead of 36%**.
Thinking reorganises what the old format already carried; it does not add information.
[thinking.md](thinking.md) has the numbers and the two claims about it I got wrong.

`peitho.html` hides the working behind a *Thought* tag and shows it only if you ask.

## large

**2,762,688 parameters. 6 layers, 192 wide, 384-character context. 3,672,574 bytes.**

The best writing and the best arithmetic, at seven times the size of `small`. Worth
knowing that this was **not** true at first: trained on a 2M-character corpus it was
the *worst* of the three on held-out text and recited 38% of its replies verbatim,
because 0.72 characters per parameter makes memorising cheaper than learning. The
corpus is now 30M characters and the ordering is the right way up. See
[training.md](training.md).

## greeter

**48,720 parameters. 2 layers, 40 wide, 128-character context. 66,561 bytes.**

A deliberately absurd little model that does one thing: open a conversation. It is
what `peitho.html` shows on an empty page, refreshed every 30 seconds.

It is trained on `corpus/greeter/greetings.py` output and nothing else — 3M characters of
greetings, 24,000 distinct in 40,000 draws — so it has never seen a question and
cannot answer one. It is excluded from the model picker for that reason.

```
Look who it is - I am not connected to anything. Your turn.
Good afternoon - I can work out small sums if you show me one. Ask me something.
Hello. Half a megabyte of text, pretending to hold a conversation.
```

Trained on CPU in seven minutes - 12,000 steps at batch 32 - because it is small
enough not to need a GPU, which is half the point of it.

Sampling temperature matters more than size here. Measured over 300 greetings:

| temperature | only real words | distinct of 300 |
|---|---|---|
| 0.9 | 98.3% | 289 |
| 0.8 | 99.0% | 281 |
| **0.7** | **99.7%** | **274** |
| 0.6 | 100% | 267 |

The page uses 0.7. At 0.9 roughly one greeting in sixty contains a mangled word - "I
am ha few hundred thousand numbers" - which is not worth 5% more variety. About 45% of
what it writes does not appear verbatim in its corpus, so it composes rather than
recites.

An earlier attempt at 3,000 steps managed 60% clean at 0.9 and produced "listeening"
and "snomething": four passes over the corpus was not enough, and the fix was steps
rather than parameters.

Two reasons it exists. A fixed list of greetings is obvious by the fourth visit,
whereas a model composes and occasionally produces something slightly odd. And it is a
useful demonstration of the floor: 47,000 parameters is enough to learn one narrow
shape well, which is the same lesson as `large`, seen from the other end.

## Version numbers

`<base>_<major>.<minor>.txt`, and a new export always takes the next free version, so
nothing is ever overwritten. Higher is newer, not necessarily better — check the
release notes. Every published version stays reachable through its git tag, which is
why the Khan Academy page can pin one and keep working while `main` moves on.

Each export also exists as a `.js` file with the same name. It holds the same weights,
wrapped so a script tag can deliver them:

```html
<script src="https://cdn.jsdelivr.net/gh/SwankyMan88/Peitho-SLM@v1.5.0/models/greeter_1.0.js"></script>
<script>var m = window.PEITHO_MODELS["greeter_1.0"];</script>
```

That form exists because some sandboxes forbid `fetch` to another host while allowing
a script tag — the two are separate permissions. `py tools/make_js_models.py`
regenerates them.

## What none of them can do

* Know anything that happened after their corpus was written, or anything about you.
  They will say so when asked directly, which is trained behaviour rather than a rule.
* Describe a real word they were never taught. Asked about a xylophone, `medium_think`
  borrowed the moon's predicate. The refusal cue was learned from invented words, so it
  keys on spelling rather than familiarity — a known gap, not yet fixed.
* Reason in several steps, or hold arithmetic with more than three digits reliably.
* Remember anything between messages beyond what is still in the visible context.
