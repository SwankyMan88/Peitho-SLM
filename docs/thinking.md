# Thinking before answering

Shipped in 1.3.0 as `medium_think_1.0`. The other exports have no thinking phase and
are unaffected — a model that never saw `◇` cannot emit one, which is what lets both
kinds sit in the same folder and the same page.

A bot turn normally carries the whole reply, working and all:

```
◀Tens: 40 + 30 = 70. Ones: 6 + 8 = 14. Add those up: 70 + 14 = 84.■
```

A thinking turn puts the working first, ends it with `◇` (U+25C7), and then says the
short thing a person actually wants:

```
◀Tens: 40 + 30 = 70. Ones: 6 + 8 = 14. Add those up: 70 + 14 = 84.◇That comes to 84.■
```

Nothing about the network changes. There is no extra pass, no second model, no
special handling at inference: the working is still generated one character at a time
and still conditions everything after it. The marker only tells a reader where the
working stops.

## What it actually buys

Two `medium` models, 20,000 steps, trained on corpora holding the *same*
conversations and differing only in whether the working is thought or spoken:

| | medium_think_1.0 | medium_plain_1.0 |
|---|---|---|
| arithmetic (225 fixed sums) | 88% | 86% |
| novelty — not recited | 66% | 65% |
| **verbatim copies** | **16%** | 36% |
| mean reply length | **54 chars** | 98 chars |
| format — turn ends properly | 84% | 84% |
| spelling / variety | 100% / 95% | 99% / 86% |

**Arithmetic is a wash.** 88 against 86 on 225 sums is four or five sums, which is
inside the noise of this measurement. Thinking reorganises information the old format
already carried; it does not add any.

What it does buy is **replies at half the length** and **verbatim copying at 16%
rather than 36%** — the answer being a short copy of a number in context leaves far
less room to fall back on a memorised phrasing. Read the copying difference with the
passes confound below in mind: the stripped corpus is smaller, so a fixed step count
makes more passes over it.

### Two earlier claims that were confounded

Worth recording, because both were wrong for the same reason.

*Thinking-medium scored 86% where medium_1.2 scored 65%, beating large_1.1's 83%.*
That comparison held the format constant and changed the corpus at the same time. The
new corpus is roughly 70% sums by construction against 28% arithmetic in the released
one, so **more arithmetic exposure** is what lifted 65% to the high eighties. The
plain half of this pair, with no thinking at all, scores 86% on the same corpus.

*Thinking cost novelty — 66% against 75%.* Also the corpus. On identical content it
is 66 against 65.

The lesson is the obvious one: change one thing at a time, and a paired build is
worth the extra training run.

## Why the format still makes sense

The old format asked one string to do two jobs — show the working *and* be a readable
answer. Under length pressure it often lost the answer:

```
medium_1.2:  Working it through. Take the tens off first: 50 -        (97 chars, cut off)
```

Splitting them lets the working run as long as it needs and makes the answer a short
copy of a number already in context. That does not show up as accuracy on these
measurements, but it does show up as shorter replies, less recitation, and a reader
who can fold the working away.

## The share matters, and mixing is worse than either extreme

`--think` sets the fraction of generated conversations that think.

| `--think` | marker emitted on 40 sums | behaviour |
|---|---|---|
| 0.0 | 0/40 | the old format |
| 0.30 | 32/40 | **leaks**: sometimes writes a plan with no marker, so an internal note prints as speech |
| 1.0 | 40/40 | one convention, no leaks in 24 chat prompts |

At 0.30 the model saw both conventions for the same kind of prompt and blended them.
The fix is not a smaller share but a consistent one. A demo of the format in a
handful of lines does not work either: the model learns proportions, and at 0.007% of
the corpus the marker is never emitted.

## Turning it off

```bash
py corpus/make_corpus.py --target_chars 20000000 --think 1.0                # with
py corpus/make_corpus.py --target_chars 20000000 --think 1.0 --no_thinking   # without
```

Both forms of every turn are built from **one** draw of the generator, so the two
corpora hold the same conversations, the same sums and the same wording — they differ
only in whether the working is thought or spoken. That makes the comparison
attributable to the format alone.

Which half survives depends on which half answers the question:

| | thinking on | thinking off |
|---|---|---|
| a sum | `Ones: 0 + 6 = 6. Tens: 2 + 0 = 2…◇Comes out at 1726.` | `Ones: 0 + 6 = 6. Tens: 2 + 0 = 2…` — the working ends on the number |
| a chat turn | `Question is the harbour. Answer plainly.◇Through the summer the harbour closes…` | `Through the summer the harbour closes…` — the reply |

Stripping the same side in both cases would be a mistake in both directions: keeping
the plan would bake internal notes in as speech, and keeping only the short answer
would leave a bare total with no working to copy it from — which is the state that
once scored 1 correct out of 180.

Note that the corpora are matched on *content*, not bytes: with thinking off the same
conversations occupy 16.4M characters rather than 20M, so a fixed number of steps
makes about 30 passes over the smaller file against 25 over the larger. That mildly
favours memorization on the no-thinking side, and novelty numbers should be read with
it in mind. Content-matching and byte-matching cannot both hold, and content is the
one that isolates the format.

## What reads it

Nothing needs a flag or a mode, because absence of the marker is the fallback:

* **`peitho.html`** shows the reply and marks the turn **Thought** beside the name.
  The working is hidden unless *Show the working* is ticked in the settings, which
  redraws the whole conversation rather than only the next reply. No marker, no tag
  and no block — the bubble looks exactly as it always has, which is why models
  trained before `◇` existed still display correctly.

  While a reply is still streaming there is no way to tell working from answer, so a
  model that knows the marker shows an ellipsis rather than streaming its working
  into the answer and then yanking it back out.
* **`benchmark.py`** grades only what follows the marker, and reports how often the
  model thought first.
* **`chat.py`** prints the working dimmed.
* Anything that ignores `◇` entirely prints both halves, which is harmless.

`◇` is not a stop character — generation continues past it to `■`.

## Open

* Whether the corpus mix, rather than thinking, is where the remaining gains are.
  Going from 28% arithmetic to ~70% moved sums from 65% to the high eighties, which
  is a much larger effect than the format. Nobody has looked for the point where
  that trade starts costing conversation.
* Whether thinking on *chat* turns is worth having at all. Sums benefit; a plan
  before a one-line answer about a kettle may be pure overhead. Making thinking
  conditional on the question rather than on a random share would keep quick answers
  for chat and remove the last of the ambiguity.
