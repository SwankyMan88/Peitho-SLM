# Teaching a model while it runs

`slm/learn.py` teaches a trained model something new, one example at a time, in
about a second per lesson. No training run, no retraining — it loads the
full-precision checkpoint, takes a handful of gradient steps, and tells you
honestly whether they worked.

```bash
py slm/learn.py --checkpoint build/medium_v15.pt
```

```
[teach] Ask: What is my dog's name?
        Say: Your dog is called Rufus.
  taught 7 wordings in 12 steps, loss 5.926 -> 0.009
  held back "remind me what my dog's name is" -> "Your dog is called Rufus."
  generalized (100% match to what you taught)
  drift from the original model: +0.0025
  sums it can still do: 4/6
```

## Two modes, one switch

**teach** — you give the question and the answer you wanted. `/mode` switches.

**chat** — an ordinary conversation with the last few turns in view. When a reply
is wrong, `/teach <the reply you wanted>` corrects it; when it is right, `/good`
promotes that exchange into a lesson.

Chat mode is the weaker of the two and it is worth knowing why. Without a target
answer there is no signal about what a *good* reply is. `--style` will train on
your own messages as text — it picks up your vocabulary, not better answers — and
training a model on its own output is the one thing this script will not do, because
that degrades it. The useful version of "it learns from chatting" is `/teach`.

## What a lesson actually does

Four things, and each one was measured against leaving it out.

**Rewords the question.** One example teaches a string; the same answer against
seven phrasings teaches a mapping. Tested on four phrasings never trained on:

| | recognized |
|---|---|
| with rewording | **4/4** |
| without (`--wordings 1`) | 1/4 |

**Holds one wording back.** The held-out phrasing is the only honest test of
whether it generalized, and it is reported after every lesson. Turn it off with
`--no_holdout` and you lose the test, not the learning.

**Masks the loss to the answer.** Everything up to the bot marker is context. It
learns to reply, not to write questions the way you type them.

**Rehearses the corpus.** Every lesson trains on 64 random slices of
`training.txt` in the same step. This is the one that keeps the model a model:

| five lessons, lr 1e-3 ×30 | drift | sums kept |
|---|---|---|
| with rehearsal | +0.053 | 0/6 |
| without | **+4.15** | 0/6 |

Without it, pushed hard, the model collapses into answering *everything* with the
last thing it was taught. `--replay_batch 0` disables it, and shouldn't be used for
anything but reproducing that failure.

**Teaches each wording at three positions.** Every exchange goes into the batch
three times: opening a conversation (with the start marker), as a bare turn, and
behind one or two earlier exchanges. Positional embeddings are learned, so the same
question at offset 0 and at offset 90 are genuinely different inputs — and a lesson
taught only at offset 0 fires only at the start of a chat. Teaching ten facts and
then asking them two exchanges deep:

| | recalled mid-conversation |
|---|---|
| opening form only | 4/10 |
| all three positions | **10/10** |

This also fixes `standalone.py`, which builds its prompt without a start marker and
so was answering taught questions from the untaught model.

**Rehearses earlier lessons.** Each lesson also trains on up to 8 pairs from
earlier ones. Without it, lesson three overwrites lesson one:

| five lessons | remembered |
|---|---|
| with lesson rehearsal | **4/5** |
| without | 1/5 — and the replies turn to word salad |

## Continuing tomorrow, and `/polish`

Lessons live in the weights, but *rehearsal* works from a list in memory — which a
reloaded checkpoint does not have. So picking a taught model up the next day and
teaching fact eleven would quietly cost facts one to ten. `--remember` fixes that:
hand back the lesson log and every earlier lesson is rehearsed with the new ones.

```bash
py slm/learn.py --checkpoint build/peitho_taught.pt \
                --log build/peitho_lessons.jsonl \
                --remember build/peitho_lessons.jsonl
```

Give each taught model its own `--log` so the two files travel together: the
checkpoint is what it knows, the log is what it was told.

Lessons are taught one at a time, so the newest is freshest and wins ties — after
ten lessons, "Who are you?" started answering with whatever came last. `/polish`
trains every lesson together for a few steps, which removes the recency because
nothing is the most recent any more. It took rephrasings from 5/6 to 6/6 without
costing a sum. Run it once at the end of a session.

## How it knows a lesson went wrong

Two measurements, because one of them is not enough.

**Drift** is the loss on a fixed sample of the corpus, drawn once from a fixed
seed. It catches wholesale damage.

**Fixed sums** are six sums the model could do before the session started, checked
again after every lesson with greedy sampling, so the result is deterministic.
These exist because drift alone missed real damage: three lessons once moved the
probe by +0.037 — nothing, apparently — while taking arithmetic from 4/6 to 0/6.
Arithmetic is the most fragile thing the model does, which makes it the best early
warning.

A lesson that costs more than `--allow_forgetting` sums (default 1) or drifts more
than `--max_drift` is undone automatically and reported. One sum is noise at this
size; two is damage. `/undo` does it by hand; `--no_guard` turns it off.

## Sampling has to be colder than ordinary chat

A lesson makes the taught reply a narrow path through the distribution, and
`chat.py`'s usual temperature of 0.8 wanders off it — the model verifies perfectly
at teach time and then answers nonsense in conversation, which looks like the
teaching failed when it did not. Four lessons, five samples each:

| | taught answer returned |
|---|---|
| temperature 0.2, top_k 5 | **20/20** |
| 0.3–0.5, top_k 5–10 | 18/20 |
| 0.7, top_k 20 | 14/20 |

So `learn.py` chats at 0.2 and top_k 5. Untaught questions still vary at that
setting — a kettle question came back three different ways — because the corpus
supports several good continuations where a taught fact supports one.

## What it can and cannot do

Five facts taught in one session, then asked cold:

| | parameters | lessons kept | sums | drift |
|---|---|---|---|---|
| `small` | 382K | 2/5 | 0/6 (from 1) | +0.033 |
| `medium` | 855K | **5/5** | 3/6 (from 4) | +0.006 |
| `large` | 2.7M | **5/5** | 3/6 (from 4) | **+0.0007** |

`small` is too small to teach — it takes the lesson and pays for it elsewhere.
`large` learns in 8 steps rather than 12 and barely moves.

Known limits, all measured:

- **Around five facts is comfortable.** At eight, recall falls to 5–6 of 8 and one
  or two sums go with it.
- **Similar questions interfere.** "What is my dog's name?" and "What is my brother
  called?" are the same shape, and the second can capture the first.
- **Mid-conversation recall is worse than cold.** Four facts: 3/4 asked cold, 2/4
  with a couple of exchanges already in view. The model attends to everything in
  its context, and taught pairs were learned without any.
- **Replacing an answer it already had generalizes badly.** A new fact carried over
  to an unseen wording 100% of the time; overwriting the kettle answer managed 37%,
  even though the exact question answers correctly.
- **Do not teach arithmetic.** A worked sum competes with the arithmetic already in
  the model and loses: it rarely carries over to a new wording, and it leaves the
  reply distribution ragged enough that unrelated questions degrade at higher
  sampling temperatures. A lesson that looks like a sum now says so and can be
  `/undo`ne.
- **Rules need several examples.** One sum teaches that sum.
- **It will not chain facts.** Teach that your dog is Rufus and that Rufus is a
  terrier, and "what breed is my dog" is still not in there.

## Keeping what it learned

Nothing is written to disk unless you ask.

- `/save [path]` writes a full-precision checkpoint — the exact thing, and the only
  format that can be taught again later.
- `/export` writes the 3-line quantized export, versioned into `models/` exactly the
  way `train.py` versions its own: `models/taught_1.0.txt`, then `taught_1.1.txt`.
  Nothing about a taught model is a special case after that — `peitho.html` finds it
  by the same probe it uses for the released models and offers it as "Taught 1.0",
  `standalone.py --model taught` runs it with no PyTorch at all, and
  `versions.resolve("taught")` accepts the bare base name. `--name` changes the base,
  `--export_on_exit` does it automatically on the way out, and
  `py tools/make_js_models.py --only taught` adds the `.js` mirror for a sandbox that
  cannot fetch.

  Verified end to end: taught on `large`, exported at 3,672,835 bytes — the same size
  as `large_1.2.txt` — the pure-Python decoder answers *"Peitho, your personal AI
  companion."*, and the browser page loads it as `Taught 1.0, 2,762,880 weights,
  8-bit` and gives the taught reply.

  Five taught facts survived export intact at every group size tried; a single
  freshly-taught fact garbled its tail once ("Rufuchles"), so check an export before
  trusting it. Taught exports are gitignored — they hold your facts, not the
  project's.
- `build/lessons.jsonl` records every lesson, whether it generalized, what it cost,
  and whether it was undone. Weights record what was learned but not what was
  taught: with the log, a session can be replayed onto the original checkpoint, and
  a run that went wrong can be rebuilt without the bad lessons.

## Things that did not work

Aiming half the rehearsal at arithmetic — findable by searching the corpus for
`+` — looked obvious, on the theory that random slices of 30M characters rarely
land on a sum. Over three runs of eight lessons it kept 3, 2 and 0 of six sums,
against 3, 4 and 4 for plain uniform sampling. Rehearsing a skill narrowly is not
the same as keeping it. Uniform sampling stayed.

Stopping a lesson when its loss dropped below a threshold looked equally
reasonable, and it quietly broke the hardest case. Replacing an answer the model
already had, the first character is the whole fight — the rest of the reply is easy
to continue — so mean loss over sixty characters reads 0.03 while the old reply
still wins the only position that decides anything. Lessons now stop when the model
actually says the new answer.
