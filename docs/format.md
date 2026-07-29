# The export format

`models/<base>_<version>.txt` is exactly 3 lines:

```
line 1  JSON header: config, vocab string, group size, tensor names + shapes
line 2  base85 of the packed quantized weights
line 3  base85 of the float16 scale factors, one per group of weights
```

No zlib, no pickle, so a port needs only base85, some bit math and a matmul. base85
rather than base64 for 25% overhead instead of 33%, and its alphabet contains no
quotes or backslashes, so a line pastes straight into a JS string literal.

## Use 8 bits, not 4

`--bits 8` is effectively lossless (+0.0001 nats). `--bits 4` halves the file but
**destroys memory** — recall drops from ~72% to ~6%, spelling from 97% to 64% —
because the copying circuits need more precision than 4 bits gives them. Scales are
per group of 32 weights for the same reason: per-tensor 4-bit cost 0.40 nats.

## Versioning

Training writes to the next free version, so nothing is ever overwritten:

```
(empty)                  -> small_1.0
small_1.0                -> small_1.1
small_1.0, small_1.1     -> small_1.2
```

The base name defaults to the preset, so `--preset small` writes `small_*` and
`--name peitho` writes `peitho_*`.

Anything that loads an export takes whichever form is convenient:

| You type | You get |
|---|---|
| `small_1.2` | that exact export |
| `small_1.2.txt` | same, filename form |
| `models/small_1.2.txt` | same, full path |
| `small` | the highest `small_*` |
| *(nothing)* | the most recent export of any name |

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

These are deliberately not on any keyboard, so they can never collide with typed
text. The end marker is why replies stop when they are finished rather than at a
character limit: `generate()` takes a `stop_id` and breaks as soon as it samples it.

If you write your own turns, never put a literal `■` inside the text of a turn — the
model would learn to stop mid-sentence. `tests/test_slm.py` checks this.

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
