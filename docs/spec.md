# The Peitho export format, version 4

A complete specification for reading a `.txt` export and running the model in it.
Everything needed is here: base85, some bit math, and a matmul. No compression
library, no serialisation library, no dependencies.

If you implement this, [check your work against the conformance
kit](#proving-a-port-correct) — a decoder can be wrong in ways that still produce
fluent-looking text.

## The file

Exactly three lines, LF-terminated, UTF-8:

```
line 1   JSON header
line 2   base85 of the packed quantized weights
line 3   base85 of the float16 group scales
```

base85 rather than base64 for 25% overhead instead of 33%. The alphabet
(RFC 1924 as implemented by Python's `base64.b85encode`) contains no `'`, `"` or
`\`, so lines 2 and 3 paste directly into a single-quoted string literal in most
languages. It does contain `` ` `` and `$`, so a JavaScript template literal will
break — use single quotes.

### Line 1: the header

```json
{"v":4,"bits":8,"group":32,
 "config":{"vocab_size":55,"block_size":16,"n_layer":2,"n_head":2,"n_embd":20},
 "vocab":"\nABC...",
 "tensors":[{"n":"tok_emb.weight","s":[55,20]}, ...]}
```

| Field | Meaning |
|---|---|
| `v` | format version — `4` |
| `bits` | `8` or `4`, the width of each quantized weight |
| `group` | weights per scale factor, normally `32` |
| `config` | model shape; `n_embd` must be divisible by `n_head` |
| `vocab` | the alphabet as one string. Index = token id, so `vocab[7]` is token 7 |
| `tensors` | every stored tensor, **in order**, with its name and shape |

The header is valid JSON *and* a valid JavaScript object literal, so it can be
pasted after `header:` unquoted.

### Line 2: the weights

Quantized values for every tensor in `tensors` order, concatenated, then packed.

* **8-bit** — one signed byte per value, two's complement, range −127…127.
* **4-bit** — two values per byte, **low nibble first**: byte `0xBA` holds value
  `0xA` then value `0xB`. Each nibble is a signed 4-bit integer, so a nibble
  greater than 7 has 16 subtracted from it, giving the range −7…7.

Each tensor is padded with zeros to a whole number of groups before packing, so a
tensor of 55×20 = 1100 values with `group` 32 contributes ceil(1100/32) = 35 groups
= 1120 packed values. **Padding belongs to the tensor it pads** — the next tensor
starts on a fresh group boundary. Skip the padding when reshaping.

### Line 3: the scales

One float16 per group, in the same order, little-endian. Decode with the usual
IEEE 754 half-precision rules: sign bit, 5-bit exponent with bias 15, 10-bit
mantissa, subnormals when the exponent field is zero.

### Reconstructing a weight

```
value  = the quantized integer at index i within its tensor
group  = floor(i / group_size)          (counted within this tensor)
weight = value * scale[group]
```

## What is not in the file

Two tensors are omitted because they are free to rebuild:

| Name | How to reconstruct |
|---|---|
| `head.weight` | tied to `tok_emb.weight` — the same matrix, used transposed |
| `*.mask` | a lower-triangular causal mask; older exports stored it, `v4` does not |

## The model

A decoder-only transformer, pre-norm, GELU, learned positional embeddings, weight
tied output head. For token id `t` at position `p`:

```
x = tok_emb[t] + pos_emb[min(p, block_size - 1)]

for each of n_layer blocks:
    x = x + attention(layernorm(x, ln1.weight, ln1.bias))
    x = x + mlp(layernorm(x, ln2.weight, ln2.bias))

x      = layernorm(x, ln_f.weight, ln_f.bias)
logits = x @ tok_emb.weight.T          # weight tying, no bias
```

**attention** — one head at a time, over the current token and all earlier ones:

```
qkv       = x @ qkv.weight.T + qkv.bias          # 3 * n_embd wide
q, k, v   = qkv split into three equal parts, in that order
per head h of size D = n_embd / n_head:
    scores  = (q_h · k_h(j)) / sqrt(D)   for every earlier position j, and itself
    weights = softmax(scores)
    out_h   = sum over j of weights[j] * v_h(j)
y         = concat(out_h for all heads) @ proj.weight.T + proj.bias
```

Causal masking is implicit: only positions up to and including the current one are
in the sum. There is no need to materialise a mask.

**mlp**

```
h = gelu(x @ fc.weight.T + fc.bias)        # 4 * n_embd wide
y = h @ proj.weight.T + proj.bias
```

**layernorm**, with `eps = 1e-5`:

```
mean = average(x)
var  = average((x - mean)^2)               # biased, divide by n not n-1
y    = (x - mean) / sqrt(var + eps) * weight + bias
```

**gelu** — the exact form, not the tanh approximation:

```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

`standalone.py` includes a short `erf` series if your language lacks one. The tanh
approximation differs by up to ~1e-3, which exceeds the conformance tolerance.

### Weight layout

PyTorch stores a linear layer's weight as `[out_features, in_features]` and computes
`x @ W.T`. Row-major, so element `(o, i)` is at `o * in_features + i`. Getting this
transposed is the single most common porting error, and it produces plausible
nonsense rather than an obvious failure.

## Generating text

Positions are clamped: once `pos` reaches `block_size`, keep using the last
positional embedding, or slide the window. Feed tokens one at a time, caching each
layer's keys and values, or each new character costs a full re-run of the context.

Sampling, as the reference implementations do it:

```
logits = logits / max(temperature, 1e-6)
keep the top_k largest, set the rest to -infinity
probabilities = softmax(logits)
draw from probabilities
```

### Conversation markers

Four characters that are not on any keyboard, so they can never collide with typed
text:

| Name | Char | Code point | Meaning |
|---|---|---|---|
| start | `◈` | U+25C8 | start of a conversation |
| user | `▶` | U+25B6 | a line the person said |
| bot | `◀` | U+25C0 | a line the model said |
| end | `■` | U+25A0 | end of turn — **stop generating** |

A turn is marker + text + end marker, one per line. Prompt the model with
`◈\n▶your text■\n◀` and stop when it emits `■`.

## Proving a port correct

`conformance/` holds a purpose-built micro model — 2 layers, 20 dimensions, 55
tokens, 17 KB — and the numbers it must produce.

```bash
py tools/make_conformance.py     # regenerate the kit
py tests/test_conformance.py     # check PyTorch, pure Python and the browser JS
node conformance/check.mjs       # just the JavaScript in peitho.html
```

`conformance/vectors.json` gives, for each of five prompts: the token ids, the
logits after the final token, and the greedy continuation for eight steps.

| Field | Use |
|---|---|
| `ids` | feed these, in order, from a fresh state |
| `logits` | compare your output after the last token, to `tolerance` (1e-4) |
| `greedy_ids` | take argmax repeatedly; catches a cache that breaks after token one |

The greedy check earns its place. A key-value cache that is correct for the first
token and wrong afterwards passes a single-logit comparison and fails this.

Three independent implementations agree on these numbers today: PyTorch
(`export.py`), pure Python with no dependencies at all (`standalone.py`), and the
JavaScript that ships inside `peitho.html`.

### If your numbers are close but not equal

In rough order of likelihood:

1. **Transposed weights.** See the layout note above.
2. **The tanh GELU approximation** instead of the `erf` form.
3. **Wrong layernorm variance** — using the unbiased estimator, or the wrong `eps`.
4. **Nibbles in the wrong order** at 4 bits, or forgetting the sign extension.
5. **Group padding treated as global** rather than per tensor.
6. **Positional embedding not clamped** past `block_size`.
7. **float16 subnormals** dropped when decoding the scales.

A port that is wrong in any of these ways still writes fluent English, which is
exactly why the vectors exist.
