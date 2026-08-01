"""Run the exported model with NO PyTorch and NO numpy - only the standard library.

This is the reference decoder for model_compressed.txt. It exists to prove the
export format needs nothing but plain arithmetic, and to serve as a line-by-line
blueprint for a JavaScript port (Khan Academy or anywhere else).

Everything a port has to reimplement, in order:

  1. read line 1 as JSON                      -> JSON.parse
  2. base85-decode lines 2 and 3              -> ~20 lines of JS, see b85_decode
  3. unpack weights: 8-bit is one signed byte each; 4-bit is two nibbles per
     byte (byte & 15, byte >> 4) with 8..15 meaning -8..-1
  4. decode float16 scale factors              -> see fp16, pure bit math
  5. weight = quantized_value * scale_of_its_group, where groups are
     `group_size` consecutive weights within a tensor
  6. forward pass: layernorm, matmul, gelu, softmax, attention

A key-value cache is used so each new character costs one small forward pass
instead of reprocessing the whole context.
"""

import argparse
import json
import math
import random
import sys

import versions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

USER_MARK, BOT_MARK, END_MARK = "▶", "◀", "■"

# Python's base64.b85 alphabet, in value order.
B85_ALPHABET = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "!#$%&()*+-;<=>?@^_`{|}~")
B85_LOOKUP = {ch: i for i, ch in enumerate(B85_ALPHABET)}


def b85_decode(text):
    """Every 5 characters carry 4 bytes, base 85. The final group may be short."""
    out = bytearray()
    for start in range(0, len(text), 5):
        group = text[start:start + 5]
        n_pad = 5 - len(group)
        group += B85_ALPHABET[-1] * n_pad  # pad with the highest digit
        value = 0
        for ch in group:
            value = value * 85 + B85_LOOKUP[ch]
        chunk = [(value >> 24) & 255, (value >> 16) & 255, (value >> 8) & 255, value & 255]
        out.extend(chunk[:4 - n_pad])
    return bytes(out)


def fp16(lo, hi):
    """Decode one IEEE half-precision float from two bytes (little endian)."""
    bits = lo | (hi << 8)
    sign = -1.0 if bits >> 15 else 1.0
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0:
        return sign * fraction * 2.0 ** -24          # subnormal
    if exponent == 31:
        return sign * float("inf")
    return sign * (1024 + fraction) * 2.0 ** (exponent - 25)


def unpack_quantized(blob, count, bits):
    """Return `count` signed integers from the packed blob."""
    if bits == 8:
        return [b - 256 if b > 127 else b for b in blob[:count]]
    out = []
    for byte in blob:
        out.append(byte & 0x0F)
        out.append(byte >> 4)
    del out[count:]
    return [v - 16 if v > 7 else v for v in out]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        weight_blob = b85_decode(f.readline().strip())
        scale_blob = b85_decode(f.readline().strip())

    bits = header["bits"]
    group = header["group"]
    if bits not in (4, 8):
        raise SystemExit(f"unsupported bit width {bits}")

    def padded(numel):
        return numel + (-numel % group)

    total = sum(padded(_numel(m["s"])) for m in header["tensors"])
    values = unpack_quantized(weight_blob, total, bits)
    scales = [fp16(scale_blob[i], scale_blob[i + 1]) for i in range(0, len(scale_blob), 2)]

    tensors = {}
    at = scale_at = 0
    for meta in header["tensors"]:
        numel = _numel(meta["s"])
        span = padded(numel)
        flat = [0.0] * numel
        for i in range(numel):
            flat[i] = values[at + i] * scales[scale_at + i // group]
        tensors[meta["n"]] = flat
        at += span
        scale_at += span // group

    return header, tensors


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ------------------------------------------------------------------ math bits

def layernorm(x, weight, bias, eps=1e-5):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(x[i] - mean) * inv * weight[i] + bias[i] for i in range(n)]


def linear(x, weight, bias, n_out, n_in):
    """weight is row-major [n_out, n_in]."""
    out = [0.0] * n_out
    for o in range(n_out):
        base = o * n_in
        acc = 0.0
        for i in range(n_in):
            acc += weight[base + i] * x[i]
        out[o] = acc + (bias[o] if bias else 0.0)
    return out


def gelu(x):
    return [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x]


def softmax(x):
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


class Model:
    def __init__(self, path):
        header, self.t = load(path)
        cfg = header["config"]
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.block_size = cfg["block_size"]
        self.vocab_size = cfg["vocab_size"]
        self.head_dim = self.n_embd // self.n_head
        self.vocab = header["vocab"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.reset()

    def reset(self):
        # cache[layer] = (list of key vectors, list of value vectors)
        self.cache = [([], []) for _ in range(self.n_layer)]
        self.pos = 0

    def step(self, token_id):
        """Feed one token, return logits over the vocabulary."""
        t = self.t
        C, H, D = self.n_embd, self.n_head, self.head_dim
        pos = min(self.pos, self.block_size - 1)

        x = [t["tok_emb.weight"][token_id * C + i] + t["pos_emb.weight"][pos * C + i]
             for i in range(C)]

        for layer in range(self.n_layer):
            p = f"blocks.{layer}."
            h = layernorm(x, t[p + "ln1.weight"], t[p + "ln1.bias"])
            qkv = linear(h, t[p + "attn.qkv.weight"], t[p + "attn.qkv.bias"], 3 * C, C)
            q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]

            keys, values = self.cache[layer]
            keys.append(k)
            values.append(v)
            if len(keys) > self.block_size:      # slide the window
                del keys[0]
                del values[0]

            attn_out = [0.0] * C
            scale = 1.0 / math.sqrt(D)
            for head in range(H):
                off = head * D
                scores = []
                for kv in keys:
                    acc = 0.0
                    for d in range(D):
                        acc += q[off + d] * kv[off + d]
                    scores.append(acc * scale)
                weights = softmax(scores)
                for j, w in enumerate(weights):
                    if w == 0.0:
                        continue
                    vv = values[j]
                    for d in range(D):
                        attn_out[off + d] += w * vv[off + d]

            proj = linear(attn_out, t[p + "attn.proj.weight"], t[p + "attn.proj.bias"], C, C)
            x = [x[i] + proj[i] for i in range(C)]

            h = layernorm(x, t[p + "ln2.weight"], t[p + "ln2.bias"])
            hidden = gelu(linear(h, t[p + "mlp.fc.weight"], t[p + "mlp.fc.bias"], 4 * C, C))
            down = linear(hidden, t[p + "mlp.proj.weight"], t[p + "mlp.proj.bias"], C, 4 * C)
            x = [x[i] + down[i] for i in range(C)]

        x = layernorm(x, t["ln_f.weight"], t["ln_f.bias"])
        # The output head shares the token embedding matrix (weight tying).
        return linear(x, t["tok_emb.weight"], None, self.vocab_size, C)

    def feed(self, text):
        logits = None
        for ch in text:
            if ch in self.stoi:
                logits = self.step(self.stoi[ch])
                self.pos += 1
        return logits

    def sample(self, logits, temperature, top_k, rng):
        scaled = [v / max(temperature, 1e-6) for v in logits]
        if top_k:
            cutoff = sorted(scaled, reverse=True)[min(top_k, len(scaled)) - 1]
            scaled = [v if v >= cutoff else -1e30 for v in scaled]
        probs = softmax(scaled)
        r = rng.random()
        acc = 0.0
        for i, prob in enumerate(probs):
            acc += prob
            if r <= acc:
                return i
        return len(probs) - 1

    def reply(self, prompt, max_new=120, temperature=0.7, top_k=40, rng=random):
        self.reset()
        logits = self.feed(prompt)
        out = []
        for _ in range(max_new):
            token = self.sample(logits, temperature, top_k, rng)
            ch = self.vocab[token]
            if ch in (END_MARK, USER_MARK, BOT_MARK, "\n"):
                break
            out.append(ch)
            logits = self.step(token)
            self.pos += 1
        return "".join(out).strip()


def main():
    p = argparse.ArgumentParser(description="Pure-Python inference from the exported text file.")
    p.add_argument("--model", default="",
                   help="Path to an export, a base name for its highest version, "
                        "or nothing for the most recent export in models/.")
    p.add_argument("--prompt", help="Answer one prompt and exit.")
    p.add_argument("--max_new", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    path = versions.resolve(args.model)
    print(f"Loading {path} with the standard library only...")
    model = Model(path)
    print(f"n_layer={model.n_layer} n_embd={model.n_embd} block_size={model.block_size} "
          f"vocab={model.vocab_size}")
    rng = random.Random(args.seed)

    def ask(text):
        prompt = f"{USER_MARK}{text}{END_MARK}\n{BOT_MARK}"
        return model.reply(prompt, args.max_new, args.temperature, args.top_k, rng)

    if args.prompt:
        print(f"Bot: {ask(args.prompt)}")
        return

    print("No PyTorch in sight. Type 'exit' to quit.\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        print(f"Bot: {ask(line)}\n")


if __name__ == "__main__":
    main()
