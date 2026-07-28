"""Compact, portable serialization of the SLM's weights.

File layout (3 lines, no compression library needed to read it):

    line 1  JSON header: config, vocab string, group size, tensor list
    line 2  base85 of the packed quantized weights
    line 3  base85 of the float16 scale factors, one per group

Decoding only needs: base85 decode -> unpack nibbles -> multiply by group scale.
No zlib, so a browser/Khan Academy port stays simple.

Quantization granularity matters a great deal. One scale per whole tensor is too
coarse for the attention circuits that carry in-context recall, costing ~0.40 nats
and most of the model's memory. Scales are therefore per group of `group_size`
consecutive weights: a few KB more of scales, nearly all of the quality kept.

base85 is used instead of base64 because it carries the same bytes in 25%
overhead instead of 33%, and its alphabet contains no quote or backslash
characters, so lines paste straight into a JS string literal without escaping.
"""

import base64
import json

import numpy as np
import torch

from model import GPT, GPTConfig

FORMAT_VERSION = 4
DEFAULT_GROUP_SIZE = 32

# Skipped because they are reconstructed for free when the model is built:
#   *.mask      -> a fixed lower-triangular matrix
#   head.weight -> tied to tok_emb.weight
SKIP_SUFFIXES = ("mask",)
SKIP_NAMES = ("head.weight",)


def encode_text(raw):
    return base64.b85encode(raw).decode("ascii")


def decode_text(text):
    return base64.b85decode(text)


def _exportable(state_dict):
    for name, tensor in state_dict.items():
        if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
            continue
        yield name, tensor


def _padded_len(numel, group_size):
    return numel + (-numel % group_size)


def _quantize(tensor, bits, group_size):
    """Symmetric quantization with one scale per group of consecutive weights.
    Groups never straddle tensors, so a tiny tensor cannot drag a big one."""
    flat = tensor.detach().cpu().float().numpy().reshape(-1)
    pad = -len(flat) % group_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    groups = flat.reshape(-1, group_size)

    qmax = (1 << (bits - 1)) - 1
    scales = np.abs(groups).max(axis=1) / qmax
    scales = np.maximum(scales, 1e-6).astype(np.float16)
    q = np.round(groups / scales[:, None].astype(np.float32)).clip(-qmax, qmax).astype(np.int8)
    return q.reshape(-1), scales


def _pack(q, bits):
    if bits == 8:
        return q.astype(np.int8).tobytes()
    nibbles = q.astype(np.uint8) & 0x0F
    if nibbles.size % 2:
        nibbles = np.append(nibbles, np.uint8(0))
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8).tobytes()


def _unpack(blob, count, bits):
    if bits == 8:
        return np.frombuffer(blob, dtype=np.int8, count=count)
    packed = np.frombuffer(blob, dtype=np.uint8)
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    vals = nibbles[:count].astype(np.int16)
    return np.where(vals > 7, vals - 16, vals).astype(np.int8)


def export_compressed(model, config, stoi, itos, out_path, bits=4,
                      group_size=DEFAULT_GROUP_SIZE):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    all_q, all_scales, meta = [], [], []
    for name, tensor in _exportable(model.state_dict()):
        q, scales = _quantize(tensor, bits, group_size)
        all_q.append(q)
        all_scales.append(scales)
        meta.append({"n": name, "s": list(tensor.shape)})

    q = np.concatenate(all_q)
    scales = np.concatenate(all_scales)

    vocab = "".join(itos[i] for i in range(len(itos)))
    header = {
        "v": FORMAT_VERSION,
        "bits": bits,
        "group": group_size,
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
        },
        "vocab": vocab,
        "tensors": meta,
    }

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        f.write(encode_text(_pack(q, bits)) + "\n")
        f.write(encode_text(scales.tobytes()) + "\n")

    return out_path


def import_compressed(path, device="cpu"):
    with open(path, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        weight_blob = decode_text(f.readline().strip())
        scale_blob = decode_text(f.readline().strip())

    if header.get("v") != FORMAT_VERSION:
        raise ValueError(f"Unsupported model file version {header.get('v')} "
                         f"(expected {FORMAT_VERSION}). Retrain to regenerate the export.")

    bits = header["bits"]
    group_size = header["group"]
    config = GPTConfig(dropout=0.0, **header["config"])
    vocab = header["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    total_padded = sum(_padded_len(int(np.prod(m["s"])), group_size) for m in header["tensors"])
    q = _unpack(weight_blob, total_padded, bits).astype(np.float32)
    scales = np.frombuffer(scale_blob, dtype=np.float16).astype(np.float32)

    state_dict = {}
    q_at = scale_at = 0
    for m in header["tensors"]:
        shape = m["s"]
        numel = int(np.prod(shape))
        padded = _padded_len(numel, group_size)
        n_groups = padded // group_size

        chunk = q[q_at:q_at + padded].reshape(n_groups, group_size)
        chunk = chunk * scales[scale_at:scale_at + n_groups][:, None]
        arr = chunk.reshape(-1)[:numel].reshape(shape)

        state_dict[m["n"]] = torch.from_numpy(arr.copy())
        q_at += padded
        scale_at += n_groups

    model = GPT(config)
    # mask buffers and the tied head.weight are rebuilt by the constructor.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected tensors in {path}: {unexpected}")
    for key in missing:
        if key not in SKIP_NAMES and not key.endswith(SKIP_SUFFIXES):
            raise ValueError(f"Missing tensor in {path}: {key}")

    return model.to(device), config, stoi, itos
