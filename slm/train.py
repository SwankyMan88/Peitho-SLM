import argparse
import contextlib
import math
import os
import random
import re
import signal
import time

import numpy as np
import torch

from model import GPT, GPTConfig, build_vocab, encode
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK
from export import export_compressed
import paths
import versions

# Architecture presets. Bigger = better text, larger export file.
PRESETS = {
    # Only ever asked to open a conversation, so it can be far smaller than
    # anything that has to answer one - small enough to sit in the page itself.
    "greeter": dict(n_layer=2, n_head=4, n_embd=40,  block_size=128),
    "tiny":   dict(n_layer=2, n_head=4, n_embd=64,  block_size=64),
    "small":  dict(n_layer=3, n_head=4, n_embd=96,  block_size=96),
    "medium": dict(n_layer=4, n_head=4, n_embd=128, block_size=128),
    "large":  dict(n_layer=6, n_head=6, n_embd=192, block_size=192),
    # Deeper rather than wider: heads stay 32 wide, like every other preset.
    "xl":     dict(n_layer=9, n_head=6, n_embd=192, block_size=192),
    # About 10M parameters and twice the context, so a conversation - or a passage
    # and the questions about it - stays in view for twice as long.
    "xxl":    dict(n_layer=8, n_head=10, n_embd=320, block_size=768),
    # ~20M parameters: deeper and wider again, heads still 32 wide. A day of training
    # on a 3060, and a 27 MB export - too big for the page to feel instant.
    "xxxl":   dict(n_layer=11, n_head=12, n_embd=384, block_size=768),
}


def load_ids(path, stoi, device):
    """Encode a text file to a device tensor, caching the result in build/.

    Encoding megabytes one character at a time in Python costs several seconds per
    run for a result that only changes when the file does."""
    # Kept in build/ rather than beside the corpus: caching next to
    # data/training.txt would drop tens of megabytes into a tracked folder.
    cache = os.path.join(paths.ensure_build(),
                         os.path.basename(path) + ".ids.npz")
    signature = np.array([os.path.getmtime(path), os.path.getsize(path), len(stoi)])

    # int16 in its own array, not concatenated onto a float64 signature: at one
    # character per element that difference is 2 bytes against 8, which is 24 MB
    # rather than 112 MB on a corpus of a few million characters.
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as stored:
            if np.array_equal(stored["signature"], signature):
                return torch.from_numpy(stored["ids"]).to(device)

    with open(path, "r", encoding="utf-8") as f:
        text = "".join(c for c in f.read() if c in stoi)
    ids = np.fromiter((stoi[c] for c in text), dtype=np.int16, count=len(text))
    np.savez(cache, signature=signature, ids=ids)
    return torch.from_numpy(ids).to(device)


def get_batch(data, block_size, batch_size, device):
    """Gather a batch entirely on the device holding `data`.

    Slicing per sequence in Python and copying to the GPU each step costs more
    than the forward pass at these model sizes, so `data` is kept resident on the
    device and indexed with one gather."""
    ix = torch.randint(data.numel() - block_size - 1, (batch_size,), device=data.device)
    window = torch.arange(block_size, device=data.device)
    rows = ix[:, None] + window[None, :]
    x = data[rows].long()
    y = data[rows + 1].long()
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20, autocast=None):
    """Accumulate on the device: calling .item() per iteration forces a host sync
    each time, which costs more than the forward passes being measured."""
    model.eval()
    ctx = autocast or contextlib.nullcontext()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        total = torch.zeros((), device=device)
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            total += loss.detach()
        out[name] = (total / eval_iters).item()
    model.train()
    return out


def make_optimizer(model, lr, weight_decay, device="cuda"):
    """Decay only matrices. Biases and LayerNorm gains are left alone, which is
    standard practice - decaying them hurts more than it regularizes."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    fused = device == "cuda"      # the fused kernels are CUDA-only
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.99), fused=fused)


def lr_at(step, total_steps, base_lr, min_lr, warmup):
    """Linear warmup then cosine decay to min_lr."""
    if step <= warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train the character-level SLM.")
    parser.add_argument("--data", default=paths.TRAINING)
    parser.add_argument("--val_data", default=paths.HELDOUT,
                        help="Separate validation file. Falls back to a slice of --data if absent.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium",
                        help="Architecture size preset (individual flags below override it).")
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Higher values fight memorization on small corpora.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Pulls weights toward zero so the model prefers simpler solutions.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="1e-3 suits models this size. Much lower and the copying/induction "
                             "circuit may never form - recall stays at 0%.")
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--select_by", choices=("val", "train"), default="val",
                        help="Which loss picks the best checkpoint. Use 'train' when the corpus "
                             "is small and curated: faithful reproduction of good hand-written "
                             "replies is then the goal, and selecting on validation loss picks an "
                             "undertrained checkpoint that writes badly.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Stop after this many evals with no improvement (0 = never).")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8,
                        help="Export width. 8 is effectively lossless. 4 halves the file but "
                             "wrecks in-context recall, so only use it "
                             "if you do not need memory.")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Weights per quantization scale. Smaller = more accurate, "
                             "slightly larger file.")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Patience is counted in evals, so a smaller interval also means "
                             "a shorter grace period before early stopping.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Where to train. CPU works and is roughly 20x slower at "
                             "these sizes, which is tolerable for the smaller presets.")
    parser.add_argument("--out_dir", default=paths.BUILD)
    parser.add_argument("--checkpoint", default="model_full.pt")
    parser.add_argument("--compressed_out", default="",
                        help="Explicit export path. Left empty, the export is written to "
                             "models/<name>_<version>.txt with the next free version, so "
                             "earlier exports are never overwritten.")
    parser.add_argument("--name", default="",
                        help="Base name for the export. Defaults to the preset name.")
    parser.add_argument("--models_dir", default=versions.MODELS_DIR)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint and train from scratch.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the weights and the batches, so the same command gives "
                             "the same run. Unseeded by default.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a paused or interrupted run exactly where it stopped: "
                             "same step, schedule, optimizer state and batches. Reads "
                             "<checkpoint>.resume, which is written at every eval. Pause a "
                             "run with Ctrl+C, or by creating <checkpoint>.pause beside it.")
    parser.add_argument("--log", default="",
                        help="Also write progress to this file, flushed every line, so "
                             "a run can be watched from another window. Shell "
                             "redirection cannot do this: Python block-buffers stdout "
                             "when it is not a terminal, and the file stays empty for "
                             "minutes at a time.")
    args = parser.parse_args()

    # Everything printed from here on goes to the terminal and, if asked, to the log.
    log_file = None
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        log_file = open(args.log, "w", encoding="utf-8", buffering=1)

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()      # buffering=1 is line-buffered; be explicit anyway

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA device available. Use --device cpu.")
    say(f"Using device: {device}")
    if args.seed is not None:
        torch.manual_seed(args.seed)      # every device's generator, before the weights exist
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, args.checkpoint)
    resume_path = checkpoint_path + ".resume"
    pause_path = checkpoint_path + ".pause"
    resumed = False
    state = None

    # What a resumed run must keep identical, or it is no longer the same run.
    SCHEDULE = ("steps", "batch_size", "lr", "min_lr", "warmup", "weight_decay", "grad_clip",
                "select_by", "patience", "eval_interval", "dropout")

    if args.resume:
        if not os.path.exists(resume_path):
            raise SystemExit(f"Nothing to resume: {resume_path} does not exist.")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        for key, value in state["schedule"].items():
            setattr(args, key, value)
        config, stoi, itos = state["config"], state["stoi"], state["itos"]
        model = GPT(config).to(device)
        model.load_state_dict(state["model"])
        resumed = True
        say(f"Resuming from step {state['step']} of {args.steps} ({resume_path})")
    elif os.path.exists(checkpoint_path) and not args.fresh:
        say(f"Found existing checkpoint at {checkpoint_path}, resuming training...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        stoi, itos = ckpt["stoi"], ckpt["itos"]
        config.dropout = args.dropout
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["state_dict"])
        resumed = True

        arch_flags = [f for f in ("block_size", "n_layer", "n_head", "n_embd")
                      if getattr(args, f) is not None]
        if arch_flags or args.preset != parser.get_default("preset"):
            say(f"  NOTE: architecture is fixed by the checkpoint "
                  f"(n_layer={config.n_layer}, n_embd={config.n_embd}, block_size={config.block_size}). "
                  f"Pass --fresh to rebuild with a different size.")

        unknown = sorted(set(text) - set(stoi))
        if unknown:
            raise ValueError(
                f"These characters in {args.data} are not in the checkpoint's vocab: {unknown!r}. "
                "Pass --fresh to retrain from scratch with the new text."
            )
    else:
        stoi, itos = build_vocab(text)
        arch = dict(PRESETS[args.preset])
        for key in ("block_size", "n_layer", "n_head", "n_embd"):
            if getattr(args, key) is not None:
                arch[key] = getattr(args, key)
        say(f"Architecture: {args.preset} preset -> {arch}")
        config = GPTConfig(vocab_size=len(stoi), dropout=args.dropout, **arch)
        model = GPT(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    say(f"Vocab size: {config.vocab_size} | Params: {n_params:,} | Training chars: {len(text):,}")
    if len(text) < n_params / 20:
        say("  WARNING: the corpus is small relative to the model. Expect memorization -- "
              "add more text or use a smaller --preset.")
    say(f"{'Resumed' if resumed else 'Fresh'} model, training for up to {args.steps} steps "
          f"(dropout={args.dropout}, weight_decay={args.weight_decay})")

    data = load_ids(args.data, stoi, device)

    if args.val_data and os.path.exists(args.val_data):
        train_data = data
        val_data = load_ids(args.val_data, stoi, device)
        say(f"Validating on {args.val_data} ({val_data.numel():,} chars, separate from training)")
    else:
        n = int((1 - args.val_frac) * len(data))
        train_data, val_data = data[:n], data[n:]
        say(f"Validating on the last {args.val_frac:.0%} of {args.data}")

    if len(val_data) <= config.block_size + 1:
        say("  WARNING: not enough validation text; validation loss will not be meaningful.")
        train_data, val_data = data, data

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
                else contextlib.nullcontext())
    if use_bf16:
        say("Using bfloat16 autocast (master weights stay float32)")

    optimizer = make_optimizer(model, args.lr, args.weight_decay, device)

    best_val = float("inf")
    best_score = None
    stale_evals = 0
    saved_any = False
    start_step = 1
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        best_val, best_score = state["best_val"], state["best_score"]
        stale_evals, saved_any = state["stale_evals"], state["saved_any"]
        start_step = state["step"] + 1
        # map_location moved these onto the GPU with everything else, but both
        # setters take a CPU ByteTensor - a CUDA resume failed here, a CPU one never did.
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())

    def save_checkpoint():
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        }, checkpoint_path)

    def save_resume(step_done):
        """Everything needed to carry on from step_done as if never stopped. Written
        to a temporary name and then renamed, so a crash mid-write - a GPU driver
        reset, say - leaves the previous resume point intact rather than half a file."""
        temp = resume_path + ".tmp"
        torch.save({
            "step": step_done, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config, "stoi": stoi, "itos": itos,
            "best_val": best_val, "best_score": best_score,
            "stale_evals": stale_evals, "saved_any": saved_any,
            "schedule": {key: getattr(args, key) for key in SCHEDULE},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }, temp)
        os.replace(temp, resume_path)

    # Ctrl+C asks for a pause rather than killing the run: it finishes the step in
    # hand, saves, and exits. A background run has no Ctrl+C, so a file does the same.
    pause_requested = []
    signal.signal(signal.SIGINT, lambda *_: pause_requested.append(True))

    def eta(step):
        per_step = (time.perf_counter() - clock_start) / max(1, step - start_step + 1)
        left = int(per_step * (args.steps - step))
        return f"{left // 3600}h{left % 3600 // 60:02d}m" if left >= 3600 else f"{left // 60}m{left % 60:02d}s"

    clock_start = time.perf_counter()
    for step in range(start_step, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.min_lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, config.block_size,
                                   args.batch_size, device, autocast=autocast)
            gap = losses["val"] - losses["train"]
            score = (-losses[args.select_by],)
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
                best_val = min(best_val, losses[args.select_by])
                stale_evals = 0
                save_checkpoint()
                saved_any = True
            else:
                stale_evals += 1

            flag = " *best*" if improved else f"  (no gain x{stale_evals})"
            say(f"step {step}/{args.steps}: train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| gap {gap:+.4f} | lr {lr:.2e} | eta {eta(step)}{flag}")
            save_resume(step)

            if args.patience and stale_evals >= args.patience:
                say(f"Early stop: validation loss has not improved in {args.patience} "
                      "evals. The best weights are already saved.")
                break

        # Last in the step, after any eval: an eval draws batches from the same
        # generator, so pausing before one would change every batch after it.
        if step < args.steps and (pause_requested
                                  or (step % 50 == 0 and os.path.exists(pause_path))):
            save_resume(step)
            if os.path.exists(pause_path):
                os.remove(pause_path)
            say(f"Paused at step {step}/{args.steps}. Continue with the same command "
                f"plus --resume.")
            return

    if not saved_any:
        save_checkpoint()
    else:
        # Reload the best weights so the export matches the saved checkpoint.
        best = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"])
        say(f"Best {args.select_by} loss: {best_val:.4f}")

    say(f"Saved full checkpoint to {checkpoint_path}")

    if args.compressed_out:
        compressed_path = os.path.join(args.out_dir, args.compressed_out)
    else:
        base = args.name or args.preset
        compressed_path = versions.next_path(base, args.models_dir)
    export_compressed(model, config, stoi, itos, compressed_path, bits=args.bits,
                      group_size=args.group_size)
    full_size = os.path.getsize(checkpoint_path)
    compressed_size = os.path.getsize(compressed_path)
    say(f"Exported compressed neural data to {compressed_path} ({args.bits}-bit, 3 lines)")
    say(f"Full checkpoint: {full_size:,} bytes | Compressed export: {compressed_size:,} bytes "
          f"({100 * compressed_size / full_size:.1f}% of full, "
          f"{8 * compressed_size / n_params:.2f} bits/param on disk)")
    # Finished and exported, so there is nothing left to resume.
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
