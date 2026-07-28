import argparse
import json
import re
import os
import sys
from datetime import datetime, timezone

import torch

# The turn markers are non-ASCII, and the default Windows console codepage cannot
# encode them. Without this, printing a marker raises UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import GPT, START_MARK, USER_MARK, BOT_MARK, END_MARK, encode, decode
from export import import_compressed
import versions

HISTORY_PATH = "chat_history.json"
HISTORY_TURNS_IN_PROMPT = 4


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def build_prompt(history, user_text, block_size):
    """Render the recent conversation in the same marker format used for training,
    then open a bot turn so the model continues it.

    Only block_size characters fit in the context, so older turns are dropped
    whole rather than sliced mid-word - a prompt that starts partway through a
    word is not something the model ever saw during training. START_MARK is
    included only when nothing had to be dropped, since it means the opening of
    the conversation is genuinely in view."""
    current = f"{USER_MARK}{user_text}{END_MARK}\n{BOT_MARK}"

    kept = []
    dropped = len(history) > HISTORY_TURNS_IN_PROMPT
    for turn in reversed(history[-HISTORY_TURNS_IN_PROMPT:]):
        mark = USER_MARK if turn["role"] == "user" else BOT_MARK
        line = f"{mark}{turn['text']}{END_MARK}"
        if len("\n".join([line] + kept + [current])) > block_size:
            dropped = True
            break
        kept.insert(0, line)

    head = [] if dropped else [START_MARK]
    prompt = "\n".join(head + kept + [current])
    return prompt[-block_size:]  # last resort, if the new message alone overflows


def clean_response(text):
    """Cut at the end marker, or at any marker/newline if the model rambled past it."""
    for stop in (END_MARK, USER_MARK, BOT_MARK, START_MARK, "\n"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return trim_loop(text).strip()


def trim_loop(text):
    """Cut a reply that has fallen into a repeating cycle.

    A model that loses the thread repeats a short unit until it runs out of room.
    Truncating at the third repeat beats printing the whole wall of it."""
    for i in range(len(text)):
        for n in range(2, 15):
            unit = text[i:i + n]
            if len(unit.strip()) > 1 and text[i:].startswith(unit * 3):
                return text[:i + n]
    # A stuttered word ("helps helps", "to to") is the same failure earlier on.
    stutter = re.search(r"\b(\w{1,12})\s+\1\b", text)
    if stutter:
        return text[:stutter.start() + len(stutter.group(1))]
    return text


def load_full_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"], ckpt["stoi"], ckpt["itos"]


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained SLM from the terminal.")
    parser.add_argument("--checkpoint", default="model_full.pt")
    parser.add_argument("--compressed", action="store_true",
                        help="Load from the compressed export instead of the full checkpoint.")
    parser.add_argument("--compressed_path", default="")
    parser.add_argument("--max_new_tokens", type=int, default=400,
                        help="Safety net only; generation normally stops at the end marker.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--show_prompt", action="store_true",
                        help="Print the marker-formatted prompt actually fed to the model.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.compressed:
        path = versions.resolve(args.compressed_path)
        print(f"Loading compressed model from {path}...")
        model, config, stoi, itos = import_compressed(path, device=device)
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}. Run train.py first.")
        print(f"Loading full checkpoint from {args.checkpoint}...")
        model, config, stoi, itos = load_full_checkpoint(args.checkpoint, device)

    model.eval()
    stop_id = stoi.get(END_MARK)
    if stop_id is None:
        print("  WARNING: this model's vocab has no end marker, so replies will run to the "
              "length limit. Retrain with --fresh to pick up the marker format.")

    history = load_history()
    if history:
        print(f"Loaded {len(history)} past turns from {HISTORY_PATH}.")
    print("Chatting with your SLM. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break

        prompt = build_prompt(history, user_text, config.block_size)
        unknown = sorted(set(prompt) - set(stoi))
        if unknown:
            print(f"  (dropping characters not seen during training: {''.join(unknown)})")
            prompt = "".join(ch for ch in prompt if ch in stoi)

        if args.show_prompt:
            print(f"--- prompt ({len(prompt)}/{config.block_size} chars) ---\n"
                  f"{prompt}\n--- end prompt ---")

        ids = encode(prompt, stoi)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             stop_id=stop_id)
        response = clean_response(decode(out[0, len(ids):].tolist(), itos))
        if not response:
            response = "(no reply)"

        print(f"Bot: {response}\n")

        now = datetime.now(timezone.utc).isoformat()
        history.append({"role": "user", "text": user_text, "timestamp": now})
        history.append({"role": "model", "text": response, "timestamp": now})
        save_history(history)


if __name__ == "__main__":
    main()
