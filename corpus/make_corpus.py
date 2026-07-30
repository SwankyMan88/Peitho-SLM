"""Build training.txt and heldout.txt from hand-written conversations.

Source:
  conversations.txt  multi-turn conversations, blank line between them

It is hand-written, which is what makes the corpus worth reading. Worked
arithmetic is generated on top by arith.py, freshly each time rather than
repeated, so what the model sees is thousands of different sums being worked
through rather than a few hundred sums to memorize.

There are no slot-filling drills here. An earlier design generated conversations
that stated a typed fact ("my favourite colour is X") and asked for it back, which
taught the model to answer every statement with "noted, I will remember that"
whether or not remembering had been asked for. Recall is handled by the page
instead, so the corpus is free to be conversation.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK

sys.path.insert(0, paths.CORPUS)
import arith
import compose
import talk
import thinking


def load_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [block.strip().split("\n") for block in text.split("\n\n") if block.strip()]


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(USER_MARK) and lines[i + 1].startswith(BOT_MARK):
            pairs.append([lines[i], lines[i + 1]])
    return pairs


def thinking_conversation(rng, thinking_on=True):
    """Turns that work something out, then say the short answer - see thinking.py.

    Both forms of every turn come from one draw, so the two corpora are comparable:
    same sentences, same sums, same wording, differing only in whether the working
    is thought or spoken.

    The working goes between the bot marker and the think marker, so an interface
    can fold it away. With thinking off, whichever half actually answers the
    question becomes the whole reply.

    Returns (what to write, the other form) so the caller can size the corpus by the
    thinking form whichever one it is writing."""
    lines, plain = [], []
    for prompt, thought, reply, without in thinking.conversation(rng):
        asked = USER_MARK + prompt + END_MARK
        lines += [asked, BOT_MARK + thought + THINK_MARK + reply + END_MARK]
        plain += [asked, BOT_MARK + without + END_MARK]
    return (lines, plain) if thinking_on else (plain, lines)


def math_conversation(rng):
    """Sums worked through rather than declined - see arith.py."""
    lines = []
    for prompt, reply in arith.conversation(rng):
        lines += [USER_MARK + prompt + END_MARK, BOT_MARK + reply + END_MARK]
    return lines


# With only a few hundred hand-written blocks, repeating them verbatim to fill a
# corpus teaches the exact strings and nothing about how a question might be
# worded. These reframe the same content: the meaning is hand-written, the
# packaging varies.
OPENERS = ["", "", "", "", "So ", "Hey, ", "Quick question, ", "I was wondering, ",
           "Can I ask, ", "Alright, ", "Okay so ", "One more thing, "]
LEAD_INS = ["", "", "", "", "Good question. ", "Short answer: ", "Honestly, ",
            "The gist is this. ", "Roughly speaking, ", "Here is the shape of it. "]
CLOSERS = ["", "", "", "", "", " Does that help?", " Was that what you meant?",
           " Ask me if that was too brief.", " There is more if you want it.",
           " That is the short version."]


# Each answer is hand-written once, but a question can be asked many ways. Without
# these the model maps one exact wording to one answer and picks badly when the
# wording shifts, which shows up as replying about the wrong topic entirely.
# Only substitutions that leave the rest of the sentence grammatical. "How come"
# and "Any idea why" work after a dropped auxiliary ("why do cats purr" -> "how
# come cats purr") but not after a dropped copula, where the verb would have to
# move ("why is the sky blue" -> "how come the sky IS blue"), so those are left out.
QUESTION_FORMS = [
    ("Why is ", ["What makes ", "Why exactly is "]),
    ("Why are ", ["What makes ", "Why exactly are "]),
    ("Why do ", ["How come ", "Any idea why ", "Why exactly do ", "What makes "]),
    ("Why does ", ["How come ", "Any idea why ", "Why exactly does ", "What makes "]),
    ("Why can ", ["How come ", "Why exactly can "]),
    ("What is ", ["Tell me about ", "Explain ", "Can you explain ", "What exactly is "]),
    ("What are ", ["Tell me about ", "Explain ", "Can you explain ", "What exactly are "]),
    ("How does ", ["How exactly does "]),
    ("How do ", ["How exactly do "]),
    ("Tell me about ", ["What do you know about ", "Explain ", "Can you explain "]),
]


# "What are stars made of?" cannot become "Tell me about stars made of." - the
# remainder is not a noun phrase. A trailing preposition is the reliable signal.
DANGLING = ("of", "in", "on", "for", "to", "from", "with", "about", "like", "at",
            "made", "used", "called", "do", "does", "is", "are", "work", "works")


def paraphrase_question(rng, line):
    """Ask the same question a different way, when the opening allows it cleanly."""
    body = line[1:-1]
    for prefix, alternatives in QUESTION_FORMS:
        if not body.startswith(prefix):
            continue
        rest = body[len(prefix):]
        if not rest:
            return line
        tail = rest.rstrip("?.!").split()
        if tail and tail[-1].lower() in DANGLING:
            keep = [a for a in alternatives if a.startswith(("What exactly", "Why exactly",
                                                             "How exactly", "How come",
                                                             "Any idea"))]
            if not keep:
                return line
            alternatives = keep
        alt = rng.choice(alternatives)
        # "Tell me about X" and "Explain X" are statements, not questions.
        if alt.rstrip().endswith(("about", "Explain", "understood")) or alt.startswith("Tell"):
            rest = rest.rstrip("?") + "."
        return USER_MARK + alt + rest + END_MARK
    return line


def reframe(rng, block):
    """Vary the packaging of a hand-written block without touching its meaning."""
    lines = list(block)
    if rng.random() < 0.55:
        lines[0] = paraphrase_question(rng, lines[0])
    if rng.random() < 0.35:
        lines[0] = USER_MARK + rng.choice(OPENERS) + lines[0][1:-1] + END_MARK
    for i, line in enumerate(lines):
        if not line.startswith(BOT_MARK):
            continue
        body = line[1:-1]
        if rng.random() < 0.3:
            lead = rng.choice(LEAD_INS)
            if lead:
                body = lead + body[0].lower() + body[1:] if lead.endswith(" ") and lead[-2] in ":," \
                    else lead + body
        if rng.random() < 0.2:
            body += rng.choice(CLOSERS)
        lines[i] = BOT_MARK + body + END_MARK
    if rng.random() < 0.18:
        u, b = compose.social_exchange(rng)
        lines = [USER_MARK + u + END_MARK, BOT_MARK + b + END_MARK] + lines
    if rng.random() < 0.18:
        u, b = compose.social_exchange(rng)
        lines = lines + [USER_MARK + u + END_MARK, BOT_MARK + b + END_MARK]
    return lines


def composed_conversation(rng):
    """A freshly generated multi-turn conversation - see talk.py.

    This is what teaches the model to build a sentence rather than recall one.
    Every one is effectively unique, so there is nothing here to memorize, and the
    subject persists across turns so continuing means reading the context."""
    lines = []
    for prompt, reply, _ in talk.dialogue(rng):
        lines += [USER_MARK + prompt + END_MARK, BOT_MARK + reply + END_MARK]
    return lines


def vary_user(rng, line):
    """Roughen a user turn the way people actually type.

    Without this every user turn is one of a fixed set of exact strings, and a
    missing full stop or a lowercase start lands off-distribution. Only user
    turns are varied; the model's own replies stay clean, since those are what it
    should imitate."""
    body = line[1:-1]
    if rng.random() < 0.3 and body.endswith("."):
        body = body[:-1]
    if rng.random() < 0.15:
        body = body[0].lower() + body[1:]
    if rng.random() < 0.08 and body.endswith("."):
        body = body[:-1] + "!"
    if rng.random() < 0.06 and body.endswith("?"):
        body += "?"
    return USER_MARK + body + END_MARK


def render(rng, blocks, target_chars, composed_share=0.0, math_share=0.0,
           think_share=0.0, thinking_on=True):
    """Fill to the target size, mixing repeated hand-written blocks with freshly
    generated ones.

    The generated share is new every time rather than repeated, so raising it adds
    unique text instead of more repetition. That is the difference between a model
    that composes a sentence and one that recalls it, and between one that works a
    sum out and one that has the answer to that particular sum by heart."""
    parts = []
    total = 0
    while total < target_chars:
        order = list(blocks)
        rng.shuffle(order)
        for block in order:
            measured = 0
            if composed_share and rng.random() < composed_share:
                # The thinking share comes out of the generated stream, so raising
                # it trades working-shown-in-the-reply for working-then-answer.
                draw = rng.random()
                if draw < think_share:
                    # Whichever form is being written, the corpus is full when the
                    # thinking form would have filled it. Measuring the stripped
                    # form instead would fit more conversations into the same bytes,
                    # and the two corpora would stop being comparable.
                    lines, other = thinking_conversation(rng, thinking_on)
                    measured = len(START_MARK + "\n" + "\n".join(other))
                elif draw < think_share + math_share:
                    lines = math_conversation(rng)
                else:
                    lines = composed_conversation(rng)
            else:
                lines = reframe(rng, block)
            lines = [vary_user(rng, l) if l.startswith(USER_MARK) else l for l in lines]
            chunk = START_MARK + "\n" + "\n".join(lines)
            parts.append(chunk)
            total += max(len(chunk), measured) + 1
            if total >= target_chars:
                break
    return "\n".join(parts) + "\n"


def main():
    p = argparse.ArgumentParser(description="Generate the conversation corpus.")
    p.add_argument("--conversations", default=paths.CONVERSATIONS)
    p.add_argument("--pairs", default="",
                   help="Optional file of single-turn pairs. Left empty by default: "
                        "short fixed exchanges repeated to fill a corpus are what the "
                        "model recites back word for word.")
    p.add_argument("--train_out", default=paths.TRAINING)
    p.add_argument("--heldout_out", default=paths.HELDOUT)
    p.add_argument("--target_chars", type=int, default=2_000_000)
    p.add_argument("--heldout_frac", type=float, default=0.10,
                   help="Fraction of blocks withheld from training entirely.")
    p.add_argument("--composed", type=float, default=0.85,
                   help="Share of the corpus generated fresh by compose.py. This is the "
                        "part that is not repeated, so it is what teaches composing "
                        "rather than reciting.")
    p.add_argument("--no_thinking", action="store_true",
                   help="Write the same corpus without the thinking phase: whichever "
                        "half of each turn answers the question becomes the whole "
                        "reply. Same sentences, so the two are comparable.")
    p.add_argument("--think", type=float, default=0.0,
                   help="Share of the generated stream where the model works the answer "
                        "out first and marks the end of its working. Off by default.")
    p.add_argument("--math", type=float, default=0.28,
                   help="Share of the generated stream that is worked arithmetic. "
                        "Generated fresh, so every sum is a different one.")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    paths.ensure_build()
    rng = random.Random(args.seed)
    conversations = load_conversations(args.conversations)
    pairs = load_pairs(args.pairs) if args.pairs else []
    blocks = conversations + pairs
    rng.shuffle(blocks)

    n_held = max(1, int(len(blocks) * args.heldout_frac))
    held, train_blocks = blocks[:n_held], blocks[n_held:]

    unique = sum(len("\n".join(b)) for b in train_blocks)
    thinking_on = not args.no_thinking
    train = render(rng, train_blocks, args.target_chars, args.composed, args.math,
                   args.think, thinking_on)
    heldout = render(random.Random(args.seed + 1), held,
                     max(50_000, int(args.target_chars * 0.06)), args.composed, args.math,
                     args.think, thinking_on)

    for path, text, label in ((args.train_out, train, "train"),
                              (args.heldout_out, heldout, "heldout")):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"{label:8} -> {paths.short(path)}: {len(text):,} chars, "
              f"{text.count(START_MARK):,} conversations, {len(set(text))} distinct characters")

    print(f"\n{len(conversations)} hand-written conversations, {len(pairs)} pairs")
    repeated_chars = int(args.target_chars * (1 - args.composed))
    passes = repeated_chars // max(1, unique)
    print(f"hand-written text: {unique:,} unique chars filling {repeated_chars:,} "
          f"chars of corpus, so ~{passes} passes over it")
    print(f"generated text: {args.composed:.0%} of the corpus, effectively all unique, "
          f"of which {args.math:.0%} is worked arithmetic and {args.think:.0%} "
          f"{'thinks before answering' if thinking_on else 'would think, stripped away'}")
    if passes > 60:
        print("  WARNING: the hand-written part is repeating heavily. Add conversations, "
              "or raise --composed so more of the corpus is fresh.")
    print(f"{len(held)} blocks held out of training entirely, for honest validation.")

    following = {}
    lines = train.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.startswith(USER_MARK) and lines[i + 1].startswith(BOT_MARK):
            counts = following.setdefault(line, {})
            counts[lines[i + 1]] = counts.get(lines[i + 1], 0) + 1
    stuck = []
    for turn, counts in following.items():
        total = sum(counts.values())
        top = max(counts.values())
        if total >= 8 and top / total > 0.5:
            stuck.append((total, top / total, turn[1:-1]))
    if stuck:
        print()
        print("  These user turns have one dominant reply, so the model will recite it:")
        for total, share, turn in sorted(stuck, reverse=True)[:8]:
            print(f"    {total:5} occurrences, {share:.0%} identical  {turn[:52]}")
    else:
        print()
        print("No user turn has a dominant reply; every prompt has several answers.")


if __name__ == "__main__":
    main()
