"""Turns where the model works something out before it answers.

A bot turn normally holds the whole reply:

    ◀Tens: 20 + 10 = 30. Ones: 4 + 5 = 9. Add those up: 39.■

A thinking turn puts the working first, ends it with the think marker, and then
says the short thing a person actually wants:

    ◀Tens: 20 + 10 = 30. Ones: 4 + 5 = 9. Add those up: 39.◇That comes to 39.■

Nothing about the model changes. The working is still generated one character at a
time and still conditions what comes after it - the marker only tells a reader
where the workings stop, so an interface can fold them away. Whether that helps or
merely looks tidy is a question for the benchmark, not for this file.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arith
import talk

# Said after the working, so the reply is short and the number is the point.
ANSWERS = [
    "That comes to {n}.", "{n}.", "It is {n}.", "I get {n}.", "So {n}.",
    "{n}, unless I dropped something.", "That gives {n}.", "The answer is {n}.",
    "{n} is what I get.", "Comes out at {n}.",
]

# For a remainder, where the answer is two numbers rather than one.
REMAINDERS = [
    "{q} with {r} left over.", "{q}, remainder {r}.", "{q} and {r} spare.",
    "It goes {q} times with {r} left.",
]

def answer_text(rng, op, lhs, rhs):
    """The short reply, once the working has been done."""
    if op == "/" and lhs % rhs:
        return rng.choice(REMAINDERS).format(q=lhs // rhs, r=lhs % rhs)
    total = (lhs + rhs if op == "+" else lhs - rhs if op == "-"
             else lhs * rhs if op == "*" else lhs // rhs)
    return rng.choice(ANSWERS).format(n=total)


def sum_turn(rng, others=0.3):
    """An arithmetic turn: sometimes a plain sum, sometimes one of the other kinds.

    Percentages, squares, three-term addition, rounding and word problems all come
    from arith.other(). With thinking off the working is the reply, exactly as for a
    plain sum, because it is the working that ends on the number."""
    if rng.random() < others:
        question, working, answer = arith.other(rng)
        return question, working, answer, working
    return plain_sum_turn(rng)


def plain_sum_turn(rng):
    """(user asks a sum, working, short answer, the same turn without thinking)

    With thinking off, the working *is* the reply - it already ends on the number,
    which is what the old corpus did and why arithmetic worked at all. Keeping the
    short answer instead would leave a bare total with nothing to copy it from."""
    while True:
        op = rng.choices(["+", "-", "*", "/"], weights=[35, 30, 25, 10])[0]
        lhs, rhs = arith.operands(rng, op)
        worked = [w for w in (m(rng, lhs, rhs) for m in arith.METHODS[op]) if w]
        if worked:
            break
    shown = f"{lhs} {op} {rhs}" if rng.random() < 0.5 else f"{lhs}{op}{rhs}"
    question = rng.choice(arith.WORDING).format(p=shown)
    working = rng.choice(worked)
    return question, working, answer_text(rng, op, lhs, rhs), working


def _sum(rng):
    """(question, answer, working) - the shape talk.conversation wants."""
    question, working, answer, _ = sum_turn(rng)
    return question, answer, working


def conversation(rng, sums=0.35):
    """[(user, thought, reply, plain), ...] for one whole conversation.

    Length and threading come from talk.conversation, which knows how to keep a
    subject going; arithmetic is handed to it as a callback so that file stays about
    conversation. Averages around nine exchanges, from two to twenty."""
    return [(ask, thought, reply, plain)
            for ask, reply, thought, plain in
            talk.conversation(rng, sum_turn=_sum, sums=sums)]


if __name__ == "__main__":
    rng = random.Random(5)
    for _ in range(6):
        for user, thought, reply, plain in conversation(rng):
            print(f"  you:            {user}")
            print(f"  thinking:       {thought}")
            print(f"  peitho:         {reply}")
            print(f"  without think:  {plain}")
        print()
