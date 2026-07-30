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
import compose
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

# A plan, not a restatement: what the reply is going to do.
PLANS = [
    "They want {topic}. Keep it short.",
    "This is about {topic}. One clear sentence, then stop.",
    "{topic}. Say the useful part first.",
    "Question is {topic}. Answer plainly, no hedging.",
    "They asked about {topic}. Give the shape of it, not the detail.",
    "{topic} - say what I actually know and leave the rest.",
    "About {topic}. Short answer, then offer more if they want it.",
    "{topic}. Do not pad this out.",
]

HONEST = [
    "I am not sure about this one. Say so.",
    "This is outside what I know. Better to admit it.",
    "I could invent something here. Do not.",
    "No idea, really. Be straight about that.",
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


def topic_of(rng, prompt):
    """Whatever the prompt is about, as the thought would put it."""
    words = [w.strip("?.!,").lower() for w in prompt.split()]
    for noun in words:
        if noun in compose.NOUNS:
            return "the " + noun
    return rng.choice(["this", "that", "what they said", "the question"])


def plan(rng, prompt):
    if rng.random() < 0.12:
        return rng.choice(HONEST)
    said = rng.choice(PLANS).format(topic=topic_of(rng, prompt))
    return said[0].upper() + said[1:]


def chat_turn(rng):
    """(user says something, a plan, the reply, the same turn without thinking)

    Here it is the other way round: the plan is not an answer, so with thinking off
    the reply stands alone and the plan is dropped. Which half survives depends on
    which half answers the question - that is the whole rule."""
    prompt, reply, thought = talk.dialogue(rng)[0]
    return prompt, thought, reply, reply


def conversation(rng, sums=0.7):
    """[(user, thought, reply, plain), ...] - mostly sums, which is what working suits."""
    turns = []
    for _ in range(rng.choices([1, 2, 3], weights=[55, 32, 13])[0]):
        turns.append(sum_turn(rng) if rng.random() < sums else chat_turn(rng))
    return turns


if __name__ == "__main__":
    rng = random.Random(5)
    for _ in range(6):
        for user, thought, reply, plain in conversation(rng):
            print(f"  you:            {user}")
            print(f"  thinking:       {thought}")
            print(f"  peitho:         {reply}")
            print(f"  without think:  {plain}")
        print()
