"""Worked arithmetic for the corpus.

Every problem is solved correctly here, and shown as a method rather than a bare
answer: place value, carrying, counting on, rounding and adjusting, doubling,
chunking. A character model cannot look up a sum, so the only thing it can learn
from an answer alone is which digits tend to follow "7 + 5 =". A method is at
least the shape of something it can generalize, and the working keeps the answer
in its own context where it can be read back off.

Each problem has several methods that reach the same number, so no wording maps
to one fixed reply, and a follow-up can rework the same problem a different way.
"""

import random

PLACES = ["ones", "tens", "hundreds", "thousands"]


def digits(n):
    """Digits of n, least significant first."""
    return [int(c) for c in str(abs(n))][::-1]


def places(n, width):
    """[8, 40, 300] style breakdown, padded with zeros to `width` places."""
    d = digits(n)
    d += [0] * (width - len(d))
    return [d[i] * 10 ** i for i in range(width)]


def width_of(*numbers):
    return max(len(digits(n)) for n in numbers)


def join(items):
    """1, 2 and 3"""
    items = [str(i) for i in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- addition

def add_place_value(rng, lhs, rhs):
    w = width_of(lhs, rhs)
    if w < 2:
        return None
    a, b = places(lhs, w), places(rhs, w)
    lines, partials = [], []
    for i in reversed(range(w)):
        if not a[i] and not b[i]:
            continue
        lines.append(f"{PLACES[i].capitalize()}: {a[i]} + {b[i]} = {a[i] + b[i]}")
        partials.append(a[i] + b[i])
    return (". ".join(lines) + f". Add those up: {' + '.join(str(p) for p in partials)}"
            f" = {lhs + rhs}.")


def add_carry(rng, lhs, rhs):
    w = width_of(lhs, rhs)
    if w < 2:
        return None
    a, b = digits(lhs) + [0] * w, digits(rhs) + [0] * w
    carry, lines = 0, []
    for i in range(w):
        total = a[i] + b[i] + carry
        line = f"{PLACES[i].capitalize()}: {a[i]} + {b[i]}"
        if carry:
            line += f" plus the {carry} carried"
        line += f" = {total}"
        if total >= 10:
            line += f", so write {total % 10} and carry {total // 10}"
        lines.append(line)
        carry = total // 10
    if carry:
        lines.append(f"The {carry} carried at the end becomes the next digit along")
    if not any("carry" in l for l in lines):
        return None
    return ". ".join(lines) + f". Reading it back: {lhs + rhs}."


def add_round(rng, lhs, rhs):
    up = (10 - rhs % 10) % 10
    if not up or rhs < 10:
        return None
    rounded = rhs + up
    return (f"{rhs} is {up} short of {rounded}. {lhs} + {rounded} = {lhs + rounded}, "
            f"then give back the {up}: {lhs + rhs}.")


def add_hop(rng, lhs, rhs):
    tens, ones = rhs - rhs % 10, rhs % 10
    if not tens or not ones:
        return None
    return (f"Start at {lhs}, add {tens} to get {lhs + tens}, then {ones} more. "
            f"That is {lhs + rhs}.")


def add_small(rng, lhs, rhs):
    if lhs > 9 or rhs > 9:
        return None
    return rng.choice([
        f"Counting {rhs} on from {lhs} lands on {lhs + rhs}.",
        f"{lhs} and {rhs} make {lhs + rhs}.",
        f"That is one of the ones I know outright: {lhs + rhs}.",
    ])


# ---------------------------------------------------------------- subtraction

def sub_count_up(rng, lhs, rhs):
    if rhs > lhs:
        return None
    steps, at = [], rhs
    to_ten = (10 - at % 10) % 10
    if to_ten and at + to_ten <= lhs:
        steps.append(to_ten)
        at += to_ten
    floor = lhs - lhs % 10
    if floor > at:
        steps.append(floor - at)
        at = floor
    if lhs > at:
        steps.append(lhs - at)
    if len(steps) < 2:
        return None
    running = rhs
    described = []
    for step in steps:
        described.append(f"{running} up to {running + step} is {step}")
        running += step
    return (f"Count up from {rhs}. " + ", ".join(described) +
            f". {' + '.join(str(s) for s in steps)} = {lhs - rhs}.")


def sub_in_parts(rng, lhs, rhs):
    if rhs > lhs:
        return None
    tens, ones = rhs - rhs % 10, rhs % 10
    if not tens or not ones:
        return None
    return (f"Take the tens off first: {lhs} - {tens} = {lhs - tens}. "
            f"Then the {ones}: {lhs - rhs}.")


def sub_borrow(rng, lhs, rhs):
    if rhs > lhs or lhs < 10 or rhs < 10:
        return None
    a, b = digits(lhs), digits(rhs) + [0] * len(digits(lhs))
    if a[0] >= b[0]:
        return None
    return (f"The ones will not go: {a[0]} - {b[0]} is short, so borrow ten from the tens. "
            f"{a[0] + 10} - {b[0]} = {a[0] + 10 - b[0]}. "
            f"Then the tens, one fewer than before. That leaves {lhs - rhs}.")


def sub_round(rng, lhs, rhs):
    if rhs > lhs:
        return None
    up = (10 - rhs % 10) % 10
    if not up or rhs < 10:
        return None
    return (f"Round {rhs} up to {rhs + up}. {lhs} - {rhs + up} = {lhs - rhs - up}, "
            f"and I took away {up} too many, so add it back: {lhs - rhs}.")


def sub_negative(rng, lhs, rhs):
    if rhs <= lhs:
        return None
    return (f"That goes below zero, so I will do it the other way round. "
            f"{rhs} - {lhs} = {rhs - lhs}, which makes the answer {lhs - rhs}.")


def sub_small(rng, lhs, rhs):
    if lhs > 9 or rhs > 9 or rhs > lhs:
        return None
    return rng.choice([
        f"Counting back {rhs} from {lhs} gives {lhs - rhs}.",
        f"{rhs} away from {lhs} leaves {lhs - rhs}.",
        f"{lhs} - {rhs} = {lhs - rhs}.",
    ])


# ---------------------------------------------------------------- multiplication

def mul_distribute(rng, lhs, rhs):
    w = width_of(rhs)
    if w < 2:
        return None
    parts = [p for p in reversed(places(rhs, w)) if p]
    pieces = [f"{lhs} * {p} = {lhs * p}" for p in parts]
    return (f"Split the {rhs}: " + ", ".join(pieces) +
            f". Add those: {' + '.join(str(lhs * p) for p in parts)} = {lhs * rhs}.")


def mul_double(rng, lhs, rhs):
    if rhs not in (2, 4, 8):
        return None
    steps, value = [], lhs
    for _ in range(rhs.bit_length() - 1):
        value *= 2
        steps.append(value)
    if rhs == 2:
        return f"Doubling {lhs}: {lhs} + {lhs} = {value}."
    return (f"{rhs} is doubling {rhs.bit_length() - 1} times. "
            f"{lhs} becomes " + join(steps) + ".")


def mul_near_ten(rng, lhs, rhs):
    if rhs % 10 != 9 or rhs < 9:
        return None
    up = rhs + 1
    return (f"{rhs} is one less than {up}. {lhs} * {up} = {lhs * up}, "
            f"then take off one {lhs}: {lhs * rhs}.")


def mul_repeated(rng, lhs, rhs):
    if rhs > 4 or lhs > 20:
        return None
    running, terms = 0, []
    for _ in range(rhs):
        running += lhs
        terms.append(running)
    return f"{rhs} lots of {lhs}, counted up: " + join(terms) + "."


def mul_small(rng, lhs, rhs):
    if lhs > 9 or rhs > 9:
        return None
    return rng.choice([
        f"{lhs} * {rhs} = {lhs * rhs}.",
        f"That is a times table one: {lhs * rhs}.",
        f"{rhs} lots of {lhs} is {lhs * rhs}.",
    ])


# ---------------------------------------------------------------- division

def div_how_many(rng, lhs, rhs):
    if lhs % rhs:
        return None
    q = lhs // rhs
    return f"How many {rhs}s fit into {lhs}? {rhs} * {q} = {lhs}, so {q}."


def div_chunk(rng, lhs, rhs):
    q = lhs // rhs
    if q < 10:
        return None
    w = width_of(q)
    parts = [p for p in reversed(places(q, w)) if p]
    if len(parts) < 2:
        return None
    lines, left = [], lhs
    for p in parts:
        lines.append(f"{rhs} * {p} = {rhs * p}, leaving {left - rhs * p}")
        left -= rhs * p
    tail = f". Together that is {' + '.join(str(p) for p in parts)} = {q}"
    if left:
        tail += f", with {left} left over"
    return f"Take it in chunks. " + ". ".join(lines) + tail + "."


def div_remainder(rng, lhs, rhs):
    q, r = divmod(lhs, rhs)
    if not r:
        return None
    if not q:
        return f"{rhs} does not fit into {lhs} even once, so none of it, with {lhs} left over."
    return (f"{rhs} * {q} = {rhs * q}, and {lhs} - {rhs * q} = {r}. "
            f"So {q} with {r} left over.")


def div_halve(rng, lhs, rhs):
    if rhs not in (2, 4) or lhs % rhs:
        return None
    if rhs == 2:
        return f"Half of {lhs} is {lhs // 2}."
    return f"Halve {lhs} to get {lhs // 2}, then halve again: {lhs // 4}."


def div_small(rng, lhs, rhs):
    if lhs > 81 or lhs % rhs:
        return None
    return rng.choice([
        f"{lhs} / {rhs} = {lhs // rhs}.",
        f"{rhs} goes into {lhs} exactly {lhs // rhs} times.",
    ])


METHODS = {
    "+": [add_place_value, add_carry, add_round, add_hop, add_small],
    "-": [sub_count_up, sub_in_parts, sub_borrow, sub_round, sub_negative, sub_small],
    "*": [mul_distribute, mul_double, mul_near_ten, mul_repeated, mul_small],
    "/": [div_how_many, div_chunk, div_remainder, div_halve, div_small],
}

OPENERS = ["", "", "", "Let me work it out. ", "Sure. ", "Working it through. ",
           "One moment. ", "I can try that. ", "Right. "]
CLOSERS = ["", "", "", "", " I think that is right.", " That is my working, anyway.",
           " Check me on it if it matters."]
WORDING = [
    "What is {p}?", "{p}", "{p} = ?", "Work out {p}.", "Can you do {p}?",
    "{p} please", "How much is {p}?", "Solve {p}.", "Whats {p}?",
    "Could you work out {p} for me?", "I need {p}.",
]
AGAIN = ["How did you get that?", "Show me another way.", "Can you explain that?",
         "Walk me through it.", "Are you sure?", "How did you work it out?",
         "Another way?", "Explain that one."]
AGAIN_LEAD = ["Another way to see it. ", "Same answer, different route. ",
              "Here it is again. ", "Sure. ", "Like this. ", ""]


def operands(rng, op):
    """Small numbers mostly, because the working has to fit in the context."""
    size = rng.choices([1, 2, 3], weights=[35, 45, 20])[0]
    lo, hi = max(1, 10 ** (size - 1)), 10 ** size - 1
    lhs, rhs = rng.randint(lo, hi), rng.randint(lo, hi)
    if op == "-" and rng.random() < 0.85:
        lhs, rhs = max(lhs, rhs), min(lhs, rhs)
    if op == "*":
        rhs = rng.randint(2, 12) if rng.random() < 0.7 else rng.randint(2, 25)
    if op == "/":
        rhs = rng.randint(2, 12)
        if rng.random() < 0.7:
            lhs = rhs * rng.randint(2, 30)
    return lhs, rhs


def problem(rng):
    """(shown problem, [worked explanations]) - every explanation gives the same answer."""
    op = rng.choices(["+", "-", "*", "/"], weights=[35, 30, 25, 10])[0]
    lhs, rhs = operands(rng, op)
    worked = [w for w in (m(rng, lhs, rhs) for m in METHODS[op]) if w]
    if not worked:
        return None
    rng.shuffle(worked)
    shown = f"{lhs} {op} {rhs}" if rng.random() < 0.5 else f"{lhs}{op}{rhs}"
    return shown, worked


def exchange(rng):
    """[(user, bot), ...] for one arithmetic problem, sometimes with a follow-up."""
    made = problem(rng)
    while not made:
        made = problem(rng)
    shown, worked = made
    turns = [(rng.choice(WORDING).format(p=shown),
              rng.choice(OPENERS) + worked[0] + rng.choice(CLOSERS))]
    if len(worked) > 1 and rng.random() < 0.35:
        turns.append((rng.choice(AGAIN), rng.choice(AGAIN_LEAD) + worked[1]))
    return turns


def conversation(rng):
    """One to three problems in a row, the way people actually ask them."""
    turns = []
    for _ in range(rng.choices([1, 2, 3], weights=[60, 30, 10])[0]):
        turns += exchange(rng)
    return turns


if __name__ == "__main__":
    rng = random.Random(3)
    for _ in range(14):
        for user, bot in conversation(rng):
            print(f"  you: {user}")
            print(f"  bot: {bot}")
        print()
