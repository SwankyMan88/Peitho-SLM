"""Conversations where something said earlier is asked about later.

Nothing else in the corpus teaches this. SQuAD passages teach reading back through
a passage, but a person telling a model their dog's name and asking two turns later
is a different shape, and xxl_1.0 failed it: told "my dog is called Pepper", asked,
it answered "My name is a dog."

An earlier design of the chat corpus dropped drills like these because they taught
the model to answer every statement with "noted, I will remember that" - so here:

  * the reply to a statement is short and varied, and often just carries on,
  * a third of conversations ask about something never mentioned, where the honest
    answer is that it was not said - otherwise the model learns to invent one,
  * a third correct a fact partway through ("no, it was Colin"), because a real chat
    broke exactly there: corrected, xxl_1.2 kept answering with the old value,
  * the facts are ordinary words in ordinary sentences, not slot-filler tokens,
  * questions come one to four turns later, with unrelated chat in between, so the
    answer is never simply the last thing said.

    py corpus/recall/make_recall.py --out build/recall.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK

NAMES = ["Colin", "Ada", "Marcus", "Priya", "Tom", "Nadia", "Femi", "Ruth", "Ivan", "Sofia",
         "Hector", "Mei", "Joel", "Anya", "Owen", "Leila", "Gus", "Bea", "Karl", "Noor"]
PETS = ["Pepper", "Biscuit", "Marlow", "Tuna", "Rufus", "Clover", "Jasper", "Nettle",
        "Bramble", "Otis", "Willow", "Domino", "Pickle", "Sable", "Mango"]
ANIMALS = ["dog", "cat", "rabbit", "parrot", "tortoise", "pony", "goldfish"]
JOBS = ["teacher", "nurse", "plumber", "baker", "bus driver", "vet", "librarian",
        "electrician", "farmer", "chef", "postman", "joiner", "gardener"]
TOWNS = ["Leeds", "Bristol", "Cork", "Hull", "Perth", "Dundee", "Bath", "Derby",
         "Galway", "Preston", "Truro", "Ely", "Stirling", "Newport"]
HOBBIES = ["climbing", "baking bread", "birdwatching", "playing the fiddle", "sea swimming",
           "restoring clocks", "growing chillies", "long-distance running"]
DRINKS = ["black coffee", "mint tea", "cold milk", "cider", "hot chocolate"]
NUMBERS = ["4172", "8305", "1190", "6621", "9047", "2538"]

# Said back when a fact arrives. Short, and mostly not a promise to remember.
TAKEN = ["Right.", "Noted.", "Got it.", "Okay.", "Fair enough.", "Good to know.",
         "Understood.", "Nice.", "Right, go on.", "Okay, I have that."]
SMALL_TALK = [
    ("What is 12 + 9?", "21."), ("What is 7 * 6?", "42."),
    ("What is a kettle for?", "A kettle boils water - that is the whole of it."),
    ("Is it raining where you are?", "No way for me to tell. I cannot see outside this chat."),
    ("What is 30 - 14?", "16."),
    ("Tell me something short about bridges.",
     "A bridge carries a load across a gap. Most of the work is in the supports."),
]
# What a correction can replace, by the kind of fact it is: getting this from the
# words of the sentence instead put a person's name where a town belonged.
POOLS = {"name": NAMES, "pet": PETS, "job": JOBS, "town": TOWNS,
         "hobby": HOBBIES, "drink": DRINKS, "number": NUMBERS}


def turn(marker, text, thought=""):
    if marker == BOT_MARK:
        return BOT_MARK + thought + THINK_MARK + text + END_MARK
    return USER_MARK + text + END_MARK


def facts_for(rng):
    """(kind, what they said, ways of asking, ways of answering)."""
    name, pet, animal = rng.choice(NAMES), rng.choice(PETS), rng.choice(ANIMALS)
    job, town = rng.choice(JOBS), rng.choice(TOWNS)
    hobby, drink, number = rng.choice(HOBBIES), rng.choice(DRINKS), rng.choice(NUMBERS)
    return [
        ("name", f"My name is {name}.",
         ["What is my name?", "Remind me - what did I say my name was?", "What was my name?"],
         [f"You said your name is {name}.", f"{name}, you said.", f"{name}."]),
        ("pet", f"I have a {animal} called {pet}.",
         [f"What is my {animal} called?", "What did I call my pet?", f"My {animal}'s name?"],
         [f"You said your {animal} is called {pet}.", f"{pet}.", f"{pet}, you said."]),
        ("job", f"I work as a {job}.",
         ["What do I do for a living?", "What was my job again?", "What is my work?"],
         [f"You said you work as a {job}.", f"A {job}.", f"{job.capitalize()}."]),
        ("town", f"I live in {town}.",
         ["Where do I live?", "Which town did I say I was in?", "What town am I in?"],
         [f"You said you live in {town}.", f"{town}.", f"{town}, you said."]),
        ("hobby", f"I am into {hobby} at the moment.",
         ["What am I into these days?", "What hobby did I mention?"],
         [f"You said {hobby}.", f"{hobby.capitalize()}, you said."]),
        ("drink", f"I drink {drink} in the mornings.",
         ["What do I drink in the mornings?", "What was my morning drink?"],
         [f"You said {drink}.", f"{drink.capitalize()}."]),
        ("number", f"Remember this number: {number}.",
         ["What was the number I gave you?", "Read the number back to me."],
         [f"{number}.", f"The number you gave me is {number}."]),
    ]


def rephrase(answers, old, new):
    return [a.replace(old, new) for a in answers]


def conversation(rng):
    """Facts given, chat in between, then a question about one of them - or about
    something never said, or about something corrected along the way."""
    facts = facts_for(rng)
    rng.shuffle(facts)
    given = facts[:rng.randint(1, 3)]

    lines = []
    for _, statement, _, _ in given:
        lines.append(turn(USER_MARK, statement))
        lines.append(turn(BOT_MARK, rng.choice(TAKEN),
                          rng.choice(["Nothing to work out - they are telling me something.",
                                      "A statement, not a question. Keep it short.",
                                      "Just take it in and say so briefly."])))
    for _ in range(rng.randint(0, 2)):
        q, a = rng.choice(SMALL_TALK)
        lines.append(turn(USER_MARK, q))
        lines.append(turn(BOT_MARK, a, "Answer it plainly, then wait."))

    # Never said: the honest answer is that, not an invented one.
    if rng.random() < 0.3:
        unsaid = [f for f in facts if f not in given]
        if unsaid:
            _, _, questions, _ = rng.choice(unsaid)
            lines.append(turn(USER_MARK, rng.choice(questions)))
            lines.append(turn(BOT_MARK, rng.choice([
                "You have not told me that one.", "That has not come up - tell me and I will use it.",
                "I do not have that. You would have to say."]),
                rng.choice(["Looking back, they never said that. Do not invent one.",
                            "That is not in what they told me. Say so plainly."])))
            return START_MARK + "\n" + "\n".join(lines)

    kind, statement, questions, answers = rng.choice(given)

    # Corrected partway through: the later value is the true one.
    if rng.random() < 0.33:
        old = next((v for v in POOLS[kind] if v in statement), None)
        new = rng.choice([v for v in POOLS[kind] if v != old])
        if old is not None:
            correction = rng.choice([f"No, it was {new}.", f"Actually it is {new}, not {old}.",
                                     f"Sorry - {new}, I mistyped."])
            lines.append(turn(USER_MARK, correction))
            lines.append(turn(BOT_MARK,
                              rng.choice([f"Got it - {new}.", f"Right, {new}.",
                                          f"Noted, {new} it is."]),
                              "They are correcting what they said. Use the new one from here."))
            lines.append(turn(USER_MARK, rng.choice(questions)))
            lines.append(turn(BOT_MARK, rng.choice(rephrase(answers, old, new)),
                              f"They corrected this: it is {new} now, not {old}."))
            return START_MARK + "\n" + "\n".join(lines)

    lines.append(turn(USER_MARK, rng.choice(questions)))
    lines.append(turn(BOT_MARK, rng.choice(answers),
                      rng.choice([f"They said it earlier: {statement[:-1]}. Read it back.",
                                  "It is further up the conversation. Say what they said.",
                                  "They told me this already. Answer from that, not from guessing."])))
    return START_MARK + "\n" + "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Generate recall conversations.")
    p.add_argument("--out", default=os.path.join(paths.BUILD, "recall.json"))
    p.add_argument("--count", type=int, default=6000)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    rng = random.Random(args.seed)
    convs = [conversation(rng) for _ in range(args.count)]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(convs, f)
    chars = sum(len(c) for c in convs)
    print(f"{len(convs):,} conversations, {chars:,} chars -> {paths.short(args.out)}")
    print(f"{len(set(convs)):,} of them distinct\n")
    print(convs[0])


if __name__ == "__main__":
    main()
