"""Greetings, and nothing else - the corpus for a very small separate model.

The page needs something to say before a conversation exists. A preset list would
do, and would be obvious after the fourth visit; a model trained only on greetings
can compose one, and being a model it will occasionally produce something slightly
odd, which is more interesting than a rotation of five fixed strings.

The whole job is one turn, so the model can be tiny - a few tens of thousands of
parameters, small enough to sit in the page itself. That only works if this file
offers enough distinct greetings that it has to learn the shape rather than the
list: the combinations below run to millions, and 30,000 draws repeat about 1% of
the time.

    py corpus/greeter/greetings.py         # look at some
    py corpus/greeter/make_greetings.py    # write the corpus
"""

import random

# Opening move. Some are time-of-day, most are not, because the page has no clock.
HELLOS = [
    "Hello", "Hi", "Hey", "Hello there", "Hi there", "Well hello", "Oh, hello",
    "Morning", "Afternoon", "Evening", "Good morning", "Good afternoon",
    "Good evening", "Hello again", "Right", "So", "Ah", "Hey there", "Greetings",
    "Welcome", "Welcome back", "There you are", "Good to see you", "Look who it is",
]

# What it says about itself, briefly and truthfully.
ABOUT = [
    "I am a small language model",
    "I am about half a megabyte of text",
    "I am a few hundred thousand numbers",
    "I run entirely in this tab",
    "I am small enough to fit in a text file",
    "I was trained on a text file somebody wrote",
    "I guess one character at a time",
    "I am not connected to anything",
    "nothing you type leaves this page",
    "I have no idea what day it is",
    "I know a little about a lot of ordinary things",
    "I can work out small sums if you show me one",
    "I am better at short questions than long ones",
    "I forget everything when you close this",
]

# An invitation to say something.
ASKS = [
    "What can I do for you?", "Ask me something.", "What would you like to know?",
    "Say something and I will do my best.", "Go ahead.", "What are we doing today?",
    "Try me with a question.", "Ask me about something ordinary.",
    "Give me a sum if you like.", "Ask me what something is.",
    "What is on your mind?", "Where shall we start?", "Start anywhere.",
    "Type something and see.", "I am listening.", "Your turn.",
    "Ask me anything small.", "What shall we talk about?",
    "Try me on something practical.", "Ask away.",
]

# Optional honesty, since it is the most useful thing to know about it.
CAVEATS = [
    "", "", "", "", "",
    "I will be wrong sometimes.",
    "I get things wrong, so check anything that matters.",
    "Expect the occasional nonsense.",
    "I am small, and it shows.",
    "If I do not know something I will say so.",
    "Do not trust me on anything important.",
    "I am more confident than I should be.",
    "Big numbers defeat me.",
    "Recent events are beyond me entirely.",
]

# A few whole greetings that do not fit the pattern, so the shape is not the only
# thing the model learns.
ONE_OFFS = [
    "Hello. Nothing has happened yet, which is where we always start.",
    "Hi. This page is empty until you type in it.",
    "Hey. I have been waiting, in the sense that nothing was running.",
    "Hello again, though I do not remember the last time.",
    "Morning, or whatever it is. I have no way of telling.",
    "Hi. Small model, short answers, no server.",
    "Hello. I am the whole of the program, weights included.",
    "Hey there. Ask me what a kettle does, if you are stuck for a question.",
    "Hi. I am quick to answer and often wrong, in that order.",
    "Hello. Half a megabyte of text, pretending to hold a conversation.",
    "Right, I am here. Nothing to report.",
    "Hello. Nothing is being sent anywhere. There is nowhere to send it.",
]

# `about` entries are clauses, so they are lowercase. Anything that puts one after a
# full stop has to use the capitalised form, or the greeting reads as a typo.
SHAPES = [
    "{hello}. {ask}",
    "{hello}. {about_cap}. {ask}",
    "{hello} - {about}. {ask}",
    "{hello}. {ask} {caveat}",
    "{hello}. {about_cap}, so {ask_lower}",
    "{hello}. {ask} {about_cap}.",
    "{hello}. {about_cap} and {about2}. {ask}",
    "{hello}. {caveat} {ask}",
    "{hello}, and {about}. {ask}",
]


def greeting(rng):
    """One greeting, composed."""
    if rng.random() < 0.06:
        return rng.choice(ONE_OFFS)

    about, about2 = rng.sample(ABOUT, 2)
    ask = rng.choice(ASKS)
    shape = rng.choice(SHAPES)
    text = shape.format(
        hello=rng.choice(HELLOS),
        about=about,
        about2=about2,
        about_cap=about[0].upper() + about[1:],
        ask=ask,
        ask_lower=ask[0].lower() + ask[1:],
        caveat=rng.choice(CAVEATS),
    )
    # shapes with an optional caveat can leave a double space or a dangling gap
    return " ".join(text.split()).replace(" .", ".")


if __name__ == "__main__":
    rng = random.Random(2)
    seen = set()
    for _ in range(12):
        print("  " + greeting(rng))
    for _ in range(30000):
        seen.add(greeting(rng))
    print(f"\n  {len(seen):,} distinct in 30,000 draws")
