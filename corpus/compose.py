"""Generate large volumes of varied, grammatical English conversation.

Hand-written text is what gives the model something worth saying, but there is only
tens of kilobytes of it. Repeated enough times to fill a corpus it gets recited
word for word: ask a question that appears in the corpus and the reply comes back
identical every time.

This module builds conversation from word pools and clause shapes instead. The
combinations run into the billions, so nearly every exchange is seen once and the
model has to learn how English fits together rather than which reply follows which
prompt. Sentences are grammatical without always being sensible - that is the
trade. The hand-written part of the corpus carries meaning; this part carries
fluency and the habit of building a sentence that has never existed before.

The shapes deliberately cover more than statements: questions back, reactions,
hedges, comparisons and follow-ups that reuse the subject under discussion. A
generator that only emits declarative sentences teaches a model that only knows
how to declare.
"""

import random

NOUNS = """harbour valley orchard meadow ridge canyon lagoon glacier prairie thicket
marsh dune reef estuary plateau gorge basin summit hollow grove
kitchen workshop library attic cellar corridor courtyard balcony terrace pantry
station platform bridge tunnel lighthouse windmill quarry foundry mill dock
kettle lantern compass anvil ladder bucket basket chisel loom hinge
violin drum flute harp bell whistle organ cello banjo tambourine
sparrow heron otter badger falcon lizard moth beetle salmon hare
willow cedar birch bramble fern moss lichen thistle clover ivy
storm drizzle frost haze gale current tide ripple ember shadow
letter journal ledger map sketch photograph postcard almanac recipe rumour
engine bearing lever pulley furnace boiler valve gasket flywheel rivet
market alley rooftop chimney gutter doorway staircase window fence gate
river stream weir pond spring waterfall channel bank shallows shoreline""".split()

ADJECTIVES = """quiet narrow crooked weathered patient stubborn restless hollow brittle
faded steady crowded empty distant familiar unlikely careful reckless plain ornate
damp warm bitter sweet coarse smooth ragged tidy tangled sunken
early late sudden gradual constant rare frequent brief endless momentary
grey golden pale bright dim silver rusted glassy dusty spotless
small vast slender heavy weightless dense sparse thick shallow abandoned
awkward elegant blunt precise fragile sturdy uneven level lopsided balanced""".split()

VERBS = """settles drifts gathers narrows widens darkens brightens softens hardens shifts
creaks hums rattles whistles murmurs echoes fades swells steadies falters
holds waits leans tilts turns spreads folds opens closes rests
weathers survives outlasts crowds thins collects scatters lingers passes empties""".split()

ADVERBS = """slowly quickly quietly steadily suddenly gradually faintly plainly oddly
neatly roughly gently sharply patiently briefly rarely often somehow
apparently mostly barely almost entirely partly clearly endlessly""".split()

PLACES = [
    "by the water", "under the eaves", "along the ridge", "behind the mill",
    "near the bridge", "past the orchard", "beneath the ice", "above the treeline",
    "inside the workshop", "across the valley", "beyond the dunes", "among the reeds",
    "against the wall", "at the far end", "near the old station", "under a grey sky",
    "over the rooftops", "in the shade", "at the edge of the field", "below the weir",
    "between the houses", "out past the wall", "up in the rafters", "down by the docks",
    "round the back", "off to one side", "further inland", "closer to the water",
]

TIMES = [
    "in winter", "by morning", "after rain", "at dusk", "before the frost",
    "through the summer", "overnight", "for a season", "within an hour",
    "long before anyone noticed", "once the wind drops", "while the tide is out",
    "on the coldest nights", "just after sunrise", "for years at a time",
    "every autumn", "on a still day", "after a long dry spell", "by the end of the week",
]

REASONS = [
    "the air holds more water than it looks like",
    "nothing there has been disturbed for a long time",
    "the ground drains faster than the rain arrives",
    "heat leaves faster than it is replaced",
    "the shape of the land funnels the wind",
    "salt works into everything eventually",
    "the light arrives at too shallow an angle",
    "what grows there has to tolerate being cut back",
    "the pressure never has anywhere to go",
    "small changes accumulate without being noticed",
    "the material was never meant to last this long",
    "whatever passes through leaves a little behind",
    "the cold gets in wherever the joint is weakest",
    "nobody has needed to repair it yet",
    "the current does the work that nobody wants to",
    "the shape was decided long before anyone asked",
    "it was built for a purpose that no longer exists",
]

OBSERVATIONS = [
    "it is easy to miss unless you are looking for it",
    "most people walk past without noticing",
    "the effect is small but it never stops",
    "it happens the same way every time",
    "nobody has a tidy explanation for it",
    "the pattern shows up at every scale",
    "you can hear it before you can see it",
    "the reverse is also true, which is stranger",
    "it took a long time for anyone to write it down",
    "the simplest description turns out to be the right one",
    "the change is obvious once it has already happened",
    "it looks deliberate and almost certainly is not",
    "the useful part and the interesting part are not the same part",
]

HEDGES = ["I think", "I would guess", "As far as I can tell", "If I had to say",
          "My sense is that", "It seems to me that", "Probably"]

REACTIONS = ["That is a good question.", "I had not thought about it that way.",
             "That is worth sitting with.", "Fair enough.", "That follows.",
             "I would not have guessed that.", "That makes sense to me.",
             "Interesting way to put it.", "I can see that."]


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def statement(rng, noun=None):
    """A declarative sentence, optionally forced to be about `noun`."""
    noun = noun or rng.choice(NOUNS)
    adj, verb = rng.choice(ADJECTIVES), rng.choice(VERBS)
    adv, place, time = rng.choice(ADVERBS), rng.choice(PLACES), rng.choice(TIMES)

    shape = rng.randint(0, 9)
    if shape == 0:
        return f"The {adj} {noun} {verb} {adv} {place}."
    if shape == 1:
        return f"{time.capitalize()}, {article(adj)} {adj} {noun} {verb} {place}."
    if shape == 2:
        return (f"{article(adj).capitalize()} {adj} {noun} {verb} {adv}, because "
                f"{rng.choice(REASONS)}.")
    if shape == 3:
        return f"The {noun} {verb} {place}, and {rng.choice(OBSERVATIONS)}."
    if shape == 4:
        return f"What makes the {adj} {noun} interesting is that {rng.choice(REASONS)}."
    if shape == 5:
        return f"{time.capitalize()} the {noun} {verb}, though {rng.choice(OBSERVATIONS)}."
    if shape == 6:
        other, other_adj = rng.choice(NOUNS), rng.choice(ADJECTIVES)
        return (f"{article(noun).capitalize()} {noun} and {article(other_adj)} "
                f"{other_adj} {other} {verb} for the same reason: "
                f"{rng.choice(REASONS)}.")
    if shape == 7:
        return (f"You would expect the {noun} to {verb.rstrip('s')} {adv}, and "
                f"{place} it does.")
    if shape == 8:
        return (f"{rng.choice(HEDGES)} the {noun} {verb} {adv} because "
                f"{rng.choice(REASONS)}.")
    return f"There is {article(adj)} {adj} {noun} {place} that {verb} {adv} {time}."


def question(rng, noun=None):
    """A question, so the model learns to ask as well as answer."""
    noun = noun or rng.choice(NOUNS)
    adj, verb = rng.choice(ADJECTIVES), rng.choice(VERBS)
    place = rng.choice(PLACES)
    shape = rng.randint(0, 6)
    if shape == 0:
        return f"Have you ever watched {article(adj)} {adj} {noun} {place}?"
    if shape == 1:
        return f"What would you do with {article(adj)} {adj} {noun}?"
    if shape == 2:
        return f"Does the {noun} {verb.rstrip('s')} where you are?"
    if shape == 3:
        return f"Would you rather have {article(adj)} {adj} {noun} or a quiet one?"
    if shape == 4:
        return f"Is there {article(adj)} {adj} {noun} anywhere near you?"
    if shape == 5:
        return f"What does the {noun} {place} look like to you?"
    return f"Do you notice the {noun} more {rng.choice(TIMES)}?"


def opinion(rng, noun=None):
    noun = noun or rng.choice(NOUNS)
    adj = rng.choice(ADJECTIVES)
    return rng.choice([
        f"{rng.choice(HEDGES)} {noun}s are more {adj} than people give them credit for.",
        f"A {noun} is the sort of thing you only notice once it is gone.",
        f"I like {noun}s for the same reason I like {rng.choice(NOUNS)}s: "
        f"{rng.choice(REASONS)}.",
        f"The {adj} ones are the ones worth looking at.",
        f"There is something {adj} about {article(noun)} {noun} nobody maintains.",
    ])


def reply(rng, noun, sentences=None):
    """One to three sentences about `noun`, sometimes ending with a question back."""
    count = sentences or rng.randint(1, 3)
    parts = [statement(rng, noun) for _ in range(count)]
    if rng.random() < 0.25:
        parts.append(question(rng, noun))
    if rng.random() < 0.12:
        parts.insert(0, rng.choice(REACTIONS))
    return " ".join(parts)


OPEN_PROMPTS = [
    "Say something", "Say anything", "Tell me a thought", "Give me a sentence",
    "Make something up", "Describe something", "Think out loud",
    "Say something interesting", "Tell me something you noticed",
    "Invent a description", "Say something odd", "Give me an image",
    "Write me a line", "Surprise me", "Anything at all", "Go on then",
]

TOPIC_PROMPTS = [
    "Describe {art} {noun}.", "Tell me about {art} {noun}.",
    "Say something about {noun}s.", "What comes to mind about {art} {noun}?",
    "Write me a line about {art} {noun}.", "Imagine {art} {noun}.",
    "Give me a picture of {art} {noun}.", "Something about {noun}s, please.",
    "What do you make of {noun}s?", "Anything about {art} {noun}?",
]

FOLLOW_UPS = [
    "Go on.", "Say more.", "What else?", "And?", "Keep going.",
    "That is interesting.", "Why though?", "How so?", "I see.",
    "What makes you say that?", "Tell me more about that.",
]


# Fixed greeting pairs are the one place recitation survives everything else: if
# "Hello" has exactly one reply in the corpus, a model that generalizes perfectly
# will still answer it identically every time. No amount of weight decay or dropout
# changes that - the variety has to exist in the data. These combine instead.
GREET_OPEN = ["Hello.", "Hi.", "Hey.", "Hey there.", "Good to see you.", "Morning.",
              "Afternoon.", "Evening.", "Oh, hello.", "Hello again.", "There you are.",
              "Hi there.", "Well, hello."]
GREET_FOLLOW = ["What is on your mind?", "Go ahead.", "What can I do for you?",
                "Where would you like to start?", "What are we doing today?",
                "Ask away.", "I am listening.", "What brings you here?",
                "How are you?", "Say what you like.", "What is it?",
                "Anything in particular?", "I have nothing else on.",
                "You have my attention.", "Start anywhere."]
THANK_REPLY = ["Any time.", "You are welcome.", "No trouble.", "Glad to help.",
               "Of course.", "Happy to.", "That is what I am for.", "No bother at all.",
               "Do not mention it.", "Whenever you like.", "Pleased to be useful."]
ACK_REPLY = ["Alright.", "Understood.", "Fair enough.", "Noted.", "Right.", "Okay.",
             "Sure.", "Whenever you are ready.", "Take your time.", "Go on.",
             "I am with you.", "Carry on."]
BYE_REPLY = ["Goodbye.", "Take care.", "See you.", "Until next time.", "So long.",
             "Come back whenever.", "Mind how you go.", "All the best."]

GREET_PROMPTS = ["Hello", "Hi", "Hey", "Hey there", "hello", "hi", "hey", "yo",
                 "Good morning", "Good evening", "Morning", "Evening", "Hello there"]
THANK_PROMPTS = ["thanks", "Thanks", "Thank you", "thanks!", "ty", "cheers",
                 "Thanks a lot", "Appreciate it"]
ACK_PROMPTS = ["ok", "Ok", "okay", "sure", "Sure", "alright", "right", "hmm", "I see",
               "fine", "got it", "mhm"]
BYE_PROMPTS = ["bye", "Goodbye", "See you", "night", "Good night", "I have to go",
               "gtg", "later"]


def social_exchange(rng):
    """A greeting, thanks, acknowledgement or farewell with a varied reply."""
    kind = rng.random()
    if kind < 0.4:
        prompt = rng.choice(GREET_PROMPTS)
        reply_text = rng.choice(GREET_OPEN)
        if rng.random() < 0.75:
            reply_text += " " + rng.choice(GREET_FOLLOW)
    elif kind < 0.6:
        prompt = rng.choice(THANK_PROMPTS)
        reply_text = rng.choice(THANK_REPLY)
        if rng.random() < 0.25:
            reply_text += " " + rng.choice(GREET_FOLLOW)
    elif kind < 0.85:
        prompt = rng.choice(ACK_PROMPTS)
        reply_text = rng.choice(ACK_REPLY)
        if rng.random() < 0.3:
            reply_text += " " + rng.choice(GREET_FOLLOW)
    else:
        prompt = rng.choice(BYE_PROMPTS)
        reply_text = rng.choice(BYE_REPLY)
    return prompt, reply_text


def dialogue(rng):
    """A short composed conversation that keeps to one subject.

    The subject persists across turns, so continuing correctly means reading what
    came before rather than starting fresh - which is the habit worth teaching."""
    lines = []
    noun = rng.choice(NOUNS)

    if rng.random() < 0.3:
        lines.append(social_exchange(rng))
    if rng.random() < 0.4:
        lines.append((rng.choice(OPEN_PROMPTS), reply(rng, noun)))
    else:
        prompt = rng.choice(TOPIC_PROMPTS).format(art=article(noun), noun=noun)
        lines.append((prompt, reply(rng, noun)))

    for _ in range(rng.randint(0, 3)):
        roll = rng.random()
        if roll < 0.55:
            lines.append((rng.choice(FOLLOW_UPS), reply(rng, noun)))
        elif roll < 0.8:
            noun = rng.choice(NOUNS)
            prompt = rng.choice(TOPIC_PROMPTS).format(art=article(noun), noun=noun)
            lines.append((prompt, reply(rng, noun)))
        else:
            lines.append((question(rng, noun), opinion(rng, noun)))
    if rng.random() < 0.2:
        lines.append(social_exchange(rng))
    return lines


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(4):
        for user, bot in dialogue(rng):
            print(f"  you: {user}")
            print(f"  slm: {bot}")
        print()
    seen = {statement(rng) for _ in range(30000)}
    print(f"30000 statement draws produced {len(seen)} distinct sentences")
