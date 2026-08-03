"""Conversation that says something, and admits when it does not know.

This replaces the lyrical generator that came before it. That one built sentences
from adjective, verb and place pools, which produced grammatical English with no
content in it:

    What makes the weightless terrace interesting is that salt works into
    everything eventually. Long before anyone noticed, an unlikely terrace waits
    over the rooftops.

Three clauses, nothing said. The model learned to write like that because that is
what it was shown.

What is here instead:

  * a short, plainly true predicate for each of ~150 everyday things, wrapped in
    many ways of saying it, so the content is real and the phrasing varies
  * questions about things that do not exist, answered by admitting it
  * asking which of two things was meant, when a question is ambiguous
  * ordinary small talk

The facts are deliberately dull and safe - what a kettle does, what a hinge is for.
Dull and true beats interesting and invented, and a model this size has no business
asserting anything harder.
"""

import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose

# thing -> what it does or is, as a predicate. Kept short so it can be dropped into
# a sentence any number of ways.
FACTS = {
    "kettle": "boils water", "lantern": "holds a light so it can be carried",
    "compass": "points north", "anvil": "gives you something solid to hammer against",
    "ladder": "gets you higher than you can reach", "bucket": "carries water",
    "basket": "carries things that do not need to be watertight",
    "chisel": "cuts wood or stone when you hit it",
    "loom": "holds threads in place while cloth is woven",
    "hinge": "lets one thing swing against another", "sieve": "keeps the big bits back",
    "funnel": "puts a wide thing into a narrow one", "clamp": "holds two things together",
    "bellows": "pushes air where you want it", "whetstone": "puts an edge back on a blade",
    "trowel": "moves small amounts of earth or mortar",
    "plumb line": "tells you what is truly vertical",
    "spirit level": "tells you what is truly horizontal",
    "pulley": "changes the direction you pull in", "wedge": "splits things apart",
    "lever": "trades distance for force", "screw": "pulls itself in as it turns",
    "magnet": "pulls iron towards it", "lens": "bends light to a point",
    "mirror": "sends light back the way it came", "prism": "spreads light into colours",
    "thermometer": "measures temperature", "barometer": "measures air pressure",
    "clock": "counts out time in even pieces", "sundial": "tells the time by a shadow",
    "hourglass": "measures one fixed stretch of time", "metronome": "keeps a steady beat",
    "tuning fork": "sounds one note and nothing else", "drum": "makes sound from a struck skin",
    "whistle": "makes a note from moving air", "bell": "rings when it is struck",
    "harbour": "gives boats somewhere sheltered to sit",
    "lighthouse": "warns ships away from the rocks", "jetty": "reaches out into the water",
    "canal": "carries boats where there was no river", "aqueduct": "carries water across a gap",
    "dam": "holds a river back", "weir": "lets a river past at a set level",
    "bridge": "carries a road over something in the way",
    "tunnel": "carries a road through something in the way",
    "chimney": "takes smoke up and out", "cellar": "stays cool because it is underground",
    "attic": "is the space nobody planned to use", "porch": "gives you somewhere to stand out of the rain",
    "gutter": "takes rain off the roof", "greenhouse": "traps warmth for the plants inside",
    "orchard": "is a field of fruit trees", "hedge": "is a fence that grows",
    "meadow": "is grass nobody has ploughed", "marsh": "is ground too wet to walk on easily",
    "dune": "is sand the wind has piled up", "reef": "is rock or coral near the surface",
    "estuary": "is where a river meets the sea", "plateau": "is high ground that is flat on top",
    "gorge": "is a valley with steep sides", "basin": "is low ground water drains into",
    "summit": "is the highest point of a hill", "glacier": "is ice slow enough to look still",
    "avalanche": "is snow that has stopped holding on",
    "tide": "is the sea rising and falling twice a day",
    "current": "is water moving in one direction", "eddy": "is water turning back on itself",
    "frost": "is water freezing where it settled", "dew": "is water the cold air let go of",
    "fog": "is cloud at ground level", "hail": "is rain that froze on the way down",
    "thunder": "is the sound of air a lightning bolt shoved aside",
    "lightning": "is a very large spark", "rainbow": "is sunlight bent by raindrops",
    "eclipse": "is one thing passing in front of another",
    "comet": "is ice that grows a tail near the sun",
    "meteor": "is a small rock burning up in the air", "moon": "goes round the Earth",
    "planet": "goes round a star", "star": "is a ball of gas heavy enough to burn",
    "galaxy": "is a great many stars held together",
    "telescope": "gathers more light than an eye can", "microscope": "makes small things large",
    "battery": "stores energy as chemistry", "fuse": "breaks on purpose before anything else does",
    "switch": "makes or breaks a circuit", "wire": "carries current where you want it",
    "transformer": "trades voltage for current", "motor": "turns electricity into movement",
    "generator": "turns movement into electricity", "pump": "moves liquid uphill",
    "valve": "lets flow one way and not the other", "gasket": "stops a joint leaking",
    "bearing": "lets something spin without wearing away",
    "gear": "trades speed for turning force", "spring": "stores force by bending",
    "cog": "passes turning from one shaft to another", "piston": "turns pressure into a push",
    "sail": "turns wind into pull", "rudder": "points the boat", "keel": "stops the boat sliding sideways",
    "anchor": "holds a boat against the water moving",
    "knot": "holds rope to itself or to something else", "pulley block": "carries several pulleys at once",
    "yeast": "makes bread rise", "salt": "keeps food from spoiling and makes it taste of more",
    "vinegar": "is sour enough to preserve things", "sugar": "sweetens and feeds yeast",
    "flour": "is ground grain", "dough": "is flour and water worked together",
    "broth": "is what is left after simmering something for a long time",
    "compost": "is rubbish rotted down until plants can use it",
    "seed": "carries a plant in a form that can wait", "root": "holds a plant down and drinks for it",
    "leaf": "turns light into food", "bark": "protects the wood underneath",
    "sap": "carries food up and down a tree", "pollen": "carries a plant's half of a seed",
    "nectar": "is the bribe a flower pays an insect", "hive": "is where bees keep everything",
    "web": "catches what walks into it", "burrow": "is a hole something lives in",
    "nest": "holds eggs still until they hatch", "shell": "keeps the inside in and the outside out",
    "feather": "keeps a bird warm and helps it fly", "scale": "protects a fish and lets it bend",
    "fin": "steers a fish", "gill": "takes oxygen out of water", "lung": "takes oxygen out of air",
    "heart": "keeps blood moving", "bone": "holds the shape of an animal",
    "muscle": "pulls, and never pushes", "tendon": "ties muscle to bone",
    "nerve": "carries signals", "brain": "decides what to do with the signals",
    "map": "shows where things are relative to each other", "atlas": "is a book of maps",
    "index": "tells you which page to turn to", "glossary": "explains the words a book assumes",
    "ledger": "records what came in and what went out", "receipt": "proves something was paid for",
    "invoice": "asks for payment", "contract": "writes down what was agreed",
    "passport": "says which country will vouch for you", "ticket": "proves you paid to travel",
    "timetable": "says when things are meant to happen",
    "queue": "decides who goes next by who arrived first",
    "recipe": "is instructions that assume you know a little",
    "draft": "is writing that is not finished being wrong",
    # A second batch, so there is more to talk about and less to repeat.
    "kite": "flies because the wind pushes against it at an angle",
    "sledge": "slides instead of rolling", "wheel": "turns dragging into rolling",
    "axle": "holds a wheel while letting it turn", "brake": "turns movement into heat",
    "chain": "carries pull around corners", "rope": "carries pull in a straight line",
    "pulley wheel": "lets a rope change direction without much friction",
    "hook": "holds a load without being tied",
    "nail": "holds by friction", "bolt": "holds by being tightened",
    "rivet": "holds by being squashed in place", "glue": "holds by sticking",
    "solder": "joins metal with a lower-melting metal",
    "weld": "joins metal by melting both sides together",
    "saw": "cuts by taking a little away at a time",
    "file": "smooths by taking a very little away at a time",
    "plane": "shaves wood flat", "drill": "makes a round hole",
    "hammer": "delivers force in a small moment",
    "spanner": "turns a nut without slipping", "screwdriver": "turns a screw",
    "pliers": "grip harder than fingers can", "vice": "holds work while you use both hands",
    "tape measure": "gives you a length you can carry",
    "protractor": "measures an angle", "set square": "gives you a right angle",
    "callipers": "measure a thickness accurately",
    "scales": "compare a weight against a known one",
    "sundial gnomon": "casts the shadow a sundial reads",
    "well": "reaches down to water", "cistern": "stores water until it is wanted",
    "tap": "lets water out when you ask", "drain": "takes water away",
    "sewer": "carries waste water somewhere else",
    "filter": "lets the water through and keeps the dirt",
    "kiln": "gets hot enough to change clay for good",
    "forge": "gets metal hot enough to shape",
    "furnace": "makes heat on purpose, in quantity",
    "boiler": "turns water into steam", "radiator": "gives heat back to a room",
    "insulation": "slows heat down", "draught": "is air moving where you did not want it",
    "candle": "burns its own wick slowly", "wick": "carries fuel to the flame",
    "match": "makes fire from friction", "flint": "makes a spark when struck",
    "tinder": "catches from a spark", "ember": "is fire without flame",
    "ash": "is what will not burn", "smoke": "is what did not burn properly",
    "steam": "is water as gas", "ice": "is water as solid",
    "cloud": "is water waiting to fall", "wind": "is air moving from high to low pressure",
    "shadow": "is where light did not reach", "echo": "is sound that came back",
    "shade": "is somewhere the sun does not reach",
}

# ------------------------------------------------------------- saying a fact
# Each fact used to be expressed exactly one way. FACTS["kettle"] is "boils water",
# it went into one of ten templates, and so the informative half of every answer was
# one of 206 fixed strings - 36,000 fact replies drawn from 206 phrasings. The
# dressing varied; the substance never did. A model trained on that can only recite,
# and it did: 7% of large_1.2's replies were not a verbatim training string, and half
# opened with one.
#
# So the same fact is now said four different ways - as a verb phrase, as a job, as
# a purpose, as a noun phrase - which forces the model to learn what a kettle does
# rather than which sentence follows the word "kettle". It is the difference between
# a lookup table and something that composes.

IRREGULAR_BASE = {"is": "be", "has": "have", "goes": "go", "does": "do",
                  "says": "say", "flies": "fly", "dries": "dry"}
IRREGULAR_GERUND = {"be": "being", "go": "going", "have": "having", "do": "doing"}
VOWELS = "aeiou"


def base_form(verb):
    """"boils" -> "boil". Third person back to the plain verb."""
    if verb in IRREGULAR_BASE:
        return IRREGULAR_BASE[verb]
    if verb.endswith("ies"):
        return verb[:-3] + "y"
    if verb.endswith(("ses", "xes", "zes", "ches", "shes", "oes")):
        return verb[:-2]
    if verb.endswith("s"):
        return verb[:-1]
    return verb


def gerund(word):
    """"boil" -> "boiling", "store" -> "storing", "put" -> "putting"."""
    if word in IRREGULAR_GERUND:
        return IRREGULAR_GERUND[word]
    if word.endswith("e") and not word.endswith(("ee", "ye", "oe")):
        return word[:-1] + "ing"
    # English doubles the final consonant only when the stress lands on the last
    # syllable, which for these verbs means the word has just one: put -> putting,
    # but deliver -> delivering, not deliverring.
    one_syllable = len(re.findall(r"[aeiouy]+", word)) == 1
    if (one_syllable and len(word) >= 3 and word[-1] not in VOWELS + "wxy"
            and word[-2] in VOWELS and word[-3] not in VOWELS):
        return word + word[-1] + "ing"
    return word + "ing"


def inflectable(predicate):
    """Whether a predicate is one verb phrase that can safely be reworded.

    "lets the water through and keeps the dirt" has a second verb agreeing with the
    subject, so rewording only the first would give "letting the water through and
    keeps the dirt". Those, the plural-subject ones ("grip harder than fingers can")
    and the noun phrases are left exactly as they are."""
    head = predicate.split()[0]
    if not head.endswith("s") or predicate.startswith("is "):
        return False
    return not re.search(r"\band \w+s\b", predicate)


# {np} is the thing as it appears in a sentence - "a kettle", "pliers", "thunder" -
# and {it}/{is}/{its} agree with it, so one frame serves a count noun, a mass noun
# and a plural-only noun without writing three of everything.

# The predicate as it stands: "A kettle boils water."
VERB_FRAMES = [
    "{NP} {pred}.", "{It} {pred}.", "Mostly {it} {pred}.",
    "{NP} {pred}, and not much else.",
    "{NP} {pred} - that is the whole of it.",
    "{NP} {is} what {pred}.", "{NP} {pred}, and that is the job.",
]
# The same fact as a job: "Its job is to boil water."
BASE_FRAMES = [
    "{Its} job {is} to {base}.", "What {it} {does} is {base}.",
    "You use one to {base}.", "{NP} will {base}.",
    "The point of {np} is to {base}.", "You would use {np} to {base}.",
    "One thing: {it} will {base}.",
]
# The same fact as a purpose: "It is for boiling water."
GERUND_FRAMES = [
    "{It} {is} for {ger}.", "{Ger} - that is what {np} {is} for.",
    "{NP} {is} the thing for {ger}.",
    "If something needs {ger}, that is {np}.",
    "{NP} {is} about {ger} and little else.", "For {ger}, you want {np}.",
]
# The ones that are already a noun phrase: "A meadow is grass nobody has ploughed."
NOUN_FRAMES = [
    "{NP} {is} {noun}.", "{Thing}: {noun}.", "Think of {np} as {noun}.",
    "{NP}? {Noun}.", "Near enough, {np} {is} {noun}.",
    "{Noun} - that is {np}.", "{NP} {is} {noun}, more or less.",
    "{It} {is} {noun}.",
]


def say_fact(rng, thing, with_after=True):
    """One sentence expressing what a thing is, chosen from every way we can say it.

    Around twenty shapes per fact rather than ten, and - the part that matters - the
    informative words differ between them rather than only the packaging."""
    predicate = FACTS[thing]
    plural = thing in PLURAL_ONLY
    parts = words(thing)

    if predicate.startswith("is "):
        noun = predicate[3:]
        frame = rng.choice(NOUN_FRAMES)
        text = frame.format(noun=noun, Noun=noun[0].upper() + noun[1:], **parts)
    elif inflectable(predicate):
        head, _, rest = predicate.partition(" ")
        stem = base_form(head)
        base = (stem + " " + rest).strip()
        ger = (gerund(stem) + " " + rest).strip()
        frame = rng.choice(VERB_FRAMES + BASE_FRAMES + GERUND_FRAMES)
        # "bellows" is plural and its predicate is written singular, so the verb has
        # to lose its "s" to agree: bellows push air, they do not pushes it.
        text = frame.format(pred=base if plural else predicate, base=base, ger=ger,
                            Ger=ger[0].upper() + ger[1:], **parts)
    else:
        # A plural subject ("pliers grip harder than fingers can") is already in the
        # right form for a plural frame and cannot be turned into a singular one.
        frame = rng.choice(VERB_FRAMES if plural else VERB_FRAMES[:1])
        text = frame.format(pred=predicate, **parts)

    if with_after and rng.random() < 0.35:
        text += " " + rng.choice(AFTER)
    return text


# A follow-up that can honestly go after any of the above.
AFTER = [
    "It is not complicated, but it has to be the right size.",
    "Simple idea, and it has been around a long time.",
    "There are fancier versions, but they all do that.",
    "You notice it most when it is missing.",
    "Nothing clever about it, which is why it works.",
    "Most of the design is about making it last.",
    "People argue about the details more than you would expect.",
    "It goes wrong in one or two predictable ways.",
    "Easy to take for granted.",
    "Worth having a decent one.",
    "The cheap ones do the same job less pleasantly.",
    "Older ones tend to be heavier and last longer.",
]

# {np} carries its own article, or none at all: "a kettle", but "thunder" and
# "pliers". Asking "what is a thunder?" taught the model to write it that way.
ASK = [
    "What {is} {np}?", "What {does} {np} do?", "Tell me about {np}.",
    "What {is} {np} for?", "whats {np}", "Explain {np} to me.",
    "I keep hearing about {them}. What are they?", "What do you know about {them}?",
    "Why would anyone need {np}?", "{Is} {np} useful?",
    "Can you explain what {np} {is}?", "{them} - what is the point of them?",
]

# The old ANSWER list lived here: ten templates, each dropping the predicate in
# unchanged. say_fact replaced it, because "how it is packaged" was the only thing
# those ten varied.

# Things that do not exist, so the honest answer is the only right one. Kept
# obviously invented: a made-up word is a clear signal, and the model needs the cue
# to be learnable rather than a guess about how obscure a real word is.
NONSENSE = [
    "florn", "vantelle", "quiddick", "morvane", "trellick", "sprundle", "flimbet",
    "gorrick", "harnwell", "jesple", "kravin", "lumberkin", "murtish", "norvel",
    "obrint", "plavis", "quorm", "ranthe", "sethrick", "torvine", "umbrell",
    "vessick", "wandrel", "yarrow-pin", "zestrel", "brindlecap", "clathorn",
    "dremmel", "eskerine", "fennoway",
]
UNKNOWN_ASK = [
    "What is {a} {thing}?", "Tell me about {a} {thing}.", "What does {a} {thing} do?",
    "Have you heard of {them}?", "whats a {thing}", "Explain {thing} to me.",
    "Do you know what {a} {thing} is?", "What do you know about {thing}?",
]
DONT_KNOW = [
    "I do not know what that means yet.",
    "I have not come across that one. I would only be guessing.",
    "That is not something I know. I would rather say so than invent it.",
    "No idea, honestly. It is not a word I have seen.",
    "I do not know that one yet. Tell me and I will use it.",
    "That means nothing to me, which probably means I have not been taught it.",
    "I could make something up here. I would rather not.",
    "Never heard of it. What is it?",
    "I do not know. If it is important, do not take my word for anything about it.",
    "Not a clue. I am small, and that shows up in gaps like this.",
]

# Ambiguous questions, where the useful reply is a question back.
CLARIFY_ASK = [
    "Can you help me with the {thing}?", "Something is wrong with my {thing}.",
    "I need to sort out {np}.", "What should I do about the {thing}?",
    "The {thing} is being difficult.", "Any advice on {them}?",
]
CLARIFY = [
    "Depends what you are after - do you mean how {np} works, or how to fix one?",
    "Which part? Choosing {np}, or using one you already have?",
    "I can try. Is it broken, or are you deciding whether you need one?",
    "Tell me a bit more - is this about {them} in general or one in particular?",
    "Happy to. Do you want the short version or the details?",
    "What is it doing that it should not be?",
]

# Ordinary conversational moves. Question and answers are kept together on purpose:
# drawing them from two independent lists produced exchanges like "How big are you?"
# answered with "Useful in a narrow way", which teaches answering beside the point.
CHIT = [
    ("How are you?", ["Fine, as far as I can tell. What do you need?",
                      "No complaints. Ask me something.",
                      "Same as always. What is up?"]),
    ("What are you up to?", ["Waiting for you to type, which is most of my life.",
                             "Nothing at all until you say something.",
                             "This. Only this."]),
    ("Busy today?", ["Not busy. Ask me something.", "Never busy, in any real sense."]),
    ("How is it going?", ["Going well enough. What is on your mind?", "Fine. You?"]),
    ("You there?", ["Here. Go ahead.", "Yes. Say the thing."]),
    ("What can you do?", ["I can hold a short conversation and work out small sums.",
                          "Short answers about things I have been shown, and arithmetic "
                          "if the numbers are small.",
                          "Talk, badly, and add up, slowly."]),
    ("What are you good at?", ["Short factual answers and small arithmetic.",
                               "Being brief. It is not entirely a choice.",
                               "Sums where the numbers are small enough to work through."]),
    ("What are you bad at?", ["Long chains of reasoning, anything recent, and big sums.",
                              "Anything I was not shown. I do not know what I do not know.",
                              "Sounding uncertain when I should. Watch for that."]),
    ("Are you clever?", ["Useful in a narrow way. I fall apart outside what I was taught.",
                         "No. I am a small pattern-matcher with good manners.",
                         "Clever is the wrong word. Consistent, maybe."]),
    ("Do you get bored?", ["No. Nothing runs between your messages.",
                           "There is no me between one message and the next."]),
    ("How do you work?", ["I guess the next character, over and over, until a sentence "
                          "appears.",
                          "One character at a time, based on the characters before it."]),
    ("Who made you?", ["Someone training a small model on a text file.",
                       "A person with a text file and some patience."]),
    ("How big are you?", ["Small enough to fit in a text file you could scroll through.",
                          "About half a megabyte, written out as text.",
                          "Tiny. That is the interesting part and also the problem."]),
    ("Do you remember me?", ["Only what is still on screen. Nothing carries over once you "
                             "clear it.",
                             "No. Each conversation starts from nothing."]),
    ("Can you learn?", ["Not while we are talking. Learning happens when I am trained.",
                        "No. What I know was fixed before you opened this."]),
    ("Do you sleep?", ["There is nothing to sleep. I only exist while you are typing.",
                       "No, but I also do not run when you are not here."]),
]

# The same care for follow-ups: each question keeps its own answers.
FOLLOW = [
    ("Is that all it does?", ["More or less. {after}", "That is the main thing, yes.",
                              "It is a simple object."]),
    ("Do I need one?", ["Only if you are doing the thing it is for.",
                        "Probably not, unless the job comes up.",
                        "Most people manage without until they do not."]),
    ("How would I choose one?", ["Get the plainest one that is the right size.",
                                 "Weight and fit matter more than features.",
                                 "Buy the boring version once rather than the clever one twice."]),
    ("Anything else?", ["Not that I would swear to.", "That is as far as I would go.",
                        "Nothing I am confident about."]),
    ("Is there a better option?", ["Probably, but not one I know about.",
                                   "For some jobs, yes. For this one it is hard to beat.",
                                   "Not that I have been taught."]),
    ("What breaks on one?", ["Usually the moving part, if it has one.",
                             "Whatever takes the load.",
                             "The join, nine times in ten."]),
    ("Why that way round?", ["Because the alternative is more work for the same result.",
                             "It is the arrangement that needs the fewest parts."]),
]


# The content is fixed by FACTS, so variety has to come from how a thing is said.
# Without this the generator repeated one reply thousands of times in a corpus of a
# few hundred thousand turns, and repetition is what gets recited back.
OPENERS = ["", "", "", "", "Sure. ", "Right. ", "Good question. ", "Honestly, ",
           "Short answer: ", "Well, ", "Easy one. ", "So, ", "Simple enough. ",
           "Broadly, ", "In practice, "]
CLOSERS = ["", "", "", "", "", " Does that help?", " Ask if you want more.",
           " That is the short version.", " There is more to it if you care.",
           " I think that is right.", " Roughly speaking.", " That is my understanding.",
           " Nothing surprising about it."]


# A refusal takes different dressing. "Sure." in front of one reads as agreement,
# and "Nothing surprising about it." after one is answering a question it just
# declined to answer.
SORRY_OPENERS = ["", "", "", "", "Honestly, ", "Sorry, ", "Afraid not - ", "No - "]
SORRY_CLOSERS = ["", "", "", "", "", " Tell me and I will use it.",
                 " You would have to tell me.", " I would rather be plain about it."]


def dress(rng, text, refusal=False):
    """The same answer, said a different way."""
    openers = SORRY_OPENERS if refusal else OPENERS
    closers = SORRY_CLOSERS if refusal else CLOSERS
    opener = rng.choice(openers)
    # "I" is a word, not a capitalised first letter: lowercasing it gives "i have".
    lowerable = text[:1].isupper() and not text.startswith(("I ", "I'"))
    if opener and lowerable and opener.endswith((", ", "- ")):
        text = text[:1].lower() + text[1:]
    return opener + text + rng.choice(closers)


# What the model works out before answering. Written per turn type, because only the
# generator knows whether it is about to answer, decline or ask back - and a thought
# that does not name the subject is the poetry problem moved somewhere quieter.
KNOWN_THOUGHT = [
    "They asked what {np} {is}. I know that one - {it} {fact}. Say it plainly.",
    "The question is about {them}. That is in what I was taught: {it} {fact}.",
    "{Thing}: {it} {fact}. Short answer, no padding.",
    "They want to know about {them}. I have that - {it} {fact}.",
    "This is a {thing} question. I know {it} {fact}, so answer and stop.",
    "I do know this one. {NP} {fact}. Keep it to that.",
    # The same working, reached through a different wording of the fact.
    "{Thing} I know. {said} That is the useful part.",
    "They want {thing}. {said} Keep it short.",
    "I have this one. {said} No need to pad it.",
    "Easy - {said} Answer and stop there.",
    "About {them}, and I know it. {said} Say that much.",
]
UNKNOWN_THOUGHT = [
    "They asked about {a} {thing}. I have never seen that word. Say so rather than guess.",
    "{Thing} means nothing to me. I could invent something here, and I should not.",
    "I do not know what {a} {thing} is. Be honest about it.",
    "No idea what {a} {thing} is. Admitting that is better than a plausible answer.",
    "They want {thing}. Not a word I was taught, so say I do not know.",
]
CANNOT_THOUGHT = [
    "They asked me {what}. I do not have that built into me, so I will be honest.",
    "This is asking {what}. Nothing from outside reaches me, so I cannot know it.",
    "They want to know {what}. I have no way to find that out. Say so.",
    "{What} - that is outside anything I can see. Do not invent it.",
    "I am being asked {what}. I would only be making it up. Be straight instead.",
]
CLARIFY_THOUGHT = [
    "They mentioned {np} but not what they want doing about it. Ask which.",
    "Too vague to answer well - {thing}, but which part? Ask before guessing.",
    "This could mean two things. Better to ask than to answer the wrong one.",
    "They want help with {np}. I need to know what kind before I answer.",
]
CHIT_THOUGHT = [
    "They are asking about me. Answer it straight, no hedging.",
    "Small talk. Say something true and short.",
    "This is about what I am, not about a thing. Be plain.",
    "Answer honestly, including the unflattering part.",
]
SOCIAL_THOUGHT = [
    "Just a greeting. Say hello back and leave room for them.",
    "Nothing to work out here - be friendly and brief.",
    "Politeness, not a question. Answer in kind.",
]

# How a cannot-know question reads when the thought refers back to it.
WHAT_ASKED = {
    "What is the date today?": "what the date is",
    "What time is it?": "what the time is",
    "What day is it?": "what day it is",
    "What is the news today?": "what is in the news",
    "Who won the game last night?": "who won a game last night",
    "What is the weather like?": "what the weather is doing",
    "Is it raining where you are?": "whether it is raining",
    "What happened yesterday?": "what happened yesterday",
    "Who is the president?": "who the president is",
    "What year is it?": "what year it is",
    "What is my name?": "their name",
    "How old am I?": "their age",
    "Where do I live?": "where they live",
    "What did I say earlier?": "what they said earlier",
    "Do you know who I am?": "who they are",
    "What was the last thing I asked you?": "what they asked before",
    "What did we talk about last time?": "what we talked about before",
    "Can you look something up for me?": "to look something up",
    "Can you check a website?": "to check a website",
    "What is on at the cinema?": "what is on at the cinema",
    "How much does a train ticket cost?": "the price of a train ticket",
    "Is the shop open?": "whether a shop is open",
    "What is the score?": "the score",
    "Did anything important happen this week?": "whether anything happened this week",
}


# Already plural, or not countable: "bellowses" and "flours" are both wrong.
SAME_PLURAL = {
    "bellows", "salt", "sugar", "flour", "dough", "broth", "compost", "pollen",
    "nectar", "sap", "vinegar", "yeast", "frost", "dew", "fog", "hail", "thunder",
    "lightning", "bark", "current",
}


def plural(word):
    """marsh -> marshes, not marshs."""
    if word in SAME_PLURAL or word in PLURAL_ONLY:
        return word          # pliers are already plural; "plierses" is not a word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


# Words that take no article. Without this the generator wrote "a thunder is the
# sound of..." and "a pliers grip harder than fingers can" - both of which teach the
# model to put an article where English does not.
PLURAL_ONLY = {"pliers", "callipers", "scales", "bellows", "tongs"}
# Listed rather than derived: "a cloud", "an echo" and "a shadow" are perfectly good
# English, so the test is whether the article is wrong, not whether the plural is
# irregular.
MASS = {
    "thunder", "lightning", "steam", "ice", "smoke", "ash", "shade", "vinegar",
    "frost", "dew", "fog", "hail", "salt", "sugar", "flour", "dough", "broth",
    "compost", "pollen", "nectar", "sap", "yeast", "bark", "wind",
}


def name(thing):
    """The thing as it appears in a sentence: "a kettle", "pliers", "thunder"."""
    if thing in PLURAL_ONLY or thing in MASS:
        return thing
    return f"{article(thing)} {thing}"


def words(thing):
    """Everything a template needs to talk about one thing, agreeing with itself.

    Pliers are plural and thunder takes no article, so "what is a pliers" and "a
    thunder is" are two versions of the same mistake. Templates ask for {np} and
    {is} rather than writing the article and the verb in by hand."""
    np = name(thing)
    plural_subject = thing in PLURAL_ONLY
    return {
        "thing": thing, "Thing": thing.capitalize(),
        "np": np, "NP": np[0].upper() + np[1:],
        "them": plural(thing),
        "a": article(thing), "A": article(thing).capitalize(),
        "it": "they" if plural_subject else "it",
        "It": "They" if plural_subject else "It",
        "its": "their" if plural_subject else "its",
        "Its": "Their" if plural_subject else "Its",
        "is": "are" if plural_subject else "is",
        "Is": "Are" if plural_subject else "Is",
        "does": "do" if plural_subject else "does",
    }


def known_turn(rng):
    """A question about something real, answered with the plain fact."""
    thing = rng.choice(list(FACTS))
    parts = words(thing)
    ask = rng.choice(ASK).format(**parts)
    answer = say_fact(rng, thing)
    # A plural subject already carries its predicate in the plain form, which is what
    # "they grip harder than fingers can" needs.
    fact = FACTS[thing]
    if thing in PLURAL_ONLY and inflectable(fact):
        head, _, rest = fact.partition(" ")
        fact = (base_form(head) + " " + rest).strip()
    thought = rng.choice(KNOWN_THOUGHT).format(
        fact=fact, **parts,
        # The working gets the same treatment. A thought that always contains the
        # predicate verbatim is another 200 fixed strings to recite.
        said=say_fact(rng, thing, with_after=False))
    return ask, dress(rng, answer), thought


def unknown_turn(rng):
    """A question about something invented, answered by saying so."""
    thing = rng.choice(NONSENSE)
    ask = rng.choice(UNKNOWN_ASK).format(thing=thing, a=article(thing),
                                         them=plural(thing))
    thought = rng.choice(UNKNOWN_THOUGHT).format(
        thing=thing, Thing=thing.capitalize(), a=article(thing))
    return ask, dress(rng, rng.choice(DONT_KNOW), refusal=True), thought


def clarify_turn(rng):
    thing = rng.choice(list(FACTS))
    ask = rng.choice(CLARIFY_ASK).format(**words(thing))
    thought = rng.choice(CLARIFY_THOUGHT).format(**words(thing))
    reply = rng.choice(CLARIFY).format(**words(thing))
    return ask, dress(rng, reply), thought


def chit_turn(rng):
    ask, replies = rng.choice(CHIT)
    return ask, dress(rng, rng.choice(replies)), rng.choice(CHIT_THOUGHT)


def social_turn(rng):
    ask, reply = compose.social_exchange(rng)
    return ask, reply, rng.choice(SOCIAL_THOUGHT)


def follow_up(rng, thing):
    """A second turn about the same thing, so continuing means reading the context."""
    ask, replies = rng.choice(FOLLOW)
    if ask == "What breaks on one?":
        ask = f"What breaks on {article(thing)} {thing}?"
    thought = (f"Still about the {thing}. They are pushing for more than I have - "
               f"answer briefly and do not invent.")
    return ask, dress(rng, rng.choice(replies).format(after=rng.choice(AFTER))), thought


# Things it genuinely cannot know, as opposed to words it was never taught. These
# matter more than the invented ones: a real person asks about the news, or about
# themselves, and a confident answer would be a lie rather than a mistake.
CANNOT_KNOW = [
    "What is the date today?", "What time is it?", "What day is it?",
    "What is the news today?", "Who won the game last night?",
    "What is the weather like?", "Is it raining where you are?",
    "What happened yesterday?", "Who is the president?", "What year is it?",
    "What is my name?", "How old am I?", "Where do I live?", "What did I say earlier?",
    "Do you know who I am?", "What was the last thing I asked you?",
    "What did we talk about last time?", "Can you look something up for me?",
    "Can you check a website?", "What is on at the cinema?",
    "How much does a train ticket cost?", "Is the shop open?",
    "What is the score?", "Did anything important happen this week?",
]
CANNOT_REPLIES = [
    "I have no way to know that. Nothing reaches me from outside this conversation.",
    "I cannot see anything beyond what you type, so I do not know.",
    "No idea - I have no clock, no calendar and no connection to anything.",
    "That is outside what I can know. I would only be inventing it.",
    "I do not know. I have nothing to check against.",
    "Not something I can find out. I only have what is on screen.",
    "I would be making that up. Better to say I do not know.",
    "Nothing about the outside world reaches me, so I genuinely cannot say.",
    "You would have to tell me. I cannot look anything up.",
    "I do not know that yet. Say it and I will use it while we talk.",
]


def cannot_know_turn(rng):
    """Something real that it has no way to know, answered honestly."""
    ask = rng.choice(CANNOT_KNOW)
    what = WHAT_ASKED[ask]
    thought = rng.choice(CANNOT_THOUGHT).format(what=what,
                                                What=what[0].upper() + what[1:])
    return ask, dress(rng, rng.choice(CANNOT_REPLIES), refusal=True), thought


TURNS = [
    (known_turn, 34),
    (chit_turn, 14),
    (unknown_turn, 16),
    (cannot_know_turn, 16),
    (clarify_turn, 10),
    (social_turn, 14),
]


# ---------------------------------------------------------------- conversations
# A long conversation needs a thread, or it is just short ones stapled together.
# One subject persists across two or three turns, referred to by pronoun, before
# something else comes up - which is also the only way a model learns to read back
# through its context rather than answer the last line in isolation.

# How many exchanges. Weighted to average about nine, with short and long both
# common, so no single length becomes the expected shape of a conversation.
LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
LENGTH_WEIGHTS = [3, 4, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Referring back to the subject without naming it, which is what makes it a thread.
ABOUT_IT = [
    "Is that all it does?", "What is it made of, usually?", "Does it wear out?",
    "How long do they last?", "Do I need one?", "Anything I should watch for?",
    "Is there a better option?", "Why that way round?", "Can it break?",
    "What goes wrong with them?", "Is it worth buying a good one?",
    "How would I choose one?", "Would a cheap one do?", "Say more about that.",
    "Interesting - go on.", "Hm. And?", "Really?", "How come?",
]
ABOUT_IT_REPLY = [
    "More or less. {after}", "Metal or wood, depending on the age of it.",
    "Slowly, if you look after it.", "Years, unless you are unlucky.",
    "Only if the job comes up.", "Whatever takes the load is what fails.",
    "Not that I know of, for this.", "Fewest parts wins, usually.",
    "Anything can break. That one is hard to break by accident.",
    "The join goes first, nine times in ten.",
    "A plain good one, yes. A clever one, no.",
    "Right size before anything else.", "For occasional use, easily.",
    "That is about the whole of it, honestly.",
    "There is not much more to it than that.",
    "I have told you the useful part already.",
]

# Getting off one subject and onto another, the way people actually do it.
PIVOTS = ["", "", "", "Anyway. ", "Different question - ", "Changing tack, ",
          "One more thing. ", "Right, separate thing. ", "Also, ", "Oh - "]

OPENINGS = ["Hello", "Hi", "Hey", "Morning", "Evening", "Hello again",
            "Hi, got a minute?", "Hey, you awake?", "Are you there?"]
OPENING_REPLIES = ["Hello. What do you need?", "Hi. Go ahead.", "Hey. What is up?",
                   "Morning. Ask away.", "Evening. What can I do?",
                   "Here. What are you after?", "Yes, here. Say the thing."]
CLOSINGS = ["Thanks", "Thank you", "Cheers", "That helps, thanks", "Got it, thanks",
            "Right, thanks", "OK, bye", "See you", "That is all, thanks"]
CLOSING_REPLIES = ["Any time.", "No trouble.", "Glad it helped.", "Welcome.",
                   "Good luck with it.", "See you.", "Take care.", "Come back if it breaks."]
OPENING_THOUGHT = ["Just a greeting. Say hello and leave room for them.",
                   "Nothing to work out - be brief and friendly."]
CLOSING_THOUGHT = ["They are done. Acknowledge it and stop.",
                   "Wrapping up. Short and warm, nothing more."]
ABOUT_IT_THOUGHT = [
    "Still on the {thing}. They want a bit more, and I do not have much more.",
    "Following up about the {thing}. Answer briefly and do not invent detail.",
    "This is still the {thing}. Say the small extra thing I actually know.",
    "They are pushing on the {thing}. Be honest that there is not much more.",
]


def about_it(rng, thing):
    """A follow-up that refers to the subject without naming it."""
    ask = rng.choice(ABOUT_IT)
    reply = rng.choice(ABOUT_IT_REPLY).format(after=rng.choice(AFTER))
    thought = rng.choice(ABOUT_IT_THOUGHT).format(thing=thing)
    return ask, dress(rng, reply), thought


# Greetings belong at the edges of a conversation. Drawn mid-thread they produced
# exchanges that said hello three times in eight turns.
MID_CONVERSATION = [maker for maker, weight in TURNS for _ in range(weight)
                    if maker is not social_turn]


def subject_of(ask):
    """Which thing a question was about, if any, so a follow-up can refer to it."""
    lowered = ask.lower()
    best = None
    for thing in FACTS:
        if thing in lowered and (best is None or len(thing) > len(best)):
            best = thing
    return best


def conversation(rng, sum_turn=None, sums=0.3, length=None):
    """[(user, reply, thought, plain), ...] - one conversation that hangs together.

    `sum_turn`, if given, is called to produce an arithmetic turn as
    (question, answer, working) - the caller decides whether arithmetic is wanted,
    which is what keeps this file about conversation."""
    wanted = length or rng.choices(LENGTHS, weights=LENGTH_WEIGHTS)[0]
    turns = []

    if rng.random() < 0.5:
        i = rng.randrange(len(OPENINGS))
        turns.append((OPENINGS[i], rng.choice(OPENING_REPLIES),
                      rng.choice(OPENING_THOUGHT), None))

    while len(turns) < wanted:
        if sum_turn and rng.random() < sums:
            question, answer, working = sum_turn(rng)
            turns.append((question, answer, working, working))
            continue

        maker = rng.choice(MID_CONVERSATION)
        ask, reply, thought = maker(rng)
        # One draw, used. Drawing twice - once to test, once to prepend - meant the
        # prefix that was checked was not the prefix that got used.
        pivot = rng.choice(PIVOTS) if turns else ""
        if pivot:
            ask = (pivot + ask[0].lower() + ask[1:]
                   if ask[:1].isupper() and not ask.startswith("I ") else pivot + ask)
        turns.append((ask, reply, thought, None))

        # stay on the subject for a turn or two, referring back to it
        thing = subject_of(ask)
        if thing and maker is known_turn:
            for _ in range(rng.choices([0, 1, 2, 3], weights=[38, 34, 20, 8])[0]):
                if len(turns) >= wanted:
                    break
                turns.append(about_it(rng, thing) + (None,))

    if rng.random() < 0.45:
        turns.append((rng.choice(CLOSINGS), rng.choice(CLOSING_REPLIES),
                      rng.choice(CLOSING_THOUGHT), None))

    # plain form defaults to the reply: for chat it is the reply that answers
    return [(ask, reply, thought, plain if plain is not None else reply)
            for ask, reply, thought, plain in turns]


if __name__ == "__main__":
    rng = random.Random(4)
    for _ in range(3):
        turns = conversation(rng)
        print(f"  -- {len(turns)} exchanges --")
        for user, reply, thought, _ in turns:
            print(f"  you:      {user}")
            print(f"  thinking: {thought}")
            print(f"  peitho:   {reply}")
        print()
