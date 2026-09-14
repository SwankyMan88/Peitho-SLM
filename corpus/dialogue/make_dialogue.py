"""Build conversations from two human-written dialogue datasets.

Sources, downloaded as repository zips from GitHub into build/datasets/:
  Schema-Guided Dialogue   google-research-datasets/dstc8-schema-guided-dialogue  CC BY-SA 4.0
  MultiWOZ 2.2             budzianowski/multiwoz                                  MIT

People wrote both, turn by turn, so the replies are phrased a thousand ways where
a generator phrases them a handful - which is the point of adding them.

Every model turn thinks first, as the rest of the corpus does. Both datasets label
what each system turn does ("request the city", "offer Bedouin"), so the thought is
built from those labels: a short plan grounded in what the reply actually does,
rather than a fixed sentence repeated over every turn.

    py corpus/dialogue/make_dialogue.py        -> build/dialogues.json
"""

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "slm"))
import paths
from model import START_MARK, USER_MARK, BOT_MARK, END_MARK, THINK_MARK

sys.path.insert(0, os.path.join(paths.CORPUS, "alpaca"))
from make_alpaca import clean

DATASETS = os.path.join(paths.BUILD, "datasets")


# Slot names that read badly as English once the underscores are gone.
RENAME = {"to": "destination", "from": "origin", "to station": "arrival station",
          "from station": "departure station", "to location": "destination",
          "from location": "starting point", "to city": "destination city",
          "from city": "departure city", "leaveat": "departure time",
          "arriveby": "arrival time", "ref": "reference number", "addr": "address"}


def words(slot):
    """"number_of_seats" -> "number of seats", "GetTrainTickets" -> "get train tickets"."""
    slot = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", slot)
    slot = slot.replace("_", " ").replace("-", " ").strip().lower()
    return RENAME.get(slot, slot)


def listing(items):
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def sgd_plan(actions):
    """Schema-Guided Dialogue labels each system turn with actions like
    {"act": "OFFER", "slot": "restaurant_name", "values": ["Sakoon"]}."""
    asks, tells, offers, notes = [], [], [], []
    for a in actions:
        act, slot, values = a["act"], words(a.get("slot", "")), a.get("values", [])
        value = values[0] if values else ""
        if act == "REQUEST":
            asks.append(slot)
        elif act in ("INFORM", "CONFIRM") and value:
            tells.append(f"the {slot} is {value}")
        elif act == "OFFER" and value:
            offers.append(value)
        elif act == "INFORM_COUNT" and value:
            notes.append(f"I found {value}.")
        elif act == "OFFER_INTENT" and value:
            notes.append(f"Offer to {words(value)}.")
        elif act == "NOTIFY_SUCCESS":
            notes.append("It went through - say so.")
        elif act == "NOTIFY_FAILURE":
            notes.append("It did not go through - say so.")
        elif act == "REQ_MORE":
            notes.append("Ask if they need anything else.")
        elif act == "GOODBYE":
            notes.append("Say goodbye.")
    return compose_plan(asks, tells, offers, notes,
                        confirm=any(a["act"] == "CONFIRM" for a in actions))


def multiwoz_plan(acts):
    """MultiWOZ labels each system turn as {"Restaurant-Inform": [["food", "Thai"]], ...}."""
    asks, tells, offers, notes = [], [], [], []
    for key, pairs in acts.items():
        act = key.split("-")[-1].lower()
        for slot, value in pairs:
            slot = words(slot)
            value = "" if value in ("?", "none", "") else value
            if act == "request":
                asks.append(slot)
            elif act in ("inform", "book", "offerbooked") and value and slot != "none":
                tells.append(f"the {slot} is {value}")
            elif act in ("recommend", "select") and value:
                offers.append(value)
        if act == "nooffer":
            notes.append("Nothing matches - say so.")
        elif act == "nobook":
            notes.append("It could not be booked - say so.")
        elif act == "offerbook":
            notes.append("Offer to book it.")
        elif act == "reqmore":
            notes.append("Ask if they need anything else.")
        elif act == "bye":
            notes.append("Say goodbye.")
        elif act in ("welcome", "greet"):
            notes.append("Be friendly.")
    return compose_plan(asks, tells, offers, notes)


def compose_plan(asks, tells, offers, notes, confirm=False):
    parts = []
    if confirm:
        parts.append("Check the details before going ahead: " + listing(tells[:4]) + ".")
        tells = []
    if offers:
        parts.append("Offer " + listing(list(dict.fromkeys(offers))[:3]) + ".")
    if tells:
        parts.append("Tell them " + listing(list(dict.fromkeys(tells))[:4]) + ".")
    if asks:
        parts.append("Ask for the " + listing(list(dict.fromkeys(asks))[:3]) + ".")
    parts += list(dict.fromkeys(notes))
    plan = " ".join(parts)
    return plan or "Answer them plainly."


def conversation(turns):
    """[(speaker, utterance, plan-or-None)] -> one finished conversation, or None."""
    lines = [START_MARK]
    for speaker, text, plan in turns:
        text = clean(text)
        if text is None:
            return None
        if speaker == "USER":
            lines.append(USER_MARK + text + END_MARK)
        else:
            plan = clean(plan)
            if plan is None:
                return None
            lines.append(BOT_MARK + plan + THINK_MARK + text + END_MARK)
    # A conversation should open with the person and give the model the last word.
    while len(lines) > 1 and not lines[1].startswith(USER_MARK):
        del lines[1]
    while len(lines) > 1 and not lines[-1].startswith(BOT_MARK):
        lines.pop()
    return "\n".join(lines) if len(lines) >= 3 else None


def load_sgd(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "dialogues_*.json"))):
        for d in json.load(open(path, encoding="utf-8")):
            turns = []
            for t in d["turns"]:
                plan = None
                if t["speaker"] == "SYSTEM":
                    plan = sgd_plan([a for f in t["frames"] for a in f.get("actions", [])])
                turns.append((t["speaker"], t["utterance"], plan))
            out.append(conversation(turns))
    return out


def load_multiwoz(root):
    acts = json.load(open(os.path.join(root, "dialog_acts.json"), encoding="utf-8"))
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "dialogues_*.json"))):
        for d in json.load(open(path, encoding="utf-8")):
            labels = acts.get(d["dialogue_id"], {})
            turns = []
            for t in d["turns"]:
                plan = None
                if t["speaker"] == "SYSTEM":
                    plan = multiwoz_plan(labels.get(t["turn_id"], {}).get("dialog_act", {}))
                turns.append((t["speaker"], t["utterance"], plan))
            out.append(conversation(turns))
    return out


def main():
    p = argparse.ArgumentParser(description="Convert SGD and MultiWOZ to the corpus format.")
    p.add_argument("--sgd", default=os.path.join(DATASETS, "sgd"))
    p.add_argument("--multiwoz", default=os.path.join(DATASETS, "multiwoz"))
    p.add_argument("--out", default=os.path.join(paths.BUILD, "dialogues.json"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    found = {}
    for name, loader, root in (("sgd", load_sgd, args.sgd),
                               ("multiwoz", load_multiwoz, args.multiwoz)):
        if not os.path.isdir(root):
            print(f"  {name}: {paths.short(root)} not found, skipped")
            continue
        convs = loader(root)
        kept = [c for c in convs if c]
        found[name] = kept
        print(f"  {name:9} {len(convs):6,} dialogues, {len(kept):6,} kept "
              f"({sum(len(c) for c in kept):,} chars)")

    everything = [c for convs in found.values() for c in convs]
    random.Random(args.seed).shuffle(everything)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(everything, f)
    print(f"\n{len(everything):,} conversations -> {paths.short(args.out)}")
    if everything:
        print("\nexample:\n" + everything[0][:700])


if __name__ == "__main__":
    main()
