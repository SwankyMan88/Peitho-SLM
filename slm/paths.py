"""Where everything lives, so no script has to guess.

Defaults are absolute and derived from this file's location, which means a script
works the same whether it is run from the repository root, from inside its own
folder, or from anywhere else.

    py slm/train.py                     from the root
    py corpus/chat/make_corpus.py       from the root
    py ../slm/train.py                  from somewhere else entirely
"""

import os
import sys

# The engine lives in slm/, so the project root is one level up.
ENGINE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENGINE)

CORPUS = os.path.join(ROOT, "corpus")          # everything that writes training text
CHAT = os.path.join(CORPUS, "chat")            # the conversational corpus and its sources
GREETER = os.path.join(CORPUS, "greeter")      # greetings, for the tiny opening model
MODELS = os.path.join(ROOT, "models")          # exports, the portable artifacts
BUILD = os.path.join(ROOT, "build")            # everything regenerable

CONVERSATIONS = os.path.join(CHAT, "conversations.txt")
GREETINGS = os.path.join(BUILD, "greetings.txt")
GREETINGS_HELD = os.path.join(BUILD, "greetings_held.txt")
TRAINING = os.path.join(BUILD, "training.txt")
HELDOUT = os.path.join(BUILD, "heldout.txt")
CHECKPOINT = os.path.join(BUILD, "model_full.pt")

PAGE = os.path.join(ROOT, "peitho.html")


def ensure_build():
    os.makedirs(BUILD, exist_ok=True)
    return BUILD


def on_path():
    """Let a script elsewhere import the engine modules."""
    if ENGINE not in sys.path:
        sys.path.insert(0, ENGINE)


def short(path):
    """A path as someone would type it, for printing."""
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return path
