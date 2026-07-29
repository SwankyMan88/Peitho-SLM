"""Where everything lives, so no script has to guess.

Defaults are absolute and derived from this file's location, which means a script
works the same whether it is run from the repository root, from inside its own
folder, or from anywhere else.

    py train.py                    from the root
    py corpus/make_corpus.py       from the root
    py ../train.py                 from somewhere else entirely
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

CORPUS = os.path.join(ROOT, "corpus")          # generators and the hand-written text
MODELS = os.path.join(ROOT, "models")          # exports, the portable artifacts
BUILD = os.path.join(ROOT, "build")            # everything regenerable

CONVERSATIONS = os.path.join(CORPUS, "conversations.txt")
TRAINING = os.path.join(BUILD, "training.txt")
HELDOUT = os.path.join(BUILD, "heldout.txt")
CHECKPOINT = os.path.join(BUILD, "model_full.pt")

PAGE = os.path.join(ROOT, "peitho.html")


def ensure_build():
    os.makedirs(BUILD, exist_ok=True)
    return BUILD


def on_path():
    """Let a script in a subfolder import the modules at the root."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def short(path):
    """A path as someone would type it, for printing."""
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return path
