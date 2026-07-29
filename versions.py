"""Naming and lookup for exported models in the models/ folder.

Exports are named `<base>_<version>.txt`, for example `small_1.0.txt`. A new export
takes the next version after the highest one present, so nothing is ever
overwritten:

    (empty folder)                  -> small_1.0
    small_1.0                       -> small_1.1
    small_1.0, small_1.1            -> small_1.2

If that name is already taken - usually because a file was renamed or copied in by
hand - the new export nests underneath rather than clobbering it:

    small_1.0, small_1.1, small_1.2 -> small_1.3
    ...and if small_1.3 also exists -> small_1.3.1, then small_1.3.2
"""

import os
import re

import paths

MODELS_DIR = paths.MODELS
NAME = re.compile(r"^(?P<base>.+)_(?P<version>\d+(?:\.\d+)*)\.txt$")


def parse(filename):
    """('small_1.2.txt') -> ('small', (1, 2)), or None if it is not an export name."""
    match = NAME.match(os.path.basename(filename))
    if not match:
        return None
    version = tuple(int(part) for part in match.group("version").split("."))
    return match.group("base"), version


def format_version(version):
    return ".".join(str(part) for part in version)


def path_for(base, version, folder=MODELS_DIR):
    return os.path.join(folder, f"{base}_{format_version(version)}.txt")


def existing(base=None, folder=MODELS_DIR):
    """Every (base, version) export present, oldest version first."""
    if not os.path.isdir(folder):
        return []
    found = []
    for name in os.listdir(folder):
        parsed = parse(name)
        if parsed and (base is None or parsed[0] == base):
            found.append(parsed)
    return sorted(found, key=lambda item: (item[0], item[1]))


def next_path(base, folder=MODELS_DIR):
    """Where the next export of `base` should be written."""
    os.makedirs(folder, exist_ok=True)
    versions = [version for _, version in existing(base, folder)]
    if not versions:
        return path_for(base, (1, 0), folder)

    highest = max(versions)
    candidate = highest[:-1] + (highest[-1] + 1,)
    while os.path.exists(path_for(base, candidate, folder)):
        if len(candidate) == len(highest):
            candidate = candidate + (1,)      # nest under the taken name
        else:
            candidate = candidate[:-1] + (candidate[-1] + 1,)
    return path_for(base, candidate, folder)


def resolve(reference, folder=MODELS_DIR):
    """Turn a user's model argument into a path.

    Accepts any of the ways someone might reasonably name one:

        models/small_1.2.txt   a path
        small_1.2.txt          a filename in the models folder
        small_1.2              a particular version
        small                  a base name, meaning its highest version
        "" or None             the most recently written export of any base
    """
    if reference and os.path.isfile(reference):
        return reference

    if reference:
        for candidate in (os.path.join(folder, reference),
                          os.path.join(folder, reference + ".txt")):
            if os.path.isfile(candidate):
                return candidate

        versions = [v for _, v in existing(reference, folder)]
        if not versions:
            known = ", ".join(f"{b}_{format_version(v)}" for b, v in existing(None, folder))
            raise SystemExit(f"No export matching {reference!r} in {folder}/. "
                             f"There is: {known or 'nothing yet'}.")
        return path_for(reference, max(versions), folder)

    candidates = [path_for(base, version, folder) for base, version in existing(None, folder)]
    if not candidates:
        raise SystemExit(f"No exports found in {folder}/. Train a model first.")
    return max(candidates, key=os.path.getmtime)


if __name__ == "__main__":
    for base, version in existing():
        print(f"  {path_for(base, version)}")
    print(f"\nnext small  -> {next_path('small')}")
    print(f"next medium -> {next_path('medium')}")
