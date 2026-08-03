"""Test everything, then push and tag a release - as you, from your machine.

Every commit here is authored by whatever `git config user.name/user.email` says,
and every push authenticates with whatever credential helper git is already using.
This script adds no identity of its own and handles no tokens; it runs the checks
in the right order and refuses to push when one fails, which is the part that is
easy to get wrong at the end of a long day.

    py tools/release.py                       checks only, nothing leaves the machine
    py tools/release.py --push                checks, then push main
    py tools/release.py --push --tag v1.6.0   checks, push, tag, push the tag
    py tools/release.py --commit "..." --push  commit everything first, then push

Commits made with --commit carry the message exactly as given and no trailers. A
Co-Authored-By line naming a tool makes that tool a contributor to the repository on
GitHub, which is not what a tool should be in someone else's project.

The GitHub Release itself is not created here. That needs an API token, and a
script that reads your credentials to post on your behalf is worse than opening a
URL - so the URL is printed instead, with the tag already filled in.
"""

import argparse
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slm"))
import paths
import versions

PY = sys.executable

# Every suite, in the order that fails fastest first.
SUITES = [
    ([PY, "-X", "utf8", "tests/learning/test_learn.py"], "live teaching"),
    ([PY, "-X", "utf8", "tests/conformance/test_conformance.py"], "decoder conformance"),
    ([PY, "-X", "utf8", "tests/pipeline/test_slm.py"], "corpus to export pipeline"),
    (["node", "tests/pages/check_pages.mjs"], "the browser pages"),
]


def run(command, **kw):
    return subprocess.run(command, cwd=paths.ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def git(*args):
    done = run(["git", *args])
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{done.stderr.strip()}")
    return done.stdout.strip()


def notes_for(tag, path=None):
    """The section of docs/releases.md that belongs to this tag.

    Used as the tag's own message, so `git show v1.6.0` explains itself even
    without GitHub."""
    path = path or os.path.join(paths.ROOT, "docs", "releases.md")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    version = tag.lstrip("v")
    match = re.search(rf"^## {re.escape(version)}\b(.*?)(?=^## |\Z)", text,
                      re.MULTILINE | re.DOTALL)
    return match.group(0).strip() if match else ""


def cdn_check(tag, owner_repo):
    """Ask jsDelivr for each export at the new tag.

    A tag that exists on GitHub is not the same as a tag jsDelivr will serve, and
    the page loads models by tag - so this is worth knowing before anyone clicks a
    release announcement."""
    base = f"https://cdn.jsdelivr.net/gh/{owner_repo}@{tag}/models/"
    for name, version in versions.existing():
        filename = os.path.basename(versions.path_for(name, version))
        url = base + filename
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=60) as answer:
                size = answer.headers.get("content-length", "?")
                print(f"  {answer.status}  {filename}  ({size} bytes)")
        except urllib.error.HTTPError as e:
            print(f"  {e.code}  {filename}  <- not served")
        except Exception as e:
            # A cold cache times out on the first request and answers on the next.
            print(f"  ??   {filename}  ({e})")


def main():
    p = argparse.ArgumentParser(description="Run every check, then push a release.")
    p.add_argument("--push", action="store_true",
                   help="Push main. Without this, nothing leaves the machine.")
    p.add_argument("--tag", default="",
                   help="Annotated tag to create and push, for example v1.6.0. Its "
                        "message comes from the matching section of docs/releases.md.")
    p.add_argument("--remote", default="origin")
    p.add_argument("--branch", default="main")
    p.add_argument("--commit", default="",
                   help="Commit everything with this message first. The message is "
                        "used exactly as given, with no trailers added - the commit is "
                        "yours and says so.")
    p.add_argument("--allow_dirty", action="store_true",
                   help="Push with uncommitted changes present. They will not be "
                        "included, which is usually not what anyone wants.")
    p.add_argument("--skip_tests", action="store_true",
                   help="Skip the suites. For a documentation-only push you are sure "
                        "about.")
    p.add_argument("--check_cdn", action="store_true",
                   help="After tagging, ask jsDelivr whether it serves the exports.")
    args = p.parse_args()

    # Who this will be. Printed rather than assumed, because the answer to "whose
    # profile did that push come from" is entirely in these three lines.
    name = git("config", "user.name")
    email = git("config", "user.email")
    remote_url = git("remote", "get-url", args.remote)
    helper = run(["git", "config", "--get", "credential.helper"]).stdout.strip()
    print(f"Author:     {name} <{email}>")
    print(f"Remote:     {remote_url}")
    print(f"Credentials: {helper or 'none configured - git will prompt'}")

    owner_repo = ""
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
    if match:
        owner_repo = match.group(1)

    dirty = git("status", "--porcelain")
    if dirty and args.commit:
        # No Co-Authored-By, no Generated-with, nothing appended. GitHub counts a
        # co-author as a contributor to the repository, so a trailer naming a tool
        # puts that tool in the contributor list of a project it does not own.
        git("add", "-A")
        git("commit", "-m", args.commit)
        print(f"\nCommitted as {name}: {git('log', '-1', '--oneline')}")
        dirty = git("status", "--porcelain")

    if dirty:
        print("\nUncommitted changes:")
        for line in dirty.splitlines():
            print(f"  {line}")
        if not args.allow_dirty:
            raise SystemExit("\nCommit them first, or pass --allow_dirty to push "
                             "without them.")

    if not args.skip_tests:
        print()
        for command, label in SUITES:
            print(f"Running {label}...", end=" ", flush=True)
            done = run(command)
            if done.returncode != 0:
                print("FAILED\n")
                print(done.stdout[-4000:] or done.stderr[-4000:])
                raise SystemExit(f"{label} failed. Nothing has been pushed.")
            summary = [l for l in done.stdout.splitlines() if "passed" in l]
            print(summary[-1].strip() if summary else "ok")

    ahead = git("log", f"{args.remote}/{args.branch}..HEAD", "--oneline")
    print(f"\nCommits to push: {len(ahead.splitlines()) if ahead else 0}")
    for line in ahead.splitlines():
        print(f"  {line}")

    if not args.push:
        print("\nChecks only - nothing pushed. Add --push when you are ready.")
        return

    print(f"\nPushing {args.branch} to {args.remote}...")
    print(git("push", args.remote, args.branch) or "  up to date")

    if args.tag:
        existing = git("tag", "--list", args.tag)
        if existing:
            raise SystemExit(f"Tag {args.tag} already exists. Pick the next version, "
                             "or delete it deliberately with git tag -d.")
        message = notes_for(args.tag) or f"{args.tag.lstrip('v')}"
        if not notes_for(args.tag):
            print(f"  NOTE: docs/releases.md has no section for "
                  f"{args.tag.lstrip('v')}; tagging with a bare version message.")
        git("tag", "-a", args.tag, "-m", message)
        git("push", args.remote, args.tag)
        print(f"Tagged and pushed {args.tag}")

        if args.check_cdn and owner_repo:
            print(f"\njsDelivr at {args.tag}:")
            cdn_check(args.tag, owner_repo)

        if owner_repo:
            print(f"\nWrite the release notes here:\n"
                  f"  https://github.com/{owner_repo}/releases/new?tag={args.tag}")
            print(f"  The body is the {args.tag.lstrip('v')} section of "
                  f"{paths.short(os.path.join(paths.ROOT, 'docs', 'releases.md'))}.")


if __name__ == "__main__":
    main()
