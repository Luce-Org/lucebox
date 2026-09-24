#!/usr/bin/env python3
"""Check that a pull request's commit messages follow the conventional commit style.

    type(scope): summary

    Optional body, separated by a blank line.

Types follow CONTRIBUTING.md plus `build` and `revert`, which main already uses.
The scope is required and lowercase. Merge commits and GitHub's `Revert "..."`
subjects are accepted as they are.

Usage: check_commit_messages.py --base origin/main --head HEAD
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "bench",
    "build",
    "ci",
    "chore",
    "revert",
)
SUBJECT = re.compile(rf"^(?:{'|'.join(TYPES)})\([a-z0-9._/,-]+\)!?: \S")
GITHUB_REVERT = re.compile(r'^Revert ".+"$')
MAX_SUBJECT = 100


def problems(message: str) -> list[str]:
    lines = message.rstrip("\n").split("\n")
    subject = lines[0]
    if GITHUB_REVERT.match(subject):
        return []
    found = []
    if not SUBJECT.match(subject):
        found.append(
            "the subject must look like `type(scope): summary` with type one of "
            + ", ".join(TYPES)
            + " and a required lowercase scope"
        )
    if len(subject) > MAX_SUBJECT:
        found.append(f"the subject is {len(subject)} characters (at most {MAX_SUBJECT})")
    if subject.endswith("."):
        found.append("the subject ends with a period")
    if len(lines) > 1 and lines[1].strip():
        found.append("a blank line must separate the subject from the body")
    return found


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args(argv)

    base = git("merge-base", args.base, args.head).strip()
    shas = git("rev-list", "--reverse", "--no-merges", f"{base}..{args.head}").split()
    bad = 0
    report = ["| Commit | Subject | Problem |", "|---|---|---|"]
    for sha in shas:
        message = git("log", "-1", "--format=%B", sha)
        subject = message.split("\n", 1)[0]
        for problem in problems(message):
            bad += 1
            print(f"::error title=Commit {sha[:10]}::{subject}: {problem}")
            report.append(f"| `{sha[:10]}` | {subject.replace('|', '/')} | {problem} |")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if bad:
        text = "\n".join(
            [
                f"## Commit style: {bad} problem(s)",
                "",
                *report,
                "",
                "Fix them with `git rebase -i` (reword), or add the `compact-commits` label: the bot "
                "folds the commits and rewrites every message in this style.",
            ]
        )
        print(text)
    else:
        text = f"## Commit style: all {len(shas)} commit(s) OK"
        print(text)
    if summary:
        with open(summary, "a") as fh:
            fh.write(text + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
