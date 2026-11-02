#!/usr/bin/env python3
"""Pick what the lucebox3 jobs run for a pull request, from the files it changes.

Prints `gpu=`, `qwen38=` and `ds41=` lines for $GITHUB_OUTPUT, each `true` or
`false`: the per-device GPU tests, and the serve checks of Qwen 3.8 27B and of
DeepSeek V4.1 Flash. Each changed path takes the first rule in RULES that matches
it, and the pull request runs every job any of its paths needs. A path no rule
names runs everything; so does --full (pushes to main, manual runs), and so does
a failure to list the changed files.

Usage: ci_plan.py --base HEAD^1 --head HEAD   (a pull request's merge commit)
       ci_plan.py --full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

JOBS = {
    "gpu": "GPU tests (R9700 and Strix Halo)",
    "qwen38": "Serve Qwen 3.8 27B",
    "ds41": "Serve DeepSeek V4.1 Flash",
}
ALL = frozenset(JOBS)
GPU = frozenset({"gpu"})
QWEN38 = frozenset({"gpu", "qwen38"})
DS41 = frozenset({"gpu", "ds41"})
NONE: frozenset[str] = frozenset()

# (pattern, jobs, what the paths are). A pattern ending in "/" matches everything
# under that directory, "*.md" matches by suffix, and anything else is one file.
RULES = [
    (".github/", ALL, "the workflow and its scripts"),
    # Each model's own code. qwen35 and deepseek4 include none of each other's
    # headers, and at link level (nm over a luce_server build, 2026-10-29) DeepSeek
    # links only gguf_target_loader.cpp from qwen35 (CpuEmbedder, which its prefill
    # and DSpark embed with); deepseek4's files are linked only by backend_factory
    # and the MoE IPC path, which dense Qwen 3.8 doesn't run. Other model
    # directories run everything: qwen35's graph includes qwen35moe and
    # bailingmoe3 headers, luce_common includes laguna's.
    ("server/src/qwen35/gguf_target_loader.cpp", ALL, "code both models link"),
    ("server/src/qwen35/", QWEN38, "Qwen 3.8 27B's code"),
    ("server/src/deepseek4/", DS41, "DeepSeek V4.1's code"),
    ("server/share/deepseek41/", DS41, "the ds41-halo profile's files"),
    # The model cards: sampler defaults and launch profiles. Only the served
    # models' cards change what the serve checks run.
    ("share/model_cards/qwen3.8-27b.json", QWEN38, "Qwen 3.8 27B's model card"),
    ("share/model_cards/deepseek-v4.1-flash.json", DS41, "DeepSeek V4.1 Flash's model card"),
    ("share/model_cards/", NONE, "cards of models the jobs don't serve"),
    ("server/test/", GPU, "tests (luce_server links none of them)"),
    # Nothing the lucebox3 jobs build or read.
    ("*.md", NONE, "documentation"),
    ("docs/", NONE, "documentation"),
    ("server/docs/", NONE, "documentation"),
    ("assets/", NONE, "images"),
    ("server/hero.png", NONE, "images"),
    ("server/hero.raw.png", NONE, "images"),
    ("server/demo.gif", NONE, "images"),
    ("harness/", NONE, "Python tooling outside the engine"),
    ("optimizations/", NONE, "Python tooling outside the engine"),
    ("scripts/", NONE, "Python tooling outside the engine"),
    ("conftest.py", NONE, "Python tooling outside the engine"),
    ("pyproject.toml", NONE, "Python tooling outside the engine"),
    ("uv.lock", NONE, "Python tooling outside the engine"),
    (".python-version", NONE, "Python tooling outside the engine"),
    ("server/pyproject.toml", NONE, "Python tooling outside the engine"),
    ("server/scripts/", NONE, "engine scripts and tools the jobs don't build"),
    ("server/tools/", NONE, "engine scripts and tools the jobs don't build"),
    ("server/eval/", NONE, "engine scripts and tools the jobs don't build"),
    ("server/examples/", NONE, "engine scripts and tools the jobs don't build"),
    ("Dockerfile", NONE, "container and repository files"),
    ("Dockerfile.rocm", NONE, "container and repository files"),
    ("docker-bake.hcl", NONE, "container and repository files"),
    (".dockerignore", NONE, "container and repository files"),
    (".gitignore", NONE, "container and repository files"),
    ("LICENSE", NONE, "container and repository files"),
    ("Makefile", NONE, "container and repository files"),
]
UNKNOWN = "code no rule narrows (shared code, other models, build files)"


def matches(path: str, pattern: str) -> bool:
    if pattern.startswith("*"):
        return path.endswith(pattern[1:])
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern


def plan(paths: list[str] | None) -> tuple[frozenset[str], dict[str, list[str]]]:
    """The jobs to run, and the changed paths grouped by what decided them."""
    if not paths:
        return ALL, {"no changed files listed": []}
    jobs: set[str] = set()
    reasons: dict[str, list[str]] = {}
    for path in paths:
        rule = next((rule for rule in RULES if matches(path, rule[0])), None)
        needs, why = (rule[1], rule[2]) if rule else (ALL, UNKNOWN)
        jobs |= needs
        reasons.setdefault(why, []).append(path)
    return frozenset(jobs), reasons


def changed_files(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z", base, head],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--full", action="store_true", help="run everything")
    args = parser.parse_args(argv)

    if args.full:
        jobs, reasons = ALL, {"a full run (a push to main or a manual run)": []}
    else:
        if not (args.base and args.head):
            parser.error("give --base and --head, or --full")
        paths = None
        try:
            paths = changed_files(args.base, args.head)
        except (OSError, subprocess.CalledProcessError) as error:
            # stdout is $GITHUB_OUTPUT, so the warning goes to stderr.
            detail = " ".join(str(getattr(error, "stderr", "") or error).split())
            print(
                f"::warning::could not list the changed files, so everything runs: {detail}",
                file=sys.stderr,
            )
        jobs, reasons = plan(paths)

    for job in JOBS:
        print(f"{job}={'true' if job in jobs else 'false'}")

    lines = ["## lucebox3 plan", "", "| Job | Runs |", "|---|---|"]
    lines += [f"| {name} | {'yes' if job in jobs else 'no'} |" for job, name in JOBS.items()]
    lines += ["", "Changed files, by what decided them:", ""]
    for why, group in reasons.items():
        shown = ", ".join(f"`{path}`" for path in group[:8])
        more = f" and {len(group) - 8} more" if len(group) > 8 else ""
        lines.append(f"- {why}" + (f" ({len(group)}): {shown}{more}" if group else ""))
    text = "\n".join(lines) + "\n"
    print(text, file=sys.stderr)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
