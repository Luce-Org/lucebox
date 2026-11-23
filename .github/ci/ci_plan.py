#!/usr/bin/env python3
"""Pick which CI jobs run for a pull request, from the files it changes.

Prints `<job>=true|false` lines for $GITHUB_OUTPUT covering Python checks,
hosted C++/megakernel/Windows builds, DGX GB10, and the lucebox3 ROCm jobs.
Each changed path takes the first rule in RULES that matches it, and the pull
request runs every job any of its paths needs. A path no rule names runs
everything; so does --full (pushes to main, merge_group, manual runs), and so
does a failure to list the changed files.

Usage: ci_plan.py --base HEAD^1 --head HEAD   (a pull request's merge commit)
       ci_plan.py --full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

JOBS = {
    "python": "uv workspace (lock + lint + Python unit tests)",
    "cpp": "Build Linux (cmake + C++ CPU unit tests)",
    "megakernel": "Build megakernel (CUDAExtension sm_75 + sm_86)",
    "windows": "Build Windows (MSVC+CUDA and ROCm HIP)",
    "dgx": "GPU tests (DGX GB10, sm_121)",
    "gpu": "GPU tests (R9700 and Strix Halo)",
    "qwen38": "Serve Qwen 3.8 27B",
    "ds41": "Serve DeepSeek V4.1 Flash",
}
ALL = frozenset(JOBS)
NONE: frozenset[str] = frozenset()
PY = frozenset({"python"})
CPP = frozenset({"cpp"})
PY_CPP = frozenset({"python", "cpp"})
PY_MK = frozenset({"python", "megakernel"})
CPP_ALL = frozenset({"cpp", "windows", "dgx", "gpu", "qwen38", "ds41"})
CPP_GPU = frozenset({"cpp", "windows", "dgx", "gpu"})
CPP_QWEN38 = frozenset({"cpp", "windows", "dgx", "gpu", "qwen38"})
CPP_DS41 = frozenset({"cpp", "windows", "dgx", "gpu", "ds41"})
DS41_PROFILE = frozenset({"cpp", "gpu", "ds41"})
CARD_QWEN38 = frozenset({"python", "cpp", "gpu", "qwen38"})
CARD_DS41 = frozenset({"python", "cpp", "gpu", "ds41"})
CARD_OTHER = frozenset({"python", "cpp"})

# (pattern, jobs, what the paths are). A pattern ending in "/" matches everything
# under that directory, "*.md" matches by suffix, and anything else is one file.
# Order matters: more specific paths must precede parent directories.
RULES = [
    (".github/workflows/ci.yml", ALL, "the CI workflow"),
    (".github/ci/", ALL, "the CI helper scripts"),
    (".github/workflows/docker.yml", NONE, "unrelated workflow (Docker prebuilds)"),
    (".github/workflows/code-review.yml", NONE, "unrelated workflow (code review bot)"),
    (".github/workflows/compact-commits.yml", NONE, "unrelated workflow (commit compaction bot)"),
    (".github/workflows/ci-fix.yml", NONE, "unrelated workflow (CI fix bot)"),
    (".github/workflows/commit-style.yml", NONE, "unrelated workflow (commit style)"),
    (".github/workflows/speed-profile.yml", NONE, "unrelated workflow (speed profile)"),
    (".github/", ALL, "other GitHub configuration"),
    # Each model's own code. qwen35 and deepseek4 include none of each other's
    # headers, and at link level (nm over a luce_server build, 2026-10-29) DeepSeek
    # links only gguf_target_loader.cpp from qwen35 (CpuEmbedder, which its prefill
    # and DSpark embed with); deepseek4's files are linked only by backend_factory
    # and the MoE IPC path, which dense Qwen 3.8 doesn't run. Other model
    # directories run everything: qwen35's graph includes qwen35moe and
    # bailingmoe3 headers, luce_common includes laguna's.
    ("server/src/qwen35/gguf_target_loader.cpp", CPP_ALL, "code both models link"),
    ("server/src/qwen35/", CPP_QWEN38, "Qwen 3.8 27B's code"),
    ("server/src/deepseek4/", CPP_DS41, "DeepSeek V4.1's code"),
    ("server/share/deepseek41/", DS41_PROFILE, "the ds41-halo profile's files"),
    # The model cards: sampler defaults and launch profiles. Only the served
    # models' cards change what the serve checks run.
    ("share/model_cards/README.md", NONE, "documentation"),
    ("share/model_cards/qwen3.8-27b.json", CARD_QWEN38, "Qwen 3.8 27B's model card"),
    ("share/model_cards/deepseek-v4.1-flash.json", CARD_DS41, "DeepSeek V4.1 Flash's model card"),
    ("share/model_cards/", CARD_OTHER, "cards of models the serve jobs don't run"),
    ("server/test/python/", PY, "Python server tests"),
    ("server/test/", CPP_GPU, "C++ tests (luce_server links none of them)"),
    # Documentation and static media affect no build or test target.
    ("*.md", NONE, "documentation"),
    ("docs/", NONE, "documentation"),
    ("server/docs/", NONE, "documentation"),
    ("assets/", NONE, "images"),
    ("server/hero.png", NONE, "images"),
    ("server/hero.raw.png", NONE, "images"),
    ("server/demo.gif", NONE, "images"),
    # Python workspace and optimizations.
    ("optimizations/megakernel/", PY_MK, "megakernel CUDA extension"),
    ("optimizations/", PY, "Python optimizations outside the C++ engine"),
    # Client launcher scripts and their shell test drivers are registered under
    # ctest `-L cpu` in server/CMakeLists.txt (not pytest).
    ("harness/clients/", PY_CPP, "client launcher scripts (covered by check-cpu shell tests)"),
    ("harness/tests/", PY_CPP, "harness tests (including check-cpu shell drivers)"),
    ("harness/", PY, "Python harness tooling"),
    ("scripts/", PY, "repository scripts"),
    ("conftest.py", PY, "pytest configuration"),
    ("pyproject.toml", PY_MK, "Python workspace configuration"),
    ("uv.lock", PY_MK, "Python lockfile"),
    (".python-version", PY_MK, "Python version pin"),
    ("server/pyproject.toml", PY_MK, "server Python package metadata"),
    ("server/scripts/entrypoint.sh", PY_CPP, "container entrypoint (covered by server_unit_entrypoint)"),
    ("server/scripts/", PY, "engine Python scripts"),
    ("server/tools/export_ds4v_mmproj.py", PY, "DS4V mmproj exporter (covered by test_export_ds4v_mmproj.py)"),
    ("server/tools/ds4_mix_converter/", CPP, "DS4 mix converter (compiled into test_ds4_mix_fitter)"),
    ("server/tools/", NONE, "standalone engine tools CI does not build"),
    ("server/eval/", NONE, "evaluation scripts CI does not run"),
    ("server/examples/chat.py", PY, "chat CLI example (covered by test_chat_stream.py)"),
    ("server/examples/", NONE, "example programs CI does not build"),
    ("Dockerfile", NONE, "container files (covered by docker.yml)"),
    ("Dockerfile.rocm", NONE, "container files (covered by docker.yml)"),
    ("docker-bake.hcl", NONE, "container files (covered by docker.yml)"),
    (".dockerignore", NONE, "container files (covered by docker.yml)"),
    (".gitignore", NONE, "repository metadata"),
    ("LICENSE", NONE, "repository metadata"),
    ("Makefile", PY, "Makefile"),
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
        jobs, reasons = ALL, {"a full run (main, merge_group, or manual dispatch)": []}
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

    lines = ["## CI plan", "", "| Job | Runs |", "|---|---|"]
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
