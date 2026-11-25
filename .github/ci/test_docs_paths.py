"""Check that quick-start shell commands in the docs resolve against this tree.

The quick starts are the first thing a contributor runs, and a moved project
(`megakernel/` -> `optimizations/megakernel/`) silently turned `cd` and
`uv run --directory ...` into "No such file or directory" (see the docs fix in
the same series). This test walks every fenced shell block, tracks the
directory each command runs in, and fails on a path that neither exists in the
tree nor is created earlier in the same block.

Deliberate limits, so the check stays quiet on commands that are correct:

* Only relative paths are checked. Absolute paths, `~`, `$VAR`, globs and
  placeholders like `<model>` are skipped.
* Directories a block creates (`git clone`, `mkdir`, `python -m venv`) count as
  existing, and a clone of this repository maps to the repo root, so
  `git clone .../lucebox.git && cd lucebox` is accepted outside a checkout.
* A small allowlist covers paths the docs create at runtime (model downloads,
  build trees, virtualenvs). Add to it only for directories a command in the
  docs actually creates.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _repo_names() -> set[str]:
    """Names a checkout may have: the workspace (CI) and the clone directory."""
    names = {REPO.name}
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=REPO, capture_output=True, text=True
    )
    if remote.returncode == 0 and remote.stdout.strip():
        url = remote.stdout.strip().rstrip("/").removesuffix(".git")
        names.add(url.rsplit("/", 1)[-1].rsplit(":", 1)[-1])
    return names


REPO_NAMES = _repo_names()

FENCE = re.compile(r"^\s*```(\w*)")
SHELL_LANGS = {"bash", "sh", "console", "shell"}
# Shell syntax this checker does not model: expansion, globs, redirection,
# placeholders, absolute paths.
UNMODELLED = ("$", "~", "*", "<", ">", "|", "&&", ";", "(", ")", "`")

# Created by commands in the docs, not present in a fresh checkout.
RUNTIME_DIRS = {
    "build",
    "server/build",
    "models",
    "server/models",
    "dist",
    "node_modules",
    ".venv",
    ".harness-work",
}


def markdown_blocks(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """Return (language, [(line number, line)]) for each fenced block."""
    blocks: list[tuple[str, list[tuple[int, str]]]] = []
    language: str | None = None
    lines: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            if language is None:
                language = fence.group(1).lower() or "text"
                lines = []
            else:
                blocks.append((language, lines))
                language = None
            continue
        if language is not None:
            lines.append((number, line))
    return blocks


def command_lines(lines: list[tuple[int, str]]):
    """Yield (line number, shlex tokens) for each command in a block."""
    for number, raw in lines:
        line = raw.strip()
        if line.startswith("$ "):
            line = line[2:].strip()
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        # Judge only the command, not a trailing comment: `uv run --directory
        # megakernel python final_bench.py  # runs pp520 (warm)` is checkable.
        if tokens and not any(marker in token for token in tokens for marker in UNMODELLED):
            yield number, tokens


def normalise(path: str) -> Path:
    """Resolve a documented path to a repo-relative one (repo name = root)."""
    candidate = Path(path)
    if candidate.parts and candidate.parts[0] in REPO_NAMES:
        candidate = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path()
    return candidate


def is_runtime_dir(path: Path) -> bool:
    text = path.as_posix().rstrip("/")
    return any(text == runtime or text.endswith("/" + runtime) for runtime in RUNTIME_DIRS)


def check_block(language: str, lines: list[tuple[int, str]]) -> list[str]:
    """Return one message per command whose directory target does not resolve."""
    if language not in SHELL_LANGS:
        return []
    cwd = Path()
    virtual: set[Path] = set()
    problems: list[str] = []

    for number, tokens in command_lines(lines):
        if tokens[:2] == ["git", "clone"]:
            url = tokens[-1].rstrip("/").removesuffix(".git")
            name = url.rsplit("/", 1)[-1]
            if name in REPO_NAMES:
                virtual.add(Path())
            elif name:
                virtual.add(cwd / name)
            continue

        if tokens[0] == "cd" and len(tokens) == 2:
            if tokens[1].startswith("/"):
                continue  # absolute paths are the reader's machine, not this tree
            target = normalise(tokens[1])
            resolved = (cwd / target).resolve() if target.is_absolute() else cwd / target
            resolved = Path(*resolved.parts)  # keep it relative
            if not ((REPO / resolved).exists() or resolved in virtual or is_runtime_dir(resolved)):
                problems.append(f"line {number}: cd {tokens[1]} does not exist")
            cwd = resolved
            continue

        if tokens[0] == "mkdir":
            for option in ("-p", "--parents"):
                if option in tokens:
                    args = tokens[tokens.index(option) + 1 :]
                    virtual.update(cwd / normalise(arg) for arg in args if not arg.startswith("-"))
                    break

        if tokens[:3] == ["python", "-m", "venv"] and len(tokens) > 3:
            virtual.add(cwd / tokens[3])

        if tokens[0] == "uv" and "--directory" in tokens:
            index = tokens.index("--directory")
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("/"):
                target = normalise(tokens[index + 1])
                resolved = cwd / target
                if not ((REPO / resolved).exists() or resolved in virtual):
                    problems.append(
                        f"line {number}: uv run --directory {tokens[index + 1]} "
                        "does not exist"
                    )
    return problems


def check_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = []
    for language, lines in markdown_blocks(text):
        for problem in check_block(language, lines):
            problems.append(f"{path.as_posix()}: {problem}")
    return problems


def tracked_markdown() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    # Vendored third-party docs under server/deps are not this project's quick starts.
    return [REPO / name for name in listed if name and not name.startswith("server/deps/")]


def test_quick_start_paths_resolve() -> None:
    problems = [problem for path in tracked_markdown() for problem in check_markdown(path)]
    assert not problems, "quick-start commands point at paths that do not exist:\n" + "\n".join(
        problems
    )


@pytest.mark.parametrize(
    "block",
    [
        # The shape of the bug: docs kept the pre-rename directory name.
        "```bash\ncd lucebox-hub\n```\n",
        # The shape of the bug: top-level megakernel/ before the move.
        "```bash\nuv run --directory megakernel python final_bench.py\n```\n",
        # Relative target under the project still assumes the old layout.
        "```bash\nuv run --directory optimizations/pflash python x.py\n"
        "uv run --directory deepseek4 python y.py\n```\n",
    ],
)
def test_rejects_stale_paths(block: str) -> None:
    assert check_block(*markdown_blocks(block)[0])


@pytest.mark.parametrize(
    "block",
    [
        # Clone of this repo maps to the repo root, then cd into it is fine.
        "```bash\ngit clone https://github.com/Luce-Org/lucebox.git\ncd lucebox\n```\n",
        # The server quick start cds into a subdirectory of the clone.
        "```bash\ngit clone https://github.com/Luce-Org/lucebox.git\ncd lucebox/server\n```\n",
        # A cloned third-party project and a venv are created by the block.
        "```bash\ngit clone https://example.com/x/y.git\ncd y\npython -m venv .venv\ncd .venv\n```\n",
        # Model downloads and build trees appear at runtime.
        "```bash\ncd server/models\ncd ../build\n```\n",
        # Absolute and expanded paths are out of scope.
        '```bash\ncd "$(mktemp -d)"\ncd /opt/models\n```\n',
        # Non-shell blocks are documentation, not commands.
        "```text\ncd not/a/real/path\n```\n",
    ],
)
def test_accepts_resolvable_paths(block: str) -> None:
    assert not check_block(*markdown_blocks(block)[0])
