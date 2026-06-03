"""``luce-bench submit-baseline`` — copy a level3 snapshot into the baselines repo.

The baselines repo (separate git checkout) is the cross-machine
ground-truth for "this is what the rig produced on this date with this
config". This subcommand:

  1. Validates that the snapshot is structurally complete + level3
     (only level3 carries the full area set we want frozen in
     baselines).
  2. Copies the dir into the baselines repo verbatim.
  3. ``git add`` + ``git commit`` with an auto-derived message.

It deliberately **does not push** — the operator pushes when they're
ready (after a final review of the diff against the previous
baseline).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench.levels import LEVELS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _validate_snapshot(snapshot_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    """Return ``(ok, reason, summary)``.

    On failure ``reason`` describes which file/field tripped the check.
    On success ``summary`` is the parsed ``_summary.json`` payload —
    callers reuse its fields to build the commit message.
    """
    if not snapshot_dir.is_dir():
        return False, f"{snapshot_dir} is not a directory", {}

    required = ("host.json", "props.json", "config.json", "_summary.json")
    for fname in required:
        if not (snapshot_dir / fname).is_file():
            return False, f"{snapshot_dir}/{fname} is missing", {}

    try:
        summary = _read_json(snapshot_dir / "_summary.json")
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"_summary.json is not readable JSON: {exc}", {}
    if not isinstance(summary, dict):
        return False, "_summary.json is not a JSON object", {}

    level = summary.get("level")
    if level != "level3":
        return False, f"snapshot level is {level!r}, want 'level3'", {}

    # Check each level3 area landed a per-area JSON. We map "smoke" / etc
    # to "<area>.json" — same convention as the snapshot writer.
    for area, _ in LEVELS["level3"]:
        area_path = snapshot_dir / f"{area}.json"
        if not area_path.is_file():
            return False, f"missing area artifact: {area_path.name}", {}

    return True, "", summary


def _resolve_baselines(arg: str | None) -> Path | None:
    """``--baselines`` flag → env var fallback. Returns ``None`` when neither set."""
    if arg:
        return Path(arg)
    env = os.environ.get("LUCE_BENCH_BASELINES_DIR", "").strip()
    if env:
        return Path(env)
    return None


def _commit_message(snapshot_dir: Path, host_info: dict[str, Any]) -> str:
    """Build ``"<host> <gpu> <label> <date> — level3 snapshot"`` from on-disk facts.

    Falls back to dir-name slices when host.json / dir name don't give
    us a full split, so the commit message always has *something*
    meaningful even on partial probes.
    """
    name = snapshot_dir.name
    gpu = str(host_info.get("gpu_name") or "?")
    # Auto-named snapshots end with -YYYY-MM-DD-XXXX; strip the trailing
    # 4-char hash so the date stays visible in the message.
    parts = name.split("-")
    date = "?"
    label = "?"
    host = "?"
    if len(parts) >= 4:
        # Pattern: host-gpu...-label-yyyy-mm-dd-cfg
        # We can't perfectly invert the slug but use simple heuristics:
        # the date is whichever 3-segment YYYY-MM-DD chunk appears.
        for i in range(len(parts) - 3, -1, -1):
            chunk = "-".join(parts[i : i + 3])
            if (
                len(chunk) == 10
                and chunk[4] == "-"
                and chunk[7] == "-"
                and chunk[:4].isdigit()
                and chunk[5:7].isdigit()
                and chunk[8:10].isdigit()
            ):
                date = chunk
                host = parts[0]
                label = parts[i - 1] if i >= 1 else "?"
                break
    return f"{host} {gpu} {label} {date} — level3 snapshot"


def _git(baselines: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(baselines), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="luce-bench submit-baseline",
        description=(
            "Copy a level3 snapshot into a baselines git repo (commit "
            "only — never pushes). Refuses to run on non-level3 "
            "snapshots or on dirs missing any of the level3 areas."
        ),
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "snapshot_dir",
        type=Path,
        help="Path to the snapshot directory (the per-name dir, not its parent).",
    )
    ap.add_argument(
        "--baselines",
        default=None,
        help=(
            "Baselines repo path. Defaults to $LUCE_BENCH_BASELINES_DIR; "
            "fails with exit 2 when neither is set."
        ),
    )
    ns = ap.parse_args(argv)

    snapshot_dir = ns.snapshot_dir.resolve()
    baselines = _resolve_baselines(ns.baselines)
    if baselines is None:
        print(
            "[lucebench submit-baseline] no baselines repo set "
            "— pass --baselines or set LUCE_BENCH_BASELINES_DIR",
            file=sys.stderr,
            flush=True,
        )
        return 2
    baselines = baselines.resolve()
    if not (baselines / ".git").exists():
        print(
            f"[lucebench submit-baseline] {baselines} is not a git checkout (missing .git)",
            file=sys.stderr,
            flush=True,
        )
        return 2

    ok, reason, _summary = _validate_snapshot(snapshot_dir)
    if not ok:
        print(f"[lucebench submit-baseline] refusing to submit: {reason}", file=sys.stderr)
        return 2

    try:
        host_info = _read_json(snapshot_dir / "host.json")
    except (OSError, json.JSONDecodeError):
        host_info = {}
    if not isinstance(host_info, dict):
        host_info = {}

    dest = baselines / snapshot_dir.name
    if dest.exists():
        print(
            f"[lucebench submit-baseline] {dest} already exists — refusing to overwrite. "
            "Rename the snapshot or remove the destination dir first.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    shutil.copytree(snapshot_dir, dest)

    add = _git(baselines, "add", snapshot_dir.name)
    if add.returncode != 0:
        print(
            f"[lucebench submit-baseline] git add failed:\n{add.stderr}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    message = _commit_message(snapshot_dir, host_info)
    commit = _git(baselines, "commit", "-m", message)
    if commit.returncode != 0:
        print(
            f"[lucebench submit-baseline] git commit failed:\n{commit.stderr}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    rev = _git(baselines, "rev-parse", "HEAD")
    sha = (rev.stdout or "").strip() or "?"
    print(
        f"[lucebench submit-baseline] committed {sha} → {dest}\n"
        f"[lucebench submit-baseline] message: {message}\n"
        f"[lucebench submit-baseline] when ready: cd {baselines} && git push origin main",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
