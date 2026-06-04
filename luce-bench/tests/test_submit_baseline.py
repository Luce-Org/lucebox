"""Tests for ``lucebench.submit_baseline`` — the level3 → baselines-repo bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lucebench import submit_baseline
from lucebench.levels import LEVELS


def _git(repo: Path, *args: str) -> str:
    """Run git inside ``repo`` and return stripped stdout."""
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _init_repo(path: Path) -> Path:
    """Init an empty git repo with a deterministic identity for commits."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _write_snapshot(
    parent: Path,
    *,
    name: str,
    level: str = "level3",
    skip_areas: list[str] | None = None,
    include_host: bool = True,
) -> Path:
    """Build a minimum-viable snapshot dir on disk for the validator to chew on."""
    snap = parent / name
    snap.mkdir(parents=True, exist_ok=True)
    if include_host:
        (snap / "host.json").write_text(
            json.dumps(
                {
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "cpu_model": "Test CPU",
                    "ram_gb": 64,
                }
            )
        )
    else:
        (snap / "host.json").write_text("{}")
    (snap / "props.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    summary: dict[str, Any] = {
        "name": name,
        "url": "http://127.0.0.1:8080",
        "model": "test",
        "level": level,
        "areas": [],
    }
    (snap / "_summary.json").write_text(json.dumps(summary))
    skip = set(skip_areas or [])
    for area, _cap in LEVELS["level3"]:
        if area in skip:
            continue
        (snap / f"{area}.json").write_text(json.dumps({"area": area, "rows": []}))
    return snap


def test_submit_baseline_happy_path(tmp_path: Path) -> None:
    """Valid level3 snapshot → copied into baselines, committed, never pushed."""
    baselines = _init_repo(tmp_path / "baselines")
    snap = _write_snapshot(
        tmp_path / "snaps",
        name="bragi-rtx-5090-profile-2026-05-28-abcd",
    )

    rc = submit_baseline.main([str(snap), "--baselines", str(baselines)])
    assert rc == 0

    # Snapshot dir copied verbatim into baselines.
    dest = baselines / snap.name
    assert (dest / "_summary.json").is_file()
    assert (dest / "smoke.json").is_file()
    # One commit landed, no remote push.
    log = _git(baselines, "log", "--oneline")
    assert "level3 snapshot" in log
    # `git status` clean (everything committed).
    assert _git(baselines, "status", "--porcelain") == ""


def test_submit_baseline_refuses_non_level3(tmp_path: Path) -> None:
    """A level1 snapshot is rejected with exit 2 — only level3 may be promoted."""
    baselines = _init_repo(tmp_path / "baselines")
    snap = _write_snapshot(tmp_path / "snaps", name="x", level="level1")
    rc = submit_baseline.main([str(snap), "--baselines", str(baselines)])
    assert rc == 2
    # Nothing copied — baselines repo still empty.
    assert list(baselines.glob("x")) == []


def test_submit_baseline_refuses_missing_area(tmp_path: Path) -> None:
    """Refusing to commit when any level3 area is missing keeps baselines coherent."""
    baselines = _init_repo(tmp_path / "baselines")
    snap = _write_snapshot(
        tmp_path / "snaps",
        name="x",
        skip_areas=["forge"],
    )
    rc = submit_baseline.main([str(snap), "--baselines", str(baselines)])
    assert rc == 2


def test_submit_baseline_refuses_no_baselines_arg_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loud failure when neither --baselines nor LUCE_BENCH_BASELINES_DIR is set."""
    monkeypatch.delenv("LUCE_BENCH_BASELINES_DIR", raising=False)
    snap = _write_snapshot(tmp_path / "snaps", name="x")
    rc = submit_baseline.main([str(snap)])
    assert rc == 2


def test_submit_baseline_uses_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LUCE_BENCH_BASELINES_DIR`` env var stands in for the --baselines flag."""
    baselines = _init_repo(tmp_path / "baselines")
    snap = _write_snapshot(tmp_path / "snaps", name="env-name")
    monkeypatch.setenv("LUCE_BENCH_BASELINES_DIR", str(baselines))
    rc = submit_baseline.main([str(snap)])
    assert rc == 0
    assert (baselines / snap.name / "_summary.json").is_file()


def test_submit_baseline_refuses_overwriting_existing_dir(tmp_path: Path) -> None:
    """A pre-existing dir in baselines isn't silently overwritten."""
    baselines = _init_repo(tmp_path / "baselines")
    snap = _write_snapshot(tmp_path / "snaps", name="dup")
    rc1 = submit_baseline.main([str(snap), "--baselines", str(baselines)])
    assert rc1 == 0
    # Re-submitting the same name now collides.
    rc2 = submit_baseline.main([str(snap), "--baselines", str(baselines)])
    assert rc2 == 2


def test_submit_baseline_refuses_non_git_baselines(tmp_path: Path) -> None:
    """A baselines path without a .git dir is rejected — no silent ``git init``."""
    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    snap = _write_snapshot(tmp_path / "snaps", name="x")
    rc = submit_baseline.main([str(snap), "--baselines", str(bare)])
    assert rc == 2


def test_commit_message_extracts_host_gpu_label_date(tmp_path: Path) -> None:
    """Commit-message synthesis pulls fields out of the host.json + dir name."""
    snap = _write_snapshot(
        tmp_path / "snaps",
        name="bragi-rtx-5090-profile-2026-05-28-abcd",
    )
    host = json.loads((snap / "host.json").read_text())
    msg = submit_baseline._commit_message(snap, host)
    assert "bragi" in msg
    assert "RTX 5090" in msg
    assert "profile" in msg
    assert "2026-05-28" in msg
    assert msg.endswith("level3 snapshot")
