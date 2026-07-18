from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.validation_v2.records import (
    ResultStore,
    RunRow,
    RunStatus,
    sha256_json,
    validate_complete_matrix,
)


def row(*, status: RunStatus = RunStatus.SUCCESS, attempt: int = 1) -> RunRow:
    return RunRow(
        config_id="cfg-1",
        attempt=attempt,
        status=status,
        config_hash=sha256_json({"mode": "exact"}),
        environment_hash="e" * 64,
        started_at="2026-07-18T00:00:00Z",
        finished_at="2026-07-18T00:00:01Z",
        return_code=0 if status is RunStatus.SUCCESS else 1,
        metrics={"decode_tok_s": 1.0} if status is RunStatus.SUCCESS else {},
        artifacts={},
        error=None if status is RunStatus.SUCCESS else "failed",
    )


def test_failed_row_is_not_complete_and_is_retried(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    store.append(row(status=RunStatus.FAILED))
    assert not store.is_complete(
        "cfg-1",
        config_hash=sha256_json({"mode": "exact"}),
        environment_hash="e" * 64,
        required_metrics=["decode_tok_s"],
    )
    store.append(row(attempt=2))
    assert store.is_complete(
        "cfg-1",
        config_hash=sha256_json({"mode": "exact"}),
        environment_hash="e" * 64,
        required_metrics=["decode_tok_s"],
    )


def test_duplicate_attempt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    raw = json.dumps(row().as_dict())
    path.write_text(raw + "\n" + raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate result attempt"):
        ResultStore(path).rows()


def test_matrix_refuses_failed_or_missing_cells() -> None:
    plan = {
        "config_id": "cfg-1",
        "config": {"mode": "exact"},
        "environment_hash": "e" * 64,
    }
    with pytest.raises(ValueError, match="matrix is incomplete"):
        validate_complete_matrix([plan], [row(status=RunStatus.FAILED)], required_metrics=[])
    with pytest.raises(ValueError, match="matrix is incomplete"):
        validate_complete_matrix([plan], [], required_metrics=[])


def test_success_requires_declared_metrics() -> None:
    plan = {
        "config_id": "cfg-1",
        "config": {"mode": "exact"},
        "environment_hash": "e" * 64,
    }
    with pytest.raises(ValueError, match="missing metrics"):
        validate_complete_matrix([plan], [row()], required_metrics=["ttft_ms"])
