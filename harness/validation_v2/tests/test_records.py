from __future__ import annotations

import dataclasses
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
        prompt_hash="a" * 64,
        protocol_hash="f" * 64,
        cell_hash=sha256_json(
            {
                "config_id": "cfg-1",
                "config": {"mode": "exact"},
                "config_hash": sha256_json({"mode": "exact"}),
                "environment_hash": "e" * 64,
                "prompt_hash": "a" * 64,
                "protocol_hash": "f" * 64,
            }
        ),
    )


def plan() -> dict[str, object]:
    value: dict[str, object] = {
        "config_id": "cfg-1",
        "config": {"mode": "exact"},
        "config_hash": sha256_json({"mode": "exact"}),
        "environment_hash": "e" * 64,
        "prompt_hash": "a" * 64,
        "protocol_hash": "f" * 64,
    }
    value["cell_hash"] = sha256_json(value)
    return value


def test_failed_row_is_not_complete_and_is_retried(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    store.append(row(status=RunStatus.FAILED))
    assert not store.is_complete(
        "cfg-1",
        config_hash=sha256_json({"mode": "exact"}),
        environment_hash="e" * 64,
        prompt_hash="a" * 64,
        protocol_hash="f" * 64,
        cell_hash=row().cell_hash,
        required_metrics=["decode_tok_s"],
    )
    store.append(row(attempt=2))
    assert store.is_complete(
        "cfg-1",
        config_hash=sha256_json({"mode": "exact"}),
        environment_hash="e" * 64,
        prompt_hash="a" * 64,
        protocol_hash="f" * 64,
        cell_hash=row().cell_hash,
        required_metrics=["decode_tok_s"],
    )


def test_duplicate_attempt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    raw = json.dumps(row().as_dict())
    path.write_text(raw + "\n" + raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate result attempt"):
        ResultStore(path).rows()


def test_matrix_refuses_failed_or_missing_cells() -> None:
    with pytest.raises(ValueError, match="matrix is incomplete"):
        validate_complete_matrix([plan()], [row(status=RunStatus.FAILED)], required_metrics=[])
    with pytest.raises(ValueError, match="matrix is incomplete"):
        validate_complete_matrix([plan()], [], required_metrics=[])


def test_success_requires_declared_metrics() -> None:
    with pytest.raises(ValueError, match="missing metrics"):
        validate_complete_matrix([plan()], [row()], required_metrics=["ttft_ms"])


def test_matrix_refuses_empty_plan_and_stale_declared_config() -> None:
    with pytest.raises(ValueError, match="planned matrix is empty"):
        validate_complete_matrix([], [], required_metrics=[])
    stale = plan()
    stale["config"] = {"mode": "changed"}
    with pytest.raises(ValueError, match="planned config hash mismatch"):
        validate_complete_matrix([stale], [], required_metrics=[])


def test_matrix_binds_prompt_protocol_and_cell_hashes() -> None:
    valid_plan = plan()
    valid_row = dataclasses.replace(
        row(),
        cell_hash=str(valid_plan["cell_hash"]),
    )
    validate_complete_matrix([valid_plan], [valid_row], required_metrics=["decode_tok_s"])
    for field in ("prompt_hash", "protocol_hash", "cell_hash"):
        changed = dataclasses.replace(valid_row, **{field: "1" * 64})
        with pytest.raises(ValueError, match=f"{field.replace('_', ' ')} mismatch"):
            validate_complete_matrix([valid_plan], [changed], required_metrics=[])
