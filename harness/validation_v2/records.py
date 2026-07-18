"""Append-only result rows and strict matrix-completeness validation."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class RunStatus(enum.StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class RunRow:
    config_id: str
    attempt: int
    status: RunStatus
    config_hash: str
    environment_hash: str
    started_at: str
    finished_at: str
    return_code: int | None
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, str]
    error: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RunRow:
        required = {
            "config_id",
            "attempt",
            "status",
            "config_hash",
            "environment_hash",
            "started_at",
            "finished_at",
            "return_code",
            "metrics",
            "artifacts",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"result row missing fields: {sorted(missing)}")
        row = cls(
            config_id=str(raw["config_id"]),
            attempt=int(raw["attempt"]),
            status=RunStatus(str(raw["status"])),
            config_hash=str(raw["config_hash"]),
            environment_hash=str(raw["environment_hash"]),
            started_at=str(raw["started_at"]),
            finished_at=str(raw["finished_at"]),
            return_code=None if raw["return_code"] is None else int(raw["return_code"]),
            metrics=dict(raw["metrics"]),
            artifacts={str(key): str(value) for key, value in dict(raw["artifacts"]).items()},
            error=None if raw.get("error") is None else str(raw["error"]),
        )
        if row.attempt < 1:
            raise ValueError("attempt must be >= 1")
        if row.status is RunStatus.SUCCESS and row.return_code != 0:
            raise ValueError("success row must have return_code=0")
        if row.status is RunStatus.SUCCESS and row.error:
            raise ValueError("success row cannot carry an error")
        return row

    def as_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["status"] = self.status.value
        return raw


class ResultStore:
    """Append-only JSONL store.

    A cell is resumable only when its newest row is a complete success whose
    configuration and environment hashes match the planned matrix.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def rows(self) -> list[RunRow]:
        if not self.path.exists():
            return []
        rows: list[RunRow] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(RunRow.from_mapping(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid result row at {self.path}:{line_number}: {exc}") from exc
        self._validate_attempts(rows)
        return rows

    @staticmethod
    def _validate_attempts(rows: Iterable[RunRow]) -> None:
        seen: set[tuple[str, int]] = set()
        latest: dict[str, int] = {}
        for row in rows:
            key = (row.config_id, row.attempt)
            if key in seen:
                raise ValueError(f"duplicate result attempt: {row.config_id}#{row.attempt}")
            seen.add(key)
            if row.attempt <= latest.get(row.config_id, 0):
                raise ValueError(f"attempts are not append-ordered for {row.config_id}")
            latest[row.config_id] = row.attempt

    def append(self, row: RunRow) -> None:
        existing = self.rows()
        previous = max((item.attempt for item in existing if item.config_id == row.config_id), default=0)
        if row.attempt != previous + 1:
            raise ValueError(
                f"attempt for {row.config_id} must be {previous + 1}, got {row.attempt}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row.as_dict()) + "\n")

    def latest(self) -> dict[str, RunRow]:
        latest: dict[str, RunRow] = {}
        for row in self.rows():
            latest[row.config_id] = row
        return latest

    def is_complete(
        self,
        config_id: str,
        *,
        config_hash: str,
        environment_hash: str,
        required_metrics: Iterable[str],
    ) -> bool:
        row = self.latest().get(config_id)
        if row is None or row.status is not RunStatus.SUCCESS:
            return False
        if row.config_hash != config_hash or row.environment_hash != environment_hash:
            return False
        return all(metric in row.metrics and row.metrics[metric] is not None for metric in required_metrics)


def validate_complete_matrix(
    planned: Iterable[Mapping[str, Any]],
    rows: Iterable[RunRow],
    *,
    required_metrics: Iterable[str],
) -> dict[str, int]:
    plans = list(planned)
    plan_by_id: dict[str, Mapping[str, Any]] = {}
    for plan in plans:
        config_id = str(plan["config_id"])
        if config_id in plan_by_id:
            raise ValueError(f"duplicate planned config_id: {config_id}")
        plan_by_id[config_id] = plan

    latest: dict[str, RunRow] = {}
    for row in rows:
        if row.config_id not in plan_by_id:
            raise ValueError(f"unplanned result row: {row.config_id}")
        latest[row.config_id] = row

    counts = {status.value: 0 for status in RunStatus}
    counts["missing"] = 0
    incomplete: list[str] = []
    required = tuple(required_metrics)
    for config_id, plan in plan_by_id.items():
        row = latest.get(config_id)
        if row is None:
            counts["missing"] += 1
            incomplete.append(f"{config_id}: missing")
            continue
        counts[row.status.value] += 1
        expected_config = str(plan.get("config_hash") or sha256_json(plan.get("config", {})))
        expected_environment = str(plan["environment_hash"])
        missing_metrics = [metric for metric in required if row.metrics.get(metric) is None]
        if row.status is not RunStatus.SUCCESS:
            incomplete.append(f"{config_id}: {row.status.value}")
        elif row.config_hash != expected_config:
            incomplete.append(f"{config_id}: config hash mismatch")
        elif row.environment_hash != expected_environment:
            incomplete.append(f"{config_id}: environment hash mismatch")
        elif missing_metrics:
            incomplete.append(f"{config_id}: missing metrics {missing_metrics}")

    if incomplete:
        raise ValueError("matrix is incomplete:\n" + "\n".join(incomplete))
    return counts
