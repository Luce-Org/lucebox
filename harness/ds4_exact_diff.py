#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "lucebox.ds4.exact-diff/v1"
TOLERANCES = {
    "routing_weights": {"atol": 1e-6, "rtol": 1e-6},
    "capture_rows": {"atol": 1e-5, "rtol": 1e-5},
    "final_logits": {"atol": 1e-4, "rtol": 1e-4},
}
WIDTHS = (1, 2, 3, 4)
PROFILES = ("reset", "snapshot")


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class Mismatch:
    profile: str
    width: int
    request: int | None
    layer: int | None
    token_position: int | None
    field: str
    oracle: Any
    candidate: Any
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "mismatch",
            "profile": self.profile,
            "width": self.width,
            "request": self.request,
            "layer": self.layer,
            "token_position": self.token_position,
            "field": self.field,
            "oracle": self.oracle,
            "candidate": self.candidate,
            "detail": self.detail,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite_constant(value: str) -> None:
    raise TraceError(f"non-finite JSON constant: {value}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=reject_nonfinite_constant)
            except (json.JSONDecodeError, TraceError) as exc:
                raise TraceError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TraceError(f"{path}:{line_number}: record is not an object")
            if record.get("schema") != SCHEMA:
                raise TraceError(f"{path}:{line_number}: missing or unknown schema")
            records.append(record)
    if not records:
        raise TraceError(f"{path}: empty trace")
    validate_records(path, records)
    return records


def require(record: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise TraceError(f"{context}: missing fields: {', '.join(missing)}")


def validate_finite(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TraceError(f"{context}: non-finite float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{context}.{key}")


def validate_float_list(value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise TraceError(f"{context}: expected a list")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TraceError(f"{context}[{index}]: expected a finite number")
        if not math.isfinite(float(item)):
            raise TraceError(f"{context}[{index}]: non-finite float")


def validate_int_list(value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise TraceError(f"{context}: expected a list")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TraceError(f"{context}[{index}]: expected an integer")
        if not -(2**31) <= item < 2**31:
            raise TraceError(f"{context}[{index}]: integer is outside int32")


def validate_records(path: Path, records: list[dict[str, Any]]) -> None:
    require(records[0], ("type", "profile", "width"), f"{path}:1")
    if records[0]["type"] != "manifest":
        raise TraceError(f"{path}: first record must be manifest")
    require(
        records[0],
        (
            "revision",
            "binary_sha256",
            "target_sha256",
            "drafter_sha256",
            "prompt_bytes_sha256",
            "request_config",
            "tolerances",
        ),
        f"{path}:manifest",
    )
    tolerances = records[0]["tolerances"]
    if not isinstance(tolerances, dict):
        raise TraceError(f"{path}:manifest:tolerances must be an object")
    for name in TOLERANCES:
        if name not in tolerances or not isinstance(tolerances[name], dict):
            raise TraceError(f"{path}:manifest:tolerances.{name} is missing")
        require(tolerances[name], ("atol", "rtol"), f"{path}:manifest:{name}")
        for field in ("atol", "rtol"):
            value = tolerances[name][field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise TraceError(
                    f"{path}:manifest:tolerances.{name}.{field} must be finite and non-negative"
                )
    if tolerances != TOLERANCES:
        raise TraceError(f"{path}:manifest:tolerances must equal the built-in contract")
    last_position: dict[int, int] = {}
    for index, record in enumerate(records, 1):
        record_type = record.get("type")
        context = f"{path}:{index}:{record_type}"
        validate_finite(record, context)
        if record.get("non_finite") is True:
            raise TraceError(f"{context}: producer reported non-finite state")
        if record_type == "model_config":
            require(record, ("n_swa", "compressor_boundaries"), context)
        elif record_type == "request_start":
            require(
                record,
                (
                    "request",
                    "prompt_token_hash",
                    "prompt_token_ids",
                    "prompt_tokens",
                    "width",
                    "restored",
                    "cache_position",
                    "n_gen",
                    "snap_slot",
                    "snap_pos",
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                ),
                context,
            )
            validate_int_list(record["prompt_token_ids"], f"{context}:prompt_token_ids")
            if len(record["prompt_token_ids"]) != record["prompt_tokens"]:
                raise TraceError(f"{context}: prompt token count does not match token IDs")
        elif record_type == "layer":
            require(
                record,
                (
                    "request",
                    "layer",
                    "position_begin",
                    "position_end",
                    "routing",
                    "hc_token_hashes",
                    "raw_kv_hash",
                    "raw_kv_bytes",
                    "compressed_kv_hash",
                    "compressed_kv_bytes",
                    "compressor_state",
                    "indexer_state",
                ),
                context,
            )
            routing = record["routing"]
            require(routing, ("mode", "ids", "weights"), f"{context}:routing")
            if not routing["ids"] or len(routing["ids"]) != len(routing["weights"]):
                raise TraceError(f"{context}: routing IDs/weights are missing or misaligned")
            validate_float_list(routing["weights"], f"{context}:routing.weights")
            if not record["hc_token_hashes"] or not record["hc_token_hashes"][0]:
                raise TraceError(f"{context}: HC token hash is missing")
            request = int(record["request"])
            position = int(record["position_end"])
            previous = last_position.get(request, -1)
            if position < previous:
                raise TraceError(f"{context}: stale position {position} follows {previous}")
            last_position[request] = position
        elif record_type == "logits":
            require(record, ("request", "position", "values"), context)
            validate_float_list(record["values"], f"{context}:values")
        elif record_type == "capture":
            require(
                record,
                (
                    "request",
                    "position_begin",
                    "position_end",
                    "layer_ids",
                    "row_hash",
                    "total_values",
                    "rows",
                ),
                context,
            )
            validate_float_list(record["rows"], f"{context}:rows")
        elif record_type in {"snapshot_save", "snapshot_restore"}:
            require(
                record,
                (
                    "request",
                    "slot",
                    "cache_position",
                    "state_hash",
                    "last_logits_hash",
                    "last_logits_count",
                    "last_logits_position",
                    "spec_feature_hash",
                    "spec_feature_count",
                ),
                context,
            )
            if record["last_logits_count"] <= 0:
                raise TraceError(f"{context}: snapshot logits are missing")
            if record["last_logits_position"] != record["cache_position"]:
                raise TraceError(f"{context}: snapshot logits position is stale")
            if record["spec_feature_count"] < 0:
                raise TraceError(f"{context}: snapshot feature count is negative")
        elif record_type == "tokens":
            require(record, ("request", "token_ids"), context)
        elif record_type == "request_end":
            require(record, ("request", "ok", "cache_position"), context)
            if record["ok"] is not True:
                raise TraceError(f"{context}: request did not complete successfully")
        elif record_type == "reset":
            require(record, ("request", "cache_position"), context)
        elif record_type == "step":
            require(
                record,
                (
                    "request",
                    "position_begin",
                    "position_end",
                    "position",
                    "width",
                    "cache_position",
                    "logits_present",
                    "relations",
                ),
                context,
            )
        elif record_type != "manifest":
            raise TraceError(f"{context}: unknown record type")

    request_starts = [
        record["request"] for record in records if record.get("type") == "request_start"
    ]
    request_ends = [record["request"] for record in records if record.get("type") == "request_end"]
    token_records = [record["request"] for record in records if record.get("type") == "tokens"]
    for label, observed in (("request_end", request_ends), ("tokens", token_records)):
        if sorted(observed) != sorted(request_starts):
            raise TraceError(f"{path}:{label} coverage does not match request_start coverage")


def close_enough(a: float, b: float, tolerance: dict[str, float]) -> bool:
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    limit = tolerance["atol"] + tolerance["rtol"] * max(abs(a), abs(b))
    return abs(a - b) <= limit


def event_index(
    records: list[dict[str, Any]], event_type: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("type") != event_type:
            continue
        if event_type == "layer":
            key = (record["request"], record["position_end"], record["layer"])
        elif event_type in {"logits", "step"}:
            key = (record["request"], record.get("position", record.get("position_end")))
        elif event_type == "capture":
            key = (record["request"], record["position_end"])
        elif event_type in {"tokens", "request_end", "request_start", "reset"}:
            key = (record["request"],)
        else:
            key = (record["request"], record.get("slot"), record.get("cache_position"))
        if key in indexed:
            raise TraceError(f"duplicate {event_type} key {key}")
        indexed[key] = record
    return indexed


def first_float_mismatch(
    oracle: list[float], candidate: list[float], tolerance: dict[str, float]
) -> tuple[int, float | None, float | None] | None:
    if len(oracle) != len(candidate):
        return min(len(oracle), len(candidate)), None, None
    for index, (left, right) in enumerate(zip(oracle, candidate, strict=True)):
        if not close_enough(float(left), float(right), tolerance):
            return index, float(left), float(right)
    return None


def compare_manifest(
    profile: str,
    width: int,
    oracle: dict[str, Any],
    candidate: dict[str, Any],
) -> Mismatch | None:
    exact_fields = (
        "revision",
        "binary_sha256",
        "target_sha256",
        "drafter_sha256",
        "prompt_bytes_sha256",
        "tolerances",
    )
    for field in exact_fields:
        if oracle[field] != candidate[field]:
            return Mismatch(
                profile,
                width,
                None,
                None,
                None,
                field,
                oracle[field],
                candidate[field],
                "manifest differs",
            )
    oracle_cfg = dict(oracle["request_config"])
    candidate_cfg = dict(candidate["request_config"])
    for allowed in ("prefill_width", "exact_bands", "port"):
        oracle_cfg.pop(allowed, None)
        candidate_cfg.pop(allowed, None)
    if oracle_cfg != candidate_cfg:
        return Mismatch(
            profile,
            width,
            None,
            None,
            None,
            "request_config",
            oracle_cfg,
            candidate_cfg,
            "non-width request configuration differs",
        )
    return None


def compare_profile(
    profile: str, width: int, oracle: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> Mismatch | None:
    mismatch = compare_manifest(profile, width, oracle[0], candidate[0])
    if mismatch:
        return mismatch
    for event_type, fields in (
        (
            "request_start",
            (
                "prompt_token_hash",
                "prompt_token_ids",
                "prompt_tokens",
                "restored",
                "cache_position",
                "n_gen",
                "snap_slot",
                "snap_pos",
                "temperature",
                "top_p",
                "top_k",
                "seed",
            ),
        ),
        ("reset", ("cache_position",)),
    ):
        refs = event_index(oracle, event_type)
        cands = event_index(candidate, event_type)
        if refs.keys() != cands.keys():
            return Mismatch(
                profile,
                width,
                None,
                None,
                None,
                event_type,
                sorted(refs),
                sorted(cands),
                f"{event_type} keys differ",
            )
        for key in sorted(cands):
            for field in fields:
                if refs[key][field] != cands[key][field]:
                    return Mismatch(
                        profile,
                        width,
                        int(key[0]),
                        None,
                        None,
                        f"{event_type}.{field}",
                        refs[key][field],
                        cands[key][field],
                        f"{event_type} differs",
                    )

    oracle_steps = event_index(oracle, "step")
    candidate_steps = event_index(candidate, "step")
    starts = event_index(candidate, "request_start")
    for label, steps in (("oracle", oracle_steps), ("candidate", candidate_steps)):
        maximum_width = 1 if label == "oracle" else width
        for request_key, start in sorted(starts.items()):
            request = int(request_key[0])
            request_steps = sorted(
                (record for key, record in steps.items() if key[0] == request),
                key=lambda record: int(record["position_end"]),
            )
            cursor = int(start["cache_position"])
            final_position = int(start["prompt_tokens"])
            if not request_steps:
                if cursor == final_position:
                    continue
                return Mismatch(
                    profile,
                    width,
                    request,
                    None,
                    None,
                    "step.coverage",
                    (cursor, final_position),
                    "missing",
                    f"{label} has no step records for request",
                )
            for record in request_steps:
                begin = int(record["position_begin"])
                end = int(record["position_end"])
                step_width = int(record["width"])
                if (
                    begin != cursor
                    or end != int(record["position"])
                    or end != int(record["cache_position"])
                    or end - begin != step_width
                    or not 1 <= step_width <= maximum_width
                    or end > final_position
                ):
                    return Mismatch(
                        profile,
                        width,
                        request,
                        None,
                        end,
                        "step.coverage",
                        {"position_begin": cursor, "final_position": final_position},
                        record,
                        f"{label} step coverage is discontinuous or malformed",
                    )
                cursor = end
            if cursor != final_position:
                return Mismatch(
                    profile,
                    width,
                    request,
                    None,
                    cursor,
                    "step.coverage",
                    final_position,
                    cursor,
                    f"{label} steps do not cover the full request",
                )
    for key in sorted(candidate_steps):
        cand = candidate_steps[key]
        ref = oracle_steps.get(key)
        if ref is None:
            return Mismatch(
                profile,
                width,
                int(key[0]),
                None,
                int(key[1]),
                "step",
                "present",
                "missing oracle alignment",
                "q=1 has no matching step",
            )
        for field in ("cache_position",):
            if ref[field] != cand[field]:
                return Mismatch(
                    profile,
                    width,
                    int(key[0]),
                    None,
                    int(key[1]),
                    f"step.{field}",
                    ref[field],
                    cand[field],
                    "step lifecycle differs",
                )
        start = starts[(key[0],)]
        final_position = start["prompt_tokens"]
        snapshot_position = int(start["snap_pos"])
        expected_logits = key[1] == final_position or (
            snapshot_position >= 0 and key[1] == snapshot_position
        )
        if cand["logits_present"] is not expected_logits:
            return Mismatch(
                profile,
                width,
                int(key[0]),
                None,
                int(key[1]),
                "step.logits_present",
                expected_logits,
                cand["logits_present"],
                "step readout policy differs",
            )

    oracle_layers = event_index(oracle, "layer")
    candidate_layers = event_index(candidate, "layer")
    if not candidate_layers:
        return Mismatch(
            profile,
            width,
            None,
            None,
            None,
            "layer",
            "present",
            "missing",
            "candidate has no layer records",
        )
    selected_positions = set(candidate_steps)
    expected_layer_keys = {key for key in oracle_layers if (key[0], key[1]) in selected_positions}
    if candidate_layers.keys() != expected_layer_keys:
        return Mismatch(
            profile,
            width,
            None,
            None,
            None,
            "layer.keys",
            sorted(expected_layer_keys),
            sorted(candidate_layers),
            "layer keys are incomplete at selected positions",
        )
    tolerances = oracle[0]["tolerances"]
    for key in sorted(expected_layer_keys):
        cand = candidate_layers[key]
        ref = oracle_layers.get(key)
        request, _, layer = key
        position = int(cand["token_position"])
        if ref is None:
            return Mismatch(
                profile,
                width,
                request,
                layer,
                position,
                "layer",
                "present",
                "missing oracle alignment",
                "q=1 has no matching layer position",
            )
        cand_routing = cand["routing"]
        ref_routing = ref["routing"]
        for field in ("mode", "ids"):
            if ref_routing[field] != cand_routing[field]:
                return Mismatch(
                    profile,
                    width,
                    request,
                    layer,
                    position,
                    f"routing.{field}",
                    ref_routing[field],
                    cand_routing[field],
                    "routing differs",
                )
        float_diff = first_float_mismatch(
            ref_routing["weights"], cand_routing["weights"], tolerances["routing_weights"]
        )
        if float_diff:
            index, left, right = float_diff
            return Mismatch(
                profile,
                width,
                request,
                layer,
                position,
                f"routing.weights[{index}]",
                left,
                right,
                "routing weight exceeds tolerance",
            )
        exact_fields = (
            "hc_token_hashes",
            "raw_kv_hash",
            "raw_kv_bytes",
            "compressed_kv_hash",
            "compressed_kv_bytes",
            "compressor_state",
            "indexer_state",
        )
        for field in exact_fields:
            if ref[field] != cand[field]:
                return Mismatch(
                    profile,
                    width,
                    request,
                    layer,
                    position,
                    field,
                    ref[field],
                    cand[field],
                    "exact state differs",
                )
    for event_type, tolerance_name, value_field in (
        ("capture", "capture_rows", "rows"),
        ("logits", "final_logits", "values"),
    ):
        refs = event_index(oracle, event_type)
        cands = event_index(candidate, event_type)
        if refs.keys() != cands.keys():
            return Mismatch(
                profile,
                width,
                None,
                None,
                None,
                event_type,
                sorted(refs),
                sorted(cands),
                f"{event_type} positions differ",
            )
        for key in sorted(cands):
            cand = cands[key]
            ref = refs.get(key)
            request = int(key[0])
            position = int(key[-1])
            if ref is None:
                return Mismatch(
                    profile,
                    width,
                    request,
                    None,
                    position,
                    event_type,
                    "present",
                    "missing oracle alignment",
                    f"q=1 has no matching {event_type}",
                )
            if event_type == "capture":
                for field in ("layer_ids", "total_values"):
                    if ref[field] != cand[field]:
                        return Mismatch(
                            profile,
                            width,
                            request,
                            None,
                            position,
                            f"capture.{field}",
                            ref[field],
                            cand[field],
                            "capture shape differs",
                        )
                if len(cand["rows"]) != cand["total_values"]:
                    return Mismatch(
                        profile,
                        width,
                        request,
                        None,
                        position,
                        "capture.rows",
                        cand["total_values"],
                        len(cand["rows"]),
                        "capture row is incomplete",
                    )
            float_diff = first_float_mismatch(
                ref[value_field], cand[value_field], tolerances[tolerance_name]
            )
            if float_diff:
                index, left, right = float_diff
                return Mismatch(
                    profile,
                    width,
                    request,
                    None,
                    position,
                    f"{event_type}.{value_field}[{index}]",
                    left,
                    right,
                    f"{event_type} exceeds tolerance",
                )
    for event_type, fields in (
        ("tokens", ("token_ids",)),
        ("request_end", ("ok", "cache_position")),
        (
            "snapshot_save",
            (
                "slot",
                "cache_position",
                "state_hash",
                "last_logits_hash",
                "last_logits_count",
                "last_logits_position",
                "spec_feature_hash",
                "spec_feature_count",
            ),
        ),
        (
            "snapshot_restore",
            (
                "slot",
                "cache_position",
                "state_hash",
                "last_logits_hash",
                "last_logits_count",
                "last_logits_position",
                "spec_feature_hash",
                "spec_feature_count",
            ),
        ),
    ):
        refs = event_index(oracle, event_type)
        cands = event_index(candidate, event_type)
        if refs.keys() != cands.keys():
            return Mismatch(
                profile,
                width,
                None,
                None,
                None,
                event_type,
                sorted(refs),
                sorted(cands),
                f"{event_type} keys differ",
            )
        for key in sorted(cands):
            if key not in refs:
                return Mismatch(
                    profile,
                    width,
                    int(key[0]),
                    None,
                    None,
                    event_type,
                    "present",
                    "missing oracle alignment",
                    f"q=1 has no matching {event_type}",
                )
            for field in fields:
                if refs[key][field] != cands[key][field]:
                    return Mismatch(
                        profile,
                        width,
                        int(key[0]),
                        None,
                        refs[key].get("cache_position"),
                        f"{event_type}.{field}",
                        refs[key][field],
                        cands[key][field],
                        f"{event_type} differs",
                    )
    return None


def matrix_failure(field: str, expected: Any, observed: Any, detail: str) -> Mismatch:
    return Mismatch("matrix", 0, None, None, None, field, expected, observed, detail)


SNAPSHOT_STATE_FIELDS = (
    "slot",
    "cache_position",
    "state_hash",
    "last_logits_hash",
    "last_logits_count",
    "last_logits_position",
    "spec_feature_hash",
    "spec_feature_count",
)

REPEATED_REQUEST_FIELDS = (
    "width",
    "prompt_token_hash",
    "prompt_token_ids",
    "prompt_tokens",
    "n_gen",
    "temperature",
    "top_p",
    "top_k",
    "seed",
)


def validate_repeated_request_lifecycle(
    profile: str,
    width: int,
    records: list[dict[str, Any]],
    starts: list[dict[str, Any]],
) -> Mismatch | None:
    ordered_starts = sorted(starts, key=lambda record: record["request"])
    first = ordered_starts[0]
    repeated = ordered_starts[1]
    for field in REPEATED_REQUEST_FIELDS:
        if first[field] != repeated[field]:
            return matrix_failure(
                f"{profile}.q{width}.repeated_request.{field}",
                first[field],
                repeated[field],
                "repeated request configuration differs from the first request",
            )

    token_records = event_index(records, "tokens")
    first_tokens = token_records[(first["request"],)]["token_ids"]
    repeated_tokens = token_records[(repeated["request"],)]["token_ids"]
    if first_tokens != repeated_tokens:
        return matrix_failure(
            f"{profile}.q{width}.repeated_tokens",
            first_tokens,
            repeated_tokens,
            "repeated request continuation tokens differ",
        )

    request_ends = event_index(records, "request_end")
    first_end = request_ends[(first["request"],)]["cache_position"]
    repeated_end = request_ends[(repeated["request"],)]["cache_position"]
    if first_end != repeated_end:
        return matrix_failure(
            f"{profile}.q{width}.repeated_cache_position",
            first_end,
            repeated_end,
            "repeated request final cache position differs",
        )

    if profile == "reset":
        resets = event_index(records, "reset")
        expected_reset_keys = {(first["request"],), (repeated["request"],)}
        if resets.keys() != expected_reset_keys or any(
            record["cache_position"] != 0 for record in resets.values()
        ):
            return matrix_failure(
                f"{profile}.q{width}.reset_lifecycle",
                {"requests": sorted(expected_reset_keys), "cache_position": 0},
                {
                    "requests": sorted(resets),
                    "cache_positions": sorted(
                        record["cache_position"] for record in resets.values()
                    ),
                },
                "reset requests must both begin from an empty cache",
            )
        logits = event_index(records, "logits")
        final_position = first["prompt_tokens"]
        first_logits = logits.get((first["request"], final_position))
        repeated_logits = logits.get((repeated["request"], final_position))
        if first_logits is None or repeated_logits is None:
            return matrix_failure(
                f"{profile}.q{width}.repeated_logits",
                "final logits for both requests",
                sorted(logits),
                "reset repetition is missing final logits",
            )
        float_diff = first_float_mismatch(
            first_logits["values"], repeated_logits["values"], TOLERANCES["final_logits"]
        )
        if float_diff:
            index, left, right = float_diff
            return matrix_failure(
                f"{profile}.q{width}.repeated_logits[{index}]",
                left,
                right,
                "repeated request final logits exceed tolerance",
            )
    return None


def validate_snapshot_lifecycle(
    profile: str,
    width: int,
    records: list[dict[str, Any]],
    starts: list[dict[str, Any]],
) -> Mismatch | None:
    saves = [record for record in records if record.get("type") == "snapshot_save"]
    restores = [record for record in records if record.get("type") == "snapshot_restore"]
    if len(saves) != 1 or len(restores) != 1:
        return matrix_failure(
            f"{profile}.q{width}.snapshot_count",
            {"save": 1, "restore": 1},
            {"save": len(saves), "restore": len(restores)},
            "snapshot profile requires exactly one save and one restore",
        )
    saved = saves[0]
    restored = restores[0]
    starts_by_request = {start["request"]: start for start in starts}
    saved_start = starts_by_request.get(saved["request"])
    restored_start = starts_by_request.get(restored["request"])
    if saved_start is None or saved_start["restored"]:
        return matrix_failure(
            f"{profile}.q{width}.snapshot_save_request",
            "fresh request",
            saved["request"],
            "snapshot save must belong to the fresh request",
        )
    if restored_start is None or not restored_start["restored"]:
        return matrix_failure(
            f"{profile}.q{width}.snapshot_restore_request",
            "restored request",
            restored["request"],
            "snapshot restore must belong to the restored request",
        )
    prompt_tokens = saved_start["prompt_tokens"]
    if (
        saved["cache_position"] != prompt_tokens
        or restored["cache_position"] != prompt_tokens
        or restored_start["cache_position"] != prompt_tokens
        or saved_start["snap_pos"] != prompt_tokens
    ):
        return matrix_failure(
            f"{profile}.q{width}.snapshot_full_prompt",
            prompt_tokens,
            {
                "save": saved["cache_position"],
                "restore": restored["cache_position"],
                "restored_start": restored_start["cache_position"],
                "fresh_snap_pos": saved_start["snap_pos"],
            },
            "snapshot profile requires a full-prompt save and restore",
        )
    for field in SNAPSHOT_STATE_FIELDS:
        if saved[field] != restored[field]:
            return matrix_failure(
                f"{profile}.q{width}.snapshot.{field}",
                saved[field],
                restored[field],
                "restored snapshot state differs from saved state",
            )
    drafter = records[0]["drafter_sha256"]
    if drafter != "none" and saved["spec_feature_count"] <= 0:
        return matrix_failure(
            f"{profile}.q{width}.snapshot.spec_feature_count",
            "positive",
            saved["spec_feature_count"],
            "DSpark snapshot features are missing",
        )
    return None


def validate_capture_coverage(
    profile: str,
    width: int,
    records: list[dict[str, Any]],
    starts: list[dict[str, Any]],
) -> Mismatch | None:
    if records[0]["drafter_sha256"] == "none":
        return None
    captures_by_request: dict[int, set[int]] = {}
    for record in records:
        if record.get("type") == "capture":
            captures_by_request.setdefault(record["request"], set()).add(record["position_end"])
    for start in starts:
        if start["restored"]:
            continue
        request = start["request"]
        final_position = start["prompt_tokens"]
        expected_ends = {final_position - 3, final_position}
        observed = captures_by_request.get(request, set())
        if not expected_ends.issubset(observed):
            return matrix_failure(
                f"{profile}.q{width}.capture.{request}",
                sorted(expected_ends),
                sorted(observed),
                "both ends of the four-token DSpark capture window are required",
            )
    return None


def validate_matrix(traces: dict[tuple[str, int], list[dict[str, Any]]]) -> Mismatch | None:
    model_configs: list[dict[str, Any]] = []
    for (profile, width), records in sorted(traces.items()):
        configs = [record for record in records if record.get("type") == "model_config"]
        if len(configs) != 1:
            return matrix_failure(
                f"{profile}.q{width}.model_config",
                1,
                len(configs),
                "each production trace must declare model boundaries exactly once",
            )
        model_configs.append(configs[0])
    canonical_config = {
        "n_swa": model_configs[0]["n_swa"],
        "compressor_boundaries": model_configs[0]["compressor_boundaries"],
    }
    for config in model_configs[1:]:
        observed = {
            "n_swa": config["n_swa"],
            "compressor_boundaries": config["compressor_boundaries"],
        }
        if observed != canonical_config:
            return matrix_failure(
                "model_config",
                canonical_config,
                observed,
                "model boundary configuration differs across traces",
            )

    reset_candidates = [traces[("reset", width)] for width in WIDTHS[1:]]
    for width, records in zip(WIDTHS[1:], reset_candidates, strict=True):
        ordinary = any(
            record.get("type") == "step"
            and record.get("width") == width
            and "ordinary" in record.get("relations", [])
            for record in records
        )
        if not ordinary:
            return matrix_failure(
                f"ordinary.q{width}", True, False, "no ordinary full-width production step"
            )

    relations = {
        relation
        for records in traces.values()
        for record in records
        if record.get("type") == "step"
        for relation in record.get("relations", [])
    }
    required_relations = {
        "tail_width_1",
        "tail_width_2",
        "tail_width_3",
        "compressor_before",
        "compressor_on",
        "compressor_after",
        "swa_before",
        "swa_on",
        "swa_after",
        "dspark_capture_window",
    }
    for boundary in canonical_config["compressor_boundaries"]:
        required_relations.update(
            f"compressor_{boundary}_{relation}" for relation in ("before", "on", "after")
        )
    if canonical_config["n_swa"] > 0:
        required_relations.update(
            f"swa_{canonical_config['n_swa']}_{relation}" for relation in ("before", "on", "after")
        )
    missing_relations = sorted(required_relations - relations)
    if missing_relations:
        return matrix_failure(
            "relations",
            sorted(required_relations),
            sorted(relations),
            f"missing matrix relations: {', '.join(missing_relations)}",
        )

    routing_modes = {
        record["routing"]["mode"]
        for records in traces.values()
        for record in records
        if record.get("type") == "layer"
    }
    if routing_modes != {"hash", "learned"}:
        return matrix_failure(
            "routing_modes",
            ["hash", "learned"],
            sorted(routing_modes),
            "both routing implementations must be observed",
        )

    for (profile, width), records in sorted(traces.items()):
        starts = [record for record in records if record.get("type") == "request_start"]
        if len(starts) != 2:
            return matrix_failure(
                f"{profile}.q{width}.repeated_requests",
                2,
                len(starts),
                "same request must be observed exactly twice",
            )
        ordered_starts = sorted(starts, key=lambda record: record["request"])
        expected_restored = [False, profile == "snapshot"]
        observed_restored = [start["restored"] for start in ordered_starts]
        if observed_restored != expected_restored:
            return matrix_failure(
                f"{profile}.q{width}.restored_requests",
                expected_restored,
                observed_restored,
                "request restore lifecycle differs from the profile contract",
            )
        tokens = [record for record in records if record.get("type") == "tokens"]
        if len(tokens) != 2 or any(len(record["token_ids"]) < 2 for record in tokens):
            return matrix_failure(
                f"{profile}.q{width}.continuations",
                "two requests with multiple tokens",
                [len(record["token_ids"]) for record in tokens],
                "multiple continuation tokens are required",
            )
        repetition_mismatch = validate_repeated_request_lifecycle(profile, width, records, starts)
        if repetition_mismatch:
            return repetition_mismatch
        if not any(record.get("type") == "logits" for record in records):
            return matrix_failure(
                f"{profile}.q{width}.logits", True, False, "final logits are missing"
            )
        if (
            profile == "reset"
            and len([record for record in records if record.get("type") == "reset"]) < 2
        ):
            return matrix_failure(f"{profile}.q{width}.reset", 2, 0, "reset events are missing")
        if profile == "snapshot":
            snapshot_mismatch = validate_snapshot_lifecycle(profile, width, records, starts)
            if snapshot_mismatch:
                return snapshot_mismatch
        capture_mismatch = validate_capture_coverage(profile, width, records, starts)
        if capture_mismatch:
            return capture_mismatch
    return None


def compare_trace_dir(trace_dir: Path) -> int:
    summaries: list[dict[str, Any]] = []
    traces: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for profile in PROFILES:
        profile_dir = trace_dir / profile
        oracle_path = profile_dir / "q1.jsonl"
        oracle = load_jsonl(oracle_path)
        traces[(profile, 1)] = oracle
        for width in WIDTHS[1:]:
            candidate = load_jsonl(profile_dir / f"q{width}.jsonl")
            traces[(profile, width)] = candidate
            mismatch = compare_profile(profile, width, oracle, candidate)
            if mismatch:
                print(json.dumps(mismatch.to_json(), sort_keys=True))
                return 1
            summaries.append({"profile": profile, "width": width, "status": "match"})
    mismatch = validate_matrix(traces)
    if mismatch:
        print(json.dumps(mismatch.to_json(), sort_keys=True))
        return 1
    print(json.dumps({"status": "match", "comparisons": summaries}, sort_keys=True))
    return 0


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before ready with status {process.returncode}")
        try:
            http_json(url, timeout=2.0)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout:.1f}s")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def git_revision(binary: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    completed = subprocess.run(
        ["git", "-C", str(binary.parent), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot infer revision; pass --revision")
    return completed.stdout.strip()


def write_manifest(
    path: Path,
    profile: str,
    width: int,
    args: argparse.Namespace,
    revision: str,
    hashes: dict[str, str],
    port: int,
) -> None:
    manifest = {
        "schema": SCHEMA,
        "type": "manifest",
        "profile": profile,
        "width": width,
        "revision": revision,
        "binary_sha256": hashes["binary"],
        "target_sha256": hashes["target"],
        "drafter_sha256": hashes["draft"],
        "prompt_bytes_sha256": hashes["prompt"],
        "request_config": {
            "prefill_mode": "exact",
            "prefill_width": width,
            "exact_bands": width > 1,
            "generated_tokens": args.generated_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "target_device": args.target_device,
            "profile": profile,
            "port": port,
            "server_args": args.server_arg,
        },
        "tolerances": TOLERANCES,
    }
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def run_one(
    profile: str,
    width: int,
    args: argparse.Namespace,
    revision: str,
    hashes: dict[str, str],
    prompt: str,
) -> None:
    profile_dir = args.output_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / f"q{width}.jsonl"
    log_path = profile_dir / f"q{width}.server.log"
    port = args.port_base + (0 if profile == "reset" else 100) + width
    write_manifest(trace_path, profile, width, args, revision, hashes, port)
    env = os.environ.copy()
    env.update(
        {
            "DFLASH_DS4_EXACT_TRACE_PATH": str(trace_path),
            "DFLASH_DS4_EXACT_TRACE_Q": str(width),
            "DFLASH_DS4_EXACT_PREFILL_BANDS": "1" if width > 1 else "0",
            "DFLASH_DS4_FUSED_VERIFY": "0",
            "DFLASH_DS4_FUSED_DECODE": "0",
            "DFLASH_DS4_FUSED_HYBRID_DECODE": "0",
        }
    )
    if args.draft:
        env.update(
            {
                "DFLASH_DS4_SPEC": "1",
                "DFLASH_DS4_DRAFT": str(args.draft),
                "DFLASH_DS4_SPEC_Q": "4",
            }
        )
    cache_slots = "0" if profile == "reset" else "2"
    command = [
        str(args.binary),
        str(args.target),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--target-device",
        args.target_device,
        "--ds4-prefill",
        "exact",
        "--chunk",
        str(width),
        "--prefix-cache-slots",
        "0",
        "--prefill-cache-slots",
        cache_slots,
        "--prefill-compression",
        "off",
        *args.server_arg,
    ]
    request = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.generated_tokens,
        "stream": False,
    }
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(port, process, args.startup_timeout)
            for _ in range(2):
                http_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    request,
                    timeout=args.request_timeout,
                )
        finally:
            stop_process(process)
    load_jsonl(trace_path)


def run_matrix(args: argparse.Namespace) -> int:
    for field in ("binary", "target", "prompt"):
        path = getattr(args, field)
        if not path.is_file():
            raise FileNotFoundError(f"{field} does not exist: {path}")
    if args.draft and not args.draft.is_file():
        raise FileNotFoundError(f"draft does not exist: {args.draft}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = git_revision(args.binary, args.revision)
    hashes = {
        "binary": sha256_file(args.binary),
        "target": sha256_file(args.target),
        "draft": sha256_file(args.draft) if args.draft else "none",
        "prompt": sha256_file(args.prompt),
    }
    prompt = args.prompt.read_text(encoding="utf-8")
    for profile in PROFILES:
        for width in WIDTHS:
            run_one(profile, width, args, revision, hashes, prompt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DS4 production exact-differential harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="generate q=1/2/3/4 production traces")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--draft", type=Path)
    run.add_argument("--prompt", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--revision")
    run.add_argument("--target-device", default="hip:0")
    run.add_argument("--generated-tokens", type=int, default=16)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--port-base", type=int, default=18100)
    run.add_argument("--startup-timeout", type=float, default=900.0)
    run.add_argument("--request-timeout", type=float, default=900.0)
    run.add_argument("--server-arg", action="append", default=[])
    compare = subparsers.add_parser("compare", help="compare all q traces against q=1")
    compare.add_argument("--trace-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            return run_matrix(args)
        return compare_trace_dir(args.trace_dir)
    except (
        FileNotFoundError,
        RuntimeError,
        TimeoutError,
        TraceError,
        urllib.error.URLError,
    ) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
