from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "ds4_exact_diff.py"
SPEC = importlib.util.spec_from_file_location("ds4_exact_diff", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def manifest(width: int = 1) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "type": "manifest",
        "profile": "reset",
        "width": width,
        "revision": "a" * 40,
        "binary_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "drafter_sha256": "d" * 64,
        "prompt_bytes_sha256": "e" * 64,
        "request_config": {
            "prefill_width": width,
            "exact_bands": width > 1,
            "port": 18000 + width,
            "temperature": 0,
        },
        "tolerances": MODULE.TOLERANCES,
    }


def layer(width: int = 1, position: int = 4) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "type": "layer",
        "request": 0,
        "layer": 2,
        "position_begin": position - width,
        "position_end": position,
        "token_position": position - 1,
        "routing": {"mode": "learned", "ids": [1, 2], "weights": [0.6, 0.4]},
        "hc_token_hashes": ["0123456789abcdef"],
        "raw_kv_hash": "1" * 16,
        "raw_kv_bytes": 16,
        "compressed_kv_hash": None,
        "compressed_kv_bytes": 0,
        "compressor_state": {"kv": None, "score": None, "n_comp": 0},
        "indexer_state": {"kv": None, "score": None, "n_comp": 0},
    }


def request_start(prompt_tokens: int = 4) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "type": "request_start",
        "request": 0,
        "width": 1,
        "restored": False,
        "cache_position": 0,
        "prompt_token_hash": "f" * 16,
        "prompt_token_ids": list(range(prompt_tokens)),
        "prompt_tokens": prompt_tokens,
        "n_gen": 16,
        "snap_slot": -1,
        "snap_pos": -1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 1,
    }


def step(width: int = 1, position: int = 4, logits_present: bool = True) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "type": "step",
        "request": 0,
        "position_begin": position - width,
        "position_end": position,
        "position": position,
        "width": width,
        "cache_position": position,
        "logits_present": logits_present,
        "relations": [],
    }


def capture(position: int = 4) -> dict[str, object]:
    values = [float(index) for index in range(96)]
    return {
        "schema": MODULE.SCHEMA,
        "type": "capture",
        "request": 0,
        "position_begin": position - 1,
        "position_end": position,
        "layer_ids": [1, 2, 3],
        "row_hash": "a" * 16,
        "total_values": len(values),
        "rows": values,
    }


def request_end(cache_position: int = 4) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "type": "request_end",
        "request": 0,
        "ok": True,
        "cache_position": cache_position,
    }


def complete_trace(
    width: int = 1,
    *,
    cache_position: int = 0,
    restored: bool = False,
    prompt_tokens: int = 4,
    compressor_ratio: int | None = None,
    snapshot_position: int = -1,
) -> list[dict[str, object]]:
    start = request_start(prompt_tokens)
    start["width"] = width
    start["cache_position"] = cache_position
    start["restored"] = restored
    start["snap_pos"] = snapshot_position
    start["snap_slot"] = 0 if snapshot_position >= 0 else -1
    records = [manifest(width), start]
    position = cache_position
    while position < prompt_tokens:
        step_width = min(width, prompt_tokens - position)
        if compressor_ratio is not None:
            step_width = min(step_width, compressor_ratio - position % compressor_ratio)
        position += step_width
        relations = []
        if (
            snapshot_position >= 0
            and position - step_width <= snapshot_position + 4
            and position >= snapshot_position - 4
        ):
            relations.append("snapshot_boundary")
        step_record = step(
            step_width,
            position,
            logits_present=position in {prompt_tokens, snapshot_position},
        )
        step_record["relations"] = relations
        records.extend(
            [
                step_record,
                layer(step_width, position),
            ]
        )
    records.append(request_end(prompt_tokens))
    return records


def write_trace(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_deterministic_sha256(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    assert MODULE.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_serialization_and_comparison_match(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = complete_trace(2)
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    loaded_oracle = MODULE.load_jsonl(oracle_path)
    loaded_candidate = MODULE.load_jsonl(candidate_path)
    assert MODULE.compare_profile("reset", 2, loaded_oracle, loaded_candidate) is None


def test_large_prompt_continuous_producer_contract_matches(tmp_path: Path) -> None:
    oracle_path = tmp_path / "q1-2k.jsonl"
    candidate_path = tmp_path / "q4-2k.jsonl"
    write_trace(oracle_path, complete_trace(prompt_tokens=2048))
    write_trace(candidate_path, complete_trace(4, prompt_tokens=2048))
    assert (
        MODULE.compare_profile(
            "reset", 4, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
        )
        is None
    )


def test_q3_compressor_split_producer_contract_matches(tmp_path: Path) -> None:
    oracle_path = tmp_path / "q1-ratio4.jsonl"
    candidate_path = tmp_path / "q3-ratio4.jsonl"
    write_trace(oracle_path, complete_trace(prompt_tokens=2048, compressor_ratio=4))
    write_trace(
        candidate_path,
        complete_trace(3, prompt_tokens=2048, compressor_ratio=4),
    )
    candidate = MODULE.load_jsonl(candidate_path)
    candidate_steps = MODULE.event_index(candidate, "step")
    assert (0, 4) in candidate_steps
    assert candidate_steps[(0, 4)]["position_begin"] == 3
    assert MODULE.compare_profile("reset", 3, MODULE.load_jsonl(oracle_path), candidate) is None


def test_snapshot_readout_uses_exact_endpoint_not_near_relation(tmp_path: Path) -> None:
    oracle_path = tmp_path / "q1-snapshot.jsonl"
    candidate_path = tmp_path / "q3-snapshot.jsonl"
    write_trace(
        oracle_path,
        complete_trace(prompt_tokens=8, compressor_ratio=4, snapshot_position=4),
    )
    write_trace(
        candidate_path,
        complete_trace(3, prompt_tokens=8, compressor_ratio=4, snapshot_position=4),
    )
    candidate = MODULE.load_jsonl(candidate_path)
    candidate_steps = MODULE.event_index(candidate, "step")
    assert "snapshot_boundary" in candidate_steps[(0, 3)]["relations"]
    assert candidate_steps[(0, 3)]["logits_present"] is False
    assert candidate_steps[(0, 4)]["logits_present"] is True
    assert MODULE.compare_profile("snapshot", 3, MODULE.load_jsonl(oracle_path), candidate) is None


def test_missing_field_fails(tmp_path: Path) -> None:
    broken = layer()
    del broken["raw_kv_hash"]
    path = tmp_path / "broken.jsonl"
    write_trace(path, [manifest(), request_start(), broken, request_end()])
    with pytest.raises(MODULE.TraceError, match="raw_kv_hash"):
        MODULE.load_jsonl(path)


def test_stale_position_fails(tmp_path: Path) -> None:
    first = layer(position=8)
    second = layer(position=4)
    second["layer"] = 3
    path = tmp_path / "stale.jsonl"
    write_trace(path, [manifest(), request_start(), first, second, request_end()])
    with pytest.raises(MODULE.TraceError, match="stale position"):
        MODULE.load_jsonl(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_fails(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonfinite.jsonl"
    text = json.dumps(manifest()) + "\n"
    text += json.dumps(request_start()) + "\n"
    text += json.dumps(layer()).replace("0.6", constant, 1) + "\n"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(MODULE.TraceError, match="non-finite"):
        MODULE.load_jsonl(path)


def test_nonfinite_comparison_never_matches() -> None:
    tolerance = {"atol": 1.0, "rtol": 1.0}
    assert not MODULE.close_enough(math.nan, math.nan, tolerance)
    assert not MODULE.close_enough(math.inf, math.inf, tolerance)


def test_producer_nonfinite_flag_fails(tmp_path: Path) -> None:
    broken = layer()
    broken["non_finite"] = True
    path = tmp_path / "producer-nonfinite.jsonl"
    write_trace(path, [manifest(), request_start(), broken, request_end()])
    with pytest.raises(MODULE.TraceError, match="producer reported non-finite"):
        MODULE.load_jsonl(path)


def test_null_numeric_value_fails(tmp_path: Path) -> None:
    broken = layer()
    broken["routing"] = {"mode": "learned", "ids": [1, 2], "weights": [0.6, None]}
    path = tmp_path / "null-number.jsonl"
    write_trace(path, [manifest(), request_start(), broken, request_end()])
    with pytest.raises(MODULE.TraceError, match="expected a finite number"):
        MODULE.load_jsonl(path)


def test_tolerance_boundary_is_inclusive() -> None:
    tolerance = {"atol": 0.125, "rtol": 0.0}
    assert MODULE.close_enough(1.0, 1.125, tolerance)
    assert not MODULE.close_enough(1.0, 1.1250001, tolerance)


def test_first_mismatch_reports_layer_token_and_field(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = complete_trace(2)
    changed = next(
        record
        for record in candidate
        if record.get("type") == "layer" and record.get("position_end") == 4
    )
    changed["routing"] = {"mode": "learned", "ids": [1, 3], "weights": [0.6, 0.4]}
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.layer == 2
    assert mismatch.token_position == 3
    assert mismatch.field == "routing.ids"


def test_missing_selected_layer_fails(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = [
        record
        for record in complete_trace(2)
        if not (record.get("type") == "layer" and record.get("position_end") == 2)
    ]
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "layer.keys"


def test_partial_step_and_layer_omission_fails(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = [
        record
        for record in complete_trace(2)
        if not (record.get("type") in {"step", "layer"} and record.get("position_end") == 2)
    ]
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "step.coverage"


def test_step_cache_position_mismatch_fails(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = complete_trace(2)
    changed_step = next(
        record
        for record in candidate
        if record.get("type") == "step" and record.get("position_end") == 4
    )
    changed_step["cache_position"] = 3
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "step.coverage"


def test_step_readout_policy_mismatch_fails(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = complete_trace(2)
    changed_step = next(
        record
        for record in candidate
        if record.get("type") == "step" and record.get("position_end") == 2
    )
    changed_step["logits_present"] = True
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "step.logits_present"


def test_restored_request_final_position_uses_full_prompt_length(tmp_path: Path) -> None:
    oracle = complete_trace(cache_position=2, restored=True)
    candidate = complete_trace(2, cache_position=2, restored=True)
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    assert (
        MODULE.compare_profile(
            "snapshot", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
        )
        is None
    )


def test_capture_middle_value_mismatch_fails(tmp_path: Path) -> None:
    oracle_capture = capture()
    candidate_capture = capture()
    candidate_capture["rows"][48] = 999.0
    oracle = complete_trace()
    candidate = complete_trace(2)
    oracle.insert(-1, oracle_capture)
    candidate.insert(-1, candidate_capture)
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "capture.rows[48]"


def test_missing_capture_position_fails(tmp_path: Path) -> None:
    oracle = complete_trace()
    candidate = complete_trace(2)
    oracle.insert(-1, capture())
    oracle_path = tmp_path / "q1.jsonl"
    candidate_path = tmp_path / "q2.jsonl"
    write_trace(oracle_path, oracle)
    write_trace(candidate_path, candidate)
    mismatch = MODULE.compare_profile(
        "reset", 2, MODULE.load_jsonl(oracle_path), MODULE.load_jsonl(candidate_path)
    )
    assert mismatch is not None
    assert mismatch.field == "capture"


def test_prompt_token_ids_are_required(tmp_path: Path) -> None:
    broken_start = request_start()
    del broken_start["prompt_token_ids"]
    path = tmp_path / "missing-prompt-token-ids.jsonl"
    write_trace(path, [manifest(), broken_start, step(), layer(), request_end()])
    with pytest.raises(MODULE.TraceError, match="prompt_token_ids"):
        MODULE.load_jsonl(path)


def test_matrix_requires_model_config_per_trace() -> None:
    traces = {
        (profile, width): [manifest(width)]
        for profile in MODULE.PROFILES
        for width in MODULE.WIDTHS
    }
    mismatch = MODULE.validate_matrix(traces)
    assert mismatch is not None
    assert mismatch.field == "reset.q1.model_config"
    assert mismatch.oracle == 1
    assert mismatch.candidate == 0
