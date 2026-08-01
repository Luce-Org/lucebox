"""Tests for bounded on-machine calibration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from lucebox.cli import app
from lucebox.types import Config, HostFacts, ModelMeta, PlacementRuntime
from typer.testing import CliRunner

from lucebox import calibration, config, download


def _qwen_config(tmp_path: Path, *, budget: int = 22, with_draft: bool = True) -> Config:
    preset = download.PRESETS["qwen3.6-27b"]
    cfg = Config(
        models_dir=tmp_path / "models",
        host=HostFacts(
            gpu_vendor="nvidia",
            gpu_name="RTX 3090",
            gpu_count=1,
            gpu_sm="86",
            vram_gb=24,
        ),
        model=ModelMeta(
            preset=preset.name,
            target_file=preset.target_file,
            draft_file=preset.draft_file or "",
        ),
    )
    cfg = replace(cfg, dflash=replace(cfg.dflash, budget=budget))
    if with_draft and preset.draft_file:
        draft = cfg.models_dir / "draft" / preset.draft_file
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_bytes(b"draft")
    return cfg


def _turn(tps: float, *, cache_hit: bool = False) -> calibration.TurnMeasurement:
    return calibration.TurnMeasurement(
        decode_tokens_per_sec=tps,
        prefill_tokens_per_sec=1000.0,
        completion_tokens=32,
        cache_hit=cache_hit,
        cached_prefix_tokens=900 if cache_hit else 0,
        prefilled_tokens=100 if cache_hit else 1000,
    )


def _result(budget: int, score: float, signature: str = "same") -> calibration.ProbeResult:
    return calibration.ProbeResult(
        budget=budget,
        model="qwen3.6-27b",
        architecture="qwen35",
        score=score,
        response_signature=signature,
        cold=_turn(score),
        warm=_turn(score, cache_hit=True),
        server={},
    )


def test_candidate_budgets_are_bounded_and_keep_baseline_first(tmp_path: Path) -> None:
    assert calibration.candidate_budgets(_qwen_config(tmp_path, budget=22)) == (22, 16, 32)
    assert calibration.candidate_budgets(_qwen_config(tmp_path, budget=8)) == (8, 4, 16)
    assert calibration.candidate_budgets(_qwen_config(tmp_path, budget=40)) == (40, 32, 64)


def test_candidate_budgets_do_not_sweep_inert_or_unavailable_runtime(tmp_path: Path) -> None:
    no_draft = _qwen_config(tmp_path, with_draft=False)
    assert calibration.candidate_budgets(no_draft) == (22,)

    deepseek = replace(
        no_draft,
        model=ModelMeta(preset="deepseek-v4-flash"),
    )
    assert calibration.candidate_budgets(deepseek) == (22,)

    spark = _qwen_config(tmp_path)
    spark = replace(spark, dflash=replace(spark.dflash, spark=True))
    assert calibration.candidate_budgets(spark) == (22,)


def test_internal_budget_protocol_requires_an_optimization_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lucebox.cli as cli

    monkeypatch.setattr(cli, "_load_or_build", lambda: _qwen_config(tmp_path))
    monkeypatch.setattr(config, "optimization_mode", lambda: "unconfigured")

    result = CliRunner().invoke(app, ["_calibration", "budgets"])

    assert result.exit_code == 2
    assert "run `lucebox optimize` first" in result.output


def test_probe_uses_server_timings_and_warm_prefix_cache(tmp_path: Path) -> None:
    cfg = _qwen_config(tmp_path)
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET" and request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "build_info": "test",
                    "model": {"arch": "qwen35"},
                    "runtime": {"backend": "cuda"},
                    "speculative": {"enabled": True, "ddtree_budget": 22},
                    "capabilities": {"tools_supported": True},
                    "pflash": {"enabled": True},
                    "prefix_cache": {"capacity": 32},
                },
            )
        assert request.method == "POST"
        body = json.loads(request.content)
        assert "tools" in body
        assert body["tool_choice"] == "none"
        posts += 1
        if posts == 4:
            assert [message["role"] for message in body["messages"][-3:]] == [
                "assistant",
                "tool",
                "user",
            ]
            assert body["messages"][-2]["tool_call_id"] == "call_test"
        completion = 8 if posts == 1 else 32
        text = "ready" if posts == 1 else f"answer-{posts}"
        message: dict[str, object] = {"role": "assistant", "content": text}
        if posts == 3:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "run_tests",
                            "arguments": '{"target": "unit"}',
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {
                    "completion_tokens": completion,
                    "timings": {
                        "decode_tokens_per_sec": 20.0 if posts < 4 else 25.0,
                        "prefill_ms": 500.0 if posts == 2 else 10.0,
                        "cache_hit": posts == 4,
                        "cached_prefix_tokens": 900 if posts == 4 else 0,
                        "prefilled_tokens": 100 if posts == 4 else 1000,
                    },
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = calibration.probe(
            cfg,
            22,
            client=client,
            base_urls=("http://test",),
            ready_timeout_s=1,
        )

    assert posts == 4
    assert result.cold.prefill_tokens_per_sec == 2000.0
    assert result.cold.decode_tokens_per_sec == 20.0
    assert result.warm.decode_tokens_per_sec == 25.0
    assert result.warm.cache_hit is True
    assert result.warm.cached_prefix_tokens == 900
    assert result.score == pytest.approx(22.222222)


def test_probe_rejects_a_server_started_with_the_wrong_budget(tmp_path: Path) -> None:
    cfg = _qwen_config(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": {"arch": "qwen35"},
                "speculative": {"enabled": True, "ddtree_budget": 16},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="expected 22"):
            calibration.probe(
                cfg,
                22,
                client=client,
                base_urls=("http://test",),
                ready_timeout_s=1,
            )


def test_tool_call_signature_ignores_transport_ids_and_json_spacing() -> None:
    def payload(call_id: str, arguments: str) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "run_tests",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                }
            ]
        }

    first_message, first_signature = calibration._assistant_turn(
        payload("call_one", '{"target":"unit"}')
    )
    second_message, second_signature = calibration._assistant_turn(
        payload("call_two", '{ "target": "unit" }')
    )

    assert first_message["tool_calls"][0]["id"] == "call_one"
    assert second_message["tool_calls"][0]["id"] == "call_two"
    assert first_signature == second_signature


def test_probe_result_rejects_corrupt_cached_measurements() -> None:
    raw = _result(22, 10.0).as_dict()
    raw["warm"]["decode_tokens_per_sec"] = "not-a-number"

    with pytest.raises(ValueError, match="malformed calibration result"):
        calibration.ProbeResult.from_dict(raw)


def test_finish_requires_quality_equivalence_and_material_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _qwen_config(tmp_path)
    result_dir = tmp_path / "results"
    for result in (
        _result(22, 10.0),
        _result(16, 10.4),
        _result(32, 15.0, signature="different"),
    ):
        calibration.write_probe(result_dir / f"budget-{result.budget}.json", result)

    applied: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        calibration,
        "apply_budget",
        lambda budget, *, final=False: applied.append((budget, final)),
    )
    monkeypatch.setattr(
        calibration,
        "calibration_record_path",
        lambda: tmp_path / "calibration.json",
    )

    summary = calibration.finish(result_dir, 22, cfg=cfg)

    assert summary.winner.budget == 22
    assert summary.rejected_budgets == (32,)
    assert applied == [(22, True)]
    assert (result_dir / "winner").read_text() == "22\n"


def test_finish_selects_a_five_percent_faster_equivalent_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _qwen_config(tmp_path)
    result_dir = tmp_path / "results"
    for result in (_result(22, 10.0), _result(16, 11.0)):
        calibration.write_probe(result_dir / f"budget-{result.budget}.json", result)
    monkeypatch.setattr(calibration, "apply_budget", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        calibration,
        "calibration_record_path",
        lambda: tmp_path / "calibration.json",
    )

    summary = calibration.finish(result_dir, 22, cfg=cfg)

    assert summary.winner.budget == 16


def test_finish_rejects_changed_warm_cache_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _qwen_config(tmp_path)
    result_dir = tmp_path / "results"
    baseline = _result(22, 10.0)
    changed_cache = replace(
        _result(16, 15.0),
        warm=replace(_turn(15.0, cache_hit=True), cached_prefix_tokens=512),
    )
    for result in (baseline, changed_cache):
        calibration.write_probe(result_dir / f"budget-{result.budget}.json", result)
    monkeypatch.setattr(calibration, "apply_budget", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        calibration,
        "calibration_record_path",
        lambda: tmp_path / "calibration.json",
    )

    summary = calibration.finish(result_dir, 22, cfg=cfg)

    assert summary.winner.budget == 22
    assert summary.rejected_budgets == (16,)


def test_apply_budget_preserves_automatic_profile_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path))
    config.write_optimization_runtime(
        _qwen_config(tmp_path).dflash,
        placement=PlacementRuntime(target_device="cuda:0"),
        mode="automatic",
    )

    calibration.apply_budget(16)

    loaded = config.load()
    assert loaded is not None
    assert loaded.dflash.budget == 16
    assert loaded.placement.target_device == "cuda:0"
    doc = config.load_doc()
    assert doc["autotune"]["mode"] == "automatic"
    assert doc["autotune"]["source"] == "calibration-candidate"


def test_cached_result_is_invalidated_by_runtime_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path))
    cfg = _qwen_config(tmp_path)
    record = {
        "schema": calibration.SCHEMA_VERSION,
        "fingerprint": calibration.fingerprint(cfg),
    }
    calibration.calibration_record_path().write_text(json.dumps(record))

    assert calibration.current_record(cfg) == record
    changed = replace(cfg, dflash=replace(cfg.dflash, max_ctx=65536))
    assert calibration.current_record(changed) is None
