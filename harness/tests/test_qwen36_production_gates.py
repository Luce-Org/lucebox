import contextlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERATION_SCRIPT = ROOT / "harness" / "benchmarks" / "generation_benchmark.py"


def load_generation_benchmark():
    spec = importlib.util.spec_from_file_location("lucebox_generation_benchmark", GENERATION_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_verdict_requires_expectations_and_gold_threshold():
    benchmark = load_generation_benchmark()
    passing = [
        {"id": "a", "expected_pass": True, "gold_correct": True},
        {"id": "b", "expected_pass": True, "gold_correct": False},
    ]
    verdict = benchmark.generation_verdict(passing, min_gold_accuracy=0.5)
    assert verdict == {
        "status": "pass",
        "expected_pass": 2,
        "expected_total": 2,
        "gold_correct": 1,
        "gold_scored": 2,
        "gold_accuracy": 0.5,
        "min_gold_accuracy": 0.5,
    }
    assert benchmark.generation_verdict(passing, min_gold_accuracy=0.6)["status"] == "fail"
    passing[0]["expected_pass"] = False
    assert benchmark.generation_verdict(passing, min_gold_accuracy=0.5)["status"] == "fail"
    with pytest.raises(ValueError, match="no gold-scored cases"):
        benchmark.generation_verdict(
            [{"id": "a", "expected_pass": True, "gold_correct": None}],
            min_gold_accuracy=0.5,
        )


def test_qwen_prompt_corpora_are_fixed_contracts():
    benchmark = load_generation_benchmark()
    smoke = benchmark.load_cases(ROOT / "harness/benchmarks/prompts/generation_smoke.jsonl")
    assert [case["id"] for case in smoke] == [
        "exact-marker",
        "arithmetic",
        "json-marker",
        "short-code",
        "needle-short",
    ]
    assert smoke[0]["expect_exact"] == "LUCEBOX_BENCH_OK"
    assert smoke[1]["expect_exact"] == "391"
    assert smoke[2]["expect_json"] == {"status": "ok", "marker": "LUCEBOX_JSON_OK"}
    assert smoke[4]["expect_exact"] == "alpha-839"

    quality = benchmark.load_cases(ROOT / "harness/benchmarks/prompts/bench_gsm.jsonl")
    assert [case["id"] for case in quality] == [f"gsm_{index:02d}" for index in range(1, 11)]
    assert all(case.get("suite") == "gsm" for case in quality)
    assert all(isinstance(case.get("gold_answer"), str) and case["gold_answer"] for case in quality)

    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml", "qwen36-27b-q4-strix-halo-ar-c4"
    )
    assert profile.minimum_gold_accuracy == 0.60
    assert profile.smoke_prompts == Path("harness/benchmarks/prompts/generation_smoke.jsonl")
    assert profile.quality_prompts == Path("harness/benchmarks/prompts/bench_gsm.jsonl")


def test_generation_expectations_support_exact_text_and_json():
    benchmark = load_generation_benchmark()
    assert benchmark.expected_pass({"expect_exact": "answer"}, " answer\n") == (True, [])
    assert benchmark.expected_pass({"expect_exact": "answer"}, "answer later")[0] is False
    assert benchmark.expected_pass({"expect_json": {"status": "ok"}}, '{"status":"ok"}') == (
        True,
        [],
    )
    assert benchmark.expected_pass({"expect_json": {"status": "ok"}}, "not-json")[0] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmups", -1, "warmups must be non-negative"),
        ("repeats", 0, "repeats must be positive"),
        ("concurrency", 0, "concurrency must be positive"),
    ],
)
def test_generation_run_validates_sampling_controls(field, value, message, tmp_path):
    benchmark = load_generation_benchmark()
    args = type(
        "Args",
        (),
        {
            "warmups": 0,
            "repeats": 1,
            "concurrency": 1,
            "prompts": str(tmp_path / "unused.jsonl"),
        },
    )()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        benchmark.cmd_run(args)


def test_generation_run_rejects_empty_prompt_corpus(tmp_path):
    benchmark = load_generation_benchmark()
    prompts = tmp_path / "empty.jsonl"
    prompts.write_text("")
    args = type(
        "Args",
        (),
        {"warmups": 0, "repeats": 1, "concurrency": 1, "prompts": str(prompts)},
    )()
    with pytest.raises(ValueError, match="at least one case"):
        benchmark.cmd_run(args)


def test_generation_verdict_can_require_byte_determinism():
    benchmark = load_generation_benchmark()
    results = [
        {"expected_pass": True, "gold_correct": None, "deterministic": True},
        {"expected_pass": True, "gold_correct": None, "deterministic": False},
    ]
    verdict = benchmark.generation_verdict(results, min_gold_accuracy=None, require_identical=True)
    assert verdict["status"] == "fail"
    assert verdict["deterministic_cases"] == 1


def test_generation_run_case_discards_warmups_and_bounds_concurrency(monkeypatch):
    benchmark = load_generation_benchmark()
    calls = []

    def fake_post_chat(**kwargs):
        calls.append(kwargs["messages"])
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        }

    monkeypatch.setattr(benchmark, "post_chat", fake_post_chat)
    result = benchmark.run_case(
        case={
            "id": "case",
            "prompt": "prompt",
            "expect_contains": ["ok"],
            "expect_exact": "ok",
        },
        base_url="http://example/v1",
        api_key="",
        model="model",
        max_tokens=8,
        temperature=0,
        timeout=1,
        repeats=3,
        warmups=2,
        concurrency=2,
    )
    assert len(calls) == 10
    assert all(run["request_count"] == 2 for run in result["runs"])
    assert len(result["runs"]) == 3
    assert result["deterministic"] is True
    assert result["expected_pass"] is True
    assert result["expect_exact"] == "ok"


def test_generation_run_case_preserves_concurrent_request_failures(monkeypatch):
    benchmark = load_generation_benchmark()
    calls = 0

    def fake_post_chat(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("request exploded")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        }

    monkeypatch.setattr(benchmark, "post_chat", fake_post_chat)
    result = benchmark.run_case(
        case={"id": "case", "prompt": "prompt", "expect_contains": ["ok"]},
        base_url="http://example/v1",
        api_key="",
        model="model",
        max_tokens=8,
        temperature=0,
        timeout=1,
        repeats=1,
        concurrency=2,
    )

    assert result["expected_pass"] is False
    assert result["deterministic"] is False
    assert len(result["runs"][0]["requests"]) == 2
    assert any("request exploded" in failure for failure in result["expected_failures"])


def test_generation_run_case_keeps_earlier_gold_failure_detail(monkeypatch):
    benchmark = load_generation_benchmark()
    responses = iter(("41", "42"))

    monkeypatch.setattr(
        benchmark,
        "post_chat",
        lambda **_kwargs: {
            "choices": [{"message": {"content": next(responses)}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        },
    )
    result = benchmark.run_case(
        case={"id": "case", "prompt": "prompt", "suite": "gsm", "gold_answer": "42"},
        base_url="http://example/v1",
        api_key="",
        model="model",
        max_tokens=8,
        temperature=0,
        timeout=1,
        repeats=2,
        concurrency=1,
    )

    assert result["gold_correct"] is False
    assert "41" in result["gold_detail"]


@pytest.mark.parametrize("concurrency", [1, 2])
def test_generation_batch_timing_excludes_local_scoring(monkeypatch, concurrency):
    benchmark = load_generation_benchmark()

    monkeypatch.setattr(
        benchmark,
        "post_chat",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        },
    )

    def slow_expected_pass(_case, _text):
        time.sleep(0.05)
        return True, []

    monkeypatch.setattr(benchmark, "expected_pass", slow_expected_pass)
    started = time.perf_counter()
    result = benchmark.run_case(
        case={"id": "case", "prompt": "prompt"},
        base_url="http://example/v1",
        api_key="",
        model="model",
        max_tokens=8,
        temperature=0,
        timeout=1,
        repeats=1,
        concurrency=concurrency,
    )
    wall_elapsed = time.perf_counter() - started
    run = result["runs"][0]
    assert wall_elapsed - run["elapsed_s"] >= 0.025
    assert run["elapsed_s"] == max(request["elapsed_s"] for request in run["requests"])
    assert run["tok_s"] == run["completion_tokens"] / run["elapsed_s"]


def test_generation_single_sample_is_not_determinism_evidence(monkeypatch):
    benchmark = load_generation_benchmark()
    monkeypatch.setattr(
        benchmark,
        "post_chat",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        },
    )
    result = benchmark.run_case(
        case={"id": "case", "prompt": "prompt"},
        base_url="http://example/v1",
        api_key="",
        model="model",
        max_tokens=8,
        temperature=0,
        timeout=1,
        repeats=1,
        concurrency=1,
    )
    assert result["deterministic"] is False


def _mutated_profile(tmp_path, mutate):
    manifest = yaml.safe_load((ROOT / "harness/qualification/qwen36/profiles.yaml").read_text())
    recipe = manifest["families"]["qwen-dense"]["recipes"]["qwen36-27b-q4-ar-c4"]
    mutate(recipe)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


@pytest.mark.parametrize("value", ["true", 1, None])
def test_profile_rejects_non_boolean_determinism(value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"]["smoke"]["workloads"][1].__setitem__(
            "require_identical", value
        ),
    )
    with pytest.raises(ProfileError, match="require_identical must be a boolean"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize("section", ["smoke", "performance"])
def test_profile_rejects_nonzero_workload_temperature(section, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"][section]["workloads"][0].__setitem__(
            "temperature", 0.5
        ),
    )
    with pytest.raises(ProfileError, match="workload temperature must be zero"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_profile_rejects_coerced_workload_warmups(value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"]["smoke"]["workloads"][0].__setitem__(
            "warmups", value
        ),
    )
    with pytest.raises(ProfileError, match="warmups must be a non-negative integer"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize("value", [None, 1, "   "])
def test_profile_rejects_non_string_or_empty_workload_names(value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"]["smoke"]["workloads"][0].__setitem__("name", value),
    )
    with pytest.raises(ProfileError, match="name must be a non-empty string"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize("value", [True, 1.5, "2", -1])
def test_profile_rejects_coerced_prefix_cache_slots(value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"].__setitem__("prefix_cache_slots", value),
    )
    with pytest.raises(ProfileError, match="prefix_cache_slots must be a non-negative integer"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_profile_requires_identical_output_at_maximum_concurrency(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"]["smoke"]["workloads"][1].__setitem__(
            "require_identical", False
        ),
    )
    with pytest.raises(ProfileError, match="must require identical output"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_profile_requires_at_least_two_concurrent_slots(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(tmp_path, lambda recipe: recipe.__setitem__("max_concurrency", 1))
    with pytest.raises(ProfileError, match="max_concurrency must be at least 2"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_profile_rejects_out_of_range_server_port(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"].__setitem__("server_port", 65536),
    )
    with pytest.raises(ProfileError, match="server_port must be at most 65535"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize("value", ["gpu", "\u0661"])
def test_profile_requires_ascii_decimal_visible_device(value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    manifest = yaml.safe_load((ROOT / "harness/qualification/qwen36/profiles.yaml").read_text())
    manifest["profiles"]["qwen36-27b-q4-strix-halo-ar-c4"]["visible_device"] = value
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ProfileError, match="ASCII decimal visible_device"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("--specla", "array of argument tokens"),
        (["value-first", "--specla"], "must start with an option"),
        (["--specla", "--specla"], "duplicate options"),
        (["--max-ctx=8192"], "cannot override profile-owned options"),
        (["--paged-attention"], "cannot override profile-owned options"),
        (["--ddtree-budget=8"], "separate option and value tokens"),
    ],
)
def test_profile_rejects_malformed_or_reserved_server_arguments(value, message, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe.__setitem__("server_arguments", value),
    )
    with pytest.raises(ProfileError, match=message):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_ar_profile_rejects_specla_flag_under_the_ar_profile(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe.__setitem__("server_arguments", ["--specla"]),
    )
    with pytest.raises(ProfileError, match="separately named profile"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_profile_requires_feature_set_in_recipe_and_profile(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe.__setitem__("feature_set", "specla"),
    )
    with pytest.raises(ProfileError, match="must include the feature_set"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_profile_rejects_parent_path_components(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe.__setitem__("target_artifact", "../outside.gguf"),
    )
    with pytest.raises(ProfileError, match="without parent-directory components"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_artifact_path_cannot_escape_models_root_through_a_symlink(tmp_path):
    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml", "qwen36-27b-q4-strix-halo-ar-c4"
    )
    models_root = tmp_path / "models"
    outside = tmp_path / "outside"
    models_root.mkdir()
    outside.mkdir()
    (models_root / profile.artifact.path).symlink_to(outside / "model.gguf")

    with pytest.raises(ValueError, match="escapes the models root"):
        profile.artifact_path(models_root)


def test_profile_requires_two_monitor_samples(tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    path = _mutated_profile(
        tmp_path,
        lambda recipe: recipe["qualification"]["drift"].__setitem__("minimum_samples", 1),
    )
    with pytest.raises(ProfileError, match="minimum_samples must be at least 2"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


@pytest.mark.parametrize(
    "field",
    [
        "monitor_interval_seconds",
        "maximum_rss_growth_mib",
        "maximum_health_latency_growth_fraction",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_profile_rejects_nonfinite_limits(field, value, tmp_path):
    from harness.qualification.qwen36.profiles import ProfileError, load_qwen_ar_profile

    def mutate(recipe):
        recipe["qualification"]["drift"][field] = value

    path = _mutated_profile(tmp_path, mutate)
    with pytest.raises(ProfileError, match="must be finite"):
        load_qwen_ar_profile(path, "qwen36-27b-q4-strix-halo-ar-c4")


def test_environment_driver_identity_preserves_internal_whitespace(monkeypatch):
    from harness.qualification.qwen36 import environment

    kernel = (
        "Linux version 6.17.0 (gcc 13.3.0, GNU ld 2.42)\n"
        "#35 SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2026\n"
    )
    monkeypatch.setattr(environment.Path, "read_text", lambda _path: kernel)
    assert environment._kernel_identity() == (
        "Linux version 6.17.0 (gcc 13.3.0, GNU ld 2.42) "
        "#35 SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2026"
    )


def test_environment_capture_times_out_with_a_stage_error(monkeypatch):
    from harness.qualification.qwen36 import environment

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["amd-smi"], 30)

    monkeypatch.setattr(environment.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="environment command timed out: amd-smi"):
        environment._capture(["amd-smi"])


def test_environment_uses_absolute_amd_smi_and_requires_string_driver(monkeypatch):
    from harness.qualification.qwen36 import environment
    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml",
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    static = {
        "gpu_data": [
            {
                "asic": {
                    "target_graphics_version": "gfx1151",
                    "market_name": "AMD Radeon Graphics",
                    "device_id": "0x1586",
                    "num_compute_units": 40,
                },
                "driver": {"name": "amdgpu", "version": "test"},
            }
        ]
    }
    live = {"card1": {"Performance Level": "auto"}}
    commands = []

    def capture(command, timeout_seconds=30.0):
        commands.append((command, timeout_seconds))
        return [json.dumps(static), json.dumps(live), "HIP test\nclang test", "cmake test"][
            len(commands) - 1
        ]

    monkeypatch.setattr(environment, "_capture", capture)
    monkeypatch.setattr(environment, "_kernel_identity", lambda: "Linux test")
    environment.observe_amd_environment(profile, [])
    assert commands[0][0][0] == "/opt/rocm/bin/amd-smi"

    static["gpu_data"][0]["driver"]["version"] = 7
    commands.clear()
    with pytest.raises(ValueError, match="driver identity is unavailable"):
        environment.observe_amd_environment(profile, [])


def test_environment_selects_only_the_requested_live_device():
    from harness.qualification.qwen36.environment import _selected_live_device

    assert _selected_live_device({"card1": {"Performance Level": "auto"}}, "1") == {
        "Performance Level": "auto"
    }
    with pytest.raises(ValueError, match="selected accelerator card1"):
        _selected_live_device(
            {
                "card0": {"Performance Level": "auto"},
                "card1": {"Performance Level": "auto"},
            },
            "1",
        )


def test_environment_requires_the_profile_accelerator_name():
    from harness.qualification.qwen36.environment import _static_identity
    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml",
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    gpu = {
        "asic": {
            "target_graphics_version": "gfx1151",
            "market_name": "AMD Radeon Graphics",
            "device_id": "0x1586",
            "num_compute_units": 40,
        },
        "driver": {"name": "amdgpu", "version": "test"},
    }
    assert _static_identity(profile, gpu)[2] == "AMD Radeon Graphics"
    gpu["asic"]["market_name"] = "Different gfx1151 accelerator"
    with pytest.raises(ValueError, match="name differs from the profile"):
        _static_identity(profile, gpu)
    gpu["asic"]["market_name"] = "AMD Radeon Graphics"
    gpu["asic"]["device_id"] = "0xffff"
    with pytest.raises(ValueError, match="hardware identity differs from the profile"):
        _static_identity(profile, gpu)
    gpu["asic"] = []
    with pytest.raises(ValueError, match="identity fields are invalid"):
        _static_identity(profile, gpu)


def test_qwen_hardware_probe_times_out_with_a_stage_error(monkeypatch):
    from harness.qualification.qwen36 import qwen36_amd

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["rocminfo"], 30)

    monkeypatch.setattr(qwen36_amd.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="qualification command timed out: rocminfo"):
        qwen36_amd._capture(["rocminfo"])


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), True])
def test_performance_conversion_rejects_invalid_samples(value):
    from harness.qualification.qwen36.performance import _samples

    report = {"cases": [{"runs": [{"elapsed_s": value}]}]}
    with pytest.raises(ValueError, match="positive finite elapsed_s"):
        _samples(report, "elapsed_s")


def test_qwen_readiness_normalizes_trailing_slash(monkeypatch):
    from harness.qualification.qwen36 import qwen36_amd

    requested = []
    process = type("Process", (), {"poll": lambda self: None})()
    monkeypatch.setattr(
        qwen36_amd.urllib.request,
        "urlopen",
        lambda url, timeout: requested.append((url, timeout)) or contextlib.nullcontext(),
    )
    qwen36_amd._wait_ready("http://127.0.0.1:8080/", process, timeout=0.1)
    assert requested == [("http://127.0.0.1:8080/props", 2)]


def test_qwen_readiness_fails_immediately_when_server_exits():
    from harness.qualification.qwen36 import qwen36_amd

    process = type("Process", (), {"poll": lambda self: 7})()
    with pytest.raises(RuntimeError, match="exited during startup with code 7"):
        qwen36_amd._wait_ready("http://127.0.0.1:8080", process)


def test_qwen_readiness_propagates_interrupt(monkeypatch):
    from harness.qualification.qwen36 import qwen36_amd

    process = type("Process", (), {"poll": lambda self: None})()

    def interrupt(*_args, **_kwargs):
        raise InterruptedError("stop")

    monkeypatch.setattr(qwen36_amd.urllib.request, "urlopen", interrupt)
    with pytest.raises(InterruptedError, match="stop"):
        qwen36_amd._wait_ready("http://127.0.0.1:8080", process)


def test_drift_gate_passes_stable_samples_and_fails_endpoint_loss():
    from harness.qualification.qwen36.drift import evaluate

    samples = [
        {
            "health_ok": True,
            "health_latency_ms": 10.0,
            "server_process": {"VmRSS": "102400 kB"},
            "accelerator": {"kind": "amd", "device": "1", "ok": True},
        },
        {
            "health_ok": True,
            "health_latency_ms": 11.0,
            "server_process": {"VmRSS": "103424 kB"},
            "accelerator": {"kind": "amd", "device": "1", "ok": True},
        },
        {
            "health_ok": True,
            "health_latency_ms": 11.0,
            "server_process": {"VmRSS": "103424 kB"},
            "accelerator": {"kind": "amd", "device": "1", "ok": True},
        },
    ]
    report = evaluate(
        samples,
        max_rss_growth_mib=2.0,
        max_health_latency_growth_fraction=0.2,
        expected_accelerator_device="1",
    )
    assert report["status"] == "pass"
    samples[-1]["health_ok"] = False
    assert (
        evaluate(
            samples,
            max_rss_growth_mib=2.0,
            max_health_latency_growth_fraction=0.2,
            expected_accelerator_device="1",
        )["status"]
        == "fail"
    )


def test_drift_gate_rejects_absent_telemetry_and_incomplete_samples():
    from harness.qualification.qwen36.drift import evaluate

    samples = [
        {
            "health_ok": True,
            "health_latency_ms": 10.0,
            "server_process": {"VmRSS": "102400 kB"},
            "accelerator": {"kind": "none", "device": "1", "ok": False},
        }
        for _ in range(6)
    ]
    report = evaluate(
        samples,
        max_rss_growth_mib=2.0,
        max_health_latency_growth_fraction=0.2,
        expected_accelerator_device="1",
        minimum_samples=6,
        steady_window_samples=2,
    )
    assert report["status"] == "fail"
    assert "accelerator telemetry failed during qualification" in report["failures"]
    del samples[-1]["server_process"]["VmRSS"]
    report = evaluate(
        samples,
        max_rss_growth_mib=2.0,
        max_health_latency_growth_fraction=0.2,
        expected_accelerator_device="1",
        minimum_samples=6,
        steady_window_samples=2,
    )
    assert "server RSS samples are unavailable" in report["failures"]


def test_drift_gate_requires_disjoint_windows():
    from harness.qualification.qwen36.drift import evaluate

    with pytest.raises(ValueError, match="at least 6"):
        evaluate(
            [
                {
                    "health_ok": True,
                    "health_latency_ms": 10.0,
                    "server_process": {"VmRSS": "102400 kB"},
                    "accelerator": {"kind": "amd", "device": "1", "ok": True},
                }
                for _ in range(5)
            ],
            max_rss_growth_mib=2.0,
            max_health_latency_growth_fraction=0.2,
            expected_accelerator_device="1",
            minimum_samples=4,
            steady_window_samples=2,
        )


def test_resource_monitor_stop_exceeds_bounded_sample_time(tmp_path):
    from harness.qualification.qwen36 import qualify as release

    joined = []
    monitor = release.ResourceMonitor(
        "http://127.0.0.1:8080",
        tmp_path / "monitor.jsonl",
        interval=1,
        pid_file=None,
        accelerator_device="1",
    )
    monitor.thread = SimpleNamespace(
        join=lambda timeout: joined.append(timeout),
        is_alive=lambda: False,
    )
    monitor.stop()
    assert joined == [15.0]


def test_resource_monitor_uses_health_and_rejects_malformed_amd_json(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    real_accelerator_status = release.accelerator_status
    endpoints = []

    def fake_probe(_url, endpoint, timeout=3.0):
        endpoints.append((endpoint, timeout))
        return 200

    monkeypatch.setattr(release, "http_probe", fake_probe)
    monkeypatch.setattr(
        release,
        "accelerator_status",
        lambda device: {"kind": "amd", "device": device, "ok": True, "data": {"card1": {}}},
    )
    sample = release.ResourceMonitor(
        "http://127.0.0.1:8080",
        tmp_path / "monitor.jsonl",
        interval=1,
        pid_file=None,
        accelerator_device="1",
    )._sample()
    assert endpoints == [("/health", 3.0)]
    assert sample["health_ok"] is True
    assert sample["accelerator"]["device"] == "1"

    monkeypatch.setattr(release.Path, "is_file", lambda _path: True)
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result",
            (),
            {"returncode": 0, "stdout": "not-json", "stderr": ""},
        )(),
    )
    telemetry = real_accelerator_status("1")
    assert telemetry["device"] == "1"
    assert "invalid rocm-smi JSON" in telemetry["error"]
    assert telemetry["ok"] is False


def test_generation_adapter_keeps_ar_concurrency_in_metric_identity(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import performance
    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    monkeypatch.setattr(performance, "git_commit", lambda: "b" * 40)
    target = tmp_path / "target.gguf"
    target.write_bytes(b"target")
    runs = [
        {"tok_s": float(value), "elapsed_s": 1.0 / value, "completion_tokens": 1}
        for value in range(20, 40)
    ]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","prompt":"hello"}\n')
    reports = [
        {
            "schema_version": 2,
            "prompts": str(prompts),
            "max_tokens": 32,
            "temperature": 0,
            "warmups": 5,
            "repeats": 20,
            "concurrency": concurrency,
            "cases": [{"runs": [{**run, "token_count_source": "usage"} for run in runs]}],
            "summary": {"status": "pass"},
        }
        for concurrency in (1, 4)
    ]
    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml",
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    report = performance.from_generation_reports(
        reports,
        profile=profile,
        target=target,
        environment={
            "accelerator": {
                "role": "target",
                "declared_name": "Strix Halo Radeon 8060S",
                "observed_name": "Radeon 8060S Graphics",
                "architecture": "gfx1151",
                "visible_device": "1",
            },
            "driver": "amdgpu test",
            "kernel": "Linux test",
            "runtime": "HIP test",
            "compiler": "clang test",
            "cmake": "cmake test",
            "power_profile": "platform-managed",
            "performance_level": "auto",
            "build_type": "Release",
            "configure_arguments": ["cmake", "-DCMAKE_BUILD_TYPE=Release"],
        },
        qualification_subject={
            "kind": "source-build",
            "server_binary_sha256": "c" * 64,
        },
    )
    assert report["comparison_identity"]["model"] == {
        "target": target.name,
        "target_sha256": performance.sha256(target),
    }
    assert report["comparison_identity"]["profile"]["feature_set"] == "ar"
    assert report["comparison_identity"]["run"]["build_flags"] == ["-DCMAKE_BUILD_TYPE=Release"]
    assert report["comparison_identity"]["run"]["environment"] == {
        "decode_mode": "autoregressive",
        "paged_attention": True,
        "maximum_context": 4096,
        "maximum_concurrency": 4,
        "prefix_cache_slots": 0,
        "server_arguments": [],
        "concurrency": [1, 4],
        "prompt_corpus": str(prompts),
        "prompt_sha256": performance.sha256(prompts),
        "max_tokens": 32,
        "temperature": 0,
    }
    assert sorted(report["metrics"]) == [
        "aggregate_tok_s_c1",
        "aggregate_tok_s_c4",
        "batch_latency_s_c1",
        "batch_latency_s_c4",
    ]
    reports[0]["warmups"] = "5"
    with pytest.raises(ValueError, match="sample plan has invalid types at c1"):
        performance.from_generation_reports(
            reports,
            profile=profile,
            target=target,
            environment={},
            qualification_subject={},
        )


def test_generation_adapter_rejects_inconsistent_throughput(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import performance
    from harness.qualification.qwen36.profiles import load_qwen_ar_profile

    monkeypatch.setattr(performance, "git_commit", lambda: "b" * 40)
    target = tmp_path / "target.gguf"
    target.write_bytes(b"target")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"p","prompt":"hello"}\n')
    reports = [
        {
            "schema_version": 2,
            "prompts": str(prompts),
            "max_tokens": 32,
            "temperature": 0,
            "warmups": 5,
            "repeats": 20,
            "concurrency": concurrency,
            "cases": [
                {
                    "runs": [
                        {
                            "tok_s": 999.0,
                            "elapsed_s": 0.5,
                            "completion_tokens": 1,
                            "token_count_source": "usage",
                        }
                    ]
                }
            ],
            "summary": {"status": "pass"},
        }
        for concurrency in (1, 4)
    ]
    profile = load_qwen_ar_profile(
        ROOT / "harness/qualification/qwen36/profiles.yaml",
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    with pytest.raises(ValueError, match="tok_s differs"):
        performance.from_generation_reports(
            reports,
            profile=profile,
            target=target,
            environment={},
            qualification_subject={},
        )
