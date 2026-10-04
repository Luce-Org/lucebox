import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.qualification.qwen36.profiles import load_qwen_ar_profile
from harness.qualification.qwen36.qualify import validate_evidence_bundle, write_reports


def test_qwen_recipe_resolves_to_distinct_hardware_profiles():
    models = Path("harness/qualification/qwen36/profiles.yaml")
    r9700 = load_qwen_ar_profile(models, "qwen36-27b-q4-r9700-ar-c4")
    strix = load_qwen_ar_profile(models, "qwen36-27b-q4-strix-halo-ar-c4")
    assert r9700.recipe_id == strix.recipe_id == "qwen36-27b-q4-ar-c4"
    assert r9700.artifact == strix.artifact
    assert r9700.smoke_workloads == strix.smoke_workloads
    assert r9700.accelerator.architecture == "gfx1201"
    assert r9700.visible_device == "0"
    assert strix.accelerator.architecture == "gfx1151"
    assert strix.visible_device == "1"
    assert r9700.feature_set == strix.feature_set == "ar"
    assert r9700.server_arguments == strix.server_arguments == ()


def test_unknown_release_profile_lists_available_profiles():
    with pytest.raises(ValueError, match="qwen36-27b-q4-r9700-ar-c4"):
        load_qwen_ar_profile(Path("harness/qualification/qwen36/profiles.yaml"), "does-not-exist")


def test_release_report_writes_json_and_markdown(tmp_path):
    report = {
        "profile": "qwen36-27b-q4-strix-halo-ar-c4",
        "family": "qwen-dense",
        "git_commit": "a" * 40,
        "server_url": "http://127.0.0.1:8080",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "verdict": "pass",
        "stages": [
            {
                "id": "R7",
                "description": "report",
                "status": "pass",
                "duration_seconds": 1.0,
                "log": "qualification.json",
            }
        ],
    }
    write_reports(report, tmp_path)
    assert json.loads((tmp_path / "qualification.json").read_text())["verdict"] == "pass"
    assert "Verdict: **PASS**" in (tmp_path / "qualification.md").read_text()
    assert "Feature set: `unknown`" in (tmp_path / "qualification.md").read_text()


def test_active_profile_uses_builtin_commands_without_environment_shells():
    from harness.qualification.qwen36.qualify import commands_for_profile

    for name in ("qwen36-27b-q4-r9700-ar-c4", "qwen36-27b-q4-strix-halo-ar-c4"):
        profile = load_qwen_ar_profile(Path("harness/qualification/qwen36/profiles.yaml"), name)
        commands = commands_for_profile(profile)
        assert sorted(commands) == [f"R{index}" for index in range(7)]
        assert all(
            "harness.qualification.qwen36.qwen36_amd" in command for command in commands.values()
        )
        assert all(
            command.startswith(f"{shlex.quote(sys.executable)} -m ")
            for command in commands.values()
        )


def test_active_profile_owns_both_supported_ar_loads():
    profile = load_qwen_ar_profile(
        Path("harness/qualification/qwen36/profiles.yaml"), "qwen36-27b-q4-strix-halo-ar-c4"
    )
    assert profile.modality == "text"
    assert profile.runner == "lucebox3"
    assert profile.feature_set == "ar"
    assert profile.server_arguments == ()
    assert profile.paged_attention is True
    assert profile.maximum_context == 4096
    assert profile.maximum_concurrency == 4
    assert profile.prefix_cache_slots == 0
    assert [workload.concurrency for workload in profile.smoke_workloads] == [1, 4]
    assert [workload.name for workload in profile.smoke_workloads] == ["paged-ar-c1", "paged-ar-c4"]
    assert profile.power_profile == "platform-managed"
    assert profile.telemetry == "amd"


def test_server_command_appends_only_profile_owned_feature_arguments():
    from harness.qualification.qwen36.qwen36_amd import _server_command

    profile = replace(
        load_qwen_ar_profile(
            Path("harness/qualification/qwen36/profiles.yaml"), "qwen36-27b-q4-strix-halo-ar-c4"
        ),
        server_arguments=("--specla", "--ddtree-budget", "8"),
    )
    command = _server_command(profile, Path("server"), Path("model.gguf"), 9000)
    assert command[-3:] == ["--specla", "--ddtree-budget", "8"]
    assert command.count("--paged-attention") == 1
    assert command.count("--max-ctx") == 1
    assert command.count("--max-concurrency") == 1


def test_generation_uses_the_workload_temperature(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qwen36_amd

    commands = []
    monkeypatch.setattr(qwen36_amd, "_run", lambda command: commands.append(command))
    qwen36_amd._generation(
        name="test",
        prompts=Path("prompts.jsonl"),
        output=tmp_path / "report.json",
        max_tokens=8,
        warmups=0,
        repeats=2,
        concurrency=1,
        temperature=0.25,
        server_url="http://127.0.0.1:8080",
    )
    command = commands[0]
    assert command[command.index("--temperature") + 1] == "0.25"


def test_server_props_must_match_the_profile():
    from harness.qualification.qwen36.qualify import _validate_server_props

    profile = load_qwen_ar_profile(
        Path("harness/qualification/qwen36/profiles.yaml"),
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    props = {
        "daemon": {"alive": True},
        "model_path": f"/opt/models/{profile.artifact.path.name}",
        "default_generation_settings": {"n_ctx": profile.maximum_context},
        "prefix_cache": {"capacity": profile.prefix_cache_slots},
        "speculative": {"enabled": False},
    }
    _validate_server_props(props, profile)
    props["speculative"]["enabled"] = True
    with pytest.raises(ValueError, match="autoregressive decoding"):
        _validate_server_props(props, profile)


def test_performance_candidate_must_match_retained_samples(tmp_path):
    from harness.qualification.qwen36.qualify import _validate_performance_candidate

    profile = load_qwen_ar_profile(
        Path("harness/qualification/qwen36/profiles.yaml"),
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    metrics = {}
    for workload in profile.performance_workloads:
        runs = [
            {"tok_s": float(index), "elapsed_s": 1.0 / index, "completion_tokens": 1}
            for index in range(1, workload.repetitions + 1)
        ]
        report = {"cases": [{"runs": runs}]}
        (tmp_path / f"r5-performance-c{workload.concurrency}.json").write_text(json.dumps(report))
        metrics[f"aggregate_tok_s_c{workload.concurrency}"] = {
            "samples": [run["tok_s"] for run in runs]
        }
        metrics[f"batch_latency_s_c{workload.concurrency}"] = {
            "samples": [run["elapsed_s"] for run in runs]
        }
    candidate = {"metrics": metrics}
    _validate_performance_candidate(tmp_path, profile, candidate)
    candidate["metrics"]["aggregate_tok_s_c1"]["samples"][0] = 999.0
    with pytest.raises(ValueError, match="samples differ"):
        _validate_performance_candidate(tmp_path, profile, candidate)


def test_drift_evidence_is_recomputed_from_retained_samples(tmp_path):
    from harness.qualification.qwen36.drift import evaluate
    from harness.qualification.qwen36.qualify import _validate_drift_evidence

    profile = load_qwen_ar_profile(
        Path("harness/qualification/qwen36/profiles.yaml"),
        "qwen36-27b-q4-strix-halo-ar-c4",
    )
    samples = [
        {
            "health_ok": True,
            "health_latency_ms": 10.0,
            "server_process": {"VmRSS": "102400 kB"},
            "accelerator": {
                "kind": "amd",
                "device": profile.visible_device,
                "ok": True,
            },
        }
        for _ in range(profile.minimum_monitor_samples)
    ]
    report = evaluate(
        samples,
        max_rss_growth_mib=profile.maximum_rss_growth_mib,
        max_health_latency_growth_fraction=(profile.maximum_health_latency_growth_fraction),
        expected_accelerator_device=profile.visible_device,
        minimum_samples=profile.minimum_monitor_samples,
        steady_window_samples=profile.steady_window_samples,
    )
    (tmp_path / "r6-drift.json").write_text(json.dumps(report))
    monitor_log = tmp_path / "resource-monitor.jsonl"
    monitor_log.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    _validate_drift_evidence(tmp_path, profile)

    samples[-1]["health_ok"] = False
    monitor_log.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    with pytest.raises(ValueError, match="differs from the retained monitor log"):
        _validate_drift_evidence(tmp_path, profile)


def test_evidence_bundle_fails_closed_and_hashes_complete_files(tmp_path):
    profile = load_qwen_ar_profile(
        Path("harness/qualification/qwen36/profiles.yaml"), "qwen36-27b-q4-strix-halo-ar-c4"
    )
    assert validate_evidence_bundle(tmp_path, profile)["status"] == "fail"
    json_files = {
        "profile.snapshot.json": {},
        "r0-identity.json": {"profile": profile.name},
        "environment.normalized.json": {},
        "environment.raw.json": {},
        "server-props.json": {},
        "r1-ar-correctness.json": {},
        "r2-generation-quality.json": {},
        "r3-concurrent-determinism.json": {},
        "r4-saturation-c1.json": {},
        "r4-saturation-c2.json": {},
        "r4-saturation-c4.json": {},
        "r6-drift.json": {"status": "pass"},
        "r5-performance-candidate.json": {"profile": profile.name},
        "r5-performance-c1.json": {},
        "r5-performance-c4.json": {},
    }
    for name, value in json_files.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "resource-monitor.jsonl").write_text("{}\n")
    evidence = validate_evidence_bundle(tmp_path, profile)
    assert evidence["status"] == "fail"
    assert "profile snapshot target SHA-256 mismatch" in evidence["failures"]
    assert {item["path"] for item in evidence["files"]} >= set(json_files)


def test_production_workflow_is_manual_and_serializes_both_profiles():
    workflow = Path(".github/workflows/production-quality.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "cron:" not in workflow
    assert "group: production-quality-lucebox3" in workflow
    assert "max-parallel: 1" in workflow
    assert workflow.count("qwen36-27b-q4-r9700-ar-c4") == 1
    assert workflow.count("qwen36-27b-q4-strix-halo-ar-c4") == 1
    assert 'harness/qualification/qwen36/qualify.sh "${{ matrix.profile }}"' in workflow
    assert "actions/checkout@v" not in workflow
    assert "actions/upload-artifact@v" not in workflow
    assert "Clean up qualification server" in workflow
    cleanup = workflow.split("- name: Clean up qualification server", 1)[1].split(
        "- name: Publish qualification summary", 1
    )[0]
    assert "uv run" not in cleanup
    assert 'kill -TERM -- "-$pid"' in cleanup
    assert "timeout-minutes: 165" in workflow
    assert "group: lucebox3-gpu-runner" not in Path(".github/workflows/ci.yml").read_text()
    assert "!${{ runner.temp }}/${{ matrix.artifact_name }}/build/**" in workflow


@pytest.mark.parametrize(
    ("server_url", "models", "message"),
    [
        (
            "http://127.0.0.1:9999",
            Path("harness/qualification/qwen36/profiles.yaml"),
            "does not support --server-url",
        ),
        ("", Path("alternate-models.yaml"), "requires harness/qualification/qwen36/profiles.yaml"),
    ],
)
def test_builtin_qualification_rejects_configuration_overrides(
    monkeypatch, tmp_path, server_url, models, message
):
    from harness.qualification.qwen36 import qualify as release

    models_path = models
    if models.name == "alternate-models.yaml":
        models_path = tmp_path / models.name
        models_path.write_text(Path("harness/qualification/qwen36/profiles.yaml").read_text())
    monkeypatch.setattr(
        release,
        "run_stage",
        lambda *_args, **_kwargs: pytest.fail("override validation must precede stage execution"),
    )
    output_dir = tmp_path / "qualification"

    with pytest.raises(ValueError, match=message):
        release.qualify(
            SimpleNamespace(
                models=models_path,
                profile="qwen36-27b-q4-strix-halo-ar-c4",
                server_url=server_url,
                output_dir=output_dir,
                monitor_interval=None,
            )
        )
    assert not output_dir.exists()


def test_stage_timeout_terminates_the_process_group(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    monkeypatch.setitem(release.STAGE_TIMEOUTS_SECONDS, "R1", 0.05)
    result = release.run_stage("R1", "exec sleep 30", tmp_path, dict(os.environ))
    assert result["status"] == "fail"
    assert result["timed_out"] is True
    assert result["returncode"] == 124


def test_server_cleanup_terminates_group_and_removes_pid_file(tmp_path):
    from harness.qualification.qwen36.qualify import (
        process_identity,
        terminate_server,
    )

    process = subprocess.Popen(["bash", "-lc", "exec sleep 30"], start_new_session=True)
    pid_file = tmp_path / "server.pid"
    pid_file.write_text(process_identity(process.pid) + "\n")
    try:
        terminate_server(pid_file)
        assert process.wait(timeout=1) != 0
        assert not pid_file.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_server_cleanup_accepts_exit_before_signal(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    pid_file = tmp_path / "server.pid"
    pid_file.write_text("123:456\n")
    monkeypatch.setattr(release, "process_start_time", lambda _pid: 456)
    monkeypatch.setattr(release.os, "getpgid", lambda _pid: 123)
    monkeypatch.setattr(
        release.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )

    release.terminate_server(pid_file)
    assert not pid_file.exists()


def test_server_cleanup_removes_pid_file_when_process_already_exited(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    pid_file = tmp_path / "server.pid"
    pid_file.write_text("123:456\n")
    monkeypatch.setattr(
        release,
        "process_start_time",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError),
    )
    monkeypatch.setattr(
        release,
        "terminate_process_group",
        lambda _pid: pytest.fail("an exited process must not be terminated"),
    )

    release.terminate_server(pid_file)
    assert not pid_file.exists()


def test_server_cleanup_refuses_a_reused_pid(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    pid_file = tmp_path / "server.pid"
    pid_file.write_text("123:456\n")
    monkeypatch.setattr(release, "process_start_time", lambda _pid: 789)
    monkeypatch.setattr(
        release,
        "terminate_process_group",
        lambda _pid: pytest.fail("reused PID must not be terminated"),
    )

    with pytest.raises(RuntimeError, match="reused process id"):
        release.terminate_server(pid_file)
    assert pid_file.exists()


def test_failed_server_cleanup_keeps_pid_file_for_retry(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    pid_file = tmp_path / "server.pid"
    pid_file.write_text("123:456\n")
    monkeypatch.setattr(release, "process_start_time", lambda _pid: 456)
    attempts = []

    def terminate(pid):
        attempts.append(pid)
        if len(attempts) == 1:
            raise RuntimeError("still running")

    monkeypatch.setattr(release, "terminate_process_group", terminate)
    with pytest.raises(RuntimeError, match="still running"):
        release.terminate_server(pid_file)
    assert pid_file.read_text() == "123:456\n"

    release.terminate_server(pid_file)
    assert attempts == [123, 123]
    assert not pid_file.exists()


@pytest.mark.parametrize("failure_point", ["pid", "ready"])
def test_server_launch_failure_terminates_the_launched_group(monkeypatch, tmp_path, failure_point):
    from harness.qualification.qwen36 import qwen36_amd

    pid_file = tmp_path / "server.pid"
    terminated = []
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(qwen36_amd.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(qwen36_amd, "process_identity", lambda _pid: "123:456")
    monkeypatch.setattr(qwen36_amd, "terminate_process_group", lambda pid: terminated.append(pid))
    if failure_point == "pid":
        monkeypatch.setattr(
            qwen36_amd,
            "atomic_write_text",
            lambda *_args: (_ for _ in ()).throw(InterruptedError("stop")),
        )
    else:
        monkeypatch.setattr(
            qwen36_amd,
            "_wait_ready",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(InterruptedError("stop")),
        )

    with pytest.raises(InterruptedError, match="stop"):
        qwen36_amd._launch_server(
            ["server"],
            server_log_path=tmp_path / "server.log",
            pid_file=pid_file,
            base_url="http://127.0.0.1:8080",
        )
    assert terminated == [123]
    assert not pid_file.exists()


def test_failed_server_launch_cleanup_keeps_pid_file_for_retry(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qwen36_amd

    pid_file = tmp_path / "server.pid"
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(qwen36_amd.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(qwen36_amd, "process_identity", lambda _pid: "123:456")
    monkeypatch.setattr(
        qwen36_amd,
        "_wait_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InterruptedError("stop")),
    )
    monkeypatch.setattr(
        qwen36_amd,
        "terminate_process_group",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("still running")),
    )

    with pytest.raises(RuntimeError, match="still running"):
        qwen36_amd._launch_server(
            ["server"],
            server_log_path=tmp_path / "server.log",
            pid_file=pid_file,
            base_url="http://127.0.0.1:8080",
        )
    assert pid_file.read_text() == "123:456\n"


def test_qualification_refuses_a_stale_pid_file(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    pid_file = tmp_path / "stale.pid"
    pid_file.write_text("123\n")
    monkeypatch.setenv("LUCE_SERVER_PID_FILE", str(pid_file))
    monkeypatch.setattr(
        release,
        "run_stage",
        lambda *_args, **_kwargs: pytest.fail("stale PID validation must precede stages"),
    )
    output_dir = tmp_path / "qualification"
    with pytest.raises(ValueError, match="PID file already exists"):
        release.qualify(
            SimpleNamespace(
                models=Path("harness/qualification/qwen36/profiles.yaml"),
                profile="qwen36-27b-q4-strix-halo-ar-c4",
                server_url="",
                output_dir=output_dir,
                monitor_interval=None,
            )
        )
    assert pid_file.read_text() == "123\n"
    assert not output_dir.exists()


def test_generation_evidence_requires_the_canonical_case_ids(tmp_path):
    from harness.qualification.qwen36.qualify import _validate_generation_evidence

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"a","prompt":"one"}\n{"id":"b","prompt":"two"}\n')
    report = {
        "schema_version": 2,
        "concurrency": 1,
        "max_tokens": 8,
        "warmups": 0,
        "repeats": 1,
        "temperature": 0,
        "prompts": str(prompts),
        "summary": {"status": "pass", "cases": 1},
        "cases": [
            {
                "id": "a",
                "expected_pass": True,
                "runs": [{"request_count": 1}],
            }
        ],
    }
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="differ from the prompt corpus"):
        _validate_generation_evidence(
            path,
            prompts=prompts,
            concurrency=1,
            maximum_tokens=8,
            warmups=0,
            repetitions=1,
        )


@pytest.mark.parametrize("accuracy", [True, float("nan"), float("-inf")])
def test_generation_evidence_rejects_invalid_gold_accuracy(tmp_path, accuracy):
    from harness.qualification.qwen36.qualify import _validate_generation_evidence

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"case","prompt":"hello"}\n')
    report = {
        "schema_version": 2,
        "concurrency": 1,
        "max_tokens": 8,
        "warmups": 0,
        "repeats": 1,
        "temperature": 0,
        "prompts": str(prompts),
        "summary": {
            "status": "pass",
            "cases": 1,
            "min_gold_accuracy": 0.5,
            "gold_accuracy": accuracy,
            "gold_scored": 1,
        },
        "cases": [
            {
                "id": "case",
                "expected_pass": True,
                "runs": [{"request_count": 1}],
            }
        ],
    }
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="quality score did not pass"):
        _validate_generation_evidence(
            path,
            prompts=prompts,
            concurrency=1,
            maximum_tokens=8,
            warmups=0,
            repetitions=1,
            minimum_gold_accuracy=0.5,
        )


def test_generation_evidence_rejects_non_object_runs(tmp_path):
    from harness.qualification.qwen36.qualify import _validate_generation_evidence

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"case","prompt":"hello"}\n')
    report = {
        "schema_version": 2,
        "concurrency": 1,
        "max_tokens": 8,
        "warmups": 0,
        "repeats": 1,
        "temperature": 0,
        "prompts": str(prompts),
        "summary": {"status": "pass", "cases": 1},
        "cases": [{"id": "case", "expected_pass": True, "runs": [None]}],
    }
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="run evidence is invalid"):
        _validate_generation_evidence(
            path,
            prompts=prompts,
            concurrency=1,
            maximum_tokens=8,
            warmups=0,
            repetitions=1,
        )


def test_unexpected_stage_error_still_writes_terminal_report(monkeypatch, tmp_path):
    from harness.qualification.qwen36 import qualify as release

    def fail_stage(*_args, **_kwargs):
        raise RuntimeError("unexpected stage failure")

    monkeypatch.setattr(release, "run_stage", fail_stage)
    output_dir = tmp_path / "qualification"
    result = release.qualify(
        SimpleNamespace(
            models=Path("harness/qualification/qwen36/profiles.yaml"),
            profile="qwen36-27b-q4-strix-halo-ar-c4",
            server_url="",
            output_dir=output_dir,
            monitor_interval=None,
        )
    )
    assert result == 1
    report = json.loads((output_dir / "qualification.json").read_text())
    assert report["verdict"] == "fail"
    assert report["finished_at"] is not None
    assert report["stages"][-1]["id"] == "R7"
    assert "unexpected stage failure" in report["orchestrator_failures"][0]
