#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from harness.qualification.qwen36.drift import evaluate as evaluate_drift
from harness.qualification.qwen36.profiles import QwenArProfile, load_qwen_ar_profile

ROOT = Path(__file__).resolve().parents[3]
STAGES = {
    "R0": "setup, exact build/config record, and long-lived server start",
    "R1": "decode-mode correctness on the canonical smoke corpus",
    "R2": "generation-based quality evaluation",
    "R3": "byte determinism at the profile concurrency",
    "R4": "bounded concurrency or single-sequence workload sweep",
    "R5": "performance evidence capture",
    "R6": "resource and health-endpoint drift verdict",
}
STAGE_TIMEOUTS_SECONDS = {
    "R0": 2700,
    "R1": 900,
    "R2": 1200,
    "R3": 900,
    "R4": 900,
    "R5": 2400,
    "R6": 300,
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _profiles(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("model manifest must contain a profiles map")
    return profiles


def list_profiles(path: Path) -> list[str]:
    manifest = yaml.safe_load(path.read_text())
    return sorted(_profiles(manifest))


def http_json(server_url: str, endpoint: str, timeout: float = 3.0) -> Any:
    request = urllib.request.Request(server_url.rstrip("/") + endpoint)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def http_probe(server_url: str, endpoint: str, timeout: float = 3.0) -> int:
    request = urllib.request.Request(server_url.rstrip("/") + endpoint)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
        return response.status


def proc_meminfo() -> dict[str, int]:
    result = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return result


def process_start_time(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat.rsplit(")", 1)[1].split()
    if len(fields) <= 19:
        raise ValueError("process stat is incomplete")
    return int(fields[19])


def process_identity(pid: int) -> str:
    return f"{pid}:{process_start_time(pid)}"


def read_process_identity(pid_file: Path) -> tuple[int, int]:
    values = pid_file.read_text().strip().split(":")
    if len(values) != 2:
        raise ValueError("server PID identity is invalid")
    pid, start_time = (int(value) for value in values)
    if pid <= 1 or start_time <= 0:
        raise ValueError("server PID identity is invalid")
    return pid, start_time


def process_status(pid_file: Path | None) -> dict[str, Any]:
    if pid_file is None or not pid_file.is_file():
        return {}
    try:
        pid, expected_start_time = read_process_identity(pid_file)
        if process_start_time(pid) != expected_start_time:
            raise ValueError("server PID identity changed")
        values: dict[str, Any] = {"pid": pid}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"VmRSS", "VmSize", "Threads"}:
                values[key] = value.strip()
        return values
    except (OSError, ValueError):
        return {"error": "server PID is unavailable"}


def accelerator_status(device: str) -> dict[str, Any]:
    rocm_smi = Path("/opt/rocm/bin/rocm-smi")
    if not rocm_smi.is_file():
        return {"kind": "amd", "device": device, "ok": False, "error": "rocm-smi missing"}
    result = subprocess.run(
        [
            str(rocm_smi),
            "-d",
            device,
            "--showmeminfo",
            "vram",
            "--showpower",
            "--showtemp",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    payload: Any = None
    error = result.stderr.strip() or None
    if result.returncode == 0:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as parse_error:
            error = f"invalid rocm-smi JSON: {parse_error}"
    ok = result.returncode == 0 and isinstance(payload, dict) and bool(payload)
    return {
        "kind": "amd",
        "device": device,
        "ok": ok,
        "data": payload,
        "error": error,
    }


class ResourceMonitor:
    def __init__(
        self,
        server_url: str,
        output: Path,
        *,
        interval: float,
        pid_file: Path | None,
        accelerator_device: str,
    ) -> None:
        self.server_url = server_url
        self.output = output
        self.interval = interval
        self.pid_file = pid_file
        self.accelerator_device = accelerator_device
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(15.0, self.interval + 5.0))

        if self.thread.is_alive():
            raise RuntimeError("resource monitor did not stop")

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "timestamp": utc_now(),
            "system_memory_kib": proc_meminfo(),
            "server_process": process_status(self.pid_file),
            "accelerator": accelerator_status(self.accelerator_device),
        }
        try:
            started = time.monotonic()
            status_code = http_probe(self.server_url, "/health")
            sample["health_latency_ms"] = (time.monotonic() - started) * 1000.0
            sample["health_ok"] = True
            sample["health_status_code"] = status_code
        except (OSError, ValueError, urllib.error.URLError) as error:
            sample["health_ok"] = False
            sample["health_error"] = str(error)
        return sample

    def _run(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", buffering=1) as stream:
            while not self.stop_event.is_set():
                try:
                    stream.write(json.dumps(self._sample(), allow_nan=False) + "\n")
                except Exception as error:
                    stream.write(
                        json.dumps(
                            {
                                "timestamp": utc_now(),
                                "monitor_error": str(error),
                            }
                        )
                        + "\n"
                    )
                self.stop_event.wait(self.interval)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_config = report.get("profile_config")
    profile_config = profile_config if isinstance(profile_config, dict) else {}
    feature_set = profile_config.get("feature_set", "unknown")
    atomic_write_text(
        output_dir / "qualification.json",
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    lines = [
        f"# Release qualification: {report['profile']}",
        "",
        f"- Verdict: **{report['verdict'].upper()}**",
        f"- Family: `{report['family']}`",
        f"- Feature set: `{feature_set}`",
        f"- Commit: `{report['git_commit']}`",
        f"- Server: `{report['server_url']}`",
        f"- Started: {report['started_at']}",
        f"- Finished: {report.get('finished_at') or 'in progress'}",
        "",
        "| Stage | Description | Status | Seconds | Log |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for stage in report["stages"]:
        duration = stage.get("duration_seconds")
        rendered_duration = f"{duration:.1f}" if isinstance(duration, int | float) else "-"
        lines.append(
            f"| {stage['id']} | {stage['description']} | "
            f"{stage['status'].upper()} | {rendered_duration} | "
            f"`{stage.get('log', '-')}` |"
        )
    lines.extend(
        [
            "",
            "This run captures performance evidence without applying a baseline gate.",
            "",
        ]
    )
    atomic_write_text(output_dir / "qualification.md", "\n".join(lines))


def commands_for_profile(profile: QwenArProfile) -> dict[str, str]:
    runner = profile.qualification_runner
    if runner != "qwen36_amd":
        raise ValueError(f"unknown built-in qualification runner: {runner}")
    python = shlex.quote(sys.executable)
    return {
        stage: f"{python} -m harness.qualification.qwen36.qwen36_amd {stage}" for stage in STAGES
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence file must contain an object: {path.name}")
    return value


def _evidence_prompt_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("generation evidence does not name its prompt corpus")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _prompt_case_ids(prompts: Path) -> list[str]:
    records = [json.loads(line) for line in prompts.read_text().splitlines() if line]
    case_ids = [record.get("id") if isinstance(record, dict) else None for record in records]
    if (
        not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("prompt corpus case IDs are invalid")
    return case_ids


def _validate_generation_evidence(
    path: Path,
    *,
    prompts: Path,
    concurrency: int,
    maximum_tokens: int,
    warmups: int,
    repetitions: int,
    require_identical: bool = False,
    minimum_gold_accuracy: float | None = None,
) -> dict[str, Any]:
    report = _read_json(path)
    expected_fields = {
        "schema_version": 2,
        "concurrency": concurrency,
        "max_tokens": maximum_tokens,
        "warmups": warmups,
        "repeats": repetitions,
        "temperature": 0,
    }
    for field, expected in expected_fields.items():
        if report.get(field) != expected:
            raise ValueError(f"{path.name}: {field} differs from the profile")
    if _evidence_prompt_path(report.get("prompts")) != prompts.resolve():
        raise ValueError(f"{path.name}: prompt corpus differs from the profile")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("status") != "pass":
        raise ValueError(f"{path.name}: generation verdict did not pass")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path.name}: generation cases are missing")
    if summary.get("cases") != len(cases):
        raise ValueError(f"{path.name}: generation case count is inconsistent")
    case_ids = [case.get("id") if isinstance(case, dict) else None for case in cases]
    if case_ids != _prompt_case_ids(prompts):
        raise ValueError(f"{path.name}: generation cases differ from the prompt corpus")
    for case in cases:
        if not isinstance(case, dict) or case.get("expected_pass") is not True:
            raise ValueError(f"{path.name}: a prompt expectation did not pass")
        if require_identical and case.get("deterministic") is not True:
            raise ValueError(f"{path.name}: a deterministic case diverged")
        runs = case.get("runs")
        if not isinstance(runs, list) or len(runs) != repetitions:
            raise ValueError(f"{path.name}: measured repetition count is inconsistent")
        if any(
            not isinstance(run, dict) or run.get("request_count") != concurrency for run in runs
        ):
            raise ValueError(f"{path.name}: run evidence is invalid")
    if require_identical:
        if summary.get("require_identical") is not True:
            raise ValueError(f"{path.name}: determinism was not required")
        if summary.get("deterministic_cases") != len(cases):
            raise ValueError(f"{path.name}: deterministic case count is inconsistent")
    if minimum_gold_accuracy is not None:
        if summary.get("min_gold_accuracy") != minimum_gold_accuracy:
            raise ValueError(f"{path.name}: quality threshold differs from the profile")
        accuracy = summary.get("gold_accuracy")
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, int | float)
            or not math.isfinite(accuracy)
            or accuracy < minimum_gold_accuracy
        ):
            raise ValueError(f"{path.name}: quality score did not pass")
        if summary.get("gold_scored") != len(cases):
            raise ValueError(f"{path.name}: not every quality case was scored")
    return report


def _validate_server_props(props: dict[str, Any], profile: QwenArProfile) -> None:
    if props.get("daemon", {}).get("alive") is not True:
        raise ValueError("server props do not report a ready model")
    if Path(str(props.get("model_path", ""))).name != profile.artifact.path.name:
        raise ValueError("server props model path differs from the profile")
    settings = props.get("default_generation_settings", {})
    if settings.get("n_ctx") != profile.maximum_context:
        raise ValueError("server props maximum context differs from the profile")
    if props.get("prefix_cache", {}).get("capacity") != profile.prefix_cache_slots:
        raise ValueError("server props prefix cache differs from the profile")
    if props.get("speculative", {}).get("enabled") is not False:
        raise ValueError("server props do not report autoregressive decoding")


def _validate_performance_candidate(
    output_dir: Path, profile: QwenArProfile, candidate: dict[str, Any]
) -> None:
    from harness.qualification.qwen36.performance import _samples, _validate_usage_throughput

    expected: dict[str, list[float]] = {}
    for workload in profile.performance_workloads:
        report = _read_json(output_dir / f"r5-performance-c{workload.concurrency}.json")
        _validate_usage_throughput(report)
        expected[f"aggregate_tok_s_c{workload.concurrency}"] = _samples(report, "tok_s")
        expected[f"batch_latency_s_c{workload.concurrency}"] = _samples(report, "elapsed_s")

    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(expected):
        raise ValueError("R5 candidate metric set differs from the retained reports")
    for name, samples in expected.items():
        metric = metrics[name]
        if not isinstance(metric, dict) or metric.get("samples") != samples:
            raise ValueError(f"R5 candidate {name} samples differ from the retained report")


def _validate_drift_evidence(output_dir: Path, profile: QwenArProfile) -> None:
    drift = _read_json(output_dir / "r6-drift.json")
    monitor_samples = [
        json.loads(line)
        for line in (output_dir / "resource-monitor.jsonl").read_text().splitlines()
        if line
    ]
    if not monitor_samples or any(not isinstance(item, dict) for item in monitor_samples):
        raise ValueError("resource monitor evidence is invalid")
    expected = evaluate_drift(
        monitor_samples,
        max_rss_growth_mib=profile.maximum_rss_growth_mib,
        max_health_latency_growth_fraction=(profile.maximum_health_latency_growth_fraction),
        expected_accelerator_device=profile.visible_device,
        minimum_samples=profile.minimum_monitor_samples,
        steady_window_samples=profile.steady_window_samples,
    )
    if drift != expected:
        raise ValueError("R6 drift verdict differs from the retained monitor log")
    if drift.get("status") != "pass":
        raise ValueError("R6 drift verdict did not pass")


def validate_evidence_bundle(output_dir: Path, profile: QwenArProfile) -> dict[str, Any]:
    performance_concurrencies = [workload.concurrency for workload in profile.performance_workloads]
    expected = {
        "profile.snapshot.json",
        "r0-identity.json",
        "environment.normalized.json",
        "environment.raw.json",
        "server-props.json",
        "r1-ar-correctness.json",
        "r2-generation-quality.json",
        "r3-concurrent-determinism.json",
        "r4-saturation-c1.json",
        "r4-saturation-c2.json",
        f"r4-saturation-c{profile.maximum_concurrency}.json",
        "r6-drift.json",
        "resource-monitor.jsonl",
        "r5-performance-candidate.json",
        *(f"r5-performance-c{concurrency}.json" for concurrency in performance_concurrencies),
    }
    missing = sorted(name for name in expected if not output_dir.joinpath(name).is_file())
    failures = [f"missing evidence: {name}" for name in missing]
    if not missing:
        try:
            snapshot = _read_json(output_dir / "profile.snapshot.json")
            snapshot_artifact = snapshot.get("artifact", {})
            if snapshot_artifact.get("sha256") != profile.artifact.sha256:
                raise ValueError("profile snapshot target SHA-256 mismatch")
            if snapshot.get("recipe_id") != profile.recipe_id:
                raise ValueError("profile snapshot recipe mismatch")
            if snapshot.get("feature_set") != profile.feature_set:
                raise ValueError("profile snapshot feature set mismatch")
            if snapshot.get("server_arguments") != list(profile.server_arguments):
                raise ValueError("profile snapshot server arguments mismatch")
            if snapshot.get("visible_device") != profile.visible_device:
                raise ValueError("profile snapshot selected device mismatch")
            environment = _read_json(output_dir / "environment.normalized.json")
            raw_environment = _read_json(output_dir / "environment.raw.json")
            server_props = _read_json(output_dir / "server-props.json")
            if not environment or not raw_environment or not server_props:
                raise ValueError("environment and server evidence must be non-empty")
            _validate_server_props(server_props, profile)

            identity = _read_json(output_dir / "r0-identity.json")
            if identity.get("profile") != profile.name:
                raise ValueError("R0 identity profile mismatch")
            if identity.get("target_sha256") != profile.artifact.sha256:
                raise ValueError("R0 target SHA-256 mismatch")
            if identity.get("recipe") != profile.recipe_id:
                raise ValueError("R0 recipe mismatch")
            if identity.get("feature_set") != profile.feature_set:
                raise ValueError("R0 feature set mismatch")
            if identity.get("server_arguments") != list(profile.server_arguments):
                raise ValueError("R0 server arguments mismatch")
            if identity.get("max_context") != profile.maximum_context:
                raise ValueError("R0 maximum context mismatch")
            if identity.get("max_concurrency") != profile.maximum_concurrency:
                raise ValueError("R0 maximum concurrency mismatch")
            if identity.get("prefix_cache_slots") != profile.prefix_cache_slots:
                raise ValueError("R0 prefix cache slots mismatch")
            if identity.get("hip_visible_devices") != profile.visible_device:
                raise ValueError("R0 selected device mismatch")
            if identity.get("accelerator_arch") != profile.accelerator.architecture:
                raise ValueError("R0 accelerator architecture mismatch")
            subject = identity.get("qualification_subject")
            if (
                not isinstance(subject, dict)
                or subject.get("kind") != "source-build"
                or not isinstance(subject.get("server_binary_sha256"), str)
            ):
                raise ValueError("R0 source-build provenance is incomplete")

            c1_smoke = next(item for item in profile.smoke_workloads if item.concurrency == 1)
            _validate_generation_evidence(
                output_dir / "r1-ar-correctness.json",
                prompts=ROOT / profile.smoke_prompts,
                concurrency=1,
                maximum_tokens=c1_smoke.maximum_tokens,
                warmups=c1_smoke.warmups,
                repetitions=c1_smoke.repetitions,
                require_identical=c1_smoke.require_identical,
            )
            _validate_generation_evidence(
                output_dir / "r2-generation-quality.json",
                prompts=ROOT / profile.quality_prompts,
                concurrency=1,
                maximum_tokens=profile.quality_maximum_tokens,
                warmups=0,
                repetitions=1,
                minimum_gold_accuracy=profile.minimum_gold_accuracy,
            )
            cmax_smoke = next(
                item
                for item in profile.smoke_workloads
                if item.concurrency == profile.maximum_concurrency
            )
            _validate_generation_evidence(
                output_dir / "r3-concurrent-determinism.json",
                prompts=ROOT / profile.smoke_prompts,
                concurrency=profile.maximum_concurrency,
                maximum_tokens=cmax_smoke.maximum_tokens,
                warmups=cmax_smoke.warmups,
                repetitions=cmax_smoke.repetitions,
                require_identical=cmax_smoke.require_identical,
            )
            saturation = profile.performance_workloads[0]
            for concurrency in (1, 2, profile.maximum_concurrency):
                _validate_generation_evidence(
                    output_dir / f"r4-saturation-c{concurrency}.json",
                    prompts=ROOT / profile.performance_prompts,
                    concurrency=concurrency,
                    maximum_tokens=saturation.maximum_tokens,
                    warmups=1,
                    repetitions=2,
                )
            for workload in profile.performance_workloads:
                _validate_generation_evidence(
                    output_dir / f"r5-performance-c{workload.concurrency}.json",
                    prompts=ROOT / profile.performance_prompts,
                    concurrency=workload.concurrency,
                    maximum_tokens=workload.maximum_tokens,
                    warmups=workload.warmups,
                    repetitions=workload.repetitions,
                )

            candidate = _read_json(output_dir / "r5-performance-candidate.json")
            if candidate.get("schema_version") != 1 or candidate.get("profile") != profile.name:
                raise ValueError("R5 candidate profile mismatch")
            if candidate.get("qualification_subject") != subject:
                raise ValueError("R5 source-build provenance differs from R0")
            comparison_identity = candidate.get("comparison_identity", {})
            profile_identity = comparison_identity.get("profile", {})
            if profile_identity.get("recipe") != profile.recipe_id:
                raise ValueError("R5 recipe mismatch")
            if profile_identity.get("feature_set") != profile.feature_set:
                raise ValueError("R5 feature set mismatch")
            run_environment = comparison_identity.get("run", {}).get("environment", {})
            if run_environment.get("server_arguments") != list(profile.server_arguments):
                raise ValueError("R5 server arguments mismatch")
            if run_environment.get("maximum_context") != profile.maximum_context:
                raise ValueError("R5 maximum context mismatch")
            if run_environment.get("maximum_concurrency") != profile.maximum_concurrency:
                raise ValueError("R5 maximum concurrency mismatch")
            if run_environment.get("prefix_cache_slots") != profile.prefix_cache_slots:
                raise ValueError("R5 prefix cache slots mismatch")
            model_identity = comparison_identity.get("model", {})
            if model_identity.get("target_sha256") != profile.artifact.sha256:
                raise ValueError("R5 target SHA-256 mismatch")
            accelerator_identity = comparison_identity.get("hardware", {}).get("accelerator", {})
            if accelerator_identity.get("visible_device") != profile.visible_device:
                raise ValueError("R5 selected device mismatch")
            if accelerator_identity.get("architecture") != profile.accelerator.architecture:
                raise ValueError("R5 accelerator architecture mismatch")

            _validate_performance_candidate(output_dir, profile, candidate)

            _validate_drift_evidence(output_dir, profile)
        except (
            AttributeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            failures.append(str(error))
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"qualification.json", "qualification.md"}:
            files.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    return {"status": "fail" if failures else "pass", "failures": failures, "files": files}


def _process_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            fields = stat.read_text().split()
            if len(fields) > 2 and fields[2] == "Z":
                return False
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(pid: int, *, grace_seconds: float = 5.0) -> None:
    if pid <= 1:
        raise ValueError(f"refusing to terminate invalid process id {pid}")
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        return
    target = -process_group if process_group == pid else pid
    try:
        os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(target, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return
        time.sleep(0.1)
    raise RuntimeError(f"process {pid} survived SIGKILL")


def terminate_server(pid_file: Path | None) -> None:
    if pid_file is None or not pid_file.is_file():
        return
    pid, expected_start_time = read_process_identity(pid_file)
    try:
        actual_start_time = process_start_time(pid)
    except FileNotFoundError:
        pid_file.unlink(missing_ok=True)
        return
    if actual_start_time != expected_start_time:
        raise RuntimeError(f"refusing to terminate reused process id {pid}")
    terminate_process_group(pid)
    pid_file.unlink(missing_ok=True)


def run_stage(
    stage: str,
    command: str,
    output_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    log = output_dir / f"{stage.lower()}.log"
    started_at = utc_now()
    started = time.monotonic()
    timeout_seconds = STAGE_TIMEOUTS_SECONDS[stage]
    timed_out = False
    with log.open("w") as stream:
        stream.write(f"$ {command}\n")
        stream.flush()
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stream.write(f"stage exceeded {timeout_seconds} seconds\n")
            stream.flush()
            terminate_process_group(process.pid)
            process.wait(timeout=1)
            returncode = 124
        except BaseException:
            terminate_process_group(process.pid)
            process.wait(timeout=1)
            raise
    return {
        "id": stage,
        "description": STAGES[stage],
        "status": "pass" if returncode == 0 else "fail",
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "started_at": started_at,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "log": log.name,
    }


def qualify(args: argparse.Namespace) -> int:
    canonical_models = (ROOT / "harness/qualification/qwen36/profiles.yaml").resolve()
    if args.models.resolve() != canonical_models:
        raise ValueError(
            "the built-in qualification runner requires harness/qualification/qwen36/profiles.yaml"
        )
    if args.server_url:
        raise ValueError("the built-in qualification runner does not support --server-url")
    profile = load_qwen_ar_profile(canonical_models, args.profile)
    family = profile.family
    profile_config = profile.snapshot()
    commands = commands_for_profile(profile)
    server_url = f"http://127.0.0.1:{profile.server_port}"
    output_dir = (
        args.output_dir
        or Path("/tmp/lucebox-qualification")
        / f"{args.profile}-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    ).resolve()
    pid_file_value = os.environ.get("LUCE_SERVER_PID_FILE", "").strip()
    pid_file = Path(pid_file_value).resolve() if pid_file_value else output_dir / "server.pid"
    if pid_file.exists():
        raise ValueError(f"qualification PID file already exists: {pid_file}")
    output_dir.mkdir(parents=True, exist_ok=False)
    monitor_log = output_dir / "resource-monitor.jsonl"

    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": args.profile,
        "family": family,
        "profile_config": profile_config,
        "git_commit": git_commit(),
        "server_url": server_url,
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "started_at": utc_now(),
        "finished_at": None,
        "verdict": "running",
        "qualification_subject": {"kind": "source-build"},
        "monitor_log": monitor_log.name,
        "stages": [],
    }
    write_reports(report, output_dir)
    atomic_write_text(
        output_dir / "profile.snapshot.json",
        json.dumps(profile_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    environment = dict(os.environ)
    environment.update(profile.environment())
    environment.update(
        {
            "LUCE_PROFILE": args.profile,
            "LUCE_FAMILY": family,
            "LUCE_SERVER_URL": server_url,
            "LUCE_OUTPUT_DIR": str(output_dir),
            "LUCE_MONITOR_LOG": str(monitor_log),
            "LUCE_SERVER_PID_FILE": str(pid_file) if pid_file else "",
        }
    )

    monitor = ResourceMonitor(
        server_url,
        monitor_log,
        interval=args.monitor_interval or profile.monitor_interval_seconds,
        pid_file=pid_file,
        accelerator_device=profile.visible_device,
    )
    monitor_started = False
    infrastructure_failed = False
    orchestrator_failures: list[str] = []

    def interrupt(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"qualification interrupted by {signal.Signals(signum).name}")

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {item: signal.getsignal(item) for item in handled_signals}
    for item in handled_signals:
        signal.signal(item, interrupt)
    try:
        try:
            for stage, command in commands.items():
                if stage == "R6" and monitor_started:
                    monitor.stop()
                    monitor_started = False
                result = run_stage(stage, command, output_dir, environment)
                infrastructure_failed = False
                if stage == "R0" and result["status"] == "pass":
                    try:
                        props = http_json(server_url, "/props", timeout=5.0)
                        (output_dir / "server-props.json").write_text(
                            json.dumps(props, indent=2, allow_nan=False) + "\n"
                        )
                    except (OSError, ValueError, urllib.error.URLError) as error:
                        result["status"] = "fail"
                        result["endpoint_error"] = str(error)
                        infrastructure_failed = True
                    if result["status"] == "pass":
                        monitor.start()
                        monitor_started = True
                elif stage != "R0":
                    try:
                        http_probe(server_url, "/health")
                    except (OSError, ValueError, urllib.error.URLError) as error:
                        result["status"] = "fail"
                        result["endpoint_error"] = (
                            f"long-lived qualification server unavailable after stage: {error}"
                        )
                        infrastructure_failed = True
                report["stages"].append(result)
                report["verdict"] = (
                    "fail"
                    if any(item["status"] == "fail" for item in report["stages"])
                    else "running"
                )
                write_reports(report, output_dir)
                if (stage == "R0" and result["status"] == "fail") or infrastructure_failed:
                    break
        except BaseException as error:
            orchestrator_failures.append(f"{type(error).__name__}: {error}")
    finally:
        for item, previous_handler in previous_handlers.items():
            signal.signal(item, previous_handler)
        if monitor_started:
            try:
                monitor.stop()
            except Exception as error:
                orchestrator_failures.append(f"monitor cleanup failed: {error}")
        try:
            terminate_server(pid_file)
        except Exception as error:
            orchestrator_failures.append(f"server cleanup failed: {error}")

    completed = {stage["id"] for stage in report["stages"]}
    for stage in STAGES:
        if stage not in completed:
            report["stages"].append(
                {
                    "id": stage,
                    "description": STAGES[stage],
                    "status": "not-run",
                    "returncode": None,
                    "log": "-",
                }
            )
    stages_passed = all(stage["status"] == "pass" for stage in report["stages"])
    try:
        evidence = validate_evidence_bundle(output_dir, profile)
    except (OSError, ValueError) as error:
        evidence = {"status": "fail", "failures": [str(error)], "files": []}
    if not stages_passed:
        evidence["status"] = "fail"
        evidence["failures"].insert(0, "one or more qualification stages failed")
    if orchestrator_failures:
        evidence["status"] = "fail"
        evidence["failures"] = [*orchestrator_failures, *evidence["failures"]]
        report["orchestrator_failures"] = orchestrator_failures
    passed = stages_passed and evidence["status"] == "pass"
    atomic_write_text(
        output_dir / "evidence-manifest.json",
        json.dumps(evidence, indent=2, allow_nan=False) + "\n",
    )
    report["stages"].append(
        {
            "id": "R7",
            "description": "combined JSON/Markdown report; promotion remains reviewed",
            "status": "pass" if passed else "fail",
            "returncode": 0 if passed else 1,
            "log": "qualification.json",
            "failures": evidence["failures"],
        }
    )
    report["verdict"] = "pass" if passed else "fail"
    report["finished_at"] = utc_now()
    write_reports(report, output_dir)
    print(f"qualification report: {output_dir / 'qualification.md'}")
    print(f"verdict: {report['verdict'].upper()}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen 3.6 R0-R7 qualification.")
    parser.add_argument("profile", nargs="?")
    parser.add_argument(
        "--models", type=Path, default=ROOT / "harness/qualification/qwen36/profiles.yaml"
    )
    parser.add_argument("--server-url", default=os.environ.get("LUCE_SERVER_URL", ""))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cleanup-pid-file", type=Path)
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=(
            float(os.environ["LUCE_MONITOR_INTERVAL"])
            if "LUCE_MONITOR_INTERVAL" in os.environ
            else None
        ),
    )
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        print("\n".join(list_profiles(args.models)))
        return 0
    if args.cleanup_pid_file:
        try:
            terminate_server(args.cleanup_pid_file)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        return 0
    if not args.profile:
        parser.error("profile is required unless a standalone action is used")
    if args.monitor_interval is not None and args.monitor_interval <= 0:
        parser.error("--monitor-interval must be positive")
    try:
        return qualify(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
