#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness.qualification.qwen36.environment import (
    observe_amd_environment,
    write_environment_evidence,
)
from harness.qualification.qwen36.profiles import QwenArProfile, load_qwen_ar_profile
from harness.qualification.qwen36.qualify import (
    ROOT,
    atomic_write_text,
    git_commit,
    process_identity,
    sha256,
    terminate_process_group,
)


def _interrupt(signum: int, _frame: object) -> None:
    raise InterruptedError(f"Qwen runner interrupted by {signal.Signals(signum).name}")


def _output_dir() -> Path:
    value = os.environ.get("LUCE_OUTPUT_DIR")
    if not value:
        raise ValueError("LUCE_OUTPUT_DIR is required")
    return Path(value)


def _profile(name: str | None = None) -> QwenArProfile:
    name = name or os.environ.get("LUCE_PROFILE", "")
    return load_qwen_ar_profile(ROOT / "harness/qualification/qwen36/profiles.yaml", name)


def _artifact(profile: QwenArProfile) -> Path:
    root = Path(os.environ.get("MODELS", "/opt/models"))
    target = profile.artifact_path(root)
    if not target.is_file():
        raise ValueError(f"required target artifact is missing: {target}")
    return target


def _run(command: list[str], *, output: Path | None = None) -> None:
    if output:
        with output.open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def _capture(command: list[str], timeout_seconds: float = 30.0) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"qualification command timed out: {command[0]}") from error
    value = result.stdout.strip()
    if not value:
        raise ValueError(f"qualification command produced no output: {command[0]}")
    return value


def _wait_ready(base_url: str, process: subprocess.Popen[str], timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"qualification server exited during startup with code {returncode}")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=2):
                return
        except InterruptedError:
            raise
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("qualification server did not become ready")


def _require_port_available(host: str, port: int) -> None:
    with socket.socket() as probe:
        probe.settimeout(0.25)
        occupied = probe.connect_ex((host, port)) == 0
    if occupied:
        raise RuntimeError(f"qualification server port {port} is already in use on {host}")


def _launch_server(
    command: list[str],
    *,
    server_log_path: Path,
    pid_file: Path,
    base_url: str,
) -> None:
    process: subprocess.Popen[str] | None = None
    try:
        with server_log_path.open("w") as server_log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        atomic_write_text(pid_file, process_identity(process.pid) + "\n")
        _wait_ready(base_url, process)
    except BaseException:
        if process is not None:
            terminate_process_group(process.pid)
            pid_file.unlink(missing_ok=True)
        raise


def stage_r0() -> None:
    profile = _profile()
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise ValueError("source qualification requires a clean tracked worktree")
    target = _artifact(profile)
    actual_sha = sha256(target)
    if actual_sha != profile.artifact.sha256:
        raise ValueError("target artifact SHA-256 does not match the profile")

    rocminfo = _capture(["/opt/rocm/bin/rocminfo"])
    expected_arch = profile.accelerator.architecture
    if expected_arch not in rocminfo:
        raise ValueError(f"visible accelerator does not include {expected_arch}")

    output_dir = _output_dir()
    build_dir = output_dir / "build"
    configure = [
        "cmake",
        "-S",
        "server",
        "-B",
        str(build_dir),
        "-DDFLASH27B_GPU_BACKEND=hip",
        f"-DDFLASH27B_HIP_ARCHITECTURES={expected_arch}",
        "-DDFLASH27B_SERVER=ON",
        "-DDFLASH27B_TESTS=ON",
        "-DGGML_HIP_GRAPHS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_HIP_FLAGS=-DDFLASH_WAVE_SIZE=32",
    ]
    _run(configure, output=output_dir / "r0-configure.log")
    _run(
        ["cmake", "--build", str(build_dir), "--target", "dflash_server", "--parallel", "8"],
        output=output_dir / "r0-build.log",
    )
    server_bin = build_dir / "dflash_server"
    if not server_bin.is_file():
        raise ValueError("release server binary was not produced")

    normalized, raw = observe_amd_environment(profile, configure)
    write_environment_evidence(output_dir, normalized, raw)

    port = profile.server_port
    command = _server_command(profile, server_bin, target, port)
    _require_port_available("127.0.0.1", port)
    pid_file = Path(os.environ["LUCE_SERVER_PID_FILE"])
    _launch_server(
        command,
        server_log_path=output_dir / "server.log",
        pid_file=pid_file,
        base_url=os.environ["LUCE_SERVER_URL"],
    )

    identity = {
        "schema_version": 2,
        "profile": os.environ["LUCE_PROFILE"],
        "recipe": profile.recipe_id,
        "git_commit": git_commit(),
        "qualification_subject": {
            "kind": "source-build",
            "server_binary_sha256": sha256(server_bin),
        },
        "target": str(target),
        "target_sha256": actual_sha,
        "decode_mode": profile.decode_mode,
        "feature_set": profile.feature_set,
        "server_arguments": list(profile.server_arguments),
        "modality": profile.modality,
        "paged_attention": profile.paged_attention,
        "max_concurrency": profile.maximum_concurrency,
        "max_context": profile.maximum_context,
        "prefix_cache_slots": profile.prefix_cache_slots,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "accelerator_arch": expected_arch,
        "configure_command": configure,
        "server_command": command,
        "environment": normalized,
    }
    atomic_write_text(output_dir / "r0-identity.json", json.dumps(identity, indent=2) + "\n")


def _generation(
    *,
    name: str,
    prompts: Path,
    output: Path,
    max_tokens: int,
    warmups: int,
    repeats: int,
    concurrency: int,
    temperature: float = 0.0,
    min_gold_accuracy: float | None = None,
    require_identical: bool = False,
    server_url: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "harness/benchmarks/generation_benchmark.py"),
        "run",
        "--name",
        name,
        "--url",
        (server_url or os.environ["LUCE_SERVER_URL"]).rstrip("/") + "/v1",
        "--model",
        "luce-dflash",
        "--prompts",
        str(prompts),
        "--json-out",
        str(output),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--timeout",
        "600",
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--concurrency",
        str(concurrency),
    ]
    if min_gold_accuracy is not None:
        command += ["--min-gold-accuracy", str(min_gold_accuracy)]
    if require_identical:
        command.append("--require-identical")
    _run(command)


def stage_r1() -> None:
    profile = _profile()
    workload = next(item for item in profile.smoke_workloads if item.concurrency == 1)
    _generation(
        name=workload.name,
        prompts=ROOT / profile.smoke_prompts,
        output=_output_dir() / "r1-ar-correctness.json",
        max_tokens=workload.maximum_tokens,
        warmups=workload.warmups,
        repeats=workload.repetitions,
        concurrency=workload.concurrency,
        temperature=workload.temperature,
        require_identical=workload.require_identical,
    )


def stage_r2() -> None:
    profile = _profile()
    _generation(
        name="ar-generation-quality",
        prompts=ROOT / profile.quality_prompts,
        output=_output_dir() / "r2-generation-quality.json",
        max_tokens=profile.quality_maximum_tokens,
        warmups=0,
        repeats=1,
        concurrency=1,
        min_gold_accuracy=profile.minimum_gold_accuracy,
    )


def stage_r3() -> None:
    profile = _profile()
    workload = next(
        item for item in profile.smoke_workloads if item.concurrency == profile.maximum_concurrency
    )
    _generation(
        name=workload.name,
        prompts=ROOT / profile.smoke_prompts,
        output=_output_dir() / "r3-concurrent-determinism.json",
        max_tokens=workload.maximum_tokens,
        warmups=workload.warmups,
        repeats=workload.repetitions,
        concurrency=workload.concurrency,
        temperature=workload.temperature,
        require_identical=workload.require_identical,
    )


def stage_r4() -> None:
    profile = _profile()
    workload = profile.performance_workloads[0]
    for concurrency in (1, 2, profile.maximum_concurrency):
        _generation(
            name=f"ar-saturation-c{concurrency}",
            prompts=ROOT / profile.performance_prompts,
            output=_output_dir() / f"r4-saturation-c{concurrency}.json",
            max_tokens=workload.maximum_tokens,
            warmups=1,
            repeats=2,
            concurrency=concurrency,
            temperature=workload.temperature,
        )


def stage_r5() -> None:
    from harness.qualification.qwen36.performance import from_generation_reports

    profile = _profile()
    output_dir = _output_dir()
    reports = []
    for workload in profile.performance_workloads:
        concurrency = workload.concurrency
        path = output_dir / f"r5-performance-c{concurrency}.json"
        _generation(
            name=f"ar-performance-c{concurrency}",
            prompts=ROOT / profile.performance_prompts,
            output=path,
            max_tokens=workload.maximum_tokens,
            warmups=workload.warmups,
            repeats=workload.repetitions,
            concurrency=int(concurrency),
            temperature=workload.temperature,
        )
        reports.append(json.loads(path.read_text()))

    identity = json.loads((output_dir / "r0-identity.json").read_text())
    target = _artifact(profile)
    candidate = from_generation_reports(
        reports,
        profile=profile,
        target=target,
        environment=identity["environment"],
        qualification_subject=identity["qualification_subject"],
    )
    candidate_path = output_dir / "r5-performance-candidate.json"
    atomic_write_text(candidate_path, json.dumps(candidate, indent=2) + "\n")


def stage_r6() -> None:
    profile = _profile()
    _run(
        [
            sys.executable,
            "-m",
            "harness.qualification.qwen36.drift",
            os.environ["LUCE_MONITOR_LOG"],
            "--max-rss-growth-mib",
            str(profile.maximum_rss_growth_mib),
            "--max-health-latency-growth-fraction",
            str(profile.maximum_health_latency_growth_fraction),
            "--accelerator-device",
            profile.visible_device,
            "--minimum-samples",
            str(profile.minimum_monitor_samples),
            "--steady-window-samples",
            str(profile.steady_window_samples),
            "--output",
            str(_output_dir() / "r6-drift.json"),
        ]
    )


def _server_command(
    profile: QwenArProfile, server_binary: Path, target: Path, port: int
) -> list[str]:
    command = [
        str(server_binary),
        str(target),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-ctx",
        str(profile.maximum_context),
        "--max-concurrency",
        str(profile.maximum_concurrency),
        "--prefix-cache-slots",
        str(profile.prefix_cache_slots),
    ]
    if profile.paged_attention:
        command.append("--paged-attention")
    command.extend(profile.server_arguments)
    return command


STAGE_FUNCTIONS = {
    "R0": stage_r0,
    "R1": stage_r1,
    "R2": stage_r2,
    "R3": stage_r3,
    "R4": stage_r4,
    "R5": stage_r5,
    "R6": stage_r6,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run built-in AMD Qwen qualification stages.")
    parser.add_argument("stage", choices=STAGE_FUNCTIONS)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    args = parser.parse_args()
    STAGE_FUNCTIONS[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
