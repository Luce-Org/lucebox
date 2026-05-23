"""Local profiling store and snapshot assembly.

The profile store is append-only: every step result is keyed by a hash of the
variables that affect that step, and multiple runs for the same hash are kept
for variance/history. Snapshot export selects the newest compatible result for
the current hash of each registered step and records missing/stale/failed
states instead of hiding them.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from lucebox.types import Config

PROFILE_SCHEMA = 1
RESULT_SCHEMA = 1
SNAPSHOT_SCHEMA = 2
LOG_TAIL_BYTES = 4096

StepStatus = Literal[
    "fresh",
    "stale",
    "missing",
    "passed",
    "failed",
    "skipped_unavailable",
]


class ProfileInfoProvider(Protocol):
    def git_info(self) -> dict[str, Any]: ...
    def live_host_info(self) -> dict[str, Any]: ...
    def docker_info(self) -> dict[str, Any]: ...
    def image_info(self, cfg: Config) -> dict[str, Any]: ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scripts_dir() -> Path:
    return _repo_root() / "dflash" / "scripts"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns() % 1_000_000_000:09d}"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _tail_text(path: Path, limit: int = LOG_TAIL_BYTES) -> tuple[str, bool]:
    try:
        data = path.read_bytes()
    except OSError:
        return "", False
    truncated = len(data) > limit
    if truncated:
        data = data[-limit:]
    return data.decode(errors="replace"), truncated


def _json_get(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return {}


def _server_base_urls(cfg: Config, base_url: str | None = None) -> list[str]:
    if base_url:
        return [base_url.rstrip("/")]
    return [
        f"http://127.0.0.1:{cfg.port}",
        f"http://host.docker.internal:{cfg.port}",
        f"http://172.17.0.1:{cfg.port}",
    ]


def _first_json(base_urls: list[str], path: str) -> tuple[dict[str, Any], str]:
    for base in base_urls:
        data = _json_get(base + path)
        if data:
            return data, base
    return {}, ""


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cmd_status(args: list[str]) -> int:
    """subprocess.run with check=False, returning 127 if the binary is missing.

    Keeps git/lspci/etc. probes from crashing the orchestrator container, which
    ships without every host inspection binary installed.
    """
    try:
        return subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    except OSError:
        return 127


def _redact_hardware_dump(value: str) -> str:
    redacted_prefixes = (
        "Serial Number",
        "GPU UUID",
        "GPU PDI",
        "GPU Fabric GUID",
        "Chassis Serial Number",
    )
    lines = []
    for line in value.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in redacted_prefixes):
            key, sep, _rest = line.partition(":")
            lines.append(f"{key}{sep} <redacted>")
        else:
            lines.append(line)
    return "\n".join(lines)


class LiveProfileInfoProvider:
    def git_info(self) -> dict[str, Any]:
        repo_head = _cmd(["git", "rev-parse", "HEAD"])
        unstaged_dirty = bool(repo_head) and _cmd_status(["git", "diff", "--quiet"]) == 1
        staged_dirty = bool(repo_head) and _cmd_status(["git", "diff", "--cached", "--quiet"]) == 1
        untracked = _cmd(["git", "ls-files", "--others", "--exclude-standard"])
        return {
            "repo_head": repo_head,
            "repo_head_subject": _cmd(["git", "log", "-1", "--format=%s"]),
            "repo_branch": _cmd(["git", "branch", "--show-current"]),
            "repo_dirty": bool(repo_head) and (unstaged_dirty or staged_dirty or bool(untracked)),
            "repo_dirty_staged": staged_dirty,
            "repo_dirty_unstaged": unstaged_dirty,
            "repo_untracked_count": len([line for line in untracked.splitlines() if line.strip()]),
        }

    def live_host_info(self) -> dict[str, Any]:
        os_release = ""
        try:
            os_release = Path("/etc/os-release").read_text(errors="replace")
        except OSError:
            pass
        pretty = ""
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip().strip('"')
                break
        nvidia = _cmd([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap,"
            "pci.bus_id,power.draw,power.limit,utilization.gpu,temperature.gpu,"
            "clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ])
        return {
            "hostname": _cmd(["hostname"]),
            "kernel": _cmd(["uname", "-a"]),
            "os_pretty_name": pretty,
            "cpu_model": _cmd([
                "sh",
                "-lc",
                "awk -F: '/model name|Hardware|Processor/ {print $2; exit}' /proc/cpuinfo",
            ]).strip(),
            "nproc": _cmd(["sh", "-lc", "nproc 2>/dev/null || true"]),
            "mem_total_kib": _cmd(["sh", "-lc", "awk '/MemTotal/ {print $2}' /proc/meminfo"]),
            "pci_display_devices": _cmd([
                "sh", "-lc", "lspci 2>/dev/null | grep -Ei 'vga|3d|display' || true",
            ]),
            "nvidia_smi_csv": nvidia,
            "nvidia_smi_full": _redact_hardware_dump(
                _cmd(["sh", "-lc", "nvidia-smi -q 2>/dev/null || true"])
            ),
            "rocm_smi_summary": _cmd([
                "sh", "-lc", "rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true",
            ]),
            "python": sys.version.replace("\n", " "),
        }

    def docker_info(self) -> dict[str, Any]:
        version = _cmd(["docker", "version", "--format", "{{.Client.Version}} {{.Server.Version}}"])
        client, _, server = version.partition(" ")
        runtimes = _cmd(["docker", "info", "--format", "{{json .Runtimes}}"])
        return {
            "docker_client_version": client,
            "docker_server_version": server,
            "docker_default_runtime": _cmd(["docker", "info", "--format", "{{.DefaultRuntime}}"]),
            "docker_has_nvidia_runtime": "nvidia" in runtimes,
        }

    def image_info(self, cfg: Config) -> dict[str, Any]:
        tag = f"{cfg.image}:{cfg.variant}"
        raw = _cmd(["docker", "image", "inspect", tag, "--format", "{{.Id}} {{.Created}}"])
        image_id, _, created = raw.partition(" ")
        return {"image": tag, "image_id": image_id, "image_created": created}


@dataclass(frozen=True, slots=True)
class ProfileContext:
    cfg: Config
    base_url: str
    profile_root: Path
    machine: dict[str, Any]
    git: dict[str, Any]
    docker: dict[str, Any]
    image: dict[str, Any]
    props: dict[str, Any]
    health: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StepDefinition:
    id: str
    version: int
    description: str
    timeout_s: int
    max_age_hours: float
    requires_live_server: bool
    argv: Callable[[ProfileContext, Path], list[str]] | None = None
    report_name: str | None = None
    trace_name: str | None = None
    csv_name: str | None = None
    collector: Callable[[ProfileContext, Path], dict[str, Any]] | None = None
    availability: Callable[[ProfileContext], tuple[bool, str]] | None = None
    tunable_keys: tuple[str, ...] = ()
    script_paths: tuple[str, ...] = ()
    row_section: str | None = None


@dataclass(frozen=True, slots=True)
class StepSelection:
    definition: StepDefinition
    fingerprint: dict[str, Any]
    hash: str
    newest: dict[str, Any] | None
    status: StepStatus
    reason: str


def profile_root(cfg: Config) -> Path:
    return cfg.models_dir / ".lucebox" / "profile"


class ProfileLock:
    def __init__(self, root: Path):
        self.root = root
        self.file = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = (self.root / "profile.lock").open("w")
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc):
        if self.file is not None:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()


def _runtime_tunables(cfg: Config) -> dict[str, Any]:
    d = cfg.dflash
    return {
        "budget": d.budget,
        "max_ctx": d.max_ctx,
        "lazy": d.lazy,
        "prefix_cache_slots": d.prefix_cache_slots,
        "prefill_cache_slots": d.prefill_cache_slots,
        "cache_type_k": d.cache_type_k,
        "cache_type_v": d.cache_type_v,
        "prefill_mode": d.prefill_mode,
        "prefill_keep_ratio": d.prefill_keep_ratio,
        "prefill_threshold": d.prefill_threshold,
        "prefill_drafter": d.prefill_drafter,
        "think_max": d.think_max,
    }


def _selected_tunables(cfg: Config, keys: tuple[str, ...]) -> dict[str, Any]:
    all_values = _runtime_tunables(cfg)
    return {key: all_values.get(key) for key in keys}


def _file_identity(path: str) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _model_metadata(props: Mapping[str, Any]) -> dict[str, Any]:
    model = props.get("model") if isinstance(props.get("model"), dict) else {}
    target = str(props.get("model_path") or "")
    draft = str(model.get("draft_path") if isinstance(model, dict) else "")
    return {
        "model_alias": props.get("model_alias"),
        "model_path": _file_identity(target),
        "draft_path": _file_identity(draft),
    }


def _props_fingerprint(props: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "server": props.get("server"),
        "build_info": props.get("build_info"),
        "runtime": props.get("runtime"),
        "default_generation_settings": props.get("default_generation_settings"),
        "speculative_mode": props.get("speculative_mode"),
        "speculative": props.get("speculative"),
        "pflash": props.get("pflash"),
    }


def _script_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    root = _repo_root()
    return {path: _sha256_file(root / path) for path in paths}


def _git_fingerprint(git: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_head": git.get("repo_head"),
    }


def _hardware_fingerprint(ctx: ProfileContext) -> dict[str, Any]:
    return {
        "host_gpu_vendor": ctx.cfg.host.gpu_vendor,
        "host_gpu_name": ctx.cfg.host.gpu_name,
        "host_gpu_sm": ctx.cfg.host.gpu_sm,
        "host_vram_gb": ctx.cfg.host.vram_gb,
        "host_driver": ctx.cfg.host.driver_version,
        "kernel": ctx.machine.get("kernel"),
        "os_pretty_name": ctx.machine.get("os_pretty_name"),
        "cpu_model": ctx.machine.get("cpu_model"),
        "nproc": ctx.machine.get("nproc"),
        "mem_total_kib": ctx.machine.get("mem_total_kib"),
        "pci_display_devices": ctx.machine.get("pci_display_devices"),
    }


def _fingerprint(step: StepDefinition, ctx: ProfileContext, result_dir: Path) -> dict[str, Any]:
    command = step.argv(ctx, result_dir) if step.argv else []
    return {
        "profile_schema": PROFILE_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "step_id": step.id,
        "step_version": step.version,
        "command": command,
        "repo": _git_fingerprint(ctx.git),
        "image": ctx.image,
        "docker": ctx.docker,
        "hardware": _hardware_fingerprint(ctx),
        "model": _model_metadata(ctx.props),
        "props": _props_fingerprint(ctx.props),
        "tunables": _selected_tunables(ctx.cfg, step.tunable_keys),
        "scripts": _script_hashes(step.script_paths),
    }


def _result_dir(root: Path, step_id: str, result_hash: str) -> Path:
    return root / "results" / step_id / result_hash


def _result_files(root: Path, step_id: str, result_hash: str) -> list[Path]:
    directory = _result_dir(root, step_id, result_hash)
    return sorted(directory.glob("*.json"))


def _load_newest_result(root: Path, step_id: str, result_hash: str) -> dict[str, Any] | None:
    files = _result_files(root, step_id, result_hash)
    for path in reversed(files):
        data = _read_json(path)
        if data:
            return data
    return None


def _status_for_result(
    step: StepDefinition, result: dict[str, Any] | None, available: bool, reason: str,
) -> tuple[StepStatus, str]:
    if not available:
        return "skipped_unavailable", reason
    if result is None:
        return "missing", "no matching result for current fingerprint"
    if result.get("status") in {"failed", "error"}:
        return "failed", str(result.get("error") or "step failed")
    completed_at = str(result.get("completed_at") or "")
    try:
        ts = time.strptime(completed_at, "%Y-%m-%dT%H:%M:%SZ")
        age_hours = (time.time() - time.mktime(ts)) / 3600
    except ValueError:
        age_hours = step.max_age_hours + 1
    if age_hours > step.max_age_hours:
        return "stale", f"older than {step.max_age_hours:g}h"
    return "fresh", "newest matching result"


def _command_step(
    step: StepDefinition,
    ctx: ProfileContext,
    result_hash: str,
    force_refresh: bool,
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    dest = _result_dir(ctx.profile_root, step.id, result_hash)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = _run_id()
    run_dir = dest / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    report_path = run_dir / (step.report_name or "report.json")
    trace_path = run_dir / step.trace_name if step.trace_name else None
    csv_path = run_dir / step.csv_name if step.csv_name else None
    assert step.argv is not None
    argv = step.argv(ctx, run_dir)
    started = _now()
    t0 = time.monotonic()
    error = ""
    try:
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            proc = subprocess.run(
                argv,
                cwd=_repo_root(),
                stdout=out,
                stderr=err,
                timeout=step.timeout_s,
                check=False,
            )
            rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        rc = 124
        error = f"timeout after {step.timeout_s}s: {e}"
    completed = _now()
    report = _read_json(report_path)
    stdout_tail, stdout_truncated = _tail_text(stdout_path)
    stderr_tail, stderr_truncated = _tail_text(stderr_path)
    status = "passed" if rc == 0 else "failed"
    payload = {
        "schema": RESULT_SCHEMA,
        "step_id": step.id,
        "step_version": step.version,
        "hash": result_hash,
        "fingerprint": fingerprint,
        "status": status,
        "exit_code": rc,
        "error": error,
        "started_at": started,
        "completed_at": completed,
        "duration_s": round(time.monotonic() - t0, 3),
        "argv": argv,
        "stdout": str(stdout_path.relative_to(dest)),
        "stderr": str(stderr_path.relative_to(dest)),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "report": report,
        "report_path": str(report_path.relative_to(dest)) if report_path.exists() else "",
        "report_sha256": _sha256_file(report_path),
        "trace_path": (
            str(trace_path.relative_to(dest)) if trace_path and trace_path.exists() else ""
        ),
        "trace_sha256": _sha256_file(trace_path) if trace_path else "",
        "csv_path": str(csv_path.relative_to(dest)) if csv_path and csv_path.exists() else "",
        "csv_sha256": _sha256_file(csv_path) if csv_path else "",
        "force_refresh": force_refresh,
    }
    _write_json(dest / f"{stamp}.json", payload)
    return payload


def _collector_step(
    step: StepDefinition,
    ctx: ProfileContext,
    result_hash: str,
    force_refresh: bool,
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    dest = _result_dir(ctx.profile_root, step.id, result_hash)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = _run_id()
    started = _now()
    t0 = time.monotonic()
    assert step.collector is not None
    try:
        report = step.collector(ctx, dest)
        status = "passed"
        error = ""
        rc = 0
    except Exception as e:
        report = {}
        status = "failed"
        error = str(e)
        rc = 1
    payload = {
        "schema": RESULT_SCHEMA,
        "step_id": step.id,
        "step_version": step.version,
        "hash": result_hash,
        "fingerprint": fingerprint,
        "status": status,
        "exit_code": rc,
        "error": error,
        "started_at": started,
        "completed_at": _now(),
        "duration_s": round(time.monotonic() - t0, 3),
        "argv": [],
        "stdout_tail": "",
        "stderr_tail": "",
        "report": report,
        "force_refresh": force_refresh,
    }
    _write_json(dest / f"{stamp}.json", payload)
    return payload


def _unavailable_step(
    step: StepDefinition,
    ctx: ProfileContext,
    result_hash: str,
    reason: str,
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    dest = _result_dir(ctx.profile_root, step.id, result_hash)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = _run_id()
    payload = {
        "schema": RESULT_SCHEMA,
        "step_id": step.id,
        "step_version": step.version,
        "hash": result_hash,
        "fingerprint": fingerprint,
        "status": "skipped_unavailable",
        "reason": reason,
        "started_at": _now(),
        "completed_at": _now(),
        "duration_s": 0,
        "argv": [],
        "report": {},
    }
    _write_json(dest / f"{stamp}.json", payload)
    return payload


def _health_collector(ctx: ProfileContext, _dest: Path) -> dict[str, Any]:
    return {
        "suite": "health.props",
        "health": ctx.health,
        "props": ctx.props,
        "ok": bool(ctx.health and ctx.props),
    }


def _autotune_collector(ctx: ProfileContext, _dest: Path) -> dict[str, Any]:
    return _read_json(ctx.cfg.models_dir / ".lucebox" / "bench-report.json")


def _has_existing_autotune(ctx: ProfileContext) -> tuple[bool, str]:
    path = ctx.cfg.models_dir / ".lucebox" / "bench-report.json"
    return (path.exists(), f"{path} is missing")


def _always_available(_ctx: ProfileContext) -> tuple[bool, str]:
    return True, ""


def _live_available(ctx: ProfileContext) -> tuple[bool, str]:
    if ctx.health.get("status") == "ok" and ctx.props:
        return True, ""
    return False, "live server did not return /health and /props"


def _capability_argv(
    area: str,
    report: str,
    trace: str,
    *,
    min_pass_rate: float = 1.0,
    max_tokens: int | None = None,
    timeout: int | None = None,
    think: bool | None = None,
) -> Callable[[ProfileContext, Path], list[str]]:
    def build(ctx: ProfileContext, dest: Path) -> list[str]:
        argv = [
            sys.executable, str(_scripts_dir() / "bench_http_capability.py"),
            "--url", ctx.base_url,
            "--area", area,
            "--min-pass-rate", str(min_pass_rate),
            "--json-out", str(dest / report),
            "--trace", str(dest / trace),
        ]
        if max_tokens is not None:
            argv += ["--max-tokens", str(max_tokens)]
        if timeout is not None:
            argv += ["--timeout", str(timeout)]
        if think is True:
            argv += ["--think"]
        elif think is False:
            argv += ["--no-think"]
        return argv
    return build


def _frontiers_argv(ctx: ProfileContext, dest: Path) -> list[str]:
    return [
        sys.executable, str(_scripts_dir() / "bench_http_frontiers.py"),
        "--url", ctx.base_url,
        "--frontiers", "2048,4096,8192,16384",
        "--gen-tokens", "64",
        "--repeat", "3",
        "--json-out", str(dest / "bench-http-frontiers.json"),
        "--csv-out", str(dest / "bench-http-frontiers.csv"),
    ]


def _agentic_tools_argv(ctx: ProfileContext, dest: Path) -> list[str]:
    return [
        sys.executable, str(_scripts_dir() / "bench_agentic_tools.py"),
        "--url", ctx.base_url,
        "--json-out", str(dest / "bench-agentic-tools.json"),
    ]


def _agentic_session_argv(ctx: ProfileContext, dest: Path) -> list[str]:
    return [
        sys.executable, str(_scripts_dir() / "bench_agentic_session.py"),
        "--url", ctx.base_url,
        "--turns", "4",
        "--sessions", "1",
        "--json-out", str(dest / "bench-agentic-session.json"),
    ]


def _pytest_argv(_ctx: ProfileContext, dest: Path) -> list[str]:
    return [
        "uv", "run", "--frozen", "--with", "pytest", "pytest",
        "dflash/scripts/test_lucebox_bench.py",
        "lucebox/tests",
        "-q",
    ]


def registry() -> list[StepDefinition]:
    live_tunables = (
        "budget", "max_ctx", "lazy", "prefix_cache_slots", "prefill_cache_slots",
        "cache_type_k", "cache_type_v", "prefill_mode", "prefill_keep_ratio",
        "prefill_threshold", "prefill_drafter", "think_max",
    )
    return [
        StepDefinition(
            id="health.props",
            version=1,
            description="Fetch /health and /props.",
            timeout_s=10,
            max_age_hours=24,
            requires_live_server=True,
            collector=_health_collector,
            availability=_live_available,
        ),
        StepDefinition(
            id="benchmark.http_frontiers",
            version=1,
            description="HTTP prompt frontier timing rows.",
            timeout_s=1800,
            max_age_hours=24,
            requires_live_server=True,
            argv=_frontiers_argv,
            report_name="bench-http-frontiers.json",
            csv_name="bench-http-frontiers.csv",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=("dflash/scripts/bench_http_frontiers.py",),
            row_section="benchmark.frontiers.row",
        ),
        StepDefinition(
            id="quality.capability_smoke",
            version=1,
            description="Short deterministic API capability smoke prompts.",
            timeout_s=300,
            max_age_hours=24,
            requires_live_server=True,
            argv=_capability_argv("smoke", "bench-capability.json", "bench-capability-trace.txt"),
            report_name="bench-capability.json",
            trace_name="bench-capability-trace.txt",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=("dflash/scripts/bench_http_capability.py",),
        ),
        StepDefinition(
            id="quality.ds4_eval",
            version=1,
            description="Full 92-case antirez/ds4 ds4-eval HTTP port.",
            timeout_s=86400,
            max_age_hours=168,
            requires_live_server=True,
            argv=_capability_argv(
                "ds4-eval",
                "bench-ds4-eval.json",
                "bench-ds4-eval-trace.txt",
                min_pass_rate=0.0,
                max_tokens=16000,
                timeout=1800,
                think=True,
            ),
            report_name="bench-ds4-eval.json",
            trace_name="bench-ds4-eval-trace.txt",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=(
                "dflash/scripts/bench_http_capability.py",
                "dflash/scripts/fixtures/ds4_eval_cases.json",
            ),
        ),
        StepDefinition(
            id="quality.capability_long",
            version=1,
            description="Long prompt recall capability check.",
            timeout_s=600,
            max_age_hours=24,
            requires_live_server=True,
            argv=_capability_argv(
                "long",
                "bench-capability-long.json",
                "bench-capability-long-trace.txt",
            ),
            report_name="bench-capability-long.json",
            trace_name="bench-capability-long-trace.txt",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=("dflash/scripts/bench_http_capability.py",),
        ),
        StepDefinition(
            id="quality.agentic_tools",
            version=1,
            description="Single-turn OpenAI tool-call reliability prompts.",
            timeout_s=600,
            max_age_hours=24,
            requires_live_server=True,
            argv=_agentic_tools_argv,
            report_name="bench-agentic-tools.json",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=("dflash/scripts/bench_agentic_tools.py",),
        ),
        StepDefinition(
            id="benchmark.agentic_session",
            version=2,
            description="Four-turn multi-tool agentic session replay.",
            timeout_s=1200,
            max_age_hours=24,
            requires_live_server=True,
            argv=_agentic_session_argv,
            report_name="bench-agentic-session.json",
            availability=_live_available,
            tunable_keys=live_tunables,
            script_paths=("dflash/scripts/bench_agentic_session.py",),
        ),
        StepDefinition(
            id="benchmark.autotune_latest",
            version=1,
            description="Latest optimizer report, if present.",
            timeout_s=10,
            max_age_hours=720,
            requires_live_server=False,
            collector=_autotune_collector,
            availability=_has_existing_autotune,
            tunable_keys=live_tunables,
        ),
        StepDefinition(
            id="test.python_unit",
            version=1,
            description="Python regression tests for the profile/autotune stack.",
            timeout_s=300,
            max_age_hours=24,
            requires_live_server=False,
            argv=_pytest_argv,
            report_name="pytest-report.json",
            availability=_always_available,
            script_paths=(
                "dflash/scripts/test_lucebox_bench.py",
                "lucebox/tests/test_profile.py",
            ),
        ),
    ]


def build_context(
    cfg: Config,
    base_url: str | None = None,
    provider: ProfileInfoProvider | None = None,
) -> ProfileContext:
    info = provider or LiveProfileInfoProvider()
    bases = _server_base_urls(cfg, base_url)
    props, props_base = _first_json(bases, "/props")
    health, health_base = _first_json(bases, "/health")
    live_base = health_base or props_base or (base_url.rstrip("/") if base_url else bases[0])
    root = profile_root(cfg)
    machine = info.live_host_info()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "machine.json", machine)
    return ProfileContext(
        cfg=cfg,
        base_url=live_base,
        profile_root=root,
        machine=machine,
        git=info.git_info(),
        docker=info.docker_info(),
        image=info.image_info(cfg),
        props=props,
        health=health,
    )


def select_step(
    step: StepDefinition,
    ctx: ProfileContext,
    force_refresh: bool = False,
) -> StepSelection:
    scratch = ctx.profile_root / ".fingerprint" / step.id
    fingerprint = _fingerprint(step, ctx, scratch)
    result_hash = _sha256_text(_stable_json(fingerprint))
    available = True
    availability_reason = ""
    if step.availability:
        available, availability_reason = step.availability(ctx)
    newest = _load_newest_result(ctx.profile_root, step.id, result_hash)
    status, reason = _status_for_result(step, newest, available, availability_reason)
    if force_refresh and available:
        status, reason = "stale", "force refresh requested"
    return StepSelection(step, fingerprint, result_hash, newest, status, reason)


def audit_steps(
    ctx: ProfileContext,
    step_filter: str | None = None,
    force_refresh: bool = False,
) -> list[StepSelection]:
    selected = []
    for step in registry():
        if step_filter and step.id != step_filter:
            continue
        selected.append(select_step(step, ctx, force_refresh=force_refresh))
    return selected


def run_profile(
    cfg: Config,
    *,
    base_url: str | None = None,
    step_filter: str | None = None,
    force_refresh: bool = False,
    dry_run: bool = False,
    provider: ProfileInfoProvider | None = None,
) -> list[dict[str, Any]]:
    ctx = build_context(cfg, base_url=base_url, provider=provider)
    with ProfileLock(ctx.profile_root):
        selections = audit_steps(ctx, step_filter=step_filter, force_refresh=force_refresh)
        results: list[dict[str, Any]] = []
        for selection in selections:
            step = selection.definition
            if dry_run:
                results.append(_selection_row(selection, ran=False))
                continue
            if selection.status == "skipped_unavailable":
                result = _unavailable_step(
                    step, ctx, selection.hash, selection.reason, selection.fingerprint
                )
            elif selection.status in {"fresh"}:
                result = selection.newest or {}
            elif step.collector:
                result = _collector_step(
                    step, ctx, selection.hash, force_refresh, selection.fingerprint
                )
            elif step.argv:
                result = _command_step(
                    step, ctx, selection.hash, force_refresh, selection.fingerprint
                )
            else:
                result = _unavailable_step(
                    step, ctx, selection.hash, "no runner configured", selection.fingerprint
                )
            results.append(
                _selection_row(selection, ran=selection.status != "fresh", result=result)
            )
        return results


def _selection_row(
    selection: StepSelection, *, ran: bool, result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = result or selection.newest or {}
    status = result.get("status") if ran and result else selection.status
    return {
        "step_id": selection.definition.id,
        "description": selection.definition.description,
        "hash": selection.hash,
        "status": status,
        "freshness_status": selection.status,
        "reason": selection.reason,
        "ran": ran,
        "duration_s": result.get("duration_s"),
        "result_completed_at": result.get("completed_at"),
        "result": result,
    }


def _snapshot_step_status(selection: StepSelection) -> dict[str, Any]:
    result = selection.newest or {}
    return {
        "step_id": selection.definition.id,
        "description": selection.definition.description,
        "hash": selection.hash,
        "status": selection.status,
        "reason": selection.reason,
        "result_status": result.get("status"),
        "completed_at": result.get("completed_at"),
        "duration_s": result.get("duration_s"),
        "report_sha256": result.get("report_sha256"),
        "trace_sha256": result.get("trace_sha256"),
        "csv_sha256": result.get("csv_sha256"),
        "stdout_tail": result.get("stdout_tail"),
        "stderr_tail": result.get("stderr_tail"),
    }


def _stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ",".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("\n", "\\n")


def _emit(lines: list[str], section: str, values: Mapping[str, Any]) -> None:
    lines.append(f"[{section}]")
    for key, value in values.items():
        lines.append(f"{key}={_stringify(value)}")
    lines.append("")


def _suite_report(selection_by_id: Mapping[str, StepSelection], step_id: str) -> dict[str, Any]:
    selection = selection_by_id.get(step_id)
    if not selection or not selection.newest:
        return {}
    report = selection.newest.get("report")
    return report if isinstance(report, dict) else {}


def _suite_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": report.get("suite"),
        "area": report.get("area"),
        "source": report.get("source"),
        "max_tokens": report.get("max_tokens"),
        "passed": report.get("passed"),
        "total": report.get("total"),
        "pass_rate": report.get("pass_rate"),
        "graded_passed": report.get("graded_passed"),
        "graded_pass_rate": report.get("graded_pass_rate"),
        "strict_passed": report.get("strict_passed"),
        "strict_pass_rate": report.get("strict_pass_rate"),
        "format_passed": report.get("format_passed"),
        "format_pass_rate": report.get("format_pass_rate"),
        "semantic_hints": report.get("semantic_hints"),
        "semantic_hint_rate": report.get("semantic_hint_rate"),
        "semantic_passed": report.get("semantic_passed"),
        "semantic_pass_rate": report.get("semantic_pass_rate"),
    }


def _row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id") or row.get("name"),
        "name": row.get("name"),
        "domain": row.get("domain"),
        "title": row.get("title"),
        "ds4_index": row.get("ds4_index"),
        "status": row.get("status"),
        "ok": row.get("ok"),
        "graded_pass": row.get("graded_pass"),
        "strict_pass": row.get("strict_pass"),
        "format_pass": row.get("format_pass"),
        "semantic_hint": row.get("semantic_hint"),
        "semantic_pass": row.get("semantic_pass"),
        "given": row.get("given"),
        "correct": row.get("correct"),
        "expected": row.get("expected"),
        "tool_names": row.get("tool_names"),
        "wall_s": row.get("wall_s"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
    }


def _frontier_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frontier_target_tokens": row.get("frontier_target_tokens"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "finish_reason": row.get("finish_reason"),
        "repeat": row.get("repeat", 1),
        "ttft_s": row.get("ttft_s"),
        "wall_s": row.get("wall_s"),
        "decode_tps": row.get("decode_tps"),
        "decode_tps_mean": row.get("decode_tps_mean", row.get("decode_tps")),
        "decode_tps_min": row.get("decode_tps_min", row.get("decode_tps")),
        "decode_tps_max": row.get("decode_tps_max", row.get("decode_tps")),
    }


def _agentic_session_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": report.get("suite"),
        "source": report.get("source"),
        "fixture_id": report.get("fixture_id"),
        "fixture_hash": report.get("fixture_hash"),
        "sessions": report.get("sessions"),
        "turns": report.get("turns"),
        "passed": report.get("passed"),
        "total": report.get("total"),
        "pass_rate": report.get("pass_rate"),
        "ok": report.get("ok"),
        "max_prompt_tokens": report.get("max_prompt_tokens"),
        "final_wall_growth_vs_turn1": report.get("final_wall_growth_vs_turn1"),
    }


def _canonical_autotune_profile(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return {
        "quick": "level1",
        "context": "level2",
        "full": "level2",
        "stress": "level3",
    }.get(value, value)


def build_snapshot(
    cfg: Config,
    *,
    base_url: str | None = None,
    provider: ProfileInfoProvider | None = None,
) -> tuple[str, dict[str, Any]]:
    ctx = build_context(cfg, base_url=base_url, provider=provider)
    selections = audit_steps(ctx)
    selection_by_id = {selection.definition.id: selection for selection in selections}
    lines: list[str] = [
        "# lucebox profile snapshot",
        f"# schema={SNAPSHOT_SCHEMA}",
        "# Snapshot rows are selected from newest matching profile results.",
        "",
    ]
    sections: list[dict[str, Any]] = []

    def emit(section: str, values: Mapping[str, Any]) -> None:
        materialized = dict(values)
        _emit(lines, section, materialized)
        sections.append({"section": section, "values": materialized})

    profile_status = "fresh"
    bad = [s for s in selections if s.status in {"missing", "stale", "failed"}]
    skipped = [s for s in selections if s.status == "skipped_unavailable"]
    if bad:
        profile_status = "incomplete"
    emit("snapshot", {
        "schema": SNAPSHOT_SCHEMA,
        "captured_at": _now(),
        "profile_status": profile_status,
        "missing_or_stale_or_failed": len(bad),
        "skipped_unavailable": len(skipped),
        "server_base_url": ctx.base_url,
    } | ctx.git)
    emit("machine", ctx.machine)
    emit("docker", ctx.docker)
    emit("image", ctx.image)
    emit("runtime.config", {
        "port": cfg.port,
        "container_name": cfg.container_name,
        "models_dir": cfg.models_dir,
        "autotune_source": cfg.autotune.source,
        "autotune_timestamp": cfg.autotune.timestamp,
    })

    for idx, selection in enumerate(selections, start=1):
        emit(f"profile.step.{idx}", _snapshot_step_status(selection))

    frontiers = _suite_report(selection_by_id, "benchmark.http_frontiers")
    if frontiers:
        emit("benchmark.frontiers", {
            "suite": frontiers.get("suite"),
            "source": frontiers.get("source"),
            "timestamp": frontiers.get("timestamp"),
            "rows": len(frontiers.get("rows") or []),
        })
        for idx, row in enumerate(frontiers.get("rows") or [], start=1):
            emit(f"benchmark.frontiers.row.{idx}", _frontier_summary(row))

    for step_id, section in (
        ("quality.capability_smoke", "quality.capability_smoke"),
        ("quality.ds4_eval", "quality.ds4_eval"),
        ("quality.capability_long", "quality.capability_long"),
        ("quality.agentic_tools", "quality.agentic_tools"),
    ):
        report = _suite_report(selection_by_id, step_id)
        if not report:
            continue
        emit(section, _suite_summary(report))
        for idx, row in enumerate(report.get("rows") or [], start=1):
            emit(f"{section}.row.{idx}", _row_summary(row))

    agentic = _suite_report(selection_by_id, "benchmark.agentic_session")
    if agentic:
        emit("benchmark.agentic_session", _agentic_session_summary(agentic))
        for idx, row in enumerate(agentic.get("summary") or [], start=1):
            emit(f"benchmark.agentic_session.turn.{idx}", row)
        for idx, row in enumerate(agentic.get("rows") or [], start=1):
            emit(f"benchmark.agentic_session.row.{idx}", row)

    autotune = _suite_report(selection_by_id, "benchmark.autotune_latest")
    if autotune:
        emit("benchmark.autotune_latest", {
            "profile": _canonical_autotune_profile(autotune.get("profile")),
            "effective_profile": _canonical_autotune_profile(autotune.get("effective_profile")),
            "target": autotune.get("target"),
            "winner": autotune.get("winner"),
        })

    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "status": profile_status,
        "steps": [_snapshot_step_status(selection) for selection in selections],
        "sections": sections,
    }
    return "\n".join(lines).rstrip() + "\n", payload


def export_snapshot(
    cfg: Config,
    out: Path,
    *,
    base_url: str | None = None,
    provider: ProfileInfoProvider | None = None,
) -> Path:
    root = profile_root(cfg)
    with ProfileLock(root):
        text, payload = build_snapshot(cfg, base_url=base_url, provider=provider)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        _write_json(out.with_suffix(".json"), payload)
        return out


def print_summary(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'STEP':34s} {'STATUS':20s} {'RAN':3s} {'TIME':>8s} NOTE"]
    for row in rows:
        duration = row.get("duration_s")
        duration_s = "-" if duration is None else f"{float(duration):.1f}s"
        lines.append(
            f"{row['step_id'][:34]:34s} {str(row['status'])[:20]:20s} "
            f"{'yes' if row.get('ran') else 'no ':3s} {duration_s:>8s} {row.get('reason') or ''}"
        )
    return "\n".join(lines)
