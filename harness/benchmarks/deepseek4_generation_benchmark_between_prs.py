#!/usr/bin/env python3
"""Build and benchmark two DeepSeek4 Lucebox pull requests on one GPU host.

This is an orchestration layer around generation_benchmark.py. It fetches two
GitHub pull-request heads, caches source worktrees and incremental builds by
immutable commit and build configuration, tests both with identical settings,
benchmarks their servers sequentially, and writes fresh JSON and Markdown
comparison artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = SCRIPT_DIR.parent.parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "generation_benchmark.py"

SUITES = {
    "generation": {
        "prompts": SCRIPT_DIR / "prompts" / "generation_smoke.jsonl",
        "max_ctx": 2048,
        "max_tokens": 64,
        "smoke_case_limit": None,
    },
    "humaneval": {
        "prompts": SCRIPT_DIR / "prompts" / "bench_he.jsonl",
        "max_ctx": 4096,
        "max_tokens": 512,
        "smoke_case_limit": 3,
    },
    "gsm8k": {
        "prompts": SCRIPT_DIR / "prompts" / "bench_gsm.jsonl",
        "max_ctx": 8192,
        "max_tokens": 1024,
        "smoke_case_limit": 3,
    },
    "math": {
        "prompts": SCRIPT_DIR / "prompts" / "bench_math.jsonl",
        "max_ctx": 8192,
        "max_tokens": 2048,
        "smoke_case_limit": 3,
    },
}

RUN_PROFILES = {
    "smoke": {"repeats": 1, "warmup_repeats": 0},
    "full": {"repeats": 5, "warmup_repeats": 1},
}

LEGACY_MODES = {
    "smoke": ("generation", "smoke"),
    "full": ("gsm8k", "full"),
}


@dataclass(frozen=True)
class TopologyPlan:
    name: str
    local_backend: str
    target_device: str | None
    target_devices: str | None
    layer_split: str | None
    remote_backend: str | None = None

    @property
    def required_backends(self) -> tuple[str, ...]:
        return (
            (self.local_backend, self.remote_backend)
            if self.remote_backend
            else (self.local_backend,)
        )


@dataclass(frozen=True)
class PrSource:
    number: int
    sha: str
    tree: Path


class FileLock:
    """Process lock for one cached worktree or build configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: IO[str] | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


TOPOLOGY_DEFAULTS: dict[str, dict[str, str]] = {
    "hip-monolithic": {
        "local_backend": "hip",
        "target_device": "hip:0",
    },
    "hip-hybrid": {
        "local_backend": "hip",
        "target_device": "hip:0",
    },
    "hip-layer-split": {
        "local_backend": "hip",
        "target_devices": "hip:0,hip:1",
        "layer_split": "1,1",
    },
    "cuda-hip-layer-split": {
        "local_backend": "cuda",
        "target_devices": "cuda:0,hip:0",
        "layer_split": "24,19",
        "remote_backend": "hip",
    },
}

SINGLE_DEVICE_HIP_TOPOLOGIES = {"hip-monolithic", "hip-hybrid"}


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_device(value: str) -> tuple[str, int]:
    backend, separator, gpu_text = value.partition(":")
    if not separator or backend not in {"cuda", "hip"}:
        raise ValueError(f"invalid target device: {value}")
    try:
        gpu = int(gpu_text)
    except ValueError as exc:
        raise ValueError(f"invalid target device: {value}") from exc
    if gpu < 0:
        raise ValueError(f"invalid target device: {value}")
    return backend, gpu


def resolve_topology(args: argparse.Namespace) -> TopologyPlan:
    defaults = TOPOLOGY_DEFAULTS[args.topology]
    if args.topology in SINGLE_DEVICE_HIP_TOPOLOGIES:
        if args.target_devices or args.layer_split:
            raise ValueError("--target-devices and --layer-split require a layer-split topology")
        target_device = args.target_device or defaults["target_device"]
        backend, _ = parse_device(target_device)
        if backend != "hip":
            raise ValueError(f"{args.topology} requires a hip:<gpu> target device")
        return TopologyPlan(
            name=args.topology,
            local_backend="hip",
            target_device=target_device,
            target_devices=None,
            layer_split=None,
        )

    if args.target_device:
        raise ValueError("--target-device is only valid with a single-device HIP topology")
    target_devices = args.target_devices or defaults["target_devices"]
    layer_split = args.layer_split or defaults["layer_split"]
    devices = csv_items(target_devices)
    weights = csv_items(layer_split)
    if len(devices) < 2:
        raise ValueError("layer-split topologies require at least two target devices")
    if len(devices) != len(weights):
        raise ValueError("--layer-split must contain one weight per target device")
    backends = [parse_device(device)[0] for device in devices]
    try:
        parsed_weights = [float(weight) for weight in weights]
    except ValueError as exc:
        raise ValueError("--layer-split weights must be numbers") from exc
    if any(weight <= 0 or not math.isfinite(weight) for weight in parsed_weights):
        raise ValueError("--layer-split weights must be positive finite numbers")

    if args.topology == "hip-layer-split":
        if any(backend != "hip" for backend in backends):
            raise ValueError("hip-layer-split requires only hip:<gpu> target devices")
    else:
        if backends[0] != "cuda" or any(backend != "hip" for backend in backends[1:]):
            raise ValueError(
                "cuda-hip-layer-split requires a CUDA local device followed by HIP devices"
            )

    return TopologyPlan(
        name=args.topology,
        local_backend=defaults["local_backend"],
        target_device=None,
        target_devices=",".join(devices),
        layer_split=",".join(weights),
        remote_backend=defaults.get("remote_backend"),
    )


def resolve_benchmark_selection(args: argparse.Namespace) -> tuple[str, str]:
    if args.mode is not None:
        if args.suite is not None or args.profile is not None:
            raise ValueError("--mode cannot be combined with --suite or --profile")
        return LEGACY_MODES[args.mode]
    return args.suite or "generation", args.profile or "smoke"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def source_environment(path: Path, base: dict[str, str]) -> dict[str, str]:
    """Source a shell environment file and return its exported environment."""
    if not path.is_file():
        raise FileNotFoundError(f"environment file not found: {path}")
    command = ['set -a; source "$1" >&2; env -0', "benchmark-env", str(path)]
    result = subprocess.run(
        ["bash", "-c", *command],
        env=base,
        capture_output=True,
        check=False,
    )
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"failed to source environment file: {path}")
    sourced = base.copy()
    for entry in result.stdout.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        sourced[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
    return sourced


def env_or_default(
    explicit: Any,
    environment: dict[str, str],
    name: str,
    default: Any,
    converter: type = str,
) -> Any:
    if explicit is not None:
        return explicit
    value = environment.get(name)
    return converter(value) if value not in (None, "") else default


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    allowed_returncodes: tuple[int, ...] = (),
) -> int:
    print(f"+ {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    if returncode != 0 and returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(returncode, command)
    return returncode


def tail(path: Path, lines: int = 100) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except OSError:
        return ""


def default_cache_dir(repo: Path, environment: dict[str, str]) -> Path:
    cache_home = environment.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    repo_key = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:12]
    return base / "lucebox" / "deepseek4-pr-benchmark" / repo_key


class DeepSeek4PrComparison:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.environment = os.environ.copy()
        if args.env_file is not None:
            self.environment = source_environment(args.env_file.resolve(), self.environment)

        self.topology = resolve_topology(args)
        if args.ds4_expert_top_k is not None and (
            self.topology.name not in SINGLE_DEVICE_HIP_TOPOLOGIES
        ):
            raise ValueError("--ds4-expert-top-k is only valid with a single-device HIP topology")
        self.ds4_expert_top_k = (
            args.ds4_expert_top_k
            if args.ds4_expert_top_k is not None
            else (4 if self.topology.name in SINGLE_DEVICE_HIP_TOPOLOGIES else 0)
        )
        self.suite, self.profile = resolve_benchmark_selection(args)
        suite = SUITES[self.suite]
        profile = RUN_PROFILES[self.profile]
        self.repeats = env_or_default(
            args.repeats, self.environment, "REPEATS", profile["repeats"], int
        )
        self.warmup_repeats = env_or_default(
            args.warmup_repeats,
            self.environment,
            "WARMUP_REPEATS",
            profile["warmup_repeats"],
            int,
        )
        self.max_ctx = env_or_default(
            args.max_ctx, self.environment, "MAX_CTX", suite["max_ctx"], int
        )
        self.max_tokens = env_or_default(
            args.max_tokens, self.environment, "MAX_TOKENS", suite["max_tokens"], int
        )
        default_case_limit = suite["smoke_case_limit"] if self.profile == "smoke" else None
        self.case_limit = args.case_limit if args.case_limit is not None else default_case_limit
        if args.prompts is not None:
            prompts_value = args.prompts
        elif args.suite is not None:
            prompts_value = suite["prompts"]
        else:
            prompts_value = env_or_default(None, self.environment, "PROMPTS", suite["prompts"])
        prompts = Path(prompts_value).expanduser()
        self.prompts = (prompts if prompts.is_absolute() else self.repo / prompts).resolve()
        self.hip_arch = env_or_default(args.hip_arch, self.environment, "HIP_ARCH", "gfx1151")
        self.wave_size = env_or_default(
            args.wave_size, self.environment, "DFLASH_WAVE_SIZE", 32, int
        )
        self.hip_visible_devices = env_or_default(
            args.hip_visible_devices,
            self.environment,
            "HIP_VISIBLE_DEVICES",
            None,
        )
        self.cuda_visible_devices = env_or_default(
            args.cuda_visible_devices,
            self.environment,
            "CUDA_VISIBLE_DEVICES",
            None,
        )
        self.cuda_architectures = env_or_default(
            args.cuda_architectures,
            self.environment,
            "DFLASH27B_USER_CUDA_ARCHITECTURES",
            None,
        )
        self.model_id = env_or_default(args.model_id, self.environment, "MODEL_ID", "ds4")

        model_value = args.model
        if model_value is None:
            model_value = next(
                (
                    self.environment[name]
                    for name in ("MODEL", "DFLASH_TARGET", "TARGET")
                    if self.environment.get(name)
                ),
                None,
            )
        if not model_value:
            raise ValueError("set --model, MODEL, DFLASH_TARGET, or TARGET to a DeepSeek4 GGUF")
        model = Path(model_value).expanduser()
        self.model = (model if model.is_absolute() else self.repo / model).resolve()

        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        default_output = (
            self.repo
            / "benchmark-results"
            / (
                f"pr{args.baseline_pr}-vs-pr{args.candidate_pr}-{self.topology.name}-"
                f"{self.suite}-{self.profile}-{timestamp}"
            )
        )
        self.output = (args.output_dir or default_output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)

        if args.no_cache and args.cache_dir is not None:
            raise ValueError("--cache-dir cannot be combined with --no-cache")
        if not args.no_cache and args.worktree_root is not None:
            raise ValueError("--worktree-root requires --no-cache")
        if not args.no_cache and args.keep_worktrees:
            raise ValueError("--keep-worktrees requires --no-cache; cached worktrees persist")

        self.cache_enabled = not args.no_cache
        if self.cache_enabled:
            self.cache_root = (
                args.cache_dir.expanduser().resolve()
                if args.cache_dir is not None
                else default_cache_dir(self.repo, self.environment)
            )
        else:
            worktree_parent = (
                args.worktree_root.expanduser().resolve() if args.worktree_root else None
            )
            if worktree_parent is not None:
                worktree_parent.mkdir(parents=True, exist_ok=True)
            self.cache_root = Path(
                tempfile.mkdtemp(
                    prefix=f"lucebox-pr-bench-{args.baseline_pr}-{args.candidate_pr}-",
                    dir=worktree_parent,
                )
            )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.sources: list[PrSource] = []
        self.build_dirs: dict[tuple[str, str], Path] = {}
        self.cache_events: list[dict[str, Any]] = []
        self.expected_failures: dict[str, list[str]] = {}
        self.server: subprocess.Popen[str] | None = None
        self.server_log = None

    def validate(self) -> None:
        if not (self.repo / ".git").exists():
            raise ValueError(f"not a Git checkout: {self.repo}")
        if not self.model.is_file():
            raise ValueError(f"model does not exist: {self.model}")
        if not self.prompts.is_file():
            raise ValueError(f"prompts do not exist: {self.prompts}")
        if not BENCHMARK_SCRIPT.is_file():
            raise ValueError(f"benchmark runner does not exist: {BENCHMARK_SCRIPT}")
        if self.repeats <= 0 or self.warmup_repeats < 0:
            raise ValueError("repeats must be positive and warmup repeats must be nonnegative")
        if self.case_limit is not None and self.case_limit <= 0:
            raise ValueError("case limit must be positive")

    def git(self, *arguments: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *arguments], check=True)

    def git_output(self, *arguments: str, cwd: Path | None = None) -> str:
        return subprocess.check_output(
            ["git", "-C", str(cwd or self.repo), *arguments], text=True
        ).strip()

    def prepare_source(self, number: int, sha: str) -> PrSource:
        tree = self.cache_root / "worktrees" / sha
        lock_path = self.cache_root / "locks" / f"worktree-{sha}.lock"
        with FileLock(lock_path):
            status = "reuse"
            if tree.exists():
                try:
                    cached_sha = self.git_output("rev-parse", "HEAD", cwd=tree)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"invalid cached worktree at {tree}; remove it or use --no-cache"
                    ) from exc
                if cached_sha != sha:
                    raise RuntimeError(
                        f"cached worktree {tree} contains {cached_sha}, expected {sha}"
                    )
            else:
                status = "cold"
                tree.parent.mkdir(parents=True, exist_ok=True)
                self.git("worktree", "prune")
                self.git("worktree", "add", "--detach", str(tree), sha)

            print(f"[cache] source pr{number} {sha[:12]}: {status} ({tree})", flush=True)
            run_logged(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=tree,
                environment=self.environment,
                log_path=self.output / f"pr{number}-submodules.log",
            )
            dirty = self.git_output(
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--ignore-submodules=none",
                cwd=tree,
            )
            if dirty:
                raise RuntimeError(
                    f"cached worktree is dirty at {tree}; remove it or use --no-cache\n{dirty}"
                )
        self.cache_events.append(
            {
                "kind": "source",
                "pr": number,
                "sha": sha,
                "status": status,
                "path": str(tree),
            }
        )
        return PrSource(number=number, sha=sha, tree=tree)

    def prepare_worktrees(self) -> tuple[PrSource, PrSource]:
        baseline_ref = f"refs/remotes/pr-bench/pr-{self.args.baseline_pr}"
        candidate_ref = f"refs/remotes/pr-bench/pr-{self.args.candidate_pr}"
        self.git(
            "fetch",
            self.args.upstream_url,
            f"+refs/pull/{self.args.baseline_pr}/head:{baseline_ref}",
            f"+refs/pull/{self.args.candidate_pr}/head:{candidate_ref}",
        )
        baseline_sha = self.git_output("rev-parse", baseline_ref)
        candidate_sha = self.git_output("rev-parse", candidate_ref)
        baseline = self.prepare_source(self.args.baseline_pr, baseline_sha)
        candidate = self.prepare_source(self.args.candidate_pr, candidate_sha)
        self.sources = [baseline, candidate]
        return baseline, candidate

    def write_metadata(self, baseline: PrSource, candidate: PrSource) -> None:
        metadata = {
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "suite": self.suite,
            "profile": self.profile,
            "legacy_mode": self.args.mode,
            "upstream_url": self.args.upstream_url,
            "baseline_pr": self.args.baseline_pr,
            "baseline_sha": baseline.sha,
            "candidate_pr": self.args.candidate_pr,
            "candidate_sha": candidate.sha,
            "model": str(self.model),
            "prompts": str(self.prompts),
            "topology": self.topology.name,
            "local_backend": self.topology.local_backend,
            "remote_backend": self.topology.remote_backend,
            "target_device": self.topology.target_device,
            "target_devices": self.topology.target_devices,
            "layer_split": self.topology.layer_split,
            "ds4_fused_decode": self.topology.name == "hip-monolithic",
            "ds4_expert_top_k": self.ds4_expert_top_k,
            "hip_arch": self.hip_arch,
            "wave_size": self.wave_size,
            "hip_visible_devices": self.hip_visible_devices,
            "cuda_visible_devices": self.cuda_visible_devices,
            "cuda_architectures": self.cuda_architectures,
            "peer_access": self.args.peer_access,
            "repeats": self.repeats,
            "warmup_repeats": self.warmup_repeats,
            "max_ctx": self.max_ctx,
            "max_tokens": self.max_tokens,
            "case_limit": self.case_limit,
            "expected_check_failures": self.expected_failures,
            "cache": {
                "enabled": self.cache_enabled,
                "root": str(self.cache_root),
                "policy": self.args.cache_policy,
                "rebuild": self.args.rebuild,
                "events": self.cache_events,
            },
        }
        (self.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_system_info(self) -> None:
        commands = [
            ["uname", "-a"],
            ["cmake", "--version"],
            ["hipcc", "--version"],
            ["rocm-smi", "--showproductname", "--showclocks", "--showpower"],
            ["nvcc", "--version"],
            ["nvidia-smi", "-L"],
        ]
        with (self.output / "system-info.txt").open("w", encoding="utf-8") as info:
            for command in commands:
                if shutil.which(command[0], path=self.environment.get("PATH")) is None:
                    continue
                info.write(f"=== {' '.join(command)} ===\n")
                result = subprocess.run(
                    command,
                    env=self.runtime_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                info.write(result.stdout)
                info.write("\n")

    def build_config(self, backend: str) -> dict[str, Any]:
        toolchain_names = (
            "CC",
            "CXX",
            "HIPCXX",
            "CUDACXX",
            "ROCM_PATH",
            "CMAKE_PREFIX_PATH",
        )
        config: dict[str, Any] = {
            "cache_schema": 1,
            "backend": backend,
            "build_type": "Release",
            "toolchain_environment": {
                name: self.environment[name]
                for name in toolchain_names
                if self.environment.get(name)
            },
        }
        if backend == "hip":
            config.update(
                {
                    "hip_arch": self.hip_arch,
                    "wave_size": self.wave_size,
                    "hip_sm80_equiv": True,
                    "cmake_hip_flags": self.environment.get("CMAKE_HIP_FLAGS", ""),
                }
            )
        elif backend == "cuda":
            config["cuda_architectures"] = self.cuda_architectures
        return config

    def build_dir_for(self, source: PrSource, backend: str) -> tuple[Path, str]:
        config_json = json.dumps(self.build_config(backend), sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(config_json.encode()).hexdigest()[:12]
        return self.cache_root / "builds" / source.sha / f"{backend}-{key}", key

    def build_backend(self, label: str, source: PrSource, backend: str) -> None:
        log = self.output / f"{label}-{backend}-build.log"
        build_dir, key = self.build_dir_for(source, backend)
        self.build_dirs[(source.sha, backend)] = build_dir
        lock_path = self.cache_root / "locks" / f"build-{source.sha}-{backend}-{key}.lock"
        with FileLock(lock_path):
            if self.args.rebuild and build_dir.exists():
                shutil.rmtree(build_dir)
                status = "rebuild"
            elif (build_dir / "CMakeCache.txt").is_file():
                status = "reuse"
            elif build_dir.exists():
                status = "resume"
            else:
                status = "cold"
            print(
                f"[cache] build pr{source.number} {backend} {key}: {status} ({build_dir})",
                flush=True,
            )
            self.configure_and_build(label, source, backend, build_dir, log)
            manifest = {
                "pr": source.number,
                "sha": source.sha,
                "key": key,
                "config": self.build_config(backend),
            }
            (build_dir / "pr-benchmark-cache.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.cache_events.append(
            {
                "kind": "build",
                "pr": source.number,
                "sha": source.sha,
                "backend": backend,
                "key": key,
                "status": status,
                "path": str(build_dir),
            }
        )

    def configure_and_build(
        self,
        label: str,
        source: PrSource,
        backend: str,
        build_dir: Path,
        log: Path,
    ) -> None:
        tree = source.tree
        configure = [
            "cmake",
            "-S",
            str(tree / "server"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DDFLASH27B_GPU_BACKEND={backend}",
        ]
        if backend == "hip":
            hip_flags = self.environment.get("CMAKE_HIP_FLAGS", "")
            hip_flags = f"{hip_flags} -DDFLASH_WAVE_SIZE={self.wave_size}".strip()
            configure.extend(
                [
                    f"-DDFLASH27B_HIP_ARCHITECTURES={self.hip_arch}",
                    "-DDFLASH27B_HIP_SM80_EQUIV=ON",
                    f"-DCMAKE_HIP_FLAGS={hip_flags}",
                ]
            )
        elif self.cuda_architectures:
            # Pass the canonical CMake variable because older PRs initialize
            # DFLASH27B_USER_CUDA_ARCHITECTURES from it before project().
            configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={self.cuda_architectures}")
        run_logged(
            configure,
            cwd=tree,
            environment=self.environment,
            log_path=log,
        )
        targets = ["test_deepseek4_unit"]
        if backend == self.topology.local_backend:
            targets.append("dflash_server")
        if backend == self.topology.remote_backend:
            targets.append("backend_ipc_daemon")
        run_logged(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                *targets,
                f"-j{os.cpu_count() or 1}",
            ],
            cwd=tree,
            environment=self.environment,
            log_path=log,
        )

    def build(self, label: str, source: PrSource) -> None:
        for backend in self.topology.required_backends:
            self.build_backend(label, source, backend)

    def read_active_cache(self) -> dict[str, Any]:
        path = self.cache_root / "active-pair.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid cache state file: {path}") from exc
        return data if isinstance(data, dict) else {}

    def write_active_cache(self, data: dict[str, Any]) -> None:
        path = self.cache_root / "active-pair.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def evict_worktree(self, tree: Path) -> None:
        print(f"[cache] evict source: {tree}", flush=True)
        try:
            self.git("worktree", "remove", "--force", str(tree))
        except subprocess.CalledProcessError:
            # A directory git no longer recognizes as a worktree (for example
            # after an interrupted eviction) must not wedge every later run.
            shutil.rmtree(tree, ignore_errors=True)
            self.git("worktree", "prune")
        self.cache_events.append({"kind": "eviction", "type": "source", "path": str(tree)})

    def evict_build(self, build_dir: Path) -> None:
        print(f"[cache] evict build: {build_dir}", flush=True)
        shutil.rmtree(build_dir)
        self.cache_events.append({"kind": "eviction", "type": "build", "path": str(build_dir)})

    def evict_ref(self, refname: str) -> None:
        print(f"[cache] evict ref: {refname}", flush=True)
        self.git("update-ref", "-d", refname)
        self.cache_events.append({"kind": "eviction", "type": "ref", "path": refname})

    def prune_cache(self) -> None:
        if not self.cache_enabled or self.args.cache_policy == "keep-all":
            return

        active_shas = {source.sha for source in self.sources}
        previous = self.read_active_cache()
        previous_builds = previous.get("builds", {})
        retained_builds: dict[str, dict[str, str]] = {}
        for sha in active_shas:
            builds = previous_builds.get(sha, {}) if isinstance(previous_builds, dict) else {}
            retained_builds[sha] = dict(builds) if isinstance(builds, dict) else {}
        for source in self.sources:
            for backend in self.topology.required_backends:
                _, key = self.build_dir_for(source, backend)
                retained_builds[source.sha][backend] = key

        worktrees_root = self.cache_root / "worktrees"
        if worktrees_root.is_dir():
            for tree in worktrees_root.iterdir():
                if tree.is_dir() and tree.name not in active_shas:
                    self.evict_worktree(tree)

        builds_root = self.cache_root / "builds"
        if builds_root.is_dir():
            for sha_dir in builds_root.iterdir():
                if not sha_dir.is_dir():
                    continue
                if sha_dir.name not in active_shas:
                    self.evict_build(sha_dir)
                    continue
                for build_dir in sha_dir.iterdir():
                    if not build_dir.is_dir():
                        continue
                    backend, separator, key = build_dir.name.partition("-")
                    if not separator or retained_builds[sha_dir.name].get(backend) != key:
                        self.evict_build(build_dir)

        # Stale pr-bench refs pin every previously fetched PR's objects in the
        # shared repository, so git gc can never reclaim them. Retained
        # worktrees keep their own objects reachable through their HEADs.
        fetched_refs = self.git_output(
            "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/pr-bench/"
        )
        for line in fetched_refs.splitlines():
            refname, _, sha = line.partition(" ")
            if sha not in active_shas:
                self.evict_ref(refname)

        # Per-SHA lock files are only ever taken under run.lock, so locks for
        # evicted SHAs cannot be held by anyone and are safe to delete.
        locks_root = self.cache_root / "locks"
        if locks_root.is_dir():
            for lock in locks_root.iterdir():
                if lock.name.startswith("worktree-") and lock.name.endswith(".lock"):
                    sha = lock.name[len("worktree-") : -len(".lock")]
                elif lock.name.startswith("build-"):
                    sha = lock.name[len("build-") :].split("-", 1)[0]
                else:
                    continue
                if len(sha) == 40 and sha not in active_shas:
                    lock.unlink(missing_ok=True)

        self.git("worktree", "prune")
        self.write_active_cache(
            {
                "schema": 1,
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                "prs": [{"number": source.number, "sha": source.sha} for source in self.sources],
                "builds": retained_builds,
            }
        )

    def runtime_environment(self) -> dict[str, str]:
        environment = self.environment.copy()
        if self.hip_visible_devices is not None:
            environment["HIP_VISIBLE_DEVICES"] = self.hip_visible_devices
        if self.cuda_visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return environment

    def unit_test_backend(self, label: str, source: PrSource, backend: str) -> None:
        build_dir = self.build_dirs[(source.sha, backend)]
        run_logged(
            [str(build_dir / "test_deepseek4_unit")],
            cwd=self.repo,
            environment=self.runtime_environment(),
            log_path=self.output / f"{label}-{backend}-unit.log",
        )

    def unit_test(self, label: str, source: PrSource) -> None:
        for backend in self.topology.required_backends:
            self.unit_test_backend(label, source, backend)

    def assert_port_available(self) -> None:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError(f"port {self.args.port} is already in use")

    def start_server(self, label: str, source: PrSource) -> None:
        self.assert_port_available()
        environment = self.runtime_environment()
        environment.pop("DFLASH_DS4_ROCMFPX_HC_GPU", None)
        environment.update(
            {
                "DFLASH_ROCMFP2_FIXED_K": "1",
                "DFLASH_ROCMFP3_FIXED_K": "1",
                "DFLASH_ROCMFP4_UNROLL2": "1",
            }
        )
        build_dir = self.build_dirs[(source.sha, self.topology.local_backend)]
        command = [
            str(build_dir / "dflash_server"),
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
            "--max-ctx",
            str(self.max_ctx),
            "--max-tokens",
            str(self.max_tokens),
            "--model-name",
            self.model_id,
            "--prefix-cache-slots",
            "0",
            "--disk-prefix-cache",
            "off",
        ]
        if self.topology.name == "hip-monolithic":
            command.append("--ds4-fused-decode")
        if self.topology.name in SINGLE_DEVICE_HIP_TOPOLOGIES:
            command.extend(["--ds4-expert-top-k", str(self.ds4_expert_top_k)])
        if self.topology.target_device:
            command.extend(["--target-device", self.topology.target_device])
        else:
            assert self.topology.target_devices is not None
            assert self.topology.layer_split is not None
            command.extend(
                [
                    "--target-devices",
                    self.topology.target_devices,
                    "--target-layer-split",
                    self.topology.layer_split,
                ]
            )
        if self.args.peer_access:
            command.append("--peer-access")
        if self.topology.remote_backend:
            remote_build = self.build_dirs[(source.sha, self.topology.remote_backend)]
            ipc_work_dir = self.output / f"{label}-target-shard-ipc"
            ipc_work_dir.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    "--target-shard-ipc-bin",
                    str(remote_build / "backend_ipc_daemon"),
                    "--target-shard-ipc-work-dir",
                    str(ipc_work_dir),
                ]
            )
        log_path = self.output / f"{label}-server.log"
        self.server_log = log_path.open("w", encoding="utf-8")
        print(f"Starting {label}: {' '.join(command)}", flush=True)
        self.server = subprocess.Popen(
            command,
            cwd=self.repo,
            env=environment,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self.args.server_timeout
        health_url = f"http://127.0.0.1:{self.args.port}/health"
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"{label} server exited during startup\n{tail(log_path)}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        self.check_startup_log(label, log_path)
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise TimeoutError(f"timed out waiting for {label} server\n{tail(log_path)}")

    def check_startup_log(self, label: str, log_path: Path) -> None:
        if self.topology.name not in SINGLE_DEVICE_HIP_TOPOLOGIES:
            return
        server_output = log_path.read_text(encoding="utf-8", errors="replace")
        if self.topology.name == "hip-monolithic":
            # PR #520 reworded the monolithic-load message; accept both variants.
            requested = (
                "fused decode requested; loading monolithic HIP model" in server_output
                or "monolithic execution requested" in server_output
            )
            if (
                not requested
                or "fused_decode=on" not in server_output
                or "[hybrid]" in server_output
            ):
                raise RuntimeError(
                    f"{label} did not activate monolithic fused decode\n{tail(log_path)}"
                )
        else:
            # PR #520 inserted the prefill mode between fused_decode=off and
            # [hybrid] on the initialized line; match the markers separately.
            if (
                "HIP target detected; using hybrid expert load path" not in server_output
                or "fused_decode=off" not in server_output
                or "[hybrid]" not in server_output
            ):
                raise RuntimeError(
                    f"{label} did not activate HIP hybrid expert placement\n{tail(log_path)}"
                )

    def stop_server(self) -> None:
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5)
        self.server = None
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

    def benchmark_command(self, label: str, repeats: int, output: Path) -> list[str]:
        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "run",
            "--name",
            label,
            "--url",
            f"http://127.0.0.1:{self.args.port}/v1",
            "--model",
            self.model_id,
            "--prompts",
            str(self.prompts),
            "--max-tokens",
            str(self.max_tokens),
            "--temperature",
            "0",
            "--repeats",
            str(repeats),
            "--json-out",
            str(output),
        ]
        if self.case_limit is not None:
            command.extend(["--case-limit", str(self.case_limit)])
        return command

    def run_benchmark(
        self, label: str, repeats: int, output: Path, log_path: Path, *, record: bool
    ) -> None:
        command = self.benchmark_command(label, repeats, output)
        returncode = run_logged(
            command,
            cwd=self.repo,
            environment=self.environment,
            log_path=log_path,
            allowed_returncodes=(1,),
        )
        if returncode == 0:
            return
        # Exit code 1 with a written report means expected-output checks
        # failed. Keep going so the candidate still runs and the comparison
        # artifacts land; run_comparison fails at the end instead.
        if not output.is_file():
            raise subprocess.CalledProcessError(returncode, command)
        report = json.loads(output.read_text(encoding="utf-8"))
        failed = sorted(
            str(case.get("id")) for case in report.get("cases", []) if not case.get("expected_pass")
        )
        if not failed:
            raise subprocess.CalledProcessError(returncode, command)
        print(f"[bench] {label}: expected checks failed: {', '.join(failed)}", flush=True)
        if record:
            self.expected_failures[label] = failed

    def benchmark(self, label: str, source: PrSource) -> None:
        self.start_server(label, source)
        try:
            if self.warmup_repeats:
                self.run_benchmark(
                    f"{label}-warmup",
                    self.warmup_repeats,
                    self.output / f"{label}-warmup.json",
                    self.output / f"{label}-benchmark.log",
                    record=False,
                )
            self.run_benchmark(
                label,
                self.repeats,
                self.output / f"{label}.json",
                self.output / f"{label}-benchmark.log",
                record=True,
            )
        finally:
            self.stop_server()

    def compare(self, baseline_label: str, candidate_label: str) -> None:
        comparison_path = self.output / "compare.json"
        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "compare",
            "--baseline",
            str(self.output / f"{baseline_label}.json"),
            "--candidate",
            str(self.output / f"{candidate_label}.json"),
            "--json-out",
            str(comparison_path),
            "--md-out",
            str(self.output / "report.md"),
        ]
        # Exit code 1 restates expected-check failures already recorded by
        # run_benchmark; the comparison artifacts are still written.
        returncode = run_logged(
            command,
            cwd=self.repo,
            environment=self.environment,
            log_path=self.output / "compare.log",
            allowed_returncodes=(1,),
        )
        if returncode != 0 and not comparison_path.is_file():
            raise subprocess.CalledProcessError(returncode, command)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        summary = comparison["summary"]
        matches = int(summary["normalized_matches"])
        cases = int(summary["cases"])
        if matches != cases and not self.args.allow_output_differences:
            mismatched = ", ".join(
                str(row.get("id"))
                for row in comparison.get("cases", [])
                if not row.get("normalized_match")
            )
            raise RuntimeError(
                f"output regression check failed: {matches}/{cases} normalized outputs match "
                f"(differing: {mismatched}); inspect report.md or pass --allow-output-differences"
            )

    def cleanup(self) -> None:
        self.stop_server()
        if self.cache_enabled:
            return
        if self.args.keep_worktrees:
            print(f"Worktrees retained at {self.cache_root}")
            return
        for tree in {source.tree for source in self.sources}:
            if tree.exists():
                subprocess.run(
                    ["git", "-C", str(self.repo), "worktree", "remove", "--force", str(tree)],
                    check=False,
                )
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def run_comparison(self) -> None:
        baseline, candidate = self.prepare_worktrees()
        self.write_metadata(baseline, candidate)
        self.write_system_info()
        workload = f"{self.suite}-{self.profile}"
        baseline_label = f"pr{self.args.baseline_pr}-{self.topology.name}-{workload}"
        candidate_label = f"pr{self.args.candidate_pr}-{self.topology.name}-{workload}"

        self.build(baseline_label, baseline)
        self.build(candidate_label, candidate)
        if not self.args.skip_unit_tests:
            self.unit_test(baseline_label, baseline)
            self.unit_test(candidate_label, candidate)
        self.prune_cache()
        self.write_metadata(baseline, candidate)
        if self.args.prepare_only:
            print("Builds prepared. Run again without --prepare-only to benchmark.")
            print(f"Results: {self.output}")
            return
        try:
            self.benchmark(baseline_label, baseline)
            self.benchmark(candidate_label, candidate)
            self.compare(baseline_label, candidate_label)
        finally:
            # Record expected-check failures even when the comparison or the
            # output-equivalence gate raises.
            self.write_metadata(baseline, candidate)
        print(f"Results: {self.output}")
        print(f"Report:  {self.output / 'report.md'}")
        if self.expected_failures:
            failures = "; ".join(
                f"{label}: {', '.join(cases)}"
                for label, cases in sorted(self.expected_failures.items())
            )
            raise RuntimeError(
                f"expected output checks failed ({failures}); "
                "the comparison artifacts above are complete"
            )

    def run(self) -> None:
        self.validate()
        if not self.cache_enabled:
            self.run_comparison()
            return
        with FileLock(self.cache_root / "locks" / "run.lock"):
            try:
                self.run_comparison()
            finally:
                self.stop_server()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_pr", type=positive_int, help="Baseline GitHub PR number")
    parser.add_argument("candidate_pr", type=positive_int, help="Candidate GitHub PR number")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        help="Benchmark suite (default: generation)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        help="Smoke uses a small sample; full runs the complete suite",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(LEGACY_MODES),
        help="Compatibility alias: smoke=generation smoke, full=GSM8K full",
    )
    parser.add_argument(
        "--topology",
        choices=sorted(TOPOLOGY_DEFAULTS),
        default="hip-monolithic",
        help="DeepSeek4 execution topology",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--upstream-url", default="https://github.com/Luce-Org/lucebox-hub.git")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional shell environment file to source",
    )
    parser.add_argument("--model", help="DeepSeek4 GGUF; overrides model environment variables")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Persistent source/build cache (default: XDG cache or ~/.cache)",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("current-pair", "keep-all"),
        default="current-pair",
        help="Retain only the active PR pair (default) or every cached PR",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Use an isolated disposable source/build directory",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete matching cached build directories before compiling",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Fetch, build, and test both PRs without running generation",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        help="Parent for disposable worktrees; requires --no-cache",
    )
    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="Retain disposable worktrees; requires --no-cache",
    )
    parser.add_argument("--target-device", help="Single-device HIP target, e.g. hip:0")
    parser.add_argument(
        "--target-devices",
        help="Layer-split targets, e.g. hip:0,hip:1 or cuda:0,hip:0",
    )
    parser.add_argument("--layer-split", help="Layer weights matching --target-devices, e.g. 24,19")
    parser.add_argument("--peer-access", action="store_true")
    parser.add_argument(
        "--ds4-expert-top-k",
        type=nonnegative_int,
        help="Routed experts for single-device HIP (default: 4; 0=model default)",
    )
    parser.add_argument("--hip-arch")
    parser.add_argument("--wave-size", type=positive_int)
    parser.add_argument("--hip-visible-devices")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument(
        "--cuda-architectures",
        help="CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES",
    )
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--case-limit", type=positive_int)
    parser.add_argument("--warmup-repeats", type=nonnegative_int)
    parser.add_argument("--max-ctx", type=positive_int)
    parser.add_argument("--max-tokens", type=positive_int)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--port", type=positive_int, default=18080)
    parser.add_argument("--server-timeout", type=positive_int, default=600)
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument(
        "--allow-output-differences",
        action="store_true",
        help="Do not fail when normalized baseline and candidate outputs differ",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    comparison: DeepSeek4PrComparison | None = None
    try:
        comparison = DeepSeek4PrComparison(args)
        comparison.run()
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if comparison is not None:
            comparison.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
