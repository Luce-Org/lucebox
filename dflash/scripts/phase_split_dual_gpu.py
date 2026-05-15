#!/usr/bin/env python3
"""Run PFlash prefill through a persistent daemon, optionally followed by target generation.

This phase-split harness is intentionally PFlash-only. It keeps the Qwen3-0.6B
PFlash drafter resident in `pflash_daemon`, optionally on a different CUDA or
HIP backend from the later target run. The cross-backend boundary is host-side
token/text data; target layer split remains inside one backend binary.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from placement.backend_device import apply_backend_visible_devices
from placement.test_dflash_args import TestDflashLaunchArgs


ROOT = Path(__file__).resolve().parent.parent


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


DEFAULT_BUILD = env_path("PFLASH_PHASE_BUILD_DIR", ROOT / "build")
DEFAULT_DRAFTER = env_path("PFLASH_PHASE_DRAFTER", ROOT / "models" / "Qwen3-0.6B-BF16.gguf")
DEFAULT_TOKENIZER = os.environ.get("PFLASH_PHASE_TOKENIZER", "Qwen/Qwen3-0.6B")
DEFAULT_TARGET = env_path("DFLASH_TARGET", ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_TARGET_DRAFT = env_path("DFLASH_DRAFT", ROOT / "models" / "draft")
DEFAULT_TARGET_TOKENIZER = os.environ.get("PFLASH_PHASE_TARGET_TOKENIZER", "Qwen/Qwen3.6-27B")


def write_counted_i32(path: Path, ids: Iterable[int]) -> None:
    values = [int(x) for x in ids]
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(values)))
        if values:
            f.write(struct.pack("<" + "i" * len(values), *values))


def write_raw_i32(path: Path, ids: Iterable[int]) -> None:
    values = [int(x) for x in ids]
    with path.open("wb") as f:
        if values:
            f.write(struct.pack("<" + "i" * len(values), *values))


def read_raw_i32(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise RuntimeError(f"raw int32 file has odd byte length: {path}")
    if not raw:
        return []
    return list(struct.unpack("<" + "i" * (len(raw) // 4), raw))


def read_stream_until_sentinel(r_fd: int) -> list[int]:
    out: list[int] = []
    while True:
        raw = os.read(r_fd, 4)
        if not raw or len(raw) < 4:
            raise RuntimeError("pflash daemon stream closed before sentinel")
        tok = struct.unpack("<i", raw)[0]
        if tok == -1:
            return out
        out.append(tok)


class ProcessLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.queue: queue.Queue[str] = queue.Queue()
        self._file = path.open("wb")
        self._thread: threading.Thread | None = None

    def attach(self, proc: subprocess.Popen[bytes]) -> None:
        def reader() -> None:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                self._file.write(raw)
                self._file.flush()
                self.queue.put(raw.decode("utf-8", errors="replace").rstrip("\n"))
            self._file.close()

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def wait_for(self, needle: str, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        tail: list[str] = []
        while time.time() < deadline:
            try:
                line = self.queue.get(timeout=0.2)
                tail.append(line)
                if needle in line:
                    return
            except queue.Empty:
                pass
        raise TimeoutError(f"timed out waiting for {needle!r}; tail={tail[-12:]}")


class PFlashDaemon:
    def __init__(self, *, binary: Path, drafter: Path, gpu: int, backend: str,
                 visible_devices: str | None, log_path: Path,
                 env: dict[str, str]) -> None:
        self.binary = binary
        self.drafter = drafter
        self.gpu = gpu
        self.backend = backend
        self.visible_devices = visible_devices
        self.log = ProcessLog(log_path)
        self.env = env
        self.proc: subprocess.Popen[bytes] | None = None
        self.r_fd: int | None = None

    def start(self) -> float:
        r_fd, w_fd = os.pipe()
        env = apply_backend_visible_devices(
            self.backend,
            visible_devices=self.visible_devices,
            fallback_device=self.gpu,
            base_env={**os.environ, **self.env},
        )
        cmd = [str(self.binary), str(self.drafter), f"--stream-fd={w_fd}"]
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(w_fd,),
            env=env,
            cwd=str(ROOT),
            bufsize=0,
        )
        os.close(w_fd)
        self.r_fd = r_fd
        self.log.attach(self.proc)
        self.log.wait_for("[pflash-daemon] ready", 180)
        return time.perf_counter() - t0

    def compress(self, counted_ids: Path, *, keep_ratio: float, lookahead: int,
                 chunk_size: int, pool_kernel: int) -> tuple[list[int], float]:
        if self.proc is None or self.proc.stdin is None or self.r_fd is None:
            raise RuntimeError("pflash daemon is not running")
        keep_x1000 = int(round(keep_ratio * 1000))
        cmd = f"compress {keep_x1000} {lookahead} {chunk_size} {pool_kernel} {counted_ids}\n"
        t0 = time.perf_counter()
        self.proc.stdin.write(cmd.encode("utf-8"))
        self.proc.stdin.flush()
        tokens = read_stream_until_sentinel(self.r_fd)
        return tokens, time.perf_counter() - t0

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b"quit\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.r_fd is not None:
            try:
                os.close(self.r_fd)
            except OSError:
                pass
        self.proc = None


class GpuMonitor:
    def __init__(self, path: Path, backend: str) -> None:
        self.path = path
        self.backend = backend
        self.phase = "init"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def start(self) -> None:
        def phase() -> str:
            with self._lock:
                return self.phase

        def loop() -> None:
            with self.path.open("w") as f:
                if self.backend == "cuda":
                    f.write("ts,phase,index,temp_c,fan_pct,power_w,power_limit_w,mem_used_mib,mem_total_mib,util_pct\n")
                else:
                    f.write("ts,phase,card,temp_edge_c,temp_junction_c,temp_memory_c,power_w,gpu_use_pct,vram_pct,vram_activity_pct,mem_activity\n")
                f.flush()
                while not self._stop.is_set():
                    ts = time.time()
                    try:
                        if self.backend == "cuda":
                            fields = "index,temperature.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total,utilization.gpu"
                            out = subprocess.check_output(
                                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                text=True,
                                stderr=subprocess.DEVNULL,
                                timeout=2,
                            )
                            for line in out.strip().splitlines():
                                parts = [p.strip() for p in line.split(",")]
                                if len(parts) == 8:
                                    f.write(",".join([f"{ts:.3f}", phase()] + parts) + "\n")
                        else:
                            out = subprocess.check_output(
                                ["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--showtemp", "--csv"],
                                text=True,
                                stderr=subprocess.DEVNULL,
                                timeout=2,
                            )
                            for line in out.strip().splitlines()[1:]:
                                parts = [p.strip() for p in line.split(",")]
                                if len(parts) >= 9:
                                    f.write(",".join([f"{ts:.3f}", phase()] + parts[:9]) + "\n")
                        f.flush()
                    except Exception as exc:
                        f.write(f"{ts:.3f},{phase()},ERR,{type(exc).__name__}\n")
                        f.flush()
                    self._stop.wait(1.0)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summarize_gpu(self, gpu: int, *, target_phase: bool | None = None) -> dict[str, float | int | None]:
        rows: list[dict[str, float | int]] = []
        if not self.path.exists():
            return {"samples": 0}
        for line in self.path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            is_target_phase = parts[1].endswith("_target")
            if target_phase is not None and is_target_phase != target_phase:
                continue
            if parts[2] == "ERR":
                continue
            try:
                if self.backend == "cuda":
                    if len(parts) != 10:
                        continue
                    idx = int(parts[2])
                    if idx != gpu:
                        continue
                    rows.append({
                        "temp": float(parts[3]),
                        "fan": float(parts[4]),
                        "power": float(parts[5]),
                        "mem": float(parts[7]),
                        "util": float(parts[9]),
                    })
                else:
                    if len(parts) < 11:
                        continue
                    idx = int(parts[2].replace("card", ""))
                    if idx != gpu:
                        continue
                    rows.append({
                        "temp": float(parts[4]),
                        "fan": 0.0,
                        "power": float(parts[6]),
                        "mem": float(parts[8]),
                        "util": float(parts[7]),
                    })
            except ValueError:
                pass
        if not rows:
            return {"samples": 0}
        return {
            "samples": len(rows),
            "mem_max_mib": max(float(r["mem"]) for r in rows),
            "temp_max_c": max(float(r["temp"]) for r in rows),
            "fan_max_pct": max(float(r["fan"]) for r in rows),
            "power_avg_w": mean(float(r["power"]) for r in rows),
            "power_max_w": max(float(r["power"]) for r in rows),
            "util_avg_pct": mean(float(r["util"]) for r in rows),
            "util_max_pct": max(float(r["util"]) for r in rows),
        }


def parse_device_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out or default


def summarize_devices(monitor: GpuMonitor, devices: list[int], *,
                      target_phase: bool | None = None) -> dict[str, dict[str, float | int | None]]:
    return {str(gpu): monitor.summarize_gpu(gpu, target_phase=target_phase)
            for gpu in devices}


@dataclass
class CompressionCase:
    name: str
    source_tokens: int
    compressed_tokens: int
    compress_wall_s: float
    compress_tok_s: float
    compression_ratio: float
    retained_key: bool | None = None
    retained_answer: bool | None = None
    target_input_tokens: int | None = None
    target_output_tokens: int | None = None
    target_wall_s: float | None = None
    target_returncode: int | None = None


def load_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )


def load_target_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.target_tokenizer,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )


def resolve_target_draft(path: Path) -> Path:
    if path.is_file():
        return path
    for candidate in path.rglob("model.safetensors"):
        return candidate
    raise SystemExit(f"target draft safetensors not found at {path}")


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"bad env override {item!r}; expected NAME=VALUE")
        name, value = item.split("=", 1)
        if not name:
            raise SystemExit(f"bad env override {item!r}; empty name")
        out[name] = value
    return out


def target_env(args) -> dict[str, str]:
    return apply_backend_visible_devices(
        args.target_backend,
        visible_devices=args.target_visible_devices,
        base_env={**os.environ, **parse_env_overrides(args.target_env)},
    )


def run_target_generation(args, case_dir: Path, compressed_text: str,
                          target_tokenizer) -> dict[str, float | int | str]:
    target_ids = target_tokenizer.encode(compressed_text, add_special_tokens=False)
    if not target_ids:
        raise RuntimeError("compressed prompt is empty after target tokenization")
    prompt_path = case_dir / "target_prompt.bin"
    out_path = case_dir / "target_out.bin"
    log_path = case_dir / "target.log"
    write_raw_i32(prompt_path, target_ids)

    cmd = [
        str(args.target_bin),
        str(args.target),
        str(args.target_draft),
        str(prompt_path),
        str(args.target_gen_tokens),
        str(out_path),
    ]
    cmd.extend(TestDflashLaunchArgs(
        target_gpus=args.target_gpus,
        target_layer_split=args.target_layer_split,
        target_split_load_draft=args.target_split_load_draft,
        target_split_dflash=args.target_split_dflash,
        max_ctx=args.target_max_ctx,
    ).to_cli_args())

    t0 = time.perf_counter()
    with log_path.open("wb") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=target_env(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_s = time.perf_counter() - t0
    out_tokens = []
    if proc.returncode == 0 and out_path.exists():
        out_tokens = read_raw_i32(out_path)
    return {
        "target_backend": args.target_backend,
        "target_input_tokens": len(target_ids),
        "target_output_tokens": max(0, len(out_tokens) - len(target_ids)),
        "target_wall_s": wall_s,
        "target_returncode": proc.returncode,
        "target_log": str(log_path),
        "target_prompt_bin": str(prompt_path),
        "target_out_bin": str(out_path),
    }


def make_niah_text(tokenizer, token_count: int, case_idx: int, needle_fraction: float) -> tuple[str, str, str, int]:
    key = f"keymark{case_idx}zeta"
    answer = f"04385{74 + case_idx:02d}"
    intro = "Below is a long passage. Keep important facts from the passage.\n\n"
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    needle = f"The special magic {key} number is {answer}. Remember this exact number. "
    question = f"\n\nQuestion: What is the special magic {key} number?\n"

    def build(reps: int) -> str:
        pos = max(0, min(reps, int(reps * needle_fraction)))
        body = filler * pos + needle + filler * (reps - pos)
        return intro + body + question

    fixed = len(tokenizer.encode(intro + needle + question, add_special_tokens=False))
    filler_tokens = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    lo = 0
    hi = max(8, (max(0, token_count - fixed) // filler_tokens) + 8)
    while len(tokenizer.encode(build(hi), add_special_tokens=False)) < token_count:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(build(mid), add_special_tokens=False)) <= token_count:
            lo = mid
        else:
            hi = mid - 1
    text = build(lo)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, key, answer, actual


def make_pflash_env(args) -> dict[str, str]:
    env = {"DFLASH_FP_ALPHA": str(args.pflash_alpha)}
    if args.pflash_use_bsa:
        env["DFLASH_FP_USE_BSA"] = "1"
    if args.pflash_k_type:
        env["DFLASH_PFLASH_K_TYPE"] = args.pflash_k_type
    return env


def fmt(value, nd: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{nd}f}"


def set_monitor_phase(monitors: list[GpuMonitor], phase: str) -> None:
    for monitor in monitors:
        monitor.set_phase(phase)


def run_cases(args, cases: list[tuple[str, str, str | None, str | None]]) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args)
    target_tokenizer = load_target_tokenizer(args) if args.run_target else None
    pflash_monitor = GpuMonitor(args.report_dir / f"pflash_gpu_monitor_{args.pflash_backend}.csv",
                                args.pflash_backend)
    target_monitor = (
        GpuMonitor(args.report_dir / f"target_gpu_monitor_{args.target_backend}.csv",
                   args.target_backend)
        if args.run_target else None
    )
    monitors = [pflash_monitor] + ([target_monitor] if target_monitor else [])
    daemon = PFlashDaemon(
        binary=args.pflash_bin,
        drafter=args.pflash_drafter,
        gpu=args.pflash_gpu,
        backend=args.pflash_backend,
        visible_devices=args.pflash_visible_devices,
        log_path=args.report_dir / "pflash_daemon.log",
        env=make_pflash_env(args),
    )
    results: list[CompressionCase] = []
    target_failures: list[dict[str, int | str]] = []
    try:
        for monitor in monitors:
            monitor.start()
        set_monitor_phase(monitors, "pflash_load")
        ready_s = daemon.start()
        for name, text, key, answer in cases:
            case_dir = args.report_dir / name
            case_dir.mkdir(parents=True, exist_ok=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            (case_dir / "prompt.txt").write_text(text, encoding="utf-8")
            counted = case_dir / "prompt_counted.bin"
            write_counted_i32(counted, ids)

            set_monitor_phase(monitors, name)
            kept, wall_s = daemon.compress(
                counted,
                keep_ratio=args.keep_ratio,
                lookahead=args.lookahead,
                chunk_size=args.chunk_size,
                pool_kernel=args.pool_kernel,
            )
            if not kept:
                raise RuntimeError(
                    f"PFlash compression returned no tokens; see {args.report_dir / 'pflash_daemon.log'}")
            compressed_text = tokenizer.decode(kept, skip_special_tokens=True)
            (case_dir / "compressed.txt").write_text(compressed_text, encoding="utf-8")
            write_counted_i32(case_dir / "compressed_counted.bin", kept)

            target_result: dict[str, float | int | str] | None = None
            if args.run_target:
                set_monitor_phase(monitors, f"{name}_target")
                assert target_tokenizer is not None
                target_result = run_target_generation(
                    args, case_dir, compressed_text, target_tokenizer)

            target_input_tokens = None
            target_output_tokens = None
            target_wall_s = None
            target_returncode = None
            if target_result is not None:
                target_input_tokens = int(target_result["target_input_tokens"])
                target_output_tokens = int(target_result["target_output_tokens"])
                target_wall_s = float(target_result["target_wall_s"])
                target_returncode = int(target_result["target_returncode"])

            results.append(CompressionCase(
                name=name,
                source_tokens=len(ids),
                compressed_tokens=len(kept),
                compress_wall_s=wall_s,
                compress_tok_s=len(ids) / wall_s if wall_s > 0 else 0.0,
                compression_ratio=(len(kept) / len(ids)) if ids else 0.0,
                retained_key=(key in compressed_text) if key else None,
                retained_answer=(answer in compressed_text) if answer else None,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                target_wall_s=target_wall_s,
                target_returncode=target_returncode,
            ))
            if target_result is not None:
                (case_dir / "target_result.json").write_text(
                    json.dumps(target_result, indent=2), encoding="utf-8")
                if target_returncode != 0:
                    target_failures.append({
                        "case": name,
                        "returncode": target_returncode,
                        "log": str(case_dir / "target.log"),
                    })

        set_monitor_phase(monitors, "cleanup")
        pflash_monitor_devices = parse_device_list(args.pflash_visible_devices,
                                                   [args.pflash_gpu])
        target_monitor_devices = parse_device_list(
            args.target_visible_devices,
            parse_device_list(args.target_gpus, [0]),
        ) if args.run_target else []
        pflash_resource_summary = summarize_devices(
            pflash_monitor, pflash_monitor_devices, target_phase=False)
        target_resource_summary = (
            summarize_devices(target_monitor, target_monitor_devices, target_phase=True)
            if target_monitor else {}
        )
        resource_summary = (
            pflash_resource_summary.get(str(pflash_monitor_devices[0]), {"samples": 0})
            if pflash_monitor_devices else {"samples": 0}
        )
        summary = {
            "date": time.strftime("%Y-%m-%d"),
            "mode": "hybrid_pflash_phase_split" if args.run_target else "pflash_phase_split",
            "pflash_backend": args.pflash_backend,
            "pflash_gpu": args.pflash_gpu,
            "pflash_visible_devices": args.pflash_visible_devices,
            "pflash_daemon_ready_s": ready_s,
            "pflash_drafter": str(args.pflash_drafter),
            "tokenizer": args.tokenizer,
            "keep_ratio": args.keep_ratio,
            "lookahead": args.lookahead,
            "chunk_size": args.chunk_size,
            "pool_kernel": args.pool_kernel,
            "pflash_k_type": args.pflash_k_type or "compute",
            "target": {
                "enabled": bool(args.run_target),
                "backend": args.target_backend,
                "bin": str(args.target_bin) if args.target_bin else None,
                "model": str(args.target),
                "draft": str(args.target_draft) if args.target_draft else None,
                "gpus": args.target_gpus,
                "layer_split": args.target_layer_split,
                "visible_devices": args.target_visible_devices,
                "max_ctx": args.target_max_ctx,
                "gen_tokens": args.target_gen_tokens,
                "tokenizer": args.target_tokenizer,
            },
            "target_failures": target_failures,
            "cases": [asdict(c) for c in results],
            "pflash_monitor_gpus": pflash_monitor_devices,
            "target_monitor_gpus": target_monitor_devices,
            "pflash_resource_summary": pflash_resource_summary,
            "target_resource_summary": target_resource_summary,
            "resource_summary": resource_summary,
            "logs": {
                "pflash": str(args.report_dir / "pflash_daemon.log"),
                "pflash_monitor": str(pflash_monitor.path),
                "target_monitor": str(target_monitor.path) if target_monitor else None,
            },
        }
        (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(args.report_dir / "summary.md", summary)
        print(json.dumps(summary, indent=2))
        if target_failures:
            first = target_failures[0]
            raise RuntimeError(
                f"target generation failed for {len(target_failures)} case(s); "
                f"first={first['case']} rc={first['returncode']} log={first['log']}")
    finally:
        for monitor in monitors:
            monitor.stop()
        daemon.stop()


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Hybrid PFlash Phase-Split Report",
        "",
        f"- PFlash backend: `{summary.get('pflash_backend', 'cuda')}`",
        f"- PFlash GPU: `{summary['pflash_gpu']}`",
        f"- PFlash daemon ready: `{fmt(summary['pflash_daemon_ready_s'])} s`",
        f"- keep ratio: `{summary['keep_ratio']}`",
        f"- lookahead: `{summary['lookahead']}`",
        f"- PFlash K cache: `{summary.get('pflash_k_type', 'compute')}`",
        "",
        "## PFlash Resource Peak",
        "",
        "| gpu | samples | peak mem MiB/% | peak temp C | avg power W | peak power W | avg util % | peak util % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def append_resource_rows(resources: dict) -> None:
        if not resources:
            lines.append("| n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            return
        for gpu, res in resources.items():
            lines.append(
                "| {gpu} | {samples} | {mem} | {temp} | {pavg} | {pmax} | {uavg} | {umax} |".format(
                    gpu=gpu,
                    samples=res.get("samples", 0),
                    mem=fmt(res.get("mem_max_mib")),
                    temp=fmt(res.get("temp_max_c")),
                    pavg=fmt(res.get("power_avg_w")),
                    pmax=fmt(res.get("power_max_w")),
                    uavg=fmt(res.get("util_avg_pct")),
                    umax=fmt(res.get("util_max_pct")),
                )
            )

    append_resource_rows(summary.get("pflash_resource_summary") or {
        str(summary["pflash_gpu"]): summary.get("resource_summary") or {}
    })
    if summary.get("target", {}).get("enabled"):
        lines.extend([
            "",
            "## Target Resource Peak",
            "",
            "| gpu | samples | peak mem MiB/% | peak temp C | avg power W | peak power W | avg util % | peak util % |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        append_resource_rows(summary.get("target_resource_summary") or {})
    lines.extend([
        "",
        "## Cases",
        "",
        "| case | source tokens | compressed tokens | ratio | PFlash s | PFlash tok/s | target input | target output | target s | target rc | key retained | answer retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ])
    for case in summary["cases"]:
        key = case.get("retained_key")
        answer = case.get("retained_answer")
        lines.append(
            "| {name} | {source} | {compressed} | {ratio} | {secs} | {tps} | {tin} | {tout} | {twall} | {trc} | {key} | {answer} |".format(
                name=case["name"],
                source=case["source_tokens"],
                compressed=case["compressed_tokens"],
                ratio=fmt(case["compression_ratio"], 4),
                secs=fmt(case["compress_wall_s"]),
                tps=fmt(case["compress_tok_s"]),
                tin=case.get("target_input_tokens") if case.get("target_input_tokens") is not None else "n/a",
                tout=case.get("target_output_tokens") if case.get("target_output_tokens") is not None else "n/a",
                twall=fmt(case.get("target_wall_s")) if case.get("target_wall_s") is not None else "n/a",
                trc=case.get("target_returncode") if case.get("target_returncode") is not None else "n/a",
                key="n/a" if key is None else ("yes" if key else "no"),
                answer="n/a" if answer is None else ("yes" if answer else "no"),
            )
        )
    lines.extend([
        "",
        "Files:",
    ])
    for key, value in summary["logs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prompt_text(args) -> str:
    if args.prompt and args.prompt_file:
        raise SystemExit("use only one of --prompt or --prompt-file")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --prompt, --prompt-file, or pipe prompt text on stdin")


def run_prompt(args) -> None:
    text = read_prompt_text(args)
    if not text.strip():
        raise SystemExit("prompt is empty")
    run_cases(args, [("prompt", text, None, None)])


def run_bench_niah(args) -> None:
    tokenizer = load_tokenizer(args)
    cases: list[tuple[str, str, str | None, str | None]] = []
    for idx, value in enumerate(x for x in args.contexts.split(",") if x.strip()):
        requested = int(value)
        text, key, answer, actual = make_niah_text(tokenizer, requested, idx, args.needle_fraction)
        cases.append((f"niah_ctx{actual}", text, key, answer))
    run_cases(args, cases)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--pflash-bin", type=Path, default=None)
    ap.add_argument("--pflash-drafter", type=Path, default=DEFAULT_DRAFTER)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--pflash-backend", choices=["cuda", "hip"], default="cuda")
    ap.add_argument("--pflash-gpu", type=int, default=0)
    ap.add_argument("--pflash-visible-devices", default=None,
                    help="Visible device list for pflash_daemon. Defaults to --pflash-gpu.")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--pool-kernel", type=int, default=13)
    ap.add_argument("--pflash-alpha", type=float, default=0.99)
    ap.add_argument("--pflash-use-bsa", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pflash-k-type", default=None,
                    choices=["f16", "bf16", "q8_0", "q4_0", "q4_1"],
                    help="persistent PFlash drafter K cache type; default follows drafter compute type")
    ap.add_argument("--report-dir", type=Path, default=Path("reports/pflash_phase_split"))
    ap.add_argument("--run-target", action="store_true",
                    help="After PFlash compression, run test_dflash target generation.")
    ap.add_argument("--target-bin", type=Path, default=None,
                    help="Path to the target test_dflash binary. Required with --run-target.")
    ap.add_argument("--target-backend", choices=["cuda", "hip"], default="cuda")
    ap.add_argument("--target-visible-devices", default=None,
                    help="Visible device list for the target binary, e.g. 0,1.")
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--target-draft", type=Path, default=DEFAULT_TARGET_DRAFT,
                    help="DFlash draft path required by test_dflash, file or directory.")
    ap.add_argument("--target-tokenizer", default=DEFAULT_TARGET_TOKENIZER)
    ap.add_argument("--target-gpus", default=None,
                    help="Comma-separated target device ids passed to --target-gpus.")
    ap.add_argument("--target-layer-split", default=None,
                    help="Comma-separated layer split weights for target layer split.")
    ap.add_argument("--target-max-ctx", type=int, default=4096)
    ap.add_argument("--target-gen-tokens", type=int, default=32)
    ap.add_argument("--target-env", action="append", default=[],
                    help="Extra NAME=VALUE environment entries for target generation.")
    ap.add_argument("--target-split-load-draft", action="store_true",
                    help="Also load DFlash draft in the target split harness for regression.")
    ap.add_argument("--target-split-dflash", action="store_true",
                    help="Run DFlash target-split decode instead of AR target decode.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    prompt = sub.add_parser("run-prompt", help="run a prompt through the PFlash phase")
    add_common_args(prompt)
    prompt.add_argument("--prompt", default=None)
    prompt.add_argument("--prompt-file", type=Path, default=None)
    prompt.set_defaults(func=run_prompt)

    bench = sub.add_parser("bench-niah", help="compress synthetic NIAH prompts")
    add_common_args(bench)
    bench.add_argument("--contexts", default="4096,8192,16384")
    bench.add_argument("--needle-fraction", type=float, default=0.5)
    bench.set_defaults(func=run_bench_niah)

    args = ap.parse_args()
    args.pflash_bin = args.pflash_bin or (args.build_dir / "pflash_daemon")
    for path in (args.pflash_bin, args.pflash_drafter):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")
    args.build_dir = args.build_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    args.pflash_bin = args.pflash_bin.resolve()
    args.pflash_drafter = args.pflash_drafter.resolve()
    if args.run_target:
        if args.target_bin is None:
            args.target_bin = args.build_dir / "test_dflash"
        if not args.target_bin.exists():
            raise SystemExit(f"missing required path: {args.target_bin}")
        if not args.target.exists():
            raise SystemExit(f"missing required path: {args.target}")
        args.target_draft = resolve_target_draft(args.target_draft)
        args.target_bin = args.target_bin.resolve()
        args.target = args.target.resolve()
        args.target_draft = args.target_draft.resolve()
        if args.target_gen_tokens <= 0:
            raise SystemExit("--target-gen-tokens must be > 0")
        if args.target_split_dflash:
            args.target_split_load_draft = True
    args.func(args)


if __name__ == "__main__":
    main()
