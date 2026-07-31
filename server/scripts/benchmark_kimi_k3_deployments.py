#!/usr/bin/env python3
"""Reproducible Kimi K3 SSD deployment comparison.

The harness starts one server at a time, runs deterministic cold/warm requests,
captures the complete server log (including MoE NVMe shutdown telemetry), and
writes machine-readable JSON.  It deliberately disables HTTP/prefix caches so
the only warm state under test is the routed-expert device cache.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_SPLIT_GGUF = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)-of-(?P<total>\d+)(?P<suffix>\.gguf)$")
_NVME_TELEMETRY = re.compile(
    r"\[moe-nvme\] io=(?P<io>\S+) requests=(?P<requests>\d+) reads=(?P<reads>\d+) "
    r"payload=(?P<payload_gib>[\d.]+) GiB physical=(?P<physical_gib>[\d.]+) GiB "
    r"active-io-rate=(?P<active_io_gib_s>[\d.]+) GiB/s cache-hit=(?P<cache_hit_pct>[\d.]+)% "
    r"mean-demand-wait=(?P<mean_demand_wait_ms>[\d.]+) ms .*?timeouts=(?P<timeouts>\d+) "
    r"errors=(?P<errors>\d+) device-cache=(?P<device_cache_mib>[\d.]+) MiB "
    r"slots=(?P<slots>\d+) hits=(?P<device_hits>\d+) misses=(?P<device_misses>\d+) "
    r"evictions=(?P<device_evictions>\d+) graphs=(?P<graphs>\d+) "
    r"graph-hits=(?P<graph_hits>\d+) graph-evictions=(?P<graph_evictions>\d+) "
    r"launches=(?P<launches>\d+)"
)


@dataclass(frozen=True)
class Deployment:
    name: str
    primary_device: int
    secondary_device: int | None


def build_deployments(
    profiles: list[str],
    strix_device: int,
    r9700_device: int,
    hetero_primary: str,
) -> list[Deployment]:
    if strix_device < 0 or r9700_device < 0:
        raise ValueError("GPU device indices must be non-negative")
    if strix_device == r9700_device and "heterogeneous" in profiles:
        raise ValueError("heterogeneous mode requires distinct R9700 and Strix devices")

    deployments: list[Deployment] = []
    for profile in profiles:
        if profile == "strix-only":
            deployments.append(Deployment("strix-only-ssd", strix_device, None))
        elif profile == "heterogeneous":
            if hetero_primary == "strix":
                deployments.append(
                    Deployment("heterogeneous-ssd", strix_device, r9700_device)
                )
            else:
                deployments.append(
                    Deployment("heterogeneous-ssd", r9700_device, strix_device)
                )
        else:
            raise ValueError(f"unknown deployment profile: {profile}")
    return deployments


def deployment_environment(
    base: dict[str, str],
    deployment: Deployment,
    nvme_backend: str,
    primary_share_per_mille: int,
    placement: Path | None,
    device_cache_mb: int | None,
    dual_trace: bool,
) -> dict[str, str]:
    """Return a clean MoE environment without inherited benchmark tuning."""
    env = dict(base)
    for key in list(env):
        if key.startswith("DFLASH_MOE_NVME_") or key in {
            "DFLASH_MOE_STORAGE",
            "DFLASH_MOE_TP_GPU",
            "DFLASH_MOE_PLACEMENT",
            "DFLASH_MOE_PRIMARY_SHARE_PER_MILLE",
            "DFLASH_MOE_DUAL_STREAM_TRACE",
        }:
            env.pop(key)

    env["DFLASH_MOE_NVME_BACKEND"] = nvme_backend
    if device_cache_mb is not None:
        env["DFLASH_MOE_NVME_DEVICE_CACHE_MB"] = str(device_cache_mb)
    if deployment.secondary_device is not None:
        env["DFLASH_MOE_TP_GPU"] = str(deployment.secondary_device)
        env["DFLASH_MOE_PRIMARY_SHARE_PER_MILLE"] = str(primary_share_per_mille)
        if placement is not None:
            env["DFLASH_MOE_PLACEMENT"] = str(placement)
        if dual_trace:
            env["DFLASH_MOE_DUAL_STREAM_TRACE"] = "1"
    return env


def server_command(
    server_bin: Path,
    model: Path,
    deployment: Deployment,
    port: int,
    max_ctx: int,
    extra_server_args: list[str],
) -> list[str]:
    return [
        str(server_bin),
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--target-device",
        f"hip:{deployment.primary_device}",
        "--max-ctx",
        str(max_ctx),
        "--moe-storage",
        "ssd",
        "--prefix-cache-slots",
        "0",
        "--prefill-cache-slots",
        "0",
        "--disk-prefix-cache",
        "off",
        *extra_server_args,
    ]


def discover_model_files(first_shard: Path) -> list[Path]:
    """Validate and return a complete split GGUF in shard order."""
    match = _SPLIT_GGUF.match(first_shard.name)
    if match is None:
        if not first_shard.is_file():
            raise FileNotFoundError(first_shard)
        return [first_shard]

    if int(match.group("index")) != 1:
        raise ValueError(f"expected the first split GGUF shard, got {first_shard.name}")

    width = len(match.group("index"))
    total_text = match.group("total")
    total = int(total_text)
    expected = [
        first_shard.with_name(
            f"{match.group('prefix')}-{index:0{width}d}-of-{total_text}{match.group('suffix')}"
        )
        for index in range(1, total + 1)
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        sample = ", ".join(path.name for path in missing[:3])
        raise FileNotFoundError(
            f"split GGUF is incomplete: missing {len(missing)}/{total} shard(s): {sample}"
        )
    return expected


def extract_nvme_telemetry(log_path: Path) -> list[dict[str, Any]]:
    telemetry: list[dict[str, Any]] = []
    if not log_path.exists():
        return telemetry
    for match in _NVME_TELEMETRY.finditer(log_path.read_text(errors="replace")):
        row: dict[str, Any] = {"io": match.group("io")}
        for key, value in match.groupdict().items():
            if key == "io":
                continue
            row[key] = float(value) if "." in value else int(value)
        telemetry.append(row)
    return telemetry


def ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"benchmark port {port} is already in use") from exc


def wait_until_ready(
    process: subprocess.Popen[bytes],
    base_url: str,
    timeout_s: float,
) -> float:
    started = time.perf_counter()
    deadline = started + timeout_s
    next_report = started + 30.0
    while time.perf_counter() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited during startup with status {return_code}")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as response:
                if 200 <= response.status < 300:
                    return time.perf_counter() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        now = time.perf_counter()
        if now >= next_report:
            print(f"  still loading model ({now - started:.0f}s)", flush=True)
            next_report = now + 30.0
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s")


def stream_chat(
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    chunks = 0
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            encoded = line[5:].strip()
            if encoded == "[DONE]":
                break
            try:
                event = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            visible = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if not visible and not thought:
                continue
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            chunks += 1
            content.append(visible)
            reasoning.append(thought)

    finished = time.perf_counter()
    wall_s = finished - started
    client_ttft_s = (first_token_at - started) if first_token_at is not None else wall_s
    completion_tokens = int(usage.get("completion_tokens") or chunks)
    event_decode_s = (
        last_token_at - first_token_at
        if first_token_at is not None and last_token_at is not None
        else 0.0
    )
    timings = usage.get("timings") or {}
    server_prefill_s = float(timings.get("prefill_ms") or 0.0) / 1000.0
    server_decode_s = float(timings.get("decode_ms") or 0.0) / 1000.0
    server_decode_tok_s = float(timings.get("decode_tokens_per_sec") or 0.0)
    if server_decode_tok_s <= 0.0:
        server_decode_tok_s = (
            (completion_tokens - 1) / event_decode_s
            if completion_tokens > 1 and event_decode_s > 0
            else 0.0
        )
    wall_tok_s = completion_tokens / wall_s if wall_s > 0 else 0.0
    event_decode_tok_s = (
        (completion_tokens - 1) / event_decode_s
        if completion_tokens > 1 and event_decode_s > 0
        else 0.0
    )
    return {
        "wall_s": wall_s,
        "wall_tok_s": wall_tok_s,
        "client_ttft_s": client_ttft_s,
        "server_prefill_s": server_prefill_s,
        "server_decode_s": server_decode_s,
        "server_decode_tok_s": server_decode_tok_s,
        "event_decode_s": event_decode_s,
        "event_decode_tok_s": event_decode_tok_s,
        "completion_tokens": completion_tokens,
        "event_chunks": chunks,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
    }


def stop_server(process: subprocess.Popen[bytes], timeout_s: float = 60.0) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait()


def selected_environment(env: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith("DFLASH_MOE_")
    }


def run_deployment(
    args: argparse.Namespace,
    deployment: Deployment,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_port_available(args.port)
    env = deployment_environment(
        os.environ,
        deployment,
        args.nvme_backend,
        args.primary_share_per_mille,
        args.placement,
        args.device_cache_mb,
        args.dual_trace,
    )
    command = server_command(
        args.server_bin,
        args.model,
        deployment,
        args.port,
        args.max_ctx,
        args.extra_server_arg,
    )
    log_path = output_dir / f"{deployment.name}.server.log"
    result: dict[str, Any] = {
        "deployment": asdict(deployment),
        "command": command,
        "environment": selected_environment(env),
        "server_log": str(log_path),
        "requests": [],
    }
    print(
        f"\n[{deployment.name}] primary=hip:{deployment.primary_device} "
        f"secondary={deployment.secondary_device}",
        flush=True,
    )
    process: subprocess.Popen[bytes] | None = None
    with log_path.open("wb") as log_file:
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            result["startup_s"] = wait_until_ready(
                process, f"http://127.0.0.1:{args.port}", args.startup_timeout
            )
            print(f"  server ready in {result['startup_s']:.1f}s", flush=True)
            for iteration in range(args.iterations):
                measurement = stream_chat(
                    f"http://127.0.0.1:{args.port}",
                    args.prompt,
                    args.max_tokens,
                    args.request_timeout,
                )
                measurement["state"] = "cold" if iteration == 0 else "warm"
                result["requests"].append(measurement)
                print(
                    f"  {measurement['state']}[{iteration}] "
                    f"wall={measurement['wall_s']:.3f}s "
                    f"prefill={measurement['server_prefill_s']:.3f}s "
                    f"decode={measurement['server_decode_tok_s']:.3f} tok/s",
                    flush=True,
                )
        except Exception as exc:  # Preserve the other profile and its evidence.
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {result['error']}", file=sys.stderr, flush=True)
        finally:
            if process is not None:
                result["server_exit_code"] = stop_server(process)
    result["nvme_telemetry"] = extract_nvme_telemetry(log_path)
    outputs = [
        row["reasoning_content"] + row["content"] for row in result["requests"]
    ]
    result["within_profile_output_match"] = bool(outputs) and len(set(outputs)) == 1
    (output_dir / f"{deployment.name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Kimi K3 Strix-only and heterogeneous SSD deployments"
    )
    parser.add_argument("model", type=Path, help="first shard of the split Kimi K3 GGUF")
    parser.add_argument(
        "--server-bin", type=Path, default=Path("server/build-hip-dual/dflash_server")
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("strix-only", "heterogeneous"),
        default=("strix-only", "heterogeneous"),
    )
    parser.add_argument("--strix-device", type=int, default=1)
    parser.add_argument("--r9700-device", type=int, default=0)
    parser.add_argument(
        "--hetero-primary",
        choices=("strix", "r9700"),
        default="strix",
        help=(
            "primary device for the heterogeneous run; current IQ1_S defaults to Strix "
            "because its non-routed tensors exceed R9700 VRAM"
        ),
    )
    parser.add_argument("--primary-share-per-mille", type=int, default=500)
    parser.add_argument("--placement", type=Path)
    parser.add_argument("--device-cache-mb", type=int)
    parser.add_argument(
        "--nvme-backend", choices=("auto", "uring", "pread", "mmap"), default="auto"
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--prompt",
        default="Explain in two short sentences why the sky appears blue.",
    )
    parser.add_argument("--dual-trace", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="print resolved commands without starting servers"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.max_ctx <= 0 or args.max_tokens <= 0 or args.iterations <= 0:
        raise ValueError("max-ctx, max-tokens, and iterations must be positive")
    if not 0 <= args.primary_share_per_mille <= 1000:
        raise ValueError("primary-share-per-mille must be in [0, 1000]")
    if args.device_cache_mb is not None and args.device_cache_mb < 0:
        raise ValueError("device-cache-mb must be non-negative")
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("deployment profiles must not be repeated")
    if args.placement is not None and not args.dry_run and not args.placement.is_file():
        raise FileNotFoundError(args.placement)


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        deployments = build_deployments(
            list(args.profiles),
            args.strix_device,
            args.r9700_device,
            args.hetero_primary,
        )
        if args.dry_run:
            resolved = []
            for deployment in deployments:
                env = deployment_environment(
                    os.environ,
                    deployment,
                    args.nvme_backend,
                    args.primary_share_per_mille,
                    args.placement,
                    args.device_cache_mb,
                    args.dual_trace,
                )
                resolved.append(
                    {
                        "deployment": asdict(deployment),
                        "command": server_command(
                            args.server_bin,
                            args.model,
                            deployment,
                            args.port,
                            args.max_ctx,
                            args.extra_server_arg,
                        ),
                        "environment": selected_environment(env),
                    }
                )
            print(json.dumps(resolved, indent=2, sort_keys=True))
            return 0

        if not args.server_bin.is_file():
            raise FileNotFoundError(args.server_bin)
        model_files = discover_model_files(args.model)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("bench-out") / f"kimi-k3-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "model_first_shard": str(args.model),
        "model_shards": len(model_files),
        "model_bytes": sum(path.stat().st_size for path in model_files),
        "results": [],
    }
    print(f"Writing benchmark evidence to {output_dir}", flush=True)
    for deployment in deployments:
        manifest["results"].append(run_deployment(args, deployment, output_dir))

    successful = [row for row in manifest["results"] if row.get("requests")]
    outputs = [
        row["requests"][0]["reasoning_content"] + row["requests"][0]["content"]
        for row in successful
    ]
    manifest["cross_profile_output_match"] = bool(outputs) and len(set(outputs)) == 1
    manifest_path = output_dir / "comparison.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nComparison: {manifest_path}")

    if any("error" in row for row in manifest["results"]):
        return 1
    deterministic = all(row["within_profile_output_match"] for row in successful)
    if len(successful) > 1:
        deterministic = deterministic and manifest["cross_profile_output_match"]
    if not deterministic:
        print("ERROR: deterministic outputs differ within or across profiles", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
