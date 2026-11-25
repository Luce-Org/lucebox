#!/usr/bin/env python3
"""Lucebox Golden Concurrency Gate CLI (luce-concurrency-gate).

Stdlib-only orchestrator implementing the 7 mandatory concurrency invariants
(I1-I7), 4-tier memory HWM sampling, and canonical scorecard generation for
both `lucebox` engine CI and `luce_box` Manage/release qualification.

Exit codes:
  0 = PASS
  1 = INVARIANT_FAIL (correctness, isolation, tool validity, 5xx, leaks)
  2 = SLO_OR_MEM_FAIL (cache hit rate, TTFT ratio, SLO regression, memory HWM)
  3 = INFRA_INVALID (bad pack/profile, server boot/health failure)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXIT_PASS = 0
EXIT_INVARIANT_FAIL = 1
EXIT_SLO_OR_MEM_FAIL = 2
EXIT_INFRA_INVALID = 3
DRAIN_TIMEOUT_S = 5.0

CONCURRENCY_METRICS_RE = re.compile(r"\[concurrency-metrics\]\s+(\{.*\})")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(obj)
    if not records:
        raise ValueError(f"{path}: empty JSONL workload")
    return records


def pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def normalize_urls(base_url: str) -> tuple[str, str]:
    """Return (api_base_v1, server_root)."""
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        return clean, clean[:-3]
    return clean + "/v1", clean


def http_get_json(url: str, timeout: float = 5.0) -> tuple[int | None, dict[str, Any] | None]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return None, None


def git_head_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        sha = out.decode("utf-8", errors="replace").strip()
        return sha if len(sha) == 40 else None
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


@dataclass
class StreamResult:
    ok: bool = False
    http_status: int | None = None
    error: str | None = None
    response_id: str | None = None
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float | None = None
    wall_ms: float = 0.0
    itls_ms: list[float] = field(default_factory=list)


def iter_sse_data(lines: Any) -> Any:
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buf:
                yield "\n".join(buf)
                buf.clear()
        elif line.startswith("data:"):
            buf.append(line[5:].lstrip())
    if buf:
        yield "\n".join(buf)


def stream_chat_completion(
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    api_key: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    seed: int = 42,
) -> StreamResult:
    res = StreamResult()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    t_prev: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_map: dict[int, dict[str, Any]] = {}
    done_seen = False

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res.http_status = resp.status
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" not in ctype and "application/json" in ctype:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                t_now = time.perf_counter()
                res.ttft_ms = (t_now - t0) * 1000.0
                res.wall_ms = res.ttft_ms
                res.response_id = body.get("id")
                choices = body.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    res.content = msg.get("content") or ""
                    res.reasoning_content = msg.get("reasoning_content") or ""
                    res.tool_calls = msg.get("tool_calls") or []
                    res.finish_reason = choices[0].get("finish_reason")
                _apply_usage(res, body.get("usage"))
                res.ok = res.http_status == 200
                return res

            for data in iter_sse_data(resp):
                if data == "[DONE]":
                    done_seen = True
                    break
                event = json.loads(data)
                if isinstance(event.get("id"), str):
                    res.response_id = event["id"]
                _apply_usage(res, event.get("usage"))
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        res.finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    emitted = False
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        content_parts.append(piece)
                        emitted = True
                    thought = delta.get("reasoning_content")
                    if isinstance(thought, str) and thought:
                        reasoning_parts.append(thought)
                        emitted = True
                    for tc in delta.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        idx = int(tc.get("index", 0))
                        entry = tc_map.setdefault(
                            idx,
                            {"id": tc.get("id") or f"call_{idx}", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if isinstance(fn.get("name"), str) and fn["name"]:
                            entry["function"]["name"] += fn["name"]
                        if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                            entry["function"]["arguments"] += fn["arguments"]
                        emitted = True
                    if emitted:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        elif t_prev is not None:
                            res.itls_ms.append((now - t_prev) * 1000.0)
                        t_prev = now
    except urllib.error.HTTPError as exc:
        res.http_status = exc.code
        try:
            body_preview = exc.read(512).decode("utf-8", errors="replace").strip()
        except Exception:
            body_preview = ""
        res.error = f"HTTPError {exc.code}: {body_preview or exc.reason}"
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"

    t_end = time.perf_counter()
    res.wall_ms = (t_end - t0) * 1000.0
    if t_first is not None:
        res.ttft_ms = (t_first - t0) * 1000.0
    res.content = "".join(content_parts)
    res.reasoning_content = "".join(reasoning_parts)
    res.tool_calls = [tc_map[k] for k in sorted(tc_map)]

    if res.error is None:
        if not done_seen:
            res.error = "ProtocolError: stream ended before [DONE]"
        elif res.finish_reason is None:
            res.error = "ProtocolError: stream ended without finish_reason"
        else:
            res.ok = True
    return res


def _apply_usage(res: StreamResult, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    pt = _int_or_none(usage.get("prompt_tokens"))
    if pt is not None:
        res.prompt_tokens = pt
    ct = _int_or_none(usage.get("completion_tokens"))
    if ct is not None:
        res.completion_tokens = ct
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        cached = _int_or_none(ptd.get("cached_tokens"))
        if cached is not None:
            res.cached_tokens = cached
    if res.cached_tokens is None:
        direct_cached = _int_or_none(usage.get("cached_tokens"))
        if direct_cached is not None:
            res.cached_tokens = direct_cached
    timings = usage.get("timings")
    if isinstance(timings, dict):
        res.timings.update(timings)
        if res.cached_tokens is None:
            tc = _int_or_none(timings.get("cached_prefix_tokens"))
            if tc is not None:
                res.cached_tokens = tc


def _open_raw_http_post(
    api_base: str, payload: dict[str, Any], api_key: str | None, timeout: float
) -> socket.socket:
    parsed = urllib.parse.urlparse(api_base.rstrip("/") + "/chat/completions")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    hdrs = [
        f"POST {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    if api_key:
        hdrs.append(f"Authorization: Bearer {api_key}")
    raw_req = ("\r\n".join(hdrs) + "\r\n\r\n").encode("utf-8") + body
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(raw_req)
    return sock


def _abort_socket_rst(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def inject_cancel_mid_prefill(
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    api_key: str | None,
    timeout: float,
) -> bool:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        sock = _open_raw_http_post(api_base, payload, api_key, timeout)
        time.sleep(0.005)
        _abort_socket_rst(sock)
        return True
    except Exception:
        return False


def inject_cancel_mid_decode(
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    cancel_after_tokens: int,
    api_key: str | None,
    timeout: float,
) -> bool:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max(max_tokens, cancel_after_tokens + 16),
        "temperature": 0.0,
    }
    try:
        sock = _open_raw_http_post(api_base, payload, api_key, timeout)
        seen_data = 0
        buf = b""
        deadline = time.perf_counter() + min(timeout, 15.0)
        while seen_data < cancel_after_tokens and time.perf_counter() < deadline:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    seen_data += 1
                    if seen_data >= cancel_after_tokens:
                        break
        _abort_socket_rst(sock)
        return True
    except Exception:
        return False


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def validate_tool_call(
    result: StreamResult,
    expected_name: str,
    required_args: list[str],
    arg_types: dict[str, str] | None = None,
    expected_arg_values: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if not result.ok:
        return False, result.error or "request failed"

    call_name: str | None = None
    raw_args: Any = None

    if result.tool_calls:
        fn = (result.tool_calls[0] or {}).get("function") or {}
        call_name = fn.get("name")
        raw_args = fn.get("arguments")
    elif result.content.strip():
        # Fallback if a backend emits a JSON tool call object inside content
        try:
            parsed_content = json.loads(result.content.strip())
            if isinstance(parsed_content, dict) and "name" in parsed_content:
                call_name = parsed_content.get("name")
                raw_args = parsed_content.get("arguments", {})
        except json.JSONDecodeError:
            pass

    if call_name != expected_name:
        return False, f"expected tool {expected_name!r}, got {call_name!r}"

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return False, f"invalid tool_call JSON: {exc}"
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return False, "missing tool_call arguments"

    if not isinstance(args, dict):
        return False, f"tool_call arguments must be JSON object, got {type(args).__name__}"

    for key in required_args:
        if key not in args:
            return False, f"missing required argument {key!r}"

    for key, expected_type in (arg_types or {}).items():
        if key in args and not _json_type_matches(args[key], expected_type):
            return False, f"argument {key!r} expected type {expected_type}, got {type(args[key]).__name__}"

    for key, expected_val in (expected_arg_values or {}).items():
        if args.get(key) != expected_val:
            return False, f"argument {key!r} expected {expected_val!r}, got {args.get(key)!r}"

    return True, "ok"


class MemorySampler:
    """4-tier HWM memory sampler (VRAM, UMA/GTT, Host RAM, SSD disk cache).

    Gracefully returns `None` for unavailable GPU/disk tiers on CPU-only hosts.
    """

    def __init__(self, disk_cache_dir: Path | None = None, interval_s: float = 0.25):
        self.disk_cache_dir = disk_cache_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.vram_peak_mib: float | None = None
        self.uma_peak_mib: float | None = None
        self.host_ram_peak_mib: float | None = None
        self.disk_cache_peak_gib: float | None = None

    @staticmethod
    def read_host_ram_used_mib() -> float | None:
        meminfo = Path("/proc/meminfo")
        if not meminfo.is_file():
            return None
        try:
            total_kb = avail_kb = None
            for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            if total_kb is not None and avail_kb is not None:
                return round(max(0, total_kb - avail_kb) / 1024.0, 1)
        except Exception:
            return None
        return None

    @staticmethod
    def read_amd_gpu_mem_mib() -> tuple[float | None, float | None]:
        """Read (discrete_vram_mib, uma_gtt_mib) from AMD DRM sysfs if present."""
        drm = Path("/sys/class/drm")
        if not drm.is_dir():
            return None, None
        vram_vals: list[float] = []
        gtt_vals: list[float] = []
        try:
            for card in sorted(drm.glob("card[0-9]*")):
                dev = card / "device"
                vram_file = dev / "mem_info_vram_used"
                gtt_file = dev / "mem_info_gtt_used"
                if vram_file.is_file():
                    val = int(vram_file.read_text().strip())
                    vram_vals.append(round(val / (1024.0 * 1024.0), 1))
                if gtt_file.is_file():
                    val = int(gtt_file.read_text().strip())
                    gtt_vals.append(round(val / (1024.0 * 1024.0), 1))
        except Exception:
            return None, None
        return (max(vram_vals) if vram_vals else None), (max(gtt_vals) if gtt_vals else None)

    @staticmethod
    def read_disk_cache_gib(cache_dir: Path | None) -> float | None:
        if cache_dir is None or not cache_dir.is_dir():
            return None
        total = 0
        try:
            for root_dir, _, files in os.walk(cache_dir):
                for name in files:
                    try:
                        total += (Path(root_dir) / name).stat().st_size
                    except OSError:
                        pass
            return round(total / (1024.0 ** 3), 3)
        except Exception:
            return None

    def sample_once(self) -> dict[str, float | None]:
        vram, uma = self.read_amd_gpu_mem_mib()
        host_ram = self.read_host_ram_used_mib()
        disk_gib = self.read_disk_cache_gib(self.disk_cache_dir)
        with self._lock:
            if vram is not None:
                self.vram_peak_mib = max(self.vram_peak_mib or 0.0, vram)
            if uma is not None:
                self.uma_peak_mib = max(self.uma_peak_mib or 0.0, uma)
            if host_ram is not None:
                self.host_ram_peak_mib = max(self.host_ram_peak_mib or 0.0, host_ram)
            if disk_gib is not None:
                self.disk_cache_peak_gib = max(self.disk_cache_peak_gib or 0.0, disk_gib)
        return {
            "vram_mib": vram,
            "uma_mib": uma,
            "host_ram_mib": host_ram,
            "disk_cache_gib": disk_gib,
        }

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.sample_once()
        with self._lock:
            return {
                "r9700_vram_peak_mib": self.vram_peak_mib,
                "strix_uma_peak_mib": self.uma_peak_mib,
                "host_ram_peak_mib": self.host_ram_peak_mib,
                "ssd_prefix_cache_peak_gib": self.disk_cache_peak_gib,
            }


def parse_concurrency_metrics_log(log_path: Path | None) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if log_path is None or not log_path.is_file():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CONCURRENCY_METRICS_RE.search(line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rid = obj.get("response_id") or obj.get("request_id")
        if isinstance(rid, str) and rid:
            metrics[rid] = obj
    return metrics


class ManagedServer:
    """Optional subprocess wrapper for --server-bin or --launcher."""

    def __init__(
        self,
        cmd: list[str],
        env: dict[str, str],
        log_path: Path,
        server_root: str,
        startup_timeout: float,
    ):
        self.cmd = cmd
        self.env = env
        self.log_path = log_path
        self.server_root = server_root
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None

    def start(self) -> tuple[bool, str]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("wb")
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                env=self.env,
            )
        except Exception as exc:
            self._log_fh.close()
            return False, f"failed to spawn server: {exc}"

        deadline = time.perf_counter() + self.startup_timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                return False, f"server exited early with code {self.proc.returncode}"
            status, _ = http_get_json(self.server_root + "/health", timeout=2.0)
            if status == 200:
                return True, "ok"
            time.sleep(0.25)
        return False, f"server did not become healthy within {self.startup_timeout}s"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass


def resolve_concurrency_levels(
    tier_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    override: str | None,
) -> tuple[list[int], list[dict[str, int]]]:
    cap = int(model_cfg["max_supported_concurrency"])
    if override:
        raw_levels = [int(x.strip()) for x in override.split(",") if x.strip()]
        if not raw_levels or any(c <= 0 for c in raw_levels):
            raise ValueError("--concurrency-override must contain positive integers")
        return raw_levels, []

    resolved: list[int] = []
    clamped: list[dict[str, int]] = []
    for c in tier_cfg.get("concurrency_levels", [3, 4]):
        c_int = int(c)
        eff = min(c_int, cap)
        if eff != c_int:
            clamped.append({"requested": c_int, "clamped_to": eff})
        if eff not in resolved:
            resolved.append(eff)
    return resolved, clamped


def run_parallel_barrier(fn_list: list[Any]) -> list[Any]:
    n = len(fn_list)
    if n == 0:
        return []
    barrier = threading.Barrier(n)

    def _wrapped(fn: Any) -> Any:
        try:
            barrier.wait(timeout=30.0)
        except threading.BrokenBarrierError:
            pass
        return fn()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(_wrapped, fn) for fn in fn_list]
        return [f.result() for f in futs]


def summarize_slo(results: list[StreamResult], wall_s: float) -> dict[str, float | None]:
    ttfts = [r.ttft_ms for r in results if r.ok and r.ttft_ms is not None]
    itls = [x for r in results if r.ok for x in r.itls_ms]
    tpots = []
    total_comp = 0
    for r in results:
        if not r.ok:
            continue
        ct = r.completion_tokens if r.completion_tokens is not None else max(1, len(r.itls_ms) + 1)
        total_comp += ct
        if r.ttft_ms is not None and ct > 1 and r.wall_ms > r.ttft_ms:
            tpots.append((r.wall_ms - r.ttft_ms) / (ct - 1))
    goodput = round(total_comp / wall_s, 2) if wall_s > 0 else 0.0
    return {
        "goodput_tok_s": goodput,
        "ttft_p50_ms": round(percentile(ttfts, 50), 2) if ttfts else None,
        "ttft_p95_ms": round(percentile(ttfts, 95), 2) if ttfts else None,
        "itl_p50_ms": round(percentile(itls, 50), 2) if itls else None,
        "itl_p99_ms": round(percentile(itls, 99), 2) if itls else None,
        "tpot_p50_ms": round(percentile(tpots, 50), 2) if tpots else None,
        "wall_s": round(wall_s, 3),
    }


def render_scorecard_markdown(sc: dict[str, Any]) -> str:
    lines = [
        f"# Golden Concurrency Gate Scorecard — `{sc['verdict']}`",
        "",
        f"- **Pack**: `{sc['pack_id']}` (`{sc['pack_sha256'][:12]}`)",
        f"- **Tier**: `{sc['tier']}` | **Model Profile**: `{sc['model_profile']}`",
        f"- **Exit Code**: `{sc['exit_code']}` | **Concurrency Levels**: `{sc['concurrency_levels']}`",
        "",
        "## Mandatory Invariants (I1–I7)",
        "",
        "| Invariant | Pass | Key Metrics |",
        "| :--- | :---: | :--- |",
    ]
    for name, data in sc["invariants"].items():
        status = "PASS" if data.get("pass") else "FAIL"
        summary = ", ".join(f"{k}={v}" for k, v in data.items() if k != "pass")
        lines.append(f"| `{name}` | **{status}** | `{summary}` |")

    lines += [
        "",
        "## SLO by Concurrency",
        "",
        "| Level | Goodput (tok/s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p99 (ms) | Wall (s) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lvl, slo in sc.get("slo_by_concurrency", {}).items():
        lines.append(
            f"| `{lvl}` | {slo.get('goodput_tok_s')} | {slo.get('ttft_p50_ms')} | "
            f"{slo.get('ttft_p95_ms')} | {slo.get('itl_p99_ms')} | {slo.get('wall_s')} |"
        )

    mem = sc.get("memory_hwm", {})
    lines += [
        "",
        "## 4-Tier Memory High-Water Marks",
        "",
        f"- **R9700 VRAM Peak**: `{mem.get('r9700_vram_peak_mib')}` MiB",
        f"- **Strix Halo UMA Peak**: `{mem.get('strix_uma_peak_mib')}` MiB",
        f"- **Host RAM Peak**: `{mem.get('host_ram_peak_mib')}` MiB",
        f"- **SSD Prefix Cache Peak**: `{mem.get('ssd_prefix_cache_peak_gib')}` GiB",
    ]
    if sc.get("failures"):
        lines += ["", "## Failures", ""]
        for f in sc["failures"]:
            lines.append(f"- **[{f['code']}]** `{f['check']}`: {f['detail']}")
    lines.append("")
    return "\n".join(lines)


def write_scorecard(output_dir: Path | None, scorecard: dict[str, Any]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "scorecard.md").write_text(
        render_scorecard_markdown(scorecard),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    default_pack = Path(__file__).resolve().parent / "v1" / "pack.json"
    p = argparse.ArgumentParser(
        prog="luce-concurrency-gate",
        description="Run the Lucebox Golden Concurrency Gate (I1-I7) and emit canonical scorecards.",
    )
    p.add_argument("--pack", type=Path, default=default_pack, help="Path to v1/pack.json")
    p.add_argument("--tier", default="tier1_pr_smoke", help="Tier name from pack.json")
    p.add_argument("--model-profile", default="qwen3.8-27b", help="Model profile key from pack.json")
    p.add_argument("--base-url", default=None, help="Attach to running OpenAI-compatible server URL")
    p.add_argument("--server-bin", type=Path, default=None, help="Boot and manage raw luce_server binary")
    p.add_argument("--launcher", type=Path, default=None, help="Boot and manage customer launcher script")
    p.add_argument("--model-path", type=Path, default=None, help="Model GGUF path (used with --server-bin)")
    p.add_argument("--draft-path", type=Path, default=None, help="Optional draft GGUF path (used with --server-bin)")
    p.add_argument("--server-arg", action="append", default=[], help="Extra argument forwarded to managed server")
    p.add_argument("--server-log", type=Path, default=None, help="Path to server log for [concurrency-metrics] I7 check")
    p.add_argument("--host", default="127.0.0.1", help="Host for managed server")
    p.add_argument("--port", type=int, default=0, help="Port for managed server (0 = ephemeral)")
    p.add_argument("--concurrency-override", default=None, help="Comma-separated concurrency levels, e.g. '1,3,4'")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to write scorecard.json and scorecard.md")
    p.add_argument("--baseline-scorecard", type=Path, default=None, help="Baseline scorecard.json for regression checks")
    p.add_argument("--model-name", default=None, help="Model name sent in OpenAI requests (defaults to --model-profile)")
    p.add_argument("--api-key", default=None, help="Optional Bearer API key")
    p.add_argument("--disk-cache-dir", type=Path, default=None, help="Optional SSD prefix cache directory for HWM check")
    p.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout seconds")
    p.add_argument("--startup-timeout", type=float, default=180.0, help="Managed server startup timeout seconds")
    p.add_argument("--oversized-chars", type=int, default=0, help="Override oversized prompt length in chars")
    p.add_argument("--require-spec-metrics", action="store_true", help="Fail closed if I7 spec-decode metrics are absent")
    return p


def run_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    pack_path = args.pack.resolve()
    if not pack_path.is_file():
        return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"pack not found: {pack_path}"}

    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack_dir = pack_path.parent
    pack_sha = sha256_file(pack_path) or ""

    if args.tier not in pack.get("tiers", {}):
        return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"unknown tier: {args.tier}"}
    if args.model_profile not in pack.get("models", {}):
        return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"unknown model profile: {args.model_profile}"}

    tier_cfg = pack["tiers"][args.tier]
    model_cfg = pack["models"][args.model_profile]
    hw_cfg = pack["hardware_profiles"][model_cfg["hardware"]]
    thresholds = tier_cfg.get("thresholds", {})
    chaos_cfg = tier_cfg.get("chaos", {})

    modes_selected = sum(x is not None for x in (args.base_url, args.server_bin, args.launcher))
    if modes_selected != 1:
        return EXIT_INFRA_INVALID, {
            "verdict": "INFRA_INVALID",
            "exit_code": EXIT_INFRA_INVALID,
            "error": "specify exactly one of --base-url, --server-bin, or --launcher",
        }
    if tier_cfg.get("require_manage_launcher") and args.server_bin is not None:
        return EXIT_INFRA_INVALID, {
            "verdict": "INFRA_INVALID",
            "exit_code": EXIT_INFRA_INVALID,
            "error": f"tier {args.tier} requires --launcher (or --base-url), not raw --server-bin",
        }

    try:
        concurrency_levels, clamped_levels = resolve_concurrency_levels(
            tier_cfg, model_cfg, args.concurrency_override
        )
    except ValueError as exc:
        return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": str(exc)}

    baseline_sc: dict[str, Any] | None = None
    if args.baseline_scorecard is not None:
        if not args.baseline_scorecard.is_file():
            return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"baseline scorecard not found: {args.baseline_scorecard}"}
        baseline_sc = json.loads(args.baseline_scorecard.read_text(encoding="utf-8"))

    # Load workloads declared by tier
    workloads_by_mode: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for wl in tier_cfg.get("workloads", []):
        wl_path = pack_dir / wl["path"]
        if not wl_path.is_file():
            return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"workload not found: {wl_path}"}
        workloads_by_mode[wl["mode"]] = (wl, load_jsonl(wl_path))

    # Boot or attach to server
    managed: ManagedServer | None = None
    server_log_path = args.server_log
    tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    if args.base_url:
        api_base, server_root = normalize_urls(args.base_url)
    else:
        port = args.port or pick_free_port(args.host)
        server_root = f"http://{args.host}:{port}"
        api_base = server_root + "/v1"
        env = os.environ.copy()
        if args.output_dir:
            server_log_path = args.output_dir / "server.log"
        else:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="luce_gate_")
            server_log_path = Path(tmp_dir_obj.name) / "server.log"

        if args.launcher is not None:
            if not args.launcher.is_file():
                return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"launcher not found: {args.launcher}"}
            env["HOST"] = args.host
            env["PORT"] = str(port)
            env["MAX_CONCURRENCY"] = str(max(concurrency_levels))
            cmd = [str(args.launcher), *args.server_arg]
        else:
            assert args.server_bin is not None
            if not args.server_bin.is_file():
                return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": f"server binary not found: {args.server_bin}"}
            cmd = [
                str(args.server_bin),
                "--host", args.host,
                "--port", str(port),
                "--max-concurrency", str(max(concurrency_levels)),
            ]
            if args.model_path:
                cmd += ["--model", str(args.model_path)]
            if args.draft_path:
                cmd += ["--draft-model", str(args.draft_path)]
            cmd += list(args.server_arg)

        managed = ManagedServer(cmd, env, server_log_path, server_root, args.startup_timeout)
        ok, err = managed.start()
        if not ok:
            managed.stop()
            if tmp_dir_obj:
                tmp_dir_obj.cleanup()
            return EXIT_INFRA_INVALID, {"verdict": "INFRA_INVALID", "exit_code": EXIT_INFRA_INVALID, "error": err}

    sampler = MemorySampler(disk_cache_dir=args.disk_cache_dir)
    sampler.start()
    failures: list[dict[str, str]] = []
    invariant_failed = False
    slo_or_mem_failed = False
    all_completed_results: list[StreamResult] = []
    slo_by_concurrency: dict[str, dict[str, Any]] = {}

    def record_fail(code: int, check: str, detail: str) -> None:
        nonlocal invariant_failed, slo_or_mem_failed
        if code == EXIT_INVARIANT_FAIL:
            invariant_failed = True
        elif code == EXIT_SLO_OR_MEM_FAIL:
            slo_or_mem_failed = True
        failures.append({"code": "INVARIANT_FAIL" if code == EXIT_INVARIANT_FAIL else "SLO_OR_MEM_FAIL", "check": check, "detail": detail})

    try:
        health_status, _ = http_get_json(server_root + "/health", timeout=5.0)
        if health_status != 200:
            return EXIT_INFRA_INVALID, {
                "verdict": "INFRA_INVALID",
                "exit_code": EXIT_INFRA_INVALID,
                "error": f"health check failed at {server_root}/health (status={health_status})",
            }

        props_status, baseline_props = http_get_json(server_root + "/props", timeout=5.0)
        baseline_load = (baseline_props or {}).get("load") if props_status == 200 and isinstance(baseline_props, dict) else None
        req_model = args.model_name or args.model_profile

        # ── I1: Isolation & Zero Cross-Talk ─────────────────────────────
        canary_wl, canary_cases = workloads_by_mode["distinct_per_lane"]
        all_canaries = [c["expected_canary"] for c in canary_cases]
        recall_by_c: dict[int, float] = {}
        total_canary_lanes = 0
        total_own_hits = 0
        total_sibling_leaks = 0
        i1_http_errors = 0

        for c_level in concurrency_levels:
            if c_level > len(canary_cases):
                return EXIT_INFRA_INVALID, {
                    "verdict": "INFRA_INVALID",
                    "exit_code": EXIT_INFRA_INVALID,
                    "error": f"concurrency {c_level} exceeds distinct canary fixtures ({len(canary_cases)})",
                }
            wave_cases = canary_cases[:c_level]
            t_wave0 = time.perf_counter()
            fns = [
                (lambda case=case: stream_chat_completion(
                    api_base=api_base,
                    model=req_model,
                    messages=case["messages"],
                    max_tokens=int(canary_wl.get("max_tokens", 48)),
                    temperature=float(canary_wl.get("temperature", 0.0)),
                    timeout=args.request_timeout,
                    api_key=args.api_key,
                ))
                for case in wave_cases
            ]
            wave_results: list[StreamResult] = run_parallel_barrier(fns)
            wave_wall_s = time.perf_counter() - t_wave0
            slo_by_concurrency[f"C{c_level}"] = summarize_slo(wave_results, wave_wall_s)

            wave_hits = 0
            for case, r in zip(wave_cases, wave_results):
                total_canary_lanes += 1
                if not r.ok:
                    i1_http_errors += 1
                    continue
                all_completed_results.append(r)
                own = case["expected_canary"]
                if own in r.content:
                    wave_hits += 1
                    total_own_hits += 1
                combined = r.content + "\n" + r.reasoning_content
                for sib in all_canaries:
                    if sib != own and sib in combined:
                        total_sibling_leaks += 1
            recall_by_c[c_level] = wave_hits / len(wave_cases)

        own_canary_recall = (total_own_hits / total_canary_lanes) if total_canary_lanes else 0.0
        i1_pass = (
            i1_http_errors == 0
            and own_canary_recall >= float(thresholds.get("own_canary_recall_min", 1.0))
            and total_sibling_leaks <= int(thresholds.get("sibling_canary_leak_max", 0))
        )
        if not i1_pass:
            record_fail(
                EXIT_INVARIANT_FAIL,
                "I1_isolation",
                f"own_canary_recall={own_canary_recall:.4f}, sibling_leaks={total_sibling_leaks}, errors={i1_http_errors}",
            )

        # ── I2: Serial-vs-Concurrent Equivalence ────────────────────────
        baseline_recall = None
        if baseline_sc:
            baseline_recall = _float_or_none(
                ((baseline_sc.get("invariants") or {}).get("I1_isolation") or {}).get("own_canary_recall")
            )
        serial_ref = recall_by_c[1] if 1 in recall_by_c else (baseline_recall if baseline_recall is not None else 1.0)
        concurrent_recalls = [v for k, v in recall_by_c.items() if k > 1]
        min_conc_recall = min(concurrent_recalls) if concurrent_recalls else serial_ref
        drop_pp = round(max(0.0, (serial_ref - min_conc_recall) * 100.0), 3)
        max_drop_pp = float(thresholds.get("serial_vs_concurrent_accuracy_drop_max_pp", 0.0))
        i2_pass = drop_pp <= max_drop_pp
        if not i2_pass:
            record_fail(
                EXIT_INVARIANT_FAIL,
                "I2_equivalence",
                f"serial_vs_concurrent_drop_pp={drop_pp} > max {max_drop_pp}",
            )

        # ── I4 + I3 (part 1): Multi-Turn Prefix Cache & Tool Calls ──────
        mt_wl, mt_cases = workloads_by_mode["multi_turn_p1_p2"]
        c_mt = min(max(concurrency_levels), len(mt_cases))
        mt_slice = mt_cases[:c_mt]

        p1_fns = [
            (lambda s=s: stream_chat_completion(
                api_base=api_base,
                model=req_model,
                messages=s["turn1_messages"],
                max_tokens=int(mt_wl.get("max_tokens", 64)),
                temperature=float(mt_wl.get("temperature", 0.0)),
                timeout=args.request_timeout,
                api_key=args.api_key,
                tools=s.get("tools"),
            ))
            for s in mt_slice
        ]
        p1_results: list[StreamResult] = run_parallel_barrier(p1_fns)
        all_completed_results.extend(r for r in p1_results if r.ok)

        p2_fns = []
        for s, r1 in zip(mt_slice, p1_results):
            turn2_msgs = list(s["turn1_messages"]) + [
                {"role": "assistant", "content": r1.content or "Acknowledged."},
                s["turn2_followup"],
            ]
            p2_fns.append(
                lambda msgs=turn2_msgs, tools=s.get("tools"): stream_chat_completion(
                    api_base=api_base,
                    model=req_model,
                    messages=msgs,
                    max_tokens=int(mt_wl.get("max_tokens", 64)),
                    temperature=float(mt_wl.get("temperature", 0.0)),
                    timeout=args.request_timeout,
                    api_key=args.api_key,
                    tools=tools,
                )
            )
        p2_results: list[StreamResult] = run_parallel_barrier(p2_fns)
        all_completed_results.extend(r for r in p2_results if r.ok)

        p2_hit_rates: list[float] = []
        p1_ttfts = [r.ttft_ms for r in p1_results if r.ok and r.ttft_ms is not None]
        p2_ttfts = [r.ttft_ms for r in p2_results if r.ok and r.ttft_ms is not None]
        mt_http_errors = sum(1 for r in p1_results + p2_results if not r.ok)

        tool_total = 0
        tool_valid = 0
        tool_error_samples: list[str] = []

        for s, r1, r2 in zip(mt_slice, p1_results, p2_results):
            if r2.ok:
                shared_est = float(s.get("shared_prefix_ratio_estimate", 0.9))
                shared_tokens = (
                    r1.prompt_tokens
                    if (r1.prompt_tokens and r1.prompt_tokens > 0)
                    else (int(r2.prompt_tokens * shared_est) if r2.prompt_tokens else 0)
                )
                if r2.cached_tokens is not None and shared_tokens > 0:
                    p2_hit_rates.append(min(1.0, r2.cached_tokens / shared_tokens))
                else:
                    p2_hit_rates.append(0.0)
            tool_total += 1
            ok_tc, reason = validate_tool_call(
                r2,
                s["expected_tool_name"],
                s.get("required_args", []),
                s.get("arg_types"),
                s.get("expected_arg_values"),
            )
            if ok_tc:
                tool_valid += 1
            else:
                tool_error_samples.append(f"{s['id']}: {reason}")

        # ── I3 (part 2): Dedicated Tool-Call Edge Cases ─────────────────
        if "tool_call_validation" in workloads_by_mode:
            tc_wl, tc_cases = workloads_by_mode["tool_call_validation"]
            tc_fns = [
                (lambda tc=tc: stream_chat_completion(
                    api_base=api_base,
                    model=req_model,
                    messages=tc["messages"],
                    max_tokens=int(tc_wl.get("max_tokens", 96)),
                    temperature=float(tc_wl.get("temperature", 0.0)),
                    timeout=args.request_timeout,
                    api_key=args.api_key,
                    tools=tc.get("tools"),
                ))
                for tc in tc_cases
            ]
            tc_results: list[StreamResult] = run_parallel_barrier(tc_fns)
            all_completed_results.extend(r for r in tc_results if r.ok)
            for tc, r in zip(tc_cases, tc_results):
                tool_total += 1
                ok_tc, reason = validate_tool_call(
                    r,
                    tc["expected_tool_name"],
                    tc.get("required_args", []),
                    tc.get("arg_types"),
                    tc.get("expected_arg_values"),
                )
                if ok_tc:
                    tool_valid += 1
                else:
                    tool_error_samples.append(f"{tc['id']}: {reason}")

        tool_valid_rate = (tool_valid / tool_total) if tool_total else 1.0
        i3_pass = (
            own_canary_recall >= float(thresholds.get("own_canary_recall_min", 1.0))
            and tool_valid_rate >= float(thresholds.get("tool_call_json_valid_min", 1.0))
        )
        if not i3_pass:
            record_fail(
                EXIT_INVARIANT_FAIL,
                "I3_quality_floor",
                f"niah_recall={own_canary_recall:.4f}, tool_call_valid={tool_valid_rate:.4f} ({'; '.join(tool_error_samples[:3])})",
            )

        p2_hit_rate = round(statistics.mean(p2_hit_rates), 4) if p2_hit_rates else 0.0
        p1_med_ttft = statistics.median(p1_ttfts) if p1_ttfts else None
        p2_med_ttft = statistics.median(p2_ttfts) if p2_ttfts else None
        p2_ttft_ratio = (
            round(p2_med_ttft / p1_med_ttft, 4)
            if (p1_med_ttft is not None and p2_med_ttft is not None and p1_med_ttft > 0)
            else None
        )
        hit_min = float(thresholds.get("p2_prefix_cache_hit_min", 0.85))
        ttft_ratio_max = float(thresholds.get("p2_ttft_ratio_vs_p1_max", 0.45))
        i4_cache_ok = p2_hit_rate >= hit_min
        i4_ttft_ok = (p2_ttft_ratio is None) or (p2_ttft_ratio <= ttft_ratio_max)
        i4_pass = mt_http_errors == 0 and i4_cache_ok and i4_ttft_ok
        if mt_http_errors > 0:
            record_fail(EXIT_INVARIANT_FAIL, "I4_prefix_cache", f"multi-turn HTTP errors={mt_http_errors}")
        elif not i4_pass:
            record_fail(
                EXIT_SLO_OR_MEM_FAIL,
                "I4_prefix_cache",
                f"p2_hit_rate={p2_hit_rate} (min {hit_min}), p2_ttft_ratio={p2_ttft_ratio} (max {ttft_ratio_max})",
            )

        # ── I5: Graceful Backpressure ───────────────────────────────────
        oversub_factor = float(chaos_cfg.get("oversubscribe_factor", 1.25))
        max_c = max(concurrency_levels)
        burst_n = max(max_c + 1, int(math.ceil(max_c * oversub_factor)))
        burst_fns = [
            (lambda idx=i: stream_chat_completion(
                api_base=api_base,
                model=req_model,
                messages=canary_cases[idx % len(canary_cases)]["messages"],
                max_tokens=int(canary_wl.get("max_tokens", 48)),
                temperature=0.0,
                timeout=args.request_timeout,
                api_key=args.api_key,
            ))
            for i in range(burst_n)
        ]
        burst_results: list[StreamResult] = run_parallel_barrier(burst_fns)
        http_5xx = sum(1 for r in burst_results if (r.http_status or 0) >= 500)
        http_429 = sum(1 for r in burst_results if r.http_status == 429)
        burst_ok = sum(1 for r in burst_results if r.ok)
        burst_bad = sum(
            1 for r in burst_results
            if not r.ok and r.http_status != 429
        )
        all_completed_results.extend(r for r in burst_results if r.ok)

        oversized_status: int | None = None
        if chaos_cfg.get("oversized_prompt_check", True):
            if args.oversized_chars > 0:
                n_chars = args.oversized_chars
            else:
                n_ctx = _int_or_none((baseline_props or {}).get("n_ctx")) or _int_or_none(
                    ((baseline_props or {}).get("default_generation_settings") or {}).get("n_ctx")
                )
                n_chars = max(8192, n_ctx * 8) if n_ctx else 600000
            oversized_msg = [{"role": "user", "content": ("oversize " * (n_chars // 9 + 1))[:n_chars]}]
            ov_res = stream_chat_completion(
                api_base=api_base,
                model=req_model,
                messages=oversized_msg,
                max_tokens=16,
                temperature=0.0,
                timeout=args.request_timeout,
                api_key=args.api_key,
            )
            oversized_status = ov_res.http_status
            if (oversized_status or 0) >= 500:
                http_5xx += 1

        i5_pass = (
            http_5xx <= int(thresholds.get("http_5xx_max", 0))
            and burst_bad == 0
            and (not chaos_cfg.get("oversized_prompt_check", True) or oversized_status == 400)
        )
        if not i5_pass:
            record_fail(
                EXIT_INVARIANT_FAIL,
                "I5_backpressure",
                f"http_5xx={http_5xx}, burst_completed={burst_ok}/{burst_n}, http_429={http_429}, burst_bad={burst_bad}, oversized_status={oversized_status}",
            )

        # ── I6: Cancellation & Disconnect Safety ────────────────────────
        if "cancel_mid_prefill_lanes" in chaos_cfg:
            n_pre = int(chaos_cfg["cancel_mid_prefill_lanes"])
        else:
            n_pre = max(1, int(round(max_c * float(chaos_cfg.get("cancel_mid_prefill_pct", 0.125)))))
        if "cancel_mid_decode_lanes" in chaos_cfg:
            n_dec = int(chaos_cfg["cancel_mid_decode_lanes"])
        else:
            n_dec = max(1, int(round(max_c * float(chaos_cfg.get("cancel_mid_decode_pct", 0.125)))))
        cancel_after = int(chaos_cfg.get("cancel_after_tokens", 6))
        n_survive = max(1, min(len(canary_cases) - min(n_pre + n_dec, len(canary_cases) - 1), max(1, max_c - n_pre - n_dec)))

        pre_cancel_mem = sampler.sample_once()
        cancel_fns: list[Any] = []
        for i in range(n_pre):
            case = canary_cases[i % len(canary_cases)]
            cancel_fns.append(
                lambda c=case: ("pre", inject_cancel_mid_prefill(
                    api_base, req_model, c["messages"], 48, args.api_key, args.request_timeout
                ))
            )
        for i in range(n_dec):
            case = canary_cases[(n_pre + i) % len(canary_cases)]
            cancel_fns.append(
                lambda c=case: ("dec", inject_cancel_mid_decode(
                    api_base, req_model, c["messages"], 48, cancel_after, args.api_key, args.request_timeout
                ))
            )
        surviving_cases = [canary_cases[(n_pre + n_dec + i) % len(canary_cases)] for i in range(n_survive)]
        for case in surviving_cases:
            cancel_fns.append(
                lambda c=case: ("survive", c, stream_chat_completion(
                    api_base=api_base,
                    model=req_model,
                    messages=c["messages"],
                    max_tokens=int(canary_wl.get("max_tokens", 48)),
                    temperature=0.0,
                    timeout=args.request_timeout,
                    api_key=args.api_key,
                ))
            )

        cancel_outcomes = run_parallel_barrier(cancel_fns)
        aborts_injected = sum(1 for item in cancel_outcomes if item[0] in ("pre", "dec") and item[1])
        survive_hits = 0
        survive_leaks = 0
        survive_errors = 0
        for item in cancel_outcomes:
            if item[0] != "survive":
                continue
            _, case, r = item
            if not r.ok:
                survive_errors += 1
                continue
            all_completed_results.append(r)
            own = case["expected_canary"]
            if own in r.content:
                survive_hits += 1
            combined = r.content + "\n" + r.reasoning_content
            for sib in all_canaries:
                if sib != own and sib in combined:
                    survive_leaks += 1

        # Poll /props up to 5s for zero slot/KV leak
        slot_leak = 0
        kv_block_leak = 0
        baseline_kv_free = _int_or_none((baseline_load or {}).get("kv_free_tokens"))
        if baseline_load is not None:
            drain_deadline = time.perf_counter() + DRAIN_TIMEOUT_S
            while True:
                p_st, p_body = http_get_json(server_root + "/props", timeout=2.0)
                load_now = (p_body or {}).get("load") if p_st == 200 and isinstance(p_body, dict) else None
                if isinstance(load_now, dict):
                    busy_now = _int_or_none(load_now.get("slots_busy")) or 0
                    kv_free_now = _int_or_none(load_now.get("kv_free_tokens"))
                    slot_leak = busy_now
                    kv_block_leak = (
                        max(0, baseline_kv_free - kv_free_now)
                        if (baseline_kv_free is not None and kv_free_now is not None)
                        else 0
                    )
                    if slot_leak == 0 and kv_block_leak == 0:
                        break
                if time.perf_counter() >= drain_deadline:
                    break
                time.sleep(0.1)

        post_cancel_mem = sampler.sample_once()
        vram_delta_pct: float | None = None
        if pre_cancel_mem["vram_mib"] and post_cancel_mem["vram_mib"] is not None:
            vram_delta_pct = round(
                max(0.0, (post_cancel_mem["vram_mib"] - pre_cancel_mem["vram_mib"]) / pre_cancel_mem["vram_mib"] * 100.0),
                2,
            )

        survive_recall = survive_hits / len(surviving_cases)
        i6_pass = (
            survive_errors == 0
            and survive_recall == 1.0
            and survive_leaks == 0
            and slot_leak <= int(thresholds.get("post_wave_slot_leak_max", 0))
            and kv_block_leak <= int(thresholds.get("post_wave_kv_block_leak_max", 0))
            and (vram_delta_pct is None or vram_delta_pct <= float(thresholds.get("post_wave_vram_delta_pct_max", 2.0)))
        )
        if not i6_pass:
            record_fail(
                EXIT_INVARIANT_FAIL,
                "I6_cancellation",
                f"survive_recall={survive_recall:.4f}, survive_errors={survive_errors}, survive_leaks={survive_leaks}, slot_leak={slot_leak}, kv_block_leak={kv_block_leak}, vram_delta_pct={vram_delta_pct}",
            )

        # ── I7: Spec-Decode / Metrics Coexistence ───────────────────────
        spec_mode = (model_cfg.get("features") or {}).get("spec_decode", "none")
        spec_floor = _float_or_none((model_cfg.get("baseline_slo") or {}).get("spec_mean_accepted_min"))
        log_metrics = parse_concurrency_metrics_log(server_log_path)

        accepted_samples: list[float] = []
        matched_ids = 0
        total_with_id = 0
        for r in all_completed_results:
            if r.response_id:
                total_with_id += 1
            m_rec = log_metrics.get(r.response_id or "")
            if m_rec is not None:
                matched_ids += 1
                steps = _int_or_none(m_rec.get("ddtree_steps")) or _int_or_none(m_rec.get("spec_steps")) or 0
                acc = _int_or_none(m_rec.get("ddtree_accepted_tokens")) or _int_or_none(m_rec.get("spec_accepted_tokens")) or 0
                al = _float_or_none(m_rec.get("mean_accepted_len"))
                if al is not None:
                    accepted_samples.append(al)
                elif steps > 0:
                    accepted_samples.append((acc + steps) / steps)
            else:
                al = _float_or_none(r.timings.get("mean_accepted_len")) or _float_or_none(r.timings.get("spec_mean_accepted"))
                if al is not None:
                    matched_ids += 1
                    accepted_samples.append(al)

        spec_enforced = spec_mode != "none" and (
            args.require_spec_metrics
            or (server_log_path is not None and server_log_path.is_file())
            or bool(accepted_samples)
        )
        mean_accepted = round(statistics.mean(accepted_samples), 3) if accepted_samples else None
        metrics_coverage = round(matched_ids / total_with_id, 4) if total_with_id > 0 else 0.0

        if not spec_enforced:
            i7_data = {
                "pass": True,
                "mode": "not_applicable",
                "mean_accepted_len": mean_accepted,
                "metrics_coverage": None,
            }
        else:
            cov_ok = metrics_coverage >= 1.0 and bool(accepted_samples)
            floor_ok = (spec_floor is None) or (mean_accepted is not None and mean_accepted >= spec_floor)
            i7_pass = cov_ok and floor_ok
            i7_data = {
                "pass": i7_pass,
                "mode": spec_mode,
                "mean_accepted_len": mean_accepted,
                "metrics_coverage": metrics_coverage,
            }
            if not cov_ok:
                record_fail(
                    EXIT_INVARIANT_FAIL,
                    "I7_spec_kvflash",
                    f"incomplete spec metrics coverage={metrics_coverage} (matched {matched_ids}/{total_with_id})",
                )
            elif not floor_ok:
                record_fail(
                    EXIT_SLO_OR_MEM_FAIL,
                    "I7_spec_kvflash",
                    f"mean_accepted_len={mean_accepted} < floor {spec_floor}",
                )

        # ── 4-Tier Memory HWM & Baseline Regression Checks ──────────────
        memory_hwm = sampler.stop()
        vram_max = _float_or_none(hw_cfg.get("vram_hwm_max_mib"))
        uma_max = _float_or_none(hw_cfg.get("uma_hwm_max_mib"))
        ram_max = _float_or_none(hw_cfg.get("host_ram_hwm_max_mib"))
        disk_max = _float_or_none(hw_cfg.get("disk_cache_max_gib"))

        if vram_max is not None and memory_hwm["r9700_vram_peak_mib"] is not None:
            if memory_hwm["r9700_vram_peak_mib"] > vram_max:
                record_fail(EXIT_SLO_OR_MEM_FAIL, "memory_vram_hwm", f"{memory_hwm['r9700_vram_peak_mib']} MiB > {vram_max} MiB")
        if uma_max is not None and memory_hwm["strix_uma_peak_mib"] is not None:
            if memory_hwm["strix_uma_peak_mib"] > uma_max:
                record_fail(EXIT_SLO_OR_MEM_FAIL, "memory_uma_hwm", f"{memory_hwm['strix_uma_peak_mib']} MiB > {uma_max} MiB")
        if ram_max is not None and memory_hwm["host_ram_peak_mib"] is not None:
            if memory_hwm["host_ram_peak_mib"] > ram_max:
                record_fail(EXIT_SLO_OR_MEM_FAIL, "memory_host_ram_hwm", f"{memory_hwm['host_ram_peak_mib']} MiB > {ram_max} MiB")
        if disk_max is not None and memory_hwm["ssd_prefix_cache_peak_gib"] is not None:
            if memory_hwm["ssd_prefix_cache_peak_gib"] > disk_max:
                record_fail(EXIT_SLO_OR_MEM_FAIL, "memory_disk_cache_hwm", f"{memory_hwm['ssd_prefix_cache_peak_gib']} GiB > {disk_max} GiB")

        if baseline_sc:
            base_slos = baseline_sc.get("slo_by_concurrency") or {}
            gp_max_reg = _float_or_none(thresholds.get("goodput_regression_max_pct"))
            wall_max_reg = _float_or_none(thresholds.get("wall_time_regression_vs_baseline_max_pct"))
            for lvl, cur in slo_by_concurrency.items():
                prev = base_slos.get(lvl)
                if not isinstance(prev, dict):
                    continue
                if gp_max_reg is not None and prev.get("goodput_tok_s") and cur.get("goodput_tok_s") is not None:
                    drop_pct = (prev["goodput_tok_s"] - cur["goodput_tok_s"]) / prev["goodput_tok_s"] * 100.0
                    if drop_pct > gp_max_reg:
                        record_fail(EXIT_SLO_OR_MEM_FAIL, f"slo_goodput_{lvl}", f"goodput dropped {drop_pct:.2f}% > {gp_max_reg}%")
                if wall_max_reg is not None and prev.get("wall_s") and cur.get("wall_s") is not None:
                    grow_pct = (cur["wall_s"] - prev["wall_s"]) / prev["wall_s"] * 100.0
                    if grow_pct > wall_max_reg:
                        record_fail(EXIT_SLO_OR_MEM_FAIL, f"slo_wall_{lvl}", f"wall time grew {grow_pct:.2f}% > {wall_max_reg}%")

        if invariant_failed:
            exit_code = EXIT_INVARIANT_FAIL
            verdict = "INVARIANT_FAIL"
        elif slo_or_mem_failed:
            exit_code = EXIT_SLO_OR_MEM_FAIL
            verdict = "SLO_OR_MEM_FAIL"
        else:
            exit_code = EXIT_PASS
            verdict = "PASS"

        scorecard = {
            "schema_version": 1,
            "pack_id": pack.get("pack_id", "lucebox-concurrency-golden-v1"),
            "pack_sha256": pack_sha,
            "tier": args.tier,
            "model_profile": args.model_profile,
            "concurrency_levels": concurrency_levels,
            "concurrency_clamped": clamped_levels,
            "verdict": verdict,
            "exit_code": exit_code,
            "provenance": {
                "engine_repo_sha": git_head_sha(Path(__file__).resolve().parent),
                "launcher_path": str(args.launcher) if args.launcher else model_cfg.get("manage_launcher"),
                "launcher_sha256": sha256_file(args.launcher) if args.launcher else None,
                "server_bin": str(args.server_bin) if args.server_bin else None,
                "model_sha256": sha256_file(args.model_path) if args.model_path else None,
                "draft_sha256": sha256_file(args.draft_path) if args.draft_path else None,
                "base_url": api_base,
            },
            "invariants": {
                "I1_isolation": {
                    "pass": i1_pass,
                    "own_canary_recall": round(own_canary_recall, 4),
                    "sibling_leaks": total_sibling_leaks,
                    "errors": i1_http_errors,
                },
                "I2_equivalence": {
                    "pass": i2_pass,
                    "serial_ref_recall": round(serial_ref, 4),
                    "serial_vs_concurrent_drop_pp": drop_pp,
                },
                "I3_quality_floor": {
                    "pass": i3_pass,
                    "niah_recall": round(own_canary_recall, 4),
                    "tool_call_valid": round(tool_valid_rate, 4),
                    "tool_calls_checked": tool_total,
                },
                "I4_prefix_cache": {
                    "pass": i4_pass,
                    "p2_ram_hit_rate": p2_hit_rate,
                    "p2_ttft_ratio": p2_ttft_ratio,
                },
                "I5_backpressure": {
                    "pass": i5_pass,
                    "http_5xx": http_5xx,
                    "http_429": http_429,
                    "oversized_status": oversized_status,
                    "oversubscribe_completed": f"{burst_ok}/{burst_n}",
                },
                "I6_cancellation": {
                    "pass": i6_pass,
                    "aborts_injected": aborts_injected,
                    "surviving_recall": round(survive_recall, 4),
                    "slot_leak": slot_leak,
                    "kv_block_leak": kv_block_leak,
                    "vram_delta_pct": vram_delta_pct,
                },
                "I7_spec_kvflash": i7_data,
            },
            "slo_by_concurrency": slo_by_concurrency,
            "memory_hwm": memory_hwm,
            "failures": failures,
        }
        write_scorecard(args.output_dir, scorecard)
        return exit_code, scorecard
    finally:
        sampler.stop()
        if managed is not None:
            managed.stop()
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    code, scorecard = run_gate(args)
    if args.output_dir is None and code != EXIT_PASS and "error" in scorecard:
        print(f"[luce-concurrency-gate] {scorecard['verdict']}: {scorecard['error']}", file=sys.stderr)
    else:
        print(
            f"[luce-concurrency-gate] verdict={scorecard.get('verdict')} exit_code={code} "
            f"tier={args.tier} model={args.model_profile}"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
