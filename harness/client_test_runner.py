#!/usr/bin/env python3
"""Client-harness compatibility and server-profile sweeps for Luce DFlash.

The goal is deliberately narrower than a full SWE-bench agent run:

1. Download the real client packages contributors care about.
2. Smoke-test their installed binaries where possible.
3. Probe the DFlash HTTP protocol shape each client depends on.
4. Sweep server settings and record latency / token / OOM signals.

The script uses only the Python standard library so it can run on fresh GPU
test machines before project Python dependencies are installed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = ROOT / ".harness-work"
MODEL = "luce-dflash"
PROBE_COUNTER = itertools.count(1)
EXPECTED_MARKER = "lucebox"


@dataclass(frozen=True)
class ClientSpec:
    name: str
    install: str
    package: str
    binary: str
    protocol: str
    version_args: tuple[str, ...] = ("--version",)
    help_args: tuple[str, ...] = ("--help",)
    notes: str = ""


CLIENTS: dict[str, ClientSpec] = {
    "claude_code": ClientSpec(
        name="claude_code",
        install="npm",
        package="@anthropic-ai/claude-code",
        binary="claude",
        protocol="anthropic_messages",
        notes="Claude Code speaks Anthropic Messages; ANTHROPIC_BASE_URL points to the server root.",
    ),
    "codex": ClientSpec(
        name="codex",
        install="npm",
        package="@openai/codex",
        binary="codex",
        protocol="responses",
        notes="Codex CLI uses OpenAI Responses and queries /v1/models?client_version=.",
    ),
    "hermes": ClientSpec(
        name="hermes",
        install="hermes",
        package="https://github.com/NousResearch/hermes-agent",
        binary="hermes",
        protocol="openai_chat",
        notes="Hermes Agent can use a named custom OpenAI-compatible provider.",
    ),
    "openclaw": ClientSpec(
        name="openclaw",
        install="npm",
        package="openclaw",
        binary="openclaw",
        protocol="openai_chat",
    ),
    "openwebui": ClientSpec(
        name="openwebui",
        install="pip",
        package="open-webui",
        binary="open-webui",
        protocol="openwebui",
    ),
    "opencode": ClientSpec(
        name="opencode",
        install="npm",
        package="opencode-ai",
        binary="opencode",
        protocol="openai_chat",
    ),
    "pi": ClientSpec(
        name="pi",
        install="npm",
        package="@mariozechner/pi-coding-agent",
        binary="pi",
        protocol="openai_chat",
    ),
}


@dataclass(frozen=True)
class ServerProfile:
    name: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    needs_prefill_drafter: bool = False
    long_prompt: bool = False


SERVER_PROFILES: dict[str, ServerProfile] = {
    "rtx3090_dflash_fast": ServerProfile(
        name="rtx3090_dflash_fast",
        args=(
            "--budget", "22",
            "--verify-mode", "ddtree",
            "--max-ctx", "4096",
            "--fa-window", "1024",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "0",
        ),
    ),
    "rtx3090_dflash_safe": ServerProfile(
        name="rtx3090_dflash_safe",
        args=(
            "--budget", "22",
            "--verify-mode", "ddtree",
            "--max-ctx", "8192",
            "--fa-window", "2048",
            "--cache-type-k", "tq3_0",
            "--cache-type-v", "tq3_0",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "0",
        ),
    ),
    "rtx3090_dflash_16k": ServerProfile(
        name="rtx3090_dflash_16k",
        args=(
            "--budget", "22",
            "--verify-mode", "ddtree",
            "--max-ctx", "16384",
            "--fa-window", "2048",
            "--cache-type-k", "tq3_0",
            "--cache-type-v", "tq3_0",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "0",
        ),
    ),
    "rtx3090_dflash_long": ServerProfile(
        name="rtx3090_dflash_long",
        args=(
            "--budget", "16",
            "--verify-mode", "ddtree",
            "--max-ctx", "32768",
            "--fa-window", "2048",
            "--cache-type-k", "tq3_0",
            "--cache-type-v", "tq3_0",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "0",
            "--lazy-draft",
        ),
        long_prompt=True,
    ),
    "rtx3090_pflash_32k": ServerProfile(
        name="rtx3090_pflash_32k",
        args=(
            "--budget", "16",
            "--verify-mode", "ddtree",
            "--max-ctx", "32768",
            "--fa-window", "2048",
            "--cache-type-k", "tq3_0",
            "--cache-type-v", "tq3_0",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "0",
            "--prefill-compression", "auto",
            "--prefill-threshold", "4096",
            "--prefill-keep-ratio", "0.10",
            "--lazy-draft",
        ),
        needs_prefill_drafter=True,
        long_prompt=True,
    ),
}


class HarnessError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def split_csv(value: str, choices: dict[str, Any]) -> list[str]:
    if value == "all":
        return list(choices)
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in out if x not in choices]
    if unknown:
        raise SystemExit(f"unknown selection(s): {', '.join(unknown)}")
    return out


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": round(time.perf_counter() - t0, 3),
            "cmd": cmd,
            "output_tail": proc.stdout[-4000:],
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "seconds": round(time.perf_counter() - t0, 3),
            "cmd": cmd,
            "output_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": 124,
            "seconds": round(time.perf_counter() - t0, 3),
            "cmd": cmd,
            "output_tail": out[-4000:] + "\nTIMEOUT",
        }


def npm_prefix(work_dir: Path, client: str) -> Path:
    return work_dir / "clients" / client / "npm"


def pip_venv(work_dir: Path, client: str) -> Path:
    return work_dir / "clients" / client / "venv"


def hermes_home(work_dir: Path) -> Path:
    return work_dir / "clients" / "hermes" / "home"


def client_bin(work_dir: Path, spec: ClientSpec) -> Path:
    if spec.install == "npm":
        return npm_prefix(work_dir, spec.name) / "bin" / spec.binary
    if spec.install == "pip":
        return pip_venv(work_dir, spec.name) / "bin" / spec.binary
    if spec.install == "hermes":
        return hermes_home(work_dir) / ".local" / "bin" / spec.binary
    raise HarnessError(f"unknown installer {spec.install}")


def client_smoke_env(work_dir: Path, spec: ClientSpec) -> dict[str, str] | None:
    if spec.install != "hermes":
        return None
    home = hermes_home(work_dir)
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "HERMES_HOME": str(home),
    })
    return env


def install_client(work_dir: Path, spec: ClientSpec) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if spec.install == "npm":
        prefix = npm_prefix(work_dir, spec.name)
        prefix.mkdir(parents=True, exist_ok=True)
        result = run_cmd(
            ["npm", "install", "--global", "--prefix", str(prefix), spec.package],
            timeout=900,
        )
    elif spec.install == "pip":
        env_dir = pip_venv(work_dir, spec.name)
        if not (env_dir / "bin" / "python").exists():
            venv.EnvBuilder(with_pip=True).create(env_dir)
        result = run_cmd(
            [str(env_dir / "bin" / "python"), "-m", "pip", "install", "-U", spec.package],
            timeout=1200,
        )
    elif spec.install == "hermes":
        root = work_dir / "clients" / spec.name
        home = hermes_home(work_dir)
        install_dir = root / "hermes-agent"
        root.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)
        script_path = root / "install.sh"
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh",
            script_path,
        )
        env = os.environ.copy()
        env.update({
            "HOME": str(home),
            "HERMES_HOME": str(home),
            "HERMES_INSTALL_DIR": str(install_dir),
        })
        result = run_cmd(
            ["bash", str(script_path), "--skip-setup", "--skip-browser"],
            env=env,
            timeout=1800,
        )
    else:
        raise HarnessError(f"unknown installer {spec.install}")

    bin_path = client_bin(work_dir, spec)
    version = binary_smoke(
        bin_path,
        spec.version_args,
        timeout=30,
        env=client_smoke_env(work_dir, spec),
    )
    result.update({
        "client": spec.name,
        "installer": spec.install,
        "package": spec.package,
        "binary": str(bin_path),
        "binary_exists": bin_path.exists(),
        "version": version,
    })
    return result


def binary_smoke(
    path: Path,
    args: tuple[str, ...],
    timeout: int = 20,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "output_tail": f"missing binary: {path}"}
    return run_cmd([str(path), *args], timeout=timeout, env=env)


def package_smoke(work_dir: Path, spec: ClientSpec) -> dict[str, Any]:
    path = client_bin(work_dir, spec)
    env = client_smoke_env(work_dir, spec)
    version = binary_smoke(path, spec.version_args, timeout=20, env=env)
    if version["ok"]:
        return {"client": spec.name, "ok": True, "binary": str(path), "version": version}
    help_result = binary_smoke(path, spec.help_args, timeout=20, env=env)
    return {
        "client": spec.name,
        "ok": bool(help_result.get("ok")),
        "binary": str(path),
        "version": version,
        "help": help_result,
    }


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 600,
) -> tuple[int, dict[str, Any] | str, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=all_headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - t0
            try:
                return resp.status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return resp.status, raw, elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - t0
        try:
            parsed: dict[str, Any] | str = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed, elapsed


def http_sse(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    expect_substring: str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        },
    )
    t0 = time.perf_counter()
    first = None
    events: list[dict[str, Any] | str] = []
    token_deltas = 0
    text = ""
    status = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    events.append(data)
                    continue
                events.append(obj)
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    if piece:
                        token_deltas += 1
                        text += str(piece)
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "seconds": round(time.perf_counter() - t0, 3),
            "error": exc.read().decode("utf-8", errors="replace")[-4000:],
        }
    elapsed = time.perf_counter() - t0
    check = text_check(text, expect_substring)
    return {
        "ok": status == 200 and token_deltas > 0 and check["generated_ok"],
        "status": status,
        "seconds": round(elapsed, 3),
        "first_token_seconds": round(first - t0, 3) if first else None,
        "events": len(events),
        "token_deltas": token_deltas,
        **check,
        "tail": events[-3:],
    }


def short_prompt() -> str:
    return (
        "You are checking a local model server. Reply with one short sentence "
        "that contains the word lucebox."
    )


def coding_prompt() -> str:
    return (
        "In a Python project, explain what this function does and mention one "
        "edge case. Include the word lucebox in your answer.\n\n"
        "def clamp(x, lo, hi):\n"
        "    if lo > hi:\n"
        "        raise ValueError('bad bounds')\n"
        "    return min(max(x, lo), hi)\n"
    )


def long_prompt() -> str:
    unit = (
        "The server is being tested for long-context harness behavior. "
        "Keep the invariant NEEDLE=lucebox-harness visible in your answer. "
        "Most of this text is filler that should survive tokenization and "
        "optional PFlash compression without crashing the server.\n"
    )
    return unit * 180


def unique_prompt(text: str, label: str) -> str:
    return f"{text}\n\nlucebox-harness request {label}-{next(PROBE_COUNTER)}"


def text_check(text: str | None, expect_substring: str | None = None) -> dict[str, Any]:
    value = text or ""
    ok = bool(value.strip())
    marker_seen = (
        expect_substring.lower() in value.lower()
        if expect_substring else None
    )
    return {
        "generated_ok": ok,
        "generated_chars": len(value),
        "generated_text_tail": value[-500:],
        "expected_substring": expect_substring,
        "expected_substring_seen": marker_seen,
    }


def openai_chat_text(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def anthropic_text(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return ""
    pieces = []
    for item in body.get("content") or []:
        if isinstance(item, dict):
            pieces.append(str(item.get("text") or item.get("thinking") or ""))
    return "".join(pieces)


def responses_text(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return ""
    if body.get("output_text"):
        return str(body["output_text"])
    pieces = []
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, dict):
                pieces.append(str(content.get("text") or ""))
    return "".join(pieces)


def record_probe(
    name: str,
    status: int,
    body: dict[str, Any] | str,
    elapsed: float,
    expect_status: int = 200,
    generated_text: str | None = None,
    expect_substring: str | None = None,
    required: bool = True,
) -> dict[str, Any]:
    usage = body.get("usage") if isinstance(body, dict) else None
    out = {
        "name": name,
        "ok": status == expect_status,
        "status": status,
        "required": required,
        "seconds": round(elapsed, 3),
        "usage": usage,
        "body_tail": body if isinstance(body, str) else json.dumps(body)[-2000:],
    }
    if generated_text is not None:
        check = text_check(generated_text, expect_substring)
        out.update(check)
        out["ok"] = out["ok"] and check["generated_ok"]
    return out


def probe_health(base_url: str) -> list[dict[str, Any]]:
    out = []
    for path in ("/health", "/v1/models"):
        status, body, elapsed = http_json("GET", base_url + path)
        out.append(record_probe(path, status, body, elapsed))
    return out


def probe_openai_chat(base_url: str, *, include_long: bool = False) -> list[dict[str, Any]]:
    prompt = unique_prompt(
        long_prompt() if include_long else coding_prompt(),
        "chat-non-stream",
    )
    probes = probe_health(base_url)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 32,
        "temperature": 0,
        "stop": ["\n\n\n"],
    }
    stream_prompt = unique_prompt(
        long_prompt() if include_long else coding_prompt(),
        "chat-stream",
    )
    stream_payload = dict(payload)
    stream_payload["messages"] = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": stream_prompt},
    ]
    stream_payload.update({"stream": True, "stream_options": {"include_usage": True}})
    probes.append({
        "name": "chat.stream",
        **http_sse(
            base_url + "/v1/chat/completions",
            stream_payload,
            expect_substring=EXPECTED_MARKER,
        ),
    })

    status, body, elapsed = http_json("POST", base_url + "/v1/chat/completions", payload)
    probes.append(record_probe(
        "chat.non_stream", status, body, elapsed,
        generated_text=openai_chat_text(body),
        expect_substring=EXPECTED_MARKER,
        required=False,
    ))

    tool_payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": unique_prompt(
                "Return a short answer and include the word lucebox.",
                "chat-tools",
            ),
        }],
        "max_tokens": 16,
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }],
        "tool_choice": "auto",
    }
    status, body, elapsed = http_json("POST", base_url + "/v1/chat/completions", tool_payload)
    probes.append(record_probe("chat.tools_accepted", status, body, elapsed))
    return probes


def probe_openwebui(base_url: str, *, include_long: bool = False) -> list[dict[str, Any]]:
    probes = probe_openai_chat(base_url, include_long=include_long)
    status, body, elapsed = http_json("GET", base_url + "/v1/models")
    model_meta_ok = False
    if isinstance(body, dict):
        data = body.get("data") or []
        model_meta_ok = bool(data and data[0].get("context_length"))
    probes.append({
        "name": "openwebui.model_metadata",
        "ok": status == 200 and model_meta_ok,
        "status": status,
        "seconds": round(elapsed, 3),
        "body_tail": body if isinstance(body, str) else json.dumps(body)[-2000:],
    })
    return probes


def probe_anthropic_messages(base_url: str, *, include_long: bool = False) -> list[dict[str, Any]]:
    prompt = unique_prompt(
        long_prompt() if include_long else short_prompt(),
        "anthropic-non-stream",
    )
    probes = probe_health(base_url)
    payload = {
        "model": MODEL,
        "system": "You are Claude Code compatibility smoke-test traffic.",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
        "stop_sequences": ["\n\n\n"],
        # Claude Code sends tool metadata. server.py currently ignores extra
        # Anthropic fields, but the request must not fail validation.
        "tools": [{
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        }],
    }
    stream_payload = dict(payload)
    stream_payload["messages"] = [{
        "role": "user",
        "content": "Reply with exactly: lucebox-stream-ok two",
    }]
    stream_payload["stream"] = True
    probes.append({
        "name": "anthropic.messages_stream",
        **probe_anthropic_sse(base_url + "/v1/messages", stream_payload),
    })

    status, body, elapsed = http_json("POST", base_url + "/v1/messages", payload)
    probes.append(record_probe(
        "anthropic.messages", status, body, elapsed,
        generated_text=anthropic_text(body),
        expect_substring=EXPECTED_MARKER,
        required=False,
    ))
    return probes


def probe_anthropic_sse(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-api-key": "lucebox-local",
            "anthropic-version": "2023-06-01",
        },
    )
    t0 = time.perf_counter()
    first = None
    events = 0
    text_deltas = 0
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("event:"):
                    events += 1
                elif line.startswith("data:") and first is None:
                    first = time.perf_counter()
                if line.startswith("data:"):
                    try:
                        obj = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        delta = obj.get("delta") or {}
                        piece = delta.get("text") or delta.get("thinking") or ""
                        if piece:
                            text += str(piece)
                            text_deltas += 1
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "seconds": round(time.perf_counter() - t0, 3),
            "error": exc.read().decode("utf-8", errors="replace")[-4000:],
        }
    elapsed = time.perf_counter() - t0
    check = text_check(text, EXPECTED_MARKER)
    return {
        "ok": status == 200 and text_deltas > 0 and check["generated_ok"],
        "status": status,
        "seconds": round(elapsed, 3),
        "first_token_seconds": round(first - t0, 3) if first else None,
        "events": events,
        "text_deltas": text_deltas,
        **check,
    }


def probe_responses(base_url: str, *, include_long: bool = False) -> list[dict[str, Any]]:
    prompt = unique_prompt(
        long_prompt() if include_long else coding_prompt(),
        "responses-non-stream",
    )
    probes = []
    status, body, elapsed = http_json("GET", base_url + "/v1/models?client_version=harness-smoke")
    models_ok = isinstance(body, dict) and bool(body.get("models"))
    probes.append({
        "name": "codex.models",
        "ok": status == 200 and models_ok,
        "status": status,
        "seconds": round(elapsed, 3),
        "body_tail": body if isinstance(body, str) else json.dumps(body)[-2000:],
    })
    payload = {
        "model": MODEL,
        "instructions": "You are a concise coding agent.",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }],
        "tool_choice": "auto",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 32,
    }
    status, body, elapsed = http_json("POST", base_url + "/v1/responses", payload)
    probes.append(record_probe(
        "responses.non_stream", status, body, elapsed,
        generated_text=responses_text(body),
        expect_substring=EXPECTED_MARKER,
    ))

    stream_prompt = unique_prompt(
        long_prompt() if include_long else coding_prompt(),
        "responses-stream",
    )
    stream_payload = dict(payload)
    stream_payload["input"] = [{
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": stream_prompt}],
    }]
    stream_payload["stream"] = True
    probes.append({
        "name": "responses.stream",
        **probe_responses_sse(base_url + "/v1/responses", stream_payload),
    })
    return probes


def probe_responses_sse(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    first = None
    events = 0
    event_types: set[str] = set()
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("event:"):
                    events += 1
                    event_types.add(line[6:].strip())
                    if first is None:
                        first = time.perf_counter()
                elif line.startswith("data:"):
                    try:
                        obj = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        piece = obj.get("delta") or ""
                        if piece:
                            text += str(piece)
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "seconds": round(time.perf_counter() - t0, 3),
            "error": exc.read().decode("utf-8", errors="replace")[-4000:],
        }
    elapsed = time.perf_counter() - t0
    check = text_check(text, EXPECTED_MARKER)
    return {
        "ok": status == 200 and events > 0 and check["generated_ok"],
        "status": status,
        "seconds": round(elapsed, 3),
        "first_event_seconds": round(first - t0, 3) if first else None,
        "events": events,
        "event_types": sorted(event_types),
        **check,
    }


def probe_wrapper(base_url: str, *, include_long: bool = False) -> list[dict[str, Any]]:
    # harness.lol itself wraps child CLIs; the server-side compatibility risk is
    # already covered by the child protocols. Keep this cheap and explicit.
    probes = probe_health(base_url)
    probes.append({
        "name": "wrapper.protocols_delegated",
        "ok": True,
        "status": 200,
        "seconds": 0.0,
        "body_tail": "harness wraps claude/codex/opencode; run those probes too.",
    })
    return probes


PROBE_BY_PROTOCOL = {
    "openai_chat": probe_openai_chat,
    "openwebui": probe_openwebui,
    "anthropic_messages": probe_anthropic_messages,
    "responses": probe_responses,
    "wrapper": probe_wrapper,
}


def probe_required(probe: dict[str, Any]) -> bool:
    if "required" in probe:
        return bool(probe["required"])
    return probe.get("name") not in {"chat.non_stream", "anthropic.messages"}


def run_client_probe(
    base_url: str,
    spec: ClientSpec,
    *,
    work_dir: Path,
    include_long: bool,
    package_check: bool,
) -> dict[str, Any]:
    started = now_ms()
    package_result = package_smoke(work_dir, spec) if package_check else None
    probe_fn = PROBE_BY_PROTOCOL[spec.protocol]
    try:
        probes = probe_fn(base_url, include_long=include_long)
    except Exception as exc:
        probes = [{
            "name": "probe_exception",
            "ok": False,
            "status": 0,
            "seconds": 0.0,
            "body_tail": repr(exc),
        }]
    return {
        "client": spec.name,
        "protocol": spec.protocol,
        "package": spec.package,
        "package_smoke": package_result,
        "ok": all((not probe_required(p)) or p.get("ok") for p in probes) and (
            package_result is None or bool(package_result.get("ok"))
        ),
        "probes": probes,
        "started_ms": started,
        "ended_ms": now_ms(),
    }


def wait_http(base_url: str, proc: subprocess.Popen | None = None, timeout: int = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            status, _body, _elapsed = http_json("GET", base_url + "/health", timeout=2)
            if status == 200:
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, socket.timeout):
            pass
        time.sleep(1)
    return False


def gpu_mem() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], timeout=10)
    if not result["ok"]:
        return {"error": result["output_tail"]}
    rows = []
    for line in result["output_tail"].strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "name": parts[0],
                "memory_used_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]),
                "gpu_util_percent": int(parts[3]),
            })
    return {"gpus": rows}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def start_server(
    profile: ServerProfile,
    *,
    target: Path,
    draft: Path,
    bin_path: Path,
    prefill_drafter: Path | None,
    port: int,
    work_dir: Path,
) -> tuple[subprocess.Popen, Path, list[str], dict[str, str]]:
    log_dir = work_dir / "server-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{profile.name}-{int(time.time())}-{port}.log"
    args = [
        sys.executable,
        "-u",
        str(ROOT / "dflash" / "scripts" / "server.py"),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--target", str(target),
        "--draft", str(draft),
        "--bin", str(bin_path),
        *profile.args,
    ]
    if profile.needs_prefill_drafter:
        if prefill_drafter is None:
            raise HarnessError(f"profile {profile.name} requires --prefill-drafter")
        args.extend(["--prefill-drafter", str(prefill_drafter)])
    env = os.environ.copy()
    env.update(profile.env)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT / "dflash"),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Keep fd alive on proc object so it does not close immediately.
    proc._lucebox_log_f = log_f  # type: ignore[attr-defined]
    return proc, log_path, args, env


def close_server_log(proc: subprocess.Popen) -> None:
    log_f = getattr(proc, "_lucebox_log_f", None)
    if log_f:
        try:
            log_f.close()
        except Exception:
            pass


def tail(path: Path, n: int = 5000) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except FileNotFoundError:
        return ""


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        print(json.dumps(payload, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def cmd_install(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    selected = split_csv(args.clients, CLIENTS)
    results = [install_client(work_dir, CLIENTS[name]) for name in selected]
    payload = {"command": "install", "clients": results, "ok": all(r["ok"] for r in results)}
    write_json(args.json_out, payload)
    return 0 if payload["ok"] else 1


def cmd_probe(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    selected = split_csv(args.clients, CLIENTS)
    install_results = []
    if args.install_packages:
        for name in selected:
            install_results.append(install_client(work_dir, CLIENTS[name]))
    results = [
        run_client_probe(
            args.url.rstrip("/"),
            CLIENTS[name],
            work_dir=work_dir,
            include_long=args.long_prompt,
            package_check=args.install_packages or args.package_smoke,
        )
        for name in selected
    ]
    payload = {
        "command": "probe",
        "url": args.url.rstrip("/"),
        "install_results": install_results,
        "clients": results,
        "ok": all(r["ok"] for r in results),
        "gpu": gpu_mem(),
    }
    write_json(args.json_out, payload)
    return 0 if payload["ok"] else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    selected_clients = split_csv(args.clients, CLIENTS)
    selected_profiles = split_csv(args.profiles, SERVER_PROFILES)
    install_results = []
    if args.install_packages:
        for name in selected_clients:
            install_results.append(install_client(work_dir, CLIENTS[name]))

    all_profiles = []
    for profile_name in selected_profiles:
        profile = SERVER_PROFILES[profile_name]
        port = args.port or free_port()
        base_url = f"http://127.0.0.1:{port}"
        before_gpu = gpu_mem()
        proc = None
        log_path = None
        profile_result: dict[str, Any] = {
            "profile": profile.name,
            "base_url": base_url,
            "server_args": [],
            "before_gpu": before_gpu,
            "clients": [],
            "ok": False,
        }
        if args.isolate_clients:
            profile_result["isolated_clients"] = True
            for client_name in selected_clients:
                client_port = args.port or free_port()
                client_base_url = f"http://127.0.0.1:{client_port}"
                client_proc = None
                client_log_path = None
                try:
                    client_proc, client_log_path, server_args, _env = start_server(
                        profile,
                        target=args.target.resolve(),
                        draft=args.draft.resolve(),
                        bin_path=args.bin.resolve(),
                        prefill_drafter=args.prefill_drafter.resolve() if args.prefill_drafter else None,
                        port=client_port,
                        work_dir=work_dir,
                    )
                    up = wait_http(client_base_url, proc=client_proc, timeout=args.start_timeout)
                    if up:
                        client_result = run_client_probe(
                            client_base_url,
                            CLIENTS[client_name],
                            work_dir=work_dir,
                            include_long=profile.long_prompt or args.long_prompt,
                            package_check=args.install_packages or args.package_smoke,
                        )
                    else:
                        client_result = {
                            "client": client_name,
                            "ok": False,
                            "probes": [],
                            "server_started": False,
                        }
                    client_result["base_url"] = client_base_url
                    client_result["server_args"] = server_args
                    client_result["server_started"] = up
                    client_result["log_path"] = str(client_log_path)
                except Exception as exc:
                    client_result = {
                        "client": client_name,
                        "ok": False,
                        "probes": [],
                        "exception": repr(exc),
                    }
                    if client_log_path:
                        client_result["server_log_tail"] = tail(client_log_path)
                finally:
                    if client_proc is not None:
                        stop_proc(client_proc)
                        close_server_log(client_proc)
                        client_result["server_returncode"] = client_proc.poll()
                    client_result["after_gpu"] = gpu_mem()
                    if client_log_path and not client_result.get("server_log_tail"):
                        client_result["server_log_tail"] = tail(client_log_path)[-2000:]
                profile_result["clients"].append(client_result)
            profile_result["ok"] = all(c["ok"] for c in profile_result["clients"])
            profile_result["after_gpu"] = gpu_mem()
            all_profiles.append(profile_result)
            continue
        try:
            proc, log_path, server_args, _env = start_server(
                profile,
                target=args.target.resolve(),
                draft=args.draft.resolve(),
                bin_path=args.bin.resolve(),
                prefill_drafter=args.prefill_drafter.resolve() if args.prefill_drafter else None,
                port=port,
                work_dir=work_dir,
            )
            profile_result["server_args"] = server_args
            up = wait_http(base_url, proc=proc, timeout=args.start_timeout)
            profile_result["server_started"] = up
            profile_result["log_path"] = str(log_path)
            if not up:
                profile_result["server_returncode"] = proc.poll()
                profile_result["server_log_tail"] = tail(log_path)
            else:
                for client_name in selected_clients:
                    profile_result["clients"].append(
                        run_client_probe(
                            base_url,
                            CLIENTS[client_name],
                            work_dir=work_dir,
                            include_long=profile.long_prompt or args.long_prompt,
                            package_check=args.install_packages or args.package_smoke,
                        )
                    )
                profile_result["ok"] = all(c["ok"] for c in profile_result["clients"])
        except Exception as exc:
            profile_result["exception"] = repr(exc)
            if log_path:
                profile_result["server_log_tail"] = tail(log_path)
        finally:
            if proc is not None:
                stop_proc(proc)
                close_server_log(proc)
                profile_result["server_returncode"] = proc.poll()
            profile_result["after_gpu"] = gpu_mem()
            if log_path:
                log_tail = tail(log_path)
                if not profile_result.get("server_log_tail"):
                    profile_result["server_log_tail"] = log_tail[-2000:]
        all_profiles.append(profile_result)

    payload = {
        "command": "sweep",
        "install_results": install_results,
        "profiles": all_profiles,
        "ok": all(p.get("ok") for p in all_profiles),
    }
    write_json(args.json_out, payload)
    return 0 if payload["ok"] else 1


def cmd_list(_args: argparse.Namespace) -> int:
    print("clients:")
    for name, spec in CLIENTS.items():
        print(f"  {name:12s} {spec.install:5s} {spec.package:36s} {spec.protocol}")
    print("\nprofiles:")
    for name, profile in SERVER_PROFILES.items():
        marker = " pflash" if profile.needs_prefill_drafter else ""
        print(f"  {name:24s}{marker}  {' '.join(profile.args)}")
    return 0


def score_client(client: dict[str, Any]) -> dict[str, Any]:
    probes = client.get("probes") or []
    failed = [p.get("name") for p in probes if not p.get("ok")]
    failed_required = [
        p.get("name") for p in probes
        if probe_required(p) and not p.get("ok")
    ]
    decode_failed = [
        p.get("name") for p in probes
        if "generated_ok" in p and not p.get("generated_ok")
    ]
    required_decode_failed = [
        p.get("name") for p in probes
        if probe_required(p) and "generated_ok" in p and not p.get("generated_ok")
    ]
    seconds = sum(float(p.get("seconds") or 0.0) for p in probes)
    generated_chars = sum(int(p.get("generated_chars") or 0) for p in probes)
    return {
        "ok": not failed_required,
        "failed_probes": failed,
        "failed_required_probes": failed_required,
        "decode_failed_probes": decode_failed,
        "required_decode_failed_probes": required_decode_failed,
        "seconds": round(seconds, 3),
        "generated_chars": generated_chars,
    }


def cmd_report(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.json_in:
        data = json.loads(path.read_text())
        for profile in data.get("profiles") or []:
            for client in profile.get("clients") or []:
                score = score_client(client)
                rows.append({
                    "source": str(path),
                    "profile": profile.get("profile"),
                    "client": client.get("client"),
                    "server_args": client.get("server_args") or profile.get("server_args"),
                    **score,
                })

    by_client: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_client.setdefault(str(row["client"]), []).append(row)

    best: dict[str, Any] = {}
    for client, client_rows in sorted(by_client.items()):
        ranked = sorted(
            client_rows,
            key=lambda r: (
                not r["ok"],
                len(r["failed_required_probes"]),
                len(r["required_decode_failed_probes"]),
                len(r["failed_probes"]),
                r["seconds"],
            ),
        )
        best[client] = ranked[0] if ranked else None

    payload = {
        "command": "report",
        "inputs": [str(p) for p in args.json_in],
        "best_by_client": best,
        "rows": rows,
        "ok": all(bool(row["ok"]) for row in best.values()),
    }
    write_json(args.json_out, payload)
    if args.json_out is not None:
        for client, row in best.items():
            status = "PASS" if row and row["ok"] else "FAIL"
            profile = row.get("profile") if row else None
            seconds = row.get("seconds") if row else None
            print(f"{client:12s} {status:4s} {profile} {seconds}s")
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List known clients and profiles")
    p_list.set_defaults(func=cmd_list)

    p_install = sub.add_parser("install", help="Download client packages")
    p_install.add_argument("--clients", default="all")
    p_install.add_argument("--json-out", type=Path, default=None)
    p_install.set_defaults(func=cmd_install)

    p_probe = sub.add_parser("probe", help="Probe an already-running server")
    p_probe.add_argument("--url", default="http://127.0.0.1:8000")
    p_probe.add_argument("--clients", default="all")
    p_probe.add_argument("--install-packages", action="store_true")
    p_probe.add_argument("--package-smoke", action="store_true")
    p_probe.add_argument("--long-prompt", action="store_true")
    p_probe.add_argument("--json-out", type=Path, default=None)
    p_probe.set_defaults(func=cmd_probe)

    p_sweep = sub.add_parser("sweep", help="Start server profiles and probe them")
    p_sweep.add_argument("--target", type=Path, required=True)
    p_sweep.add_argument("--draft", type=Path, required=True)
    p_sweep.add_argument("--bin", type=Path, required=True)
    p_sweep.add_argument("--prefill-drafter", type=Path, default=None)
    p_sweep.add_argument("--profiles", default="rtx3090_dflash_fast,rtx3090_dflash_safe")
    p_sweep.add_argument("--clients", default="all")
    p_sweep.add_argument("--install-packages", action="store_true")
    p_sweep.add_argument("--package-smoke", action="store_true")
    p_sweep.add_argument("--long-prompt", action="store_true")
    p_sweep.add_argument("--isolate-clients", action="store_true",
                         help="Restart the server for each client probe")
    p_sweep.add_argument("--port", type=int, default=None)
    p_sweep.add_argument("--start-timeout", type=int, default=240)
    p_sweep.add_argument("--json-out", type=Path, default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_report = sub.add_parser("report", help="Summarize sweep JSON and pick per-client profiles")
    p_report.add_argument("json_in", nargs="+", type=Path)
    p_report.add_argument("--json-out", type=Path, default=None)
    p_report.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
