"""Liveness smoke test.

Checks that the running server reports a healthy `/props` shape, streams text,
and can emit an OpenAI-format tool call. Semantic quality is the benchmark's
job; smoke proves the API surface is wired to the intended lucebox server.

Talks to the server via the host docker socket — the server container's port
is mapped to host port `cfg.port`, and the orchestrator container reaches it
via `host.docker.internal` on Docker Desktop or via the docker bridge
gateway on Linux. We resolve to `host.docker.internal` first and fall back.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass

import httpx

from lucebox.types import Config

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_PROMPT = "Reply with exactly one word: hello"


@dataclass(frozen=True, slots=True)
class SmokeResult:
    ok: bool
    http_status: int
    n_tokens: int
    wall_s: float
    props_ok: bool = False
    tool_ok: bool = False
    error: str = ""


def _server_base_url(cfg: Config) -> str:
    """Where to reach the server from inside the orchestrator container.

    Tries `host.docker.internal` (Docker Desktop, also added by recent
    docker-ce via --add-host), then falls back to the default bridge
    gateway 172.17.0.1.
    """
    host = "host.docker.internal"
    try:
        socket.gethostbyname(host)
    except OSError:
        host = "172.17.0.1"
    return f"http://{host}:{cfg.port}"


def _check_props(client: httpx.Client, base_url: str) -> tuple[bool, str]:
    try:
        resp = client.get(base_url + "/props")
    except httpx.HTTPError as e:
        return False, f"/props failed: {e}"
    if resp.status_code != 200:
        return False, f"/props HTTP {resp.status_code}"
    try:
        props = resp.json()
    except ValueError as e:
        return False, f"/props invalid JSON: {e}"
    required_top = (
        "default_generation_settings",
        "model_alias",
        "model_path",
        "build_info",
        "speculative_mode",
    )
    missing = [k for k in required_top if k not in props]
    if missing:
        return False, f"/props missing {', '.join(missing)}"
    dgs = props.get("default_generation_settings")
    if not isinstance(dgs, dict) or not all(
        k in dgs for k in ("n_ctx", "temperature", "top_p", "top_k", "min_p")
    ):
        return False, "/props default_generation_settings incomplete"
    runtime = props.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("backend"):
        return False, "/props runtime.backend missing"
    if props.get("speculative_mode") not in {"off", "mtp", "dflash", "pflash"}:
        return False, "/props speculative_mode invalid"
    return True, ""


def _check_tool_call(client: httpx.Client, base_url: str, timeout_s: float) -> tuple[bool, str]:
    body = {
        "model": "luce-dflash",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the provided tool now. Call report_status with "
                    'status="ok". Do not answer in prose.'
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "report_status",
                    "description": "Report smoke-test status.",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "report_status"}},
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err = ""
    for attempt in range(1, 4):
        try:
            resp = client.post(base_url + "/v1/chat/completions", json=body, timeout=timeout_s)
        except httpx.HTTPError as e:
            last_err = f"tool call request failed: {e}"
            continue
        if resp.status_code != 200:
            last_err = f"tool call HTTP {resp.status_code}: {resp.text[:300]}"
            continue
        try:
            data = resp.json()
        except ValueError as e:
            last_err = f"tool call invalid JSON: {e}"
            continue
        choices = data.get("choices") or []
        if not choices:
            last_err = "tool call response had no choices"
            continue
        msg = choices[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        if not calls:
            finish = choices[0].get("finish_reason")
            content = (msg.get("content") or "")[:300]
            last_err = (
                f"attempt {attempt}: no tool_calls emitted (finish={finish}, content={content!r})"
            )
            continue
        names = [((c.get("function") or {}).get("name")) for c in calls]
        if "report_status" in names:
            return True, ""
        last_err = f"attempt {attempt}: wrong tool call names: {names!r}"
    return False, last_err


def run(
    cfg: Config,
    *,
    prompt: str = DEFAULT_PROMPT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    check_tools: bool = True,
) -> SmokeResult:
    base_url = _server_base_url(cfg)
    url = base_url + "/v1/chat/completions"
    body = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "stream": True,
    }

    t0 = time.perf_counter()
    n_tokens = 0
    try:
        with httpx.Client(timeout=timeout_s) as client:
            props_ok, props_err = _check_props(client, base_url)
            if not props_ok:
                return SmokeResult(
                    ok=False,
                    http_status=0,
                    n_tokens=0,
                    wall_s=time.perf_counter() - t0,
                    props_ok=False,
                    tool_ok=False,
                    error=props_err,
                )

            with client.stream("POST", url, json=body) as resp:
                status = resp.status_code
                if status != 200:
                    return SmokeResult(
                        ok=False,
                        http_status=status,
                        n_tokens=0,
                        wall_s=time.perf_counter() - t0,
                        props_ok=props_ok,
                        tool_ok=False,
                        error=f"HTTP {status}",
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        n_tokens += 1
            if n_tokens < 1:
                return SmokeResult(
                    ok=False,
                    http_status=status,
                    n_tokens=n_tokens,
                    wall_s=time.perf_counter() - t0,
                    props_ok=props_ok,
                    tool_ok=False,
                    error="no tokens streamed",
                )

            tool_ok = True
            tool_err = ""
            if check_tools:
                tool_ok, tool_err = _check_tool_call(client, base_url, timeout_s)
            wall = time.perf_counter() - t0
            return SmokeResult(
                ok=(status == 200 and n_tokens >= 1 and props_ok and tool_ok),
                http_status=status,
                n_tokens=n_tokens,
                wall_s=wall,
                props_ok=props_ok,
                tool_ok=tool_ok,
                error=tool_err,
            )
    except httpx.HTTPError as e:
        return SmokeResult(
            ok=False,
            http_status=0,
            n_tokens=n_tokens,
            wall_s=time.perf_counter() - t0,
            error=str(e),
        )
