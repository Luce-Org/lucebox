"""Execute pre-generated validation cells against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .records import ResultStore, RunRow, RunStatus, canonical_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(raw)
    return rows


def parse_sse_lines(lines: list[bytes]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="strict").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ValueError("stream returned a non-SSE data line")
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            raise ValueError("stream chunk is not a JSON object")
        chunks.append(chunk)
        if isinstance(chunk.get("usage"), dict):
            usage.update(chunk["usage"])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
    return "".join(text_parts), usage, chunks


def _content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("non-streaming response must contain exactly one choice")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("response has no string final content")
    return content


def request_chat(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[str, dict[str, Any], Any, float | None, float]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    if payload.get("stream"):
        chunks: list[bytes] = []
        first_content_at: float | None = None
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            while True:
                line = response.readline()
                if not line:
                    break
                chunks.append(line)
                if first_content_at is None and b'"content"' in line and b'"content":""' not in line:
                    first_content_at = time.perf_counter()
        text, usage, parsed = parse_sse_lines(chunks)
        elapsed = time.perf_counter() - started
        ttft = None if first_content_at is None else first_content_at - started
        return text, usage, parsed, ttft, elapsed
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        parsed = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return _content(parsed), dict(parsed.get("usage") or {}), parsed, None, elapsed


def _artifact_name(config_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in config_id)
    return safe[:220] + ".json"


def execute_cell(
    cell: dict[str, Any],
    *,
    prompt: dict[str, Any],
    api_key: str,
    timeout: float,
    artifacts_dir: Path,
    attempt: int,
) -> RunRow:
    config = dict(cell["config"])
    endpoint = str(config["endpoint_url"]).rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": prompt.get("messages")
        or [{"role": "user", "content": str(prompt["prompt"])}],
        "max_tokens": int(config.get("max_tokens", 384)),
        "temperature": float(config.get("temperature", 0.0)),
        "stream": bool(config.get("stream", False)),
    }
    for key in ("seed", "top_p", "top_k", "min_p", "stop"):
        if key in config:
            payload[key] = config[key]

    started_at = dt.datetime.now(dt.UTC)
    status = RunStatus.SUCCESS
    return_code: int | None = 0
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    error: str | None = None
    try:
        text, usage, raw_response, ttft_s, elapsed_s = request_chat(
            url=endpoint, api_key=api_key, payload=payload, timeout=timeout
        )
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            raise ValueError("response lacks an authoritative positive completion token count")
        metrics = {
            "elapsed_s": elapsed_s,
            "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tok_s": completion_tokens / elapsed_s,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "output_bytes": len(text.encode()),
            "effective_decode_mode": usage.get("effective_decode_mode"),
            "configured_contract": usage.get("configured_contract"),
            "fallback_reason": usage.get("fallback_reason"),
        }
        artifact_path = artifacts_dir / _artifact_name(str(cell["config_id"]))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "config_id": cell["config_id"],
            "prompt_id": cell["prompt_id"],
            "request": payload,
            "response": raw_response,
            "final_content": text,
        }
        artifact_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
        artifacts = {
            "response": str(artifact_path),
            "response_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
    except TimeoutError as exc:
        status = RunStatus.TIMEOUT
        return_code = None
        error = type(exc).__name__
    except urllib.error.HTTPError as exc:
        status = RunStatus.FAILED
        return_code = exc.code
        error = f"HTTPError:{exc.code}"
    except Exception as exc:  # noqa: BLE001 - failures are append-only evidence
        status = RunStatus.FAILED
        return_code = 1
        error = type(exc).__name__
    finished_at = dt.datetime.now(dt.UTC)
    return RunRow(
        config_id=str(cell["config_id"]),
        attempt=attempt,
        status=status,
        config_hash=str(cell["config_hash"]),
        environment_hash=str(cell["environment_hash"]),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        return_code=return_code,
        metrics=metrics,
        artifacts=artifacts,
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="VALIDATION_API_KEY")
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    cells = _jsonl(args.matrix)
    prompts = {str(row["id"]): row for row in _jsonl(args.prompts)}
    store = ResultStore(args.results)
    latest = store.latest()
    api_key = os.environ.get(args.api_key_env, "")
    for cell in cells:
        config_id = str(cell["config_id"])
        previous = latest.get(config_id)
        required = ("decode_tok_s", "completion_tokens", "output_sha256")
        if store.is_complete(
            config_id,
            config_hash=str(cell["config_hash"]),
            environment_hash=str(cell["environment_hash"]),
            required_metrics=required,
        ):
            continue
        prompt_id = str(cell["prompt_id"])
        if prompt_id not in prompts:
            raise ValueError(f"matrix references missing prompt: {prompt_id}")
        attempt = 1 if previous is None else previous.attempt + 1
        row = execute_cell(
            cell,
            prompt=prompts[prompt_id],
            api_key=api_key,
            timeout=args.timeout,
            artifacts_dir=args.artifacts_dir,
            attempt=attempt,
        )
        store.append(row)
        latest[config_id] = row
        print(canonical_json({"config_id": config_id, "attempt": attempt, "status": row.status.value}))


if __name__ == "__main__":
    main()
