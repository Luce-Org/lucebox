"""Execute pre-generated validation cells against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
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
    saw_done = False
    saw_finish_reason = False
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="strict").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ValueError("stream returned a non-SSE data line")
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            break
        chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            raise ValueError("stream chunk is not a JSON object")
        chunks.append(chunk)
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason") is not None:
                saw_finish_reason = True
        if isinstance(chunk.get("usage"), dict):
            usage.update(chunk["usage"])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
    if not saw_done:
        raise ValueError("stream ended without [DONE]")
    if not saw_finish_reason:
        raise ValueError("stream ended without a finish reason")
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


def _assistant_output(raw_response: Any) -> dict[str, str]:
    """Reconstruct the assistant fields that carry generated model text.

    Output parity must include reasoning text as well as final content.  Quality
    scorers continue to consume final content separately.
    """

    if isinstance(raw_response, dict):
        choices = raw_response.get("choices") or []
        if len(choices) != 1:
            raise ValueError("response must contain exactly one choice")
        message = choices[0].get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content", "")
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError("response assistant text fields must be strings")
        return {"reasoning_content": reasoning, "content": content}
    if isinstance(raw_response, list):
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in raw_response:
            if not isinstance(chunk, dict):
                raise ValueError("stream chunk is not a JSON object")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content is not None:
                    if not isinstance(content, str):
                        raise ValueError("stream content delta is not a string")
                    content_parts.append(content)
                if reasoning is not None:
                    if not isinstance(reasoning, str):
                        raise ValueError("stream reasoning delta is not a string")
                    reasoning_parts.append(reasoning)
        return {
            "reasoning_content": "".join(reasoning_parts),
            "content": "".join(content_parts),
        }
    raise ValueError("response is neither a JSON object nor SSE chunks")


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


def request_props(*, endpoint_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch authenticated launch evidence from the server's root /props route."""

    parsed_url = urllib.parse.urlsplit(endpoint_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("endpoint_url must be an absolute HTTP(S) URL")
    props_url = urllib.parse.urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, "/props", "", "")
    )
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(props_url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        props = json.loads(response.read().decode("utf-8"))
    if not isinstance(props, dict):
        raise ValueError("/props response is not a JSON object")
    return props


def validate_speculative_evidence(
    *, usage: dict[str, Any], props: dict[str, Any], config: dict[str, Any]
) -> dict[str, str]:
    """Fail closed unless response and launch evidence prove the planned route."""

    expected_fields = (
        "expected_configured_contract",
        "expected_effective_decode_mode",
        "expected_fallback_reason",
        "expected_draft_attached",
        "expected_draft_loaded",
    )
    missing_expectations = [field for field in expected_fields if field not in config]
    if missing_expectations:
        raise ValueError(
            "configuration lacks speculative evidence expectations: "
            + ", ".join(missing_expectations)
        )

    speculative = usage.get("speculative")
    if not isinstance(speculative, dict):
        raise ValueError("response usage lacks nested speculative evidence")
    props_speculative = props.get("speculative")
    if not isinstance(props_speculative, dict):
        raise ValueError("/props lacks speculative launch evidence")

    evidence: dict[str, str] = {}
    for field in ("configured_contract", "effective_decode_mode", "fallback_reason"):
        value = speculative.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"response usage.speculative lacks {field}")
        evidence[field] = value

    expected_contract = str(config["expected_configured_contract"])
    expected_mode = str(config["expected_effective_decode_mode"])
    expected_fallback = str(config["expected_fallback_reason"])
    if evidence["configured_contract"] != expected_contract:
        raise ValueError("response configured contract does not match the matrix")
    if evidence["effective_decode_mode"] != expected_mode:
        raise ValueError("response effective decode mode does not match the matrix")
    if evidence["fallback_reason"] != expected_fallback:
        raise ValueError("response fallback reason does not match the matrix")
    if props_speculative.get("configured_contract") != expected_contract:
        raise ValueError("/props configured contract does not match the matrix")
    for field in ("draft_attached", "draft_loaded"):
        expected = config[f"expected_{field}"]
        if not isinstance(expected, bool):
            raise ValueError(f"expected_{field} must be a boolean")
        if props_speculative.get(field) is not expected:
            raise ValueError(f"/props {field} does not match the matrix")
    return evidence


def _response_timings(raw_response: Any) -> dict[str, Any]:
    """Return the server timing object from a JSON response or final SSE chunk."""

    candidates = (
        [raw_response]
        if isinstance(raw_response, dict)
        else list(reversed(raw_response))
        if isinstance(raw_response, list)
        else []
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("timings"), dict):
            return dict(candidate["timings"])
    raise ValueError("response lacks authoritative server timings")


def validate_llama_cpp_evidence(
    *, props: dict[str, Any], raw_response: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate observable upstream llama.cpp identity and speculative activity.

    llama.cpp does not currently expose the configured speculative route in
    ``/props``.  We therefore require an exact build/model identity match and
    prove speculative execution from per-response draft counters.  This is
    deliberately labelled externally unverified rather than exact.
    """

    expected_fields = (
        "expected_build_info",
        "expected_model_alias",
        "expected_speculative_type",
        "expected_configured_contract",
        "expected_effective_decode_mode",
        "expected_fallback_reason",
    )
    missing = [field for field in expected_fields if field not in config]
    if missing:
        raise ValueError(
            "llama.cpp configuration lacks evidence expectations: "
            + ", ".join(missing)
        )
    if props.get("build_info") != config["expected_build_info"]:
        raise ValueError("/props build_info does not match the matrix")
    if props.get("model_alias") != config["expected_model_alias"]:
        raise ValueError("/props model_alias does not match the matrix")

    speculative_type = str(config["expected_speculative_type"])
    allowed_types = {"none", "draft-dflash", "draft-mtp"}
    if speculative_type not in allowed_types:
        raise ValueError("unsupported expected_speculative_type")
    timings = _response_timings(raw_response)
    draft_n = timings.get("draft_n", 0)
    draft_n_accepted = timings.get("draft_n_accepted", 0)
    if not isinstance(draft_n, int) or not isinstance(draft_n_accepted, int):
        raise ValueError("llama.cpp draft counters must be integers")
    if draft_n < 0 or draft_n_accepted < 0 or draft_n_accepted > draft_n:
        raise ValueError("llama.cpp draft counters are inconsistent")

    if speculative_type == "none":
        if draft_n != 0:
            raise ValueError("target-only llama.cpp cell unexpectedly drafted tokens")
        configured_contract = "autoregressive"
        effective_mode = "autoregressive"
    else:
        if draft_n <= 0:
            raise ValueError("llama.cpp speculative cell produced no draft evidence")
        configured_contract = "external-unverified"
        effective_mode = "llama_cpp_" + speculative_type.removeprefix("draft-")
    fallback_reason = "none"

    expected = {
        "configured_contract": str(config["expected_configured_contract"]),
        "effective_decode_mode": str(config["expected_effective_decode_mode"]),
        "fallback_reason": str(config["expected_fallback_reason"]),
    }
    observed = {
        "configured_contract": configured_contract,
        "effective_decode_mode": effective_mode,
        "fallback_reason": fallback_reason,
    }
    for field, value in observed.items():
        if value != expected[field]:
            raise ValueError(f"llama.cpp {field} does not match the matrix")
    return {
        **observed,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
    }


def _artifact_name(config_id: str, attempt: int) -> str:
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in config_id)
    return safe[:210] + f".attempt-{attempt}.json"


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
    prompt_hash = hashlib.sha256(canonical_json(prompt).encode()).hexdigest()
    if prompt_hash != cell.get("prompt_hash"):
        raise ValueError("prompt row does not match the matrix prompt hash")
    endpoint = str(config["endpoint_url"]).rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": prompt.get("messages")
        or [{"role": "user", "content": str(prompt["prompt"])}],
        "max_tokens": int(config.get("max_tokens", 384)),
        "temperature": float(config.get("temperature", 0.0)),
        "stream": bool(config.get("stream", False)),
    }
    measurement_kind = str(config.get("measurement_kind", "natural_stop"))
    if measurement_kind not in {"fixed_work", "natural_stop"}:
        raise ValueError("measurement_kind must be fixed_work or natural_stop")
    if measurement_kind == "fixed_work":
        if config.get("ignore_eos") is not True:
            raise ValueError("fixed_work cells must explicitly request ignore_eos")
        payload["ignore_eos"] = True
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
        props = request_props(
            endpoint_url=str(config["endpoint_url"]), api_key=api_key, timeout=timeout
        )
        text, usage, raw_response, ttft_s, elapsed_s = request_chat(
            url=endpoint, api_key=api_key, payload=payload, timeout=timeout
        )
        assistant_output = _assistant_output(raw_response)
        if assistant_output["content"] != text:
            raise ValueError("reconstructed final content does not match the response")
        runtime_family = str(config.get("runtime_family", "lucebox"))
        if runtime_family == "lucebox":
            speculative: dict[str, Any] = validate_speculative_evidence(
                usage=usage, props=props, config=config
            )
            timings = usage.get("timings")
            if not isinstance(timings, dict):
                raise ValueError("response usage lacks authoritative server timings")
            decode_tok_s = timings.get("decode_tokens_per_sec")
            decode_ms = timings.get("decode_ms")
        elif runtime_family == "llama.cpp":
            speculative = validate_llama_cpp_evidence(
                props=props, raw_response=raw_response, config=config
            )
            timings = _response_timings(raw_response)
            decode_tok_s = timings.get("predicted_per_second")
            decode_ms = timings.get("predicted_ms")
        else:
            raise ValueError("runtime_family must be lucebox or llama.cpp")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            raise ValueError("response lacks an authoritative positive completion token count")
        if measurement_kind == "fixed_work" and completion_tokens != payload["max_tokens"]:
            raise ValueError("fixed_work response stopped before the requested token count")
        if not isinstance(decode_tok_s, (int, float)) or decode_tok_s <= 0:
            raise ValueError("response lacks a positive server decode rate")
        if not isinstance(decode_ms, (int, float)) or decode_ms <= 0:
            raise ValueError("response lacks a positive server decode duration")
        generated_bytes = canonical_json(assistant_output).encode()
        metrics = {
            "elapsed_s": elapsed_s,
            "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tok_s": float(decode_tok_s),
            "decode_ms": float(decode_ms),
            "measurement_kind": measurement_kind,
            "output_sha256": hashlib.sha256(generated_bytes).hexdigest(),
            "output_bytes": len(generated_bytes),
            "effective_decode_mode": speculative["effective_decode_mode"],
            "configured_contract": speculative["configured_contract"],
            "fallback_reason": speculative["fallback_reason"],
            "runtime_family": runtime_family,
            "props_sha256": hashlib.sha256(canonical_json(props).encode()).hexdigest(),
        }
        if "draft_n" in speculative:
            draft_n = int(speculative["draft_n"])
            draft_n_accepted = int(speculative["draft_n_accepted"])
            metrics.update(
                {
                    "draft_n": draft_n,
                    "draft_n_accepted": draft_n_accepted,
                    "draft_acceptance_rate": draft_n_accepted / draft_n
                    if draft_n
                    else None,
                }
            )
        artifact_path = artifacts_dir / _artifact_name(str(cell["config_id"]), attempt)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "config_id": cell["config_id"],
            "prompt_id": cell["prompt_id"],
            "prompt_hash": prompt_hash,
            "request": payload,
            "response": raw_response,
            "props": props,
            "assistant_output": assistant_output,
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
        prompt_hash=str(cell["prompt_hash"]),
        protocol_hash=str(cell["protocol_hash"]),
        cell_hash=str(cell["cell_hash"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="VALIDATION_API_KEY")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    cells = _jsonl(args.matrix)
    prompts = {str(row["id"]): row for row in _jsonl(args.prompts)}
    store = ResultStore(args.results)
    latest = store.latest()
    api_key = os.environ.get(args.api_key_env, "")
    if args.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    pending: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for cell in cells:
        config_id = str(cell["config_id"])
        previous = latest.get(config_id)
        required = (
            "decode_tok_s",
            "completion_tokens",
            "output_sha256",
            "configured_contract",
            "effective_decode_mode",
            "fallback_reason",
            "props_sha256",
        )
        if store.is_complete(
            config_id,
            config_hash=str(cell["config_hash"]),
            environment_hash=str(cell["environment_hash"]),
            prompt_hash=str(cell["prompt_hash"]),
            protocol_hash=str(cell["protocol_hash"]),
            cell_hash=str(cell["cell_hash"]),
            required_metrics=required,
        ):
            continue
        prompt_id = str(cell["prompt_id"])
        if prompt_id not in prompts:
            raise ValueError(f"matrix references missing prompt: {prompt_id}")
        planned_concurrency = int(cell["config"].get("concurrency", 1))
        if planned_concurrency != args.concurrency:
            raise ValueError(
                f"cell {config_id} requires concurrency={planned_concurrency}, "
                f"runner was started with {args.concurrency}"
            )
        attempt = 1 if previous is None else previous.attempt + 1
        pending.append((cell, prompts[prompt_id], attempt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                execute_cell,
                cell,
                prompt=prompt,
                api_key=api_key,
                timeout=args.timeout,
                artifacts_dir=args.artifacts_dir,
                attempt=attempt,
            )
            for cell, prompt, attempt in pending
        ]
        for (cell, _, attempt), future in zip(pending, futures, strict=True):
            row = future.result()
            store.append(row)
            config_id = str(cell["config_id"])
            latest[config_id] = row
            print(
                canonical_json(
                    {"config_id": config_id, "attempt": attempt, "status": row.status.value}
                )
            )


if __name__ == "__main__":
    main()
