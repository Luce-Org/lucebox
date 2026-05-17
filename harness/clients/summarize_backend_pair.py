#!/usr/bin/env python3
"""Summarize one client harness Lucebox vs llama.cpp backend-pair run."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


LUCEBOX_DONE_RE = re.compile(r"(?:chat|responses|messages) DONE .*? in=(?P<prompt>\d+) out=(?P<out>\d+)")
LUCEBOX_DECODE_RE = re.compile(r"decode=[^(]*\((?P<tps>[0-9.]+)tok/s\)")
LUCEBOX_OVERALL_RE = re.compile(r"\s(?P<tps>[0-9.]+) tok/s\s+finish=")
LUCEBOX_FINISH_RE = re.compile(r"\sfinish=(?P<finish>\S+)")
LLAMA_PROMPT_RE = re.compile(
    r"prompt eval time\s*=.*?/\s*(?P<tokens>\d+)\s+tokens.*?,\s*(?P<tps>[0-9.]+)\s+tokens per second",
    re.I,
)
LLAMA_DECODE_RE = re.compile(
    r"^\s*eval time\s*=.*?/\s*(?P<tokens>\d+)\s+tokens.*?,\s*(?P<tps>[0-9.]+)\s+tokens per second",
    re.I | re.M,
)
PRIMARY_OUTPUT_NAMES = {
    "claude-code.out",
    "codex.out",
    "hermes.out",
    "opencode.out",
    "openclaw.out",
    "openwebui.out",
    "openwebui-tools.out",
    "pi.out",
}
MARKERS = ("OK_DONE", "lucebox-client-ok", "OPENWEBUI_TOOL_OK")
MISSING_JSON = object()


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_lucebox_calls(log: str) -> list[dict]:
    calls = []
    lines = [line for line in log.splitlines() if " DONE " in line]
    for line in lines:
        m = LUCEBOX_DONE_RE.search(line)
        if not m:
            continue
        decode = LUCEBOX_DECODE_RE.search(line)
        overall = LUCEBOX_OVERALL_RE.search(line)
        finish = LUCEBOX_FINISH_RE.search(line)
        calls.append(
            {
                "prompt_tokens": int(m.group("prompt")),
                "output_tokens": int(m.group("out")),
                "decode_tok_s": float((decode or overall).group("tps")) if (decode or overall) else None,
                "finish": finish.group("finish") if finish else "",
                "line": line.strip(),
            }
        )
    return calls


def parse_llama_calls(log: str) -> list[dict]:
    events = []
    for m in LLAMA_PROMPT_RE.finditer(log):
        events.append(
            (
                m.start(),
                "prompt",
                {
                    "prompt_tokens": int(m.group("tokens")),
                    "prompt_tok_s": float(m.group("tps")),
                },
            )
        )
    for m in LLAMA_DECODE_RE.finditer(log):
        events.append(
            (
                m.start(),
                "decode",
                {
                    "output_tokens": int(m.group("tokens")),
                    "decode_tok_s": float(m.group("tps")),
                },
            )
        )
    calls: list[dict] = []
    current: dict = {}
    for _, kind, values in sorted(events, key=lambda item: item[0]):
        if kind == "prompt":
            if current:
                calls.append(current)
            current = values
        elif current:
            current.update(values)
            calls.append(current)
            current = {}
        else:
            calls.append(values)
    if current:
        calls.append(current)
    return calls


def pick_primary_call(calls: list[dict]) -> dict:
    if not calls:
        return {}
    return max(calls, key=lambda call: (call.get("prompt_tokens", 0), call.get("output_tokens", 0)))


def primary_output_paths(run_dir: Path) -> list[Path]:
    preferred = [run_dir / name for name in sorted(PRIMARY_OUTPUT_NAMES)]
    found = [path for path in preferred if path.exists()]
    return found or sorted(run_dir.glob("*.out"))


def first_json_value(text: str):
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    if not stripped:
        return MISSING_JSON
    try:
        value, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return MISSING_JSON
    return value


def extract_generated_text(text: str) -> str:
    parts: list[str] = []
    value = first_json_value(text)
    if isinstance(value, dict):
        choices = value.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    parts.append(content)
                tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        fn = call.get("function", {}) if isinstance(call, dict) else {}
                        name = fn.get("name") if isinstance(fn, dict) else None
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        if name:
                            parts.append(f"tool_call:{name}")
                        if isinstance(args, str):
                            parts.append(args)
        payloads = value.get("payloads")
        if isinstance(payloads, list):
            for payload in payloads:
                if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                    parts.append(payload["text"])
        result = value.get("result")
        if isinstance(result, str):
            parts.append(result)
    elif isinstance(value, str):
        parts.append(value)
    elif value is None:
        return ""

    if not parts:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                part = event.get("part", {}) if isinstance(event, dict) else {}
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(event.get("message"), dict):
                    message = event["message"]
                    content = message.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                parts.append(item["text"])

    if not parts:
        # Plain-text clients such as Hermes write the generated answer directly.
        trimmed = []
        for line in text.splitlines():
            if line.startswith(("rc=", "model_server=", "run_dir=", "client_out=", "server_log=", "--- ")):
                break
            trimmed.append(line)
        parts.append("\n".join(trimmed).strip())
    return "\n".join(part for part in parts if part)


def marker_ok(text: str) -> bool:
    return any(marker in text for marker in MARKERS)


def tool_call_ok(text: str) -> bool:
    return "tool_call:" in text or '"tool_calls"' in text


def preview(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def find_run(pair_dir: Path, backend: str) -> Path | None:
    matches = sorted(p for p in pair_dir.iterdir() if p.is_dir() and p.name.endswith(f"-{backend}"))
    return matches[-1] if matches else None


def summarize_backend(pair_dir: Path, backend: str) -> dict:
    run_dir = find_run(pair_dir, backend)
    if run_dir is None:
        return {"backend": backend, "status": "missing"}
    server_log = read(run_dir / "server.log")
    output_paths = primary_output_paths(run_dir)
    client_text = "\n".join(read(p) for p in output_paths)
    generated_text = extract_generated_text(client_text)
    calls = parse_lucebox_calls(server_log) if backend == "lucebox" else parse_llama_calls(server_log)
    metrics = pick_primary_call(calls)
    rc = "unknown"
    backend_out = read(pair_dir / f"{backend}.out")
    m = re.search(r"^rc=(\d+)$", backend_out, flags=re.M)
    if m:
        rc = m.group(1)
    observed_tool_call = tool_call_ok(generated_text) or any(call.get("finish") == "tool_calls" for call in calls)
    return {
        "backend": backend,
        "run_dir": str(run_dir),
        "rc": rc,
        "marker_ok": marker_ok(generated_text),
        "tool_call_ok": observed_tool_call,
        "generated_preview": preview(generated_text),
        "calls": calls,
        **metrics,
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_backend_pair.py PAIR_DIR", file=sys.stderr)
        return 2
    pair_dir = Path(sys.argv[1])
    rows = [summarize_backend(pair_dir, "llamacpp"), summarize_backend(pair_dir, "lucebox")]
    by_backend = {row["backend"]: row for row in rows}
    llama_tps = by_backend.get("llamacpp", {}).get("decode_tok_s")
    luce_tps = by_backend.get("lucebox", {}).get("decode_tok_s")
    speed_valid = all((row.get("output_tokens") or 0) >= 32 for row in rows)
    speedup = (
        luce_tps / llama_tps
        if speed_valid and isinstance(luce_tps, float) and isinstance(llama_tps, float) and llama_tps
        else None
    )

    print("# Backend Pair Summary")
    print()
    print(f"Pair dir: `{pair_dir}`")
    if speedup is not None:
        print(f"Lucebox / llama.cpp decode speedup: **{speedup:.2f}x**")
    else:
        print("Lucebox / llama.cpp decode speedup: not reported for this run (too little generated output on one side).")
    print()
    print("| Backend | rc | marker | tool call | prompt tok | output tok | prompt tok/s | decode tok/s | calls | generated preview |")
    print("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            "| {backend} | {rc} | {marker} | {tool_call} | {prompt_tokens} | {output_tokens} | {prompt_tok_s} | {decode_tok_s} | {calls} | {preview} |".format(
                backend=row["backend"],
                rc=fmt(row.get("rc")),
                marker="yes" if row.get("marker_ok") else "no",
                tool_call="yes" if row.get("tool_call_ok") else "no",
                prompt_tokens=fmt(row.get("prompt_tokens")),
                output_tokens=fmt(row.get("output_tokens")),
                prompt_tok_s=fmt(row.get("prompt_tok_s")),
                decode_tok_s=fmt(row.get("decode_tok_s")),
                calls=len(row.get("calls", [])),
                preview=fmt(row.get("generated_preview", "")).replace("|", "\\|"),
            )
        )
    print()
    print("```json")
    print(json.dumps(rows, indent=2, sort_keys=True))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
