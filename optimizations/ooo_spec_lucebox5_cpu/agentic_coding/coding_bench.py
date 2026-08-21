#!/usr/bin/env python3
"""Correctness-gated A/B benchmark for coding-agent turn-cache reuse."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from coding_tools import (
    SKIP_DIRS,
    TOOLS,
    TOOL_NAMES,
    execute_tool,
    format_tool_result,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = SCRIPT_DIR / "fixtures" / "repo"
DEFAULT_TASKS = SCRIPT_DIR / "tasks.json"
DEFAULT_URL = "http://127.0.0.1:18145/v1/chat/completions"
MAX_CALLS_PER_TURN = 16
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SYSTEM = """You are a coding agent inspecting an unfamiliar repository.
Use the provided read-only tools before answering every task. Base the answer
only on returned repository evidence. Search for short literal identifiers,
not prose phrases. When tracing behavior, search for a symbol's callers and
inspect the relevant files. When the task names files, call read_file for those
exact paths. Omit optional bounds instead of sending empty values. Once the
evidence answers the task, stop calling tools and answer
concisely with exact symbol and path names. Never request a write or shell
command."""
NUDGE = (
    "If successful tool evidence already answers the task, give the concise "
    "final answer now. Otherwise call a repository tool; correct invalid "
    "arguments by omitting empty optional values."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="output tag under agentic_coding/results")
    parser.add_argument(
        "--run-id",
        default=os.environ.get("DFLASH_BENCH_RUN_ID") or secrets.token_hex(8),
        help="recorded cache-isolation nonce; defaults to a fresh value",
    )
    parser.add_argument("--url", default=os.environ.get("DFLASH_SERVER_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("DFLASH_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--tasks", type=int, default=0, help="0 runs every task")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="run only this task ID; repeat to select multiple tasks",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--first", choices=("control", "cache"), default="control")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--hardware-note", default=os.environ.get("DFLASH_BENCH_HARDWARE", "")
    )
    parser.add_argument(
        "--server-build-id", default=os.environ.get("DFLASH_SERVER_BUILD_ID", "")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for current_root, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories
            if name not in SKIP_DIRS and not name.startswith(".")
        )
        for filename in sorted(filenames):
            if not filename.startswith("."):
                paths.append(Path(current_root) / filename)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8"))
        else:
            digest.update(b"file\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_revision(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout != ""
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def validate_tasks(tasks: Any) -> list[dict[str, Any]]:
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("tasks file must contain a non-empty JSON array")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise SystemExit("every task must be a JSON object")
        task_id = task.get("id")
        prompt = task.get("prompt")
        if not isinstance(task_id, str) or not SAFE_TAG.fullmatch(task_id):
            raise SystemExit("every task id must be a short safe identifier")
        if task_id in seen:
            raise SystemExit(f"duplicate task id: {task_id}")
        seen.add(task_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"task {task_id} has no prompt")
        for field in ("answer_terms", "evidence_terms"):
            terms = task.get(field)
            if not isinstance(terms, list) or not terms or not all(
                isinstance(term, str) and term for term in terms
            ):
                raise SystemExit(f"task {task_id} has invalid {field}")
    return tasks


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def post(url: str, body: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, (time.perf_counter() - started) * 1000.0


def fetch_server_props(request_url: str, timeout: int) -> tuple[str, dict[str, Any]]:
    parsed = urllib.parse.urlsplit(request_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit("benchmark URL must be an absolute HTTP(S) URL")
    props_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/props", "", "")
    )
    try:
        with urllib.request.urlopen(props_url, timeout=timeout) as response:
            props = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot record server properties from {props_url}: {exc}") from exc
    if not isinstance(props, dict):
        raise SystemExit("server /props response must be a JSON object")
    return props_url, props


def _parse_calls(message: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return [("", "", None)]
    parsed = []
    for call in raw_calls:
        if not isinstance(call, dict):
            parsed.append(("", "", None))
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            parsed.append((call.get("id") or "", "", None))
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                arguments = None
        except (TypeError, json.JSONDecodeError):
            arguments = None
        parsed.append((call.get("id") or "", function.get("name") or "", arguments))
    return parsed


def _usage_timings(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    timings = usage.get("timings")
    return timings if isinstance(timings, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value >= 0 else None


def _trace_sha256(trace: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tool_trace_sha256(trace: list[dict[str, Any]]) -> str:
    trajectory = [
        {"tool_calls": event["tool_calls"]}
        for event in trace
        if event.get("tool_calls")
    ]
    return _trace_sha256(trajectory)


def pair_identifier(run_id: str, task_id: str, repetition: int) -> str:
    """Opaque prompt salt shared by both arms of one pair in one run."""
    material = f"{run_id}\0{task_id}\0{repetition}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def validate(task: dict[str, Any], final: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    final_folded = final.casefold()
    missing_answer = [term for term in task["answer_terms"] if term.casefold() not in final_folded]
    successful_results = [result for result in results if result.get("ok") is True]
    evidence = json.dumps(successful_results, sort_keys=True, ensure_ascii=False)
    missing_evidence = [term for term in task["evidence_terms"] if term not in evidence]
    successful_tools = len(successful_results)
    failed_tools = len(results) - successful_tools
    evidence_collected = successful_tools > 0
    return {
        "ok": (
            bool(final.strip())
            and not missing_answer
            and not missing_evidence
            and evidence_collected
        ),
        "missing_answer_terms": missing_answer,
        "missing_evidence_terms": missing_evidence,
        "successful_tool_calls": successful_tools,
        "failed_tool_calls": failed_tools,
        "successful_evidence_collected": evidence_collected,
        "all_tools_succeeded": failed_tools == 0 and evidence_collected,
    }


def run_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    arm: str,
    repetition: int,
) -> dict[str, Any]:
    pair_id = pair_identifier(args.run_id, task["id"], repetition)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                SYSTEM
                + "\nOpaque benchmark pair ID: "
                + pair_id
                + ". Ignore this ID in the answer."
            ),
        },
        {"role": "user", "content": task["prompt"]},
    ]
    turns: list[dict[str, Any]] = []
    canonical_trace: list[dict[str, Any]] = []
    evidence_results: list[dict[str, Any]] = []
    final = ""
    nudge_count = 0
    model_ms = 0.0
    tool_wait_ms = 0.0
    calls = 0
    eligible_followup_turns = 0
    agent_turn_cache_hits = 0
    unexpected_agent_turn_cache_hits = 0
    eligible_prefill_ms = 0.0
    eligible_model_wall_ms = 0.0
    eligible_prefilled_tokens = 0
    eligible_effective_prompt_tokens = 0
    timing_records_complete = True
    expect_agent_turn_cache = False
    started = time.perf_counter()

    for turn_index in range(args.max_turns):
        successful_evidence = any(
            result.get("ok") is True for result in evidence_results
        )
        body = {
            "model": args.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto" if successful_evidence else "required",
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
            "automatic_tool_speculation": False,
            "agent_turn_cache": arm == "cache",
        }
        response, request_ms = post(args.url, body, args.timeout)
        model_ms += request_ms
        timings = _usage_timings(response)
        prefill_ms = _number(timings.get("prefill_ms"))
        prefilled_tokens = _number(timings.get("prefilled_tokens"))
        effective_prompt_tokens = _number(timings.get("effective_prompt_tokens"))
        cache_hit_value = timings.get("agent_turn_cache_hit")
        cache_hit = cache_hit_value is True
        timing_complete = (
            prefill_ms is not None
            and prefilled_tokens is not None
            and effective_prompt_tokens is not None
            and isinstance(cache_hit_value, bool)
        )
        timing_records_complete = timing_records_complete and timing_complete
        cache_eligible = expect_agent_turn_cache
        if cache_eligible:
            eligible_followup_turns += 1
            eligible_model_wall_ms += request_ms
            if timing_complete:
                eligible_prefill_ms += prefill_ms
                eligible_prefilled_tokens += int(prefilled_tokens)
                eligible_effective_prompt_tokens += int(effective_prompt_tokens)
            if cache_hit:
                agent_turn_cache_hits += 1
        elif cache_hit:
            unexpected_agent_turn_cache_hits += 1
        choices = response.get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else {}
        ) or {}
        if not isinstance(message, dict):
            message = {}
        raw_calls = message.get("tool_calls") or []
        parsed = _parse_calls(message)
        turn: dict[str, Any] = {
            "turn": turn_index,
            "model_wall_ms": request_ms,
            "cache_eligible": cache_eligible,
            "timings": timings,
            "tools": [],
        }
        content = message.get("content")
        if not isinstance(content, str):
            content = ""
        reasoning = message.get("reasoning_content", message.get("reasoning", ""))
        if not isinstance(reasoning, str):
            reasoning = ""
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if isinstance(raw_calls, list) and raw_calls:
            assistant_message["tool_calls"] = raw_calls
        messages.append(assistant_message)
        trace_event: dict[str, Any] = {
            "assistant_reasoning": reasoning,
            "assistant_content": content,
            "tool_calls": [],
        }
        if not parsed:
            canonical_trace.append(trace_event)
            if evidence_results and content.strip():
                final = content
                turns.append(turn)
                break
            if nudge_count < 2 and turn_index + 1 < args.max_turns:
                nudge_count += 1
                messages.append({"role": "user", "content": NUDGE})
                turn["nudge_count"] = nudge_count
                turns.append(turn)
                continue
            final = content
            turns.append(turn)
            break
        nudge_count = 0

        if len(parsed) > MAX_CALLS_PER_TURN:
            turn["protocol_error"] = (
                f"model emitted {len(parsed)} calls; limit is {MAX_CALLS_PER_TURN}"
            )
            turns.append(turn)
            break

        call_ids = [call_id for call_id, _, _ in parsed]
        if any(not isinstance(call_id, str) or not call_id for call_id in call_ids) or (
            len(set(call_ids)) != len(call_ids)
        ):
            turn["protocol_error"] = "tool call IDs must be non-empty and unique"
            turns.append(turn)
            break

        calls += len(parsed)
        results: list[dict[str, Any] | None] = [None] * len(parsed)
        contents: list[str | None] = [None] * len(parsed)
        pending = list(range(len(parsed)))
        tool_started = time.perf_counter()

        def execute(index: int) -> tuple[int, dict[str, Any]]:
            _, name, arguments = parsed[index]
            if name not in TOOL_NAMES or arguments is None:
                return index, {"tool_name": name, "ok": False, "error": "invalid tool call"}
            return index, execute_tool(name, arguments, args.workspace)

        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                for index, result in pool.map(execute, pending):
                    results[index] = result
                    contents[index] = format_tool_result(result)
        wait_ms = (time.perf_counter() - tool_started) * 1000.0
        tool_wait_ms += wait_ms

        for index, (call_id, name, arguments) in enumerate(parsed):
            result = results[index]
            assert result is not None and contents[index] is not None
            evidence_results.append(result)
            turn["tools"].append(
                {
                    "name": name,
                    "arguments": arguments,
                    "source": "client_exec",
                    "ok": result.get("ok"),
                }
            )
            trace_event["tool_calls"].append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": contents[index],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": contents[index]}
            )
        canonical_trace.append(trace_event)
        expect_agent_turn_cache = True
        turn["tool_wait_ms"] = wait_ms
        turns.append(turn)

    wall_ms = (time.perf_counter() - started) * 1000.0
    validation = validate(task, final, evidence_results)
    return {
        "task": task["id"],
        "repetition": repetition,
        "pair_id": pair_id,
        "arm": arm,
        "prompt": task["prompt"],
        "final": final,
        "correct": validation["ok"],
        "validation": validation,
        "wall_ms": wall_ms,
        "model_ms": model_ms,
        "tool_wait_ms": tool_wait_ms,
        "calls": calls,
        "eligible_followup_turns": eligible_followup_turns,
        "agent_turn_cache_hits": agent_turn_cache_hits,
        "unexpected_agent_turn_cache_hits": unexpected_agent_turn_cache_hits,
        "eligible_prefill_ms": eligible_prefill_ms,
        "eligible_model_wall_ms": eligible_model_wall_ms,
        "eligible_prefilled_tokens": eligible_prefilled_tokens,
        "eligible_effective_prompt_tokens": eligible_effective_prompt_tokens,
        "timing_records_complete": timing_records_complete,
        "trace_sha256": _tool_trace_sha256(canonical_trace),
        "assistant_trace_sha256": _trace_sha256(canonical_trace),
        "trace": canonical_trace,
        "turns": len(turns),
        "turn_log": turns,
    }


def print_completion_summary(results: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["task"], result["repetition"]), {})[result["arm"]] = result
    complete_pairs = [
        arms
        for arms in grouped.values()
        if set(arms) == {"control", "cache"}
    ]
    correct_pairs = sum(
        pair["control"]["correct"] and pair["cache"]["correct"]
        for pair in complete_pairs
    )
    cache_eligible = sum(
        pair["cache"]["eligible_followup_turns"] for pair in complete_pairs
    )
    cache_hits = sum(
        pair["cache"]["agent_turn_cache_hits"] for pair in complete_pairs
    )
    tool_trace_matches = sum(
        pair["control"]["trace_sha256"] == pair["cache"]["trace_sha256"]
        for pair in complete_pairs
    )
    transcript_matches = sum(
        pair["control"]["assistant_trace_sha256"] ==
            pair["cache"]["assistant_trace_sha256"]
        for pair in complete_pairs
    )
    print(
        "\n=== RUN COMPLETION (not a performance claim)\n"
        f"complete_pairs={len(complete_pairs)}/{len(grouped)} "
        f"correct_pairs={correct_pairs}/{len(complete_pairs)} "
        f"tool_trace_matches={tool_trace_matches}/{len(complete_pairs)} "
        f"transcript_matches={transcript_matches}/{len(complete_pairs)} "
        f"agent_turn_cache_hits={cache_hits}/{cache_eligible}"
    )
    print("Run coding_summary.py to evaluate every pair and the publication gates.")


def main() -> int:
    args = parse_args()
    if not SAFE_TAG.fullmatch(args.tag):
        raise SystemExit("tag must use 1-64 letters, digits, dots, underscores, or dashes")
    if not SAFE_TAG.fullmatch(args.run_id):
        raise SystemExit("run ID must use 1-64 letters, digits, dots, underscores, or dashes")
    if args.max_turns <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--max-turns, --max-tokens, and --timeout must be positive")
    args.workspace = args.workspace.resolve(strict=True)
    if not args.workspace.is_dir():
        raise SystemExit("--workspace must be a directory")
    args.tasks_file = args.tasks_file.resolve(strict=True)
    with args.tasks_file.open(encoding="utf-8") as task_file:
        tasks = validate_tasks(json.load(task_file))
    if args.tasks < 0:
        raise SystemExit("--tasks cannot be negative")
    if args.tasks and args.task_id:
        raise SystemExit("--tasks and --task-id cannot be combined")
    if len(args.task_id) != len(set(args.task_id)):
        raise SystemExit("--task-id values must be unique")
    if args.task_id:
        selected = set(args.task_id)
        available = {task["id"] for task in tasks}
        missing = sorted(selected - available)
        if missing:
            raise SystemExit("unknown task ID: " + ", ".join(missing))
        tasks = [task for task in tasks if task["id"] in selected]
    if args.tasks > 0:
        tasks = tasks[: args.tasks]
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")

    props_url, server_props = fetch_server_props(args.url, args.timeout)

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"coding_{args.tag}.json"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"artifact already exists: {output_path}; pass --overwrite to replace it")
    results: list[dict[str, Any]] = []
    artifact: dict[str, Any] = {
        "schema": "lucebox.agent-turn-cache-benchmark.v1",
        "tag": args.tag,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_checkout": git_revision(SCRIPT_DIR.parents[2]),
        "workspace_revision": git_revision(args.workspace),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "model": args.model,
            "server_build_id": args.server_build_id or None,
            "hardware_note": args.hardware_note or None,
            "url": args.url,
            "props_url": props_url,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
        },
        "workspace": str(args.workspace),
        "server_props": server_props,
        "workspace_sha256": workspace_sha256(args.workspace),
        "tasks_file": str(args.tasks_file),
        "tasks_sha256": sha256_file(args.tasks_file),
        "tool_schema_sha256": hashlib.sha256(
            json.dumps(TOOLS, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "repetitions": args.repetitions,
        "task_count": len(tasks),
        "task_ids": [task["id"] for task in tasks],
        "arm_order": "alternating by task and repetition",
        "arm_isolation": (
            "identical initial prompts within each pair; exact normalized assistant/tool "
            "transcript parity and equal served token counts gate every follow-up; a "
            "recorded run-unique opaque pair ID prevents reuse from earlier pairs or "
            "benchmark invocations; control "
            "requests exclude generated-turn slots and treatment requests exclude "
            "ordinary control-origin snapshots; arm order alternates"
        ),
        "results": results,
    }
    for repetition in range(args.repetitions):
        for task_index, task in enumerate(tasks):
            control_first = (task_index + repetition) % 2 == 0
            if args.first == "cache":
                control_first = not control_first
            arms = ("control", "cache") if control_first else ("cache", "control")
            for arm in arms:
                result = run_task(args, task, arm, repetition)
                results.append(result)
                print(
                    f"{task['id']:18} rep={repetition} {arm:7} "
                    f"wall={result['wall_ms']/1000:7.2f}s calls={result['calls']:2} "
                    f"cache_hits={result['agent_turn_cache_hits']:2}/"
                    f"{result['eligible_followup_turns']:<2} "
                    f"correct={result['correct']}"
                )
                write_artifact(output_path, artifact)
    print_completion_summary(results)
    print(f"artifact: {output_path}")
    print(f"sha256: {sha256_file(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
