#!/usr/bin/env python3
"""Execute a trace-compiled read-only workflow through the tool-spec protocol.

The engine treats the compiled workflow like any other predicted tool: Qwen
proposes its typed arguments, DS4 remains authoritative, and the private result
is released only after an exact call match. Independent workflow branches run
concurrently inside the CPU-pinned executor process.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from benchmark_trace_compiled_workflows import CompiledPattern, load_training_traces, mine_pattern
from bfcl_replay_tool_executor import (
    PROTOCOL,
    call_ref,
    call_sha256,
    execute as execute_leaf,
)


MAX_BRANCHES = 4
TRAINING_REPORT_ENV = "DFLASH_TRACE_TRAINING_REPORT"
WORKFLOW_REGISTRY_ENV = "DFLASH_TRACE_WORKFLOW_REGISTRY"
LeafExecutor = Callable[[dict[str, Any]], dict[str, Any]]


def load_pattern() -> CompiledPattern:
    default = (
        Path(__file__).with_name("results")
        / "multiturn-cached-wordref-production-6tasks.json"
    )
    report = Path(os.environ.get(TRAINING_REPORT_ENV, str(default)))
    return mine_pattern(load_training_traces(report, required_steps=5))


def load_registry() -> dict[str, Any]:
    default = Path(__file__).with_name("results") / "trace-workflow-registry.json"
    path = Path(os.environ.get(WORKFLOW_REGISTRY_ENV, str(default)))
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("compiled workflow registry is invalid")
    return registry


def validate_items(items: Any, pattern: CompiledPattern) -> list[dict[str, str]]:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_BRANCHES:
        raise ValueError(f"customers must contain between 1 and {MAX_BRANCHES} items")
    expected_fields = set(pattern.root_fields)
    validated = []
    for item in items:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("customer fields do not match the compiled workflow")
        if not all(isinstance(value, str) and value for value in item.values()):
            raise ValueError("compiled workflow inputs must be non-empty strings")
        validated.append(dict(item))
    return validated


def resolve_items(
    arguments: Any, pattern: CompiledPattern, registry: dict[str, Any]
) -> list[dict[str, str]]:
    if not isinstance(arguments, dict) or set(arguments) != {"workflow_ref"}:
        raise ValueError("compiled workflow requires only a workflow_ref")
    workflow_ref = arguments["workflow_ref"]
    if not isinstance(workflow_ref, str) or re.fullmatch(
        r"workflow_[a-z]+", workflow_ref
    ) is None:
        raise ValueError("workflow_ref is malformed")
    if registry.get("pattern_fingerprint") != pattern.fingerprint:
        raise ValueError("workflow registry pattern does not match the executor")
    workflows = registry.get("workflows")
    entry = workflows.get(workflow_ref) if isinstance(workflows, dict) else None
    if (
        not isinstance(entry, dict)
        or entry.get("pattern_fingerprint") != pattern.fingerprint
    ):
        raise ValueError("workflow_ref is unknown or bound to another pattern")
    return validate_items(entry.get("items"), pattern)


def execute_branch(
    request: dict[str, Any],
    pattern: CompiledPattern,
    root: dict[str, str],
    leaf_executor: LeafExecutor,
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    steps = []
    for index in range(len(pattern.steps)):
        call = pattern.instantiate(root, previous, index)
        leaf_request = {**request, "call": call}
        envelope = leaf_executor(leaf_request)
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if (
            not isinstance(envelope, dict)
            or not envelope.get("ok")
            or not isinstance(result, dict)
        ):
            raise RuntimeError("leaf tool returned an invalid result")
        if (
            result.get("call_sha256") != call_sha256(call)
            or result.get("call_ref") != call_ref(call)
            or result.get("tool_name") != call["name"]
            or result.get("side_effects") is not False
        ):
            raise RuntimeError("leaf tool result did not match its compiled call")
        previous = result
        steps.append({"call": call, "tool_result": result})
    return {"root": root, "steps": steps, "final_ref": steps[-1]["tool_result"]["call_ref"]}


def execute_macro(
    request: dict[str, Any],
    pattern: CompiledPattern,
    leaf_executor: LeafExecutor = execute_leaf,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    call = request.get("call")
    if not isinstance(call, dict) or call.get("name") != pattern.macro_name:
        raise ValueError("request is not for the compiled workflow")
    items = resolve_items(
        call.get("arguments"), pattern, registry if registry is not None else load_registry()
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(items), thread_name_prefix="compiled-workflow") as pool:
        futures = [
            pool.submit(execute_branch, request, pattern, root, leaf_executor)
            for root in items
        ]
        branches = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "ok": True,
        "result": {
            "call_sha256": call_sha256(call),
            "call_ref": call_ref(call),
            "tool_name": pattern.macro_name,
            "workflow_fingerprint": pattern.fingerprint,
            "branches": branches,
            "call_count": len(items) * len(pattern.steps),
            "elapsed_ms": elapsed_ms,
            "cpu_affinity": affinity,
            "side_effects": False,
        },
    }


def execute(request: dict[str, Any], pattern: CompiledPattern) -> dict[str, Any]:
    call = request.get("call")
    if isinstance(call, dict) and call.get("name") == pattern.macro_name:
        return execute_macro(request, pattern)
    return execute_leaf(request)


def main() -> int:
    if sys.argv[1:] != ["--dflash-tool-spec-v1"]:
        print("expected --dflash-tool-spec-v1", file=sys.stderr)
        return 2
    try:
        line = sys.stdin.readline()
        if not line:
            raise ValueError("missing request")
        request = json.loads(line)
        if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
            raise ValueError("unsupported tool-speculation request")
        print(json.dumps(execute(request, load_pattern()), separators=(",", ":")), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
