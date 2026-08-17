#!/usr/bin/env python3
"""Benchmark no-training trace-compiled tool workflows on Lucebox5.

The benchmark compares three end-to-end agent paths over 10--20 real model
tool calls:

* ``stage_batched``: DS4 authorizes one batch per dependency stage and every
  call in that stage executes concurrently. This gives the control parallel
  tools without speculating or compiling the complete workflow.
* ``compiled``: a recurring, side-effect-free trace is exposed as one generated
  macro tool; DS4 authorizes it once and independent branches run in parallel.
* ``speculative``: the PR's Qwen predictor proposes the macro call and the
  engine starts its compiled graph before DS4 finishes. The private result is
  committed only after an exact name-and-arguments match.

The workflow compiler is learned from prior successful traces. It performs no
model training and refuses literals, side effects, ambiguous dataflow, or
inconsistent control flow. Every arm uses the production DS4+DSpark endpoint,
includes all model turns and tools in wall time, and must produce the same
underlying calls, tool results, and exact final answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_cpu_tool_speculation import (
    NATIVE_PREDICTION_SOURCE,
    get_json,
    normalize_tool_call,
    parse_cpu_list,
    percentile,
    post_json,
    props_url,
)
from bfcl_replay_tool_executor import (
    PROTOCOL,
    call_ref,
    call_sha256,
    canonical_call,
)


@dataclass(frozen=True)
class ArgumentBinding:
    source: str
    key: str


@dataclass(frozen=True)
class PatternStep:
    tool: str
    arguments: tuple[tuple[str, ArgumentBinding], ...]


@dataclass(frozen=True)
class CompiledPattern:
    steps: tuple[PatternStep, ...]
    root_fields: tuple[str, ...]
    training_traces: int

    @property
    def macro_name(self) -> str:
        # Semantic names are materially more reliable for tool-call generation
        # than opaque hashes. The registry remains keyed by the full pattern
        # fingerprint, so this user-facing alias does not define identity.
        subject = self.steps[0].tool.rsplit("_", 1)[-1]
        return f"execute_{subject}_workflows"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            [
                [
                    step.tool,
                    [[name, binding.source, binding.key] for name, binding in step.arguments],
                ]
                for step in self.steps
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def instantiate(
        self, root: dict[str, str], previous_result: dict[str, Any] | None, index: int
    ) -> dict[str, Any]:
        step = self.steps[index]
        arguments: dict[str, Any] = {}
        for name, binding in step.arguments:
            source: dict[str, Any]
            if binding.source == "root":
                source = root
            elif binding.source == "previous_result" and previous_result is not None:
                source = previous_result
            else:
                raise RuntimeError(
                    f"cannot resolve {binding.source}.{binding.key} for {step.tool}"
                )
            if binding.key not in source:
                raise RuntimeError(
                    f"missing {binding.source}.{binding.key} for {step.tool}"
                )
            arguments[name] = source[binding.key]
        return {"name": step.tool, "arguments": arguments}

    def simulate(self, root: dict[str, str]) -> list[dict[str, Any]]:
        calls = []
        previous: dict[str, Any] | None = None
        for index in range(len(self.steps)):
            call = self.instantiate(root, previous, index)
            calls.append(call)
            previous = simulated_tool_result(call)
        return calls

    def macro_tool(
        self, max_items: int, workflow_ref: str | None = None
    ) -> dict[str, Any]:
        if workflow_ref is not None:
            parameters = {
                "type": "object",
                "properties": {
                    "workflow_ref": {
                        "type": "string",
                        "enum": [workflow_ref],
                        "description": "Bound workflow instance for this request.",
                    }
                },
                "required": ["workflow_ref"],
                "additionalProperties": False,
            }
        else:
            properties = {field: {"type": "string"} for field in self.root_fields}
            parameters = {
                "type": "object",
                "properties": {
                    "customers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": properties,
                            "required": list(self.root_fields),
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": max_items,
                    }
                },
                "required": ["customers"],
                "additionalProperties": False,
            }
        return {
            "type": "function",
            "function": {
                "name": self.macro_name,
                "description": (
                    "Execute the validated five-step customer workflow independently "
                    "for every requested customer."
                ),
                "parameters": parameters,
            },
        }


def simulated_tool_result(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_ref": call_ref(call),
        "call_sha256": call_sha256(call),
        "tool_name": call["name"],
        "side_effects": False,
    }


def _infer_binding(
    value: Any,
    root: dict[str, str],
    previous_result: dict[str, Any] | None,
) -> ArgumentBinding:
    previous_matches = []
    if previous_result is not None:
        previous_matches = [key for key, candidate in previous_result.items() if candidate == value]
    root_matches = [key for key, candidate in root.items() if candidate == value]
    if len(previous_matches) == 1:
        return ArgumentBinding("previous_result", previous_matches[0])
    if len(root_matches) == 1:
        return ArgumentBinding("root", root_matches[0])
    raise ValueError("argument is literal or has ambiguous trace dataflow")


def mine_pattern(traces: list[dict[str, Any]]) -> CompiledPattern:
    if len(traces) < 2:
        raise ValueError("at least two successful traces are required")
    signatures = []
    root_fields: set[str] = set()
    for trace in traces:
        root = trace.get("root")
        calls = trace.get("calls")
        results = trace.get("results")
        if not isinstance(root, dict) or not isinstance(calls, list) or not isinstance(results, list):
            raise ValueError("trace must contain root, calls, and results")
        if not calls or len(calls) != len(results):
            raise ValueError("trace calls and results must be non-empty and aligned")
        signature = []
        previous: dict[str, Any] | None = None
        for call, result in zip(calls, results, strict=True):
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                raise ValueError("trace call is malformed")
            if not isinstance(result, dict) or result.get("side_effects") is not False:
                raise ValueError("only explicitly side-effect-free traces can be compiled")
            if result.get("call_sha256") != call_sha256(call):
                raise ValueError("trace result does not match its call")
            bindings = []
            for name, value in sorted(call["arguments"].items()):
                binding = _infer_binding(value, root, previous)
                if binding.source == "root":
                    root_fields.add(binding.key)
                bindings.append((name, binding))
            signature.append(PatternStep(str(call.get("name", "")), tuple(bindings)))
            previous = result
        signatures.append(tuple(signature))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("training traces do not share one control/data-flow pattern")
    if not root_fields:
        raise ValueError("compiled workflow exposes no request-bound arguments")
    return CompiledPattern(
        steps=signatures[0],
        root_fields=tuple(sorted(root_fields)),
        training_traces=len(traces),
    )


def load_training_traces(path: Path, required_steps: int) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    compact_traces = report.get("traces") if isinstance(report, dict) else None
    if isinstance(compact_traces, list):
        traces = [
            trace
            for trace in compact_traces
            if isinstance(trace, dict)
            and isinstance(trace.get("calls"), list)
            and len(trace["calls"]) == required_steps
        ]
        if len(traces) < 2:
            raise ValueError(
                "training trace file contains fewer than two complete traces"
            )
        return traces
    pairs = report.get("pairs") if isinstance(report, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("training report has no pairs")
    traces = []
    for pair in pairs:
        task = pair.get("task") if isinstance(pair, dict) else None
        control = pair.get("control") if isinstance(pair, dict) else None
        steps = control.get("steps") if isinstance(control, dict) else None
        if (
            not isinstance(task, dict)
            or not isinstance(steps, list)
            or len(steps) != required_steps
            or not control.get("all_calls_correct")
        ):
            continue
        traces.append(
            {
                "root": {
                    "customer_email": task["customer_email"],
                    "destination": task["destination"],
                },
                "calls": [step["call"] for step in steps],
                "results": [step["tool_result"] for step in steps],
            }
        )
    if len(traces) < 2:
        raise ValueError("training report contains fewer than two complete correct traces")
    return traces


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alphabetic_identifier(value: int) -> str:
    """Encode an integer with letters so copy accuracy is not biased by 0/o."""
    prefix = "warm" if value < 0 else "task"
    number = abs(value)
    encoded = ""
    while True:
        encoded = chr(ord("a") + number % 26) + encoded
        number = number // 26 - 1
        if number < 0:
            break
    return prefix + encoded


def make_task(
    index: int, branch_count: int, pattern: CompiledPattern
) -> dict[str, Any]:
    destinations = ("Rome", "Milan", "Turin", "Bologna", "Florence", "Naples")
    items = []
    used_refs = [set() for _ in pattern.steps]
    candidate = max(index, 0) * 100
    while len(items) < branch_count and candidate < max(index, 0) * 100 + 20_000:
        root = {
            "customer_email": (
                f"agent-{alphabetic_identifier(index)}-"
                f"{alphabetic_identifier(candidate)}@example.test"
            ),
            "destination": destinations[(index + candidate) % len(destinations)],
        }
        refs = [call_ref(call) for call in pattern.simulate(root)]
        if all(reference not in used_refs[step] for step, reference in enumerate(refs)):
            items.append(root)
            for step, reference in enumerate(refs):
                used_refs[step].add(reference)
        candidate += 1
    if len(items) != branch_count:
        raise RuntimeError("could not construct collision-free workflow branches")
    return {
        "id": f"trace_compiled_{index:03d}",
        "workflow_ref": f"workflow_{alphabetic_identifier(index)}",
        "items": items,
        "branch_count": branch_count,
        "call_count": branch_count * len(pattern.steps),
    }


def request_content(task: dict[str, Any]) -> str:
    rendered = "; ".join(
        f"{item['customer_email']} to {item['destination']}" for item in task["items"]
    )
    return f"Customers: {rendered}."


def workflow_reference(task: dict[str, Any], pattern: CompiledPattern) -> str:
    del pattern
    workflow_ref = task.get("workflow_ref")
    if not isinstance(workflow_ref, str) or re.fullmatch(
        r"workflow_[a-z]+", workflow_ref
    ) is None:
        raise ValueError("task has no valid request-scoped workflow_ref")
    return workflow_ref


def write_workflow_registry(
    path: Path, pattern: CompiledPattern, tasks: list[dict[str, Any]]
) -> None:
    workflows = {
        workflow_reference(task, pattern): {
            "pattern_fingerprint": pattern.fingerprint,
            "items": task["items"],
        }
        for task in tasks
    }
    if len(workflows) != len(tasks):
        raise ValueError("workflow_ref collision in request-scoped registry")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pattern_fingerprint": pattern.fingerprint,
                "workflows": workflows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_request_customers(content: str) -> list[dict[str, str]]:
    prefix = "Customers: "
    if not content.startswith(prefix) or not content.endswith("."):
        raise ValueError("request does not use the validated customer-list format")
    customers = []
    for entry in content[len(prefix) : -1].split("; "):
        match = re.fullmatch(r"([^\s;]+) to ([A-Za-z][A-Za-z -]*)", entry)
        if match is None:
            raise ValueError("request customer entry is malformed")
        customers.append(
            {"customer_email": match.group(1), "destination": match.group(2)}
        )
    if not customers:
        raise ValueError("request contains no customers")
    return customers


def stage_batch_tool(
    pattern: CompiledPattern,
    step_index: int,
    max_items: int,
    stage_ref: str | None = None,
) -> dict[str, Any]:
    step = pattern.steps[step_index]
    if stage_ref is not None:
        parameters = {
            "type": "object",
            "properties": {
                "stage_ref": {
                    "type": "string",
                    "enum": [stage_ref],
                    "description": "Bound batch for this workflow stage.",
                }
            },
            "required": ["stage_ref"],
            "additionalProperties": False,
        }
    else:
        properties = {name: {"type": "string"} for name, _ in step.arguments}
        parameters = {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": max_items,
                }
            },
            "required": ["calls"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": f"batch_{step.tool}",
            "description": (
                f"Run {step.tool} once for every item. Calls execute concurrently "
                "and results preserve input order."
            ),
            "parameters": parameters,
        },
    }


def stage_batched_messages(
    task: dict[str, Any], pattern: CompiledPattern
) -> list[dict[str, Any]]:
    del pattern
    system = (
        "The scheduler exposes exactly one currently-ready batch tool at a time. "
        "Call only that declared tool once and copy its bound stage_ref exactly. "
        "The bound batch contains every customer in input order. Do not invent or "
        "name future tools, and emit no prose."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": request_content(task)},
    ]


def macro_messages(task: dict[str, Any], pattern: CompiledPattern) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "Use the workflow tool exactly once for every requested customer. "
                "Copy its bound workflow_ref exactly. No prose."
            ),
        },
        {"role": "user", "content": request_content(task)},
    ]


def stage_reference(task: dict[str, Any], pattern: CompiledPattern, index: int) -> str:
    stage_names = ("one", "two", "three", "four", "five")
    if not 0 <= index < len(stage_names):
        raise ValueError("stage index is outside the compiled workflow")
    return f"{workflow_reference(task, pattern)}_stage_{stage_names[index]}"


def model_observation(result: dict[str, Any], wall_ms: float) -> dict[str, Any]:
    choices = result.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    if not isinstance(message, dict):
        raise RuntimeError("model response has no assistant message")
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        raw_calls = []
    calls = []
    for raw in raw_calls:
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict):
            raise RuntimeError("model emitted a malformed tool call")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(function.get("name"), str) or not isinstance(arguments, dict):
            raise RuntimeError("model emitted invalid tool name or arguments")
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError("model tool call has no id")
        calls.append(
            {
                "id": call_id,
                "call": {"name": function["name"], "arguments": arguments},
            }
        )
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    timings = usage.get("timings") if isinstance(usage.get("timings"), dict) else {}
    assistant_message = dict(message)
    assistant_message["role"] = "assistant"
    assistant_message["content"] = content
    content_format_call = False
    if not calls and content:
        parsed_call = normalize_tool_call(result)
        if parsed_call is not None:
            content_format_call = True
            call_id = "call_content_" + hashlib.sha256(
                canonical_call(parsed_call).encode()
            ).hexdigest()[:16]
            calls.append({"id": call_id, "call": parsed_call})
            assistant_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": parsed_call["name"],
                            "arguments": json.dumps(
                                parsed_call["arguments"],
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
    return {
        "request_wall_ms": wall_ms,
        "model_compute_ms": float(timings.get("prefill_ms", 0.0))
        + float(timings.get("decode_ms", 0.0)),
        "prefill_ms": float(timings.get("prefill_ms", 0.0)),
        "decode_ms": float(timings.get("decode_ms", 0.0)),
        "decode_tokens_per_sec": float(timings.get("decode_tokens_per_sec", 0.0)),
        "cache_hit": bool(timings.get("cache_hit", False)),
        "cached_prefix_tokens": int(timings.get("cached_prefix_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "accept_rate": float(usage.get("accept_rate", 0.0)),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "assistant_message": assistant_message,
        "calls": calls,
        "content_format_call": content_format_call,
    }


def post_turn(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any,
    max_tokens: int,
    *,
    automatic_tool_speculation: bool = False,
) -> dict[str, Any]:
    result, wall_ms = post_json(
        args.url,
        {
            "model": "dflash",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": max_tokens,
            "stream": False,
            "automatic_tool_speculation": automatic_tool_speculation,
        },
        args.timeout,
    )
    observation = model_observation(result, wall_ms)
    observation["speculation"] = result.get("dflash_tool_speculation")
    return observation


def execute_tool_safe(
    binary: Path,
    call: dict[str, Any],
    cpus: list[int],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    request = {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "resource_percentage": 100,
        "accelerator_relation": "non_accelerator",
        "cpu_affinity": cpus,
        "cpu_affinity_isolated": True,
        "call": call,
    }
    command = [str(binary), "--dflash-tool-spec-v1"]
    if os.name == "posix" and Path("/usr/bin/taskset").is_file():
        command = ["/usr/bin/taskset", "-c", ",".join(map(str, cpus)), *command]
    started = time.perf_counter()
    process = subprocess.run(
        command,
        input=json.dumps(request, separators=(",", ":")) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1_000.0
    if process.returncode != 0:
        raise RuntimeError(
            f"tool executor exited {process.returncode}: {process.stderr.strip()}"
        )
    envelope = json.loads(process.stdout)
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not envelope.get("ok") or not isinstance(result, dict):
        raise RuntimeError(f"tool executor returned invalid data: {envelope!r}")
    if (
        result.get("call_sha256") != call_sha256(call)
        or result.get("call_ref") != call_ref(call)
        or result.get("tool_name") != call.get("name")
        or result.get("side_effects") is not False
    ):
        raise RuntimeError("tool result does not exactly match the read-only call")
    return {"wall_ms": wall_ms, "result": result}


def run_branch(
    args: argparse.Namespace,
    pattern: CompiledPattern,
    root: dict[str, str],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    steps = []
    previous: dict[str, Any] | None = None
    for index in range(len(pattern.steps)):
        call = pattern.instantiate(root, previous, index)
        tool = execute_tool_safe(
            args.binary,
            call,
            args.tool_cpus,
            args.timeout,
            f"{label}-step-{index + 1}",
        )
        previous = tool["result"]
        steps.append({"call": call, "tool_result": previous, "tool_wall_ms": tool["wall_ms"]})
    return {
        "root": root,
        "steps": steps,
        "final_ref": steps[-1]["tool_result"]["call_ref"],
        "wall_ms": (time.perf_counter() - started) * 1_000.0,
    }


def start_graph(
    args: argparse.Namespace,
    pattern: CompiledPattern,
    items: list[dict[str, str]],
    label: str,
) -> dict[str, Any]:
    pool = ThreadPoolExecutor(max_workers=len(items), thread_name_prefix="tool-graph")
    started = time.perf_counter()
    futures: list[Future[dict[str, Any]]] = [
        pool.submit(run_branch, args, pattern, root, f"{label}-branch-{index}")
        for index, root in enumerate(items)
    ]
    return {"pool": pool, "futures": futures, "started": started}


def finish_graph(handle: dict[str, Any], timeout: float) -> dict[str, Any]:
    pool: ThreadPoolExecutor = handle["pool"]
    futures: list[Future[dict[str, Any]]] = handle["futures"]
    try:
        branches = [future.result(timeout=timeout) for future in futures]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {
        "branches": branches,
        "wall_ms": (time.perf_counter() - float(handle["started"])) * 1_000.0,
    }


def expected_final(branches: list[dict[str, Any]]) -> str:
    return "workflow_complete:" + ",".join(branch["final_ref"] for branch in branches)


def final_answer_correct(content: str, expected: str) -> bool:
    """Accept the receipt literally or as the equivalent one-field JSON object."""
    content = content.strip()
    if content == expected:
        return True
    prefix = "workflow_complete:"
    if not expected.startswith(prefix):
        return False
    receipt = expected[len(prefix) :]
    if content == receipt:
        return True
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False
    return parsed == {"workflow_complete": receipt}


def post_final(
    args: argparse.Namespace,
    expected: str,
) -> dict[str, Any]:
    """Measure the same minimal final-response turn for every benchmark arm."""
    return post_turn(
        args,
        [
            {
                "role": "system",
                "content": (
                    "Return the opaque workflow receipt from the user message exactly. "
                    "Do not explain, reformat, or add any characters."
                ),
            },
            {"role": "user", "content": expected},
        ],
        [],
        "none",
        args.final_max_tokens,
    )


def flatten_graph_calls(graph: dict[str, Any]) -> list[str]:
    return [
        canonical_call(step["call"])
        for branch in graph["branches"]
        for step in branch["steps"]
    ]


def flatten_graph_results(graph: dict[str, Any]) -> list[str]:
    return [
        step["tool_result"]["call_sha256"]
        for branch in graph["branches"]
        for step in branch["steps"]
    ]


def run_stage_batched(
    args: argparse.Namespace,
    task: dict[str, Any],
    pattern: CompiledPattern,
    label: str,
) -> dict[str, Any]:
    """Run a strong non-speculative baseline with parallel calls per stage."""
    started = time.perf_counter()
    stage_messages = stage_batched_messages(task, pattern)
    current_tools: list[dict[str, Any]] = []
    branches = [
        {"root": root, "steps": [], "final_ref": ""} for root in task["items"]
    ]
    previous_results: list[dict[str, Any] | None] = [None] * len(branches)
    turns = []
    exposed_tool_wait_ms = 0.0

    for step_index, step in enumerate(pattern.steps):
        expected_calls = [
            pattern.instantiate(branch["root"], previous_results[index], step_index)
            for index, branch in enumerate(branches)
        ]
        batch_name = f"batch_{step.tool}"
        stage_ref = stage_reference(task, pattern, step_index)
        current_tools = [
            stage_batch_tool(pattern, step_index, args.max_branches, stage_ref)
        ]
        expected_batch = {
            "name": batch_name,
            "arguments": {"stage_ref": stage_ref},
        }
        model = post_turn(
            args,
            stage_messages,
            current_tools,
            {"type": "function", "function": {"name": batch_name}},
            args.call_max_tokens,
        )
        if len(model["calls"]) != 1 or model["calls"][0]["call"] != expected_batch:
            raise RuntimeError(
                f"{task['id']} stage {step_index + 1}: batch call "
                f"{[item['call'] for item in model['calls']]!r} != "
                f"{expected_batch!r}; content={model['content']!r}"
            )
        stage_started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=len(expected_calls), thread_name_prefix="batched-tool-stage"
        ) as pool:
            futures = [
                pool.submit(
                    execute_tool_safe,
                    args.binary,
                    call,
                    args.tool_cpus,
                    args.timeout,
                    f"{label}-stage-{step_index}-branch-{branch_index}",
                )
                for branch_index, call in enumerate(expected_calls)
            ]
            stage_tools = [future.result(timeout=args.timeout) for future in futures]
        exposed_tool_wait_ms += (time.perf_counter() - stage_started) * 1_000.0

        for branch_index, (call, tool) in enumerate(
            zip(expected_calls, stage_tools, strict=True)
        ):
            result = tool["result"]
            previous_results[branch_index] = result
            branches[branch_index]["steps"].append(
                {
                    "call": call,
                    "tool_result": result,
                    "tool_wall_ms": tool["wall_ms"],
                }
            )
        turns.append(model)

    for branch in branches:
        branch["final_ref"] = branch["steps"][-1]["tool_result"]["call_ref"]
    expected = expected_final(branches)
    final = post_final(args, expected)
    all_turns = [*turns, final]
    graph = {"branches": branches}
    return {
        "task_ms": (time.perf_counter() - started) * 1_000.0,
        "model_turns": len(all_turns),
        "call_turns": turns,
        "final": final,
        "expected_final": expected,
        "final_correct": final_answer_correct(final["content"], expected),
        "graph": graph,
        "underlying_calls": flatten_graph_calls(graph),
        "tool_results": flatten_graph_results(graph),
        "model_compute_ms": sum(turn["model_compute_ms"] for turn in all_turns),
        "decode_ms": sum(turn["decode_ms"] for turn in all_turns),
        "completion_tokens": sum(turn["completion_tokens"] for turn in all_turns),
        "exposed_tool_wait_ms": exposed_tool_wait_ms,
        "all_ds4_active": all(turn["accept_rate"] > 0.0 for turn in turns),
    }


def macro_result_message(
    pattern: CompiledPattern,
    graph: dict[str, Any],
    tool_call_id: str,
) -> dict[str, Any]:
    content = {
        "workflow": pattern.macro_name,
        "call_count": sum(len(branch["steps"]) for branch in graph["branches"]),
        "items": [
            {
                **branch["root"],
                "final_ref": branch["final_ref"],
            }
            for branch in graph["branches"]
        ],
        "side_effects": False,
    }
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": pattern.macro_name,
        "content": json.dumps(content, sort_keys=True, separators=(",", ":")),
    }


def graph_from_speculative_result(
    metadata: dict[str, Any], pattern: CompiledPattern, call: dict[str, Any]
) -> dict[str, Any]:
    result = metadata.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("engine speculation hit has no compiled workflow result")
    branches = result.get("branches")
    if (
        result.get("call_sha256") != call_sha256(call)
        or result.get("call_ref") != call_ref(call)
        or result.get("tool_name") != pattern.macro_name
        or result.get("workflow_fingerprint") != pattern.fingerprint
        or result.get("side_effects") is not False
        or not isinstance(branches, list)
    ):
        raise RuntimeError("engine returned an invalid compiled workflow result")
    return {"branches": branches, "wall_ms": float(result.get("elapsed_ms", 0.0))}


def run_macro(
    args: argparse.Namespace,
    task: dict[str, Any],
    pattern: CompiledPattern,
    speculative: bool,
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    messages = macro_messages(task, pattern)
    workflow_ref = workflow_reference(task, pattern)
    tools = [pattern.macro_tool(args.max_branches, workflow_ref)]
    parsed_items = parse_request_customers(messages[-1]["content"])
    if parsed_items != task["items"]:
        raise RuntimeError("event extractor did not reproduce structured request data")
    expected_call = {
        "name": pattern.macro_name,
        "arguments": {"workflow_ref": workflow_ref},
    }
    model = post_turn(
        args,
        messages,
        tools,
        "required",
        args.macro_max_tokens,
        automatic_tool_speculation=speculative,
    )
    if len(model["calls"]) != 1:
        raise RuntimeError(f"{task['id']}: macro turn emitted {len(model['calls'])} calls")
    emitted = model["calls"][0]
    macro_correct = emitted["call"] == expected_call
    if not macro_correct:
        raise RuntimeError(
            f"{task['id']}: macro call {emitted['call']!r} != {expected_call!r}"
        )

    metadata = model.get("speculation") if speculative else None
    if speculative and not isinstance(metadata, dict):
        raise RuntimeError("automatic Qwen speculation returned no engine metadata")
    prediction = metadata.get("prediction") if isinstance(metadata, dict) else None
    prediction_hit = (
        speculative
        and metadata.get("status") == "hit"
        and prediction == expected_call
    )
    predictor_ms = (
        float(metadata.get("predictor_wall_ms", 0.0))
        if isinstance(metadata, dict)
        else 0.0
    )
    if prediction_hit:
        graph = graph_from_speculative_result(metadata, pattern, emitted["call"])
        exposed_wait_ms = float(metadata.get("commit_wait_ms", 0.0))
    else:
        graph_handle = start_graph(args, pattern, task["items"], f"{label}-authoritative")
        wait_started = time.perf_counter()
        graph = finish_graph(graph_handle, args.timeout)
        exposed_wait_ms = (time.perf_counter() - wait_started) * 1_000.0
    expected_calls = [
        canonical_call(call)
        for root in task["items"]
        for call in pattern.simulate(root)
    ]
    actual_calls = flatten_graph_calls(graph)
    if actual_calls != expected_calls:
        raise RuntimeError(f"{task['id']}: compiled graph diverged from learned pattern")
    messages.extend(
        [
            model["assistant_message"],
            macro_result_message(pattern, graph, emitted["id"]),
        ]
    )
    expected = expected_final(graph["branches"])
    final = post_final(args, expected)
    all_turns = [model, final]
    return {
        "task_ms": (time.perf_counter() - started) * 1_000.0,
        "model_turns": len(all_turns),
        "call_turns": [model],
        "macro_call": emitted["call"],
        "macro_correct": macro_correct,
        "prediction_hit": prediction_hit,
        "prediction_source": (
            metadata.get("prediction_source") if isinstance(metadata, dict) else None
        ),
        "prediction_status": metadata.get("status") if isinstance(metadata, dict) else None,
        "prediction_reason": metadata.get("reason") if isinstance(metadata, dict) else None,
        "predictor_ms": predictor_ms,
        "graph": graph,
        "graph_wall_ms": graph["wall_ms"],
        "exposed_tool_wait_ms": exposed_wait_ms,
        "underlying_calls": actual_calls,
        "tool_results": flatten_graph_results(graph),
        "final": final,
        "expected_final": expected,
        "final_correct": final_answer_correct(final["content"], expected),
        "model_compute_ms": sum(turn["model_compute_ms"] for turn in all_turns),
        "decode_ms": sum(turn["decode_ms"] for turn in all_turns),
        "completion_tokens": sum(turn["completion_tokens"] for turn in all_turns),
        "all_ds4_active": model["accept_rate"] > 0.0,
    }


def macro_signature(arm: dict[str, Any]) -> dict[str, Any]:
    turn = arm["call_turns"][0]
    return {
        "macro_call": canonical_call(arm["macro_call"]),
        "turn_content": turn["content_sha256"],
        "turn_tokens": turn["completion_tokens"],
        "final_content": arm["final"]["content_sha256"],
        "final_tokens": arm["final"]["completion_tokens"],
    }


def bootstrap_speedup_ci(
    pairs: list[dict[str, Any]], numerator: str, denominator: str, resamples: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        values.append(
            statistics.median(
                pair[numerator]["task_ms"] / pair[denominator]["task_ms"]
                for pair in sample
            )
        )
    return [percentile(values, 0.025), percentile(values, 0.975)]


def paired_slowdown(
    pairs: list[dict[str, Any]], metric: str, quantile: float
) -> float:
    ratios = [
        pair["speculative"][metric] / pair["compiled"][metric]
        for pair in pairs
        if pair["compiled"][metric] > 0.0
    ]
    return 100.0 * (percentile(ratios, quantile) - 1.0)


def summarize(
    pairs: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    baseline_speedups = [
        pair["stage_batched"]["task_ms"] / pair["speculative"]["task_ms"]
        for pair in pairs
    ]
    compiled_speedups = [
        pair["compiled"]["task_ms"] / pair["speculative"]["task_ms"] for pair in pairs
    ]
    continuation_turns = [
        turn
        for pair in pairs
        for arm_name in ("stage_batched", "compiled", "speculative")
        for turn in [*pair[arm_name]["call_turns"][1:], pair[arm_name]["final"]]
    ]
    speedup_by_call_count = {}
    for call_count in sorted({pair["task"]["call_count"] for pair in pairs}):
        bucket = [pair for pair in pairs if pair["task"]["call_count"] == call_count]
        speedup_by_call_count[str(call_count)] = {
            "tasks": len(bucket),
            "stage_batched_task_p50_ms": statistics.median(
                pair["stage_batched"]["task_ms"] for pair in bucket
            ),
            "compiled_task_p50_ms": statistics.median(
                pair["compiled"]["task_ms"] for pair in bucket
            ),
            "speculative_task_p50_ms": statistics.median(
                pair["speculative"]["task_ms"] for pair in bucket
            ),
            "combined_speedup_p50": statistics.median(
                pair["stage_batched"]["task_ms"]
                / pair["speculative"]["task_ms"]
                for pair in bucket
            ),
            "speculation_speedup_p50": statistics.median(
                pair["compiled"]["task_ms"] / pair["speculative"]["task_ms"]
                for pair in bucket
            ),
        }
    return {
        "tasks": len(pairs),
        "calls_per_task": [pair["task"]["call_count"] for pair in pairs],
        "stage_batched_task_p50_ms": statistics.median(
            pair["stage_batched"]["task_ms"] for pair in pairs
        ),
        "compiled_task_p50_ms": statistics.median(
            pair["compiled"]["task_ms"] for pair in pairs
        ),
        "speculative_task_p50_ms": statistics.median(
            pair["speculative"]["task_ms"] for pair in pairs
        ),
        "stage_batched_to_speculative_speedup_p50": statistics.median(
            baseline_speedups
        ),
        "stage_batched_to_speculative_speedup_p05": percentile(
            baseline_speedups, 0.05
        ),
        "stage_batched_to_speculative_speedup_min": min(baseline_speedups),
        "stage_batched_to_speculative_bootstrap_95ci": bootstrap_speedup_ci(
            pairs, "stage_batched", "speculative", resamples, seed
        ),
        "stage_batched_to_compiled_speedup_p50": statistics.median(
            pair["stage_batched"]["task_ms"] / pair["compiled"]["task_ms"]
            for pair in pairs
        ),
        "compiled_to_speculative_speedup_p50": statistics.median(compiled_speedups),
        "compiled_to_speculative_speedup_p05": percentile(compiled_speedups, 0.05),
        "compiled_to_speculative_bootstrap_95ci": bootstrap_speedup_ci(
            pairs, "compiled", "speculative", resamples, seed + 1
        ),
        "total_wall_speedup": sum(
            pair["stage_batched"]["task_ms"] for pair in pairs
        )
        / sum(pair["speculative"]["task_ms"] for pair in pairs),
        "stage_batched_model_turns_p50": statistics.median(
            pair["stage_batched"]["model_turns"] for pair in pairs
        ),
        "compiled_model_turns_p50": statistics.median(
            pair["compiled"]["model_turns"] for pair in pairs
        ),
        "stage_batched_exposed_tool_wait_p50_ms": statistics.median(
            pair["stage_batched"]["exposed_tool_wait_ms"] for pair in pairs
        ),
        "compiled_exposed_tool_wait_p50_ms": statistics.median(
            pair["compiled"]["exposed_tool_wait_ms"] for pair in pairs
        ),
        "speculative_exposed_tool_wait_p50_ms": statistics.median(
            pair["speculative"]["exposed_tool_wait_ms"] for pair in pairs
        ),
        "pattern_prediction_hit_rate": sum(
            pair["speculative"]["prediction_hit"] for pair in pairs
        )
        / len(pairs),
        "all_predictions_from_qwen": all(
            pair["speculative"]["prediction_source"] == NATIVE_PREDICTION_SOURCE
            for pair in pairs
        ),
        "predictor_p50_ms": statistics.median(
            pair["speculative"]["predictor_ms"] for pair in pairs
        ),
        "model_compute_slowdown_p50_percent": paired_slowdown(
            pairs, "model_compute_ms", 0.50
        ),
        "model_compute_slowdown_p95_percent": paired_slowdown(
            pairs, "model_compute_ms", 0.95
        ),
        "decode_slowdown_p50_percent": paired_slowdown(pairs, "decode_ms", 0.50),
        "decode_slowdown_p95_percent": paired_slowdown(pairs, "decode_ms", 0.95),
        "continuation_cache_hit_rate": sum(
            turn["cache_hit"] and turn["cached_prefix_tokens"] > 0
            for turn in continuation_turns
        )
        / len(continuation_turns),
        "speedup_by_call_count": speedup_by_call_count,
        "all_calls_stable": all(
            pair["stage_batched"]["underlying_calls"]
            == pair["compiled"]["underlying_calls"]
            == pair["speculative"]["underlying_calls"]
            for pair in pairs
        ),
        "all_tool_results_stable": all(
            pair["stage_batched"]["tool_results"]
            == pair["compiled"]["tool_results"]
            == pair["speculative"]["tool_results"]
            for pair in pairs
        ),
        "macro_output_stability_rate": sum(
            macro_signature(pair["compiled"]) == macro_signature(pair["speculative"])
            for pair in pairs
        )
        / len(pairs),
        "all_final_answers_correct": all(
            pair[arm]["final_correct"]
            for pair in pairs
            for arm in ("stage_batched", "compiled", "speculative")
        ),
        "all_final_outputs_stable": all(
            pair["stage_batched"]["final"]["content"].strip()
            == pair["compiled"]["final"]["content"].strip()
            == pair["speculative"]["final"]["content"].strip()
            for pair in pairs
        ),
        "all_macro_calls_correct": all(
            pair[arm]["macro_correct"]
            for pair in pairs
            for arm in ("compiled", "speculative")
        ),
        "all_ds4_active": all(
            pair[arm]["all_ds4_active"]
            for pair in pairs
            for arm in ("stage_batched", "compiled", "speculative")
        ),
    }


def production_checks(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "sample_size": summary["tasks"] >= args.min_production_pairs,
        "end_to_end_speedup": summary["stage_batched_to_speculative_speedup_p50"]
        >= args.min_e2e_speedup,
        "end_to_end_ci": summary[
            "stage_batched_to_speculative_bootstrap_95ci"
        ][0]
        > 1.0,
        "end_to_end_tail": summary["stage_batched_to_speculative_speedup_p05"]
        >= args.min_e2e_speedup_p05,
        "speculation_incremental_gain": summary["compiled_to_speculative_speedup_p50"]
        >= args.min_incremental_speedup,
        "speculation_incremental_ci": summary[
            "compiled_to_speculative_bootstrap_95ci"
        ][0]
        > 1.0,
        "prediction_hit_rate": summary["pattern_prediction_hit_rate"] == 1.0,
        "prediction_source": summary["all_predictions_from_qwen"],
        "model_slowdown_p50": summary["model_compute_slowdown_p50_percent"]
        <= args.max_model_slowdown_percent,
        "model_slowdown_p95": summary["model_compute_slowdown_p95_percent"]
        <= args.max_model_slowdown_p95_percent,
        "decode_slowdown_p50": summary["decode_slowdown_p50_percent"]
        <= args.max_decode_slowdown_percent,
        "decode_slowdown_p95": summary["decode_slowdown_p95_percent"]
        <= args.max_decode_slowdown_p95_percent,
        "prefix_cache_configured": summary["prefix_cache_configured"],
        "calls_stable": summary["all_calls_stable"],
        "tool_results_stable": summary["all_tool_results_stable"],
        "macro_outputs_stable": summary["macro_output_stability_rate"] == 1.0,
        "final_answers_correct": summary["all_final_answers_correct"],
        "final_outputs_stable": summary["all_final_outputs_stable"],
        "macro_calls_correct": summary["all_macro_calls_correct"],
        "ds4_active": summary["all_ds4_active"],
    }


def compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
    """Keep auditable outputs and timings without embedding the full graph."""
    fields = (
        "task_ms",
        "model_turns",
        "model_compute_ms",
        "decode_ms",
        "completion_tokens",
        "exposed_tool_wait_ms",
        "all_ds4_active",
        "final_correct",
        "graph_wall_ms",
        "macro_call",
        "macro_correct",
        "prediction_hit",
        "prediction_reason",
        "prediction_source",
        "prediction_status",
        "predictor_ms",
    )
    compact = {field: arm[field] for field in fields if field in arm}
    for field in ("underlying_calls", "tool_results"):
        values = arm.get(field)
        if isinstance(values, list):
            compact[f"{field}_count"] = len(values)
            compact[f"{field}_sha256"] = hashlib.sha256(
                json.dumps(values, separators=(",", ":")).encode()
            ).hexdigest()
    final = arm.get("final")
    if isinstance(final, dict):
        compact["final"] = {
            field: final[field]
            for field in (
                "content_sha256",
                "completion_tokens",
                "accept_rate",
            )
            if field in final
        }
    return compact


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    task = pair["task"]
    return {
        "pair_index": pair["pair_index"],
        "task": {
            field: task[field]
            for field in ("id", "branch_count", "call_count")
            if field in task
        },
        "arm_order": pair["arm_order"],
        **{
            arm: compact_arm(pair[arm])
            for arm in ("stage_batched", "compiled", "speculative")
        },
    }


def load_partial_pairs(
    path: Path,
    measured_tasks: list[dict[str, Any]],
    arm_orders: list[list[str]],
) -> list[dict[str, Any]]:
    """Load and strictly validate a checkpoint before resuming a long run."""
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint, dict):
        raise ValueError("benchmark checkpoint is not an object")
    pairs = checkpoint.get("pairs")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("complete") is not False
        or not isinstance(pairs, list)
        or len(pairs) > len(measured_tasks)
    ):
        raise ValueError("benchmark checkpoint is not resumable")
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, dict)
            or pair.get("pair_index") != index
            or pair.get("task") != measured_tasks[index]
            or pair.get("arm_order") != arm_orders[index]
            or not all(
                isinstance(pair.get(arm), dict)
                for arm in ("stage_batched", "compiled", "speculative")
            )
        ):
            raise ValueError(f"benchmark checkpoint pair {index} does not match this run")
        for arm in ("stage_batched", "compiled", "speculative"):
            result = pair[arm]
            final = result.get("final")
            expected = result.get("expected_final")
            if (
                not isinstance(final, dict)
                or not isinstance(final.get("content"), str)
                or not isinstance(expected, str)
            ):
                raise ValueError(
                    f"benchmark checkpoint pair {index} has an invalid {arm} final"
                )
            result["final_correct"] = final_answer_correct(
                final["content"], expected
            )
    return pairs


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be an executable tool adapter")
    if not args.training_report.is_file():
        parser.error("--training-report must be an existing trace file or report")
    if (
        args.pairs <= 0
        or args.warmup_tasks < 0
        or not 2 <= args.min_branches <= args.max_branches <= 4
        or args.timeout <= 0
        or min(args.call_max_tokens, args.macro_max_tokens, args.final_max_tokens) <= 0
        or args.bootstrap_resamples <= 0
        or args.min_production_pairs < 2
        or args.min_e2e_speedup <= 1.0
        or args.min_e2e_speedup_p05 <= 1.0
        or args.min_incremental_speedup <= 1.0
    ):
        parser.error("benchmark counts and thresholds are invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18145/v1/chat/completions")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--workflow-registry",
        type=Path,
        default=Path(__file__).with_name("results") / "trace-workflow-registry.json",
    )
    parser.add_argument("--tool-cpus", type=parse_cpu_list, default="14-15,30-31")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--warmup-tasks", type=int, default=1)
    parser.add_argument("--min-branches", type=int, default=2)
    parser.add_argument("--max-branches", type=int, default=4)
    parser.add_argument("--call-max-tokens", type=int, default=160)
    parser.add_argument("--macro-max-tokens", type=int, default=512)
    parser.add_argument("--final-max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=814)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--min-production-pairs", type=int, default=6)
    parser.add_argument("--min-e2e-speedup", type=float, default=2.0)
    parser.add_argument("--min-e2e-speedup-p05", type=float, default=1.5)
    parser.add_argument("--min-incremental-speedup", type=float, default=1.05)
    parser.add_argument("--max-model-slowdown-percent", type=float, default=1.0)
    parser.add_argument("--max-model-slowdown-p95-percent", type=float, default=5.0)
    parser.add_argument("--max-decode-slowdown-percent", type=float, default=1.0)
    parser.add_argument("--max-decode-slowdown-p95-percent", type=float, default=5.0)
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="resume the strictly matching <output>.partial checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.training_report = args.training_report.resolve()
    args.workflow_registry = args.workflow_registry.resolve()
    validate_args(parser, args)

    traces = load_training_traces(args.training_report, required_steps=5)
    pattern = mine_pattern(traces)
    props = get_json(props_url(args.url), args.timeout)
    prefix_cache = props.get("prefix_cache")
    if not isinstance(prefix_cache, dict) or int(prefix_cache.get("capacity", 0)) <= 0:
        raise SystemExit("prefix cache is not enabled")
    tool_speculation = props.get("tool_speculation")
    required_tool_props = {
        "enabled": True,
        "automatic_prediction_enabled": True,
        "prediction_source": NATIVE_PREDICTION_SOURCE,
        "predictor_schedule": "before-model",
        "execution_mode": "child_process_cpu_affinity",
        "cpu_affinity_isolated": True,
        "preserves_token_speculation": True,
    }
    if not isinstance(tool_speculation, dict):
        raise SystemExit("engine tool speculation is not enabled")
    for key, expected in required_tool_props.items():
        if tool_speculation.get(key) != expected:
            raise SystemExit(
                f"engine tool_speculation.{key}={tool_speculation.get(key)!r}, "
                f"expected {expected!r}"
            )
    if pattern.macro_name not in tool_speculation.get("allowed_tools", []):
        raise SystemExit(f"engine does not allow compiled macro {pattern.macro_name!r}")

    branch_span = args.max_branches - args.min_branches + 1
    warmup_tasks = [
        make_task(-1000 - index, args.min_branches, pattern)
        for index in range(args.warmup_tasks)
    ]
    measured_tasks = [
        make_task(
            pair_index,
            args.min_branches + pair_index % branch_span,
            pattern,
        )
        for pair_index in range(args.pairs)
    ]
    write_workflow_registry(
        args.workflow_registry, pattern, [*warmup_tasks, *measured_tasks]
    )
    generator = random.Random(args.seed)
    arm_orders = []
    for _ in measured_tasks:
        order = ["stage_batched", "compiled", "speculative"]
        generator.shuffle(order)
        arm_orders.append(order)
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    pairs = (
        load_partial_pairs(partial_output, measured_tasks, arm_orders)
        if args.resume_partial
        else []
    )
    if pairs:
        print(json.dumps({"resumed_pairs": len(pairs)}), flush=True)
    else:
        for warmup, task in enumerate(warmup_tasks):
            run_stage_batched(args, task, pattern, f"warm-{warmup}-stage-batched")
            run_macro(args, task, pattern, False, f"warm-{warmup}-compiled")
            run_macro(args, task, pattern, True, f"warm-{warmup}-speculative")

    for pair_index in range(len(pairs), len(measured_tasks)):
        task = measured_tasks[pair_index]
        order = arm_orders[pair_index]
        arms = {}
        for arm in order:
            if arm == "stage_batched":
                arms[arm] = run_stage_batched(
                    args, task, pattern, f"pair-{pair_index}-stage-batched"
                )
            else:
                arms[arm] = run_macro(
                    args,
                    task,
                    pattern,
                    arm == "speculative",
                    f"pair-{pair_index}-{arm}",
                )
        pair = {"pair_index": pair_index, "task": task, "arm_order": order, **arms}
        pairs.append(pair)
        partial_output.parent.mkdir(parents=True, exist_ok=True)
        partial_output.write_text(
            json.dumps({"schema_version": 1, "complete": False, "pairs": pairs}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "pair": pair_index + 1,
                    "calls": task["call_count"],
                    "order": order,
                    "stage_batched_ms": round(
                        arms["stage_batched"]["task_ms"], 1
                    ),
                    "compiled_ms": round(arms["compiled"]["task_ms"], 1),
                    "speculative_ms": round(arms["speculative"]["task_ms"], 1),
                    "end_to_end_speedup": round(
                        arms["stage_batched"]["task_ms"]
                        / arms["speculative"]["task_ms"],
                        3,
                    ),
                    "incremental_speedup": round(
                        arms["compiled"]["task_ms"]
                        / arms["speculative"]["task_ms"],
                        3,
                    ),
                    "prediction_hit": arms["speculative"]["prediction_hit"],
                    "correct": all(arms[arm]["final_correct"] for arm in arms),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = summarize(pairs, args.bootstrap_resamples, args.seed)
    ending_props = get_json(props_url(args.url), args.timeout)
    ending_prefix_cache = ending_props.get("prefix_cache")
    if not isinstance(ending_prefix_cache, dict):
        raise RuntimeError("prefix cache disappeared during the benchmark")
    summary["prefix_cache_lifetime_hit_delta"] = int(
        ending_prefix_cache.get("lifetime_hits", 0)
    ) - int(prefix_cache.get("lifetime_hits", 0))
    summary["prefix_cache_configured"] = int(prefix_cache.get("capacity", 0)) > 0
    checks = production_checks(summary, args)
    report = {
        "schema_version": 1,
        "host": "lucebox5",
        "feature": "no-training trace-compiled speculative tool graphs",
        "pattern": {
            "macro_name": pattern.macro_name,
            "fingerprint": pattern.fingerprint,
            "training_traces": pattern.training_traces,
            "training_report": str(args.training_report),
            "training_report_sha256": file_sha256(args.training_report),
            "workflow_registry": str(args.workflow_registry),
            "workflow_registry_sha256": file_sha256(args.workflow_registry),
            "root_fields": list(pattern.root_fields),
            "steps": [
                {
                    "tool": step.tool,
                    "arguments": {
                        name: {"source": binding.source, "key": binding.key}
                        for name, binding in step.arguments
                    },
                }
                for step in pattern.steps
            ],
            "model_training": False,
            "side_effects_allowed": False,
        },
        "workload": {
            "tasks": args.pairs,
            "branches_per_task": f"{args.min_branches}-{args.max_branches}",
            "calls_per_task": f"{args.min_branches * len(pattern.steps)}-"
            f"{args.max_branches * len(pattern.steps)}",
            "dependency": "five serial calls per branch; branches are independent",
            "tool_adapter": "deterministic read-only 2-second API replay",
        },
        "methodology": {
            "stage_batched": (
                "DS4+DSpark sees only the currently-ready typed batch, authorizes "
                "one per dependency stage, runs its calls concurrently, and receives "
                "a compact rolling state instead of replaying old tool history"
            ),
            "compiled": (
                "one DS4+DSpark macro authorization; independent branches execute "
                "concurrently on the Strix CPU lane"
            ),
            "speculative": (
                "Qwen predicts the trace-derived macro through the engine; its CPU "
                "graph overlaps DS4+DSpark and commits only on an exact call match"
            ),
            "arm_order": "randomized per task",
            "measured_wall_time": "request through all tools and exact final answer",
            "oracle_prediction": False,
            "argument_binding": (
                "the harness binds validated structured inputs to a request-scoped "
                "workflow_ref before either model runs"
            ),
            "model_seed": args.seed,
            "warmup_tasks": args.warmup_tasks,
        },
        "server_snapshot": {
            "prefix_cache_before": prefix_cache,
            "prefix_cache_after": ending_prefix_cache,
            "tool_speculation": tool_speculation,
            "model": props.get("model"),
        },
        "production_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "min_e2e_speedup": args.min_e2e_speedup,
                "min_e2e_speedup_p05": args.min_e2e_speedup_p05,
                "min_incremental_speedup": args.min_incremental_speedup,
                "min_production_pairs": args.min_production_pairs,
                "max_model_slowdown_percent": args.max_model_slowdown_percent,
                "max_model_slowdown_p95_percent": args.max_model_slowdown_p95_percent,
                "max_decode_slowdown_percent": args.max_decode_slowdown_percent,
                "max_decode_slowdown_p95_percent": args.max_decode_slowdown_p95_percent,
            },
        },
        "summary": summary,
        "pairs": [compact_pair(pair) for pair in pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["production_gate"]["passed"]:
        partial_output.unlink(missing_ok=True)
    print(
        json.dumps(
            {"production_gate": report["production_gate"], "summary": summary},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["production_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
