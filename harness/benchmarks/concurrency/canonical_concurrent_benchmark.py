#!/usr/bin/env python3
"""Run a complete repository benchmark suite in fixed-concurrency waves."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "concurrent_benchmark", HERE / "concurrent_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


CONCURRENCY_METRICS_MARKER = re.compile(r"\[concurrency-metrics\]\s+(\{.*\})\s*$")
SERVER_DONE_MARKER = re.compile(r"\[server\] chat DONE\s+(\S+)")
DDTREE_METRICS_MARKER = re.compile(r"\[concurrency-metrics\]\s+(\{.*\})")
DDTREE_COUNTERS = ("ddtree_steps", "ddtree_accepted_tokens", "target_forwards")


def retired_response_ids(text: str) -> set[str]:
    retired: set[str] = set()
    for line in text.splitlines():
        marker = CONCURRENCY_METRICS_MARKER.search(line)
        if marker:
            try:
                value = json.loads(marker.group(1))
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                response_id = value.get("response_id") or value.get("request_id")
                if isinstance(response_id, str) and response_id:
                    retired.add(response_id)
        done = SERVER_DONE_MARKER.search(line)
        if done:
            retired.add(done.group(1))
    return retired


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{path}:{line_no}: missing or duplicate string id")
        if isinstance(record.get("prompt"), str) and record["prompt"]:
            prompt = record["prompt"]
        else:
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_no}: 'messages' must be an array")
            prompt = base.prompt_messages(messages)
        seen.add(case_id)
        cases.append({"id": case_id, "prompt": prompt})
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def wait_for_retirement(path: Path, response_ids: list[str], timeout: float) -> float:
    started = time.perf_counter()
    pending = set(response_ids)
    deadline = started + timeout
    while pending and time.perf_counter() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            pending -= retired_response_ids(text)
        if pending:
            time.sleep(0.05)
    if pending:
        raise TimeoutError(f"scheduler did not retire responses: {sorted(pending)}")
    return time.perf_counter() - started


def load_ddtree_metrics(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DDTREE_METRICS_MARKER.search(line)
        if not match:
            continue
        value = json.loads(match.group(1))
        response_id = value.get("response_id") or value.get("request_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("concurrency metric is missing response_id")
        if response_id in found:
            raise ValueError(f"duplicate concurrency metric for {response_id}")
        for key in DDTREE_COUNTERS:
            if (
                isinstance(value.get(key), bool)
                or not isinstance(value.get(key), int)
                or value[key] < 0
            ):
                raise ValueError(f"{response_id}: invalid {key}")
        found[response_id] = value
    return found


def attach_ddtree_proof(
    report: dict[str, Any], metrics: dict[str, dict[str, Any]]
) -> None:
    levels = report.get("levels", [])
    details = [
        request
        for level in levels
        for wave in level.get("wave_results", [])
        for request in wave.get("requests_detail", [])
    ]
    if (
        any(level.get("failures") for level in levels)
        or any(request.get("error") is not None for request in details)
    ):
        raise ValueError("DDTree proof requires a complete level with no failed requests")
    requests = [
        request
        for request in details
    ]
    totals = {key: 0 for key in DDTREE_COUNTERS}
    for request in requests:
        response_id = request.get("response_id")
        if not isinstance(response_id, str) or response_id not in metrics:
            raise ValueError(f"missing concurrency metric for response {response_id!r}")
        value = metrics[response_id]
        if value["ddtree_steps"] <= 0:
            raise ValueError(f"{response_id}: ddtree_steps must be positive")
        request["ddtree_metrics"] = {key: value[key] for key in DDTREE_COUNTERS}
        for key in DDTREE_COUNTERS:
            totals[key] += value[key]
    steps = totals["ddtree_steps"]
    if not requests or steps <= 0:
        raise ValueError("DDTree proof requires at least one successful request and step")
    emitted = totals["ddtree_accepted_tokens"] + steps
    report["ddtree_proof"] = {
        **totals,
        "speculative_emitted_tokens": emitted,
        "mean_accepted_length": emitted / steps,
        "acceptance_rate": emitted / (16 * steps),
        "acceptance_denominator_tokens_per_step": 16,
        "requests_proven": len(requests),
    }


def aggregate_waves(clients: int, waves: list[dict[str, Any]]) -> dict[str, Any]:
    details = [record for wave in waves for record in wave["requests_detail"]]
    ok = [record for record in details if record["error"] is None]
    completion = [record["completion_tokens"] for record in ok]
    prompts = [record["prompt_tokens"] for record in ok]
    rates = [record["request_decode_tok_s"] for record in ok]
    ttfts = [record["ttft_s"] for record in ok]
    completion_complete = bool(ok) and all(isinstance(value, int) for value in completion)
    prompt_complete = bool(ok) and all(isinstance(value, int) for value in prompts)
    wall = sum(wave["wall_s"] for wave in waves)
    output_window_values = [wave.get("output_window_s") for wave in waves]
    output_window = (
        sum(output_window_values)
        if all(isinstance(value, (int, float)) for value in output_window_values)
        else None
    )
    prompt_window_values = [wave.get("prompt_to_first_token_s") for wave in waves]
    prompt_window = (
        sum(prompt_window_values)
        if all(isinstance(value, (int, float)) for value in prompt_window_values)
        else None
    )
    native_tokens = [
        wave.get("server_native_prefilled_tokens_total") for wave in waves
    ]
    native_windows_ms = [
        wave.get("server_native_prefill_window_ms") for wave in waves
    ]
    native_token_count_complete = bool(waves) and all(
        wave.get("server_native_prefill_token_count_complete") is True
        and isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for wave, value in zip(waves, native_tokens, strict=True)
    )
    native_timing_complete = bool(waves) and all(
        wave.get("server_native_prefill_timing_complete") is True
        and isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        for wave, value in zip(waves, native_windows_ms, strict=True)
    )
    native_tokens_total = (
        sum(native_tokens) if native_token_count_complete else None
    )
    native_window_ms = (
        sum(native_windows_ms) if native_timing_complete else None
    )
    failures = sum(wave["failures"] for wave in waves)
    return {
        "clients": clients,
        "waves": len(waves),
        "requests": len(details),
        "requests_ok": len(ok),
        "failures": failures,
        "wall_s": wall,
        "completion_tokens_total": sum(completion) if completion_complete else None,
        "token_count_complete": completion_complete,
        "prompt_tokens_total": sum(prompts) if prompt_complete else None,
        "prompt_tokens_min": min(prompts) if prompt_complete else None,
        "prompt_tokens_max": max(prompts) if prompt_complete else None,
        "prompt_token_count_complete": prompt_complete,
        "prompt_to_first_token_s": prompt_window,
        "prompt_tokens_per_s_to_first_token": (
            sum(prompts) / prompt_window
            if prompt_complete and prompt_window is not None and prompt_window > 0 else None
        ),
        "aggregate_tok_s": sum(completion) / wall if completion_complete and wall > 0 else None,
        "output_window_tok_s": (
            sum(completion) / output_window
            if completion_complete and output_window is not None and output_window > 0 else None
        ),
        "request_decode_tok_s_median": (
            statistics.median(rates)
            if len(rates) == len(ok) and ok else None
        ),
        "server_native_prefilled_tokens_total": native_tokens_total,
        "server_native_prefill_window_ms": native_window_ms,
        "server_native_prefill_token_count_complete": native_token_count_complete,
        "server_native_prefill_timing_complete": native_timing_complete,
        "server_native_prefill_tokens_per_s": (
            native_tokens_total * 1000.0 / native_window_ms
            if native_tokens_total is not None
            and native_window_ms is not None
            and native_window_ms > 0 else None
        ),
        "server_native_prefill_metric": (
            "sum_prefilled_tokens_per_sum_wave_common_origin_"
            "prefill_window_second"
        ),
        "ttft_median_s": statistics.median(ttfts) if len(ttfts) == len(ok) and ok else None,
        "ttft_max_s": max(ttfts) if len(ttfts) == len(ok) and ok else None,
        "wave_results": waves,
    }


def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.prompt_file)
    if args.case_limit is not None:
        if args.case_limit < 1 or args.case_limit > len(cases):
            raise ValueError(
                f"--case-limit must be between 1 and the suite size {len(cases)}"
            )
        cases = cases[:args.case_limit]
    if args.clients < 1 or len(cases) % args.clients:
        raise ValueError(
            f"suite size {len(cases)} must be divisible by --clients={args.clients}; "
            "refusing a lower-concurrency tail wave"
        )
    waves = []
    for offset in range(0, len(cases), args.clients):
        selected = cases[offset:offset + args.clients]
        wave = base.run_level(args.clients, args, [case["prompt"] for case in selected], 0)
        for case, detail in zip(selected, wave["requests_detail"], strict=True):
            detail["case_id"] = case["id"]
        if args.retire_log and wave["failures"] == 0:
            response_ids = [
                detail["response_id"] for detail in wave["requests_detail"]
                if isinstance(detail.get("response_id"), str)
            ]
            if len(response_ids) != args.clients:
                raise ValueError("successful wave is missing response IDs")
            wait_s = wait_for_retirement(args.retire_log, response_ids, args.timeout)
            wave["retirement_wait_s"] = wait_s
            wave["wall_s"] += wait_s
        waves.append(wave)
    level = aggregate_waves(args.clients, waves)
    level["fixed_token_workload_valid"] = (
        level["failures"] == 0
        and level["requests_ok"] == len(cases)
        and level["token_count_complete"] is True
        and level["prompt_token_count_complete"] is True
        and all(wave["fixed_token_workload_valid"] is True for wave in waves)
    ) if args.ignore_eos else None
    metadata = (
        json.loads(args.server_metadata_json.read_text(encoding="utf-8"))
        if args.server_metadata_json else {}
    )
    report = {
        "schema_version": 1,
        "label": args.label,
        "suite": args.suite,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
        "case_limit": args.case_limit,
        "prompt_file_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
        "server_metadata": metadata,
        "levels": [level],
    }
    ddtree_proof_attached = False
    if getattr(args, "ddtree_proof", False):
        if args.retire_log is None:
            raise ValueError("--ddtree-proof requires --retire-log")
        if level["failures"] == 0:
            attach_ddtree_proof(report, load_ddtree_metrics(args.retire_log))
            ddtree_proof_attached = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(base.markdown(report), end="")
    if ddtree_proof_attached:
        proof = report["ddtree_proof"]
        print(
            f"DDTree AL={proof['mean_accepted_length']:.2f} "
            f"acceptance={100 * proof['acceptance_rate']:.1f}% "
            f"steps={proof['ddtree_steps']}"
        )
    return 1 if base.level_failed(level, args.ignore_eos) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="luce-dflash")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--server-metadata-json", type=Path)
    parser.add_argument("--retire-log", type=Path)
    parser.add_argument(
        "--ddtree-proof",
        action="store_true",
        help="validate and attach DDTree telemetry from --retire-log",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"[canonical-bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
