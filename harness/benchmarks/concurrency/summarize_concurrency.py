#!/usr/bin/env python3
"""Summarize ragged or canonical Lucebox concurrency benchmark reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_reports(root: Path) -> list[dict]:
    reports = []
    for path in sorted(root.rglob("bench.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = report.get("server_metadata") or {}
        if len(report.get("levels", [])) != 1:
            raise ValueError(f"{path}: expected exactly one client level")
        level = report["levels"][0]
        if (
            level.get("failures")
            or not level.get("token_count_complete")
            or not level.get("prompt_token_count_complete")
        ):
            raise ValueError(f"{path}: failed or incomplete token accounting")
        if report.get("ignore_eos") is not True:
            raise ValueError(f"{path}: fixed-token protocol is required")
        if level.get("fixed_token_workload_valid") is not True:
            raise ValueError(f"{path}: fixed-token validation failed")
        reports.append({"path": path, "report": report, "level": level, "meta": meta})
    if not reports:
        raise ValueError(f"{root}: no bench.json files found")
    return reports


def median(values: list[float]) -> float:
    return statistics.median(values)


def native_metric_median(
    levels: list[dict[str, Any]], key: str, context: str,
) -> float | None:
    """Require native telemetry on every repeat or on none of them."""
    available: list[bool] = []
    values: list[float] = []
    for level in levels:
        token_count_complete = (
            level.get("server_native_prefill_token_count_complete") is True
        )
        timing_complete = (
            level.get("server_native_prefill_timing_complete") is True
        )
        if token_count_complete != timing_complete:
            raise ValueError(
                f"{context}: mismatched native prefill completeness flags"
            )
        complete = token_count_complete and timing_complete
        value = level.get(key)
        if complete:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{context}: invalid complete native metric {key}")
            values.append(float(value))
        elif value is not None:
            raise ValueError(
                f"{context}: native metric {key} present without complete telemetry"
            )
        available.append(complete)
    if any(available) and not all(available):
        raise ValueError(f"{context}: partial native prefill telemetry across repeats")
    return median(values) if values else None


def paired_delta(
    grouped: dict[tuple[str, int, str], list[dict]],
    workload: str,
    clients: int,
    variant: str,
    items: list[dict],
    prompt_hashes: set[str],
    other: str,
    metric: str,
) -> str:
    peers = grouped.get((workload, clients, other), [])
    if not peers:
        return "n/a"
    output_hashes = {
        item["level"].get("selected_output_set_sha256") for item in items
    }
    peer_output_hashes = {
        item["level"].get("selected_output_set_sha256") for item in peers
    }
    if len(output_hashes) > 1 or len(peer_output_hashes) > 1:
        return "n/a"
    peer_hashes = {p["level"]["selected_prompt_set_sha256"] for p in peers}
    if peer_hashes != prompt_hashes:
        raise ValueError(f"{workload} C={clients}: {variant}/{other} prompts differ")
    by_repeat = {int(item["meta"]["repeat"]): item for item in items}
    peers_by_repeat = {int(item["meta"]["repeat"]): item for item in peers}
    if by_repeat.keys() != peers_by_repeat.keys():
        raise ValueError(
            f"{workload} C={clients}: {variant}/{other} repeat sets differ"
        )
    ratios = []
    for repeat in sorted(by_repeat):
        value = by_repeat[repeat]["level"].get(metric)
        base = peers_by_repeat[repeat]["level"].get(metric)
        if value is None or base is None:
            return "n/a"
        if base <= 0:
            raise ValueError(
                f"{workload} C={clients} repeat={repeat}: "
                f"non-positive {other} {metric}"
            )
        ratios.append(value / base - 1.0)
    return f"{median(ratios) * 100:+.1f}%"


def summarize(reports: list[dict]) -> str:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for item in reports:
        meta, level = item["meta"], item["level"]
        key = (str(meta["workload"]), int(level["clients"]), str(meta["variant"]))
        grouped[key].append(item)
    for key, items in grouped.items():
        repeats = [int(item["meta"]["repeat"]) for item in items]
        if len(repeats) != len(set(repeats)):
            raise ValueError(f"{key}: duplicate repeat")

    lines = [
        "# Concurrency benchmark summary", "",
        "Aggregate output goodput includes queueing, prefill, and decode. "
        "Output-window goodput starts at the first observed output and is decode-facing, "
        "but it can include staggered prefill. Prompt tok/s to first token includes "
        "admission and TTFT. Native prefill metrics come only from terminal "
        "usage.timings; they remain n/a when a server does not expose those fields. "
        "Native prefill tok/s divides total prefilled_tokens by the common-origin "
        "prefill window, so overlapping request times are not summed.", "",
        "| Workload | C | Variant | Repeats | Output goodput tok/s | "
        "Output-window tok/s | Request decode tok/s | Prompt tok/s to first | "
        "TTFT median s | TTFT max s | Stable output | vs llama | Decode vs llama | "
        "K8 vs K1 | Native prefill tok/s | Native prefill window ms |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workload, clients, variant in sorted(grouped):
        items = grouped[(workload, clients, variant)]
        hashes = {item["level"]["selected_prompt_set_sha256"] for item in items}
        if len(hashes) != 1:
            raise ValueError(f"{workload} C={clients} {variant}: prompt sets differ")
        goodput = median([item["level"]["aggregate_tok_s"] for item in items])
        output_window_values = [
            item["level"].get("output_window_tok_s") for item in items
            if item["level"].get("output_window_tok_s") is not None
        ]
        output_window = median(output_window_values) if output_window_values else None
        request_decode_values = [
            item["level"].get("request_decode_tok_s_median") for item in items
            if item["level"].get("request_decode_tok_s_median") is not None
        ]
        request_decode = median(request_decode_values) if request_decode_values else None
        prompt_rate_values = [
            item["level"]["prompt_tokens_per_s_to_first_token"] for item in items
            if item["level"].get("prompt_tokens_per_s_to_first_token") is not None
        ]
        prompt_rate = median(prompt_rate_values) if prompt_rate_values else None
        native_levels = [item["level"] for item in items]
        native_context = f"{workload} C={clients} {variant}"
        native_prefill_rate = native_metric_median(
            native_levels, "server_native_prefill_tokens_per_s", native_context,
        )
        native_prefill_window_ms = native_metric_median(
            native_levels, "server_native_prefill_window_ms", native_context,
        )
        ttft_median = median([item["level"]["ttft_median_s"] for item in items])
        ttft_max = median([item["level"]["ttft_max_s"] for item in items])
        output_hashes = {
            item["level"].get("selected_output_set_sha256") for item in items
        }
        stable = (
            "n/a" if len(items) < 2
            else "yes" if len(output_hashes) == 1
            else "NO"
        )

        vs_llama = (
            paired_delta(
                grouped, workload, clients, variant, items, hashes,
                "llama", "aggregate_tok_s",
            )
            if variant == "luce-k8" else "—"
        )
        decode_vs_llama = (
            paired_delta(
                grouped, workload, clients, variant, items, hashes,
                "llama", "output_window_tok_s",
            )
            if variant == "luce-k8" else "—"
        )
        vs_k1 = (
            paired_delta(
                grouped, workload, clients, variant, items, hashes,
                "luce-k1", "aggregate_tok_s",
            )
            if variant == "luce-k8" else "—"
        )
        output_window_text = f"{output_window:.2f}" if output_window is not None else "n/a"
        request_decode_text = f"{request_decode:.2f}" if request_decode is not None else "n/a"
        prompt_rate_text = f"{prompt_rate:.2f}" if prompt_rate is not None else "n/a"
        native_rate_text = (
            f"{native_prefill_rate:.2f}" if native_prefill_rate is not None else "n/a"
        )
        native_ms_text = (
            f"{native_prefill_window_ms:.1f}"
            if native_prefill_window_ms is not None else "n/a"
        )
        lines.append(
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"{output_window_text} | {request_decode_text} | {prompt_rate_text} | "
            f"{ttft_median:.3f} | {ttft_max:.3f} | {stable} | {vs_llama} | "
            f"{decode_vs_llama} | {vs_k1} | {native_rate_text} | {native_ms_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def output_signature(report: dict[str, Any]) -> tuple[tuple[str, str, str], ...] | None:
    level = report["levels"][0]
    rows = []
    for wave in level.get("wave_results", []):
        for request in wave.get("requests_detail", []):
            try:
                rows.append((
                    request["case_id"], request["content_sha256"],
                    request["reasoning_content_sha256"],
                ))
            except KeyError:
                return None
    if len(rows) != level["requests"]:
        return None
    return tuple(rows)


def summarize_canonical(root: Path) -> str:
    GroupKey = tuple[str, int, str, int | None, int]
    FamilyKey = tuple[str, int, int | None, int]
    PromptFamilyKey = tuple[str, int, int | None]
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    repeat_ids: dict[GroupKey, set[int]] = defaultdict(set)
    variant_repeats: dict[FamilyKey, dict[str, set[int]]] = defaultdict(dict)
    prompt_hashes: dict[PromptFamilyKey, str] = {}
    for path in root.glob("*/c*/r*/*/bench.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        levels = report.get("levels")
        if not isinstance(levels, list) or len(levels) != 1:
            raise ValueError(f"{path}: expected exactly one client level")
        level = levels[0]
        metadata = report["server_metadata"]
        suite = report.get("suite")
        variant = metadata.get("variant")
        clients = level.get("clients")
        requests = level.get("requests")
        repeat = metadata.get("repeat")
        case_limit = report.get("case_limit")
        if not isinstance(suite, str) or not isinstance(variant, str):
            raise ValueError(f"invalid suite or variant metadata: {path}")
        if isinstance(clients, bool) or not isinstance(clients, int) or clients < 1:
            raise ValueError(f"invalid client count: {path}")
        if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
            raise ValueError(f"invalid request count: {path}")
        if case_limit is not None and (
            isinstance(case_limit, bool) or not isinstance(case_limit, int) or case_limit < 1
        ):
            raise ValueError(f"invalid case_limit: {path}")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"missing or invalid repeat id: {path}")
        if (
            level["failures"]
            or level["fixed_token_workload_valid"] is not True
            or level.get("token_count_complete") is not True
            or level.get("prompt_token_count_complete") is not True
        ):
            raise ValueError(f"failed or incomplete token accounting: {path}")
        prompt_hash = report.get("prompt_file_sha256")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            raise ValueError(f"missing prompt_file_sha256: {path}")
        prompt_family = (suite, clients, case_limit)
        previous_hash = prompt_hashes.setdefault(prompt_family, prompt_hash)
        if previous_hash != prompt_hash:
            raise ValueError(
                f"{suite} C={clients} case_limit={case_limit}: prompt files differ"
            )
        if variant.endswith("ddtree"):
            proof = report.get("ddtree_proof")
            if not isinstance(proof, dict):
                raise ValueError(f"missing positive DDTree proof: {path}")
            steps = proof.get("ddtree_steps")
            if (
                isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
                or proof.get("requests_proven") != requests
            ):
                raise ValueError(f"missing positive DDTree proof: {path}")
        key = (suite, clients, variant, case_limit, requests)
        if repeat in repeat_ids[key]:
            raise ValueError(f"duplicate repeat id {repeat} for {key}")
        repeat_ids[key].add(repeat)
        groups[key].append(report)
        family = (suite, clients, case_limit, requests)
        variant_repeats.setdefault(family, {}).setdefault(variant, set()).add(repeat)
    for family, variants in variant_repeats.items():
        if len({frozenset(repeats) for repeats in variants.values()}) > 1:
            raise ValueError(f"mismatched repeat sets for {family}")
    if not groups:
        raise ValueError(f"no canonical reports under {root}")
    lines = [
        "# Canonical Qwen3.6 concurrency benchmark", "",
        "| Suite | C | Cases | Variant | Repeats | Goodput tok/s | Output-window tok/s | "
        "Prompt tok/s to first | Request decode tok/s | TTFT median s | TTFT max s | "
        "DDTree AL | Acceptance | Stable output | Native prefill tok/s | "
        "Native prefill window ms |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    sort_key = lambda item: (
        item[0][0], item[0][1], item[0][2], item[0][3] is not None,
        item[0][3] or 0, item[0][4],
    )
    for (suite, clients, variant, _case_limit, _requests), reports in sorted(
        groups.items(), key=sort_key
    ):
        levels = [report["levels"][0] for report in reports]
        proofs = [report.get("ddtree_proof") for report in reports]
        med = lambda key, current_levels=levels: statistics.median(
            level[key] for level in current_levels
        )
        native_context = f"{suite} C={clients} {variant} canonical"
        native_prefill_rate = native_metric_median(
            levels, "server_native_prefill_tokens_per_s", native_context,
        )
        native_prefill_window_ms = native_metric_median(
            levels, "server_native_prefill_window_ms", native_context,
        )
        al = (
            statistics.median(proof["mean_accepted_length"] for proof in proofs)
            if all(proof is not None for proof in proofs) else None
        )
        acceptance = (
            statistics.median(proof["acceptance_rate"] for proof in proofs)
            if all(proof is not None for proof in proofs) else None
        )
        acceptance_text = f"{100 * acceptance:.1f}%" if acceptance is not None else "n/a"
        signatures = [output_signature(report) for report in reports]
        complete = len(reports) >= 2 and all(signature is not None for signature in signatures)
        stable = "YES" if complete and len(set(signatures)) == 1 else "NO" if complete else "n/a"
        lines.append(
            f"| {suite} | {clients} | {levels[0]['requests']} | {variant} | {len(reports)} | "
            f"{fmt(med('aggregate_tok_s'))} | {fmt(med('output_window_tok_s'))} | "
            f"{fmt(med('prompt_tokens_per_s_to_first_token'))} | "
            f"{fmt(med('request_decode_tok_s_median'))} | "
            f"{fmt(med('ttft_median_s'), 3)} | {fmt(med('ttft_max_s'), 3)} | "
            f"{fmt(al)} | {acceptance_text} | {stable} | "
            f"{fmt(native_prefill_rate)} | {fmt(native_prefill_window_ms, 1)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--format", choices=("ragged", "canonical"), default="ragged")
    args = parser.parse_args()
    text = (
        summarize_canonical(args.root)
        if args.format == "canonical"
        else summarize(load_reports(args.root))
    )
    if args.out:
        args.out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
