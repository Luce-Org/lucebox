#!/usr/bin/env python3
"""Run deterministic generation checks and compare Lucebox against llama.cpp.

The client launchers answer "can this real client talk to Lucebox?". This file
answers a different question: "does Lucebox generate the same kind of output as
a llama.cpp baseline, and how fast is it on the same prompts?".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Shared math-scoring helpers (canonical copy in harness/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from math_scoring import _extract_boxed, _math_equiv


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            if "id" not in case:
                raise ValueError(f"{path}:{line_no}: missing id")
            if "messages" not in case and "prompt" not in case:
                raise ValueError(f"{path}:{line_no}: missing messages or prompt")
            cases.append(case)
    return cases


def messages_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in case:
        return case["messages"]
    return [{"role": "user", "content": case["prompt"]}]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def approx_token_count(text: str) -> int:
    # Fallback only. Prefer server usage.completion_tokens when available.
    return max(1, len(re.findall(r"\S+", text)))


def _extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer from model output for GSM-style problems."""
    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>") :] if think_end >= 0 else text

    # #### <number>
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")

    # \boxed{<number>}
    boxed = _extract_boxed(answer_text)
    if boxed:
        cleaned = boxed.replace(",", "").strip()
        if re.match(r"^[+-]?\d+\.?\d*$", cleaned):
            return cleaned

    # "the answer is <number>"
    m = re.search(
        r"(?:answer\s+is|result\s+is|equals?|there\s+are|we\s+get)\s*\$?\s*\\?(?:boxed\{)?([+-]?\d[\d,]*\.?\d*)",
        answer_text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")

    # **<number>**
    m = re.search(r"\*\*([+-]?\d[\d,]*\.?\d*)\*\*", answer_text)
    if m:
        return m.group(1).replace(",", "")

    # Last standalone number
    nums = re.findall(r"(?<![.\d])([+-]?\d[\d,]*\.?\d*)(?![.\d])", answer_text)
    if nums:
        return nums[-1].replace(",", "")

    return None


def score_gold_answer(case: dict[str, Any], text: str) -> tuple[bool | None, str]:
    """Score model output against gold_answer if present.

    Returns (correct_or_None, detail_str). None means no gold_answer to check.
    """
    gold = case.get("gold_answer")
    if gold is None:
        return None, ""

    suite = case.get("suite", "")
    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>") :] if think_end >= 0 else text

    if suite == "gsm":
        pred = _extract_numeric_answer(text)
        if pred is None:
            return False, f"no numeric answer found, gold={gold}"
        try:
            correct = abs(float(pred) - float(gold)) < 1e-6
        except (ValueError, TypeError):
            correct = pred.strip() == gold.strip()
        return correct, f"pred={pred} gold={gold}"
    else:
        # Math-style: extract \boxed{} and compare
        pred = _extract_boxed(answer_text)
        if not pred:
            pred = _extract_boxed(text)
        if not pred:
            # Fallback: bold pattern
            m = re.search(
                r"(?:answer\s+is|result\s+is|equals?)\s*\*\*(.+?)\*\*", answer_text, re.IGNORECASE
            )
            if m:
                pred = m.group(1).strip().rstrip(".")
        if not pred:
            return False, f"no answer found, gold={gold}"
        correct = _math_equiv(pred, gold)
        return correct, f"pred={pred} gold={gold}"


def expected_pass(case: dict[str, Any], text: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    expected_exact = case.get("expect_exact")
    if expected_exact is not None:
        if not isinstance(expected_exact, str):
            raise ValueError("expect_exact must be a string")
        if normalize_text(text) != normalize_text(expected_exact):
            failures.append(f"normalized output differs from {expected_exact!r}")
    expected_json = case.get("expect_json")
    if expected_json is not None:
        if not isinstance(expected_json, dict):
            raise ValueError("expect_json must be an object")
        try:
            actual_json = json.loads(text)
        except json.JSONDecodeError:
            failures.append("output is not valid JSON")
        else:
            if actual_json != expected_json:
                failures.append(f"JSON output differs from {expected_json!r}")
    for needle in case.get("expect_contains", []):
        if needle not in text:
            failures.append(f"missing {needle!r}")
    for pattern in case.get("expect_regex", []):
        if not re.search(pattern, text, flags=re.MULTILINE):
            failures.append(f"regex did not match {pattern!r}")
    return not failures, failures


def post_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def run_case(
    case: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    repeats: int,
    warmups: int = 0,
    concurrency: int = 1,
) -> dict[str, Any]:
    def execute_request() -> dict[str, Any]:
        start = time.perf_counter()
        try:
            response = post_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages_for_case(case),
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as error:
            elapsed = time.perf_counter() - start
            detail = f"{type(error).__name__}: {error}"
            return {
                "elapsed_s": elapsed,
                "completion_tokens": 0,
                "prompt_tokens": None,
                "tok_s": 0.0,
                "token_count_source": "unavailable",
                "expected_pass": False,
                "expected_failures": [detail],
                "gold_correct": False if case.get("gold_answer") is not None else None,
                "gold_detail": detail,
                "text": "",
                "usage": {},
                "error": detail,
            }

        elapsed = time.perf_counter() - start
        text = extract_text(response)
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        token_source = "usage"
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            completion_tokens = approx_token_count(text)
            token_source = "approx_words"
        pass_expected, failures = expected_pass(case, text)
        gold_correct, gold_detail = score_gold_answer(case, text)
        return {
            "elapsed_s": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
            "token_count_source": token_source,
            "expected_pass": pass_expected,
            "expected_failures": failures,
            "gold_correct": gold_correct,
            "gold_detail": gold_detail,
            "text": text,
            "usage": usage,
        }

    def execute_batch() -> dict[str, Any]:
        if concurrency == 1:
            requests = [execute_request()]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                ready = threading.Barrier(concurrency + 1)

                def synchronized_request() -> dict[str, Any]:
                    ready.wait()
                    return execute_request()

                futures = [executor.submit(synchronized_request) for _ in range(concurrency)]
                ready.wait()
                requests = [future.result() for future in futures]

        elapsed = max(request["elapsed_s"] for request in requests)
        completion_tokens = sum(request["completion_tokens"] for request in requests)
        prompt_values = [request["prompt_tokens"] for request in requests]
        gold_results = [
            request["gold_correct"] for request in requests if request["gold_correct"] is not None
        ]
        failures = sorted(
            {failure for request in requests for failure in request["expected_failures"]}
        )
        token_sources = {request["token_count_source"] for request in requests}
        gold_details = [
            request["gold_detail"]
            for request in requests
            if request["gold_correct"] is False and request["gold_detail"]
        ]
        return {
            "elapsed_s": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": (
                sum(prompt_values)
                if all(isinstance(value, int) for value in prompt_values)
                else None
            ),
            "tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
            "token_count_source": (
                next(iter(token_sources)) if len(token_sources) == 1 else "mixed"
            ),
            "request_count": concurrency,
            "expected_pass": all(request["expected_pass"] for request in requests),
            "expected_failures": failures,
            "gold_correct": all(gold_results) if gold_results else None,
            "gold_detail": "; ".join(gold_details) or requests[-1]["gold_detail"],
            "text": requests[-1]["text"],
            "requests": requests,
        }

    for _ in range(warmups):
        execute_batch()
    runs = [execute_batch() for _ in range(repeats)]

    tok_s_values = [run["tok_s"] for run in runs]
    elapsed_values = [run["elapsed_s"] for run in runs]
    gold_results = [run["gold_correct"] for run in runs if run["gold_correct"] is not None]
    requests = [request for run in runs for request in run["requests"]]
    successful_texts = [
        request["text"]
        for request in requests
        if not request.get("error") and request["text"]
    ]
    failures = sorted({failure for run in runs for failure in run["expected_failures"]})
    gold_details = [
        run["gold_detail"] for run in runs if run["gold_correct"] is False and run["gold_detail"]
    ]
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "expect_contains": case.get("expect_contains", []),
        "expect_regex": case.get("expect_regex", []),
        "expect_exact": case.get("expect_exact"),
        "expect_json": case.get("expect_json"),
        "gold_answer": case.get("gold_answer"),
        "runs": runs,
        "mean_tok_s": statistics.mean(tok_s_values),
        "median_tok_s": statistics.median(tok_s_values),
        "mean_elapsed_s": statistics.mean(elapsed_values),
        "deterministic": (
            len(requests) >= 2
            and len(successful_texts) == len(requests)
            and len(set(successful_texts)) == 1
        ),
        "expected_pass": all(run["expected_pass"] for run in runs),
        "expected_failures": failures,
        "gold_correct": all(gold_results) if gold_results else None,
        "gold_detail": "; ".join(gold_details) or runs[-1].get("gold_detail", ""),
        "text": runs[-1]["text"],
        "completion_tokens": runs[-1]["completion_tokens"],
        "prompt_tokens": runs[-1]["prompt_tokens"],
        "token_count_source": runs[-1]["token_count_source"],
    }


def generation_verdict(
    results: list[dict[str, Any]],
    min_gold_accuracy: float | None,
    require_identical: bool = False,
) -> dict[str, Any]:
    if min_gold_accuracy is not None and not 0 <= min_gold_accuracy <= 1:
        raise ValueError("min_gold_accuracy must be between 0 and 1")
    scored = [result for result in results if result["gold_correct"] is not None]
    if min_gold_accuracy is not None and not scored:
        raise ValueError("minimum gold accuracy set but there are no gold-scored cases")
    expected_pass = sum(1 for result in results if result["expected_pass"])
    gold_correct = sum(1 for result in scored if result["gold_correct"])
    gold_accuracy = gold_correct / len(scored) if scored else None
    passed = expected_pass == len(results)
    if min_gold_accuracy is not None:
        passed = passed and gold_accuracy is not None and gold_accuracy >= min_gold_accuracy
    deterministic_cases = sum(1 for result in results if result.get("deterministic"))
    if require_identical:
        passed = passed and deterministic_cases == len(results)
    verdict = {
        "status": "pass" if passed else "fail",
        "expected_pass": expected_pass,
        "expected_total": len(results),
        "gold_correct": gold_correct,
        "gold_scored": len(scored),
        "gold_accuracy": gold_accuracy,
        "min_gold_accuracy": min_gold_accuracy,
    }
    if require_identical:
        verdict["deterministic_cases"] = deterministic_cases
        verdict["require_identical"] = True
    return verdict


def cmd_run(args: argparse.Namespace) -> int:
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    cases = load_cases(Path(args.prompts))
    if not cases:
        raise ValueError("prompt corpus must contain at least one case")
    results = []
    for case in cases:
        print(f"[bench] {args.name}: {case['id']}", end="", flush=True)
        result = run_case(
            case=case,
            base_url=args.url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            repeats=args.repeats,
            warmups=args.warmups,
            concurrency=args.concurrency,
        )
        results.append(result)
        if result["gold_correct"] is not None:
            mark = "🎯" if result["gold_correct"] else "✗"
            print(f"  {mark} {result['gold_detail']}", flush=True)
        else:
            print(flush=True)

    verdict = generation_verdict(results, args.min_gold_accuracy, args.require_identical)
    report = {
        "schema_version": 2,
        "name": args.name,
        "url": args.url,
        "model": args.model,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompts": str(Path(args.prompts)),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "cases": results,
        "summary": {
            **verdict,
            "cases": len(results),
            "mean_tok_s": statistics.mean([r["mean_tok_s"] for r in results]) if results else 0.0,
        },
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {out}")
    if verdict["gold_scored"]:
        print(
            f"[bench] correctness: {verdict['gold_correct']}/{verdict['gold_scored']} "
            f"({verdict['gold_accuracy'] * 100:.0f}%)"
        )
    print(f"[bench] verdict: {verdict['status'].upper()}")
    return 0 if verdict["status"] == "pass" else 1


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in report.get("cases", [])}


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_report(Path(args.baseline))
    candidate = load_report(Path(args.candidate))
    base_cases = case_map(baseline)
    cand_cases = case_map(candidate)
    rows = []
    for case_id in sorted(base_cases.keys() & cand_cases.keys()):
        base = base_cases[case_id]
        cand = cand_cases[case_id]
        base_tps = float(base.get("mean_tok_s", 0.0))
        cand_tps = float(cand.get("mean_tok_s", 0.0))
        rows.append(
            {
                "id": case_id,
                "baseline_tok_s": base_tps,
                "candidate_tok_s": cand_tps,
                "speedup": cand_tps / base_tps if base_tps > 0 else None,
                "baseline_expected_pass": bool(base.get("expected_pass")),
                "candidate_expected_pass": bool(cand.get("expected_pass")),
                "normalized_match": normalize_text(base.get("text", ""))
                == normalize_text(cand.get("text", "")),
                "baseline_text": base.get("text", ""),
                "candidate_text": cand.get("text", ""),
            }
        )

    summary = {
        "cases": len(rows),
        "baseline": baseline.get("name"),
        "candidate": candidate.get("name"),
        "baseline_expected_pass": sum(1 for r in rows if r["baseline_expected_pass"]),
        "candidate_expected_pass": sum(1 for r in rows if r["candidate_expected_pass"]),
        "normalized_matches": sum(1 for r in rows if r["normalized_match"]),
        "baseline_mean_tok_s": statistics.mean([r["baseline_tok_s"] for r in rows])
        if rows
        else 0.0,
        "candidate_mean_tok_s": statistics.mean([r["candidate_tok_s"] for r in rows])
        if rows
        else 0.0,
    }
    if summary["baseline_mean_tok_s"] > 0:
        summary["mean_speedup"] = summary["candidate_mean_tok_s"] / summary["baseline_mean_tok_s"]
    else:
        summary["mean_speedup"] = None

    report = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "baseline_report": str(Path(args.baseline)),
        "candidate_report": str(Path(args.candidate)),
        "summary": summary,
        "cases": rows,
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        write_markdown(report, Path(args.md_out))
    print(f"[bench] wrote {json_out}")
    return 0 if summary["candidate_expected_pass"] == summary["cases"] else 1


def fmt_speedup(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.2f}x"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Lucebox vs llama.cpp Generation Benchmark",
        "",
        f"Baseline: `{summary['baseline']}`",
        f"Candidate: `{summary['candidate']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Baseline mean tok/s | {summary['baseline_mean_tok_s']:.2f} |",
        f"| Candidate mean tok/s | {summary['candidate_mean_tok_s']:.2f} |",
        f"| Mean speedup | {fmt_speedup(summary['mean_speedup'])} |",
        f"| Candidate expected checks | {summary['candidate_expected_pass']}/{summary['cases']} |",
        f"| Normalized output matches | {summary['normalized_matches']}/{summary['cases']} |",
        "",
        "| Case | llama.cpp tok/s | Lucebox tok/s | Speedup | Expected | Same normalized text |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["cases"]:
        expected = "pass" if row["candidate_expected_pass"] else "fail"
        match = "yes" if row["normalized_match"] else "no"
        lines.append(
            f"| `{row['id']}` | {row['baseline_tok_s']:.2f} | "
            f"{row['candidate_tok_s']:.2f} | {fmt_speedup(row['speedup'])} | "
            f"{expected} | {match} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one OpenAI-compatible endpoint")
    run.add_argument("--name", required=True)
    run.add_argument("--url", required=True, help="Base URL ending in /v1")
    run.add_argument("--api-key", default="")
    run.add_argument("--model", required=True)
    run.add_argument(
        "--prompts", default=str(Path(__file__).with_name("prompts") / "generation_smoke.jsonl")
    )
    run.add_argument("--json-out", required=True)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--warmups", type=int, default=0)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--min-gold-accuracy", type=float)
    run.add_argument("--require-identical", action="store_true")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="Compare two endpoint reports")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--json-out", required=True)
    compare.add_argument("--md-out", default="")
    compare.set_defaults(func=cmd_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"[bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
