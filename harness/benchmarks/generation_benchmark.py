#!/usr/bin/env python3
"""Run deterministic generation checks and compare Lucebox against llama.cpp.

The client launchers answer "can this real client talk to Lucebox?". This file
answers a different question: "does Lucebox generate the same kind of output as
a llama.cpp baseline, and how fast is it on the same prompts?".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
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
    answer_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    # #### <number>
    m = re.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', answer_text)
    if m:
        return m.group(1).replace(",", "")

    # \boxed{<number>}
    boxed = _extract_boxed(answer_text)
    if boxed:
        cleaned = boxed.replace(",", "").strip()
        if re.match(r'^[+-]?\d+\.?\d*$', cleaned):
            return cleaned

    # "the answer is <number>"
    m = re.search(
        r'(?:answer\s+is|result\s+is|equals?|there\s+are|we\s+get)\s*\$?\s*\\?(?:boxed\{)?([+-]?\d[\d,]*\.?\d*)',
        answer_text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # **<number>**
    m = re.search(r'\*\*([+-]?\d[\d,]*\.?\d*)\*\*', answer_text)
    if m:
        return m.group(1).replace(",", "")

    # Last standalone number
    nums = re.findall(r'(?<![.\d])([+-]?\d[\d,]*\.?\d*)(?![.\d])', answer_text)
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
    answer_text = text[think_end + len("</think>"):] if think_end >= 0 else text

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
                r'(?:answer\s+is|result\s+is|equals?)\s*\*\*(.+?)\*\*',
                answer_text, re.IGNORECASE)
            if m:
                pred = m.group(1).strip().rstrip(".")
        if not pred:
            return False, f"no answer found, gold={gold}"
        correct = _math_equiv(pred, gold)
        return correct, f"pred={pred} gold={gold}"


def expected_pass(case: dict[str, Any], text: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
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
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        start = time.perf_counter()
        response = post_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages_for_case(case),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        text = extract_text(response)
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        token_source = "usage"
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            completion_tokens = approx_token_count(text)
            token_source = "approx_words"
        prompt_tokens = usage.get("prompt_tokens")
        pass_expected, failures = expected_pass(case, text)
        gold_correct, gold_detail = score_gold_answer(case, text)
        runs.append(
            {
                "elapsed_s": elapsed,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
                "token_count_source": token_source,
                "expected_pass": pass_expected,
                "expected_failures": failures,
                "gold_correct": gold_correct,
                "gold_detail": gold_detail,
                "text": text,
                "usage": usage,
            }
        )

    tok_s_values = [r["tok_s"] for r in runs]
    elapsed_values = [r["elapsed_s"] for r in runs]
    gold_results = [r["gold_correct"] for r in runs if r["gold_correct"] is not None]
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "expect_contains": case.get("expect_contains", []),
        "expect_regex": case.get("expect_regex", []),
        "gold_answer": case.get("gold_answer"),
        "runs": runs,
        "mean_tok_s": statistics.mean(tok_s_values),
        "median_tok_s": statistics.median(tok_s_values),
        "mean_elapsed_s": statistics.mean(elapsed_values),
        "expected_pass": all(r["expected_pass"] for r in runs),
        "gold_correct": all(gold_results) if gold_results else None,
        "gold_detail": runs[-1].get("gold_detail", ""),
        "text": runs[-1]["text"],
        "completion_tokens": runs[-1]["completion_tokens"],
        "prompt_tokens": runs[-1]["prompt_tokens"],
        "token_count_source": runs[-1]["token_count_source"],
    }


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.prompts))
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
        )
        results.append(result)
        if result["gold_correct"] is not None:
            mark = "🎯" if result["gold_correct"] else "✗"
            print(f"  {mark} {result['gold_detail']}", flush=True)
        else:
            print(flush=True)

    scored = [r for r in results if r["gold_correct"] is not None]
    report = {
        "name": args.name,
        "url": args.url,
        "model": args.model,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompts": str(Path(args.prompts)),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "cases": results,
        "summary": {
            "cases": len(results),
            "expected_pass": sum(1 for r in results if r["expected_pass"]),
            "gold_correct": sum(1 for r in scored if r["gold_correct"]),
            "gold_scored": len(scored),
            "mean_tok_s": statistics.mean([r["mean_tok_s"] for r in results]) if results else 0.0,
        },
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {out}")
    if scored:
        print(f"[bench] correctness: {report['summary']['gold_correct']}/{len(scored)}"
              f" ({report['summary']['gold_correct']/len(scored)*100:.0f}%)")
    return 0 if report["summary"]["expected_pass"] == len(results) else 1


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
        "baseline_mean_tok_s": statistics.mean([r["baseline_tok_s"] for r in rows]) if rows else 0.0,
        "candidate_mean_tok_s": statistics.mean([r["candidate_tok_s"] for r in rows]) if rows else 0.0,
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
    run.add_argument("--prompts", default=str(Path(__file__).with_name("prompts") / "generation_smoke.jsonl"))
    run.add_argument("--json-out", required=True)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--repeats", type=int, default=1)
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
