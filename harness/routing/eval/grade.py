"""Grade collected answers -> runs/<name>/grades.jsonl.

Row: {id, model, thinking, correct, score, grader, detail}

Deterministic graders run locally (python_tests in a sandboxed subprocess). Prompts
with grader type ``judge`` -- and ``math`` misses with ``judge_fallback`` -- go to the
LLM judge (default: the 27B on :8216, thinking off, temperature 0). For judge
prompts the reference is the prompt's own ``reference`` if set, else the 27B
thinking-on answer from the same run when one was collected.

Resumable: keys already graded (and not ``judge_error``) are skipped. ``--regrade``
starts over.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import (
    PROMPTS_PATH,
    REFERENCE_MODEL,
    RUNS_DIR,
    append_jsonl,
    latest_by_key,
    load_backends,
    prompt_text,
    read_jsonl,
    strip_think,
)
from .graders import GradeResult, grade_deterministic
from .judge import Judge, JudgeConfig


def reference_answers(answers: dict[tuple[str, str, bool], dict[str, Any]], ref_model: str) -> dict[str, str]:
    """{prompt id: reference text} from the reference model's thinking-on answers."""
    refs = {}
    for (pid, model, thinking), row in answers.items():
        if model == ref_model and thinking and not row.get("error") and row.get("finish_reason") != "length":
            text = strip_think(row.get("text"))
            if text:
                refs[pid] = text
    return refs


def needs_regrade(grade: dict[str, Any], refs: dict[str, str]) -> bool:
    """Judge failures, and judge grades made before a reference answer existed."""
    if grade["grader"] == "judge_error":
        return True
    is_ref_row = grade["model"] == REFERENCE_MODEL and grade["thinking"]
    return (grade["grader"] == "judge" and not grade["detail"].get("reference")
            and grade["id"] in refs and not is_ref_row)


def error_result(row: dict[str, Any]) -> GradeResult:
    return GradeResult(False, 0.0, "no_answer", {"error": row.get("error")})


async def grade_rows(
    prompts: dict[str, dict[str, Any]],
    answers: list[dict[str, Any]],
    judge: Judge | None,
    refs: dict[str, str],
    out_path: Path,
    judge_concurrency: int = 4,
    test_workers: int = 8,
) -> dict[str, int]:
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=test_workers)
    sem = asyncio.Semaphore(judge_concurrency)
    counts = {"graded": 0, "judge_calls": 0, "skipped_no_judge": 0}

    async def run_judge(prompt: dict[str, Any], text: str, reference: str | None) -> GradeResult:
        counts["judge_calls"] += 1
        async with sem:
            return await judge.grade(prompt_text(prompt["messages"]), text, reference,
                                     prompt["grader"].get("rubric"))

    async def one(row: dict[str, Any]) -> None:
        prompt = prompts.get(row["id"])
        if prompt is None:
            return
        g = prompt["grader"]
        if row.get("error"):
            res = error_result(row)
        elif g["type"] == "judge":
            if judge is None:
                counts["skipped_no_judge"] += 1
                return
            is_ref_row = row["model"] == REFERENCE_MODEL and row["thinking"]
            reference = g.get("reference") or (None if is_ref_row else refs.get(row["id"]))
            res = await run_judge(prompt, row["text"], reference)
        else:
            res = await loop.run_in_executor(pool, grade_deterministic, row["text"], g)
            if not res.correct and g.get("judge_fallback") and judge is not None:
                jr = await run_judge(prompt, row["text"], f"Final answer: {g['answer']}")
                if jr.grader != "judge_error":
                    res = GradeResult(jr.correct, jr.score, f"{res.grader}+judge",
                                      {**res.detail, "judge": jr.detail})
        out = {"id": row["id"], "model": row["model"], "thinking": row["thinking"],
               "correct": res.correct, "score": res.score, "grader": res.grader,
               "detail": res.detail, "finish_reason": row.get("finish_reason")}
        append_jsonl(out_path, out)
        counts["graded"] += 1

    try:
        await asyncio.gather(*(one(r) for r in answers))
    finally:
        pool.shutdown(wait=True)
    return counts


async def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / args.run
    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    answers = latest_by_key(read_jsonl(run_dir / "answers.jsonl"))
    out_path = run_dir / "grades.jsonl"
    if args.regrade and out_path.exists():
        out_path.unlink()
    refs = reference_answers(answers, args.reference_model)
    done = {k for k, row in latest_by_key(read_jsonl(out_path)).items() if not needs_regrade(row, refs)}
    todo = [row for k, row in answers.items() if k not in done]
    if args.models:
        keep = set(args.models.split(","))
        todo = [r for r in todo if r["model"] in keep]

    judge = None
    if not args.no_judge:
        backends = load_backends(args.backends)
        spec = backends.get(args.judge_model, {})
        base_url = args.judge_url or spec.get("base_url", "http://127.0.0.1:8216/v1")
        judge = Judge(JudgeConfig(base_url=base_url, model=spec.get("served_name", args.judge_model)))
    print(f"{len(todo)} answers to grade ({len(refs)} judge references) -> {out_path}", flush=True)
    try:
        counts = await grade_rows(prompts, todo, judge, refs, out_path,
                                  args.judge_concurrency, args.test_workers)
    finally:
        if judge is not None:
            await judge.aclose()
    print(counts)
    summary: dict[tuple[str, bool], list[float]] = {}
    for row in latest_by_key(read_jsonl(out_path)).values():
        summary.setdefault((row["model"], row["thinking"]), []).append(row["score"])
    for (model, thinking), scores in sorted(summary.items()):
        print(f"  {model:<14} think={'on ' if thinking else 'off'} n={len(scores):>4} "
              f"mean={sum(scores) / len(scores):.3f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--backends", default=None)
    ap.add_argument("--models", default=None, help="only grade these models")
    ap.add_argument("--judge-model", default=REFERENCE_MODEL)
    ap.add_argument("--judge-url", default=None, help="override judge base_url (.../v1)")
    ap.add_argument("--reference-model", default=REFERENCE_MODEL)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--test-workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="deterministic graders only")
    ap.add_argument("--regrade", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.run or args.run_dir):
        raise SystemExit("--run or --run-dir required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
