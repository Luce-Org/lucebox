"""Build the routing eval prompt set -> data/prompts.jsonl.

Mixes public benchmarks (downloaded with HF ``datasets`` and cached under
data/cache/) with the hand-written easy / hard prompts in data/. Sampling is
deterministic for a given ``--seed``; the train/test split is stratified by source.

    uv run --no-project --with datasets --with pyarrow \
        python -m eval.build_set            # from harness/routing/
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import DATA_DIR, PROMPTS_PATH, read_jsonl, write_jsonl
from .ifeval_checks import SUPPORTED as IFEVAL_SUPPORTED

CACHE_DIR = DATA_DIR / "cache" / "hf"

# Default number of prompts per public source (handwritten files are taken whole).
DEFAULT_COUNTS = {
    "gsm8k": 80,
    "mmlu_pro": 84,  # 6 per subject x 14 subjects
    "ifeval": 60,
    "math500": 60,  # 12 per difficulty level x 5
    "humaneval": 40,
    "mbpp": 30,
}


def _load(name: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, cache_dir=str(CACHE_DIR))


def _user(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def _maybe_list(v: Any) -> Any:
    """Some dataset viewers stringify list columns; accept both forms."""
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _stratified_sample(
    rows: list[dict[str, Any]], n: int, key: Callable[[dict[str, Any]], str], rng: random.Random
) -> list[dict[str, Any]]:
    """Round-robin over strata (sorted) so every stratum gets ~n/k rows."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out: list[dict[str, Any]] = []
    names = sorted(groups)
    i = 0
    while len(out) < n and any(groups.values()):
        g = groups[names[i % len(names)]]
        if g:
            out.append(g.pop())
        i += 1
    return out


# --------------------------------------------------------------------------- sources


def gsm8k(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("openai/gsm8k", "main", "test")
    idx = sorted(rng.sample(range(len(ds)), n))
    out = []
    for i in idx:
        r = ds[i]
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "id": f"gsm8k-{i}",
            "source": "gsm8k",
            "category": "math_word",
            "messages": _user(
                f"{r['question']}\n\nSolve step by step, then give the final answer as "
                "\\boxed{N} where N is a number."),
            "grader": {"type": "numeric", "answer": float(answer), "tolerance": 1e-4},
        })
    return out


LETTERS = "ABCDEFGHIJ"


def mmlu_pro(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("TIGER-Lab/MMLU-Pro", None, "test")
    rows = [dict(r, _i=i) for i, r in enumerate(ds)]
    picked = _stratified_sample(rows, n, lambda r: r["category"], rng)
    out = []
    for r in sorted(picked, key=lambda r: r["_i"]):
        options = _maybe_list(r["options"])
        opts = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
        out.append({
            "id": f"mmlu_pro-{r['question_id']}",
            "source": "mmlu_pro",
            "category": "knowledge_mc",
            "meta": {"subject": r["category"]},
            "messages": _user(
                "Answer the following multiple choice question. Think briefly, then finish with "
                f"the line \"The answer is (X)\" where X is the letter of the correct option.\n\n"
                f"Question: {r['question']}\n\nOptions:\n{opts}"),
            "grader": {"type": "choice", "answer": r["answer"], "n_options": len(options)},
        })
    return out


def ifeval(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("google/IFEval", None, "train")
    ok = []
    for r in ds:
        ids = _maybe_list(r["instruction_id_list"])
        if all(i in IFEVAL_SUPPORTED for i in ids):
            ok.append(r)
    picked = rng.sample(ok, min(n, len(ok)))
    out = []
    for r in sorted(picked, key=lambda r: int(r["key"])):
        kwargs = [
            {k: v for k, v in kw.items() if v is not None}
            for kw in _maybe_list(r["kwargs"])
        ]
        out.append({
            "id": f"ifeval-{r['key']}",
            "source": "ifeval",
            "category": "instruction_following",
            "messages": _user(r["prompt"]),
            "grader": {
                "type": "ifeval",
                "instruction_id_list": list(_maybe_list(r["instruction_id_list"])),
                "kwargs": kwargs,
                "prompt": r["prompt"],
            },
        })
    return out


def math500(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("HuggingFaceH4/MATH-500", None, "test")
    rows = [dict(r, _i=i) for i, r in enumerate(ds)]
    picked = _stratified_sample(rows, n, lambda r: str(r["level"]), rng)
    out = []
    for r in sorted(picked, key=lambda r: r["_i"]):
        uid = r["unique_id"].removeprefix("test/").removesuffix(".json").replace("/", "-")
        out.append({
            "id": f"math500-{uid}",
            "source": "math500",
            "category": "math_hard",
            "meta": {"subject": r["subject"], "level": int(r["level"])},
            "messages": _user(f"{r['problem']}\n\nPut your final answer in \\boxed{{}}."),
            # Deterministic LaTeX-normalised match first; the judge re-checks misses
            # against the gold answer (equivalent forms like 0.5 vs \frac12).
            "grader": {"type": "math", "answer": r["answer"], "judge_fallback": True},
        })
    return out


def humaneval(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("openai/openai_humaneval", None, "test")
    idx = sorted(rng.sample(range(len(ds)), n))
    out = []
    for i in idx:
        r = ds[i]
        out.append({
            "id": f"humaneval-{r['task_id'].split('/')[-1]}",
            "source": "humaneval",
            "category": "code",
            "messages": _user(
                "Complete the following Python function. Reply with the complete function "
                "(signature, imports and body) in a single ```python code block.\n\n"
                f"```python\n{r['prompt']}```"),
            "grader": {
                "type": "python_tests",
                "tests": r["test"],
                "entry_point": r["entry_point"],
                "prompt_prefix": r["prompt"],
                "timeout": 10,
            },
        })
    return out


def mbpp(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _load("google-research-datasets/mbpp", "sanitized", "test")
    idx = sorted(rng.sample(range(len(ds)), n))
    out = []
    for i in idx:
        r = ds[i]
        tests = list(_maybe_list(r["test_list"]))
        imports = list(_maybe_list(r["test_imports"]))
        out.append({
            "id": f"mbpp-{r['task_id']}",
            "source": "mbpp",
            "category": "code",
            "messages": _user(
                f"{r['prompt']}\nYour code should pass these tests:\n\n" + "\n".join(tests)
                + "\n\nReply with the code in a single ```python code block."),
            "grader": {
                "type": "python_tests",
                "tests": "\n".join(imports + tests),
                "entry_point": None,
                "timeout": 10,
            },
        })
    return out


def handwritten(path: Path, source: str) -> list[dict[str, Any]]:
    return [dict(r, source=source) for r in read_jsonl(path)]


PUBLIC = {
    "gsm8k": gsm8k,
    "mmlu_pro": mmlu_pro,
    "ifeval": ifeval,
    "math500": math500,
    "humaneval": humaneval,
    "mbpp": mbpp,
}


# --------------------------------------------------------------------------- split


def assign_split(rows: list[dict[str, Any]], train_frac: float, seed: int) -> None:
    """Stratified by (source, category): shuffle each stratum, first round(frac*n) -> train."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        strata[(r["source"], r["category"])].append(r)
    for (source, category), group in sorted(strata.items()):
        ids = sorted(r["id"] for r in group)
        random.Random(f"{seed}:{source}:{category}").shuffle(ids)
        n_train = round(train_frac * len(ids))
        train = set(ids[:n_train])
        for r in group:
            r["split"] = "train" if r["id"] in train else "test"


def build(
    counts: dict[str, int], seed: int, train_frac: float, include_handwritten: bool = True
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, fn in PUBLIC.items():
        n = counts.get(source, 0)
        if n > 0:
            rows.extend(fn(n, random.Random(f"{seed}:{source}")))
    if include_handwritten:
        rows.extend(handwritten(DATA_DIR / "handwritten_easy.jsonl", "handwritten_easy"))
        rows.extend(handwritten(DATA_DIR / "handwritten_hard.jsonl", "handwritten_hard"))
    ids = [r["id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate prompt ids: {dup[:5]}")
    assign_split(rows, train_frac, seed)
    key_order = ["id", "source", "category", "split", "messages", "grader", "meta"]
    return [{k: r[k] for k in key_order if k in r} for r in rows]


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = [f"total {len(rows)}"]
    by = Counter((r["source"], r["category"], r["split"]) for r in rows)
    agg: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (s, c, sp), n in by.items():
        agg[(s, c)][sp] += n
    lines.append(f"{'source':<18}{'category':<24}{'train':>6}{'test':>6}")
    for (s, c), cnt in sorted(agg.items()):
        lines.append(f"{s:<18}{c:<24}{cnt['train']:>6}{cnt['test']:>6}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--count", action="append", default=[], metavar="SOURCE=N",
                    help=f"override per-source counts (defaults: {DEFAULT_COUNTS})")
    ap.add_argument("--no-handwritten", action="store_true")
    args = ap.parse_args(argv)
    counts = dict(DEFAULT_COUNTS)
    for item in args.count:
        k, v = item.split("=", 1)
        if k not in PUBLIC:
            ap.error(f"unknown source {k!r}")
        counts[k] = int(v)
    rows = build(counts, args.seed, args.train_frac, not args.no_handwritten)
    write_jsonl(args.out, rows)
    print(summarize(rows))
    print(f"wrote {args.out}")
    meta = {"seed": args.seed, "train_frac": args.train_frac, "counts": counts, "n": len(rows)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
