#!/usr/bin/env python3
"""Command-line entry point for validation-v2 preparation and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .niah import generate_control_cases, generate_primary_cases, write_cases
from .records import ResultStore, canonical_json, validate_complete_matrix
from .scoring import score_niah_records
from .statistics import paired_prompt_values, summarize_paired


def _pinned_tokenizer_counter(identifier: str, revision: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        revision=revision,
        trust_remote_code=False,
    )
    resolved = tokenizer.init_kwargs.get("_commit_hash")
    if resolved != revision:
        raise ValueError(
            f"tokenizer resolved revision {resolved!r}, expected immutable {revision!r}"
        )

    def count(prompt: str) -> int:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("pinned tokenizer returned no token IDs")
        return len(token_ids)

    return count


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(raw)
    return rows


def command_niah(args: argparse.Namespace) -> None:
    token_counter = _pinned_tokenizer_counter(args.tokenizer, args.tokenizer_revision)
    cases = generate_primary_cases(
        context_tokens=args.context_tokens,
        tokenizer_revision=args.tokenizer_revision,
        seed_base=args.seed_base,
        token_counter=token_counter,
    )
    if args.controls:
        cases.extend(
            generate_control_cases(
                context_tokens=args.context_tokens,
                tokenizer_revision=args.tokenizer_revision,
                seed_base=args.seed_base,
                token_counter=token_counter,
            )
        )
    digest = write_cases(args.out, cases)
    print(canonical_json({"cases": len(cases), "path": str(args.out), "sha256": digest}))


def command_validate_matrix(args: argparse.Namespace) -> None:
    planned = _jsonl(args.plan)
    rows = ResultStore(args.results).rows()
    counts = validate_complete_matrix(planned, rows, required_metrics=args.require_metric)
    print(canonical_json({"complete": True, "counts": counts}))


def command_score_niah(args: argparse.Namespace) -> None:
    summary = score_niah_records(_jsonl(args.cases), _jsonl(args.responses))
    rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def command_summarize(args: argparse.Namespace) -> None:
    rows = _jsonl(args.results)
    pairs = paired_prompt_values(
        rows,
        baseline=args.baseline,
        candidate=args.candidate,
        metric=args.metric,
        expected_repetitions=args.expected_repetitions,
    )
    summary = summarize_paired(
        pairs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    niah = commands.add_parser("niah-generate", help="generate a sealed NIAH-v2 case set")
    niah.add_argument("--context-tokens", type=int, required=True, choices=(32768, 65536, 131072))
    niah.add_argument("--tokenizer-revision", required=True)
    niah.add_argument("--tokenizer", required=True)
    niah.add_argument("--seed-base", type=int, default=20260718)
    niah.add_argument("--controls", action="store_true")
    niah.add_argument("--out", type=Path, required=True)
    niah.set_defaults(function=command_niah)

    matrix = commands.add_parser("validate-matrix", help="fail unless every planned cell succeeded")
    matrix.add_argument("--plan", type=Path, required=True)
    matrix.add_argument("--results", type=Path, required=True)
    matrix.add_argument("--require-metric", action="append", default=[])
    matrix.set_defaults(function=command_validate_matrix)

    score = commands.add_parser("score-niah", help="strictly score a complete NIAH response set")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--out", type=Path)
    score.set_defaults(function=command_score_niah)

    summary = commands.add_parser("summarize-paired", help="paired prompt-level bootstrap summary")
    summary.add_argument("--results", type=Path, required=True)
    summary.add_argument("--baseline", required=True)
    summary.add_argument("--candidate", required=True)
    summary.add_argument("--metric", required=True)
    summary.add_argument("--expected-repetitions", type=int, default=5)
    summary.add_argument("--bootstrap-samples", type=int, default=10_000)
    summary.add_argument("--seed", type=int, default=20260718)
    summary.add_argument("--out", type=Path)
    summary.set_defaults(function=command_summarize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
