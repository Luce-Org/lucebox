#!/usr/bin/env python3
"""Run the publication decode workload at exact input-context lengths.

The DeepSeek tokenizer is read from the target GGUF through the project's C++
tokenizer harness.  Prompts are padded inside the inert reference block so the
generation task remains identical at every length.  The server log remains the
authoritative source for model-side throughput; this client records request
ordering, exact usage, response hashes, TTFT, and transport-side throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from ds4_publication_decode_client import SYSTEM_MESSAGE, build_prompt, stream_request


class TokenizerHarness:
    def __init__(self, executable: Path, model: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def encode_count(self, text: str) -> int:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps({"cmd": "encode", "text": text}) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return len(response["ids"])

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            self.process.stdin.write('{"cmd":"quit"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)


def fit_prompt(
    tokenizer: TokenizerHarness,
    target_server_tokens: int,
    chat_template_overhead: int,
) -> tuple[str, dict[str, int]]:
    """Construct a prompt whose encoded length plus chat overhead is exact."""
    target_raw_tokens = target_server_tokens - chat_template_overhead
    low = 0
    high = max(1, target_server_tokens)
    best_words = 0
    best_count = tokenizer.encode_count(build_prompt(0))
    while low <= high:
        middle = (low + high) // 2
        count = tokenizer.encode_count(build_prompt(middle))
        if count <= target_raw_tokens:
            best_words = middle
            best_count = count
            low = middle + 1
        else:
            high = middle - 1

    # Leave enough room for the filler label, then use one-token " x" units.
    while best_words > 0 and tokenizer.encode_count(build_prompt(best_words, 1)) > target_raw_tokens:
        best_words -= 1
    low = 0
    high = max(64, target_raw_tokens - best_count + 64)
    exact: tuple[str, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt = build_prompt(best_words, middle)
        count = tokenizer.encode_count(prompt)
        if count == target_raw_tokens:
            exact = (prompt, middle)
            break
        if count < target_raw_tokens:
            low = middle + 1
        else:
            high = middle - 1
    if exact is None:
        # Sentence-size plateaus can leave a small gap. Search nearby word
        # counts and filler sizes exhaustively; this runs before model loading.
        for words in range(max(0, best_words - 64), best_words + 1):
            for filler_words in range(0, 192):
                prompt = build_prompt(words, filler_words)
                if tokenizer.encode_count(prompt) == target_raw_tokens:
                    exact = (prompt, filler_words)
                    best_words = words
                    break
            if exact is not None:
                break
    if exact is None:
        raise RuntimeError(f"could not construct exact {target_server_tokens}-token prompt")
    prompt, filler_words = exact
    return prompt, {
        "target_server_tokens": target_server_tokens,
        "raw_prompt_tokens": target_raw_tokens,
        "chat_template_overhead": chat_template_overhead,
        "padding_words": best_words,
        "filler_words": filler_words,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_numeric_fields = (
        "client_decode_tok_s",
        "client_decode_s",
        "prompt_tokens",
        "completion_tokens",
    )
    valid = [
        row
        for row in rows
        if row.get("ok")
        and all(
            isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
            for field in required_numeric_fields
        )
        and float(row["client_decode_s"]) > 0
    ]
    rates = [float(row["client_decode_tok_s"]) for row in valid]
    decode_seconds = sum(float(row["client_decode_s"]) for row in valid)
    completion_tokens = sum(int(row["completion_tokens"]) for row in valid)
    return {
        "n": len(rows),
        "n_ok": len(valid),
        "actual_prompt_tokens": sorted({int(row["prompt_tokens"]) for row in valid}),
        "actual_completion_tokens": sorted({int(row["completion_tokens"]) for row in valid}),
        "client_decode_tok_s_median": round(statistics.median(rates), 3) if rates else None,
        "client_decode_tok_s_min": round(min(rates), 3) if rates else None,
        "client_decode_tok_s_max": round(max(rates), 3) if rates else None,
        "client_decode_tok_s_weighted": (
            round(completion_tokens / decode_seconds, 3) if decode_seconds else None
        ),
        "response_hashes": sorted({str(row["response_sha256"]) for row in valid}),
    }


def compact_group_report(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep sweep order and repeated context lengths in terminal output."""
    return [
        {
            "target_context": row["target_context"],
            "measured_summary": row["measured_summary"],
        }
        for row in groups
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_targets = [
        int(value)
        for value in os.environ.get("CONTEXT_SWEEP_TARGETS", "2048 4096 8192 16384").split()
    ]
    parser.add_argument("--url", default="http://127.0.0.1:18109")
    parser.add_argument("--model", default="dflash")
    parser.add_argument("--model-gguf", type=Path, required=True)
    parser.add_argument("--tokenizer-harness", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=default_targets)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    # Accepted for compatibility with benchmark_publication_suite.sh.
    parser.add_argument("--padding-words", type=int, default=1300)
    parser.add_argument("--calibration-server-tokens", type=int, default=1956)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if not args.model_gguf.is_file():
        parser.error(f"--model-gguf is not a file: {args.model_gguf}")
    if not args.tokenizer_harness.is_file() or not os.access(
        args.tokenizer_harness, os.X_OK
    ):
        parser.error(
            f"--tokenizer-harness is not an executable file: {args.tokenizer_harness}"
        )
    if not args.targets or any(target < 1 for target in args.targets):
        parser.error("--targets must contain positive context lengths")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.padding_words < 0:
        parser.error("--padding-words must be non-negative")
    if args.calibration_server_tokens < 1:
        parser.error("--calibration-server-tokens must be positive")
    if args.expected_sha256 is not None and (
        len(args.expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_sha256)
    ):
        parser.error(
            "--expected-sha256 must contain exactly 64 lowercase hexadecimal characters"
        )

    tokenizer = TokenizerHarness(args.tokenizer_harness, args.model_gguf)
    try:
        calibration_raw = tokenizer.encode_count(build_prompt(args.padding_words))
        overhead = args.calibration_server_tokens - calibration_raw
        if overhead < 0:
            raise RuntimeError("invalid chat-template overhead calibration")
        fitted = {
            target: fit_prompt(tokenizer, target, overhead) for target in args.targets
        }
    finally:
        tokenizer.close()

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "calibration_raw_tokens": calibration_raw,
                    "chat_template_overhead": overhead,
                    "contexts": {str(target): fitted[target][1] for target in args.targets},
                },
                indent=2,
            )
        )
        return 0

    groups: list[dict[str, Any]] = []
    failed = False
    for target in args.targets:
        prompt, construction = fitted[target]
        records: list[dict[str, Any]] = []
        total = args.warmup + args.runs
        print(f"[context-sweep] target={target} requests={total}", flush=True)
        for index in range(total):
            measured = index >= args.warmup
            label = "measure" if measured else "warmup"
            print(
                f"[context-sweep] target={target} {label} {index + 1}/{total}",
                flush=True,
            )
            result = stream_request(args.url, args.model, prompt, args.max_tokens)
            result.update({"index": index, "measured": measured, "target_context": target})
            hash_ok = (
                args.expected_sha256 is None
                or result.get("response_sha256") == args.expected_sha256
            )
            result["expected_hash_match"] = hash_ok
            records.append(result)
            print(
                "[context-sweep] "
                f"ok={result.get('ok')} prompt={result.get('prompt_tokens')} "
                f"output={result.get('completion_tokens')} "
                f"client_decode={result.get('client_decode_tok_s')} tok/s "
                f"sha={result.get('response_sha256')} hash_ok={hash_ok}",
                flush=True,
            )
            if (
                not result.get("ok")
                or int(result.get("prompt_tokens") or -1) != target
                or int(result.get("completion_tokens") or -1) != args.max_tokens
                or not hash_ok
            ):
                failed = True
                break
        measured_rows = [row for row in records if row["measured"]]
        groups.append(
            {
                "target_context": target,
                "construction": construction,
                "records": records,
                "measured_summary": summarize(measured_rows),
            }
        )
        if failed:
            break

    payload = {
        "schema_version": 1,
        "workload": "deterministic-exact-context-sweep",
        "targets": args.targets,
        "temperature": 0,
        "batch_size": 1,
        "max_tokens": args.max_tokens,
        "warmup_per_context": args.warmup,
        "runs_per_context": args.runs,
        "expected_sha256": args.expected_sha256,
        "calibration_server_tokens": args.calibration_server_tokens,
        "calibration_raw_tokens": calibration_raw,
        "chat_template_overhead": overhead,
        "system_message": SYSTEM_MESSAGE,
        "groups": groups,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact_group_report(groups), indent=2))
    return 1 if failed or len(groups) != len(args.targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
