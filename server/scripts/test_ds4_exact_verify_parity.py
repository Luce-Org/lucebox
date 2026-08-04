#!/usr/bin/env python3
"""Model-backed token parity gate for DS4 reference-exact verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

TRACE_RE = re.compile(
    r"^\[ds4-exact-verify-trace\] reference_exact=([01]) speculation=([01]) "
    r"n=(\d+) ids=\[([0-9 ]*)\]$",
    re.MULTILINE,
)
SPEC_SUMMARY_RE = re.compile(
    r"^\[ds4-spec\] gen=(\d+) steps=(\d+) matched=(\d+) offered=(\d+) "
    r".*\bfull_snap=([01])$",
    re.MULTILINE,
)

SANITIZED_POLICY_PREFIXES = (
    "DFLASH_DS4_",
    "DFLASH_EXPERT_",
    "DFLASH_MOE_",
)
SANITIZED_POLICY_NAMES = {
    "DFLASH_MMQ_SUB_BATCH",
    "GGML_BATCH_PEER_COPIES",
    "GGML_CUDA_BATCH_PEER_COPIES",
    "LUCE_MMVQ_MAX_NCOLS",
}


class TokenTrace(NamedTuple):
    reference_exact: bool
    speculation: bool
    tokens: tuple[int, ...]


class SpeculationSummary(NamedTuple):
    generated: int
    steps: int
    matched: int
    offered: int
    full_snapshot: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def parse_token_trace(
    log: str,
    *,
    expected_reference_exact: bool,
    expected_speculation: bool,
) -> TokenTrace:
    matches = list(TRACE_RE.finditer(log))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one token trace, found {len(matches)}")
    match = matches[0]
    declared_count = int(match.group(3))
    raw_ids = match.group(4)
    tokens = tuple(int(token) for token in raw_ids.split()) if raw_ids else ()
    if declared_count != len(tokens):
        raise RuntimeError(f"token trace declared {declared_count} IDs but contained {len(tokens)}")
    if not tokens:
        raise RuntimeError("token trace is empty")
    trace = TokenTrace(
        reference_exact=match.group(1) == "1",
        speculation=match.group(2) == "1",
        tokens=tokens,
    )
    if trace.reference_exact != expected_reference_exact:
        raise RuntimeError(
            "reference-exact mode mismatch: "
            f"expected {int(expected_reference_exact)}, got {int(trace.reference_exact)}"
        )
    if trace.speculation != expected_speculation:
        raise RuntimeError(
            "speculation mode mismatch: "
            f"expected {int(expected_speculation)}, got {int(trace.speculation)}"
        )
    return trace


def require_speculation_work(log: str) -> SpeculationSummary:
    matches = list(SPEC_SUMMARY_RE.finditer(log))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one DS4 speculation summary, found {len(matches)}")
    match = matches[0]
    summary = SpeculationSummary(
        generated=int(match.group(1)),
        steps=int(match.group(2)),
        matched=int(match.group(3)),
        offered=int(match.group(4)),
        full_snapshot=match.group(5) == "1",
    )
    if summary.generated <= 0 or summary.steps <= 0 or summary.offered <= 0:
        raise RuntimeError(
            "DS4 speculation did no work: "
            f"generated={summary.generated}, steps={summary.steps}, "
            f"offered={summary.offered}"
        )
    if summary.matched >= summary.offered:
        raise RuntimeError(
            "reference-exact run did not exercise rejection rollback: "
            f"matched={summary.matched}, offered={summary.offered}"
        )
    if not summary.full_snapshot:
        raise RuntimeError("reference-exact run did not enable full rollback snapshots")
    if "[ds4-spec] reference-exact verifier:" not in log:
        raise RuntimeError("reference-exact verifier activation banner is missing")
    return summary


def token_mismatch_message(ar_tokens: tuple[int, ...], exact_tokens: tuple[int, ...]) -> str | None:
    common_length = min(len(ar_tokens), len(exact_tokens))
    first_mismatch = next(
        (index for index in range(common_length) if ar_tokens[index] != exact_tokens[index]),
        None,
    )
    if first_mismatch is not None:
        return (
            f"first token mismatch at {first_mismatch}: "
            f"ar={ar_tokens[first_mismatch : first_mismatch + 4]} "
            f"exact={exact_tokens[first_mismatch : first_mismatch + 4]}"
        )
    if len(ar_tokens) != len(exact_tokens):
        return (
            f"token trace length mismatch after common prefix of {common_length}: "
            f"ar={len(ar_tokens)} exact={len(exact_tokens)}"
        )
    return None


def retain_failed_attempt(attempt_dir: Path) -> Path:
    failed_dir = attempt_dir.with_name(attempt_dir.name.replace("attempt-", "failed-", 1))
    attempt_dir.replace(failed_dir)
    return failed_dir


def promote_evidence(staged_paths: list[Path], final_paths: list[Path]) -> None:
    promoted: list[Path] = []
    try:
        for staged, final in zip(staged_paths, final_paths, strict=True):
            os.link(staged, final)
            promoted.append(final)
    except Exception:
        for final in reversed(promoted):
            final.unlink(missing_ok=True)
        raise
    for staged in staged_paths:
        try:
            staged.unlink()
        except OSError:
            pass


def wait_ready(port: int, proc: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness: {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout:.0f}s")


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def case_environment(args: argparse.Namespace, reference_exact: bool) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name in SANITIZED_POLICY_NAMES or name.startswith(SANITIZED_POLICY_PREFIXES):
            env.pop(name)
    env.update(
        {
            "DFLASH_DS4_ADAPTIVE_WIDTH": "0",
            "DFLASH_DS4_EXACT_VERIFY_TRACE": "1",
            "DFLASH_DS4_SPEC_Q": str(args.spec_q),
            "LUCE_MMVQ_MAX_NCOLS": str(args.mmvq_max_ncols),
        }
    )
    if reference_exact:
        env.update(
            {
                "DFLASH_DS4_DRAFT": str(args.draft),
                "DFLASH_DS4_SPEC": "1",
                "DFLASH_DS4_SPEC_REFERENCE_EXACT": "1",
            }
        )
    return env


def run_case(
    args: argparse.Namespace,
    *,
    reference_exact: bool,
    prompt: str,
    log_path: Path,
) -> TokenTrace:
    command = [
        str(args.server_bin),
        str(args.target),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-ctx",
        str(args.max_ctx),
        "--chunk",
        str(args.prefill_chunk),
        "--target-device",
        args.target_device,
        "--ds4-fused-decode",
        "--ds4-prefill",
        "exact",
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            env=case_environment(args, reference_exact),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(args.port, proc, args.startup_timeout)
            body = json.dumps(
                {
                    "model": "dflash",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                }
            ).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"generation returned HTTP {response.status}")
                json.load(response)
        finally:
            stop_server(proc)

    log_text = log_path.read_text(errors="replace")
    trace = parse_token_trace(
        log_text,
        expected_reference_exact=reference_exact,
        expected_speculation=reference_exact,
    )
    if reference_exact:
        require_speculation_work(log_text)
    return trace


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DS4 reference-exact speculative decode with greedy AR",
    )
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--draft-sha256", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-sha256", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--target-device", default="hip:0")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spec-q", type=int, default=4)
    parser.add_argument("--mmvq-max-ncols", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=600)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    for path in (args.server_bin, args.target, args.draft, args.prompt_file):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2 so speculation can run")
    if not 2 <= args.spec_q <= 4:
        parser.error("--spec-q must be between 2 and 4")
    if args.mmvq_max_ncols < args.spec_q:
        parser.error("--mmvq-max-ncols must be at least --spec-q")

    for path, expected in (
        (args.server_bin, args.server_sha256),
        (args.target, args.target_sha256),
        (args.draft, args.draft_sha256),
        (args.prompt_file, args.prompt_sha256),
    ):
        require_sha256(path, expected)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        parser.error("--prompt-file must not be empty")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    evidence_paths = [
        args.log_dir / "ar.log",
        args.log_dir / "reference-exact.log",
        args.log_dir / "manifest.json",
    ]
    existing = [str(path) for path in evidence_paths if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite evidence files: {', '.join(existing)}")
    attempt_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.log_dir))
    staged_paths = [attempt_dir / path.name for path in evidence_paths]
    try:
        ar = run_case(
            args,
            reference_exact=False,
            prompt=prompt,
            log_path=staged_paths[0],
        )
        exact = run_case(
            args,
            reference_exact=True,
            prompt=prompt,
            log_path=staged_paths[1],
        )
        manifest = {
            "server": {"path": str(args.server_bin), "sha256": args.server_sha256.lower()},
            "target": {"path": str(args.target), "sha256": args.target_sha256.lower()},
            "draft": {"path": str(args.draft), "sha256": args.draft_sha256.lower()},
            "prompt": {"path": str(args.prompt_file), "sha256": args.prompt_sha256.lower()},
            "target_device": args.target_device,
            "max_ctx": args.max_ctx,
            "prefill_chunk": args.prefill_chunk,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "spec_q": args.spec_q,
            "mmvq_max_ncols": args.mmvq_max_ncols,
            "ar_tokens": list(ar.tokens),
            "reference_exact_tokens": list(exact.tokens),
        }
        staged_paths[2].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        mismatch = token_mismatch_message(ar.tokens, exact.tokens)
        if mismatch is not None:
            failed_dir = retain_failed_attempt(attempt_dir)
            print(f"FAIL: {mismatch}; diagnostics retained in {failed_dir}")
            return 1
        promote_evidence(staged_paths, evidence_paths)
    except Exception as error:
        failed_dir = retain_failed_attempt(attempt_dir)
        print(f"FAIL: {error}; diagnostics retained in {failed_dir}", file=sys.stderr)
        return 1
    try:
try:
    attempt_dir.rmdir()
except OSError as error:
    print(f"warning: could not remove staging dir {attempt_dir}: {error}", file=sys.stderr)
    print(
        f"PASS: {len(ar.tokens)} generated token IDs are identical; "
        "reference-exact speculation executed at least one step"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
