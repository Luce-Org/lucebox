#!/usr/bin/env python3
"""Model-backed token parity regression for the public DS4 fused-verify flag."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


TOKEN_RE = re.compile(r"\[ds4-parity-tokens\] n=\d+ ids=\[([0-9 ]*)\]")


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


def run_case(args: argparse.Namespace, fused_requested: bool, log_path: Path) -> list[int]:
    env = os.environ.copy()
    for name in (
        "DFLASH_DS4_FUSED_VERIFY",
        "DFLASH_DS4_ALLOW_APPROX_FUSED_VERIFY",
        "DFLASH_DS4_SEQ_VERIFY",
        "DFLASH_DS4_DSPARK_DEBUG",
    ):
        env.pop(name, None)
    env.update({
        "DFLASH_DS4_SPEC": "1",
        "DFLASH_DS4_DRAFT": str(args.draft),
        "DFLASH_DS4_SPEC_Q": "4",
        "DFLASH_DS4_ADAPTIVE_WIDTH": "0",
        "DFLASH_DS4_PARITY_TRACE": "1",
        "LUCE_MMVQ_MAX_NCOLS": "4",
    })
    if fused_requested:
        env["DFLASH_DS4_FUSED_VERIFY"] = "1"

    cmd = [
        str(args.server_bin), str(args.target),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-ctx", str(args.max_ctx), "--chunk", "512",
        "--target-device", "hip:0", "--ds4-fused-decode",
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(args.port, proc, args.startup_timeout)
            body = json.dumps({
                "model": "dflash",
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0,
                "seed": 1234,
                "max_tokens": args.max_tokens,
                "stream": False,
            }).encode()
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

    matches = TOKEN_RE.findall(log_path.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"token trace missing from {log_path}")
    return [int(token) for token in matches[-1].split()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--prompt",
        default="Explain why a bicycle stays upright while moving.",
    )
    args = parser.parse_args()
    for path in (args.server_bin, args.target, args.draft):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    with tempfile.TemporaryDirectory(prefix="ds4-fused-parity-") as tmp:
        tmp_path = Path(tmp)
        normal = run_case(args, False, tmp_path / "normal.log")
        fused_flag = run_case(args, True, tmp_path / "fused-flag.log")

    if normal != fused_flag:
        limit = min(len(normal), len(fused_flag))
        first = next((i for i in range(limit) if normal[i] != fused_flag[i]), limit)
        print(f"FAIL: first token mismatch at {first}: normal={normal[first:first+4]} "
              f"fused_flag={fused_flag[first:first+4]}")
        return 1
    print(f"PASS: {len(normal)} generated token IDs are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
