#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / "server/build/dflash_server"
DEFAULT_LOG = REPO_ROOT / ".harness-work/compaction_test_server.log"


def http_get(url: str):
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        return resp.status, resp.read().decode("utf-8")


def http_post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def wait_for_health(base_url: str, deadline_s: float = 120.0) -> None:
    deadline = time.time() + deadline_s
    last_error = None
    while time.time() < deadline:
        try:
            status, _ = http_get(f"{base_url}/health")
            if status == 200:
                return
        except Exception as exc:  # pragma: no cover - harness convenience
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def make_tool_payload(repeats: int, size: int, path: str) -> list[dict]:
    tool_blob = json.dumps({"path": path, "content": "X" * size})
    items: list[dict] = [
        {"role": "developer", "content": "Answer with the single word OK."},
        {"role": "user", "content": "Use prior tool results and answer tersely."},
    ]
    for idx in range(repeats):
        items.append(
            {
                "role": "tool",
                "content": tool_blob,
                "tool_call_id": f"tool_{idx}",
            }
        )
    items.append({"role": "assistant", "content": "<think>hidden reasoning</think>Ready."})
    items.append({"role": "user", "content": "Final answer only."})
    return items


def assert_no_compaction(resp: dict) -> None:
    usage = resp.get("usage", {})
    saved = usage.get("compacted_tokens_saved", 0)
    assert saved == 0, f"expected no compaction, got compacted_tokens_saved={saved}"


def assert_compaction(resp: dict, label: str) -> None:
    usage = resp.get("usage", {})
    saved = usage.get("compacted_tokens_saved", 0)
    assert saved > 0, f"expected compaction for {label}, got usage={usage}"


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.server_bin),
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-name",
        args.model_name,
        "--max-ctx",
        str(args.max_ctx),
        "--max-tokens",
        str(args.max_output_tokens),
        "--compaction-threshold",
        "0.5",
    ]
    if args.draft:
        cmd += ["--draft", args.draft]
    if args.prefill_drafter:
        cmd += ["--prefill-drafter", args.prefill_drafter]
    if args.extra_server_args:
        cmd += shlex.split(args.extra_server_args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an end-to-end compaction smoke test.")
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--model", default=os.getenv("TARGET") or os.getenv("MODEL_PATH"))
    parser.add_argument("--draft", default=os.getenv("DRAFT"))
    parser.add_argument("--prefill-drafter", default=os.getenv("PREFILL_DRAFTER"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("COMPACTION_TEST_PORT", "18081")))
    parser.add_argument("--model-name", default=os.getenv("MODEL_ID", "luce-dflash"))
    parser.add_argument("--max-ctx", type=int, default=int(os.getenv("MAX_CTX", "2048")))
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--extra-server-args", default=os.getenv("COMPACTION_SERVER_EXTRA_ARGS", ""))
    parser.add_argument("--server-log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    if not args.model:
        print("Set --model or TARGET/MODEL_PATH before running this harness.", file=sys.stderr)
        return 2
    if not args.server_bin.exists():
        print(f"Server binary not found: {args.server_bin}", file=sys.stderr)
        return 2

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    cmd = build_command(args)

    print("[compaction-test] starting server:", " ".join(shlex.quote(part) for part in cmd))
    with args.server_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(base_url)

            short_resp = http_post_json(
                f"{base_url}/v1/responses",
                {
                    "model": args.model_name,
                    "stream": False,
                    "max_output_tokens": args.max_output_tokens,
                    "input": [
                        {"role": "developer", "content": "Answer with OK."},
                        {"role": "user", "content": "Say OK."},
                    ],
                },
            )
            assert_no_compaction(short_resp)
            print("[compaction-test] short request passed")

            long_resp = http_post_json(
                f"{base_url}/v1/responses",
                {
                    "model": args.model_name,
                    "stream": False,
                    "max_output_tokens": args.max_output_tokens,
                    "input": make_tool_payload(repeats=10, size=1000, path="/repo/a.cpp"),
                },
            )
            assert_compaction(long_resp, "long request")
            print("[compaction-test] long request passed")

            override_resp = http_post_json(
                f"{base_url}/v1/responses",
                {
                    "model": args.model_name,
                    "stream": False,
                    "max_output_tokens": args.max_output_tokens,
                    "context_management": [
                        {"type": "compaction", "compact_threshold": 128}
                    ],
                    "input": make_tool_payload(repeats=4, size=700, path="/repo/b.cpp"),
                },
            )
            assert_compaction(override_resp, "context_management override")
            print("[compaction-test] override request passed")
            return 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
