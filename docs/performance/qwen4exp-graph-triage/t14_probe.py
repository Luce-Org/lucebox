#!/usr/bin/env python3
"""Direct, lock-owned qwen4exp 13k token-stream probe."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

REPO = Path("/home/duster/lucebox-qwen4exp")
OUT = Path("/tmp/qwen4exp-graph-triage")
MODEL = Path("/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf")
PROMPTS = REPO / "harness/benchmarks/prompts/bench_recall.jsonl"
BASE_ENV = {
    "HIP_VISIBLE_DEVICES": "1", "DFLASH_HIP_NO_AUTO_UMA": "1",
    "GGML_CUDA_MMB": "1", "QWEN4EXP_QSA": "1",
    "QWEN4EXP_MMB_CUBLAS": "5", "DFLASH_MMB_SHADOW": "1",
    "LLAMA_MMB_HC16": "2", "QWEN4EXP_LAST_TOKEN_FFN": "1",
    "QWEN4EXP_DENSE_TABLE": "1", "QWEN4EXP_HC_TILE16": "1",
    "QWEN4EXP_DECODE_REUSE": "1",
}


def prompt_13k():
    with PROMPTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] == "recall_13k":
                return row["messages"], row["max_tokens"]
    raise RuntimeError("recall_13k prompt missing")


def request(port, messages, max_tokens):
    payload = {
        "model": "dflash", "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"effort": "none"},
    }
    pieces, usage = [], {}
    started = time.perf_counter()
    with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload,
                       stream=True, timeout=(30, 1800)) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1):
            if not raw or not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("reasoning_content") or delta.get("content")
                if piece is not None:
                    pieces.append(piece)
    return {"pieces": pieces, "text": "".join(pieces), "usage": usage,
            "wall_s": time.perf_counter() - started}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--stable", choices=("0", "1"), required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--warm-small", action="store_true")
    ap.add_argument("--logit-trace", action="store_true")
    ap.add_argument("--qsa", choices=("0", "1"), default="1")
    ap.add_argument("--mmb", choices=("0", "1"), default="1")
    ap.add_argument("--upstream", action="store_true")
    ap.add_argument("--gdn-no-tiled", action="store_true")
    ap.add_argument("--harness-suite")
    args = ap.parse_args()
    if subprocess.run(["pgrep", "-x", "dflash_server"], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("another dflash_server is resident")
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(BASE_ENV, QWEN4EXP_DECODE_STABLEGRAPH=args.stable,
               QWEN4EXP_STABLEGRAPH_TELEMETRY="1",
               GGML_CUDA_GRAPH_STATS="1", GGML_CUDA_GRAPH_STATS_EVERY="32")
    if args.qsa == "0":
        env.pop("QWEN4EXP_QSA")
    env["GGML_CUDA_MMB"] = args.mmb
    if args.upstream:
        env["QWEN4EXP_UPSTREAM"] = "1"
    if args.gdn_no_tiled:
        env["DFLASH_GDN_NO_TILED"] = "1"
    if args.logit_trace:
        env["QWEN4EXP_LOGIT_TRACE"] = "1"
    cmd = [args.binary, "--model", str(MODEL), "--host", "127.0.0.1",
           "--port", str(args.port), "--target-device", "hip:0",
           "--max-ctx", "32768", "--chunk", "16384", "--prefix-cache-slots", "0"]
    log = (OUT / f"{args.tag}.server.log").open("w")
    proc = subprocess.Popen(cmd, cwd=REPO, env=dict(os.environ, **env), stdout=log,
                            stderr=log, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                if requests.get(f"http://127.0.0.1:{args.port}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"server failed to become ready: {proc.poll()}")
        live = dict(x.decode().split("=", 1) for x in Path(f"/proc/{proc.pid}/environ").read_bytes().split(b"\0") if b"=" in x)
        got = {k: live.get(k) for k in env}
        if got != env:
            raise RuntimeError(f"live environment mismatch: {got}")
        (OUT / f"{args.tag}.environ.json").write_text(json.dumps(got, indent=2) + "\n")
        messages, max_tokens = prompt_13k()
        warm_rows = []
        if args.warm_small:
            for prompt in (
                "Answer with only the city name: What is the capital of France?",
                "Answer with only the integer: What is 17 multiplied by 19?",
                "Write a Python function named square that returns the square of its argument.",
                "The calibration code is ORCHID-7429. " +
                    "Background information about systems and measurements. " * 300 +
                    "\nWhat is the calibration code? Answer verbatim.",
            ):
                warm_rows.append(request(args.port, [{"role": "user", "content": prompt}], 96))
        rows = []
        for i in range(args.repeats):
            row = request(args.port, messages, max_tokens)
            row["repeat"] = i + 1
            row["has_key"] = "QUINCE-AMBER-7731" in row["text"]
            row["has_54"] = "54" in row["text"]
            rows.append(row)
            print(json.dumps({"repeat": i + 1, "completion_tokens": row["usage"].get("completion_tokens"),
                              "has_key": row["has_key"], "has_54": row["has_54"],
                              "text": row["text"]}), flush=True)
        harness = None
        if args.harness_suite:
            quality_path = OUT / f"{args.tag}.quality.json"
            quality_log = OUT / f"{args.tag}.quality.log"
            quality_cmd = ["python3", "harness/client_test_runner.py", "bench", "--url",
                           f"http://127.0.0.1:{args.port}", "--suite", args.harness_suite,
                           "--model", "dflash", "--json-out", str(quality_path)]
            with quality_log.open("w") as output:
                run = subprocess.run(quality_cmd, cwd=REPO, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=7200)
            harness = {"command": quality_cmd, "exit": run.returncode,
                       "result": json.loads(quality_path.read_text())}
        result = {"tag": args.tag, "binary": args.binary, "command": cmd,
                  "stable": args.stable, "warm_small": warm_rows, "rows": rows,
                  "harness": harness,
                  "all_text_equal": len({r["text"] for r in rows}) == 1,
                  "all_pieces_equal": not rows or all(r["pieces"] == rows[0]["pieces"] for r in rows)}
        (OUT / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
        log.close()


if __name__ == "__main__":
    main()
