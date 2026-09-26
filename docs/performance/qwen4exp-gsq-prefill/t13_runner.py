#!/usr/bin/env python3
"""Locked qwen4exp follow-up measurements; the caller owns gpu.lock."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests


REPO = Path("/home/duster/lucebox-qwen4exp")
BIN = REPO / "server/build-hip/dflash_server"
OUT = Path("/tmp/q4exp-gsq-prefill")
PROF = Path("/tmp/sdkfull/opt/rocm-7.2.2/bin/rocprofv3")
MODELS = {
    "gsq": Path("/home/duster/models/qwen4exp/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    "iq4": Path("/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf"),
}
DELIVERED = {
    "HIP_VISIBLE_DEVICES": "1",
    "DFLASH_HIP_NO_AUTO_UMA": "1",
    "GGML_CUDA_MMB": "1",
    "QWEN4EXP_QSA": "1",
    "QWEN4EXP_MMB_CUBLAS": "5",
    "DFLASH_MMB_SHADOW": "1",
    "LLAMA_MMB_HC16": "2",
    "QWEN4EXP_LAST_TOKEN_FFN": "1",
    "QWEN4EXP_DENSE_TABLE": "1",
    "QWEN4EXP_HC_TILE16": "1",
}
if "T13_DENSE_TABLE" in os.environ:
    DELIVERED["QWEN4EXP_DENSE_TABLE"] = os.environ["T13_DENSE_TABLE"]


def prompt_exact_16366():
    filler = "Lucebox is a GPU inference engine for local language models. It runs on a single integrated GPU with unified memory. "
    reps = max(1, (8875 - 40) // 13)
    plant = "OPERATIONS NOTE 47-B: the calibration key for the Strix Halo test rig is QUINCE-AMBER-7731, and the rig thermal ceiling is recorded as 54 degrees Celsius. "
    return plant + filler * reps + "\nQuestion: What is the calibration key? Answer verbatim."


def prompt_depth(target, trial):
    filler = "Lucebox is a GPU inference engine for local language models. It runs on a single integrated GPU with unified memory. "
    start = f"Trial {trial:08d}. OPERATIONS NOTE: the calibration key is JUNIPER-COPPER-5826; the thermal ceiling is 57 degrees Celsius. "
    return start + filler * max(1, (target - 110) // 24) + "\nWrite a long numbered list of practical observations about local computing. Continue for at least 200 items."


def request(port, prompt, max_tokens):
    payload = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"effort": "none"},
    }
    t0 = time.perf_counter()
    first = None
    usage = {}
    with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, stream=True, timeout=(30, 1800)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue
            text = line.decode("utf-8", "replace")
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter()
    end = time.perf_counter()
    if not usage:
        raise RuntimeError("stream ended without usage")
    return {"usage": usage, "ttft_s": first - t0 if first else None, "wall_s": end - t0}


def wait_ready(port, process):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"server failed to become ready, exit={process.poll()}")


def server_pid(port):
    for pid in subprocess.run(
        ["pgrep", "-x", "dflash_server"], text=True, capture_output=True
    ).stdout.split():
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if f"--port {port}" in cmdline:
            return int(pid)
    raise RuntimeError(f"dflash_server on port {port} not found")


def save_env(pid, path):
    values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    env = dict(item.decode().split("=", 1) for item in values if b"=" in item)
    got = {key: env.get(key) for key in DELIVERED}
    path.write_text(json.dumps(got, indent=2) + "\n")
    if got != DELIVERED:
        raise RuntimeError(f"live environment mismatch: {got}")


def start(model, port, tag, traced=False, op_prof=False):
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **DELIVERED, QWEN4EXP_FA_TELEMETRY="1")
    if op_prof:
        env["GGML_CUDA_OP_PROF"] = "1"
        env["QWEN4EXP_PROF"] = "1"
    args = [str(BIN), "--model", str(MODELS[model]), "--host", "127.0.0.1", "--port", str(port),
            "--target-device", "hip:0", "--max-ctx", "40000", "--chunk", "16384"]
    if traced:
        trace_dir = OUT / f"{tag}-trace"
        trace_dir.mkdir(exist_ok=True)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.2.2/lib:/tmp/sdkfull/opt/rocm-7.2.2/lib:/tmp/aqlp/opt/rocm-7.2.2/lib"
        args = [str(PROF), "--kernel-trace", "--memory-copy-trace", "-f", "csv", "-d", str(trace_dir), "-o", tag, "--"] + args
    log = (OUT / f"{tag}.server.log").open("w")
    process = subprocess.Popen(args, cwd=REPO, env=env, stdout=log, stderr=log, start_new_session=True)
    wait_ready(port, process)
    pid = server_pid(port)
    save_env(pid, OUT / f"{tag}.environ.json")
    return process, log, pid


def stop(process, log, pid):
    os.kill(pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait()
    log.close()


def run_control(model, kind, repeats, port, tag):
    process, log, pid = start(model, port, tag)
    rows = []
    try:
        if kind == "prefill":
            prompt = prompt_exact_16366()
            request(port, prompt, 1)
            for i in range(repeats):
                row = request(port, prompt, 1)
                row.update(rep=i + 1, model=model, kind=kind)
                rows.append(row)
                print(json.dumps(row), flush=True)
        else:
            for depth in (2048, 16384):
                request(port, prompt_depth(depth, 90000000 + depth), 128)
                for i in range(repeats):
                    row = request(port, prompt_depth(depth, depth * 100 + i), 128)
                    row.update(rep=i + 1, model=model, kind=kind, target_depth=depth)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        stop(process, log, pid)
    (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=2) + "\n")


def run_trace(model, kind, depth, port, tag, op_prof=False):
    process, log, pid = start(model, port, tag, traced=not op_prof, op_prof=op_prof)
    try:
        if kind == "prefill":
            prompt = prompt_exact_16366()
            warm = request(port, prompt, 1)
            time.sleep(3)
            measured = request(port, prompt, 1)
        else:
            warm = request(port, prompt_depth(depth, 80000000 + depth), 128)
            time.sleep(3)
            measured = request(port, prompt_depth(depth, 70000000 + depth), 128)
        result = {"model": model, "kind": kind, "depth": depth, "warmup": warm, "measured": measured}
        (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        stop(process, log, pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("control", "trace", "op-prof"))
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("kind", choices=("prefill", "decode"))
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if subprocess.run(["pgrep", "-x", "dflash_server"], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("another dflash_server is resident")
    if args.mode == "control":
        run_control(args.model, args.kind, args.repeats, args.port, args.tag)
    else:
        run_trace(args.model, args.kind, args.depth, args.port, args.tag, op_prof=args.mode == "op-prof")


if __name__ == "__main__":
    main()
