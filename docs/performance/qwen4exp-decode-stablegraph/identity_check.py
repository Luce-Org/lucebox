#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path

import requests

spec = importlib.util.spec_from_file_location("runner", "/tmp/q4exp-followup-runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def request_tokens(port, prompt):
    payload = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"effort": "none"},
    }
    pieces = []
    usage = {}
    with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                       json=payload, stream=True, timeout=(30, 1800)) as response:
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
                if "content" in delta or "reasoning_content" in delta:
                    pieces.append(delta.get("content") or delta.get("reasoning_content") or "")
    return {"pieces": pieces, "usage": usage}


def run(enabled, tag, port):
    if enabled:
        os.environ["QWEN4EXP_DECODE_STABLEGRAPH"] = "1"
        os.environ["QWEN4EXP_STABLEGRAPH_TELEMETRY"] = "1"
        os.environ["GGML_CUDA_GRAPH_STATS"] = "1"
        os.environ["GGML_CUDA_GRAPH_STATS_EVERY"] = "32"
    else:
        for key in ("QWEN4EXP_DECODE_STABLEGRAPH", "QWEN4EXP_STABLEGRAPH_TELEMETRY",
                    "GGML_CUDA_GRAPH_STATS", "GGML_CUDA_GRAPH_STATS_EVERY"):
            os.environ.pop(key, None)
    process, log, pid = runner.start("iq4", port, tag)
    try:
        result = {}
        filler = "Lucebox executes local inference on unified memory. "
        for depth in (2048, 16384):
            prompt = ("OPERATIONS RECORD: the calibration key is QUINCE-AMBER-7731. " +
                      filler * max(1, (depth - 100) // 11) +
                      "\nReturn exactly this text and nothing else: "
                      "QUINCE-AMBER-7731 | QUINCE-AMBER-7731")
            result[str(depth)] = request_tokens(port, prompt)
        return result
    finally:
        runner.stop(process, log, pid)


control = run(False, "stablegraph-identity-control", 18732)
candidate = run(True, "stablegraph-identity-candidate", 18733)
result = {"control": control, "candidate": candidate, "equal": {}}
for depth in ("2048", "16384"):
    a, b = control[depth], candidate[depth]
    result["equal"][depth] = (
        a["pieces"] == b["pieces"] and
        a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"]
    )
Path("/tmp/q4exp-decode-stablegraph").mkdir(exist_ok=True)
Path("/tmp/q4exp-decode-stablegraph/identity.json").write_text(
    json.dumps(result, indent=2) + "\n")
print(json.dumps({"equal": result["equal"],
                  "tokens": {d: control[d]["usage"]["completion_tokens"]
                             for d in result["equal"]}}))
raise SystemExit(0 if all(result["equal"].values()) else 1)
