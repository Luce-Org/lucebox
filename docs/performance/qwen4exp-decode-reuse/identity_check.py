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


def run(enabled, tag):
    if enabled:
        os.environ["QWEN4EXP_DECODE_REUSE"] = "1"
    else:
        os.environ.pop("QWEN4EXP_DECODE_REUSE", None)
    process, log, pid = runner.start("iq4", 8895, tag)
    try:
        result = {}
        filler = "Lucebox executes local inference on unified memory. "
        for depth in (2048, 16384):
            prompt = ("OPERATIONS RECORD: the calibration key is QUINCE-AMBER-7731. " +
                      filler * max(1, (depth - 100) // 11) +
                      "\nReturn only the calibration key, exactly as written, with no punctuation.")
            result[str(depth)] = request_tokens(8895, prompt)
        return result
    finally:
        runner.stop(process, log, pid)


control = run(False, "decode-reuse-identity-control")
candidate_enabled = os.getenv("IDENTITY_CANDIDATE_REUSE", "1") != "0"
candidate = run(candidate_enabled, "decode-reuse-identity-candidate")
result = {"control": control, "candidate": candidate, "equal": {}}
for depth in ("2048", "16384"):
    a = control[depth]["pieces"]
    b = candidate[depth]["pieces"]
    result["equal"][depth] = a == b
Path("/tmp/q4exp-decode-reuse").mkdir(exist_ok=True)
Path("/tmp/q4exp-decode-reuse/identity.json").write_text(json.dumps(result, indent=2) + "\n")
for depth in ("2048", "16384"):
    if not result["equal"][depth]:
        a = control[depth]["pieces"]
        b = candidate[depth]["pieces"]
        first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
        raise SystemExit(f"decode mismatch at {depth}: first={first} control={a[first:first+3]!r} candidate={b[first:first+3]!r}")
print(json.dumps({"equal": result["equal"],
                  "piece_counts": {d: len(control[d]["pieces"]) for d in result["equal"]}}))
