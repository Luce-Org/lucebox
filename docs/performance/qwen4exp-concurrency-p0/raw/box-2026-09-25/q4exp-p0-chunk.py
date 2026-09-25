#!/usr/bin/env python3
"""Queue a short streamed decode behind a ~30k-token single-sequence prefill."""
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
PROMPT_REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 29900

def stream_chat(messages, max_tokens):
    body = json.dumps({
        "model": "luce", "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    points = []
    status = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=2400) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    item = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = item.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    points.append(time.monotonic())
    except Exception as e:
        return {"start": start, "status": status, "error": repr(e), "points": points,
                "done": time.monotonic()}
    return {"start": start, "status": status, "points": points,
            "done": time.monotonic()}

long_messages = [{"role": "user", "content": " the" * PROMPT_REPEATS +
                  " Summarize the repeated text in one word."}]
short_messages = [{"role": "user", "content":
                   "Write a comma-separated list of every integer from 1 through 100. "
                   "Do not stop until you have written 100."}]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    pre = pool.submit(stream_chat, long_messages, 1)
    time.sleep(2.0)
    decode_submit = time.monotonic()
    dec = pool.submit(stream_chat, short_messages, 64)
    pre_result = pre.result()
    dec_result = dec.result()

def intervals(res):
    return [b-a for a,b in zip(res["points"], res["points"][1:])]

gaps = intervals(dec_result)
ordered = sorted(gaps)
p99 = ordered[min(len(ordered)-1, int(0.99*len(ordered)))] if ordered else None
result = {
    "prompt_repeats": PROMPT_REPEATS,
    "prefill_request": {"status": pre_result["status"], "error": pre_result.get("error"),
        "http_start_to_first_delta_s": (pre_result["points"][0]-pre_result["start"])
            if pre_result["points"] else None,
        "wall_s": pre_result["done"]-pre_result["start"],
        "events": len(pre_result["points"])},
    "decode_request": {"status": dec_result["status"], "error": dec_result.get("error"),
        "enqueue_to_first_delta_s": (dec_result["points"][0]-decode_submit)
            if dec_result["points"] else None,
        "first_delta_to_last_delta_s": (dec_result["points"][-1]-dec_result["points"][0])
            if len(dec_result["points"]) > 1 else None,
        "events": len(dec_result["points"]),
        "gap_median_s": statistics.median(gaps) if gaps else None,
        "gap_p99_s": p99,
        "gap_max_s": max(gaps) if gaps else None,
        "gap_n": len(gaps)},
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(json.dumps(result, indent=2))
