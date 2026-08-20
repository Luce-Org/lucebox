#!/usr/bin/env python3
"""BFCL-style multi-step tasks over real public REST APIs: control vs speculative arm.

Both arms hit the same dflash_server (PR#614 build); the ONLY difference is the
request field `automatic_tool_speculation`. Tool calls are executed client-side
(concurrently when the model emits several), except when the server returns a
speculative hit for the single call the model made, in which case the client
consumes that result instead of calling the API itself.

Usage: rest_bench.py <tag> [--single-call] [--tasks N] [--first control|spec]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/lucebox5/tbspec")
import rest_tools  # noqa: E402

URL = "http://127.0.0.1:18145/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_TURNS = 8
MAX_TOKENS = 512

SYSTEM = (
    "You are a helpful assistant with access to live data tools (geocoding, weather, country facts, "
    "Wikipedia, exchange rates). Use the tools to get real data; do not guess numbers. "
    "When a step depends on a previous result, call the next tool with the exact values returned. "
    "When you have everything, reply with a short final answer containing the facts and numbers."
)
SINGLE_CALL_RULE = " Make exactly one tool call per response."

TASKS = [
    ("t01", "What is the current temperature in Tokyo, Japan?", ["tokyo"]),
    ("t02", "What is the capital of Peru, and what is the weather there right now?", ["lima"]),
    ("t03", "Convert 250 EUR to JPY using today's ECB rate.", ["jpy"]),
    ("t04", "How much is 1000 USD in CHF today, and what is the population of Switzerland?", ["chf", "switzerland"]),
    ("t05", "Give me a one-sentence summary of the Wikipedia article about the capital of Australia.", ["canberra"]),
    ("t06", "What is 100 Norwegian kroner (NOK) worth in EUR right now, and what is the population of Norway?", ["eur", "norway"]),
    ("t07", "Compare the current temperature in Rome and in Paris.", ["rome", "paris"]),
    ("t08", "What are the latest population and the income level of Kenya according to the World Bank?", ["kenya"]),
    ("t09", "Summarize the Wikipedia page for Mount Everest in one sentence.", ["everest"]),
    ("t10", "What is the wind speed right now in Reykjavik, Iceland?", ["reykjav"]),
    ("t11", "Use the country tool to find the capital of Belgium, then tell me the current weather there.", ["brussels"]),
    ("t12", "Is it currently warmer in Cairo or in Madrid?", ["cairo", "madrid"]),
    ("t13", "How many GBP is 500 CAD today, and what is the capital of Canada?", ["gbp", "ottawa"]),
    ("t14", "Give me the latitude and longitude of Nairobi, Kenya and a one-sentence Wikipedia summary of the city.", ["nairobi"]),
    ("t15", "What is the time zone of Brazil's capital, and what is the current temperature there?", ["bras"]),
    ("t16", "What is 75 AUD in USD, and 75 AUD in EUR, at today's rates?", ["usd", "eur"]),
    ("t17", "Summarize the Wikipedia article on the Eiffel Tower in one sentence, and tell me the current weather in Paris.", ["eiffel", "paris"]),
    ("t18", "What region and subregion is Vietnam in, and what is the current temperature in Hanoi?", ["asia", "hanoi"]),
    ("t19", "How many people live in Iceland, and what is the current temperature in Reykjavik?", ["iceland"]),
    ("t20", "What is today's EUR to USD exchange rate, and what is the population of the United States?", ["usd", "united states"]),
]


def post(body: dict, timeout: int = 1800) -> tuple[dict, float]:
    started = time.perf_counter()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data, (time.perf_counter() - started) * 1000.0


def prime(messages: list[dict]) -> None:
    for _ in range(2):
        post({"model": MODEL, "messages": messages, "tools": rest_tools.TOOLS, "tool_choice": "auto", "temperature": 0,
              "max_tokens": 1, "stream": False, "automatic_tool_speculation": False})


def run_task(task_id: str, prompt: str, expect: list[str], arm: str, single_call: bool) -> dict:
    system = SYSTEM + (SINGLE_CALL_RULE if single_call else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    turns = []
    t_start = time.perf_counter()
    final = None
    tool_wait_ms = 0.0; model_ms = 0.0; pred_ms = 0.0; hits = 0; calls = 0; multi_call_turns = 0
    spec_statuses = []
    for turn_index in range(MAX_TURNS):
        body = {"model": MODEL, "messages": messages, "tools": rest_tools.TOOLS, "tool_choice": "auto", "temperature": 0,
                "max_tokens": MAX_TOKENS, "stream": False, "automatic_tool_speculation": arm == "spec"}
        resp, wall = post(body)
        model_ms += wall
        usage = resp.get("usage") or {}
        spec = resp.get("dflash_tool_speculation")
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        turn = {"turn": turn_index, "model_wall_ms": wall, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "timings": usage.get("timings"), "spec": spec, "n_calls": len(tool_calls)}
        if isinstance(spec, dict):
            pred_ms += float(spec.get("predictor_wall_ms") or 0.0)
            spec_statuses.append(f"{spec.get('status')}/{spec.get('reason') or ''}/{str(spec.get('detail') or '').split(':')[0]}")
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            final = msg.get("content") or ""
            turns.append(turn)
            break
        calls += len(tool_calls)
        if len(tool_calls) > 1:
            multi_call_turns += 1
        parsed = []
        for c in tool_calls:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = None
            parsed.append((c.get("id"), fn.get("name"), args))
        results = [None] * len(parsed)
        sources = [None] * len(parsed)
        # speculative hit applies only to a single-call response
        if (len(parsed) == 1 and isinstance(spec, dict) and spec.get("status") == "hit" and isinstance(spec.get("result"), dict)
                and parsed[0][2] is not None and spec["result"].get("call_sha256") == rest_tools.call_sha256(parsed[0][1], parsed[0][2])):
            results[0] = spec["result"]; sources[0] = "speculative_hit"; hits += 1
        # early-dispatch hits (prototype): one entry per closed <function_call> block
        early = resp.get("dflash_early_dispatch") or []
        turn["early"] = [{k: e.get(k) for k in ("name", "status", "reason", "detail", "launched_at_token", "executor_wall_ms", "commit_wait_ms")} for e in early] if early else None
        early_hits = {e["result"]["call_sha256"]: e["result"] for e in early if e.get("status") == "hit" and isinstance(e.get("result"), dict)}
        early_content = {e["result"]["call_sha256"]: e["tool_message_content"] for e in early if e.get("status") == "hit" and isinstance(e.get("result"), dict) and e.get("tool_message_content")}
        for i, (cid, name, args) in enumerate(parsed):
            if results[i] is None and args is not None and name in rest_tools.TOOL_NAMES:
                sha = rest_tools.call_sha256(name, args)
                if sha in early_hits:
                    results[i] = early_hits[sha]; sources[i] = "early_dispatch_hit"; hits += 1
        todo = [i for i, r in enumerate(results) if r is None]
        t0 = time.perf_counter()
        if todo:
            def _exec(i):
                cid, name, args = parsed[i]
                if args is None or name not in rest_tools.TOOL_NAMES:
                    return {"tool_name": name, "ok": False, "value": {"error": "invalid tool call"}, "elapsed_ms": 0.0}
                return rest_tools.run_tool(name, args)
            with ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
                for i, r in zip(todo, pool.map(_exec, todo)):
                    results[i] = r; sources[i] = "client_exec"
        wait = (time.perf_counter() - t0) * 1000.0
        tool_wait_ms += wait
        turn["tools"] = [{"name": parsed[i][1], "args": parsed[i][2], "source": sources[i], "elapsed_ms": results[i].get("elapsed_ms"), "ok": results[i].get("ok")} for i in range(len(parsed))]
        turn["tool_wait_ms"] = wait
        for i, (cid, name, args) in enumerate(parsed):
            sha = rest_tools.call_sha256(name, args) if args is not None else None
            # Echo the server's canonical tool text so prefetch-prefill hits.
            content = early_content.get(sha) or rest_tools.format_result(results[i])
            messages.append({"role": "tool", "tool_call_id": cid, "content": content})
        turns.append(turn)
    wall_ms = (time.perf_counter() - t_start) * 1000.0
    correct = final is not None and all(e.lower() in final.lower() for e in expect)
    return {"task": task_id, "arm": arm, "prompt": prompt, "final": final, "correct": correct, "wall_ms": wall_ms, "model_ms": model_ms,
            "tool_wait_ms": tool_wait_ms, "predictor_ms": pred_ms, "hits": hits, "calls": calls, "turns": len(turns),
            "multi_call_turns": multi_call_turns, "spec_statuses": spec_statuses, "turn_log": turns}


def main() -> None:
    tag = sys.argv[1]
    single_call = "--single-call" in sys.argv
    n = int(sys.argv[sys.argv.index("--tasks") + 1]) if "--tasks" in sys.argv else len(TASKS)
    first = sys.argv[sys.argv.index("--first") + 1] if "--first" in sys.argv else "control"
    out_path = f"/home/lucebox5/tbspec/results/rest_{tag}.json"
    results = []
    for idx, (tid, prompt, expect) in enumerate(TASKS[:n]):
        order = ["control", "spec"] if (idx % 2 == 0) == (first == "control") else ["spec", "control"]
        system = SYSTEM + (SINGLE_CALL_RULE if single_call else "")
        prime([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        for arm in order:
            r = run_task(tid, prompt, expect, arm, single_call)
            results.append(r)
            print(f"{tid} {arm:7} wall={r['wall_ms']/1000:6.1f}s model={r['model_ms']/1000:6.1f}s tool_wait={r['tool_wait_ms']/1000:5.2f}s pred={r['predictor_ms']/1000:4.1f}s "
                  f"calls={r['calls']} hits={r['hits']} turns={r['turns']} multi={r['multi_call_turns']} correct={r['correct']} | {(r['final'] or '')[:90]!r}", flush=True)
            json.dump({"tag": tag, "single_call": single_call, "results": results}, open(out_path, "w"), indent=1)
    # summary
    by = {}
    for r in results:
        by.setdefault(r["task"], {})[r["arm"]] = r
    ratios = []; c_tot = s_tot = 0.0
    for tid, arms in by.items():
        if "control" in arms and "spec" in arms:
            ratios.append(arms["control"]["wall_ms"] / arms["spec"]["wall_ms"]); c_tot += arms["control"]["wall_ms"]; s_tot += arms["spec"]["wall_ms"]
    spec_rs = [r for r in results if r["arm"] == "spec"]; ctrl_rs = [r for r in results if r["arm"] == "control"]
    print("\n=== SUMMARY", tag, "single_call=", single_call)
    print(f"pairs={len(ratios)} paired speedup median x{statistics.median(ratios):.3f} mean x{statistics.mean(ratios):.3f} aggregate x{c_tot/s_tot:.3f}")
    print(f"control: wall sum {c_tot/1000:.1f}s tool_wait sum {sum(r['tool_wait_ms'] for r in ctrl_rs)/1000:.1f}s calls {sum(r['calls'] for r in ctrl_rs)} correct {sum(r['correct'] for r in ctrl_rs)}/{len(ctrl_rs)}")
    print(f"spec:    wall sum {s_tot/1000:.1f}s tool_wait sum {sum(r['tool_wait_ms'] for r in spec_rs)/1000:.1f}s calls {sum(r['calls'] for r in spec_rs)} hits {sum(r['hits'] for r in spec_rs)} predictor sum {sum(r['predictor_ms'] for r in spec_rs)/1000:.1f}s correct {sum(r['correct'] for r in spec_rs)}/{len(spec_rs)} multi-call turns {sum(r['multi_call_turns'] for r in spec_rs)}")


if __name__ == "__main__":
    main()

