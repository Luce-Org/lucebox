#!/usr/bin/env python3
"""Many-call tasks over real public APIs: tool-call orchestration A/B/C.

Arms (same server, same model, same tools):
  react    - one tool call per model turn (sequential; PR#614-style agent loop)
  parallel - model may emit several independent calls per turn; client runs them concurrently
  dag      - LLMCompiler-style: ONE planning turn emits the whole call graph with
             "$k.field" placeholders; the executor resolves dependencies, runs each
             wave concurrently (server early-dispatch already started the
             placeholder-free calls mid-stream), then ONE joiner turn.

Usage: dag_bench.py <tag> [--tasks N] [--arms react,parallel,dag]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/lucebox5/tbspec")
import rest_tools  # noqa: E402

URL = "http://127.0.0.1:18145/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_TURNS = {"react": 30, "parallel": 10, "dag": 8}
MAX_TOKENS = {"react": 512, "parallel": 1024, "dag": 1024}

BASE = (
    "You are a helpful assistant with access to live data tools (geocoding, weather, country facts, "
    "Wikipedia, exchange rates). Use the tools to get real data; do not guess numbers. "
    "When you have everything, reply with a short final answer listing the facts and numbers."
)
SYSTEM = {
    "react": BASE + " Make exactly one tool call per response, then wait for its result.",
    "parallel": BASE + " You may call several tools in ONE response when they do not depend on each other. "
                "When a call needs a value from a previous result, wait for that result first.",
    "dag": BASE + " Plan ALL the tool calls the task needs in ONE response, including calls that depend on earlier calls. "
           "Calls are numbered 1..N in the order you write them. When an argument depends on the output of an earlier call k, "
           "write the placeholder string \"$k.field\" (for example \"$1.latitude\", \"$3.capital_longitude\"). "
           "Output fields you may reference: geocode_city -> latitude, longitude; country_info -> capital, capital_latitude, "
           "capital_longitude, population; exchange_rate -> rate; get_weather -> temperature_c, wind_speed_kmh, relative_humidity_pct; "
           "wikipedia_summary -> extract. Emit every call now; the results will all come back together, then write the final answer.",
}

TASKS = [
    ("d01", "What is the current temperature in each of these cities: Tokyo, Paris, Cairo, Lima, Sydney, Nairobi, Toronto, Oslo?",
     ["tokyo", "paris", "cairo", "lima", "sydney", "nairobi", "toronto", "oslo"]),
    ("d02", "For the countries with ISO codes PE, KE, NO, VN, IS, BR: give the latest population and the current temperature in the capital (use the capital coordinates returned by the country tool).",
     ["peru", "kenya", "norway", "viet", "iceland", "brazil"]),
    ("d03", "Give today's ECB rate from EUR to each of: USD, JPY, GBP, CHF, NOK, CAD, AUD, SEK.",
     ["usd", "jpy", "gbp", "chf", "nok", "cad", "aud", "sek"]),
    ("d04", "Give a one-sentence Wikipedia summary for: Eiffel Tower, Colosseum, Machu Picchu, Mount Fuji; and the current temperature in Paris, Rome, Cusco and Tokyo.",
     ["eiffel", "colosseum", "machu", "fuji", "paris", "rome", "cusco", "tokyo"]),
    ("d05", "What is the current wind speed in: Reykjavik, Wellington, Cape Town, Buenos Aires, Madrid, Bangkok, Anchorage, Honolulu, Mumbai, Lagos?",
     ["reykjav", "wellington", "cape town", "buenos aires", "madrid", "bangkok", "anchorage", "honolulu", "mumbai", "lagos"]),
    ("d06", "For Switzerland (CH), Japan (JP), Canada (CA), Australia (AU) and the United Kingdom (GB): give the latest population, and how much 100 EUR is in the local currency (CHF, JPY, CAD, AUD, GBP) at today's rate.",
     ["switzerland", "japan", "canada", "australia", "united kingdom", "chf", "jpy", "cad", "aud", "gbp"]),
    ("d07", "Is it warmer right now in the capital of Egypt (EG) or in the capital of Argentina (AR)? Use the capital coordinates from the country tool, and also give both populations.",
     ["egypt", "argentina"]),
    ("d08", "Give the latitude, longitude and current relative humidity for: Rome, Athens, Istanbul, Lisbon, Dublin, Vienna.",
     ["rome", "athens", "istanbul", "lisbon", "dublin", "vienna"]),
    ("d09", "Give a one-sentence Wikipedia summary of each: Great Barrier Reef, Sahara, Amazon rainforest, Lake Baikal, Grand Canyon, Mount Kilimanjaro.",
     ["barrier", "sahara", "amazon", "baikal", "grand canyon", "kilimanjaro"]),
    ("d10", "Convert 500 USD into EUR, GBP, JPY and CHF at today's rates, and give the latest population of the United States (US), United Kingdom (GB), Japan (JP) and Switzerland (CH).",
     ["eur", "gbp", "jpy", "chf", "united states", "united kingdom", "japan", "switzerland"]),
]

TASKS_SMALL = [
    ("s01", "What is the current temperature in Tokyo, Paris and Cairo?", ["tokyo", "paris", "cairo"]),
    ("s02", "Give today's ECB rate from EUR to USD, JPY, GBP and CHF.", ["usd", "jpy", "gbp", "chf"]),
    ("s03", "For Peru (PE) and Kenya (KE): latest population and the current temperature in the capital (use the capital coordinates from the country tool).", ["peru", "kenya"]),
    ("s04", "Give a one-sentence Wikipedia summary of the Colosseum and of Machu Picchu, and the current wind speed in Rome and in Cusco.", ["colosseum", "machu", "rome", "cusco"]),
]

PLACEHOLDER = re.compile(r"^\$(\d+)\.([A-Za-z_][A-Za-z0-9_]*)$")
EMBEDDED = re.compile(r"\$(\d+)\.([A-Za-z_][A-Za-z0-9_]*)")


def wait_server(max_wait: float = 1200.0) -> None:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            urllib.request.urlopen("http://127.0.0.1:18145/v1/models", timeout=5).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise RuntimeError("server did not come back")


def post(body: dict, timeout: int = 1800) -> tuple[dict, float]:
    for attempt in range(4):
        started = time.perf_counter()
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data, (time.perf_counter() - started) * 1000.0
        except (ConnectionError, urllib.error.URLError, OSError) as exc:
            print(f"[post] server error ({exc!r}); waiting for server and retrying ({attempt + 1}/4)", flush=True)
            wait_server()
    raise RuntimeError("request failed after retries")


def prime(messages: list[dict]) -> None:
    for _ in range(2):
        post({"model": MODEL, "messages": messages, "tools": rest_tools.TOOLS, "tool_choice": "auto", "temperature": 0,
              "max_tokens": 1, "stream": False, "automatic_tool_speculation": False})


def resolve_args(args: dict, results: dict[int, dict]) -> tuple[dict, list[int], str | None]:
    """Substitute $k.field placeholders. Returns (resolved_args, deps, error)."""
    deps: list[int] = []
    out = {}
    err = None
    for key, value in args.items():
        if isinstance(value, str):
            m = PLACEHOLDER.match(value.strip())
            if m:
                k, field = int(m.group(1)), m.group(2)
                deps.append(k)
                res = results.get(k)
                if res is None:
                    out[key] = value
                    continue
                val = (res.get("value") or {}).get(field)
                if val is None:
                    err = f"missing field {field} in result of call {k}"
                    out[key] = value
                else:
                    out[key] = val
                continue
            if "$" in value and EMBEDDED.search(value):
                def _sub(mm):
                    k, field = int(mm.group(1)), mm.group(2)
                    deps.append(k)
                    res = results.get(k)
                    v = (res.get("value") or {}).get(field) if res else None
                    return str(v) if v is not None else mm.group(0)
                out[key] = EMBEDDED.sub(_sub, value)
                continue
        out[key] = value
    return out, deps, err


def run_task(task_id: str, prompt: str, expect: list[str], arm: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM[arm]}, {"role": "user", "content": prompt}]
    t_start = time.perf_counter()
    turns = []
    final = None
    stats = {"model_ms": 0.0, "tool_wait_ms": 0.0, "calls": 0, "early_hits": 0, "waves": 0, "max_parallel": 0, "dep_calls": 0, "errors": 0}
    nudges = 0
    for turn_index in range(MAX_TURNS[arm]):
        body = {"model": MODEL, "messages": messages, "tools": rest_tools.TOOLS, "tool_choice": "auto", "temperature": 0,
                "max_tokens": MAX_TOKENS[arm], "stream": False, "automatic_tool_speculation": True}
        resp, wall = post(body)
        stats["model_ms"] += wall
        usage = resp.get("usage") or {}
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        early = resp.get("dflash_early_dispatch") or []
        early_hits = {e["result"]["call_sha256"]: e["result"] for e in early if e.get("status") == "hit" and isinstance(e.get("result"), dict)}
        turn = {"turn": turn_index, "model_wall_ms": wall, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "timings": usage.get("timings"), "n_calls": len(tool_calls), "early": [(e.get("name"), e.get("status")) for e in early],
                "finish": (resp.get("choices") or [{}])[0].get("finish_reason")}
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            content = msg.get("content") or ""
            looks_unfinished = (not re.search(r"\d", content) or len(content.strip()) < 20 or
                                re.match(r"^\s*(Now|Next|Let me|I'll|I will|I need|Then|First|Okay|OK)\b", content) is not None)
            if looks_unfinished and nudges < 4:
                nudges += 1
                messages.append({"role": "user", "content": "Do not describe what you will do. Call the tool now (use the function-call format), or if you already have every value from the tools, give the final answer with the numbers."})
                turn["nudged"] = True
                turns.append(turn)
                continue
            final = content
            turns.append(turn)
            break
        stats["calls"] += len(tool_calls)
        # parse calls (1-based index within this response)
        parsed = []
        for i, c in enumerate(tool_calls, start=1):
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("args not object")
            except Exception:  # noqa: BLE001
                args = None
            parsed.append({"idx": i, "id": c.get("id"), "name": fn.get("name"), "args": args, "raw_args": args, "result": None, "source": None})
        t0 = time.perf_counter()
        results_by_idx: dict[int, dict] = {}
        pending = [p for p in parsed]
        wave = 0
        while pending:
            ready = []
            for p in pending:
                if p["args"] is None or p["name"] not in rest_tools.TOOL_NAMES:
                    p["result"] = {"tool_name": p["name"], "ok": False, "value": {"error": "invalid tool call"}, "elapsed_ms": 0.0}
                    p["source"] = "invalid"; results_by_idx[p["idx"]] = p["result"]; stats["errors"] += 1
                    continue
                resolved, deps, err = resolve_args(p["args"], results_by_idx)
                if deps:
                    stats["dep_calls"] += 0  # counted once below
                unresolved = [k for k in deps if k not in results_by_idx]
                if unresolved:
                    continue  # wait for a later wave
                p["resolved_args"] = resolved
                p["deps"] = deps
                if err:
                    p["result"] = {"tool_name": p["name"], "ok": False, "value": {"error": err}, "elapsed_ms": 0.0}
                    p["source"] = "dep_error"; results_by_idx[p["idx"]] = p["result"]; stats["errors"] += 1
                    continue
                ready.append(p)
            pending = [p for p in pending if p["result"] is None and p not in ready]
            if not ready:
                # deadlock: unresolved refs to unknown calls -> mark errors
                for p in pending:
                    p["result"] = {"tool_name": p["name"], "ok": False, "value": {"error": "unresolvable placeholder"}, "elapsed_ms": 0.0}
                    p["source"] = "dep_error"; results_by_idx[p["idx"]] = p["result"]; stats["errors"] += 1
                pending = []
                break
            wave += 1
            stats["max_parallel"] = max(stats["max_parallel"], len(ready))
            # early-dispatch hits (only for calls whose args had no placeholders)
            todo = []
            for p in ready:
                sha = rest_tools.call_sha256(p["name"], p["resolved_args"])
                if p["deps"]:
                    stats["dep_calls"] += 1
                if not p["deps"] and sha in early_hits:
                    p["result"] = early_hits[sha]; p["source"] = "early_dispatch_hit"; stats["early_hits"] += 1
                    results_by_idx[p["idx"]] = p["result"]
                else:
                    todo.append(p)
            if todo:
                with ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
                    outs = list(pool.map(lambda p: rest_tools.run_tool(p["name"], p["resolved_args"]), todo))
                for p, r in zip(todo, outs):
                    p["result"] = r; p["source"] = "client_exec"; results_by_idx[p["idx"]] = r
                    if not r.get("ok"):
                        stats["errors"] += 1
        stats["waves"] += wave
        wait = (time.perf_counter() - t0) * 1000.0
        stats["tool_wait_ms"] += wait
        turn["tool_wait_ms"] = wait
        turn["waves"] = wave
        turn["tools"] = [{"name": p["name"], "args": p.get("resolved_args", p["args"]), "source": p["source"], "ok": (p["result"] or {}).get("ok"),
                          "elapsed_ms": (p["result"] or {}).get("elapsed_ms"), "deps": p.get("deps")} for p in parsed]
        for p in parsed:
            messages.append({"role": "tool", "tool_call_id": p["id"], "content": rest_tools.format_result(p["result"])})
        turns.append(turn)
    wall_ms = (time.perf_counter() - t_start) * 1000.0
    correct = (final is not None and all(e.lower() in final.lower() for e in expect)
               and re.search(r"\d", final) is not None and stats["calls"] > 0)
    return {"task": task_id, "arm": arm, "prompt": prompt, "final": final, "correct": correct, "wall_ms": wall_ms,
            "turns": len(turns), **stats, "turn_log": turns}


def main() -> None:
    tag = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--tasks") + 1]) if "--tasks" in sys.argv else len(TASKS)
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else ["dag", "parallel", "react"]
    out_path = f"/home/lucebox5/tbspec/results/dag_{tag}.json"
    results = []
    tasks = TASKS_SMALL if "--small" in sys.argv else TASKS
    for tid, prompt, expect in tasks[:n]:
        for arm in arms:
            prime([{"role": "system", "content": SYSTEM[arm]}, {"role": "user", "content": prompt}])
            r = run_task(tid, prompt, expect, arm)
            results.append(r)
            print(f"{tid} {arm:8} wall={r['wall_ms']/1000:6.1f}s turns={r['turns']:2} calls={r['calls']:2} waves={r['waves']} maxpar={r['max_parallel']:2} "
                  f"early_hits={r['early_hits']:2} tool_wait={r['tool_wait_ms']/1000:5.2f}s model={r['model_ms']/1000:6.1f}s errors={r['errors']} correct={r['correct']} | {(r['final'] or '')[:80]!r}", flush=True)
            json.dump({"tag": tag, "results": results}, open(out_path, "w"), indent=1)
    by = {}
    for r in results:
        by.setdefault(r["task"], {})[r["arm"]] = r
    print("\n=== SUMMARY", tag)
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        print(f"{arm:8}: wall sum {sum(r['wall_ms'] for r in rs)/1000:7.1f}s  turns {sum(r['turns'] for r in rs):3}  calls {sum(r['calls'] for r in rs):3}  "
              f"tool_wait {sum(r['tool_wait_ms'] for r in rs)/1000:5.1f}s  early_hits {sum(r['early_hits'] for r in rs):3}  correct {sum(r['correct'] for r in rs)}/{len(rs)}")
    for a, b in (("react", "parallel"), ("react", "dag"), ("parallel", "dag")):
        ratios = [by[t][a]["wall_ms"] / by[t][b]["wall_ms"] for t in by if a in by[t] and b in by[t]]
        if ratios:
            print(f"{a} / {b}: median x{statistics.median(ratios):.2f} mean x{statistics.mean(ratios):.2f} aggregate x{sum(by[t][a]['wall_ms'] for t in by if a in by[t] and b in by[t]) / sum(by[t][b]['wall_ms'] for t in by if a in by[t] and b in by[t]):.2f}")


if __name__ == "__main__":
    main()

