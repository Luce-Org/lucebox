#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: rest_summary.py <tag>")

result_path = Path(__file__).resolve().parent / "results" / f"rest_{sys.argv[1]}.json"
with result_path.open(encoding="utf-8") as result_file:
    d = json.load(result_file)
rs = d["results"]
by = {}
for r in rs:
    by.setdefault(r["task"], {})[r["arm"]] = r
print(f"{'task':5} {'calls':>5} {'ctrl_s':>7} {'spec_s':>7} {'ratio':>6} {'hits':>4} {'pred_s':>6} {'twait_c':>7} {'twait_s':>7} {'ok_c':>4} {'ok_s':>4} multi  first")
ratios = []; ct = st = 0; hits = 0; calls = 0; pred = 0; twc = tws = 0; okc = oks = 0; multi = 0; n_ctrl_first = 0
order = {}
for r in rs:
    order.setdefault(r["task"], r["arm"])
for t, a in sorted(by.items()):
    if "control" in a and "spec" in a:
        c, s = a["control"], a["spec"]
        spec_wall_ms = s.get("wall_ms", 0)
        ratio = c["wall_ms"] / spec_wall_ms if spec_wall_ms > 0 else None
        if ratio is not None:
            ratios.append(ratio)
            ct += c["wall_ms"]
            st += spec_wall_ms
        hits += s["hits"]; calls += s["calls"]; pred += s["predictor_ms"]
        twc += c["tool_wait_ms"]; tws += s["tool_wait_ms"]; okc += c["correct"]; oks += s["correct"]; multi += s["multi_call_turns"]
        ratio_text = f"{ratio:6.3f}" if ratio is not None else "   n/a"
        print(f"{t:5} {s['calls']:>5} {c['wall_ms']/1000:7.1f} {spec_wall_ms/1000:7.1f} {ratio_text} {s['hits']:>4} {s['predictor_ms']/1000:6.1f} {c['tool_wait_ms']/1000:7.2f} {s['tool_wait_ms']/1000:7.2f} {str(c['correct']):>4} {str(s['correct']):>4} {s['multi_call_turns']:>5}  {order[t]}")
if ratios and st > 0:
    speedup = (f"median x{statistics.median(ratios):.3f} "
               f"mean x{statistics.mean(ratios):.3f} aggregate x{ct/st:.3f}")
else:
    speedup = "no complete non-zero pairs"
print(f"pairs={len(ratios)} {speedup} | spec hits {hits}/{calls} calls | predictor {pred/1000:.1f}s total | tool wait control {twc/1000:.1f}s spec {tws/1000:.1f}s | correct control {okc} spec {oks} | multi-call turns {multi}")
# hits detail + predictor outcomes
import collections
c = collections.Counter()
for r in rs:
    if r["arm"] != "spec":
        continue
    for t in r["turn_log"]:
        sp = t.get("spec") or {}
        if sp:
            c[(sp.get("status"), sp.get("reason") or "", str(sp.get("detail") or "").split(":")[0])] += 1
        for x in t.get("tools", []):
            if x["source"] == "speculative_hit":
                print("  HIT", r["task"], x["name"], x["args"], "saved_ms", round(x["elapsed_ms"] or 0))
print("spec outcomes:", dict(c))
