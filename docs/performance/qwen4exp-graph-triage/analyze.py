#!/usr/bin/env python3
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw"


def med_span(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


summary = {}
prefill = json.loads((RAW / "e44-prefill-graphs-on.json").read_text())
summary["prefill_graphs_on"] = med_span([
    row["usage"]["timings"]["prefilled_tokens"] /
    (row["usage"]["timings"]["prefill_ms"] / 1000)
    for row in prefill
])

for mode in ("off", "on"):
    rows = json.loads((RAW / f"e45-decode-graphs-{mode}.json").read_text())
    for depth in (2048, 16384):
        selected = [r for r in rows if r["target_depth"] == depth]
        summary[f"decode_graphs_{mode}_{depth}"] = med_span([
            r["usage"]["timings"]["decode_tokens_per_sec"] for r in selected
        ])
        summary[f"decode_ms_graphs_{mode}_{depth}"] = med_span([
            r["usage"]["timings"]["decode_ms"] for r in selected
        ])

for depth in (2048, 16384):
    off = summary[f"decode_graphs_off_{depth}"]["median"]
    on = summary[f"decode_graphs_on_{depth}"]["median"]
    summary[f"decode_delta_pct_{depth}"] = 100 * (on / off - 1)

line_re = re.compile(
    r"\[qwen4exp-logit\] request=(\d+) step=(\d+) pos=(\d+) selected=(\d+) "
    r"best=(\d+) best_logit=([^ ]+) second=(\d+) second_logit=([^ ]+) "
    r"margin=([^ ]+) hash=([0-9a-f]+)"
)


def parse_logits(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = line_re.search(line)
        if match:
            req, step, pos, selected, best, best_v, second, second_v, margin, hashv = match.groups()
            rows.append({
                "request": int(req), "step": int(step), "pos": int(pos),
                "selected": int(selected), "best": int(best), "best_logit": float(best_v),
                "second": int(second), "second_logit": float(second_v),
                "margin": float(margin), "hash": hashv,
            })
    return rows


def compare_requests(rows, a=1, b=2):
    left = [r for r in rows if r["request"] == a]
    right = [r for r in rows if r["request"] == b]
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    return {
        "steps": n,
        "selected_equal": sum(x["selected"] == y["selected"] for x, y in zip(left, right)),
        "top2_equal": sum(
            (x["best"], x["second"]) == (y["best"], y["second"])
            for x, y in zip(left, right)
        ),
        "hash_equal": sum(x["hash"] == y["hash"] for x, y in zip(left, right)),
        "step0": {"a": left[0], "b": right[0]} if n else None,
        "best_logit_abs_diff": med_span([
            abs(x["best_logit"] - y["best_logit"]) for x, y in zip(left, right)
        ]) if n else None,
        "margin_abs_diff": med_span([
            abs(x["margin"] - y["margin"]) for x, y in zip(left, right)
        ]) if n else None,
    }


for tag in (
    "e35-logits-graphs-off", "e35-logits-graphs-on", "e36-qsa-off-graphs-off",
    "e37-mmb-off-graphs-off", "e38-upstream-graphs-off", "e39-gdn-scalar-graphs-off",
):
    summary[tag] = compare_requests(parse_logits(RAW / f"{tag}.server.log"))

# The current upstream dumper unexpectedly omitted I32 callbacks. Compare the
# clean candidate IDs with the last valid upstream capture produced by the same
# checked-in differential harness and model.
old_upstream = ROOT.parent / "qwen4exp-decode-stablegraph" / "reference" / "upstream.log"
current_ours = RAW / "e46-reference-diff-clean" / "ours.log"
up_re = re.compile(r"IDS ffn_moe_topk-(\d+)\s+n=\s*\d+ ([^\n]+)")
our_re = re.compile(r"\[dump\] L(\d+)\.mid[^\n]*ids:([^\n]+)")
up_ids = {}
our_ids = {}
for layer, values in up_re.findall(old_upstream.read_text()):
    up_ids.setdefault(int(layer), values.split())
for layer, values in our_re.findall(current_ours.read_text()):
    our_ids.setdefault(int(layer), values.split())
layers = sorted(up_ids.keys() & our_ids.keys())
summary["reference_ids_historical_capture"] = {
    "upstream_layers": len(up_ids), "candidate_layers": len(our_ids),
    "compared_layers": len(layers),
    "mismatches": sum(up_ids[layer] != our_ids[layer] for layer in layers),
    "upstream_source": str(old_upstream),
    "candidate_source": str(current_ours),
}

(ROOT / "measurement-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
