"""MT-Bench quality comparator.

Reads all results_*.json in the given directory (or current dir),
treats baseline_off as reference, and prints a markdown comparison table.

Usage:
    python eval_quality_compare.py [--dir PATH] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict[tuple[int, int], str]:
    """Returns {(question_id, turn_num): reply} for turn_num in {1, 2}."""
    mapping = {}
    with open(path) as f:
        records = json.load(f)
    for r in records:
        qid = r["question_id"]
        mapping[(qid, 1)] = r["turn_1"]
        mapping[(qid, 2)] = r["turn_2"]
    return mapping


def lcp_ratio(a: str, b: str) -> float:
    """Longest common prefix length / min(len(a), len(b))."""
    denom = min(len(a), len(b))
    if denom == 0:
        return 1.0 if a == b else 0.0
    i = 0
    while i < denom and a[i] == b[i]:
        i += 1
    return i / denom


def compare(ref: dict, cand: dict) -> dict:
    """Compute comparison metrics between ref and cand reply maps."""
    keys = sorted(set(ref) & set(cand))
    if not keys:
        return {"exact_match_rate": 0.0, "mean_lcp_ratio": 0.0,
                "divergence_count": 0, "total_pairs": 0,
                "first_5_divergences": []}

    exact = 0
    lcp_sum = 0.0
    divergences = []

    for k in keys:
        r, c = ref[k], cand[k]
        if r == c:
            exact += 1
        else:
            if len(divergences) < 5:
                qid, turn = k
                divergences.append((qid, turn, r[:50], c[:50]))
        lcp_sum += lcp_ratio(r, c)

    n = len(keys)
    return {
        "exact_match_rate":   exact / n,
        "mean_lcp_ratio":     lcp_sum / n,
        "divergence_count":   n - exact,
        "total_pairs":        n,
        "first_5_divergences": divergences,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MT-Bench quality comparator")
    ap.add_argument("--dir", type=Path, default=Path("."),
                    help="Directory containing results_*.json files")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "eval/summary.md",
                    help="Output markdown summary path")
    args = ap.parse_args()

    result_files = sorted(args.dir.glob("results_*.json"))
    if not result_files:
        print(f"ERROR: no results_*.json found in {args.dir}", file=sys.stderr)
        return 1

    # Map config name -> result file
    configs: dict[str, Path] = {}
    for f in result_files:
        # strip "results_" prefix and ".json" suffix
        name = f.stem[len("results_"):]
        configs[name] = f

    if "baseline_off" not in configs:
        print("ERROR: baseline_off results not found — cannot compare", file=sys.stderr)
        return 1

    ref = load_results(configs["baseline_off"])

    rows = []
    for name, path in configs.items():
        cand = load_results(path)
        m = compare(ref, cand)
        m["config"] = name
        rows.append(m)

    # Sort: baseline_off first, then alphabetical
    def sort_key(r):
        if r["config"] == "baseline_off":
            return (0, r["config"])
        return (1, r["config"])
    rows.sort(key=sort_key)

    # Sanity check: baseline_off_2 vs baseline_off
    sanity_row = next((r for r in rows if r["config"] == "baseline_off_2"), None)
    sanity_warning = ""
    if sanity_row and sanity_row["exact_match_rate"] < 0.99:
        sanity_warning = (
            f"WARNING: baseline_off_2 exact_match_rate={sanity_row['exact_match_rate']:.3f} "
            f"< 0.99 — SERVER IS NONDETERMINISTIC. All other comparisons are suspect.\n\n"
        )

    # Build markdown table
    lines = []
    if sanity_warning:
        lines.append(f"> {sanity_warning.strip()}\n")

    lines.append("| config | exact_match_rate | mean_lcp_ratio | divergence_count | total_pairs |")
    lines.append("|--------|-----------------|----------------|-----------------|-------------|")
    for r in rows:
        lines.append(
            f"| {r['config']} "
            f"| {r['exact_match_rate']:.3f} "
            f"| {r['mean_lcp_ratio']:.3f} "
            f"| {r['divergence_count']} "
            f"| {r['total_pairs']} |"
        )

    lines.append("")
    lines.append("## First 5 divergences per config (vs baseline_off)")
    for r in rows:
        if r["config"] == "baseline_off" or not r["first_5_divergences"]:
            continue
        lines.append(f"\n### {r['config']}")
        lines.append("| qid | turn | ref (first 50) | cand (first 50) |")
        lines.append("|-----|------|----------------|-----------------|")
        for qid, turn, ref50, cand50 in r["first_5_divergences"]:
            ref50_s  = ref50.replace("|", "\\|").replace("\n", " ")
            cand50_s = cand50.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {qid} | {turn} | {ref50_s!r} | {cand50_s!r} |")

    table = "\n".join(lines)

    # Print to stdout
    if sanity_warning:
        print(f"\n{'!'*70}")
        print(sanity_warning.strip())
        print(f"{'!'*70}\n")
    print(table)

    # Write summary file
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    print(f"\nSummary written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
