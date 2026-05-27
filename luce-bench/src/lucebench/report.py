"""`luce-bench-report` — summarize and compare snapshot directories.

A snapshot dir is what `luce-bench --sweep` writes: per-area JSON
files (ds4-eval.json, code.json, …) plus `_summary.json`. This tool
aggregates one or many such dirs into:

  * a single-snapshot summary table (--summary), or
  * a multi-snapshot comparison matrix (--compare).

Default mode: one positional argument → summary; multiple → compare.

Output: stdout markdown table. Pipe to `tee` / a file as needed. We
deliberately keep this stdlib-only so the report runs anywhere
lucebench installs.

The compare mode also runs against single-area JSON files (not just
full sweep dirs) — useful for one-off ad-hoc comparisons of two ds4
runs without rebuilding into the sweep layout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from lucebench import __version__


def _row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case rows into a per-area stat block."""
    if not rows:
        return {
            "n": 0,
            "pass": 0,
            "rate": 0.0,
            "wall_total": 0,
            "wall_median": 0.0,
            "tok_per_s": 0.0,
            "comp_median": 0,
        }
    passes = sum(1 for r in rows if r.get("pass") or r.get("graded_pass"))
    walls = [r.get("wall_seconds") or r.get("wall_s") or 0 for r in rows]
    comp = [r.get("completion_tokens") or 0 for r in rows]
    decode_tps = [(r.get("timings") or {}).get("decode_tokens_per_sec") or 0 for r in rows]
    decode_tps = [t for t in decode_tps if t > 0]
    return {
        "n": len(rows),
        "pass": passes,
        "rate": 100 * passes / len(rows),
        "wall_total": sum(walls),
        "wall_median": statistics.median(walls) if walls else 0,
        "comp_median": statistics.median(comp) if comp else 0,
        # End-to-end throughput (includes prefill + HTTP). When the
        # server surfaces usage.timings.decode_tokens_per_sec we also
        # carry that as decode_tps_median.
        "tok_per_s": (sum(comp) / sum(walls)) if comp and walls and sum(walls) else 0,
        "decode_tps_median": statistics.median(decode_tps) if decode_tps else 0,
    }


def load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Load a snapshot dir or a single-area JSON file.

    Returns ``{area_name: row_stats_dict}``. For a single JSON file,
    the area name comes from the file's `area` field (or stem).
    """
    out: dict[str, dict[str, Any]] = {}
    if path.is_dir():
        for f in sorted(path.glob("*.json")):
            if f.name.startswith("_"):
                continue  # skip _summary.json
            data = json.loads(f.read_text())
            rows = data.get("rows") or data.get("results") or []
            area = data.get("area") or f.stem
            out[area] = _row_stats(rows)
        return out
    # Single file
    data = json.loads(path.read_text())
    rows = data.get("rows") or data.get("results") or []
    area = data.get("area") or path.stem
    out[area] = _row_stats(rows)
    return out


def fmt_summary_md(name: str, snapshot: dict[str, dict[str, Any]]) -> str:
    lines = [f"# {name}", ""]
    lines += [
        "| area | n | pass | rate | wall_total | wall_median | tok/s | decode_tps (median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for area, s in sorted(snapshot.items()):
        lines.append(
            f"| {area} | {s['n']} | {s['pass']} | {s['rate']:.1f}% | "
            f"{s['wall_total']:.0f}s | {s['wall_median']:.1f}s | "
            f"{s['tok_per_s']:.1f} | "
            f"{s['decode_tps_median']:.1f} |"
        )
    return "\n".join(lines)


def fmt_compare_md(snapshots: list[tuple[str, dict[str, dict[str, Any]]]]) -> str:
    """One row per (snapshot, area), grouped by area for easy scanning."""
    all_areas: set[str] = set()
    for _name, snap in snapshots:
        all_areas.update(snap.keys())
    lines = ["# luce-bench compare", ""]
    lines += [f"- {len(snapshots)} snapshots: " + ", ".join(n for n, _ in snapshots), ""]
    for area in sorted(all_areas):
        lines += [
            "",
            f"## {area}",
            "",
            "| snapshot | n | pass | rate | wall_total | wall_median | "
            "tok/s | decode_tps (median) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, snap in snapshots:
            s = snap.get(area)
            if not s:
                lines.append(f"| {name} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {name} | {s['n']} | {s['pass']} | {s['rate']:.1f}% | "
                f"{s['wall_total']:.0f}s | {s['wall_median']:.1f}s | "
                f"{s['tok_per_s']:.1f} | {s['decode_tps_median']:.1f} |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="luce-bench-report",
        description=(
            "Summarize and compare luce-bench snapshot directories. "
            "Pass one path for a summary table; multiple for a side-by-side "
            "comparison. Accepts snapshot dirs (`./snapshots/<name>/`) or "
            "individual area JSON files (`./snapshots/<name>/ds4-eval.json`)."
        ),
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "paths", nargs="+", type=Path, help="Snapshot directories or per-area JSON files."
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="Write the markdown to this file instead of stdout."
    )
    args = ap.parse_args()

    loaded = []
    for p in args.paths:
        if not p.exists():
            print(f"luce-bench-report: {p} does not exist", file=sys.stderr)
            return 2
        snap = load_snapshot(p)
        loaded.append((p.name or str(p), snap))

    if len(loaded) == 1:
        text = fmt_summary_md(loaded[0][0], loaded[0][1])
    else:
        text = fmt_compare_md(loaded)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"luce-bench-report: wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
