"""``luce-bench report`` — summarize and compare snapshot directories.

A snapshot dir is what ``luce-bench --areas all`` (or the ``snapshot``
subcommand) writes: per-area JSON files (ds4-eval.json, code.json, …)
plus ``_summary.json``. This tool aggregates one or many such dirs into:

  * a single-snapshot summary table (--summary), or
  * a multi-snapshot comparison matrix (--compare).

Default mode: one positional argument → summary; multiple → compare.

Output: stdout markdown table. Pipe to `tee` / a file as needed. We
deliberately keep this stdlib-only so the report runs anywhere
lucebench installs.

The compare mode also runs against single-area JSON files (not just
full sweep dirs) — useful for one-off ad-hoc comparisons of two ds4
runs without rebuilding into the sweep layout.

Host-identity (schema v2+) routing
----------------------------------
``load_snapshot`` now routes every per-area JSON through
``normalize.normalize_result`` so the ``host`` block reaches the
comparison table. The compare path surfaces a ``Host`` column with
``host.wsl_version`` / ``host.kernel`` / ``host.gpus[0].name`` (+ "+N"
when N>1) and prints a confounder warning above the table when rows
differ on the WSL version, primary GPU name, or primary GPU power
limit. Timestamps (``host.collected_at`` first, falling back to
``started_at``) ride along the row identity since the same config can
land in multiple snapshots and the user has explicitly asked for
side-by-side cross-host comparison with explicit labelling.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench.normalize import normalize_result
from lucebench.schema import CanonicalResult, HostInfo


def _row_stats(canon: CanonicalResult) -> dict[str, Any]:
    """Aggregate per-case rows on a CanonicalResult into a per-area stat block.

    Reads from CanonicalRow (case_id/content/wall_seconds/graded shape)
    so legacy and current result.json layouts collapse onto the same
    output. Pass-rate is computed straight from rows (graded.pass /
    graded.strict_pass), bypassing the writer's possibly-wrong-unit
    aggregate.
    """
    all_rows = canon.rows
    # client_abort (Tier-2 budgeted) rows are a SEPARATE mode — keep them out
    # of the single-pass pool entirely and surface them under `budgeted`.
    rows = [r for r in all_rows if not _row_is_client_abort(r)]
    budgeted = budgeted_mode_stats(all_rows)
    if not rows:
        return {
            "n": 0,
            "pass": 0,
            "rate": 0.0,
            "wall_total": 0,
            "wall_median": 0.0,
            "tok_per_s": 0.0,
            "comp_median": 0,
            "decode_tps_median": 0,
            "budgeted": budgeted,
        }
    passes = sum(
        1
        for r in rows
        if r.graded.get("pass") or r.graded.get("strict_pass")
    )
    walls = [r.wall_seconds or 0 for r in rows]
    comp = [r.completion_tokens or 0 for r in rows]
    decode_tps = [r.decode_tokens_per_sec or 0 for r in rows]
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
        "budgeted": budgeted,
    }


def _row_is_client_abort(r: Any) -> bool:
    """True iff a CanonicalRow's client_thinking block is in client_abort mode.

    Tier-2 budgeted runs are a SEPARATE benchmark mode — these rows must not
    be pooled with single-pass think/nothink rows in any aggregate.
    """
    ct = getattr(r, "client_thinking", None)
    return ct is not None and getattr(ct, "mode", "off") == "client_abort"


def budgeted_mode_stats(rows: list[Any]) -> dict[str, Any]:
    """Aggregate ONLY the client_abort (Tier-2 budgeted) rows, honestly.

    Comparability rules from docs/client-thinking-budget.md:

      * client_abort rows are a distinct mode — callers select them with
        :func:`_row_is_client_abort` and never pool them with single-pass
        think/nothink rows.
      * Rows with ``continuation == "unsupported"`` (provider rejected the
        prefill / empty answer) or ``answer_started_before_abort == True``
        (a re-prompt could duplicate/corrupt the answer) are EXCLUDED from
        the aggregate accuracy — a provider-capability failure or corrupted
        re-prompt must not be conflated with the model over-thinking.
      * The headline always carries COVERAGE (graded / total) so a route with
        many excluded rows can't look artificially strong on a shrunken sample.

    Returns ``{total, graded, excluded, coverage, pass, rate}`` where ``rate``
    is over the GRADED denominator (a fraction in [0,1]); ``coverage`` is
    ``graded / total``. ``rate`` is 0.0 when nothing graded.
    """
    abort_rows = [r for r in rows if _row_is_client_abort(r)]
    total = len(abort_rows)
    graded_rows = []
    for r in abort_rows:
        ct = r.client_thinking
        if ct.continuation == "unsupported" or ct.answer_started_before_abort:
            continue
        graded_rows.append(r)
    graded = len(graded_rows)
    passes = sum(
        1
        for r in graded_rows
        if r.graded.get("pass") or r.graded.get("strict_pass")
    )
    return {
        "total": total,
        "graded": graded,
        "excluded": total - graded,
        "coverage": (graded / total) if total else 0.0,
        "pass": passes,
        "rate": (passes / graded) if graded else 0.0,
    }


def _host_short(host: HostInfo | None) -> str:
    """Compact one-cell host descriptor for the comparison table.

    Format: ``"<wsl|host-os> · <gpu names joined by +>"``. Every GPU is
    rendered so a ``5090+3090`` mixed rig is visibly different from
    ``5090+5090`` — material when comparing benchmark rows across rigs.
    Empty when nothing is known about the host (pre-v2 result with no
    host block).
    """
    if host is None:
        return "—"
    parts: list[str] = []
    tag = host.wsl_version or (host.kernel.split("-")[0] if host.kernel else None)
    if not tag and host.os_pretty:
        tag = host.os_pretty.split()[0]
    if tag:
        parts.append(tag)
    if host.gpus:
        # Trim "NVIDIA GeForce " / "NVIDIA " prefixes for readability;
        # join every GPU with "+" so multi-GPU rigs are visibly distinct.
        short_names = [
            (g.name or "?").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            for g in host.gpus
        ]
        parts.append("+".join(short_names))
    if not parts:
        # Fall back to the provenance label so an "unknown" row still
        # carries the source it came from.
        return host.source or "—"
    return " · ".join(parts)


def _host_confounders(snapshots: list[tuple[str, dict[str, dict[str, Any]], HostInfo | None]]) -> list[str]:
    """Return a list of confounder-warning lines for the compare header.

    Empty when all rows agree on wsl_version, full GPU lineup (every
    name + power limit, in index order), and CUDA_VISIBLE_DEVICES.
    Comparing the full lineup catches the case where a 5090+3090 rig
    and a 5090+5090 rig would otherwise look identical via gpus[0]
    alone. We deliberately don't refuse to render — the user has
    explicitly said cross-host comparison is allowed with explicit
    labelling.
    """
    wsl_versions: set[str | None] = set()
    # The "lineup" is the ordered tuple of (name, power_limit_w) for
    # every GPU. Sets of tuples compare structurally — `(a, b) == (a, b)`
    # but `(a, b) != (b, a)` (index order matters; CUDA_VISIBLE_DEVICES
    # references indices).
    gpu_lineups: set[tuple[tuple[str | None, int | None], ...]] = set()
    cuda_visible: set[str | None] = set()
    for _name, _snap, host in snapshots:
        if host is None:
            continue
        wsl_versions.add(host.wsl_version)
        if host.gpus:
            lineup = tuple((g.name, g.power_limit_w) for g in host.gpus)
        else:
            lineup = ()
        gpu_lineups.add(lineup)
        cuda_visible.add(host.cuda_visible_devices)
    warnings: list[str] = []
    if len(wsl_versions) > 1:
        warnings.append(
            "⚠ confounder: hosts differ on host.wsl_version "
            f"(values: {sorted(str(v) for v in wsl_versions)})"
        )
    if len(gpu_lineups) > 1:
        # Render each lineup compactly: "5090@175W+3090@250W". This
        # surfaces both the GPU mix AND the power limit difference in
        # a single warning rather than splitting into two.
        def _fmt(lineup: tuple[tuple[str | None, int | None], ...]) -> str:
            if not lineup:
                return "—"
            return "+".join(
                f"{(name or '?')}@{plw}W" if plw is not None else (name or "?")
                for name, plw in lineup
            )
        warnings.append(
            "⚠ confounder: hosts differ on host.gpus lineup "
            f"(values: {sorted(_fmt(lineup) for lineup in gpu_lineups)})"
        )
    if len(cuda_visible) > 1:
        warnings.append(
            "⚠ confounder: hosts differ on host.cuda_visible_devices "
            f"(values: {sorted(repr(v) for v in cuda_visible)})"
        )
    return warnings


def _row_identity(name: str, host: HostInfo | None) -> str:
    """Render the per-row identity used in the compare header.

    Includes the snapshot name and its collected_at timestamp. The user
    explicitly called out that multiple copies of "the same" snapshot
    can land in the baselines repo — so the timestamp is part of the
    key. Empty timestamp falls back to just the name.
    """
    if host is None or not host.collected_at:
        return name
    return f"{name}@{host.collected_at}"


def load_snapshot(path: Path) -> tuple[dict[str, dict[str, Any]], HostInfo | None]:
    """Load a snapshot dir (or single-area JSON) into per-area stats + host.

    Returns ``({area_name: row_stats}, host)``. The host block is
    pulled from the first per-area file (they all carry the same block
    because the snapshot writes it into each), or from ``host.json`` as
    a fallback. For a single-file input we read the host directly off
    that file's normalized result.
    """
    out: dict[str, dict[str, Any]] = {}
    host: HostInfo | None = None
    if path.is_dir():
        for f in sorted(path.glob("*.json")):
            if f.name.startswith("_"):
                continue  # skip _summary.json
            if f.name in {"host.json", "props.json", "config.json"}:
                continue
            try:
                data = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            canon = normalize_result(data, source_path=f)
            area = canon.area or f.stem
            out[area] = _row_stats(canon)
            if host is None and canon.host is not None:
                # Only take a non-"unknown" host as authoritative — an
                # explicit unknown means the JSON had no host context, so
                # keep looking through siblings.
                if canon.host.source and canon.host.source != "unknown":
                    host = canon.host
                elif host is None:
                    host = canon.host
        # Fall back to host.json if nothing in the area files spoke up.
        if host is None or host.source == "unknown":
            host_json_path = path / "host.json"
            if host_json_path.is_file():
                try:
                    raw = json.loads(host_json_path.read_text())
                except (OSError, json.JSONDecodeError):
                    raw = None
                if isinstance(raw, dict):
                    from lucebench.schema import host_from_dict

                    candidate = host_from_dict(raw)
                    if candidate is not None and candidate.source != "unknown":
                        host = candidate
        return out, host
    # Single file path.
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return out, None
    if not isinstance(data, dict):
        return out, None
    canon = normalize_result(data, source_path=path)
    area = canon.area or path.stem
    out[area] = _row_stats(canon)
    host = canon.host
    return out, host


def fmt_summary_md(name: str, snapshot: dict[str, dict[str, Any]], host: HostInfo | None = None) -> str:
    lines = [f"# {name}"]
    if host is not None:
        lines.append(f"_host:_ {_host_short(host)} (source={host.source or 'unknown'})")
    lines.append("")
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


def fmt_compare_md(
    snapshots: list[tuple[str, dict[str, dict[str, Any]], HostInfo | None]],
) -> str:
    """One row per (snapshot, area), grouped by area for easy scanning.

    Adds a ``Host`` column with the compact host descriptor. Confounder
    warnings (host.wsl_version / primary-GPU name / primary-GPU power
    limit drift) print above the table — informational, never an exit.
    """
    all_areas: set[str] = set()
    for _name, snap, _h in snapshots:
        all_areas.update(snap.keys())
    lines = ["# luce-bench compare", ""]
    lines += [
        f"- {len(snapshots)} snapshots: "
        + ", ".join(_row_identity(n, h) for n, _s, h in snapshots),
        "",
    ]
    warns = _host_confounders(snapshots)
    if warns:
        lines += warns
        lines.append("")
    for area in sorted(all_areas):
        lines += [
            "",
            f"## {area}",
            "",
            "| snapshot | host | n | pass | rate | wall_total | wall_median | "
            "tok/s | decode_tps (median) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, snap, host in snapshots:
            host_label = _host_short(host)
            row_id = _row_identity(name, host)
            s = snap.get(area)
            if not s:
                lines.append(
                    f"| {row_id} | {host_label} | — | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| {row_id} | {host_label} | {s['n']} | {s['pass']} | {s['rate']:.1f}% | "
                f"{s['wall_total']:.0f}s | {s['wall_median']:.1f}s | "
                f"{s['tok_per_s']:.1f} | {s['decode_tps_median']:.1f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="luce-bench report",
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
    args = ap.parse_args(argv)

    loaded: list[tuple[str, dict[str, dict[str, Any]], HostInfo | None]] = []
    for p in args.paths:
        if not p.exists():
            print(f"luce-bench report: {p} does not exist", file=sys.stderr)
            return 2
        snap, host = load_snapshot(p)
        loaded.append((p.name or str(p), snap, host))

    if len(loaded) == 1:
        name, snap, host = loaded[0]
        text = fmt_summary_md(name, snap, host)
    else:
        text = fmt_compare_md(loaded)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"luce-bench report: wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
