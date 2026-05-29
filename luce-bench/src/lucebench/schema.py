"""Canonical luce-bench result schema (v1).

Background
----------
luce-bench result.json grew organically and now has two incompatible
in-the-wild shapes:

  * legacy (pre-0.2.5)  → rows have ``id, output, ok, graded_pass,
    strict_pass, thinking_tokens, wall_s``; top-level ``pass_rate`` is a
    FRACTION (0.5761).
  * current (0.2.7+)    → rows have ``case_id, content, pass, graded{},
    reasoning_tokens, wall_seconds``; top-level ``pass_rate`` is a
    PERCENT (77.17).

That divergence (plus a dead ``semantic_pass_rate`` column that's always
0.0 and a ``semantic_hint_rate`` diagnostic that gets misread as the
headline score) makes cross-version comparisons untrustworthy.

This module defines a single canonical shape both loaders and the
``luce-bench regrade`` CLI converge on, with explicit unit choices:

  * all ``*_rate`` fields are FRACTIONS in [0.0, 1.0]; the regrade CLI
    formats them as percents only at the markdown layer.
  * ``semantic_pass_rate`` is intentionally absent (no semantic judge is
    plumbed today). If you wire a judge in the future, emit results
    under ``metrics["semantic_judge"][<judge_id>]``.
  * ``semantic_hint_rate`` is preserved as a diagnostic only — call
    sites and docs MUST NOT treat it as the headline score.

Versioning
----------
``schema_version`` bumps when this file's shape changes. ``grader_version``
is composed from the per-area ``GRADER_VERSION`` constants — bump those
when a grader's extractor / line-match / hint semantics change, and the
re-grade CLI refuses to put runs with different ``grader_version``s in
the same comparison row.

We keep the structure as plain dataclasses (and ``to_dict`` helpers) so
the rest of luce-bench's stdlib-only / no-pydantic style holds.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Bump when the CanonicalResult / CanonicalRow shape changes in a way
# loaders/consumers care about. Loaders should refuse (or warn loudly)
# on a higher schema_version than they know about.
#
# v2 (additive over v1): CanonicalResult gains a `host` field — the
# host-identity facts captured by the C++ server at startup and
# surfaced via /props.host (props_schema 4+), or probed client-side
# by lucebench.hostinfo.probe_host_info when /props.host isn't
# available. Mirrors the JSON shape written to
# /opt/lucebox-hub/HOST_INFO. Pre-v2 result files load with
# host=None — see normalize.py for the fallback rules.
SCHEMA_VERSION = 2


@dataclass
class GpuInfo:
    """One GPU's identity facts — mirrors `/props.host.gpus[*]`.

    Sourced from the host wrapper's nvidia-smi probe (host wrapper exports
    LUCEBOX_HOST_GPU_LIST_CSV; entrypoint.sh parses it into the JSON the
    C++ server surfaces). Multi-GPU rigs surface one entry per device.
    """

    index: int | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None
    name: str | None = None
    sm: str | None = None
    vram_gb: int | None = None
    power_limit_w: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HostInfo:
    """Host-identity facts mirroring the `/props.host` JSON shape.

    Every field is Optional with a None default so a partial probe (or a
    historical result.json that pre-dates the v2 schema) still constructs
    cleanly. ``source`` carries provenance: ``"props"`` when the C++ server
    surfaced /props.host (schema 4+); ``"client-fallback"`` when the
    snapshot subcommand probed locally (server pre-4 or non-lucebox);
    ``"unknown"`` when neither path was available; ``"lucebox.sh"`` when
    the JSON was read straight from /opt/lucebox-hub/HOST_INFO.

    Written into the snapshot dir's ``host.json`` and into every per-area
    ``<area>.json`` so individual area files self-describe even after
    being moved out of the snapshot dir.
    """

    os_pretty: str | None = None
    kernel: str | None = None
    wsl_version: str | None = None
    docker_version: str | None = None
    nvidia_driver: str | None = None
    nvidia_ctk_version: str | None = None
    cpu_model: str | None = None
    nproc: int | None = None
    ram_gb: int | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda_visible_devices: str | None = None
    source: str | None = None
    collector: str | None = None
    collected_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalRow:
    """One per-case row in canonical form.

    Mirrors what ``lucebench.runner.run_case`` writes today, with legacy
    field names remapped:

      * ``id``             → ``case_id``
      * ``output``         → ``content``
      * ``thinking_tokens``→ ``reasoning_tokens``
      * ``wall_s``         → ``wall_seconds``

    The ``graded`` dict is whatever the area's ``grade_case`` returned.
    Preserves ``content`` + ``reasoning_content`` verbatim so a re-grade
    has the raw input it needs.
    """

    case_id: str
    source: str = ""
    kind: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    ttft_seconds: float | None = None
    wall_seconds: float = 0.0
    prefill_ms: float | None = None
    decode_ms: float | None = None
    decode_tokens_per_sec: float | None = None
    content: str = ""
    reasoning_content: str | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None
    graded: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalResult:
    """One canonical result file.

    Top-level rate fields are always fractions in [0.0, 1.0]. The CLI
    formats them as percents for the markdown table at render time.

    ``thinking_control_requested`` is ``"think"`` or ``"nothink"`` — what
    the runner asked for. ``thinking_control_honored`` is whether the
    server actually complied. When the post-run thinking-control verify
    pass hasn't shipped, the normalizer infers from rows (``nothink`` +
    any row with ``reasoning_tokens > 0`` → not honored).
    """

    schema_version: int = SCHEMA_VERSION
    lucebench_version: str = ""
    grader_version: str = ""  # e.g. "ds4-eval=1" or "ds4-eval=1+gsm8k=1"
    area: str = ""
    model: str = ""
    quant: str | None = None
    serving_url: str = ""
    mode: Literal["think", "nothink"] = "nothink"
    seed: int | None = None
    n: int = 0
    started_at: str = ""
    thinking_control_requested: str = "nothink"
    thinking_control_honored: bool = True
    contradicting_rows: int = 0
    strict_pass_rate: float = 0.0
    format_pass_rate: float = 0.0
    semantic_hint_rate: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    rows: list[CanonicalRow] = field(default_factory=list)
    # v2 addition. Optional so historical result.json (pre-host-block)
    # loads cleanly with host=None; the normalizer back-fills a stub
    # `HostInfo(source="unknown")` when the input has no host context.
    host: HostInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict recurses into the rows, which is what we want.
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def from_dict(d: dict[str, Any]) -> CanonicalResult:
    """Build a CanonicalResult from a plain dict (e.g. JSON-loaded).

    Tolerant of missing optional fields — only top-level keys present in
    the canonical dataclass are picked up; everything else is ignored.
    Used by the tests and the regrade CLI when reading already-canonical
    JSON back in.
    """
    row_dicts = d.get("rows") or []
    rows = [_row_from_dict(r) for r in row_dicts]
    kept = {
        f.name: d.get(f.name)
        for f in dataclasses.fields(CanonicalResult)
        if f.name in d and f.name not in {"rows", "host"}
    }
    host_raw = d.get("host")
    host = host_from_dict(host_raw) if isinstance(host_raw, dict) else None
    return CanonicalResult(rows=rows, host=host, **kept)


def host_from_dict(d: dict[str, Any] | None) -> HostInfo | None:
    """Build a HostInfo from a plain dict (e.g. /props.host or
    HOST_INFO file). Tolerant of missing fields and extra keys; the
    canonical field set is fixed by the dataclass.
    """
    if not isinstance(d, dict):
        return None
    gpus_raw = d.get("gpus") or []
    gpus: list[GpuInfo] = []
    if isinstance(gpus_raw, list):
        for entry in gpus_raw:
            if isinstance(entry, dict):
                kept_g = {
                    f.name: entry.get(f.name)
                    for f in dataclasses.fields(GpuInfo)
                    if f.name in entry
                }
                gpus.append(GpuInfo(**kept_g))
    kept = {
        f.name: d.get(f.name)
        for f in dataclasses.fields(HostInfo)
        if f.name in d and f.name != "gpus"
    }
    return HostInfo(gpus=gpus, **kept)


def _row_from_dict(d: dict[str, Any]) -> CanonicalRow:
    kept = {f.name: d.get(f.name) for f in dataclasses.fields(CanonicalRow) if f.name in d}
    # case_id is required by the dataclass; if it's missing fall back to
    # the legacy `id` so a half-normalised dict still loads cleanly.
    kept.setdefault("case_id", d.get("case_id") or d.get("id") or "")
    return CanonicalRow(**kept)


__all__ = [
    "SCHEMA_VERSION",
    "CanonicalResult",
    "CanonicalRow",
    "GpuInfo",
    "HostInfo",
    "from_dict",
    "host_from_dict",
]
