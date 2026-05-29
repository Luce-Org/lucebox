"""``luce-bench snapshot`` — capture a coherent, baseline-grade run.

This is the consolidation point for what used to be three half-overlapping
tools (``lucebox profile`` / ``luce-bench --sweep`` / ``luce-bench-report``).
A snapshot is a single directory that records:

  * ``host.json`` — CPU, RAM, GPU, driver, CUDA runtime (see ``hostinfo``).
  * ``props.json`` — verbatim ``<url>/props`` response (server identity).
  * ``config.json`` — the strict 11-field runtime tunable allowlist
    extracted from ``props.runtime`` (mirrors the 11-field DflashRuntime
    allowlist persisted by ``lucebox autotune --apply``).
  * one ``<area>.json`` per level area, written by the same code path
    the main CLI uses for ``--areas`` (no duplication).
  * ``_summary.json`` + ``_summary.md`` — area roll-up plus the
    requested ``level`` field (used by ``submit-baseline``).
  * ``command.sh`` — the exact ``luce-bench snapshot ...`` invocation
    that reproduces this dir.
  * ``bench.stdout`` / ``bench.stderr`` — captured stdio.

The default ``--name`` collapses identical configs and branches on
different ones: ``{host}-{gpu_slug}-{label}-{date}-{config_hash[:4]}``.
An explicit ``--name`` is used verbatim.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench.cli import (
    AREAS,
    _run_forge_area_to_dir,
    _run_standard_area_to_dir,
    list_models,
    write_sweep_summary,
)
from lucebench.hostinfo import host_info_to_canonical, probe_host_info
from lucebench.levels import LEVELS, resolve_level

# Strict allowlist for the runtime config block. Mirrors the canonical
# field set in ``lucebox.types.DflashRuntime`` (the 11 fields persisted
# by ``lucebox autotune --apply``). Adding or removing a knob there
# should be reflected here so snapshots stay a faithful record of
# "what the server was running".
CONFIG_FIELDS: tuple[str, ...] = (
    "budget",
    "max_ctx",
    "lazy",
    "prefix_cache_slots",
    "prefill_cache_slots",
    "cache_type_k",
    "cache_type_v",
    "prefill_mode",
    "prefill_keep_ratio",
    "prefill_threshold",
    "prefill_drafter",
)


def _slug(value: str) -> str:
    """Lowercase, de-space, drop non-snapshot-name-safe characters."""
    out = value.lower().strip().replace(" ", "-")
    out = "".join(c for c in out if c.isalnum() or c in ("-", "_"))
    return out.strip("-_") or "unknown"


def _fetch_props(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    """Curl ``<url>/props``. Empty dict on failure — the caller decides what to do."""
    base = url.rstrip("/")
    req = urllib.request.Request(base + "/props", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _derive_config(props: dict[str, Any]) -> dict[str, Any]:
    """Pull the 11 allowlisted fields out of ``props.runtime``.

    Missing fields surface as ``None`` so the config remains
    structurally identical across servers and the hash stays stable
    (a server that doesn't report ``cache_type_k`` always hashes to
    the same null marker).
    """
    runtime = props.get("runtime") if isinstance(props.get("runtime"), dict) else {}
    return {field: runtime.get(field) for field in CONFIG_FIELDS}


def _config_hash(config: dict[str, Any]) -> str:
    """Stable sha256 prefix of a canonicalized config dict."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _derive_name(
    *,
    explicit_name: str | None,
    label: str,
    host_info: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Compute the default snapshot dir name (or echo ``explicit_name`` verbatim).

    Auto format: ``{host}-{gpu_slug}-{label}-{YYYY-MM-DD}-{cfg4}`` where
    ``cfg4`` is the first 4 hex chars of the canonical config hash — so
    re-running the same config on the same day reuses the same dir and
    a tunable change branches into a sibling.

    ``gpu_slug`` encodes every GPU in the rig (not just gpus[0]) so a
    `5090+3090` mixed rig and a `5090+5090` matched rig produce different
    snapshot dir names — material difference for benchmark comparability.
    Reads the legacy ``gpu_name`` scalar OR the new ``gpus[]`` array.
    """
    if explicit_name:
        return explicit_name
    host = _slug(socket.gethostname() or "host")
    gpu_names: list[str] = []
    gpus = host_info.get("gpus")
    if isinstance(gpus, list) and gpus:
        for g in gpus:
            if isinstance(g, dict):
                name = g.get("name")
                if name:
                    gpu_names.append(str(name))
    if not gpu_names:
        legacy = host_info.get("gpu_name")
        if legacy:
            gpu_names.append(str(legacy))
    if not gpu_names:
        gpu_slug = "no-gpu"
    else:
        gpu_slug = "+".join(_slug(n) for n in gpu_names)
    date = _dt.date.today().isoformat()
    cfg4 = _config_hash(config)[:4]
    return f"{host}-{gpu_slug}-{_slug(label)}-{date}-{cfg4}"


def _build_command(argv: list[str]) -> str:
    """Render the original argv as a copy-pasteable ``luce-bench snapshot`` line."""
    import shlex

    return "luce-bench " + " ".join(shlex.quote(a) for a in argv)


class _Tee(io.TextIOBase):
    """Mirror writes to two underlying streams.

    Used to capture stdout/stderr into the snapshot dir while still
    surfacing them on the operator's terminal. We deliberately *don't*
    swallow the original stream (logs would vanish on long runs).
    """

    def __init__(self, primary: Any, secondary: Any) -> None:
        super().__init__()
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        n = self._primary.write(s)
        try:
            self._secondary.write(s)
        except Exception:
            pass
        return n

    def flush(self) -> None:  # type: ignore[override]
        with contextlib.suppress(Exception):
            self._primary.flush()
        with contextlib.suppress(Exception):
            self._secondary.flush()


def _run_areas(
    level_tuples: list[tuple[str, int | None]],
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    think: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    no_fail_fast: bool,
    prompt_thinking_control: str,
    server_honors_api_flags: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Iterate ``(area, n_cap)`` tuples, returning ``(summary_areas, aborted)``."""
    summary_areas: list[dict[str, Any]] = []
    for area, n_cap in level_tuples:
        if area == "forge":
            row = _run_forge_area_to_dir(
                out_root=out_root,
                url=url,
                model=model,
                auth_header=auth_header,
                timeout=timeout,
                max_tokens=max_tokens,
                questions=n_cap,
            )
            if row is not None:
                summary_areas.append(row)
            continue
        if area not in AREAS:
            print(
                f"[lucebench snapshot] unknown area {area!r} in level definition — skipping",
                file=sys.stderr,
                flush=True,
            )
            continue
        row, aborted = _run_standard_area_to_dir(
            area,
            out_root=out_root,
            url=url,
            model=model,
            auth_header=auth_header,
            timeout=timeout,
            max_tokens=max_tokens,
            think=think,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            questions=n_cap,
            no_fail_fast=no_fail_fast,
            prompt_thinking_control=prompt_thinking_control,
            server_honors_api_flags=server_honors_api_flags,
        )
        if aborted:
            return summary_areas, True
        if row is not None:
            summary_areas.append(row)
    return summary_areas, False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="luce-bench snapshot",
        description=(
            "Capture a coherent, baseline-grade snapshot: host info + "
            "server /props + the area set for the requested level. Writes "
            "<out-dir>/<name>/ with host.json, props.json, config.json, "
            "per-area JSON, _summary.json, _summary.md, command.sh, "
            "bench.stdout, bench.stderr."
        ),
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--level",
        required=True,
        choices=sorted(LEVELS.keys()),
        help="Snapshot tier — picks the area set (see lucebench.levels).",
    )
    ap.add_argument(
        "--url",
        "--base-url",
        dest="url",
        required=True,
        help="Server base URL (e.g. http://127.0.0.1:8080).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./snapshots"),
        help="Root directory for snapshots. Default: ./snapshots",
    )
    ap.add_argument(
        "--name",
        default=None,
        help=(
            "Snapshot dir name (used verbatim when set). When omitted, "
            "auto-derived: {host}-{gpu_slug}-{label}-{date}-{cfg4}. "
            "Pass --label to set the {label} segment of the auto name."
        ),
    )
    ap.add_argument(
        "--label",
        default="profile",
        help=(
            "Label segment used by the auto-derived name when --name is "
            "not supplied. Default: 'profile'."
        ),
    )
    ap.add_argument(
        "--host-info",
        type=Path,
        default=None,
        help=(
            "Read host info from this JSON file instead of probing. Useful "
            "when running inside a container that can't see the host's /proc."
        ),
    )
    ap.add_argument(
        "--model",
        default="default",
        help="Model identifier sent in the request body. Default: 'default' (auto-resolve).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request wall timeout (s).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-request decode cap (overrides area default).",
    )
    ap.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=None,
        help="Force think=true across all areas.",
    )
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument(
        "--auth-env",
        default=None,
        help="Env var name to read auth bearer token from.",
    )
    ap.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Keep going past the first connection-refused-style error.",
    )
    ap.add_argument(
        "--prompt-thinking-control",
        choices=["auto", "on", "off"],
        default="auto",
        help="Client-side prompt-level thinking-control fallback (matches the main CLI knob).",
    )
    ap.add_argument(
        "--baselines",
        action="store_true",
        help=(
            "Refuse to run unless --level is level3. Use this as a guard "
            "in pipelines that promote snapshots into the baselines repo."
        ),
    )
    ns = ap.parse_args(argv)

    if ns.baselines and ns.level != "level3":
        print(
            f"[lucebench snapshot] --baselines requires --level level3 (got {ns.level!r})",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        host_info = probe_host_info(ns.host_info)
    except ValueError as exc:
        print(f"[lucebench snapshot] {exc}", file=sys.stderr, flush=True)
        return 2

    props = _fetch_props(ns.url)
    if not props:
        print(
            f"[lucebench snapshot] WARNING: no /props at {ns.url} — config will be empty",
            file=sys.stderr,
            flush=True,
        )
    config = _derive_config(props)

    # Host block resolution — three sources in priority order:
    #   1. /props.host present → use verbatim, tag source="props".
    #      The C++ server has already done the probe (entrypoint.sh
    #      wrote /opt/lucebox-hub/HOST_INFO from LUCEBOX_HOST_*); we
    #      just lift the JSON across the wire.
    #   2. /props present but no `host` block (older lucebox image,
    #      pre-schema-4) → client-side probe, tag "client-fallback".
    #      `host_info` is already the local probe at this point;
    #      relabel its source field.
    #   3. /props missing entirely (non-lucebox server like OpenRouter
    #      / vLLM) → client-side probe, source="client-fallback" with a
    #      note that nothing was negotiated.
    host_block: dict[str, Any]
    props_host = props.get("host") if isinstance(props, dict) else None
    if isinstance(props_host, dict):
        host_block = dict(props_host)
        # Two-layer provenance:
        #   source    = how THIS snapshot got the data (props / fallback / unknown)
        #   collector = how the upstream got it (lucebox.sh / entrypoint.sh / ...)
        # Real entrypoint.sh writes both. Older lucebox builds only
        # wrote `source`. Preserve the upstream's identification by
        # promoting source → collector when collector is missing,
        # THEN overwrite source to "props".
        if not host_block.get("collector") and host_block.get("source"):
            host_block["collector"] = host_block["source"]
        host_block["source"] = "props"
    else:
        host_block = host_info_to_canonical(host_info)
        if props:
            # Server answered /props but had no host block — pre-schema-4
            # lucebox-shaped image. Mark accordingly so consumers can
            # tell this apart from "remote with no /props at all".
            host_block["source"] = "client-fallback"
            host_block.setdefault("collector", "luce-bench")
        else:
            # No /props at all: remote / non-lucebox server. We still
            # write our local probe but tag the source so a comparison
            # can warn about mixing host-of-record sources.
            host_block["source"] = "client-fallback"
            host_block.setdefault("collector", "luce-bench")

    name = _derive_name(
        explicit_name=ns.name,
        label=ns.label,
        host_info=host_info,
        config=config,
    )
    out_root = ns.out_dir / name
    out_root.mkdir(parents=True, exist_ok=True)

    # Write the identity trio first so the dir is recognizable even if
    # the bench gets killed halfway through. ``host.json`` carries the
    # canonical HostInfo-shaped block (props.host when available, local
    # probe otherwise); ``props.json`` is the raw /props snapshot.
    (out_root / "host.json").write_text(json.dumps(host_block, indent=2, default=str) + "\n")
    (out_root / "props.json").write_text(json.dumps(props, indent=2, default=str) + "\n")
    (out_root / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    # Record the reproducer.
    reproducer_argv = ["snapshot", *(argv if argv is not None else sys.argv[2:])]
    (out_root / "command.sh").write_text("#!/bin/sh\n" + _build_command(reproducer_argv) + "\n")
    os.chmod(out_root / "command.sh", 0o755)

    # Resolve auth.
    auth_header = ""
    if ns.auth_env:
        token = os.environ.get(ns.auth_env, "")
        if not token:
            print(
                f"[lucebench snapshot] --auth-env {ns.auth_env}: env var empty or unset",
                file=sys.stderr,
                flush=True,
            )
            return 2
        auth_header = f"Bearer {token}"

    # Auto-resolve model if left at default — the area runners forward
    # whatever string we hand them, so failing to resolve here would
    # send a literal "default" the server will likely 404 on.
    model = ns.model
    if model == "default":
        resolved, _ = list_models(ns.url, auth_header=auth_header)
        if resolved:
            model = resolved
            print(f"[lucebench snapshot] --model default → '{resolved}'", flush=True)

    server_honors = bool(props.get("model_card_source")) if props else False

    # Capture stdio to the snapshot dir for post-mortem while still
    # streaming to the user's terminal. The captures are best-effort —
    # we always restore the originals in `finally` so a flush error
    # doesn't leave the process with a half-installed tee.
    stdout_file = (out_root / "bench.stdout").open("w")
    stderr_file = (out_root / "bench.stderr").open("w")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_out, stdout_file)
    sys.stderr = _Tee(real_err, stderr_file)
    try:
        level_tuples = resolve_level(ns.level)
        print(
            f"[lucebench snapshot] level={ns.level} url={ns.url} model={model} "
            f"out={out_root} areas={[a for a, _ in level_tuples]}",
            flush=True,
        )
        summary_areas, aborted = _run_areas(
            level_tuples,
            out_root=out_root,
            url=ns.url,
            model=model,
            auth_header=auth_header,
            timeout=ns.timeout,
            max_tokens=ns.max_tokens,
            think=ns.think,
            temperature=ns.temperature,
            top_p=ns.top_p,
            top_k=ns.top_k,
            no_fail_fast=ns.no_fail_fast,
            prompt_thinking_control=ns.prompt_thinking_control,
            server_honors_api_flags=server_honors,
        )
        # Inject the host block into every per-area JSON so individual
        # files self-describe after being moved out of the snapshot dir
        # (the report.py compare path can land on a single
        # `<snap>/ds4-eval.json` with no surrounding context). Done
        # before write_sweep_summary so the summary's per-area pointers
        # don't drift from the on-disk files.
        for area_json in sorted(out_root.glob("*.json")):
            if area_json.name.startswith("_") or area_json.name in {
                "host.json",
                "props.json",
                "config.json",
            }:
                continue
            try:
                payload = json.loads(area_json.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            payload["host"] = host_block
            area_json.write_text(
                json.dumps(payload, indent=2, default=str) + "\n"
            )

        write_sweep_summary(
            out_root,
            name=name,
            url=ns.url,
            model=model,
            summary_areas=summary_areas,
            extra={"level": ns.level},
        )
        md_text = (out_root / "_summary.md").read_text()
        print(f"\n[lucebench snapshot] complete → {out_root}", flush=True)
        print(md_text.rstrip(), flush=True)
        if aborted:
            return 3
        return 0
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
        with contextlib.suppress(Exception):
            stdout_file.flush()
            stdout_file.close()
        with contextlib.suppress(Exception):
            stderr_file.flush()
            stderr_file.close()


if __name__ == "__main__":
    sys.exit(main())
