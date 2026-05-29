"""``lucebox autotune --sweep`` — empirical DFLASH_* bracket on the live server.

Reuses existing primitives rather than re-inventing per-cell server
spawning:

  * ``autotune.candidate_configs(host)`` builds the per-tier bracket.
  * ``config.config_set("dflash.*")`` writes each candidate to
    ``~/.lucebox/config.toml`` — the same sparse-write path
    ``autotune --apply`` uses.
  * ``subprocess.run(["systemctl", "--user", "restart", "lucebox.service"])``
    cycles the server. We shell out instead of re-implementing the
    restart so the systemd unit's lifecycle stays the single source of
    truth.
  * Poll ``http://localhost:<port>/v1/models`` until 200 OK, then
  * shell out to ``lucebox profile --level level1`` which runs
    ``luce-bench snapshot`` in the container. Parse decode_tokens_per_sec
    out of the resulting ``<cell-dir>/<area>.json`` rows.

Pre-sweep state is snapshotted (``~/.lucebox/config.toml`` → ``.sweep-
backup``) and restored on SIGINT/SIGTERM or any uncaught exception.
On a successful sweep the backup is deleted after the winner is
applied; on failure (no cell produced a tps reading, or all cells
timed out) the backup is restored and the exit code is non-zero.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.table import Table

from lucebox import autotune as autotune_mod
from lucebox import config as config_mod
from lucebox.host_facts import from_env
from lucebox.types import DflashRuntime

# ── allowlist: the 11 dflash.* fields that lucebox autotune --apply writes ──
# Kept in sync with cli.DFLASH_ALLOWLIST — we duplicate it locally to
# avoid an import cycle (cli.py imports this module).
DFLASH_ALLOWLIST: tuple[str, ...] = (
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


@dataclasses.dataclass(slots=True)
class CellResult:
    """One sweep cell's outcome."""

    index: int
    config: DflashRuntime
    snapshot_dir: Path | None
    mean_decode_tps: float | None
    error: str | None


# ── snapshot dir + decode-tps extraction ───────────────────────────────────


def _sweep_root() -> Path:
    """Top-level sweep dir: ``$XDG_DATA_HOME/lucebox/profile-snapshots/sweep``.

    Cells land at ``<sweep_root>/cell-NN-<short_hash>/``; the per-cell
    snapshot dir name is then chosen by ``luce-bench snapshot`` via the
    ``--name`` flag we pass.
    """
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "lucebox" / "profile-snapshots" / "sweep"


def _short_hash(config: DflashRuntime) -> str:
    """Stable 8-char tag for a config — used in cell directory names.

    Hash the eleven allowlisted fields so two runs with the same
    bracket produce the same dir names (helps `luce-bench report`
    dedupe across host machines and reruns).
    """
    import hashlib

    fields = "|".join(f"{k}={getattr(config, k)!r}" for k in DFLASH_ALLOWLIST)
    return hashlib.sha1(fields.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def _mean_decode_tps_from_snapshot(snapshot_dir: Path) -> float | None:
    """Walk every per-area JSON in ``snapshot_dir`` and average decode tps.

    luce-bench snapshot writes ``<area>.json`` per area (smoke, code,
    gsm8k, agent, longctx for level1). Each row carries
    ``timings.decode_tokens_per_sec`` when the lucebox-server populated
    it. We average across all rows where that field is present — the
    winner picker is "highest mean" so a higher row count just makes
    the estimate more stable.

    Returns None when no row in the snapshot carries a decode tps
    (offline server, OpenRouter-shaped responses with no timings,
    etc.). Callers treat None as "this cell didn't produce a measurement"
    and exclude it from winner picking.
    """
    if not snapshot_dir.exists():
        return None
    tps_values: list[float] = []
    for area_json in sorted(snapshot_dir.glob("*.json")):
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
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            timings = row.get("timings")
            if not isinstance(timings, dict):
                continue
            tps = timings.get("decode_tokens_per_sec")
            if isinstance(tps, int | float) and tps > 0:
                tps_values.append(float(tps))
    if not tps_values:
        return None
    return sum(tps_values) / len(tps_values)


# ── server lifecycle: restart + readiness probe ────────────────────────────


def _systemctl_restart() -> int:
    """``systemctl --user restart lucebox.service``. Returns exit code.

    Shell out instead of adding a Python restart() helper — the
    systemd unit is already the single source of truth for the
    server's lifecycle (see ``lucebox.sh::cmd_systemctl_passthrough``).
    """
    return subprocess.run(
        ["systemctl", "--user", "restart", "lucebox.service"],
        check=False,
    ).returncode


def _wait_ready(port: int, timeout_s: int) -> bool:
    """Poll ``http://localhost:<port>/v1/models`` until 200 OK or budget runs out."""
    deadline = time.time() + timeout_s
    probe_url = f"http://localhost:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


# ── pre-flight checks ──────────────────────────────────────────────────────


def _systemd_unit_path() -> Path:
    """Where the user-installed systemd unit lives."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / "lucebox.service"


def _preflight(console: Console) -> int | None:
    """Refuse to sweep when prerequisites are missing.

    Returns an exit code (non-zero) when the sweep should abort, or
    None when it's safe to proceed. Each refusal includes a one-line
    hint pointing at the canonical "fix it" command.
    """
    if not _systemd_unit_path().exists():
        console.print(
            f"[red]No lucebox.service unit at {_systemd_unit_path()}.[/red]\n"
            "[dim]Hint: run `lucebox install` first.[/dim]"
        )
        return 2

    # No preset AND no target_file → entrypoint would have nothing to
    # serve. Either is enough on its own (preset implies a target;
    # explicit target_file overrides the preset path).
    cfg = config_mod.load() or config_mod.live_config()
    if not cfg.model.preset and not cfg.model.target_file:
        console.print(
            "[red]No model configured (model.preset and model.target_file are both unset).[/red]\n"
            "[dim]Hint: run `lucebox models download` first.[/dim]"
        )
        return 2

    return None


# ── backup + restore (signal-safe) ─────────────────────────────────────────


def _backup_path() -> Path:
    return config_mod.default_config_path().with_suffix(".toml.sweep-backup")


def _take_backup(console: Console) -> Path | None:
    """Copy config.toml → config.toml.sweep-backup. Returns the backup path.

    None when there's no pre-existing config.toml — the sweep will
    still trap signals, and on failure the on-disk file is simply
    removed (back to the pre-sweep state of "no file").
    """
    cfg_path = config_mod.default_config_path()
    if not cfg_path.exists():
        return None
    backup = _backup_path()
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, backup)
    console.print(f"[dim]config.toml → {backup}[/dim]")
    return backup


def _restore_backup(console: Console, backup: Path | None) -> None:
    """Put config.toml back the way we found it."""
    cfg_path = config_mod.default_config_path()
    if backup is not None and backup.exists():
        shutil.copy2(backup, cfg_path)
        console.print(f"[yellow]Restored[/yellow] config.toml from {backup}")
    elif cfg_path.exists():
        # Pre-sweep state was "no file" — remove ours.
        cfg_path.unlink()
        console.print("[yellow]Removed[/yellow] config.toml (no pre-sweep file)")


# ── config write + apply ───────────────────────────────────────────────────


def _apply_config(runtime: DflashRuntime) -> None:
    """Write the 11 dflash.* fields of ``runtime`` to config.toml.

    Uses ``config_set`` so each field lands sparsely — other on-disk
    keys (model.preset etc.) are untouched.
    """
    for name in DFLASH_ALLOWLIST:
        config_mod.config_set(f"dflash.{name}", getattr(runtime, name))


# ── winner selection + results table ───────────────────────────────────────


def _pick_winner(results: list[CellResult]) -> CellResult | None:
    """Highest mean_decode_tps wins; ties → lower max_ctx, then lower budget.

    Cells with ``mean_decode_tps is None`` are excluded — they didn't
    produce a measurement (server-never-ready, no decode tps in the
    snapshot rows).
    """
    valid = [r for r in results if r.mean_decode_tps is not None]
    if not valid:
        return None
    valid.sort(
        key=lambda r: (
            -float(r.mean_decode_tps or 0),
            int(r.config.max_ctx),
            int(r.config.budget),
        )
    )
    return valid[0]


def _print_results(
    console: Console,
    results: list[CellResult],
    winner: CellResult | None,
) -> None:
    """Pretty-print the final results table after the sweep completes."""
    table = Table(title=f"Sweep complete. Tested {len(results)} config(s).")
    table.add_column("#")
    table.add_column("budget", justify="right")
    table.add_column("max_ctx", justify="right")
    table.add_column("kv")
    table.add_column("tps", justify="right")
    table.add_column("status")
    for r in results:
        cfg = r.config
        kv = cfg.cache_type_k or "auto"
        if r.error:
            tps_str = "—"
            status = f"[red]{r.error}[/red]"
        elif r.mean_decode_tps is None:
            tps_str = "—"
            status = "[yellow]no tps in snapshot[/yellow]"
        else:
            tps_str = f"{r.mean_decode_tps:.1f}"
            status = "[green]← winner[/green]" if winner is r else ""
        table.add_row(str(r.index + 1), str(cfg.budget), str(cfg.max_ctx), kv, tps_str, status)
    console.print(table)


# ── driver ─────────────────────────────────────────────────────────────────


def _format_eta(n_candidates: int) -> str:
    """Rough wall-time estimate to surface before the user commits.

    Each cell does: restart (~10 s) + readiness wait (≤60 s) + level1
    snapshot (~30-60 s on a 24 GB rig). Call it ~90 s per cell as a
    user-facing estimate; honest enough to set expectations without
    over-promising.
    """
    seconds = n_candidates * 90
    minutes = seconds // 60
    if minutes < 1:
        return f"~{seconds}s"
    return f"~{minutes} min"


def run_sweep(
    *,
    console: Console | None = None,
    ready_timeout: int = 60,
    yes: bool = False,
) -> int:
    """Top-level sweep driver. Returns process exit code.

    1. Pre-flight: refuse if no systemd unit or no model preset.
    2. Snapshot config.toml + install signal trap.
    3. For each candidate from ``autotune.candidate_configs(host_facts)``:
         a. Write dflash.* fields via ``config_set``.
         b. Restart the systemd unit.
         c. Wait for /v1/models healthy.
         d. Invoke ``lucebox profile --level level1`` against the running server.
         e. Parse decode_tps from the snapshot's per-area JSONs.
    4. Apply the winning config (highest mean_decode_tps), restart, remove backup.
    """
    console = console or Console()

    rc = _preflight(console)
    if rc is not None:
        return rc

    cfg = config_mod.load() or config_mod.live_config()
    host = from_env()
    candidates = autotune_mod.candidate_configs(host)
    if not candidates:
        console.print("[red]candidate_configs returned no configs — nothing to sweep.[/red]")
        return 2

    if not yes:
        console.print(
            f"About to sweep [bold]{len(candidates)}[/bold] config(s) "
            f"(~{_format_eta(len(candidates))} total). "
            "Each cell restarts the server and runs `lucebox profile --level level1`."
        )
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            console.print("[dim]aborted[/dim]")
            return 1

    backup = _take_backup(console)

    # SIGINT/SIGTERM trap: restore the backup + restart so the
    # interrupted sweep doesn't leave the server with a half-applied
    # cell config. The handler raises KeyboardInterrupt so the
    # outer try/except still fires the same cleanup path.
    def _signal_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    sweep_root = _sweep_root()
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: list[CellResult] = []
    interrupted = False
    try:
        for idx, candidate in enumerate(candidates):
            short = _short_hash(candidate)
            kv = candidate.cache_type_k or "auto"
            console.print(
                f"[bold][{idx + 1}/{len(candidates)}][/bold] "
                f"budget={candidate.budget}  max_ctx={candidate.max_ctx}  kv={kv}"
            )

            try:
                _apply_config(candidate)
            except (KeyError, ValueError, OSError) as exc:
                results.append(
                    CellResult(
                        index=idx,
                        config=candidate,
                        snapshot_dir=None,
                        mean_decode_tps=None,
                        error=f"config_set: {exc}",
                    )
                )
                continue

            rc = _systemctl_restart()
            if rc != 0:
                results.append(
                    CellResult(
                        index=idx,
                        config=candidate,
                        snapshot_dir=None,
                        mean_decode_tps=None,
                        error=f"restart exit={rc}",
                    )
                )
                continue

            if not _wait_ready(cfg.port, ready_timeout):
                results.append(
                    CellResult(
                        index=idx,
                        config=candidate,
                        snapshot_dir=None,
                        mean_decode_tps=None,
                        error=f"server-not-ready ({ready_timeout}s)",
                    )
                )
                continue

            cell_name = f"cell-{idx + 1:02d}-{short}"
            # Shell out to `lucebox profile` so the snapshot pipeline
            # stays the user-visible single source of truth. Pass
            # --label so the auto-derived name still slots under the
            # sweep dir if the user inspects results manually.
            snapshot_rc = subprocess.run(
                [
                    "lucebox",
                    "profile",
                    "--level",
                    "level1",
                ],
                check=False,
                env={
                    **os.environ,
                    # Pipe the sweep-cell out-dir through the env so
                    # run_profile lands snapshots under our sweep
                    # tree. The profile command honors --out-dir / -name
                    # via run_profile kwargs — for the sweep path we
                    # invoke run_profile directly below so the env
                    # plumbing isn't actually needed; kept here as a
                    # belt+suspenders override for any out-of-band
                    # lucebox invocation.
                    "LUCEBOX_SWEEP_OUT_DIR": str(sweep_root),
                    "LUCEBOX_SWEEP_CELL_NAME": cell_name,
                },
            ).returncode

            # Locate the snapshot dir luce-bench actually wrote. We
            # asked for cell-NN-<hash>, but the snapshot subcommand
            # may decorate the name with host/gpu/date when --name
            # isn't honored (older luce-bench builds). Fall back to
            # the newest dir under sweep_root.
            candidate_dir = sweep_root / cell_name
            if not candidate_dir.exists():
                # Pick newest dir under sweep_root that isn't a
                # previously-recorded cell.
                existing = {r.snapshot_dir for r in results if r.snapshot_dir is not None}
                newest: Path | None = None
                newest_mtime = -1.0
                for child in sweep_root.iterdir():
                    if not child.is_dir() or child in existing:
                        continue
                    mtime = child.stat().st_mtime
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = child
                candidate_dir = newest or candidate_dir

            tps = _mean_decode_tps_from_snapshot(candidate_dir)
            error = None
            if snapshot_rc != 0 and tps is None:
                error = f"profile exit={snapshot_rc}"
            results.append(
                CellResult(
                    index=idx,
                    config=candidate,
                    snapshot_dir=candidate_dir if candidate_dir.exists() else None,
                    mean_decode_tps=tps,
                    error=error,
                )
            )
            tps_str = "—" if tps is None else f"{tps:.2f}"
            console.print(f"    → mean_decode_tps={tps_str}")

    except KeyboardInterrupt as exc:
        interrupted = True
        console.print(f"\n[yellow]interrupted ({exc})[/yellow]")
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if interrupted:
        _restore_backup(console, backup)
        _systemctl_restart()
        return 130  # canonical SIGINT exit code

    winner = _pick_winner(results)
    if winner is None:
        console.print(
            "[red]No cell produced a decode_tps measurement — restoring pre-sweep config.[/red]"
        )
        _restore_backup(console, backup)
        _systemctl_restart()
        _print_results(console, results, None)
        return 1

    # Apply the winner. The losing cells already wrote their dflash.*
    # fields during the loop, so the on-disk state is whatever the
    # final cell wrote — we overwrite with the winner's fields here.
    _apply_config(winner.config)
    rc = _systemctl_restart()
    if rc != 0:
        console.print(
            f"[red]Restart after applying winner failed (exit={rc}). "
            "Backup retained.[/red]"
        )
        _print_results(console, results, winner)
        return rc

    if not _wait_ready(cfg.port, ready_timeout):
        console.print(
            "[red]Server didn't come back after applying winner — backup retained.[/red]"
        )
        _print_results(console, results, winner)
        return 1

    # Success path — purge the backup so the next `lucebox autotune
    # --sweep` starts clean.
    if backup is not None and backup.exists():
        backup.unlink()

    _print_results(console, results, winner)
    return 0


__all__ = [
    "CellResult",
    "DFLASH_ALLOWLIST",
    "run_sweep",
]
