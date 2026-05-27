"""Typer app — the user-facing subcommands.

Layout follows the host wrapper's dispatch table. Anything `lucebox.sh`
doesn't intercept (everything outside the systemd surface) ends up here.

Subcommand inventory:
    check                  — readiness report
    configure              — write .lucebox/config.toml from heuristic autotune
    pull                   — docker pull the cuda12 image
    print-run              — emit the docker-run command for the server
    print-serve-argv       — same, raw argv lines (consumed by lucebox.sh serve)
    benchmark              — sweep DFLASH_* knobs and validation suites
    profile                — update local profile evidence and export snapshots
    smoke                  — hit /props + /v1/chat/completions on a running server
    download-models        — fetch target + draft via the container
    claude                 — launch Claude Code pointed at the running server
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

import lucebox.autotune as autotune_mod
import lucebox.config as config_mod
import lucebox.docker_run as docker_run
import lucebox.download as download_mod
import lucebox.host_check as host_check
import lucebox.profile as profile_mod
import lucebox.smoke as smoke_mod
from lucebox import __version__
from lucebox.host_facts import from_env
from lucebox.types import AutotuneMeta, BenchmarkMeta, Config, Variant

app = typer.Typer(
    name="lucebox",
    help="Host CLI for the lucebox-hub container. Invoked by lucebox.sh.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# ── helpers ────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_variant_from_driver(driver_major: int, gpu_sm: str) -> Variant:
    # Mirrors lucebox.sh::pick_variant. Centralized so Python and bash agree.
    return os.environ.get("LUCEBOX_VARIANT", "cuda12")


def _build_default_config() -> Config:
    """Build a fresh Config from current host facts + heuristic autotune."""
    host = from_env()
    variant = _pick_variant_from_driver(host.driver_major, host.gpu_sm)
    dflash = autotune_mod.runtime_from_host(host)
    default = Config()
    return Config(
        variant=variant,
        image=os.environ.get("LUCEBOX_IMAGE", default.image),
        container_name=os.environ.get("LUCEBOX_CONTAINER", default.container_name),
        port=int(os.environ.get("LUCEBOX_PORT", str(default.port))),
        models_dir=Path(os.environ.get("LUCEBOX_MODELS", str(default.models_dir))),
        dflash=dflash,
        host=host,
        autotune=AutotuneMeta(source="heuristic", timestamp=_now()),
    )


def _load_or_build() -> Config:
    cfg = config_mod.load()
    if cfg is not None:
        return cfg
    return _build_default_config()


# ── subcommands ────────────────────────────────────────────────────────────

@app.command()
def check() -> None:
    """Print a readiness report (driver, docker, CTK, RAM, VRAM, systemd)."""
    host = from_env()
    results = host_check.run_checks(host)
    worst = host_check.render(console, host, results)
    if worst == "fail":
        raise typer.Exit(code=1)


@app.command()
def configure(
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Replace any existing config.toml")
    ] = False,
) -> None:
    """Pick image tag + autotune defaults, write .lucebox/config.toml."""
    path = config_mod.default_config_path()
    if path.exists() and not overwrite:
        console.print(f"[yellow]{path} already exists — pass --overwrite to replace[/yellow]")
        raise typer.Exit(code=1)

    cfg = _build_default_config()
    written = config_mod.save(cfg, path)
    console.print(f"[green]Wrote[/green] {written}")
    console.print(f"  variant     [bold]{cfg.variant}[/bold]")
    console.print(f"  image       {cfg.image}:{cfg.variant}")
    console.print(f"  models      {cfg.models_dir}")
    console.print(f"  budget      {cfg.dflash.budget}")
    console.print(f"  max_ctx     {cfg.dflash.max_ctx}")
    console.print(f"  lazy_draft  {cfg.dflash.lazy}")
    console.print()
    console.print("Next:")
    console.print("  [dim]lucebox.sh pull[/dim]            # fetch the image")
    console.print("  [dim]lucebox.sh start[/dim]           # start via systemd")
    console.print("  [dim]lucebox.sh serve[/dim]           # foreground, no systemd")


@app.command()
def pull() -> None:
    """`docker pull` the image variant from config.toml."""
    cfg = _load_or_build()
    tag = f"{cfg.image}:{cfg.variant}"
    console.print(f"[bold]Pulling {tag}[/bold] (~14 GB; takes a while)…")
    rc = docker_run.docker_pull(tag)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("print-run")
def print_run() -> None:
    """Print the docker-run command for the server (copy-pasteable)."""
    cfg = _load_or_build()
    spec = docker_run.server_run_spec(cfg)
    print(spec.printable())


@app.command("print-serve-argv")
def print_serve_argv() -> None:
    """Emit the server docker-run argv, one token per line.

    Consumed by lucebox.sh's `serve` subcommand and the systemd unit. Kept as
    a separate command from `print-run` so the bash side has a guaranteed
    machine-readable contract that's independent of the pretty formatter.
    """
    cfg = _load_or_build()
    spec = docker_run.server_run_spec(cfg)
    for tok in spec.argv():
        print(tok)


@app.command()
def benchmark(
    profile: Annotated[
        str,
        typer.Option(
            help="Optimizer profile: level1, level2, or level3.",
        ),
    ] = "level1",
    budgets: Annotated[
        str, typer.Option(help="Comma-separated DFLASH_BUDGET values to sweep.")
    ] = "8,16,22,32",
    ctx_values: Annotated[
        str,
        typer.Option(
            "--ctx-values",
            help="Comma-separated DFLASH_MAX_CTX values to sweep.",
        ),
    ] = "",
    min_context_speed_ratio: Annotated[
        float,
        typer.Option(
            help="For level2/level3, keep candidates at least this fraction of fastest speed.",
        ),
    ] = 0.85,
    lazy_values: Annotated[
        str, typer.Option(help="Comma-separated lazy-draft values to sweep: 0,1.")
    ] = "",
    prefix_cache_slots_values: Annotated[
        str, typer.Option(help="Comma-separated prefix-cache slot counts to sweep.")
    ] = "",
    prefill_cache_slots_values: Annotated[
        str, typer.Option(help="Comma-separated prefill-cache slot counts to sweep.")
    ] = "",
    kv_values: Annotated[
        str,
        typer.Option(
            help="Comma-separated KV modes: auto,f16,q4_0,q4_1,q5_0,q5_1,q8_0,tq3_0."
        ),
    ] = "",
    prefill_modes: Annotated[
        str, typer.Option(help="Comma-separated PFlash modes to sweep: off,auto,always.")
    ] = "",
    prefill_keep_ratios: Annotated[
        str, typer.Option(help="Comma-separated PFlash keep ratios to sweep.")
    ] = "",
    prefill_thresholds: Annotated[
        str, typer.Option(help="Comma-separated PFlash auto thresholds to sweep.")
    ] = "",
    prefill_drafter: Annotated[
        str, typer.Option(help="PFlash drafter GGUF path for non-off prefill modes.")
    ] = "",
    n_prompts: Annotated[
        int, typer.Option("--n-prompts", help="HE prompts per budget cell.")
    ] = 5,
    n_gen: Annotated[
        int, typer.Option("--n-gen", help="Generated tokens per prompt.")
    ] = 256,
    ready_timeout: Annotated[
        int, typer.Option(help="Seconds to wait for each server cell to become ready.")
    ] = 180,
    cell_timeout: Annotated[
        int, typer.Option(help="Hard wall-clock budget per benchmark cell.")
    ] = 240,
    extra_suites: Annotated[
        str,
        typer.Option(
            "--extra-suites",
            help=(
                "Comma-separated post-optimizer suites: http-frontiers,capability,"
                "ds4-eval,capability-long,agentic-tools,agentic-session."
            ),
        ),
    ] = "",
    frontiers: Annotated[
        str, typer.Option(help="Prompt frontiers for the http-frontiers extra suite.")
    ] = "2048,4096,8192,16384",
    frontiers_repeat: Annotated[
        int,
        typer.Option(
            "--frontiers-repeat",
            help="Samples per frontier in the http-frontiers extra suite.",
        ),
    ] = 1,
    agentic_repeat: Annotated[
        int,
        typer.Option("--agentic-repeat", help="Repeats per prompt for agentic-tools."),
    ] = 1,
    agentic_session_turns: Annotated[
        int,
        typer.Option(
            "--agentic-session-turns",
            help="Turns per session for the agentic-session extra suite.",
        ),
    ] = 6,
    agentic_session_sessions: Annotated[
        int,
        typer.Option(
            "--agentic-session-sessions",
            help="Independent sessions for the agentic-session extra suite.",
        ),
    ] = 1,
    allow_extra_suite_failures: Annotated[
        bool,
        typer.Option(
            "--allow-extra-suite-failures",
            help="Persist the winner even if post-winner validation suites fail.",
        ),
    ] = False,
) -> None:
    """Sweep DFLASH_* knobs inside the container, merge winner into config."""
    cfg = _load_or_build()
    bench_args = [
        "--profile", profile,
        "--budgets", budgets,
        "--ctx-values", ctx_values,
        "--min-context-speed-ratio", str(min_context_speed_ratio),
        "--lazy-values", lazy_values,
        "--prefix-cache-slots-values", prefix_cache_slots_values,
        "--prefill-cache-slots-values", prefill_cache_slots_values,
        "--kv-values", kv_values,
        "--prefill-modes", prefill_modes,
        "--prefill-keep-ratios", prefill_keep_ratios,
        "--prefill-thresholds", prefill_thresholds,
        "--prefill-drafter", prefill_drafter,
        "--n-prompts", str(n_prompts),
        "--n-gen", str(n_gen),
        "--ready-timeout", str(ready_timeout),
        "--cell-timeout", str(cell_timeout),
        "--extra-suites", extra_suites,
        "--frontiers", frontiers,
        "--frontiers-repeat", str(frontiers_repeat),
        "--agentic-repeat", str(agentic_repeat),
        "--agentic-session-turns", str(agentic_session_turns),
        "--agentic-session-sessions", str(agentic_session_sessions),
    ]
    if allow_extra_suite_failures:
        bench_args.append("--allow-extra-suite-failures")
    spec = docker_run.benchmark_run_spec(cfg, tuple(bench_args))
    console.print(f"[bold]Running optimizer in {spec.image}[/bold]")
    rc = docker_run.run(spec.argv(), check=False).returncode
    if rc != 0:
        raise typer.Exit(code=rc)

    report_path = Path(cfg.models_dir) / ".lucebox" / "bench-report.json"
    try:
        report = json.loads(report_path.read_text())
    except OSError as e:
        console.print(f"[red]Benchmark finished but report is missing:[/red] {e}")
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        console.print(f"[red]Benchmark report is invalid JSON:[/red] {e}")
        raise typer.Exit(code=1) from e

    winner = report.get("winner")
    if not isinstance(winner, dict) or "budget" not in winner or "max_ctx" not in winner:
        console.print("[red]Benchmark did not produce a winning cell[/red]")
        raise typer.Exit(code=1)

    try:
        budget = int(winner["budget"])
        max_ctx = int(winner["max_ctx"])
        lazy = bool(winner.get("lazy", cfg.dflash.lazy))
        prefix_cache_slots = int(winner.get("prefix_cache_slots", cfg.dflash.prefix_cache_slots))
        prefill_cache_slots = int(winner.get("prefill_cache_slots", cfg.dflash.prefill_cache_slots))
        cache_type_k = str(winner.get("cache_type_k", ""))
        cache_type_v = str(winner.get("cache_type_v", ""))
        prefill_mode = str(winner.get("prefill_mode", cfg.dflash.prefill_mode))
        prefill_keep_ratio = float(winner.get("prefill_keep_ratio", cfg.dflash.prefill_keep_ratio))
        prefill_threshold = int(winner.get("prefill_threshold", cfg.dflash.prefill_threshold))
        prefill_drafter_winner = str(winner.get("prefill_drafter", cfg.dflash.prefill_drafter))
    except (TypeError, ValueError) as e:
        console.print(f"[red]Benchmark winner has invalid fields:[/red] {winner!r}")
        raise typer.Exit(code=1) from e

    mean_tps = winner.get("mean_decode_tps")
    mean_tps_f = float(mean_tps) if mean_tps is not None else None
    tuned = replace(
        cfg,
        dflash=autotune_mod.merge_benchmark_winner(
            cfg.dflash,
            budget=budget,
            max_ctx=max_ctx,
            lazy=lazy,
            prefix_cache_slots=prefix_cache_slots,
            prefill_cache_slots=prefill_cache_slots,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            prefill_mode=prefill_mode,
            prefill_keep_ratio=prefill_keep_ratio,
            prefill_threshold=prefill_threshold,
            prefill_drafter=prefill_drafter_winner,
        ),
        autotune=AutotuneMeta(source="benchmark", timestamp=_now()),
        benchmark=BenchmarkMeta(
            ran_at=_now(),
            profile=str(report.get("profile") or profile),
            winner_budget=budget,
            winner_max_ctx=max_ctx,
            winner_lazy=lazy,
            winner_prefix_cache_slots=prefix_cache_slots,
            winner_prefill_cache_slots=prefill_cache_slots,
            winner_cache_type_k=cache_type_k,
            winner_cache_type_v=cache_type_v,
            winner_prefill_mode=prefill_mode,
            mean_tps=mean_tps_f,
            report_path=str(report_path),
        ),
    )
    written = config_mod.save(tuned)
    console.print(f"[green]Updated[/green] {written}")
    console.print(f"  DFLASH_BUDGET [bold]{budget}[/bold]")
    console.print(f"  DFLASH_MAX_CTX [bold]{max_ctx}[/bold]")
    console.print(f"  DFLASH_LAZY [bold]{lazy}[/bold]")
    console.print(f"  prefix slots [bold]{prefix_cache_slots}[/bold]")
    console.print(f"  KV cache     [bold]{cache_type_k or 'auto'}[/bold]")
    console.print(f"  PFlash       [bold]{prefill_mode}[/bold]")
    if mean_tps_f is not None:
        console.print(f"  mean decode   [bold]{mean_tps_f:.2f} tok/s[/bold]")
    console.print(f"  report        {report_path}")


@app.command()
def smoke(
    timeout: Annotated[float, typer.Option(help="Per-request timeout (seconds).")] = 60.0,
    tools: Annotated[
        bool,
        typer.Option("--tools/--no-tools", help="Also require a tool-call response."),
    ] = True,
) -> None:
    """Hit /props + /v1/chat/completions on the running server; report PASS/FAIL.

    Talks to the server container via the host docker bridge (port is mapped
    from cfg.port → 8080 inside). Pass criteria: valid /props, HTTP 200,
    at least one streamed content token, and by default one tool call.
    """
    cfg = _load_or_build()
    result = smoke_mod.run(cfg, timeout_s=timeout, check_tools=tools)
    console.print(
        f"props={result.props_ok}  tools={result.tool_ok}  "
        f"http={result.http_status}  tokens={result.n_tokens}  "
        f"wall={result.wall_s:.2f}s"
    )
    if result.ok:
        console.print("[green]PASS[/green]")
        return
    console.print(f"[red]FAIL[/red]  {result.error}")
    raise typer.Exit(code=1)


@app.command("download-models")
def download_models() -> None:
    """Fetch the default target + DFlash draft into the models dir.

    Streams `hf download` progress to the user. Resume-safe: re-running after
    an interrupted download picks up where it left off.
    """
    cfg = _load_or_build()
    pres = download_mod.DEFAULT_PRESET
    current = download_mod.status(cfg, pres)
    console.print(f"Models dir: [bold]{cfg.models_dir}[/bold]")
    console.print(f"  target ({pres.target_repo}/{pres.target_file}):"
                  f"  {'present' if current['target_present'] else 'will download'}")
    console.print(f"  draft  ({pres.draft_repo}/{pres.draft_file}):"
                  f"  {'present' if current['draft_present'] else 'will download'}")
    if current["target_present"] and current["draft_present"]:
        console.print("[green]Both present. Nothing to do.[/green]")
        return

    console.print(f"[bold]Downloading[/bold] (~{pres.approx_total_gb} GB total)…")
    rc = download_mod.download_preset(cfg, pres)
    if rc != 0:
        raise typer.Exit(code=rc)
    console.print("[green]Done.[/green]")


@app.command()
def profile(
    export_snapshot: Annotated[
        bool,
        typer.Option(
            "--export-snapshot",
            help="Write a profile snapshot after updating/auditing profile results.",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Snapshot output path when --export-snapshot is set.",
        ),
    ] = Path("profile-snapshot.txt"),
    url: Annotated[
        str,
        typer.Option(
            "--url",
            help="Optional running server base URL. Auto-detects localhost/Docker host routes.",
        ),
    ] = "",
    force_refresh: Annotated[
        bool,
        typer.Option(
            "--force-refresh",
            help="Regenerate matching step results even if fresh artifacts exist.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show freshness decisions without running steps."),
    ] = False,
    step: Annotated[
        str,
        typer.Option("--step", help="Run/audit only one registered step id."),
    ] = "",
) -> None:
    """Update local profile evidence; optionally export the comparable snapshot."""
    cfg = _load_or_build()
    rows = profile_mod.run_profile(
        cfg,
        base_url=url or None,
        step_filter=step or None,
        force_refresh=force_refresh,
        dry_run=dry_run,
    )
    console.print(profile_mod.print_summary(rows))
    if export_snapshot and not dry_run:
        written = profile_mod.export_snapshot(cfg, out, base_url=url or None)
        console.print(f"[green]Wrote[/green] {written}")


@app.command()
def claude(
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="One-shot prompt (non-interactive). Omit for the TUI.",
        ),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(help="Model ID to advertise to Claude Code."),
    ] = "luce-dflash",
) -> None:
    """Launch Claude Code pointed at the running Lucebox server.

    The Anthropic Messages compatibility lives at <base>/v1; we set
    ANTHROPIC_BASE_URL + a placeholder API key so Claude Code talks to
    Lucebox instead of api.anthropic.com. Telemetry + the
    nonstreaming-fallback retry are disabled (privacy + reliability —
    local models don't need the fallback).

    Interactive by default — exec into the full TUI. Pass --prompt for a
    one-shot non-interactive run (matches `claude --print` mode).

    Delegates to ``harness.clients.claude_code.launch`` so this and the
    test-harness path (``harness/clients/run_claude_code.sh``) go through
    one env-config source.
    """
    from harness.clients import claude_code as claude_launcher

    cfg = _load_or_build()
    base_url = url
    if base_url is None:
        bases = profile_mod._server_base_urls(cfg)
        # Take the first that answers /health within a second.
        for candidate in bases:
            if profile_mod._json_get(candidate + "/health", timeout_s=1.0):
                base_url = candidate
                break
        if base_url is None:
            base_url = bases[0]  # fall through; let claude error if down
            console.print(
                f"[yellow]warning:[/yellow] no /health response at {bases[0]} "
                f"— starting claude anyway (server may be down)."
            )

    try:
        rc = claude_launcher.launch(
            base_url=base_url,
            model=model,
            prompt=prompt,
            interactive=prompt is None,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=127) from e
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command()
def version() -> None:
    """Print lucebox version."""
    print(__version__)


def main() -> None:
    """Module entrypoint — `python -m lucebox`."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
