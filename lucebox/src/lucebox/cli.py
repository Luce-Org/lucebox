"""Typer app — the user-facing subcommands.

Layout follows the host wrapper's dispatch table. Anything `lucebox`
doesn't intercept (everything outside the systemd surface) ends up here.

Subcommand inventory:
    check                  — readiness report
    config get/set/unset   — read / write a single key in config.toml
    pull                   — docker pull the selected CUDA or ROCm image
    print-run              — emit the docker-run command for the server
    print-serve-argv       — same, raw argv lines (consumed by `lucebox serve`)
    models                 — list / download presets, activate one
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

import lucebox.config as config_mod
import lucebox.docker_run as docker_run
import lucebox.download as download_mod
import lucebox.host_check as host_check
from lucebox import __version__
from lucebox.config import config_get, config_set, config_unset, live_config
from lucebox.host_facts import from_env
from lucebox.types import Config

app = typer.Typer(
    name="lucebox",
    help="Host CLI for the lucebox-hub container. Invoked by lucebox.sh.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def root_options(
    version_flag: Annotated[
        bool,
        typer.Option("--version", help="Print lucebox version and exit.", is_eager=True),
    ] = False,
) -> None:
    """Apply options shared by the top-level CLI."""
    if version_flag:
        print(__version__)
        raise typer.Exit()


# ── helpers ────────────────────────────────────────────────────────────────


def _load_or_build() -> Config:
    """env > config.toml > dataclass defaults — the canonical precedence.

    Only the five documented top-level scalars have environment overrides;
    dflash, host, and model settings intentionally remain config-driven.
    """
    try:
        cfg = config_mod.load()
        if cfg is None:
            return live_config()
        # Host facts exported by the wrapper take precedence over an absent or
        # stale persisted snapshot. A zero-filled environment means the CLI was
        # invoked directly, so retain any snapshot already in the config.
        live_host = from_env()
        host = live_host if live_host.vram_gb > 0 or live_host.nproc > 0 else cfg.host
        return config_mod.overlay_env(replace(cfg, host=host))
    except (OSError, ValueError) as exc:
        error_console.print(f"[red]Invalid configuration:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc


def _server_spec() -> docker_run.DockerRunSpec:
    """Build the server command and turn user-path errors into clean CLI output."""
    cfg = _load_or_build()
    try:
        return docker_run.server_run_spec(cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]Cannot build server command:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc


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
def pull() -> None:
    """`docker pull` the image variant from config.toml."""
    cfg = _load_or_build()
    tag = f"{cfg.image}:{cfg.variant}"
    console.print(f"[bold]Pulling {escape(tag)}[/bold] (~14 GB; takes a while)…")
    rc = docker_run.docker_pull(tag)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("print-run")
def print_run() -> None:
    """Print the docker-run command for the server (copy-pasteable)."""
    print(_server_spec().printable())


@app.command("print-serve-argv")
def print_serve_argv() -> None:
    """Emit the server docker-run argv, one token per line.

    Consumed by lucebox.sh's `serve` subcommand and the systemd unit. Kept as
    a separate command from `print-run` so the bash side has a guaranteed
    machine-readable contract that's independent of the pretty formatter.
    """
    for tok in _server_spec().argv():
        print(tok)


# ── config sub-app ─────────────────────────────────────────────────────────


config_app = typer.Typer(no_args_is_help=True, help="Read/write keys in config.toml.")
app.add_typer(config_app, name="config")


@config_app.command("get")
def config_get_cmd(
    key: Annotated[str, typer.Argument(help="Dotted key (omit to list every key).")] = "",
) -> None:
    """Print a single key (or every reachable key) with its origin annotation."""
    try:
        entries = config_get(key or None)
    except (KeyError, OSError, ValueError) as exc:
        error_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    for k, (value, origin) in entries.items():
        console.print(f"{k} = {escape(repr(value))} ([dim]from {origin}[/dim])")


@config_app.command("set")
def config_set_cmd(
    kv: Annotated[str, typer.Argument(help='"key=value" pair (e.g. "model.preset=qwen3.6-27b")')],
) -> None:
    """Set one dotted key. Auto-creates config.toml when missing.

    Only the named key is written — other on-disk keys are preserved
    untouched, unset keys stay implicit. Use `lucebox config unset` to
    remove a key (next read falls back to the live default).
    """
    if "=" not in kv:
        console.print("[red]argument must be key=value[/red]")
        raise typer.Exit(code=2)
    key, _, value = kv.partition("=")
    key = key.strip()
    value = value.strip()
    try:
        config_set(key, value)
    except (KeyError, OSError, ValueError) as exc:
        error_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]Set[/green] {escape(key)} = {escape(value)}")


@config_app.command("unset")
def config_unset_cmd(
    key: Annotated[str, typer.Argument(help="Dotted key to remove from config.toml.")],
) -> None:
    """Remove a key from config.toml. Next read uses the live default."""
    try:
        changed = config_unset(key)
    except (KeyError, OSError, ValueError) as exc:
        error_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    if changed:
        console.print(f"[green]Unset[/green] {escape(key)}")
    else:
        console.print(f"[dim]{escape(key)} was not in config.toml; nothing to do[/dim]")


# ── models sub-app ─────────────────────────────────────────────────────────


models_app = typer.Typer(
    no_args_is_help=False, help="Manage local model presets (list, download, activate)."
)
app.add_typer(models_app, name="models")


def _print_installed_presets() -> None:
    cfg = _load_or_build()
    installed = download_mod.installed_presets(cfg)
    active = cfg.model.preset
    console.print(f"Models dir: [bold]{cfg.models_dir}[/bold]")
    if not installed:
        console.print("[dim]No presets installed yet — try `lucebox models download`.[/dim]")
        return
    table = Table()
    table.add_column("preset")
    table.add_column("status")
    table.add_column("size (GB)")
    for pres in installed:
        marker = "* " if pres.name == active else "  "
        size_gb = download_mod.installed_size_gb(cfg, pres)
        table.add_row(f"{marker}{pres.name}", "installed", f"{size_gb:.1f}")
    console.print(table)
    total = sum(download_mod.installed_size_gb(cfg, p) for p in installed)
    console.print(f"[dim]Total disk usage: {total:.1f} GB[/dim]")


@models_app.callback(invoke_without_command=True)
def models_default(ctx: typer.Context) -> None:
    """Default action: list installed presets, mark active with `*`."""
    if ctx.invoked_subcommand is None:
        _print_installed_presets()


@models_app.command("list")
def models_list() -> None:
    """Show every registered preset (installed or not) with status + size."""
    cfg = _load_or_build()
    active = cfg.model.preset
    table = Table()
    table.add_column("preset")
    table.add_column("status")
    table.add_column("size (GB)")
    table.add_column("description")
    for name in sorted(download_mod.PRESETS):
        pres = download_mod.PRESETS[name]
        marker = "* " if name == active else "  "
        status = download_mod.installed_status(cfg, pres)
        size = download_mod.installed_size_gb(cfg, pres)
        size_text = f"{size:.1f}" if size > 0 else f"~{pres.approx_total_gb}*"
        table.add_row(f"{marker}{name}", status, size_text, pres.description or "")
    console.print(table)


@models_app.command("download")
def models_download(
    preset: Annotated[str, typer.Argument(help="Preset name (empty = recommend)")] = "",
    activate: Annotated[
        bool, typer.Option("--activate", help="Also set as active preset (model.preset).")
    ] = False,
) -> None:
    """Fetch a preset's GGUFs into the models dir.

    With no argument and no preset configured, recommends one for this
    host's VRAM tier and auto-activates it (the first-install path).
    Otherwise the named preset is downloaded; pass ``--activate`` to
    also flip `model.preset` to it.
    """
    cfg = _load_or_build()
    if not preset:
        if cfg.model.preset:
            console.print(
                "[yellow]No preset specified and one is already active. "
                "Pass an explicit preset name (or use --activate to switch).[/yellow]"
            )
            raise typer.Exit(code=2)
        recommended = download_mod.recommend_preset(cfg.host)
        if recommended is None:
            console.print(
                "[red]Cannot recommend a preset for this host. "
                "Run `lucebox models list` and pick one explicitly.[/red]"
            )
            raise typer.Exit(code=2)
        preset = recommended
        activate = True
        console.print(
            f"[bold]Recommended preset: {preset}[/bold] "
            "(no preset configured; auto-activating after download)"
        )

    try:
        pres = download_mod.resolve_preset(preset)
    except KeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    current = download_mod.status(cfg, pres)
    console.print(f"Models dir: [bold]{cfg.models_dir}[/bold]")
    console.print(f"Preset:     [bold]{pres.name}[/bold]")
    console.print(
        f"  target ({pres.target_repo}/{pres.target_file}):"
        f"  {'present' if current['target_present'] else 'will download'}"
    )
    if pres.has_draft:
        console.print(
            f"  draft  ({pres.draft_repo}/{pres.draft_file}):"
            f"  {'present' if current['draft_present'] else 'will download'}"
        )
    else:
        console.print("  draft  [dim](none — target-only preset)[/dim]")

    if current["target_present"] and current["draft_present"]:
        console.print("[green]Already present.[/green]")
    else:
        console.print(f"[bold]Downloading[/bold] (~{pres.approx_total_gb} GB total)…")
        rc = download_mod.download_preset(cfg, pres)
        if rc != 0:
            raise typer.Exit(code=rc)
        console.print("[green]Done.[/green]")

    if activate:
        config_set("model.preset", preset)
        if pres.target_file:
            config_set("model.target_file", pres.target_file)
        if pres.has_draft and pres.draft_file:
            config_set("model.draft_file", pres.draft_file)
        else:
            # Drop any stale draft_file from a previous activation; the
            # active preset has no draft.
            config_unset("model.draft_file")
        console.print(f"[green]Activated:[/green] model.preset = {preset}")
        # First-time setup: bake the VRAM-tier DFLASH_* heuristic into
        # config.toml so `lucebox serve` is auto-tuned to this host instead
        # of falling back to the conservative class defaults. Never clobbers
        # an existing [dflash] section.
        if config_mod.seed_dflash_from_host(cfg.host):
            max_ctx = config_get("dflash.max_ctx")["dflash.max_ctx"][0]
            console.print(
                f"[green]Auto-tuned:[/green] VRAM-tier DFLASH_* defaults "
                f"(max_ctx={max_ctx}) written to config.toml"
            )


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
