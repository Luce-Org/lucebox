"""Typer app — the user-facing subcommands.

Layout follows the host wrapper's dispatch table. Anything `lucebox`
doesn't intercept (everything outside the systemd surface) ends up here.

Subcommand inventory:
    check                  — readiness report
    config get/set/unset   — read / write a single key in config.toml
    pull                   — docker pull the cuda12 image
    print-run              — emit the docker-run command for the server
    print-serve-argv       — same, raw argv lines (consumed by `lucebox serve`)
    models                 — list / download presets, activate one
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import lucebox.config as config_mod
import lucebox.docker_run as docker_run
import lucebox.download as download_mod
import lucebox.host_check as host_check
from lucebox import __version__
from lucebox.config import config_get, config_set, config_unset, live_config
from lucebox.host_facts import from_env

app = typer.Typer(
    name="lucebox",
    help="Host CLI for the lucebox-hub container. Invoked by lucebox.sh.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# ── helpers ────────────────────────────────────────────────────────────────


def _load_or_build() -> config_mod.Config:  # type: ignore[name-defined]
    """env > config.toml > dataclass defaults — the canonical precedence.

    Without the env-overlay step below, `config_mod.load()` returned the
    persisted config verbatim and `LUCEBOX_IMAGE` / `LUCEBOX_VARIANT` /
    `LUCEBOX_PORT` / `LUCEBOX_CONTAINER` / `LUCEBOX_MODELS` from the
    systemd unit's `Environment=` (or any one-shot shell export) were
    silently dropped. That contradicted the precedence lucebox.sh
    documents and applies — and bit sindri when its config.toml had
    `[image]` without `registry`, so the dataclass default
    `ghcr.io/luce-org/lucebox-hub` won over the unit's
    `LUCEBOX_IMAGE=ghcr.io/easel/lucebox-hub`.

    Fix: overlay env on top of the loaded config (or the live_config
    fallback when config.toml is absent). Only the five top-level
    scalars have env hooks — dflash/host/model don't, by design.
    """
    cfg = config_mod.load()
    if cfg is None:
        cfg = live_config()
    # Overlay live host facts. When ``config.toml`` exists without a
    # ``[host]`` block (the common case — operators don't hand-edit
    # host facts), ``cfg.host`` defaults to a zero-filled ``HostFacts``
    # and the DFLASH_* serve heuristic silently falls through to the
    # "no VRAM signal" path. Re-probe from env so the wrapper-exported
    # LUCEBOX_HOST_* facts always win over the persisted (possibly
    # absent) snapshot.
    live_host = from_env()
    host = live_host if live_host.vram_gb > 0 or live_host.nproc > 0 else cfg.host
    return replace(
        cfg,
        variant=os.environ.get("LUCEBOX_VARIANT", cfg.variant),
        image=os.environ.get("LUCEBOX_IMAGE", cfg.image),
        container_name=os.environ.get("LUCEBOX_CONTAINER", cfg.container_name),
        port=int(os.environ.get("LUCEBOX_PORT", str(cfg.port))),
        models_dir=Path(os.environ.get("LUCEBOX_MODELS", str(cfg.models_dir))),
        host=host,
    )


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
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    for k, (value, origin) in entries.items():
        console.print(f"{k} = {value!r} ([dim]from {origin}[/dim])")


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
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]Set[/green] {key} = {value}")


@config_app.command("unset")
def config_unset_cmd(
    key: Annotated[str, typer.Argument(help="Dotted key to remove from config.toml.")],
) -> None:
    """Remove a key from config.toml. Next read uses the live default."""
    try:
        changed = config_unset(key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if changed:
        console.print(f"[green]Unset[/green] {key}")
    else:
        console.print(f"[dim]{key} was not in config.toml; nothing to do[/dim]")


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
        console.print(f"[red]{exc}[/red]")
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
