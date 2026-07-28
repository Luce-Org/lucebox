"""Typer app — the user-facing subcommands.

Layout follows the host wrapper's dispatch table. Anything `lucebox`
doesn't intercept (everything outside the systemd surface) ends up here.

Subcommand inventory:
    (no command)           — branded interactive menu
    check                  — readiness report
    config get/set/unset   — read / write a single key in config.toml
    optimize               — apply the recommended hardware profile
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

import lucebox.autotune as autotune_mod
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
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def root_options(
    ctx: typer.Context,
    version_flag: Annotated[
        bool,
        typer.Option("--version", help="Print lucebox version and exit.", is_eager=True),
    ] = False,
) -> None:
    """Apply options shared by the top-level CLI."""
    if version_flag:
        print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            _package_menu()
        else:
            _print_logo()
            console.print(ctx.get_help())
        raise typer.Exit()


# ── helpers ────────────────────────────────────────────────────────────────


_LOGO = r"""
     · ╱
  ·──✦──·  █    █ █ ▄▀▀ █▀▀  █▀▀▄ ▄▀▀▄ █ █
    ╱ ·    █  █ █ █   █▀▀   █▀▀▄ █  █  █
   ·        ▀▀  ▀▀  ▀▀ ▀▀▀  ▀▀▀  ▀▀  ▀ ▀
""".strip("\n")


def _print_logo() -> None:
    """Render the compact Lucebox mark used by both interactive surfaces."""
    console.print(f"[bold gold1]{_LOGO}[/bold gold1]")
    console.print("[dim]             local inference, made simple[/dim]\n")


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


def _package_menu() -> None:
    """Small in-package menu for direct installs and contributor workflows.

    Service lifecycle remains host-owned by ``lucebox.sh``. This menu covers
    the package's own responsibilities and makes a direct ``python -m lucebox``
    invocation useful instead of dropping users into a wall of help text.
    """
    while True:
        _print_logo()
        cfg = _load_or_build()
        active = cfg.model.preset or "not selected"
        console.print(f"Model:        [bold]{escape(active)}[/bold]")
        console.print("Optimization: [bold]Automatic[/bold] (safe defaults for this GPU)\n")
        console.print("  [bold cyan]1[/bold cyan]  Choose or download a model")
        console.print("  [bold cyan]2[/bold cyan]  Apply automatic optimization")
        console.print("  [bold cyan]3[/bold cyan]  Show configuration")
        console.print("  [bold cyan]4[/bold cyan]  Show Docker launch command")
        console.print("  [bold cyan]q[/bold cyan]  Quit")
        try:
            choice = typer.prompt("\nChoose", default="1").strip().lower()
        except (EOFError, typer.Abort):
            return
        if choice in {"q", "quit", "exit"}:
            return
        if choice == "1":
            models_select()
        elif choice == "2":
            optimize()
        elif choice == "3":
            config_get_cmd()
        elif choice == "4":
            print_run()
        else:
            console.print("[yellow]Choose 1–4 or q.[/yellow]")
        console.print()


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


def _activate_preset(cfg: Config, preset: download_mod.ModelPreset) -> None:
    """Persist one selected preset and seed first-run tuning."""
    config_set("model.preset", preset.name)
    config_set("model.target_file", preset.target_file)
    if preset.has_draft and preset.draft_file:
        config_set("model.draft_file", preset.draft_file)
    else:
        # Drop any stale draft_file from a previous activation; the selected
        # preset is explicitly target-only.
        config_unset("model.draft_file")
    console.print(f"[green]Activated:[/green] model.preset = {preset.name}")
    # Never clobber a user-edited [dflash] section. Explicit profile resets
    # go through `lucebox optimize` instead.
    if config_mod.seed_dflash_from_host(cfg.host):
        max_ctx = config_get("dflash.max_ctx")["dflash.max_ctx"][0]
        console.print(
            f"[green]Auto-tuned:[/green] VRAM-tier DFLASH_* defaults "
            f"(max_ctx={max_ctx}) written to config.toml"
        )


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


@models_app.command("select")
def models_select(
    preset: Annotated[
        str,
        typer.Argument(help="Preset name (omit for a numbered menu)."),
    ] = "",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Download and activate without confirmation."),
    ] = False,
) -> None:
    """Choose, download, and activate a model in one guided step."""
    cfg = _load_or_build()
    if not preset:
        names = sorted(download_mod.PRESETS)
        recommended = download_mod.recommend_preset(cfg.host)
        default_name = cfg.model.preset or recommended or names[0]
        default_index = names.index(default_name) + 1 if default_name in names else 1

        table = Table(title="Choose a model", show_lines=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("model")
        table.add_column("download")
        table.add_column("status")
        table.add_column("notes")
        for index, name in enumerate(names, start=1):
            candidate = download_mod.PRESETS[name]
            labels: list[str] = []
            if name == cfg.model.preset:
                labels.append("active")
            if name == recommended:
                labels.append("recommended")
            table.add_row(
                str(index),
                name,
                f"~{candidate.approx_total_gb} GB",
                download_mod.installed_status(cfg, candidate),
                ", ".join(labels) or candidate.description,
            )
        console.print(table)
        try:
            answer = typer.prompt(
                "Model number or name",
                default=str(default_index),
            ).strip()
        except (EOFError, typer.Abort):
            console.print("[dim]No model changed.[/dim]")
            return
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            preset = names[int(answer) - 1]
        else:
            preset = answer

    try:
        selected = download_mod.resolve_preset(preset)
    except KeyError as exc:
        error_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    state = download_mod.installed_status(cfg, selected)
    if state == "installed":
        # Selection must work on a preloaded buyer appliance even when it is
        # offline. Do not query Hugging Face merely to activate files that are
        # already present locally.
        _activate_preset(cfg, selected)
        return
    if state != "installed" and not yes:
        if not typer.confirm(
            f"Download about {selected.approx_total_gb} GB and activate {selected.name}?",
            default=True,
        ):
            console.print("[dim]No model changed.[/dim]")
            return
    models_download(selected.name, activate=True)


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
        _activate_preset(cfg, pres)


@app.command()
def optimize(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Apply without confirmation."),
    ] = False,
) -> None:
    """Apply the recommended hardware-aware inference profile.

    Stable CUDA/ROCm fast paths remain enabled by the engine itself. This
    profile selects a safe context size, cache format, and DFlash budget from
    detected VRAM while leaving experimental features off.
    """
    cfg = _load_or_build()
    recommended = autotune_mod.runtime_from_host(cfg.host)
    console.print("[bold]Automatic optimization (recommended)[/bold]")
    console.print("  DFlash decode       automatic when the model has a matching draft")
    console.print("  GPU fast paths      enabled by the engine on CUDA and ROCm")
    console.print(f"  Maximum context     {recommended.max_ctx:,} tokens")
    cache = recommended.cache_type_k or "model default"
    console.print(f"  KV cache            {cache}")
    console.print("  Experimental paths  off")
    if not yes and not typer.confirm("Apply this profile?", default=True):
        console.print("[dim]Optimization unchanged.[/dim]")
        return
    config_mod.seed_dflash_from_host(cfg.host, force=True)
    console.print("[green]Automatic optimization applied.[/green]")
    console.print("[dim]Advanced users can still use `lucebox config set dflash.KEY=VALUE`.[/dim]")


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
