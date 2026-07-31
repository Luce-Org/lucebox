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
from typing import Annotated, Literal

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
from lucebox.placement import PlacementPlan
from lucebox.types import Config, DflashRuntime, ModelMeta

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

    Only the five documented top-level scalars have environment overrides.
    Optimization, placement, and model settings remain config-driven, while
    fresh host facts replace an absent or stale persisted snapshot.
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


def _active_optimization_names(cfg: Config) -> list[str]:
    """Return the product-level features represented by the loaded config."""
    active: list[str] = []
    preset = download_mod.PRESETS.get(cfg.model.preset)
    if (
        cfg.dflash.speculative_decode
        and preset is not None
        and autotune_mod.draft_available(cfg, preset)
    ):
        active.append("DFlash")
    if cfg.dflash.prefill_mode != "off":
        active.append("PFlash")
    if cfg.dflash.kvflash != "off":
        active.append("KVFlash")
    if cfg.dflash.spark:
        active.append("Spark")
    return active


def _optimization_label(cfg: Config) -> str:
    mode = config_mod.optimization_mode()
    mode_label = {
        "automatic": "Automatic",
        "custom": "Custom",
        "unconfigured": "Not configured",
    }[mode]
    active = _active_optimization_names(cfg)
    return f"{mode_label} ({', '.join(active) if active else 'standard engine'})"


def _placement_label(cfg: Config) -> str:
    placement = cfg.placement
    target = placement.target_device or ", ".join(placement.target_devices)
    if placement.remote_expert_device:
        return f"{target} target + {placement.remote_expert_device} Spark experts"
    if placement.draft_device:
        return f"{target} target + {placement.draft_device} draft/scorer"
    if placement.target_devices:
        return f"{target} target split"
    return target or "server default"


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
        console.print(f"Optimization: [bold]{escape(_optimization_label(cfg))}[/bold]")
        console.print(f"Execution:    [bold]{escape(_placement_label(cfg))}[/bold]\n")
        console.print("  [bold cyan]1[/bold cyan]  Choose or download a model")
        console.print("  [bold cyan]2[/bold cyan]  Review optimizations")
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


def _model_meta(preset: download_mod.ModelPreset) -> ModelMeta:
    """Build the persisted model selection for one catalog preset."""
    return ModelMeta(
        preset=preset.name,
        target_file=preset.target_file,
        draft_file=preset.draft_file if preset.has_draft and preset.draft_file else "",
    )


def _preset_placement(cfg: Config, preset: download_mod.ModelPreset) -> PlacementPlan:
    """Plan the placement that activating ``preset`` would persist."""
    selected_cfg = replace(cfg, model=_model_meta(preset))
    if config_mod.optimization_mode() == "custom":
        return autotune_mod.placement_for_runtime(selected_cfg, cfg.dflash)
    return autotune_mod.automatic_plan(selected_cfg).placement


def _require_runnable_preset(cfg: Config, preset: download_mod.ModelPreset) -> None:
    """Stop before a large download when this machine cannot run the model."""
    placement = _preset_placement(cfg, preset)
    if placement.runnable:
        return
    error_console.print(
        f"[red]{escape(preset.label)} cannot run on the detected hardware.[/red]\n"
        f"[dim]{escape(placement.reason)}[/dim]\n"
        "[dim]No model files were downloaded. Run `lucebox check` to review "
        "the detected accelerators.[/dim]"
    )
    raise typer.Exit(code=2)


def _activate_preset(cfg: Config, preset: download_mod.ModelPreset) -> None:
    """Activate one preset together with a runnable execution profile."""
    selected_model = _model_meta(preset)
    selected_cfg = replace(
        cfg,
        model=selected_model,
    )

    mode = config_mod.optimization_mode()
    if mode == "custom":
        placement = autotune_mod.placement_for_runtime(selected_cfg, cfg.dflash)
        if not placement.runnable:
            error_console.print(
                "[red]Model not activated: the custom profile has no runnable "
                f"placement on this machine.[/red]\n[dim]{escape(placement.reason)}[/dim]"
            )
            raise typer.Exit(code=2)
        runtime = placement.optimization_runtime
        config_mod.write_model_profile(
            selected_model,
            runtime,
            placement.runtime,
            mode="custom",
            source="model-switch",
        )
        console.print(f"[green]Activated:[/green] model.preset = {preset.name}")
        if runtime != cfg.dflash:
            console.print(
                "[yellow]Custom profile adjusted.[/yellow] Features incompatible "
                "with the new placement were disabled; run `lucebox optimize "
                "--advanced` to review it."
            )
        else:
            console.print(
                "[yellow]Custom optimization profile kept.[/yellow] "
                "Execution placement was recalculated for this model."
            )
        return

    plan = autotune_mod.automatic_plan(selected_cfg)
    if not plan.placement.runnable:
        error_console.print(
            "[red]Model not activated: Automatic found no runnable placement "
            f"on this machine.[/red]\n[dim]{escape(plan.placement.reason)}[/dim]"
        )
        raise typer.Exit(code=2)
    config_mod.write_model_profile(
        selected_model,
        plan.runtime,
        plan.placement.runtime,
    )
    console.print(f"[green]Activated:[/green] model.preset = {preset.name}")
    active = ", ".join(plan.active_names) or "standard engine"
    console.print(
        f"[green]Optimized:[/green] Automatic profile for {preset.name} "
        f"({active}; max_ctx={plan.runtime.max_ctx})"
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
    table.add_column("model")
    table.add_column("preset")
    table.add_column("status")
    table.add_column("size (GB)")
    table.add_column("description")
    for pres in download_mod.catalog_presets():
        marker = "* " if pres.name == active else "  "
        status = download_mod.installed_status(cfg, pres)
        size = download_mod.installed_size_gb(cfg, pres)
        size_text = f"{size:.1f}" if size > 0 else f"~{pres.approx_total_gb}*"
        table.add_row(
            f"{marker}{pres.label}",
            pres.name,
            status,
            size_text,
            pres.description or "",
        )
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
        candidates = download_mod.catalog_presets(featured_only=True)
        names = [candidate.name for candidate in candidates]
        recommended = download_mod.recommend_preset(cfg.host)
        default_name = cfg.model.preset or recommended or names[0]
        default_index = names.index(default_name) + 1 if default_name in names else 1

        table = Table(title="Choose a model", show_lines=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("model")
        table.add_column("download")
        table.add_column("status")
        table.add_column("this machine")
        for index, candidate in enumerate(candidates, start=1):
            labels: list[str] = []
            if candidate.name == cfg.model.preset:
                labels.append("active")
            if candidate.name == recommended:
                labels.append("recommended")
            placement = _preset_placement(cfg, candidate)
            fit = "ready" if placement.runnable else "not enough compatible memory"
            table.add_row(
                str(index),
                candidate.label,
                f"~{candidate.approx_total_gb} GB",
                download_mod.installed_status(cfg, candidate),
                ", ".join((*labels, fit)) if labels else fit,
            )
        console.print(table)
        extra = [item.name for item in download_mod.catalog_presets() if not item.featured]
        if extra:
            console.print(
                "[dim]More supported presets: "
                f"{escape(', '.join(extra))}. See `lucebox models list`.[/dim]"
            )
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
    _require_runnable_preset(cfg, selected)
    if state != "installed" and not yes:
        if not typer.confirm(
            f"Download about {selected.approx_total_gb} GB and activate {selected.label}?",
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

    if activate:
        _require_runnable_preset(cfg, pres)

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


def _render_optimization_plan(
    plan: autotune_mod.OptimizationPlan,
    *,
    title: str = "Automatic optimization (recommended)",
) -> None:
    console.print(f"[bold]{title}[/bold]")
    console.print(f"Model: {escape(plan.model_name)}")
    placement_state = "[green]ready[/green]" if plan.placement.runnable else "[red]blocked[/red]"
    console.print(f"Execution: [bold]{escape(plan.placement.summary)}[/bold] ({placement_state})")
    console.print(f"[dim]{escape(plan.placement.reason)}[/dim]")
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Optimization", style="bold")
    table.add_column("State")
    table.add_column("Why")
    for decision in plan.decisions:
        if decision.enabled:
            state = "[green]ON[/green]"
        elif decision.available:
            state = "[dim]off[/dim]"
        else:
            state = "[dim]not available[/dim]"
        table.add_row(decision.name, state, escape(decision.reason))
    console.print(table)
    cache = plan.runtime.cache_type_k or "model default"
    console.print(
        f"Context: [bold]{plan.runtime.max_ctx:,}[/bold] tokens  ·  "
        f"KV format: [bold]{cache}[/bold]  ·  "
        f"DFlash budget: [bold]{plan.runtime.budget}[/bold]"
    )


def _render_custom_runtime(
    runtime: DflashRuntime,
    placement: PlacementPlan | None = None,
) -> None:
    """Show the exact product choices before a custom profile is committed."""
    console.print("\n[bold]Your custom profile[/bold]")
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Optimization", style="bold")
    table.add_column("Selected")
    selections = (
        ("DFlash", runtime.speculative_decode),
        ("PFlash", runtime.prefill_mode != "off"),
        ("KVFlash", runtime.kvflash != "off"),
        ("Spark", runtime.spark),
    )
    for name, enabled in selections:
        table.add_row(name, "[green]ON[/green]" if enabled else "[dim]off[/dim]")
    console.print(table)
    if runtime.cache_type_k == runtime.cache_type_v:
        cache = runtime.cache_type_k or "model default"
    else:
        cache = f"K={runtime.cache_type_k or 'default'}, V={runtime.cache_type_v or 'default'}"
    console.print(
        f"Context: [bold]{runtime.max_ctx:,}[/bold] tokens  ·  "
        f"KV format: [bold]{cache}[/bold]  ·  "
        f"DFlash budget: [bold]{runtime.budget}[/bold]"
    )
    if placement is not None:
        console.print(
            f"Execution: [bold]{escape(placement.summary)}[/bold]  ·  {escape(placement.reason)}"
        )


def _ensure_optimizer_drafter(cfg: Config, *, assume_yes: bool = False) -> bool:
    """Offer the one shared scorer asset and degrade cleanly when offline."""
    if download_mod.optimizer_drafter_installed(cfg):
        return True
    question = (
        f"Install the shared PFlash/KVFlash scorer "
        f"(~{download_mod.OPTIMIZER_DRAFTER_APPROX_GB:g} GB)?"
    )
    if not assume_yes and not typer.confirm(question, default=True):
        return False
    console.print("[bold]Installing the shared long-context scorer…[/bold]")
    if download_mod.download_optimizer_drafter(cfg) != 0:
        console.print(
            "[yellow]Continuing without the scorer; the rest of the automatic "
            "profile is still safe.[/yellow]"
        )
        return False
    if not download_mod.optimizer_drafter_installed(cfg):
        console.print("[yellow]The scorer download completed but its file is missing.[/yellow]")
        return False
    console.print("[green]Shared optimizer installed.[/green]")
    return True


def _customize_runtime(
    cfg: Config,
    plan: autotune_mod.OptimizationPlan,
) -> DflashRuntime:
    """Prompt only for product-level choices; keep low-level knobs automatic."""
    preset = download_mod.PRESETS.get(cfg.model.preset)
    runtime = plan.runtime
    console.print("\n[bold]Customize optimizations[/bold]")
    console.print("[dim]Context, cache format, and budgets remain hardware-tuned.[/dim]\n")

    if plan.dflash.available:
        speculative_decode = typer.confirm(
            "Enable DFlash speculative decode?", default=plan.dflash.enabled
        )
    else:
        speculative_decode = False
        console.print(f"DFlash: [dim]unavailable — {escape(plan.dflash.reason)}[/dim]")

    pflash = False
    if plan.pflash.available:
        pflash = typer.confirm(
            "Enable PFlash automatically for long prompts?", default=plan.pflash.enabled
        )
        if pflash and not _ensure_optimizer_drafter(cfg):
            pflash = False
            console.print("[yellow]PFlash left off because its scorer is unavailable.[/yellow]")
    else:
        console.print(f"PFlash: [dim]unavailable — {escape(plan.pflash.reason)}[/dim]")

    kvflash = False
    kvflash_policy: Literal["drafter", "lru", "qk"] = "drafter"
    if preset is not None and plan.kvflash.available:
        kvflash = typer.confirm(
            "Enable KVFlash bounded long-context memory?", default=plan.kvflash.enabled
        )
    else:
        console.print(f"KVFlash: [dim]unavailable — {escape(plan.kvflash.reason)}[/dim]")
    if kvflash:
        has_scorer = download_mod.optimizer_drafter_installed(cfg)
        if has_scorer:
            policy_choices = "drafter"
            if preset is not None and preset.architecture == "qwen35":
                policy_choices += "/qk/lru"
            else:
                policy_choices += "/lru"
            answer = (
                typer.prompt(f"KVFlash policy ({policy_choices})", default="drafter")
                .strip()
                .lower()
            )
            if answer not in policy_choices.split("/"):
                fallback_policy = policy_choices.split("/")[0]
                console.print(f"[yellow]Unknown policy; using {fallback_policy}.[/yellow]")
                answer = fallback_policy
            if answer == "qk":
                kvflash_policy = "qk"
            elif answer == "lru":
                kvflash_policy = "lru"
            else:
                kvflash_policy = "drafter"
        elif preset is not None and preset.architecture == "qwen35":
            kvflash_policy = "qk"
            console.print("KVFlash policy: [bold]qk[/bold] (no extra scorer required)")
        else:
            console.print(
                "[yellow]KVFlash would use recency-only LRU without the shared scorer; "
                "older context can be evicted.[/yellow]"
            )
            if typer.confirm("Install the scorer instead?", default=True):
                if _ensure_optimizer_drafter(cfg):
                    kvflash_policy = "drafter"
                else:
                    kvflash_policy = "lru"
            else:
                kvflash_policy = "lru"

    spark = False
    if plan.spark.available:
        spark = typer.confirm(
            "Enable Spark self-tuning MoE expert offload?", default=plan.spark.enabled
        )
    else:
        console.print(f"Spark: [dim]unavailable — {escape(plan.spark.reason)}[/dim]")

    uses_scorer = pflash or (kvflash and kvflash_policy == "drafter")
    prefill_drafter = download_mod.optimizer_drafter_container_path() if uses_scorer else ""
    if kvflash and runtime.fa_window > 0:
        console.print("[dim]KVFlash selected: disabling the incompatible FA window.[/dim]")

    return replace(
        runtime,
        speculative_decode=speculative_decode,
        lazy=False if not speculative_decode else runtime.lazy,
        prefill_mode="auto" if pflash else "off",
        prefill_keep_ratio=0.10 if pflash else runtime.prefill_keep_ratio,
        prefill_threshold=32768,
        prefill_drafter=prefill_drafter,
        kvflash="auto" if kvflash else "off",
        kvflash_policy=kvflash_policy,
        spark=spark,
        spark_vram_gb=0.0,
        fa_window=0 if kvflash else runtime.fa_window,
    )


@app.command()
def optimize(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Apply Automatic without prompts."),
    ] = False,
    advanced: Annotated[
        bool,
        typer.Option("--advanced", help="Review DFlash/PFlash/KVFlash/Spark choices."),
    ] = False,
) -> None:
    """Choose and apply a model-aware, hardware-aware optimization profile."""
    if yes and advanced:
        error_console.print(
            "[red]--advanced is interactive and cannot be combined with --yes[/red]"
        )
        raise typer.Exit(code=2)

    cfg = _load_or_build()
    plan = autotune_mod.automatic_plan(cfg)
    _render_optimization_plan(plan)

    custom = advanced
    if not yes and not advanced:
        console.print("\n  [bold cyan]1[/bold cyan]  Apply Automatic (recommended)")
        console.print("  [bold cyan]2[/bold cyan]  Customize")
        console.print("  [bold cyan]q[/bold cyan]  Cancel")
        choice = typer.prompt("\nChoose", default="1").strip().lower()
        if choice in {"q", "quit", "cancel"}:
            console.print("[dim]Optimization unchanged.[/dim]")
            return
        if choice not in {"1", "2"}:
            error_console.print("[red]Choose 1, 2, or q.[/red]")
            raise typer.Exit(code=2)
        custom = choice == "2"

    if custom:
        runtime = _customize_runtime(cfg, plan)
        placement = autotune_mod.placement_for_runtime(cfg, runtime)
        runtime = placement.optimization_runtime
        _render_custom_runtime(runtime, placement)
        if not placement.runnable:
            error_console.print(
                "[red]This profile has no runnable placement on the detected hardware.[/red]"
            )
            raise typer.Exit(code=2)
        if not typer.confirm("Apply this custom profile?", default=True):
            console.print("[dim]Optimization unchanged.[/dim]")
            return
        config_mod.write_optimization_runtime(
            runtime,
            placement=placement.runtime,
            mode="custom",
            source="guided",
        )
        console.print("[green]Custom optimization profile applied.[/green]")
        return

    if plan.needs_optimizer_drafter and _ensure_optimizer_drafter(cfg, assume_yes=yes):
        plan = autotune_mod.automatic_plan(cfg)
        _render_optimization_plan(plan, title="Updated automatic plan")
    if not yes and not typer.confirm("Apply this automatic profile?", default=True):
        console.print("[dim]Optimization unchanged.[/dim]")
        return
    if not plan.placement.runnable:
        error_console.print(
            "[red]Automatic cannot produce a runnable placement for this model and machine.[/red]"
        )
        raise typer.Exit(code=2)
    config_mod.write_optimization_runtime(
        plan.runtime,
        placement=plan.placement.runtime,
    )
    console.print("[green]Automatic optimization applied.[/green]")
    console.print(
        "[dim]Run `lucebox optimize --advanced` anytime to review individual features.[/dim]"
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
