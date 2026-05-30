"""Typer app — the user-facing subcommands.

Layout follows the host wrapper's dispatch table. Anything `lucebox`
doesn't intercept (everything outside the systemd surface) ends up here.

Subcommand inventory:
    check                  — readiness report
    config get/set/unset   — read / write a single key in config.toml
    pull                   — docker pull the cuda12 image
    print-run              — emit the docker-run command for the server
    print-serve-argv       — same, raw argv lines (consumed by `lucebox serve`)
    autotune               — print/persist VRAM-tier DFLASH_* defaults; `--sweep`
                              empirically tests a per-tier bracket and persists the winner
    profile                — run a luce-bench snapshot via the running container
    smoke                  — hit /props + /v1/chat/completions on a running server
    models                 — list / download presets, activate one
    claude                 — launch Claude Code pointed at the running server
    codex                  — launch Codex pointed at the running server
    opencode               — launch OpenCode pointed at the running server
    hermes                 — launch Hermes pointed at the running server
    pi                     — launch Pi pointed at the running server
    openclaw               — launch OpenClaw pointed at the running server
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

import lucebox.autotune as autotune_mod
import lucebox.config as config_mod
import lucebox.docker_run as docker_run
import lucebox.download as download_mod
import lucebox.host_check as host_check
import lucebox.profile as profile_mod
import lucebox.smoke as smoke_mod
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


# The strict 11-field allowlist that mirrors lucebench's snapshot
# config.json. Used by `autotune --apply` to write dflash.* keys.
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
    return replace(
        cfg,
        variant=os.environ.get("LUCEBOX_VARIANT", cfg.variant),
        image=os.environ.get("LUCEBOX_IMAGE", cfg.image),
        container_name=os.environ.get("LUCEBOX_CONTAINER", cfg.container_name),
        port=int(os.environ.get("LUCEBOX_PORT", str(cfg.port))),
        models_dir=Path(os.environ.get("LUCEBOX_MODELS", str(cfg.models_dir))),
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


# ── autotune ───────────────────────────────────────────────────────────────


@app.command()
def autotune(
    apply_: Annotated[
        bool,
        typer.Option("--apply", help="Write the 11 dflash.* keys to config.toml."),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable output (the asdict of DflashRuntime)."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "With --apply: overwrite even when a persisted dflash.* key "
                "already differs from the recommendation (e.g. a sweep-tuned "
                "value)."
            ),
        ),
    ] = False,
    sweep: Annotated[
        bool,
        typer.Option(
            "--sweep",
            help=(
                "Empirically test a per-VRAM-tier bracket of dflash.* configs "
                "against the live server and persist the winner. Uses "
                "`lucebox config set` + `lucebox restart` + `luce-bench "
                "snapshot` for each cell."
            ),
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="With --sweep: skip the confirmation prompt before starting.",
        ),
    ] = False,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help=(
                "With --sweep: which workload profile to use. "
                "'heuristic' (default) brackets KV-quant axes and scores by "
                "mean decode_tps from a luce-bench snapshot. "
                "'coding-agent-loop' brackets max_ctx × fa_window × budget × "
                "pflash and scores by pass-rate on a real recorded agentic "
                "session replay, then speed. See `lucebox autotune "
                "--list-profiles` for the full set."
            ),
        ),
    ] = "heuristic",
    list_profiles: Annotated[
        bool,
        typer.Option(
            "--list-profiles",
            help="Print registered autotune profiles + descriptions and exit.",
        ),
    ] = False,
) -> None:
    """Compute the recommended DflashRuntime for this host.

    By default prints a Rich table comparing live defaults vs the
    heuristic recommendation. Pass ``--apply`` to persist every value
    in the 11-field allowlist to config.toml (sparse — only those keys
    land on disk). ``--json`` dumps the recommendation as JSON for
    scripting.

    Guard: when ``--apply`` would overwrite a value the user has
    already persisted (typically from a sweep) with a different
    recommendation, the command refuses and lists the affected keys.
    Pass ``--force`` to overwrite anyway.

    ``--sweep`` is the empirical mode: builds a per-tier bracket of
    candidate dflash.* configs (see ``autotune.candidate_configs``),
    cycles the server through each one via ``lucebox restart`` +
    readiness probe, runs ``lucebox profile --level level1`` to capture
    decode_tps, picks the highest-tps cell as winner, and persists it.
    Pre-sweep config.toml is backed up to ``.sweep-backup`` and restored
    on interrupt or failure. ``--sweep`` is mutually exclusive with
    ``--apply`` (sweep applies its own winner) and ``--json`` (sweep
    is interactive). Pass ``--yes`` / ``-y`` to skip the confirmation
    prompt.
    """
    if list_profiles:
        table = Table(title="Autotune profiles")
        table.add_column("name")
        table.add_column("scorer")
        table.add_column("description")
        for name in sorted(autotune_mod.PROFILES):
            p = autotune_mod.PROFILES[name]
            table.add_row(p.name, p.scorer, p.description)
        console.print(table)
        return

    if sweep and (apply_ or json_out):
        console.print(
            "[red]--sweep is mutually exclusive with --apply and --json[/red]"
        )
        raise typer.Exit(code=2)
    if sweep:
        from lucebox.sweep import run_sweep

        rc = run_sweep(console=console, yes=yes, profile=profile)
        if rc != 0:
            raise typer.Exit(code=rc)
        return

    host = from_env()
    runtime = autotune_mod.runtime_from_host(host)
    if json_out:
        print(json.dumps(asdict(runtime), indent=2))
        return

    table = Table(title="Recommended DflashRuntime")
    table.add_column("key")
    table.add_column("recommendation")
    for name in DFLASH_ALLOWLIST:
        table.add_row(name, str(getattr(runtime, name)))
    console.print(table)

    if apply_:
        # Drift guard. config_get with no key returns every reachable
        # dflash.* entry tagged "file" (persisted) or "default" (in-
        # memory only). Compare the persisted value to the
        # recommendation; refuse on any drift unless --force.
        if not force:
            entries = config_mod.config_get()
            drift: list[tuple[str, Any, Any]] = []
            for name in DFLASH_ALLOWLIST:
                key = f"dflash.{name}"
                current, origin = entries.get(key, (None, "default"))
                if origin != "file":
                    continue  # not persisted → nothing to overwrite
                recommended = getattr(runtime, name)
                if current != recommended:
                    drift.append((name, current, recommended))
            if drift:
                console.print(
                    "[yellow]The following config keys already differ from "
                    "the recommendation:[/yellow]"
                )
                width = max(len(name) for name, _, _ in drift)
                for name, current, recommended in drift:
                    console.print(
                        f"  dflash.{name:<{width}}  current={current!r}  "
                        f"recommended={recommended!r}"
                    )
                console.print("[dim]Pass --force to overwrite.[/dim]")
                raise typer.Exit(code=1)
        for name in DFLASH_ALLOWLIST:
            config_set(f"dflash.{name}", getattr(runtime, name))
        console.print(
            f"[green]Applied[/green] {len(DFLASH_ALLOWLIST)} dflash.* keys to "
            f"{config_mod.default_config_path()}"
        )


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
        recommended = autotune_mod.recommend_preset(cfg.host)
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


# ── profile (collapsed wrapper) ────────────────────────────────────────────


@app.command()
def profile(
    level: Annotated[
        str,
        typer.Option("--level", help="Snapshot tier: level0 / level1 / level2 / level3."),
    ] = "level1",
    url: Annotated[
        str,
        typer.Option("--url", help="Server base URL; auto-detects when empty."),
    ] = "",
) -> None:
    """Run a luce-bench snapshot via the running container.

    Thin wrapper that probes the host, picks an output dir under
    $XDG_DATA_HOME/lucebox/profile-snapshots, and exec's
    ``luce-bench snapshot`` inside the running lucebox container. Errors
    clearly when no container is up (hint: ``lucebox start`` first).
    """
    cfg = _load_or_build()
    rc = profile_mod.run_profile(cfg, level=level, url=url or None, console=console)
    if rc != 0:
        raise typer.Exit(code=rc)


# ── smoke ──────────────────────────────────────────────────────────────────


@app.command()
def smoke(
    timeout: Annotated[float, typer.Option(help="Per-request timeout (seconds).")] = 60.0,
    tools: Annotated[
        bool,
        typer.Option("--tools/--no-tools", help="Also require a tool-call response."),
    ] = True,
) -> None:
    """Hit /props + /v1/chat/completions on the running server; report PASS/FAIL."""
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


# ── client launchers ───────────────────────────────────────────────────────


def _detect_server_url(cfg_url: str | None) -> str:
    """Auto-detect a live Lucebox server URL.

    Tries an explicit override first, otherwise probes the standard
    localhost/docker-host base URLs from profile_mod and takes the first
    that answers /health within 1s. Falls back to the first probe candidate
    if nothing answers — lets the client fail with a clearer "server down"
    error than the auto-detect can give.
    """
    if cfg_url:
        return cfg_url
    cfg = _load_or_build()
    bases = profile_mod._server_base_urls(cfg)
    for candidate in bases:
        if profile_mod._json_get(candidate + "/health", timeout_s=1.0):
            return candidate
    console.print(
        f"[yellow]warning:[/yellow] no /health response at {bases[0]} "
        f"— starting client anyway (server may be down)."
    )
    return bases[0]


def _exec_client(launcher_mod, *, url: str | None, model: str, prompt: str | None) -> None:
    """Common entry: probe server, exec the harness client launcher."""
    base_url = _detect_server_url(url)
    try:
        rc = launcher_mod.launch(
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
def claude(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch Claude Code pointed at the running Lucebox server."""
    from harness.clients import claude_code as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def codex(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch Codex pointed at the running Lucebox server."""
    from harness.clients import codex as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def opencode(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch OpenCode pointed at the running Lucebox server."""
    from harness.clients import opencode as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def hermes(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch Hermes Agent pointed at the running Lucebox server."""
    from harness.clients import hermes as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def pi(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch Pi pointed at the running Lucebox server."""
    from harness.clients import pi as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def openclaw(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="One-shot prompt (non-interactive)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(help="Lucebox base URL. Auto-detects localhost / docker host."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model ID to advertise.")] = "luce-dflash",
) -> None:
    """Launch OpenClaw pointed at the running Lucebox server."""
    from harness.clients import openclaw as launcher

    _exec_client(launcher, url=url, model=model, prompt=prompt)


@app.command()
def version() -> None:
    """Print lucebox version."""
    print(__version__)


def _pick_variant_from_driver(driver_major: int, gpu_sm: str) -> config_mod.Variant:  # type: ignore[name-defined]
    """Mirrors lucebox.sh::pick_variant. Centralized so Python and bash agree.

    Kept as a thin wrapper around the LUCEBOX_VARIANT env var because
    the variant tag is picked by the shell wrapper before Python runs;
    this function exists so legacy callers and tests still resolve.
    """
    del driver_major, gpu_sm  # variant pick lives in lucebox.sh
    return os.environ.get("LUCEBOX_VARIANT", "cuda12")


def main() -> None:
    """Module entrypoint — `python -m lucebox`."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
