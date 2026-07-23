"""Build and execute `docker run` argv for the server and download containers.

We shell out to the `docker` CLI rather than using the docker SDK because
(a) the CLI is the user-visible contract — errors look the same whether
issued by lucebox or the user; (b) zero import cost; (c) trivially mockable
via subprocess in tests. Wrap everything in one module so swapping to the
SDK later is a single-file change.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lucebox.types import Config


def _host_facts_env() -> list[tuple[str, str]]:
    """Forward LUCEBOX_HOST_* from the orchestrator's env into the server.

    lucebox.sh's probe_host() exports every host-identity fact (OS,
    kernel, GPU list CSV, CTK version, …) before invoking ``docker run``
    on the orchestrator. The orchestrator inherits them and we pass
    them through verbatim so the server entrypoint can write
    /opt/lucebox-hub/HOST_INFO without re-probing inside the container
    (where /proc and nvidia-smi see the container's view, not the
    rig's). See entrypoint.sh::write_host_info and http_server.cpp's
    /props.host block.
    """
    out: list[tuple[str, str]] = []
    for key, value in sorted(os.environ.items()):
        if key.startswith("LUCEBOX_HOST_"):
            out.append((key, value))
    return out


def _resolve_model_files(cfg: Config) -> tuple[str, str, str]:
    """Return (target_file, draft_file, draft_dir) for DFLASH_TARGET / DFLASH_DRAFT.

    Resolution order — first non-empty wins per field:
        1. cfg.model.target_file / draft_file (explicit override in config.toml)
        2. PRESETS[cfg.model.preset].target_file / draft_file / speculator_dir (registry)
        3. "" (entrypoint autodetect path runs unchanged).

    ``draft_dir`` is a directory name under ``models/draft/`` holding a
    safetensors speculator (e.g. ``laguna-xs2-speculator``). It is only set
    when the preset declares one AND the directory exists on disk; otherwise
    it is empty. When non-empty, docker_run_spec uses it as DFLASH_DRAFT
    (a directory path) instead of the GGUF-file path, allowing the entrypoint
    to discover the safetensors file inside it.

    Imported lazily to avoid the lucebox.types ↔ lucebox.download circular
    import that surfaces when this module is imported from ``__init__``.
    """
    target = cfg.model.target_file
    draft = cfg.model.draft_file
    draft_dir = ""
    if (not target or not draft) and cfg.model.preset:
        from lucebox.download import PRESETS

        pres = PRESETS.get(cfg.model.preset)
        if pres is not None:
            if not target:
                target = pres.target_file
            if not draft and pres.has_draft and pres.draft_file:
                draft = pres.draft_file
            if not draft and pres.speculator_dir:
                spec_path = cfg.models_dir / "draft" / pres.speculator_dir
                if spec_path.is_dir():
                    draft_dir = pres.speculator_dir
    return target, draft, draft_dir


def _runtime_volumes(cfg: Config) -> tuple[tuple[str, str], ...]:
    """Mount models plus $HOME so absolute symlink targets remain valid."""
    home = str(Path.home())
    models = str(cfg.models_dir)
    volumes = [(models, "/opt/lucebox-hub/server/models")]
    if home != models:
        volumes.append((home, home))
    return tuple(volumes)


@dataclass(frozen=True, slots=True)
class DockerRunSpec:
    """Pre-render of a docker-run command. Render via `argv()` or `printable()`."""

    image: str
    name: str
    gpus: bool = True
    detach: bool = False
    remove: bool = True
    port_publish: tuple[int, int] | None = None  # (host, container)
    volumes: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    entrypoint_args: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        out = ["docker", "run"]
        if self.remove:
            out.append("--rm")
        if self.detach:
            out.append("-d")
        out += ["--name", self.name]
        if self.gpus:
            out += ["--gpus", "all"]
        if self.port_publish is not None:
            host, container = self.port_publish
            out += ["-p", f"{host}:{container}"]
        for host_path, container_path in self.volumes:
            out += ["-v", f"{host_path}:{container_path}"]
        for k, v in self.env:
            out += ["-e", f"{k}={v}"]
        out += list(self.extra)
        out.append(self.image)
        out += list(self.entrypoint_args)
        return out

    def printable(self) -> str:
        """Human-readable, one-flag-per-line docker run. Copy-pasteable."""
        argv = self.argv()
        if not argv:
            return ""
        out = argv[0]
        i = 1
        while i < len(argv):
            tok = argv[i]
            out += " \\\n    " + tok
            # Glue value-taking flags onto the same line.
            if tok in {
                "-p",
                "-v",
                "-e",
                "--name",
                "--gpus",
                "--env",
                "--volume",
                "--publish",
                "--entrypoint",
            } and i + 1 < len(argv):
                i += 1
                out += " " + shlex.quote(argv[i])
            i += 1
        return out


# ── server argv from Config ────────────────────────────────────────────────


def server_run_spec(cfg: Config) -> DockerRunSpec:
    """Long-running OpenAI-compatible server. Foreground (systemd manages
    lifecycle), --gpus all, models bind-mounted, DFLASH_* propagated.
    """
    # LUCEBOX_HOST_* first so they ride out front in the rendered argv,
    # making it obvious in `print-run` output what host facts get forwarded.
    env: list[tuple[str, str]] = list(_host_facts_env())
    env += [
        ("DFLASH_BUDGET", str(cfg.dflash.budget)),
        ("DFLASH_MAX_CTX", str(cfg.dflash.max_ctx)),
        ("DFLASH_PREFIX_CACHE_SLOTS", str(cfg.dflash.prefix_cache_slots)),
        ("DFLASH_PREFILL_CACHE_SLOTS", str(cfg.dflash.prefill_cache_slots)),
        ("DFLASH_THINK_MAX", str(cfg.dflash.think_max)),
        ("DFLASH_PORT", "8080"),
    ]
    # Resolve target/draft GGUFs in priority order:
    #   1. cfg.model.target_file / draft_file (explicit override in config.toml)
    #   2. PRESETS[cfg.model.preset].target_file / draft_file / speculator_dir (registry)
    #   3. unset — entrypoint's autodetect path runs unchanged.
    # Container view of the models dir is /opt/lucebox-hub/server/models
    # (see _runtime_volumes); the entrypoint reads DFLASH_TARGET / DFLASH_DRAFT.
    # draft_dir is a subdirectory of models/draft/ holding a safetensors speculator;
    # it takes effect only when draft_file is empty and the directory exists on disk.
    target_file, draft_file, draft_dir = _resolve_model_files(cfg)
    if target_file:
        env.append(("DFLASH_TARGET", f"/opt/lucebox-hub/server/models/{target_file}"))
    if draft_file:
        env.append(("DFLASH_DRAFT", f"/opt/lucebox-hub/server/models/draft/{draft_file}"))
    elif draft_dir:
        env.append(("DFLASH_DRAFT", f"/opt/lucebox-hub/server/models/draft/{draft_dir}"))
    if cfg.dflash.lazy:
        env.append(("DFLASH_LAZY", "1"))
    if cfg.dflash.cache_type_k:
        env.append(("DFLASH_CACHE_TYPE_K", cfg.dflash.cache_type_k))
    if cfg.dflash.cache_type_v:
        env.append(("DFLASH_CACHE_TYPE_V", cfg.dflash.cache_type_v))
    if cfg.dflash.prefill_mode != "off":
        env += [
            ("DFLASH_PREFILL_MODE", cfg.dflash.prefill_mode),
            ("DFLASH_PREFILL_KEEP", str(cfg.dflash.prefill_keep_ratio)),
            ("DFLASH_PREFILL_THRESHOLD", str(cfg.dflash.prefill_threshold)),
        ]
        if cfg.dflash.prefill_drafter:
            env.append(("DFLASH_PREFILL_DRAFTER", cfg.dflash.prefill_drafter))
    # fa_window=0 is the server's own default (full attention); only emit
    # the env when the operator has selected a sparse decode window. The
    # entrypoint mirrors this guard so an unset env reproduces the
    # server's stock behavior.
    if cfg.dflash.fa_window > 0:
        env.append(("DFLASH_FA_WINDOW", str(cfg.dflash.fa_window)))
    # Soft-close ratio: 0.0 is server-side disabled (byte-identical
    # to pre-PR-#326 behavior). Emit only when nonzero to keep the
    # docker env minimal and mirror the entrypoint's `case` guard.
    if cfg.dflash.think_soft_close_min_ratio > 0.0:
        env.append((
            "DFLASH_THINK_SOFT_CLOSE_MIN_RATIO",
            f"{cfg.dflash.think_soft_close_min_ratio:g}",
        ))
    if cfg.dflash.debug_thinking_logits:
        env.append(("DFLASH_DEBUG_THINKING_LOGITS", "1"))

    return DockerRunSpec(
        image=f"{cfg.image}:{cfg.variant}",
        name=cfg.container_name,
        gpus=True,
        remove=True,
        detach=False,
        port_publish=(cfg.port, 8080),
        volumes=_runtime_volumes(cfg),
        env=tuple(env),
    )


# ── subprocess helpers ─────────────────────────────────────────────────────


def docker_pull(image_tag: str) -> int:
    """Pull an image, streaming progress. Returns docker's exit code."""
    return subprocess.call(["docker", "pull", image_tag])
