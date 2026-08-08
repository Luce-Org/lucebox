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
from pathlib import Path, PurePosixPath

from lucebox.types import Config, GpuVendor

_CONTAINER_MODELS = "/opt/lucebox-hub/server/models"
_CONTAINER_RESOLVED_MODELS = "/opt/lucebox-resolved"


@dataclass(frozen=True, slots=True)
class BindMount:
    """One explicit Docker bind mount.

    ``read_only`` is part of the type so sensitive compatibility mounts cannot
    accidentally become writable during argv rendering.
    """

    source: str
    target: str
    read_only: bool = False

    def __post_init__(self) -> None:
        if not Path(self.source).is_absolute():
            raise ValueError(f"bind-mount source must be absolute, got {self.source!r}")
        if not PurePosixPath(self.target).is_absolute():
            raise ValueError(f"bind-mount target must be absolute, got {self.target!r}")
        for label, value in (("source", self.source), ("target", self.target)):
            if "," in value or any(char in value for char in "\n\r\0"):
                raise ValueError(
                    f"bind-mount {label} contains a character unsupported by "
                    f"Docker --mount: {value!r}"
                )

    def argument(self) -> str:
        option = f"type=bind,source={self.source},target={self.target}"
        return f"{option},readonly" if self.read_only else option


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

    The preset registry is imported lazily so constructing a minimal Docker
    spec does not import the Hugging Face download surface unnecessarily.
    """
    target = cfg.model.target_file
    draft = cfg.model.draft_file
    draft_dir = ""
    if (not target or not draft) and cfg.model.preset:
        from lucebox.download import PRESETS, local_artifact_present

        pres = PRESETS.get(cfg.model.preset)
        if pres is not None:
            if not target:
                target = pres.target_file
            if not draft and pres.has_draft and pres.draft_file:
                draft = pres.draft_file
            if not draft and pres.speculator_dir:
                spec_path = cfg.models_dir / "draft" / pres.speculator_dir
                # is_dir() follows valid directory symlinks. It also rejects
                # dangling links and links to files, which cannot satisfy the
                # speculator-directory contract.
                required = tuple(spec_path / filename for filename in pres.speculator_files)
                if spec_path.is_dir() and required and all(
                    local_artifact_present(path) for path in required
                ):
                    draft_dir = pres.speculator_dir
    return target, draft, draft_dir


def _selected_model_architecture(cfg: Config) -> str:
    """Return the engine architecture declared by the active preset."""
    if not cfg.model.preset:
        return ""
    from lucebox.download import PRESETS

    preset = PRESETS.get(cfg.model.preset)
    return preset.architecture if preset is not None else ""


def _runtime_volumes(cfg: Config) -> tuple[BindMount, ...]:
    """Mount only the writable application data needed by the server."""
    models = str(cfg.models_dir.absolute())
    config_home = Path(os.environ.get("LUCEBOX_HOME") or Path.home() / ".lucebox").absolute()
    return (
        BindMount(models, _CONTAINER_MODELS),
        BindMount(str(config_home), str(config_home)),
    )


def _placement_env(cfg: Config) -> list[tuple[str, str]]:
    """Translate the portable placement profile to the entrypoint contract."""
    placement = cfg.placement
    env: list[tuple[str, str]] = [("DFLASH_PLACEMENT_MODE", placement.mode)]
    if placement.target_device:
        env.append(("DFLASH_TARGET_DEVICE", placement.target_device))
    if placement.target_devices:
        env.append(("DFLASH_TARGET_DEVICES", ",".join(placement.target_devices)))
        env.append(
            (
                "DFLASH_TARGET_LAYER_SPLIT",
                ",".join(f"{weight:g}" for weight in placement.target_layer_split),
            )
        )
    if placement.draft_device:
        env.append(("DFLASH_DRAFT_DEVICE", placement.draft_device))
    if placement.remote_draft:
        env.append(("DFLASH_REMOTE_DRAFT", "1"))
    if placement.remote_target_shard:
        env.append(("DFLASH_REMOTE_TARGET_SHARD", "1"))
    if placement.peer_access:
        env.append(("DFLASH_PEER_ACCESS", "1"))
    if placement.remote_expert_device:
        env.append(("DFLASH_REMOTE_EXPERT_DEVICE", placement.remote_expert_device))
    return env


def _validate_model_relative_path(value: str, field: str) -> PurePosixPath:
    """Validate a config-provided path intended to live below models_dir."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value in {"", "."}:
        raise ValueError(f"{field} must be a path below models_dir, got {value!r}")
    return relative


def _selected_model_path(
    cfg: Config,
    value: str,
    *,
    field: str,
    role: str,
    under_draft: bool,
    directory: bool,
    mounts: list[BindMount],
) -> str:
    """Return the container path for one explicitly selected model artifact.

    Normal files are already covered by the models-directory bind mount.  For
    a symlink, bind only its resolved file (or the selected directory for a
    speculator) into a dedicated read-only location. This preserves symlinked
    model workflows without exposing the user's home or adjacent model files.
    """
    relative = _validate_model_relative_path(value, field)
    base = cfg.models_dir / "draft" if under_draft else cfg.models_dir
    host_path = base.joinpath(*relative.parts)
    container_base = PurePosixPath(_CONTAINER_MODELS)
    if under_draft:
        container_base /= "draft"
    canonical = str(container_base.joinpath(*relative.parts))
    resolved = host_path.resolve(strict=False)
    # A symlink in models_dir or one of its ancestors is covered by the root bind.
    # Resolve that root before comparing so only symlinks *within* the selected
    # model path need a separate narrow mount.
    expected = cfg.models_dir.resolve(strict=False)
    if under_draft:
        expected /= "draft"
    expected = expected.joinpath(*relative.parts)
    if resolved == expected:
        return canonical

    mount_target = f"{_CONTAINER_RESOLVED_MODELS}/{role}"
    if directory:
        mounts.append(BindMount(str(resolved), mount_target, read_only=True))
        return mount_target

    container_path = f"{mount_target}/{resolved.name}"
    mounts.append(BindMount(str(resolved), container_path, read_only=True))
    return container_path


@dataclass(frozen=True, slots=True)
class DockerRunSpec:
    """Pre-render of a docker-run command. Render via `argv()` or `printable()`."""

    image: str
    name: str
    gpus: bool = True
    gpu_vendor: GpuVendor = "nvidia"
    detach: bool = False
    remove: bool = True
    port_publish: tuple[int, int] | None = None  # (host, container)
    volumes: tuple[BindMount, ...] = ()
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
        if self.gpus and self.gpu_vendor == "nvidia":
            out += ["--gpus", "all"]
        elif self.gpus and self.gpu_vendor == "amd":
            out += [
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video",
                "--group-add",
                "render",
                "--security-opt",
                "seccomp=unconfined",
            ]
        if self.port_publish is not None:
            host, container = self.port_publish
            out += ["-p", f"{host}:{container}"]
        for mount in self.volumes:
            out += ["--mount", mount.argument()]
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
                "--mount",
                "--publish",
                "--entrypoint",
                "--device",
                "--group-add",
                "--security-opt",
            } and i + 1 < len(argv):
                i += 1
                out += " " + shlex.quote(argv[i])
            i += 1
        return out


# ── server argv from Config ────────────────────────────────────────────────


def server_run_spec(cfg: Config) -> DockerRunSpec:
    """Long-running OpenAI-compatible server. Foreground (systemd manages
    lifecycle), vendor-specific GPU devices, models bind-mounted, DFLASH_* propagated.
    """
    if cfg.placement.requires_hybrid_runtime:
        raise ValueError(
            "cross-vendor placement requires the paired native Lucebox runtime; "
            "a single-backend Docker image cannot launch it"
        )

    # LUCEBOX_HOST_* first so they ride out front in the rendered argv,
    # making it obvious in `print-run` output what host facts get forwarded.
    env: list[tuple[str, str]] = list(_host_facts_env())
    config_home = str(Path(os.environ.get("LUCEBOX_HOME") or Path.home() / ".lucebox").absolute())
    env += [
        ("LUCEBOX_HOME", config_home),
        ("DFLASH_BUDGET", str(cfg.dflash.budget)),
        ("DFLASH_MAX_CTX", str(cfg.dflash.max_ctx)),
        ("DFLASH_PREFIX_CACHE_SLOTS", str(cfg.dflash.prefix_cache_slots)),
        ("DFLASH_PREFILL_CACHE_SLOTS", str(cfg.dflash.prefill_cache_slots)),
        ("DFLASH_PORT", "8080"),
    ]
    if cfg.dflash.think_max is not None:
        env.append(("DFLASH_THINK_MAX", str(cfg.dflash.think_max)))
    env += _placement_env(cfg)
    # Keep the API model catalog aligned with the preset selected by the host
    # CLI. Client connectors use this id for explicit model selection and
    # clients such as Codex also discover it through ``/v1/models``.
    if cfg.model.preset:
        env.append(("DFLASH_MODEL_NAME", cfg.model.preset))
    # Resolve target/draft GGUFs in priority order:
    #   1. cfg.model.target_file / draft_file (explicit override in config.toml)
    #   2. PRESETS[cfg.model.preset].target_file / draft_file / speculator_dir (registry)
    #   3. unset — entrypoint's autodetect path runs unchanged.
    # Container view of the models dir is /opt/lucebox-hub/server/models
    # (see _runtime_volumes); the entrypoint reads DFLASH_TARGET / DFLASH_DRAFT.
    # draft_dir is a subdirectory of models/draft/ holding a safetensors speculator;
    # it takes effect only when draft_file is empty and the directory exists on disk.
    target_file, draft_file, draft_dir = _resolve_model_files(cfg)
    model_architecture = _selected_model_architecture(cfg)
    volumes = list(_runtime_volumes(cfg))
    if target_file:
        target_path = _selected_model_path(
            cfg,
            target_file,
            field="model.target_file",
            role="target",
            under_draft=False,
            directory=False,
            mounts=volumes,
        )
        env.append(("DFLASH_TARGET", target_path))
    if model_architecture == "deepseek4":
        # DeepSeek's DSpark path is architecture-specific. Passing its GGUF as
        # generic --draft is explicitly inert for deepseek4; the backend reads
        # these two variables instead. Keep generic draft discovery disabled
        # so a stale Qwen/Gemma draft cannot be attached by the entrypoint.
        env.append(("DFLASH_DRAFT", "/opt/lucebox-hub/server/models/.lucebox-no-draft"))
        if cfg.dflash.speculative_decode and draft_file:
            draft_path = _selected_model_path(
                cfg,
                draft_file,
                field="model.draft_file",
                role="ds4-draft",
                under_draft=True,
                directory=False,
                mounts=volumes,
            )
            env += [
                ("DFLASH_DS4_SPEC", "1"),
                ("DFLASH_DS4_DRAFT", draft_path),
            ]
    elif not cfg.dflash.speculative_decode:
        env.append(("DFLASH_DRAFT", "/opt/lucebox-hub/server/models/.lucebox-no-draft"))
    elif draft_file:
        draft_path = _selected_model_path(
            cfg,
            draft_file,
            field="model.draft_file",
            role="draft",
            under_draft=True,
            directory=False,
            mounts=volumes,
        )
        env.append(("DFLASH_DRAFT", draft_path))
    elif draft_dir:
        draft_path = _selected_model_path(
            cfg,
            draft_dir,
            field="preset.speculator_dir",
            role="draft-dir",
            under_draft=True,
            directory=True,
            mounts=volumes,
        )
        env.append(("DFLASH_DRAFT", draft_path))
    elif cfg.model.preset:
        # An active target-only preset is an explicit choice, not permission
        # for entrypoint.sh to scan models/draft and attach an unrelated stale
        # draft left by a previously active model. The entrypoint treats this
        # guaranteed-missing path as "run target-only".
        env.append(("DFLASH_DRAFT", "/opt/lucebox-hub/server/models/.lucebox-no-draft"))
    if cfg.dflash.lazy:
        env.append(("DFLASH_LAZY", "1"))
    if cfg.dflash.cache_type_k:
        env.append(("DFLASH_CACHE_TYPE_K", cfg.dflash.cache_type_k))
    if cfg.dflash.cache_type_v:
        env.append(("DFLASH_CACHE_TYPE_V", cfg.dflash.cache_type_v))
    if cfg.dflash.prefill_drafter:
        env.append(("DFLASH_PREFILL_DRAFTER", cfg.dflash.prefill_drafter))
    if cfg.dflash.prefill_mode != "off":
        env += [
            ("DFLASH_PREFILL_MODE", cfg.dflash.prefill_mode),
            ("DFLASH_PREFILL_KEEP", str(cfg.dflash.prefill_keep_ratio)),
            ("DFLASH_PREFILL_THRESHOLD", str(cfg.dflash.prefill_threshold)),
        ]
    if cfg.dflash.kvflash != "off":
        env += [
            ("DFLASH_KVFLASH", cfg.dflash.kvflash),
            ("DFLASH_KVFLASH_POLICY", cfg.dflash.kvflash_policy),
            ("DFLASH_KVFLASH_TAU", str(cfg.dflash.kvflash_tau)),
        ]
    if cfg.dflash.spark:
        env.append(("DFLASH_SPARK", "1"))
        if cfg.dflash.spark_vram_gb > 0.0:
            env.append(("DFLASH_SPARK_VRAM_GB", f"{cfg.dflash.spark_vram_gb:g}"))
    if cfg.dflash.ds4_prefill != "exact":
        env.append(("DFLASH_DS4_PREFILL", cfg.dflash.ds4_prefill))
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
        env.append(
            (
                "DFLASH_THINK_SOFT_CLOSE_MIN_RATIO",
                f"{cfg.dflash.think_soft_close_min_ratio:g}",
            )
        )
    if cfg.dflash.debug_thinking_logits:
        env.append(("DFLASH_DEBUG_THINKING_LOGITS", "1"))

    # The chosen image variant is the runtime contract. This matters on a
    # heterogeneous RTX + Strix host: the probe records both vendors, while an
    # explicit ``variant=rocm`` must still receive AMD device flags. Unknown
    # custom variants fall back to the probe's selected/default vendor.
    variant_lower = cfg.variant.lower()
    if "rocm" in variant_lower:
        gpu_vendor: GpuVendor = "amd"
    elif "cuda" in variant_lower:
        gpu_vendor = "nvidia"
    else:
        gpu_vendor = cfg.host.gpu_vendor if cfg.host.gpu_vendor != "none" else "nvidia"

    # Keep execution on the same primary device used by host detection and
    # automatic tuning. This matters on an R9700 + Strix build where ROCm may
    # enumerate the integrated GPU before the larger discrete card. Explicit
    # CUDA/ROCR/HIP visibility supplied by an advanced user remains authoritative.
    placement_is_explicit = bool(cfg.placement.target_device or cfg.placement.target_devices)
    if gpu_vendor == "amd" and not placement_is_explicit:
        rocr_visible = os.environ.get("LUCEBOX_HOST_ROCR_VISIBLE_DEVICES", "").strip()
        hip_visible = os.environ.get("LUCEBOX_HOST_HIP_VISIBLE_DEVICES", "").strip()
        if rocr_visible:
            env.append(("ROCR_VISIBLE_DEVICES", rocr_visible))
        elif hip_visible:
            env.append(("HIP_VISIBLE_DEVICES", hip_visible))
    elif gpu_vendor == "nvidia" and not placement_is_explicit:
        visible = os.environ.get("LUCEBOX_HOST_CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            env.append(("CUDA_VISIBLE_DEVICES", visible))

    return DockerRunSpec(
        image=f"{cfg.image}:{cfg.variant}",
        name=cfg.container_name,
        gpus=True,
        gpu_vendor=gpu_vendor,
        remove=True,
        detach=False,
        port_publish=(cfg.port, 8080),
        volumes=tuple(volumes),
        env=tuple(env),
    )


# ── subprocess helpers ─────────────────────────────────────────────────────


def docker_pull(image_tag: str) -> int:
    """Pull an image, streaming progress. Returns docker's exit code."""
    return subprocess.call(["docker", "pull", image_tag])
