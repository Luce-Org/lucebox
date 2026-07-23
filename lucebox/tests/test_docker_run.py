"""Tests for the docker-run serve-argv builder.

This is the core's whole job: turn a Config into the exact `docker run`
command (and DFLASH_* env) that launches the server. The argv contract is
what `lucebox serve` / the systemd unit / `print-run` all consume, so it is
pinned field-by-field here rather than only smoke-tested.
"""

from __future__ import annotations

from pathlib import Path

from lucebox.download import PRESETS
from lucebox.types import Config, DflashRuntime, ModelMeta

from lucebox import docker_run


def _env(spec) -> dict[str, str]:
    return dict(spec.env)


# ── DockerRunSpec.argv ───────────────────────────────────────────────────────


def test_argv_minimal_defaults() -> None:
    spec = docker_run.DockerRunSpec(image="img:tag", name="box")
    argv = spec.argv()
    assert argv[:2] == ["docker", "run"]
    assert "--rm" in argv  # remove defaults True
    assert ["--name", "box"] == argv[argv.index("--name") : argv.index("--name") + 2]
    assert ["--gpus", "all"] == argv[argv.index("--gpus") : argv.index("--gpus") + 2]
    # image is the last positional (no entrypoint_args here)
    assert argv[-1] == "img:tag"
    assert "-d" not in argv  # detach defaults False


def test_argv_flags_and_ordering() -> None:
    spec = docker_run.DockerRunSpec(
        image="img:tag",
        name="box",
        gpus=False,
        detach=True,
        remove=False,
        port_publish=(8080, 8080),
        volumes=(("/host/models", "/opt/lucebox-hub/server/models"),),
        env=(("DFLASH_BUDGET", "22"),),
        entrypoint_args=("serve",),
        extra=("--shm-size", "1g"),
    )
    argv = spec.argv()
    assert "--rm" not in argv  # remove=False
    assert "-d" in argv  # detach
    assert "--gpus" not in argv  # gpus=False
    assert ["-p", "8080:8080"] == argv[argv.index("-p") : argv.index("-p") + 2]
    assert ["-v", "/host/models:/opt/lucebox-hub/server/models"] == argv[
        argv.index("-v") : argv.index("-v") + 2
    ]
    assert ["-e", "DFLASH_BUDGET=22"] == argv[argv.index("-e") : argv.index("-e") + 2]
    # extra flags precede the image; entrypoint_args follow it.
    assert argv[-1] == "serve"
    assert argv[-2] == "img:tag"
    assert argv.index("--shm-size") < argv.index("img:tag")


def test_printable_glues_value_taking_flags() -> None:
    spec = docker_run.DockerRunSpec(
        image="img:tag",
        name="box",
        port_publish=(8080, 8080),
        env=(("K", "v"),),
    )
    out = spec.printable()
    # one flag per line, continued with backslash-newline
    assert out.startswith("docker \\\n    run")
    # value-taking flags keep their value on the same line
    assert "--name box" in out
    assert "--gpus all" in out
    assert "-p 8080:8080" in out
    assert "-e K=v" in out


# ── _runtime_volumes ─────────────────────────────────────────────────────────


def test_runtime_volumes_mounts_models_and_home(tmp_path: Path) -> None:
    cfg = Config(models_dir=tmp_path / "models")
    vols = docker_run._runtime_volumes(cfg)
    assert (str(tmp_path / "models"), "/opt/lucebox-hub/server/models") in vols
    # $HOME is also mounted so absolute symlink targets resolve in-container.
    assert any(host == str(Path.home()) for host, _ in vols)


def test_runtime_volumes_dedupes_when_models_is_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    cfg = Config(models_dir=tmp_path)
    vols = docker_run._runtime_volumes(cfg)
    # models_dir == home → only the models mount, no duplicate home mount.
    assert len(vols) == 1


# ── _resolve_model_files ─────────────────────────────────────────────────────


def test_resolve_model_files_explicit_override_wins(tmp_path: Path) -> None:
    cfg = Config(
        models_dir=tmp_path,
        model=ModelMeta(preset="qwen3.6-27b", target_file="custom.gguf", draft_file="d.gguf"),
    )
    target, draft, draft_dir = docker_run._resolve_model_files(cfg)
    assert target == "custom.gguf"
    assert draft == "d.gguf"
    assert draft_dir == ""


def test_resolve_model_files_falls_back_to_preset_registry(tmp_path: Path) -> None:
    pres = PRESETS["qwen3.6-27b"]
    cfg = Config(models_dir=tmp_path, model=ModelMeta(preset="qwen3.6-27b"))
    target, draft, draft_dir = docker_run._resolve_model_files(cfg)
    assert target == pres.target_file
    assert draft == (pres.draft_file or "")
    assert draft_dir == ""  # no speculator dir on disk


def test_resolve_model_files_no_preset_no_override(tmp_path: Path) -> None:
    cfg = Config(models_dir=tmp_path)  # ModelMeta() defaults: all empty
    assert docker_run._resolve_model_files(cfg) == ("", "", "")


# ── server_run_spec ──────────────────────────────────────────────────────────


def test_server_run_spec_top_level_shape(tmp_path: Path) -> None:
    cfg = Config(
        image="ghcr.io/x/lucebox-hub",
        variant="cuda12",
        container_name="lucebox",
        port=9000,
        models_dir=tmp_path,
    )
    spec = docker_run.server_run_spec(cfg)
    assert spec.image == "ghcr.io/x/lucebox-hub:cuda12"
    assert spec.name == "lucebox"
    assert spec.gpus is True
    assert spec.remove is True
    assert spec.detach is False
    assert spec.port_publish == (9000, 8080)
    assert (str(tmp_path), "/opt/lucebox-hub/server/models") in spec.volumes


def test_server_run_spec_always_emits_core_dflash_env(tmp_path: Path) -> None:
    cfg = Config(models_dir=tmp_path, dflash=DflashRuntime(budget=22, max_ctx=32768))
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_BUDGET"] == "22"
    assert env["DFLASH_MAX_CTX"] == "32768"
    assert env["DFLASH_PREFIX_CACHE_SLOTS"] == "0"
    assert env["DFLASH_PREFILL_CACHE_SLOTS"] == "0"
    assert env["DFLASH_THINK_MAX"] == "15488"
    assert env["DFLASH_PORT"] == "8080"


def test_server_run_spec_optional_env_off_by_default(tmp_path: Path) -> None:
    env = _env(docker_run.server_run_spec(Config(models_dir=tmp_path)))
    for absent in (
        "DFLASH_LAZY",
        "DFLASH_CACHE_TYPE_K",
        "DFLASH_CACHE_TYPE_V",
        "DFLASH_PREFILL_MODE",
        "DFLASH_FA_WINDOW",
        "DFLASH_THINK_SOFT_CLOSE_MIN_RATIO",
        "DFLASH_DEBUG_THINKING_LOGITS",
        "DFLASH_TARGET",
        "DFLASH_DRAFT",
    ):
        assert absent not in env


def test_server_run_spec_optional_env_emitted_when_set(tmp_path: Path) -> None:
    cfg = Config(
        models_dir=tmp_path,
        dflash=DflashRuntime(
            lazy=True,
            cache_type_k="tq3_0",
            cache_type_v="tq3_0",
            prefill_mode="auto",
            prefill_keep_ratio=0.1,
            prefill_threshold=20000,
            prefill_drafter="drafter.gguf",
            fa_window=512,
            think_soft_close_min_ratio=0.5,
            debug_thinking_logits=True,
        ),
    )
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_LAZY"] == "1"
    assert env["DFLASH_CACHE_TYPE_K"] == "tq3_0"
    assert env["DFLASH_CACHE_TYPE_V"] == "tq3_0"
    assert env["DFLASH_PREFILL_MODE"] == "auto"
    assert env["DFLASH_PREFILL_KEEP"] == "0.1"
    assert env["DFLASH_PREFILL_THRESHOLD"] == "20000"
    assert env["DFLASH_PREFILL_DRAFTER"] == "drafter.gguf"
    assert env["DFLASH_FA_WINDOW"] == "512"
    assert env["DFLASH_THINK_SOFT_CLOSE_MIN_RATIO"] == "0.5"
    assert env["DFLASH_DEBUG_THINKING_LOGITS"] == "1"


def test_server_run_spec_resolves_target_and_draft_paths(tmp_path: Path) -> None:
    pres = PRESETS["qwen3.6-27b"]
    cfg = Config(models_dir=tmp_path, model=ModelMeta(preset="qwen3.6-27b"))
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_TARGET"] == f"/opt/lucebox-hub/server/models/{pres.target_file}"
    if pres.draft_file:
        assert env["DFLASH_DRAFT"] == (
            f"/opt/lucebox-hub/server/models/draft/{pres.draft_file}"
        )


def test_server_run_spec_forwards_host_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUCEBOX_HOST_OS_PRETTY", "Ubuntu 22.04")
    monkeypatch.setenv("LUCEBOX_HOST_GPU_NAME", "RTX 5090")
    env = _env(docker_run.server_run_spec(Config(models_dir=tmp_path)))
    assert env["LUCEBOX_HOST_OS_PRETTY"] == "Ubuntu 22.04"
    assert env["LUCEBOX_HOST_GPU_NAME"] == "RTX 5090"


def test_large_preset_serves_at_safe_default_ctx(tmp_path: Path) -> None:
    """Regression guard for the preset-cap analysis (#5).

    Activating a preset writes only [model], never [dflash], so a loaded
    Config keeps the conservative DflashRuntime() floor (max_ctx=16384).
    The VRAM-tier heuristic's higher caps only apply via `autotune --apply`
    (which threads cfg.model.preset and is a separate PR). This test pins
    that a large preset does NOT silently serve at a high, OOM-prone ctx
    through the default serve path.
    """
    cfg = Config(models_dir=tmp_path, model=ModelMeta(preset="qwen3.6-27b"))
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_MAX_CTX"] == "16384"


# ── docker_pull ──────────────────────────────────────────────────────────────


def test_docker_pull_shells_out_and_returns_code(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_call(argv: list[str]) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(docker_run.subprocess, "call", fake_call)
    rc = docker_run.docker_pull("img:tag")
    assert rc == 7
    assert seen["argv"] == ["docker", "pull", "img:tag"]
