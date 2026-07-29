"""Tests for the docker-run serve-argv builder.

This is the core's whole job: turn a Config into the exact `docker run`
command (and DFLASH_* env) that launches the server. The argv contract is
what `lucebox serve` / the systemd unit / `print-run` all consume, so it is
pinned field-by-field here rather than only smoke-tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lucebox.download import PRESETS
from lucebox.types import Config, DflashRuntime, HostFacts, ModelMeta

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
        volumes=(
            docker_run.BindMount("/host/models", "/opt/lucebox-hub/server/models"),
        ),
        env=(("DFLASH_BUDGET", "22"),),
        entrypoint_args=("serve",),
        extra=("--shm-size", "1g"),
    )
    argv = spec.argv()
    assert "--rm" not in argv  # remove=False
    assert "-d" in argv  # detach
    assert "--gpus" not in argv  # gpus=False
    assert ["-p", "8080:8080"] == argv[argv.index("-p") : argv.index("-p") + 2]
    mount_arg = "type=bind,source=/host/models,target=/opt/lucebox-hub/server/models"
    assert ["--mount", mount_arg] == argv[
        argv.index("--mount") : argv.index("--mount") + 2
    ]
    assert ["-e", "DFLASH_BUDGET=22"] == argv[argv.index("-e") : argv.index("-e") + 2]
    # extra flags precede the image; entrypoint_args follow it.
    assert argv[-1] == "serve"
    assert argv[-2] == "img:tag"
    assert argv.index("--shm-size") < argv.index("img:tag")


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("relative", "/container"),
        ("/host", "relative"),
        ("/host,comma", "/container"),
        ("/host", "/container\nnewline"),
    ],
)
def test_bind_mount_rejects_ambiguous_paths(source: str, target: str) -> None:
    with pytest.raises(ValueError, match="bind-mount"):
        docker_run.BindMount(source, target)


def test_argv_amd_uses_rocm_device_contract() -> None:
    spec = docker_run.DockerRunSpec(image="img:rocm", name="box", gpu_vendor="amd")
    argv = spec.argv()
    assert "--gpus" not in argv
    assert ["--device", "/dev/kfd"] == argv[
        argv.index("--device") : argv.index("--device") + 2
    ]
    assert "/dev/dri" in argv
    assert ["--group-add", "video"] == argv[
        argv.index("--group-add") : argv.index("--group-add") + 2
    ]
    assert "render" in argv
    assert ["--security-opt", "seccomp=unconfined"] == argv[
        argv.index("--security-opt") : argv.index("--security-opt") + 2
    ]


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


def test_runtime_volumes_mounts_only_models_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("LUCEBOX_HOME", raising=False)
    cfg = Config(models_dir=tmp_path / "models")
    vols = docker_run._runtime_volumes(cfg)
    assert docker_run.BindMount(
        str(tmp_path / "models"), "/opt/lucebox-hub/server/models"
    ) in vols
    assert docker_run.BindMount(
        str(home / ".lucebox"), str(home / ".lucebox")
    ) in vols
    assert all(mount.source != str(home) for mount in vols)


def test_runtime_volumes_dedupes_when_models_is_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("LUCEBOX_HOME", raising=False)
    cfg = Config(models_dir=tmp_path)
    vols = docker_run._runtime_volumes(cfg)
    # models_dir is mounted at the image's canonical path, while config keeps
    # its same-path mount. The parent home directory itself is never exposed.
    assert len(vols) == 2
    assert docker_run.BindMount(
        str(tmp_path / ".lucebox"), str(tmp_path / ".lucebox")
    ) in vols
    assert all(
        mount.source != str(tmp_path) or mount.target != str(tmp_path)
        for mount in vols
    )


def test_runtime_volumes_mounts_custom_config_home(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    config_home = tmp_path / "config-outside-home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("LUCEBOX_HOME", str(config_home))

    vols = docker_run._runtime_volumes(Config(models_dir=tmp_path / "models"))

    assert docker_run.BindMount(str(config_home), str(config_home)) in vols


def test_empty_lucebox_home_uses_default(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("LUCEBOX_HOME", "")

    cfg = Config(models_dir=tmp_path / "models")
    spec = docker_run.server_run_spec(cfg)

    assert _env(spec)["LUCEBOX_HOME"] == str(home / ".lucebox")
    assert str(Path.cwd()) != _env(spec)["LUCEBOX_HOME"]


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


@pytest.mark.parametrize("invalid_target", ["file", "missing"])
def test_resolve_model_files_ignores_invalid_speculator_symlink(
    invalid_target: str, tmp_path: Path
) -> None:
    draft_root = tmp_path / "draft"
    draft_root.mkdir()
    target = tmp_path / "external-speculator"
    if invalid_target == "file":
        target.write_bytes(b"not a directory")
    (draft_root / "laguna-xs2-speculator").symlink_to(
        target, target_is_directory=invalid_target == "missing"
    )
    cfg = Config(models_dir=tmp_path, model=ModelMeta(preset="laguna-xs.2"))

    _, _, draft_dir = docker_run._resolve_model_files(cfg)

    assert draft_dir == ""


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
    assert docker_run.BindMount(
        str(tmp_path), "/opt/lucebox-hub/server/models"
    ) in spec.volumes


def test_server_run_spec_mounts_selected_symlink_target_narrowly_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    models = tmp_path / "models"
    external = home / "model-cache"
    models.mkdir()
    external.mkdir(parents=True)
    target = external / "target.gguf"
    target.write_bytes(b"model")
    (models / "selected.gguf").symlink_to(target)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    spec = docker_run.server_run_spec(
        Config(models_dir=models, model=ModelMeta(target_file="selected.gguf"))
    )

    assert docker_run.BindMount(
        str(target),
        "/opt/lucebox-resolved/target/target.gguf",
        read_only=True,
    ) in spec.volumes
    assert _env(spec)["DFLASH_TARGET"] == "/opt/lucebox-resolved/target/target.gguf"
    assert all(mount.source != str(home) for mount in spec.volumes)
    argv = spec.argv()
    mount_args = [argv[i + 1] for i, token in enumerate(argv) if token == "--mount"]
    assert any(
        "target=/opt/lucebox-resolved/target/target.gguf,readonly" in argument
        for argument in mount_args
    )


def test_server_run_spec_resolves_symlinked_model_parent(tmp_path: Path) -> None:
    models = tmp_path / "models"
    external = tmp_path / "external"
    models.mkdir()
    external.mkdir()
    target = external / "nested.gguf"
    target.write_bytes(b"model")
    (models / "selected").symlink_to(external, target_is_directory=True)

    spec = docker_run.server_run_spec(
        Config(models_dir=models, model=ModelMeta(target_file="selected/nested.gguf"))
    )

    assert docker_run.BindMount(
        str(target),
        "/opt/lucebox-resolved/target/nested.gguf",
        read_only=True,
    ) in spec.volumes
    assert _env(spec)["DFLASH_TARGET"] == "/opt/lucebox-resolved/target/nested.gguf"


def test_server_run_spec_does_not_remount_file_for_symlinked_models_dir(
    tmp_path: Path,
) -> None:
    actual_models = tmp_path / "actual-models"
    models = tmp_path / "models"
    actual_models.mkdir()
    (actual_models / "target.gguf").write_bytes(b"model")
    models.symlink_to(actual_models, target_is_directory=True)

    spec = docker_run.server_run_spec(
        Config(models_dir=models, model=ModelMeta(target_file="target.gguf"))
    )

    assert _env(spec)["DFLASH_TARGET"] == "/opt/lucebox-hub/server/models/target.gguf"
    assert all(
        mount.target != "/opt/lucebox-resolved/target/target.gguf"
        for mount in spec.volumes
    )


def test_server_run_spec_mounts_symlinked_speculator_directory_read_only(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    draft_root = models / "draft"
    external = tmp_path / "external-speculator"
    draft_root.mkdir(parents=True)
    external.mkdir()
    (external / "model.safetensors").write_bytes(b"speculator")
    (draft_root / "laguna-xs2-speculator").symlink_to(
        external, target_is_directory=True
    )

    spec = docker_run.server_run_spec(
        Config(models_dir=models, model=ModelMeta(preset="laguna-xs.2"))
    )

    assert docker_run.BindMount(
        str(external),
        "/opt/lucebox-resolved/draft-dir",
        read_only=True,
    ) in spec.volumes
    assert _env(spec)["DFLASH_DRAFT"] == "/opt/lucebox-resolved/draft-dir"


def test_server_run_spec_rejects_model_path_traversal(tmp_path: Path) -> None:
    cfg = Config(models_dir=tmp_path, model=ModelMeta(target_file="../secret.gguf"))

    with pytest.raises(ValueError, match="below models_dir"):
        docker_run.server_run_spec(cfg)


def test_server_run_spec_rocm_uses_amd_devices_on_heterogeneous_host(tmp_path: Path) -> None:
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        # The generic probe may select NVIDIA by default on RTX + Strix. The
        # explicit image variant must still control Docker's device contract.
        host=HostFacts(gpu_vendor="nvidia", has_nvidia_gpu=True, has_amd_gpu=True),
    )
    spec = docker_run.server_run_spec(cfg)
    assert spec.gpu_vendor == "amd"
    argv = spec.argv()
    assert "--gpus" not in argv
    assert "/dev/kfd" in argv
    assert "/dev/dri" in argv


def test_server_run_spec_always_emits_core_dflash_env(tmp_path: Path) -> None:
    cfg = Config(models_dir=tmp_path, dflash=DflashRuntime(budget=22, max_ctx=32768))
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_BUDGET"] == "22"
    assert env["DFLASH_MAX_CTX"] == "32768"
    assert env["DFLASH_PREFIX_CACHE_SLOTS"] == "0"
    assert env["DFLASH_PREFILL_CACHE_SLOTS"] == "0"
    assert env["DFLASH_THINK_MAX"] == "15488"
    assert env["DFLASH_PORT"] == "8080"
    assert env["LUCEBOX_HOME"]


def test_server_run_spec_target_only_preset_disables_stale_draft(tmp_path: Path) -> None:
    cfg = Config(
        models_dir=tmp_path,
        model=ModelMeta(preset="qwen3.6-moe"),
    )

    env = _env(docker_run.server_run_spec(cfg))

    assert env["DFLASH_TARGET"].endswith("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    assert env["DFLASH_DRAFT"].endswith("/.lucebox-no-draft")
    assert env["DFLASH_MODEL_NAME"] == "qwen3.6-moe"


def test_server_run_spec_optional_env_off_by_default(tmp_path: Path) -> None:
    env = _env(docker_run.server_run_spec(Config(models_dir=tmp_path)))
    for absent in (
        "DFLASH_LAZY",
        "DFLASH_CACHE_TYPE_K",
        "DFLASH_CACHE_TYPE_V",
        "DFLASH_PREFILL_MODE",
        "DFLASH_PREFILL_DRAFTER",
        "DFLASH_KVFLASH",
        "DFLASH_KVFLASH_POLICY",
        "DFLASH_KVFLASH_TAU",
        "DFLASH_SPARK",
        "DFLASH_SPARK_VRAM_GB",
        "DFLASH_FA_WINDOW",
        "DFLASH_THINK_SOFT_CLOSE_MIN_RATIO",
        "DFLASH_DEBUG_THINKING_LOGITS",
        "DFLASH_TARGET",
        "DFLASH_DRAFT",
        "DFLASH_MODEL_NAME",
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
            kvflash="auto",
            kvflash_policy="qk",
            kvflash_tau=96,
            spark=True,
            spark_vram_gb=14.5,
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
    assert env["DFLASH_KVFLASH"] == "auto"
    assert env["DFLASH_KVFLASH_POLICY"] == "qk"
    assert env["DFLASH_KVFLASH_TAU"] == "96"
    assert env["DFLASH_SPARK"] == "1"
    assert env["DFLASH_SPARK_VRAM_GB"] == "14.5"
    assert env["DFLASH_THINK_SOFT_CLOSE_MIN_RATIO"] == "0.5"
    assert env["DFLASH_DEBUG_THINKING_LOGITS"] == "1"


def test_server_run_spec_rejects_kvflash_with_fa_window() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DflashRuntime(kvflash="auto", fa_window=512)


def test_server_run_spec_forwards_primary_rocm_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LUCEBOX_HOST_ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("LUCEBOX_HOST_HIP_VISIBLE_DEVICES", raising=False)
    cfg = Config(variant="rocm", models_dir=tmp_path)

    env = _env(docker_run.server_run_spec(cfg))

    assert env["ROCR_VISIBLE_DEVICES"] == "1"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_server_run_spec_resolves_target_and_draft_paths(tmp_path: Path) -> None:
    pres = PRESETS["qwen3.6-27b"]
    cfg = Config(models_dir=tmp_path, model=ModelMeta(preset="qwen3.6-27b"))
    env = _env(docker_run.server_run_spec(cfg))
    assert env["DFLASH_TARGET"] == f"/opt/lucebox-hub/server/models/{pres.target_file}"
    if pres.draft_file:
        assert env["DFLASH_DRAFT"] == (
            f"/opt/lucebox-hub/server/models/draft/{pres.draft_file}"
        )


def test_server_run_spec_can_disable_preset_dflash_draft(tmp_path: Path) -> None:
    cfg = Config(
        models_dir=tmp_path,
        model=ModelMeta(preset="qwen3.6-27b"),
        dflash=DflashRuntime(speculative_decode=False),
    )

    env = _env(docker_run.server_run_spec(cfg))

    assert env["DFLASH_DRAFT"].endswith("/.lucebox-no-draft")


def test_server_run_spec_forwards_host_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUCEBOX_HOST_OS_PRETTY", "Ubuntu 22.04")
    monkeypatch.setenv("LUCEBOX_HOST_GPU_NAME", "RTX 5090")
    env = _env(docker_run.server_run_spec(Config(models_dir=tmp_path)))
    assert env["LUCEBOX_HOST_OS_PRETTY"] == "Ubuntu 22.04"
    assert env["LUCEBOX_HOST_GPU_NAME"] == "RTX 5090"


def test_large_preset_serves_at_safe_default_ctx(tmp_path: Path) -> None:
    """A bare low-level Config retains the conservative 16K context floor."""
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
