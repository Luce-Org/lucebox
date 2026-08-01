import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from lucebox.autotune import automatic_plan
from lucebox.capabilities import ARCHITECTURE_CAPABILITIES
from lucebox.docker_run import server_run_spec
from lucebox.host_facts import compatible_variant, for_variant, from_env, nvidia_variant
from lucebox.placement import automatic_placement
from lucebox.topology import from_config
from lucebox.types import Config, DflashRuntime, HostFacts, ModelMeta

from lucebox import download


def _install_optimizer_drafter(cfg: Config) -> None:
    path = download.optimizer_drafter_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-scorer")


def _nvidia_csv(*memory_mib: int) -> str:
    return "\n".join(
        f"{index}, GPU-{index}, 0000:{index:02x}:00.0, RTX {index}, 8.6, {memory} MiB, 350 W"
        for index, memory in enumerate(memory_mib)
    )


def _amd_csv(*rows: tuple[str, str, int]) -> str:
    return "\n".join(
        f"{index}, , , {name}, {architecture}, {memory_mib} MiB,"
        for index, (name, architecture, memory_mib) in enumerate(rows)
    )


def _set_host_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str | int]) -> None:
    for key in tuple(os.environ):
        if key.startswith("LUCEBOX_HOST_"):
            monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


def test_r9700_strix_keeps_fitting_qwen_target_on_r9700(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="AMD Radeon AI PRO R9700",
        gpu_count=2,
        vram_gb=31,
        gpu_sm="gfx1201",
        ram_gb=125,
        amd_gpu_name="AMD Radeon AI PRO R9700",
        amd_gpu_count=2,
        amd_vram_gb=31,
        amd_gpu_arch="gfx1201",
        amd_gpu_list_csv=_amd_csv(
            ("AMD Radeon AI PRO R9700", "gfx1201", 32624),
            ("AMD Radeon Graphics", "gfx1151", 512),
        ),
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="qwen3.6-27b"),
    )

    topology = from_config(cfg)
    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert topology.primary is not None
    assert topology.primary.name == "AMD Radeon AI PRO R9700"
    assert topology.companions[0].unified_memory is True
    assert topology.companions[0].effective_memory_gb == 109
    assert plan.placement.runtime.target_device == "hip:0"
    assert plan.placement.runtime.uses_multiple_devices is False
    assert "full stack fits" in plan.placement.reason


def test_strix_128gb_does_not_double_reserve_memory_for_deepseek(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="Radeon 8060S",
        gpu_count=1,
        vram_gb=125,
        gpu_sm="gfx1151",
        ram_gb=125,
        amd_gpu_name="Radeon 8060S",
        amd_gpu_count=1,
        amd_vram_gb=125,
        amd_gpu_arch="gfx1151",
        amd_gpu_list_csv=_amd_csv(("Radeon 8060S", "gfx1151", 0)),
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="deepseek-v4-flash"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.placement.runnable is True
    assert plan.placement.runtime.mode == "single"
    assert plan.placement.runtime.target_device == "hip:0"
    assert plan.prefill_alternative is not None
    assert plan.prefill_alternative.available is True


def test_three_same_backend_gpus_are_used_only_when_capacity_requires_them(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        gpu_name="RTX 0",
        gpu_count=3,
        vram_gb=24,
        gpu_sm="86",
        nvidia_gpu_name="RTX 0",
        nvidia_gpu_count=3,
        nvidia_vram_gb=24,
        nvidia_gpu_arch="86",
        nvidia_gpu_list_csv=_nvidia_csv(24576, 24576, 24576),
    )
    cfg = Config(variant="cuda12", models_dir=tmp_path, host=host)
    preset = SimpleNamespace(
        architecture="qwen35",
        target_file="large.gguf",
        approx_target_gb=55.0,
        approx_draft_gb=0.0,
    )

    plan = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=False),
        preset,
        has_draft=False,
        optimizer_drafter_available=False,
    )

    assert plan.runnable is True
    assert plan.runtime.mode == "layer-split"
    assert plan.runtime.target_devices == ("cuda:0", "cuda:1", "cuda:2")
    assert sum(plan.runtime.target_layer_split) == pytest.approx(1.0)
    assert plan.runtime.peer_access is False


def test_undersized_secondary_does_not_claim_draft_offload(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        gpu_name="RTX 0",
        gpu_count=2,
        vram_gb=18,
        gpu_sm="86",
        nvidia_gpu_name="RTX 0",
        nvidia_gpu_count=2,
        nvidia_vram_gb=18,
        nvidia_gpu_arch="86",
        nvidia_gpu_list_csv=_nvidia_csv(18432, 1024),
    )
    cfg = Config(
        variant="cuda12",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="qwen3.6-27b"),
    )
    preset = download.PRESETS["qwen3.6-27b"]

    placement = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=True),
        preset,
        has_draft=True,
        optimizer_drafter_available=False,
    )

    draft_option = next(option for option in placement.options if option.key == "draft-offload")
    assert placement.runnable is False
    assert draft_option.available is False
    assert "0 GB of safe capacity" in draft_option.reason


def test_layer_split_never_assigns_target_to_a_zero_capacity_primary(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        gpu_name="RTX 0",
        gpu_count=2,
        vram_gb=4,
        gpu_sm="86",
        nvidia_gpu_name="RTX 0",
        nvidia_gpu_count=2,
        nvidia_vram_gb=4,
        nvidia_gpu_arch="86",
        nvidia_gpu_list_csv=_nvidia_csv(4096, 24576),
    )
    cfg = Config(variant="cuda12", models_dir=tmp_path, host=host)
    preset = SimpleNamespace(
        architecture="qwen35",
        target_file="large.gguf",
        approx_target_gb=20.0,
        approx_draft_gb=3.0,
    )

    placement = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=True),
        preset,
        has_draft=True,
        optimizer_drafter_available=False,
    )

    split_option = next(option for option in placement.options if option.key == "layer-split")
    assert placement.runnable is False
    assert split_option.available is False


def test_rtx_strix_moe_uses_paired_runtime_for_remote_spark_experts(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        gpu_count=1,
        vram_gb=24,
        gpu_sm="86",
        ram_gb=125,
        nvidia_gpu_name="RTX 3090",
        nvidia_gpu_count=1,
        nvidia_vram_gb=24,
        nvidia_gpu_arch="86",
        nvidia_gpu_list_csv=("0, GPU-0, 0000:01:00.0, RTX 3090, 8.6, 24576 MiB, 350 W"),
        amd_gpu_name="AMD Radeon Graphics",
        amd_gpu_count=1,
        amd_vram_gb=125,
        amd_gpu_arch="gfx1151",
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 512)),
        hybrid_runtime=True,
    )
    cfg = Config(
        variant="cuda12",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="qwen3.6-moe"),
    )
    _install_optimizer_drafter(cfg)

    plan = automatic_plan(cfg)

    assert plan.runtime.spark is True
    assert plan.placement.runtime.target_device == "cuda:0"
    assert plan.placement.runtime.remote_expert_device == "hip:0"
    assert plan.placement.runtime.requires_hybrid_runtime is True
    assert "Spark experts" in plan.placement.summary


def test_rtx_strix_without_paired_runtime_keeps_spark_on_cpu(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        vram_gb=24,
        gpu_sm="86",
        ram_gb=125,
        nvidia_gpu_list_csv=("0, GPU-0, 0000:01:00.0, RTX 3090, 8.6, 24576 MiB, 350 W"),
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 512)),
    )
    cfg = Config(
        variant="cuda12",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="qwen3.6-moe"),
    )
    _install_optimizer_drafter(cfg)

    plan = automatic_plan(cfg)

    assert plan.runtime.spark is True
    assert plan.placement.runtime.target_device == "cuda:0"
    assert plan.placement.runtime.remote_expert_device == ""
    assert plan.placement.runtime.requires_hybrid_runtime is False
    remote_option = next(
        option for option in plan.placement.options if option.key == "remote-experts"
    )
    assert remote_option.available is False


def test_paired_runtime_does_not_claim_unsupported_hip_to_cuda_direction(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        vram_gb=24,
        gpu_sm="86",
        ram_gb=125,
        nvidia_gpu_list_csv=("0, GPU-0, 0000:01:00.0, RTX 3090, 8.6, 24576 MiB, 350 W"),
        amd_gpu_name="AMD Radeon Graphics",
        amd_gpu_count=1,
        amd_vram_gb=125,
        amd_gpu_arch="gfx1151",
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 512)),
        hybrid_runtime=True,
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="qwen3.6-moe"),
    )
    _install_optimizer_drafter(cfg)

    plan = automatic_plan(cfg)

    assert plan.placement.runtime.target_device == "hip:0"
    assert plan.placement.runtime.remote_expert_device == ""
    assert plan.placement.runtime.requires_hybrid_runtime is False


def test_strix_is_uma_even_when_driver_reports_large_aperture(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="AMD Radeon Graphics",
        vram_gb=16,
        gpu_sm="gfx1151",
        ram_gb=128,
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 16384)),
    )

    topology = from_config(Config(variant="rocm", models_dir=tmp_path, host=host))

    assert topology.primary is not None
    assert topology.primary.unified_memory is True
    assert topology.primary.physical_vram_gb == 16
    assert topology.primary.effective_memory_gb == 112


def test_gb10_nvml_na_memory_is_normalized_as_shared_memory(tmp_path: Path) -> None:
    """DGX Spark reports ``[N/A]`` for NVML memory despite CUDA UMA."""
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        gpu_name="NVIDIA GB10",
        gpu_count=1,
        vram_gb=105,
        gpu_sm="121",
        ram_gb=121,
        nvidia_gpu_name="NVIDIA GB10",
        nvidia_gpu_count=1,
        nvidia_vram_gb=105,
        nvidia_gpu_arch="121",
        nvidia_gpu_list_csv=(
            "0, GPU-test, 00000000:01:00.0, NVIDIA GB10, 12.1, [N/A], [N/A]"
        ),
        nvidia_unified_memory=True,
    )

    topology = from_config(Config(variant="cuda13", models_dir=tmp_path, host=host))

    assert topology.primary is not None
    assert topology.primary.backend == "cuda"
    assert topology.primary.architecture == "121"
    assert topology.primary.unified_memory is True
    assert topology.primary.physical_vram_gb == 0
    assert topology.primary.effective_memory_gb == 105
    assert "105 GB shared" in topology.primary.label


def test_gb10_unified_memory_flag_is_read_from_host_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_host_env(
        monkeypatch,
        {
            "LUCEBOX_HOST_GPU_VENDOR": "nvidia",
            "LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY": 1,
        },
    )

    assert from_env().nvidia_unified_memory is True


def test_backend_projection_uses_the_selected_vendor_inventory() -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        gpu_count=1,
        vram_gb=24,
        gpu_sm="86",
        nvidia_gpu_name="RTX 3090",
        nvidia_gpu_count=1,
        nvidia_vram_gb=24,
        nvidia_gpu_arch="86",
        amd_gpu_name="AMD Radeon Graphics",
        amd_gpu_count=1,
        amd_vram_gb=125,
        amd_gpu_arch="gfx1151",
    )

    selected = for_variant(host, "rocm")

    assert selected.gpu_vendor == "amd"
    assert selected.gpu_name == "AMD Radeon Graphics"
    assert selected.gpu_count == 1
    assert selected.vram_gb == 125
    assert selected.gpu_sm == "gfx1151"


@pytest.mark.parametrize(
    ("architecture", "name", "expected"),
    [
        ("86", "NVIDIA GeForce RTX 3090", "cuda12"),
        ("90", "NVIDIA H100", "cuda12"),
        ("120", "NVIDIA GeForce RTX 5090", "cuda128"),
        ("121", "NVIDIA GB10", "cuda13"),
    ],
)
def test_nvidia_image_variant_tracks_the_toolkit_required_by_the_architecture(
    architecture: str,
    name: str,
    expected: str,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        gpu_name=name,
        gpu_sm=architecture,
        nvidia_gpu_name=name,
        nvidia_gpu_arch=architecture,
    )

    assert nvidia_variant(host) == expected
    assert compatible_variant(host, "cuda12") == expected


def test_rtx_strix_uses_remote_target_shard_when_capacity_requires_it(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        vram_gb=24,
        gpu_sm="86",
        ram_gb=128,
        nvidia_gpu_list_csv=("0, GPU-0, 0000:01:00.0, RTX 3090, 8.6, 24576 MiB, 350 W"),
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 512)),
        hybrid_runtime=True,
    )
    cfg = Config(variant="cuda12", models_dir=tmp_path, host=host)
    preset = SimpleNamespace(
        architecture="qwen35",
        target_file="large.gguf",
        approx_target_gb=40.0,
        approx_draft_gb=0.0,
    )

    plan = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=False),
        preset,
        has_draft=False,
        optimizer_drafter_available=False,
    )

    assert plan.runnable is True
    assert plan.runtime.mode == "heterogeneous"
    assert plan.runtime.target_devices == ("cuda:0", "hip:0")
    assert plan.runtime.remote_target_shard is True
    assert plan.runtime.peer_access is False


def test_deepseek_uses_r9700_and_strix_capacity_automatically(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="AMD Radeon AI PRO R9700",
        gpu_count=2,
        vram_gb=31,
        gpu_sm="gfx1201",
        ram_gb=125,
        amd_gpu_list_csv=_amd_csv(
            ("AMD Radeon AI PRO R9700", "gfx1201", 32624),
            ("AMD Radeon Graphics", "gfx1151", 512),
        ),
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="deepseek-v4-flash"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.placement.runnable is True
    assert plan.placement.runtime.mode == "layer-split"
    assert plan.placement.runtime.target_devices == ("hip:0", "hip:1")
    assert sum(plan.placement.runtime.target_layer_split) == pytest.approx(1.0)
    assert plan.runtime.speculative_decode is False
    assert plan.prefill_alternative is not None
    assert plan.prefill_alternative.available is False
    assert "requires the target to fit one HIP device" in plan.prefill_alternative.reason


def test_deepseek_sparse_prefill_runs_only_when_target_fits_one_hip_gpu(
    tmp_path: Path,
) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="Large HIP GPU",
        vram_gb=128,
        gpu_sm="gfx1201",
        ram_gb=256,
        amd_gpu_list_csv=_amd_csv(("Large HIP GPU", "gfx1201", 131072)),
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="deepseek-v4-flash"),
    )

    placement = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=False, ds4_prefill="sparse"),
        download.PRESETS["deepseek-v4-flash"],
        has_draft=False,
        optimizer_drafter_available=False,
    )

    assert placement.runnable is True
    assert placement.runtime.mode == "single"
    assert placement.runtime.target_device == "hip:0"


def test_deepseek_sparse_prefill_rejects_required_layer_split(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="amd",
        has_amd_gpu=True,
        gpu_name="AMD Radeon AI PRO R9700",
        gpu_count=2,
        vram_gb=31,
        gpu_sm="gfx1201",
        ram_gb=125,
        amd_gpu_list_csv=_amd_csv(
            ("AMD Radeon AI PRO R9700", "gfx1201", 32624),
            ("AMD Radeon Graphics", "gfx1151", 512),
        ),
    )
    cfg = Config(variant="rocm", models_dir=tmp_path, host=host)

    placement = automatic_placement(
        cfg,
        DflashRuntime(speculative_decode=False, ds4_prefill="sparse"),
        download.PRESETS["deepseek-v4-flash"],
        has_draft=False,
        optimizer_drafter_available=False,
    )

    assert placement.runnable is False
    assert "requires the target to fit one HIP device" in placement.reason


def test_deepseek_uses_rtx_strix_paired_runtime_automatically(tmp_path: Path) -> None:
    host = HostFacts(
        gpu_vendor="nvidia",
        has_nvidia_gpu=True,
        has_amd_gpu=True,
        gpu_name="RTX 3090",
        vram_gb=24,
        gpu_sm="86",
        ram_gb=128,
        nvidia_gpu_list_csv=("0, GPU-0, 0000:01:00.0, RTX 3090, 8.6, 24576 MiB, 350 W"),
        amd_gpu_list_csv=_amd_csv(("AMD Radeon Graphics", "gfx1151", 512)),
        hybrid_runtime=True,
    )
    cfg = Config(
        variant="cuda12",
        models_dir=tmp_path,
        host=host,
        model=ModelMeta(preset="deepseek-v4-flash"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.placement.runnable is True
    assert plan.placement.runtime.mode == "heterogeneous"
    assert plan.placement.runtime.target_devices == ("cuda:0", "hip:0")
    assert plan.placement.runtime.remote_target_shard is True
    assert plan.placement.runtime.requires_hybrid_runtime is True


def test_python_architecture_capabilities_match_engine_table() -> None:
    header = (Path(__file__).parents[2] / "server/src/common/model_capabilities.h").read_text()
    row_pattern = re.compile(
        r'\{"(?P<arch>[^\"]+)",\s*'
        r"(?P<split>true|false),\s*(?P<remote>true|false),\s*"
        r"(?P<pflash>true|false),\s*(?P<offload>true|false),\s*"
        r"(?P<draft>kNever|kMono|kBoth),"
    )
    engine_rows = {match["arch"]: match.groupdict() for match in row_pattern.finditer(header)}

    assert set(ARCHITECTURE_CAPABILITIES) == set(engine_rows)
    for architecture, capability in ARCHITECTURE_CAPABILITIES.items():
        row = engine_rows[architecture]
        split = row["split"] == "true"
        assert capability.layer_split is split
        assert capability.remote_draft is (row["remote"] == "true")
        assert capability.draft_on_layer_split is (row["draft"] == "kBoth")
        assert capability.pflash_on_layer_split is (split and row["pflash"] == "true")
        assert capability.expert_offload is (row["offload"] == "true")


def test_env_driven_dual_nvidia_plan_reaches_server_launch_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_host_env(
        monkeypatch,
        {
            "LUCEBOX_HOST_NPROC": 16,
            "LUCEBOX_HOST_RAM_GB": 64,
            "LUCEBOX_HOST_GPU_VENDOR": "nvidia",
            "LUCEBOX_HOST_HAS_NVIDIA_GPU": 1,
            "LUCEBOX_HOST_GPU_NAME": "RTX 0",
            "LUCEBOX_HOST_GPU_COUNT": 2,
            "LUCEBOX_HOST_VRAM_GB": 12,
            "LUCEBOX_HOST_GPU_SM": "86",
            "LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV": _nvidia_csv(12288, 12288),
        },
    )
    cfg = Config(
        variant="cuda12",
        models_dir=tmp_path,
        host=from_env(),
        model=ModelMeta(preset="qwen3.6-27b"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)
    launch = server_run_spec(
        Config(
            variant=cfg.variant,
            models_dir=cfg.models_dir,
            host=cfg.host,
            model=cfg.model,
            dflash=plan.runtime,
            placement=plan.placement.runtime,
        )
    )
    launch_env = dict(launch.env)

    assert plan.placement.runnable is True
    assert plan.placement.runtime.mode == "layer-split"
    assert plan.placement.runtime.target_devices == ("cuda:0", "cuda:1")
    assert launch_env["DFLASH_TARGET_DEVICES"] == "cuda:0,cuda:1"
    assert launch_env["DFLASH_TARGET_LAYER_SPLIT"]
    assert launch_env["DFLASH_KVFLASH"] == "auto"


def test_env_driven_lucebox_plan_uses_r9700_without_unnecessary_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_host_env(
        monkeypatch,
        {
            "LUCEBOX_HOST_NPROC": 32,
            "LUCEBOX_HOST_RAM_GB": 125,
            "LUCEBOX_HOST_GPU_VENDOR": "amd",
            "LUCEBOX_HOST_HAS_AMD_GPU": 1,
            "LUCEBOX_HOST_GPU_NAME": "AMD Radeon AI PRO R9700",
            "LUCEBOX_HOST_GPU_COUNT": 2,
            "LUCEBOX_HOST_VRAM_GB": 31,
            "LUCEBOX_HOST_GPU_SM": "gfx1201",
            "LUCEBOX_HOST_AMD_GPU_LIST_CSV": _amd_csv(
                ("AMD Radeon AI PRO R9700", "gfx1201", 32624),
                ("AMD Radeon Graphics", "gfx1151", 512),
            ),
        },
    )
    cfg = Config(
        variant="rocm",
        models_dir=tmp_path,
        host=from_env(),
        model=ModelMeta(preset="qwen3.6-27b"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)
    launch = server_run_spec(
        Config(
            variant=cfg.variant,
            models_dir=cfg.models_dir,
            host=cfg.host,
            model=cfg.model,
            dflash=plan.runtime,
            placement=plan.placement.runtime,
        )
    )
    launch_env = dict(launch.env)

    assert plan.placement.runnable is True
    assert plan.placement.runtime.mode == "single"
    assert plan.placement.runtime.target_device == "hip:0"
    assert plan.placement.runtime.uses_multiple_devices is False
    assert plan.placement.topology.companions[0].unified_memory is True
    assert launch.gpu_vendor == "amd"
    assert launch_env["DFLASH_TARGET_DEVICE"] == "hip:0"
    assert "DFLASH_TARGET_DEVICES" not in launch_env
