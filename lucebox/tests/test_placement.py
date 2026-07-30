import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from lucebox.autotune import automatic_plan
from lucebox.placement import ARCHITECTURE_CAPABILITIES, automatic_placement
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
