"""Tests for ``lucebench.snapshot`` — the snapshot subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lucebench import snapshot


@pytest.fixture
def host_info_file(tmp_path: Path) -> Path:
    payload = {
        "cpu_model": "Test CPU",
        "nproc": 16,
        "ram_gb": 64,
        "gpu_name": "Test GPU 5090",
        "gpu_count": 1,
        "vram_gb": 32,
        "gpu_sm": "12.0",
        "gpu_power_limit_w": 600.0,
        "driver_version": "595.71.05",
        "cuda_runtime_version": "12.8",
        "nvidia_smi_csv": "Test GPU 5090, 32760 MiB",
    }
    p = tmp_path / "host.json"
    p.write_text(json.dumps(payload))
    return p


def _fake_props() -> dict[str, Any]:
    return {
        "model_card_source": "/etc/luce/cards/qwen3.6-27b.json",
        "runtime": {
            "budget": 22,
            "max_ctx": 65536,
            "lazy": False,
            "prefix_cache_slots": 0,
            "prefill_cache_slots": 0,
            "cache_type_k": "auto",
            "cache_type_v": "auto",
            "prefill_mode": "off",
            "prefill_keep_ratio": 0.05,
            "prefill_threshold": 32000,
            "prefill_drafter": "",
            "extra_ignored_field": "should not appear in config.json",
        },
    }


def _stub_area_runners(monkeypatch: pytest.MonkeyPatch, areas_written: list[str]) -> None:
    """Replace the real area runners with cheap stubs that just emit per-area JSON."""

    def fake_standard(
        area: str,
        *,
        out_root: Path,
        url: str,
        model: str,
        auth_header: str,
        timeout: int,
        max_tokens: int | None,
        think: bool | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        questions: int | None,
        no_fail_fast: bool,
        prompt_thinking_control: str,
        server_honors_api_flags: bool,
    ) -> tuple[dict[str, Any], bool]:
        areas_written.append(area)
        n = questions if questions is not None else 3
        payload = {
            "area": area,
            "n": n,
            "pass": n,
            "pass_rate": 100.0,
            "wall_total": n * 0.1,
            "wall_median": 0.1,
        }
        (out_root / f"{area}.json").write_text(json.dumps(payload))
        return (
            {
                "area": area,
                "n": n,
                "pass": n,
                "rate": 100.0,
                "wall_total": n * 0.1,
                "wall_median": 0.1,
            },
            False,
        )

    def fake_forge(
        *,
        out_root: Path,
        url: str,
        model: str,
        auth_header: str,
        timeout: int,
        max_tokens: int | None,
        questions: int | None,
    ) -> dict[str, Any]:
        areas_written.append("forge")
        (out_root / "forge.json").write_text(json.dumps({"area": "forge", "n": 7}))
        return {
            "area": "forge",
            "n": 7,
            "pass": 7,
            "rate": 100.0,
            "wall_total": 0.7,
            "wall_median": 0.1,
        }

    monkeypatch.setattr(snapshot, "_run_standard_area_to_dir", fake_standard)
    monkeypatch.setattr(snapshot, "_run_forge_area_to_dir", fake_forge)


def test_snapshot_level0_writes_required_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """level0 runs smoke only — dir contains the identity trio + summary + reproducer."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: _fake_props())
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    areas_written: list[str] = []
    _stub_area_runners(monkeypatch, areas_written)

    out_root = tmp_path / "snaps"
    argv = [
        "--level",
        "level0",
        "--url",
        "http://127.0.0.1:8080",
        "--out-dir",
        str(out_root),
        "--name",
        "fixed-name",
        "--host-info",
        str(host_info_file),
    ]
    rc = snapshot.main(argv)
    assert rc == 0
    snap_dir = out_root / "fixed-name"
    for fname in (
        "host.json",
        "props.json",
        "config.json",
        "_summary.json",
        "_summary.md",
        "command.sh",
        "bench.stdout",
        "bench.stderr",
        "smoke.json",
    ):
        assert (snap_dir / fname).is_file(), f"missing {fname}"

    # smoke only — level0 has exactly one area.
    assert areas_written == ["smoke"]

    # config.json carries the strict 11-field allowlist (no extras).
    config = json.loads((snap_dir / "config.json").read_text())
    assert set(config.keys()) == set(snapshot.CONFIG_FIELDS)
    assert "extra_ignored_field" not in config
    assert config["budget"] == 22
    assert config["max_ctx"] == 65536

    # _summary.json records the level so submit-baseline can validate it.
    summary = json.loads((snap_dir / "_summary.json").read_text())
    assert summary["level"] == "level0"
    assert summary["name"] == "fixed-name"

    # command.sh contains the actual snapshot invocation.
    cmd = (snap_dir / "command.sh").read_text()
    assert "luce-bench snapshot" in cmd
    assert "--level" in cmd
    assert "level0" in cmd


def test_snapshot_baselines_flag_requires_level3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """``--baselines`` is a guard: anything below level3 exits nonzero."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: {})
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    rc = snapshot.main(
        [
            "--level",
            "level1",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(tmp_path / "snaps"),
            "--name",
            "x",
            "--host-info",
            str(host_info_file),
            "--baselines",
        ]
    )
    assert rc == 2


def test_snapshot_default_name_is_stable_for_same_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """Auto-naming: same config → same dir name (modulo wall-clock date)."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: _fake_props())
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    _stub_area_runners(monkeypatch, [])

    out_root = tmp_path / "snaps"
    argv = [
        "--level",
        "level0",
        "--url",
        "http://127.0.0.1:8080",
        "--out-dir",
        str(out_root),
        "--host-info",
        str(host_info_file),
        "--label",
        "myrun",
    ]
    rc1 = snapshot.main(argv)
    assert rc1 == 0
    names_after_first = {p.name for p in out_root.iterdir()}
    assert len(names_after_first) == 1
    only_name = next(iter(names_after_first))
    # The label and the date should appear in the auto-name.
    assert "myrun" in only_name
    # A second run with the same args should hit the same dir.
    rc2 = snapshot.main(argv)
    assert rc2 == 0
    assert {p.name for p in out_root.iterdir()} == names_after_first


def test_snapshot_default_name_branches_on_config_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """Different /props.runtime → different auto-derived name (last 4 hex chars)."""
    _stub_area_runners(monkeypatch, [])
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))

    def props_a(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
        return _fake_props()

    def props_b(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
        d = _fake_props()
        d["runtime"]["budget"] = 12  # different config
        return d

    out_root = tmp_path / "snaps"
    monkeypatch.setattr(snapshot, "_fetch_props", props_a)
    snapshot.main(
        [
            "--level",
            "level0",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(out_root),
            "--host-info",
            str(host_info_file),
        ]
    )
    monkeypatch.setattr(snapshot, "_fetch_props", props_b)
    snapshot.main(
        [
            "--level",
            "level0",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(out_root),
            "--host-info",
            str(host_info_file),
        ]
    )
    names = sorted(p.name for p in out_root.iterdir())
    assert len(names) == 2, f"expected two dirs, got {names}"


def test_snapshot_level1_runs_full_area_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """level1 dispatches all five areas through the area-runner stub."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: _fake_props())
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    areas_written: list[str] = []
    _stub_area_runners(monkeypatch, areas_written)
    rc = snapshot.main(
        [
            "--level",
            "level1",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(tmp_path / "snaps"),
            "--name",
            "lvl1",
            "--host-info",
            str(host_info_file),
        ]
    )
    assert rc == 0
    # smoke, code, gsm8k, agent, longctx — in that order, per LEVELS.
    assert areas_written == ["smoke", "code", "gsm8k", "agent", "longctx"]


def test_derive_config_drops_unknown_fields() -> None:
    cfg = snapshot._derive_config(_fake_props())
    assert set(cfg.keys()) == set(snapshot.CONFIG_FIELDS)


def test_derive_config_handles_missing_runtime() -> None:
    cfg = snapshot._derive_config({})
    assert set(cfg.keys()) == set(snapshot.CONFIG_FIELDS)
    assert all(v is None for v in cfg.values())


def test_config_hash_is_order_independent() -> None:
    """Reordered keys hash to the same prefix."""
    a = {"budget": 22, "max_ctx": 1024, "lazy": False}
    b = {"max_ctx": 1024, "lazy": False, "budget": 22}
    assert snapshot._config_hash(a) == snapshot._config_hash(b)


def test_snapshot_writes_host_block_from_props(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """When ``/props.host`` is present, host.json is the verbatim block tagged source="props"."""
    props_with_host = _fake_props()
    props_with_host["host"] = {
        "os_pretty": "Ubuntu 22.04.3 LTS",
        "kernel": "6.6.87.2-microsoft-standard-WSL2",
        "wsl_version": "wsl2",
        "docker_version": "29.1.3",
        "nvidia_driver": "596.36",
        "nvidia_ctk_version": "1.16.2",
        "cpu_model": "Test CPU",
        "nproc": 24,
        "ram_gb": 64,
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-abc",
                "pci_bus_id": "00000000:01:00.0",
                "name": "NVIDIA RTX 5090",
                "sm": "12.0",
                "vram_gb": 24,
                "power_limit_w": 175,
            }
        ],
        "cuda_visible_devices": "0",
        "source": "lucebox.sh",
        "collector": "lucebox.sh",
        "collected_at": "2026-05-28T20:31:42Z",
    }
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: props_with_host)
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    areas_written: list[str] = []
    _stub_area_runners(monkeypatch, areas_written)

    out_root = tmp_path / "snaps"
    rc = snapshot.main(
        [
            "--level",
            "level0",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(out_root),
            "--name",
            "with-props-host",
            "--host-info",
            str(host_info_file),
        ]
    )
    assert rc == 0
    snap_dir = out_root / "with-props-host"
    host_json = json.loads((snap_dir / "host.json").read_text())
    # When data crossed the wire via /props, the snapshot relabels
    # source="props" — the entrypoint's source label ("lucebox.sh") is
    # preserved in `collector`. Two layers, two fields.
    assert host_json["source"] == "props"
    assert host_json["collector"] == "lucebox.sh"
    assert host_json["wsl_version"] == "wsl2"
    assert host_json["gpus"][0]["name"] == "NVIDIA RTX 5090"

    # Per-area JSON gets the same host block embedded.
    smoke_payload = json.loads((snap_dir / "smoke.json").read_text())
    assert "host" in smoke_payload
    assert smoke_payload["host"]["wsl_version"] == "wsl2"
    assert smoke_payload["host"]["gpus"][0]["name"] == "NVIDIA RTX 5090"


def test_snapshot_writes_host_block_client_fallback_when_props_lacks_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """``/props`` present but no ``host`` block → client-fallback probe used + tagged."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: _fake_props())
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    _stub_area_runners(monkeypatch, [])

    out_root = tmp_path / "snaps"
    rc = snapshot.main(
        [
            "--level",
            "level0",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(out_root),
            "--name",
            "client-fallback",
            "--host-info",
            str(host_info_file),
        ]
    )
    assert rc == 0
    snap_dir = out_root / "client-fallback"
    host_json = json.loads((snap_dir / "host.json").read_text())
    assert host_json["source"] == "client-fallback"
    # Canonical HostInfo-shaped fields are present (None when probe didn't fire,
    # but the key set is there).
    for key in (
        "os_pretty",
        "kernel",
        "wsl_version",
        "gpus",
        "source",
        "collector",
        "collected_at",
    ):
        assert key in host_json
    # Per-area JSON still gets the block.
    smoke_payload = json.loads((snap_dir / "smoke.json").read_text())
    assert smoke_payload["host"]["source"] == "client-fallback"


def test_snapshot_writes_host_block_unknown_when_no_props(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host_info_file: Path
) -> None:
    """``/props`` missing entirely (remote/OpenRouter) → still emit a client-fallback host block."""
    monkeypatch.setattr(snapshot, "_fetch_props", lambda url, timeout_s=10.0: {})
    monkeypatch.setattr(snapshot, "list_models", lambda url, auth_header="": (None, []))
    _stub_area_runners(monkeypatch, [])

    out_root = tmp_path / "snaps"
    rc = snapshot.main(
        [
            "--level",
            "level0",
            "--url",
            "http://127.0.0.1:8080",
            "--out-dir",
            str(out_root),
            "--name",
            "no-props",
            "--host-info",
            str(host_info_file),
        ]
    )
    assert rc == 0
    snap_dir = out_root / "no-props"
    host_json = json.loads((snap_dir / "host.json").read_text())
    # source is the client-side probe even when /props is missing.
    assert host_json["source"] == "client-fallback"
