"""Tests for ``luce-bench report`` — summary + compare markdown output.

The contract under test:

* per-area JSON files in a snapshot dir route through
  ``normalize.normalize_result``, so the ``host`` block reaches the
  comparison table;
* compare tables surface a ``Host`` column built from
  ``host.wsl_version`` / ``host.gpus[0].name`` (+ "+N" for extras);
* when snapshots in a compare disagree on ``host.wsl_version`` /
  ``host.gpus[*].name`` / ``host.gpus[*].power_limit_w`` a confounder
  warning prints above the table (informational — never an exit);
* timestamps ride along the row identity since the same config can
  land in multiple snapshots over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lucebench import report


def _fake_area_payload(area: str, *, host: dict[str, Any] | None = None, n: int = 3, passes: int = 3) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "area": area,
        "rows": [
            {
                "case_id": f"c{i}",
                "content": "Answer: B",
                "graded": {"pass": i < passes, "strict_pass": i < passes},
                "wall_seconds": 0.1,
                "completion_tokens": 50,
            }
            for i in range(n)
        ],
    }
    if host is not None:
        payload["host"] = host
    return payload


def _bragi_host(*, power: int = 175) -> dict[str, Any]:
    return {
        "os_pretty": "Ubuntu 22.04.3 LTS",
        "kernel": "6.6.87.2-microsoft-standard-WSL2",
        "wsl_version": "wsl2",
        "docker_version": "29.1.3",
        "nvidia_driver": "596.36",
        "nvidia_ctk_version": "1.16.2",
        "cpu_model": "Test",
        "nproc": 24,
        "ram_gb": 64,
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-bragi",
                "pci_bus_id": "00000000:01:00.0",
                "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
                "sm": "12.0",
                "vram_gb": 24,
                "power_limit_w": power,
            }
        ],
        "cuda_visible_devices": "0",
        "source": "props",
        "collector": "lucebox.sh",
        "collected_at": "2026-05-28T20:31:42Z",
    }


def _sindri_host() -> dict[str, Any]:
    return {
        "os_pretty": "Ubuntu 24.04",
        "kernel": "6.6.0-generic",
        "wsl_version": None,
        "docker_version": "29.1.3",
        "nvidia_driver": "595.71.05",
        "nvidia_ctk_version": "1.16.2",
        "cpu_model": "AMD",
        "nproc": 64,
        "ram_gb": 128,
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-sindri",
                "pci_bus_id": "00000000:01:00.0",
                "name": "NVIDIA GeForce RTX 3090 Ti",
                "sm": "8.6",
                "vram_gb": 24,
                "power_limit_w": 350,
            }
        ],
        "cuda_visible_devices": None,
        "source": "props",
        "collector": "lucebox.sh",
        "collected_at": "2026-05-27T18:10:00Z",
    }


def _write_snap(root: Path, *, host: dict[str, Any] | None, areas: dict[str, dict[str, Any]]) -> None:
    """Write a fake snapshot dir with host.json + per-area JSON files."""
    root.mkdir(parents=True, exist_ok=True)
    if host is not None:
        (root / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    (root / "props.json").write_text("{}\n")
    (root / "config.json").write_text("{}\n")
    (root / "_summary.json").write_text("{}\n")
    for area, payload in areas.items():
        (root / f"{area}.json").write_text(json.dumps(payload, indent=2) + "\n")


def test_load_snapshot_extracts_host_from_per_area_json(tmp_path: Path) -> None:
    snap = tmp_path / "snap-a"
    _write_snap(
        snap,
        host=None,  # only embedded in the per-area json
        areas={
            "smoke": _fake_area_payload("smoke", host=_bragi_host()),
            "ds4-eval": _fake_area_payload("ds4-eval", host=_bragi_host()),
        },
    )
    by_area, host = report.load_snapshot(snap)
    assert set(by_area.keys()) == {"smoke", "ds4-eval"}
    assert host is not None
    assert host.wsl_version == "wsl2"
    assert host.gpus[0].name == "NVIDIA GeForce RTX 5090 Laptop GPU"


def test_load_snapshot_falls_back_to_host_json(tmp_path: Path) -> None:
    """No per-area host block but host.json is present → use host.json."""
    snap = tmp_path / "snap-b"
    _write_snap(
        snap,
        host=_sindri_host(),
        areas={"smoke": _fake_area_payload("smoke")},  # no host inside
    )
    by_area, host = report.load_snapshot(snap)
    assert by_area["smoke"]["n"] == 3
    assert host is not None
    # The per-area payload didn't carry host → normalize gave it
    # source="unknown", and load_snapshot routed to host.json after the
    # area scan, so the authoritative host is the sindri one.
    assert host.gpus[0].name == "NVIDIA GeForce RTX 3090 Ti"


def test_compare_table_includes_host_column(tmp_path: Path) -> None:
    snap_a = tmp_path / "snap-a"
    snap_b = tmp_path / "snap-b"
    _write_snap(
        snap_a,
        host=_bragi_host(),
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host())},
    )
    _write_snap(
        snap_b,
        host=_bragi_host(),  # same host → no confounder
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host())},
    )
    a, host_a = report.load_snapshot(snap_a)
    b, host_b = report.load_snapshot(snap_b)
    md = report.fmt_compare_md([("snap-a", a, host_a), ("snap-b", b, host_b)])
    assert "| snapshot | host |" in md
    # Compact GPU descriptor without the verbose "NVIDIA GeForce " prefix.
    assert "RTX 5090 Laptop GPU" in md
    # No confounder banner — same host on both rows.
    assert "confounder" not in md
    # Row identity includes the collected_at timestamp.
    assert "snap-a@2026-05-28T20:31:42Z" in md


def test_compare_table_warns_on_host_confounder_wsl(tmp_path: Path) -> None:
    """bragi (WSL2) vs sindri (bare Linux) → confounder banner on wsl_version + GPU name."""
    snap_a = tmp_path / "bragi"
    snap_b = tmp_path / "sindri"
    _write_snap(
        snap_a,
        host=_bragi_host(),
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host())},
    )
    _write_snap(
        snap_b,
        host=_sindri_host(),
        areas={"smoke": _fake_area_payload("smoke", host=_sindri_host())},
    )
    a, host_a = report.load_snapshot(snap_a)
    b, host_b = report.load_snapshot(snap_b)
    md = report.fmt_compare_md([("bragi", a, host_a), ("sindri", b, host_b)])
    assert "⚠ confounder" in md
    assert "host.wsl_version" in md
    # The full GPU lineup is what we compare now, so different GPU names
    # fold into the single "host.gpus lineup" confounder line.
    assert "host.gpus lineup" in md


def test_compare_table_warns_on_power_limit_diff(tmp_path: Path) -> None:
    """Same GPU model, different power caps → confounder line on GPU lineup."""
    snap_a = tmp_path / "bragi-175"
    snap_b = tmp_path / "bragi-150"
    _write_snap(
        snap_a,
        host=_bragi_host(power=175),
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host(power=175))},
    )
    _write_snap(
        snap_b,
        host=_bragi_host(power=150),
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host(power=150))},
    )
    a, host_a = report.load_snapshot(snap_a)
    b, host_b = report.load_snapshot(snap_b)
    md = report.fmt_compare_md([("bragi-175", a, host_a), ("bragi-150", b, host_b)])
    assert "⚠ confounder" in md
    # Power limit is now part of the lineup tuple; the formatted value
    # carries the `@<N>W` annotation.
    assert "host.gpus lineup" in md
    assert "175W" in md and "150W" in md


def test_host_short_handles_no_host() -> None:
    assert report._host_short(None) == "—"


def test_summary_includes_host_line(tmp_path: Path) -> None:
    snap = tmp_path / "summary-host"
    _write_snap(
        snap,
        host=_bragi_host(),
        areas={"smoke": _fake_area_payload("smoke", host=_bragi_host())},
    )
    by_area, host = report.load_snapshot(snap)
    md = report.fmt_summary_md("summary-host", by_area, host)
    assert "_host:_" in md
    assert "RTX 5090 Laptop GPU" in md


def test_multi_gpu_compact_descriptor() -> None:
    """N>1 GPUs render every name joined by `+` so 5090+3090 ≠ 5090+5090."""
    # Same-model dual GPU rig.
    same_host = _bragi_host()
    same_host["gpus"].append(
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "vram_gb": 24,
            "power_limit_w": 175,
        }
    )
    from lucebench.schema import host_from_dict

    h_same = host_from_dict(same_host)
    same_label = report._host_short(h_same)
    # Both names appear (no truncation to gpus[0]+count).
    assert same_label.count("RTX 5090 Laptop GPU") == 2

    # Mixed dual GPU rig: result must differ.
    mixed_host = _bragi_host()
    mixed_host["gpus"].append(
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 3090 Ti",
            "vram_gb": 24,
            "power_limit_w": 350,
        }
    )
    h_mixed = host_from_dict(mixed_host)
    mixed_label = report._host_short(h_mixed)
    assert "RTX 5090 Laptop GPU" in mixed_label
    assert "RTX 3090 Ti" in mixed_label
    # The whole point: matched vs mixed lineups must render differently.
    assert same_label != mixed_label
