"""Integration-style tests for ``lucebox.sweep.run_sweep``.

All side-effects (subprocess.run, urllib, config_set, signal handlers)
are mocked. The goal is to verify the orchestration contract rather
than exercise real systemd / docker / urllib — the underlying
primitives (config.config_set, autotune.candidate_configs,
profile.run_profile) have their own tests.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from unittest import mock

import pytest
from lucebox.types import DflashRuntime, HostFacts

from lucebox import sweep as sweep_mod

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def stub_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin LUCEBOX_HOME + XDG_DATA_HOME + XDG_CONFIG_HOME under tmp.

    Also creates a fake systemd unit file so the pre-flight check
    passes, and seeds config.toml with a model preset so we don't
    bail on "no model configured".
    """
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path / "lucebox"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Fake systemd unit
    unit_path = tmp_path / "config" / "systemd" / "user" / "lucebox.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("# fake unit\n")
    # Seed config.toml with a model preset.
    cfg_path = tmp_path / "lucebox" / "config.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        '[model]\npreset = "qwen3.6-27b"\n'
        'target_file = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"\n'
        '\n[dflash]\nbudget = 22\nmax_ctx = 16384\n'
    )
    # Force HostFacts to a known 24 GB tier (the test bracket has 6 cells).
    monkeypatch.setattr(
        "lucebox.host_facts.from_env",
        lambda: HostFacts(vram_gb=24, gpu_name="RTX 5090", gpu_count=1),
    )
    monkeypatch.setattr(
        "lucebox.sweep.from_env",
        lambda: HostFacts(vram_gb=24, gpu_name="RTX 5090", gpu_count=1),
    )
    return tmp_path


def _write_synthetic_snapshot(snapshot_dir: Path, decode_tps: float) -> None:
    """Drop a tiny smoke.json into ``snapshot_dir`` carrying ``decode_tps``.

    Schema mirrors what ``luce-bench snapshot --level level1`` writes —
    `_mean_decode_tps_from_snapshot` averages every row that carries
    `timings.decode_tokens_per_sec`.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "area": "smoke",
        "n": 1,
        "rows": [
            {
                "pass": True,
                "timings": {
                    "decode_tokens_per_sec": decode_tps,
                    "prefill_ms": 100,
                    "decode_ms": 1000,
                },
            }
        ],
    }
    (snapshot_dir / "smoke.json").write_text(json.dumps(payload))


# ── unit tests for the helpers ─────────────────────────────────────────────


def test_mean_decode_tps_averages_per_area_rows(tmp_path: Path) -> None:
    snap = tmp_path / "cell"
    snap.mkdir()
    (snap / "smoke.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"timings": {"decode_tokens_per_sec": 40.0}},
                    {"timings": {"decode_tokens_per_sec": 50.0}},
                ]
            }
        )
    )
    (snap / "code.json").write_text(
        json.dumps({"rows": [{"timings": {"decode_tokens_per_sec": 60.0}}]})
    )
    tps = sweep_mod._mean_decode_tps_from_snapshot(snap)
    assert tps == pytest.approx(50.0)  # (40+50+60)/3


def test_mean_decode_tps_ignores_underscore_and_identity_files(tmp_path: Path) -> None:
    snap = tmp_path / "cell"
    snap.mkdir()
    (snap / "_summary.json").write_text(
        json.dumps({"rows": [{"timings": {"decode_tokens_per_sec": 999.0}}]})
    )
    (snap / "host.json").write_text(
        json.dumps({"rows": [{"timings": {"decode_tokens_per_sec": 999.0}}]})
    )
    (snap / "smoke.json").write_text(
        json.dumps({"rows": [{"timings": {"decode_tokens_per_sec": 40.0}}]})
    )
    tps = sweep_mod._mean_decode_tps_from_snapshot(snap)
    assert tps == pytest.approx(40.0)


def test_mean_decode_tps_returns_none_when_no_rows(tmp_path: Path) -> None:
    snap = tmp_path / "empty"
    snap.mkdir()
    (snap / "smoke.json").write_text(json.dumps({"rows": []}))
    assert sweep_mod._mean_decode_tps_from_snapshot(snap) is None


def test_pick_winner_breaks_ties_by_max_ctx_then_budget() -> None:
    r1 = sweep_mod.CellResult(
        index=0,
        config=DflashRuntime(budget=22, max_ctx=131072),
        snapshot_dir=None,
        mean_decode_tps=50.0,
        error=None,
    )
    r2 = sweep_mod.CellResult(
        index=1,
        config=DflashRuntime(budget=22, max_ctx=65536),
        snapshot_dir=None,
        mean_decode_tps=50.0,
        error=None,
    )
    r3 = sweep_mod.CellResult(
        index=2,
        config=DflashRuntime(budget=32, max_ctx=65536),
        snapshot_dir=None,
        mean_decode_tps=50.0,
        error=None,
    )
    winner = sweep_mod._pick_winner([r1, r2, r3])
    # All tied tps. Lower max_ctx wins, then lower budget.
    assert winner is r2


def test_pick_winner_returns_none_when_all_failed() -> None:
    r = sweep_mod.CellResult(
        index=0,
        config=DflashRuntime(),
        snapshot_dir=None,
        mean_decode_tps=None,
        error="server-not-ready",
    )
    assert sweep_mod._pick_winner([r]) is None


def test_pick_winner_picks_highest_tps() -> None:
    r1 = sweep_mod.CellResult(
        index=0, config=DflashRuntime(budget=8), snapshot_dir=None, mean_decode_tps=10.0, error=None
    )
    r2 = sweep_mod.CellResult(
        index=1, config=DflashRuntime(budget=22), snapshot_dir=None,
        mean_decode_tps=50.0, error=None,
    )
    r3 = sweep_mod.CellResult(
        index=2, config=DflashRuntime(budget=32), snapshot_dir=None,
        mean_decode_tps=30.0, error=None,
    )
    assert sweep_mod._pick_winner([r1, r2, r3]) is r2


# ── pre-flight ─────────────────────────────────────────────────────────────


def test_preflight_refuses_when_no_systemd_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path / "lucebox"))
    from rich.console import Console

    rc = sweep_mod._preflight(Console())
    assert rc == 2  # noqa: PLR2004


def test_preflight_refuses_when_no_model_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path / "lucebox"))
    # Install the systemd unit so we get past that check.
    unit_path = tmp_path / "config" / "systemd" / "user" / "lucebox.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("# fake unit\n")
    # No config.toml at all → live_config returns model.preset="" and target_file=""
    from rich.console import Console

    # Force HostFacts so live_config doesn't try to read env.
    monkeypatch.setattr(
        "lucebox.host_facts.from_env",
        lambda: HostFacts(vram_gb=24),
    )
    rc = sweep_mod._preflight(Console())
    assert rc == 2  # noqa: PLR2004


# ── full sweep flow ────────────────────────────────────────────────────────


def test_run_sweep_happy_path_picks_winner_and_applies(
    stub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three candidates → 3 restarts → 3 snapshots → winner applied + restart."""
    # Force a 3-cell bracket so the test runs fast.
    base = DflashRuntime(budget=22, max_ctx=65536)
    candidates = [
        DflashRuntime(budget=8, max_ctx=65536),
        base,
        DflashRuntime(budget=32, max_ctx=65536),
    ]
    monkeypatch.setattr(
        "lucebox.sweep.autotune_mod.candidate_configs",
        lambda host: candidates,  # noqa: ARG005
    )

    # Restart always succeeds.
    restart_calls = []

    def fake_run(argv, check=False, env=None, **kw):  # noqa: ARG001
        restart_calls.append(argv)
        # The sweep does TWO kinds of subprocess.run:
        #   1. systemctl --user restart lucebox.service  → success
        #   2. lucebox profile --level level1            → success, writes
        #      a snapshot dir into LUCEBOX_SWEEP_OUT_DIR / LUCEBOX_SWEEP_CELL_NAME
        if argv[0] == "lucebox" and len(argv) > 1 and argv[1] == "profile":
            sweep_dir = Path(env["LUCEBOX_SWEEP_OUT_DIR"]) if env else None
            cell_name = env.get("LUCEBOX_SWEEP_CELL_NAME") if env else None
            if sweep_dir and cell_name:
                # tps tied to budget so we can assert the winner is the
                # higher-tps cell. budget * 2 → 16, 44, 64. Winner: 64
                # (budget=32).
                budget_marker = cell_name  # encoded via _short_hash, not budget directly
                # Pick tps based on the call index — restart_calls
                # tracks ALL calls including restarts. Counting only
                # profile calls is more robust.
                profile_calls = [c for c in restart_calls if c[0] == "lucebox"]
                idx = len(profile_calls) - 1
                tps = [16.0, 44.0, 64.0][idx]
                _write_synthetic_snapshot(sweep_dir / cell_name, tps)
                del budget_marker
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr("lucebox.sweep.subprocess.run", fake_run)
    monkeypatch.setattr("lucebox.sweep._wait_ready", lambda port, timeout_s: True)

    rc = sweep_mod.run_sweep(yes=True)
    assert rc == 0

    # 3 cell restarts + 3 profile calls + 1 final winner restart = 7 calls.
    restart_argvs = [c for c in restart_calls if c[0] == "systemctl"]
    profile_argvs = [c for c in restart_calls if c[0] == "lucebox"]
    assert len(restart_argvs) == 4  # 3 cells + 1 final  # noqa: PLR2004
    assert len(profile_argvs) == 3  # noqa: PLR2004

    # Winner = budget=32 (tps=64). It must be persisted as the final
    # on-disk config.
    from lucebox import config as config_mod

    entries = config_mod.config_get()
    assert entries["dflash.budget"][0] == 32  # noqa: PLR2004
    assert entries["dflash.budget"][1] == "file"

    # Backup should be cleaned up on success.
    backup = sweep_mod._backup_path()
    assert not backup.exists(), f"backup not removed: {backup}"


def test_run_sweep_all_cells_fail_restores_backup(
    stub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cell times out on wait_ready → backup restored, exit non-zero."""
    candidates = [
        DflashRuntime(budget=8),
        DflashRuntime(budget=22),
        DflashRuntime(budget=32),
    ]
    monkeypatch.setattr(
        "lucebox.sweep.autotune_mod.candidate_configs",
        lambda host: candidates,  # noqa: ARG005
    )

    # Capture the pre-sweep config.toml.
    from lucebox import config as config_mod

    cfg_path = config_mod.default_config_path()
    pre_text = cfg_path.read_text()

    monkeypatch.setattr(
        "lucebox.sweep.subprocess.run",
        lambda argv, **kw: mock.MagicMock(returncode=0),  # noqa: ARG005
    )
    # Every readiness probe times out.
    monkeypatch.setattr("lucebox.sweep._wait_ready", lambda port, timeout_s: False)

    rc = sweep_mod.run_sweep(yes=True)
    assert rc == 1

    # Backup was restored — config.toml should match the pre-sweep state.
    assert cfg_path.read_text() == pre_text


def test_run_sweep_keyboard_interrupt_restores_backup(
    stub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt mid-sweep → backup restored, exit 130.

    The cleanup-restart at the end of ``run_sweep`` also flows through
    ``subprocess.run``; the fake here only raises mid-loop, so the
    final cleanup call lands in the "happy" branch.
    """
    candidates = [DflashRuntime(budget=8), DflashRuntime(budget=22), DflashRuntime(budget=32)]
    monkeypatch.setattr(
        "lucebox.sweep.autotune_mod.candidate_configs",
        lambda host: candidates,  # noqa: ARG005
    )

    from lucebox import config as config_mod

    cfg_path = config_mod.default_config_path()
    pre_text = cfg_path.read_text()

    state = {"calls": 0, "raised": False}

    def fake_run(argv, **kw):  # noqa: ARG001
        state["calls"] += 1
        # Raise only the FIRST time we hit the loop-body restart (call
        # #1 is the very first cell's restart). After we've raised
        # once we fall through to the cleanup restart inside the
        # finally / signal-handler path; that one must succeed so the
        # restore completes cleanly.
        if not state["raised"] and state["calls"] >= 2:  # noqa: PLR2004
            state["raised"] = True
            raise KeyboardInterrupt("simulated ctrl+c")
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr("lucebox.sweep.subprocess.run", fake_run)
    monkeypatch.setattr("lucebox.sweep._wait_ready", lambda port, timeout_s: True)
    # Don't actually install signal handlers — pytest already has its own.
    monkeypatch.setattr("lucebox.sweep.signal.signal", lambda sig, h: signal.SIG_DFL)

    rc = sweep_mod.run_sweep(yes=True)
    assert rc == 130  # noqa: PLR2004

    # Backup was restored — config.toml matches pre-sweep.
    assert cfg_path.read_text() == pre_text
