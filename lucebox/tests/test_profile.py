import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from lucebox.types import Config, DflashRuntime, HostFacts

from lucebox import profile


class StubProfileInfoProvider:
    def __init__(self, *, dirty: bool = False, nvidia_smi_csv: str | None = None):
        self.dirty = dirty
        self.nvidia_smi_csv = nvidia_smi_csv

    def git_info(self):
        return {
            "repo_head": "abc123",
            "repo_head_subject": "test commit",
            "repo_branch": "test",
            "repo_dirty": self.dirty,
            "repo_dirty_staged": self.dirty,
            "repo_dirty_unstaged": False,
            "repo_untracked_count": 2 if self.dirty else 0,
        }

    def live_host_info(self):
        return {
            "hostname": "host-a",
            "kernel": "Linux host-a",
            "os_pretty_name": "Test Linux",
            "cpu_model": "Test CPU",
            "nproc": "8",
            "mem_total_kib": "123456",
            "nvidia_smi_csv": self.nvidia_smi_csv or "0, Test GPU, 24564, 1024, 999.1, 8.6",
            "nvidia_smi_full": "GPU test dump",
        }

    def docker_info(self):
        return {
            "docker_client_version": "1",
            "docker_server_version": "1",
            "docker_default_runtime": "runc",
            "docker_has_nvidia_runtime": False,
        }

    def image_info(self, _cfg):
        return {
            "image": "example/lucebox:cuda12",
            "image_id": "sha256:test",
            "image_created": "today",
        }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            payload = {"status": "ok"}
        elif self.path == "/props":
            payload = {
                "server": {"name": "luce-dflash", "version": "0.1.0", "props_schema": 1},
                "build_info": "luce-dflash v0.1.0 props_schema=1",
                "model_alias": "luce-dflash",
                "model_path": "/models/target.gguf",
                "model": {"draft_path": "/models/draft.gguf"},
                "runtime": {"backend": "cuda", "kv_cache_k": "q4_0", "kv_cache_v": "q4_0"},
                "default_generation_settings": {"n_ctx": 4096},
                "speculative_mode": "dflash",
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return


def _server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _cfg(tmp_path: Path) -> Config:
    return Config(
        image="example/lucebox",
        port=1236,
        models_dir=tmp_path,
        dflash=DflashRuntime(max_ctx=4096, budget=8),
        host=HostFacts(gpu_vendor="nvidia", gpu_name="Test GPU", vram_gb=24),
    )


def test_profile_keeps_versioned_results_and_exports_audited_snapshot(tmp_path):
    srv, url = _server()
    try:
        cfg = _cfg(tmp_path)
        provider = StubProfileInfoProvider()

        first = profile.run_profile(
            cfg,
            base_url=url,
            step_filter="health.props",
            force_refresh=True,
            provider=provider,
        )
        reused = profile.run_profile(
            cfg,
            base_url=url,
            step_filter="health.props",
            provider=provider,
        )
        second = profile.run_profile(
            cfg,
            base_url=url,
            step_filter="health.props",
            force_refresh=True,
            provider=provider,
        )

        assert first[0]["status"] == "passed"
        assert reused[0]["ran"] is False
        assert reused[0]["freshness_status"] == "fresh"
        assert second[0]["status"] == "passed"
        result_files = list(
            (tmp_path / ".lucebox" / "profile" / "results" / "health.props").glob("*/*.json")
        )
        assert len(result_files) == 2

        out = tmp_path / "snapshot.txt"
        written = profile.export_snapshot(cfg, out, base_url=url, provider=provider)

        assert written == out
        text = out.read_text()
        assert "[profile.step.1]" in text
        assert "step_id=health.props" in text
        assert "status=fresh" in text
        snapshot_json = json.loads((tmp_path / "snapshot.json").read_text())
        assert any(s["section"] == "profile.step.1" for s in snapshot_json["sections"])
    finally:
        srv.shutdown()


def test_profile_dry_run_reports_missing_without_running(tmp_path):
    cfg = _cfg(tmp_path)
    rows = profile.run_profile(
        cfg,
        base_url="http://127.0.0.1:1",
        step_filter="health.props",
        dry_run=True,
        provider=StubProfileInfoProvider(),
    )

    assert rows[0]["step_id"] == "health.props"
    assert rows[0]["ran"] is False
    assert rows[0]["status"] == "skipped_unavailable"


def test_luce_bench_argv_shape_for_each_area(tmp_path):
    """Each bench-running StepDefinition builds a luce-bench argv with the
    right area, base-url, model, and per-step knobs (think mode, max_tokens,
    timeout). Catches regressions in the argv builder shape — the framework
    later writes the produced JSON back through the hash/dedup pipeline."""
    srv, url = _server()
    try:
        cfg = _cfg(tmp_path)
        ctx = profile.build_context(cfg, base_url=url, provider=StubProfileInfoProvider())
        # Steps that drive luce-bench (4 of the 7 in the current registry).
        expected_areas = {
            "benchmark.code": ("code", None),
            "benchmark.longctx": ("longctx", None),
            "benchmark.agent": ("agent", "4096"),
            "quality.ds4_eval": ("ds4-eval", "16000"),
        }
        for step in profile.registry():
            if step.id not in expected_areas:
                continue
            area, max_tokens = expected_areas[step.id]
            assert step.argv is not None
            argv = step.argv(ctx, tmp_path)
            # Every luce-bench step delegates to `python -m harness.bench`,
            # which composes the lucebench argv internally. Single source
            # of truth for "run a bench against a Lucebox server".
            assert argv[1] == "-m"
            assert argv[2] == "harness.bench"
            # Area + base-url + model are mandatory.
            assert "--area" in argv
            assert argv[argv.index("--area") + 1] == area
            assert argv[argv.index("--base-url") + 1] == url
            assert argv[argv.index("--model") + 1] == "default"
            # JSON output lands in the step's owned dest dir.
            json_out = argv[argv.index("--json-out") + 1]
            assert str(tmp_path) in json_out
            # max_tokens only set when the step asked for it (agent, ds4_eval).
            if max_tokens is None:
                assert "--max-tokens" not in argv
            else:
                assert argv[argv.index("--max-tokens") + 1] == max_tokens
        # quality.ds4_eval is the slow score-only step; verify its specific knobs.
        ds4 = next(s for s in profile.registry() if s.id == "quality.ds4_eval")
        argv = ds4.argv(ctx, tmp_path)
        assert "--think" in argv  # ds4-eval runs with thinking enabled
        assert ds4.timeout_s == 86400  # 24h ceiling for the full 92-case run
        assert "--timeout" in argv  # per-case timeout (1800s)
        assert argv[argv.index("--timeout") + 1] == "1800"
    finally:
        srv.shutdown()


def test_profile_hash_ignores_volatile_machine_and_dirty_state(tmp_path):
    srv, url = _server()
    try:
        cfg = _cfg(tmp_path)
        clean = profile.build_context(
            cfg,
            base_url=url,
            provider=StubProfileInfoProvider(
                dirty=False,
                nvidia_smi_csv=(
                    "0, Test GPU, 24564, 1024, 999.1, 8.6, 0000:01:00.0, 70, 450, 3, 41, 1500, 9500"
                ),
            ),
        )
        busy = profile.build_context(
            cfg,
            base_url=url,
            provider=StubProfileInfoProvider(
                dirty=True,
                nvidia_smi_csv=(
                    "0, Test GPU, 24564, 20000, 999.1, 8.6, 0000:01:00.0, "
                    "390, 450, 99, 76, 1800, 10500"
                ),
            ),
        )

        step = next(s for s in profile.registry() if s.id == "benchmark.autotune_latest")

        assert profile.select_step(step, clean).hash == profile.select_step(step, busy).hash
        assert clean.git["repo_dirty"] is False
        assert busy.git["repo_dirty"] is True
        assert clean.machine["nvidia_smi_csv"] != busy.machine["nvidia_smi_csv"]
    finally:
        srv.shutdown()


def test_command_step_keeps_append_only_artifacts_and_fingerprint(tmp_path):
    srv, url = _server()
    try:
        cfg = _cfg(tmp_path)
        ctx = profile.build_context(cfg, base_url=url, provider=StubProfileInfoProvider())
        code = (
            "import json, pathlib, sys; "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'ok': True}) + '\\n')"
        )

        def argv(_ctx, dest):
            return [sys.executable, "-c", code, str(dest / "report.json")]

        step = profile.StepDefinition(
            id="test.command_artifacts",
            version=1,
            description="command artifact append-only test",
            timeout_s=10,
            max_age_hours=24,
            requires_live_server=False,
            argv=argv,
            report_name="report.json",
        )
        selection = profile.select_step(step, ctx, force_refresh=True)

        first = profile._command_step(step, ctx, selection.hash, True, selection.fingerprint)
        second = profile._command_step(step, ctx, selection.hash, True, selection.fingerprint)

        assert first["status"] == "passed"
        assert second["status"] == "passed"
        assert first["report_path"] != second["report_path"]
        assert first["fingerprint"] == selection.fingerprint
        result_dir = tmp_path / ".lucebox" / "profile" / "results" / step.id / selection.hash
        assert (result_dir / first["report_path"]).exists()
        assert (result_dir / second["report_path"]).exists()
        assert len(list(result_dir.glob("*.json"))) == 2
    finally:
        srv.shutdown()


def test_hardware_dump_redaction_removes_device_identifiers():
    redacted = profile._redact_hardware_dump(
        "Serial Number                                      : 12345\n"
        "GPU UUID                                           : GPU-secret\n"
        "GPU PDI                                            : secret-pdi\n"
        "Product Name                                       : Test GPU\n"
    )

    assert "12345" not in redacted
    assert "GPU-secret" not in redacted
    assert "secret-pdi" not in redacted
    assert "Product Name                                       : Test GPU" in redacted


def test_snapshot_canonicalizes_legacy_autotune_profile_names():
    assert profile._canonical_autotune_profile("quick") == "level1"
    assert profile._canonical_autotune_profile("full") == "level2"
    assert profile._canonical_autotune_profile("stress") == "level3"
    assert profile._canonical_autotune_profile("level2") == "level2"


def test_missing_binary_status_does_not_mark_unknown_repo_dirty():
    assert profile._cmd_status(["/definitely/missing/lucebox-test-git"]) == 127

    provider = profile.LiveProfileInfoProvider()
    info = provider.git_info()

    if info["repo_head"]:
        assert isinstance(info["repo_dirty"], bool)
    else:
        assert info["repo_dirty"] is False
        assert info["repo_dirty_staged"] is False
        assert info["repo_dirty_unstaged"] is False
