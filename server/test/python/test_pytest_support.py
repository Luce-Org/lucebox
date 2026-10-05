"""Exercise the shared pytest options and real child-process lifetimes."""

import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def suite(pytester, monkeypatch):
    for name in (
        "LUCE_MAX_CONCURRENCY", "LUCE_TEST_SERVER_URL", "LUCE_TEST_MODEL",
        "LUCE_SERVER_BIN", "LUCE_SERVER_EXTRA_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    pytester.makeconftest((ROOT / "conftest.py").read_text())
    return pytester


@pytest.mark.parametrize("cli, env", [
    ("0", None), ("0", "3"), ("-1", None), ("65", None), (None, "invalid"),
])
def test_invalid_concurrency_is_a_usage_error(suite, monkeypatch, cli, env):
    if env is not None:
        monkeypatch.setenv("LUCE_MAX_CONCURRENCY", env)
    suite.makepyfile("def test_unused(): pass")
    args = ["--max-concurrency", cli] if cli is not None else []
    result = suite.runpytest_subprocess(*args)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--max-concurrency must be*"])


@pytest.mark.parametrize("cli, env, expected", [
    ("1", "invalid", 1), (None, "64", 64),
])
def test_concurrency_cli_overrides_environment(suite, monkeypatch, cli, env, expected):
    monkeypatch.setenv("LUCE_MAX_CONCURRENCY", env)
    suite.makepyfile(f"""
        def test_slots(max_concurrency):
            assert max_concurrency == {expected}
    """)
    args = ["--max-concurrency", cli] if cli is not None else []
    suite.runpytest_subprocess(*args).assert_outcomes(passed=1)


@pytest.mark.parametrize("fail_first_module", [False, True])
def test_spawned_servers_stop_between_modules(suite, fail_first_module):
    # sys.executable is the binary and this script is the --launch "model",
    # so this exercises real startup/HTTP/teardown on Windows as well as Unix.
    fake = suite.makepyfile(fake_server="""
        import argparse
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, required=True)
        args, _ = parser.parse_known_args()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"status": "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with HTTPServer(("127.0.0.1", args.port), Handler) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    """)
    suite.makepyfile(state="handles = []")
    suite.makepyfile(test_a=f"""
        import state

        def test_launched(server_handle):
            state.handles.append(server_handle)
            assert server_handle.client.get("/health") == {{"status": "ok"}}
            assert {not fail_first_module}, "exercise teardown after failure"
    """)
    suite.makepyfile(test_b=f"""
        import sys
        import state

        def test_custom_server(spawn_luce_server):
            previous = state.handles[-1]
            assert previous.proc.poll() is not None, "previous module kept its server alive"
            assert previous.log_file.closed
            handle = spawn_luce_server([sys.executable, {str(fake)!r}], timeout=10)
            state.handles.append(handle)
            assert handle.client.get("/health") == {{"status": "ok"}}
    """)
    suite.makepyfile(test_c="""
        import state

        def test_all_stopped():
            assert len(state.handles) == 2
            for handle in state.handles:
                assert handle.proc.poll() is not None
                assert handle.log_file.closed
                handle.stop()  # stopping an already stopped server is harmless
    """)
    # Keep custom-option paths in the same argv entry. Before conftest is
    # loaded, pytest can mistake a repo-local interpreter for a test path
    # and load the real repository conftest as well as this isolated copy.
    suite.runpytest_subprocess(
        f"--server-bin={sys.executable}", f"--launch={fake}",
    ).assert_outcomes(passed=2 if fail_first_module else 3, failed=int(fail_first_module))


def test_server_binary_path_is_relative_to_invocation(suite):
    binary = suite.path / "local-server"
    binary.touch()
    suite.makepyfile(f"""
        from pathlib import Path

        def test_path(server_bin):
            assert server_bin == Path({str(binary)!r})
    """)
    suite.runpytest_subprocess("--server-bin", "./local-server").assert_outcomes(passed=1)


def test_startup_failure_reports_child_log(suite):
    suite.makepyfile("""
        import sys
        import pytest

        def test_failed_start(spawn_luce_server):
            with pytest.raises(pytest.fail.Exception, match="invalid model fixture"):
                spawn_luce_server([
                    sys.executable, "-c",
                    "raise SystemExit('invalid model fixture')",
                ], timeout=10)
    """)
    suite.runpytest_subprocess().assert_outcomes(passed=1)
