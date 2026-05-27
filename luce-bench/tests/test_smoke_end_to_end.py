"""End-to-end smoke tests against an in-process HTTP fixture server.

Exercises the full CLI path through a real socket-level OpenAI-shape
server — driven via ``subprocess`` so we catch packaging /
console-script / argparse regressions that unit tests with mocked
urlopen would miss.

The fixture server is defined in ``conftest.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Invoke `python -m lucebench.cli ...` and capture stdout/stderr."""
    cmd = [sys.executable, "-m", "lucebench.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        env={**({} if env is None else env)},
    )


def _run_cli_passthrough(*args: str) -> subprocess.CompletedProcess:
    """Same as _run_cli but inherits the parent env (for $PATH / $HOME)."""
    import os

    env = {k: v for k, v in os.environ.items()}
    return _run_cli(*args, env=env)


def test_single_area_smoke(mock_openai_server, tmp_path):
    """`luce-bench --area code` against the mock server produces a JSON-out row."""
    url, captured, _ = mock_openai_server
    out_json = tmp_path / "code.json"
    result = _run_cli_passthrough(
        "--area",
        "code",
        "--base-url",
        url,
        "--model",
        "mock-model",
        "--questions",
        "1",
        "--timeout",
        "10",
        "--json-out",
        str(out_json),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert out_json.exists()

    data = json.loads(out_json.read_text())
    assert data["area"] == "code"
    assert data["n"] == 1
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    # Mock server returned content "42\nAnswer: 42"; the row should carry it.
    assert "42" in (row.get("content") or "")
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    # Server populated timings — they should flow through to the row.
    assert row["timings"]["decode_tokens_per_sec"] == 100.0

    # The server saw exactly one POST to /v1/chat/completions.
    posts = [c for c in captured if c["path"] == "/v1/chat/completions"]
    assert len(posts) == 1
    assert posts[0]["body"]["model"] == "mock-model"


def test_v1_models_autoresolve(mock_openai_server, tmp_path):
    """`--model default` should hit /v1/models and pick the single exposed model."""
    url, captured, _ = mock_openai_server
    out_json = tmp_path / "code.json"
    result = _run_cli_passthrough(
        "--area",
        "code",
        "--base-url",
        url,
        # NOTE: not passing --model → CLI defaults to "default" → /v1/models hit.
        "--questions",
        "1",
        "--timeout",
        "10",
        "--json-out",
        str(out_json),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # Stderr should mention the autoresolve.
    assert "resolved to 'mock-model'" in (result.stdout + result.stderr)
    # And the actual POST should ship the resolved model id.
    posts = [c for c in captured if c["path"] == "/v1/chat/completions"]
    assert posts[0]["body"]["model"] == "mock-model"


def test_sweep_smoke(mock_openai_server, tmp_path):
    """`--sweep` writes per-area JSON + _summary files."""
    url, _captured, _ = mock_openai_server
    out_dir = tmp_path / "snapshots"
    result = _run_cli_passthrough(
        "--sweep",
        "--name",
        "ci-smoke",
        "--base-url",
        url,
        "--model",
        "mock-model",
        "--questions",
        "1",
        "--timeout",
        "10",
        "--out-dir",
        str(out_dir),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    snap = out_dir / "ci-smoke"
    assert snap.exists()
    # All four stdlib areas should have written a per-area JSON.
    for area in ("ds4-eval", "code", "longctx", "agent"):
        f = snap / f"{area}.json"
        assert f.exists(), f"missing {f}"
        data = json.loads(f.read_text())
        assert data["area"] == area
        assert data["n"] == 1
    # Combined summary files.
    assert (snap / "_summary.json").exists()
    summary = json.loads((snap / "_summary.json").read_text())
    assert summary["name"] == "ci-smoke"
    assert len(summary["areas"]) == 4
    assert (snap / "_summary.md").exists()
    md = (snap / "_summary.md").read_text()
    assert "ci-smoke" in md
    assert "| ds4-eval |" in md


def test_sweep_fail_fast_on_unreachable(tmp_path):
    """`--sweep` aborts cleanly when the URL is unreachable (no live server)."""
    out_dir = tmp_path / "snapshots"
    # Use a port that's almost certainly closed.
    result = _run_cli_passthrough(
        "--sweep",
        "--name",
        "fail-fast-smoke",
        "--base-url",
        "http://127.0.0.1:1",  # Reserved / closed in practice
        "--model",
        "mock-model",
        "--questions",
        "1",
        "--timeout",
        "5",
        "--out-dir",
        str(out_dir),
    )
    # Exit code 3 = sweep aborted by fail-fast guard.
    assert result.returncode == 3, (
        f"expected exit 3 (fail-fast), got {result.returncode}\nstderr: {result.stderr}"
    )
    assert "appears" in result.stderr or "unreachable" in result.stderr


def test_no_fail_fast_keeps_going(tmp_path):
    """`--no-fail-fast` should NOT abort even when the server is down.

    The sweep will still run to completion; every row is an error row.
    Exit code 0 because pass_n == 0 isn't itself a failure signal.
    """
    out_dir = tmp_path / "snapshots"
    result = _run_cli_passthrough(
        "--sweep",
        "--name",
        "no-ff-smoke",
        "--base-url",
        "http://127.0.0.1:1",
        "--model",
        "mock-model",
        "--no-fail-fast",
        "--questions",
        "1",
        "--timeout",
        "5",
        "--out-dir",
        str(out_dir),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # All four area files exist with error rows.
    snap = out_dir / "no-ff-smoke"
    for area in ("ds4-eval", "code", "longctx", "agent"):
        data = json.loads((snap / f"{area}.json").read_text())
        assert data["n"] == 1
        assert data["pass"] == 0
        assert data["rows"][0]["error"]
