"""Unit tests for small pure helpers extracted out of ``lucebench.cli``.

These cover branches that the subprocess-level smoke tests exercise only
indirectly (auth-header resolution, JSON-out envelope shape, terse-row
filtering) so regressions surface as fast, targeted failures.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lucebench.cli import (
    _AuthEnvEmpty,
    _preflight,
    _resolve_auth_header,
    _terse_rows,
    _write_area_json,
)


def _args(auth_env: str | None) -> SimpleNamespace:
    return SimpleNamespace(auth_env=auth_env)


def test_resolve_auth_header_unset_returns_empty():
    assert _resolve_auth_header(_args(None), on_empty="error") == ""
    assert _resolve_auth_header(_args(""), on_empty="ignore") == ""


def test_resolve_auth_header_with_token(monkeypatch):
    monkeypatch.setenv("LB_TEST_TOKEN", "sekret")
    args = _args("LB_TEST_TOKEN")
    assert _resolve_auth_header(args, on_empty="error") == "Bearer sekret"
    assert _resolve_auth_header(args, on_empty="ignore") == "Bearer sekret"


def test_resolve_auth_header_empty_error_mode_raises(monkeypatch):
    monkeypatch.delenv("LB_TEST_MISSING", raising=False)
    args = _args("LB_TEST_MISSING")
    with pytest.raises(_AuthEnvEmpty) as ei:
        _resolve_auth_header(args, on_empty="error")
    # Message names the offending env var so the CLI diagnostic stays useful.
    assert "LB_TEST_MISSING" in str(ei.value)


def test_resolve_auth_header_empty_ignore_mode_tolerated(monkeypatch):
    monkeypatch.setenv("LB_TEST_EMPTY", "")
    args = _args("LB_TEST_EMPTY")
    assert _resolve_auth_header(args, on_empty="ignore") == ""


def test_terse_rows_drops_heavy_keys():
    rows = [
        {
            "id": "q1",
            "content": "answer",
            "_response": {"big": "blob"},
            "_thinking_injection": {"echo": 1},
        }
    ]
    out = _terse_rows(rows)
    assert out == [{"id": "q1", "content": "answer"}]
    # Original rows are untouched (no in-place mutation).
    assert "_response" in rows[0]


def test_write_area_json_envelope(tmp_path):
    import json

    out = tmp_path / "code.json"
    _write_area_json(
        out,
        area="code",
        url="http://x/v1",
        model="m1",
        summary={"n": 2, "pass": 1},
        rows=[{"id": "q1"}],
    )
    data = json.loads(out.read_text())
    assert data["area"] == "code"
    assert data["url"] == "http://x/v1"
    assert data["model"] == "m1"
    # Summary keys are spliced in at the top level.
    assert data["n"] == 2
    assert data["pass"] == 1
    assert data["rows"] == [{"id": "q1"}]
    assert "lucebench_version" in data


def test_preflight_live_against_mock_server(mock_openai_server):
    """Preflight passes liveness + /v1/models against the conftest mock server.

    The mock server has no /props endpoint, so the soft /props check
    degrades gracefully (ok stays True, server_honors_api_flags False).
    """
    url, _captured, _ = mock_openai_server
    ok, lines, server_honors, card = _preflight(url, requested_model="mock-model")
    assert ok is True
    # mock-model is the only exposed id and should be the `*`-marked pick.
    models_line = next(line for line in lines if "/v1/models" in line)
    assert "*mock-model" in models_line
    # No /props on the mock server → soft-skip, no server-side enforcement.
    props_line = next(line for line in lines if "/props" in line)
    assert "absent" in props_line
    assert server_honors is False
    assert card is None
