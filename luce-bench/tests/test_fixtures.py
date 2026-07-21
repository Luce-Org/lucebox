"""Smoke tests — fixtures load and basic graders behave as expected.

No network. No model. These guard against packaging accidents (missing
fixture data, broken imports).
"""

from __future__ import annotations

from lucebench import __version__
from lucebench.areas import ds4_eval, humaneval


def test_version_exposed():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1


def test_ds4_eval_cases_load():
    cases = ds4_eval.load_ds4_eval_cases()
    assert len(cases) >= 90, f"expected ~92 ds4-eval cases, got {len(cases)}"
    sources = {c["source"] for c in cases}
    # All four canonical ds4 source families should be present.
    assert {"GPQA Diamond", "SuperGPQA", "AIME2025", "COMPSEC"} <= sources, sources


def test_ds4_eval_cases_have_required_fields():
    for c in ds4_eval.load_ds4_eval_cases():
        assert "id" in c, c
        assert "source" in c, c
        assert "answer" in c, c
        assert "kind" in c, c


def test_humaneval_cases_load():
    cases = humaneval.load_humaneval_cases()
    assert len(cases) >= 5, f"expected at least 5 HumanEval cases, got {len(cases)}"
    for c in cases:
        assert c["area"] == "code"
        assert c["kind"] == "code-completion"
        assert "prompt" in c


def test_grade_ds4_choice_pass():
    case = {
        "id": "x",
        "source": "GPQA Diamond",
        "kind": "choice",
        "answer": "B",
        "choices": ["A1", "A2", "A3", "A4"],
    }
    row = {"content": "I think the answer is B because...\nAnswer: B"}
    g = ds4_eval.grade_case(case, row)
    assert g["pass"] is True
    assert g["given"] == "B"


def test_grade_ds4_integer_fail():
    case = {"id": "x", "source": "AIME2025", "kind": "integer", "answer": 42}
    row = {"content": "After computing it carefully, Answer: 41"}
    g = ds4_eval.grade_case(case, row)
    assert g["pass"] is False
    assert g["given"] == "41"


def test_grade_ds4_integer_pass():
    case = {"id": "x", "source": "AIME2025", "kind": "integer", "answer": 42}
    row = {"content": "Step 1...\nStep 2...\nAnswer: 42"}
    g = ds4_eval.grade_case(case, row)
    assert g["pass"] is True


def test_grade_ds4_format_error():
    case = {"id": "x", "source": "AIME2025", "kind": "integer", "answer": 42}
    row = {"content": "no useful content"}
    g = ds4_eval.grade_case(case, row)
    # No integer in text at all -> format error.
    assert g["pass"] is False
    assert g["status"] == "format_error"
    assert g["given"] == "?"


def test_grade_ds4_integer_unmarked_permissive():
    # antirez/ds4's find_ds4_integer_answer falls back to the last
    # integer in text if there's no "Answer:" marker. That's intentional —
    # tracks here so any future tightening doesn't silently regress.
    case = {"id": "x", "source": "AIME2025", "kind": "integer", "answer": 42}
    row = {"content": "I think it's 42 but I'm not sure."}
    g = ds4_eval.grade_case(case, row)
    assert g["given"] == "42"
    assert g["pass"] is True


def test_grade_humaneval_pass():
    case = {"id": "x", "area": "code", "kind": "code-completion", "prompt": "def add(a, b):\n    "}
    row = {"content": "return a + b\n"}
    g = humaneval.grade_humaneval_case(case, row)
    assert g["pass"] is True
    assert g["given"] == "parse_ok"


def test_grade_humaneval_fail():
    case = {"id": "x", "area": "code", "kind": "code-completion", "prompt": "def add(a, b):\n    "}
    row = {"content": "this is not python @@"}
    g = humaneval.grade_humaneval_case(case, row)
    assert g["pass"] is False


def test_longctx_cases_load():
    from lucebench.areas import longctx

    assert len(longctx.LONGCTX_CASES) >= 5
    for c in longctx.LONGCTX_CASES:
        assert c["kind"] == "longctx-frontier"
        assert "prompt" in c
        assert "target_tokens" in c


def test_agent_cases_load():
    from lucebench.areas import agent

    cases = agent.load_agent_cases()
    assert len(cases) >= 1
    for c in cases:
        assert c["kind"] == "agent-prompt"
        # system_prompt should be loaded from disk fixture
        assert c.get("system_prompt"), c
        assert c.get("user_message"), c


def test_grade_longctx_pass():
    from lucebench.areas import longctx

    case = {"id": "x", "kind": "longctx-frontier", "prompt": "irrelevant", "target_tokens": 2048}
    row = {"content": "Risk: the haystack contains nothing actionable."}
    g = longctx.grade_longctx_case(case, row)
    assert g["pass"] is True


def test_grade_longctx_fail():
    from lucebench.areas import longctx

    case = {"id": "x", "kind": "longctx-frontier", "prompt": "irrelevant", "target_tokens": 2048}
    row = {"content": "I think the risk is..."}  # missing "Risk:" prefix
    g = longctx.grade_longctx_case(case, row)
    assert g["pass"] is False


def test_grade_agent_codeblock_pass():
    from lucebench.areas import agent

    case = {
        "id": "x",
        "kind": "agent-prompt",
        "system_prompt": "be an agent",
        "user_message": "fix foo.py",
    }
    row = {"content": "Here's a fix:\n```python\nprint('hi')\n```"}
    g = agent.grade_agent_case(case, row)
    assert g["pass"] is True


# ────────────────────────────────────────────────────────────────────
# Sweep helpers (v0.2.1): fail-fast detection + forge availability.
# These are pure functions so they don't need an HTTP fixture; the
# runner-level integration tests live in v0.2.2's test_runner.py.
# ────────────────────────────────────────────────────────────────────


def test_row_is_unreachable_connection_refused():
    from lucebench.cli import _row_is_unreachable

    row = {"error": "ConnectionRefusedError: [Errno 111] Connection refused"}
    assert _row_is_unreachable(row) is True


def test_row_is_unreachable_dns_failure():
    from lucebench.cli import _row_is_unreachable

    row = {"error": "URLError: <urlopen error [Errno -2] Name or service not known>"}
    assert _row_is_unreachable(row) is True


def test_row_is_unreachable_timeout_not_unreachable():
    """Timeouts are per-request failures, NOT 'server down' signals."""
    from lucebench.cli import _row_is_unreachable

    row = {"error": "TimeoutError: timed out"}
    assert _row_is_unreachable(row) is False


def test_row_is_unreachable_http_500_not_unreachable():
    from lucebench.cli import _row_is_unreachable

    row = {"error": "HTTPError: HTTP Error 500: Internal Server Error"}
    assert _row_is_unreachable(row) is False


def test_row_is_unreachable_no_error_field():
    from lucebench.cli import _row_is_unreachable

    assert _row_is_unreachable({}) is False
    assert _row_is_unreachable({"error": None}) is False
    assert _row_is_unreachable({"error": ""}) is False


def test_forge_available_returns_two_tuple():
    """Whether anthropic is installed or not, the API shape is stable."""
    from lucebench.cli import _forge_available

    ok, reason = _forge_available()
    assert isinstance(ok, bool)
    if ok:
        assert reason is None
    else:
        assert isinstance(reason, str) and "anthropic" in reason.lower()


# ────────────────────────────────────────────────────────────────────
# _preflight /props display — pinned against the lucebox /props schema
# (see ../../docs/specs/props-endpoint.md). Tests stub urlopen so they
# don't need a live server.
# ────────────────────────────────────────────────────────────────────


def _stub_preflight_urlopen(monkeypatch, models_payload, props_payload):
    """Wire `urllib.request.urlopen` to return models then props bodies."""
    import io
    import json as _json
    import urllib.request
    from contextlib import contextmanager

    queue = [models_payload, props_payload]

    @contextmanager
    def fake_urlopen(req, timeout=0):
        # Each call pops the next body. The preflight calls /v1/models
        # first, then /props — order matters.
        body = queue.pop(0)
        yield io.BytesIO(_json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_preflight_props_schema3_image_and_target_displayed(monkeypatch):
    """Schema 3 server: preflight surfaces image=<tag>@<sha7> and target=<basename>."""
    from lucebench.cli import _preflight

    models = {"data": [{"id": "dflash"}]}
    props = {
        "build": {
            "server_name": "luce-dflash",
            "server_version": "0.0.0+cpp",
            "props_schema": 3,
            "git_sha": "6d12378abcdef01234567890abcdef0123456789",
            "image_tag": "cuda12",
            "image_digest": None,
            "build_time": "2026-05-28T13:43:57Z",
        },
        "model": {
            "arch": "qwen35",
            "alias": "dflash",
            "target": {
                "path": "/opt/models/Qwen3.6-27B-Q4_K_M.gguf",
                "size_bytes": 17134510080,
                "sha256": "a" * 64,
                "gguf": {"general.architecture": "qwen35"},
            },
            "draft": None,
        },
        "budget_envelope": {
            "model_card_source": "share/model_cards/qwen3.6-27b.json",
            "hard_limit_reply_budget": 4096,
        },
    }
    _stub_preflight_urlopen(monkeypatch, models, props)
    ok, lines, _server_honors, _card = _preflight("http://localhost:8080")
    assert ok is True
    # Find the /props line.
    props_line = next(line for line in lines if "/props" in line)
    # Image identity: `image=cuda12@6d12378`.
    assert "image=cuda12@6d12378" in props_line, props_line
    # Target basename, .gguf-stripped.
    assert "target=Qwen3.6-27B-Q4_K_M" in props_line, props_line
    # Existing budget_envelope hits still surfaced.
    assert "model_card=share/model_cards/qwen3.6-27b.json" in props_line
    assert "reply_budget=4096" in props_line


def test_preflight_props_schema2_falls_back(monkeypatch):
    """Pre-schema-3 server: no `build` / `model.target` → fallback display."""
    from lucebench.cli import _preflight

    models = {"data": [{"id": "dflash"}]}
    props = {
        # No `build` block — schema 2.
        "model": {"arch": "qwen35"},
        "budget_envelope": {
            "model_card_source": "share/model_cards/qwen3.6-27b.json",
            "hard_limit_reply_budget": 4096,
        },
    }
    _stub_preflight_urlopen(monkeypatch, models, props)
    ok, lines, _server_honors, _card = _preflight("http://localhost:8080")
    assert ok is True
    props_line = next(line for line in lines if "/props" in line)
    # No image=/target= bits when those fields are absent.
    assert "image=" not in props_line
    assert "target=" not in props_line
    # Schema-2 bits still present.
    assert "model_card=share/model_cards/qwen3.6-27b.json" in props_line
    assert "reply_budget=4096" in props_line


def test_preflight_props_build_image_fields_null(monkeypatch):
    """Bare-metal build: `build` block present but image_* are null.

    The preflight line should NOT carry an `image=` token in that case —
    showing `image=None@None` would be worse than just hiding the bit.
    """
    from lucebench.cli import _preflight

    models = {"data": [{"id": "dflash"}]}
    props = {
        "build": {
            "server_name": "luce-dflash",
            "server_version": "0.0.0+cpp",
            "props_schema": 3,
            "git_sha": None,
            "image_tag": None,
            "image_digest": None,
            "build_time": None,
        },
        "model": {"arch": "qwen35"},
        "budget_envelope": {},
    }
    _stub_preflight_urlopen(monkeypatch, models, props)
    ok, lines, _server_honors, _card = _preflight("http://localhost:8080")
    assert ok is True
    props_line = next(line for line in lines if "/props" in line)
    assert "image=" not in props_line
