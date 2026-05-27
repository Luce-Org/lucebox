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
