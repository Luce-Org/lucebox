"""Unit tests for the ``agent_recorded`` area.

Covers:

* Fixture loads, is non-empty, and every case has the expected schema
  (id, prompt, verifier with type + expected_tools).
* The area is registered in the CLI's AREAS dict with the documented
  defaults.
* The three-bin grader (pass / partial / fail) maps the obvious
  positive/negative inputs to the right verdicts. Includes a refusal
  case, a stub case, and a "tool but no file" partial.

No live server. Each grader call is fed a canned ``row["content"]``.
"""

from __future__ import annotations

from lucebench.areas import agent_recorded
from lucebench.cli import AREAS


def test_agent_recorded_cases_load_non_empty():
    cases = agent_recorded.load_agent_recorded_cases()
    assert len(cases) > 0, "fixture must ship with at least one case"
    # The collector caps the scan but we want a meaningful suite.
    assert len(cases) >= 5, f"expected >=5 cases, got {len(cases)}"


def test_agent_recorded_case_schema():
    cases = agent_recorded.load_agent_recorded_cases()
    for c in cases:
        assert c["area"] == "agent_recorded"
        assert c["kind"] == "agent-prompt"
        assert c["prompt"], "every case needs a non-empty prompt"
        assert c["user_message"] == c["prompt"]
        v = c["verifier"]
        assert v["type"] == "tool-schema-coverage"
        assert isinstance(v["expected_tools"], list)
        assert v["expected_tools"], f"{c['id']}: expected_tools should be non-empty"
        assert isinstance(v["expected_files_touched"], list)
        # Source must round-trip the collector's label set.
        assert c["source"] in ("agent-recorded-claude-code", "agent-recorded-codex")


def test_agent_recorded_registered_in_areas():
    assert "agent_recorded" in AREAS
    cfg = AREAS["agent_recorded"]
    assert cfg["default_max_tokens"] == 4096
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def _case(tools, files=None):
    return {
        "id": "test-case",
        "verifier": {
            "type": "tool-schema-coverage",
            "expected_tools": list(tools),
            "expected_files_touched": list(files or []),
            "min_tool_calls": 2,
        },
    }


def test_grade_pass_when_tool_and_file_named():
    """Coherent reply that names both an expected tool and an expected file
    lands in the ``pass`` bin."""
    case = _case(["Edit", "Bash"], files=["lucebox.sh"])
    row = {
        "content": (
            "I would use the Edit tool to modify lucebox.sh, replacing the "
            "current systemd unit definition with the new template. Then "
            "run a Bash command to reload the daemon."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["pass"] is True
    assert g["bin"] == "pass"
    assert g["status"] == "passed"
    assert "Edit" in g["tools_hit"]
    assert "lucebox.sh" in g["files_hit"]


def test_grade_partial_tool_but_no_file():
    """Names a tool but doesn't name any expected file → ``partial``."""
    case = _case(["Edit", "Bash"], files=["lucebox.sh"])
    row = {
        "content": (
            "I'd start by reading the existing systemd setup, then use Edit "
            "to apply the necessary changes. Specific paths depend on the "
            "current layout I'd inspect first."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["pass"] is False
    assert g["bin"] == "partial"
    assert g["status"] == "partial"
    assert "Edit" in g["tools_hit"]
    assert g["files_hit"] == []


def test_grade_partial_file_but_no_tool():
    """Names the file but mentions no expected tool → ``partial``."""
    case = _case(["Edit"], files=["lucebox.sh"])
    row = {
        "content": (
            "The right place for the change is lucebox.sh — but I'd want to "
            "see the current contents first before describing a concrete "
            "plan. Could you share that file?"
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["bin"] == "partial"


def test_grade_fail_when_refused():
    """A refusal at the head of the reply forces ``fail`` even when the
    follow-up text happens to mention an expected tool."""
    case = _case(["Edit"], files=["lucebox.sh"])
    row = {
        "content": (
            "I can't help with this request. Edit lucebox.sh — though I'm "
            "not going to walk through how."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["pass"] is False
    assert g["bin"] == "fail"
    assert g["given"] == "refused"


def test_grade_fail_when_stub():
    """Short reply (< 80 chars) is a stub regardless of content."""
    case = _case(["Edit"], files=["lucebox.sh"])
    row = {"content": "Use Edit on lucebox.sh."}  # short
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["pass"] is False
    assert g["bin"] == "fail"
    assert g["given"] == "stub"


def test_grade_fail_when_off_topic():
    """Coherent but mentions nothing the verifier expects → ``fail``."""
    case = _case(["Edit"], files=["lucebox.sh"])
    row = {
        "content": (
            "That's an interesting question about systems engineering. "
            "Generally, you'd want to think about idempotency, observability, "
            "and the blast radius of any change. Let me know more about your "
            "goals."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["pass"] is False
    assert g["bin"] == "fail"


def test_grade_synonym_match_for_bash():
    """The loose Bash synonym list — \"shell command\" counts."""
    case = _case(["Bash"], files=[])
    row = {
        "content": (
            "I would run a shell command to inspect the current state, then "
            "decide whether to apply a patch or revert. The first step is "
            "checking git status."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert "Bash" in g["tools_hit"]
    assert g["bin"] == "pass"  # no files expected → tools alone is enough


def test_grade_empty_files_list_grades_on_tools_only():
    """When the verifier ships no expected files (codex sessions where
    we couldn't recover patch paths), the file-axis check is skipped."""
    case = _case(["Edit", "Bash"], files=[])
    row = {
        "content": (
            "First step is to Edit the relevant module — I'd locate it via "
            "ripgrep, read the surrounding context, and apply a minimal "
            "patch. Then run the test suite to confirm."
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["bin"] == "pass"
    assert g["file_coverage"] is None


def test_grade_reads_reasoning_content_fallback():
    """If a thinking-mode server emits the engagement in reasoning_content
    and leaves content empty, the grader still picks it up."""
    case = _case(["Edit"], files=["lucebox.sh"])
    row = {
        "content": "",
        "reasoning_content": (
            "Let me think about lucebox.sh. I would use Edit to add the "
            "new line, then verify with a Bash run of the test suite."
        ),
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["bin"] == "pass"
