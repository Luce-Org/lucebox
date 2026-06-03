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


# ── multi-turn replay loader + prefill-and-decode verifier ──────────


def test_multi_turn_fixture_loads_when_present():
    """The harvested multi-turn fixture ships in-tree; loader returns
    bucketed cases sorted ascending by target bucket."""
    cases = agent_recorded.load_agent_recorded_multi_turn_cases()
    if not cases:
        # If the fixture wasn't generated (e.g. a downstream consumer
        # snapshot without --multi-turn) the loader returns []. Don't
        # fail packaging tests on that; the harvested fixture is
        # expected to be present in this branch.
        return
    assert len(cases) >= 1
    targets = [c["target_bucket_tokens"] for c in cases]
    assert targets == sorted(targets), "loader must return cases sorted by bucket"
    for c in cases:
        assert c["kind"] == "multi-turn-replay"
        assert c["verifier"]["type"] == "prefill-and-decode"
        assert isinstance(c["messages"], list) and c["messages"], c["id"]
        # Contract from the extractor: every case fits under its bucket.
        assert c["context_tokens_approx"] <= c["target_bucket_tokens"], c["id"]


def test_pick_multi_turn_case_for_budget():
    cases = [
        {"id": "small", "context_tokens_approx": 7800, "target_bucket_tokens": 8192},
        {"id": "med", "context_tokens_approx": 31000, "target_bucket_tokens": 32768},
        {"id": "big", "context_tokens_approx": 130000, "target_bucket_tokens": 131072},
    ]
    # Plenty of budget — should pick the largest fitting (default
    # safety_factor=0.7 → effective 140K; big at 130K fits).
    assert agent_recorded.pick_multi_turn_case_for_budget(cases, 200_000)["id"] == "big"
    # Mid budget — 60K * 0.7 = 42K effective; small (7800) and med
    # (31000) fit, big (130000) doesn't. Picker returns the larger.
    assert agent_recorded.pick_multi_turn_case_for_budget(cases, 60_000)["id"] == "med"
    # Below the smallest — returns None.
    assert agent_recorded.pick_multi_turn_case_for_budget(cases, 1_000) is None


def test_pick_multi_turn_case_for_budget_safety_factor_excludes_near_ceiling():
    """Regression for the 2026-05-30 gemma4 sweep failure: a case whose
    ``context_tokens_approx`` is just under the raw prompt_budget
    overshoots the real tokenizer count by ~40% and trips the server's
    effort-tier ceiling. The default 0.7 safety_factor must reject it."""
    cases = [
        {"id": "small", "context_tokens_approx": 64000, "target_bucket_tokens": 65536},
        # ``approx=102000`` ≤ raw budget 126976, but real tokens ≈ 140K
        # would exceed; safety_factor must keep this case OUT.
        {"id": "borderline", "context_tokens_approx": 102000, "target_bucket_tokens": 102400},
    ]
    # raw_budget=126976 → effective=88883. small (64K) fits; borderline (102K) does not.
    picked = agent_recorded.pick_multi_turn_case_for_budget(cases, 126976)
    assert picked["id"] == "small", f"safety_factor must exclude the borderline case, picked {picked}"

    # Operator can override the safety factor — e.g. when the fixture
    # has been re-tokenized accurately and the approx is the truth.
    picked2 = agent_recorded.pick_multi_turn_case_for_budget(
        cases, 126976, safety_factor=1.0
    )
    assert picked2["id"] == "borderline", "safety_factor=1.0 should allow borderline"


def test_grade_prefill_and_decode_pass():
    row = {"content": "Sure, here's a plan: read the file, edit, run tests.", "wall_s": 12.3}
    g = agent_recorded.grade_prefill_and_decode(row)
    assert g["pass"] is True
    assert "ok" in g["reason"]


def test_grade_prefill_and_decode_fails_on_error():
    row = {"content": "", "error": "HTTP 500: out of memory"}
    g = agent_recorded.grade_prefill_and_decode(row)
    assert g["pass"] is False
    assert "server error" in g["reason"]


def test_grade_prefill_and_decode_fails_on_wall_budget():
    row = {"content": "ok", "wall_s": 600.0}
    g = agent_recorded.grade_prefill_and_decode(row, max_wall_seconds=300.0)
    assert g["pass"] is False
    assert "wall" in g["reason"]


def test_grade_prefill_and_decode_falls_back_to_reasoning_content():
    """Thinking-mode response with empty content but populated
    reasoning_content still passes — the verifier is about the server
    serving N tokens, not where in the response shape the model put
    them."""
    row = {"content": "", "reasoning_content": "I'd start by reading the relevant file.", "wall_s": 5.0}
    g = agent_recorded.grade_prefill_and_decode(row, min_response_chars=10)
    assert g["pass"] is True


# ── call:<verb>{} structured-tool-call recognition ──────────────────


def test_tool_mentioned_picks_up_hyphenated_call_verb():
    """Regression for the 2026-05-30 gemma sweep: model emitted
    ``call:execute-bead:read-file{path:...}`` which is real tool
    engagement but the original grader missed it because no synonym
    list contained ``read-file``."""
    text = (
        'call:execute-bead:read-file{path: "crates/foo/src/lib.rs"}\n\n'
        'call:execute-bead:list-files{path: "src/"}\n\n'
    )
    assert agent_recorded._tool_mentioned(text, "Read") is True, (
        "verb 'read-file' from a call:<ns>:<verb>{} emission should match Read"
    )
    assert agent_recorded._tool_mentioned(text, "Glob") is True, (
        "verb 'list-files' should match Glob"
    )


def test_tool_mentioned_picks_up_snake_case_call_verb():
    """The snake_case variant (read_file / list_files) is also common."""
    text = 'call:exec-bead:read_file{path:"foo.txt"} call:exec-bead:list_files{}'
    assert agent_recorded._tool_mentioned(text, "Read") is True
    assert agent_recorded._tool_mentioned(text, "Glob") is True


def test_tool_mentioned_call_verb_with_no_namespace_prefix():
    """A bare ``call:read_file{}`` without namespace prefix still matches."""
    text = 'call:read_file{path: "foo.txt"}'
    assert agent_recorded._tool_mentioned(text, "Read") is True


def test_tool_mentioned_call_verb_is_case_insensitive_on_verb():
    """Verbs come back lowercased in the call: extractor regardless of
    how the model spelled them."""
    text = 'call:execute-bead:Read-File{path: "foo.txt"}'
    assert agent_recorded._tool_mentioned(text, "Read") is True


def test_tool_mentioned_unrelated_verb_still_misses():
    """The expander should not give false positives on unrelated verbs."""
    text = 'call:execute-bead:dance{}'
    assert agent_recorded._tool_mentioned(text, "Read") is False
    assert agent_recorded._tool_mentioned(text, "Bash") is False


def test_grade_full_pass_with_call_verb_emission():
    """End-to-end: a response built entirely of call:<verb>{} lines
    (no plain English) lands in the pass bin when verbs cover the
    expected tools and an expected file is named."""
    case = _case(["Read", "Glob"], files=["lucebox.sh"])
    row = {
        "content": (
            'call:execute-bead:list-files{path: "."}\n\n'
            'call:execute-bead:read-file{path: "lucebox.sh"}\n'
        )
    }
    g = agent_recorded.grade_agent_recorded_case(case, row)
    assert g["bin"] == "pass", g
    assert "Read" in g["tools_hit"]
    assert "Glob" in g["tools_hit"]
    assert "lucebox.sh" in g["files_hit"]
