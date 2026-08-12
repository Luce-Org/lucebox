"""Unit tests for the ``agent_recorded`` + ``agent_recorded_v1`` areas.

Covers:

* Legacy v1 area (single-turn tool-schema coverage):
  - Fixture loads, every case has the expected schema.
  - Area registered in AREAS as ``agent_recorded_v1``.
  - Three-bin grader (pass / partial / fail) maps the obvious
    positive/negative inputs to the right verdicts.
* New multi-turn area:
  - Fixture loads with v1 schema (carries reference_response).
  - Mocked judge round-trips through the area runner end-to-end.
  - Cache speedup arithmetic.
* LLM judge:
  - Disk cache hit/miss with mocked client.
  - Verdict parsing (strict JSON + tolerant Markdown fence).
  - Missing API key path raises JudgeUnavailable.

No live server, no live judge — every API call is mocked.
"""

from __future__ import annotations

from lucebench.areas import agent_recorded
from lucebench.cli import AREAS


def test_agent_recorded_v1_cases_load_non_empty():
    cases = agent_recorded.load_agent_recorded_v1_cases()
    assert len(cases) > 0, "fixture must ship with at least one case"
    # The collector caps the scan but we want a meaningful suite.
    assert len(cases) >= 5, f"expected >=5 cases, got {len(cases)}"


def test_agent_recorded_v1_case_schema():
    cases = agent_recorded.load_agent_recorded_v1_cases()
    for c in cases:
        assert c["area"] == "agent_recorded_v1"
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


def test_agent_recorded_v1_registered_in_areas():
    assert "agent_recorded_v1" in AREAS
    cfg = AREAS["agent_recorded_v1"]
    assert cfg["default_max_tokens"] == 4096
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def test_agent_recorded_new_area_registered():
    """The new multi-turn area is registered as ``agent_recorded``."""
    assert "agent_recorded" in AREAS
    cfg = AREAS["agent_recorded"]
    # Different defaults than v1 (the judge cost dominates so we cap
    # decode tokens lower).
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    # The grade entry is a stub (real grading happens in
    # run_agent_recorded_area); it should still be callable.
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
        # v1 schema: cache-and-quality. v0 fixtures (legacy) used
        # prefill-and-decode; we accept either to keep loaders working
        # against snapshots produced by older extractors.
        assert c["verifier"]["type"] in ("cache-and-quality", "prefill-and-decode")
        assert isinstance(c["messages"], list) and c["messages"], c["id"]
        # Contract from the extractor: every case fits under its bucket.
        assert c["context_tokens_approx"] <= c["target_bucket_tokens"], c["id"]
        # v1 schema also requires the messages list to end on a user
        # turn (so the model under test is generating the next assistant
        # response). Tolerate v0 (no reference_response) by skipping the
        # tail-role check for those.
        if c.get("reference_response"):
            assert c["messages"][-1]["role"] == "user", c["id"]
            assert isinstance(c["reference_response"], str)
            assert c["reference_response"].strip(), c["id"]


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


# ── New multi-turn area: runner + judge integration ──────────────────


def test_multi_turn_runner_with_mocked_judge_and_http(tmp_path, monkeypatch):
    """Drive run_agent_recorded_area end-to-end with both the HTTP call
    and the judge stubbed. Verifies sequential growing-prefix replay,
    cache-hit accounting, and the row schema the operator sees."""
    from lucebench.areas import agent_recorded as ar
    from lucebench.grading.llm_judge import JudgeVerdict

    responses = [
        {
            "choices": [{"message": {"content": "Turn one response."}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 50,
                "timings": {
                    "prefill_ms": 3500.0,
                    "decode_ms": 2400.0,
                    "decode_tokens_per_sec": 20.8,
                    "prompt_n_cached": 0,
                },
            },
        },
        {
            "choices": [{"message": {"content": "Turn two response."}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5600,
                "completion_tokens": 50,
                "timings": {
                    "prefill_ms": 500.0,
                    "decode_ms": 2400.0,
                    "decode_tokens_per_sec": 20.8,
                    "prompt_n_cached": 5100,
                },
            },
        },
        {
            "choices": [{"message": {"content": "Final response, here's my plan."}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 8000,
                "completion_tokens": 50,
                "timings": {
                    "prefill_ms": 200.0,
                    "decode_ms": 2400.0,
                    "decode_tokens_per_sec": 20.8,
                    "prompt_n_cached": 7950,
                },
            },
        },
    ]

    seen_message_counts: list[int] = []

    def fake_post(*, url, model, messages, max_tokens, auth_header="", timeout_s=600):
        seen_message_counts.append(len(messages))
        raw = responses[len(seen_message_counts) - 1]
        return raw, [5.9, 2.8, 2.6][len(seen_message_counts) - 1]

    monkeypatch.setattr(ar, "_http_post_chat_completions", fake_post)
    monkeypatch.setattr(ar, "_restart_lucebox_service", lambda **kw: (False, "test"))
    # When restart fails we fall back to warm/warm; but we still want
    # to test the cold/warm path. Pretend restart works.
    monkeypatch.setattr(ar, "_restart_lucebox_service", lambda **kw: (True, "ok"))
    monkeypatch.setattr(ar, "_health_wait", lambda url, timeout_s=120: True)

    # Mock the judge: pass on cases mentioning "plan", fail otherwise.
    def fake_judge(*, case_id, messages, reference_response, candidate_response):
        passed = "plan" in candidate_response.lower()
        return JudgeVerdict(
            passed=passed,
            verdict="suitable_match" if passed else "off_track",
            rationale="mocked",
            cached=False,
            model="mock",
        )

    # Build a one-case fake fixture so the runner doesn't depend on the
    # shipped 48-case fixture in this unit test.
    fake_fixture = tmp_path / "multi_turn_cases.json"
    import json
    fake_fixture.write_text(json.dumps({
        "schema": "lucebox-bench-agent-recorded-multi-turn-v1",
        "buckets": [8192],
        "cases": [
            {
                "id": "test-case-1",
                "source": "claude-code",
                "kind": "multi-turn-replay",
                "messages": [
                    {"role": "user", "content": "Help me debug this"},
                    {"role": "assistant", "content": "Sure, what error?"},
                    {"role": "user", "content": "It crashes on startup"},
                    {"role": "assistant", "content": "I would inspect the startup logs."},
                    {"role": "user", "content": "The logs mention a missing config file."},
                ],
                "reference_response": "Let me look at the startup sequence and grep for any obvious crash paths.",
                "context_tokens_approx": 7900,
                "target_bucket_tokens": 8192,
                "n_messages": 5,
                "initial_state": {},
                "verifier": {"type": "cache-and-quality"},
            }
        ],
    }))

    rows, summary = ar.run_agent_recorded_area(
        url="http://test.invalid",
        model="test-model",
        max_tokens=128,
        restart_between_cases=True,
        fixture_path=fake_fixture,
        mock_judge=fake_judge,
    )

    assert seen_message_counts == [1, 3, 5]
    assert len(rows) == 1
    row = rows[0]
    assert row["replay_mode"] == "sequential"
    assert row["case_id"] == "test-case-1"
    assert row["bucket_tokens"] == 8192
    assert row["restart"]["status"] == "ok"
    assert len(row["turns"]) == 3
    assert row["cold"]["prefill_s"] == 3.5
    assert row["warm"]["prefill_s"] == 0.2  # final turn alias
    assert row["turns"][1]["restore"] is True
    assert row["turns"][2]["restore"] is True
    assert row["cache_eligible_turns"] == 2
    assert row["cache_hit_turns"] == 2
    assert row["cache_speedup_x"] is None
    assert row["judge"]["pass"] is True
    assert row["judge"]["verdict"] == "suitable_match"
    assert row["pass"] is True

    assert summary["n"] == 1
    assert summary["n_judged"] == 1
    assert summary["n_pass"] == 1
    assert summary["pass_rate"] == 100.0
    assert summary["replay_mode"] == "sequential"
    assert summary["cache_speedup_p50"] is None
    assert summary["cache_hit_turns"] == 2
    assert summary["cache_eligible_turns"] == 2
    assert summary["cache_hit_rate"] == 100.0
    assert summary["cache_eligible_prefill_p50"] == 0.35
    assert summary["final_prefill_p50"] == 0.2


def test_multi_turn_runner_exact_repeat_mode_keeps_full_prompt_probe(tmp_path, monkeypatch):
    """The opt-in exact-repeat mode keeps the historical cold/warm probe."""
    from lucebench.areas import agent_recorded as ar
    from lucebench.grading.llm_judge import JudgeVerdict

    cold_response = {
        "choices": [{"message": {"content": "Cold response, here's my plan."}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 8000,
            "completion_tokens": 50,
            "timings": {
                "prefill_ms": 3500.0,
                "decode_ms": 2400.0,
                "decode_tokens_per_sec": 20.8,
                "prompt_n_cached": 0,
            },
        },
    }
    warm_response = {
        "choices": [{"message": {"content": "Warm response, same content."}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 8000,
            "completion_tokens": 50,
            "timings": {
                "prefill_ms": 200.0,
                "decode_ms": 2400.0,
                "decode_tokens_per_sec": 20.8,
                "prompt_n_cached": 7950,
            },
        },
    }

    seen_message_counts: list[int] = []

    def fake_post(*, url, model, messages, max_tokens, auth_header="", timeout_s=600):
        seen_message_counts.append(len(messages))
        raw = cold_response if len(seen_message_counts) == 1 else warm_response
        return raw, 5.9 if len(seen_message_counts) == 1 else 2.6

    monkeypatch.setattr(ar, "_http_post_chat_completions", fake_post)
    monkeypatch.setattr(ar, "_restart_lucebox_service", lambda **kw: (True, "ok"))
    monkeypatch.setattr(ar, "_health_wait", lambda url, timeout_s=120: True)

    def fake_judge(*, case_id, messages, reference_response, candidate_response):
        return JudgeVerdict(
            passed=True,
            verdict="suitable_match",
            rationale="mocked",
            cached=False,
            model="mock",
        )

    fake_fixture = tmp_path / "multi_turn_cases.json"
    import json
    fake_fixture.write_text(json.dumps({
        "schema": "lucebox-bench-agent-recorded-multi-turn-v1",
        "buckets": [8192],
        "cases": [
            {
                "id": "test-case-1",
                "source": "claude-code",
                "kind": "multi-turn-replay",
                "messages": [
                    {"role": "user", "content": "Help me debug this"},
                    {"role": "assistant", "content": "Sure, what error?"},
                    {"role": "user", "content": "It crashes on startup"},
                ],
                "reference_response": "Let me look at the startup sequence.",
                "context_tokens_approx": 7900,
                "target_bucket_tokens": 8192,
                "n_messages": 3,
                "initial_state": {},
                "verifier": {"type": "cache-and-quality"},
            }
        ],
    }))

    rows, summary = ar.run_agent_recorded_area(
        url="http://test.invalid",
        model="test-model",
        max_tokens=128,
        restart_between_cases=True,
        fixture_path=fake_fixture,
        mock_judge=fake_judge,
        replay_mode="exact-repeat",
    )

    assert seen_message_counts == [3, 3]
    assert rows[0]["replay_mode"] == "exact-repeat"
    assert rows[0]["cache_speedup_x"] == 17.5
    assert rows[0]["warm"]["restore"] is True
    assert summary["replay_mode"] == "exact-repeat"
    assert summary["cache_speedup_p50"] == 17.5
    assert summary["cache_eligible_turns"] == 1
    assert summary["cache_hit_turns"] == 1
    assert summary["cache_hit_rate"] == 100.0
    assert summary["cache_eligible_prefill_p50"] == 0.2
    assert summary["cache_hit_prefill_p50"] == 0.2
    assert summary["prefix_len_p50"] == 7950


def test_prefill_speedup_uses_one_millisecond_floor_for_zero_warm_prefill():
    """A fully restored prefill can round to 0ms; report a lower-bound speedup."""
    assert agent_recorded._prefill_speedup({"prefill_s": 3.5}, {"prefill_s": 0.0}) == 3500.0


def test_multi_turn_runner_marks_judge_pending_without_reference(tmp_path, monkeypatch):
    """A v0 fixture case without ``reference_response`` returns
    ``judge_pending`` rather than failing — quality wasn't checkable."""
    from lucebench.areas import agent_recorded as ar

    def fake_post(*, url, model, messages, max_tokens, auth_header="", timeout_s=600):
        return ({
            "choices": [{"message": {"content": "fine"}}],
            "usage": {"completion_tokens": 1, "timings": {"prefill_ms": 100, "decode_ms": 50}},
        }, 0.5)

    monkeypatch.setattr(ar, "_http_post_chat_completions", fake_post)
    monkeypatch.setattr(ar, "_restart_lucebox_service", lambda **kw: (True, "ok"))
    monkeypatch.setattr(ar, "_health_wait", lambda url, timeout_s=120: True)

    import json
    fake_fixture = tmp_path / "no_ref.json"
    fake_fixture.write_text(json.dumps({
        "schema": "lucebox-bench-agent-recorded-multi-turn-v0",
        "cases": [{
            "id": "v0-case",
            "source": "claude-code",
            "kind": "multi-turn-replay",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens_approx": 100,
            "target_bucket_tokens": 8192,
            "n_messages": 1,
        }],
    }))

    # Judge should NEVER be called for a case lacking reference_response.
    def boom(**kw):  # pragma: no cover - asserted by absence of call
        raise AssertionError("judge should not be invoked without a reference")

    rows, summary = ar.run_agent_recorded_area(
        url="http://test.invalid", model="test",
        fixture_path=fake_fixture, mock_judge=boom,
    )
    assert len(rows) == 1
    assert rows[0]["judge"]["verdict"] == "judge_pending"
    assert rows[0]["pass"] is False
    # Pass rate denominator excludes judge_pending → pass_rate = 0 of 0 → 0.0.
    assert summary["n_judged"] == 0


# ── LLM judge unit tests (mocked SDK client) ─────────────────────────


def test_judge_parses_strict_json(tmp_path):
    """The judge parses a strict-JSON verdict and returns the expected
    JudgeVerdict; the cache file lands at the expected path."""
    from lucebench.grading import llm_judge

    class _FakeBlock:
        def __init__(self, text):
            self.text = text

    class _FakeResp:
        def __init__(self, text):
            self.content = [_FakeBlock(text)]

    class _FakeMessagesAPI:
        def create(self, **kwargs):
            return _FakeResp('{"verdict": "suitable_match", "rationale": "matches well"}')

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessagesAPI()

    v = llm_judge.judge_response(
        case_id="t1",
        messages=[{"role": "user", "content": "x"}],
        reference_response="abc",
        candidate_response="xyz",
        cache_root=tmp_path,
        _client_factory=_FakeClient,
    )
    assert v.passed is True
    assert v.verdict == "suitable_match"
    assert v.cached is False

    # Re-run with same inputs hits the cache.
    v2 = llm_judge.judge_response(
        case_id="t1",
        messages=[{"role": "user", "content": "x"}],
        reference_response="abc",
        candidate_response="xyz",
        cache_root=tmp_path,
        _client_factory=_FakeClient,
    )
    assert v2.cached is True
    assert v2.verdict == "suitable_match"


def test_judge_handles_markdown_fenced_json(tmp_path):
    """Judges sometimes wrap their JSON in ```json fences; the parser
    strips them rather than mapping to off_track."""
    from lucebench.grading import llm_judge

    class _FakeBlock:
        def __init__(self, text):
            self.text = text

    class _FakeResp:
        def __init__(self, text):
            self.content = [_FakeBlock(text)]

    class _FakeMessagesAPI:
        def create(self, **kwargs):
            return _FakeResp('```json\n{"verdict": "divergent_but_valid", "rationale": "ok"}\n```')

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessagesAPI()

    v = llm_judge.judge_response(
        case_id="t-fenced",
        messages=[{"role": "user", "content": "x"}],
        reference_response="a",
        candidate_response="b",
        cache_root=tmp_path,
        _client_factory=_FakeClient,
    )
    assert v.passed is True  # divergent_but_valid maps to pass
    assert v.verdict == "divergent_but_valid"


def test_judge_off_track_fails(tmp_path):
    """off_track maps to fail."""
    from lucebench.grading import llm_judge

    class _FakeBlock:
        def __init__(self, text):
            self.text = text

    class _FakeResp:
        def __init__(self, text):
            self.content = [_FakeBlock(text)]

    class _FakeMessagesAPI:
        def create(self, **kwargs):
            return _FakeResp('{"verdict": "off_track", "rationale": "irrelevant"}')

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessagesAPI()

    v = llm_judge.judge_response(
        case_id="t-off",
        messages=[{"role": "user", "content": "x"}],
        reference_response="a",
        candidate_response="b",
        cache_root=tmp_path,
        _client_factory=_FakeClient,
    )
    assert v.passed is False
    assert v.verdict == "off_track"


def test_judge_malformed_response_maps_to_off_track(tmp_path):
    """A junk judge response biases against the candidate."""
    from lucebench.grading import llm_judge

    class _FakeBlock:
        def __init__(self, text):
            self.text = text

    class _FakeResp:
        def __init__(self, text):
            self.content = [_FakeBlock(text)]

    class _FakeMessagesAPI:
        def create(self, **kwargs):
            return _FakeResp("I'm not going to answer that.")

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessagesAPI()

    v = llm_judge.judge_response(
        case_id="t-junk",
        messages=[{"role": "user", "content": "x"}],
        reference_response="a",
        candidate_response="b",
        cache_root=tmp_path,
        _client_factory=_FakeClient,
    )
    assert v.passed is False
    assert v.verdict == "off_track"


def test_judge_unavailable_when_no_key_and_no_factory(monkeypatch, tmp_path):
    """Without ANTHROPIC_API_KEY and no _client_factory, judge raises
    JudgeUnavailable so the area can surface a clear error."""
    from lucebench.grading import llm_judge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        llm_judge.judge_response(
            case_id="t-noenv",
            messages=[{"role": "user", "content": "x"}],
            reference_response="a",
            candidate_response="b",
            cache_root=tmp_path,
        )
    except llm_judge.JudgeUnavailable as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:  # pragma: no cover - test should always raise
        raise AssertionError("expected JudgeUnavailable")


def test_extract_row_from_response_includes_cache_signal():
    """The row extractor pulls prefill/decode/prefix_len from the
    server's ``usage.timings`` block."""
    from lucebench.areas import agent_recorded as ar
    raw = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "completion_tokens": 10,
            "timings": {"prefill_ms": 1200, "decode_ms": 500, "prompt_n_cached": 3000},
        },
    }
    row = ar._extract_row_from_response(raw, wall=1.8)
    assert row["prefill_s"] == 1.2
    assert row["decode_s"] == 0.5
    assert row["prefix_len"] == 3000
    assert row["restore"] is True
    assert row["content"] == "ok"
