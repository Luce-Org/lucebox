import sys
from types import SimpleNamespace

import pytest

from harness.validation_v2.cli import _pinned_tokenizer_counter
from harness.validation_v2.niah import generate_control_cases, generate_primary_cases


def test_primary_design_is_balanced_and_deterministic() -> None:
    counter = len
    first = generate_primary_cases(
        context_tokens=32768, tokenizer_revision="a" * 40, token_counter=counter
    )
    second = generate_primary_cases(
        context_tokens=32768, tokenizer_revision="a" * 40, token_counter=counter
    )
    assert len(first) == 100
    assert first == second
    assert len({case.case_id for case in first}) == 100
    assert {case.depth_fraction for case in first} == {0.05, 0.25, 0.5, 0.75, 0.95}
    assert len({case.prompt_sha256 for case in first}) == 100
    assert all(case.actual_context_tokens == 32768 for case in first)
    assert all(abs(case.actual_depth_fraction - case.depth_fraction) <= 0.02 for case in first)


def test_controls_cover_all_adversarial_families() -> None:
    controls = generate_control_cases(
        context_tokens=32768, tokenizer_revision="a" * 40, token_counter=len
    )
    assert len(controls) == 20
    assert {case.control for case in controls} == {
        "answer-absent",
        "misleading",
        "multiple",
        "prefix",
        "prompt-echo",
    }
    assert {case.answer for case in controls} == {"NONE"}
    assert all(case.actual_context_tokens == 32768 for case in controls)


def test_publication_generation_refuses_character_only_estimates() -> None:
    with pytest.raises(ValueError, match="actual tokenizer counter"):
        generate_primary_cases(context_tokens=32768, tokenizer_revision="a" * 40)
    with pytest.raises(ValueError, match="immutable lowercase SHA"):
        generate_primary_cases(
            context_tokens=32768,
            tokenizer_revision="latest",
            token_counter=len,
        )


def test_cli_constructs_counter_from_exact_pinned_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40

    class FakeTokenizer:
        init_kwargs = {"_commit_hash": revision}

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize and add_generation_prompt
            return [1, 2, 3]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier, *, revision, trust_remote_code):
            assert identifier == "repo/tokenizer"
            assert revision == "a" * 40
            assert trust_remote_code is False
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    assert _pinned_tokenizer_counter("repo/tokenizer", revision)("hello") == 3
