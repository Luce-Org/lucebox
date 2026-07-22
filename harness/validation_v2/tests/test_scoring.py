import argparse
import json
from pathlib import Path

import pytest

from harness.validation_v2.cli import command_score_niah
from harness.validation_v2.records import sha256_json
from harness.validation_v2.scoring import score_exact_final_content, score_niah_records


def test_exact_match_normalizes_unicode_and_outer_whitespace() -> None:
    score = score_exact_final_content("café", "  cafe\u0301\n")
    assert score.passed


def test_substring_prefix_superset_and_extra_prose_fail() -> None:
    for response in ("12345670", "x1234567", "The answer is 1234567", "1234567 and 7654321"):
        assert not score_exact_final_content("1234567", response).passed


def test_only_native_final_content_is_considered() -> None:
    response = {
        "choices": [
            {"message": {"content": "amber-birch", "reasoning_content": "wrong answer"}}
        ]
    }
    assert score_exact_final_content("amber-birch", response).passed


def test_ambiguous_or_missing_content_fails() -> None:
    assert not score_exact_final_content("answer", {"choices": []}).passed
    assert not score_exact_final_content("answer", {"choices": [{}, {}]}).passed


def test_niah_scoring_is_complete_hash_bound_and_exact() -> None:
    cases = [
        {"case_id": "p1", "answer": "amber", "context_tokens": 32, "control": None},
        {"case_id": "p2", "answer": "NONE", "context_tokens": 32, "control": "prefix"},
    ]
    responses = [
        {"prompt_id": "p1", "prompt_hash": sha256_json(cases[0]), "final_content": "amber"},
        {
            "prompt_id": "p2",
            "prompt_hash": sha256_json(cases[1]),
            "final_content": "NONE with explanation",
        },
    ]
    result = score_niah_records(cases, responses)
    assert result["summary"] == {
        "cases": 2,
        "passed": 1,
        "failed": 1,
        "exact_match_rate": 0.5,
    }
    assert result["primary"]["exact_match_rate"] == 1.0
    assert result["controls"]["exact_match_rate"] == 0.0


def test_niah_scoring_rejects_missing_or_unbound_responses() -> None:
    case = {"case_id": "p1", "answer": "amber"}
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        score_niah_records(
            [case], [{"prompt_id": "p1", "prompt_hash": "x" * 64, "final_content": "amber"}]
        )
    with pytest.raises(ValueError, match="incomplete"):
        score_niah_records([case], [])


def test_score_niah_cli_writes_structured_output(tmp_path: Path) -> None:
    case = {"case_id": "p1", "answer": "amber", "control": None}
    response = {
        "prompt_id": "p1",
        "prompt_hash": sha256_json(case),
        "final_content": "amber",
    }
    cases_path = tmp_path / "cases.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "scores.json"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    responses_path.write_text(json.dumps(response) + "\n", encoding="utf-8")
    command_score_niah(
        argparse.Namespace(cases=cases_path, responses=responses_path, out=output_path)
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["schema"] == "lucebox.validation-v2.niah-scores/1"
    assert output["summary"]["exact_match_rate"] == 1.0
