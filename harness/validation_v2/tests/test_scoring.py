from harness.validation_v2.scoring import score_exact_final_content


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
