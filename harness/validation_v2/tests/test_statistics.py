from harness.validation_v2.statistics import paired_prompt_values, summarize_paired


def test_paired_summary_uses_prompts_as_independent_units() -> None:
    rows = []
    for prompt in ("p1", "p2"):
        for repetition in range(5):
            rows.append(
                {"prompt_id": prompt, "config": "ar", "status": "success", "tok_s": 10 + repetition}
            )
            rows.append(
                {"prompt_id": prompt, "config": "spec", "status": "success", "tok_s": 15 + repetition}
            )
    pairs = paired_prompt_values(rows, baseline="ar", candidate="spec", metric="tok_s")
    summary = summarize_paired(pairs, bootstrap_samples=100, seed=1)
    assert summary["independent_unit"] == "prompt"
    assert summary["prompt_count"] == 2
    assert summary["slower_prompt_count"] == 0


def test_failed_prompt_is_not_silently_dropped() -> None:
    rows = [
        {"prompt_id": "p1", "config": "ar", "status": "success", "tok_s": 10},
        {"prompt_id": "p1", "config": "spec", "status": "failed", "tok_s": None},
    ]
    try:
        paired_prompt_values(rows, baseline="ar", candidate="spec", metric="tok_s")
    except ValueError as exc:
        assert "failed/incomplete prompts" in str(exc)
    else:
        raise AssertionError("failed prompt was silently dropped")
