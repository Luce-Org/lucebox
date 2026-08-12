from harness.validation_v2.statistics import paired_prompt_values, summarize_paired


def test_paired_summary_uses_prompts_as_independent_units() -> None:
    rows = []
    for prompt in ("p1", "p2"):
        for repetition in range(5):
            rows.append(
                {
                    "prompt_id": prompt,
                    "config": "ar",
                    "repetition": repetition + 1,
                    "status": "success",
                    "tok_s": 10 + repetition,
                }
            )
            rows.append(
                {
                    "prompt_id": prompt,
                    "config": "spec",
                    "repetition": repetition + 1,
                    "status": "success",
                    "tok_s": 15 + repetition,
                }
            )
    pairs = paired_prompt_values(
        rows,
        baseline="ar",
        candidate="spec",
        metric="tok_s",
        expected_repetitions=5,
    )
    summary = summarize_paired(pairs, bootstrap_samples=100, seed=1)
    assert summary["independent_unit"] == "prompt"
    assert summary["prompt_count"] == 2
    assert summary["slower_prompt_count"] == 0


def test_failed_prompt_is_not_silently_dropped() -> None:
    rows = [
        {
            "prompt_id": "p1",
            "config": "ar",
            "repetition": 1,
            "status": "success",
            "tok_s": 10,
        },
        {
            "prompt_id": "p1",
            "config": "spec",
            "repetition": 1,
            "status": "failed",
            "tok_s": None,
        },
    ]
    try:
        paired_prompt_values(rows, baseline="ar", candidate="spec", metric="tok_s")
    except ValueError as exc:
        assert "failed/incomplete prompts" in str(exc)
    else:
        raise AssertionError("failed prompt was silently dropped")


def test_latest_successful_retry_replaces_prior_failure() -> None:
    rows = [
        {
            "config_id": "ar-p1-r1",
            "attempt": 1,
            "prompt_id": "p1",
            "config": "ar",
            "repetition": 1,
            "status": "failed",
            "tok_s": None,
        },
        {
            "config_id": "ar-p1-r1",
            "attempt": 2,
            "prompt_id": "p1",
            "config": "ar",
            "repetition": 1,
            "status": "success",
            "tok_s": 10,
        },
        {
            "config_id": "spec-p1-r1",
            "attempt": 1,
            "prompt_id": "p1",
            "config": "spec",
            "repetition": 1,
            "status": "success",
            "tok_s": 12,
        },
    ]
    assert paired_prompt_values(
        rows,
        baseline="ar",
        candidate="spec",
        metric="tok_s",
        expected_repetitions=1,
    ) == {"p1": ([10.0], [12.0])}


def test_paired_analysis_requires_same_complete_repetition_denominator() -> None:
    rows = [
        {
            "prompt_id": "p1",
            "config": "ar",
            "repetition": 1,
            "status": "success",
            "tok_s": 10,
        },
        {
            "prompt_id": "p1",
            "config": "spec",
            "repetition": 2,
            "status": "success",
            "tok_s": 12,
        },
    ]
    try:
        paired_prompt_values(rows, baseline="ar", candidate="spec", metric="tok_s")
    except ValueError as exc:
        assert "failed/incomplete prompts" in str(exc)
    else:
        raise AssertionError("mismatched paired repetitions were accepted")


def test_nested_bootstrap_recomputes_ratio_of_means() -> None:
    summary = summarize_paired({"p": ([1, 100], [2, 101])}, bootstrap_samples=1000, seed=1)
    assert summary["geometric_mean_ratio"] == 103 / 101
    assert summary["ratio_ci95"][0] >= 1.0
    assert summary["ratio_ci95"][1] <= 2.0
