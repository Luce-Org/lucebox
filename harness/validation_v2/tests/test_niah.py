from harness.validation_v2.niah import generate_control_cases, generate_primary_cases


def test_primary_design_is_balanced_and_deterministic() -> None:
    first = generate_primary_cases(context_tokens=32768, tokenizer_revision="a" * 40)
    second = generate_primary_cases(context_tokens=32768, tokenizer_revision="a" * 40)
    assert len(first) == 100
    assert first == second
    assert len({case.case_id for case in first}) == 100
    assert {case.depth_fraction for case in first} == {0.05, 0.25, 0.5, 0.75, 0.95}
    assert len({case.prompt_sha256 for case in first}) == 100


def test_controls_cover_all_adversarial_families() -> None:
    controls = generate_control_cases(context_tokens=32768, tokenizer_revision="a" * 40)
    assert len(controls) == 20
    assert {case.control for case in controls} == {
        "answer-absent",
        "misleading",
        "multiple",
        "prefix",
        "prompt-echo",
    }
