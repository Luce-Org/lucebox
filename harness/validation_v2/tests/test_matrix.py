from harness.validation_v2.matrix import build_cells


def test_matrix_ids_are_stable_and_unique() -> None:
    cells = build_cells(
        prompt_ids=["p1", "p2"],
        platform="dual-v100s",
        environment_hash="e" * 64,
        configurations=[{"name": "target-ar", "contract": "autoregressive"}],
        repetitions=2,
        phase="publication",
    )
    assert len(cells) == 4
    assert len({cell["config_id"] for cell in cells}) == 4
    assert cells[0]["config_hash"] == cells[-1]["config_hash"]
