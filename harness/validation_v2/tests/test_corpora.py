import json
from pathlib import Path

import pytest

from harness.validation_v2.corpora import (
    build_tuning40,
    build_validation100,
    validate_frozen_hashes,
)


def test_tuning_builder_requires_exactly_40_unique_source_rows() -> None:
    raw = json.dumps([["family", f"prompt {index}"] for index in range(40)]).encode()
    rows = build_tuning40(raw)
    assert len(rows) == 40
    assert len({row["id"] for row in rows}) == 40


def test_validation_builder_balances_ten_declared_families() -> None:
    examples = {
        "code": "Write a Python function for this task",
        "math": "Calculate the probability in this problem",
        "summarization": "Summarize the following passage",
        "classification": "Classify the following document",
        "extraction": "Extract the names from the text",
        "rewriting": "Rewrite the following paragraph",
        "brainstorming": "Brainstorm ideas for a workshop",
        "question-answering": "What is the purpose of the device?",
        "reasoning": "Analyze the evidence and infer the result",
        "creative": "Write a creative short story",
    }
    source = []
    for instruction in examples.values():
        source.extend({"instruction": f"{instruction} #{index}", "input": ""} for index in range(10))
    rows = build_validation100(json.dumps(source).encode())
    assert len(rows) == 100
    assert {row["family"] for row in rows} == set(examples)
    assert all(row["source_revision"] for row in rows)
    assert {row["source_license"] for row in rows} == {"CC-BY-NC-4.0"}


def test_generated_hashes_must_match_protocol(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "decode_corpora": {
                    "tuning40_sha256": "a" * 64,
                    "validation100_sha256": "b" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    result = {
        "tuning40": {"sha256": "a" * 64},
        "validation100": {"sha256": "b" * 64},
    }
    validate_frozen_hashes(result, protocol)
    result["validation100"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_frozen_hashes(result, protocol)
