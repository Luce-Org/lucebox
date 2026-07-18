import json

from harness.validation_v2.corpora import build_tuning40, build_validation100


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
