"""build_set.py split contract and the hand-written prompt files it ships."""

from collections import Counter

import pytest
from eval.build_set import assign_split
from eval.common import DATA_DIR, read_jsonl
from eval.graders import DETERMINISTIC


def test_split_is_stratified_and_deterministic():
    rows = [{"id": f"{s}-{i}", "source": s, "category": "c"} for s, n in (("big", 100), ("small", 10))
            for i in range(n)]
    assign_split(rows, 0.7, seed=1)
    first = {r["id"]: r["split"] for r in rows}
    counts = Counter((r["source"], r["split"]) for r in rows)
    assert counts == {("big", "train"): 70, ("big", "test"): 30, ("small", "train"): 7, ("small", "test"): 3}
    for r in rows:
        r.pop("split")
    assign_split(list(reversed(rows)), 0.7, seed=1)   # input order must not matter
    assert {r["id"]: r["split"] for r in rows} == first


@pytest.mark.parametrize("name", ["handwritten_easy.jsonl", "handwritten_hard.jsonl"])
def test_handwritten_prompts_are_well_formed(name):
    rows = read_jsonl(DATA_DIR / name)
    assert rows and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["messages"][-1]["role"] == "user" and r["messages"][-1]["content"].strip()
        g = r["grader"]
        assert g["type"] == "judge" or g["type"] in DETERMINISTIC, r["id"]
        if g["type"] == "label":
            assert g["answer"] in g["labels"], r["id"]
        if g["type"] == "json_keys":
            assert set(g.get("values", {})) <= set(g["keys"]), r["id"]
