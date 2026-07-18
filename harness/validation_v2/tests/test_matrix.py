import hashlib
import json
from pathlib import Path

import pytest

from harness.validation_v2.matrix import build_cells, load_prompt_hashes, write_cells
from harness.validation_v2.records import sha256_json


def test_matrix_ids_are_stable_and_unique() -> None:
    cells = build_cells(
        prompt_hashes={"p1": "a" * 64, "p2": "b" * 64},
        platform="dual-v100s",
        environment_hash="e" * 64,
        protocol_hash="f" * 64,
        configurations=[{"name": "target-ar", "contract": "autoregressive"}],
        repetitions=2,
        phase="publication",
    )
    assert len(cells) == 4
    assert len({cell["config_id"] for cell in cells}) == 4
    assert cells[0]["config_hash"] == cells[-1]["config_hash"]
    assert cells[0]["prompt_hash"] != cells[-1]["prompt_hash"]
    assert all(cell["cell_hash"] for cell in cells)


def test_prompt_rows_are_content_addressed(tmp_path: Path) -> None:
    prompt = {"id": "p1", "prompt": "hello"}
    prompt["prompt_sha256"] = hashlib.sha256(b"hello").hexdigest()
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    first = load_prompt_hashes(path)
    prompt["prompt"] = "changed"
    path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        load_prompt_hashes(path)
    assert first == {"p1": sha256_json({**prompt, "prompt": "hello"})}


def test_matrix_rejects_empty_zero_rep_and_stale_hashes(tmp_path: Path) -> None:
    kwargs = {
        "prompt_hashes": {"p": "a" * 64},
        "platform": "rtx3090",
        "environment_hash": "e" * 64,
        "protocol_hash": "f" * 64,
        "configurations": [{"name": "ar"}],
        "phase": "publication",
    }
    with pytest.raises(ValueError, match="repetitions"):
        build_cells(**kwargs, repetitions=0)
    with pytest.raises(ValueError, match="configuration list is empty"):
        build_cells(**{**kwargs, "configurations": []}, repetitions=1)
    with pytest.raises(ValueError, match="prompt corpus is empty"):
        build_cells(**{**kwargs, "prompt_hashes": {}}, repetitions=1)
    cells = build_cells(**kwargs, repetitions=1)
    cells[0]["config"]["mode"] = "changed"
    with pytest.raises(ValueError, match="config hash mismatch"):
        write_cells(tmp_path / "matrix.jsonl", cells)
