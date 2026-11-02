"""Tests for ci_plan.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from ci_plan import ALL, DS41, GPU, NONE, QWEN38, RULES, main, plan

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["server/src/qwen35/qwen35_backend.cpp", "server/test/unit/test_qwen35_x.cpp"], QWEN38),
        (["server/src/deepseek4/deepseek4_dspark.cpp"], DS41),
        (["server/share/deepseek41/placement_lucebox.json", "server/docs/DS41.md"], DS41),
        (["server/test/unit/test_server_unit.cpp"], GPU),
        (["share/model_cards/qwen3.8-27b.json"], QWEN38),
        (["share/model_cards/deepseek-v4.1-flash.json", "share/model_cards/README.md"], DS41),
        (["share/model_cards/gemma-4-31b-it.json", "share/model_cards/_schema.json"], NONE),
        (["README.md", "server/docs/DS41.md", "harness/clients/run_pi.sh"], NONE),
        (["server/tools/ds4_mix_converter/ds4_mix_converter.cpp", "uv.lock"], NONE),
        # Both models, shared code, other models and unlisted paths run everything.
        (["server/src/qwen35/a.cpp", "server/src/deepseek4/b.cpp"], ALL),
        (["server/src/qwen35/a.cpp", "server/src/common/dspark_head.cpp"], ALL),
        (["server/src/qwen35/gguf_target_loader.cpp"], ALL),
        (["server/src/qwen35moe/qwen35moe_ffn.h"], ALL),
        (["server/src/laguna/laguna_internal.h"], ALL),
        (["server/deps/llama.cpp/ggml/src/ggml-cuda/mmq.cu"], ALL),
        (["server/CMakeLists.txt"], ALL),
        ([".github/ci/serve_smoke.sh"], ALL),
        ([".github/workflows/ci.yml", "README.md"], ALL),
        (["server/share/status.html"], ALL),
        (["a/new/top/level/dir.cpp"], ALL),
        ([], ALL),
        (None, ALL),
    ],
)
def test_plan(paths: list[str] | None, expected: frozenset[str]) -> None:
    assert plan(paths)[0] == expected


def test_a_directory_rule_needs_the_whole_name() -> None:
    assert plan(["server/src/qwen35moe/x.cpp"])[0] == ALL
    assert plan(["server/src/deepseek4_extra.cpp"])[0] == ALL


@pytest.mark.parametrize("pattern", [rule[0] for rule in RULES if not rule[0].startswith("*")])
def test_every_rule_names_a_path_in_the_tree(pattern: str) -> None:
    # A renamed directory would otherwise leave a rule that never matches.
    assert (REPO / pattern).exists(), f"{pattern} is gone: update ci_plan.RULES"


def test_full_runs_everything(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--full"]) == 0
    assert capsys.readouterr().out.splitlines() == ["gpu=true", "qwen38=true", "ds41=true"]


def test_a_failed_listing_runs_everything(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--base", "no-such-ref", "--head", "HEAD"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["gpu=true", "qwen38=true", "ds41=true"]
    assert "::warning::" in captured.err
