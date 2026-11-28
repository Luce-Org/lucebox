"""Tests for ci_plan.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from ci_plan import (
    ALL,
    CARD_DS41,
    CARD_OTHER,
    CARD_QWEN38,
    CPP,
    CPP_ALL,
    CPP_DS41,
    CPP_GPU,
    CPP_QWEN38,
    DS41_PROFILE,
    JOBS,
    NONE,
    PY,
    PY_CPP,
    PY_MK,
    RULES,
    main,
    plan,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["server/src/qwen35/qwen35_backend.cpp", "server/test/unit/test_qwen35_x.cpp"], CPP_QWEN38),
        (["server/src/deepseek4/deepseek4_dspark.cpp"], CPP_DS41),
        # The doc also runs the Python lane, which holds the quick-start path check.
        (["server/share/deepseek41/placement_lucebox.json", "server/docs/DS41.md"], DS41_PROFILE | PY),
        (["server/test/unit/test_server_unit.cpp"], CPP_GPU),
        (["server/test/python/test_bench_ds4_decode.py"], PY),
        (["share/model_cards/qwen3.8-27b.json"], CARD_QWEN38),
        (["share/model_cards/deepseek-v4.1-flash.json", "share/model_cards/README.md"], CARD_DS41 | PY),
        (["share/model_cards/gemma-4-31b-it.json", "share/model_cards/_schema.json"], CARD_OTHER),
        # Markdown runs the Python lane, which holds the quick-start path check.
        (["README.md", "server/docs/DS41.md", "docs/guide.md"], PY),
        ([".github/workflows/docker.yml", "Dockerfile"], NONE),
        (["harness/clients/run_pi.sh"], PY_CPP),
        (["harness/clients/common.sh"], PY_CPP),
        (["harness/tests/test_client_launcher_timeouts.sh"], PY_CPP),
        (["server/scripts/entrypoint.sh"], PY_CPP),
        (["server/tools/export_ds4v_mmproj.py"], PY),
        (["server/tools/ds4_mix_converter/ds4_mix_converter.cpp"], CPP),
        (["server/examples/chat.py"], PY),
        (["optimizations/pflash/tests/test_pflash.py"], PY),
        (["optimizations/megakernel/setup.py"], PY_MK),
        (["uv.lock"], PY_MK),
        # Both models run all C++/GPU/serve jobs.
        (["server/src/qwen35/a.cpp", "server/src/deepseek4/b.cpp"], CPP_ALL),
        (["server/src/qwen35/gguf_target_loader.cpp"], CPP_ALL),
        # Shared code, other models, build files, CI scripts, and unlisted paths run everything.
        (["server/src/qwen35/a.cpp", "server/src/common/dspark_head.cpp"], ALL),
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


def test_hidden_shared_change_alongside_docs_still_runs_full_suite() -> None:
    # Negative verification: hiding a real shared engine change inside a
    # docs/python PR must expand selection to ALL so CI cannot go falsely green.
    jobs, _ = plan(["README.md", "harness/README.md", "server/include/luce/engine.h"])
    assert jobs == ALL


@pytest.mark.parametrize("pattern", [rule[0] for rule in RULES if not rule[0].startswith("*")])
def test_every_rule_names_a_path_in_the_tree(pattern: str) -> None:
    # A renamed directory would otherwise leave a rule that never matches.
    assert (REPO / pattern).exists(), f"{pattern} is gone: update ci_plan.RULES"


def test_every_markdown_change_runs_the_docs_check() -> None:
    # A directory rule (server/test/, share/model_cards/) shadows "*.md"; the
    # quick-start check would otherwise be skipped on those files.
    files = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    assert files
    missing = [name for name in files if not PY <= plan([name])[0]]
    assert not missing, f"markdown does not select the Python lane: {missing}"


def test_full_runs_everything(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--full"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{job}=true" for job in JOBS]


def test_a_failed_listing_runs_everything(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--base", "no-such-ref", "--head", "HEAD"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"{job}=true" for job in JOBS]
    assert "::warning::" in captured.err
