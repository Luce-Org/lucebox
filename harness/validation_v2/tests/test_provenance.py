from pathlib import Path

import pytest

import harness.validation_v2.provenance as provenance
from harness.validation_v2.provenance import (
    build_manifest,
    redact_command,
    redact_environment,
    repository_identity,
    sha256_file,
)


def test_sha256_file_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert sha256_file(first) == sha256_file(second)
    second.write_bytes(b"changed")
    assert sha256_file(first) != sha256_file(second)


def test_environment_redaction_is_allowlist_only() -> None:
    result = redact_environment(
        {"CUDA_VISIBLE_DEVICES": "0", "LANG": "C.UTF-8", "API_KEY": "do-not-copy"}
    )
    assert result == {"CUDA_VISIBLE_DEVICES": "0", "LANG": "C.UTF-8"}


def test_missing_file_hash_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")


def test_repository_rejects_uninitialized_or_mismatched_submodules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(args: list[str], _cwd: Path) -> str:
        if args[1:3] == ["status", "--porcelain"]:
            return ""
        if args[1:3] == ["submodule", "status"]:
            return "-" + "a" * 40 + " deps/example"
        return "a" * 40

    monkeypatch.setattr(provenance, "_run", fake_run)
    with pytest.raises(ValueError, match="initialized, pinned submodules"):
        repository_identity(tmp_path)


def test_manifest_requires_real_immutable_sha(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="commit-like revisions"):
        build_manifest(
            root=tmp_path,
            files={"input": source},
            immutable_inputs={"model": "this-is-definitely-latest"},
            build={},
            command=[],
        )


def test_manifest_redacts_commands_urls_and_nested_build_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    source.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        provenance,
        "repository_identity",
        lambda _root: {"commit": "a" * 40, "clean": True, "submodules": []},
    )
    manifest = build_manifest(
        root=tmp_path,
        files={"input": source},
        immutable_inputs={"model": "a" * 40},
        build={"flags": {"api_key": "secret", "max_tokens": 384}},
        command=[
            "runner",
            "--api-key",
            "secret",
            "--max-tokens",
            "384",
            "https://user:password@example.com/run?token=secret",
        ],
    )
    assert manifest["command"] == [
        "runner",
        "--api-key",
        "[REDACTED]",
        "--max-tokens",
        "384",
        "https://example.com/run",
    ]
    assert manifest["build"]["flags"] == {
        "api_key": "[REDACTED]",
        "max_tokens": 384,
    }
    assert redact_command(["HF_TOKEN=secret"]) == ["HF_TOKEN=[REDACTED]"]
