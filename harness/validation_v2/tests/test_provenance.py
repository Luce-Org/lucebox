from pathlib import Path

import pytest

from harness.validation_v2.provenance import redact_environment, sha256_file


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
