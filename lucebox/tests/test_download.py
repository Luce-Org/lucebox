"""Tests for the model-download orchestration.

The downloader now drives `huggingface_hub.hf_hub_download` directly
(no subprocess) and verifies size + sha256 against the repo metadata
before re-fetching. The tests stub out the network calls so the
behavior contract — what gets requested, when downloads are skipped —
stays pinned without actually talking to the Hub.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from lucebox.download import DEFAULT_PRESET, PRESETS, resolve_preset, status

from lucebox import download


def test_default_preset_uses_quantized_gguf_draft():
    assert DEFAULT_PRESET.draft_repo == "spiritbuun/Qwen3.6-27B-DFlash-GGUF"
    assert DEFAULT_PRESET.draft_file == "dflash-draft-3.6-q4_k_m.gguf"


def test_default_preset_is_registered_under_qwen_name():
    assert DEFAULT_PRESET is PRESETS["qwen3.6-27b"]
    assert DEFAULT_PRESET.name == "qwen3.6-27b"


def test_resolve_preset_returns_default_on_none():
    assert resolve_preset(None) is DEFAULT_PRESET
    assert resolve_preset("") is DEFAULT_PRESET


def test_resolve_preset_picks_gemma_target_and_draft():
    pres = resolve_preset("gemma-4-26b")
    assert pres.name == "gemma-4-26b"
    assert pres.target_repo == "bartowski/google_gemma-4-26B-A4B-it-GGUF"
    assert pres.target_file == "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    assert pres.draft_repo == "Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF"
    assert pres.draft_file == "gemma-4-26B-A4B-it-DFlash-q8_0.gguf"
    assert pres.has_draft


def test_resolve_preset_supports_target_only_laguna():
    pres = resolve_preset("laguna-xs.2")
    assert pres.target_repo == "Lucebox/Laguna-XS.2-GGUF"
    assert pres.draft_repo is None
    assert not pres.has_draft


def test_resolve_preset_picks_qwen36_moe_target_only():
    """Qwen3.6 MoE preset routes to unsloth's UD-Q4_K_M file, no draft.

    The MoE variant has no published DFlash draft GGUF (verified against
    HfApi.repo_info 2026-05-28), so it runs target-only like Laguna. The
    file stem is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — the unsloth repo only
    publishes the UD ("unsloth dynamic") family at Q4_K_M, not a plain
    `Q4_K_M.gguf`.
    """
    pres = resolve_preset("qwen3.6-moe")
    assert pres.name == "qwen3.6-moe"
    assert pres.target_repo == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert pres.target_file == "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    assert pres.draft_repo is None
    assert pres.draft_file is None
    assert not pres.has_draft


def test_download_preset_target_only_qwen36_moe_skips_draft(tmp_path, monkeypatch):
    """qwen3.6-moe behaves identically to laguna-xs.2: target only, no draft fetch."""
    cfg = SimpleNamespace(models_dir=tmp_path)
    pres = resolve_preset("qwen3.6-moe")
    assert not pres.has_draft
    fetches: list[tuple[str, str]] = []

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        return 10, None

    def _stub_fetch(api, repo_id, filename, local_dir, console):  # noqa: ARG001
        fetches.append((repo_id, filename))
        out = local_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.truncate(10)
        return out

    monkeypatch.setattr(download, "_file_meta", _meta)
    monkeypatch.setattr(download, "_fetch", _stub_fetch)

    assert download.download_preset(cfg, pres) == 0
    # Only the target — no draft attempt at all.
    assert fetches == [(pres.target_repo, pres.target_file)]


def test_status_qwen36_moe_reports_draft_present_when_target_only(tmp_path, monkeypatch):
    """No published draft → status reports draft_present=True (nothing to fetch)."""
    cfg = SimpleNamespace(models_dir=tmp_path)
    pres = resolve_preset("qwen3.6-moe")

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        return 22 * 10**9, None

    monkeypatch.setattr(download, "_file_meta", _meta)
    # Target absent → target_present False, draft_present True (no draft).
    assert status(cfg, pres) == {"target_present": False, "draft_present": True}


def test_resolve_preset_unknown_name_lists_known_options():
    with pytest.raises(KeyError) as exc_info:
        resolve_preset("qwen-99b")
    msg = str(exc_info.value)
    # Every registered preset must appear in the suggestion list so the
    # user can copy-paste the right name.
    for name in PRESETS:
        assert name in msg


def _stub_file_meta(target_size: int, draft_size: int):
    """Build a `_file_meta` replacement that returns (size, None) per repo+file.

    sha256 is left None so tests don't need to compute real hashes; the
    real metadata path is exercised by the live `models download`
    invocation, not the unit tests.
    """

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        if repo_id == DEFAULT_PRESET.target_repo and filename == DEFAULT_PRESET.target_file:
            return target_size, None
        if repo_id == DEFAULT_PRESET.draft_repo and filename == DEFAULT_PRESET.draft_file:
            return draft_size, None
        raise FileNotFoundError(f"unexpected ({repo_id}, {filename})")

    return _meta


def test_status_checks_default_draft_gguf(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    target = tmp_path / DEFAULT_PRESET.target_file
    draft = draft_dir / DEFAULT_PRESET.draft_file

    monkeypatch.setattr(download, "_file_meta", _stub_file_meta(target_size=1024, draft_size=512))

    # Neither file exists yet.
    assert status(cfg) == {"target_present": False, "draft_present": False}

    # Write files at the expected sizes.
    with target.open("wb") as f:
        f.truncate(1024)
    with draft.open("wb") as f:
        f.truncate(512)
    assert status(cfg) == {"target_present": True, "draft_present": True}


def test_status_rejects_partial_model_files(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    target = tmp_path / DEFAULT_PRESET.target_file
    draft = draft_dir / DEFAULT_PRESET.draft_file
    target.write_bytes(b"partial")
    draft.write_bytes(b"partial")

    # Repo says the target is 1 GB; a 7-byte file is partial, not present.
    monkeypatch.setattr(
        download, "_file_meta", _stub_file_meta(target_size=10**9, draft_size=10**6)
    )
    assert status(cfg) == {"target_present": False, "draft_present": False}


def test_current_bytes_reads_xet_staging_path(tmp_path):
    """Regression: progress polling must see hf-xet's hashed staging file.

    huggingface_hub 1.x writes partial Xet downloads to
    ``{local_dir}/.cache/huggingface/download/{short_hash}.{etag}.incomplete``
    — NOT to ``{local_dir}/{filename}.incomplete``. Before the fix the
    polling code only checked the latter (which never appears) so the
    Rich progress bar sat at 0 bytes for the entire transfer.
    """
    filename = "model.gguf"
    etag = "abc123"
    candidates = download._incomplete_path_candidates(tmp_path, filename, etag)
    # The first candidate must point at the actual hf-xet staging path.
    xet_path: Path = candidates[0]
    assert xet_path.parent == tmp_path / ".cache" / "huggingface" / "download"
    assert xet_path.name.endswith(f".{etag}.incomplete")

    # Now: writing to that path must be observed by _current_bytes.
    xet_path.parent.mkdir(parents=True, exist_ok=True)
    xet_path.write_bytes(b"x" * 4096)
    target = tmp_path / filename
    assert download._current_bytes(target, candidates) == 4096


def test_current_bytes_falls_back_to_glob_without_etag(tmp_path):
    """When sha256 is unknown we still find growing .incomplete files."""
    filename = "model.gguf"
    candidates = download._incomplete_path_candidates(tmp_path, filename, etag=None)
    target = tmp_path / filename

    staging = tmp_path / ".cache" / "huggingface" / "download"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "deadbeef.deadbeef.incomplete").write_bytes(b"x" * 8192)
    assert download._current_bytes(target, candidates) == 8192


def test_current_bytes_prefers_final_target_when_complete(tmp_path):
    filename = "model.gguf"
    candidates = download._incomplete_path_candidates(tmp_path, filename, etag="abc")
    target = tmp_path / filename
    target.write_bytes(b"x" * 1234)
    assert download._current_bytes(target, candidates) == 1234


def test_download_preset_fetches_exact_draft_file(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    fetches: list[tuple[str, str, str]] = []

    monkeypatch.setattr(download, "_file_meta", _stub_file_meta(target_size=10, draft_size=10))

    # Stub the actual download to record what was requested + create a stub
    # file of the expected size so `_local_matches` would pass on a re-run.
    def _stub_fetch(api, repo_id, filename, local_dir, console):  # noqa: ARG001
        fetches.append((repo_id, filename, str(local_dir)))
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            f.truncate(10)
        return target

    monkeypatch.setattr(download, "_fetch", _stub_fetch)

    assert download.download_preset(cfg) == 0
    assert (DEFAULT_PRESET.target_repo, DEFAULT_PRESET.target_file, str(tmp_path)) in fetches
    assert (
        DEFAULT_PRESET.draft_repo,
        DEFAULT_PRESET.draft_file,
        str(tmp_path / "draft"),
    ) in fetches


def test_download_preset_routes_gemma_preset_to_correct_repos(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    pres = resolve_preset("gemma-4-26b")
    fetches: list[tuple[str, str, str]] = []

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        return 10, None

    def _stub_fetch(api, repo_id, filename, local_dir, console):  # noqa: ARG001
        fetches.append((repo_id, filename, str(local_dir)))
        out = local_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.truncate(10)
        return out

    monkeypatch.setattr(download, "_file_meta", _meta)
    monkeypatch.setattr(download, "_fetch", _stub_fetch)

    assert download.download_preset(cfg, pres) == 0
    assert (pres.target_repo, pres.target_file, str(tmp_path)) in fetches
    assert (pres.draft_repo, pres.draft_file, str(tmp_path / "draft")) in fetches


def test_download_preset_target_only_skips_draft_fetch(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    pres = resolve_preset("laguna-xs.2")
    assert not pres.has_draft
    fetches: list[tuple[str, str]] = []

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        return 10, None

    def _stub_fetch(api, repo_id, filename, local_dir, console):  # noqa: ARG001
        fetches.append((repo_id, filename))
        out = local_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.truncate(10)
        return out

    monkeypatch.setattr(download, "_file_meta", _meta)
    monkeypatch.setattr(download, "_fetch", _stub_fetch)

    assert download.download_preset(cfg, pres) == 0
    # Target fetched, no draft fetch attempted at all.
    assert fetches == [(pres.target_repo, pres.target_file)]


def test_status_target_only_preset_reports_draft_as_present(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    pres = resolve_preset("laguna-xs.2")

    def _meta(_api, repo_id: str, filename: str) -> tuple[int, None]:
        return 1024, None

    monkeypatch.setattr(download, "_file_meta", _meta)
    # Target absent → target_present False, draft_present True (nothing to download).
    assert status(cfg, pres) == {"target_present": False, "draft_present": True}
