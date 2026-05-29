"""Model download orchestration.

Runs *inside* the orchestrator container. Uses `huggingface_hub` directly
(no subprocess) so we can:

  * drive a Rich progress bar based on real byte counts (the previous
    `uvx hf download` subprocess produced no visible progress inside the
    container — hf-xet's TTY detection misfires there),
  * verify each candidate file's size and sha256 against the repo
    metadata BEFORE downloading, so a re-run on a host that already has
    the target GGUF (e.g. previous download into the same models_dir)
    skips the multi-GB fetch entirely.

The :data:`PRESETS` registry encodes the canonical (target_repo,
target_file, draft_repo, draft_file) tuple per model — selectable via
``lucebox models download <name>``. ``DEFAULT_PRESET`` stays pinned to
Qwen3.6-27B for back-compat with callers that pre-date the registry.
Drafts are optional: presets that have no published DFlash draft
(e.g. Laguna's speculator is safetensors, not GGUF) carry
``draft_repo=None`` and run target-only.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# hf-xet (huggingface_hub ≥ 1.16) streams the entire file in one final
# burst — the polling-based progress bar sits at 0% for ~14 minutes
# then snaps to 100% on a 17 GB GGUF. Force the chunked Python
# downloader instead so bytes grow continuously and the Rich bar tracks
# reality. Set before importing hf_hub_download so the import picks
# the env up. `setdefault` lets a user override on the command line.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub._local_folder import get_local_download_paths  # noqa: E402
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from lucebox.types import Config


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """Canonical (target, draft) repo+filename pair for a supported model.

    ``draft_repo`` and ``draft_file`` may both be ``None`` for models
    where no GGUF DFlash draft is published (e.g. Laguna's safetensors
    speculator). In that case the entrypoint runs target-only — DFlash
    speculative decoding is disabled but the server still works.
    """

    name: str
    target_repo: str
    target_file: str
    draft_repo: str | None
    draft_file: str | None
    approx_total_gb: int
    description: str = ""

    @property
    def has_draft(self) -> bool:
        return bool(self.draft_repo and self.draft_file)


# Registry of supported models. Keyed by preset name; the CLI surface
# exposes these via `lucebox models download <name>` and the
# `lucebox models list` table. The values come straight from the model
# cards under share/model_cards/ — keep them in sync.
PRESETS: dict[str, ModelPreset] = {
    "qwen3.6-27b": ModelPreset(
        name="qwen3.6-27b",
        target_repo="unsloth/Qwen3.6-27B-GGUF",
        target_file="Qwen3.6-27B-Q4_K_M.gguf",
        draft_repo="spiritbuun/Qwen3.6-27B-DFlash-GGUF",
        draft_file="dflash-draft-3.6-q4_k_m.gguf",
        approx_total_gb=17,
        description="Qwen3.6 27B dense (Q4_K_M) + Qwen3.6 DFlash draft. Lucebox default.",
    ),
    "gemma-4-26b": ModelPreset(
        name="gemma-4-26b",
        target_repo="bartowski/google_gemma-4-26B-A4B-it-GGUF",
        target_file="google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        draft_repo="Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF",
        draft_file="gemma-4-26B-A4B-it-DFlash-q8_0.gguf",
        approx_total_gb=18,
        description="Gemma 4 26B-A4B IT MoE (Q4_K_M) + Lucebox DFlash q8_0 draft.",
    ),
    "gemma-4-31b": ModelPreset(
        name="gemma-4-31b",
        target_repo="bartowski/google_gemma-4-31B-it-GGUF",
        target_file="google_gemma-4-31B-it-Q4_K_M.gguf",
        draft_repo="Lucebox/gemma-4-31B-it-DFlash-GGUF",
        draft_file="gemma-4-31B-it-DFlash-q8_0.gguf",
        approx_total_gb=21,
        description="Gemma 4 31B IT dense (Q4_K_M) + Lucebox DFlash q8_0 draft.",
    ),
    "laguna-xs.2": ModelPreset(
        name="laguna-xs.2",
        target_repo="Lucebox/Laguna-XS.2-GGUF",
        target_file="laguna-xs2-Q4_K_M.gguf",
        # Laguna's published DFlash speculator is safetensors
        # (poolside/Laguna-XS.2-speculator.dflash), not GGUF — the
        # download command doesn't fetch it. Target-only here; users
        # who want the speculator pull it manually.
        draft_repo=None,
        draft_file=None,
        approx_total_gb=20,
        description=(
            "Laguna-XS.2 MoE code model (Q4_K_M), target-only. "
            "DFlash speculator is safetensors — download manually if needed."
        ),
    ),
    "qwen3.6-moe": ModelPreset(
        name="qwen3.6-moe",
        target_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        # Unsloth's MoE repo publishes both a "UD" (dynamic) and a plain
        # Q4_K_M family. Verified 2026-05-28 via HfApi.repo_info: the
        # `-UD-Q4_K_M.gguf` variant (22.1 GB) is the canonical Q4_K_M
        # release — there is no plain `Q4_K_M.gguf` on the MoE repo.
        target_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        # No DFlash draft GGUF has been published for the MoE variant
        # (probed Lucebox/* and spiritbuun/* repos 2026-05-28 — none
        # exist). Target-only, mirroring laguna-xs.2's wiring. The
        # lucebox C++ server speaks the `qwen35moe` arch natively
        # (server/src/qwen35moe/) so this runs without a draft.
        draft_repo=None,
        draft_file=None,
        approx_total_gb=22,
        description=(
            "Qwen3.6 35B-A3B MoE (3B active per token), Q4_K_M unsloth "
            "dynamic quant. Target-only — no DFlash MoE draft published "
            "yet. Uses lucebox's qwen35moe arch backend."
        ),
    ),
}

DEFAULT_PRESET = PRESETS["qwen3.6-27b"]


def resolve_preset(name: str | None) -> ModelPreset:
    """Look up a preset by name, with a friendly error on typos.

    ``None`` (or empty string) resolves to :data:`DEFAULT_PRESET` so
    callers and the CLI default both flow through one code path.
    """
    if not name:
        return DEFAULT_PRESET
    if name in PRESETS:
        return PRESETS[name]
    # Build a suggestion list — show every known preset; the user's
    # search space is small (4 entries today) so listing them all is
    # cheaper and clearer than a fuzzy-match heuristic.
    known = ", ".join(sorted(PRESETS.keys()))
    raise KeyError(f"unknown preset {name!r}. Known presets: {known}")


def _file_meta(api: HfApi, repo_id: str, filename: str) -> tuple[int, str | None]:
    """Return (expected_size, lfs_sha256_or_None) for filename in repo_id."""
    info = api.repo_info(repo_id, files_metadata=True)
    for sib in info.siblings or []:
        if sib.rfilename == filename:
            sha = getattr(sib.lfs, "sha256", None) if sib.lfs else None
            return int(sib.size or 0), sha
    raise FileNotFoundError(f"{filename} not present in repo {repo_id}")


def _sha256(path: Path, chunk_mb: int = 16) -> str:
    h = hashlib.sha256()
    chunk = chunk_mb * 1024 * 1024
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _local_matches(path: Path, size: int, sha256: str | None, console: Console) -> bool:
    """True iff a local file at `path` matches the expected size + sha256.

    Size mismatch shortcircuits (cheap). Sha256 is verified for LFS files
    (multi-GB GGUFs always carry one) and skipped when the repo doesn't
    expose a hash. Hashing 17 GB takes ~30s on a fast SSD — worth it to
    avoid a multi-GB re-download on rate-limited / metered links.
    """
    if not path.exists():
        return False
    actual_size = path.stat().st_size
    if actual_size != size:
        console.print(
            f"  [yellow]✗[/yellow] {path.name} present but size {actual_size:,} != "
            f"expected {size:,} — will re-download"
        )
        return False
    if sha256:
        console.print(f"  [dim]verifying sha256 of {path.name} ({actual_size / 1e9:.1f} GB)…[/dim]")
        actual_sha = _sha256(path)
        if actual_sha != sha256:
            console.print(
                f"  [yellow]✗[/yellow] {path.name} sha256 {actual_sha[:12]}… != "
                f"expected {sha256[:12]}… — will re-download"
            )
            return False
    return True


def _incomplete_path_candidates(local_dir: Path, filename: str, etag: str | None) -> list[Path]:
    """Return likely paths of the partial file currently being written.

    huggingface_hub 1.x (with hf-xet) stages downloads under
    ``{local_dir}/.cache/huggingface/download/`` using a *hashed* name —
    ``{short_hash(metadata_filename)}.{etag}.incomplete`` — so a naive
    ``{filename}.incomplete`` poll never sees any growth and the
    progress bar sits at 0 % for the whole multi-GB transfer.

    We get the *exact* expected staging path from
    ``get_local_download_paths().incomplete_path(etag)`` when we already
    know the LFS sha256 (which acts as the etag for Xet downloads), and
    fall back to globbing every ``*.incomplete`` in the staging dir
    otherwise. The legacy non-Xet downloader writes a ``.incomplete``
    next to the destination blob in ``~/.cache/huggingface/hub`` — but
    when ``local_dir`` is set hf-hub always uses the local staging dir,
    so the two candidates above cover every code path we hit.
    """
    paths = get_local_download_paths(local_dir, filename)
    candidates: list[Path] = []
    if etag:
        candidates.append(paths.incomplete_path(etag))
    # Fallback: every .incomplete file in the staging dir. This is what
    # rescues us when sha256 is unknown (non-LFS file) or when hf-hub
    # changes the etag derivation again in some future release.
    candidates.append(paths.metadata_path.parent)  # sentinel: glob this dir
    return candidates


def _current_bytes(target: Path, candidates: list[Path]) -> int:
    """Best-effort byte count of the file currently being written."""
    if target.exists():
        try:
            return target.stat().st_size
        except OSError:
            pass
    for c in candidates:
        if c.is_dir():
            # Glob every .incomplete in the staging dir; return the
            # largest (there's typically only one in-flight transfer).
            largest = 0
            try:
                for p in c.glob("*.incomplete"):
                    try:
                        largest = max(largest, p.stat().st_size)
                    except OSError:
                        continue
            except OSError:
                continue
            if largest:
                return largest
        else:
            try:
                if c.exists():
                    return c.stat().st_size
            except OSError:
                continue
    return 0


def _download_with_progress(
    repo_id: str,
    filename: str,
    local_dir: Path,
    expected_size: int,
    console: Console,
    etag: str | None = None,
) -> Path:
    """Download a single HF file with a Rich progress bar.

    Runs hf_hub_download in a worker thread; the main thread polls the
    growing file size and updates the Rich progress bar. The polled
    target is computed via ``get_local_download_paths`` so we hit the
    actual hf-xet staging path (a hashed filename under
    ``.cache/huggingface/download/``), not a guess.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    candidates = _incomplete_path_candidates(local_dir, filename, etag)

    result: list[str | None] = [None]
    error: list[BaseException | None] = [None]

    def _worker() -> None:
        try:
            result[0] = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
        except BaseException as exc:  # propagate to main thread
            error[0] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(filename, total=expected_size or 1)
        while t.is_alive():
            current = _current_bytes(target, candidates)
            # Always tick the bar — even at 0 bytes — so Rich repaints
            # the spinner/ETA and the user sees the UI is alive within
            # the first poll tick rather than a blank "Downloading…" line.
            progress.update(task, completed=min(current, expected_size or current or 1))
            time.sleep(0.5)
        # Final tick after the worker finishes so the bar paints 100%.
        if target.exists():
            progress.update(task, completed=target.stat().st_size)

    t.join(timeout=5)
    if error[0] is not None:
        raise error[0]
    if result[0] is None:
        raise RuntimeError(f"hf_hub_download returned no path for {filename}")
    return Path(result[0])


def _fetch(
    api: HfApi,
    repo_id: str,
    filename: str,
    local_dir: Path,
    console: Console,
) -> Path:
    """Verify-or-download a single file. Skips when the local copy matches."""
    size, sha = _file_meta(api, repo_id, filename)
    target = local_dir / filename
    if _local_matches(target, size, sha, console):
        console.print(f"  [green]✓[/green] {filename} already present (size + sha256 match)")
        return target
    # `sha` doubles as the etag for hf-xet's staging path
    # ({local_dir}/.cache/huggingface/download/{hash}.{etag}.incomplete);
    # passing it through is what makes the Rich progress bar see real
    # byte counts during the multi-GB transfer.
    return _download_with_progress(repo_id, filename, local_dir, size, console, etag=sha)


def download_preset(cfg: Config, preset: ModelPreset | None = None) -> int:
    """Fetch the target GGUF + (optional) DFlash draft into cfg.models_dir.

    Returns 0 on success, non-zero on failure. Verifies each file's size
    and (LFS) sha256 against the repo metadata before downloading, so a
    repeat run with the files already on disk is a no-op + sha256 walk.

    ``preset=None`` resolves to :data:`DEFAULT_PRESET` for back-compat;
    presets with ``has_draft=False`` (e.g. Laguna) skip the draft fetch
    entirely and let the server run target-only.
    """
    preset = preset or DEFAULT_PRESET
    console = Console()
    api = HfApi()
    models = cfg.models_dir
    models.mkdir(parents=True, exist_ok=True)
    draft = models / "draft"
    draft.mkdir(exist_ok=True)

    try:
        _fetch(api, preset.target_repo, preset.target_file, models, console)
        if preset.has_draft:
            # Narrow the optionals for the type-checker — has_draft is
            # exactly the predicate that proves these aren't None.
            assert preset.draft_repo is not None and preset.draft_file is not None
            _fetch(api, preset.draft_repo, preset.draft_file, draft, console)
        else:
            console.print(
                f"  [dim]no DFlash draft published for {preset.name} — running target-only[/dim]"
            )
    except Exception as exc:
        console.print(f"[red]download failed:[/red] {exc}")
        return 1
    return 0


def _local_target_path(cfg: Config, preset: ModelPreset) -> Path:
    return cfg.models_dir / preset.target_file


def _local_draft_path(cfg: Config, preset: ModelPreset) -> Path | None:
    if not (preset.has_draft and preset.draft_file):
        return None
    return cfg.models_dir / "draft" / preset.draft_file


def installed_status(cfg: Config, preset: ModelPreset) -> str:
    """Return ``"installed"`` / ``"partial"`` / ``"absent"`` for a preset.

    Size-only — doesn't hash. ``"installed"`` requires the target (and
    draft when one is published) to exist on disk; ``"partial"`` means
    at least one of the two is present but the set is incomplete.
    """
    target_exists = _local_target_path(cfg, preset).exists()
    draft_path = _local_draft_path(cfg, preset)
    if draft_path is None:
        return "installed" if target_exists else "absent"
    draft_exists = draft_path.exists()
    if target_exists and draft_exists:
        return "installed"
    if target_exists or draft_exists:
        return "partial"
    return "absent"


def installed_size_gb(cfg: Config, preset: ModelPreset) -> float:
    """Sum of on-disk byte sizes for the preset's files, in GB (binary 1e9)."""
    total = 0
    target = _local_target_path(cfg, preset)
    if target.exists():
        try:
            total += target.stat().st_size
        except OSError:
            pass
    draft = _local_draft_path(cfg, preset)
    if draft is not None and draft.exists():
        try:
            total += draft.stat().st_size
        except OSError:
            pass
    return total / 1e9


def installed_presets(cfg: Config) -> list[ModelPreset]:
    """Return every preset whose files are currently present in cfg.models_dir.

    "Present" follows ``installed_status`` — fully installed only.
    Partial states (target without draft, etc.) are excluded so the
    default ``lucebox models`` view stays uncluttered.
    """
    out: list[ModelPreset] = []
    for name in sorted(PRESETS):
        pres = PRESETS[name]
        if installed_status(cfg, pres) == "installed":
            out.append(pres)
    return out


def status(cfg: Config, preset: ModelPreset | None = None) -> dict[str, bool]:
    """Quick presence check — what's already on disk? Size-only, no sha256.

    For presets without a published DFlash draft, ``draft_present`` is
    reported as ``True`` (nothing to fetch → nothing missing). That
    keeps the "all present, nothing to do" UX path uniform whether or
    not a draft exists.
    """
    preset = preset or DEFAULT_PRESET
    api = HfApi()
    out: dict[str, bool] = {}
    try:
        size, _ = _file_meta(api, preset.target_repo, preset.target_file)
        local = cfg.models_dir / preset.target_file
        out["target_present"] = local.exists() and local.stat().st_size == size
    except Exception:
        out["target_present"] = False

    if preset.has_draft:
        assert preset.draft_repo is not None and preset.draft_file is not None
        try:
            size, _ = _file_meta(api, preset.draft_repo, preset.draft_file)
            local = cfg.models_dir / "draft" / preset.draft_file
            out["draft_present"] = local.exists() and local.stat().st_size == size
        except Exception:
            out["draft_present"] = False
    else:
        out["draft_present"] = True
    return out
