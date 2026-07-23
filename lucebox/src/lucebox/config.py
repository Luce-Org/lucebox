"""Sparse TOML persistence for .lucebox/config.toml.

Single source of truth for user-overridden configuration. We track which
dotted keys were explicitly set by the user (or by commands acting on
their behalf) and serialize ONLY those keys back to disk — defaults
stay implicit, so `config.toml` reads like a diff against live defaults
and upgrades that add new fields don't gratuitously rewrite every file.

The dotted-key surface area is small and flat:
  model.preset, model.target_file, model.draft_file
  port, models_dir, variant, image, container_name
  dflash.<field>  for each of the 11 DflashRuntime knobs + think_max

Load resolves the TOML file → ``Config`` object, with anything absent
filled from ``Config()`` defaults. Save writes back only the keys that
appear in the TOML doc (tracked on ``Config._user_set``). The TOML doc
itself is a plain ``dict[str, Any]`` carrying only the set keys.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC
from pathlib import Path
from typing import Any

import tomli_w

from lucebox.types import (
    Config,
    DflashRuntime,
    HostFacts,
    ModelMeta,
    Variant,
    default_models_dir,
)


def default_config_path() -> Path:
    """Where .lucebox/config.toml lives.

    Convention: under $LUCEBOX_HOME if set, otherwise $HOME/.lucebox. Lives in
    the bind-mounted host home dir so the config survives container teardown
    and is editable from the host.
    """
    base = os.environ.get("LUCEBOX_HOME")
    if base:
        return Path(base) / "config.toml"
    return Path.home() / ".lucebox" / "config.toml"


# ── dotted-key registry ────────────────────────────────────────────────────

def _cast_prefill_mode(v: Any) -> str:
    s = str(v)
    if s not in {"off", "auto", "always"}:
        raise ValueError(f"prefill_mode must be off/auto/always, got {s!r}")
    return s


def _cast_bool(v: Any) -> bool:
    """Strict-ish boolean coercion for config values.

    - Native booleans pass through.
    - Strings: 1/true/yes/on → True; 0/false/no/off/"" → False (case-insensitive).
    - Anything else raises ``ValueError`` rather than silently coercing,
      because that's what bit ``dflash.debug_thinking_logits`` — the
      built-in ``bool`` caster turned ``"false"`` into ``True``.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"cannot parse boolean: {v!r}")
    if isinstance(v, int):
        return bool(v)
    raise ValueError(f"cannot parse boolean: {v!r}")


# Each entry: dotted-key → (toml_path, type_caster, default_getter).
# ``toml_path`` is the (section, field) pair on disk; ``"_root"`` means the
# key lives at the top level (no [section]). ``default_getter`` returns the
# in-memory default so ``config get`` can annotate origin.
KEY_REGISTRY: dict[str, tuple[tuple[str, str], Callable[[Any], Any]]] = {
    "variant": (("image", "variant"), str),
    "image": (("image", "registry"), str),
    "container_name": (("runtime", "container_name"), str),
    "port": (("runtime", "port"), int),
    "models_dir": (("paths", "models"), str),
    "model.preset": (("model", "preset"), str),
    "model.target_file": (("model", "target_file"), str),
    "model.draft_file": (("model", "draft_file"), str),
    "dflash.budget": (("dflash", "budget"), int),
    "dflash.max_ctx": (("dflash", "max_ctx"), int),
    "dflash.lazy": (("dflash", "lazy"), _cast_bool),
    "dflash.prefix_cache_slots": (("dflash", "prefix_cache_slots"), int),
    "dflash.prefill_cache_slots": (("dflash", "prefill_cache_slots"), int),
    "dflash.cache_type_k": (("dflash", "cache_type_k"), str),
    "dflash.cache_type_v": (("dflash", "cache_type_v"), str),
    "dflash.prefill_mode": (("dflash", "prefill_mode"), _cast_prefill_mode),
    "dflash.prefill_keep_ratio": (("dflash", "prefill_keep_ratio"), float),
    "dflash.prefill_threshold": (("dflash", "prefill_threshold"), int),
    "dflash.prefill_drafter": (("dflash", "prefill_drafter"), str),
    "dflash.think_max": (("dflash", "think_max"), int),
    "dflash.fa_window": (("dflash", "fa_window"), int),
    "dflash.think_soft_close_min_ratio": (
        ("dflash", "think_soft_close_min_ratio"), float),
    "dflash.debug_thinking_logits": (
        ("dflash", "debug_thinking_logits"), _cast_bool),
}


def _doc_get(doc: dict[str, Any], section: str, field: str) -> Any:
    if section == "_root":
        return doc.get(field)
    sub = doc.get(section)
    if isinstance(sub, dict):
        return sub.get(field)
    return None


def _doc_set(doc: dict[str, Any], section: str, field: str, value: Any) -> None:
    if section == "_root":
        doc[field] = value
        return
    doc.setdefault(section, {})[field] = value


def _doc_unset(doc: dict[str, Any], section: str, field: str) -> bool:
    """Remove a dotted key from the doc. Returns True iff something was removed."""
    if section == "_root":
        if field in doc:
            del doc[field]
            return True
        return False
    sub = doc.get(section)
    if isinstance(sub, dict) and field in sub:
        del sub[field]
        if not sub:
            del doc[section]
        return True
    return False


# ── load ───────────────────────────────────────────────────────────────────


def load(path: Path | None = None) -> Config | None:
    """Load config.toml, or return None if missing.

    If a legacy `.env` sits next to it (or in place of it), migrate that
    first and write back as TOML.
    """
    path = path or default_config_path()
    if path.exists():
        return _load_toml(path)

    legacy = path.with_suffix(".env")
    if legacy.exists():
        cfg, doc = _load_legacy_env(legacy)
        save(cfg, path, doc=doc)
        return cfg

    return None


def _load_toml(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    return _from_dict(raw)


def load_doc(path: Path | None = None) -> dict[str, Any]:
    """Return the raw TOML doc (a dict). Empty when no file or empty file."""
    path = path or default_config_path()
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


_LEGACY_KEY_MAP: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "DFLASH_BUDGET": ("dflash", "budget", int),
    "DFLASH_MAX_CTX": ("dflash", "max_ctx", int),
    "DFLASH_LAZY": ("dflash", "lazy",
                    lambda v: str(v).strip().lower() in ("1", "true", "yes", "on")),
    "DFLASH_PREFIX_CACHE_SLOTS": ("dflash", "prefix_cache_slots", int),
    "DFLASH_PORT": ("runtime", "port", int),
    "LUCEBOX_VARIANT": ("image", "variant", str),
    "LUCEBOX_IMAGE": ("image", "registry", str),
    "LUCEBOX_MODELS": ("paths", "models", str),
}


def _load_legacy_env(path: Path) -> tuple[Config, dict[str, Any]]:
    """Best-effort migration from the bash-era .lucebox/config.env."""
    raw: dict[str, Any] = {}
    line_re = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
        if key not in _LEGACY_KEY_MAP:
            continue
        section, field, cast_fn = _LEGACY_KEY_MAP[key]
        try:
            raw.setdefault(section, {})[field] = cast_fn(val)
        except (TypeError, ValueError):
            continue
    return _from_dict(raw), raw


def _from_dict(raw: dict[str, Any]) -> Config:
    img = raw.get("image", {})
    variant: Variant = str(img.get("variant", "cuda12"))
    registry = img.get("registry", "ghcr.io/luce-org/lucebox-hub")

    runtime = raw.get("runtime", {})
    port = int(runtime.get("port", 8080))
    container_name = str(runtime.get("container_name", "lucebox"))

    paths = raw.get("paths", {})
    models_dir = Path(paths.get("models", str(default_models_dir())))

    df = raw.get("dflash", {})
    dflash = DflashRuntime(
        budget=int(df.get("budget", 22)),
        max_ctx=int(df.get("max_ctx", 16384)),
        lazy=bool(df.get("lazy", False)),
        prefix_cache_slots=int(df.get("prefix_cache_slots", 0)),
        prefill_cache_slots=int(df.get("prefill_cache_slots", 0)),
        cache_type_k=str(df.get("cache_type_k", "")),
        cache_type_v=str(df.get("cache_type_v", "")),
        prefill_mode=df.get("prefill_mode", "off"),
        prefill_keep_ratio=float(df.get("prefill_keep_ratio", 0.05)),
        prefill_threshold=int(df.get("prefill_threshold", 32000)),
        prefill_drafter=str(df.get("prefill_drafter", "")),
        think_max=int(df.get("think_max", 15488)),
        fa_window=int(df.get("fa_window", 0)),
        think_soft_close_min_ratio=float(
            df.get("think_soft_close_min_ratio", 0.0)),
        debug_thinking_logits=bool(df.get("debug_thinking_logits", False)),
    )

    host_raw = raw.get("host", {})
    host = HostFacts(
        nproc=int(host_raw.get("nproc", 0)),
        ram_gb=int(host_raw.get("ram_gb", 0)),
        gpu_vendor=host_raw.get("gpu_vendor", "none"),
        gpu_name=str(host_raw.get("gpu_name", "")),
        gpu_count=int(host_raw.get("gpu_count", 0)),
        vram_gb=int(host_raw.get("vram_gb", 0)),
        gpu_sm=str(host_raw.get("gpu_sm", "")),
        driver_version=str(host_raw.get("driver_version", "")),
        driver_major=int(host_raw.get("driver_major", 0)),
        has_systemd=bool(host_raw.get("has_systemd", False)),
        is_wsl=bool(host_raw.get("is_wsl", False)),
        has_docker=bool(host_raw.get("has_docker", False)),
        docker_version=str(host_raw.get("docker_version", "")),
        ctk=host_raw.get("ctk", "none"),
    )

    # `[model]` is optional — legacy configs (pre-multi-model) carry no
    # such section and we want them to keep working unchanged. If
    # `preset` is set but `target_file` / `draft_file` isn't, derive
    # them from the registry so users only have to write one key.
    mdl = raw.get("model", {})
    preset_name = str(mdl.get("preset", ""))
    target_file = str(mdl.get("target_file", ""))
    draft_file = str(mdl.get("draft_file", ""))
    if preset_name and (not target_file or not draft_file):
        from lucebox.download import PRESETS

        if preset_name in PRESETS:
            pres = PRESETS[preset_name]
            if not target_file:
                target_file = pres.target_file
            if not draft_file and pres.has_draft and pres.draft_file:
                draft_file = pres.draft_file
    model = ModelMeta(preset=preset_name, target_file=target_file, draft_file=draft_file)

    return Config(
        variant=variant,
        image=registry,
        container_name=container_name,
        port=port,
        models_dir=models_dir,
        dflash=dflash,
        host=host,
        model=model,
    )


# ── save ───────────────────────────────────────────────────────────────────


def _atomic_write_doc(path: Path, doc: dict[str, Any]) -> None:
    """Serialize ``doc`` to TOML and write it to ``path`` atomically.

    Write to a sibling ``.toml.tmp`` then ``replace`` so a crash mid-write
    never leaves a truncated config.toml. Caller ensures ``path.parent`` exists.
    """
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_bytes(tomli_w.dumps(doc).encode("utf-8"))
    tmp.replace(path)


def save(cfg: Config, path: Path | None = None, *, doc: dict[str, Any] | None = None) -> Path:
    """Persist a Config to ``path``. Only keys present in ``doc`` are written.

    ``doc`` is the raw TOML mapping returned by ``load_doc`` — it carries
    exactly the keys the user (or a command on their behalf) has set. When
    ``doc=None`` and the file exists we re-use the on-disk doc; when both
    are absent we write an empty file.
    """
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if doc is None:
        doc = load_doc(path)
    _atomic_write_doc(path, doc)
    # Silence unused-arg: cfg is the on-disk representation's source of
    # truth for callers that want to round-trip through a Config object,
    # but the sparse write never re-derives keys from it.
    del cfg
    return path


def seed_dflash_from_host(host: HostFacts, *, path: Path | None = None) -> bool:
    """Persist the VRAM-tier DFLASH_* heuristic to config.toml on first setup.

    Returns True when it wrote. No-op (returns False) when a ``[dflash]``
    section already exists, so it never clobbers values a prior tune or the
    user set. Called when a preset is first activated: without it a fresh
    install serves at the conservative ``DflashRuntime`` class defaults
    (``load()`` returns those for a config.toml that has no ``[dflash]``),
    ignoring the host's VRAM tier. The ``live_config`` heuristic only fires
    when there is no config.toml at all, which stops being true the moment a
    model is activated — so the heuristic is persisted here instead.
    """
    from datetime import datetime

    import lucebox.autotune as autotune_mod

    path = path or default_config_path()
    doc = load_doc(path)
    if "dflash" in doc:
        return False
    runtime = autotune_mod.runtime_from_host(host)
    for field, value in asdict(runtime).items():
        _doc_set(doc, "dflash", field, _value_to_toml(value))
    _doc_set(doc, "autotune", "source", "heuristic")
    _doc_set(
        doc, "autotune", "timestamp",
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_doc(path, doc)
    return True


# ── dotted-key API ─────────────────────────────────────────────────────────


def _value_to_toml(value: Any) -> Any:
    """Make a Python value safe for tomli_w (no None, Path→str)."""
    if isinstance(value, Path):
        return str(value)
    return value


def _live_default(key: str) -> Any:
    """Return the in-memory default for ``key`` (from a fresh Config())."""
    cfg = Config()
    section_field = KEY_REGISTRY[key][0]
    section, field = section_field
    if section == "image":
        return {"variant": cfg.variant, "registry": cfg.image}[field]
    if section == "runtime":
        return {"port": cfg.port, "container_name": cfg.container_name}[field]
    if section == "paths":
        return str(cfg.models_dir) if field == "models" else None
    if section == "dflash":
        return getattr(cfg.dflash, field)
    if section == "model":
        return getattr(cfg.model, field)
    return None


def config_set(key: str, value: Any, *, path: Path | None = None) -> None:
    """Set one dotted key and write the file. Auto-creates a missing file."""
    if key not in KEY_REGISTRY:
        raise KeyError(f"unknown config key {key!r}; known: {sorted(KEY_REGISTRY)}")
    section_field, caster = KEY_REGISTRY[key]
    section, field = section_field
    try:
        cast_value = caster(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot coerce {value!r} for {key}: {exc}") from exc
    path = path or default_config_path()
    doc = load_doc(path) if path.exists() else {}
    _doc_set(doc, section, field, _value_to_toml(cast_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_doc(path, doc)


def config_unset(key: str, *, path: Path | None = None) -> bool:
    """Remove a dotted key from the file. Returns True if something changed."""
    if key not in KEY_REGISTRY:
        raise KeyError(f"unknown config key {key!r}; known: {sorted(KEY_REGISTRY)}")
    section_field, _ = KEY_REGISTRY[key]
    section, field = section_field
    path = path or default_config_path()
    if not path.exists():
        return False
    doc = load_doc(path)
    changed = _doc_unset(doc, section, field)
    if changed:
        # Leave the file in place even when empty — `config set` will
        # repopulate; deleting would surprise users who expect their
        # config dir to exist.
        _atomic_write_doc(path, doc)
    return changed


def config_get(key: str | None = None, *, path: Path | None = None) -> dict[str, tuple[Any, str]]:
    """Return ``{key: (value, origin)}``. ``origin`` is ``"file"`` or ``"default"``.

    When ``key`` is None or empty, every registered key is returned.
    Otherwise just that one key (still as a single-item dict, for caller
    uniformity).
    """
    path = path or default_config_path()
    doc = load_doc(path) if path.exists() else {}
    keys = [key] if key else list(KEY_REGISTRY)
    out: dict[str, tuple[Any, str]] = {}
    for k in keys:
        if k not in KEY_REGISTRY:
            raise KeyError(f"unknown config key {k!r}; known: {sorted(KEY_REGISTRY)}")
        section_field, _ = KEY_REGISTRY[k]
        section, field = section_field
        in_file = _doc_get(doc, section, field)
        if in_file is not None:
            out[k] = (in_file, "file")
        else:
            out[k] = (_live_default(k), "default")
    return out


def live_config() -> Config:
    """Build a fresh Config from current host facts + the DFLASH_* heuristic.

    Used as the no-config fallback in ``cli._load_or_build`` and reused by
    the ``models`` sub-app, so the host probe + heuristic + env-override
    logic lives in one place rather than being duplicated per caller.
    """
    # Lazy import to avoid the autotune ↔ config import cycle the importer
    # would hit if this moved to module scope.
    import lucebox.autotune as autotune_mod
    from lucebox.host_facts import from_env

    host = from_env()
    default = Config()
    return replace(
        default,
        variant=os.environ.get("LUCEBOX_VARIANT", "cuda12"),
        image=os.environ.get("LUCEBOX_IMAGE", default.image),
        container_name=os.environ.get("LUCEBOX_CONTAINER", default.container_name),
        port=int(os.environ.get("LUCEBOX_PORT", str(default.port))),
        models_dir=Path(os.environ.get("LUCEBOX_MODELS", str(default.models_dir))),
        dflash=autotune_mod.runtime_from_host(host),
        host=host,
    )
