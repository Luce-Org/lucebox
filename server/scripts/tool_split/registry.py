"""Adapter registry and user plugin discovery."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Type

from tool_split.base import ToolSplitAdapter

if TYPE_CHECKING:
    from tool_split.config import ToolSplitConfig

_REGISTRY: dict[str, Type[ToolSplitAdapter]] = {}


def register_adapter(name: str):
    """Decorator: register a :class:`ToolSplitAdapter` subclass under *name*."""

    def _wrap(cls: Type[ToolSplitAdapter]) -> Type[ToolSplitAdapter]:
        if not issubclass(cls, ToolSplitAdapter):
            raise TypeError(f"{cls!r} must subclass ToolSplitAdapter")
        key = name.strip().lower()
        if not key:
            raise ValueError("adapter name must be non-empty")
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"tool-split adapter {key!r} is already registered as {existing.__name__}")
        cls.profile_name = key
        _REGISTRY[key] = cls
        return cls

    return _wrap


def builtin_adapter_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_adapter_class(name: str) -> Type[ToolSplitAdapter]:
    key = name.strip().lower()
    if key.startswith("plugin:"):
        key = key.split(":", 1)[1]
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown tool-split profile {name!r}; "
            f"built-ins: {', '.join(builtin_adapter_names()) or '(none loaded)'}"
        )
    return _REGISTRY[key]


def load_plugin_dir(plugin_dir: Path | None) -> list[str]:
    """Import ``*.py`` modules from *plugin_dir* that call ``register_adapter``."""
    if plugin_dir is None or not plugin_dir.is_dir():
        return []
    loaded: list[str] = []
    for path in sorted(plugin_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"_tool_split_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
            loaded.append(path.name)
        except Exception as exc:
            print(f"[tool-split] plugin {path.name} failed: {exc}", flush=True)
    return loaded


def resolve_adapter(
    cfg: "ToolSplitConfig",
    *,
    arch: str,
    tokenizer_id: str,
) -> ToolSplitAdapter | None:
    """Pick an adapter instance from config (``auto`` uses ``detect()``)."""
    if not cfg.enabled:
        return None

    if cfg.plugin_dir:
        load_plugin_dir(cfg.plugin_dir)

    profile = cfg.profile.strip().lower()
    if profile == "auto":
        for cls in _REGISTRY.values():
            try:
                if cls.detect(arch=arch, tokenizer_id=tokenizer_id):
                    return cls()
            except Exception:
                continue
        print(
            f"[tool-split] auto: no adapter matched arch={arch!r} "
            f"tokenizer={tokenizer_id!r}; disabled",
            flush=True,
        )
        return None

    cls = get_adapter_class(profile)
    return cls()
