"""Configuration for split tool KV + conversation PFlash."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolSplitConfig:
    """Parsed ``--tool-split-*`` flags / ``DFLASH_TOOL_SPLIT_*`` env vars."""

    enabled: bool = False
    #: ``auto`` | built-in name (``qwen3``, ``laguna``) | ``plugin:name``
    profile: str = "auto"
    plugin_dir: Path | None = None
    #: Dedicated daemon slots pinned for tool KV (not LRU-evicted with conv).
    pinned_tool_slots: int = 2
    #: Allow PFlash on the conversation suffix when tools are present.
    compress_conversation: bool = True

    def __post_init__(self) -> None:
        if self.pinned_tool_slots < 0:
            raise ValueError("pinned_tool_slots must be >= 0")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def config_from_env_and_args(args=None) -> ToolSplitConfig:
    """Merge env vars with optional argparse namespace."""
    enabled = _env_bool("DFLASH_TOOL_SPLIT_ENABLED", False)
    profile = os.environ.get("DFLASH_TOOL_SPLIT_PROFILE", "auto")
    plugin_dir_raw = os.environ.get("DFLASH_TOOL_SPLIT_PLUGIN_DIR")
    plugin_dir = Path(plugin_dir_raw).expanduser() if plugin_dir_raw else None
    pinned_raw = os.environ.get("DFLASH_TOOL_SPLIT_PINNED_SLOTS", "2")
    try:
        pinned = int(pinned_raw)
    except ValueError:
        pinned = 2
    compress_conv = _env_bool("DFLASH_TOOL_SPLIT_COMPRESS_CONV", True)

    if args is not None:
        if getattr(args, "tool_split", None) is not None:
            enabled = args.tool_split != "off"
        if getattr(args, "tool_split_profile", None):
            profile = args.tool_split_profile
        if getattr(args, "tool_split_plugin_dir", None):
            plugin_dir = Path(args.tool_split_plugin_dir).expanduser()
        if getattr(args, "tool_split_pinned_slots", None) is not None:
            pinned = args.tool_split_pinned_slots
        if getattr(args, "tool_split_compress_conv", None) is not None:
            compress_conv = args.tool_split_compress_conv

    return ToolSplitConfig(
        enabled=enabled,
        profile=profile,
        plugin_dir=plugin_dir,
        pinned_tool_slots=pinned,
        compress_conversation=compress_conv,
    )


def add_cli_flags(ap) -> None:
    """Attach ``--tool-split-*`` flags to an argparse parser."""
    ap.add_argument(
        "--tool-split",
        choices=["off", "on", "auto"],
        default=None,
        help="Split tool KV from conversation for PFlash + pinned tool cache. "
             "'auto' enables when a matching adapter is found. "
             "Env: DFLASH_TOOL_SPLIT_ENABLED=1.",
    )
    ap.add_argument(
        "--tool-split-profile",
        default=None,
        metavar="NAME",
        help="Adapter profile: auto (default), qwen3, laguna, or plugin:name. "
             "Env: DFLASH_TOOL_SPLIT_PROFILE.",
    )
    ap.add_argument(
        "--tool-split-plugin-dir",
        type=Path,
        default=None,
        help="Directory of user .py plugins calling register_adapter(). "
             "Env: DFLASH_TOOL_SPLIT_PLUGIN_DIR.",
    )
    ap.add_argument(
        "--tool-split-pinned-slots",
        type=int,
        default=None,
        help="Daemon slots reserved for tool KV snapshots (default 2). "
             "Env: DFLASH_TOOL_SPLIT_PINNED_SLOTS.",
    )
    ap.add_argument(
        "--tool-split-compress-conv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="PFlash-compress conversation suffix when tools are present "
             "(default on). Env: DFLASH_TOOL_SPLIT_COMPRESS_CONV.",
    )
