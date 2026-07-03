"""Split tool KV from conversation for PFlash + pinned tool cache.

Built-in profiles: ``qwen3``, ``laguna``. User plugins: drop a ``.py`` file
in ``--tool-split-plugin-dir`` that calls ``register_adapter("name")``.

See ``example_user_plugin.py`` for a template.
"""
from tool_split.base import PromptSplit, ToolSplitAdapter, ToolSplitPlan
from tool_split.config import ToolSplitConfig, add_cli_flags, config_from_env_and_args
from tool_split.orchestrator import ToolSlotCache, ToolSplitOrchestrator
from tool_split.registry import (
    builtin_adapter_names,
    get_adapter_class,
    load_plugin_dir,
    register_adapter,
    resolve_adapter,
)

# Import built-ins so they self-register.
from tool_split import laguna as _laguna  # noqa: F401
from tool_split import qwen3 as _qwen3  # noqa: F401

__all__ = [
    "PromptSplit",
    "ToolSplitAdapter",
    "ToolSplitConfig",
    "ToolSplitOrchestrator",
    "ToolSplitPlan",
    "ToolSlotCache",
    "add_cli_flags",
    "builtin_adapter_names",
    "config_from_env_and_args",
    "get_adapter_class",
    "load_plugin_dir",
    "register_adapter",
    "resolve_adapter",
]
