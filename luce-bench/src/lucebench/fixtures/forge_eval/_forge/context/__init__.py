# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Context management for the forge library.

Provides compaction strategies, context budget management, and
hardware detection for VRAM-based budget estimation.
"""

from .hardware import (
    HardwareProfile,
    detect_hardware,
)
from .manager import CompactEvent, ContextManager, default_context_warning
from .strategies import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)

__all__ = [
    "CompactEvent",
    "CompactStrategy",
    "ContextManager",
    "default_context_warning",
    "HardwareProfile",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    "detect_hardware",
]
