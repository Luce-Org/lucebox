# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports;
# llamafile/ollama/sampling_defaults clients dropped (not used by the eval harness).
"""Client adapters for LLM backends.

Only the subset needed by ``bench_http_capability.py --area forge`` is
vendored: the base client interface and the Anthropic adapter (loaded
lazily by ``_forge.clients.anthropic``). The upstream llamafile/ollama
clients and ``sampling_defaults`` helpers were not vendored because the
dflash eval driver does not use them.
"""

from .base import ChunkType, LLMClient, StreamChunk

__all__ = [
    "ChunkType",
    "LLMClient",
    "StreamChunk",
]
