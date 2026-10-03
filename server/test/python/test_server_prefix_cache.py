"""End-to-end Phase A test: prefix cache integration.

Sends 3 chat completions sharing a 2K-token system prompt, asserts turns 2/3
have noticeably faster prefill than turn 1.

Spawns its own luce_server with the prefix cache enabled (skips when the
binary or model files are missing):
    pytest server/test/python/test_server_prefix_cache.py -v

Model overrides: LUCE_TARGET, LUCE_DRAFT; binary: --server-bin / LUCE_SERVER_BIN.
"""
import os
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.server,
    pytest.mark.model,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def prefix_cache_server(server_bin, spawn_luce_server):
    """A dedicated server with --prefix-cache-slots 2."""
    target = Path(os.environ.get(
        "LUCE_TARGET",
        Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"))
    draft = Path(os.environ.get(
        "LUCE_DRAFT", Path.home() / "models/qwen3.6-27b-dflash"))
    for p, label in [(target, "target GGUF"), (draft, "draft dir/GGUF")]:
        if not p.exists():
            pytest.skip(f"{label} missing at {p}")

    handle = spawn_luce_server(
        [str(server_bin), str(target),
         "--draft", str(draft),
         "--max-ctx", "4096",
         "--prefix-cache-slots", "2"],
        name="prefix_cache")
    yield handle
    handle.stop()


def test_prefix_cache_warm_turns(prefix_cache_server):
    client = prefix_cache_server.client
    # Large system prompt (~2K tokens) to make the prefill cost measurable.
    system = "You are a precise coding assistant. " * 200

    def chat(user_msg, max_tokens=8):
        payload = {
            "model": "luce-dflash",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens, "stream": False,
        }
        t0 = time.time()
        data = client.post("/v1/chat/completions", payload, timeout=600)
        return time.time() - t0, data["choices"][0]["message"]["content"]

    # Turn 1: cold (cache miss → snapshot taken at end)
    t1, r1 = chat("What is 2+2?")
    # Turns 2/3: same system prompt → cache HIT, only the suffix is prefilled
    t2, _ = chat("What is the capital of France?")
    t3, _ = chat("Tell me about Mars.")
    print(f"turn_1: {t1:.2f}s  turn_2: {t2:.2f}s (ratio {t2/t1:.2f})  "
          f"turn_3: {t3:.2f}s (ratio {t3/t1:.2f})")

    assert r1, "cold turn returned an empty reply"
    # Expect turn 2 and 3 prefill to be much faster (~2K-token system prompt
    # cached). Total wall is prefill + decode; decode is ~constant (small
    # max_tokens). Conservative gate: ratio < 0.85 (>=15% faster).
    assert t2 / t1 < 0.85, f"turn 2 not faster than cold turn ({t2/t1:.2f}x)"
    assert t3 / t1 < 0.85, f"turn 3 not faster than cold turn ({t3/t1:.2f}x)"

