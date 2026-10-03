"""Phase B.3 end-to-end test: multi-slot THICK LRU prefix cache.

Sends 5 conversation turns with a shared (large) system prompt and a growing
history. Asserts:

  - Turn 1: cold (cache miss).
  - Turns 2-5: each finds a progressively deeper cache hit so only the new
    user message (+ short assistant reply header) needs prefilling.
  - No warm turn is slower than the cold turn (<= 5% regression).

Spawns its own luce_server with a 4-slot prefix cache (skips when the
binary or model files are missing):
    pytest server/test/python/test_multi_turn_prefix_cache.py -v

Model overrides: LUCE_TARGET, LUCE_DRAFT; binary: --server-bin / LUCE_SERVER_BIN.
"""
import os
import re
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.server,
    pytest.mark.model,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def mt_prefix_cache_server(server_bin, spawn_luce_server):
    """A dedicated server with --prefix-cache-slots 4."""
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
         "--max-ctx", "8192",
         "--prefix-cache-slots", "4"],
        name="multi_turn_prefix_cache")
    yield handle
    handle.stop()


def test_multi_turn_deeper_hits(mt_prefix_cache_server):
    server = mt_prefix_cache_server
    client = server.client
    # Large system prompt (~2K tokens) to make the prefill cost measurable.
    system = "You are a helpful coder. " * 200

    history: list[dict] = []
    latencies: list[float] = []

    def turn(user: str) -> str:
        history.append({"role": "user", "content": user})
        payload = {
            "model": "luce-dflash",
            "messages": [{"role": "system", "content": system}, *history],
            "max_tokens": 8,
            "stream": False,
        }
        t0 = time.time()
        data = client.post("/v1/chat/completions", payload, timeout=600)
        dt = time.time() - t0
        reply = data["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        latencies.append(dt)
        print(f"turn {len(latencies)}: latency={dt:.2f}s reply={reply!r}")
        return reply

    r1 = turn("Q1: what is 2+2?")                        # cold
    turn("Q2: what is the capital of France?")           # hits system boundary
    turn("Q3: what is the square root of 144?")          # hits user1+asst1
    turn("Q4: what is the largest planet?")              # hits end-of-asst2
    turn("Q5: what is the speed of light?")              # hits end-of-asst3

    t1, *warm = latencies
    for n, t in enumerate(warm, start=2):
        print(f"turn {n} ratio={t / t1:.2f}")

    assert r1, "turn 1 reply must be non-empty"

    hit_lens = [
        int(m.group(1))
        for ln in server.read_log().splitlines()
        for m in [re.search(r"lookup hit slot=\d+ prefix_len=(\d+)", ln)]
        if m
    ]
    print(f"hit prefix_lens (turns 2..5): {hit_lens}")
    assert len(hit_lens) >= 4, \
        f"expected >=4 [pc] lookup hit lines (turns 2..5), got {len(hit_lens)}"
    assert hit_lens[-1] > hit_lens[0], \
        "cache did not walk deeper across turns: " + str(hit_lens)

    # Non-regression latency gate: warm turns should not be SLOWER than cold.
    for n, t in enumerate(warm, start=2):
        assert t <= t1 * 1.05, \
            f"warm turn {n} was >5% slower than cold turn 1 ({t1:.2f}s)"

