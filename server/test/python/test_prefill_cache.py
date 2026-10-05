"""End-to-end proof for exact prefill/full-prompt cache.

Starts luce_server with inline prefix cache disabled and prefill cache
enabled, sends the same long chat prompt three times, and asserts:

  - /props.full_cache reports enabled capacity.
  - the first request commits a full-cache entry.
  - requests 2 and 3 hit that entry.
  - warm prefill time is at least 5x faster than cold prefill.

Model overrides: LUCE_TARGET, LUCE_DRAFT; binary: --server-bin / LUCE_SERVER_BIN.

Run:  pytest server/test/python/test_prefill_cache.py -v -m server
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

TARGET = Path(os.environ.get("LUCE_TARGET", Path.home() / "models/Qwen3.6-27B-Q4_K_M.gguf"))
DRAFT = Path(os.environ.get("LUCE_DRAFT", Path.home() / "models/draft/dflash-draft-3.6-q4_k_m.gguf"))


def _extract_prefill_s(resp: dict) -> float:
    usage = resp.get("usage") or {}
    timings = usage.get("timings") or {}
    for key in ("prefill_s", "prompt_s", "prefill_seconds"):
        val = timings.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    val = timings.get("prefill_ms")
    if isinstance(val, (int, float)):
        return float(val) / 1000.0
    raise RuntimeError(f"response did not include prefill timing: usage={usage}")


@pytest.fixture(scope="module")
def prefill_cache_server(server_bin, spawn_luce_server):
    """luce_server with the inline prefix cache off and the prefill cache on."""
    for p, label in [(TARGET, "target GGUF"), (DRAFT, "draft GGUF")]:
        if not p.exists():
            pytest.skip(f"{label} missing at {p}")
    if not os.access(server_bin, os.X_OK):
        pytest.fail(f"luce_server is not executable at {server_bin}")

    handle = spawn_luce_server(
        [
            str(server_bin),
            str(TARGET),
            "--draft", str(DRAFT),
            "--max-ctx", "16384",
            "--prefix-cache-slots", "0",
            "--prefill-cache-slots", "2",
            "--ddtree",
            "--ddtree-budget", "16",
            "--cache-type-k", "tq3_0",
            "--cache-type-v", "tq3_0",
            "--fa-window", "0",
        ],
        name="prefill_cache")
    return handle


class TestPrefillCache:
    """Both tests share one dedicated server."""

    def test_full_cache_props(self, prefill_cache_server):
        props = prefill_cache_server.client.get("/props", timeout=5)
        full_cache = props.get("full_cache", {})
        assert full_cache.get("enabled"), f"full cache not enabled: {full_cache}"
        assert full_cache.get("capacity") == 2, \
            f"unexpected full-cache capacity: {full_cache}"

    def test_full_cache_hit_and_speedup(self, prefill_cache_server):
        client = prefill_cache_server.client

        filler = (
            "The repository contains an inference server, a benchmark harness, "
            "and cache implementations. This sentence is deterministic filler. "
        )
        prompt = (filler * 240) + "\n\nQuestion: Reply with exactly the word cached."
        payload = {
            "model": "luce",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }

        prefill = []
        for i in range(3):
            t0 = time.time()
            resp = client.post("/v1/chat/completions", payload, timeout=900)
            wall = time.time() - t0
            pf = _extract_prefill_s(resp)
            prefill.append(pf)
            print(f"turn {i + 1}: wall={wall:.3f}s prefill={pf:.3f}s")

        full_after = client.get("/props", timeout=5).get("full_cache", {})
        log_text = prefill_cache_server.read_log()
        commits = log_text.count("[pc] full-cache committed")
        hits = log_text.count("[pc] full-cache hit")
        print(f"after full_cache={full_after}")
        print(f"log commits={commits} hits={hits}")

        assert full_after.get("in_use", 0) >= 1, \
            f"full cache did not retain an entry: {full_after}"
        assert full_after.get("lifetime_hits", 0) >= 2 and hits >= 2, \
            f"full cache did not hit twice: props={full_after} log_hits={hits}"
        assert commits >= 1, "full cache did not log a committed entry"

        cold = prefill[0]
        warm_best = min(prefill[1:])
        speedup = cold / max(warm_best, 0.001)
        print(f"best warm speedup={speedup:.2f}x")
        assert speedup >= 5.0, \
            f"expected >=5x prefill speedup, got {speedup:.2f}x"
