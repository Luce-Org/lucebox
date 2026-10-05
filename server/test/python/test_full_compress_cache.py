"""Option 3 integration test: full-compress-result cache.

Sends an identical ~40K-token NIAH prompt 3 times to a server with both
pFlash compression AND the full-compress-result cache enabled.

Expected behaviour:
  - Turn 1: cold compress + cold prefill + cache registration.
  - Turn 2: full-cache hit — skips BOTH the drafter dance AND the prefill.
            Latency should be < 30% of Turn 1.
  - Turn 3: same full-cache hit, same speedup.

Correctness: all three replies must be identical.

Skipped automatically if any prerequisite is missing:
  - target GGUF        (LUCE_TARGET or ~/models/qwen3.6-27b/...)
  - draft safetensors dir or GGUF            (LUCE_DRAFT)
  - Qwen3-0.6B-BF16 pFlash drafter GGUF      (LUCE_PREFILL_DRAFTER)
  - luce_server binary (--server-bin / LUCE_SERVER_BIN)

Run:  pytest server/test/python/test_full_compress_cache.py -v -m server
"""
import os
import random
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.server,
    pytest.mark.model,
    pytest.mark.slow,
]

# ─── NIAH prompt builder ──────────────────────────────────────────────

def build_long_prompt(target_tokens: int, seed: int = 42) -> tuple[str, str]:
    """Needle-In-A-Haystack-shaped prompt for the cache test.

    Uses a coarse `target_tokens * 4.0` sizing — this script only needs a
    long, deterministic, mostly-filler prompt and does not care about exact
    token counts (cf. pflash/tests/niah_gen.py, which is the canonical
    NIAH generator with tokenizer-aware sizing and a hard <=target cap).
    Keep the FILLER / NEEDLE / QUESTION text in rough sync with niah_gen.py
    so a reader sees the same NIAH shape; full deduplication would require
    moving the shared text into a package both files can import.
    """
    rng = random.Random(seed)
    key   = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    value = "".join(rng.choices("0123456789", k=7))
    filler = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again. ")
    target_chars = int(target_tokens * 4.0)
    body = (filler * (target_chars // len(filler) + 1))[:target_chars]
    insert = rng.randint(target_chars // 4, 3 * target_chars // 4)
    needle = f"The special magic {key} number is: {value}."
    body = body[:insert] + " " + needle + " " + body[insert:]
    prompt = (
        "Below is a long passage. Answer the question at the end "
        "based ONLY on information in the passage.\n\n"
        f"{body}\n\nQuestion: What is the special magic {key} number? "
        "Answer in one short sentence.\nAnswer:"
    )
    return prompt, value


PROMPT, ANSWER = build_long_prompt(target_tokens=40000)


# ─── Server fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def compress_server(server_bin, spawn_luce_server):
    """luce_server with pFlash compression + full-compress-result cache."""
    target = Path(os.environ.get(
        "LUCE_TARGET",
        Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"))
    draft = Path(os.environ.get(
        "LUCE_DRAFT", Path.home() / "models/qwen3.6-27b-dflash"))
    drafter = Path(os.environ.get(
        "LUCE_PREFILL_DRAFTER", Path.home() / "models/Qwen3-0.6B-BF16.gguf"))

    for p, label in [
        (target, "target GGUF"),
        (draft, "draft dir/GGUF"),
        (drafter, "drafter GGUF"),
    ]:
        if not p.exists():
            pytest.skip(f"{label} missing at {p}")

    # --prefix-cache-slots 4: prefix-cache pool (slots 0-3, unused when
    #   compression fires because compressed tokens lack chat-template
    #   markers).
    # --prefill-cache-slots 4: full-cache pool (slots 4-7).
    # Total = 8 == daemon hard cap (PREFIX_CACHE_SLOTS in test_dflash.cpp).
    handle = spawn_luce_server(
        [
            str(server_bin), str(target),
            "--draft", str(draft),
            "--max-ctx", "8192",
            "--prefix-cache-slots", "4",
            "--prefill-cache-slots", "4",
            "--prefill-compression", "auto",
            "--prefill-threshold", "32000",
            "--prefill-keep-ratio", "0.05",
            "--prefill-drafter", str(drafter),
        ],
        name="full_compress_cache")
    return handle


def _chat(client, max_tokens: int = 16) -> tuple[float, str]:
    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.time()
    data = client.post("/v1/chat/completions", payload, timeout=900)
    return time.time() - t0, data["choices"][0]["message"]["content"]


# ─── Test ─────────────────────────────────────────────────────────────

def test_full_cache_hit(compress_server):
    client = compress_server.client

    t1, r1 = _chat(client)   # cold: compress + prefill + register
    t2, r2 = _chat(client)   # full-cache hit
    t3, r3 = _chat(client)   # full-cache hit again

    full_cache_hits = compress_server.read_log().count("[pc] full-cache hit")

    print(f"t1={t1:.2f}s  t2={t2:.2f}s  t3={t3:.2f}s")
    print(f"  ratio_2/1={t2/t1:.3f}  ratio_3/1={t3/t1:.3f}")
    print(f"  log [pc] full-cache hits: {full_cache_hits}")
    print(f"  needle retrieved turn 1: {ANSWER in r1} (looking for {ANSWER!r})")

    # Needle retrieval is informational, not gated. It is a quality property
    # of pFlash's compression (keep-ratio + importance scoring), not of the
    # cache; identity-of-replies + log-confirmed hits is the right gate here.
    assert r1 == r2 == r3, "cached replies differ from the cold reply"
    assert t2 < t1 * 0.30, f"turn 2 not a full-cache speedup ({t2/t1:.2f}x)"
    assert t3 < t1 * 0.30, f"turn 3 not a full-cache speedup ({t3/t1:.2f}x)"
    assert full_cache_hits >= 2, \
        f"expected >=2 [pc] full-cache hit log lines, got {full_cache_hits}"
