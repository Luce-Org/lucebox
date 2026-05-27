import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from _prefill_hook import PrefillConfig
from prefix_cache import FullCacheEntry, PrefixCache, hash_prefix
from server import build_app


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "<|im_end|>": [1],
            "<|im_start|>": [2],
            "system": [3],
        }
        return mapping.get(text, [99])


def make_cache(tmp_path: Path, *, cap: int = 2, full_cap: int = 2,
               kv_k_type: str = "q8_0", fa_window: int = 2048,
               budget_bytes: int = 0) -> PrefixCache:
    async def await_reply(prefix: str, timeout: float = 10.0) -> str:
        return prefix

    cache = PrefixCache(
        daemon_stdin=SimpleNamespace(write=lambda *_: None, flush=lambda: None),
        await_reply=await_reply,
        daemon_lock=asyncio.Lock(),
        tokenizer=FakeTokenizer(),
        kv_k_type=kv_k_type,
        fa_window=fa_window,
        cap=cap,
    )
    cache.init_full_cache(full_cap, cache_dir=str(tmp_path), budget_bytes=budget_bytes)
    return cache


def write_meta(cache: PrefixCache, key: bytes, *, cur_ids_len: int,
               last_used_ns: int, raw_prompt_len: int | None = None,
               kv_k_type: str | None = None,
               fa_window: int | None = None) -> None:
    meta = {
        "version": cache.FULL_META_VERSION,
        "key_hex": key.hex(),
        "kv_k_type": kv_k_type or cache.kv_k_type,
        "fa_window": cache.fa_window if fa_window is None else fa_window,
        "cur_ids_len": cur_ids_len,
        "raw_prompt_len": cur_ids_len if raw_prompt_len is None else raw_prompt_len,
        "last_used_ns": last_used_ns,
    }
    cache._full_meta_path(key).write_text(json.dumps(meta), encoding="utf-8")


def test_rehydrate_full_cache_restores_valid_entries(tmp_path):
    prompt_ids = [11, 22, 33]
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"cached")

    cache = make_cache(tmp_path)
    cache.confirm_full_snap(cache._full_slot_base, prompt_ids, source_path, 7)

    restored_cache = make_cache(tmp_path)
    replay = AsyncMock(return_value=True)

    restored = asyncio.run(restored_cache.rehydrate_full_cache(replay))

    key = hash_prefix(prompt_ids, restored_cache.kv_k_type, restored_cache.fa_window)
    assert restored == 1
    replay.assert_awaited_once_with(
        restored_cache._full_slot_base,
        str(restored_cache._full_bin_path(key)),
        7,
    )
    entry = restored_cache.full_entries[key]
    assert entry.slot == restored_cache._full_slot_base
    assert entry.cur_bin_path == str(restored_cache._full_bin_path(key))
    assert entry.cur_ids_len == 7
    assert entry.raw_prompt_len == len(prompt_ids)


def test_rehydrate_full_cache_skips_stale_metadata(tmp_path):
    cache = make_cache(tmp_path)
    key = b"\x01" * 16
    cache._full_bin_path(key).write_bytes(b"stale")
    write_meta(cache, key, cur_ids_len=5, last_used_ns=1, kv_k_type="q4_0")

    replay = AsyncMock(return_value=True)
    restored = asyncio.run(cache.rehydrate_full_cache(replay))

    assert restored == 0
    replay.assert_not_called()
    assert cache.full_entries == {}


def test_rehydrate_full_cache_skips_malformed_integer_metadata(tmp_path):
    cache = make_cache(tmp_path)
    bad_key = b"\x02" * 16
    good_prompt = [7, 7, 7]
    good_key = hash_prefix(good_prompt, cache.kv_k_type, cache.fa_window)
    cache._full_bin_path(bad_key).write_bytes(b"bad")
    cache._full_meta_path(bad_key).write_text(json.dumps({
        "version": [],
        "key_hex": bad_key.hex(),
        "kv_k_type": cache.kv_k_type,
        "fa_window": "oops",
        "cur_ids_len": 5,
        "raw_prompt_len": 5,
        "last_used_ns": 1,
    }), encoding="utf-8")
    cache._full_bin_path(good_key).write_bytes(b"good")
    write_meta(cache, good_key, cur_ids_len=6, raw_prompt_len=len(good_prompt), last_used_ns=2)

    replay = AsyncMock(return_value=True)
    restored = asyncio.run(cache.rehydrate_full_cache(replay))

    assert restored == 1
    replay.assert_awaited_once_with(cache._full_slot_base, str(cache._full_bin_path(good_key)), 6)
    assert list(cache.full_entries.keys()) == [good_key]


def test_rehydrate_full_cache_keeps_most_recent_entries_within_cap(tmp_path):
    cache = make_cache(tmp_path, full_cap=2)
    keys = [bytes([i]) * 16 for i in range(1, 4)]
    for idx, key in enumerate(keys, start=1):
        cache._full_bin_path(key).write_bytes(f"bin-{idx}".encode())
        write_meta(cache, key, cur_ids_len=idx, last_used_ns=idx)

    replay_calls = []

    async def replay(slot: int, cur_bin_path: str, cur_ids_len: int) -> bool:
        replay_calls.append((slot, Path(cur_bin_path).name, cur_ids_len))
        return True

    restored = asyncio.run(cache.rehydrate_full_cache(replay))

    assert restored == 2
    assert replay_calls == [
        (cache._full_slot_base, f"{keys[1].hex()}.bin", 2),
        (cache._full_slot_base + 1, f"{keys[2].hex()}.bin", 3),
    ]
    assert list(cache.full_entries.keys()) == [keys[1], keys[2]]


def test_score_victim_selector_prefers_lower_usefulness(tmp_path):
    cache = make_cache(tmp_path)
    low_key = b"\x10" * 16
    high_key = b"\x20" * 16
    cache._full_bin_path(low_key).write_bytes(b"a" * 100)
    cache._full_meta_path(low_key).write_text("{}", encoding="utf-8")
    cache._full_bin_path(high_key).write_bytes(b"b" * 10)
    cache._full_meta_path(high_key).write_text("{}", encoding="utf-8")
    cache.full_entries[low_key] = FullCacheEntry(
        slot=cache._full_slot_base,
        cur_bin_path=str(cache._full_bin_path(low_key)),
        cur_ids_len=40,
        raw_prompt_len=40,
        last_used_ns=10,
        hits=0,
    )
    cache.full_entries[high_key] = FullCacheEntry(
        slot=cache._full_slot_base + 1,
        cur_bin_path=str(cache._full_bin_path(high_key)),
        cur_ids_len=80,
        raw_prompt_len=80,
        last_used_ns=5,
        hits=3,
    )

    victim_key, _ = cache._select_full_victim_from_entries(cache.full_entries)
    assert victim_key == low_key


def test_score_victim_selector_breaks_ties_by_oldest_last_used(tmp_path):
    cache = make_cache(tmp_path)
    key_a = b"\x31" * 16
    key_b = b"\x32" * 16
    for key in (key_a, key_b):
        cache._full_bin_path(key).write_bytes(b"x" * 20)
        cache._full_meta_path(key).write_text("{}", encoding="utf-8")
    cache.full_entries[key_a] = FullCacheEntry(
        slot=cache._full_slot_base,
        cur_bin_path=str(cache._full_bin_path(key_a)),
        cur_ids_len=20,
        raw_prompt_len=20,
        last_used_ns=1,
        hits=0,
    )
    cache.full_entries[key_b] = FullCacheEntry(
        slot=cache._full_slot_base + 1,
        cur_bin_path=str(cache._full_bin_path(key_b)),
        cur_ids_len=20,
        raw_prompt_len=20,
        last_used_ns=2,
        hits=0,
    )

    victim_key, _ = cache._select_full_victim_from_entries(cache.full_entries)
    assert victim_key == key_a


def test_live_prefix_penalty_makes_strict_prefix_easier_to_evict(tmp_path):
    cache = make_cache(tmp_path)
    prompt_a = [1, 2, 3, 4]
    prompt_b = [1, 2, 3, 4, 5, 6]
    key_a = hash_prefix(prompt_a, cache.kv_k_type, cache.fa_window)
    key_b = hash_prefix(prompt_b, cache.kv_k_type, cache.fa_window)
    for key in (key_a, key_b):
        cache._full_bin_path(key).write_bytes(b"x" * 20)
        cache._full_meta_path(key).write_text("{}", encoding="utf-8")
    cache.full_entries[key_a] = FullCacheEntry(
        slot=cache._full_slot_base,
        cur_bin_path=str(cache._full_bin_path(key_a)),
        cur_ids_len=4,
        raw_prompt_len=len(prompt_a),
        last_used_ns=10,
        hits=0,
    )
    cache.full_entries[key_b] = FullCacheEntry(
        slot=cache._full_slot_base + 1,
        cur_bin_path=str(cache._full_bin_path(key_b)),
        cur_ids_len=6,
        raw_prompt_len=len(prompt_b),
        last_used_ns=5,
        hits=0,
    )

    victim_key, _ = cache._select_full_victim_from_entries(cache.full_entries, prompt_b)
    assert victim_key == key_a


def test_budget_trim_retires_lowest_score_entry(tmp_path):
    cache = make_cache(tmp_path)
    key_a = b"\x41" * 16
    key_b = b"\x42" * 16
    cache._full_bin_path(key_a).write_bytes(b"a" * 90)
    write_meta(cache, key_a, cur_ids_len=10, raw_prompt_len=10, last_used_ns=1)
    cache._full_bin_path(key_b).write_bytes(b"b" * 20)
    write_meta(cache, key_b, cur_ids_len=50, raw_prompt_len=50, last_used_ns=2)
    cache.full_entries[key_a] = FullCacheEntry(
        slot=cache._full_slot_base,
        cur_bin_path=str(cache._full_bin_path(key_a)),
        cur_ids_len=10,
        raw_prompt_len=10,
        last_used_ns=1,
        hits=0,
    )
    cache.full_entries[key_b] = FullCacheEntry(
        slot=cache._full_slot_base + 1,
        cur_bin_path=str(cache._full_bin_path(key_b)),
        cur_ids_len=50,
        raw_prompt_len=50,
        last_used_ns=2,
        hits=0,
    )
    cache._full_budget_bytes = cache._full_entry_artifact_size(key_b, cache.full_entries[key_b]) + 1

    cache._enforce_full_budget()

    assert list(cache.full_entries.keys()) == [key_b]
    assert not cache._full_bin_path(key_a).exists()
    assert not cache._full_meta_path(key_a).exists()


def test_budget_trim_recomputes_next_free_slot(tmp_path):
    cache = make_cache(tmp_path, full_cap=3)
    keys = [bytes([0x51 + i]) * 16 for i in range(3)]
    sizes = [120, 20, 20]
    for idx, (key, size) in enumerate(zip(keys, sizes), start=1):
        cache._full_bin_path(key).write_bytes(b"x" * size)
        write_meta(cache, key, cur_ids_len=10 * idx, raw_prompt_len=10 * idx, last_used_ns=idx)
        cache.full_entries[key] = FullCacheEntry(
            slot=cache._full_slot_base + (idx - 1),
            cur_bin_path=str(cache._full_bin_path(key)),
            cur_ids_len=10 * idx,
            raw_prompt_len=10 * idx,
            last_used_ns=idx,
            hits=0,
        )
    cache._full_next_slot = 0
    cache._full_budget_bytes = sum(
        cache._full_entry_artifact_size(key, cache.full_entries[key]) for key in keys[1:]
    ) + 1

    cache._enforce_full_budget()

    remaining_slots = {entry.slot for entry in cache.full_entries.values()}
    assert cache._full_slot_base not in remaining_slots
    prep = cache.prepare_full_snap([9, 9, 9, 9])
    assert prep is not None
    slot, _ = prep
    assert slot == cache._full_slot_base


def test_lookup_full_updates_hits_in_memory_only(tmp_path):
    prompt_ids = [7, 8, 9, 10]
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"payload")

    cache = make_cache(tmp_path)
    cache.confirm_full_snap(cache._full_slot_base, prompt_ids, source_path, 5)
    key = hash_prefix(prompt_ids, cache.kv_k_type, cache.fa_window)
    before = json.loads(cache._full_meta_path(key).read_text(encoding="utf-8"))

    hit = cache.lookup_full(prompt_ids)

    assert hit is not None
    assert cache.full_entries[key].hits == 1
    after = json.loads(cache._full_meta_path(key).read_text(encoding="utf-8"))
    assert "hits" not in after
    assert after["last_used_ns"] >= before["last_used_ns"]


def test_server_startup_rehydrates_full_cache(tmp_path):
    tokenizer = FakeTokenizer()
    prefill_cfg = PrefillConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.05,
        drafter_gguf=Path("drafter.gguf"),
        drafter_tokenizer_id="Qwen/Qwen3-0.6B",
    )

    with patch("server.subprocess.Popen") as mock_popen, \
            patch("server.PrefixCache.startup_sync", new_callable=AsyncMock) as mock_sync, \
            patch("server.PrefixCache.rehydrate_full_cache",
                  new_callable=AsyncMock, return_value=1) as mock_rehydrate:
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.stdout.readline.return_value = b""

        app = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=4096,
            tokenizer=tokenizer,
            stop_ids={2},
            prefill_cfg=prefill_cfg,
            drafter_tokenizer=tokenizer,
            prefill_cache_slots=2,
        )

        with TestClient(app):
            pass

    mock_sync.assert_awaited_once()
    mock_rehydrate.assert_awaited_once()
