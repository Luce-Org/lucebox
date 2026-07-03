"""Unit tests: prefix cache slot depth vs lookup cut (cross-session safety)."""
import unittest
from unittest.mock import MagicMock, patch

from prefix_cache import PrefixCache, hash_prefix


class _FakeTokenizer:
  pass


class PrefixCacheSlotDepthTests(unittest.TestCase):
    def _make_cache(self) -> PrefixCache:
        pc = PrefixCache(
            daemon_stdin=MagicMock(),
            await_reply=MagicMock(),
            daemon_lock=MagicMock(),
            tokenizer=_FakeTokenizer(),
            kv_k_type="f16",
            fa_window=0,
            cap=4,
        )
        self.assertFalse(pc.disabled)
        return pc

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_lookup_rejects_stale_shallow_hash(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]

        shallow_key = hash_prefix(ids[:376], pc.kv_k_type, pc.fa_window)
        pc.entries[shallow_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 2411

        self.assertIsNone(pc.lookup(ids))
        self.assertNotIn(shallow_key, pc.entries)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_lookup_accepts_matching_depth(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]

        key = hash_prefix(ids[:376], pc.kv_k_type, pc.fa_window)
        pc.entries[key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376

        self.assertEqual(pc.lookup(ids), (0, 376))

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_confirm_evicts_stale_keys_for_slot(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(500))
        mock_bounds.return_value = [100, 376, 480]

        shallow = hash_prefix(ids[:376], pc.kv_k_type, pc.fa_window)
        deep = hash_prefix(ids[:480], pc.kv_k_type, pc.fa_window)
        pc.entries[shallow] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376

        pc.confirm_inline_snap(0, 480, ids)

        self.assertNotIn(shallow, pc.entries)
        self.assertIn(deep, pc.entries)
        self.assertEqual(pc.entries[deep], 0)
        self.assertEqual(pc._slot_prefix_len[0], 480)


if __name__ == "__main__":
    unittest.main()
