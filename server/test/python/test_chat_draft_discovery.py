import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CHAT_PATH = Path(__file__).resolve().parents[2] / "examples" / "chat.py"
SPEC = importlib.util.spec_from_file_location("chat_example", CHAT_PATH)
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


class DraftDiscoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def create_file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def resolve(self, root=None, target="Qwen3.6-27B-Q4_K_M.gguf"):
        with patch.object(chat, "DRAFT_ROOT", root or self.root):
            with patch.object(chat, "TARGET", Path(target)):
                return chat.resolve_draft()

    def test_discovers_nested_gguf(self):
        draft = self.create_file("nested/dflash-draft-3.6-q4_k_m.gguf")
        self.assertEqual(self.resolve(), draft)

    def test_named_draft_takes_priority(self):
        self.create_file("aaa.gguf")
        self.create_file("model.safetensors")
        draft = self.create_file("dflash-draft-3.6-q4_k_m.gguf")
        self.assertEqual(self.resolve(), draft)

    def test_generic_gguf_takes_priority_over_safetensors(self):
        self.create_file("model.safetensors")
        draft = self.create_file("custom.gguf")
        self.assertEqual(self.resolve(), draft)

    def test_prefers_matching_target_version(self):
        self.create_file("dflash-draft-3.5-q4_k_m.gguf")
        draft = self.create_file("dflash-draft-3.6-q4_k_m.gguf")
        self.assertEqual(self.resolve(), draft)

    def test_falls_back_to_safetensors(self):
        draft = self.create_file("nested/model.safetensors")
        self.assertEqual(self.resolve(), draft)

    def test_accepts_explicit_file(self):
        draft = self.create_file("custom.gguf")
        self.assertEqual(self.resolve(root=draft), draft)

    def test_selection_is_deterministic_without_version_match(self):
        self.create_file("z.gguf")
        draft = self.create_file("a.gguf")
        self.assertEqual(self.resolve(target="other-model.gguf"), draft)

    def test_ignores_directories_named_like_weights(self):
        (self.root / "fake.gguf").mkdir()
        draft = self.create_file("model.safetensors")
        self.assertEqual(self.resolve(), draft)

    def test_empty_directory_reports_missing_weights(self):
        with self.assertRaisesRegex(SystemExit, "no DFlash draft"):
            self.resolve()

    def test_missing_directory_reports_missing_weights(self):
        with self.assertRaisesRegex(SystemExit, "no DFlash draft"):
            self.resolve(root=self.root / "missing")

    def test_warns_when_target_version_has_no_match(self):
        draft = self.create_file("dflash-draft-3.5-q4_k_m.gguf")
        with patch.object(chat.sys, "stderr", new_callable=io.StringIO) as output:
            self.assertEqual(self.resolve(), draft)

        self.assertIn("no draft filename matches Qwen3.6", output.getvalue())
        self.assertIn(str(draft), output.getvalue())

    def test_warns_when_multiple_candidates_remain(self):
        draft = self.create_file("dflash-draft-3.6-bf16.gguf")
        self.create_file("dflash-draft-3.6-q4_k_m.gguf")
        with patch.object(chat.sys, "stderr", new_callable=io.StringIO) as output:
            self.assertEqual(self.resolve(), draft)

        self.assertIn("multiple draft files found", output.getvalue())
        self.assertIn(str(draft), output.getvalue())

    def test_single_matching_draft_has_no_warning(self):
        draft = self.create_file("dflash-draft-3.6-q4_k_m.gguf")
        with patch.object(chat.sys, "stderr", new_callable=io.StringIO) as output:
            self.assertEqual(self.resolve(), draft)

        self.assertEqual(output.getvalue(), "")

    def test_safetensors_fallback_has_no_version_warning(self):
        draft = self.create_file("model.safetensors")
        with patch.object(chat.sys, "stderr", new_callable=io.StringIO) as output:
            self.assertEqual(self.resolve(), draft)

        self.assertEqual(output.getvalue(), "")

if __name__ == "__main__":
    unittest.main()