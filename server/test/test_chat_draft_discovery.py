import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CHAT_PATH = Path(__file__).resolve().parents[1] / "examples" / "chat.py"
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


if __name__ == "__main__":
    unittest.main()