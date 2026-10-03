"""Unit test for the C++ BPE tokenizer.

Compares the C++ tokenizer (via test_tokenizer_harness) against the
HuggingFace reference tokenizer for Qwen3-0.6B. Tests encode, decode,
token_text, special tokens, and edge cases.

Usage:
  pytest server/test/python/test_tokenizer.py -v

Overrides:
  LUCE_TOKENIZER_MODEL    Qwen3-0.6B GGUF path
                          (default: server/models/Qwen3-0.6B-BF16.gguf)
  LUCE_TOKENIZER_HARNESS  path to test_tokenizer_harness binary
                          (default: server/build/test_tokenizer_harness)

HF-comparison tests skip automatically when transformers/HF access is
unavailable; everything skips when the harness binary or model is missing.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

SERVER_DIR = Path(__file__).resolve().parents[2]
HARNESS = Path(os.environ.get(
    "LUCE_TOKENIZER_HARNESS", SERVER_DIR / "build/test_tokenizer_harness"))
MODEL = Path(os.environ.get(
    "LUCE_TOKENIZER_MODEL", SERVER_DIR / "models/Qwen3-0.6B-BF16.gguf"))


# ---------------------------------------------------------------------------
# C++ tokenizer harness (subprocess)
# ---------------------------------------------------------------------------
class CppTokenizer:
    def __init__(self, harness_path: str, model_path: str):
        self.proc = subprocess.Popen(
            [harness_path, model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ready signal on stderr.
        while True:
            line = self.proc.stderr.readline().decode()
            if "ready" in line:
                break
            if not line:
                raise RuntimeError("Harness exited before ready")

    def _call(self, obj: dict) -> dict:
        line = json.dumps(obj) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().decode().strip()
        if not resp:
            raise RuntimeError("Empty response from harness")
        return json.loads(resp)

    def encode(self, text: str) -> list[int]:
        return self._call({"cmd": "encode", "text": text})["ids"]

    def decode(self, ids: list[int]) -> str:
        return self._call({"cmd": "decode", "ids": ids})["text"]

    def token_text(self, id: int) -> str:
        return self._call({"cmd": "token_text", "id": id})["text"]

    def raw_token(self, id: int) -> str:
        return self._call({"cmd": "raw_token", "id": id})["text"]

    def info(self) -> dict:
        return self._call({"cmd": "info"})

    def close(self):
        try:
            self.proc.stdin.write(b'{"cmd":"quit"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture(scope="module")
def cpp():
    if not HARNESS.is_file():
        pytest.skip(f"test_tokenizer_harness binary not found at {HARNESS}")
    if not MODEL.is_file():
        pytest.skip(f"model file not found at {MODEL}")
    tok = CppTokenizer(str(HARNESS), str(MODEL))
    yield tok
    tok.close()


@pytest.fixture(scope="module")
def hf():
    """HF reference tokenizer, or None when unavailable (HF-dependent checks
    skip themselves)."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",
                                             trust_remote_code=True)
    except Exception as e:
        print(f"WARNING: Could not load HF tokenizer: {e}")
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
ENCODE_VS_HF_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's not butter!",
    "2+2=4",
    "def foo(x):\n    return x * 2\n",
    "こんにちは世界",
    "Hello\n\nWorld",
    " leading space",
    "trailing space ",
    "  multiple   spaces  ",
    "tab\there",
    "newline\nhere",
    "Hello, how are you? I'm fine, thanks!",
    "<think>This is reasoning</think>",
    '{"name": "test", "value": 42}',
    "A" * 100,
    "1234567890",
    "!@#$%^&*()",
    "mixed 123 CAPS lower πλατφόρμα",
    "",
]


class TestTokenizer:
    def test_info(self, cpp):
        info = cpp.info()
        assert info["vocab_size"] == 151936, f"got {info['vocab_size']}"
        assert info["eos_id"] == 151645, f"got {info['eos_id']}"

    @pytest.mark.parametrize("text, expected", [
        ("Hello, world!", [9707, 11, 1879, 0]),
        ("", []),
        ("Hello", [9707]),
    ])
    def test_encode_basic(self, cpp, text, expected):
        ids = cpp.encode(text)
        assert ids == expected, f"encode({text!r}) = {ids}"

    def test_encode_single_char_nonempty(self, cpp):
        ids = cpp.encode("a")
        assert len(ids) > 0, f"encode('a') empty: {ids}"

    @pytest.mark.parametrize("text", ENCODE_VS_HF_STRINGS)
    def test_encode_vs_hf(self, cpp, hf, text):
        if hf is None:
            pytest.skip("no HF tokenizer available")
        cpp_ids = cpp.encode(text)
        hf_ids = hf.encode(text, add_special_tokens=False)
        if cpp_ids != hf_ids:
            pos = next(i for i in range(max(len(cpp_ids), len(hf_ids)))
                       if i >= len(cpp_ids) or i >= len(hf_ids)
                       or cpp_ids[i] != hf_ids[i])
            c = cpp_ids[pos] if pos < len(cpp_ids) else "END"
            h = hf_ids[pos] if pos < len(hf_ids) else "END"
            pytest.fail(f"encode({text[:30]!r}) differ at pos {pos}: "
                        f"cpp={c} hf={h} "
                        f"(cpp_len={len(cpp_ids)} hf_len={len(hf_ids)})")

    def test_decode_basic(self, cpp):
        assert cpp.decode([9707, 11, 1879, 0]) == "Hello, world!"

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "def main():\n    print('hello')\n",
        "The temperature is -5°C today.",
        "Привет, мир!",
        "🎉 Party time! 🎊",
        "   spaces   ",
    ])
    def test_roundtrip(self, cpp, text):
        decoded = cpp.decode(cpp.encode(text))
        assert decoded == text, \
            f"roundtrip({text[:40]!r}) decoded={decoded!r}"

    def test_token_text_gpt2_decode(self, cpp):
        # Token 1879 is "Ġworld" in GPT-2 encoding → " world" decoded
        assert cpp.token_text(1879) == " world", \
            f"got {cpp.token_text(1879)!r}"

        # Token 9707 is "Hello" — no GPT-2 encoding needed (all printable)
        assert cpp.token_text(9707) == "Hello", f"got {cpp.token_text(9707)!r}"

    # Common tokens with the Ġ (space) prefix.
    @pytest.mark.parametrize("tid", [220, 262, 383, 279, 198])
    def test_token_text_vs_hf(self, cpp, hf, tid):
        if hf is None:
            pytest.skip("no HF tokenizer available")
        cpp_text = cpp.token_text(tid)
        hf_text = hf.decode([tid])
        assert cpp_text == hf_text, \
            f"token_text({tid}): cpp={cpp_text!r} hf={hf_text!r}"

    def test_special_tokens(self, cpp):
        # <|im_start|> should be returned as-is
        assert cpp.token_text(151644) == "<|im_start|>"
        assert cpp.token_text(151645) == "<|im_end|>"

        # raw_token should return the GPT-2 encoded form
        assert cpp.raw_token(151644) == "<|im_start|>", \
            f"got {cpp.raw_token(151644)!r}"

    def test_out_of_range(self, cpp):
        assert cpp.token_text(-1) == "", f"got {cpp.token_text(-1)!r}"
        assert cpp.token_text(999999) == "", f"got {cpp.token_text(999999)!r}"
        assert cpp.encode("") == [], f"got {cpp.encode('')}"

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "I'm going to the store.",
        "int main() { return 0; }",
        "The price is $19.99.",
    ])
    def test_decode_vs_hf(self, cpp, hf, text):
        # Use HF to encode, then compare both decoders.
        if hf is None:
            pytest.skip("no HF tokenizer available")
        hf_ids = hf.encode(text, add_special_tokens=False)
        cpp_decoded = cpp.decode(hf_ids)
        hf_decoded = hf.decode(hf_ids)
        assert cpp_decoded == hf_decoded, \
            f"{text[:30]!r}: cpp={cpp_decoded!r} hf={hf_decoded!r}"
