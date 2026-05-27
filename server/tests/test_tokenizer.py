#!/usr/bin/env python3
"""Unit test for the C++ BPE tokenizer.

Compares the C++ tokenizer (via test_tokenizer_harness) against the
HuggingFace reference tokenizer for Qwen3-0.6B. Tests encode, decode,
token_text, special tokens, and edge cases.

Usage:
  python3 dflash/tests/test_tokenizer.py [--model dflash/models/Qwen3-0.6B-BF16.gguf]
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# HuggingFace reference tokenizer
# ---------------------------------------------------------------------------
def load_hf_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    except Exception as e:
        print(f"WARNING: Could not load HF tokenizer: {e}")
        return None


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TokenizerTest:
    def __init__(self, cpp: CppTokenizer, hf):
        self.cpp = cpp
        self.hf = hf
        self.passed = 0
        self.failed = 0

    def _report(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {detail}")

    def test_info(self):
        print("\n[1] Tokenizer info")
        info = self.cpp.info()
        self._report("vocab_size == 151936",
                     info["vocab_size"] == 151936,
                     f"got {info['vocab_size']}")
        self._report("eos_id == 151645",
                     info["eos_id"] == 151645,
                     f"got {info['eos_id']}")

    def test_encode_basic(self):
        print("\n[2] Basic encode")
        cases = [
            ("Hello, world!", [9707, 11, 1879, 0]),
            ("", []),
            ("a", None),  # just check non-empty
            ("Hello", [9707]),
        ]
        for text, expected in cases:
            ids = self.cpp.encode(text)
            if expected is None:
                self._report(f"encode({text!r}) non-empty",
                             len(ids) > 0, f"got {ids}")
            else:
                self._report(f"encode({text!r}) == {expected}",
                             ids == expected,
                             f"got {ids}")

    def test_encode_vs_hf(self):
        print("\n[3] Encode vs HuggingFace")
        if not self.hf:
            print("  ⏭ Skipped (no HF tokenizer)")
            return

        test_strings = [
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

        mismatches = 0
        for text in test_strings:
            cpp_ids = self.cpp.encode(text)
            hf_ids = self.hf.encode(text, add_special_tokens=False)

            ok = cpp_ids == hf_ids
            if not ok:
                mismatches += 1
                # Show first difference
                for i in range(max(len(cpp_ids), len(hf_ids))):
                    c = cpp_ids[i] if i < len(cpp_ids) else "END"
                    h = hf_ids[i] if i < len(hf_ids) else "END"
                    if c != h:
                        self._report(
                            f"encode({text[:30]!r}...)" if len(text) > 30 else f"encode({text!r})",
                            False,
                            f"differ at pos {i}: cpp={c} hf={h} "
                            f"(cpp_len={len(cpp_ids)} hf_len={len(hf_ids)})")
                        break
            else:
                self._report(
                    f"encode({text[:40]!r}...)" if len(text) > 40 else f"encode({text!r})",
                    True)

        if mismatches == 0:
            print(f"    → All {len(test_strings)} strings match HF tokenizer!")

    def test_decode_basic(self):
        print("\n[4] Basic decode")
        cases = [
            ([9707, 11, 1879, 0], "Hello, world!"),
        ]
        for ids, expected in cases:
            text = self.cpp.decode(ids)
            self._report(f"decode({ids}) == {expected!r}",
                         text == expected,
                         f"got {text!r}")

    def test_roundtrip(self):
        print("\n[5] Encode → decode roundtrip")
        test_strings = [
            "Hello, world!",
            "def main():\n    print('hello')\n",
            "The temperature is -5°C today.",
            "Привет, мир!",
            "🎉 Party time! 🎊",
            "   spaces   ",
        ]
        for text in test_strings:
            ids = self.cpp.encode(text)
            decoded = self.cpp.decode(ids)
            self._report(f"roundtrip({text[:40]!r})",
                         decoded == text,
                         f"decoded={decoded!r}")

    def test_token_text_gpt2_decode(self):
        print("\n[6] GPT-2 byte decode in token_text()")
        # Token 1879 is "Ġworld" in GPT-2 encoding → " world" decoded
        text = self.cpp.token_text(1879)
        self._report('token_text(1879) == " world"',
                     text == " world",
                     f"got {text!r}")

        # Token 9707 is "Hello" — no GPT-2 encoding needed (all printable)
        text = self.cpp.token_text(9707)
        self._report('token_text(9707) == "Hello"',
                     text == "Hello",
                     f"got {text!r}")

        # Check a few more tokens with spaces
        if self.hf:
            for tid in [220, 262, 383, 279, 198]:  # common tokens with Ġ prefix
                cpp_text = self.cpp.token_text(tid)
                hf_text = self.hf.decode([tid])
                self._report(f"token_text({tid}) matches HF",
                             cpp_text == hf_text,
                             f"cpp={cpp_text!r} hf={hf_text!r}")

    def test_special_tokens(self):
        print("\n[7] Special tokens")
        # <|im_start|> should be returned as-is
        im_start_id = 151644
        text = self.cpp.token_text(im_start_id)
        self._report('token_text(151644) == "<|im_start|>"',
                     text == "<|im_start|>",
                     f"got {text!r}")

        im_end_id = 151645
        text = self.cpp.token_text(im_end_id)
        self._report('token_text(151645) == "<|im_end|>"',
                     text == "<|im_end|>",
                     f"got {text!r}")

        # raw_token should return the GPT-2 encoded form
        raw = self.cpp.raw_token(im_start_id)
        self._report('raw_token(151644) == "<|im_start|>"',
                     raw == "<|im_start|>",
                     f"got {raw!r}")

    def test_out_of_range(self):
        print("\n[8] Edge cases")
        # Out of range
        text = self.cpp.token_text(-1)
        self._report('token_text(-1) == ""', text == "", f"got {text!r}")

        text = self.cpp.token_text(999999)
        self._report('token_text(999999) == ""', text == "", f"got {text!r}")

        # Empty encode
        ids = self.cpp.encode("")
        self._report('encode("") == []', ids == [], f"got {ids}")

    def test_decode_vs_hf(self):
        print("\n[9] Decode vs HuggingFace")
        if not self.hf:
            print("  ⏭ Skipped (no HF tokenizer)")
            return

        # Use HF to encode some strings, then compare both decoders
        test_strings = [
            "Hello, world!",
            "I'm going to the store.",
            "int main() { return 0; }",
            "The price is $19.99.",
        ]
        for text in test_strings:
            hf_ids = self.hf.encode(text, add_special_tokens=False)
            cpp_decoded = self.cpp.decode(hf_ids)
            hf_decoded = self.hf.decode(hf_ids)
            self._report(f"decode HF tokens for {text[:30]!r}",
                         cpp_decoded == hf_decoded,
                         f"cpp={cpp_decoded!r} hf={hf_decoded!r}")

    def run_all(self):
        print("=" * 60)
        print("Tokenizer Unit Tests")
        print("=" * 60)

        self.test_info()
        self.test_encode_basic()
        self.test_encode_vs_hf()
        self.test_decode_basic()
        self.test_roundtrip()
        self.test_token_text_gpt2_decode()
        self.test_special_tokens()
        self.test_out_of_range()
        self.test_decode_vs_hf()

        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="Tokenizer unit test")
    parser.add_argument("--model", default="dflash/models/Qwen3-0.6B-BF16.gguf",
                        help="GGUF model path")
    parser.add_argument("--harness", default=None,
                        help="Path to test_tokenizer_harness binary")
    args = parser.parse_args()

    # Find harness binary.
    harness = args.harness
    if not harness:
        candidates = [
            "dflash/build/test_tokenizer_harness",
            "build/test_tokenizer_harness",
        ]
        for c in candidates:
            if os.path.isfile(c):
                harness = c
                break
    if not harness or not os.path.isfile(harness):
        print(f"ERROR: Could not find test_tokenizer_harness binary")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print(f"Harness: {harness}")
    print(f"Model:   {args.model}")

    cpp = CppTokenizer(harness, args.model)
    hf = load_hf_tokenizer()
    if hf:
        print(f"HF ref:  Qwen/Qwen3-0.6B (vocab={hf.vocab_size})")

    try:
        tester = TokenizerTest(cpp, hf)
        ok = tester.run_all()
        sys.exit(0 if ok else 1)
    finally:
        cpp.close()


if __name__ == "__main__":
    main()
