#!/usr/bin/env python3
"""Regression tests for Q8 draft conversion metadata profiles."""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2]
SCRIPT = SERVER_DIR / "scripts" / "quantize_draft_q8.py"
SPEC = importlib.util.spec_from_file_location("quantize_draft_q8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Qwen36SwaMetadataTest(unittest.TestCase):
    @staticmethod
    def _write_minimal_safetensors(path):
        header = json.dumps(
            {
                "hidden_norm.weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }
            },
            separators=(",", ":"),
        ).encode()
        payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        path.write_bytes(payload)

    def test_converter_cli_writes_profile_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "minimal.safetensors"
            self._write_minimal_safetensors(source)
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    output = tmp / f"draft-{enabled}.gguf"
                    command = [sys.executable, str(SCRIPT), str(source), str(output)]
                    if enabled:
                        command.append("--qwen36-swa")
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    reader = MODULE.gguf.GGUFReader(output)
                    window = reader.get_field(
                        "qwen35-dflash-draft.attention.sliding_window"
                    )
                    pattern = reader.get_field(
                        "qwen35-dflash-draft.attention.sliding_window_pattern"
                    )
                    if enabled:
                        self.assertEqual(window.contents(), 2048)
                        self.assertEqual(
                            pattern.contents(), [True, True, True, True, False]
                        )
                        # The C++ draft loader reads these GGUF types.
                        self.assertEqual([kind.name for kind in window.types], ["UINT32"])
                        self.assertEqual(
                            [kind.name for kind in pattern.types], ["ARRAY", "BOOL"]
                        )
                    else:
                        self.assertIsNone(window)
                        self.assertIsNone(pattern)


if __name__ == "__main__":
    unittest.main()