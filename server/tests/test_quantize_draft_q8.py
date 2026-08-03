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


SERVER_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SERVER_DIR / "scripts" / "quantize_draft_q8.py"
SPEC = importlib.util.spec_from_file_location("quantize_draft_q8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RecordingWriter:
    def __init__(self):
        self.calls = []

    def add_uint32(self, key, value):
        self.calls.append(("uint32", key, value))

    def add_array(self, key, value):
        self.calls.append(("array", key, value))


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

    def test_profile_is_opt_in(self):
        writer = RecordingWriter()
        MODULE.add_qwen36_swa_metadata(writer, False)
        self.assertEqual(writer.calls, [])

    def test_profile_writes_exact_loader_keys(self):
        writer = RecordingWriter()
        MODULE.add_qwen36_swa_metadata(writer, True)
        self.assertEqual(
            writer.calls,
            [
                (
                    "uint32",
                    "qwen35-dflash-draft.attention.sliding_window",
                    2048,
                ),
                (
                    "array",
                    "qwen35-dflash-draft.attention.sliding_window_pattern",
                    [True, True, True, True, False],
                ),
            ],
        )

    def test_cli_default_preserves_qwen35_behavior(self):
        args = MODULE.build_arg_parser().parse_args(["in.safetensors", "out.gguf"])
        self.assertFalse(args.qwen36_swa)

    def test_cli_enables_qwen36_profile(self):
        args = MODULE.build_arg_parser().parse_args(
            ["in.safetensors", "out.gguf", "--qwen36-swa"]
        )
        self.assertTrue(args.qwen36_swa)

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
                    else:
                        self.assertIsNone(window)
                        self.assertIsNone(pattern)

    def test_gguf_round_trip_preserves_types_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.gguf"
            writer = MODULE.gguf.GGUFWriter(path, MODULE.ARCH)
            MODULE.add_qwen36_swa_metadata(writer, True)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = MODULE.gguf.GGUFReader(path)
            window = reader.get_field(
                "qwen35-dflash-draft.attention.sliding_window"
            )
            pattern = reader.get_field(
                "qwen35-dflash-draft.attention.sliding_window_pattern"
            )
            self.assertIsNotNone(window)
            self.assertIsNotNone(pattern)
            self.assertEqual(window.contents(), 2048)
            self.assertEqual(pattern.contents(), [True, True, True, True, False])
            self.assertEqual([kind.name for kind in window.types], ["UINT32"])
            self.assertEqual(
                [kind.name for kind in pattern.types], ["ARRAY", "BOOL"]
            )


if __name__ == "__main__":
    unittest.main()