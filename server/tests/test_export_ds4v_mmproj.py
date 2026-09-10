#!/usr/bin/env python3
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = REPO_ROOT / "server" / "tools" / "export_ds4v_mmproj.py"
SPEC = importlib.util.spec_from_file_location("export_ds4v_mmproj", EXPORTER_PATH)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


TEST_SHAPES = {
    "aligner.tiny": (2,),
    "image_start": (3,),
    "vision.tiny": (2, 2),
}


def exact_config():
    return dict(exporter.EXPECTED_CONFIG)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors(path, entries, payload):
    raw_header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def default_entries():
    entries = {}
    payload = bytearray()
    payloads = {}
    for number, (name, shape) in enumerate(sorted(TEST_SHAPES.items()), start=1):
        nbytes = 2
        for dim in shape:
            nbytes *= dim
        raw = bytes(((number * 17 + offset) % 256 for offset in range(nbytes)))
        start = len(payload)
        payload += raw
        entries[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [start, len(payload)]}
        payloads[name] = raw
    return entries, bytes(payload), payloads


def make_source(root, entries=None, payload=None, weight_map=None, config=None):
    source = root / "parent"
    source.mkdir()
    write_json(source / "config.json", exact_config() if config is None else config)
    if entries is None or payload is None:
        default_header, default_payload, _ = default_entries()
        entries = default_header if entries is None else entries
        payload = default_payload if payload is None else payload
    write_safetensors(source / "model-00001-of-00001.safetensors", entries, payload)
    if weight_map is None:
        weight_map = {name: "model-00001-of-00001.safetensors" for name in TEST_SHAPES}
    write_json(source / "model.safetensors.index.json", {"weight_map": weight_map})
    return source


def read_string(handle):
    length = struct.unpack("<Q", handle.read(8))[0]
    return handle.read(length).decode("utf-8")


def read_gguf(path):
    metadata = {}
    infos = []
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise AssertionError("bad GGUF magic")
        version, tensor_count, kv_count = struct.unpack("<IQQ", handle.read(20))
        for _ in range(kv_count):
            key = read_string(handle)
            value_type = struct.unpack("<I", handle.read(4))[0]
            if value_type == exporter.GGUF_TYPE_UINT32:
                value = struct.unpack("<I", handle.read(4))[0]
            elif value_type == exporter.GGUF_TYPE_FLOAT32:
                value = struct.unpack("<f", handle.read(4))[0]
            elif value_type == exporter.GGUF_TYPE_STRING:
                value = read_string(handle)
            elif value_type == exporter.GGUF_TYPE_ARRAY:
                subtype, count = struct.unpack("<IQ", handle.read(12))
                if subtype != exporter.GGUF_TYPE_FLOAT32:
                    raise AssertionError("unexpected array subtype")
                value = struct.unpack("<%df" % count, handle.read(4 * count))
            else:
                raise AssertionError("unexpected metadata type %d" % value_type)
            metadata[key] = value
        for _ in range(tensor_count):
            name = read_string(handle)
            dimensions = struct.unpack("<I", handle.read(4))[0]
            ggml_shape = struct.unpack("<%dQ" % dimensions, handle.read(8 * dimensions))
            tensor_type, offset = struct.unpack("<IQ", handle.read(12))
            infos.append((name, tuple(reversed(ggml_shape)), tensor_type, offset))
        data_start = exporter._align(handle.tell())
    raw = path.read_bytes()
    return version, metadata, infos, data_start, raw


class ExportDs4vMmprojTest(unittest.TestCase):
    def export_small(self, source, output):
        with mock.patch.object(exporter, "expected_shapes", return_value=TEST_SHAPES):
            return exporter.export_mmproj(source, output)

    def test_lossless_export_has_names_shapes_types_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries, payload, payloads = default_entries()
            source = make_source(root, entries, payload)
            output = root / "vision.gguf"

            count, payload_bytes = self.export_small(source, output)
            version, metadata, infos, data_start, raw = read_gguf(output)

            self.assertEqual(count, len(TEST_SHAPES))
            self.assertEqual(payload_bytes, sum(map(len, payloads.values())))
            self.assertEqual(version, 3)
            self.assertEqual(metadata["general.architecture"], "deepseek4_vision")
            self.assertEqual(metadata["deepseek4.vision.block_count"], 32)
            self.assertEqual(metadata["deepseek4.vision.embedding_length"], 1024)
            self.assertEqual(metadata["deepseek4.vision.attention.head_count"], 16)
            self.assertEqual(metadata["deepseek4.vision.patch_size"], 14)
            self.assertEqual(metadata["deepseek4.vision.feed_forward_length"], 2816)
            self.assertEqual(metadata["deepseek4.vision.downsample_ratio"], 3)
            self.assertEqual(metadata["deepseek4.vision.language_embedding_length"], 4096)
            self.assertEqual(metadata["deepseek4.vision.vocabulary_size"], 129280)
            self.assertAlmostEqual(metadata["deepseek4.vision.rope.freq_base"], 10000.0)
            self.assertAlmostEqual(metadata["deepseek4.vision.attention.layer_norm_rms_epsilon"], 1e-6)
            self.assertEqual(metadata["deepseek4.vision.image.max_tokens"], 384)
            self.assertEqual(metadata["deepseek4.vision.image.min_pixels"], 147456)
            self.assertEqual(metadata["deepseek4.vision.image.max_aspect_ratio"], 8.0)
            self.assertEqual(metadata["deepseek4.vision.image.normalization_mean"], (0.5, 0.5, 0.5))
            self.assertEqual(metadata["deepseek4.vision.image.normalization_std"], (0.5, 0.5, 0.5))
            self.assertEqual(metadata["deepseek4.vision.image.layout_version"], 1)
            self.assertEqual(metadata["deepseek4.vision.image.layout_recipe"], "row-pair-column-interleave")
            self.assertEqual(metadata["deepseek4.vision.image.compression_alignment"], 4)

            self.assertEqual([info[0] for info in infos], sorted(TEST_SHAPES))
            for name, shape, tensor_type, offset in infos:
                self.assertEqual(shape, TEST_SHAPES[name])
                self.assertEqual(tensor_type, exporter.GGML_TYPE_BF16)
                self.assertEqual(raw[data_start + offset:data_start + offset + len(payloads[name])], payloads[name])

    def test_vendored_gguf_reader_accepts_output(self):
        gguf_path = REPO_ROOT / "server" / "deps" / "llama.cpp" / "gguf-py"
        sys.path.insert(0, str(gguf_path))
        try:
            import gguf
        except ImportError as exc:
            self.skipTest("vendored GGUF reader dependency unavailable: %s" % exc)
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries, payload, payloads = default_entries()
            output = root / "vision.gguf"
            self.export_small(make_source(root, entries, payload), output)

            reader = gguf.GGUFReader(output)
            self.assertEqual(reader.get_field("general.architecture").contents(), "deepseek4_vision")
            self.assertEqual(len(reader.tensors), len(TEST_SHAPES))
            for tensor in reader.tensors:
                self.assertEqual(tensor.tensor_type.name, "BF16")
                self.assertEqual(tuple(reversed(tensor.shape.tolist())), TEST_SHAPES[tensor.name])
                self.assertEqual(tensor.data.tobytes(), payloads[tensor.name])

    def test_missing_tensor_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source(root)
            index_path = source / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"]["image_start"]
            write_json(index_path, index)
            output = root / "vision.gguf"
            with self.assertRaisesRegex(exporter.ExportError, "incomplete projector inventory"):
                self.export_small(source, output)
            self.assertFalse(output.exists())

    def test_extra_projector_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source(root)
            index_path = source / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["vision.unexpected"] = "model-00001-of-00001.safetensors"
            write_json(index_path, index)
            with self.assertRaisesRegex(exporter.ExportError, "unexpected: vision.unexpected"):
                self.export_small(source, root / "vision.gguf")

    def test_wrong_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries, payload, _ = default_entries()
            entries["image_start"]["shape"] = [1, 3]
            source = make_source(root, entries, payload)
            with self.assertRaisesRegex(exporter.ExportError, "shape must be"):
                self.export_small(source, root / "vision.gguf")

    def test_non_bf16_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries, payload, _ = default_entries()
            entries["image_start"]["dtype"] = "F16"
            source = make_source(root, entries, payload)
            with self.assertRaisesRegex(exporter.ExportError, "must remain BF16"):
                self.export_small(source, root / "vision.gguf")

    def test_out_of_bounds_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries, payload, _ = default_entries()
            entries["image_start"]["data_offsets"] = [0, len(payload) + 2]
            source = make_source(root, entries, payload)
            with self.assertRaisesRegex(exporter.ExportError, "exceed safetensors bounds"):
                self.export_small(source, root / "vision.gguf")

    def test_overlapping_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = {
                "aligner.tiny": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                "image_start": {"dtype": "BF16", "shape": [3], "data_offsets": [2, 8]},
                "vision.tiny": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]},
            }
            source = make_source(root, entries, bytes(range(16)))
            with self.assertRaisesRegex(exporter.ExportError, "overlapping projector tensors"):
                self.export_small(source, root / "vision.gguf")

    def test_shard_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight_map = {name: "../outside.safetensors" for name in TEST_SHAPES}
            source = make_source(root, weight_map=weight_map)
            with self.assertRaisesRegex(exporter.ExportError, "unsafe safetensors shard path"):
                self.export_small(source, root / "vision.gguf")

    def test_output_collision_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source(root)
            output = root / "vision.gguf"
            output.write_bytes(b"keep me")
            with self.assertRaisesRegex(exporter.ExportError, "refusing to overwrite"):
                self.export_small(source, output)
            self.assertEqual(output.read_bytes(), b"keep me")

    def test_config_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = exact_config()
            config["vision_patch_size"] = 16
            source = make_source(root, config=config)
            with self.assertRaisesRegex(exporter.ExportError, "vision_patch_size must be 14"):
                self.export_small(source, root / "vision.gguf")

    def test_copy_failure_has_no_partial_final_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source(root)
            output = root / "vision.gguf"

            def fail_after_bytes(handle, tensor):
                handle.write(b"partial")
                raise exporter.ExportError("injected copy failure")

            with mock.patch.object(exporter, "expected_shapes", return_value=TEST_SHAPES), \
                    mock.patch.object(exporter, "_copy_tensor", side_effect=fail_after_bytes):
                with self.assertRaisesRegex(exporter.ExportError, "injected copy failure"):
                    exporter.export_mmproj(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".vision.gguf.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
