#!/usr/bin/env python3
"""Extract the DeepSeek-V4 vision projector from parent safetensors.

The exporter is intentionally standard-library-only. It copies the BF16 tensor
payloads byte for byte and writes a small GGUF v3 header around them; it never
loads, converts, or quantizes tensor values.
"""

import argparse
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_BF16 = 30

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class ExportError(RuntimeError):
    """A malformed source or unsafe output prevented export."""


@dataclass(frozen=True)
class SourceTensor:
    name: str
    path: Path
    shape: Tuple[int, ...]
    dtype: str
    file_offset: int
    nbytes: int
    file_size: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class OutputTensor:
    source: SourceTensor
    data_offset: int


def _is_projector_tensor(name: str) -> bool:
    return name.startswith("vision.") or name.startswith("aligner.") or name.startswith("image_")


def expected_shapes() -> Dict[str, Tuple[int, ...]]:
    """Return the complete, source-name-preserving DS4V projector inventory."""
    shapes = {
        "aligner.w1.bias": (4096,),
        "aligner.w1.weight": (4096, 9216),
        "aligner.w2.bias": (4096,),
        "aligner.w2.weight": (4096, 4096),
        "image_end": (4096,),
        "image_newline": (4096,),
        "image_pad": (4096,),
        "image_start": (4096,),
        "vision.norm.weight": (1024,),
        "vision.patch_embed.proj.bias": (1024,),
        "vision.patch_embed.proj.weight": (1024, 3 * 14 * 14),
    }
    for block in range(32):
        prefix = "vision.blocks.%d" % block
        shapes.update({
            prefix + ".attn.wo.bias": (1024,),
            prefix + ".attn.wo.weight": (1024, 1024),
            prefix + ".attn.wqkv.bias": (3 * 1024,),
            prefix + ".attn.wqkv.weight": (3 * 1024, 1024),
            prefix + ".mlp.w1.weight": (2 * 2816, 1024),
            prefix + ".mlp.w2.weight": (1024, 2816),
            prefix + ".norm1.weight": (1024,),
            prefix + ".norm2.weight": (1024,),
        })
    return shapes


EXPECTED_CONFIG = {
    "vision_n_layers": 32,
    "vision_dim": 1024,
    "vision_n_heads": 16,
    "vision_inter_dim": 2816,
    "vision_patch_size": 14,
    "vision_rope_theta": 10000.0,
    "vision_downsample_ratio": 3,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "vision_max_n_token": 384,
    "vision_min_pixels": 147456,
    "vision_max_wh_ratio": 8,
}


def metadata() -> List[Tuple[str, str, object]]:
    """Metadata contract consumed by the future DS4V vision loader."""
    return [
        ("general.architecture", "string", "deepseek4_vision"),
        ("general.name", "string", "DeepSeek-V4 vision projector"),
        ("general.type", "string", "mmproj"),
        ("general.alignment", "uint32", GGUF_ALIGNMENT),
        ("deepseek4.vision.schema_version", "uint32", 1),
        ("deepseek4.vision.block_count", "uint32", 32),
        ("deepseek4.vision.embedding_length", "uint32", 1024),
        ("deepseek4.vision.attention.head_count", "uint32", 16),
        ("deepseek4.vision.attention.head_dimension", "uint32", 64),
        ("deepseek4.vision.attention.rope_layout", "string", "2d-half-split-height-width"),
        ("deepseek4.vision.feed_forward_length", "uint32", 2816),
        ("deepseek4.vision.patch_size", "uint32", 14),
        ("deepseek4.vision.rope.freq_base", "float32", 10000.0),
        ("deepseek4.vision.downsample_ratio", "uint32", 3),
        ("deepseek4.vision.aligner_input_length", "uint32", 9216),
        ("deepseek4.vision.language_embedding_length", "uint32", 4096),
        ("deepseek4.vision.vocabulary_size", "uint32", 129280),
        ("deepseek4.vision.attention.layer_norm_rms_epsilon", "float32", 1e-6),
        ("deepseek4.vision.image.max_tokens", "uint32", 384),
        ("deepseek4.vision.image.min_pixels", "uint32", 147456),
        ("deepseek4.vision.image.max_aspect_ratio", "float32", 8.0),
        ("deepseek4.vision.image.normalization_mean", "float32_array", (0.5, 0.5, 0.5)),
        ("deepseek4.vision.image.normalization_std", "float32_array", (0.5, 0.5, 0.5)),
        ("deepseek4.vision.image.patch_layout", "string", "channel-major"),
        ("deepseek4.vision.image.layout", "string", "n"),
        ("deepseek4.vision.image.layout_version", "uint32", 1),
        ("deepseek4.vision.image.layout_recipe", "string", "row-pair-column-interleave"),
        ("deepseek4.vision.image.compression_alignment", "uint32", 4),
        ("deepseek4.vision.image.sentinel_types", "string", "start,pad,image,newline,end"),
        ("deepseek4.vision.image.sentinel_type_count", "uint32", 5),
        ("deepseek4.vision.aligner.padding", "string", "bottom-right"),
        ("deepseek4.vision.aligner.patch_layout", "string", "channel-first-unfold"),
        ("deepseek4.vision.aligner.activation", "string", "gelu-exact"),
    ]


def _unique_object(pairs: Iterable[Tuple[str, object]]) -> Dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExportError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _read_json(path: Path, limit: int, description: str) -> Mapping[str, object]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExportError("cannot stat %s %s: %s" % (description, path, exc)) from exc
    if size > limit:
        raise ExportError("%s exceeds %d-byte limit: %s" % (description, limit, path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportError("cannot read %s %s: %s" % (description, path, exc)) from exc
    if not isinstance(value, dict):
        raise ExportError("%s must contain a JSON object: %s" % (description, path))
    return value


def _source_child(root: Path, name: str, description: str) -> Path:
    if not name or name in (".", "..") or "/" in name or "\\" in name or Path(name).name != name:
        raise ExportError("unsafe %s path: %r" % (description, name))
    candidate = root / name
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ExportError("%s escapes or is missing from source directory: %r" % (description, name)) from exc
    if not resolved.is_file():
        raise ExportError("%s is not a regular file: %s" % (description, resolved))
    return resolved


def _validate_config(config: Mapping[str, object]) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config:
            raise ExportError("config.json is missing required key %s" % key)
        actual = config[key]
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool) or float(actual) != expected:
                raise ExportError("config %s must be %r, got %r" % (key, expected, actual))
        elif actual != expected or isinstance(actual, bool):
            raise ExportError("config %s must be %r, got %r" % (key, expected, actual))


def _parse_safetensors_header(path: Path) -> Tuple[int, Mapping[str, object], os.stat_result]:
    try:
        stat_before = path.stat()
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ExportError("truncated safetensors size header: %s" % path)
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise ExportError("invalid safetensors JSON header size %d: %s" % (header_size, path))
            if 8 + header_size > stat_before.st_size:
                raise ExportError("safetensors JSON header exceeds file bounds: %s" % path)
            raw_header = handle.read(header_size)
        stat_after = path.stat()
    except OSError as exc:
        raise ExportError("cannot read safetensors header %s: %s" % (path, exc)) from exc
    if (stat_before.st_dev, stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_dev, stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns):
        raise ExportError("safetensors file changed while reading its header: %s" % path)
    try:
        header = json.loads(raw_header, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExportError("invalid safetensors JSON header %s: %s" % (path, exc)) from exc
    if not isinstance(header, dict):
        raise ExportError("safetensors header must be an object: %s" % path)
    return header_size, header, stat_after


def _validate_tensor_entry(name: str, entry: object, data_bytes: int) -> Tuple[str, Tuple[int, ...], int, int]:
    if not isinstance(entry, dict):
        raise ExportError("tensor %s header entry must be an object" % name)
    if set(entry) != {"dtype", "shape", "data_offsets"}:
        raise ExportError("tensor %s header entry has unexpected fields" % name)
    dtype = entry["dtype"]
    shape = entry["shape"]
    offsets = entry["data_offsets"]
    if dtype != "BF16":
        raise ExportError("tensor %s must remain BF16, got %r" % (name, dtype))
    if not isinstance(shape, list) or not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise ExportError("tensor %s has an invalid shape %r" % (name, shape))
    if not isinstance(offsets, list) or len(offsets) != 2 or any(
            not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets):
        raise ExportError("tensor %s has invalid data offsets %r" % (name, offsets))
    start, end = offsets
    if start < 0 or end <= start or end > data_bytes:
        raise ExportError("tensor %s data offsets %r exceed safetensors bounds" % (name, offsets))
    elements = 1
    for dim in shape:
        elements *= dim
    expected_bytes = elements * 2
    if end - start != expected_bytes:
        raise ExportError("tensor %s has %d bytes; BF16 shape %r requires %d" % (
            name, end - start, shape, expected_bytes))
    return dtype, tuple(shape), start, end


def discover_tensors(source_dir: Path) -> List[SourceTensor]:
    try:
        root = source_dir.resolve(strict=True)
    except OSError as exc:
        raise ExportError("source directory does not exist: %s" % source_dir) from exc
    if not root.is_dir():
        raise ExportError("source is not a directory: %s" % root)

    config_path = _source_child(root, "config.json", "config")
    index_path = _source_child(root, "model.safetensors.index.json", "safetensors index")
    _validate_config(_read_json(config_path, MAX_CONFIG_BYTES, "config"))
    index = _read_json(index_path, MAX_INDEX_BYTES, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ExportError("safetensors index is missing object-valued weight_map")
    if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()):
        raise ExportError("safetensors weight_map keys and values must be strings")

    expected = expected_shapes()
    selected = {name: shard for name, shard in weight_map.items() if _is_projector_tensor(name)}
    missing = sorted(set(expected) - set(selected))
    extra = sorted(set(selected) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
        if extra:
            details.append("unexpected: " + ", ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))
        raise ExportError("incomplete projector inventory (%s)" % "; ".join(details))

    shard_paths = {}
    for shard_name in sorted(set(selected.values())):
        shard_paths[shard_name] = _source_child(root, shard_name, "safetensors shard")

    headers = {}
    for shard_name, path in shard_paths.items():
        header_size, header, file_stat = _parse_safetensors_header(path)
        headers[shard_name] = (header_size, header, file_stat)
        indexed_here = {name for name, mapped_shard in selected.items() if mapped_shard == shard_name}
        present_here = {name for name in header if name != "__metadata__" and _is_projector_tensor(name)}
        if present_here != indexed_here:
            missing_here = sorted(indexed_here - present_here)
            unindexed_here = sorted(present_here - indexed_here)
            raise ExportError("projector/index disagreement in %s (missing=%r, unindexed=%r)" % (
                shard_name, missing_here[:8], unindexed_here[:8]))

    tensors = []
    ranges_by_shard = {}
    for name in sorted(expected):
        shard_name = selected[name]
        path = shard_paths[shard_name]
        header_size, header, file_stat = headers[shard_name]
        if name not in header:
            raise ExportError("index tensor %s is absent from %s" % (name, shard_name))
        data_bytes = file_stat.st_size - 8 - header_size
        dtype, shape, start, end = _validate_tensor_entry(name, header[name], data_bytes)
        if shape != expected[name]:
            raise ExportError("tensor %s shape must be %r, got %r" % (name, expected[name], shape))
        ranges_by_shard.setdefault(shard_name, []).append((start, end, name))
        tensors.append(SourceTensor(
            name=name,
            path=path,
            shape=shape,
            dtype=dtype,
            file_offset=8 + header_size + start,
            nbytes=end - start,
            file_size=file_stat.st_size,
            device=file_stat.st_dev,
            inode=file_stat.st_ino,
            mtime_ns=file_stat.st_mtime_ns,
        ))

    for shard_name, ranges in ranges_by_shard.items():
        previous_end = -1
        previous_name = ""
        for start, end, name in sorted(ranges):
            if start < previous_end:
                raise ExportError("overlapping projector tensors in %s: %s and %s" % (
                    shard_name, previous_name, name))
            previous_end = end
            previous_name = name
    return tensors


def _pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_metadata_entry(key: str, kind: str, value: object) -> bytes:
    result = bytearray(_pack_string(key))
    if kind == "uint32":
        result += struct.pack("<II", GGUF_TYPE_UINT32, int(value))
    elif kind == "float32":
        result += struct.pack("<If", GGUF_TYPE_FLOAT32, float(value))
    elif kind == "string":
        result += struct.pack("<I", GGUF_TYPE_STRING)
        result += _pack_string(str(value))
    elif kind == "float32_array":
        values = tuple(float(item) for item in value)
        result += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32, len(values))
        result += struct.pack("<%df" % len(values), *values)
    else:
        raise AssertionError("unsupported metadata kind %s" % kind)
    return bytes(result)


def _align(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _build_header(tensors: Sequence[SourceTensor]) -> Tuple[bytes, List[OutputTensor]]:
    metadata_entries = metadata()
    output_tensors = []
    offset = 0
    tensor_info = bytearray()
    for tensor in tensors:
        output_tensors.append(OutputTensor(tensor, offset))
        tensor_info += _pack_string(tensor.name)
        tensor_info += struct.pack("<I", len(tensor.shape))
        for dim in reversed(tensor.shape):
            tensor_info += struct.pack("<Q", dim)
        tensor_info += struct.pack("<IQ", GGML_TYPE_BF16, offset)
        offset += _align(tensor.nbytes)

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata_entries))
    for key, kind, value in metadata_entries:
        header += _pack_metadata_entry(key, kind, value)
    header += tensor_info
    header += bytes(_align(len(header)) - len(header))
    return bytes(header), output_tensors


def _check_source_identity(handle, tensor: SourceTensor) -> None:
    current = os.fstat(handle.fileno())
    actual = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    expected = (tensor.device, tensor.inode, tensor.file_size, tensor.mtime_ns)
    if actual != expected:
        raise ExportError("source shard changed after validation: %s" % tensor.path)


def _copy_tensor(output, tensor: SourceTensor) -> None:
    try:
        with tensor.path.open("rb") as source:
            _check_source_identity(source, tensor)
            source.seek(tensor.file_offset)
            remaining = tensor.nbytes
            while remaining:
                chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
                if not chunk:
                    raise ExportError("source tensor became truncated: %s" % tensor.name)
                output.write(chunk)
                remaining -= len(chunk)
            _check_source_identity(source, tensor)
    except OSError as exc:
        raise ExportError("cannot copy tensor %s from %s: %s" % (tensor.name, tensor.path, exc)) from exc
    padding = _align(tensor.nbytes) - tensor.nbytes
    if padding:
        output.write(bytes(padding))


def _validate_output_path(output_path: Path) -> Path:
    if output_path.name in ("", ".", ".."):
        raise ExportError("output must name a GGUF file")
    try:
        parent = output_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ExportError("output directory does not exist: %s" % output_path.parent) from exc
    if not parent.is_dir():
        raise ExportError("output parent is not a directory: %s" % parent)
    output = parent / output_path.name
    if os.path.lexists(str(output)):
        raise ExportError("refusing to overwrite existing output: %s" % output)
    return output


def export_mmproj(source_dir: Path, output_path: Path) -> Tuple[int, int]:
    tensors = discover_tensors(source_dir)
    header, output_tensors = _build_header(tensors)
    output = _validate_output_path(output_path)

    temp_fd = -1
    temp_name = ""
    linked = False
    try:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=".%s." % output.name,
            suffix=".tmp",
            dir=str(output.parent),
        )
        with os.fdopen(temp_fd, "wb") as handle:
            temp_fd = -1
            handle.write(header)
            for output_tensor in output_tensors:
                _copy_tensor(handle, output_tensor.source)
            handle.flush()
            os.fsync(handle.fileno())

        # Hard-linking is an atomic no-overwrite publish on the same filesystem.
        # Readers can observe only the complete, fsynced file.
        os.link(temp_name, output)
        linked = True
        os.unlink(temp_name)
        temp_name = ""
        directory_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ExportError("refusing to overwrite existing output: %s" % output) from exc
    except OSError as exc:
        if linked:
            raise ExportError("output was published but final directory sync failed: %s" % exc) from exc
        raise ExportError("cannot write output %s: %s" % (output, exc)) from exc
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    return len(tensors), sum(tensor.nbytes for tensor in tensors)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly extract the DeepSeek-V4 vision projector into standalone GGUF v3.")
    parser.add_argument("source_dir", type=Path,
                        help="parent model directory containing config.json, index, and shards")
    parser.add_argument("output", type=Path, help="new output GGUF path; existing paths are refused")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        tensor_count, payload_bytes = export_mmproj(args.source_dir, args.output)
    except ExportError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("exported %d BF16 tensors (%d payload bytes) to %s" % (
        tensor_count, payload_bytes, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
