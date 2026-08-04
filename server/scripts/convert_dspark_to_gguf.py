#!/usr/bin/env python3
"""Convert a config-described DSpark/DFlash checkpoint to a Q8_0 GGUF.

The converter intentionally knows the DFlash/DSpark tensor contract, not a
specific target model.  Target geometry, capture layers, draft geometry, RoPE,
and auxiliary-head dimensions come from ``config.json`` and are checked against
the safetensors shapes before any output is committed.

Large matrices are encoded as Q8_0.  Norms and the tiny confidence head remain
F32.  Unknown tensors, malformed safetensors, inconsistent configs, hash
mismatches, and unexpected Q8 alignment fail closed.  Output and the optional
JSON report are written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

import gguf
from gguf.quants import dequantize, quantize

ARCH = "dflash-draft"
CONVERTER_VERSION = 1
Q8_BLOCK_SIZE = 32
MAX_HEADER_BYTES = 128 * 1024 * 1024
SUPPORTED_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}


class ConversionError(RuntimeError):
    """Raised when the source cannot safely satisfy the GGUF contract."""


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def n_bytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ModelSpec:
    hidden: int
    draft_layers: int
    target_layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    context_length: int
    rms_eps: float
    rope_theta: float
    rope_type: str
    rope_factor: float
    rope_original_context: int
    capture_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    markov_rank: int
    markov_type: str
    confidence_enabled: bool
    confidence_with_markov: bool
    confidence_dim: int
    sliding_window: int
    sliding_pattern: tuple[bool, ...]

    @property
    def capture_count(self) -> int:
        return len(self.capture_layer_ids)


@dataclass(frozen=True)
class ConversionOptions:
    model_dir: Path
    output: Path
    report: Path | None = None
    name: str | None = None
    source_repo: str | None = None
    source_revision: str | None = None
    target_repo: str | None = None
    expected_sha256: str | None = None
    max_relative_rmse: float = 0.01
    sample_elements: int = 1_000_000
    force: bool = False


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversionError(f"config.{key} must be a positive integer, got {value!r}")
    return value


def _optional_token_id(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversionError(f"config.{key} must be a non-negative integer, got {value!r}")
    return value


def load_model_spec(config: dict[str, Any]) -> ModelSpec:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(value, str) and "DSparkDraftModel" in value for value in architectures
    ):
        raise ConversionError("config.architectures must identify a DSparkDraftModel")
    if config.get("attention_bias", False):
        raise ConversionError("attention_bias=true is not supported by the DFlash GGUF contract")
    if config.get("hidden_act", "silu") != "silu":
        raise ConversionError("only the SiLU DFlash MLP contract is currently supported")

    hidden = _positive_int(config, "hidden_size")
    draft_layers = _positive_int(config, "num_hidden_layers")
    target_layers = _positive_int(config, "num_target_layers")
    heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    intermediate = _positive_int(config, "intermediate_size")
    vocab = _positive_int(config, "vocab_size")
    context_length = _positive_int(config, "max_position_embeddings")
    block_size = _positive_int(config, "block_size")
    markov_rank = _positive_int(config, "markov_rank")

    if heads % kv_heads:
        raise ConversionError(
            f"num_attention_heads={heads} is not divisible by num_key_value_heads={kv_heads}"
        )

    rms_eps = config.get("rms_norm_eps")
    if not isinstance(rms_eps, (int, float)) or isinstance(rms_eps, bool) or rms_eps <= 0:
        raise ConversionError(f"config.rms_norm_eps must be positive, got {rms_eps!r}")

    dflash = config.get("dflash_config")
    if not isinstance(dflash, dict):
        raise ConversionError("config.dflash_config must be an object")
    capture_ids = dflash.get("target_layer_ids")
    if (
        not isinstance(capture_ids, list)
        or not capture_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capture_ids)
    ):
        raise ConversionError("config.dflash_config.target_layer_ids must be a non-empty int array")
    capture_layer_ids = tuple(capture_ids)
    if tuple(sorted(set(capture_layer_ids))) != capture_layer_ids:
        raise ConversionError("target_layer_ids must be unique and strictly increasing")
    if capture_layer_ids[0] < 0 or capture_layer_ids[-1] >= target_layers:
        raise ConversionError(
            f"target_layer_ids must fall inside target block range [0, {target_layers})"
        )
    mask_token_id = dflash.get("mask_token_id")
    if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int):
        raise ConversionError("config.dflash_config.mask_token_id must be an integer")
    if not 0 <= mask_token_id < vocab:
        raise ConversionError(f"mask_token_id={mask_token_id} is outside vocab_size={vocab}")

    rope = config.get("rope_parameters") or {}
    if not isinstance(rope, dict):
        raise ConversionError("config.rope_parameters must be an object")
    rope_theta = rope.get("rope_theta", config.get("rope_theta", 10_000.0))
    rope_type = str(rope.get("rope_type", "none")).lower()
    rope_factor = rope.get("factor", 1.0)
    rope_original_context = rope.get("original_max_position_embeddings", context_length)
    if not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
        raise ConversionError(f"RoPE theta must be positive, got {rope_theta!r}")
    if not isinstance(rope_factor, (int, float)) or rope_factor <= 0:
        raise ConversionError(f"RoPE factor must be positive, got {rope_factor!r}")
    if (
        isinstance(rope_original_context, bool)
        or not isinstance(rope_original_context, int)
        or rope_original_context <= 0
    ):
        raise ConversionError("original_max_position_embeddings must be a positive integer")
    if rope_type not in {"none", "yarn"}:
        raise ConversionError(f"unsupported rope_type={rope_type!r}; supported: none, yarn")

    markov_type = str(config.get("markov_head_type", "vanilla")).lower()
    if markov_type != "vanilla":
        raise ConversionError(f"unsupported markov_head_type={markov_type!r}")
    confidence_enabled = bool(config.get("enable_confidence_head", False))
    confidence_with_markov = bool(config.get("confidence_head_with_markov", False))
    if confidence_with_markov and not confidence_enabled:
        raise ConversionError("confidence_head_with_markov requires enable_confidence_head")
    confidence_dim = hidden + (markov_rank if confidence_with_markov else 0)

    raw_layer_types = config.get("layer_types") or ["full_attention"] * draft_layers
    if not isinstance(raw_layer_types, list) or len(raw_layer_types) != draft_layers:
        raise ConversionError(f"config.layer_types must contain {draft_layers} entries")
    unknown_layer_types = set(raw_layer_types) - {"full_attention", "sliding_attention"}
    if unknown_layer_types:
        raise ConversionError(f"unsupported layer_types: {sorted(unknown_layer_types)}")
    sliding_pattern = tuple(value == "sliding_attention" for value in raw_layer_types)
    raw_sliding_window = config.get("sliding_window")
    if any(sliding_pattern):
        if (
            isinstance(raw_sliding_window, bool)
            or not isinstance(raw_sliding_window, int)
            or raw_sliding_window <= 0
        ):
            raise ConversionError("sliding_attention layers require a positive sliding_window")
        sliding_window = raw_sliding_window
    else:
        sliding_window = 0

    return ModelSpec(
        hidden=hidden,
        draft_layers=draft_layers,
        target_layers=target_layers,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        vocab=vocab,
        context_length=context_length,
        rms_eps=float(rms_eps),
        rope_theta=float(rope_theta),
        rope_type=rope_type,
        rope_factor=float(rope_factor),
        rope_original_context=rope_original_context,
        capture_layer_ids=capture_layer_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
        bos_token_id=_optional_token_id(config, "bos_token_id"),
        eos_token_id=_optional_token_id(config, "eos_token_id"),
        pad_token_id=_optional_token_id(config, "pad_token_id"),
        markov_rank=markov_rank,
        markov_type=markov_type,
        confidence_enabled=confidence_enabled,
        confidence_with_markov=confidence_with_markov,
        confidence_dim=confidence_dim,
        sliding_window=sliding_window,
        sliding_pattern=sliding_pattern,
    )


def load_safetensors_header(path: Path) -> tuple[int, dict[str, TensorEntry]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        encoded_size = handle.read(8)
        if len(encoded_size) != 8:
            raise ConversionError("safetensors file is too short to contain a header")
        header_size = struct.unpack("<Q", encoded_size)[0]
        if header_size == 0 or header_size > MAX_HEADER_BYTES:
            raise ConversionError(f"invalid safetensors header size: {header_size}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ConversionError("truncated safetensors header")
    try:
        decoded = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ConversionError("safetensors header must be a JSON object")

    data_size = file_size - 8 - header_size
    if data_size < 0:
        raise ConversionError("safetensors header extends beyond the file")
    entries: dict[str, TensorEntry] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, info in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ConversionError("invalid tensor entry in safetensors header")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if dtype not in SUPPORTED_DTYPES:
            raise ConversionError(f"{name}: unsupported source dtype {dtype!r}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise ConversionError(f"{name}: shape must contain positive integers")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise ConversionError(f"{name}: invalid data_offsets")
        start, end = offsets
        expected_bytes = math.prod(shape) * SUPPORTED_DTYPES[dtype]
        if start < 0 or end <= start or end > data_size or end - start != expected_bytes:
            raise ConversionError(
                f"{name}: invalid byte range [{start}, {end}) for {dtype} shape={shape}"
            )
        entry = TensorEntry(name, dtype, tuple(shape), start, end)
        entries[name] = entry
        intervals.append((start, end, name))

    if not entries:
        raise ConversionError("safetensors contains no tensors")
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != data_size:
        raise ConversionError("safetensors tensor data does not cover the payload exactly")
    for (_, previous_end, previous_name), (start, _, name) in zip(
        intervals, intervals[1:], strict=False
    ):
        if start != previous_end:
            relation = "overlaps" if start < previous_end else "leaves a gap after"
            raise ConversionError(f"{name} {relation} {previous_name} in the data payload")
    return header_size, entries


def map_tensor_name(name: str) -> str | None:
    singleton_map = {
        "fc.weight": "dflash.fc.weight",
        "hidden_norm.weight": "dflash.hidden_norm.weight",
        "norm.weight": "output_norm.weight",
        "markov_head.markov_w1.weight": "dflash.dspark.markov.w1",
        "markov_head.markov_w2.weight": "dflash.dspark.markov.w2",
        "dspark_markov_head.markov_w1.weight": "dflash.dspark.markov.w1",
        "dspark_markov_head.markov_w2.weight": "dflash.dspark.markov.w2",
        "mtp.2.markov_head.markov_w1.weight": "dflash.dspark.markov.w1",
        "mtp.2.markov_head.markov_w2.weight": "dflash.dspark.markov.w2",
        "confidence_head.proj.weight": "dflash.dspark.confidence.weight",
        "confidence_head.proj.bias": "dflash.dspark.confidence.bias",
        "dspark_confidence_head.weight": "dflash.dspark.confidence.weight",
        "dspark_confidence_head.bias": "dflash.dspark.confidence.bias",
        "mtp.2.confidence_head.proj.weight": "dflash.dspark.confidence.weight",
        "mtp.2.confidence_head.proj.bias": "dflash.dspark.confidence.bias",
    }
    if name in singleton_map:
        return singleton_map[name]
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
    if not match:
        return None
    layer = int(match.group(1))
    suffix_map = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    suffix = suffix_map.get(match.group(2))
    return f"blk.{layer}.{suffix}" if suffix else None


def expected_source_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (spec.hidden, spec.capture_count * spec.hidden),
        "hidden_norm.weight": (spec.hidden,),
        "norm.weight": (spec.hidden,),
        "markov_head.markov_w1.weight": (spec.vocab, spec.markov_rank),
        "markov_head.markov_w2.weight": (spec.vocab, spec.markov_rank),
    }
    if spec.confidence_enabled:
        shapes["confidence_head.proj.weight"] = (1, spec.confidence_dim)
        shapes["confidence_head.proj.bias"] = (1,)
    q_dim = spec.heads * spec.head_dim
    kv_dim = spec.kv_heads * spec.head_dim
    for layer in range(spec.draft_layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (spec.hidden,),
                prefix + "post_attention_layernorm.weight": (spec.hidden,),
                prefix + "self_attn.q_proj.weight": (q_dim, spec.hidden),
                prefix + "self_attn.k_proj.weight": (kv_dim, spec.hidden),
                prefix + "self_attn.v_proj.weight": (kv_dim, spec.hidden),
                prefix + "self_attn.o_proj.weight": (spec.hidden, q_dim),
                prefix + "self_attn.q_norm.weight": (spec.head_dim,),
                prefix + "self_attn.k_norm.weight": (spec.head_dim,),
                prefix + "mlp.gate_proj.weight": (spec.intermediate, spec.hidden),
                prefix + "mlp.up_proj.weight": (spec.intermediate, spec.hidden),
                prefix + "mlp.down_proj.weight": (spec.hidden, spec.intermediate),
            }
        )
    return shapes


def validate_tensor_contract(entries: dict[str, TensorEntry], spec: ModelSpec) -> None:
    expected = expected_source_shapes(spec)
    # Accept historical aliases only after mapping them to the canonical contract.
    actual_by_gguf: dict[str, TensorEntry] = {}
    for source_name, entry in entries.items():
        gguf_name = map_tensor_name(source_name)
        if gguf_name is None:
            raise ConversionError(f"unmapped source tensor: {source_name}")
        if gguf_name in actual_by_gguf:
            raise ConversionError(f"multiple source tensors map to {gguf_name}")
        actual_by_gguf[gguf_name] = entry

    expected_by_gguf = {map_tensor_name(name): (name, shape) for name, shape in expected.items()}
    missing = sorted(set(expected_by_gguf) - set(actual_by_gguf))
    extra = sorted(set(actual_by_gguf) - set(expected_by_gguf))
    if missing or extra:
        raise ConversionError(f"tensor contract mismatch: missing={missing}, extra={extra}")
    for gguf_name, entry in actual_by_gguf.items():
        _, expected_shape = expected_by_gguf[gguf_name]
        if entry.shape != expected_shape:
            raise ConversionError(
                f"{entry.name}: shape {entry.shape} does not match expected {expected_shape}"
            )


def _read_tensor(path: Path, header_size: int, entry: TensorEntry) -> np.ndarray:
    with path.open("rb") as handle:
        handle.seek(8 + header_size + entry.start)
        raw = handle.read(entry.n_bytes)
    if len(raw) != entry.n_bytes:
        raise ConversionError(f"short read for {entry.name}")
    if entry.dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").reshape(entry.shape)
        return (words.astype("<u4") << 16).view("<f4").reshape(entry.shape)
    if entry.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(entry.shape).astype("<f4")
    return np.frombuffer(raw, dtype="<f4").reshape(entry.shape).copy()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _quantization_kind(gguf_name: str, shape: tuple[int, ...]) -> str:
    if len(shape) == 1 or gguf_name.startswith("dflash.dspark.confidence."):
        return "F32"
    if shape[-1] % Q8_BLOCK_SIZE:
        raise ConversionError(f"{gguf_name}: last dimension {shape[-1]} is not Q8_0 block-aligned")
    return "Q8_0"


def _sort_tensor(item: tuple[str, TensorEntry]) -> tuple[int, int, str]:
    gguf_name, _ = item
    if gguf_name.startswith("dflash."):
        return (0, 0, gguf_name)
    if gguf_name == "output_norm.weight":
        return (1, 0, gguf_name)
    match = re.match(r"blk\.(\d+)\.", gguf_name)
    return (2, int(match.group(1)) if match else 0, gguf_name)


def _sample_error(reference: np.ndarray, encoded: np.ndarray, limit: int) -> tuple[float, int]:
    restored = dequantize(encoded, gguf.GGMLQuantizationType.Q8_0).reshape(reference.shape)
    stride = max(1, math.ceil(reference.size / limit))
    ref = reference.reshape(-1)[::stride][:limit].astype(np.float64)
    got = restored.reshape(-1)[::stride][:limit].astype(np.float64)
    error_energy = float(np.dot(got - ref, got - ref))
    reference_energy = float(np.dot(ref, ref))
    relative_rmse = math.sqrt(error_energy / max(reference_energy, np.finfo(np.float64).tiny))
    return relative_rmse, ref.size


def _add_metadata(
    writer: gguf.GGUFWriter,
    spec: ModelSpec,
    config: dict[str, Any],
    options: ConversionOptions,
    source_hash: str,
) -> None:
    name = options.name or f"{options.model_dir.name}-Q8_0"
    writer.add_name(name)
    writer.add_type("model")
    writer.add_description("Q8_0 DSpark/DFlash speculative drafter")
    writer.add_quantized_by("Lucebox")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if options.source_repo:
        writer.add_source_repo_url(f"https://huggingface.co/{options.source_repo}")
        writer.add_string("general.source.repository", options.source_repo)
    if options.source_revision:
        writer.add_string("general.source.revision", options.source_revision)
    writer.add_string("general.source.file", "model.safetensors")
    writer.add_string("general.source.sha256", source_hash)

    writer.add_uint32(f"{ARCH}.context_length", spec.context_length)
    writer.add_uint32(f"{ARCH}.embedding_length", spec.hidden)
    writer.add_uint32(f"{ARCH}.block_count", spec.draft_layers)
    writer.add_uint32(f"{ARCH}.feed_forward_length", spec.intermediate)
    writer.add_uint32(f"{ARCH}.attention.head_count", spec.heads)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", spec.kv_heads)
    writer.add_uint32(f"{ARCH}.attention.key_length", spec.head_dim)
    writer.add_uint32(f"{ARCH}.attention.value_length", spec.head_dim)
    writer.add_uint32(f"{ARCH}.vocab_size", spec.vocab)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", spec.rms_eps)
    writer.add_uint32(f"{ARCH}.rope.dimension_count", spec.head_dim)
    writer.add_float32(f"{ARCH}.rope.freq_base", spec.rope_theta)
    if spec.rope_type == "yarn":
        writer.add_string(f"{ARCH}.rope.scaling.type", "yarn")
        writer.add_float32(f"{ARCH}.rope.scaling.factor", spec.rope_factor)
        writer.add_uint32(
            f"{ARCH}.rope.scaling.original_context_length", spec.rope_original_context
        )
    if spec.sliding_window:
        writer.add_uint32(f"{ARCH}.attention.sliding_window", spec.sliding_window)
        writer.add_array(f"{ARCH}.attention.sliding_window_pattern", spec.sliding_pattern)

    # n_target_layers is the number of captured features consumed by fc, not
    # the target network's block count.  They happen to be equal for some old
    # DFlash checkpoints but are 5 and 93 respectively for Kimi K3.
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", spec.capture_count)
    writer.add_uint32(f"{ARCH}.dflash.n_target_features", spec.capture_count * spec.hidden)
    writer.add_uint32(f"{ARCH}.dflash.target.block_count", spec.target_layers)
    writer.add_uint32(f"{ARCH}.dflash.block_size", spec.block_size)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id", spec.mask_token_id)
    writer.add_array(f"{ARCH}.dflash.target_layer_ids", spec.capture_layer_ids)
    if options.target_repo:
        writer.add_string(f"{ARCH}.dflash.target.repository", options.target_repo)

    writer.add_uint32(f"{ARCH}.dflash.dspark.enabled", 1)
    writer.add_uint32(f"{ARCH}.dflash.dspark.markov_rank", spec.markov_rank)
    writer.add_uint32(f"{ARCH}.dflash.dspark.vocab_size", spec.vocab)
    writer.add_string(f"{ARCH}.dflash.dspark.markov_type", spec.markov_type)
    writer.add_bool(f"{ARCH}.dflash.dspark.confidence.enabled", spec.confidence_enabled)
    writer.add_bool(f"{ARCH}.dflash.dspark.confidence.with_markov", spec.confidence_with_markov)
    if spec.confidence_enabled:
        writer.add_uint32(f"{ARCH}.dflash.dspark.confidence_dim", spec.confidence_dim)

    if spec.bos_token_id is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", spec.bos_token_id)
    if spec.eos_token_id is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", spec.eos_token_id)
    if spec.pad_token_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", spec.pad_token_id)
    writer.add_string(f"{ARCH}.source.config_json", json.dumps(config, sort_keys=True))


def _write_json_atomic(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise ConversionError(f"report already exists: {path} (pass --force to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_model(options: ConversionOptions) -> dict[str, Any]:
    if options.sample_elements <= 0:
        raise ConversionError("sample_elements must be positive")
    if not 0 < options.max_relative_rmse < 1:
        raise ConversionError("max_relative_rmse must fall inside (0, 1)")
    if bool(options.source_repo) != bool(options.source_revision):
        raise ConversionError("source_repo and source_revision must be supplied together")
    if options.output.exists() and not options.force:
        raise ConversionError(
            f"output already exists: {options.output} (pass --force to replace it)"
        )
    if options.report and options.report.exists() and not options.force:
        raise ConversionError(
            f"report already exists: {options.report} (pass --force to replace it)"
        )

    config_path = options.model_dir / "config.json"
    source_path = options.model_dir / "model.safetensors"
    if not config_path.is_file() or not source_path.is_file():
        raise ConversionError(f"{options.model_dir} must contain config.json and model.safetensors")
    protected_paths = {source_path.resolve(), config_path.resolve()}
    if options.output.resolve() in protected_paths:
        raise ConversionError("output must not overwrite a source model file")
    if options.report and options.report.resolve() in protected_paths | {options.output.resolve()}:
        raise ConversionError("report must be distinct from the source and GGUF paths")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise ConversionError("config.json must contain an object")

    spec = load_model_spec(config)
    header_size, entries = load_safetensors_header(source_path)
    validate_tensor_contract(entries, spec)

    source_hash = _sha256(source_path)
    expected_hash = options.expected_sha256.lower() if options.expected_sha256 else None
    if expected_hash and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ConversionError("expected_sha256 must be 64 lowercase or uppercase hex characters")
    if expected_hash and source_hash != expected_hash:
        raise ConversionError(
            f"source SHA256 mismatch: expected {expected_hash}, measured {source_hash}"
        )

    mapped = [(map_tensor_name(name), entry) for name, entry in entries.items()]
    if any(name is None for name, _ in mapped):
        raise AssertionError("validate_tensor_contract allowed an unmapped tensor")
    ordered = sorted(((name, entry) for name, entry in mapped if name), key=_sort_tensor)

    options.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.output.with_name(f".{options.output.name}.partial.{os.getpid()}")
    writer: gguf.GGUFWriter | None = None
    quantized_bytes = 0
    quantized_params = 0
    q8_metrics: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {"Q8_0": 0, "F32": 0}
    type_bytes: dict[str, int] = {"Q8_0": 0, "F32": 0}
    try:
        writer = gguf.GGUFWriter(temporary, ARCH)
        _add_metadata(writer, spec, config, options, source_hash)
        for gguf_name, entry in ordered:
            values = _read_tensor(source_path, header_size, entry)
            kind = _quantization_kind(gguf_name, entry.shape)
            if kind == "Q8_0":
                encoded = quantize(values, gguf.GGMLQuantizationType.Q8_0)
                relative_rmse, samples = _sample_error(values, encoded, options.sample_elements)
                if not math.isfinite(relative_rmse) or relative_rmse > options.max_relative_rmse:
                    raise ConversionError(
                        f"{gguf_name}: sampled relative RMSE {relative_rmse:.6g} exceeds "
                        f"limit {options.max_relative_rmse:.6g}"
                    )
                writer.add_tensor(gguf_name, encoded, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                q8_metrics.append(
                    {
                        "name": gguf_name,
                        "relative_rmse": relative_rmse,
                        "sample_elements": samples,
                    }
                )
                quantized_params += entry.n_elements
            else:
                encoded = values.astype("<f4", copy=False)
                writer.add_tensor(gguf_name, encoded, raw_dtype=gguf.GGMLQuantizationType.F32)
            type_counts[kind] += 1
            type_bytes[kind] += encoded.nbytes
            quantized_bytes += encoded.nbytes
            print(
                f"[tensor] {gguf_name:48s} {entry.dtype:4s}->{kind:4s} "
                f"shape={entry.shape} bytes={encoded.nbytes:,}"
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, options.output)
    except Exception:
        if writer is not None:
            writer.close()
        raise
    finally:
        temporary.unlink(missing_ok=True)

    output_hash = _sha256(options.output)
    source_bytes = source_path.stat().st_size
    output_bytes = options.output.stat().st_size
    weighted_error = math.sqrt(
        sum(metric["relative_rmse"] ** 2 * metric["sample_elements"] for metric in q8_metrics)
        / max(1, sum(metric["sample_elements"] for metric in q8_metrics))
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "converter": {"name": Path(__file__).name, "version": CONVERTER_VERSION},
        "source": {
            "repository": options.source_repo,
            "revision": options.source_revision,
            "file": source_path.name,
            "sha256": source_hash,
            "bytes": source_bytes,
            "tensor_count": len(entries),
            "parameter_count": sum(entry.n_elements for entry in entries.values()),
        },
        "target": {
            "repository": options.target_repo,
            "block_count": spec.target_layers,
            "capture_layer_ids": list(spec.capture_layer_ids),
        },
        "output": {
            "file": options.output.name,
            "sha256": output_hash,
            "bytes": output_bytes,
            "architecture": ARCH,
            "file_type": "MOSTLY_Q8_0",
            "source_size_ratio": output_bytes / source_bytes,
        },
        "quantization": {
            "tensor_counts": type_counts,
            "payload_bytes": type_bytes,
            "payload_total_bytes": quantized_bytes,
            "q8_parameter_count": quantized_params,
            "sampled_relative_rmse_rms": weighted_error,
            "sampled_relative_rmse_max": max(
                (metric["relative_rmse"] for metric in q8_metrics), default=0.0
            ),
            "max_allowed_relative_rmse": options.max_relative_rmse,
            "tensors": q8_metrics,
        },
    }
    if options.report:
        _write_json_atomic(options.report, report, options.force)
    print(
        f"[done] {source_bytes / 1e9:.3f} GB -> {output_bytes / 1e9:.3f} GB "
        f"({output_bytes / source_bytes:.1%}), sampled relative RMSE={weighted_error:.6g}"
    )
    print(f"[sha256] source={source_hash}")
    print(f"[sha256] output={output_hash}")
    return report


def parse_args(argv: list[str] | None = None) -> ConversionOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--source-repo")
    parser.add_argument("--source-revision")
    parser.add_argument("--target-repo")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-relative-rmse", type=float, default=0.01)
    parser.add_argument("--sample-elements", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return ConversionOptions(
        model_dir=args.model_dir,
        output=args.output,
        report=args.report,
        name=args.name,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        target_repo=args.target_repo,
        expected_sha256=args.expected_sha256,
        max_relative_rmse=args.max_relative_rmse,
        sample_elements=args.sample_elements,
        force=args.force,
    )


def main() -> int:
    try:
        convert_model(parse_args())
    except (ConversionError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
