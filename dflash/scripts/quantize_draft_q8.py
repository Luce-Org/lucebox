#!/usr/bin/env python3
"""
Quantize a z-lab DFlash draft (safetensors, bf16) to a Q8_0 GGUF.

Supports both Qwen and Gemma4 draft architectures via --arch.
When config.json is present alongside the safetensors file, dimensions are
auto-detected from it; hardcoded defaults are used as fallback.

Projection weights (fc, wq, wk, wv, wo, gate, up, down) are quantized
to Q8_0 (~50% size reduction vs BF16).  Norm weights stay F32
(precision-critical, tiny).

The output GGUF uses the same arch and tensor naming as
convert_dflash_to_gguf.py so gguf_draft_loader.cpp can load it.

Usage:
    # Qwen3.5 draft (auto-detects arch from config.json when present)
    python3 scripts/quantize_draft_q8.py --arch qwen \
        models/draft/model.safetensors \
        models/draft/draft-q8_0.gguf

    # Gemma4 draft
    python3 scripts/quantize_draft_q8.py --arch gemma4 \
        models/draft-gemma4-31b/model.safetensors \
        models/draft-gemma4-31b/draft-q8_0.gguf

    # Auto-detect arch from config.json (requires model_type field)
    python3 scripts/quantize_draft_q8.py \
        models/draft/model.safetensors \
        models/draft/draft-q8_0.gguf
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import gguf

Q8_0_BLOCK_SIZE = 32   # elements per Q8_0 block

# ──────────────────────────────────────────────────────────────────────
# Per-arch defaults  (used when config.json is absent or incomplete)
# ──────────────────────────────────────────────────────────────────────

_QWEN_DEFAULTS = dict(
    ARCH            = "qwen35-dflash-draft",
    HIDDEN          = 5120,
    N_LAYER         = 5,
    N_HEAD          = 32,
    N_HEAD_KV       = 8,
    HEAD_DIM        = 128,
    INTERMEDIATE    = 17408,
    VOCAB           = 248320,
    ROPE_THETA      = 1_000_000.0,
    RMS_EPS         = 1e-6,
    MASK_TOKEN_ID   = 248070,
    BLOCK_SIZE      = 16,
    CTX_LEN         = 32768,
    N_TARGET_LAYERS = 5,
    MODEL_SIZE_TAG  = "27B",
    # Qwen-specific (no sliding window or logit softcap)
    LOGIT_SOFTCAP   = None,
    SLIDING_WINDOW  = None,
    TARGET_LAYER_IDS = None,
)

_GEMMA4_DEFAULTS = dict(
    ARCH            = "gemma4-dflash-draft",
    HIDDEN          = 2816,
    N_LAYER         = 5,
    N_HEAD          = 32,
    N_HEAD_KV       = 8,
    HEAD_DIM        = 128,
    INTERMEDIATE    = 5632,
    VOCAB           = 262144,
    ROPE_THETA      = 1_000_000.0,
    RMS_EPS         = 1e-6,
    MASK_TOKEN_ID   = 4,
    BLOCK_SIZE      = 16,
    CTX_LEN         = 262144,
    LOGIT_SOFTCAP   = 30.0,
    SLIDING_WINDOW  = 2048,
    TARGET_LAYER_IDS = [1, 6, 11, 17, 22, 27],
    MODEL_SIZE_TAG  = "26B",
)

_ARCH_DEFAULTS = {
    "qwen":   _QWEN_DEFAULTS,
    "gemma4": _GEMMA4_DEFAULTS,
}

# config.json model_type -> arch key
_MODEL_TYPE_MAP = {
    "qwen3":  "qwen",
    "gemma4": "gemma4",
}


# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────

def detect_arch_from_config(cfg_path: Path) -> str | None:
    """Return 'qwen' or 'gemma4' by reading model_type from config.json."""
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        raw = json.load(f)
    model_type = raw.get("model_type", "").lower()
    for prefix, arch in _MODEL_TYPE_MAP.items():
        if model_type.startswith(prefix):
            return arch
    architectures = raw.get("architectures", [])
    for a in architectures:
        a_lower = a.lower()
        for prefix, arch in _MODEL_TYPE_MAP.items():
            if prefix in a_lower:
                return arch
    return None


def load_config(safetensors_path: Path, arch: str) -> dict:
    """
    Load dimensions from config.json next to the safetensors file.
    Returns a merged cfg dict, falling back to per-arch defaults for missing keys.
    """
    defaults = dict(_ARCH_DEFAULTS[arch])
    cfg_path = safetensors_path.parent / "config.json"

    if not cfg_path.exists():
        print(f"[info] no config.json found at {cfg_path}, using {arch} hardcoded defaults")
        return defaults

    print(f"[info] reading config from {cfg_path}")
    with open(cfg_path) as f:
        raw = json.load(f)

    dflash_cfg = raw.get("dflash_config", {})

    # Derive model size tag from directory name (e.g. "draft-gemma4-31b" -> "31B")
    dir_name = safetensors_path.parent.name
    m = re.search(r"(\d+[bBmM])", dir_name)
    model_size_tag = m.group(1).upper() if m else defaults["MODEL_SIZE_TAG"]

    cfg = dict(defaults)
    cfg.update(dict(
        HIDDEN          = raw.get("hidden_size",            defaults["HIDDEN"]),
        N_LAYER         = raw.get("num_hidden_layers",      defaults["N_LAYER"]),
        N_HEAD          = raw.get("num_attention_heads",    defaults["N_HEAD"]),
        N_HEAD_KV       = raw.get("num_key_value_heads",    defaults["N_HEAD_KV"]),
        HEAD_DIM        = raw.get("head_dim",               defaults["HEAD_DIM"]),
        INTERMEDIATE    = raw.get("intermediate_size",      defaults["INTERMEDIATE"]),
        VOCAB           = raw.get("vocab_size",             defaults["VOCAB"]),
        ROPE_THETA      = float(raw.get("rope_theta",       defaults["ROPE_THETA"])),
        RMS_EPS         = float(raw.get("rms_norm_eps",     defaults["RMS_EPS"])),
        MASK_TOKEN_ID   = dflash_cfg.get("mask_token_id",  defaults["MASK_TOKEN_ID"]),
        BLOCK_SIZE      = raw.get("block_size",             defaults["BLOCK_SIZE"]),
        CTX_LEN         = raw.get("max_position_embeddings", defaults["CTX_LEN"]),
        MODEL_SIZE_TAG  = model_size_tag,
    ))

    if arch == "gemma4":
        target_layer_ids = dflash_cfg.get("target_layer_ids", defaults["TARGET_LAYER_IDS"])
        cfg.update(dict(
            LOGIT_SOFTCAP   = float(raw.get("final_logit_softcapping", defaults["LOGIT_SOFTCAP"])),
            SLIDING_WINDOW  = raw.get("sliding_window",               defaults["SLIDING_WINDOW"]),
            TARGET_LAYER_IDS = target_layer_ids,
        ))

    print(f"[info] detected model size tag: {model_size_tag}")
    print(f"[info] hidden={cfg['HIDDEN']}  n_layers={cfg['N_LAYER']}  "
          f"n_head={cfg['N_HEAD']}  n_head_kv={cfg['N_HEAD_KV']}  "
          f"head_dim={cfg['HEAD_DIM']}")
    print(f"[info] intermediate={cfg['INTERMEDIATE']}  vocab={cfg['VOCAB']}")
    if arch == "gemma4":
        print(f"[info] target_layer_ids={cfg['TARGET_LAYER_IDS']}")
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# (Identical to convert_dflash_to_gguf.py; shared across both arches)
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1])
        rest = parts[2]
        layer_map = {
            "input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
            "self_attn.q_norm.weight":         f"blk.{i}.attn_q_norm.weight",
            "self_attn.k_norm.weight":         f"blk.{i}.attn_k_norm.weight",
            "mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
        }
        return layer_map.get(rest)
    return None


def is_norm_tensor(gguf_name: str) -> bool:
    return (
        gguf_name.endswith("_norm.weight") or
        gguf_name == "output_norm.weight" or
        gguf_name == "dflash.hidden_norm.weight"
    )


# ──────────────────────────────────────────────────────────────────────
# safetensors reader
# ──────────────────────────────────────────────────────────────────────

def load_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path: Path, header_size: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bf16_bytes_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view("<f4").reshape(shape)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Quantize DFlash draft BF16 safetensors to Q8_0 GGUF (qwen or gemma4)")
    ap.add_argument("safetensors", type=Path,
                    help="Input BF16 safetensors (e.g. models/draft/model.safetensors)")
    ap.add_argument("out_gguf", type=Path,
                    help="Output Q8_0 GGUF (e.g. models/draft/draft-q8_0.gguf)")
    ap.add_argument("--arch", choices=["qwen", "gemma4"],
                    help="Draft model architecture. Auto-detected from config.json "
                         "model_type when omitted.")
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    # Resolve arch: explicit flag > auto-detect from config.json
    arch = args.arch
    cfg_path = args.safetensors.parent / "config.json"
    if arch is None:
        arch = detect_arch_from_config(cfg_path)
        if arch is None:
            print(
                "[error] --arch not specified and could not auto-detect from "
                f"config.json (model_type not in {list(_MODEL_TYPE_MAP)}).\n"
                "        Pass --arch qwen or --arch gemma4 explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[info] auto-detected arch: {arch}")
    else:
        print(f"[info] arch: {arch}")

    cfg = load_config(args.safetensors, arch)
    ARCH             = cfg["ARCH"]
    HIDDEN           = cfg["HIDDEN"]
    N_LAYER          = cfg["N_LAYER"]
    N_HEAD           = cfg["N_HEAD"]
    N_HEAD_KV        = cfg["N_HEAD_KV"]
    HEAD_DIM         = cfg["HEAD_DIM"]
    INTERMEDIATE     = cfg["INTERMEDIATE"]
    VOCAB            = cfg["VOCAB"]
    ROPE_THETA       = cfg["ROPE_THETA"]
    RMS_EPS          = cfg["RMS_EPS"]
    MASK_TOKEN_ID    = cfg["MASK_TOKEN_ID"]
    BLOCK_SIZE       = cfg["BLOCK_SIZE"]
    CTX_LEN          = cfg["CTX_LEN"]
    MODEL_SIZE_TAG   = cfg["MODEL_SIZE_TAG"]

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    # Compute N_TARGET_LAYERS / TARGET_HIDDEN from fc.weight shape
    fc_info = header.get("fc.weight")
    if fc_info is None:
        print("[error] fc.weight not found in safetensors", file=sys.stderr)
        sys.exit(1)
    fc_shape = fc_info["shape"]   # [hidden, n_target_layers * target_hidden]

    if arch == "qwen":
        N_TARGET_LAYERS = cfg["N_TARGET_LAYERS"]
        if fc_shape[1] % N_TARGET_LAYERS != 0:
            print(f"[error] fc.weight columns ({fc_shape[1]}) not divisible by "
                  f"N_TARGET_LAYERS ({N_TARGET_LAYERS})", file=sys.stderr)
            sys.exit(1)
    else:  # gemma4
        TARGET_LAYER_IDS = cfg["TARGET_LAYER_IDS"]
        if not TARGET_LAYER_IDS:
            print("[error] target_layer_ids is empty; cannot compute N_TARGET_LAYERS "
                  "(check config.json or _DEFAULTS)", file=sys.stderr)
            sys.exit(1)
        N_TARGET_LAYERS = len(TARGET_LAYER_IDS)
        if fc_shape[1] % N_TARGET_LAYERS != 0:
            print(f"[error] fc.weight columns ({fc_shape[1]}) not divisible by "
                  f"N_TARGET_LAYERS ({N_TARGET_LAYERS})", file=sys.stderr)
            sys.exit(1)

    TARGET_HIDDEN = fc_shape[1] // N_TARGET_LAYERS
    print(f"[info] fc.weight shape {fc_shape}  ->  "
          f"N_TARGET_LAYERS={N_TARGET_LAYERS}  TARGET_HIDDEN={TARGET_HIDDEN}")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata (identical to convert_dflash_to_gguf.py)
    if arch == "qwen":
        model_name = f"Qwen3.5-{MODEL_SIZE_TAG}-DFlash-Draft-Q8_0"
    else:
        model_name = f"Gemma4-{MODEL_SIZE_TAG}-DFlash-Draft-Q8_0"
    writer.add_string("general.name", model_name)
    print(f"[info] general.name = {model_name}")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             N_LAYER)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    writer.add_uint32(f"{ARCH}.attention.key_length",    HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         ROPE_THETA)

    # DFlash-specific hyperparameters (shared)
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    # Gemma4-specific hyperparameters
    if arch == "gemma4":
        writer.add_uint32(f"{ARCH}.dflash.sliding_window",  cfg["SLIDING_WINDOW"])
        writer.add_float32(f"{ARCH}.dflash.logit_softcap",  cfg["LOGIT_SOFTCAP"])
        writer.add_uint32(f"{ARCH}.dflash.target_hidden",   TARGET_HIDDEN)
        writer.add_array(f"{ARCH}.dflash.target_layer_ids", cfg["TARGET_LAYER_IDS"])

    # Collect and sort tensors (same order as convert_dflash_to_gguf.py)
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        if info["dtype"] not in ("BF16", "F16", "F32"):
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}",
                  file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, st_name, info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):   return (0, n)
        if n.startswith("output_"):   return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    total_bf16 = 0
    total_q8   = 0

    for gguf_name, st_name, info in pending:
        shape = info["shape"]
        raw = read_tensor_bytes(args.safetensors, header_size, info)

        # Convert to F32 from whatever source dtype
        if info["dtype"] == "BF16":
            arr = bf16_bytes_to_f32(raw, shape)
        elif info["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype="<f2").reshape(shape).astype("<f4")
        else:
            arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()

        src_bytes = len(raw)
        total_bf16 += src_bytes

        if is_norm_tensor(gguf_name):
            # Norm weights: keep F32
            writer.add_tensor(gguf_name, arr,
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            total_q8 += arr.nbytes
            print(f"[tensor] {gguf_name:50s} BF16->F32  {tuple(shape)}"
                  f"  ({arr.nbytes:,} bytes)")
        else:
            # Projection weights: quantize to Q8_0
            # Verify alignment: last dim must be multiple of 32
            last_dim = shape[-1]
            assert last_dim % Q8_0_BLOCK_SIZE == 0, \
                f"{gguf_name}: last dim {last_dim} not divisible by {Q8_0_BLOCK_SIZE}"
            q8_data = gguf.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name, q8_data,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            total_q8 += q8_data.nbytes
            ratio = q8_data.nbytes / src_bytes
            print(f"[tensor] {gguf_name:50s} BF16->Q8_0 {tuple(shape)}"
                  f"  ({q8_data.nbytes:,} bytes, {ratio:.1%} of BF16)")

    print(f"\n[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[done] wrote {args.out_gguf}")
    print(f"[size] BF16 source: {total_bf16 / 1e9:.2f} GB")
    print(f"[size] Q8_0 output: {total_q8 / 1e9:.2f} GB")
    print(f"[size] compression: {total_q8 / total_bf16:.1%}")


if __name__ == "__main__":
    main()
