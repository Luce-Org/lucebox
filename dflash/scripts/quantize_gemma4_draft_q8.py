#!/usr/bin/env python3
"""
Quantize the z-lab DFlash Gemma4 draft (safetensors, bf16) to a Q8_0 GGUF.

Projection weights (fc, wq, wk, wv, wo, gate, up, down) are quantized
to Q8_0 (~50% size reduction vs BF16).  Norm weights stay F32
(precision-critical, tiny).

The output GGUF uses the same arch and tensor naming as
convert_dflash_to_gguf.py so gguf_draft_loader.cpp can load it.

Dimensions are auto-detected from config.json in the same directory as the
safetensors file.  Falls back to hardcoded 26B constants if no config.json
is present.

Usage:
    python3 scripts/quantize_gemma4_draft_q8.py \
        models/draft-gemma4-31b/model.safetensors \
        models/draft-gemma4-31b/draft-q8_0.gguf
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash Gemma4 draft architecture constants — 26B fallback defaults
# (used when no config.json is found alongside the safetensors file)
# ──────────────────────────────────────────────────────────────────────

ARCH                = "gemma4-dflash-draft"

_DEFAULTS = dict(
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

Q8_0_BLOCK_SIZE = 32   # elements per Q8_0 block


# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────

def load_config(safetensors_path: Path) -> dict:
    """
    Load dimensions from config.json next to the safetensors file.
    Returns a dict with the same keys as _DEFAULTS, merged over defaults.
    Falls back to _DEFAULTS entirely if config.json is absent.
    """
    cfg_path = safetensors_path.parent / "config.json"
    if not cfg_path.exists():
        print(f"[info] no config.json found at {cfg_path}, using 26B hardcoded defaults")
        return dict(_DEFAULTS)

    print(f"[info] reading config from {cfg_path}")
    with open(cfg_path) as f:
        raw = json.load(f)

    dflash_cfg = raw.get("dflash_config", {})
    target_layer_ids = dflash_cfg.get("target_layer_ids", _DEFAULTS["TARGET_LAYER_IDS"])

    # Derive model size tag from directory name (e.g. "draft-gemma4-31b" -> "31B")
    dir_name = safetensors_path.parent.name
    m = re.search(r"(\d+[bBmM])", dir_name)
    model_size_tag = m.group(1).upper() if m else _DEFAULTS["MODEL_SIZE_TAG"]

    cfg = dict(
        HIDDEN          = raw.get("hidden_size",            _DEFAULTS["HIDDEN"]),
        N_LAYER         = raw.get("num_hidden_layers",      _DEFAULTS["N_LAYER"]),
        N_HEAD          = raw.get("num_attention_heads",    _DEFAULTS["N_HEAD"]),
        N_HEAD_KV       = raw.get("num_key_value_heads",    _DEFAULTS["N_HEAD_KV"]),
        HEAD_DIM        = raw.get("head_dim",               _DEFAULTS["HEAD_DIM"]),
        INTERMEDIATE    = raw.get("intermediate_size",      _DEFAULTS["INTERMEDIATE"]),
        VOCAB           = raw.get("vocab_size",             _DEFAULTS["VOCAB"]),
        ROPE_THETA      = float(raw.get("rope_theta",       _DEFAULTS["ROPE_THETA"])),
        RMS_EPS         = float(raw.get("rms_norm_eps",     _DEFAULTS["RMS_EPS"])),
        MASK_TOKEN_ID   = dflash_cfg.get("mask_token_id",  _DEFAULTS["MASK_TOKEN_ID"]),
        BLOCK_SIZE      = raw.get("block_size",             _DEFAULTS["BLOCK_SIZE"]),
        CTX_LEN         = raw.get("max_position_embeddings", _DEFAULTS["CTX_LEN"]),
        LOGIT_SOFTCAP   = float(raw.get("final_logit_softcapping", _DEFAULTS["LOGIT_SOFTCAP"])),
        SLIDING_WINDOW  = raw.get("sliding_window",         _DEFAULTS["SLIDING_WINDOW"]),
        TARGET_LAYER_IDS = target_layer_ids,
        MODEL_SIZE_TAG  = model_size_tag,
    )

    print(f"[info] detected model size tag: {model_size_tag}")
    print(f"[info] hidden={cfg['HIDDEN']}  n_layers={cfg['N_LAYER']}  "
          f"n_head={cfg['N_HEAD']}  n_head_kv={cfg['N_HEAD_KV']}  "
          f"head_dim={cfg['HEAD_DIM']}")
    print(f"[info] intermediate={cfg['INTERMEDIATE']}  vocab={cfg['VOCAB']}")
    print(f"[info] target_layer_ids={cfg['TARGET_LAYER_IDS']}")
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# (Identical to convert_dflash_to_gguf.py)
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
        description="Quantize DFlash Gemma4 draft BF16 safetensors to Q8_0 GGUF")
    ap.add_argument("safetensors", type=Path,
                    help="Input BF16 safetensors (e.g. models/draft-gemma4-31b/model.safetensors)")
    ap.add_argument("out_gguf", type=Path,
                    help="Output Q8_0 GGUF (e.g. models/draft-gemma4-31b/draft-q8_0.gguf)")
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    # Load dimensions from config.json (or fall back to defaults)
    cfg = load_config(args.safetensors)
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
    LOGIT_SOFTCAP    = cfg["LOGIT_SOFTCAP"]
    SLIDING_WINDOW   = cfg["SLIDING_WINDOW"]
    TARGET_LAYER_IDS = cfg["TARGET_LAYER_IDS"]
    MODEL_SIZE_TAG   = cfg["MODEL_SIZE_TAG"]
    N_TARGET_LAYERS  = len(TARGET_LAYER_IDS)

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    # Compute TARGET_HIDDEN from fc.weight shape
    fc_info = header.get("fc.weight")
    if fc_info is None:
        print("[error] fc.weight not found in safetensors", file=sys.stderr)
        sys.exit(1)
    fc_shape = fc_info["shape"]          # [hidden, n_target_layers * target_hidden]
    if fc_shape[1] % N_TARGET_LAYERS != 0:
        print(f"[error] fc.weight columns ({fc_shape[1]}) not divisible by "
              f"N_TARGET_LAYERS ({N_TARGET_LAYERS})", file=sys.stderr)
        sys.exit(1)
    TARGET_HIDDEN = fc_shape[1] // N_TARGET_LAYERS
    print(f"[info] fc.weight shape {fc_shape}  ->  "
          f"N_TARGET_LAYERS={N_TARGET_LAYERS}  TARGET_HIDDEN={TARGET_HIDDEN}")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata (identical to convert_dflash_to_gguf.py)
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

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    # Gemma4-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.sliding_window",  SLIDING_WINDOW)
    writer.add_float32(f"{ARCH}.dflash.logit_softcap",  LOGIT_SOFTCAP)
    writer.add_uint32(f"{ARCH}.dflash.target_hidden",   TARGET_HIDDEN)
    writer.add_array(f"{ARCH}.dflash.target_layer_ids", TARGET_LAYER_IDS)

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
