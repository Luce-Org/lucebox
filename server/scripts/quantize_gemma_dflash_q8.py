#!/usr/bin/env python3
"""
Quantize a z-lab Gemma-4 DFlash draft (safetensors, bf16) to Q8_0 GGUF.

Reads hyperparams from the model directory's config.json so it works for any
DFlash drafter (gemma-4-31B-it, gemma-4-26B-A4B-it, ...). Projection weights
become Q8_0; norm weights stay F32.

Output GGUF arch is `gemma4-dflash-draft` (distinct from `qwen35-dflash-draft`
used by Qwen3.5 drafter), so future dflash loader work can route on it.

Usage:
    python3 quantize_gemma_dflash_q8.py <model_dir> <out_gguf> [name]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

import gguf

ARCH = "gemma4-dflash-draft"
Q8_0_BLOCK_SIZE = 32


def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name == "embed_tokens.weight": return "token_embd.weight"
    if name == "lm_head.weight":      return "output.weight"
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


def is_norm_tensor(name: str) -> bool:
    return (name.endswith("_norm.weight") or
            name == "output_norm.weight" or
            name == "dflash.hidden_norm.weight")


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


def bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view("<f4").reshape(shape).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="Directory with config.json + model.safetensors")
    ap.add_argument("out_gguf",  type=Path)
    ap.add_argument("--name",    default=None, help="general.name for the GGUF")
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    sf  = args.model_dir / "model.safetensors"
    if not sf.exists():
        print(f"[error] {sf} missing", file=sys.stderr); sys.exit(1)

    HIDDEN        = int(cfg["hidden_size"])
    N_LAYER       = int(cfg["num_hidden_layers"])
    N_HEAD        = int(cfg["num_attention_heads"])
    N_HEAD_KV     = int(cfg["num_key_value_heads"])
    HEAD_DIM      = int(cfg["head_dim"])
    INTERMEDIATE  = int(cfg["intermediate_size"])
    VOCAB         = int(cfg["vocab_size"])
    ROPE_THETA    = float(cfg["rope_theta"])
    RMS_EPS       = float(cfg["rms_norm_eps"])
    CTX_LEN       = int(cfg.get("max_position_embeddings", 8192))
    SLIDING_WIN   = int(cfg.get("sliding_window", 0))
    SOFTCAP       = float(cfg.get("final_logit_softcapping", 0.0))
    N_TGT_LAYERS  = int(cfg.get("num_target_layers", 0))
    TGT_LAYER_IDS = list(cfg.get("dflash_config", {}).get("target_layer_ids", []))
    MASK_TOK      = int(cfg.get("dflash_config", {}).get("mask_token_id", 4))
    BLOCK_SIZE    = int(cfg.get("block_size", 16))
    TIE_EMBED     = bool(cfg.get("tie_word_embeddings", False))

    name = args.name or args.model_dir.name + "-Q8_0"
    print(f"[info] {name}: hidden={HIDDEN} n_layer={N_LAYER} n_head={N_HEAD}/{N_HEAD_KV} "
          f"head_dim={HEAD_DIM} ff={INTERMEDIATE} vocab={VOCAB} softcap={SOFTCAP} "
          f"n_tgt_layers={N_TGT_LAYERS} target_layer_ids={TGT_LAYER_IDS}")

    header_size, header = load_safetensors_header(sf)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info] {n_entries} tensors in safetensors")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)
    writer.add_string("general.name", name)
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
    if SLIDING_WIN:
        writer.add_uint32(f"{ARCH}.attention.sliding_window", SLIDING_WIN)
    if SOFTCAP > 0:
        writer.add_float32(f"{ARCH}.final_logit_softcapping", SOFTCAP)
    writer.add_bool(f"{ARCH}.tie_word_embeddings", TIE_EMBED)
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TGT_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOK)
    if TGT_LAYER_IDS:
        writer.add_array(f"{ARCH}.dflash.target_layer_ids", TGT_LAYER_IDS)

    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        if info["dtype"] not in ("BF16", "F16", "F32"):
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, st_name, info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):   return (0, n)
        if n in ("token_embd.weight", "output_norm.weight", "output.weight"):
            return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    total_bf16 = 0
    total_q8   = 0
    for gguf_name, st_name, info in pending:
        shape = info["shape"]
        raw = read_tensor_bytes(sf, header_size, info)
        if info["dtype"] == "BF16":
            arr = bf16_bytes_to_f32(raw, shape)
        elif info["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype="<f2").reshape(shape).astype("<f4")
        else:
            arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        total_bf16 += len(raw)
        if is_norm_tensor(gguf_name):
            writer.add_tensor(gguf_name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            total_q8 += arr.nbytes
            print(f"[tensor] {gguf_name:50s} BF16->F32  {tuple(shape)} ({arr.nbytes:,}B)")
        else:
            last_dim = shape[-1]
            if last_dim % Q8_0_BLOCK_SIZE != 0:
                # fall back to F16 for misaligned tensors
                arr16 = arr.astype("<f2")
                writer.add_tensor(gguf_name, arr16, raw_dtype=gguf.GGMLQuantizationType.F16)
                total_q8 += arr16.nbytes
                print(f"[tensor] {gguf_name:50s} BF16->F16  {tuple(shape)} ({arr16.nbytes:,}B, last_dim={last_dim} not %32)")
            else:
                q8 = gguf.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(gguf_name, q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                total_q8 += q8.nbytes
                print(f"[tensor] {gguf_name:50s} BF16->Q8_0 {tuple(shape)} ({q8.nbytes:,}B, {q8.nbytes/len(raw):.1%})")

    print(f"[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] BF16 src={total_bf16/1e9:.2f}GB  Q8_0 out={total_q8/1e9:.2f}GB  ratio={total_q8/total_bf16:.1%}")


if __name__ == "__main__":
    main()
