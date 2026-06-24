#!/usr/bin/env python3
"""
Convert the z-lab DFlash draft (safetensors, bf16) to a Q4_K_M GGUF.

This is a corrected variant of convert_dflash_to_gguf.py with two fixes:

  FIX 1 — rope_theta resolution:
    The original pick("rope_theta") only searched the top-level config.json.
    Modern Qwen3.6 drafter repos publish theta nested under "rope_parameters":
        { "rope_parameters": { "rope_theta": 10000000 } }
    The original code silently fell through to the ROPE_THETA=1_000_000 default,
    baking the wrong frequency base into the GGUF → spec-decode acceptance
    collapse on 35B-a3b targets. This version checks rope_parameters first,
    then top-level rope_theta, then sys.exit(1) — no silent wrong default.

  FIX 2 — Q4_K_M output:
    Projection weights are quantized to Q4_K_M (mirrors the Q8_0 mechanism
    in quantize_draft_q8.py: gguf.quantize() + raw_dtype override).
    Norm weights and the dflash.hidden_norm singleton remain F32.
    # ponytail: gguf.quantize(Q4_K_M) is a pure-Python reference path —
    # llama-quantize would be faster for large models but is offline here.

Usage:
  PYTHONPATH=../../dflash_ggml/deps/llama.cpp/gguf-py python convert_modal_dflash_to_gguf.py \\
    models/draft/model.safetensors \\
    models/draft/draft-q4_k_m.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
# gguf is imported lazily inside main() so that resolve_rope_theta() and its
# tests remain importable without gguf installed (offline / pure-unit-test use).


# ──────────────────────────────────────────────────────────────────────
# Architecture defaults (27B drafter fallback when config.json absent).
# rope_theta intentionally absent — must come from config; see FIX 1.
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
HIDDEN              = 5120
N_LAYER             = 5
N_HEAD              = 32
N_HEAD_KV           = 8
HEAD_DIM            = 128
INTERMEDIATE        = 17408
VOCAB               = 248320
N_TARGET_LAYERS     = 5
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768

Q4_K_BLOCK_SIZE     = 256   # Q4_K_M: 256 elements per superblock


# ──────────────────────────────────────────────────────────────────────
# FIX 1: rope_theta resolution — pure helper, testable without IO.
# ──────────────────────────────────────────────────────────────────────

def resolve_rope_theta(config_dict: dict) -> float:
    """Resolve rope_theta from a parsed config.json dict.

    Priority:
      1. config["rope_parameters"]["rope_theta"]  (Qwen3.6 drafter layout)
      2. config["rope_theta"]                     (legacy top-level layout)
      3. sys.exit(1)  — no silent default

    Returns float. Raises SystemExit on missing.
    """
    c = config_dict
    rp = c.get("rope_parameters", {})
    # Guard: rope_parameters may be present but null or a non-dict scalar;
    # calling .get() on a non-dict would raise AttributeError.
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    if c.get("rope_theta") is not None:
        return float(c["rope_theta"])
    print(
        "[error] rope_theta not found in config.json. "
        "Checked: config['rope_parameters']['rope_theta'] and config['rope_theta']. "
        "Add one of these fields to config.json before converting.",
        file=sys.stderr,
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Norm tensor predicate — shared with quantize_draft_q8.py consumers.
# ──────────────────────────────────────────────────────────────────────

def is_norm_tensor(gguf_name: str) -> bool:
    return (
        gguf_name.endswith("_norm.weight")
        or gguf_name == "output_norm.weight"
        or gguf_name == "dflash.hidden_norm.weight"
    )


# ──────────────────────────────────────────────────────────────────────
# Architecture loader
# ──────────────────────────────────────────────────────────────────────

def load_arch(safetensors: Path, header: dict) -> dict:
    """Resolve arch scalars from config.json + tensor shapes.

    rope_theta is resolved via resolve_rope_theta() (FIX 1).
    All other fields follow the same pattern as the original converter.
    """
    a = dict(
        hidden=HIDDEN, n_layer=N_LAYER, n_head=N_HEAD, n_head_kv=N_HEAD_KV,
        head_dim=HEAD_DIM, intermediate=INTERMEDIATE, vocab=VOCAB,
        n_target_layers=N_TARGET_LAYERS,
        rms_eps=RMS_EPS, mask_token_id=MASK_TOKEN_ID, block_size=BLOCK_SIZE,
        ctx_len=CTX_LEN,
        sliding_window=0, sliding_window_pattern=[],
    )
    # rope_theta intentionally absent from defaults — will be set from config
    # or sys.exit(1) below.

    cfg_path = safetensors.parent / "config.json"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        dc = c.get("dflash_config", {})

        def pick(*keys):
            for k in keys:
                if k in c and c[k] is not None:
                    return c[k]
            return None

        def pick_dflash(*keys):
            for k in keys:
                if k in dc and dc[k] is not None:
                    return dc[k]
            for k in keys:
                if k in c and c[k] is not None:
                    return c[k]
            return None

        for dst, val in (
            ("hidden",          pick("hidden_size")),
            ("n_layer",         pick("num_hidden_layers")),
            ("n_head",          pick("num_attention_heads")),
            ("n_head_kv",       pick("num_key_value_heads")),
            ("head_dim",        pick("head_dim")),
            ("intermediate",    pick("intermediate_size")),
            ("vocab",           pick("vocab_size")),
            ("rms_eps",         pick("rms_norm_eps")),
            ("n_target_layers", pick_dflash("n_target_layers", "num_target_layers")),
            ("mask_token_id",   pick_dflash("mask_token_id")),
            ("block_size",      pick_dflash("block_size", "draft_block_size")),
            ("ctx_len",         pick("max_position_embeddings")),
        ):
            if val is not None:
                a[dst] = val

        # FIX 1: resolve rope_theta through scoped helper (loud on missing).
        a["rope_theta"] = resolve_rope_theta(c)

        _tli = (
            pick_dflash("target_layer_ids")
            or c.get("aux_hidden_state_layer_ids")
        )
        if _tli:
            a["capture_layer_ids"] = [int(x) for x in _tli]

        sw = pick("sliding_window")
        if sw is not None:
            a["sliding_window"] = sw
        layer_types = pick("layer_types")
        if layer_types:
            a["sliding_window_pattern"] = [lt == "sliding_attention" for lt in layer_types]

        print(f"[info] read arch from {cfg_path}")
    else:
        print(
            "[error] no config.json found next to safetensors — cannot resolve "
            "rope_theta. Provide a config.json with rope_theta or "
            "rope_parameters.rope_theta.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tensor-shape overrides (ground truth beats config for derived values).
    def shape_of(st_name):
        e = header.get(st_name)
        return e["shape"] if e else None

    k0 = shape_of("layers.0.self_attn.k_proj.weight")
    if k0 and a["n_head_kv"]:
        derived_hd = k0[0] // a["n_head_kv"]
        cfg_text = (cfg_path.read_text() if cfg_path.exists() else "{}")
        if "head_dim" not in json.loads(cfg_text):
            a["head_dim"] = derived_hd

    g0 = shape_of("layers.0.mlp.gate_proj.weight")
    if g0:
        a["intermediate"] = g0[0]

    fc = shape_of("fc.weight")
    if fc and a["hidden"]:
        a["n_target_layers"] = max(fc) // a["hidden"]

    n_blocks = 1 + max(
        (int(n.split(".")[1]) for n in header
         if n.startswith("layers.") and n.split(".")[1].isdigit()),
        default=a["n_layer"] - 1,
    )
    a["n_layer"] = n_blocks

    if k0:
        exp_kv = a["n_head_kv"] * a["head_dim"]
        if exp_kv != k0[0]:
            print(
                f"[error] config n_head_kv*head_dim={exp_kv} != "
                f"k_proj.weight dim {k0[0]}; fix config.json",
                file=sys.stderr,
            )
            sys.exit(1)

    swa_n = sum(1 for x in a.get("sliding_window_pattern", []) if x)
    print(
        f"[info] arch: hidden={a['hidden']} n_layer={a['n_layer']} "
        f"n_head={a['n_head']} n_head_kv={a['n_head_kv']} "
        f"head_dim={a['head_dim']} ff={a['intermediate']} vocab={a['vocab']} "
        f"n_target_layers={a['n_target_layers']} rope_theta={a['rope_theta']:.0f} "
        f"swa={swa_n}/{a['n_layer']} window={a.get('sliding_window', 0)}"
    )
    return a


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping — DFlash safetensors -> llama.cpp GGUF
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
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


# ──────────────────────────────────────────────────────────────────────
# safetensors I/O helpers
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
    u32 = u16.astype(np.uint32) << 16
    return u32.view("<f4").reshape(shape)


def to_f32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        return bf16_bytes_to_f32(raw, shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype("<f4")
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    raise ValueError(f"unsupported safetensors dtype {dtype}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    import gguf  # lazy import so pure helpers (resolve_rope_theta) work offline

    ap = argparse.ArgumentParser(
        description="Convert DFlash draft BF16 safetensors to Q4_K_M GGUF"
    )
    ap.add_argument("safetensors", type=Path,
                    help="Input BF16 safetensors (e.g. models/draft/model.safetensors)")
    ap.add_argument("out_gguf", type=Path,
                    help="Output Q4_K_M GGUF (e.g. models/draft/draft-q4_k_m.gguf)")
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    a = load_arch(args.safetensors, header)

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata — identical keys to convert_dflash_to_gguf.py.
    writer.add_string("general.name",
                      f"DFlash-Draft-{a['hidden']}h-{a['n_layer']}L-Q4_K_M")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{ARCH}.context_length",          a["ctx_len"])
    writer.add_uint32(f"{ARCH}.embedding_length",        a["hidden"])
    writer.add_uint32(f"{ARCH}.block_count",             a["n_layer"])
    writer.add_uint32(f"{ARCH}.feed_forward_length",     a["intermediate"])
    writer.add_uint32(f"{ARCH}.attention.head_count",    a["n_head"])
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", a["n_head_kv"])
    writer.add_uint32(f"{ARCH}.attention.key_length",    a["head_dim"])
    writer.add_uint32(f"{ARCH}.attention.value_length",  a["head_dim"])
    writer.add_uint32(f"{ARCH}.vocab_size",              a["vocab"])
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", a["rms_eps"])
    writer.add_float32(f"{ARCH}.rope.freq_base",         a["rope_theta"])

    # DFlash-specific hyperparameters.
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", a["n_target_layers"])
    writer.add_uint32(f"{ARCH}.dflash.block_size",      a["block_size"])
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   a["mask_token_id"])

    _cap_ids = a.get("capture_layer_ids")
    if _cap_ids and len(_cap_ids) == a["n_target_layers"]:
        writer.add_array(f"{ARCH}.dflash.target_layer_ids",
                         [int(x) for x in _cap_ids])
    elif _cap_ids:
        print(
            f"[warn] capture_layer_ids len {len(_cap_ids)} != n_target_layers "
            f"{a['n_target_layers']}; not embedding ids",
            file=sys.stderr,
        )

    if a.get("sliding_window"):
        writer.add_uint32(f"{ARCH}.attention.sliding_window", a["sliding_window"])
    if a.get("sliding_window_pattern"):
        writer.add_array(f"{ARCH}.attention.sliding_window_pattern",
                         a["sliding_window_pattern"])

    # Collect + sort tensors (same ordering as original converter).
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

    total_src  = 0
    total_out  = 0

    for gguf_name, st_name, info in pending:
        shape  = info["shape"]
        raw    = read_tensor_bytes(args.safetensors, header_size, info)
        arr    = to_f32(raw, info["dtype"], shape)
        total_src += len(raw)

        if is_norm_tensor(gguf_name):
            # Norm weights: keep F32 (ggml-cuda elementwise asserts on BF16/F16 src1).
            writer.add_tensor(gguf_name, arr,
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            total_out += arr.nbytes
            print(f"[tensor] {gguf_name:50s} {info['dtype']:4s}->F32   {tuple(shape)}")
        else:
            # FIX 2: projection weights → Q4_K_M.
            # Mirrors the Q8_0 mechanism in quantize_draft_q8.py:
            #   gguf.quantize(arr, type) + add_tensor(..., raw_dtype=type).
            # Q4_K_M requires the last dim to be a multiple of 256 (superblock).
            last_dim = shape[-1]
            if last_dim % Q4_K_BLOCK_SIZE != 0:
                print(
                    f"[error] {gguf_name}: last dim {last_dim} not divisible by "
                    f"{Q4_K_BLOCK_SIZE} (Q4_K_M superblock); cannot quantize",
                    file=sys.stderr,
                )
                sys.exit(1)
            q4_data = gguf.quantize(arr, gguf.GGMLQuantizationType.Q4_K)
            writer.add_tensor(gguf_name, q4_data,
                              raw_dtype=gguf.GGMLQuantizationType.Q4_K)
            total_out += q4_data.nbytes
            ratio = q4_data.nbytes / len(raw)
            print(
                f"[tensor] {gguf_name:50s} {info['dtype']:4s}->Q4_K_M {tuple(shape)}"
                f"  ({q4_data.nbytes:,} bytes, {ratio:.1%} of src)"
            )

    print(f"\n[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[done] wrote {args.out_gguf}")
    print(f"[size] source:      {total_src / 1e9:.2f} GB")
    print(f"[size] Q4_K_M out:  {total_out / 1e9:.2f} GB")
    if total_src:
        print(f"[size] compression: {total_out / total_src:.1%}")


if __name__ == "__main__":
    main()
