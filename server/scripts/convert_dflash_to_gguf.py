#!/usr/bin/env python3
"""
Convert the z-lab DFlash draft (safetensors, bf16) to a GGUF that
llama.cpp can load.

Uses the vendored `deps/llama.cpp/gguf-py` package — the same GGUF tooling lineage
used by llama.cpp — with no hand-rolled binary writer. The library
handles header layout, alignment, BF16 storage, and tensor info offsets
correctly.

DFlash draft is a 5-layer Qwen-style transformer with two extra
model-level singletons specific to the spec-decode block-diffusion
algorithm:
  - `fc.weight`           [hidden, 5*hidden]  — fuses 5 captured target
                                                 hidden states into the
                                                 draft's input
  - `hidden_norm.weight`  [hidden]            — RMSNorm applied right after
                                                 the fc projection

These are stored under the `dflash.` prefix so llama.cpp can fetch them
via a custom arch loader without colliding with any upstream tensor
name.

Usage:
  python convert_dflash_to_gguf.py \
    models/draft/model.safetensors \
    qwen3.5-27b-dflash-draft.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Keep the converter self-contained: use the vendored gguf-py copy instead of
# requiring a separately installed Python package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))
# Use the vendored GGUF Python package — adds bf16 / metadata / alignment
# correctness without any hand-rolled code.
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash draft architecture constants — DEFAULTS ONLY.
#
# These are the qwen35-27B draft's values; they are used as a fallback when
# the source model has no config.json. Any other draft (A3B, gemma, ...) has
# a different head/dim/layer config, so the real scalars are read from the
# source config.json + derived from the tensor shapes in load_arch(). A
# converter that hardcoded these silently produced GGUFs with correct
# weights but 27B metadata, which the strict draft loader then rejected.
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
HIDDEN              = 5120
N_LAYER             = 5
N_HEAD              = 32          # query heads
N_HEAD_KV           = 8
HEAD_DIM            = 128
INTERMEDIATE        = 17408
VOCAB               = 248320
N_TARGET_LAYERS     = 5            # fc projects N_TARGET_LAYERS*hidden -> hidden
ROPE_THETA          = 1_000_000.0
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768


def load_arch(safetensors: Path, header: dict) -> dict:
    """Resolve the draft's architecture scalars. config.json (next to the
    safetensors) is authoritative for the transformer hparams; the tensor
    shapes are authoritative for the rest, so the result always matches the
    weights even when config.json is partial or absent."""
    a = dict(hidden=HIDDEN, n_layer=N_LAYER, n_head=N_HEAD, n_head_kv=N_HEAD_KV,
             head_dim=HEAD_DIM, intermediate=INTERMEDIATE, vocab=VOCAB,
             n_target_layers=N_TARGET_LAYERS, rope_theta=ROPE_THETA,
             rms_eps=RMS_EPS, mask_token_id=MASK_TOKEN_ID, block_size=BLOCK_SIZE,
             ctx_len=CTX_LEN)

    cfg_path = safetensors.parent / "config.json"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        def pick(*keys):
            for k in keys:
                if k in c and c[k] is not None:
                    return c[k]
            return None
        for dst, val in (
            ("hidden",       pick("hidden_size")),
            ("n_layer",      pick("num_hidden_layers")),
            ("n_head",       pick("num_attention_heads")),
            ("n_head_kv",    pick("num_key_value_heads")),
            ("head_dim",     pick("head_dim")),
            ("intermediate", pick("intermediate_size")),
            ("vocab",        pick("vocab_size")),
            ("rope_theta",   pick("rope_theta")),
            ("rms_eps",      pick("rms_norm_eps")),
            ("n_target_layers", pick("n_target_layers", "num_target_layers")),
            ("mask_token_id",   pick("mask_token_id")),
            ("block_size",      pick("block_size", "draft_block_size")),
            ("ctx_len",         pick("max_position_embeddings")),
        ):
            if val is not None:
                a[dst] = val
        dfc = c.get("dflash_config") or {}
        _tli = (dfc.get("target_layer_ids") or c.get("target_layer_ids")
                or c.get("aux_hidden_state_layer_ids"))
        if _tli:
            a["capture_layer_ids"] = [int(x) for x in _tli]
        print(f"[info] read arch from {cfg_path}")
    else:
        print(f"[warn] no config.json next to safetensors; using 27B defaults")

    # Weights are ground truth — derive/verify from tensor shapes.
    def shape_of(st_name):
        e = header.get(st_name)
        return e["shape"] if e else None

    # hidden absent in config: k-proj is [n_head_kv*head_dim, hidden] -> ne[1].
    k0 = shape_of("layers.0.self_attn.k_proj.weight")
    if (not cfg_path.exists()) and k0:
        a["hidden"] = k0[1]
    # head_dim absent in config: derive from k-proj (n_head_kv * head_dim).
    if k0 and a["n_head_kv"]:
        derived_hd = k0[0] // a["n_head_kv"]
        if not cfg_path.exists() or "head_dim" not in json.loads(cfg_path.read_text() if cfg_path.exists() else "{}"):
            a["head_dim"] = derived_hd
    # intermediate: ffn gate/up is [intermediate, hidden] — ne[0].
    g0 = shape_of("layers.0.mlp.gate_proj.weight")
    if g0:
        a["intermediate"] = g0[0]
    # n_target_layers: fc.weight is [hidden, n_target*hidden]; ne[0] (the
    # larger dim) / hidden is the capture count the loader checks.
    fc = shape_of("fc.weight")
    if fc and a["hidden"]:
        a["n_target_layers"] = max(fc) // a["hidden"]
    # n_layer: count the actual blocks present.
    n_blocks = 1 + max((int(n.split(".")[1]) for n in header
                        if n.startswith("layers.") and n.split(".")[1].isdigit()),
                       default=a["n_layer"] - 1)
    a["n_layer"] = n_blocks

    # Consistency check against the k-proj weight.
    if k0:
        exp_kv = a["n_head_kv"] * a["head_dim"]
        if exp_kv != k0[0]:
            print(f"[error] config n_head_kv*head_dim={exp_kv} != "
                  f"k_proj.weight dim {k0[0]}; fix config.json", file=sys.stderr)
            sys.exit(1)
    print(f"[info] arch: hidden={a['hidden']} n_layer={a['n_layer']} "
          f"n_head={a['n_head']} n_head_kv={a['n_head_kv']} "
          f"head_dim={a['head_dim']} ff={a['intermediate']} vocab={a['vocab']} "
          f"n_target_layers={a['n_target_layers']}")
    return a


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
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


# ──────────────────────────────────────────────────────────────────────
# safetensors reader  —  header parse + raw byte slice
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


def bytes_to_np(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # Convert BF16 -> F16 on the host. Several ggml-cuda ops (mul,
        # binbcast) only accept F32 / F16 inputs, and llama.cpp's
        # build_norm path multiplies normalised activations by the norm
        # weight tensor. Storing the draft as F16 throughout sidesteps
        # the unsupported BF16 path entirely. Quality impact ~0 for
        # weight tensors (BF16 -> F16 keeps 10/8 mantissa bits anyway
        # after the implicit cast).
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        # bf16 = sign(1) + exp(8) + mantissa(7); reinterpret as f32 by
        # putting it in the high half, then narrow to f16.
        u32 = (u16.astype(np.uint32) << 16)
        f32 = u32.view("<f4").reshape(shape)
        return f32.astype("<f2")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


SAFETENSORS_DTYPE_TO_GGUF = {
    "F32":  gguf.GGMLQuantizationType.F32,
    "F16":  gguf.GGMLQuantizationType.F16,
    # BF16 in safetensors -> we narrow to F16 in bytes_to_np above.
    "BF16": gguf.GGMLQuantizationType.F16,
}


DOMINO_TENSOR_MAP = {
    "domino_start":               ("dflash.domino.start",         gguf.GGMLQuantizationType.F32),
    "domino_gru.weight_ih_l0":    ("dflash.domino.gru.weight_ih", gguf.GGMLQuantizationType.F16),
    "domino_gru.weight_hh_l0":    ("dflash.domino.gru.weight_hh", gguf.GGMLQuantizationType.F16),
    "domino_gru.bias_ih_l0":      ("dflash.domino.gru.bias_ih",   gguf.GGMLQuantizationType.F32),
    "domino_gru.bias_hh_l0":      ("dflash.domino.gru.bias_hh",   gguf.GGMLQuantizationType.F32),
    "domino_head.W1.weight":      ("dflash.domino.head.w1",       gguf.GGMLQuantizationType.F16),
    "domino_head.W1.bias":        ("dflash.domino.head.b1",       gguf.GGMLQuantizationType.F32),
    "domino_head.W2.weight":      ("dflash.domino.head.w2",       gguf.GGMLQuantizationType.F16),
    "domino_head.W2.bias":        ("dflash.domino.head.b2",       gguf.GGMLQuantizationType.F32),
}


DSPARK_TENSOR_MAP = {
    ("dspark_markov_head.markov_w1.weight",
     "mtp.2.markov_head.markov_w1.weight"): ("dflash.dspark.markov.w1", gguf.GGMLQuantizationType.F16),
    ("dspark_markov_head.markov_w2.weight",
     "mtp.2.markov_head.markov_w2.weight"): ("dflash.dspark.markov.w2", gguf.GGMLQuantizationType.F16),
}

DSPARK_CONFIDENCE_TENSOR_MAP = {
    ("dspark_confidence_head.weight",
     "mtp.2.confidence_head.proj.weight"): ("dflash.dspark.confidence.weight", gguf.GGMLQuantizationType.F16),
    ("dspark_confidence_head.bias",
     "mtp.2.confidence_head.proj.bias"): ("dflash.dspark.confidence.bias", gguf.GGMLQuantizationType.F32),
}


def load_aux_state(aux_path: Path, label: str, wanted_names: set[str] | None = None):
    if aux_path.suffix == ".safetensors":
        header_size, header = load_safetensors_header(aux_path)
        state = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if wanted_names is not None and name not in wanted_names:
                continue
            raw = read_tensor_bytes(aux_path, header_size, info)
            state[name] = bytes_to_np(raw, info["dtype"], info["shape"])
        return state

    try:
        import torch
    except ImportError as exc:
        print(f"[error] --aux-heads requires torch for {label} .pt files: {exc}", file=sys.stderr)
        sys.exit(1)

    state = torch.load(aux_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        print(f"[error] {label} aux heads file is not a tensor dict: {aux_path}", file=sys.stderr)
        sys.exit(1)
    return state


def tensor_to_np(t, raw_dtype):
    if hasattr(t, "detach"):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.asarray(t)
        if arr.dtype.kind not in ("f", "i", "u"):
            arr = arr.astype(np.float32)
    if raw_dtype == gguf.GGMLQuantizationType.F16:
        return arr.astype("<f2")
    return arr.astype("<f4")


def find_state_tensor(state: dict, names: tuple[str, ...]):
    for name in names:
        if name in state:
            return name, state[name]
    return None, None


def flat_alias_names(alias_map: dict[tuple[str, ...], tuple[str, object]]) -> set[str]:
    out = set()
    for names in alias_map:
        out.update(names)
    return out


def add_domino_aux_heads(writer, arch: str, aux_path: Path | None):
    if aux_path is None:
        return
    if not aux_path.exists():
        print(f"[warn] Domino aux heads not found: {aux_path}")
        return

    print(f"[info] reading Domino aux heads from {aux_path}")
    state = load_aux_state(aux_path, "Domino", set(DOMINO_TENSOR_MAP))

    if not any(k in state for k in DOMINO_TENSOR_MAP):
        return

    missing = [k for k in DOMINO_TENSOR_MAP if k not in state]
    if missing:
        print(f"[warn] incomplete Domino aux heads; missing {missing}; skipping Domino tensors")
        return

    w_hh = state["domino_gru.weight_hh_l0"]
    w1 = state["domino_head.W1.weight"]
    w2 = state["domino_head.W2.weight"]
    gru_hidden_dim = int(w_hh.shape[1])
    emb_dim = int(w1.shape[0])
    vocab = int(w2.shape[0])
    writer.add_uint32(f"{arch}.dflash.domino.enabled", 1)
    writer.add_uint32(f"{arch}.dflash.domino.gru_hidden_dim", gru_hidden_dim)
    writer.add_uint32(f"{arch}.dflash.domino.emb_dim", emb_dim)
    writer.add_uint32(f"{arch}.dflash.domino.vocab_size", vocab)

    for st_name, (gguf_name, raw_dtype) in DOMINO_TENSOR_MAP.items():
        arr = tensor_to_np(state[st_name], raw_dtype)
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} aux ->{raw_dtype.name:4s} {tuple(arr.shape)}")


def add_dspark_aux_heads(writer, arch: str, aux_path: Path | None):
    if aux_path is None:
        return
    if not aux_path.exists():
        return

    wanted_names = flat_alias_names(DSPARK_TENSOR_MAP) | flat_alias_names(DSPARK_CONFIDENCE_TENSOR_MAP)
    state = load_aux_state(aux_path, "DSpark", wanted_names)
    resolved = {}
    missing = []
    for names, spec in DSPARK_TENSOR_MAP.items():
        found_name, tensor = find_state_tensor(state, names)
        if tensor is None:
            missing.append("/".join(names))
            continue
        resolved[names] = (found_name, tensor, spec)
    if missing:
        return

    print(f"[info] reading DSpark aux heads from {aux_path}")
    w1 = resolved[("dspark_markov_head.markov_w1.weight", "mtp.2.markov_head.markov_w1.weight")][1]
    w2 = resolved[("dspark_markov_head.markov_w2.weight", "mtp.2.markov_head.markov_w2.weight")][1]
    vocab = int(w1.shape[0])
    rank = int(w1.shape[1])
    if tuple(w2.shape) != (vocab, rank):
        print(f"[error] DSpark markov_w2 shape {tuple(w2.shape)} != {(vocab, rank)}", file=sys.stderr)
        sys.exit(1)

    writer.add_uint32(f"{arch}.dflash.dspark.enabled", 1)
    writer.add_uint32(f"{arch}.dflash.dspark.markov_rank", rank)
    writer.add_uint32(f"{arch}.dflash.dspark.vocab_size", vocab)

    for _names, (st_name, t, (gguf_name, raw_dtype)) in resolved.items():
        arr = tensor_to_np(t, raw_dtype)
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} aux:{st_name} ->{raw_dtype.name:4s} {tuple(arr.shape)}")

    conf_resolved = {}
    conf_missing = []
    for names, spec in DSPARK_CONFIDENCE_TENSOR_MAP.items():
        found_name, tensor = find_state_tensor(state, names)
        if tensor is None:
            conf_missing.append(names)
            continue
        conf_resolved[names] = (found_name, tensor, spec)
    weight_names = ("dspark_confidence_head.weight", "mtp.2.confidence_head.proj.weight")
    bias_names = ("dspark_confidence_head.bias", "mtp.2.confidence_head.proj.bias")
    if weight_names not in conf_resolved:
        if conf_missing:
            print("[warn] incomplete DSpark confidence head; Markov head will still load")
        return
    if bias_names not in conf_resolved:
        conf_resolved[bias_names] = (
            "<zero-bias>",
            np.zeros((1,), dtype=np.float32),
            DSPARK_CONFIDENCE_TENSOR_MAP[bias_names],
        )
        print("[warn] DSpark confidence head has no bias tensor; writing a zero bias")

    conf_w = conf_resolved[weight_names][1]
    conf_b = conf_resolved[bias_names][1]
    confidence_dim = int(conf_w.shape[1])
    if int(conf_w.shape[0]) != 1 or tuple(conf_b.shape) != (1,):
        print(
            f"[error] DSpark confidence shapes weight={tuple(conf_w.shape)} bias={tuple(conf_b.shape)}",
            file=sys.stderr,
        )
        sys.exit(1)
    writer.add_uint32(f"{arch}.dflash.dspark.confidence_dim", confidence_dim)
    writer.add_uint32(f"{arch}.dflash.dspark.confidence.enabled", 1)
    for _names, (st_name, t, (gguf_name, raw_dtype)) in conf_resolved.items():
        arr = tensor_to_np(t, raw_dtype)
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} aux:{st_name} ->{raw_dtype.name:4s} {tuple(arr.shape)}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("out_gguf",     type=Path)
    ap.add_argument("--aux-heads", type=Path, default=None,
                    help="optional Domino/DSpark aux-head .pt or DS4 MTP .safetensors file; defaults to dflash_aux_heads.pt next to the safetensors")
    ap.add_argument("--no-aux-heads", action="store_true",
                    help="do not auto-embed Domino/DSpark aux-head tensors")
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

    # Architecture metadata (resolved from config.json + tensor shapes)
    writer.add_string("general.name", f"DFlash-Draft-{a['hidden']}h-{a['n_layer']}L")
    writer.add_uint32(f"{ARCH}.context_length",          a["ctx_len"])
    writer.add_uint32(f"{ARCH}.embedding_length",        a["hidden"])
    writer.add_uint32(f"{ARCH}.block_count",             a["n_layer"])
    writer.add_uint32(f"{ARCH}.feed_forward_length",     a["intermediate"])
    writer.add_uint32(f"{ARCH}.attention.head_count",    a["n_head"])
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", a["n_head_kv"])
    # key_length / value_length override the n_embd/n_head heuristic, which
    # is wrong for DFlash drafts (n_head*head_dim != n_embd).
    writer.add_uint32(f"{ARCH}.attention.key_length",    a["head_dim"])
    writer.add_uint32(f"{ARCH}.attention.value_length",  a["head_dim"])
    writer.add_uint32(f"{ARCH}.vocab_size",              a["vocab"])
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", a["rms_eps"])
    writer.add_float32(f"{ARCH}.rope.freq_base",         a["rope_theta"])

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", a["n_target_layers"])
    writer.add_uint32(f"{ARCH}.dflash.block_size",      a["block_size"])
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   a["mask_token_id"])
    # Explicit captured target-layer ids (from the drafter's config.json). Travels
    # with the model so the server reads which target layers to capture instead of
    # hardcoding a per-arch set. Emitted only when it matches the fc-derived count.
    _cap_ids = a.get("capture_layer_ids")
    if _cap_ids and len(_cap_ids) == a["n_target_layers"]:
        writer.add_array(f"{ARCH}.dflash.target_layer_ids", [int(x) for x in _cap_ids])
    elif _cap_ids:
        print(f"[warn] capture_layer_ids len {len(_cap_ids)} != n_target_layers "
              f"{a['n_target_layers']}; not embedding ids", file=sys.stderr)

    # Walk + add tensors. Sort: dflash.* singletons first, then output_*,
    # then per-layer in numeric order — keeps the on-disk layout stable.
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        dtype = SAFETENSORS_DTYPE_TO_GGUF.get(info["dtype"])
        if dtype is None:
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, info["dtype"], info["shape"], info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):     return (0, n)
        if n.startswith("output_"):     return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    for gguf_name, st_dtype, shape, info in pending:
        raw = read_tensor_bytes(args.safetensors, header_size, info)
        arr = bytes_to_np(raw, st_dtype, shape)
        raw_dtype = SAFETENSORS_DTYPE_TO_GGUF[st_dtype]
        # Norm weights and the dflash hidden_norm singleton must be F32:
        # the ggml-cuda mul path that build_norm emits asserts on
        # src1's element size alignment (binbcast.cu nb10 % sizeof) and
        # the F32 path is the safest cross-quant fallback.
        is_norm = (
            gguf_name.endswith("_norm.weight") or
            gguf_name == "output_norm.weight" or
            gguf_name == "dflash.hidden_norm.weight"
        )
        if is_norm:
            arr = arr.astype("<f4")
            raw_dtype = gguf.GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} {st_dtype:4s}->{raw_dtype.name:4s} {tuple(shape)}")

    aux_path = None
    if not args.no_aux_heads:
        aux_path = args.aux_heads if args.aux_heads is not None else args.safetensors.parent / "dflash_aux_heads.pt"
    add_domino_aux_heads(writer, ARCH, aux_path)
    add_dspark_aux_heads(writer, ARCH, aux_path)

    print(f"[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] wrote {args.out_gguf}")


if __name__ == "__main__":
    main()
