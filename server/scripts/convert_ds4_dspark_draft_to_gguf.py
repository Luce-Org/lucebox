#!/usr/bin/env python3
"""Convert the DSpark draft block of a DeepSeek V4.1 checkpoint to a GGUF.

The DeepSeek V4.1 Flash release stores its speculative draft block next to
the backbone, under the `mtp.<stage>.*` namespace: `num_nextn_predict_layers`
decoder stages with their own MoE (`dspark_n_routed_experts` experts, top
`dspark_num_experts_per_tok`), a feature projection on the first stage
(`main_proj`, `main_norm`) and the Markov / confidence heads on the last
stage. The token embedding and the lm head are tied to the backbone, so the
draft GGUF carries neither; the engine reads them from the target model.

Output: a `deepseek41-dflash-draft` GGUF that `load_deepseek4_dspark_drafter`
(server/src/deepseek4/deepseek4_dspark.cpp) loads. Only the three safetensors
shards that hold `mtp.*` are needed.

Precision (default): routed experts in MXFP4, a bit-exact repack of the
checkpoint's FP4 E2M1 values and per-32 E8M0 scales at the same 4.25 bits per
weight as the DS4 drafter's ROCmFP4 experts (re-encoding them to ROCmFP4
instead costs about 5% relative RMS error); dense projections in Q8_0 (the
checkpoint stores them as FP8); router, norms, hyper-connection and head
tensors in F32/F16.

Usage:
  python3 convert_ds4_dspark_draft_to_gguf.py HF_DIR OUT.gguf \\
      --ggml-lib server/build/deps/llama.cpp/ggml/src/libggml-base.so

HF_DIR must hold config.json, model.safetensors.index.json and the shards
the index lists for `mtp.*`.
"""
import argparse
import ctypes
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ARCH = "deepseek41-dflash-draft"

# ggml type ids (server/deps/llama.cpp/ggml/include/ggml.h)
T_F32, T_F16, T_Q8_0, T_MXFP4, T_ROCMFP4_FAST = 0, 1, 8, 39, 101
BLOCK = {T_F32: (1, 4), T_F16: (1, 2), T_Q8_0: (32, 34), T_MXFP4: (32, 17),
         T_ROCMFP4_FAST: (32, 17)}
TYPE_NAMES = {"q8_0": T_Q8_0, "mxfp4": T_MXFP4, "rocmfp4_fast": T_ROCMFP4_FAST, "f16": T_F16}

# ── safetensors ──────────────────────────────────────────────────────────


class Shards:
    def __init__(self, hf_dir: Path):
        self.dir = hf_dir
        self.index = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def _header(self, fn):
        if fn not in self.headers:
            with open(self.dir / fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[fn] = (8 + n, json.loads(f.read(n)))
        return self.headers[fn]

    def has(self, name):
        return name in self.index

    def raw(self, name):
        fn = self.index[name]
        base, h = self._header(fn)
        info = h[name]
        a, b = info["data_offsets"]
        with open(self.dir / fn, "rb") as f:
            f.seek(base + a)
            data = f.read(b - a)
        return info["dtype"], info["shape"], data


# FP8 E4M3 (finite, no inf) decode table.
def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for v in range(256):
        s = -1.0 if v & 0x80 else 1.0
        e = (v >> 3) & 0x0F
        m = v & 0x07
        if e == 0x0F and m == 0x07:
            t[v] = np.nan
        elif e == 0:
            t[v] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[v] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e8m0(raw: bytes, shape):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.int32)
    return np.ldexp(np.float32(1.0), b - 127).astype(np.float32)


def repack_mxfp4(st: Shards, name: str) -> bytes:
    """Packed FP4 E2M1 + E8M0 scales -> ggml block_mxfp4 rows, bit-exact.

    The checkpoint packs elements (2k, 2k+1) into byte k (low nibble first);
    a ggml MXFP4 block stores element j in the low nibble and j + 16 in the
    high nibble of byte j, after its E8M0 scale byte."""
    dtype, shape, data = st.raw(name)
    assert dtype == "I8", (name, dtype)
    _, sshape, sdata = st.raw(name[: -len("weight")] + "scale")
    q = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    out, half = shape
    nib = np.empty((out, half * 2), dtype=np.uint8)
    nib[:, 0::2] = q & 0x0F
    nib[:, 1::2] = q >> 4
    nb = half * 2 // 32
    assert sshape == [out, nb], (name, sshape)
    nib = nib.reshape(out, nb, 32)
    blocks = np.empty((out, nb, 17), dtype=np.uint8)
    blocks[:, :, 0] = np.frombuffer(sdata, dtype=np.uint8).reshape(out, nb)
    blocks[:, :, 1:] = nib[:, :, :16] | (nib[:, :, 16:] << 4)
    return blocks.tobytes()


def load_f32(st: Shards, name: str) -> np.ndarray:
    """Dequantize one checkpoint tensor to f32 in its torch [out, in] layout."""
    dtype, shape, data = st.raw(name)
    if dtype == "F32":
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    if dtype == "BF16":
        u = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    scale_name = name[: -len("weight")] + "scale"
    _, sshape, sdata = st.raw(scale_name)
    scale = e8m0(sdata, sshape)
    if dtype == "F8_E4M3":
        w = E4M3[np.frombuffer(data, dtype=np.uint8)].reshape(shape)
        out, inp = shape
        bo, bi = out // sshape[0], inp // sshape[1]
        w = w.reshape(sshape[0], bo, sshape[1], bi) * scale[:, None, :, None]
        return w.reshape(out, inp)
    if dtype == "I8":  # packed FP4 E2M1, low nibble first, one E8M0 scale per 32 inputs
        q = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        out, half = shape
        w = np.empty((out, half * 2), dtype=np.float32)
        w[:, 0::2] = E2M1[q & 0x0F]
        w[:, 1::2] = E2M1[q >> 4]
        blk = (half * 2) // sshape[1]
        return (w.reshape(out, sshape[1], blk) * scale[:, :, None]).reshape(out, half * 2)
    raise ValueError(f"{name}: unsupported dtype {dtype}")


# ── ggml quantization through libggml-base ──────────────────────────────


class Quantizer:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        # ggml_quantize_chunk does not dispatch the ROCmFP types; their
        # quantizers are exported directly.
        self.lib.rocmfp4_quantize_q4_0_fast.restype = ctypes.c_size_t
        self.lib.rocmfp4_quantize_q4_0_fast.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

    def __call__(self, x: np.ndarray, ttype: int) -> bytes:
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, cols = x.shape
        blk, bsz = BLOCK[ttype]
        assert cols % blk == 0, (x.shape, ttype)
        dst = np.empty(rows * cols // blk * bsz, dtype=np.uint8)
        if ttype == T_ROCMFP4_FAST:
            n = self.lib.rocmfp4_quantize_q4_0_fast(x.ctypes.data, dst.ctypes.data, rows, cols, None)
        else:
            n = self.lib.ggml_quantize_chunk(ttype, x.ctypes.data, dst.ctypes.data, 0, rows, cols, None)
        assert n == dst.size, (n, dst.size)
        return dst.tobytes()


def encode(q: Quantizer, x: np.ndarray, ttype: int) -> bytes:
    if ttype == T_F32:
        return np.ascontiguousarray(x, dtype=np.float32).tobytes()
    if ttype == T_F16:
        return np.ascontiguousarray(x, dtype=np.float16).tobytes()
    return q(x.reshape(-1, x.shape[-1]), ttype)


def nbytes(ne, ttype):
    blk, bsz = BLOCK[ttype]
    n = 1
    for d in ne:
        n *= d
    assert ne[0] % blk == 0
    return n // blk * bsz


# ── minimal GGUF v3 writer (streams tensor data) ─────────────────────────

GGUF_U32, GGUF_I32, GGUF_F32, GGUF_STR, GGUF_ARR = 4, 5, 6, 8, 9


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def w_kv(f, key, value):
    w_str(f, key)
    if isinstance(value, str):
        f.write(struct.pack("<I", GGUF_STR))
        w_str(f, value)
    elif isinstance(value, float):
        f.write(struct.pack("<If", GGUF_F32, value))
    elif isinstance(value, list):
        f.write(struct.pack("<IIQ", GGUF_ARR, GGUF_I32, len(value)))
        f.write(struct.pack(f"<{len(value)}i", *value))
    else:
        f.write(struct.pack("<II", GGUF_U32, int(value)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ggml-lib", required=True, help="path to libggml-base.so of an engine build")
    ap.add_argument("--expert-type", default="mxfp4", choices=["mxfp4", "rocmfp4_fast", "q8_0"])
    ap.add_argument("--dense-type", default="q8_0", choices=["q8_0", "rocmfp4_fast", "f16"])
    args = ap.parse_args()

    cfg = json.loads((args.hf_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    st = Shards(args.hf_dir)
    q = Quantizer(args.ggml_lib)
    t_exp, t_dense = TYPE_NAMES[args.expert_type], TYPE_NAMES[args.dense_type]

    n_stage = int(tc["num_nextn_predict_layers"])
    n_embd = int(tc["hidden_size"])
    n_ff = int(tc["moe_intermediate_size"])
    n_exp = int(tc["dspark_n_routed_experts"])
    n_used = int(tc["dspark_num_experts_per_tok"])
    n_head, head_dim = int(tc["num_attention_heads"]), int(tc["head_dim"])
    n_hc = int(tc["hc_mult"])
    targets = [int(x) for x in tc["dspark_target_layer_ids"]]
    rank = int(tc["dspark_markov_rank"])
    vocab = int(tc["vocab_size"])
    last = n_stage - 1

    # Each entry: (gguf name, checkpoint tensors, ggml type). Routed experts
    # stack into one [in, out, n_expert] tensor per projection.
    plan = []
    for s in range(n_stage):
        p, b = f"mtp.{s}.", f"blk.{s}."
        for gname, src, ttype in [
            ("attn_norm.weight", "attn_norm.weight", T_F32),
            ("attn_q_a.weight", "attn.wq_a.weight", t_dense),
            ("attn_q_a_norm.weight", "attn.q_norm.weight", T_F32),
            ("attn_q_b.weight", "attn.wq_b.weight", t_dense),
            ("attn_kv.weight", "attn.wkv.weight", t_dense),
            ("attn_kv_a_norm.weight", "attn.kv_norm.weight", T_F32),
            ("attn_sinks.weight", "attn.attn_sink", T_F32),
            ("attn_output_a.weight", "attn.wo_a.weight", t_dense),
            ("attn_output_b.weight", "attn.wo_b.weight", t_dense),
            ("hc_attn_fn.weight", "hc_attn_fn", T_F16),
            ("hc_attn_scale.weight", "hc_attn_scale", T_F32),
            ("hc_attn_base.weight", "hc_attn_base", T_F32),
            ("ffn_norm.weight", "ffn_norm.weight", T_F32),
            ("ffn_gate_inp.weight", "ffn.gate.weight", T_F32),
            ("exp_probs_b.bias", "ffn.gate.bias", T_F32),
            ("ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight", t_dense),
            ("ffn_up_shexp.weight", "ffn.shared_experts.w3.weight", t_dense),
            ("ffn_down_shexp.weight", "ffn.shared_experts.w2.weight", t_dense),
            ("hc_ffn_fn.weight", "hc_ffn_fn", T_F16),
            ("hc_ffn_scale.weight", "hc_ffn_scale", T_F32),
            ("hc_ffn_base.weight", "hc_ffn_base", T_F32),
        ]:
            plan.append((b + gname, [p + src], ttype))
        for gname, w in [("ffn_gate_exps.weight", "w1"), ("ffn_up_exps.weight", "w3"),
                         ("ffn_down_exps.weight", "w2")]:
            plan.append((b + gname, [f"{p}ffn.experts.{e}.{w}.weight" for e in range(n_exp)], t_exp))
    plan += [
        ("dflash.fc.weight", ["mtp.0.main_proj.weight"], t_dense),
        ("dflash.hidden_norm.weight", ["mtp.0.main_norm.weight"], T_F32),
        ("output_norm.weight", [f"mtp.{last}.norm.weight"], T_F32),
        ("dflash.dspark.markov.w1", [f"mtp.{last}.markov_head.embed.weight"], T_F16),
        ("dflash.dspark.markov.w2", [f"mtp.{last}.markov_head.head.weight"], T_F16),
        ("dflash.dspark.confidence.weight", [f"mtp.{last}.confidence_head.proj.weight"], T_F16),
    ]

    # Shapes from the safetensors headers (no data read yet); ggml ne lists
    # the innermost (input) dimension first, the reverse of the torch shape.
    tensors = []
    for gname, srcs, ttype in plan:
        for s in srcs:
            if not st.has(s):
                sys.exit(f"[error] {s} is not in the checkpoint index")
        _, h = st._header(st.index[srcs[0]])
        shape = list(h[srcs[0]]["shape"])
        if h[srcs[0]]["dtype"] == "I8":
            shape[-1] *= 2  # packed FP4
        ne = list(reversed(shape))
        if len(srcs) > 1:
            ne.append(len(srcs))
        tensors.append((gname, srcs, ttype, ne))
    tensors.append(("dflash.dspark.confidence.bias", [], T_F32, [1]))

    kv = [
        ("general.architecture", ARCH),
        ("general.name", f"{cfg.get('model_type', 'deepseek')}-dspark-draft"),
        (f"{ARCH}.block_count", n_stage),
        (f"{ARCH}.embedding_length", n_embd),
        (f"{ARCH}.vocab_size", vocab),
        (f"{ARCH}.attention.head_count", n_head),
        (f"{ARCH}.attention.head_count_kv", int(tc.get("num_key_value_heads", 1))),
        (f"{ARCH}.attention.key_length", head_dim),
        (f"{ARCH}.attention.value_length", head_dim),
        (f"{ARCH}.rope.dimension_count", int(tc["qk_rope_head_dim"])),
        (f"{ARCH}.rope.freq_base", float(tc["rope_theta"])),
        (f"{ARCH}.attention.q_lora_rank", int(tc["q_lora_rank"])),
        (f"{ARCH}.attention.output_lora_rank", int(tc["o_lora_rank"])),
        (f"{ARCH}.attention.output_group_count", int(tc["o_groups"])),
        (f"{ARCH}.attention.sliding_window", int(tc["sliding_window"])),
        (f"{ARCH}.attention.layer_norm_rms_epsilon", float(tc["rms_norm_eps"])),
        (f"{ARCH}.expert_count", n_exp),
        (f"{ARCH}.expert_used_count", n_used),
        (f"{ARCH}.expert_shared_count", int(tc["n_shared_experts"])),
        (f"{ARCH}.expert_feed_forward_length", n_ff),
        (f"{ARCH}.hash_layer_count", 0),
        (f"{ARCH}.expert_weights_scale", float(tc["routed_scaling_factor"])),
        (f"{ARCH}.expert_gating_func", str(tc["scoring_func"])),
        (f"{ARCH}.swiglu_clamp_exp", float(tc["swiglu_limit"])),
        (f"{ARCH}.hyper_connection.count", n_hc),
        (f"{ARCH}.hyper_connection.epsilon", float(tc["hc_eps"])),
        (f"{ARCH}.hyper_connection.sinkhorn_iterations", int(tc["hc_sinkhorn_iters"])),
        (f"{ARCH}.dflash.n_target_layers", len(targets)),
        (f"{ARCH}.dflash.block_size", int(tc["dspark_block_size"])),
        (f"{ARCH}.dflash.mask_token_id", int(tc["dspark_noise_token_id"])),
        # The reference feeds the draft the residual entering each target
        # layer; the engine captures the residual after a layer, i.e. the
        # input of the next one.
        (f"{ARCH}.dflash.target_layer_ids", targets),
        (f"{ARCH}.dflash.capture_layer_ids", [t - 1 for t in targets]),
        (f"{ARCH}.dflash.head_hc_enabled", 0),
        (f"{ARCH}.dflash.dspark.enabled", 1),
        (f"{ARCH}.dflash.dspark.markov_rank", rank),
        (f"{ARCH}.dflash.dspark.vocab_size", vocab),
        (f"{ARCH}.dflash.dspark.confidence_dim", n_embd + rank),
        (f"{ARCH}.dflash.dspark.confidence.enabled", 1),
    ]

    align = 32
    infos, offset = [], 0
    for gname, srcs, ttype, ne in tensors:
        size = nbytes(ne, ttype)
        infos.append((gname, ne, ttype, offset, size))
        offset += (size + align - 1) // align * align

    t0 = time.time()
    with open(args.out, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(infos), len(kv)))
        for k, v in kv:
            w_kv(f, k, v)
        for gname, ne, ttype, off, _ in infos:
            w_str(f, gname)
            f.write(struct.pack("<I", len(ne)))
            f.write(struct.pack(f"<{len(ne)}Q", *ne))
            f.write(struct.pack("<IQ", ttype, off))
        f.write(b"\0" * ((-f.tell()) % align))
        data_start = f.tell()
        for (gname, srcs, ttype, ne), (_, _, _, off, size) in zip(tensors, infos):
            assert f.tell() - data_start == off
            if not srcs:
                blob = np.zeros(ne, dtype=np.float32).tobytes()
            else:
                parts = [repack_mxfp4(st, s) if ttype == T_MXFP4 else encode(q, load_f32(st, s), ttype)
                         for s in srcs]
                blob = b"".join(parts)
            assert len(blob) == size, (gname, len(blob), size)
            f.write(blob)
            f.write(b"\0" * ((-size) % align))
            print(f"[info] {gname:40s} {str(ne):28s} type={ttype:3d} {size / 2**20:9.1f} MiB  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(f"[info] wrote {args.out} ({offset / 2**30:.2f} GiB, {len(infos)} tensors)")


if __name__ == "__main__":
    main()
