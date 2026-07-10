#!/usr/bin/env python3
"""
Requantize a DFlash drafter GGUF in place of llama-quantize (which does not
know the drafter arch). Big 2D projection weights are re-encoded; norms,
gates, aux heads and all metadata are copied verbatim, so any drafter GGUF
the engine loads today keeps loading.

Verification keeps the committed text greedy-exact regardless of drafter
precision; the only quantity at risk is acceptance length. Measured on
Laguna XS 2.1 + official DFlash (RTX 3090, HE screen): q4 drafter = same
acceptance within noise, ~+3% end-to-end (the drafter is latency-bound,
not bandwidth-bound, so gains are modest).

Usage:
    python3 requant_dflash_draft.py IN.gguf OUT.gguf [--variant q8|q4]

    q8: all projection mats Q8_0 (~2x smaller, lossless in practice)
    q4: projection mats Q4_0, dflash.fc kept Q8_0 (~3.4x smaller; fc is the
        feature projection with known overflow sensitivity, keep it higher)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import dequantize, quantize

QUANT_SUBSTR = ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight", "dflash.fc.weight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variant", choices=("q8", "q4"), default="q4")
    args = ap.parse_args()

    if os.path.realpath(args.src) == os.path.realpath(args.dst):
        sys.exit("src and dst are the same file: the writer truncates dst "
                 "while tensor data is still mmap-read from src")

    r = GGUFReader(args.src)
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = str(bytes(f.parts[f.data[0]]), "utf-8")
    if not arch:
        sys.exit("no general.architecture in source gguf")
    w = GGUFWriter(args.dst, arch)

    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for f in r.fields.values():
        if f.name in skip:
            continue
        ftype = f.types[0]
        if ftype == gguf.GGUFValueType.STRING:
            w.add_string(f.name, str(bytes(f.parts[f.data[0]]), "utf-8"))
        elif ftype == gguf.GGUFValueType.ARRAY:
            itype = f.types[1]
            if itype == gguf.GGUFValueType.STRING:
                vals = [str(bytes(f.parts[i]), "utf-8") for i in f.data]
            else:
                vals = [f.parts[i].tolist()[0] for i in f.data]
            w.add_array(f.name, vals)
        else:
            val = f.parts[f.data[0]].tolist()[0]
            adders = {
                gguf.GGUFValueType.UINT8: w.add_uint8,   gguf.GGUFValueType.INT8: w.add_int8,
                gguf.GGUFValueType.UINT16: w.add_uint16, gguf.GGUFValueType.INT16: w.add_int16,
                gguf.GGUFValueType.UINT32: w.add_uint32, gguf.GGUFValueType.INT32: w.add_int32,
                gguf.GGUFValueType.UINT64: w.add_uint64, gguf.GGUFValueType.INT64: w.add_int64,
                gguf.GGUFValueType.FLOAT32: w.add_float32, gguf.GGUFValueType.FLOAT64: w.add_float64,
                gguf.GGUFValueType.BOOL: w.add_bool,
            }
            adders[ftype](f.name, val)

    n_q = 0
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        do_q = (len(shape) == 2 and any(s in t.name for s in QUANT_SUBSTR)
                and shape[0] % 256 == 0)
        if do_q:
            f32 = dequantize(t.data, t.tensor_type).reshape(shape[::-1])
            qt = GGMLQuantizationType.Q8_0
            if args.variant == "q4" and "dflash.fc" not in t.name:
                qt = GGMLQuantizationType.Q4_0
            w.add_tensor(t.name, quantize(f32, qt), raw_dtype=qt)
            n_q += 1
        else:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"requantized {n_q} tensors -> {args.dst} "
          f"({Path(args.dst).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
