#!/usr/bin/env python3
"""Requantize a DFlash draft GGUF (f16) for faster spec-decode drafting.

Schemes:
  q8_0    - all 2D matmul weights to q8_0 (~47% smaller, acceptance-neutral)
  q4-mix  - backbone (blk.*) weights to q4_0, dflash.* head weights kept at
            q8_0 (~67% smaller; the head split protects the Markov/projection
            bias precision that near-tie corrections depend on).

Norms, biases and non-f16 tensors are copied as-is. Metadata is preserved,
so the output loads anywhere the input does.

Measured on Laguna-XS.2 + v24 drafter (RTX 3090, HumanEval, verify width 6):
f16 236.6 tok/s -> q8_0 241.3 -> q4-mix 249.6, acceptance unchanged.
"""
import argparse
import sys

from pathlib import Path
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.quants import quantize  # noqa: E402


def pick_type(name: str, scheme: str) -> GGMLQuantizationType:
    if scheme == "q8_0":
        return GGMLQuantizationType.Q8_0
    # q4-mix: protect the dflash.* heads
    if name.startswith("dflash."):
        return GGMLQuantizationType.Q8_0
    return GGMLQuantizationType.Q4_0


def copy_metadata(r: GGUFReader, w: GGUFWriter) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    T = gguf.GGUFValueType
    for f in r.fields.values():
        if f.name in skip:
            continue
        ftype = f.types[0]
        val = f.parts[f.data[0]]
        if ftype == T.STRING:
            w.add_string(f.name, bytes(val).decode())
        elif ftype == T.ARRAY:
            sub = f.types[1]
            vals = [f.parts[i] for i in f.data]
            if sub == T.STRING:
                w.add_array(f.name, [bytes(v).decode() for v in vals])
            else:
                w.add_array(f.name, [np.asarray(v)[0].item() for v in vals])
        elif ftype == T.BOOL:
            w.add_bool(f.name, bool(val[0]))
        elif ftype == T.FLOAT32:
            w.add_float32(f.name, float(val[0]))
        elif ftype == T.FLOAT64:
            w.add_float64(f.name, float(val[0]))
        else:
            fn = {T.UINT32: w.add_uint32, T.INT32: w.add_int32,
                  T.UINT64: w.add_uint64, T.INT64: w.add_int64,
                  T.UINT8: w.add_uint8, T.INT8: w.add_int8,
                  T.UINT16: w.add_uint16, T.INT16: w.add_int16}[ftype]
            fn(f.name, val[0].item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="f16 draft GGUF")
    ap.add_argument("output", help="output GGUF path")
    ap.add_argument("--scheme", choices=["q8_0", "q4-mix"], default="q4-mix")
    args = ap.parse_args()

    r = GGUFReader(args.input)
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode()
    if not arch:
        print("error: no general.architecture in input", file=sys.stderr)
        return 1

    w = GGUFWriter(args.output, arch)
    copy_metadata(r, w)

    n_q = n_keep = 0
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        if (t.tensor_type == GGMLQuantizationType.F16 and len(shape) == 2
                and shape[0] % 32 == 0 and "norm" not in t.name):
            qt = pick_type(t.name, args.scheme)
            arr = np.array(t.data, dtype=np.float16).reshape(shape[::-1]).astype(np.float32)
            w.add_tensor(t.name, quantize(arr, qt), raw_dtype=qt)
            n_q += 1
        else:
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
            n_keep += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.scheme}: quantized {n_q} tensors, kept {n_keep} as-is -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
