#!/usr/bin/env python3
"""Comparator for test_paged_attn_wmma.cpp two-mode dumps.

Usage: compare_paged_attn.py ref.bin cand.bin [--tol 6e-3]
       (the test writes ref=paged_attn_out_vdot2.bin,
        cand=paged_attn_out_wmma.bin, one file per route)
Exits non-zero when the global max exceeds --tol, so CI can gate on it.

The dump layout is case order from the test source; the test prints the
per-case shape line. The V_DOT2 reference dequantizes K/V via dp4a-integer
paths while the WMMA kernel dequantizes through a half2 tile, a wider gap
than the contiguous fattn differential. Per-type bounds: f16 2e-3,
q8_0 3e-3, q4_0 6e-3 (the 4-bit lattice is coarser, so the same
dequant-path difference lands ~2x higher; measured 3.4-4.3e-3 max, 3e-4
mean, <2e-4 of elements over 3e-3).

This comparator reports the global max over all concatenated cases, so the
default is the 6e-3 bound of the shipped case set (which includes q4_0).
Pass --tol 3e-3 to keep the strict f16/q8_0 bound on dumps without q4_0.
"""
import sys
import numpy as np

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = 6e-3
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    ref = np.fromfile(args[0], dtype=np.float32)
    cand = np.fromfile(args[1], dtype=np.float32)
    assert ref.shape == cand.shape, f"shape mismatch: {ref.shape} vs {cand.shape}"
    diff = np.abs(ref - cand)
    maxd = float(diff.max())
    # per-case comparison would need shape metadata; report the global max.
    ok = maxd < tol
    print(f"elements={ref.size} max_abs_diff={maxd:.6e} tol={tol:.1e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
