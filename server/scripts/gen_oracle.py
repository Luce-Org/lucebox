"""
Generate deterministic oracle inputs + expected output for the DFlash draft
forward pass, using the validated Phase 0 PyTorch reference in
../../megaqwen3_27b_dflash/reference/.

Output: a directory with three float32 binary files:
    noise.bin          shape [1, q_len=16, hidden=5120]  (contiguous row-major)
    target.bin         shape [1, ctx_len, 5*hidden=25600]
    expected.bin       shape [1, q_len=16, hidden=5120]

All files are flat float32 arrays, no header. The C++ test loads them with
the same shape convention (ggml col-major = PyTorch row-major for these
specific layouts since everything is contiguous).

Usage:
    python gen_oracle.py --out /tmp/dflash_oracle --ctx-len 64
"""

import argparse
import os
import sys
import struct

import torch

# Import the existing reference implementation
HERE = os.path.abspath(os.path.dirname(__file__))
REF_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "megaqwen3_27b_dflash", "reference"))
sys.path.insert(0, REF_DIR)

from dflash_reference import (
    DFlashConfig,
    dflash_forward_core,
)
from load_weights import load_dflash_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ctx-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    ref_dtype = torch.bfloat16  # reference was trained in bf16
    cfg: DFlashConfig
    weights: 'DFlashWeights'
    cfg, weights = load_dflash_weights(device=device, dtype=ref_dtype)

    # Quick sanity print
    print(f"loaded weights: fc={tuple(weights.fc.shape)}, "
          f"layers={len(weights.layers)}")

    q_len = cfg.block_size                                 # 16
    hidden = cfg.hidden_size                                # 5120
    ctx_len = args.ctx_len
    fc_in_dim = len(cfg.target_layer_ids) * hidden          # 25600

    torch.manual_seed(args.seed)
    noise = (torch.randn(1, q_len, hidden, device=device, dtype=ref_dtype)
             * 0.02)
    target_hidden_cat = (
        torch.randn(1, ctx_len, fc_in_dim, device=device, dtype=ref_dtype)
        * 0.02)
    position_ids = torch.arange(
        0, ctx_len + q_len, device=device, dtype=torch.long
    ).unsqueeze(0).expand(1, ctx_len + q_len)

    # Run the reference forward (matches the C++ graph semantically)
    with torch.inference_mode():
        out_hidden = dflash_forward_core(
            noise_embedding=noise,
            target_hidden_concat=target_hidden_cat,
            position_ids=position_ids,
            weights=weights,
            cfg=cfg,
        )

    print(f"reference out: shape={tuple(out_hidden.shape)} "
          f"dtype={out_hidden.dtype} "
          f"mean={out_hidden.float().mean().item():.6g} "
          f"std={out_hidden.float().std().item():.6g}")

    # Save as float32 row-major contiguous
    noise_f32  = noise.to(torch.float32).contiguous().cpu().numpy()
    target_f32 = target_hidden_cat.to(torch.float32).contiguous().cpu().numpy()
    out_f32    = out_hidden.to(torch.float32).contiguous().cpu().numpy()

    noise_f32.tofile(os.path.join(args.out, "noise.bin"))
    target_f32.tofile(os.path.join(args.out, "target.bin"))
    out_f32.tofile(os.path.join(args.out, "expected.bin"))

    # Also write a small metadata file for the C++ test to read
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"ctx_len={ctx_len}\n")
        f.write(f"q_len={q_len}\n")
        f.write(f"hidden={hidden}\n")
        f.write(f"fc_in={fc_in_dim}\n")
        f.write(f"seed={args.seed}\n")

    print(f"wrote: {args.out}/{{noise,target,expected}}.bin + meta.txt")


if __name__ == "__main__":
    main()
