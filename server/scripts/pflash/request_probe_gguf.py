#!/usr/bin/env python3
"""Write a PFlash request-probe GGUF (PFLASH_REQUEST_PROBE_GGUF).

The runtime contract (server/src/pflash/pflash_request_probe.h): general.architecture "requestprobe", schema
"qwen3_5_0_8b_request_probe_v1", F32 tensors requestprobe.{norm,fc1,fc2}.{weight,bias} and F32 metadata
requestprobe.norm_eps and requestprobe.tau (raw-logit threshold, comparator >=), and string
requestprobe.extraction_version (the run-extraction rule the tau was calibrated for; currently
request_runs_v2_minrun2_ge_raw_logit). Torch weights are written as-is:
LayerNorm(1024), Linear(1024, W).weight [W, 1024], Linear(W, 1).weight [1, W].

  request_probe_gguf.py OUT.gguf --drafter-sha256 HEX --tau T --weights probe.npz
      npz keys: norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias (torch shapes)
  request_probe_gguf.py OUT.gguf --drafter-sha256 HEX --tau T --random-seed S [--width W]
      random weights, for runtime plumbing tests only (not a trained probe)

Needs numpy and gguf-py (PYTHONPATH=server/deps/llama.cpp/gguf-py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deps/llama.cpp/gguf-py"))
import gguf

HIDDEN = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--drafter-sha256", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--norm-eps", type=float, default=1e-5)
    # kRequestExtractionVersion in pflash_request_probe.h.
    ap.add_argument("--extraction-version", default="request_runs_v2_minrun2_ge_raw_logit")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights")
    src.add_argument("--random-seed", type=int)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    if args.weights:
        z = np.load(args.weights)
        t = {k: np.asarray(z[k], dtype=np.float32) for k in
             ("norm_weight", "norm_bias", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")}
    else:
        rng = np.random.default_rng(args.random_seed)
        w = args.width
        t = {
            "norm_weight": np.ones(HIDDEN, np.float32) + 0.1 * rng.standard_normal(HIDDEN).astype(np.float32),
            "norm_bias": 0.1 * rng.standard_normal(HIDDEN).astype(np.float32),
            "fc1_weight": (rng.standard_normal((w, HIDDEN)) / np.sqrt(HIDDEN)).astype(np.float32),
            "fc1_bias": 0.1 * rng.standard_normal(w).astype(np.float32),
            "fc2_weight": (rng.standard_normal((1, w)) / np.sqrt(w)).astype(np.float32),
            "fc2_bias": np.zeros(1, np.float32),
        }
    width = t["fc1_weight"].shape[0]
    expect = {"norm_weight": (HIDDEN,), "norm_bias": (HIDDEN,), "fc1_weight": (width, HIDDEN),
              "fc1_bias": (width,), "fc2_weight": (1, width), "fc2_bias": (1,)}
    for k, shape in expect.items():
        if t[k].shape != shape:
            print(f"{k}: shape {t[k].shape}, expected {shape}", file=sys.stderr)
            return 1

    w = gguf.GGUFWriter(args.out, "requestprobe")
    w.add_string("requestprobe.schema", "qwen3_5_0_8b_request_probe_v1")
    w.add_string("requestprobe.base_model", "Qwen/Qwen3.5-0.8B")
    w.add_string("requestprobe.runtime_gguf_sha256", args.drafter_sha256)
    w.add_string("requestprobe.feature_tap", "post_block14_residual_before_block15")
    w.add_float32("requestprobe.norm_eps", args.norm_eps)
    w.add_float32("requestprobe.tau", args.tau)  # raw logit, comparator >=
    w.add_string("requestprobe.extraction_version", args.extraction_version)
    w.add_tensor("requestprobe.norm.weight", t["norm_weight"])
    w.add_tensor("requestprobe.norm.bias", t["norm_bias"])
    w.add_tensor("requestprobe.fc1.weight", t["fc1_weight"])
    w.add_tensor("requestprobe.fc1.bias", t["fc1_bias"])
    w.add_tensor("requestprobe.fc2.weight", t["fc2_weight"].reshape(width))
    w.add_tensor("requestprobe.fc2.bias", t["fc2_bias"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (width {width}, tau {args.tau})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
