#!/usr/bin/env python3
"""Compare the engine's DeepSeek V4.1 image path with DeepSeek's reference code.

Runs the checkpoint's own inference/image_processor.py (load_image) and
inference/vision.py (ViT, Aligner) on one image, with the encoder weights read
from the checkpoint's first two shards, and compares them with what
test_ds41v_encoder_dump wrote for the same image:

  patches     the BF16 ViT input: must match bit for bit (same Pillow resize,
              same normalization, same patch layout);
  features    the ViT output, before the aligner;
  embeddings  the aligner rows the decoder receives.

    LUCE_DS41V_MMPROJ=... LUCE_DS41V_IMAGE=img.jpg LUCE_DS41V_DUMP=out ./test_ds41v_encoder_dump
    python ds41v_reference_check.py --inference <checkpoint>/inference \\
        --checkpoint <checkpoint> --image img.jpg --dump out

The reference runs in float32 (exact math) and in bfloat16 (the precision
the checkpoint runs at); the report gives, for each, the relative L2 error and
the worst per-row cosine similarity of the engine's output against it.
Needs torch, Pillow, numpy and safetensors.
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encoder_weights(checkpoint: Path):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in index.items() if k.startswith(("vision.", "aligner."))}
    out = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(str(checkpoint / shard), framework="pt") as f:
            for name in f.keys():
                if name in wanted:
                    out[name] = f.get_tensor(name)
    return out


def compare(ours: np.ndarray, ref: np.ndarray) -> dict:
    a = ours.astype(np.float64)
    b = ref.astype(np.float64)
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
    return {"rel_l2": rel, "min_row_cosine": float(cos.min()), "mean_row_cosine": float(cos.mean()),
            "max_abs": float(np.abs(a - b).max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inference", type=Path, required=True, help="the checkpoint's inference/ folder")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with the index and shards 1-2")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True, help="test_ds41v_encoder_dump's output folder")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    config = json.loads((args.inference / "config.json").read_text())
    vision_args = types.SimpleNamespace(**config)
    sys.path.insert(0, str(args.inference))
    processor = load_module(args.inference / "image_processor.py", "image_processor")
    vision = load_module(args.inference / "vision.py", "vision")

    plan = json.loads((args.dump / "plan.json").read_text())
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = processor.load_image({"url": str(args.image)}, vision_args)
    report = {"image": args.image.name, "engine_device": plan["device"], "engine_encode_ms": plan["encode_ms"],
              "grid": [n_vit_h, n_vit_w], "aligner_grid": [n_llm_h, n_llm_w]}
    plan_matches = [plan["vit_rows"], plan["vit_cols"], plan["aligner_rows"], plan["aligner_cols"]] == \
        [n_vit_h, n_vit_w, n_llm_h, n_llm_w]
    report["plan_matches"] = plan_matches
    ref_bits = patches.contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)
    our_bits = np.fromfile(args.dump / "patches.bf16", dtype=np.uint16)
    report["patches_identical"] = bool(our_bits.shape == ref_bits.shape and np.array_equal(our_bits, ref_bits))
    if not plan_matches or our_bits.shape != ref_bits.shape:
        print(json.dumps(report, indent=1))
        return 1
    report["patch_words_differing"] = int((our_bits != ref_bits).sum())

    weights = encoder_weights(args.checkpoint)
    rows = n_llm_h * n_llm_w
    ours_features = np.fromfile(args.dump / "features.f32", dtype=np.float32).reshape(n_vit_h * n_vit_w, -1)
    ours_embeddings = np.fromfile(args.dump / "embeddings.f32", dtype=np.float32).reshape(rows, -1)
    refs = {}
    for dtype in (torch.float32, torch.bfloat16):
        torch.set_default_dtype(dtype)
        vit, aligner = vision.ViT(vision_args), vision.Aligner(vision_args)
        vit.load_state_dict({k[len("vision."):]: v.to(dtype) for k, v in weights.items() if k.startswith("vision.")})
        aligner.load_state_dict({k[len("aligner."):]: v.to(dtype) for k, v in weights.items() if k.startswith("aligner.")})
        with torch.inference_mode():
            features = vit(patches.to(dtype), n_vit_h, n_vit_w)
            embeddings = aligner(features, n_vit_h, n_vit_w)
        name = "float32" if dtype == torch.float32 else "bfloat16"
        refs[name] = (features.float().numpy(), embeddings.float().numpy())
        report[f"features_vs_{name}"] = compare(ours_features, refs[name][0])
        report[f"embeddings_vs_{name}"] = compare(ours_embeddings, refs[name][1])
    # The noise floor: the reference's own bfloat16 run against its float32 run.
    report["reference_bfloat16_vs_float32"] = {
        "features": compare(refs["bfloat16"][0], refs["float32"][0]),
        "embeddings": compare(refs["bfloat16"][1], refs["float32"][1]),
    }
    # The rows furthest from the float32 reference, with their sizes.
    ref32 = refs["float32"][0].astype(np.float64)
    a = ours_features.astype(np.float64)
    cos = (a * ref32).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(ref32, axis=1), 1e-30)
    worst = np.argsort(cos)[:5]
    report["worst_feature_rows"] = [
        {"row": int(r), "patch": [int(r) // n_vit_w, int(r) % n_vit_w], "cosine": float(cos[r]),
         "norm_ours": float(np.linalg.norm(a[r])), "norm_ref": float(np.linalg.norm(ref32[r])),
         "median_norm": float(np.median(np.linalg.norm(ref32, axis=1)))} for r in worst]
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
