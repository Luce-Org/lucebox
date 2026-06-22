#!/usr/bin/env python3
"""
TQ3_0 bit-plane ranking experiment (phase0 kill-test).

Reads real Q/K tensors dumped by the dflash_server with PHASE0_DUMP=1
and --kvflash-policy qk. Measures whether the MSB (bit-plane 2) of the
TQ3_0 Lloyd-Max codes ranks attention keys as well as the full QK dot.

Usage:
    python3 bench/bitplane_lsh_experiment.py \
        --k /tmp/phase0_dump/phase0_K.bin \
        --q /tmp/phase0_dump/phase0_Q.bin \
        --out-dir bench/qwen35moe_dflash/ctxsweep/

Tensor layout (from server dump):
    K: [N_tokens, head_dim=256] float32, layer=0 head=0, FWHT-rotated domain
       (dequanted from TQ3_0 cache — values = norm * centroid_i)
    Q: [n_layers=16, n_q_heads=24, head_dim=256] float32, FWHT-rotated domain
       Experiment uses layer=0, head=0 only for per-token analysis.

TQ3_0 codebook (from tq3-quant.cuh):
    centroids: [-0.190685, -0.117832, -0.065717, -0.021460,
                 0.021460,  0.065717,  0.117832,  0.190685]
    code 0-3: negative half, code 4-7: positive half
    bit layout: bits[1:0] = low 2 bits, bit[2] = sign/high bit

Bit-plane approximations:
    1-bit (MSB=bit2): keep only sign, dequant → -c_mean or +c_mean
    2-bit (bits[1:0]): keep low 2 bits, dequant from 2-bit sub-codebook
    3-bit (all): full dequant = baseline
    RANDOM: uniform random scores (floor)
    FULL: true QK dot (ceiling)
"""

import argparse
import json
import math
import os
import sys

import numpy as np


# TQ3_0 Lloyd-Max centroids (from tq3-quant.cuh)
TQ3_CENTROIDS = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
], dtype=np.float32)

TQ3_MIDS = np.array([
    -0.154259, -0.091775, -0.043589, 0.0,
     0.043589,  0.091775,  0.154259
], dtype=np.float32)


def tq3_quantize_normalized(x_normed: np.ndarray) -> np.ndarray:
    """Quantize normalized values (pre-scaled by 1/norm) to 3-bit codes 0-7.
    x_normed: arbitrary shape, values in range of centroids.
    Returns: int array, same shape, values in [0,7].
    """
    codes = np.zeros(x_normed.shape, dtype=np.uint8)
    codes[x_normed >= TQ3_MIDS[0]] = 1
    codes[x_normed >= TQ3_MIDS[1]] = 2
    codes[x_normed >= TQ3_MIDS[2]] = 3
    codes[x_normed >= TQ3_MIDS[3]] = 4
    codes[x_normed >= TQ3_MIDS[4]] = 5
    codes[x_normed >= TQ3_MIDS[5]] = 6
    codes[x_normed >= TQ3_MIDS[6]] = 7
    return codes


def norm_and_quantize(K_tokens: np.ndarray, block_size: int = 32):
    """
    Re-quantize K rows using TQ3_0 per-block norm.
    K_tokens: [N, head_dim], already in rotated domain.
    Returns codes: [N, head_dim] uint8 in [0,7]
    """
    N, D = K_tokens.shape
    codes = np.zeros((N, D), dtype=np.uint8)
    for start in range(0, D, block_size):
        end = min(start + block_size, D)
        block = K_tokens[:, start:end]  # [N, bs]
        norms = np.linalg.norm(block, axis=1, keepdims=True)  # [N,1]
        norms = np.maximum(norms, 1e-6)
        normed = block / norms  # [N, bs] in centroid range
        codes[:, start:end] = tq3_quantize_normalized(normed)
    return codes


def dequant_1bit(codes: np.ndarray, K_tokens: np.ndarray, block_size: int = 32) -> np.ndarray:
    """1-bit approximation: use only MSB (bit 2 = sign).
    Dequant value = sign(code) * mean(|centroids|).
    Returns: [N, head_dim] float32
    """
    # MSB: codes >= 4 → positive half, else negative
    sign = np.where(codes >= 4, 1.0, -1.0).astype(np.float32)
    mean_abs_c = np.mean(np.abs(TQ3_CENTROIDS))  # ≈ 0.100736
    out = np.zeros_like(K_tokens)
    for start in range(0, K_tokens.shape[1], block_size):
        end = min(start + block_size, K_tokens.shape[1])
        block = K_tokens[:, start:end]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        out[:, start:end] = sign[:, start:end] * mean_abs_c * norms
    return out


def dequant_2bit(codes: np.ndarray, K_tokens: np.ndarray, block_size: int = 32) -> np.ndarray:
    """2-bit approximation: use only low 2 bits (bits 1:0).
    Maps low2 → mean centroid of that 2-bit bucket:
        00 → mean(c0,c4) = 0 (or use mean of the 2 centroids)
        01 → mean(c1,c5)
        10 → mean(c2,c6)
        11 → mean(c3,c7)
    But we don't know the sign without bit2, so we use mean magnitude.
    Actually: reconstruct using |centroid[low2]| with centroid sign from 0.
    Better: use the mean of the two centroids at each low2 value.
    """
    # Two centroids per low2 value: one positive, one negative
    # Mean abs centroids for each low2 bucket:
    # low2=0: (|c0| + |c4|)/2 = (0.190685+0.021460)/2 = 0.106073
    # low2=1: (|c1| + |c5|)/2 = (0.117832+0.065717)/2 = 0.091775
    # low2=2: (|c2| + |c6|)/2 = (0.065717+0.117832)/2 = 0.091775
    # low2=3: (|c3| + |c7|)/2 = (0.021460+0.190685)/2 = 0.106073
    low2 = (codes & 0x3).astype(np.int32)
    # Use the abs mean centroid (2-bit recovers magnitude, not sign)
    mag_lut = np.array([0.106073, 0.091775, 0.091775, 0.106073], dtype=np.float32)
    # For the 2-bit approximation we LOSE sign info — this tests just magnitude recovery
    mag = mag_lut[low2]  # [N, D]
    out = np.zeros_like(K_tokens)
    for start in range(0, K_tokens.shape[1], block_size):
        end = min(start + block_size, K_tokens.shape[1])
        block = K_tokens[:, start:end]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        out[:, start:end] = mag[:, start:end] * norms
    return out


def dequant_3bit(codes: np.ndarray, K_tokens: np.ndarray, block_size: int = 32) -> np.ndarray:
    """3-bit dequant: full TQ3_0 reconstruction (should recover K_tokens closely)."""
    vals = TQ3_CENTROIDS[codes]  # [N, D] float32
    out = np.zeros_like(K_tokens)
    for start in range(0, K_tokens.shape[1], block_size):
        end = min(start + block_size, K_tokens.shape[1])
        block = K_tokens[:, start:end]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        out[:, start:end] = vals[:, start:end] * norms
    return out


def compute_attention_scores(K: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute s_true[j] = K[j] · q / sqrt(D). Returns [N]."""
    D = q.shape[0]
    scores = K.dot(q) / math.sqrt(D)
    return scores


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def mass_recall_at_k(true_weights: np.ndarray, approx_scores: np.ndarray,
                     k_frac: float) -> float:
    """Fraction of total softmax weight captured by top-k% tokens by approx score.
    Mass-recall = sum(w[top-k by approx]) / sum(w[all]).
    k_frac: fraction of N (e.g. 0.05 = top 5%).
    """
    N = len(true_weights)
    k = max(1, int(N * k_frac))
    top_idx = np.argpartition(approx_scores, -k)[-k:]
    return float(true_weights[top_idx].sum() / true_weights.sum())


def count_recall_at_k(true_scores: np.ndarray, approx_scores: np.ndarray,
                      k_frac: float) -> float:
    """Fraction of true top-k% tokens that appear in approx top-k%.
    Count-recall = |top-k-true ∩ top-k-approx| / k.
    """
    N = len(true_scores)
    k = max(1, int(N * k_frac))
    true_top = set(np.argpartition(true_scores, -k)[-k:].tolist())
    approx_top = set(np.argpartition(approx_scores, -k)[-k:].tolist())
    return len(true_top & approx_top) / k


def run_experiment(k_path: str, q_path: str, out_dir: str):
    print(f"Loading K from {k_path}")
    K_flat = np.fromfile(k_path, dtype=np.float32)
    head_dim = 256
    N = K_flat.shape[0] // head_dim
    K = K_flat.reshape(N, head_dim)
    print(f"  K shape: {K.shape}  (N={N} tokens, {N//64} chunks of 64)")

    print(f"Loading Q from {q_path}")
    Q_flat = np.fromfile(q_path, dtype=np.float32)
    # Q layout: [n_layers=16, n_q_heads=24, head_dim=256]
    n_layers, n_q_heads, d = 16, 24, 256
    assert Q_flat.shape[0] == n_layers * n_q_heads * d, \
        f"Q size {Q_flat.shape[0]} != {n_layers*n_q_heads*d}"
    Q_3d = Q_flat.reshape(n_layers, n_q_heads, d)
    # Use layer=0, head=0 (same layer/head as K dump)
    q = Q_3d[0, 0]  # [256]
    print(f"  Q shape: {Q_3d.shape}  using layer=0 head=0")

    # Quantize K to 3-bit codes
    print("Quantizing K to TQ3_0 codes...")
    codes = norm_and_quantize(K)

    # True scores and softmax weights
    s_true = compute_attention_scores(K, q)
    w_true = softmax(s_true)
    print(f"  s_true: min={s_true.min():.4f} max={s_true.max():.4f} mean={s_true.mean():.4f}")
    print(f"  w_true entropy: {float(-np.sum(w_true * np.log(w_true + 1e-30))):.3f} nats")
    print(f"  w_true max (concentration): {float(w_true.max()):.6f}")

    # Dequant approximations
    K_1bit = dequant_1bit(codes, K)
    K_2bit = dequant_2bit(codes, K)
    K_3bit = dequant_3bit(codes, K)

    # Approximate scores
    s_1bit = K_1bit.dot(q) / math.sqrt(head_dim)
    s_2bit = K_2bit.dot(q) / math.sqrt(head_dim)
    s_3bit = K_3bit.dot(q) / math.sqrt(head_dim)
    s_random = np.random.RandomState(42).randn(N).astype(np.float32)

    k_fracs = [0.01, 0.02, 0.05, 0.10, 0.20]

    arms = [
        ("RANDOM",  s_random),
        ("1-bit",   s_1bit),
        ("2-bit",   s_2bit),
        ("3-bit",   s_3bit),
        ("FULL-QK", s_true),
    ]

    results = {}
    print("\n=== Mass-recall@k (captured softmax weight fraction) ===")
    header = f"{'Arm':<10}" + "".join(f"  k={int(f*100):>2}%" for f in k_fracs)
    print(header)
    for name, scores in arms:
        row = {}
        line = f"{name:<10}"
        for frac in k_fracs:
            mr = mass_recall_at_k(w_true, scores, frac)
            row[f"mass_recall@{int(frac*100)}pct"] = round(mr, 4)
            line += f"  {mr:.4f}    "
        print(line)
        results[name] = row

    print("\n=== Count-recall@k (top-k% overlap fraction) ===")
    print(header)
    for name, scores in arms:
        line = f"{name:<10}"
        for frac in k_fracs:
            cr = count_recall_at_k(s_true, scores, frac)
            results[name][f"count_recall@{int(frac*100)}pct"] = round(cr, 4)
            line += f"  {cr:.4f}    "
        print(line)

    # Sink+recent always-kept analysis
    # Force-keep: first 64 tokens (1 sink chunk) + last 256 tokens (4 tail chunks)
    sink_toks = 64
    tail_toks = 256
    always_kept = np.zeros(N, dtype=bool)
    always_kept[:sink_toks] = True
    always_kept[max(0, N-tail_toks):] = True
    n_always = always_kept.sum()
    print(f"\n=== With sink+recent force-keep ({n_always}/{N} = {n_always/N:.1%} tokens) ===")
    # After force-keep, score remaining with approximation, pick top-k of remainder
    remaining = ~always_kept
    n_remaining = remaining.sum()

    print(f"{'Arm':<10}" + "".join(f"  k={int(f*100):>2}%(eff)" for f in k_fracs))
    for name, scores in arms:
        line = f"{name:<10}"
        for frac in k_fracs:
            k_total = max(1, int(N * frac))
            k_extra = max(0, k_total - n_always)
            # Top k_extra from remaining by approx
            rem_scores = scores[remaining]
            if k_extra > 0 and len(rem_scores) > k_extra:
                rem_top_idx = np.where(remaining)[0][
                    np.argpartition(rem_scores, -k_extra)[-k_extra:]
                ]
            else:
                rem_top_idx = np.where(remaining)[0]
            kept_idx = np.concatenate([np.where(always_kept)[0], rem_top_idx])
            mr = float(w_true[kept_idx].sum() / w_true.sum())
            line += f"  {mr:.4f}      "
            results[name][f"mass_recall_with_sink@{int(frac*100)}pct"] = round(mr, 4)
        print(line)

    # Per-head mass-recall at k=10% for 1-bit vs 3-bit
    print("\n=== Per-head mass-recall@10% (1-bit vs 3-bit vs FULL-QK) ===")
    head_results = []
    for h in range(n_q_heads):
        q_h = Q_3d[0, h]
        s_true_h = compute_attention_scores(K, q_h)
        w_h = softmax(s_true_h)
        s_1bit_h = K_1bit.dot(q_h) / math.sqrt(head_dim)
        s_3bit_h = K_3bit.dot(q_h) / math.sqrt(head_dim)
        mr_1bit = mass_recall_at_k(w_h, s_1bit_h, 0.10)
        mr_3bit = mass_recall_at_k(w_h, s_3bit_h, 0.10)
        mr_full = mass_recall_at_k(w_h, s_true_h, 0.10)
        mr_rand = mass_recall_at_k(w_h, np.random.RandomState(h).randn(N).astype(np.float32), 0.10)
        head_results.append((h, mr_1bit, mr_3bit, mr_full, mr_rand))

    mr1_vals = [r[1] for r in head_results]
    mr3_vals = [r[2] for r in head_results]
    mrf_vals = [r[3] for r in head_results]
    mrr_vals = [r[4] for r in head_results]
    print(f"          {'min':>6} {'median':>8} {'max':>6}")
    print(f"  1-bit   {min(mr1_vals):6.4f} {np.median(mr1_vals):8.4f} {max(mr1_vals):6.4f}")
    print(f"  3-bit   {min(mr3_vals):6.4f} {np.median(mr3_vals):8.4f} {max(mr3_vals):6.4f}")
    print(f"  FULL-QK {min(mrf_vals):6.4f} {np.median(mrf_vals):8.4f} {max(mrf_vals):6.4f}")
    print(f"  RANDOM  {min(mrr_vals):6.4f} {np.median(mrr_vals):8.4f} {max(mrr_vals):6.4f}")

    # Find at what k does 1-bit mass-recall hit 0.9 and beat random
    print("\n=== 1-bit: at what k% does mass-recall hit 0.9 / beat random? ===")
    for frac_search in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        mr1 = mass_recall_at_k(w_true, s_1bit, frac_search)
        mrr = mass_recall_at_k(w_true, s_random, frac_search)
        beat = "BEATS-RANDOM" if mr1 > mrr else "below-random"
        print(f"  k={int(frac_search*100):>3}%: 1-bit={mr1:.4f}  random={mrr:.4f}  → {beat}")
        if mr1 >= 0.9:
            print(f"  *** 1-bit mass-recall hits 0.9 at k={int(frac_search*100)}% ***")
            break

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho_1bit, _ = spearmanr(s_true, s_1bit)
    rho_2bit, _ = spearmanr(s_true, s_2bit)
    rho_3bit, _ = spearmanr(s_true, s_3bit)
    rho_rand, _ = spearmanr(s_true, s_random)
    print(f"\n=== Spearman ρ vs true QK score ===")
    print(f"  RANDOM:  {rho_rand:.4f}")
    print(f"  1-bit:   {rho_1bit:.4f}")
    print(f"  2-bit:   {rho_2bit:.4f}")
    print(f"  3-bit:   {rho_3bit:.4f}")
    print(f"  FULL-QK: 1.0000")

    results["_meta"] = {
        "N_tokens": int(N),
        "n_chunks": int(N // 64),
        "head_dim": int(head_dim),
        "k_file": k_path,
        "q_file": q_path,
        "layer": 0,
        "head": 0,
        "spearman_rho_1bit": round(float(rho_1bit), 4),
        "spearman_rho_2bit": round(float(rho_2bit), 4),
        "spearman_rho_3bit": round(float(rho_3bit), 4),
        "spearman_rho_random": round(float(rho_rand), 4),
        "per_head_mass_recall_10pct": {
            "1bit_min": round(min(mr1_vals), 4),
            "1bit_median": round(float(np.median(mr1_vals)), 4),
            "1bit_max": round(max(mr1_vals), 4),
            "3bit_min": round(min(mr3_vals), 4),
            "3bit_median": round(float(np.median(mr3_vals)), 4),
            "3bit_max": round(max(mr3_vals), 4),
            "full_min": round(min(mrf_vals), 4),
            "full_median": round(float(np.median(mrf_vals)), 4),
            "full_max": round(max(mrf_vals), 4),
            "random_median": round(float(np.median(mrr_vals)), 4),
        }
    }

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "phase0_bitplane_lsh_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default="/tmp/phase0_dump/phase0_K.bin")
    parser.add_argument("--q", default="/tmp/phase0_dump/phase0_Q.bin")
    parser.add_argument("--out-dir",
                        default="/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep/")
    args = parser.parse_args()
    run_experiment(args.k, args.q, args.out_dir)


if __name__ == "__main__":
    main()
