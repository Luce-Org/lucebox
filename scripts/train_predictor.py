#!/usr/bin/env python3
"""
train_predictor.py — Train and evaluate an expert routing predictor.

Reads binary routing data collected by `dflash_server --collect-routing FILE`,
trains a predictor to predict which K experts will be activated from the
hidden state, and reports prediction accuracy per layer and overall.

Objective: False-Positive Suppression at Confidence Threshold
-------------------------------------------------------------
The loss combines BCEWithLogitsLoss with a hinge penalty on "false positives
above threshold": non-top-4 experts the model predicts with probability >=
--threshold (default 0.75).

  L = BCE(logits, targets)
    + beta × mean( relu(logit - t*) × (1 - target) )

where t* = log(threshold / (1-threshold)) ≈ 1.099 for threshold=0.75.

Goal: any expert predicted with probability >= 75% must truly be in the
actual top-4. When the model is confident, it must be right.

Binary format per sample:
  int32  layer_idx
  int32  K
  float32[HIDDEN_DIM]  hidden_state (gate input for this layer)
  int32[K]  expert_indices (actual routing result)

Model types (--model-type):
  mlp           2-layer MLP, same-layer: h[l] -> experts[l]  (baseline)
  linear        Single linear layer, same-layer: h[l] -> experts[l]
  linear-cross  Fate cross-layer linear: h[l] -> experts[l+1]  (recommended)
  mlp-cross     Fate cross-layer MLP: h[l] -> experts[l+1]  (ablation)

Fate insight (arXiv:2502.12224): gate inputs of adjacent layers have >93%%
cosine similarity, so h[l] predicts experts[l+1] as well as h[l+1] does.
The actual MoE gate is a single linear layer, so linear-cross matches it.

Data size guidance:
  Linear model  — saturates around 2k–5k tokens; 500 tokens is a useful min.
  MLP model     — needs 5k–20k tokens for good generalisation; diversity
                  (code, math, prose, tool-calls) matters more than raw count.
  Each token produces (num_layers-1) cross-layer pairs or num_layers same-layer
  pairs.  1k tokens ≈ 59k pairs for a 60-layer model.

Usage:
    # Single file (Qwen3.6-35B-A3B defaults: HIDDEN_DIM=2048, 256 experts, 40 layers):
    python train_predictor.py routing.bin --hidden-dim 2048 --num-experts 256 --num-layers 40

    # Multiple files:
    python train_predictor.py a.bin b.bin c.bin --model-type linear-cross

    # Whole folder (loads every *.bin):
    python train_predictor.py --data-dir ./routing_data/ --model-type linear-cross

    # Laguna-XS.2 (HIDDEN_DIM=3072, 256 experts, 40 layers, top-8):
    python train_predictor.py routing.bin --hidden-dim 3072 --num-experts 256 --num-layers 40 \\
        --model-type linear-cross --epochs 20

    # Stricter false-positive suppression (higher beta):
    python train_predictor.py routing.bin --fp-penalty-weight 10.0

    # Custom threshold (e.g. 90% confidence must be right):
    python train_predictor.py routing.bin --threshold 0.90 --fp-penalty-weight 8.0

    # Save trained model:
    python train_predictor.py --data-dir ./routing_data/ --save-model predictor.pt

    # Finetune existing model with new data:
    python train_predictor.py new_data.bin --load-model predictor.pt --save-model predictor.pt \\
        --epochs 10 --lr 1e-4
"""

import random
import struct
import sys
from pathlib import Path

import numpy as np

HIDDEN_DIM = 2048   # Qwen3.6-35B-A3B default; override via --hidden-dim
NUM_EXPERTS = 256   # Qwen3.6-35B-A3B default; override via --num-experts
NUM_LAYERS = 40     # Qwen3.6-35B-A3B default; override via --num-layers


def load_routing_data(path, hidden_dim):
    """Load a single binary routing file into numpy arrays."""
    data = Path(path).read_bytes()
    offset = 0
    layers = []
    hiddens = []
    experts = []

    while offset < len(data) - 8:  # at least header
        layer_idx = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        K = struct.unpack_from('<i', data, offset)[0]
        offset += 4

        # Check for truncated record (e.g. server killed mid-write)
        needed = hidden_dim * 4 + K * 4
        if offset + needed > len(data):
            break

        h = np.frombuffer(data, dtype=np.float32, count=hidden_dim, offset=offset).copy()
        offset += hidden_dim * 4

        ei = np.frombuffer(data, dtype=np.int32, count=K, offset=offset).copy()
        offset += K * 4

        layers.append(layer_idx)
        hiddens.append(h)
        experts.append(ei)

    # Empty or fully-truncated file: return well-formed empty arrays rather
    # than crashing in np.stack([]) / max([]).
    if not experts:
        return (np.array([], dtype=np.int32),
                np.zeros((0, hidden_dim), dtype=np.float32),
                np.zeros((0, 0), dtype=np.int32), 0)

    layers = np.array(layers, dtype=np.int32)
    hiddens = np.stack(hiddens)
    max_K = max(len(e) for e in experts)
    # Pad with -1 (not 0): 0 is a valid expert index, so a 0 sentinel would be
    # indistinguishable from a real top-0 routing and corrupt labels/eval when
    # K varies across records. Consumers filter out negative indices.
    experts_padded = np.full((len(experts), max_K), -1, dtype=np.int32)
    for i, e in enumerate(experts):
        experts_padded[i, :len(e)] = e

    return layers, hiddens, experts_padded, max_K


def load_multiple_routing_files(paths, hidden_dim, num_layers, seed=42):
    """Load multiple routing files and return token-shuffled concatenation.

    Each file is sliced into per-token blocks of ``num_layers`` records.
    Tokens from all files are shuffled together so the train/eval split
    draws from every file, preventing any single-source bias in evaluation.

    Returns (layers, hiddens, experts, K, n_tokens).
    """
    print(f"Loading {len(paths)} routing file(s)...")
    token_layers  = []   # list of (num_layers,) arrays
    token_hiddens = []   # list of (num_layers, hidden_dim) arrays
    token_experts = []   # list of (num_layers, K) arrays
    max_K_global  = 0

    for path in paths:
        layers, hiddens, experts, K = load_routing_data(path, hidden_dim=hidden_dim)
        n_records = len(layers)
        if n_records == 0:
            print(f"  WARNING: {Path(path).name}: no usable records (empty or "
                  "truncated); skipping.")
            continue
        if n_records % num_layers != 0:
            n_trunc = (n_records // num_layers) * num_layers
            print(f"  WARNING: {path}: {n_records} records not divisible by "
                  f"num_layers={num_layers}; truncating to {n_trunc}.")
            layers  = layers[:n_trunc]
            hiddens = hiddens[:n_trunc]
            experts = experts[:n_trunc]
            n_records = n_trunc

        n_tokens = n_records // num_layers
        if n_tokens == 0:
            # Fewer than one full token's worth of records after truncation:
            # contributes no tokens, so don't let its K inflate the global pad
            # width (which would force needless -1 padding on every other file).
            print(f"  WARNING: {Path(path).name}: <1 token after truncation; skipping.")
            continue
        max_K_global = max(max_K_global, K)

        for t in range(n_tokens):
            base = t * num_layers
            token_layers.append(layers[base:base + num_layers])
            token_hiddens.append(hiddens[base:base + num_layers])
            token_experts.append(experts[base:base + num_layers])

        print(f"  {Path(path).name}: {n_tokens} tokens ({n_records} records), K={K}")

    if not token_layers:
        print("ERROR: no usable routing records loaded from any file.")
        sys.exit(1)

    # Pad all expert arrays to global max_K so they can be stacked. Pad with -1
    # (not the np.pad default of 0): 0 is a valid expert index. Consumers filter
    # out negative indices.
    token_experts = [
        np.pad(e, ((0, 0), (0, max_K_global - e.shape[1])), constant_values=-1)
        if e.shape[1] < max_K_global else e
        for e in token_experts
    ]

    # Shuffle token order across files (keeps per-token layer ordering intact)
    rng = random.Random(seed)
    indices = list(range(len(token_layers)))
    rng.shuffle(indices)

    layers_out  = np.concatenate([token_layers[i]  for i in indices])
    hiddens_out = np.concatenate([token_hiddens[i] for i in indices])
    experts_out = np.concatenate([token_experts[i] for i in indices])

    n_tokens_total = len(indices)
    print(f"  Total: {n_tokens_total} tokens "
          f"({len(layers_out)} records), K={max_K_global}")
    return layers_out, hiddens_out, experts_out, max_K_global, n_tokens_total


def fp_threshold_loss(logits, targets, threshold=0.75, beta=5.0):
    """Loss that penalises false positives above a confidence threshold.

    Combines standard BCEWithLogitsLoss with a hinge penalty on non-top-4
    experts whose predicted logit exceeds the threshold logit t*.

      L = BCE(logits, targets)
        + beta * mean( relu(logit - t*) * (1 - target) )

    t* = log(threshold / (1 - threshold))  e.g. log(3) ≈ 1.099 for threshold=0.75

    The ReLU fires only when a *negative* expert is predicted above the threshold,
    so there is zero gradient on negatives that are already safely below it.
    beta controls the trade-off: higher beta → fewer FP above threshold, lower recall.
    """
    import math

    import torch.nn.functional as F
    t_star = math.log(threshold / (1.0 - threshold))
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    neg_mask = 1.0 - targets                       # 1 for non-top-4 experts
    hinge = F.relu(logits - t_star) * neg_mask
    return bce + beta * hinge.mean()


def eval_threshold_metrics(logits_np, E_test, threshold=0.75):
    """Compute precision, recall, and per-sample FP rate at a confidence threshold.

    Returns a dict with keys:
      precision  — of experts predicted ≥ threshold, fraction truly in top-4
      recall     — of true top-4 experts, fraction predicted ≥ threshold
      fp_rate    — fraction of samples that contain ≥ 1 false positive above threshold
      n_pred     — total number of experts predicted above threshold
    """
    probs = 1.0 / (1.0 + np.exp(-logits_np))   # sigmoid
    above = probs >= threshold                   # (N, num_experts) bool

    true_pos = 0
    false_pos = 0
    true_top4_total = 0
    true_pos_recall = 0
    samples_with_fp = 0

    for i in range(len(E_test)):
        actual = set(int(x) for x in E_test[i] if x >= 0)
        predicted_above = set(int(j) for j in range(above.shape[1]) if above[i, j])
        tp = len(predicted_above & actual)
        fp = len(predicted_above - actual)
        true_pos += tp
        false_pos += fp
        true_top4_total += len(actual)
        true_pos_recall += tp
        if fp > 0:
            samples_with_fp += 1

    n_pred = true_pos + false_pos
    precision = true_pos / n_pred if n_pred > 0 else 1.0
    recall    = true_pos_recall / true_top4_total if true_top4_total > 0 else 0.0
    fp_rate   = samples_with_fp / len(E_test) if len(E_test) > 0 else 0.0
    return dict(precision=precision, recall=recall, fp_rate=fp_rate, n_pred=n_pred)


def build_target_multilabel(expert_indices, num_experts):
    """Convert expert indices to multi-label binary targets."""
    N = len(expert_indices)
    targets = np.zeros((N, num_experts), dtype=np.float32)
    for i in range(N):
        for j in range(expert_indices.shape[1]):
            e = expert_indices[i, j]
            if e >= 0:  # skip -1 padding (0 is a valid expert index)
                targets[i, e] = 1.0
    return targets


def build_cross_layer_pairs(layers, hiddens, experts, num_layers):
    """Build Fate-style cross-layer training pairs: (h[l], experts[l+1]).

    Records are token-major: layers 0..num_layers-1 per token in sequence.
    For each token t and each layer l in 0..num_layers-2, we pair:
      input:  hiddens[t, l]      (gate input at layer l)
      target: experts[t, l+1]   (expert activations at layer l+1)
      label:  l+1                (target layer index for layer embedding)

    Returns (X, L, E) arrays with shape (N_tokens*(num_layers-1), ...).
    """
    n_total = len(layers)
    assert n_total % num_layers == 0, \
        f"Expected multiple of {num_layers} records, got {n_total}"
    n_tokens = n_total // num_layers

    X_list, L_list, E_list = [], [], []
    for t in range(n_tokens):
        base = t * num_layers
        for l in range(num_layers - 1):
            X_list.append(hiddens[base + l])       # h[l]
            L_list.append(layers[base + l + 1])    # target layer l+1
            E_list.append(experts[base + l + 1])   # experts at l+1

    return (np.stack(X_list),
            np.array(L_list, dtype=np.int32),
            np.stack(E_list))


def analyze_cosine_similarity(layers, hiddens, num_layers):
    """Measure cosine similarity between h[l] and h[l+1] (Fate §4.2).

    High similarity validates cross-layer prediction: if cos(h[l], h[l+1]) ≈ 1,
    then h[l] is a good substitute for h[l+1] as gate input.
    """
    n_total = len(layers)
    assert n_total % num_layers == 0
    n_tokens = n_total // num_layers

    layer_sims = np.zeros(num_layers - 1)
    layer_counts = np.zeros(num_layers - 1)

    for t in range(n_tokens):
        base = t * num_layers
        for l in range(num_layers - 1):
            h0 = hiddens[base + l].astype(np.float64)
            h1 = hiddens[base + l + 1].astype(np.float64)
            cos = np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-12)
            layer_sims[l] += cos
            layer_counts[l] += 1

    layer_sims /= np.maximum(layer_counts, 1)
    return layer_sims


def make_model(model_type, hidden_dim, num_experts, num_layers, hidden_size, device,
               dropout=0.1):
    """Construct the predictor model and move it to device."""
    import torch.nn as nn

    class ExpertPredictor(nn.Module):
        """Original 2-layer MLP, same-layer: h[l] -> experts[l]."""
        def __init__(self):
            super().__init__()
            self.layer_emb = nn.Embedding(num_layers, 32)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + 32, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts),
            )

        def forward(self, x, layer_ids):
            return self.net(torch.cat([x, self.layer_emb(layer_ids)], dim=-1))

    class FateLinearPredictor(nn.Module):
        """Single linear layer: h[l] -> experts[l] or experts[l+1] (cross)."""
        def __init__(self):
            super().__init__()
            self.layer_emb = nn.Embedding(num_layers, 64)
            self.proj = nn.Linear(hidden_dim + 64, num_experts)

        def forward(self, x, layer_ids):
            return self.proj(torch.cat([x, self.layer_emb(layer_ids)], dim=-1))

    class FateMLPPredictor(nn.Module):
        """One-hidden-layer MLP (cross-layer ablation): h[l] -> experts[l+1]."""
        def __init__(self):
            super().__init__()
            self.layer_emb = nn.Embedding(num_layers, 64)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts),
            )

        def forward(self, x, layer_ids):
            return self.net(torch.cat([x, self.layer_emb(layer_ids)], dim=-1))

    constructors = {
        'mlp':          ExpertPredictor,
        'linear':       FateLinearPredictor,
        'linear-cross': FateLinearPredictor,
        'mlp-cross':    FateMLPPredictor,
    }
    if model_type not in constructors:
        print(f"ERROR: unknown --model-type '{model_type}'")
        sys.exit(1)
    return constructors[model_type]().to(device)


def train_and_evaluate(layers, hiddens, experts, K, n_tokens,
                       hidden_size=256, epochs=20, lr=1e-3, K_pred=4,
                       model_type='mlp',
                       hidden_dim=HIDDEN_DIM,
                       num_experts=NUM_EXPERTS,
                       num_layers=NUM_LAYERS,
                       load_model_path=None,
                       save_model_path=None,
                       threshold=0.75,
                       fp_penalty_weight=5.0,
                       skip_layers=None,
                       dropout=0.1):
    """Train expert predictor and evaluate prediction accuracy.

    Accepts pre-loaded numpy arrays (from load_multiple_routing_files).
    Supports loading a checkpoint for finetuning via load_model_path.

    The loss combines standard BCE with a hinge penalty on false positives above
    `threshold` (default 0.75). Any expert the model predicts with probability
    >= threshold must be in the actual top-4; `fp_penalty_weight` (beta) controls
    how hard the model is pushed to suppress confident-but-wrong predictions.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    cross_layer = model_type.endswith('-cross')

    print(f"  {len(layers)} samples, K={K}, layers 0-{layers.max()}")
    print(f"  Hidden state shape: {hiddens.shape}")
    print(f"  Hidden RMS: {np.sqrt(np.mean(hiddens**2)):.4f}")
    print(f"  Model type: {model_type}")

    # Temporal locality baseline
    print("\n=== Temporal Locality Baseline ===")
    prev_experts = {}
    temporal_hits = 0
    temporal_total = 0
    for i in range(len(layers)):
        li = int(layers[i])
        ei = set(int(x) for x in experts[i] if x >= 0)  # drop -1 padding
        if li in prev_experts:
            temporal_hits += len(ei & prev_experts[li])
            temporal_total += len(ei)  # real experts only, not padded K
        prev_experts[li] = ei
    if temporal_total > 0:
        print(f"  Temporal hit rate: {temporal_hits}/{temporal_total} = "
              f"{temporal_hits/temporal_total*100:.1f}%")

    # Cosine similarity analysis (Fate §4.2)
    print("\n=== Adjacent-Layer Cosine Similarity (Fate §4.2) ===")
    layer_sims = analyze_cosine_similarity(layers, hiddens, num_layers)
    avg_sim = layer_sims.mean()
    print(f"  Mean cos(h[l], h[l+1]): {avg_sim:.4f}  Min: {layer_sims.min():.4f}")
    print("  Per-layer: " +
          "  ".join(f"L{l}:{layer_sims[l]:.3f}" for l in range(0, num_layers - 1, 5)))
    if avg_sim > 0.7:
        print("  ✓ High similarity confirms cross-layer prediction should work well.")
    else:
        print("  ✗ Low similarity — cross-layer approach may not improve much.")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nTraining on: {device}")

    model = make_model(model_type, hidden_dim, num_experts, num_layers, hidden_size, device,
                       dropout=dropout)

    # Load checkpoint for finetuning
    if load_model_path is not None:
        print(f"Loading model checkpoint from {load_model_path} ...")
        ckpt = torch.load(load_model_path, map_location=device)
        ckpt_cfg = ckpt.get('config', {})
        if (ckpt_cfg.get('model_type') not in (None, model_type) or
                ckpt_cfg.get('hidden_dim') not in (None, hidden_dim) or
                ckpt_cfg.get('num_experts') not in (None, num_experts) or
                ckpt_cfg.get('num_layers') not in (None, num_layers)):
            print(f"  WARNING: checkpoint config {ckpt_cfg} may not match current settings. "
                  "Attempting to load anyway.")
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded: {ckpt_cfg}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model ({model_type}): {param_count:,} parameters ({param_count*4/1024/1024:.2f} MB)")

    # Build train / test split (80/20 by token to avoid data leakage)
    split_tokens = int(n_tokens * 0.8)
    skip_set = set(skip_layers) if skip_layers else set()
    if skip_set:
        print(f"  Skipping layers: {sorted(skip_set)}")

    if cross_layer:
        X_all, L_all, E_all = build_cross_layer_pairs(layers, hiddens, experts, num_layers)
        pairs_per_token = num_layers - 1
        split = split_tokens * pairs_per_token
        train_idx = np.arange(split)
        test_idx  = np.arange(split, len(X_all))

        if skip_set:
            train_idx = train_idx[~np.isin(L_all[train_idx], list(skip_set))]
            test_idx  = test_idx[ ~np.isin(L_all[test_idx],  list(skip_set))]

        X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
        L_train = torch.tensor(L_all[train_idx], dtype=torch.long)
        Y_train = torch.tensor(build_target_multilabel(E_all[train_idx], num_experts),
                               dtype=torch.float32)
        X_test      = torch.tensor(X_all[test_idx], dtype=torch.float32)
        L_test      = torch.tensor(L_all[test_idx], dtype=torch.long)
        E_test      = E_all[test_idx]
        layers_test = L_all[test_idx]
    else:
        targets = build_target_multilabel(experts, num_experts)
        split = split_tokens * num_layers
        train_idx = np.arange(split)
        test_idx  = np.arange(split, len(layers))

        if skip_set:
            train_idx = train_idx[~np.isin(layers[train_idx], list(skip_set))]
            test_idx  = test_idx[ ~np.isin(layers[test_idx],  list(skip_set))]

        X_train = torch.tensor(hiddens[train_idx], dtype=torch.float32)
        L_train = torch.tensor(layers[train_idx],  dtype=torch.long)
        Y_train = torch.tensor(targets[train_idx], dtype=torch.float32)
        X_test      = torch.tensor(hiddens[test_idx], dtype=torch.float32)
        L_test      = torch.tensor(layers[test_idx],  dtype=torch.long)
        E_test      = experts[test_idx]
        layers_test = layers[test_idx]

    if len(train_idx) == 0 or len(test_idx) == 0:
        print(f"ERROR: empty split (train={len(train_idx)}, test={len(test_idx)} "
              f"samples). Need more data, fewer --skip-layers, or a different "
              f"train/test ratio (n_tokens={n_tokens}).")
        sys.exit(1)

    train_ds = TensorDataset(X_train, L_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import math as _math
    t_star = _math.log(threshold / (1.0 - threshold))
    print(f"Loss: BCE + {fp_penalty_weight}×hinge  "
          f"(threshold={threshold}, t*={t_star:.3f})\n"
          f"  → penalises non-top-4 experts predicted above p={threshold}")

    print(f"\nTrain: {len(train_idx)} samples, Test: {len(test_idx)} samples"
          + (f"  (layers {sorted(skip_set)} excluded)" if skip_set else ""))
    print(f"Training for {epochs} epochs...\n")

    def eval_hit_rate(logits_np, E, k):
        preds = np.argsort(-logits_np, axis=1)[:, :k]
        hits = 0
        total = 0
        for i in range(len(E)):
            actual = set(int(x) for x in E[i] if x >= 0)  # drop -1 padding
            hits += len(actual & set(preds[i].tolist()))
            total += len(actual)  # real experts only, not padded K
        return hits, total

    def run_eval():
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(X_test), 512):
                chunks.append(model(X_test[i:i+512].to(device),
                                    L_test[i:i+512].to(device)).cpu())
        return torch.cat(chunks, dim=0).numpy()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, lb, yb in train_dl:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            loss = fp_threshold_loss(model(xb, lb), yb,
                                     threshold=threshold,
                                     beta=fp_penalty_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logits = run_eval()
            hits, total = eval_hit_rate(logits, E_test, K_pred)
            acc = hits / total * 100 if total > 0 else 0
            avg_loss = total_loss / len(train_idx)
            tm = eval_threshold_metrics(logits, E_test, threshold)
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  "
                  f"top-{K_pred} hit={acc:.1f}%  "
                  f"prec@{threshold:.0%}={tm['precision']*100:.1f}%  "
                  f"fp_rate={tm['fp_rate']*100:.1f}%  "
                  f"recall@{threshold:.0%}={tm['recall']*100:.1f}%")

    # --- Save checkpoint ---
    if save_model_path is not None:
        ckpt = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_type': model_type,
                'hidden_dim': hidden_dim,
                'num_experts': num_experts,
                'num_layers': num_layers,
                'hidden_size': hidden_size,
            },
        }
        torch.save(ckpt, save_model_path)
        print(f"\nModel saved to {save_model_path}")

    # --- Final detailed evaluation ---
    print("\n=== Final Evaluation ===")
    test_logits = run_eval()
    pred_indices = np.argsort(-test_logits, axis=1)[:, :K_pred]

    # --- Threshold precision / FP report (primary objective) ---
    tm = eval_threshold_metrics(test_logits, E_test, threshold)
    print(f"\n--- False-Positive Suppression @ threshold={threshold:.0%} ---")
    print(f"  Precision@{threshold:.0%} : {tm['precision']*100:.1f}%  "
          f"({tm['n_pred']} experts predicted above threshold)")
    print(f"  Recall@{threshold:.0%}    : {tm['recall']*100:.1f}%  "
          f"(of true top-4 experts, fraction above threshold)")
    print(f"  FP rate         : {tm['fp_rate']*100:.1f}%  "
          f"(samples with ≥1 false positive above threshold)")
    if tm['precision'] >= 0.999:
        print(f"  ✓ Zero false positives above {threshold:.0%} threshold.")
    else:
        print(f"  ✗ {(1-tm['precision'])*100:.1f}% of high-confidence predictions are wrong. "
              f"Consider increasing --fp-penalty-weight (current: {fp_penalty_weight}).")

    layer_hits  = np.zeros(num_layers)
    layer_total = np.zeros(num_layers)
    for i in range(len(E_test)):
        li = int(layers_test[i])
        actual = set(int(x) for x in E_test[i] if x >= 0)  # drop -1 padding
        layer_hits[li]  += len(actual & set(pred_indices[i].tolist()))
        layer_total[li] += len(actual)  # real experts only, not padded K

    print(f"\nPer-layer top-{K_pred} hit rates:")
    for li in range(num_layers):
        if layer_total[li] > 0:
            rate = layer_hits[li] / layer_total[li] * 100
            print(f"  Layer {li:2d}: {rate:5.1f}%  {'#' * int(rate / 2)}")

    overall_hits  = int(layer_hits.sum())
    overall_total = int(layer_total.sum())
    overall_rate  = overall_hits / overall_total * 100 if overall_total > 0 else 0
    print(f"\n  OVERALL top-{K_pred} hit: {overall_rate:.1f}% ({overall_hits}/{overall_total})")
    if temporal_total > 0:
        print(f"  Temporal baseline: {temporal_hits/temporal_total*100:.1f}%")
    if cross_layer:
        print("  Note: cross-layer hit rates are for predicted layer l+1 given h[l].")

    for k in [4, 6, 8, 12, 16]:
        h, _ = eval_hit_rate(test_logits, E_test, k)
        print(f"  Top-{k:2d} predictions: {h/overall_total*100:.1f}% hit rate")

    # Potential speedup estimate
    print("\n=== Speedup Estimate ===")
    baseline_io_ms = 2.4
    for hr in [overall_rate, 50, 60, 70, 80]:
        miss_rate    = 1.0 - hr / 100.0
        miss_io_ms   = baseline_io_ms * (miss_rate * K / K) ** 0.7
        new_io_ms    = 0.1 + miss_io_ms
        savings_ms   = baseline_io_ms - new_io_ms
        new_total    = 4.0 - savings_ms
        new_toks     = 1000.0 / (new_total * 60) if new_total > 0 else 0
        label        = "MEASURED" if hr == overall_rate else "target"
        print(f"  {hr:.0f}% hits ({label}): "
              f"expert_io {new_io_ms:.1f}ms → {new_toks:.1f} tok/s "
              f"(saves {savings_ms:.1f}ms/layer)")

    return model, overall_rate


if __name__ == '__main__':
    import argparse

    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_files', nargs='*',
                        help='One or more binary routing data files. '
                             'Omit when using --data-dir.')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='Load all *.bin files from this directory '
                             '(combined with any explicit data_files).')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=256,
                        help='Hidden size for MLP models (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model-type', default='mlp',
                        choices=['mlp', 'linear', 'linear-cross', 'mlp-cross'],
                        help='Predictor architecture (default: mlp). '
                             'linear-cross = Fate cross-layer approach (recommended).')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help=f'Hidden state dimension (default: {HIDDEN_DIM}). '
                             'Use 2048 for 35B, 3072 for 122B.')
    parser.add_argument('--num-experts', type=int, default=None,
                        help=f'Number of experts (default: {NUM_EXPERTS}). '
                             'Use 256 for 35B/122B.')
    parser.add_argument('--num-layers', type=int, default=None,
                        help=f'Number of layers (default: {NUM_LAYERS}). '
                             'Use 40 for 35B, 48 for 122B.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for token shuffle (default: 42).')
    parser.add_argument('--save-model', metavar='PATH',
                        help='Save trained model checkpoint to this path.')
    parser.add_argument('--load-model', metavar='PATH',
                        help='Load existing checkpoint and finetune with new data.')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Confidence threshold for false-positive suppression '
                             '(default: 0.75). Any expert predicted above this '
                             'probability must be in the actual top-4.')
    parser.add_argument('--fp-penalty-weight', type=float, default=5.0,
                        metavar='BETA',
                        help='Weight on the hinge penalty for false positives above '
                             '--threshold (default: 5.0). Higher → fewer FP above '
                             'threshold but lower overall recall.')
    parser.add_argument('--skip-layers', type=int, nargs='*', default=[0, 1, 2],
                        metavar='N',
                        help='Layer indices to exclude from training and evaluation '
                             '(default: 0 1 2). Pass no arguments to include all layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate applied after each hidden layer in MLP models '
                             '(default: 0.1). Set to 0 to disable.')
    args = parser.parse_args()

    # --threshold feeds log(threshold / (1 - threshold)); guard the domain so
    # values <= 0 or >= 1 fail fast instead of crashing mid-training.
    if not 0.0 < args.threshold < 1.0:
        parser.error(f"--threshold must be in the open interval (0, 1), got {args.threshold}")

    # Resolve model dimensions (CLI overrides > defaults)
    hidden_dim   = args.hidden_dim   if args.hidden_dim   is not None else HIDDEN_DIM
    num_experts  = args.num_experts  if args.num_experts  is not None else NUM_EXPERTS
    num_layers   = args.num_layers   if args.num_layers   is not None else NUM_LAYERS

    # Collect input files
    paths = [Path(f) for f in args.data_files]
    if args.data_dir:
        dir_files = sorted(Path(args.data_dir).glob('*.bin'))
        if not dir_files:
            print(f"ERROR: no *.bin files found in {args.data_dir}")
            sys.exit(1)
        paths.extend(dir_files)
    if not paths:
        parser.error("Provide at least one data file or use --data-dir.")

    # Load and shuffle data
    layers, hiddens, experts, K, n_tokens = load_multiple_routing_files(
        paths, hidden_dim=hidden_dim, num_layers=num_layers, seed=args.seed)

    train_and_evaluate(
        layers, hiddens, experts, K, n_tokens,
        hidden_size=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model_type,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        load_model_path=args.load_model,
        save_model_path=args.save_model,
        threshold=args.threshold,
        fp_penalty_weight=args.fp_penalty_weight,
        skip_layers=args.skip_layers,
        dropout=args.dropout,
    )
