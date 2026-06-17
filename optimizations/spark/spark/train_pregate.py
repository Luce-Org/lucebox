#!/usr/bin/env python3
"""(Research) Train a per-layer pre-gate predictor from a routing trace.

A pre-gate predicts a layer's selected experts from the *block input* hidden,
i.e. one step early, which is what you need to prefetch experts and fuse the
attention+FFN graphs (Pre-gated MoE, arXiv:2308.12066).

Trains a predictor per layer from the trace captured by the engine
(`DFLASH_LAGUNA_PREGATE_TRACE`, fixed-size records: int16 layer, int16 n_sel,
int32[8] selected, float32[n_embd] hidden) and reports recall@K: of the experts
a layer actually selects, how many are in the predicted top-K. High recall@K at
small K means prefetch+fusion is viable.

Loss (`--loss`): recall@K is a ranking metric, so the predictor is trained with a
ranking-aware objective by default rather than per-expert binary cross-entropy.
  - `listnet`  listwise: cross-entropy between softmax(scores) and the normalized
    selected-expert distribution (ListNet, Cao et al. 2007).
  - `ranknet`  pairwise logistic on every (selected, unselected) score pair, so
    every selected expert is pushed above every unselected one (RankNet, Burges
    et al. 2005).
  - `bce`      the original per-expert binary cross-entropy baseline.
A ranking objective optimizes the *order* of the expert scores, which is exactly
what top-K prefetch consumes; BCE optimizes each expert independently and is only
a surrogate for it (ranking-aware pre-gating, arXiv:2511.10676). `--loss all`
trains every objective on identical splits/init and prints a recall@K comparison.

Result on Laguna-XS.2 (RTX 3090, Claude Code traces) with BCE: linear ~50% / MLP
~53% recall@8 -- a richer model barely helps, because the pre-gate sees the
pre-attention hidden while the router decides on the post-attention hidden. That
information gap caps a fitted predictor; a ranking loss aligns the objective with
recall@K but does not create information the input lacks, so the headroom over
BCE is bounded. Reaching high recall needs fine-tuning the model's gate, not a
bigger predictor or a different loss alone. Re-measure with `--loss all`; see
RESULTS.md.

    python -m spark.train_pregate --trace pregate_trace.bin --model mlp
    python -m spark.train_pregate --trace pregate_trace.bin --loss all
    python -m spark.train_pregate --self-test        # synthetic, no trace/GPU
"""
import argparse

import numpy as np

LOSSES = ("bce", "listnet", "ranknet")


def build_net(model, n_embd, n_expert, dev):
    import torch.nn as nn
    if model == "linear":
        return nn.Linear(n_embd, n_expert).to(dev)
    return nn.Sequential(nn.Linear(n_embd, 512), nn.GELU(), nn.Linear(512, n_expert)).to(dev)


def loss_listnet(scores, target):
    """Top-1 ListNet: CE between softmax(scores) and the normalized relevance."""
    import torch
    tgt = target / target.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return -(tgt * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()


def loss_ranknet(scores, target, sel):
    """Pairwise RankNet logistic loss over (selected, unselected) score pairs.

    `sel` is the padded (N, 8) selected-expert index matrix (-1 = empty slot);
    `target` is the (N, E) multi-hot of the same selection.
    """
    import torch.nn.functional as F
    valid = (sel >= 0).unsqueeze(2)                       # (N, P, 1)
    pos_scores = scores.gather(1, sel.clamp(min=0))       # (N, P)
    neg = (target == 0).unsqueeze(1)                      # (N, 1, E)
    diff = scores.unsqueeze(1) - pos_scores.unsqueeze(2)  # (N, P, E): s_neg - s_pos
    weight = valid & neg                                  # count only pos-vs-neg pairs
    return (F.softplus(diff) * weight).sum() / weight.sum().clamp(min=1)


def step_loss(name, out, Ytr, Str):
    import torch.nn as nn
    if name == "bce":
        return nn.functional.binary_cross_entropy_with_logits(out, Ytr)
    if name == "listnet":
        return loss_listnet(out, Ytr)
    if name == "ranknet":
        return loss_ranknet(out, Ytr, Str)
    raise ValueError(name)


def recall_at_k(net, Xte, Yte, ks=(8, 16, 24)):
    import torch
    with torch.no_grad():
        order = torch.argsort(net(Xte), dim=1, descending=True)
        den = Yte.sum(1).clamp(min=1)
        return {k: (100 * (Yte.gather(1, order[:, :k]).sum(1) / den).mean()).item() for k in ks}


def train_eval(model, name, Xtr, Ytr, Str, Xte, Yte, n_expert, epochs, lr, seed):
    import torch
    torch.manual_seed(seed)
    net = build_net(model, Xtr.shape[1], n_expert, Xtr.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        step_loss(name, net(Xtr), Ytr, Str).backward()
        opt.step()
    return recall_at_k(net, Xte, Yte)


def to_multihot(sel, n_expert, dev):
    import torch
    Y = torch.zeros(len(sel), n_expert, device=dev)
    for j in range(sel.shape[1]):
        c = torch.tensor(sel[:, j].astype(np.int64), device=dev)
        m = c >= 0
        Y[torch.arange(len(sel), device=dev)[m], c[m]] = 1.0
    return Y


def report(model, agg):
    for name in LOSSES:
        d = agg.get(name)
        if not d or not d[8]:
            continue
        print(f"=== {model} pre-gate recall@K [{name}] (held-out, mean over layers) ===")
        for k in sorted(d):
            print(f"  recall@{k:2d}: {np.mean(d[k]):5.1f}%  (worst layer {min(d[k]):.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace")
    ap.add_argument("--n-embd", type=int, default=2048)
    ap.add_argument("--n-expert", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=40)
    ap.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--loss", choices=[*LOSSES, "all"], default="listnet")
    ap.add_argument("--max-per-layer", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="run a synthetic separable-data check (no trace/GPU needed)")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args))
    if not args.trace:
        ap.error("--trace is required (or pass --self-test)")

    import torch
    torch.manual_seed(args.seed)
    H, E = args.n_embd, args.n_expert
    losses = list(LOSSES) if args.loss == "all" else [args.loss]
    dt = np.dtype([("layer", "<i2"), ("nsel", "<i2"), ("sel", "<i4", (8,)), ("hid", "<f4", (H,))])
    arr = np.fromfile(args.trace, dtype=dt)
    print(f"records={len(arr)}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    agg = {name: {8: [], 16: [], 24: []} for name in losses}
    for L in range(1, args.n_layer):
        idx = np.where(arr["layer"] == L)[0][:args.max_per_layer]
        if len(idx) < 400:
            continue
        X = torch.tensor(np.ascontiguousarray(arr["hid"][idx]), device=dev)
        s = arr["sel"][idx]
        Y = to_multihot(s, E, dev)
        sel_t = torch.tensor(s.astype(np.int64), device=dev)
        ntr = int(len(idx) * 0.8)
        Xtr, Xte, Ytr, Yte, Str = X[:ntr], X[ntr:], Y[:ntr], Y[ntr:], sel_t[:ntr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        for name in losses:
            rec = train_eval(args.model, name, Xtr, Ytr, Str, Xte, Yte, E,
                             args.epochs, args.lr, args.seed + L)
            for k in agg[name]:
                agg[name][k].append(rec[k])

    report(args.model, agg)


def self_test(args):
    """Synthetic separable trace: a fixed linear gate decides the top-8 experts,
    so a fitted predictor must reach high recall. Validates every loss path,
    gradient flow, and the recall@K eval without the real 1.2 GB trace or a GPU.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    H, E, N, topk = 64, 32, 4000, 8
    W = torch.randn(E, H, device=dev)
    X = torch.randn(N, H, device=dev)
    sel = torch.argsort(X @ W.t(), dim=1, descending=True)[:, :topk]  # (N, 8) ground truth
    Y = torch.zeros(N, E, device=dev).scatter_(1, sel, 1.0)
    ntr = int(N * 0.8)
    Xtr, Xte, Ytr, Yte, Str = X[:ntr], X[ntr:], Y[:ntr], Y[ntr:], sel[:ntr]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    failures = []
    for name in LOSSES:
        rec = train_eval("mlp", name, Xtr, Ytr, Str, Xte, Yte, E,
                         args.epochs, args.lr, seed=1)
        ok = all(np.isfinite(v) for v in rec.values()) and rec[8] > 70.0
        print(f"  {name:8s} recall@8={rec[8]:5.1f}%  recall@16={rec[16]:5.1f}%  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    print("self-test PASS" if not failures else f"self-test FAIL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    main()
