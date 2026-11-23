#!/usr/bin/env python3
"""Step 3: validate a placement profile on HELD-OUT chunks.

Drives the daemon over held-out corpus chunks and reports mean decode tok/s and
the cold-hit rate from the per-token profiler. Compare runs with/without
`--hotness` (uniform vs calibrated) and with/without the expert cache to see the
ladder in RESULTS.md.

    # all-GPU baseline
    python -m spark.validate --bin ... --gguf ... --tok ... \
        --corpus corpus/test.jsonl --budget-pct 0

    # calibrated 60% + bounded expert cache
    python -m spark.validate --bin ... --gguf ... --tok ... \
        --corpus corpus/test.jsonl --budget-pct 60 \
        --hotness spark_profile.csv --cache-slots 32
"""
import argparse
import json
import os
import re
import statistics
import struct
from pathlib import Path

from tokenizers import Tokenizer

try:
    from ._daemon import Daemon
except ImportError:  # allow direct `python validate.py`
    from _daemon import Daemon


def write_counted_i32(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<%di" % len(ids), *(int(t) for t in ids)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--corpus", required=True, help="test.jsonl (held-out)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--budget-pct", type=int, default=60, help="0 = all-GPU")
    ap.add_argument("--hotness", default="", help="placement CSV (empty = uniform)")
    ap.add_argument("--cache-slots", type=int, default=0, help="bounded expert cache slots/layer")
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--n-chunks", type=int, default=30)
    ap.add_argument("--max-tok", type=int, default=1024)
    ap.add_argument("--natural", action="store_true", help="stop at EOS (no forced length)")
    ap.add_argument("--ready-timeout", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=120)
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tok)
    chunks = [json.loads(l) for l in open(args.corpus)][:args.n_chunks]

    env = dict(os.environ)
    env["LUCE_LAGUNA_PROFILE"] = "1"
    if not args.natural:
        env["LUCE_IGNORE_EOS"] = "1"
    if args.budget_pct > 0:
        env["LUCE_EXPERT_BUDGET_PCT"] = str(args.budget_pct)
    if args.hotness:
        env["LUCE_LAGUNA_HOTNESS"] = args.hotness
    if args.cache_slots > 0:
        env["LUCE_LAGUNA_GPU_REMAP"] = "1"
        env["LUCE_LAGUNA_EXPERT_CACHE"] = "1"
        env["LUCE_LAGUNA_CACHE_SLOTS"] = str(args.cache_slots)

    placement = []
    daemon = Daemon([args.bin, args.gguf, "--max-ctx", str(args.max_ctx)], env, capture_stderr=True)
    daemon.wait_ready(
        timeout=args.ready_timeout,
        on_line=lambda l: placement.append(l.rstrip())
        if ("storage ready" in l or "placement result" in l) else None)

    pp = Path("/tmp/spark_val_chunk.bin")
    op = Path("/tmp/spark_val_out.bin")
    toks = []
    for c in chunks:
        ids = tk.encode(c).ids[:args.max_tok]
        if len(ids) < 8:
            continue
        write_counted_i32(pp, ids)
        reply = daemon.request(f"generate {pp} {args.n_gen} {op}", timeout=args.gen_timeout)
        if reply is None:
            print("daemon stalled/closed; stopping early")
            break
        if reply.startswith("ok "):
            m = re.search(r"decode_tok_s=([0-9.]+)", reply)
            if m:
                toks.append(float(m.group(1)))

    daemon.quit()

    for p in placement:
        print("PLACEMENT:", p.strip())
    label = f"budget={args.budget_pct} hotness={'yes' if args.hotness else 'no'} cache={args.cache_slots}"
    print(f"\n===== VALIDATION  {label} =====")
    if toks:
        print(f"decode tok/s: mean={statistics.mean(toks):.1f} median={statistics.median(toks):.1f} "
              f"over {len(toks)} chunks")
    colds = [float(m.group(1)) for l in daemon.stderr_lines
             for m in [re.search(r"cold_experts/tok=([0-9.]+)", l)] if m]
    if colds:
        print(f"cold_experts/tok: mean={statistics.mean(colds):.2f} max={max(colds):.2f}")


if __name__ == "__main__":
    main()
