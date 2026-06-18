#!/usr/bin/env python3
"""Position-weighted aggregate of per-prompt [topk-calib] tables.

  python3 calib/aggregate.py calib/out/calib.log
"""
import re, sys
from pathlib import Path

def main():
    txt = Path(sys.argv[1]).read_text()
    blocks = re.split(r'\[topk-calib\] positions=', txt)[1:]
    set_metrics  = ["greedy (target argmax)", "sampling top_k=8",
                    "sampling top_k=20", "sampling top_p=0.95 nucleus"]
    set_agg  = {m: {} for m in set_metrics}     # metric -> {M: weighted%}
    mass_agg = {}                                # M -> weighted%
    totpos = 0; nblk = 0
    for b in blocks:
        pos = int(re.match(r'(\d+)', b).group(1)); totpos += pos; nblk += 1
        cur = None
        for line in b.splitlines():
            if "set-coverage" in line:
                for m in set_metrics:
                    if m in line: cur = m
            if "MASS coverage" in line:
                cur = "__mass__"
            cm = re.search(r'coverage@M=(\d+)\s*:\s*([\d.]+)%', line)
            if cm and cur and cur != "__mass__":
                M, v = int(cm.group(1)), float(cm.group(2))
                set_agg[cur][M] = set_agg[cur].get(M, 0.0) + pos * v
            mm = re.search(r'mass@M=(\d+)\s*:\s*([\d.]+)%', line)
            if mm:
                M, v = int(mm.group(1)), float(mm.group(2))
                mass_agg[M] = mass_agg.get(M, 0.0) + pos * v
    print(f"prompts={nblk}  total_positions={totpos}\n")
    Ms = sorted({M for d in set_agg.values() for M in d} | set(mass_agg))
    hdr = "M".ljust(8) + "".join(f"{M:>9}" for M in Ms)
    print(hdr)
    for m in set_metrics:
        row = m[:26].ljust(8) + "".join(
            f"{set_agg[m].get(M,0)/totpos:9.1f}" for M in Ms)
        print(row)
    if mass_agg:
        print("\n-- top_p=0.95 nucleus MASS coverage (mean %) --")
        print("mass".ljust(8) + "".join(f"{mass_agg.get(M,0)/totpos:9.2f}" for M in Ms))

if __name__ == "__main__":
    main()
