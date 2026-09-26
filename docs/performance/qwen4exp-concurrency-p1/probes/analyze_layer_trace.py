#!/usr/bin/env python3
"""Compare row 0 of batched capture against solo captures from the P1 probe."""
import glob
import os
import struct
import sys


def read(path, fmt):
    with open(path, "rb") as f:
        data = f.read()
    width = struct.calcsize(fmt)
    if len(data) % width:
        raise ValueError(f"bad byte count in {path}: {len(data)}")
    return list(struct.unpack("=" + fmt * (len(data) // width), data))


def main(root, prefix):
    labels = ("hcmix", "att", "ffnxn", "fmix", "mid", "moe", "res", "ple")
    def solo_path(arm, layer, label):
        for slot in range(4):
            path = f"{root}/q4trace_{arm}_solo_s{slot}_step0_L{layer:02d}.{label}_row0.bin"
            if os.path.exists(path): return path
        return None
    for label in labels:
        first = None
        max_seen = (0.0, None)
        set_changes = []
        for layer in range(48):
            stem_s = solo_path(prefix, layer, label)
            stem_b = f"{root}/q4trace_{prefix}_batch_step0_L{layer:02d}.{label}_row0.bin"
            if not (stem_s and os.path.exists(stem_b)):
                continue
            if label == "mid":
                a, b = read(stem_s, "i"), read(stem_b, "i")
                sa, sb = set(a), set(b)
                if sa != sb:
                    set_changes.append((layer, sorted(sa - sb), sorted(sb - sa)))
                continue
            a, b = read(stem_s, "f"), read(stem_b, "f")
            if len(a) != len(b):
                print(f"{prefix} L{layer:02d}.{label}: shape-length mismatch {len(a)} vs {len(b)}")
                continue
            delta = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
            if delta > max_seen[0]:
                max_seen = (delta, layer)
            if first is None and delta > 1e-3:
                first = (layer, delta)
        if label == "mid":
            print(f"{prefix} routed-expert set first-change={set_changes[0] if set_changes else 'none'}; layers-changed={len(set_changes)}")
        else:
            print(f"{prefix} {label}: first >1e-3={first}; max={max_seen}")

    # Attention/control comparison: solo stable vs solo exact and exact/no-trim.
    if prefix == "stable":
        for other in ("exact", "exact0"):
            print(f"solo {prefix} vs {other}:")
            for label in ("hcmix", "att", "ffnxn", "fmix", "moe", "res"):
                first = None
                max_seen = (0.0, None)
                for layer in range(48):
                    a_path = solo_path(prefix, layer, label)
                    b_path = f"{root}/q4trace_{other}_solo_s0_step0_L{layer:02d}.{label}_row0.bin"
                    if not a_path:
                        continue
                    if not os.path.exists(b_path):
                        continue
                    a, b = read(a_path, "f"), read(b_path, "f")
                    d = max((abs(x-y) for x, y in zip(a,b)), default=0.0)
                    if d > max_seen[0]: max_seen = (d, layer)
                    if first is None and d > 1e-3: first = (layer,d)
                print(f"  {label}: first >1e-3={first}; max={max_seen}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp", sys.argv[2])
