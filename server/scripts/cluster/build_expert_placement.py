#!/usr/bin/env python3
"""Build a cluster expert placement JSON from a routing hotness CSV.

Input: the CSV written by MoeHybridRoutingStats::save_csv
(DFLASH_DS4_ROUTING_STATS_OUT):

    # hotness table: n_layer=43 n_expert=256 n_expert_used=6
    # format: one row per layer, columns are expert activation counts (expert 0..N-1)
    12,0,5,...            <- layer 0
    ...

Output: the JSON consumed by ClusterExpertPlacement::load_json
(server/src/cluster/cluster_expert_placement.cpp) and passed to
--cluster-expert-placement <file.json>:

    {"n_ranks":N,"n_layer":L,"n_expert":E,"n_expert_used":K,
     "replicate_hot":R,"owner":[[...E ints...] x L]}

owner[layer][expert] is the owning rank or -1 (replicated on every rank; the
rank that evaluates expert e for token slot t is (e + t) % N).

The algorithms mirror the C++ builders exactly (uniform: e % N; balanced:
hottest replicate_hot experts replicated, the rest sorted by count descending
and greedily assigned to the least-loaded rank below ceil(remaining/N) owned
experts, ties to the lowest rank) so a JSON produced here hashes identically
to a placement built in-process from the same CSV.

stdlib only.
"""

import argparse
import json
import re
import sys


def load_hotness_csv(path):
    n_layer = n_expert = n_expert_used = None
    rows = []
    header_re = re.compile(r"#\s*hotness table:\s*n_layer=(\d+)\s+n_expert=(\d+)\s+n_expert_used=(\d+)")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = header_re.match(line)
                if m:
                    n_layer, n_expert, n_expert_used = (int(m.group(i)) for i in (1, 2, 3))
                continue
            rows.append([int(x) for x in line.split(",") if x.strip() != ""])
    if n_layer is None:
        raise ValueError("missing '# hotness table:' header")
    if len(rows) != n_layer:
        raise ValueError("expected %d layer rows, found %d" % (n_layer, len(rows)))
    for il, row in enumerate(rows):
        if len(row) != n_expert:
            raise ValueError("layer %d has %d columns, expected %d" % (il, len(row), n_expert))
    return n_layer, n_expert, n_expert_used, rows


def ranked_experts(counts):
    # count descending, expert id ascending (stable, same as C++ ranked_experts)
    return sorted(range(len(counts)), key=lambda e: (-counts[e], e))


def build_uniform(n_ranks, n_layer, n_expert, rows, replicate_hot):
    owner = []
    for il in range(n_layer):
        row = [e % n_ranks for e in range(n_expert)]
        if replicate_hot > 0:
            for e in ranked_experts(rows[il])[:replicate_hot]:
                row[e] = -1
        owner.append(row)
    return owner


def build_balanced(n_ranks, n_layer, n_expert, rows, replicate_hot):
    owner = []
    for il in range(n_layer):
        counts = rows[il]
        ranked = ranked_experts(counts)
        row = [0] * n_expert
        for e in ranked[:replicate_hot]:
            row[e] = -1
        remaining = n_expert - replicate_hot
        cap = -(-remaining // n_ranks)  # ceil
        load = [0] * n_ranks
        owned = [0] * n_ranks
        for e in ranked[replicate_hot:]:
            best = -1
            for r in range(n_ranks):
                if owned[r] >= cap:
                    continue
                if best < 0 or load[r] < load[best]:
                    best = r
            if best < 0:
                raise RuntimeError("no rank below cap at layer %d" % il)
            row[e] = best
            load[best] += counts[e]
            owned[best] += 1
        owner.append(row)
    return owner


def build_all_on_rank(rank, n_ranks, n_layer, n_expert):
    # Diagnostic: one rank owns every expert, the others hold empty shards.
    # The cluster must then reproduce the single-node output exactly.
    if rank < 0 or rank >= n_ranks:
        raise ValueError("--all-on-rank must be in [0, ranks)")
    return [[rank] * n_expert for _ in range(n_layer)]


def validate(owner, n_ranks, n_expert, allow_empty=False):
    for il, row in enumerate(owner):
        replicated = sum(1 for o in row if o == -1)
        owned = [0] * n_ranks
        for e, o in enumerate(row):
            if o == -1:
                continue
            if o < 0 or o >= n_ranks:
                raise ValueError("layer %d expert %d owner %d out of range" % (il, e, o))
            owned[o] += 1
        if n_expert - replicated >= n_ranks and min(owned) == 0:
            if allow_empty:
                print("warning: layer %d: rank %d owns no expert (diagnostic placement)"
                      % (il, owned.index(0)), file=sys.stderr)
            else:
                raise ValueError("layer %d: rank %d owns no expert" % (il, owned.index(0)))


def report(owner, rows, n_ranks, n_layer, n_expert):
    total_owned = [0] * n_ranks
    replicated_total = 0
    worst = 0.0
    ratio_sum = 0.0
    layers_counted = 0
    per_layer = []
    for il in range(n_layer):
        row = owner[il]
        counts = rows[il]
        load = [0.0] * n_ranks
        owned = [0] * n_ranks
        total = float(sum(counts))
        for e, o in enumerate(row):
            if o == -1:
                replicated_total += 1
                for r in range(n_ranks):
                    load[r] += counts[e] / n_ranks
            else:
                owned[o] += 1
                total_owned[o] += 1
                load[o] += counts[e]
        if total > 0:
            mean = total / n_ranks
            mx = max(load)
            ratio = mx / mean if mean > 0 else 1.0
            worst = max(worst, ratio)
            ratio_sum += ratio
            layers_counted += 1
            per_layer.append((il, owned, mx, mean, ratio))
        else:
            per_layer.append((il, owned, 0.0, 0.0, 1.0))

    print("ranks=%d layers=%d experts=%d replicated_total=%d" % (n_ranks, n_layer, n_expert, replicated_total))
    print("owned experts per rank (all layers): %s" % total_owned)
    print("layer  owned_per_rank            max_load     mean_load   max/mean")
    for il, owned, mx, mean, ratio in per_layer:
        print("%5d  %-24s %11.0f %11.0f   %.3f" % (il, owned, mx, mean, ratio))
    if layers_counted:
        print("expected imbalance (max/mean): worst=%.3f avg=%.3f over %d layers"
              % (worst, ratio_sum / layers_counted, layers_counted))
    else:
        print("no routing counts in CSV (all zero): load report skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="routing hotness CSV (MoeHybridRoutingStats::save_csv); "
                    "optional with --dims")
    ap.add_argument("--dims", help="L,E,K (n_layer,n_expert,n_expert_used) when no CSV is given, "
                    "e.g. 43,256,6 for DeepSeek V4 Flash")
    ap.add_argument("--all-on-rank", type=int, default=None,
                    help="diagnostic: rank R owns every expert, other ranks get empty shards")
    ap.add_argument("--ranks", "-n", type=int, required=True, help="cluster size N (2..8)")
    ap.add_argument("--mode", choices=("uniform", "balanced"), default="balanced")
    ap.add_argument("--replicate-hot", type=int, default=0, help="replicate the k hottest experts per layer")
    ap.add_argument("--out", "-o", required=True, help="output placement JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.ranks < 1:
        sys.exit("--ranks must be >= 1")
    if args.replicate_hot < 0:
        sys.exit("--replicate-hot must be >= 0")

    if args.csv:
        n_layer, n_expert, n_expert_used, rows = load_hotness_csv(args.csv)
    elif args.dims:
        n_layer, n_expert, n_expert_used = (int(x) for x in args.dims.split(","))
        rows = [[0] * n_expert for _ in range(n_layer)]
    else:
        sys.exit("give a hotness CSV or --dims L,E,K")
    if args.replicate_hot > n_expert - args.ranks:
        sys.exit("--replicate-hot leaves fewer experts than ranks")

    if args.all_on_rank is not None:
        owner = build_all_on_rank(args.all_on_rank, args.ranks, n_layer, n_expert)
        args.replicate_hot = 0
    elif args.mode == "uniform":
        owner = build_uniform(args.ranks, n_layer, n_expert, rows, args.replicate_hot)
    else:
        if not args.csv:
            sys.exit("balanced placement needs a hotness CSV")
        owner = build_balanced(args.ranks, n_layer, n_expert, rows, args.replicate_hot)
    validate(owner, args.ranks, n_expert, allow_empty=args.all_on_rank is not None)

    doc = {
        "n_ranks": args.ranks,
        "n_layer": n_layer,
        "n_expert": n_expert,
        "n_expert_used": n_expert_used,
        "replicate_hot": args.replicate_hot,
        "owner": owner,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        # one owner row per line keeps the file diffable
        f.write("{\n")
        f.write('  "n_ranks": %d,\n' % doc["n_ranks"])
        f.write('  "n_layer": %d,\n' % doc["n_layer"])
        f.write('  "n_expert": %d,\n' % doc["n_expert"])
        f.write('  "n_expert_used": %d,\n' % doc["n_expert_used"])
        f.write('  "replicate_hot": %d,\n' % doc["replicate_hot"])
        f.write('  "owner": [\n')
        for il, row in enumerate(owner):
            f.write("    " + json.dumps(row, separators=(",", ":")) + (",\n" if il + 1 < n_layer else "\n"))
        f.write("  ]\n}\n")

    if not args.quiet:
        report(owner, rows, args.ranks, n_layer, n_expert)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
