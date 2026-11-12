#!/usr/bin/env python3
"""usage: report.py RUNDIR SCRIPTS_DIR -- speeds per engine arm (median/mean/min/max of the per-request arrays) and the
determinism repro (W1fix vs W1fix2 picks, Amendment 3 rule 1: picked/resident lines identical after masking 'N ms')."""
import datetime, glob, json, os, statistics, sys

ARMS = ("W0", "W1old", "W1fix", "W1fix2")
MIN_DECODE_TOKENS = 8   # history turns are 1-token greedy: no decode speed to read


def turns(d, arm):
    out = []
    for f in sorted(glob.glob(f"{d}/{arm}.lane*.turns.jsonl")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def speeds(ts):
    """Pure. (prefill tok/s, decode tok/s, wall s) arrays from turn records with timings."""
    pre, dec, wall = [], [], []
    for t in ts:
        tm = t.get("timings") or {}
        if t.get("error"):
            continue
        if tm.get("prefill_ms") and tm.get("prefilled_tokens"):
            pre.append(tm["prefilled_tokens"] / tm["prefill_ms"] * 1000)
        if tm.get("decode_ms") and (t.get("completion_tokens") or 0) >= MIN_DECODE_TOKENS:
            dec.append(t["completion_tokens"] / tm["decode_ms"] * 1000)
        if t.get("wall_s"):
            wall.append(t["wall_s"])
    return pre, dec, wall


def stats(xs):
    if not xs:
        return "n=0"
    return f"n={len(xs)} median={statistics.median(xs):.1f} mean={statistics.fmean(xs):.1f} min={min(xs):.1f} max={max(xs):.1f}"


def arm_wall(d, arm):
    """Seconds from 'gpu.lock held' to the arm's last status line, from <arm>.run.txt."""
    p = f"{d}/{arm}.run.txt"
    ts = [l for l in open(p) if l.startswith("=== ")] if os.path.exists(p) else []
    if len(ts) < 2:
        return None
    at = lambda l: datetime.datetime.fromisoformat(l.split()[1]).timestamp()
    held = [l for l in ts if "gpu.lock held" in l]
    return at(ts[-1]) - at(held[0] if held else ts[0])


def main(d, scripts):
    sys.path.insert(0, scripts)
    import conc_acc_amend3 as A, conc_acc_readout as R
    print(f"# WMMA determinism bench readout: {d}")
    for arm in ARMS:
        ts = turns(d, arm)
        if not ts:
            print(f"{arm:7s} NOT RUN")
            continue
        pre, dec, wall = speeds(ts)
        aw = arm_wall(d, arm)
        print(f"{arm:7s} requests={len(ts)} errors={sum(1 for t in ts if t.get('error'))} arm_wall={aw and f'{aw:.0f}s'}")
        print(f"        prefill tok/s {stats(pre)}")
        print(f"        decode  tok/s {stats(dec)}")
        print(f"        wall s        {stats(wall)}")
    for a, b in (("W1fix", "W1fix2"), ("W0", "W1fix"), ("W1old", "W1fix")):
        ga, gb = R.load_arm(d, a), R.load_arm(d, b)
        if not ga["turns"] or not gb["turns"]:
            continue
        ka, kb = A._by_key(ga), A._by_key(gb)
        keys = sorted(set(ka) | set(kb))
        bad = [k for k in keys if k not in ka or k not in kb or ka[k] != kb[k]]
        verdict = "PASS" if keys and not bad else "FAIL"
        tag = "DETERMINISM" if (a, b) == ("W1fix", "W1fix2") else "info"
        print(f"picks {a} vs {b} [{tag}]: {verdict} compared={len(keys)} differ={len(bad)} first={[f'{k[0][-6:]}:{k[1]}' for k in bad[:5]]}")
    for f in sorted(glob.glob(f"{d}/micro_*.txt")):
        for l in open(f):
            if " ms " in l:
                print(f"{os.path.basename(f)[6:-4]:16s} {l.strip()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
