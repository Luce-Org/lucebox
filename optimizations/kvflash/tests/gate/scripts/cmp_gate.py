"""Compare arms A and B of a gate run. usage: python3 -I cmp_gate.py ROOT
Prints identity per turn and SERVER-SIDE medians (from 'chat DONE' lines; warm-up request dropped):
prefill s, decode tok/s, wall s; plus spec-decode acceptance when present."""
import json, re, sys, statistics as S
R = sys.argv[1].rstrip("/") + "/"
J = {k: json.load(open(R + f"gate4_{k}.json")) for k in "AB"}
DONE = re.compile(r"chat DONE .*? out=(\d+) ([\d.]+)s .*?prefix_len=(-?\d+) prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")


def server(k):
    rows = []
    for l in open(R + f"gate4_{k}.log", errors="replace"):
        m = DONE.search(l)
        if m:
            rows.append(dict(out=int(m[1]), wall=float(m[2]), prefix=int(m[3]), pf=float(m[4]), tps=float(m[6])))
    return rows[1:]


def rp(r):
    for l in r["live"]:
        m = re.search(r"live restore: L=(\d+) C=(-?\d+) replay=(\d+)", l)
        if m:
            return "L=%s C=%s replay=%s" % m.groups()
    return "-"


def spec(k):
    rows = []
    for l in open(R + f"gate4_{k}.log", errors="replace"):
        m = re.search(r"\[spec-decode\] tokens=(\d+).*accepted=(\d+)/(\d+)", l)
        if m:
            rows.append(tuple(map(int, m.groups())))
    return rows[1:]


SV = {k: server(k) for k in "AB"}
print("turn ptok A_pf B_pf A_tps B_tps B_restore identical")
ok = len(J["A"]) == len(J["B"]) and len(J["A"]) > 0
for i, (a, b) in enumerate(zip(J["A"], J["B"])):
    same = a["answer"] == b["answer"]
    ok &= same
    sa, sb = (SV[k][i] if i < len(SV[k]) else {} for k in "AB")
    print(a["turn"], a["prompt_tokens"], sa.get("pf"), sb.get("pf"), sa.get("tps"), sb.get("tps"), rp(b), same)
for k in "AB":
    X = SV[k]
    if not X:
        print(k, "no server-side rows")
        continue
    print(k, "server-side medians: prefill %.2f s, decode %.1f tok/s, wall %.2f s (n=%d)" % (
        S.median(r["pf"] for r in X), S.median(r["tps"] for r in X), S.median(r["wall"] for r in X), len(X)))
    sp = spec(k)
    if sp:
        tot = sum(s[2] for s in sp)
        print(k, "spec accepted/proposed", sum(s[1] for s in sp), tot, "pooled %.1f%%" % (100 * sum(s[1] for s in sp) / tot))
n_same = sum(a["answer"] == b["answer"] for a, b in zip(J["A"], J["B"]))
print(("IDENTICAL %d/%d" if ok else "DIFF %d/%d") % (n_same, len(J["A"])))
sys.exit(0 if ok else 1)
