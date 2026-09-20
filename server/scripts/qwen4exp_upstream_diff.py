#!/usr/bin/env python3
"""Differential harness: compare upstream llama.cpp qwen4exp node activations
against our hand-rolled qwen4exp forward, layer by layer, on the box.

Aligned pairs (ours L%02d.<kind> vs upstream <up>-N; prefill pass; both are
[n_embd, T] activations, so element counts must match exactly):
  L%02d.moe <-> ffn_out-N       MoE block output, layers 0..n_layer-2
                                (the final layer is skipped: upstream
                                computes the last layer's FFN for the final
                                token only, n = n_embd vs n_embd*T)
  L%02d.att <-> attn_output-N   attention output, full-attention layers only
                                (identified by ours' L%02d.Q dumps). At
                                linear-attention layers upstream attn_output
                                is the pre-projection [6144, T] tensor while
                                ours att is post-projection [n_embd, T], so
                                those layers are out of scope by geometry.

Deliberately NOT aligned:
  L%02d.res <-> l_last-N  REMOVED: ours .res is the packed [n_embd,hc,T,2]
  hyper-compact stream (or the fused combine node), while upstream l_last-N
  is the wide [n_embd,T] residual. No pure channel slice recovers one from
  the other (recovery needs the hc up-mix matmul), so their scalar stats are
  semantically incomparable; the old alignment failed every run and could
  hijack the onset.

Token parity: both engines feed tok[i] = (i*7919+13) % n_vocab for a
length-S prefill (upstream: qwen4exp_upstream_nodes.cpp; ours:
server/test/smoke_qwen4exp_forward.cpp). The upstream dumper always prints
a TOKENS line with its first/last ids and vocab; the harness recomputes the
recipe using the vocab size our binary reports and cross-checks. Our smoke
binary does not print its token ids, so ours' ids are a source-verified
assumption, not a runtime observation.

Onset (informational): the first aligned pair whose absmax ratio leaves
[0.99,1.01]. A single run of each engine cannot separate FP noise from real
divergence; pass --calibrate to rerun the upstream engine A-A and print its
run-to-run noise floor for interpretation. The exit code ignores the onset
band and fails only on hard errors: element-count mismatch on an in-scope
aligned pair, non-finite values on either side, token parity failure, or a
two-scalar divergence (absmax ratio outside [0.9,1.1] AND mean drift > 0.1
relative to absmax — an absmax-only excursion with stable mean is reported
as informational outlier-driven drift; max-abs difference would need full
tensor dumps, not scalars, so it is not computed).
"""
import argparse
import os
import re
import subprocess
import sys


def run(cmd, env=None, timeout=600):
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return p.stdout + p.stderr


def build_upstream_dumper(script_dir, llama_tree, out_bin):
    cpp = os.path.join(script_dir, "qwen4exp_upstream_nodes.cpp")
    cmd = [
        "g++", "-O2",
        f"-I{llama_tree}/include", f"-I{llama_tree}/ggml/include",
        cpp,
        f"-L{llama_tree}/build/bin", "-lllama", "-lggml", "-lggml-base",
        f"-Wl,-rpath,{llama_tree}/build/bin",
        "-o", out_bin,
    ]
    out = run(cmd)
    if not os.path.exists(out_bin):
        print(out, file=sys.stderr)
        sys.exit("upstream dumper build failed")
    return out_bin


def build_ours(repo):
    build_dir = os.path.join(repo, "server", "build-hip")
    bin_path = os.path.join(build_dir, "smoke_qwen4exp_forward")
    if not os.path.exists(bin_path):
        out = run(["ninja", "-C", build_dir, "smoke_qwen4exp_forward"], timeout=1200)
        if not os.path.exists(bin_path):
            print(out, file=sys.stderr)
            sys.exit("our smoke build failed")
    return bin_path


UP_RE = re.compile(
    r"NODE (?P<name>\S+)\s+n=(?P<n>\d+)\s+finite=(?P<finite>\d)\s+"
    r"absmax=(?P<absmax>\S+)\s+mean=(?P<mean>\S+)"
)
OUR_RE = re.compile(
    r"\[dump\] (?P<name>\S+)\s+n=(?P<n>\d+)\s+finite=(?P<finite>\d)\s+"
    r"absmax=(?P<absmax>\S+)\s+mean=(?P<mean>\S+)"
)
UP_TOKENS_RE = re.compile(r"TOKENS n=(\d+) n_vocab=(\d+) first=(-?\d+) last=(-?\d+)")
OUR_VOCAB_RE = re.compile(r"\[smoke\] load .*\bvocab=(\d+)")


def _stats(m):
    return {
        "n": int(m.group("n")),
        "finite": m.group("finite") == "1",
        "absmax": float(m.group("absmax")),
        "mean": float(m.group("mean")),
    }


def parse_upstream(text):
    # kind -> {layer: stats}; first occurrence = prefill
    out = {}
    for m in UP_RE.finditer(text):
        mm = re.match(r"(?P<kind>[a-zA-Z_]+)-(?P<layer>\d+)$", m.group("name"))
        if not mm:
            continue
        kind, layer = mm.group("kind"), int(mm.group("layer"))
        out.setdefault(kind, {})
        if layer not in out[kind]:
            out[kind][layer] = _stats(m)
    return out


def parse_ours(text):
    out = {}
    for m in OUR_RE.finditer(text):
        mm = re.match(r"L(?P<layer>\d+)\.(?P<kind>\w+)$", m.group("name"))
        if not mm:
            continue
        layer, kind = int(mm.group("layer")), mm.group("kind")
        out.setdefault(kind, {})
        if layer not in out[kind]:
            out[kind][layer] = _stats(m)
    return out


ALIGN = {
    "moe": "ffn_out",
    "att": "attn_output",
}

ONSET_BAND = (0.99, 1.01)
BAD_BAND = (0.9, 1.1)
MEAN_DRIFT_MAX = 0.1


def check_tokens(up_out, our_out, seq):
    """Hard-check token parity. Upstream prints its ids; our smoke binary does
    not, so ours' tok[i]=(i*7919+13)%n_vocab recipe (smoke_qwen4exp_forward.cpp)
    is verified against the upstream ids and the vocab both engines report."""
    tm = UP_TOKENS_RE.search(up_out)
    vm = OUR_VOCAB_RE.search(our_out)
    if not tm or not vm:
        print(f"[tokens] WARNING: cannot verify token parity "
              f"(upstream TOKENS line found: {bool(tm)}, our vocab line found: {bool(vm)})")
        return False
    s, n_vocab = int(tm.group(1)), int(tm.group(2))
    first, last = int(tm.group(3)), int(tm.group(4))
    problems = []
    if n_vocab != int(vm.group(1)):
        problems.append(f"vocab mismatch: upstream {n_vocab} vs ours {vm.group(1)}")
    if s != seq:
        problems.append(f"upstream ran {s} tokens, harness expected {seq}")
    expected = [(i * 7919 + 13) % n_vocab for i in range(seq)]
    if os.environ.get("QWEN4EXP_TOKEN_FILE"):
        with open(os.environ["QWEN4EXP_TOKEN_FILE"]) as f:
            expected = list(map(int, f.read().split()))
        if len(expected) != seq:
            print("[tokens] ERROR: token file length differs from --seq")
            return True
    exp_first, exp_last = expected[0], expected[-1]
    if first != exp_first or last != exp_last:
        problems.append(f"upstream tokens first/last {first}/{last} != expected {exp_first}/{exp_last}")
    for p in problems:
        print(f"[tokens] ERROR: {p}")
    if problems:
        return True
    source = "token file" if os.environ.get("QWEN4EXP_TOKEN_FILE") else "synthetic recipe"
    print(f"[tokens] OK: upstream first/last = {first}/{last}, matches {source}; "
          "ours uses the same input (source-verified)")
    return False


def compare(up_data, our_data, seq):
    bad = False
    onset = None
    onset_pair = None
    for our_kind, up_kind in ALIGN.items():
        print(f"\n=== {up_kind} (upstream) vs L%02d.{our_kind} (ours) ===")
        up_layers = up_data.get(up_kind, {})
        our_layers = our_data.get(our_kind, {})
        common = set(up_layers) & set(our_layers)
        if our_kind == "att":
            full_attn = set(our_data.get("Q", {}))
            if not full_attn:
                print("  <-- ERROR: cannot identify full-attention layers (no L%02d.Q dumps)")
                bad = True
                continue
            layers = sorted(common & full_attn)
            skipped = sorted(common - full_attn)
            if skipped:
                print(f"  (scope: {len(layers)} full-attention layers; {len(skipped)} linear-attention "
                      f"layers excluded: upstream attn_output there is the pre-projection tensor, "
                      f"ours att is post-projection)")
        else:
            layers = sorted(common)
        if not layers:
            print(f"  <-- ERROR: no aligned layers found for {up_kind} <-> {our_kind}")
            bad = True
        for layer in layers:
            u = up_layers[layer]
            o = our_layers[layer]
            pair = f"{up_kind}-{layer}/L{layer:02d}.{our_kind}"
            if our_kind == "moe" and layer == layers[-1] and u["n"] != o["n"] and u["n"] * seq == o["n"]:
                print(f"L{layer:02d}  SKIPPED: upstream {up_kind} covers a final-token-only subset "
                      f"(n={u['n']} vs ours n={o['n']}); n parity inapplicable")
                continue
            errors = []
            if u["n"] != o["n"]:
                errors.append(f"element-count mismatch: upstream n={u['n']}, ours n={o['n']}")
            if not u["finite"]:
                errors.append("upstream has non-finite values")
            if not o["finite"]:
                errors.append("ours has non-finite values")
            ratio = o["absmax"] / u["absmax"] if u["absmax"] else float("nan")
            scale = max(u["absmax"], o["absmax"], 1e-12)
            mdrift = abs(o["mean"] - u["mean"]) / scale
            info = ""
            if not (BAD_BAND[0] <= ratio <= BAD_BAND[1]):
                if mdrift > MEAN_DRIFT_MAX:
                    errors.append(f"absmax ratio {ratio:.4f} outside [{BAD_BAND[0]},{BAD_BAND[1]}] AND "
                                  f"mean drift {mdrift:.4f} > {MEAN_DRIFT_MAX}: divergent on both scalars")
                else:
                    info = "  <-- absmax excursion outside [0.9,1.1] (mean stable: outlier-driven drift, informational)"
            flags = "".join(f"  <-- ERROR [{pair}]: {e}" for e in errors)
            if errors:
                bad = True
            if not (ONSET_BAND[0] <= ratio <= ONSET_BAND[1]) and (onset is None or layer < onset):
                onset, onset_pair = layer, pair
            print(f"L{layer:02d}  n={u['n']}/{o['n']}  "
                  f"absmax up={u['absmax']:.5g} ours={o['absmax']:.5g} ratio={ratio:.4f}  "
                  f"mean up={u['mean']:.5g} ours={o['mean']:.5g} drift={mdrift:.4f}{info}{flags}")
    return bad, onset, onset_pair


def calibrate(up_bin, args, env, up_data):
    """A-A rerun of the upstream engine: measures its run-to-run absmax noise
    floor so small ours-vs-upstream excursions can be interpreted."""
    up_out2 = run([up_bin, args.model, str(args.seq)], env=env, timeout=600)
    up_data2 = parse_upstream(up_out2)
    worst, worst_pair = 0.0, "-"
    for up_kind in ALIGN.values():
        for layer in sorted(set(up_data.get(up_kind, {})) & set(up_data2.get(up_kind, {}))):
            a = up_data[up_kind][layer]["absmax"]
            b = up_data2[up_kind][layer]["absmax"]
            if a > 0:
                d = abs(a - b) / a
                if d > worst:
                    worst, worst_pair = d, f"{up_kind}-{layer}"
    print(f"[calibrate] upstream A-A rerun: max |dabsmax|/absmax = {worst:.4g} ({worst_pair})")
    print("[calibrate] excursions within this floor are run-to-run kernel/FP noise, not divergence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-tree", default="/home/duster/llama-qwen4")
    ap.add_argument("--repo", default="/home/duster/lucebox-qwen4exp")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--reference", action="store_true",
                    help="enable upstream graph/attention/RoPE paths and require exact expert IDs")
    ap.add_argument("--output-dir", help="save raw node and expert-id logs for exact comparison")
    ap.add_argument("--calibrate", action="store_true",
                    help="rerun the upstream engine once (A-A) and print its run-to-run noise floor")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    up_bin = "/tmp/qwen4exp_nodes_up"
    build_upstream_dumper(script_dir, args.llama_tree, up_bin)
    our_bin = build_ours(args.repo)

    env = dict(os.environ, HIP_VISIBLE_DEVICES="1")
    up_out = run([up_bin, args.model, str(args.seq)], env=env, timeout=600)
    up_data = parse_upstream(up_out)

    env2 = dict(env, QWEN4EXP_DUMP="1", GGML_CUDA_MMB="0")
    if args.reference:
        env2["QWEN4EXP_UPSTREAM"] = "1"
    our_out = run([our_bin, args.model, str(args.seq)], env=env2, timeout=600)
    our_data = parse_ours(our_out)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, output in (("upstream.log", up_out), ("ours.log", our_out)):
            with open(os.path.join(args.output_dir, name), "w") as f:
                f.write(output)

    if "MODEL_LOAD_FAIL" in up_out or "DECODE_FAIL" in up_out or "CTX_FAIL" in up_out:
        print(up_out)
        sys.exit("upstream run failed")

    bad = check_tokens(up_out, our_out, args.seq)
    bad_tables, onset, onset_pair = compare(up_data, our_data, args.seq)
    bad = bad or bad_tables
    if args.reference:
        up_ids, our_ids = {}, {}
        for m in re.finditer(r"IDS ffn_moe_topk-(\d+)\s+n=\s*\d+ ([^\n]+)", up_out):
            up_ids.setdefault(int(m[1]), list(map(int, m[2].split())))
        for m in re.finditer(r"\[dump\] L(\d+)\.mid[^\n]*ids:([^\n]+)", our_out):
            our_ids.setdefault(int(m[1]), list(map(int, m[2].split())))
        if not up_ids or up_ids.keys() != our_ids.keys():
            print("[ids] ERROR: missing expert-id layers")
            bad = True
        for layer in sorted(up_ids.keys() & our_ids.keys()):
            a, b = up_ids[layer], our_ids[layer]
            mismatches = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            print(f"[ids] L{layer:02d} mismatches={mismatches}/{len(a)}")
            bad |= mismatches != 0

    print("\n(note) res<->l_last alignment removed: ours L%02d.res is the packed "
          "[n_embd,hc,T,2] stream, not the wide residual")
    print("(note) a single run per engine cannot separate FP noise from divergence; "
          "ratios within the A-A noise floor (see --calibrate) are not evidence of drift")
    if args.calibrate:
        calibrate(up_bin, args, env, up_data)

    if onset is not None:
        print(f"\nDivergence onset (informational, [{ONSET_BAND[0]},{ONSET_BAND[1]}] band): "
              f"layer {onset} ({onset_pair})")
    else:
        print(f"\nNo onset (informational): all aligned pairs within [{ONSET_BAND[0]},{ONSET_BAND[1]}]")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
