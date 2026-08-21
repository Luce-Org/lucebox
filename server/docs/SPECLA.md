# SpecLA for the Qwen3.6 Gated-DeltaNet target

This is the Qwen runtime implementation of *SpecLA: Efficient Speculative
Decoding for Linear-Attention Models* (arXiv:2607.16673). It is enabled only
for the single-device Qwen target with exact fast rollback; tensor-parallel,
layer-split, and KVFlash targets keep their established state paths.

The important distinction from the old experimental path is that the normal
route is now the paper's state-resident, chain-decomposed verifier with delayed
raw factors. The UT-factorized verifier and immediate DeltaConstruct commit
remain only as a compatibility/reference route.

## Paper-to-code map

### §4.1: state-resident serial verification

`ggml_gated_delta_net_specla` processes every adjacent node in a chain inside
one kernel while its recurrent-state tile stays in registers. The kernel tiles
the value dimension and keeps the full key dimension inside a warp, avoiding a
cross-block reduction for `S^T q`. It reads durable state once and does not
write it between adjacent candidate tokens.

`ggml_ssm_conv_specla` does the same for the causal depthwise-convolution
window. This matters for Qwen: correct GDN recurrence alone is insufficient if
siblings accidentally inherit convolution history from DFS neighbours.

Chain verification is represented by one chain and one dependency wave, so it
uses these same kernels without tree-specific overhead.

### §4.2: fully factorized tree verification

`build_delta_net_specla` implements the topology-masked UT transform. Host
inputs provide strict-ancestor, inclusive-ancestor, and identity masks; the
builder returns candidate outputs plus corrected values and cumulative gates.
It is retained as a numerical reference and fallback for callers without an
HLD schedule. It is not the production Qwen route because its short-window
setup and reduction cost lose to the serial kernel at this shape.

### §4.3: chain-decomposed hybrid verification

`make_specla_hld_schedule` performs deterministic heavy-light decomposition:

- the largest child subtree is the heavy continuation;
- heavy edges stay in one state-resident chain;
- light edges create chain boundaries;
- chains are grouped into dependency waves; and
- all ready chains in a wave launch in parallel.

Only states at light-edge boundaries are materialized. A chain loads either
durable root state or its parent boundary, executes serially, emits per-node
outputs/factors, and writes only boundary states needed by later waves. Both
GDN and convolution consume the same packed parent topology.

### §5.1: accepted-factor buffering

Dense per-candidate recurrent checkpoints are replaced by two consolidated
FP32 factor banks. The HLD kernels write directly into the current bank:

- normalized key `k`;
- the already-computed Delta-rule residual/update vector;
- log decay `g`; and
- raw convolution input.

This is sufficient to replay the exact serial recurrence for an accepted path
without regenerating projections or storing a dense state per node. A chain
acceptance rotates banks with no device copy. A branch acceptance gathers its
arbitrary DFS indices into path order with one compaction kernel.

### §5.2: delayed fused update and verify

After selection, accepted factors remain pending. At the next verification,
each state-resident kernel:

1. loads its durable state tile;
2. applies the preceding accepted GDN/conv factors;
3. writes that committed state once;
4. immediately verifies the current chain from the same live tile; and
5. records current candidates in the other bank.

There is no standalone commit kernel or recurrent-state snapshot in the normal
loop. If generation ends, switches to autoregressive decode, or is cancelled
before another verification, `finish_speculative_state()` materializes the one
remaining pending path exactly once.

The older immediate `specla_commit_accepted` implementation remains for the
factorized fallback and tests; it is not used by the normal HLD route.

### §6.1: confidence-guided pruning

`build_ddtree` and `build_ddtree_conditional` score a node by cumulative path
log probability and keep

```
q(v) >= q* - tau_tree
```

before applying the node budget. Expansion is best-first and the retained set
is ancestor-closed. A finite `--ddtree-tau` also shrinks the actual target
batch; pruned nodes are not replaced by fake padding in SpecLA mode.

### §6.2: target-aligned drafting

The Qwen DFlash draft consumes the target's recurrent execution features, so
the feature-alignment interface exists. Its released checkpoint is a five-layer
block-diffusion drafter, however, not the paper's specially trained one-layer
EAGLE drafter.

The runtime includes exact prefix-conditioned tree construction. Set
`DFLASH_SPECLA_CONDITIONAL_DRAFT=1` to rerun the draft for every expanded
prefix. This is the faithful algorithmic experiment, but it is intentionally
off by default: per-node reruns of this five-layer draft cost more than a 27B
target verification. Realizing the paper's §6.2 speedup requires training the
small recurrent-feature EAGLE checkpoint; a runtime patch cannot synthesize
that model artifact.

SpecLA defaults to the paper's top-k=4 tree width. Use
`--specla-top-k <K>` when a checkpoint or workload benefits from a different
width. `DFLASH_SPECLA_TOPK=<K>` remains available for non-CLI harnesses.

## Running

```sh
# Enable SpecLA and let the runtime select a compatible proposal adapter.
# Qwen3.6 currently selects DDTree with budget 22, tau 6, and top-k 4.
# Older drafts without embedded sliding-window metadata may additionally need
# --draft-swa=2048.
build/test_dflash TARGET.gguf DRAFT.gguf prompt.bin 128 out.bin --specla

# Expensive exact branch-conditioned drafting experiment.
DFLASH_SPECLA=1 DFLASH_SPECLA_CONDITIONAL_DRAFT=1 build/dflash_server ...
```

Tau 6 was best in the current ten-prompt probe and is the `--specla` default.
Use `--ddtree-tau` to tune it for the intended checkpoint and workload; a
tighter margin changed batching at numerically sensitive logits and was not
consistently faster. `--ddtree-budget`, `--specla-top-k`, and `--draft-swa`
remain explicit for the same setup-dependent reason.

`DFLASH_SPECLA=1` remains a compatibility switch for non-CLI integrations.
`DFLASH_SPECLA_CONDITIONAL_DRAFT=1` and `DFLASH_SPECLA_FUSED_COMMIT=0` are
advanced algorithm/debug controls, not required for normal use.

KVFlash uses a pager-backed attention cache that cannot migrate SpecLA factor
state. Combining `--specla` with `--kvflash <tokens|auto>` therefore prints a
warning and falls back to ordinary DDTree verification; startup reports SpecLA
as off.

SpecLA does not intrinsically require DDTree. It consumes speculative
candidates plus their parent topology; a chain is a degenerate tree. DDTree is
the proposal adapter currently connected for Qwen3.6. A future DSpark adapter
can feed the same user-facing `--specla` mode when its target recurrence and
factor-capture path are supported.

## Correctness coverage

- CPU tests cover cumulative-score pruning, ancestor closure, budget caps,
  exact-prefix conditional queries, and deterministic HLD schedules.
- GPU tests compare HLD chains and trees with the sequential GDN reference at
  the Qwen state shape, including non-empty pending factors and sibling
  boundaries.
- Convolution tests cover tree ancestry, delayed pending windows, and direct
  factor capture.
- Factorized UT output and DeltaConstruct remain cross-checked against the
  serial recurrence.
- Runtime exit, tree sibling, greedy chain/tree, and sampled bonus paths all
  use the same double-bank lifecycle; the final flush is GPU-tested directly.

On the current gfx1151 probe, 16-node and 22-node HLD trees produced identical
128-token hashes even though their accepted boundaries differed. The legacy
F16 checkpoint rollback did not match that stream. Dynamic tree sizes can
still perturb near-tie logits in Qwen's full-attention layers; that is a
batched floating-point issue outside the GDN recurrence and is why confidence
margin changes require output checks.

## Current Qwen3.6-27B measurements

Target: Qwen3.6-27B Q4_K_M. Draft: five-layer DFlash Q8_0 with SWA 2048.
Workload: ten cached HumanEval-style prompts, 128 generated tokens, gfx1151.

| route | tree | mean accepted/step | mean decode |
|---|---:|---:|---:|
| SpecLA off reference | 22, top-8 | 5.64 | 25.14 tok/s |
| completed HLD, no pruning | 22, top-4 | 5.58 | 26.05 tok/s |
| completed HLD, `tau=6` | 22, top-4 | 5.95 | **27.19 tok/s** |
| completed HLD, `tau=6` | 22, top-8 | 5.90 | 26.92 tok/s |

The recommended route is 8.2% faster than the SpecLA-off reference, but that
reference runs the same 22-node tree at **top-8** while the recommended route
runs at **top-4**, so the headline number mixes the tree-width change into the
SpecLA/pruning effect. The comparable pairs are: at top-8, completed HLD with
`tau=6` reaches 26.92 tok/s versus 25.14 tok/s for the off reference (+7.1%);
within completed HLD, `tau=6` raises top-4 throughput from 26.05 to 27.19
tok/s (+4.4%); and with HLD plus `tau=6` held fixed, top-4 is 1.0% faster than
top-8 (27.19 versus 26.92 tok/s). The table does not include a pair that
isolates HLD alone. On one fixed 22-node step, HLD reduced target verification
from 191.53 ms to 184.11 ms; the full-model gain is smaller than the paper's
GDN-1.3B result because Qwen has full-attention layers and large
projections/FFNs that HLD does not accelerate, and this draft's acceptance is
below the paper's best workloads.

The paper reports 1.42x mixed, 1.70x GSM8K, and 1.06x HumanEval end-to-end
speedups on an H100 with a pure GDN-1.3B target and its trained EAGLE-style
drafter. Those figures are not directly transferable to this 27B hybrid model.
