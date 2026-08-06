# DS4 heterogeneous attention TP

This experiment extends the existing in-process expert split with real
attention parallelism across an R9700 and Strix Halo. It is opt-in and limited
to q=5 fused verification.

## Qualified layout

- R9700 owns 6 of the 8 output groups: 48 of 64 attention heads.
- Strix owns 2 groups: 16 heads (25%).
- Only compression-ratio-4 layers are split (21 of 43 layers at this model
  shape). Their longer KV span is large enough to repay the device fork.
- QR and the assembled KV matrix are packed into one persistent F32 staging
  packet. The packet is copied on the Strix stream after a main-stream event,
  so the R9700 immediately continues its independent 75% branch.
- Strix performs its Q projection, score/softmax/value attention, inverse RoPE,
  output-A, and its partial output-B projection.
- The peer returns one small partial residual. The existing HC-post operation
  adds the main and peer results; there is no intermediate head join.

`GGML_HIP_GRAPHS=ON` is mandatory. Disabling it reduced the same control from
88.538 to 83.208 tok/s and invalidated comparisons with the qualified runtime.

## Reproduce

Configure and build:

```bash
cmake -S server -B server/build-hip-dual \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON
cmake --build server/build-hip-dual --target dflash_server \
  test_deepseek4_unit -j 12
```

Set `TARGET_MODEL`, `DRAFT_MODEL`, `HOTNESS_CSV`, and optionally
`DECODE_HOTNESS_CSV`, then run:

```bash
server/scripts/qualify_ds4_q5_amd_attention_tp.sh
```

The wrapper checks the CMake cache before starting and defaults to two warmups,
seven measured 2K/128-token runs. Override `TARGETS` to run the full context
sweep.

## 2026-08-06 result on lucebox5

All accepted runs generated the exact expected SHA-256:
`0f785a7ffa406498aafb14553966eaed0f52220fed0f7cc016b66921d104d194`.

| Configuration | 2K median client decode | Result |
| --- | ---: | --- |
| Same binary, attention TP off, HIP graphs on | 88.538 tok/s | control |
| 25%, packed fork, explicit attention, output-B on peer | 88.704 tok/s | initial three-run screen |
| Same 25% configuration, final seven-run confirmation | 88.472 tok/s | retained and stable |
| 25%, same split, output-B on main | 88.531 tok/s | correct, neutral |
| 12.5%, packed fork | 87.968 tok/s | under-filled peer |
| 25%, fused peer attention | 83.797 tok/s | rejected |
| 25%, graph-native late fork | 75.231 tok/s | rejected: peer starts too late |

The first three-run screen measured 88.704 tok/s. The final two-warmup,
seven-measurement confirmation measured **88.472 tok/s** (88.309–88.611), with
7/7 correct hashes and no failed requests. Against the 88.538 same-binary
control, this is a tie within run noise. Describe the result as “more attention
overlap without a decode regression,” not as a large throughput win.
Raw result directories are under:

`/home/lucebox5/ds4-attention-full-tp-proven-20260806/results/attention_tp/`

The final confirmation run ID is
`attention-full-25-outputb-final-confirm-r7-teardownfix-2k-20260806ak`.

The first long run exposed a cache-eviction teardown race after four measured
requests. Multi-backend scheduler events and gallocr buffers were being freed
without first quiescing both GPU streams; the next graph build then crashed in
the attention node list. Scheduler teardown and native-graph invalidation now
synchronize both backends first. The identical nine-request rerun completed.

## Important rejected paths

- Copying ordinary graph temporaries on the destination stream can race the
  allocator. Only explicitly marked persistent staging tensors are safe.
- A normal dependency-tree join reduces scheduler segments, but records the
  fork event after the complete main branch has already been queued. That
  serializes Strix behind the R9700 despite the smaller graph.
- Global single-copy generation fences are unnecessary for the retained path
  and lowered its median.
- Moving 50% of the heads overloads Strix; moving 12.5% leaves useful Strix
  bandwidth idle. Two output groups are the measured balance point.
- Peer flash attention is exact for this qualification but materially slower
  than the explicit score/softmax/value kernels on gfx1151.
