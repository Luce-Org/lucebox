# Native DS4V tower unit

Start only after the mmproj export has passed its original-byte and native GGUF parser checks. This unit is the reusable native vision runtime, not the HTTP or decoder integration.

## Scope

Own new `server/src/deepseek4/deepseek4_vision.h` and `.cpp`, a standalone CPU validation target under `server/tools/ds4v_vision`, and focused runtime tests. Add main server build wiring only if required to link the reusable unit. Do not edit HTTP, decoder routing, attention, or chunk selection in this unit.

The public data shape is a validated VisionConfig, a patch grid with height and width, and a VisionRuntime that owns projector weights and graph resources on a caller-selected GGML backend. Encoding consumes channel-major patches and returns raster-order aligner rows. Sentinel vectors remain accessible through typed sentinel identities. The later request layer owns N-layout and span placement.

## Grounding

Read selected `design.md`, `native-tower-grounding.md`, and the complete parent `source-reference/vision.py`. Load the verified projector metadata and preserve original tensor names. Reject missing tensors, wrong shapes, unsupported schema/recipe strings, and wrong language vocabulary or dimension before backend allocation.

Implement the actual 32-block tower, final RMSNorm, and aligner. Parent attention is fully bidirectional. RoPE splits each 64-dimensional head into two 32-value halves and uses height frequencies followed by width frequencies. The aligner pads bottom and right to multiples of three, gathers channel-first 3x3 patches, and uses exact GELU.

The parent executes BF16 linear outputs, activations, and residuals, with F32 normalization and rotary arithmetic followed by BF16 casts. Establish how the native primitives represent this arithmetic. Measure differences rather than widening a tolerance until a test passes. Use stage outputs to isolate layout mistakes from expected kernel rounding.

## Remote verification

Mac authors only. Create another isolated soulf worktree and build directory from the pushed fork branch. Do not change the active converter worktree or binary. Do not start any GPU lane in this unit until the parent releases the GPU pair after text load proof.

Use the complete CPU parent fixtures at `~/lucebox-ds4v-mix-fix/artifacts/vision-reference`. Its manifest records hashes, dimensions, and the original code and images. Existing CPU Python environment is `~/lucebox-ds4v-mix-fix/.venv-vision-reference`. Do not change it. The projector path comes from the verified exporter evidence.

First test small deterministic geometry for half-split RoPE, bidirectional attention, and channel-first padded unfolding. Then execute the full native tower on CPU for both carrots and corn. Compare raster features and aligner embeddings with the parent fixture, reporting maximum absolute error, RMS error, cosine similarity, finite values, and shape. Add stage comparisons when needed to locate a mismatch. These prove components, not complete decoder logits.

Return exact commits, commands, proof paths, numerical results, resource ownership, and unresolved limits. Parent reviews the diff independently before accepting. No HTTP lane or server startup is part of this unit.
