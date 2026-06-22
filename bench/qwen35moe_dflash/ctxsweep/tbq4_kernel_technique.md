# TBQ4 fused-dequant FlashAttention — extracted technique (Indras-Mirror/llama.cpp-turboq-mtp)

Source fork: https://github.com/Indras-Mirror/llama.cpp-turboq-mtp (public, MIT, fork of ggml-org/llama.cpp)
Extracted 2026-06-22 via WebFetch on raw.githubusercontent URLs.

## TL;DR — the 3 techniques that made it fast (ranked by contribution)

1. **Fuse dequant INTO the FA tile-load, eliminating the separate dequant pass + extra
   global round-trip.** The FA kernel reads raw `block_tbq4_0` K/V blocks straight from
   global memory and writes `half2` directly into the shared-memory tiles that the existing
   FP16 MMA path consumes. There is no materialized FP16 KV cache in HBM. This is THE win:
   the K/V are never written back to global memory as FP16, so the "dequant tax" collapses to
   the cost of a LUT load already overlapped with the matmul.
2. **Column-group thread distribution for 100% lane utilization on the load.** Each thread owns
   one `elem_idx` (one nibble-pair / one `half2`) across rows, instead of one-row-per-thread.
   The top-of-loop comment / repo notes claim this lifts load utilization from ~25% (row-based)
   to 100%.
3. **Centroid LUT in `__constant__` + the matmul stays on FP16 tensor cores (NOT dp4a).**
   The 16-entry centroid table and the two ±1 Hadamard sign rows live in `__constant__`
   memory (broadcast, cached). Dequant is `centroid[nibble] * norm` → `half2`. The actual
   Q·Kᵀ and softmax·V run on the **stock `flash_attn_ext_f16` FP16 tensor-core MMA kernel**.
   They did NOT invent an integer-dot low-bit path; they reach ~q8 speed by making dequant
   free and reusing the FP16 MMA engine.

So "~99% of q8_0" does not come from a clever dp4a low-bit dot. q8_0 in upstream FA also
dequants to FP16 tiles and runs FP16 MMA; tbq4 matches it because the *only* extra work vs q8
is a `__constant__` LUT indexing per nibble, which is hidden under the MMA.

---

## 1. The format: `block_tbq4_0` (`ggml/src/ggml-common.h`)

```c
#define QK_TBQ4 128
typedef struct {
    ggml_half d;            // per-block scale (norm)
    uint8_t qs[QK_TBQ4 / 2]; // 64 bytes, two 4-bit indices per byte
} block_tbq4_0;
static_assert(sizeof(block_tbq4_0) == sizeof(ggml_half) + QK_TBQ4 / 2, "wrong tbq4_0 block size/padding");
```

- **128 elements / block**, 64 packed bytes + 2-byte half scale = **66 bytes → 4.125 bpw**
  (repo rounds to "4.25 bpv").
- **4-bit, 16-centroid Lloyd-Max codebook** (symmetric), NOT ternary.
- A **FWHT (Walsh-Hadamard) rotation with two ±1 sign masks** is applied so the values are
  near-Gaussian before centroid quantization — same accuracy idea as our tq3_0, just 4-bit/128
  vs our 3-bit/32.

### Centroid LUT + WHT sign tables — all `__constant__` (`ggml/src/ggml-cuda/tbq4-cuda.cuh`)

```c
static __constant__ float d_tbq4_centroids[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

static __constant__ float d_tbq4_wht_s1[128] = { /* ±1 sign mask, 128 entries */ };
static __constant__ float d_tbq4_wht_s2[128] = { /* ±1 sign mask, 128 entries */ };
```

(`s1`/`s2` are the pre- and post-FWHT sign-flip masks; rotate_forward(x) = s2 · (1/√128) · FWHT(s1 · x).)

---

## 2. Inner-loop byte batching — the load (`ggml/src/ggml-cuda/fattn-mma-tbq4.cuh`)

Top-of-file rationale comment:

```
// Fused MMA-native TBQ4 flash attention: reads raw TBQ4_0 K/V directly,
// dequants (centroid*norm) into half2 shmem tiles inside the attention loop.
// Q is pre-rotated (rotate_forward) so K attention works in the rotated domain.
// V accumulates in the rotated domain; output is inverse-rotated after the kernel.
```

### Synchronous tile loader — the hot path

```cuda
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_tbq4_load_tile(
        const char * __restrict__ data_raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row = D / 128;
    constexpr int elems_per_block = 64;
    constexpr int elems_per_row = blocks_per_row * elems_per_block;
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int base_row = 0; base_row < nbatch_fa; base_row += (nthreads + elems_per_row - 1) / elems_per_row) {
        const int idx = tid;
        const int elem_idx = idx % elems_per_row;   // column-group distribution
        const int row_offset = idx / elems_per_row;
        if (row_offset + base_row >= nbatch_fa) continue;

        const int blk_idx = elem_idx / elems_per_block;
        const int b       = elem_idx % elems_per_block;

        const char * row_ptr = data_raw + (int64_t)(base_row + row_offset) * stride_bytes;
        const block_tbq4_0 * blk = (const block_tbq4_0 *)(row_ptr) + blk_idx;
        const float norm = __half2float(__ldg(&blk->d));
        const uint8_t byte = __ldg(&blk->qs[b]);                       // 1 byte = 2 elems
        const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
        const half hi = __float2half(d_tbq4_centroids[byte >> 4]  * norm);
        tile[(base_row + row_offset) * stride_tile + elem_idx] = __halves2half2(lo, hi);
    }
    // ... oob_check zero-fill omitted ...
}
```

Key memory-access facts:
- **Load granularity in the sync path is 1 byte (`__ldg(&blk->qs[b])`) = 2 quant elements → 1 `half2` written.**
  It is NOT a `uint4`/`int4` wide vector load in the sync loader; the batching win is the
  **thread→element mapping** (one `half2` per thread, full-warp coverage), not a wide vector load.
- `__ldg` forces the read-only/L1 path; `blk->d` is read once per thread but is broadcast-friendly.

### Async/staged variant — here is where wide(r) loads appear

```cuda
template<int D, int nbatch_fa, int nwarps>
static __device__ __forceinline__ void flash_attn_ext_tbq4_load_raw_async(
        const char * __restrict__ data_raw, char * __restrict__ staging,
        const int stride_bytes, const int i_sup) {
    constexpr int raw_row_bytes = (D/128) * (int)sizeof(block_tbq4_0);
    constexpr int ints_per_row  = raw_row_bytes / 4;     // 4-byte (int) staging chunks
    constexpr int total_ints    = nbatch_fa * ints_per_row;
    constexpr int nthreads      = nwarps * warp_size;
    for (int idx = threadIdx.y*warp_size+threadIdx.x; idx < total_ints; idx += nthreads) {
        const int row = idx / ints_per_row;
        const int off = (idx % ints_per_row) * 4;
        if (row < i_sup) {
            const int src = __ldg((const int *)(data_raw + (int64_t)row*stride_bytes + off)); // 4-byte load
            *(int *)(staging + (int64_t)row*tbq4_staging_row_bytes<D>() + off) = src;
        }
    }
}
```

The staged path copies raw bytes in **4-byte (`int`) chunks** into a 16-byte-aligned staging
buffer (`tbq4_staging_row_bytes` rounds up to multiple of 16), then `flash_attn_ext_tbq4_dequant_staging`
runs the identical centroid·norm → half2 expansion out of the staging buffer. Used to dodge
`cp.async` alignment constraints on the 66-byte block stride.

```cuda
template<int D>
static constexpr __host__ __device__ int tbq4_staging_row_bytes() {
    constexpr int raw_bytes = (D/128) * (int)sizeof(block_tbq4_0);
    return (raw_bytes + 15) & ~15;   // 16-byte align for vectorized staging
}
```

> Transfer note: if you want a wide-vector load for tq3_0, the staged path is the model —
> stage raw 14-byte-block rows into a 16-byte-aligned scratch with `int`/`int4` copies, then
> dequant out of scratch. The sync path proves you do NOT need wide loads to hit q8 speed;
> per-thread `half2` output coverage is enough.

---

## 3. Codebook / centroid handling

- **Where:** `__constant__ float d_tbq4_centroids[16]` (broadcast through the constant cache;
  all lanes hitting the same nibble coalesce to one constant-cache transaction).
- **Precompute:** NONE per-tile. They do **not** pre-multiply centroid×scale into shared memory.
  The hot loop does `d_tbq4_centroids[nibble] * norm` inline, once per element. `norm` is loaded
  per block via `__ldg(&blk->d)`. Because the LUT is `__constant__` and only 16 floats, there's
  no LUT serialization pressure to amortize — that's why no shared-mem precompute exists.
- **LUT access in hot loop:** `d_tbq4_centroids[byte & 0xF]` and `d_tbq4_centroids[byte >> 4]`.

---

## 4. Q handling + accumulation (dp4a vs FP) — CRITICAL for the port

- **Q stays FP16.** `tbq4_rotate_Q_tile()` keeps Q as `half2` and applies the *forward* FWHT
  rotation (warp-shuffle stages 0–4, shared-mem stages 5–6) so Q lives in the same rotated
  domain as the (rotation-baked) K. Q is **never** int8-quantized.
- **The dot product is FP16 tensor-core MMA, NOT dp4a.** The launcher
  (`fattn-mma-tbq4-launch.cuh`) sets `fattn_kernel = flash_attn_ext_f16<...>` and its header
  comment requires `fattn-mma-f16.cuh` "(provides flash_attn_ext_f16, launch_fattn, ... MMA
  helpers)". After the tbq4 loader fills the `half2` K/V tiles, the **stock FP16 MMA flash-attn
  kernel** does Q·Kᵀ and softmax·V. There is **zero dp4a anywhere in the tbq4 files.**
- **Launcher knobs:** `nstages=0, V_is_K_view=false, need_f16_K=false, need_f16_V=false` — raw
  tbq4 bytes pass through to the kernel; the kernel owns dequant.

This is the crux for our tq3_0 port: **they reach ~q8 speed without an integer dot.** q8_0's FA
path also dequants to FP16 and runs FP16 MMA; the tbq4 match is because the marginal cost over q8
is only a 16-entry `__constant__` LUT index per nibble, fully hidden by the MMA. We do **not**
need to find an int4/dp4a trick for tq3_0 — we need to make our dequant fused + free the same way.

---

## 5. Accuracy / rotation

- **FWHT (Walsh–Hadamard) rotation with two ±1 sign masks** (`d_tbq4_wht_s1`, `d_tbq4_wht_s2`),
  scaled by 1/√128. K is quantized in the rotated domain; Q is rotated forward at load
  (`tbq4_rotate_Q_tile`); V accumulates rotated and the **output is inverse-rotated after the
  kernel**. So the rotation is symmetric across Q/K and undone once on the final O.
- The rotation is what lets a *coarse* 16-centroid 4-bit codebook stay near-lossless — it
  Gaussianizes the per-128 block so the fixed centroid table fits.
- Interaction with Q-int8: because Q is rotated and kept FP16, there's no int8 path to worry
  about; the rotation does NOT force or block any Q quantization here.

---

## 6. tbq4 vs OUR tq3_0 — what transfers, what's format-specific

| Aspect | tbq4_0 (theirs) | tq3_0 (ours) |
|---|---|---|
| Bits / block | 4-bit, 128 elems, 66 B → 4.125 bpw | 3-bit, 32 elems, 14 B → 3.5 bpw |
| Codebook | 16 centroids, Lloyd-Max, `__constant__` | 8 centroids, Lloyd-Max, `__constant__` |
| Rotation | FWHT + 2 ±1 sign masks, 1/√128 | FWHT |
| Packing | 2 nibbles/byte | 3-bit packed (not nibble-aligned) |
| Dot | FP16 MMA (no dp4a) | (target) FP16 MMA |

**Transfers directly (format-agnostic):**
1. **Fused dequant-in-the-FA-tile-loader** — read raw blocks, emit `half2` straight into the MMA
   shared tile. No materialized FP16 KV in HBM. (The single biggest lever.)
2. **Column-group thread distribution** — map one thread → one output `half2`/element across
   rows for 100% load-lane utilization.
3. **Centroid LUT in `__constant__`, indexed inline, no per-tile precompute** — our 8-entry tq3
   LUT is even cheaper than their 16.
4. **Reuse the existing FP16 MMA kernel** (`flash_attn_ext_f16`) for the actual matmul; only swap
   the tile loader. Launcher pattern: force `nstages=0`, `need_f16_K/V=false`, pass raw bytes.
5. **Inverse-rotate the output once after the kernel**, rotate Q forward at load — both are
   FWHT-family and we already do FWHT for tq3_0.
6. **Staged 16-byte-aligned scratch** (`tbq4_staging_row_bytes` + 4-byte `int` raw copies) as the
   `cp.async`-safe path when the block stride isn't 16-aligned — **directly relevant to tq3_0's
   14-byte blocks** (also not 16-aligned), so we'd want this staging variant, not naive wide loads.

**Format-specific (must adapt for 3-bit/32):**
- Nibble unpack (`byte & 0xF`, `byte >> 4`) → must become **3-bit cross-byte unpacking** for
  tq3_0's 14B/32 layout (indices straddle byte boundaries; no clean 2-per-byte). This is the one
  non-trivial rewrite.
- `elems_per_block = 64` (= 128/2) → for tq3_0 it's 32 elems over 14 bytes; the thread→element
  map and the staging int-count math (`ints_per_row`) change accordingly. 14 B isn't a multiple
  of 4, so the `int`-chunk staging copy needs a tail-byte guard.
- `QK_TBQ4 128` / `1/√128` rotation scale → ours is per-32 (or our existing tq3 FWHT block size).

---

## 7. Measured numbers + hardware (from repo description + README, AS QUOTED)

- **"82+ tok/s with lossless 4.25 bpv KV cache at 200K context on RTX 4090"** (repo description).
- README: **"80–179 tok/s decode (325 effective)"**, **"82–93 tok/s"**.
- **262K context on 24GB VRAM** vs upstream's ~131K limit; **tensor sharing saves 682 MiB**.
- Baseline framing "~99% of q8_0 @262K on 24GB" comes from the repo's own positioning; the exact
  "37% → <1.5% dequant tax" phrasing was **NOT found verbatim** in README, top-of-file comments,
  or the visible commit list (see Caveats). The mechanism that *produces* that result — fused
  dequant, no HBM FP16 KV — is confirmed in code.

## License / attribution

- **MIT** (llama.cpp upstream license; fork shows an MIT badge). Reusing the kernel is permitted
  **with the MIT copyright notice + permission notice retained**. Attribute
  `Indras-Mirror/llama.cpp-turboq-mtp` (and upstream ggml-org/llama.cpp) in any ported file
  header. Per our external-repo rules: if we later upstream a port, open an issue first and keep
  the attribution explicit.

## Caveats / what I could NOT confirm verbatim

- The exact phrase **"37% → <1.5% dequant tax"** was not located in README / file comments /
  commit titles via WebFetch (GitHub's rendered search returned 0 and WebFetch summarizes long
  pages). It may live in a PR body, a docs/ file not at the guessed path (`docs/turboq.md` = 404),
  or an issue. The *code* substantiates the claim's mechanism regardless.
- WebFetch paraphrases long files; the ±1 contents of `d_tbq4_wht_s1/s2` were returned but should
  be re-pulled exactly from `tbq4-cuda.cuh` before copying into a port (sign tables must be bit-exact).

## Source URLs (all fetched 2026-06-22)

- Repo: https://github.com/Indras-Mirror/llama.cpp-turboq-mtp
- Kernel: https://raw.githubusercontent.com/Indras-Mirror/llama.cpp-turboq-mtp/master/ggml/src/ggml-cuda/fattn-mma-tbq4.cuh
- Launcher: https://raw.githubusercontent.com/Indras-Mirror/llama.cpp-turboq-mtp/master/ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh
- Centroids/WHT: https://raw.githubusercontent.com/Indras-Mirror/llama.cpp-turboq-mtp/master/ggml/src/ggml-cuda/tbq4-cuda.cuh
- Format struct: https://raw.githubusercontent.com/Indras-Mirror/llama.cpp-turboq-mtp/master/ggml/src/ggml-common.h
- README: https://raw.githubusercontent.com/Indras-Mirror/llama.cpp-turboq-mtp/master/README.md
