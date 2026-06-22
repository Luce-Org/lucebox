// Unit tests for KvFlashPager serialize/deserialize round-trip and critical-
// chunk pinning, on a CPU ggml backend (no CUDA).
#include "../src/common/kvflash_pager.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

// Small synthetic cache: head_dim=4, pool=512 tok, n_head_kv=2, 1 layer,
// chunk_tokens=64 -> 8 blocks. sink=1 + tail=4 -> min_pool = (1+4+2)*64=448 <= 512.
struct Harness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    Harness(int head_dim, int pool, int n_head_kv, int n_layer) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < n_layer; l++) {
            ggml_tensor * kt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, pool, n_head_kv);
            ggml_tensor * vt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, pool, n_head_kv);
            k.push_back(kt); v.push_back(vt);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    KvFlashConfig cfg(int pool, int chunk) const {
        KvFlashConfig c; c.pool_tokens = pool; c.chunk_tokens = chunk;
        c.sink_chunks = 1; c.tail_window_chunks = 4; return c;
    }
};

static void test_serialize_roundtrip() {
    const int head_dim = 4, pool = 512, nkv = 2, nlayer = 1, chunk = 64;
    Harness A(head_dim, pool, nkv, nlayer);
    KvFlashPager pa;
    expect(pa.attach(A.cfg(pool, chunk), A.k, A.v), "attach A");
    // Map the whole pool and stamp a recognizable ramp into the K/V buffers.
    expect(pa.alloc_span(0, pool), "alloc_span fills pool A");
    const size_t kbytes = ggml_nbytes(A.k[0]);
    std::vector<uint8_t> ramp(kbytes);
    for (size_t i = 0; i < kbytes; i++) ramp[i] = (uint8_t)(i * 31 + 7);
    ggml_backend_tensor_set(A.k[0], ramp.data(), 0, kbytes);
    ggml_backend_tensor_set(A.v[0], ramp.data(), 0, kbytes);

    std::vector<uint8_t> blob = pa.serialize();
    expect(!blob.empty(), "serialize produces a blob");

    // Deserialize into a fresh pager over fresh (zeroed) tensors.
    Harness B(head_dim, pool, nkv, nlayer);
    KvFlashPager pb;
    expect(pb.attach(B.cfg(pool, chunk), B.k, B.v), "attach B");
    expect(pb.deserialize(blob.data(), blob.size()), "deserialize succeeds");

    // The restored pool must contain the same KV bytes as the source.
    std::vector<uint8_t> kb(kbytes), vb(kbytes);
    ggml_backend_tensor_get(B.k[0], kb.data(), 0, kbytes);
    ggml_backend_tensor_get(B.v[0], vb.data(), 0, kbytes);
    expect(kb == ramp, "K restored byte-identical after round-trip");
    expect(vb == ramp, "V restored byte-identical after round-trip");

    // A blob from a mismatched layout must be rejected (header validation).
    KvFlashConfig bad = B.cfg(pool, chunk);
    blob[8] ^= 0xFF;  // corrupt a header byte
    KvFlashPager pc;
    Harness C(head_dim, pool, nkv, nlayer);
    expect(pc.attach(C.cfg(pool, chunk), C.k, C.v), "attach C");
    expect(!pc.deserialize(blob.data(), blob.size()), "corrupt-header blob rejected");
    (void)bad;
    std::printf("ok: serialize/deserialize round-trip + header guard\n");
}

// ── New tests for the pooled-prefill snapshot fix ────────────────────────
//
// RED: floor_to_chunk helper and serialize(max_chunks=k) partial-serialize
// do NOT exist yet -> test_floor_to_chunk and test_serialize_partial MUST fail
// to compile / link / assert before the fix lands.

static void test_floor_to_chunk() {
    // floor_to_chunk(pos, chunk) must return the largest multiple of chunk_tokens
    // that is <= pos — the chunk-aligned boundary used by the pooled-prefill
    // snapshot path.
    expect(dflash::common::floor_to_chunk(0, 64)   == 0,  "floor(0,64)==0");
    expect(dflash::common::floor_to_chunk(63, 64)  == 0,  "floor(63,64)==0");
    expect(dflash::common::floor_to_chunk(64, 64)  == 64, "floor(64,64)==64");
    expect(dflash::common::floor_to_chunk(65, 64)  == 64, "floor(65,64)==64");
    expect(dflash::common::floor_to_chunk(127, 64) == 64, "floor(127,64)==64");
    expect(dflash::common::floor_to_chunk(128, 64) == 128,"floor(128,64)==128");
    // snap_pos=34077, chunk=64: floor = 34048 (531*64)
    expect(dflash::common::floor_to_chunk(34077, 64) == 34048, "floor(34077,64)==34048");
    std::printf("ok: floor_to_chunk\n");
}

static void test_serialize_partial() {
    // Pool 8 chunks (8*64=512 tokens). Fill all 8, then partial-serialize
    // only the first k=5 chunks. Deserialize into a fresh pager; assert:
    //   - exactly 5 chunks are restored
    //   - cur_pos == 5*64 == 320
    //   - KV bytes for chunks 0..4 are bit-identical to source
    //   - the pager reports 5 chunks (not 8)
    const int head_dim = 4, pool = 512, nkv = 2, nlayer = 1, chunk = 64;
    const int k = 5;  // only serialize first k chunks
    Harness A(head_dim, pool, nkv, nlayer);
    KvFlashPager pa;
    expect(pa.attach(A.cfg(pool, chunk), A.k, A.v), "partial: attach A");
    expect(pa.alloc_span(0, pool), "partial: alloc_span fills pool A");

    // Stamp a distinct ramp into each chunk's region so we can tell them apart.
    const size_t bytes_per_chunk = (size_t)chunk * (size_t)head_dim * 2 * nkv; // F16=2 bytes
    const size_t total_bytes = ggml_nbytes(A.k[0]);
    std::vector<uint8_t> ramp(total_bytes);
    for (size_t i = 0; i < total_bytes; i++) ramp[i] = (uint8_t)((i * 37 + 11) & 0xFF);
    ggml_backend_tensor_set(A.k[0], ramp.data(), 0, total_bytes);
    ggml_backend_tensor_set(A.v[0], ramp.data(), 0, total_bytes);

    // Partial serialize: max_chunks=k means only chunks [0, k) go into the blob.
    std::vector<uint8_t> blob = pa.serialize(k);
    expect(!blob.empty(), "partial: serialize(k) produces a blob");

    // Deserialize into a fresh pager.
    Harness B(head_dim, pool, nkv, nlayer);
    KvFlashPager pb;
    expect(pb.attach(B.cfg(pool, chunk), B.k, B.v), "partial: attach B");
    expect(pb.deserialize(blob.data(), blob.size()), "partial: deserialize succeeds");

    // The pager must report exactly k chunks (not the full 8).
    expect(pb.n_chunks() == k, "partial: n_chunks == k after restore");

    // Verify KV bytes for chunks 0..k-1 are restored identically.
    // Physical layout: chunk c is at pool slot c (identity for a fresh pager
    // after alloc_span from position 0).  The K tensor bytes at
    // offset [c*chunk*head_dim*2 .. (c+1)*chunk*head_dim*2) for each head.
    // We just verify the first k chunks' byte regions match the source.
    std::vector<uint8_t> kb(total_bytes, 0), vb(total_bytes, 0);
    ggml_backend_tensor_get(B.k[0], kb.data(), 0, total_bytes);
    ggml_backend_tensor_get(B.v[0], vb.data(), 0, total_bytes);

    // With identity slot assignment (chunk c -> block c), the first k*chunk rows
    // of each head should be identical.  head stride = pool*head_dim*2.
    const size_t row_bytes = (size_t)head_dim * 2;  // F16
    const size_t head_stride = (size_t)pool * row_bytes;  // nb[2] stride
    bool k_ok = true, v_ok = true;
    for (int h = 0; h < nkv; h++) {
        for (int r = 0; r < k * chunk; r++) {
            const size_t off = (size_t)h * head_stride + (size_t)r * row_bytes;
            for (size_t b = 0; b < row_bytes; b++) {
                if (kb[off + b] != ramp[off + b]) { k_ok = false; break; }
                if (vb[off + b] != ramp[off + b]) { v_ok = false; break; }
            }
        }
    }
    expect(k_ok, "partial: K bytes [0..k*chunk) restored bit-identical");
    expect(v_ok, "partial: V bytes [0..k*chunk) restored bit-identical");

    std::printf("ok: serialize(max_chunks=%d) partial round-trip\n", k);
}

static void test_pinning() {
    const int head_dim = 4, pool = 512, nkv = 2, nlayer = 1, chunk = 64;
    Harness H(head_dim, pool, nkv, nlayer);
    KvFlashPager p;
    expect(p.attach(H.cfg(pool, chunk), H.k, H.v), "attach pin");

    // No pins -> nothing pinned.
    expect(!p.is_pinned(2), "pre: chunk 2 unpinned");

    // Pin a single mid chunk (tokens 128..191 -> chunk 2). 8 blocks, sink1+tail4
    // +pinned1+2 = 8 <= 8 -> allowed.
    p.pin_range(128, 191);
    expect(p.is_pinned(2), "chunk 2 pinned");
    expect(!p.is_pinned(1), "chunk 1 not pinned");
    expect(!p.is_pinned(3), "chunk 3 not pinned");

    // unpin clears it.
    p.unpin_all();
    expect(!p.is_pinned(2), "unpin_all clears chunk 2");

    // Deadlock guard: pinning the whole pool (8 chunks) would leave no evictable
    // block (1+4+8+2 = 15 > 8) -> refused, nothing pinned.
    p.pin_range(0, pool - 1);
    expect(!p.is_pinned(0) && !p.is_pinned(4) && !p.is_pinned(7),
           "over-pin refused by deadlock guard");

    // reset() also clears pins.
    p.pin_range(128, 191);
    expect(p.is_pinned(2), "re-pin chunk 2");
    p.reset();
    expect(!p.is_pinned(2), "reset clears pins");
    std::printf("ok: pinning + deadlock guard + reset\n");
}

int main() {
    test_serialize_roundtrip();
    test_pinning();
    test_floor_to_chunk();
    test_serialize_partial();
    std::printf("PASS: kvflash pager (serde + pinning + partial-serialize)\n");
    return 0;
}
