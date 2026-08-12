// Loads DeepSeek V4 Flash from a GGUF file.
//
// Tensor naming follows the ds4 GGUF conversion:
//   token_embd.weight, output_norm.weight, output.weight,
//   output_hc_base.weight, output_hc_fn.weight, output_hc_scale.weight
//   blk.<i>.attn_norm.weight, blk.<i>.attn_q_a.weight, attn_q_a_norm,
//   attn_q_b, attn_kv, attn_kv_a_norm, attn_sinks, attn_output_a, attn_output_b,
//   attn_compressor_{ape,kv,gate,norm}, indexer.{attn_q_b, proj},
//   indexer_compressor_{ape,kv,gate,norm},
//   hc_attn_fn, hc_attn_scale, hc_attn_base,
//   ffn_norm, ffn_gate_inp, exp_probs_b (bias), ffn_gate_tid2eid,
//   ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//   ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
//   hc_ffn_fn, hc_ffn_scale, hc_ffn_base

#include "deepseek4_internal.h"
#include "internal.h"
#include "dflash27b.h"
#include "common/gguf_bounds.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_types.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>   // SIZE_MAX, used by the portable checked-size helpers below
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <fcntl.h>

extern "C" bool ggml_backend_cuda_buffer_is_managed(ggml_backend_buffer_t buffer);

// Runtime decode registration for GGML_TYPE_Q3_1_ROCMFP3_MIX (105). Defined in
// ggml-cuda/rocmfp3_mix.cu; declared here (no HIP include in the loader) so the
// deepseek4 loader can stage per-expert codebooks/modes to device and register
// each fused down-expert tensor's base pointer + stride with the decoder.
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
// Remove a qtype-105/106 tensor's registry entry and free its device side-data.
// Called from free_deepseek4_weights before the GPU buffer is released so
// stale base-pointer ranges can't survive an unload/reload.
extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base);
extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

struct DS4Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool is_fd_open() const {
#if defined(_WIN32)
        return hFile != INVALID_HANDLE_VALUE;
#else
        return fd >= 0;
#endif
    }
    void close_fd() {
#if defined(_WIN32)
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        // Convert UTF-8 path to UTF-16 for CreateFileW — CreateFileA uses the
        // active ANSI code page and fails on non-ASCII paths (e.g. CJK chars).
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen == 0) {
            err = "MultiByteToWideChar: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        std::wstring wpath(wlen, L'\0');
        if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen) == 0) {
            err = "MultiByteToWideChar (fill): " + path + ": error " + std::to_string(GetLastError());
            return false;
        }

        hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileW: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingW: error " + std::to_string(GetLastError());
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            CloseHandle(hMap); hMap = nullptr;
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + " " + strerror(errno); return false; }
        struct stat st;
        if (fstat(fd, &st) < 0) { err = "fstat"; ::close(fd); fd = -1; return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap"; addr = nullptr; ::close(fd); fd = -1; return false; }
#endif
        return true;
    }
    void close_map() {
#if defined(_WIN32)
        if (addr) { UnmapViewOfFile(addr); addr = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (addr) { ::munmap(addr, len); addr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }
};

uint32_t get_u32_or(gguf_context * g, const char * key, uint32_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const uint32_t *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_u32(g, id);
}

uint64_t get_u64_or(gguf_context * g, const char * key, uint64_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    // Handle both u32 and u64 storage in GGUF
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) {
        return (uint64_t)gguf_get_val_u32(g, id);
    }
    return (uint64_t)gguf_get_val_u64(g, id);
}

float get_f32_or(gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const float *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

bool get_u32_arr(gguf_context * g, const char * key, std::vector<uint32_t> & out,
                 std::string * err = nullptr) {
    out.clear();
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return true;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) {
        if (err) *err = std::string(key) + " must be an array";
        return false;
    }
    const enum gguf_type arr_type = gguf_get_arr_type(g, id);
    if (arr_type != GGUF_TYPE_INT32 && arr_type != GGUF_TYPE_UINT32) {
        if (err) {
            *err = std::string(key) + " array element type must be i32 or u32";
        }
        return false;
    }

    const size_t n = gguf_get_arr_n(g, id);
    const void * raw = gguf_get_arr_data(g, id);
    out.resize(n);
    if (arr_type == GGUF_TYPE_INT32) {
        const int32_t * vals = static_cast<const int32_t *>(raw);
        for (size_t i = 0; i < n; ++i) {
            if (vals[i] < 0) {
                if (err) *err = std::string(key) + " array values must be non-negative";
                out.clear();
                return false;
            }
            out[i] = (uint32_t)vals[i];
        }
    } else {
        const uint32_t * vals = static_cast<const uint32_t *>(raw);
        out.assign(vals, vals + n);
    }
    return true;
}

ggml_tensor * find_tensor(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

static size_t align_up_size(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    return r == 0 ? x : x + (a - r);
}

static bool parse_block_tensor_name(const char * name, int & layer_id) {
    const char prefix[] = "blk.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0 || v > INT32_MAX) return false;
    layer_id = (int)v;
    return true;
}

static bool is_expert_tensor(const char * name) {
    return std::strstr(name, "ffn_gate_exps") != nullptr ||
           std::strstr(name, "ffn_up_exps") != nullptr ||
           std::strstr(name, "ffn_down_exps") != nullptr;
}

static bool should_keep_ds4_tensor(const char * name,
                                   const TargetLoadPlan & plan) {
    int layer_id = -1;
    if (plan.expert_metadata_only) {
        return parse_block_tensor_name(name, layer_id) &&
               layer_id >= plan.layer_begin &&
               layer_id < plan.layer_end &&
               is_expert_tensor(name);
    }

    // Global tensors
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "output_norm.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0 ||
        std::strcmp(name, "output_hc_base.weight") == 0 ||
        std::strcmp(name, "output_hc_fn.weight") == 0 ||
        std::strcmp(name, "output_hc_scale.weight") == 0) {
        return plan.load_output;
    }

    if (!parse_block_tensor_name(name, layer_id)) return false;
    return layer_id >= plan.layer_begin && layer_id < plan.layer_end;
}

static bool should_upload_ds4_tensor(const char * name,
                                     const TargetLoadPlan & plan) {
    if (!should_keep_ds4_tensor(name, plan)) return false;
    if (plan.expert_metadata_only) return false;
    // token_embd stays on CPU for embedding lookup
    if (std::strcmp(name, "token_embd.weight") == 0) return false;
    return !(plan.skip_expert_tensors && is_expert_tensor(name));
}

static int ds4_dense_tp_mask() {
    const char * value = std::getenv("DFLASH_DS4_DENSE_TP_MASK");
    if (!value || !value[0]) return 0;
    return std::max(0, std::atoi(value));
}

static bool should_split_ds4_dense_tensor(const char * name, int mask) {
    if (mask == 0 || !name) return false;
    // Each selected tensor is consumed as src0 of a plain 2-D MUL_MAT. Expert
    // tensors and the grouped output-A view are deliberately excluded: the
    // CUDA/HIP split-buffer implementation cannot split 3-D weight views.
    if ((mask & 1) && std::strcmp(name, "output.weight") == 0) return true;
    if ((mask & 2) && std::strstr(name, ".attn_q_b.weight") &&
        !std::strstr(name, ".indexer.attn_q_b.weight")) return true;
    if ((mask & 4) && std::strstr(name, ".attn_output_b.weight")) return true;
    if ((mask & 8) && (std::strstr(name, ".attn_q_a.weight") ||
                       std::strstr(name, ".attn_kv.weight") ||
                       std::strstr(name, ".indexer.attn_q_b.weight") ||
                       std::strstr(name, ".attn_compressor_kv.weight") ||
                       std::strstr(name, ".attn_compressor_gate.weight") ||
                       std::strstr(name, ".indexer_compressor_kv.weight") ||
                       std::strstr(name, ".indexer_compressor_gate.weight"))) {
        return true;
    }
    return false;
}

struct DS4TensorAlloc {
    ggml_tensor * tensor = nullptr;
    size_t tensor_offset = 0;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
    bool upload_to_backend = true;
    bool dense_split = false;
};
}  // namespace

// ─── Compute per-layer compression ratios (matches ds4.c logic) ─────────
static std::vector<uint32_t> compute_compress_ratios(int n_layer) {
    std::vector<uint32_t> ratios(n_layer, 0);
    for (int il = 0; il < n_layer; il++) {
        if (il < 2) {
            ratios[il] = 0;  // First 2 layers: no compression
        } else if ((il & 1) == 0) {
            ratios[il] = 4;  // Even layers ≥2: ratio 4
        } else {
            ratios[il] = 128;  // Odd layers ≥2: ratio 128
        }
    }
    return ratios;
}

namespace {
constexpr int DS4_QTYPE_ROCMFP3_MIX = 105;  // GGML_TYPE_Q3_1_ROCMFP3_MIX


// 64-bit-safe forward seek. std::fseek takes long, which is 32 bits on Windows, so a
// size_t payload that passed the checked arithmetic above it could still truncate in
// the cast. Current dimension ceilings keep sidecar payloads KB-scale, but the seek
// must not be the one unchecked conversion in a chain built on checked arithmetic.
static bool ds4_seek_fwd(std::FILE * f, size_t bytes) {
#ifdef _WIN32
    return _fseeki64(f, (long long) bytes, SEEK_CUR) == 0;
#else
    return fseeko(f, (off_t) bytes, SEEK_CUR) == 0;
#endif
}

// Read the "<gguf>.p4mix.bin" sidecar (produced on the H100) and register each
// qtype-105 fused down-expert tensor's device base + per-expert codebooks/modes
// with the CUDA/HIP decoder. No-op for uniform (qtype-104) models or when no
// sidecar is present. Called once after weights are uploaded, when the tensor
// data pointers (which the mul_mat_id -> to_fp16 fallback consults) are final.
// Returns false (and the caller must fail the load) if this model carries
// qtype-105 down-experts that cannot be fully and validly registered — an
// unregistered qtype-105 tensor has no working decode path (the to_fp16 shim
// GGML_ABORTs), so a partial/garbage registration would only turn into a
// delayed crash or silently-wrong output at first decode.
// Sanity bounds for a corrupt/hostile sidecar entry, checked BEFORE any buffer
// is sized from its fields. Far above any real DeepSeek MoE (256–512 experts,
// n_ff ~ 2k–18k) but small enough that E*C*K*2 bytes can't overflow size_t.
constexpr uint32_t DS4_P4MIX_MAX_EXPERTS = 1u << 16;   // 65536
constexpr uint32_t DS4_P4MIX_MAX_DIM     = 1u << 20;   // 1,048,576
constexpr uint32_t DS4_P4MIX_C           = 2;          // codebooks per expert
constexpr uint32_t DS4_P4MIX_K           = 8;          // levels per codebook
constexpr uint32_t DS4_P4MIX_QK          = 32;         // MIX_QK block width (rocmfp3_mix.cu)
constexpr uint8_t  DS4_P4MIX_MAX_MODE    = 1;          // 0 = fixed, 1 = adaptive

// ---- learned-codebook sidecars: embedded in the GGUF, or a loose file beside it ----
// The adaptive mix qtypes keep per-expert codebooks out of band because a ggml block has
// nowhere to put a learned table. Shipping that as a second file makes the model a
// two-part download whose halves can be separated -- and it is REQUIRED, so a model
// without it is undecodable. Prefer a copy embedded in the GGUF KV block; fall back to
// the loose file so already-published two-file artifacts keep loading unchanged.
//
// fmemopen is what keeps this small: the callers' existing FILE* parsers read the
// embedded bytes with no change, so the sidecar layout still has exactly one parser.
// A read-only FILE* over a memory buffer, portably.
//
// fmemopen is POSIX and absent from the MSVC CRT, and this file carries real Windows support
// (several _WIN32 branches above), so calling it unconditionally would break that build. The
// fallback writes the blob to a tmpfile and rewinds: one extra copy of ~375 KB, once per
// load, against keeping a single FILE*-based parser for the layout. Duplicating the parsers
// to take byte buffers instead would trade a trivial cost for exactly the drift risk the
// single-parser design exists to avoid.
static FILE * ds4_fopen_memory(std::vector<uint8_t> & backing) {
#if defined(_WIN32)
    FILE * f = std::tmpfile();
    if (!f) {
        return nullptr;
    }
    if (!backing.empty() &&
        std::fwrite(backing.data(), 1, backing.size(), f) != backing.size()) {
        std::fclose(f);
        return nullptr;
    }
    std::rewind(f);
    return f;
#else
    return fmemopen(backing.data(), backing.size(), "rb");
#endif
}

// Checked size arithmetic, portably. The __builtin_*_overflow forms these replace are
// GCC/Clang-only; MSVC has neither, and the sidecar parsers use them on every header field
// that sizes a read or an allocation, so they cannot simply be dropped on that compiler.
static inline bool ds4_size_mul_overflow(size_t a, size_t b, size_t * out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > (SIZE_MAX / a)) {
        return true;
    }
    *out = a * b;
    return false;
#endif
}

static inline bool ds4_size_add_overflow(size_t a, size_t b, size_t * out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if (b > (SIZE_MAX - a)) {
        return true;
    }
    *out = a + b;
    return false;
#endif
}

static FILE * ds4_open_sidecar(const std::string & gguf_path,
                               const char * kv_key,
                               const char * suffix,
                               std::vector<uint8_t> & backing) {
    struct gguf_init_params gip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gip);
    if (g) {
        const int64_t id = gguf_find_key(g, kv_key);
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(g, id) == GGUF_TYPE_UINT8) {
            const size_t n = gguf_get_arr_n(g, id);
            const uint8_t * d = (const uint8_t *) gguf_get_arr_data(g, id);
            backing.assign(d, d + n);
            gguf_free(g);
            std::fprintf(stderr, "[deepseek4] %s: using codebooks embedded in the GGUF "
                         "(%zu bytes, no sidecar file needed)\n", kv_key, backing.size());
            return ds4_fopen_memory(backing);
        }
        gguf_free(g);
    }
    const std::string p = gguf_path + suffix;
    return std::fopen(p.c_str(), "rb");
}

static bool ds4_register_p4mix_sidecar(const std::string & gguf_path,
                                       const TargetLoadPlan & plan,
                                       DeepSeek4Weights & out) {
    // required[layer] == true for a qtype-105 down-expert resident on this shard.
    std::vector<bool> required(out.layers.size(), false);
    int n_qtype105 = 0;
    for (size_t li = 0; li < out.layers.size(); ++li) {
        const ggml_tensor * dt = out.layers[li].ffn_down_exps;
        if (dt && (int) dt->type == DS4_QTYPE_ROCMFP3_MIX) { required[li] = true; n_qtype105++; }
    }
    if (n_qtype105 == 0) return true;  // uniform (qtype-104) model — nothing to do

    // Metadata-only / expert-skipping shards keep the expert tensor allocated but
    // never upload its weights (see should_upload_ds4_tensor), so its device base
    // is unfilled. Such shards don't decode experts locally, so skip registration
    // rather than register a garbage base. (Hybrid/cold placement of a qtype-105
    // model — which likewise sets skip_expert_tensors — is rejected earlier in
    // DeepSeek4Backend::init_hybrid_model, since it has no working 105 decode.)
    if (plan.skip_expert_tensors) {
        std::fprintf(stderr, "[deepseek4] qtype-105 down-experts not resident on this "
                     "shard (skip_expert_tensors) — fused decode disabled here\n");
        return true;
    }

    const std::string sc_path = gguf_path + ".p4mix.bin";
    std::vector<uint8_t> sc_embedded;  // outlives f when the blob came from GGUF KV
    FILE * f = ds4_open_sidecar(gguf_path, "deepseek4.p4mix.sidecar", ".p4mix.bin", sc_embedded);
    if (!f) {
        std::fprintf(stderr, "[deepseek4] qtype-105 down-experts require sidecar but it "
                     "is missing: %s\n", sc_path.c_str());
        return false;
    }
    char magic[8];
    uint32_t n_layers = 0, reserved = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "P4MIXv1\0", 8) != 0 ||
        std::fread(&n_layers, 4, 1, f) != 1 ||
        std::fread(&reserved, 4, 1, f) != 1) {
        std::fprintf(stderr, "[deepseek4] bad p4mix sidecar header: %s\n", sc_path.c_str());
        std::fclose(f);
        return false;
    }
    (void) reserved;

    std::vector<bool> done(out.layers.size(), false);  // resident layers registered
    std::vector<const void *> registered_bases;        // for unwind on failure
    bool ok = true;
    for (uint32_t i = 0; i < n_layers && ok; i++) {
        uint32_t hdr[6];
        if (std::fread(hdr, 4, 6, f) != 6) {
            std::fprintf(stderr, "[deepseek4] truncated p4mix sidecar (entry %u header)\n", i);
            ok = false; break;
        }
        const uint32_t layer = hdr[0], E = hdr[1], odim = hdr[2], idim = hdr[3],
                       C = hdr[4], K = hdr[5];

        // ── Structural validation BEFORE sizing any allocation (finding: reject
        //    oversized/overflowing dims without allocating) ──
        if (C != DS4_P4MIX_C || K != DS4_P4MIX_K) {
            std::fprintf(stderr, "[deepseek4] p4mix entry %u (layer %u) unexpected C=%u K=%u\n",
                         i, layer, C, K);
            ok = false; break;
        }
        if (E == 0 || E > DS4_P4MIX_MAX_EXPERTS ||
            odim == 0 || odim > DS4_P4MIX_MAX_DIM ||
            idim == 0 || idim > DS4_P4MIX_MAX_DIM) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u dims out of range: E=%u out=%u in=%u\n",
                         layer, E, odim, idim);
            ok = false; break;
        }
        // Checked payload size = modes(E) + rots(E) + books(E*C*K * 2 bytes).
        size_t book_elems = 0, book_bytes = 0, payload = 0;
        if (ds4_size_mul_overflow((size_t) E, (size_t) (C * K), &book_elems) ||
            ds4_size_mul_overflow(book_elems, (size_t) sizeof(uint16_t), &book_bytes) ||
            ds4_size_add_overflow((size_t) E + (size_t) E, book_bytes, &payload)) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u payload size overflow\n", layer);
            ok = false; break;
        }

        // Layers outside this shard's range aren't resident — skip their payload
        // without allocating or reading it.
        ggml_tensor * dt = (layer < out.layers.size())
                               ? out.layers[layer].ffn_down_exps : nullptr;
        const bool resident105 = dt && (int) dt->type == DS4_QTYPE_ROCMFP3_MIX;
        if (!resident105) {
            if (!ds4_seek_fwd(f, payload)) {
                std::fprintf(stderr, "[deepseek4] p4mix seek past layer %u failed\n", layer);
                ok = false; break;
            }
            continue;
        }

        // ── Dimensions must match the resident tensor so routed expert ids and
        //    decode offsets stay in-bounds. Tensor ne = [in, out, E]. ──
        if ((int64_t) E != dt->ne[2] || (int64_t) odim != dt->ne[1] ||
            (int64_t) idim != dt->ne[0]) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u dim mismatch: sidecar "
                         "E=%u out=%u in=%u vs tensor ne2=%lld ne1=%lld ne0=%lld\n",
                         layer, E, odim, idim, (long long) dt->ne[2],
                         (long long) dt->ne[1], (long long) dt->ne[0]);
            ok = false; break;
        }
        if (idim % DS4_P4MIX_QK != 0) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u in=%u not a multiple of %u\n",
                         layer, idim, DS4_P4MIX_QK);
            ok = false; break;
        }
        if (!dt->data) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u expert data not resident\n", layer);
            ok = false; break;
        }
        // Reject a duplicate entry for an already-registered layer (a repeated
        // layer could otherwise mask an omitted one and still match the count).
        if (done[layer]) {
            std::fprintf(stderr, "[deepseek4] p4mix layer %u appears more than once\n", layer);
            ok = false; break;
        }

        // Safe to allocate now: E == ne[2] and bounded above.
        std::vector<uint8_t> modes(E), rots(E);
        std::vector<uint16_t> books(book_elems);
        if (std::fread(modes.data(), 1, E, f) != E ||
            std::fread(rots.data(), 1, E, f) != E ||
            std::fread(books.data(), sizeof(uint16_t), books.size(), f) != books.size()) {
            std::fprintf(stderr, "[deepseek4] truncated p4mix entry (layer %u)\n", layer);
            ok = false; break;
        }
        // Value validation: only modes 0/1 are decoded, and rotation is not yet
        // implemented (the CUDA kernel ignores e.rotations), so a nonzero rotation
        // would silently produce wrong output — reject it.
        for (uint32_t e = 0; e < E && ok; ++e) {
            if (modes[e] > DS4_P4MIX_MAX_MODE) {
                std::fprintf(stderr, "[deepseek4] p4mix layer %u expert %u unsupported mode %u\n",
                             layer, e, modes[e]);
                ok = false;
            } else if (rots[e] != 0) {
                std::fprintf(stderr, "[deepseek4] p4mix layer %u expert %u nonzero rotation %u "
                             "(rotation not implemented)\n", layer, e, rots[e]);
                ok = false;
            }
        }
        if (!ok) break;

        ggml_cuda_rocmfp3_mix_register_host(
            dt->data, dt->nb[2], (int) E, (int) odim, (int) idim,
            books.data(), modes.data(), rots.data());
        registered_bases.push_back(dt->data);
        done[layer] = true;
    }
    std::fclose(f);

    // Every resident qtype-105 layer must be covered exactly once (finding:
    // duplicate/missing-layer sidecars must fail the load).
    if (ok) {
        for (size_t li = 0; li < required.size(); ++li) {
            if (required[li] && !done[li]) {
                std::fprintf(stderr, "[deepseek4] p4mix sidecar missing resident qtype-105 "
                             "layer %zu; refusing to load\n", li);
                ok = false; break;
            }
        }
    }

    if (!ok) {
        // Unwind partial registration so a failed load leaves no live entries or
        // device allocations behind.
        for (const void * b : registered_bases) ggml_cuda_rocmfp3_mix_unregister(b);
        return false;
    }
    std::fprintf(stderr, "[deepseek4] registered %d qtype-105 down-expert layer(s) "
                 "from %s\n", (int) registered_bases.size(), sc_path.c_str());
    return true;
}
constexpr int DS4_QTYPE_ROCMFP2_MIX = 106;  // GGML_TYPE_Q2_1_ROCMFP2_MIX

// Read the "<gguf>.gumix.bin" sidecar and register each qtype-106 gate/up expert
// tensor's device base + per-expert codebooks/modes with the CUDA/HIP decoder.
// Mirrors ds4_register_p4mix_sidecar's contract and validation discipline; three
// things genuinely differ, and each is why this is a separate function rather than
// a parameterised one:
//
//   1. TWO tensors per layer. The serving GGUF stores gate and up separately
//      (ffn_gate_exps / ffn_up_exps), so an entry carries a `surface` selector and
//      a layer is only covered when BOTH halves are registered.
//   2. Codebooks are SHARED between a layer's halves. One pair is fitted per fused
//      expert covering both, so the sidecar repeats it per tensor and each half
//      registers its own device copy. That duplication is deliberate: the registry
//      is keyed by base pointer and frees what it owns, so sharing one buffer
//      between two entries would double-free on unregister.
//   3. No rotation field. The qtype-106 encoder never emits rotation, so the wire
//      omits it entirely rather than carrying a byte that must always be zero. The
//      kernel's register_host still takes a rotations argument, so zeros are
//      passed explicitly.
// ---- "<gguf>.dmix.bin": per-tensor codebooks for DENSE mix-qtype tensors ----
// Registers dense (non-MoE) qtype-105/106 tensors -- the attention stack -- with the
// device decoder. Motivated by measurement rather than symmetry: on attention the learned
// codebook is worth -25%/-40% ppl damage at byte-identical size, and attention is 46.2% of
// per-token read bytes, both measured. Without registration the fused kernels
// return false and every dense mix tensor falls to dequantize->cuBLAS, which reads MORE
// bytes than the f16 it replaced.
//
// Three things differ from the p4mix / gumix readers, and each is why this is its own
// function rather than a parameterisation:
//
//   1. ONE codebook per TENSOR, not per expert. A dense weight is a single matrix, so the
//      wire carries exactly C*K levels and this reader replicates them across the slice
//      count below. Cheap (a few hundred bytes) and it means the kernels need no
//      "dense" special case at all.
//   2. SLICES ARE A VIEW, not storage. attn_output_a is stored 2-D and reshaped to
//      [group_dim, n_lora_o, n_out_group] at graph build time (deepseek4_graph.cpp:2122),
//      so src1->ne[2] > 1 and only the 3-D fused entry point can serve it. The registry
//      is keyed by base pointer with expert = (p - base)/nb02, so registering nslices
//      entries of stride ggml_nbytes/nslices makes the view's slices resolve correctly
//      while the 2-D base still resolves to slice 0. `nslices` comes from the SIDECAR,
//      not from a hardcoded class list here -- the exporter knows which classes the graph
//      views as 3-D, and baking that knowledge into two places is how they drift apart.
//   3. Classes, not surfaces. Five attention weight classes per layer, any subset of
//      which may be a mix qtype in a given artifact.
constexpr uint32_t DS4_DMIX_CLASSES = 5;   // q_a, q_b, kv, output_a, output_b
constexpr uint32_t DS4_DMIX_QK      = 32;  // MIX_QK block width (rocmfp{2,3}_mix.cu)
// The wide load-from-floor staging in mix_block_accum reads a 16 B window per 10/14 B
// block, which only stays in bounds on the final block when the row length is a multiple
// of 128. register_host aborts on violation; checking here gives a message naming the
// tensor instead of a bare abort.
constexpr int64_t  DS4_DMIX_ROW_ALIGN = 128;

static const char * ds4_dmix_class_name(uint32_t cls) {
    switch (cls) {
        case 0: return "attn_q_a";
        case 1: return "attn_q_b";
        case 2: return "attn_kv";
        case 3: return "attn_output_a";
        case 4: return "attn_output_b";
        default: return "?";
    }
}

// Header validation for one dmix sidecar entry, factored out so the malformed cases are
// reachable from a unit test: the parser that calls it is static and reads from a FILE*, so
// covering "rejects mode 2" or "rejects a duplicate" through it would need a fixture model
// and a hand-forged sidecar on disk for every case.
//
// `already_covered` is the caller's covered[layer][cls]; passing it in keeps the duplicate
// rule here with the rest of the entry rules rather than split across two places.
// Returns nullptr when the entry is acceptable, else a short reason for the caller to log.
const char * ds4_dmix_entry_reject_reason_impl(
        uint32_t layer, uint32_t cls, uint32_t qtype, uint32_t nslices,
        uint32_t C, uint32_t K, uint8_t mode,
        uint32_t n_layers, bool already_covered) {
    if (layer >= n_layers || cls >= DS4_DMIX_CLASSES) {
        return "out of range";
    }
    const uint32_t want_K = (qtype == (uint32_t) DS4_QTYPE_ROCMFP3_MIX) ? 8u : 4u;
    if (C != 2 || K != want_K ||
        (qtype != (uint32_t) DS4_QTYPE_ROCMFP3_MIX &&
         qtype != (uint32_t) DS4_QTYPE_ROCMFP2_MIX)) {
        return "unexpected qtype/C/K";
    }
    if (nslices == 0 || nslices > 4096) {
        return "bad nslices";
    }
    // The kernels branch on mode != 0, so an unrecognised future mode would be silently
    // decoded as adaptive against a codebook that means something else.
    if (mode > DS4_P4MIX_MAX_MODE) {
        return "unsupported mode";
    }
    // Exact coverage once per tensor is the loader's contract; the caller enforces the
    // "at least once" half. Without this a second entry silently replaces the first
    // registration -- later codebook wins, earlier one leaks, nothing reported.
    if (already_covered) {
        return "duplicate (layer, class)";
    }
    return nullptr;
}

static ggml_tensor * ds4_dmix_class_tensor(const DeepSeek4Layer & L, uint32_t cls) {
    switch (cls) {
        case 0: return L.attn_q_a;
        case 1: return L.attn_q_b;
        case 2: return L.attn_kv;
        case 3: return L.attn_output_a;
        case 4: return L.attn_output_b;
        default: return nullptr;
    }
}

static bool ds4_register_dmix_sidecar(const std::string & gguf_path,
                                      DeepSeek4Weights & out) {
    // covered[layer][cls]: a dense mix-qtype attention tensor resident on this shard.
    const size_t n_layers_out = out.layers.size();
    std::vector<std::array<bool, DS4_DMIX_CLASSES>> required(
        n_layers_out, std::array<bool, DS4_DMIX_CLASSES>{});
    std::vector<std::array<bool, DS4_DMIX_CLASSES>> covered(
        n_layers_out, std::array<bool, DS4_DMIX_CLASSES>{});
    int n_dense_mix = 0;
    for (size_t li = 0; li < n_layers_out; ++li) {
        for (uint32_t c = 0; c < DS4_DMIX_CLASSES; ++c) {
            const ggml_tensor * t = ds4_dmix_class_tensor(out.layers[li], c);
            if (t && ((int) t->type == DS4_QTYPE_ROCMFP3_MIX ||
                      (int) t->type == DS4_QTYPE_ROCMFP2_MIX)) {
                required[li][c] = true; n_dense_mix++;
            }
        }
    }
    if (n_dense_mix == 0) return true;  // uniform dense (101/104/107) -- nothing to do

    const std::string sc_path = gguf_path + ".dmix.bin";
    std::vector<uint8_t> sc_embedded;  // outlives f when the blob came from GGUF KV
    FILE * f = ds4_open_sidecar(gguf_path, "deepseek4.dmix.sidecar", ".dmix.bin", sc_embedded);
    if (!f) {
        std::fprintf(stderr, "[deepseek4] %d dense mix-qtype attention tensors require "
                     "sidecar but it is missing: %s\n", n_dense_mix, sc_path.c_str());
        return false;
    }
    char magic[8];
    uint32_t n_entries = 0, reserved = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "DMIXs1\0\0", 8) != 0 ||
        std::fread(&n_entries, 4, 1, f) != 1 ||
        std::fread(&reserved, 4, 1, f) != 1) {
        std::fprintf(stderr, "[deepseek4] bad dmix sidecar header: %s\n", sc_path.c_str());
        std::fclose(f);
        return false;
    }

    std::vector<const void *> registered_bases;  // unwound on any failure below
    bool ok = true;
    for (uint32_t i = 0; i < n_entries && ok; ++i) {
        uint32_t layer = 0, cls = 0, qtype = 0, nslices = 0, C = 0, K = 0;
        uint8_t  mode = 0, pad[3] = {0, 0, 0};
        if (std::fread(&layer, 4, 1, f) != 1 || std::fread(&cls, 4, 1, f) != 1 ||
            std::fread(&qtype, 4, 1, f) != 1 || std::fread(&nslices, 4, 1, f) != 1 ||
            std::fread(&C, 4, 1, f) != 1 || std::fread(&K, 4, 1, f) != 1 ||
            std::fread(&mode, 1, 1, f) != 1 || std::fread(pad, 1, 3, f) != 3) {
            std::fprintf(stderr, "[deepseek4] truncated dmix sidecar (entry %u header)\n", i);
            ok = false; break;
        }
        // Bound everything BEFORE it is used to size a read or an allocation. Shared with
        // the unit test so the rules cannot drift from what is covered.
        const bool dup = layer < n_layers_out && cls < DS4_DMIX_CLASSES
                         && covered[layer][cls];
        if (const char * why = ds4_dmix_entry_reject_reason_impl(
                layer, cls, qtype, nslices, C, K, mode, n_layers_out, dup)) {
            std::fprintf(stderr, "[deepseek4] dmix entry %u (L%u %s) rejected: %s "
                         "(qtype=%u C=%u K=%u nslices=%u mode=%u)\n",
                         i, layer,
                         cls < DS4_DMIX_CLASSES ? ds4_dmix_class_name(cls) : "?",
                         why, qtype, C, K, nslices, (unsigned) mode);
            ok = false; break;
        }
        std::vector<uint16_t> book_one((size_t) C * K);
        if (std::fread(book_one.data(), 2, book_one.size(), f) != book_one.size()) {
            std::fprintf(stderr, "[deepseek4] truncated dmix sidecar (entry %u payload)\n", i);
            ok = false; break;
        }

        ggml_tensor * t = ds4_dmix_class_tensor(out.layers[layer], cls);
        if (!t) continue;                       // class not resident on this shard
        if (!required[layer][cls]) {
            // The sidecar describes a tensor that is NOT a mix qtype here. Registering it
            // would attach a codebook to a tensor whose decoder never consults one, so the
            // artifact and the sidecar disagree about the plan -- refuse rather than guess.
            std::fprintf(stderr, "[deepseek4] dmix entry %u describes L%u %s but its type "
                         "is %d, not a mix qtype\n", i, layer, ds4_dmix_class_name(cls),
                         (int) t->type);
            ok = false; break;
        }
        if ((uint32_t) t->type != qtype) {
            std::fprintf(stderr, "[deepseek4] dmix L%u %s qtype mismatch: sidecar %u, "
                         "tensor %d\n", layer, ds4_dmix_class_name(cls), qtype,
                         (int) t->type);
            ok = false; break;
        }
        if (t->ne[0] % DS4_DMIX_ROW_ALIGN != 0) {
            std::fprintf(stderr, "[deepseek4] dmix L%u %s row length %lld is not a multiple "
                         "of %lld; the fused kernel's wide block load would read past the "
                         "tensor\n", layer, ds4_dmix_class_name(cls),
                         (long long) t->ne[0], (long long) DS4_DMIX_ROW_ALIGN);
            ok = false; break;
        }
        if (t->ne[1] % (int64_t) nslices != 0) {
            std::fprintf(stderr, "[deepseek4] dmix L%u %s: nslices=%u does not divide "
                         "ne[1]=%lld\n", layer, ds4_dmix_class_name(cls), nslices,
                         (long long) t->ne[1]);
            ok = false; break;
        }
        const size_t total_bytes = ggml_nbytes(t);
        if (total_bytes % nslices != 0) {
            std::fprintf(stderr, "[deepseek4] dmix L%u %s: %zu bytes not divisible by "
                         "nslices=%u\n", layer, ds4_dmix_class_name(cls), total_bytes,
                         nslices);
            ok = false; break;
        }
        const size_t nb02 = total_bytes / nslices;

        // Replicate the single per-tensor codebook/mode across slices, so the kernels'
        // `codebooks + slice*C*K` and `modes[slice]` striding resolves to the same values
        // for every slice with no dense-specific branch in device code.
        std::vector<uint16_t> books((size_t) nslices * C * K);
        for (uint32_t sl = 0; sl < nslices; ++sl) {
            std::memcpy(books.data() + (size_t) sl * C * K, book_one.data(),
                        book_one.size() * 2);
        }
        std::vector<uint8_t> modes(nslices, mode), rots(nslices, 0);

        const int in_dim  = (int) t->ne[0];
        const int out_dim = (int) (t->ne[1] / (int64_t) nslices);
        if (qtype == (uint32_t) DS4_QTYPE_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_register_host(t->data, nb02, (int) nslices,
                                                out_dim, in_dim, books.data(),
                                                modes.data(), rots.data());
        } else {
            ggml_cuda_rocmfp2_mix_register_host(t->data, nb02, (int) nslices,
                                                out_dim, in_dim, books.data(),
                                                modes.data(), rots.data());
        }
        registered_bases.push_back(t->data);
        covered[layer][cls] = true;
    }
    std::fclose(f);

    // Every resident dense mix tensor must be covered exactly once. An uncovered one
    // silently falls back to dequant->cuBLAS, which is slower than the f16 it replaced
    // AND decodes its adaptive levels with the uniform fallback -- wrong numbers, not just
    // slow. Refuse the load instead.
    if (ok) {
        for (size_t li = 0; li < n_layers_out && ok; ++li) {
            for (uint32_t c = 0; c < DS4_DMIX_CLASSES; ++c) {
                if (required[li][c] && !covered[li][c]) {
                    std::fprintf(stderr, "[deepseek4] dmix sidecar does not cover L%zu %s\n",
                                 li, ds4_dmix_class_name(c));
                    ok = false; break;
                }
            }
        }
    }
    if (!ok) {
        for (const void * b : registered_bases) {
            ggml_cuda_rocmfp3_mix_unregister(b);
            ggml_cuda_rocmfp2_mix_unregister(b);
        }
        return false;
    }
    std::fprintf(stderr, "[deepseek4] dmix: registered %d dense mix-qtype attention "
                 "tensors from %s\n", n_dense_mix, sc_path.c_str());
    return true;
}

constexpr uint32_t DS4_GUMIX_C        = 2;   // codebooks per expert
constexpr uint32_t DS4_GUMIX_K        = 4;   // levels per codebook (2-bit codes)
constexpr uint32_t DS4_GUMIX_SURFACES = 3;   // 0 = gate, 1 = up, 2 = down
constexpr uint32_t DS4_GUMIX_QK       = 32;  // MIX_QK block width (rocmfp2_mix.cu)

static bool ds4_register_gumix_sidecar(const std::string & gguf_path,
                                      const TargetLoadPlan & plan,
                                      DeepSeek4Weights & out) {
    // required[layer][surface]: a qtype-106 gate/up expert resident on this shard.
    const size_t n_layers_out = out.layers.size();
    std::vector<std::array<bool, DS4_GUMIX_SURFACES>> required(
        n_layers_out, std::array<bool, DS4_GUMIX_SURFACES>{false, false, false});
    int n_qtype106 = 0;
    for (size_t li = 0; li < n_layers_out; ++li) {
        const ggml_tensor * ts[DS4_GUMIX_SURFACES] = {
            out.layers[li].ffn_gate_exps, out.layers[li].ffn_up_exps,
            out.layers[li].ffn_down_exps };
        for (uint32_t s = 0; s < DS4_GUMIX_SURFACES; ++s) {
            if (ts[s] && (int) ts[s]->type == DS4_QTYPE_ROCMFP2_MIX) {
                required[li][s] = true; n_qtype106++;
            }
        }
    }
    if (n_qtype106 == 0) return true;  // uniform (qtype-107) gate/up — nothing to do

    if (plan.skip_expert_tensors) {
        std::fprintf(stderr, "[deepseek4] qtype-106 gate/up experts not resident on this "
                     "shard (skip_expert_tensors) — fused decode disabled here\n");
        return true;
    }

    const std::string sc_path = gguf_path + ".gumix.bin";
    std::vector<uint8_t> sc_embedded;  // outlives f when the blob came from GGUF KV
    FILE * f = ds4_open_sidecar(gguf_path, "deepseek4.gumix.sidecar", ".gumix.bin", sc_embedded);
    if (!f) {
        std::fprintf(stderr, "[deepseek4] qtype-106 gate/up experts require sidecar but it "
                     "is missing: %s\n", sc_path.c_str());
        return false;
    }
    char magic[8];
    uint32_t n_entries = 0, reserved = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "GUMIXs1\0", 8) != 0 ||
        std::fread(&n_entries, 4, 1, f) != 1 ||
        std::fread(&reserved, 4, 1, f) != 1) {
        // 's' = split form. A FUSED sidecar (GUMIXv1) describes one tensor per layer
        // and would silently mis-register against the split GGUF, so its magic is
        // rejected here rather than tolerated.
        std::fprintf(stderr, "[deepseek4] bad gumix sidecar header (need split-form "
                     "GUMIXs1): %s\n", sc_path.c_str());
        std::fclose(f);
        return false;
    }
    (void) reserved;

    std::vector<std::array<bool, DS4_GUMIX_SURFACES>> done(
        n_layers_out, std::array<bool, DS4_GUMIX_SURFACES>{false, false, false});
    std::vector<const void *> registered_bases;
    bool ok = true;
    for (uint32_t i = 0; i < n_entries && ok; i++) {
        uint32_t hdr[7];
        if (std::fread(hdr, 4, 7, f) != 7) {
            std::fprintf(stderr, "[deepseek4] truncated gumix sidecar (entry %u header)\n", i);
            ok = false; break;
        }
        const uint32_t layer = hdr[0], surface = hdr[1], E = hdr[2], odim = hdr[3],
                       idim = hdr[4], C = hdr[5], K = hdr[6];

        // ── Structural validation BEFORE sizing any allocation ──
        if (surface >= DS4_GUMIX_SURFACES) {
            std::fprintf(stderr, "[deepseek4] gumix entry %u (layer %u) bad surface %u\n",
                         i, layer, surface);
            ok = false; break;
        }
        if (C != DS4_GUMIX_C || K != DS4_GUMIX_K) {
            std::fprintf(stderr, "[deepseek4] gumix entry %u (layer %u) unexpected C=%u K=%u\n",
                         i, layer, C, K);
            ok = false; break;
        }
        if (E == 0 || E > DS4_P4MIX_MAX_EXPERTS ||
            odim == 0 || odim > DS4_P4MIX_MAX_DIM ||
            idim == 0 || idim > DS4_P4MIX_MAX_DIM) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u dims out of range: "
                         "E=%u out=%u in=%u\n", layer, surface, E, odim, idim);
            ok = false; break;
        }
        // Checked payload = modes(E) + books(E*C*K * 2 bytes). No rotation field.
        size_t book_elems = 0, book_bytes = 0, payload = 0;
        if (ds4_size_mul_overflow((size_t) E, (size_t) (C * K), &book_elems) ||
            ds4_size_mul_overflow(book_elems, (size_t) sizeof(uint16_t), &book_bytes) ||
            ds4_size_add_overflow((size_t) E, book_bytes, &payload)) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u payload size overflow\n", layer);
            ok = false; break;
        }

        ggml_tensor * gt = nullptr;
        if (layer < n_layers_out) {
            gt = (surface == 0) ? out.layers[layer].ffn_gate_exps
               : (surface == 1) ? out.layers[layer].ffn_up_exps
                                : out.layers[layer].ffn_down_exps;
        }
        const bool resident106 = gt && (int) gt->type == DS4_QTYPE_ROCMFP2_MIX;
        if (!resident106) {
            if (!ds4_seek_fwd(f, payload)) {
                std::fprintf(stderr, "[deepseek4] gumix seek past layer %u surface %u failed\n",
                             layer, surface);
                ok = false; break;
            }
            continue;
        }

        // Tensor ne = [in, out, E].
        if ((int64_t) E != gt->ne[2] || (int64_t) odim != gt->ne[1] ||
            (int64_t) idim != gt->ne[0]) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u dim mismatch: sidecar "
                         "E=%u out=%u in=%u vs tensor ne2=%lld ne1=%lld ne0=%lld\n",
                         layer, surface, E, odim, idim, (long long) gt->ne[2],
                         (long long) gt->ne[1], (long long) gt->ne[0]);
            ok = false; break;
        }
        if (idim % DS4_GUMIX_QK != 0) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u in=%u not a multiple "
                         "of %u\n", layer, surface, idim, DS4_GUMIX_QK);
            ok = false; break;
        }
        if (!gt->data) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u expert data not "
                         "resident\n", layer, surface);
            ok = false; break;
        }
        if (done[layer][surface]) {
            std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u appears more than "
                         "once\n", layer, surface);
            ok = false; break;
        }

        std::vector<uint8_t> modes(E);
        std::vector<uint8_t> rots(E, 0);   // wire carries none; kernel ignores it
        std::vector<uint16_t> books(book_elems);
        if (std::fread(modes.data(), 1, E, f) != E ||
            std::fread(books.data(), sizeof(uint16_t), books.size(), f) != books.size()) {
            std::fprintf(stderr, "[deepseek4] truncated gumix entry (layer %u surface %u)\n",
                         layer, surface);
            ok = false; break;
        }
        for (uint32_t e = 0; e < E && ok; ++e) {
            if (modes[e] > DS4_P4MIX_MAX_MODE) {
                std::fprintf(stderr, "[deepseek4] gumix layer %u surface %u expert %u "
                             "unsupported mode %u\n", layer, surface, e, modes[e]);
                ok = false;
            }
        }
        if (!ok) break;

        ggml_cuda_rocmfp2_mix_register_host(
            gt->data, gt->nb[2], (int) E, (int) odim, (int) idim,
            books.data(), modes.data(), rots.data());
        registered_bases.push_back(gt->data);
        done[layer][surface] = true;
    }
    std::fclose(f);

    // Every resident qtype-106 half must be covered exactly once. A layer with only
    // one half registered is worse than none: the unregistered tensor's to_fp16 shim
    // GGML_ABORTs at first decode.
    if (ok) {
        for (size_t li = 0; li < required.size() && ok; ++li) {
            for (uint32_t s = 0; s < DS4_GUMIX_SURFACES; ++s) {
                if (required[li][s] && !done[li][s]) {
                    std::fprintf(stderr, "[deepseek4] gumix sidecar missing resident qtype-106 "
                                 "layer %zu %s; refusing to load\n",
                                 li, s == 0 ? "gate" : s == 1 ? "up" : "down");
                    ok = false; break;
                }
            }
        }
    }

    if (!ok) {
        for (const void * b : registered_bases) ggml_cuda_rocmfp2_mix_unregister(b);
        return false;
    }
    std::fprintf(stderr, "[deepseek4] registered %d qtype-106 gate/up expert tensor(s) "
                 "from %s\n", (int) registered_bases.size(), sc_path.c_str());
    return true;
}

}  // namespace

// Exported with C linkage purely so the unit test can reach the dmix entry rules: they live
// in the anonymous namespace above (internal linkage), which is right for the parser but
// makes them unlinkable from outside the TU. A thin forwarding wrapper keeps one definition
// of the rules rather than a second copy that could drift from what the loader enforces.
extern "C" const char * ds4_dmix_entry_reject_reason(
        uint32_t layer, uint32_t cls, uint32_t qtype, uint32_t nslices,
        uint32_t C, uint32_t K, uint8_t mode, uint32_t n_layers, bool already_covered) {
    return ds4_dmix_entry_reject_reason_impl(layer, cls, qtype, nslices, C, K, mode,
                                             n_layers, already_covered);
}


bool load_deepseek4_gguf(const std::string & path,
                          ggml_backend_t backend,
                          DeepSeek4Weights & out) {
    TargetLoadPlan plan;
    return load_deepseek4_gguf_partial(path, backend, plan, out);
}

bool load_deepseek4_gguf_partial(const std::string & path,
                                  ggml_backend_t backend,
                                  const TargetLoadPlan & plan_in,
                                  DeepSeek4Weights & out) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init failed: " + path); return false; }

    // Validate arch
    {
        int64_t aid = gguf_find_key(gctx, "general.architecture");
        if (aid < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, aid);
        if (std::string(arch) != "deepseek4") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected deepseek4)");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    static const char * kRequiredU32Keys[] = {
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.vocab_size",
        "deepseek4.attention.head_count",
        "deepseek4.attention.head_count_kv",
        "deepseek4.attention.key_length",
        "deepseek4.rope.dimension_count",
        "deepseek4.attention.q_lora_rank",
        "deepseek4.attention.output_lora_rank",
        "deepseek4.attention.output_group_count",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_shared_count",
        "deepseek4.expert_feed_forward_length",
        "deepseek4.hash_layer_count",
        "deepseek4.attention.sliding_window",
        "deepseek4.attention.indexer.head_count",
        "deepseek4.attention.indexer.key_length",
        "deepseek4.attention.indexer.top_k",
        "deepseek4.hyper_connection.count",
        "deepseek4.hyper_connection.sinkhorn_iterations",
    };
    for (const char * key : kRequiredU32Keys) {
        if (gguf_find_key(gctx, key) < 0) {
            set_last_error(std::string("missing required key: ") + key);
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    // ── Read hyperparameters ────────────────────────────────────────────
    const uint32_t n_layer        = get_u32_or(gctx, "deepseek4.block_count", 43);
    const uint32_t n_embd         = get_u32_or(gctx, "deepseek4.embedding_length", 4096);
    const uint32_t n_vocab        = get_u32_or(gctx, "deepseek4.vocab_size", 129280);
    const uint32_t n_head         = get_u32_or(gctx, "deepseek4.attention.head_count", 64);
    const uint32_t n_head_kv      = get_u32_or(gctx, "deepseek4.attention.head_count_kv", 1);
    const uint32_t head_dim       = get_u32_or(gctx, "deepseek4.attention.key_length", 512);
    const uint32_t n_rot          = get_u32_or(gctx, "deepseek4.rope.dimension_count", 64);
    const uint32_t n_lora_q       = get_u32_or(gctx, "deepseek4.attention.q_lora_rank", 1024);
    const uint32_t n_lora_o       = get_u32_or(gctx, "deepseek4.attention.output_lora_rank", 1024);
    const uint32_t n_out_group    = get_u32_or(gctx, "deepseek4.attention.output_group_count", 8);
    const uint32_t n_expert       = get_u32_or(gctx, "deepseek4.expert_count", 256);
    const uint32_t n_expert_used  = get_u32_or(gctx, "deepseek4.expert_used_count", 6);
    const uint32_t n_expert_shared = get_u32_or(gctx, "deepseek4.expert_shared_count", 1);
    const uint32_t n_ff_exp       = get_u32_or(gctx, "deepseek4.expert_feed_forward_length", 2048);
    const uint32_t n_hash_layer   = get_u32_or(gctx, "deepseek4.hash_layer_count", 3);
    const uint32_t n_swa          = get_u32_or(gctx, "deepseek4.attention.sliding_window", 128);
    const uint32_t n_indexer_head = get_u32_or(gctx, "deepseek4.attention.indexer.head_count", 64);
    const uint32_t n_indexer_head_dim = get_u32_or(gctx, "deepseek4.attention.indexer.key_length", 128);
    const uint32_t n_indexer_top_k = get_u32_or(gctx, "deepseek4.attention.indexer.top_k", 512);
    const uint32_t n_hc           = get_u32_or(gctx, "deepseek4.hyper_connection.count", 4);
    const uint32_t n_hc_sinkhorn  = get_u32_or(gctx, "deepseek4.hyper_connection.sinkhorn_iterations", 20);

    // RoPE parameters
    const float rope_freq_base    = get_f32_or(gctx, "deepseek4.rope.freq_base", 10000.0f);
    const float rope_scale_factor = get_f32_or(gctx, "deepseek4.rope.scaling.factor", 16.0f);
    const float rope_yarn_beta_fast = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_fast", 32.0f);
    const float rope_yarn_beta_slow = get_f32_or(gctx, "deepseek4.rope.scaling.yarn_beta_slow", 1.0f);
    const float compress_rope_freq_base = get_f32_or(gctx, "deepseek4.attention.compress_rope_freq_base", 160000.0f);
    const uint64_t rope_orig_ctx  = get_u64_or(gctx, "deepseek4.rope.scaling.original_context_length", 65536);

    // Other parameters
    const float rms_eps           = get_f32_or(gctx, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    const float hc_eps            = get_f32_or(gctx, "deepseek4.hyper_connection.epsilon", 1e-6f);
    const float expert_weight_scale = get_f32_or(gctx, "deepseek4.expert_weights_scale", 1.5f);
    const float swiglu_clamp      = get_f32_or(gctx, "deepseek4.swiglu_clamp_exp", 10.0f);

    if (n_vocab == 0) {
        set_last_error("deepseek4.vocab_size must be > 0");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // Compression ratios from metadata (or compute default)
    std::vector<uint32_t> compress_ratios_meta;
    std::string compress_ratios_err;
    if (!get_u32_arr(gctx, "deepseek4.attention.compress_ratios",
                     compress_ratios_meta, &compress_ratios_err)) {
        set_last_error(compress_ratios_err);
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    std::vector<uint32_t> compress_ratios;
    if (compress_ratios_meta.size() == n_layer) {
        compress_ratios = compress_ratios_meta;
    } else {
        compress_ratios = compute_compress_ratios((int)n_layer);
    }

    const uint32_t kMissingSpecial = 0xFFFFFFFFu;
    const uint32_t raw_eos = get_u32_or(gctx, "tokenizer.ggml.eos_token_id", kMissingSpecial);
    const uint32_t raw_eot = get_u32_or(gctx, "tokenizer.ggml.eot_token_id", kMissingSpecial);

    std::fprintf(stderr, "[deepseek4] model: layers=%u embd=%u heads=%u head_dim=%u "
                 "lora_q=%u lora_o=%u out_groups=%u\n",
                 n_layer, n_embd, n_head, head_dim, n_lora_q, n_lora_o, n_out_group);
    std::fprintf(stderr, "[deepseek4] moe: experts=%u used=%u shared=%u ff=%u hash_layers=%u\n",
                 n_expert, n_expert_used, n_expert_shared, n_ff_exp, n_hash_layer);
    std::fprintf(stderr, "[deepseek4] attention: swa=%u rot=%u indexer_heads=%u top_k=%u hc=%u\n",
                 n_swa, n_rot, n_indexer_head, n_indexer_top_k, n_hc);

    // Fill output metadata
    out.n_layer         = (int)n_layer;
    out.n_embd          = (int)n_embd;
    out.n_vocab         = (int)n_vocab;
    out.n_head          = (int)n_head;
    out.n_head_kv       = (int)n_head_kv;
    out.head_dim        = (int)head_dim;
    out.n_rot           = (int)n_rot;
    out.n_out_group     = (int)n_out_group;
    out.n_lora_q        = (int)n_lora_q;
    out.n_lora_o        = (int)n_lora_o;
    out.n_expert        = (int)n_expert;
    out.n_expert_used   = (int)n_expert_used;
    out.n_expert_shared = (int)n_expert_shared;
    out.n_ff_exp        = (int)n_ff_exp;
    out.n_hash_layer    = (int)n_hash_layer;
    out.n_swa           = (int)n_swa;
    out.n_indexer_head  = (int)n_indexer_head;
    out.n_indexer_head_dim = (int)n_indexer_head_dim;
    out.n_indexer_top_k = (int)n_indexer_top_k;
    out.n_hc            = (int)n_hc;
    out.n_hc_sinkhorn_iter = (int)n_hc_sinkhorn;
    out.compress_ratios = compress_ratios;
    out.expert_weight_scale = expert_weight_scale;
    out.rope_freq_base  = rope_freq_base;
    out.rope_scale_factor = rope_scale_factor;
    out.rope_yarn_beta_fast = rope_yarn_beta_fast;
    out.rope_yarn_beta_slow = rope_yarn_beta_slow;
    out.compress_rope_freq_base = compress_rope_freq_base;
    out.rope_orig_ctx   = rope_orig_ctx;
    out.rms_eps         = rms_eps;
    out.hc_eps          = hc_eps;
    out.swiglu_clamp_exp = swiglu_clamp;
    out.eos_id          = (raw_eos == kMissingSpecial) ? -1 : (int32_t)raw_eos;
    out.eos_chat_id     = (raw_eot == kMissingSpecial) ? -1 : (int32_t)raw_eot;

    out.layers.resize(n_layer);
    out.backend = backend;

    // ── Build load plan ─────────────────────────────────────────────────
    TargetLoadPlan plan = plan_in;
    if (plan.layer_end < 0) plan.layer_end = (int)n_layer;
    if (plan.expert_metadata_only) {
        plan.load_output = false;
        plan.skip_expert_tensors = true;
    }

    // ── Collect tensors for allocation ──────────────────────────────────
    const int n_tensors = gguf_get_n_tensors(gctx);
    const size_t data_offset = gguf_get_data_offset(gctx);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    int dense_tp_mask = ds4_dense_tp_mask();
    if (dense_tp_mask != 0) {
        const char * fused_verify = std::getenv("DFLASH_DS4_FUSED_VERIFY");
        if (fused_verify && fused_verify[0] &&
            std::strcmp(fused_verify, "0") != 0) {
            std::fprintf(stderr,
                "[deepseek4-dense-tp] disabling mask=%d: split-buffer dense "
                "weights are incompatible with fused verifier graph replay\n",
                dense_tp_mask);
            dense_tp_mask = 0;
        }
    }
    ggml_backend_buffer_type_t split_buft = nullptr;
    size_t split_alignment = 1;
    if (dense_tp_mask != 0 && ggml_backend_is_cuda(backend) &&
        ggml_backend_cuda_get_device_count() >= 2) {
        float strix_fraction = 0.28f;
        if (const char * value = std::getenv("DFLASH_DS4_DENSE_TP_STRIX_FRACTION")) {
            const float parsed = std::strtof(value, nullptr);
            if (parsed > 0.0f && parsed < 1.0f) strix_fraction = parsed;
        }
        float tensor_split[GGML_CUDA_MAX_DEVICES] = {};
        tensor_split[0] = 1.0f - strix_fraction;
        tensor_split[1] = strix_fraction;
        split_buft = ggml_backend_cuda_split_buffer_type(0, tensor_split);
        split_alignment = ggml_backend_buft_get_alignment(split_buft);
        std::fprintf(stderr,
                     "[deepseek4-dense-tp] mask=%d row split R9700=%.3f Strix=%.3f\n",
                     dense_tp_mask, 1.0f - strix_fraction, strix_fraction);
    }

    std::vector<DS4TensorAlloc> allocs;
    allocs.reserve(n_tensors);
    size_t total_buf_size = 0;
    size_t split_total_buf_size = 0;
    size_t tok_embd_alloc_idx = SIZE_MAX;

    for (int ti = 0; ti < n_tensors; ti++) {
        const char * tname = gguf_get_tensor_name(gctx, ti);
        if (!should_keep_ds4_tensor(tname, plan)) continue;

        ggml_tensor * t = find_tensor(meta_ctx, tname);
        if (!t) continue;

        const size_t tensor_offset = gguf_get_tensor_offset(gctx, ti);
        const bool upload_to_backend = should_upload_ds4_tensor(tname, plan);

        DS4TensorAlloc a;
        a.tensor = t;
        a.tensor_offset = tensor_offset;
        a.file_size = gguf_get_tensor_size(gctx, ti);
        a.upload_to_backend = upload_to_backend;
        a.dense_split = upload_to_backend && split_buft &&
                        should_split_ds4_dense_tensor(tname, dense_tp_mask);
        if (upload_to_backend) {
            if (a.dense_split) {
                split_total_buf_size = align_up_size(
                    split_total_buf_size, split_alignment);
                a.buffer_offset = split_total_buf_size;
                split_total_buf_size +=
                    ggml_backend_buft_get_alloc_size(split_buft, t);
            } else {
                total_buf_size = align_up_size(total_buf_size, alignment);
                a.buffer_offset = total_buf_size;
                total_buf_size += ggml_backend_buft_get_alloc_size(buft, t);
            }
        }
        allocs.push_back(a);
        if (std::strcmp(tname, "token_embd.weight") == 0) {
            tok_embd_alloc_idx = allocs.size() - 1;
        }
    }

    // ── Allocate GPU buffer ─────────────────────────────────────────────
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t split_buf = nullptr;
    if (total_buf_size > 0) {
        buf = ggml_backend_alloc_buffer(backend, total_buf_size);
        if (!buf) {
            set_last_error("failed to allocate GPU buffer (" + std::to_string(total_buf_size) + " bytes)");
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (split_total_buf_size > 0) {
        split_buf = ggml_backend_buft_alloc_buffer(
            split_buft, split_total_buf_size);
        if (!split_buf) {
            set_last_error("failed to allocate dense TP split buffer (" +
                           std::to_string(split_total_buf_size) + " bytes)");
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(
            split_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // ── Assign tensors from meta_ctx to the backend buffer ──────────────
    // Use ggml_backend_tensor_alloc to properly set the buffer association.
    char * buf_base = buf ? (char *)ggml_backend_buffer_get_base(buf) : nullptr;
    char * split_buf_base = split_buf
        ? (char *)ggml_backend_buffer_get_base(split_buf) : nullptr;
    for (auto & a : allocs) {
        if (!a.upload_to_backend) continue;
        ggml_backend_buffer_t tensor_buf = a.dense_split ? split_buf : buf;
        char * tensor_base = a.dense_split ? split_buf_base : buf_base;
        if (!tensor_buf ||
            ggml_backend_tensor_alloc(tensor_buf, a.tensor,
                                      tensor_base + a.buffer_offset) !=
                GGML_STATUS_SUCCESS) {
            set_last_error("ggml_backend_tensor_alloc failed");
            if (split_buf) ggml_backend_buffer_free(split_buf);
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
    }

    // ── Memory-map the file and copy tensor data ────────────────────────
    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        set_last_error("mmap: " + mmap_err);
        if (split_buf) ggml_backend_buffer_free(split_buf);
        if (buf) ggml_backend_buffer_free(buf);
        gguf_free(gctx);
        ggml_free(meta_ctx);
        return false;
    }
    for (auto & a : allocs) {
        if (!gguf_tensor_in_file(data_offset, a.tensor_offset, a.file_size, mmap.len)) {
            set_last_error(gguf_bounds_error("deepseek4 GGUF",
                                             ggml_get_name(a.tensor),
                                             ggml_type_name(a.tensor->type),
                                             data_offset,
                                             a.tensor_offset,
                                             a.file_size,
                                             mmap.len));
            mmap.close_map();
            if (split_buf) ggml_backend_buffer_free(split_buf);
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
        a.file_offset = data_offset + a.tensor_offset;
    }

#if !defined(_WIN32)
    bool fast_managed = (buf != nullptr) && ggml_backend_cuda_buffer_is_managed(buf) && (getenv("DFLASH_NO_PREAD") == nullptr);
#else
    // pread/posix_fadvise not available on Windows; fall back to mmap path.
    bool fast_managed = false;
#endif
    if (fast_managed) {
        // Unified/managed buffer: read weights straight off disk into it in parallel at
        // disk bandwidth, instead of mmap page-faults (~5x slower). Drop the cached file
        // pages afterward so the page cache does not double a near-RAM-size model.
        unsigned nth = std::thread::hardware_concurrency();
        if (nth == 0) nth = 4;
        if (nth > 8)  nth = 8;
        std::atomic<size_t> next{0};
        std::atomic<bool> read_ok{true};
        auto worker = [&]() {
            size_t i;
            while ((i = next.fetch_add(1)) < allocs.size()) {
                auto & a = allocs[i];
                if (!a.upload_to_backend || a.dense_split) continue;
                char * dst = (char *) a.tensor->data;
                size_t done = 0;
                while (done < a.file_size) {
#if !defined(_WIN32)
                    ssize_t r = pread(mmap.fd, dst + done, a.file_size - done,
                              (off_t) (a.file_offset + done));
#else
                    int r = -1;  // not reached: fast_managed is false on Windows
#endif
                    if (r <= 0) { read_ok = false; return; }
                    done += (size_t) r;
                }
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nth; t++) pool.emplace_back(worker);
        for (auto & th : pool) th.join();
#if !defined(_WIN32)
        posix_fadvise(mmap.fd, 0, (off_t) mmap.len, POSIX_FADV_DONTNEED);
#endif
        ggml_backend_synchronize(backend);  // make CPU-written managed pages visible to GPU
        // Split tensors have no single CPU-visible base address. Upload them
        // through the split buffer, which distributes whole rows to each GPU.
        for (auto & a : allocs) {
            if (!a.upload_to_backend || !a.dense_split) continue;
            const void * src_data = (const char *)mmap.addr + a.file_offset;
            ggml_backend_tensor_set(a.tensor, src_data, 0, a.file_size);
        }
        if (!read_ok) {
            set_last_error("parallel weight read failed");
            mmap.close_map();
            if (split_buf) ggml_backend_buffer_free(split_buf);
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
    } else {
        for (auto & a : allocs) {
            if (!a.upload_to_backend) continue;
            const void * src_data = (const char *)mmap.addr + a.file_offset;
            ggml_backend_tensor_set(a.tensor, src_data, 0, a.file_size);
        }
    }
    mmap.close_map();

    // ── Set up CPU embedder ─────────────────────────────────────────────
    // The embedder is set up using the mmap data directly (like gemma4).
    // For now, we use an owned copy of the token embedding table bytes.
    if (tok_embd_alloc_idx != SIZE_MAX) {
        const auto & a = allocs[tok_embd_alloc_idx];
        out.embedder.tok_embd_owned.resize(a.file_size);
        // Re-read from mmap (already closed). Use the GPU tensor instead:
        // Actually, we need the raw bytes for dequantization. Reopen mmap briefly.
        DS4Mmap emb_mmap;
        std::string emb_err;
        if (emb_mmap.open_ro(path, emb_err)) {
            std::memcpy(out.embedder.tok_embd_owned.data(),
                        (const char *)emb_mmap.addr + a.file_offset, a.file_size);
            emb_mmap.close_map();
        } else {
            set_last_error("embedder mmap: " + emb_err);
            if (split_buf) ggml_backend_buffer_free(split_buf);
            if (buf) ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(meta_ctx);
            return false;
        }
        out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
        out.embedder.tok_embd_type  = a.tensor->type;
        out.embedder.n_embd         = n_embd;
        out.embedder.n_vocab        = (int64_t)n_vocab;
        out.embedder.row_bytes      = a.file_size / (size_t)n_vocab;
    }

    // ── Bind tensors to weight struct fields ────────────────────────────
    for (auto & a : allocs) {
        const char * name = ggml_get_name(a.tensor);

        // Global tensors
        if (std::strcmp(name, "token_embd.weight") == 0) { out.tok_embd = a.tensor; continue; }
        if (std::strcmp(name, "output_norm.weight") == 0) { out.out_norm = a.tensor; continue; }
        if (std::strcmp(name, "output.weight") == 0) { out.output = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_base.weight") == 0) { out.output_hc_base = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_fn.weight") == 0) { out.output_hc_fn = a.tensor; continue; }
        if (std::strcmp(name, "output_hc_scale.weight") == 0) { out.output_hc_scale = a.tensor; continue; }

        // Per-layer tensors
        int il = -1;
        if (!parse_block_tensor_name(name, il) || il < 0 || il >= (int)n_layer) continue;
        DeepSeek4Layer & L = out.layers[il];

        // Find the suffix after "blk.<il>."
        const char * p = name;
        while (*p && *p != '.') p++;  // skip "blk"
        if (*p == '.') p++;           // skip first '.'
        while (*p && *p != '.') p++;  // skip layer number
        if (*p == '.') p++;           // skip second '.'
        const std::string suffix(p);

        // Attention
        if (suffix == "attn_norm.weight")          { L.attn_norm = a.tensor; continue; }
        if (suffix == "attn_q_a.weight")           { L.attn_q_a = a.tensor; continue; }
        if (suffix == "attn_q_a_norm.weight")      { L.attn_q_a_norm = a.tensor; continue; }
        if (suffix == "attn_q_b.weight")           { L.attn_q_b = a.tensor; continue; }
        if (suffix == "attn_kv.weight")            { L.attn_kv = a.tensor; continue; }
        if (suffix == "attn_kv_a_norm.weight")     { L.attn_kv_a_norm = a.tensor; continue; }
        if (suffix == "attn_sinks.weight")         { L.attn_sinks = a.tensor; continue; }
        if (suffix == "attn_output_a.weight")      { L.attn_output_a = a.tensor; continue; }
        if (suffix == "attn_output_b.weight")      { L.attn_output_b = a.tensor; continue; }

        // Compressor
        if (suffix == "attn_compressor_ape.weight")  { L.attn_compressor_ape = a.tensor; continue; }
        if (suffix == "attn_compressor_kv.weight")   { L.attn_compressor_kv = a.tensor; continue; }
        if (suffix == "attn_compressor_gate.weight") { L.attn_compressor_gate = a.tensor; continue; }
        if (suffix == "attn_compressor_norm.weight") { L.attn_compressor_norm = a.tensor; continue; }

        // Indexer
        if (suffix == "indexer.attn_q_b.weight")     { L.indexer_attn_q_b = a.tensor; continue; }
        if (suffix == "indexer.proj.weight")          { L.indexer_proj = a.tensor; continue; }
        if (suffix == "indexer_compressor_ape.weight")  { L.indexer_compressor_ape = a.tensor; continue; }
        if (suffix == "indexer_compressor_kv.weight")   { L.indexer_compressor_kv = a.tensor; continue; }
        if (suffix == "indexer_compressor_gate.weight") { L.indexer_compressor_gate = a.tensor; continue; }
        if (suffix == "indexer_compressor_norm.weight") { L.indexer_compressor_norm = a.tensor; continue; }

        // HC attention
        if (suffix == "hc_attn_fn.weight")         { L.hc_attn_fn = a.tensor; continue; }
        if (suffix == "hc_attn_scale.weight")      { L.hc_attn_scale = a.tensor; continue; }
        if (suffix == "hc_attn_base.weight")       { L.hc_attn_base = a.tensor; continue; }

        // FFN
        if (suffix == "ffn_norm.weight")           { L.ffn_norm = a.tensor; continue; }
        if (suffix == "ffn_gate_inp.weight")       { L.ffn_gate_inp = a.tensor; continue; }
        if (suffix == "exp_probs_b.bias")          { L.ffn_exp_probs_b = a.tensor; continue; }
        if (suffix == "ffn_gate_tid2eid.weight")   { L.ffn_gate_tid2eid = a.tensor; continue; }
        if (suffix == "ffn_gate_exps.weight")      { L.ffn_gate_exps = a.tensor; continue; }
        if (suffix == "ffn_up_exps.weight")        { L.ffn_up_exps = a.tensor; continue; }
        if (suffix == "ffn_down_exps.weight")      { L.ffn_down_exps = a.tensor; continue; }
        if (suffix == "ffn_gate_shexp.weight")     { L.ffn_gate_shexp = a.tensor; continue; }
        if (suffix == "ffn_up_shexp.weight")       { L.ffn_up_shexp = a.tensor; continue; }
        if (suffix == "ffn_down_shexp.weight")     { L.ffn_down_shexp = a.tensor; continue; }

        // HC FFN
        if (suffix == "hc_ffn_fn.weight")          { L.hc_ffn_fn = a.tensor; continue; }
        if (suffix == "hc_ffn_scale.weight")       { L.hc_ffn_scale = a.tensor; continue; }
        if (suffix == "hc_ffn_base.weight")        { L.hc_ffn_base = a.tensor; continue; }
    }

    out.ctx = meta_ctx;
    out.buf = buf;
    out.dense_split_buf = split_buf;
    gguf_free(gctx);
    // Note: meta_ctx is now owned by out.ctx — do NOT free it here.

    // qtype-105 mixed-policy down-experts: register per-expert codebooks now
    // that tensor data pointers are final (no-op for uniform models). A qtype-105
    // model with a missing/invalid/mismatched sidecar has no working decode path,
    // so fail the load here rather than aborting at first decode.
    // qtype-106 mixed-policy gate/up experts: same contract as the qtype-105 call
    // below, on the other MoE surface. Registered first so a gate/up failure unwinds
    // before any down-expert entries exist.
    // Dense (attention) mix-qtype tensors first: a dense failure unwinds before any MoE
    // entries exist, keeping the teardown order the reverse of registration.
    if (!ds4_register_dmix_sidecar(path, out)) {
        std::fprintf(stderr, "[deepseek4] dense mix-qtype sidecar registration failed "
                     "for %s\n", path.c_str());
        free_deepseek4_weights(out);
        return false;
    }

    if (!ds4_register_gumix_sidecar(path, plan, out)) {
        std::fprintf(stderr, "[deepseek4] qtype-106 gate/up sidecar registration failed "
                     "for %s\n", path.c_str());
        free_deepseek4_weights(out);
        return false;
    }

    if (!ds4_register_p4mix_sidecar(path, plan, out)) {
        std::fprintf(stderr, "[deepseek4] qtype-105 sidecar registration failed for %s\n",
                     path.c_str());
        // out.ctx / out.buf are already live at this point. Release them (and any
        // remaining registry entries) so the failed load leaves `out` empty —
        // otherwise load_model()'s fallback to init_hybrid_model() reuses the same
        // DeepSeek4Weights and overwrites out.ctx/out.buf, permanently leaking the
        // context + GPU buffer allocated above. free_deepseek4_weights is safe to
        // call here (it null-checks and its qtype-105 unregister loop is a no-op
        // for the entries the sidecar path already unwound).
        free_deepseek4_weights(out);
        return false;
    }

    std::fprintf(stderr, "[deepseek4] loaded %zu tensors, %.1f MB GPU buffer, %.1f MB dense TP split%s\n",
                 allocs.size(), (double)total_buf_size / (1024.0 * 1024.0),
                 (double)split_total_buf_size / (1024.0 * 1024.0),
                 plan.expert_metadata_only ? " [expert-metadata-only]" : "");
    return true;
}

namespace {

static MoeHybridColdBackend ds4_cold_backend_from_env() {
    const char * value = std::getenv("DFLASH_MOE_COLD_BACKEND");
    if (!value || !value[0]) return MoeHybridColdBackend::Cpu;
    if (std::strcmp(value, "gpu") == 0 || std::strcmp(value, "hip") == 0 ||
        std::strcmp(value, "rocm") == 0) {
        return MoeHybridColdBackend::Gpu;
    }
    return MoeHybridColdBackend::Cpu;
}

static MoeHybridConfig make_ds4_moe_hybrid_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.cold_expert_backend = ds4_cold_backend_from_env();
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

}  // namespace

bool build_deepseek4_moe_hybrid_storage_from_file(
        const std::string & path,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const MoeHybridPlacement & placement,
        const MoeHybridConfig * cfg_override,
        MoeHybridStorage & out,
        std::string * err) {
    ggml_context * expert_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &expert_meta;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        if (err) *err = "failed to re-open GGUF for expert loading";
        return false;
    }

    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        gguf_free(gctx);
        if (expert_meta) ggml_free(expert_meta);
        if (err) *err = mmap_err;
        return false;
    }

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = static_cast<const uint8_t *>(mmap.addr);
    std::vector<LayerExpertFileData> layer_file_data((size_t)w.n_layer);
    bool bad_bounds = false;
    std::string bounds_err;

    for (int il = 0; il < w.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            const size_t tensor_off = gguf_get_tensor_offset(gctx, tid);
            const size_t sz = gguf_get_tensor_size(gctx, tid);
            if (!gguf_tensor_in_file(data_start, tensor_off, sz, mmap.len)) {
                bad_bounds = true;
                bounds_err = gguf_bounds_error("deepseek4 expert GGUF", name,
                                               ggml_type_name(gguf_get_tensor_type(gctx, tid)),
                                               data_start, tensor_off, sz, mmap.len);
                return {};
            }
            const size_t off = data_start + tensor_off;
            return { file_bytes + off, sz };
        };

        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
        if (bad_bounds) {
            mmap.close_map();
            gguf_free(gctx);
            if (expert_meta) ggml_free(expert_meta);
            if (err) *err = bounds_err;
            return false;
        }
    }

    std::vector<MoeLayerDesc> layer_descs((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w.layers[(size_t)il]);
    }

    const MoeHybridConfig cfg = cfg_override ? *cfg_override : make_ds4_moe_hybrid_config(w);
    const bool ok = build_moe_hybrid_storage_from_file(
        cfg, backend, placement, layer_descs, layer_file_data, out, err);

    mmap.close_map();
    gguf_free(gctx);
    if (expert_meta) ggml_free(expert_meta);
    return ok;
}

bool build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
        const std::string & path,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const MoeHybridPlacement & placement,
        const MoeHybridConfig * cfg_override,
        MoeHybridStorage & out,
        std::string * err,
        ggml_backend_t cold_gpu_backend) {
    ggml_context * expert_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &expert_meta;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        if (err) *err = "failed to re-open GGUF for expert loading";
        return false;
    }

    DS4Mmap mmap;
    std::string mmap_err;
    if (!mmap.open_ro(path, mmap_err)) {
        gguf_free(gctx);
        if (expert_meta) ggml_free(expert_meta);
        if (err) *err = mmap_err;
        return false;
    }
    mmap.close_fd();

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = static_cast<const uint8_t *>(mmap.addr);
    std::vector<LayerExpertFileData> layer_file_data((size_t)w.n_layer);
    bool bad_bounds = false;
    std::string bounds_err;

    for (int il = 0; il < w.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            const size_t tensor_off = gguf_get_tensor_offset(gctx, tid);
            const size_t sz = gguf_get_tensor_size(gctx, tid);
            if (!gguf_tensor_in_file(data_start, tensor_off, sz, mmap.len)) {
                bad_bounds = true;
                bounds_err = gguf_bounds_error("deepseek4 expert GGUF", name,
                                               ggml_type_name(gguf_get_tensor_type(gctx, tid)),
                                               data_start, tensor_off, sz, mmap.len);
                return {};
            }
            const size_t off = data_start + tensor_off;
            return { file_bytes + off, sz };
        };

        layer_file_data[(size_t)il].gate_exps = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps = find_tensor_data("ffn_down_exps");
        if (bad_bounds) {
            mmap.close_map();
            gguf_free(gctx);
            if (expert_meta) ggml_free(expert_meta);
            if (err) *err = bounds_err;
            return false;
        }
    }

    std::vector<MoeLayerDesc> layer_descs((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        layer_descs[(size_t)il] = make_ds4_moe_layer_desc(w.layers[(size_t)il]);
    }

    const MoeHybridConfig cfg = cfg_override ? *cfg_override : make_ds4_moe_hybrid_config(w);
    const bool ok = build_moe_hybrid_storage_from_file_with_mmap(
        cfg, backend, placement, layer_descs, layer_file_data,
        mmap.addr, mmap.len, out, err, 0, cold_gpu_backend);

    if (!ok) {
        mmap.close_map();
    } else {
        mmap.addr = nullptr;
        mmap.len = 0;
    }
    gguf_free(gctx);
    if (expert_meta) ggml_free(expert_meta);
    return ok;
}

bool build_deepseek4_moe_hybrid_storage_from_file(
        const std::string & path,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const MoeHybridPlacement & placement,
        MoeHybridStorage & out,
        std::string * err) {
    return build_deepseek4_moe_hybrid_storage_from_file(
        path, backend, w, placement, nullptr, out, err);
}

void free_deepseek4_weights(DeepSeek4Weights & w) {
    deepseek4_release_runtime_graphs(w);
    // Drop qtype-105 registry entries (and free their device side-data) while the
    // tensors are still valid — BEFORE ggml_free destroys them and the GPU buffer
    // is released. Otherwise the entries leak device memory across park/unpark and
    // reload, and a later load reusing the same device address would resolve to a
    // stale range. Keyed by dt->data, so read it here.
    for (auto & L : w.layers) {
        ggml_tensor * dt = L.ffn_down_exps;
        if (dt && dt->type == GGML_TYPE_Q3_1_ROCMFP3_MIX && dt->data) {
            ggml_cuda_rocmfp3_mix_unregister(dt->data);
        }
    }
    // Same contract for qtype-106 gate/up, and it needs both halves: the gumix
    // sidecar registers the SPLIT form, one entry per tensor, and each entry owns
    // its own device copy of the (shared) codebooks — duplicated deliberately so
    // that unregister can free per entry without double-freeing. Dropping only one
    // half would still leak the other's codebooks and leave a stale range behind.
    for (auto & L : w.layers) {
        ggml_tensor * const gu[3] = { L.ffn_gate_exps, L.ffn_up_exps,
                                      L.ffn_down_exps };
        for (ggml_tensor * t : gu) {
            if (t && t->type == GGML_TYPE_Q2_1_ROCMFP2_MIX && t->data) {
                ggml_cuda_rocmfp2_mix_unregister(t->data);
            }
        }
    }
    // And the DENSE mix tensors. ds4_register_dmix_sidecar registers up to five attention
    // classes per layer through the same two registries, but they were never torn down: the
    // expert loops above key off the ffn_* members and cannot reach attn_q_a/q_b/kv/
    // output_a/output_b. Every load/free cycle therefore leaked their device-side codebooks
    // and left registry ranges pointing at released buffers -- and a later allocation landing
    // on one of those addresses resolves to the stale side data rather than failing. The
    // dispatch is by TENSOR TYPE, not by class, because a dense artifact may mix 105 and 106
    // across classes; the sidecar records a qtype per entry precisely so it can.
    for (auto & L : w.layers) {
        for (uint32_t c = 0; c < DS4_DMIX_CLASSES; ++c) {
            ggml_tensor * t = ds4_dmix_class_tensor(L, c);
            if (!t || !t->data) {
                continue;
            }
            if (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
                ggml_cuda_rocmfp3_mix_unregister(t->data);
            } else if (t->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
                ggml_cuda_rocmfp2_mix_unregister(t->data);
            }
        }
    }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    if (w.dense_split_buf) {
        ggml_backend_buffer_free(w.dense_split_buf);
        w.dense_split_buf = nullptr;
    }
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    w.layers.clear();
    w.embedder.tok_embd_owned.clear();
    w.embedder.tok_embd_bytes = nullptr;
    w.moe_hybrid = false;
}

}  // namespace dflash::common
