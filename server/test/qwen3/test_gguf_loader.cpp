#include "dflash27b.h"
#include "qwen3/qwen3_drafter_model.h"
#include "common/gguf_bounds.h"
#include "common/gguf_inspect.h"
#include "qwen35/prefill_helpers.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// Qwen3-0.6B drafter loader: truncated GGUF guard (bug #438)
// ═══════════════════════════════════════════════════════════════════════
//
// Builds a minimal but structurally valid Qwen3-0.6B-style GGUF on disk, then
// verifies that load_qwen3_drafter_model:
//   (1) loads the full, untruncated file successfully (positive control), and
//   (2) fails cleanly with a "truncated or corrupt" error when the tensor-data
//       section is truncated — instead of letting the H2D copy read past the
//       end of the mmap and SIGSEGV inside the device copy.

// Write a tiny valid drafter GGUF to the caller-owned temporary file. The loader fixes
// n_vocab at 151936 (Qwen3DrafterWeights default), so token_embd stays the
// largest tensor (~2.4 MB BF16) while every other tensor is minimal.
static bool write_qwen3_drafter_fixture_gguf(const char * path) {
    const int n_embd    = 8;
    const int n_head    = 2;
    const int head_dim  = 4;
    const int n_head_kv = 1;
    const int n_ff      = 16;
    const int n_layer   = 1;
    const int n_vocab   = 151936;                // must match the loader default
    const int q_dim     = n_head * head_dim;     // 8
    const int kv_dim    = n_head_kv * head_dim;  // 4

    ggml_init_params ip{};
    ip.mem_size   = (size_t)16 * 1024 * 1024;    // headroom for token_embd + 13 tensors
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "qwen3.embedding_length",        (uint32_t)n_embd);
    gguf_set_val_u32(g, "qwen3.feed_forward_length",     (uint32_t)n_ff);
    gguf_set_val_u32(g, "qwen3.attention.head_count",    (uint32_t)n_head);
    gguf_set_val_u32(g, "qwen3.attention.head_count_kv", (uint32_t)n_head_kv);
    gguf_set_val_u32(g, "qwen3.block_count",             (uint32_t)n_layer);
    gguf_set_val_u32(g, "qwen3.context_length",          (uint32_t)64);
    gguf_set_val_u32(g, "qwen3.attention.key_length",    (uint32_t)head_dim);
    gguf_set_val_f32(g, "qwen3.rope.freq_base",          1000000.0f);

    auto add_tensor = [&](const char * name, ggml_type t, int n_dims,
                          int64_t ne0, int64_t ne1) {
        ggml_tensor * w = (n_dims == 1)
            ? ggml_new_tensor_1d(ctx, t, ne0)
            : ggml_new_tensor_2d(ctx, t, ne0, ne1);
        ggml_set_name(w, name);
        std::memset(w->data, 0, ggml_nbytes(w));
        gguf_add_tensor(g, w);
    };

    // Top-level tensors. output.weight is intentionally omitted so the loader
    // exercises its tied-weights path (and the fixture stays small).
    add_tensor("token_embd.weight",  GGML_TYPE_BF16, 2, n_embd, n_vocab);
    add_tensor("output_norm.weight", GGML_TYPE_F32,  1, n_embd, 0);

    // The single transformer block: the 11 per-layer tensors the loader copies.
    add_tensor("blk.0.attn_norm.weight",   GGML_TYPE_F32,  1, n_embd,   0);
    add_tensor("blk.0.attn_q.weight",      GGML_TYPE_BF16, 2, n_embd,   q_dim);
    add_tensor("blk.0.attn_k.weight",      GGML_TYPE_BF16, 2, n_embd,   kv_dim);
    add_tensor("blk.0.attn_v.weight",      GGML_TYPE_BF16, 2, n_embd,   kv_dim);
    add_tensor("blk.0.attn_output.weight", GGML_TYPE_BF16, 2, q_dim,    n_embd);
    add_tensor("blk.0.attn_q_norm.weight", GGML_TYPE_F32,  1, head_dim, 0);
    add_tensor("blk.0.attn_k_norm.weight", GGML_TYPE_F32,  1, head_dim, 0);
    add_tensor("blk.0.ffn_norm.weight",    GGML_TYPE_F32,  1, n_embd,   0);
    add_tensor("blk.0.ffn_gate.weight",    GGML_TYPE_BF16, 2, n_embd,   n_ff);
    add_tensor("blk.0.ffn_up.weight",      GGML_TYPE_BF16, 2, n_embd,   n_ff);
    add_tensor("blk.0.ffn_down.weight",    GGML_TYPE_BF16, 2, n_ff,     n_embd);

    const bool written = gguf_write_to_file(g, path, /*only_meta=*/false);

    gguf_free(g);
    ggml_free(ctx);
    return written;
}

TEST_CASE(ServerUnitFixture, test_qwen3_drafter_rejects_truncated_gguf) {
    char temp_path[] = "/tmp/dflash_test_qwen3_drafter_438_XXXXXX";
    const int fd = mkstemp(temp_path);
    TEST_ASSERT(fd >= 0);
    struct RemoveFile {
        const char * path;
        ~RemoveFile() { unlink(path); }
    } cleanup{temp_path};
    TEST_ASSERT(close(fd) == 0);
    TEST_ASSERT(write_qwen3_drafter_fixture_gguf(temp_path));
    const std::string path = temp_path;

    ggml_backend_t backend = ggml_backend_cpu_init();
    TEST_ASSERT(backend != nullptr);

    // Positive control: the full, untruncated file loads cleanly.
    {
        Qwen3DrafterWeights w;
        bool ok = load_qwen3_drafter_model(path, backend, w);
        TEST_ASSERT_MSG(ok, dflash27b_last_error());
        free_qwen3_drafter_model(w);
    }

    // Truncate inside the tensor-data section. The header, kv block, and tensor
    // info table all live before the data offset, so gguf_init_from_file still
    // succeeds and we reach the EOF guard rather than a parse failure.
    struct stat st{};
    TEST_ASSERT(stat(path.c_str(), &st) == 0);
    const off_t truncated_size = (off_t)st.st_size - 4096;
    TEST_ASSERT(truncated_size > 0);
    TEST_ASSERT(truncate(path.c_str(), truncated_size) == 0);

    // The loader must fail cleanly (no SIGSEGV) with a descriptive error.
    {
        Qwen3DrafterWeights w;
        bool ok = load_qwen3_drafter_model(path, backend, w);
        TEST_ASSERT(!ok);
        const std::string err = dflash27b_last_error();
        TEST_ASSERT_MSG(err.find("truncated or corrupt") != std::string::npos,
                        err.c_str());
        free_qwen3_drafter_model(w);
    }

    ggml_backend_free(backend);
}

// ─── GGUF tensor bounds (gguf_tensor_in_file / gguf_bounds_error) ───────
//
// The shared overflow-safe bounds check used by every GGUF loader
// (draft/target/laguna) to reject truncated/corrupt files before a copy reads
// past the mapping (#438), without wrongly rejecting valid files (#318). These
// tests pin the boundary and, critically, the size_t overflow behaviour that a
// naive `data_off + tensor_off + tensor_sz > file_size` test gets wrong.
TEST_CASE(ServerUnitFixture, test_gguf_tensor_in_file_bounds) {
    // Typical layout: 100-byte header/kv, 900-byte data section, 1000-byte file.
    const size_t data_off = 100;
    const size_t file     = 1000;

    // Fully inside, including the exact end-of-file boundary.
    TEST_ASSERT(gguf_tensor_in_file(data_off, 0,   900, file));   // fills the data section
    TEST_ASSERT(gguf_tensor_in_file(data_off, 0,   0,   file));   // zero-size tensor
    TEST_ASSERT(gguf_tensor_in_file(data_off, 899, 1,   file));   // last byte
    TEST_ASSERT(gguf_tensor_in_file(data_off, 900, 0,   file));   // zero-size at EOF

    // One byte past EOF must be rejected.
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 0,   901, file));
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 900, 1,   file));

    // Data section offset itself past EOF (corrupt header).
    TEST_ASSERT(!gguf_tensor_in_file(2000, 0, 0, file));
    // data_off == file: only a zero-size tensor at offset 0 fits.
    TEST_ASSERT(gguf_tensor_in_file(file, 0, 0, file));
    TEST_ASSERT(!gguf_tensor_in_file(file, 0, 1, file));

    // Whole-file data section (data_off == 0), valid full read.
    TEST_ASSERT(gguf_tensor_in_file(0, 0, file, file));
    TEST_ASSERT(!gguf_tensor_in_file(0, 0, file + 1, file));

    // Overflow safety: a malformed offset/size must not wrap and slip through.
    const size_t kMax = std::numeric_limits<size_t>::max();
    TEST_ASSERT(!gguf_tensor_in_file(data_off, kMax, 10, file));   // huge tensor_off
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 0, kMax, file));    // huge tensor_sz
    // The case a naive `off + sz > file` check fails: off + sz wraps below file.
    // off = kMax - 10, sz = 20  →  off + sz == 9 (< file) would FALSE-PASS.
    TEST_ASSERT(!gguf_tensor_in_file(0, kMax - 10, 20, file));
    TEST_ASSERT(!gguf_tensor_in_file(kMax, 10, 10, file));         // huge data_off
}

TEST_CASE(ServerUnitFixture, test_gguf_bounds_error_reports_operands) {
    // A normal (non-overflowing) rejection: the message must surface every
    // operand so a false positive on a valid file (#318) is diagnosable.
    const std::string e = gguf_bounds_error("target GGUF", "blk.56.ssm_out.weight",
                                            "q5_K", 100, 5000, 200, 1000);
    TEST_ASSERT(e.find("blk.56.ssm_out.weight") != std::string::npos);
    TEST_ASSERT(e.find("q5_K") != std::string::npos);
    TEST_ASSERT(e.find("data_off=100") != std::string::npos);
    TEST_ASSERT(e.find("tensor_off=5000") != std::string::npos);
    TEST_ASSERT(e.find("size=200") != std::string::npos);
    TEST_ASSERT(e.find("5300") != std::string::npos);      // 100 + 5000 + 200
    TEST_ASSERT(e.find("1000") != std::string::npos);      // file size

    // Null name/type must not crash and must produce placeholders.
    const std::string n = gguf_bounds_error("draft GGUF", nullptr, nullptr,
                                            0, 0, 1, 0);
    TEST_ASSERT(n.find("(null)") != std::string::npos);
    TEST_ASSERT(n.find("(unknown)") != std::string::npos);

    // When the end offset itself overflows size_t, report "overflow" rather
    // than a wrapped number.
    const size_t kMax = std::numeric_limits<size_t>::max();
    const std::string o = gguf_bounds_error("target GGUF", "t", "f32",
                                            kMax, 10, 10, 100);
    TEST_ASSERT(o.find("overflow") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_qwen35_embedded_mtp_target_layer_count) {
    uint32_t target_layers = 0;
    std::string error;

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35", 64, 0, target_layers, error));
    TEST_ASSERT(target_layers == 64);

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35", 65, 1, target_layers, error));
    TEST_ASSERT(target_layers == 64);

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35moe", 81, 1, target_layers, error));
    TEST_ASSERT(target_layers == 80);

    TEST_ASSERT(derive_effective_target_layer_count(
        "bailingmoe3", 43, 1, target_layers, error));
    TEST_ASSERT(target_layers == 42);

    TEST_ASSERT(!derive_effective_target_layer_count(
        "qwen35", 1, 1, target_layers, error));
    TEST_ASSERT(error.find("smaller than block_count") != std::string::npos);

    TEST_ASSERT(!derive_effective_target_layer_count(
        "qwen35", 0, 0, target_layers, error));
    TEST_ASSERT(error.find("greater than zero") != std::string::npos);

    // Do not reinterpret similarly named metadata for unrelated architectures.
    TEST_ASSERT(derive_effective_target_layer_count(
        "laguna", 65, 1, target_layers, error));
    TEST_ASSERT(target_layers == 65);
}
