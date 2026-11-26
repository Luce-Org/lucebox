// Qwen3.5-0.8B drafter loading: the drafter GGUF, the optional trained
// block-15 scoring head and the optional segment probe.
//
// The scorer is built on the Qwen3.5 target weights
// (load_target_gguf_partial).

#include "qwen35_drafter.h"

#include "common/gguf_inspect.h"
#include "common/kvflash_pool_flag.h"
#include "common/sha256.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <limits>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <atomic>
#  include <fcntl.h>
#  include <io.h>
#  include <windows.h>
#endif

namespace luce::common {

bool qwen35_head_block_available(const TargetWeights & w, std::string & error) {
    if (w.n_layer <= kQwen35HeadBlock || (size_t)kQwen35HeadBlock >= w.layers.size()) {
        error = "qwen35 scoring head needs at least 16 blocks";
        return false;
    }
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];
    if (((kQwen35HeadBlock + 1) % w.full_attention_interval) != 0 ||
        !L.wq || !L.wk || !L.attn_norm || !L.q_norm || !L.k_norm) {
        error = "qwen35 scoring head block 15 is not a full-attention block";
        return false;
    }
    return true;
}

namespace {

static constexpr const char * kQwen35HeadSchema = "qwen3_5_0_8b_nope_qk_mass_v1";
static constexpr const char * kQwen35HeadBaseModel = "Qwen/Qwen3.5-0.8B";
static constexpr const char * kQwen35HeadFeatureTap =
    "post_block14_residual_before_block15";

static void free_qwen35_head(Qwen35DrafterState & st) {
    if (st.head_buf) { ggml_backend_buffer_free(st.head_buf); st.head_buf = nullptr; }
    if (st.head_ctx) { ggml_free(st.head_ctx); st.head_ctx = nullptr; }
    st.head_wq = st.head_wk = nullptr;
    st.head_file_id.clear();
    st.head_sha256.clear();
    st.head_loaded = false;
}

static void free_qwen35_segment_probe(Qwen35DrafterState & st) {
    if (st.probe_buf) { ggml_backend_buffer_free(st.probe_buf); st.probe_buf = nullptr; }
    if (st.probe_ctx) { ggml_free(st.probe_ctx); st.probe_ctx = nullptr; }
    st.probe_norm_w = st.probe_norm_b = st.probe_fc1_w = st.probe_fc1_b =
        st.probe_fc2_w = st.probe_fc2_b =
        st.probe_sub_fc2_w = st.probe_sub_fc2_b =
        st.probe_cont_fc2_w = st.probe_cont_fc2_b = nullptr;
    st.probe_conv_w.clear();
    st.probe_sub_conv_w.clear();
    st.probe_cont_conv_w.clear();
    st.probe_cont_conv_b = 0.0f;
    st.probe_loaded = false;
    st.probe_line_aligned = false;
}

static bool qwen35_metadata_equals(gguf_context * g, const char * key,
                                   const std::string & expected) {
    const int id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
           expected == gguf_get_val_str(g, id);
}

static FILE * open_snapshot_file_readonly(std::vector<uint8_t> & raw) {
    if (raw.empty()) return nullptr;
#if !defined(_WIN32)
    return ::fmemopen(raw.data(), raw.size(), "rb");
#else
    char tmp_dir[MAX_PATH] = {};
    const DWORD n = GetTempPathA(MAX_PATH, tmp_dir);
    if (n == 0 || n >= MAX_PATH) return nullptr;
    static std::atomic<uint64_t> seq{0};
    HANDLE h = INVALID_HANDLE_VALUE;
    const DWORD pid = GetCurrentProcessId();
    for (int attempt = 0; attempt < 32; ++attempt) {
        const unsigned long long id = (unsigned long long) seq.fetch_add(1, std::memory_order_relaxed);
        const unsigned long long tick = (unsigned long long) GetTickCount64();
        char tmp_path[MAX_PATH] = {};
        const int written = std::snprintf(
            tmp_path, sizeof(tmp_path), "%sluce_head_%lu_%llu_%llu.tmp",
            tmp_dir, (unsigned long) pid, tick, id);
        if (written <= 0 || (size_t) written >= sizeof(tmp_path)) return nullptr;
        h = CreateFileA(
            tmp_path,
            GENERIC_READ | GENERIC_WRITE,
            0,  // exclusive: no other handle can open, read, or overwrite
            nullptr,
            CREATE_NEW,
            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
            nullptr);
        if (h != INVALID_HANDLE_VALUE) break;
        if (GetLastError() != ERROR_FILE_EXISTS) return nullptr;
    }
    if (h == INVALID_HANDLE_VALUE) return nullptr;
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        CloseHandle(h);
        return nullptr;
    }
    FILE * snap_fp = _fdopen(fd, "w+b");
    if (!snap_fp) {
        _close(fd);
        return nullptr;
    }
    if (std::fwrite(raw.data(), 1, raw.size(), snap_fp) != raw.size() ||
        std::fflush(snap_fp) != 0) {
        std::fclose(snap_fp);
        return nullptr;
    }
    std::rewind(snap_fp);
    return snap_fp;
#endif
}

// Optional trained head for the block-15 tap. Fails closed on any contract
// mismatch.
static bool load_qwen35_scoring_head(const std::string & path,
                                     Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return false;
    }
    if (st.gguf_sha256.empty()) {
        set_last_error("scoring head requires the drafter GGUF identity hash");
        return false;
    }
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        set_last_error("scoring head GGUF could not be opened: " + path);
        return false;
    }
    const std::string head_fd_id = pflash_file_identity_from_fd(path, fileno(fp));
    // Read the entire scoring head once into memory so the SHA-256 digest and the
    // parsed GGUF weights come from the exact same immutable byte snapshot.
    static constexpr size_t kMaxScoringHeadBytes = 256u * 1024u * 1024u;
    std::vector<uint8_t> raw;
    {
        uint8_t buf[65536];
        for (;;) {
            const size_t n = std::fread(buf, 1, sizeof(buf), fp);
            if (n > 0) {
                if (raw.size() + n > kMaxScoringHeadBytes) {
                    std::fclose(fp);
                    set_last_error("scoring head GGUF exceeds size limit: " + path);
                    return false;
                }
                raw.insert(raw.end(), buf, buf + n);
            }
            if (n < sizeof(buf)) {
                if (std::ferror(fp)) {
                    std::fclose(fp);
                    set_last_error("scoring head GGUF read failed: " + path);
                    return false;
                }
                break;
            }
        }
    }
    const std::string head_fd_id_after = pflash_file_identity_from_fd(path, fileno(fp));
    std::fclose(fp);
    if (head_fd_id_after != head_fd_id) {
        set_last_error("scoring head GGUF changed on disk while loading: " + path);
        return false;
    }
    if (raw.empty()) {
        set_last_error("scoring head GGUF is empty: " + path);
        return false;
    }
    Sha256Ctx sha_ctx;
    sha256_init(sha_ctx);
    sha256_update(sha_ctx, raw.data(), raw.size());
    const std::string head_sha = sha256_final(sha_ctx);
    FILE * snap_fp = open_snapshot_file_readonly(raw);
    if (!snap_fp) {
        set_last_error("scoring head snapshot stream failed: " + path);
        return false;
    }
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file_ptr(snap_fp, params);
    std::fclose(snap_fp);
    if (!g) {
        if (data_ctx) ggml_free(data_ctx);
        set_last_error("scoring head GGUF could not be opened: " + path);
        return false;
    }
    auto fail = [&](const std::string & message) {
        free_qwen35_head(st);
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    // GGUF contract of a scoring-head file: architecture `pflash_scoring_head`,
    // metadata and tensors under `scoringhead.*`.
    if (!qwen35_metadata_equals(g, "general.architecture", "pflash_scoring_head") ||
        !qwen35_metadata_equals(g, "scoringhead.schema", kQwen35HeadSchema) ||
        !qwen35_metadata_equals(g, "scoringhead.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "scoringhead.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "scoringhead.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("scoring head metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    struct Contract {
        const char * name;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"scoringhead.attn_q.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head * w.n_embd_head_k, &st.head_wq},
        {"scoringhead.attn_k.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head_kv * w.n_embd_head_k, &st.head_wk},
    };
    ggml_init_params head_params{};
    head_params.mem_size = 4 * ggml_tensor_overhead();
    head_params.no_alloc = true;
    st.head_ctx = ggml_init(head_params);
    if (!st.head_ctx) return fail("scoring head context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = data_ctx ? ggml_get_tensor(data_ctx, contract.name) : nullptr;
        if (!source || source->type != GGML_TYPE_F32 || ggml_n_dims(source) != 2 ||
            source->ne[0] != contract.ne0 || source->ne[1] != contract.ne1) {
            return fail(std::string("scoring head tensor contract mismatch: ") +
                        contract.name);
        }
        *contract.destination =
            ggml_new_tensor_2d(st.head_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    st.head_buf = ggml_backend_alloc_ctx_tensors(st.head_ctx, w.backend);
    if (!st.head_buf) return fail("scoring head buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.head_file_id = head_fd_id;
    st.head_sha256 = head_sha;
    st.head_loaded = true;
    std::fprintf(stderr, "[qwen35-drafter] loaded scoring head: %s\n", path.c_str());
    std::fflush(stderr);
    return true;
}

static constexpr const char * kQwen35ProbeSchema = "qwen3_5_0_8b_segment_probe_v1";
static constexpr const char * kQwen35ProbeSchemaV2 = "qwen3_5_0_8b_segment_probe_v2";

static bool qwen35_metadata_f32(gguf_context * g, const char * key, float & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    out = gguf_get_val_f32(g, id);
    return true;
}

static bool qwen35_metadata_u32(gguf_context * g, const char * key, int & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return false;
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

// Optional segment probe for the block-14 tap: LayerNorm -> Linear -> GELU ->
// Linear on the GPU, a 5-tap smoothing on the CPU, sigmoid, cut above the
// threshold. Fails closed on any contract mismatch, like the head loader.
static bool load_qwen35_segment_probe(const std::string & path,
                                      Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    if (st.gguf_sha256.empty()) {
        set_last_error("segment probe requires the drafter GGUF identity hash");
        return false;
    }
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        set_last_error("segment probe GGUF could not be opened: " + path);
        return false;
    }
    const std::string probe_fd_id = pflash_file_identity_from_fd(path, fileno(fp));
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file_ptr(fp, params);
    std::fclose(fp);
    if (!g) {
        set_last_error("segment probe GGUF could not be opened: " + path);
        return false;
    }
    auto fail = [&](const std::string & message) {
        free_qwen35_segment_probe(st);
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "segmentprobe") ||
        (!qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchema) &&
         !qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchemaV2)) ||
        !qwen35_metadata_equals(g, "segmentprobe.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "segmentprobe.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "segmentprobe.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("segment probe metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    if (!qwen35_metadata_f32(g, "segmentprobe.threshold", st.probe_threshold) ||
        !qwen35_metadata_f32(g, "segmentprobe.norm_eps", st.probe_norm_eps) ||
        !qwen35_metadata_u32(g, "segmentprobe.min_segment", st.probe_min_segment) ||
        !qwen35_metadata_u32(g, "segmentprobe.max_segment", st.probe_max_segment) ||
        !(st.probe_threshold > 0.0f && st.probe_threshold < 1.0f) ||
        st.probe_min_segment < 1 || st.probe_max_segment < st.probe_min_segment) {
        return fail("segment probe parameters are missing or out of range");
    }
    {
        // Optional oversize split rule; an unknown value refuses the probe.
        st.probe_line_aligned = false;
        const int id = gguf_find_key(g, "segmentprobe.oversize_split");
        if (id >= 0) {
            if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
                return fail("segmentprobe.oversize_split must be a string");
            }
            const std::string rule = gguf_get_val_str(g, id);
            if (rule == "line_aligned") {
                st.probe_line_aligned = true;
            } else if (rule != "best_score") {
                return fail("unknown segmentprobe.oversize_split: " + rule);
            }
        }
    }
    ggml_tensor * fc1 = data_ctx ? ggml_get_tensor(data_ctx, "segmentprobe.fc1.weight") : nullptr;
    if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
        fc1->ne[0] != w.n_embd || fc1->ne[1] < 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.fc1.weight");
    }
    st.probe_width = (int) fc1->ne[1];
    struct Contract {
        const char * name;
        int n_dims;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"segmentprobe.norm.weight", 1, (int64_t) w.n_embd, 1, &st.probe_norm_w},
        {"segmentprobe.norm.bias", 1, (int64_t) w.n_embd, 1, &st.probe_norm_b},
        {"segmentprobe.fc1.weight", 2, (int64_t) w.n_embd, (int64_t) st.probe_width, &st.probe_fc1_w},
        {"segmentprobe.fc1.bias", 1, (int64_t) st.probe_width, 1, &st.probe_fc1_b},
        // ggml drops trailing unit dimensions: the [width, 1] output row is 1-D.
        {"segmentprobe.fc2.weight", 1, (int64_t) st.probe_width, 1, &st.probe_fc2_w},
        {"segmentprobe.fc2.bias", 1, 1, 1, &st.probe_fc2_b},
    };
    ggml_init_params probe_params{};
    probe_params.mem_size = 12 * ggml_tensor_overhead();
    probe_params.no_alloc = true;
    st.probe_ctx = ggml_init(probe_params);
    if (!st.probe_ctx) return fail("segment probe context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        if (!source || source->type != GGML_TYPE_F32 ||
            ggml_n_dims(source) != contract.n_dims ||
            source->ne[0] != contract.ne0 ||
            (contract.n_dims == 2 && source->ne[1] != contract.ne1)) {
            return fail(std::string("segment probe tensor contract mismatch: ") + contract.name);
        }
        *contract.destination = contract.n_dims == 1
            ? ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, contract.ne0)
            : ggml_new_tensor_2d(st.probe_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    ggml_tensor * conv_w = ggml_get_tensor(data_ctx, "segmentprobe.conv.weight");
    ggml_tensor * conv_b = ggml_get_tensor(data_ctx, "segmentprobe.conv.bias");
    if (!conv_w || conv_w->type != GGML_TYPE_F32 || ggml_n_dims(conv_w) != 1 ||
        conv_w->ne[0] < 1 || conv_w->ne[0] % 2 != 1 ||
        !conv_b || conv_b->type != GGML_TYPE_F32 || ggml_n_dims(conv_b) != 1 || conv_b->ne[0] != 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.conv");
    }
    st.probe_conv_w.assign((const float *) conv_w->data,
                           (const float *) conv_w->data + conv_w->ne[0]);
    st.probe_conv_b = ((const float *) conv_b->data)[0];
    // Optional sub-unit head (schema v2): scores feed only the oversize
    // split rule's interior argmax. All four tensors ship together or none.
    ggml_tensor * sub_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.fc2.weight");
    ggml_tensor * sub_b_src = nullptr;
    if (sub_src) {
        sub_b_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.fc2.bias");
        ggml_tensor * sub_cw_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.conv.weight");
        ggml_tensor * sub_cb_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.conv.bias");
        if (sub_src->type != GGML_TYPE_F32 || ggml_n_dims(sub_src) != 1 ||
            sub_src->ne[0] != (int64_t) st.probe_width ||
            !sub_b_src || sub_b_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_b_src) != 1 || sub_b_src->ne[0] != 1 ||
            !sub_cw_src || sub_cw_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_cw_src) != 1 || sub_cw_src->ne[0] != conv_w->ne[0] ||
            !sub_cb_src || sub_cb_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_cb_src) != 1 || sub_cb_src->ne[0] != 1) {
            return fail("segment probe tensor contract mismatch: segmentprobe.subunit");
        }
        st.probe_sub_fc2_w = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, st.probe_width);
        ggml_set_name(st.probe_sub_fc2_w, "segmentprobe.subunit.fc2.weight");
        st.probe_sub_fc2_b = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(st.probe_sub_fc2_b, "segmentprobe.subunit.fc2.bias");
        st.probe_sub_conv_w.assign((const float *) sub_cw_src->data,
                                   (const float *) sub_cw_src->data + sub_cw_src->ne[0]);
        st.probe_sub_conv_b = ((const float *) sub_cb_src->data)[0];
    }
    // Optional container head (schema v2): top-level container break scores
    // feed boundary-continuity features to the segment reranker.
    // All four tensors ship together or none.
    ggml_tensor * cont_src = ggml_get_tensor(data_ctx, "segmentprobe.container.fc2.weight");
    ggml_tensor * cont_b_src = nullptr;
    if (cont_src) {
        cont_b_src = ggml_get_tensor(data_ctx, "segmentprobe.container.fc2.bias");
        ggml_tensor * cont_cw_src = ggml_get_tensor(data_ctx, "segmentprobe.container.conv.weight");
        ggml_tensor * cont_cb_src = ggml_get_tensor(data_ctx, "segmentprobe.container.conv.bias");
        if (cont_src->type != GGML_TYPE_F32 || ggml_n_dims(cont_src) != 1 ||
            cont_src->ne[0] != (int64_t) st.probe_width ||
            !cont_b_src || cont_b_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(cont_b_src) != 1 || cont_b_src->ne[0] != 1 ||
            !cont_cw_src || cont_cw_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(cont_cw_src) != 1 || cont_cw_src->ne[0] != conv_w->ne[0] ||
            !cont_cb_src || cont_cb_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(cont_cb_src) != 1 || cont_cb_src->ne[0] != 1) {
            return fail("segment probe tensor contract mismatch: segmentprobe.container");
        }
        st.probe_cont_fc2_w = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, st.probe_width);
        ggml_set_name(st.probe_cont_fc2_w, "segmentprobe.container.fc2.weight");
        st.probe_cont_fc2_b = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(st.probe_cont_fc2_b, "segmentprobe.container.fc2.bias");
        st.probe_cont_conv_w.assign((const float *) cont_cw_src->data,
                                    (const float *) cont_cw_src->data + cont_cw_src->ne[0]);
        st.probe_cont_conv_b = ((const float *) cont_cb_src->data)[0];
    }
    st.probe_buf = ggml_backend_alloc_ctx_tensors(st.probe_ctx, w.backend);
    if (!st.probe_buf) return fail("segment probe buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    if (st.probe_sub_fc2_w) {
        ggml_backend_tensor_set(st.probe_sub_fc2_w, sub_src->data, 0, ggml_nbytes(sub_src));
        ggml_backend_tensor_set(st.probe_sub_fc2_b, sub_b_src->data, 0, ggml_nbytes(sub_b_src));
    }
    if (st.probe_cont_fc2_w) {
        ggml_backend_tensor_set(st.probe_cont_fc2_w, cont_src->data, 0, ggml_nbytes(cont_src));
        ggml_backend_tensor_set(st.probe_cont_fc2_b, cont_b_src->data, 0, ggml_nbytes(cont_b_src));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.probe_file_id = probe_fd_id;
    st.probe_loaded = true;
    std::fprintf(stderr,
        "[qwen35-drafter] loaded segment probe: %s (width %d, threshold %.3f, "
        "segments %d-%d tokens, %zu conv taps%s%s%s)\n",
        path.c_str(), st.probe_width, st.probe_threshold,
        st.probe_min_segment, st.probe_max_segment, st.probe_conv_w.size(),
        st.probe_sub_fc2_w ? ", sub-unit head" : "",
        st.probe_cont_fc2_w ? ", container head" : "",
        st.probe_line_aligned ? ", line-aligned oversize split" : "");
    std::fflush(stderr);
    return true;
}

static void free_qwen35_request_probe(Qwen35DrafterState & st) {
    if (st.request_buf) { ggml_backend_buffer_free(st.request_buf); st.request_buf = nullptr; }
    if (st.request_ctx) { ggml_free(st.request_ctx); st.request_ctx = nullptr; }
    st.request_norm_w = st.request_norm_b = st.request_fc1_w = st.request_fc1_b =
        st.request_fc2_w = st.request_fc2_b = nullptr;
    st.request_width = 0;
    st.request_loaded = false;
    st.request_turns = false;
    st.request_v5 = false;
    st.request_checkpoint_sha256.clear();
    st.request_policy = luce::pflash::PFlashRequestPolicy{};
}

static constexpr const char * kQwen35RequestProbeSchema = "qwen3_5_0_8b_request_probe_v1";

// Request probe: its own LayerNorm -> fc1 -> GELU(erf) -> fc2 trunk on the
// block-14 tap, from PFLASH_REQUEST_PROBE_GGUF. Fails closed on any contract
// mismatch, like the segment probe.
static bool load_qwen35_request_probe(const char * path,
                                      Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    ggml_context * data_ctx = nullptr;
    gguf_context * g = nullptr;
    auto fail = [&](const std::string & message) {
        free_qwen35_request_probe(st);
        if (g) gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    if (!path || !*path) {
        return fail("PFLASH_REQUEST_PROBE_GGUF is empty");
    }
    if (st.gguf_sha256.empty()) {
        return fail("request probe requires the drafter GGUF identity hash");
    }
    FILE * fp = std::fopen(path, "rb");
    if (!fp) return fail(std::string("request probe GGUF could not be opened: ") + path);
    const std::string req_fd_id = pflash_file_identity_from_fd(path, fileno(fp));
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    g = gguf_init_from_file_ptr(fp, params);
    std::fclose(fp);
    if (!g) return fail(std::string("request probe GGUF could not be opened: ") + path);
    if (!qwen35_metadata_equals(g, "general.architecture", "requestprobe") ||
        !qwen35_metadata_equals(g, "requestprobe.schema", kQwen35RequestProbeSchema) ||
        !qwen35_metadata_equals(g, "requestprobe.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "requestprobe.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "requestprobe.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("request probe metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    if (!qwen35_metadata_f32(g, "requestprobe.norm_eps", st.request_norm_eps) ||
        !(st.request_norm_eps > 0.0f) ||
        !qwen35_metadata_f32(g, "requestprobe.tau", st.request_tau) ||
        std::isnan(st.request_tau)) {
        return fail("request probe metadata requestprobe.norm_eps / requestprobe.tau "
                    "is missing or out of range");
    }
    if (!qwen35_metadata_equals(g, "requestprobe.extraction_version",
                                luce::pflash::kRequestExtractionVersion)) {
        return fail(std::string("request probe metadata requestprobe.extraction_version "
                    "is missing or is not ") + luce::pflash::kRequestExtractionVersion +
                    " (the runtime's request-run extraction rule)");
    }
    st.request_turns = false;
    st.request_v5 = false;
    if (gguf_find_key(g, luce::pflash::kRequestCandidateRuleKey) >= 0) {
        if (qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey,
                                   luce::pflash::kRequestCandidateRule)) {
            st.request_turns = true;
        } else if (qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey,
                                          luce::pflash::kRequestCandidateRuleV5)) {
            st.request_v5 = true;
        } else {
            return fail(std::string("request probe metadata ") +
                        luce::pflash::kRequestCandidateRuleKey + " is not " +
                        luce::pflash::kRequestCandidateRule + " or " + luce::pflash::kRequestCandidateRuleV5 +
                        " (the runtime's candidate rules)");
        }
    }
    st.request_checkpoint_sha256.clear();
    {
        const int64_t id = gguf_find_key(g, luce::pflash::kRequestCheckpointShaKey);
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING) {
            st.request_checkpoint_sha256 = gguf_get_val_str(g, id);
        }
    }
    ggml_tensor * fc1 = ggml_get_tensor(data_ctx, "requestprobe.fc1.weight");
    if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
        fc1->ne[0] != w.n_embd || fc1->ne[1] < 1) {
        return fail("request probe tensor contract mismatch: requestprobe.fc1.weight");
    }
    st.request_width = (int) fc1->ne[1];
    struct Contract {
        const char * name;
        int n_dims;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"requestprobe.norm.weight", 1, (int64_t) w.n_embd, 1, &st.request_norm_w},
        {"requestprobe.norm.bias", 1, (int64_t) w.n_embd, 1, &st.request_norm_b},
        {"requestprobe.fc1.weight", 2, (int64_t) w.n_embd, (int64_t) st.request_width, &st.request_fc1_w},
        {"requestprobe.fc1.bias", 1, (int64_t) st.request_width, 1, &st.request_fc1_b},
        {"requestprobe.fc2.weight", 1, (int64_t) st.request_width, 1, &st.request_fc2_w},
        {"requestprobe.fc2.bias", 1, 1, 1, &st.request_fc2_b},
    };
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    st.request_ctx = ggml_init(ip);
    if (!st.request_ctx) return fail("request probe context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        if (!source || source->type != GGML_TYPE_F32 ||
            ggml_n_dims(source) != contract.n_dims ||
            source->ne[0] != contract.ne0 ||
            (contract.n_dims == 2 && source->ne[1] != contract.ne1)) {
            return fail(std::string("request probe tensor contract mismatch: ") + contract.name);
        }
        *contract.destination = contract.n_dims == 1
            ? ggml_new_tensor_1d(st.request_ctx, GGML_TYPE_F32, contract.ne0)
            : ggml_new_tensor_2d(st.request_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    st.request_buf = ggml_backend_alloc_ctx_tensors(st.request_ctx, w.backend);
    if (!st.request_buf) return fail("request probe buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    luce::pflash::PFlashRequestPolicy policy;
    const struct { const char * key; int * dst; } ints[] = {
        {"requestprobe.policy.min_prompt_tokens", &policy.min_prompt_tokens},
        {"requestprobe.policy.max_prompt_tokens", &policy.max_prompt_tokens},
        {"requestprobe.policy.max_runs", &policy.max_runs},
        {"requestprobe.policy.recency_pin_tokens", &policy.recency_pin_tokens},
    };
    for (const auto & k : ints) {
        if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_u32(g, k.key, *k.dst)) {
            return fail(std::string("request probe metadata ") + k.key + " must be a u32");
        }
    }
    const struct { const char * key; float * dst; } floats[] = {
        {"requestprobe.policy.min_mean_margin", &policy.min_mean_margin},
        {"requestprobe.policy.min_max_margin", &policy.min_max_margin},
        {"requestprobe.policy.min_latest_share", &policy.min_latest_share},
        {"requestprobe.policy.max_latest_share", &policy.max_latest_share},
    };
    for (const auto & k : floats) {
        if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_f32(g, k.key, *k.dst)) {
            return fail(std::string("request probe metadata ") + k.key + " must be an f32");
        }
    }
    std::string policy_error;
    if (!luce::pflash::pflash_request_policy_valid(policy, policy_error)) return fail(policy_error);
    st.request_policy = policy;
    gguf_free(g);
    ggml_free(data_ctx);
    st.request_probe_file_id = req_fd_id;
    st.request_loaded = true;
    std::fprintf(stderr, "[qwen35-drafter] request probe candidate rule: %s\n",
                 st.request_v5 ? luce::pflash::kRequestCandidateRuleV5
                 : st.request_turns ? luce::pflash::kRequestCandidateRule : "human messages only");
    std::fprintf(stderr,
        "[qwen35-drafter] loaded request probe: %s (width %d, tau %g; policy: prompt [%d,%d], "
        "min mean/max margin %g/%g, latest share [%g,%g], max runs %d, recency pin %d)\n",
        path, st.request_width, (double) st.request_tau, policy.min_prompt_tokens, policy.max_prompt_tokens,
        (double) policy.min_mean_margin, (double) policy.min_max_margin, (double) policy.min_latest_share,
        (double) policy.max_latest_share, policy.max_runs, policy.recency_pin_tokens);
    std::fflush(stderr);
    return true;
}

// A file replaced in place keeps neither its inode nor its ctime, even when
// size and mtime are preserved (cp -p, rsync -t, tar).
struct DrafterFileStatKey {
    std::string path;
    int64_t size = -1;
    int64_t mtime_sec = 0;
    int64_t mtime_nsec = 0;
    int64_t ctime_sec = 0;
    int64_t ctime_nsec = 0;
    uint64_t dev = 0;
    uint64_t ino = 0;
    bool valid = false;

    bool operator==(const DrafterFileStatKey & o) const {
        return valid && o.valid && path == o.path && size == o.size &&
               mtime_sec == o.mtime_sec && mtime_nsec == o.mtime_nsec &&
               ctime_sec == o.ctime_sec && ctime_nsec == o.ctime_nsec &&
               dev == o.dev && ino == o.ino;
    }
};

static DrafterFileStatKey stat_from_buf(const std::string & path, const struct stat & st) {
    DrafterFileStatKey k;
    k.path = path;
    k.size = (int64_t) st.st_size;
    k.mtime_sec = (int64_t) st.st_mtime;
    k.ctime_sec = (int64_t) st.st_ctime;
    k.dev = (uint64_t) st.st_dev;
    k.ino = (uint64_t) st.st_ino;
#if defined(__APPLE__)
    k.mtime_nsec = (int64_t) st.st_mtimespec.tv_nsec;
    k.ctime_nsec = (int64_t) st.st_ctimespec.tv_nsec;
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200809L || defined(__linux__)
    k.mtime_nsec = (int64_t) st.st_mtim.tv_nsec;
    k.ctime_nsec = (int64_t) st.st_ctim.tv_nsec;
#endif
    k.valid = true;
    return k;
}

static DrafterFileStatKey stat_drafter_file(const std::string & path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) return stat_from_buf(path, st);
    DrafterFileStatKey k;
    k.path = path;
    return k;
}

static std::string format_stat_identity(const DrafterFileStatKey & fk) {
    if (fk.path.empty()) return {};
    std::string out = fk.path;
    if (fk.valid) {
        out += "@" + std::to_string((long long) fk.size) + ":" +
               std::to_string((long long) fk.mtime_sec) + "." +
               std::to_string((long long) fk.mtime_nsec) + ":" +
               std::to_string((long long) fk.ctime_sec) + "." +
               std::to_string((long long) fk.ctime_nsec) + ":" +
               std::to_string((unsigned long long) fk.ino);
    }
    return out;
}

static GgufMetadata cached_drafter_metadata(const std::string & path, bool compute_sha256) {
    struct Entry {
        DrafterFileStatKey key;
        GgufMetadata meta;
        bool has_sha256 = false;
    };
    static std::mutex mu;
    static std::unordered_map<std::string, Entry> cache;

    const DrafterFileStatKey fk = stat_drafter_file(path);
    if (fk.valid) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.key == fk &&
            (!compute_sha256 || it->second.has_sha256)) {
            return it->second.meta;
        }
    }
    // Never trust a size-only `.sha256` sidecar for drafter validation: a cold-start
    // same-size replacement could otherwise inherit the previous drafter's digest.
    GgufMetadata meta = read_gguf_metadata(path, /*compute_sha256=*/false);
    if (compute_sha256 && meta.ok) {
        meta.sha256 = luce::common::sha256_of_file(path);
        if (meta.sha256.empty()) meta.ok = false;
    }
    const DrafterFileStatKey fk_after = stat_drafter_file(path);
    if (!fk.valid || !(fk_after == fk)) {
        meta.ok = false;
        meta.sha256.clear();
        return meta;
    }
    if (meta.ok) {
        std::lock_guard<std::mutex> lk(mu);
        cache[path] = Entry{fk, meta, compute_sha256 && !meta.sha256.empty()};
    }
    return meta;
}

// Hashes the file itself: a retrained head has the same size as the one it
// replaces, which a size-guarded sidecar cannot tell apart.
static std::string cached_file_sha256(const std::string & path) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::pair<DrafterFileStatKey, std::string>> cache;

    const DrafterFileStatKey fk = stat_drafter_file(path);
    if (fk.valid) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.first == fk) return it->second.second;
    }
    std::string sha = luce::common::sha256_of_file(path);
    const DrafterFileStatKey fk_after = stat_drafter_file(path);
    if (!fk.valid || !(fk_after == fk)) return {};
    if (!sha.empty()) {
        std::lock_guard<std::mutex> lk(mu);
        cache[path] = {fk, sha};
    }
    return sha;
}

static std::shared_ptr<const Tokenizer> cached_drafter_tokenizer(const std::string & path) {
    struct Entry {
        DrafterFileStatKey key;
        std::shared_ptr<const Tokenizer> tok;
    };
    static std::mutex mu;
    static std::unordered_map<std::string, Entry> cache;

    const DrafterFileStatKey fk = stat_drafter_file(path);
    if (fk.valid) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.key == fk && it->second.tok) {
            return it->second.tok;
        }
    }
    auto tok = std::make_shared<Tokenizer>();
    if (!tok->load_from_gguf(path.c_str())) {
        return {};
    }
    std::shared_ptr<const Tokenizer> out = tok;
    if (fk.valid) {
        std::lock_guard<std::mutex> lk(mu);
        cache[path] = Entry{fk, out};
    }
    return out;
}

static bool drafter_only_needs_head_prefix(const char * head_path) {
    if (!head_path || !*head_path) return false;
    // KVFlash's drafter residency scorer can attach to this drafter via
    // window.loaded whenever KVFlash is active (including policy=lru) and
    // runs all drafter blocks, so keep the full drafter loaded under KVFlash.
    if (luce::common::kvflash_pool_requested(std::getenv("LUCE_KVFLASH"))) {
        return false;
    }
    return true;
}

static bool check_gguf_full_data_extent(
        gguf_context * g,
        uint64_t file_size,
        std::string & error) {
    const uint64_t data_offset = (uint64_t) gguf_get_data_offset(g);
    const uint64_t alignment = (uint64_t) gguf_get_alignment(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    uint64_t total_data = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const uint64_t off = (uint64_t) gguf_get_tensor_offset(g, i);
        const uint64_t sz = (uint64_t) gguf_get_tensor_size(g, i);
        const uint64_t rem = alignment > 1 ? (sz % alignment) : 0;
        const uint64_t pad = rem ? (alignment - rem) : 0;
        if (sz > UINT64_MAX - pad || off > UINT64_MAX - (sz + pad)) {
            error = "GGUF tensor size/offset overflow";
            return false;
        }
        const uint64_t end = off + sz + pad;
        if (end > total_data) total_data = end;
    }
    if (data_offset > file_size || total_data > file_size - data_offset) {
        error = "GGUF truncated data section";
        return false;
    }
    return true;
}

static bool open_gguf_header_only(
        const std::string & path,
        gguf_context *& g_out,
        ggml_context *& ctx_out,
        uint64_t & file_size_out) {
    g_out = nullptr;
    ctx_out = nullptr;
    file_size_out = 0;
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    const long fsz = std::ftell(fp);
    if (fsz < 0) {
        std::fclose(fp);
        return false;
    }
    file_size_out = (uint64_t) fsz;
    std::rewind(fp);
    gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ &ctx_out };
    g_out = gguf_init_from_file_ptr(fp, params);
    std::fclose(fp);
    if (!g_out || !ctx_out) {
        if (g_out) { gguf_free(g_out); g_out = nullptr; }
        if (ctx_out) { ggml_free(ctx_out); ctx_out = nullptr; }
        return false;
    }
    return true;
}

static bool validate_head_gguf_header(
        const std::string & path,
        const std::string & drafter_sha256,
        int n_embd,
        int n_head,
        int n_head_kv,
        int n_embd_head_k,
        std::string & error) {
    gguf_context * g = nullptr;
    ggml_context * data_ctx = nullptr;
    uint64_t file_size = 0;
    if (!open_gguf_header_only(path, g, data_ctx, file_size)) {
        error = "scoring head GGUF could not be opened: " + path;
        return false;
    }
    auto fail = [&](const std::string & msg) {
        gguf_free(g);
        ggml_free(data_ctx);
        error = msg;
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "pflash_scoring_head") ||
        !qwen35_metadata_equals(g, "scoringhead.schema", kQwen35HeadSchema) ||
        !qwen35_metadata_equals(g, "scoringhead.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "scoringhead.runtime_gguf_sha256", drafter_sha256) ||
        !qwen35_metadata_equals(g, "scoringhead.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("scoring head metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    const struct { const char * name; int64_t ne0; int64_t ne1; } reqs[] = {
        {"scoringhead.attn_q.weight", (int64_t) n_embd, (int64_t) n_head * n_embd_head_k},
        {"scoringhead.attn_k.weight", (int64_t) n_embd, (int64_t) n_head_kv * n_embd_head_k},
    };
    for (const auto & r : reqs) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, r.name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) != 2 ||
            t->ne[0] != r.ne0 || t->ne[1] != r.ne1) {
            return fail(std::string("scoring head tensor contract mismatch: ") + r.name);
        }
    }
    std::string ext_err;
    if (!check_gguf_full_data_extent(g, file_size, ext_err)) {
        return fail("scoring head " + ext_err);
    }
    gguf_free(g);
    ggml_free(data_ctx);
    return true;
}

static bool validate_segment_probe_gguf_header(
        const std::string & path,
        const std::string & drafter_sha256,
        int n_embd,
        std::string & error) {
    gguf_context * g = nullptr;
    ggml_context * data_ctx = nullptr;
    uint64_t file_size = 0;
    if (!open_gguf_header_only(path, g, data_ctx, file_size)) {
        error = "segment probe GGUF could not be opened: " + path;
        return false;
    }
    auto fail = [&](const std::string & msg) {
        gguf_free(g);
        ggml_free(data_ctx);
        error = msg;
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "segmentprobe") ||
        (!qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchema) &&
         !qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchemaV2)) ||
        !qwen35_metadata_equals(g, "segmentprobe.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "segmentprobe.runtime_gguf_sha256", drafter_sha256) ||
        !qwen35_metadata_equals(g, "segmentprobe.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("segment probe metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    float threshold = 0.0f, norm_eps = 0.0f;
    int min_seg = 0, max_seg = 0;
    if (!qwen35_metadata_f32(g, "segmentprobe.threshold", threshold) ||
        !qwen35_metadata_f32(g, "segmentprobe.norm_eps", norm_eps) ||
        !qwen35_metadata_u32(g, "segmentprobe.min_segment", min_seg) ||
        !qwen35_metadata_u32(g, "segmentprobe.max_segment", max_seg) ||
        !(threshold > 0.0f && threshold < 1.0f) ||
        min_seg < 1 || max_seg < min_seg) {
        return fail("segment probe parameters are missing or out of range");
    }
    const int ov_id = gguf_find_key(g, "segmentprobe.oversize_split");
    if (ov_id >= 0) {
        if (gguf_get_kv_type(g, ov_id) != GGUF_TYPE_STRING) {
            return fail("segmentprobe.oversize_split must be a string");
        }
        const std::string rule = gguf_get_val_str(g, ov_id);
        if (rule != "line_aligned" && rule != "best_score") {
            return fail("unknown segmentprobe.oversize_split: " + rule);
        }
    }
    ggml_tensor * fc1 = ggml_get_tensor(data_ctx, "segmentprobe.fc1.weight");
    if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
        fc1->ne[0] != n_embd || fc1->ne[1] < 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.fc1.weight");
    }
    const int64_t width = fc1->ne[1];
    const struct { const char * name; int n_dims; int64_t ne0; int64_t ne1; } reqs[] = {
        {"segmentprobe.norm.weight", 1, (int64_t) n_embd, 1},
        {"segmentprobe.norm.bias", 1, (int64_t) n_embd, 1},
        {"segmentprobe.fc1.weight", 2, (int64_t) n_embd, width},
        {"segmentprobe.fc1.bias", 1, width, 1},
        {"segmentprobe.fc2.weight", 1, width, 1},
        {"segmentprobe.fc2.bias", 1, 1, 1},
    };
    for (const auto & r : reqs) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, r.name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) != r.n_dims ||
            t->ne[0] != r.ne0 || (r.n_dims == 2 && t->ne[1] != r.ne1)) {
            return fail(std::string("segment probe tensor contract mismatch: ") + r.name);
        }
    }
    ggml_tensor * conv_w = ggml_get_tensor(data_ctx, "segmentprobe.conv.weight");
    ggml_tensor * conv_b = ggml_get_tensor(data_ctx, "segmentprobe.conv.bias");
    if (!conv_w || conv_w->type != GGML_TYPE_F32 || ggml_n_dims(conv_w) != 1 ||
        conv_w->ne[0] < 1 || conv_w->ne[0] % 2 != 1 ||
        !conv_b || conv_b->type != GGML_TYPE_F32 || ggml_n_dims(conv_b) != 1 || conv_b->ne[0] != 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.conv");
    }
    // Validate optional v2 subunit and container heads identically to load_qwen35_segment_probe.
    const auto check_optional_head = [&](const char * group) -> bool {
        const std::string p = std::string("segmentprobe.") + group + ".";
        ggml_tensor * w_t = ggml_get_tensor(data_ctx, (p + "fc2.weight").c_str());
        if (!w_t) return true;
        ggml_tensor * b_t = ggml_get_tensor(data_ctx, (p + "fc2.bias").c_str());
        ggml_tensor * cw_t = ggml_get_tensor(data_ctx, (p + "conv.weight").c_str());
        ggml_tensor * cb_t = ggml_get_tensor(data_ctx, (p + "conv.bias").c_str());
        if (w_t->type != GGML_TYPE_F32 || ggml_n_dims(w_t) != 1 || w_t->ne[0] != width ||
            !b_t || b_t->type != GGML_TYPE_F32 || ggml_n_dims(b_t) != 1 || b_t->ne[0] != 1 ||
            !cw_t || cw_t->type != GGML_TYPE_F32 || ggml_n_dims(cw_t) != 1 || cw_t->ne[0] != conv_w->ne[0] ||
            !cb_t || cb_t->type != GGML_TYPE_F32 || ggml_n_dims(cb_t) != 1 || cb_t->ne[0] != 1) {
            error = std::string("segment probe tensor contract mismatch: segmentprobe.") + group;
            return false;
        }
        return true;
    };
    if (!check_optional_head("subunit") || !check_optional_head("container")) {
        return fail(error);
    }
    std::string ext_err;
    if (!check_gguf_full_data_extent(g, file_size, ext_err)) {
        return fail("segment probe " + ext_err);
    }
    gguf_free(g);
    ggml_free(data_ctx);
    return true;
}

static bool validate_request_probe_gguf_header(
        const std::string & path,
        const std::string & drafter_sha256,
        int n_embd,
        std::string & error) {
    gguf_context * g = nullptr;
    ggml_context * data_ctx = nullptr;
    uint64_t file_size = 0;
    if (!open_gguf_header_only(path, g, data_ctx, file_size)) {
        error = "request probe GGUF could not be opened: " + path;
        return false;
    }
    auto fail = [&](const std::string & msg) {
        gguf_free(g);
        ggml_free(data_ctx);
        error = msg;
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "requestprobe") ||
        !qwen35_metadata_equals(g, "requestprobe.schema", kQwen35RequestProbeSchema) ||
        !qwen35_metadata_equals(g, "requestprobe.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "requestprobe.runtime_gguf_sha256", drafter_sha256) ||
        !qwen35_metadata_equals(g, "requestprobe.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("request probe metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    float norm_eps = 0.0f, tau = 0.0f;
    if (!qwen35_metadata_f32(g, "requestprobe.norm_eps", norm_eps) || !(norm_eps > 0.0f) ||
        !qwen35_metadata_f32(g, "requestprobe.tau", tau) || std::isnan(tau)) {
        return fail("request probe metadata requestprobe.norm_eps / requestprobe.tau is missing or out of range");
    }
    if (!qwen35_metadata_equals(g, "requestprobe.extraction_version",
                                luce::pflash::kRequestExtractionVersion)) {
        return fail(std::string("request probe metadata requestprobe.extraction_version is missing or is not ") +
                    luce::pflash::kRequestExtractionVersion);
    }
    if (gguf_find_key(g, luce::pflash::kRequestCandidateRuleKey) >= 0 &&
        !qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey, luce::pflash::kRequestCandidateRule) &&
        !qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey, luce::pflash::kRequestCandidateRuleV5)) {
        return fail(std::string("request probe metadata ") + luce::pflash::kRequestCandidateRuleKey + " is unsupported");
    }
    ggml_tensor * fc1 = ggml_get_tensor(data_ctx, "requestprobe.fc1.weight");
    if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
        fc1->ne[0] != n_embd || fc1->ne[1] < 1) {
        return fail("request probe tensor contract mismatch: requestprobe.fc1.weight");
    }
    const int64_t width = fc1->ne[1];
    const struct { const char * name; int n_dims; int64_t ne0; int64_t ne1; } reqs[] = {
        {"requestprobe.norm.weight", 1, (int64_t) n_embd, 1},
        {"requestprobe.norm.bias", 1, (int64_t) n_embd, 1},
        {"requestprobe.fc1.weight", 2, (int64_t) n_embd, width},
        {"requestprobe.fc1.bias", 1, width, 1},
        {"requestprobe.fc2.weight", 1, width, 1},
        {"requestprobe.fc2.bias", 1, 1, 1},
    };
    for (const auto & r : reqs) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, r.name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) != r.n_dims ||
            t->ne[0] != r.ne0 || (r.n_dims == 2 && t->ne[1] != r.ne1)) {
            return fail(std::string("request probe tensor contract mismatch: ") + r.name);
        }
    }
    luce::pflash::PFlashRequestPolicy policy;
    const struct { const char * key; int * dst; } ints[] = {
        {"requestprobe.policy.min_prompt_tokens", &policy.min_prompt_tokens},
        {"requestprobe.policy.max_prompt_tokens", &policy.max_prompt_tokens},
        {"requestprobe.policy.max_runs", &policy.max_runs},
        {"requestprobe.policy.recency_pin_tokens", &policy.recency_pin_tokens},
    };
    for (const auto & k : ints) {
        if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_u32(g, k.key, *k.dst)) {
            return fail(std::string("request probe metadata ") + k.key + " must be a u32");
        }
    }
    const struct { const char * key; float * dst; } floats[] = {
        {"requestprobe.policy.min_mean_margin", &policy.min_mean_margin},
        {"requestprobe.policy.min_max_margin", &policy.min_max_margin},
        {"requestprobe.policy.min_latest_share", &policy.min_latest_share},
        {"requestprobe.policy.max_latest_share", &policy.max_latest_share},
    };
    for (const auto & k : floats) {
        if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_f32(g, k.key, *k.dst)) {
            return fail(std::string("request probe metadata ") + k.key + " must be an f32");
        }
    }
    std::string policy_err;
    if (!luce::pflash::pflash_request_policy_valid(policy, policy_err)) return fail(policy_err);
    std::string ext_err;
    if (!check_gguf_full_data_extent(g, file_size, ext_err)) {
        return fail("request probe " + ext_err);
    }
    gguf_free(g);
    ggml_free(data_ctx);
    return true;
}

} // namespace

std::string pflash_file_identity(const std::string & path) {
    if (path.empty()) return {};
    return format_stat_identity(stat_drafter_file(path));
}

std::string pflash_file_identity_from_fd(const std::string & path, int fd) {
    if (path.empty()) return {};
#if !defined(_WIN32)
    struct stat st{};
    if (fd >= 0 && ::fstat(fd, &st) == 0) {
        return format_stat_identity(stat_from_buf(path, st));
    }
#else
    (void) fd;
#endif
    return pflash_file_identity(path);
}

bool validate_pflash_component_ggufs(
        const std::string & drafter_path,
        const std::string & head_path,
        const std::string & segment_probe_path,
        const std::string & request_probe_path,
        const std::string & seg_reranker_path,
        std::string & error) {
    error.clear();
    if (drafter_path.empty() || head_path.empty() || segment_probe_path.empty() ||
        request_probe_path.empty() || seg_reranker_path.empty()) {
        error = "all four PFlash component paths and the drafter path are required";
        return false;
    }
    gguf_context * dg = nullptr;
    ggml_context * dctx = nullptr;
    uint64_t drafter_file_size = 0;
    if (!open_gguf_header_only(drafter_path, dg, dctx, drafter_file_size)) {
        error = "PFlash drafter GGUF could not be opened: " + drafter_path;
        return false;
    }
    std::string arch;
    const int arch_id = gguf_find_key(dg, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(dg, arch_id) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(dg, arch_id);
    }
    int n_embd = 0, n_head = 0, n_head_kv = 0, n_embd_head_k = 0, n_layer = 0, fai = 0;
    const auto d_u32 = [&](const std::string & key, int & out) {
        return qwen35_metadata_u32(dg, key.c_str(), out);
    };
    const bool drafter_ok = !arch.empty() &&
        d_u32(arch + ".embedding_length", n_embd) && n_embd > 0 &&
        d_u32(arch + ".attention.head_count", n_head) && n_head > 0 &&
        d_u32(arch + ".attention.head_count_kv", n_head_kv) && n_head_kv > 0 &&
        d_u32(arch + ".attention.key_length", n_embd_head_k) && n_embd_head_k > 0 &&
        d_u32(arch + ".block_count", n_layer) && n_layer > kQwen35HeadBlock &&
        d_u32(arch + ".full_attention_interval", fai) && fai > 0 &&
        ((kQwen35HeadBlock + 1) % fai) == 0;
    std::string drafter_ext_err;
    const bool drafter_extent_ok = drafter_ok &&
        check_gguf_full_data_extent(dg, drafter_file_size, drafter_ext_err);
    gguf_free(dg);
    ggml_free(dctx);
    if (!drafter_ok) {
        error = "PFlash drafter GGUF metadata missing or incompatible (need Qwen3.5 with >= 16 blocks): " +
                drafter_path;
        return false;
    }
    if (!drafter_extent_ok) {
        error = "PFlash drafter " + drafter_ext_err + ": " + drafter_path;
        return false;
    }
    const GgufMetadata drafter_meta = cached_drafter_metadata(drafter_path, /*compute_sha256=*/true);
    if (!drafter_meta.ok || drafter_meta.sha256.empty()) {
        error = "PFlash drafter GGUF could not be opened or hashed: " + drafter_path;
        return false;
    }

    if (!validate_head_gguf_header(head_path, drafter_meta.sha256,
                                   n_embd, n_head, n_head_kv, n_embd_head_k, error)) {
        return false;
    }
    if (!validate_segment_probe_gguf_header(segment_probe_path, drafter_meta.sha256, n_embd, error)) {
        return false;
    }
    if (!validate_request_probe_gguf_header(request_probe_path, drafter_meta.sha256, n_embd, error)) {
        return false;
    }
    const std::string head_sha256 = cached_file_sha256(head_path);
    const int k_dim = n_embd_head_k * n_head_kv;
    if (!validate_pflash_seg_reranker_gguf(seg_reranker_path, drafter_meta.sha256,
                                           head_sha256, k_dim, error)) {
        return false;
    }
    return true;
}

bool load_qwen35_drafter(const std::string & gguf_path,
                         DrafterContext & out) {
    auto * st = new Qwen35DrafterState();
    st->gguf_path = gguf_path;
    const bool skip_assets = (out.asset_mode == DrafterAssetMode::SkipAll);
    const bool require_all = (out.asset_mode == DrafterAssetMode::RequireAll);
    const char * head_path = skip_assets ? nullptr : std::getenv("PFLASH_SCORING_HEAD_GGUF");
    const char * probe_path = skip_assets ? nullptr : std::getenv("PFLASH_SEGMENT_PROBE_GGUF");
    const char * request_path = skip_assets ? nullptr : std::getenv("PFLASH_REQUEST_PROBE_GGUF");
    const char * reranker_path = skip_assets ? nullptr : std::getenv("PFLASH_SEG_RERANKER_GGUF");
    if (require_all) {
        if (!head_path || !*head_path || !probe_path || !*probe_path ||
            !request_path || !*request_path || !reranker_path || !*reranker_path) {
            set_last_error("PFlash requires all four component env vars (PFLASH_SCORING_HEAD_GGUF, "
                           "PFLASH_SEGMENT_PROBE_GGUF, PFLASH_REQUEST_PROBE_GGUF, PFLASH_SEG_RERANKER_GGUF)");
            delete st;
            return false;
        }
    }
    const bool need_sha256 = (head_path != nullptr) || (probe_path != nullptr) ||
                             (request_path != nullptr) || (reranker_path != nullptr);
    const DrafterFileStatKey drafter_fk_before = stat_drafter_file(gguf_path);
    st->drafter_file_id = format_stat_identity(drafter_fk_before);
    const GgufMetadata meta = cached_drafter_metadata(gguf_path, need_sha256);
    st->ctx_train = meta.ok && meta.context_length > 0 ? meta.context_length : 0;
    if (need_sha256) {
        st->gguf_sha256 = meta.ok ? meta.sha256 : std::string();
    }
    // The scorer never needs logits, and tied-embedding Qwen3.5-0.8B
    // exports omit output.weight, so skip the lm_head entirely. The scoring
    // head reads block 15, so blocks 16..N-1 stay off the GPU unless KVFlash
    // needs the full drafter (drafter_only_needs_head_prefix).
    TargetLoadPlan plan;
    plan.load_output = false;
    if (!skip_assets && drafter_only_needs_head_prefix(head_path) &&
        meta.ok && meta.block_count > kQwen35HeadBlock + 1) {
        plan.layer_end = kQwen35HeadBlock + 1;
    }
    if (!load_target_gguf_partial(gguf_path, out.backend, plan, st->weights)) {
        delete st;
        return false;
    }
    const DrafterFileStatKey drafter_fk_after = stat_drafter_file(gguf_path);
    if (!drafter_fk_before.valid || !(drafter_fk_after == drafter_fk_before)) {
        set_last_error("drafter GGUF changed on disk while loading");
        free_target_weights(st->weights);
        delete st;
        return false;
    }
    if (probe_path) {
        if (!*probe_path || !load_qwen35_segment_probe(probe_path, *st)) {
            if (!*probe_path) {
                set_last_error("PFLASH_SEGMENT_PROBE_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: segment probe load failed, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (request_path) {
        if (!*request_path || !load_qwen35_request_probe(request_path, *st)) {
            if (!*request_path) {
                set_last_error("PFLASH_REQUEST_PROBE_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: request probe load failed (%s), "
                "refusing to serve without it\n", luce_last_error());
            std::fflush(stderr);
            free_qwen35_segment_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (head_path) {
        if (!*head_path || !load_qwen35_scoring_head(head_path, *st)) {
            if (!*head_path) {
                set_last_error("PFLASH_SCORING_HEAD_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: scoring head load failed, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_qwen35_request_probe(*st);
            free_qwen35_segment_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    // The server's startup gate resolved this path (beside the head when unset) and checked it opens.
    if (reranker_path) {
        if (!*reranker_path) {
            set_last_error("PFLASH_SEG_RERANKER_GGUF is empty");
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: PFLASH_SEG_RERANKER_GGUF is empty, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
        const std::string resolved_reranker_path = reranker_path;
        std::string rr_err;
        // Validate against the exact scoring-head bytes loaded into memory (not a late path reopen).
        const std::string head_sha256 = !st->head_sha256.empty()
            ? st->head_sha256
            : (head_path && *head_path ? cached_file_sha256(head_path) : std::string());
        const int k_dim = st->weights.n_embd_head_k * st->weights.n_head_kv;
        bool rr_ok = load_pflash_seg_reranker(resolved_reranker_path, st->weights.backend,
                                              st->gguf_sha256, head_sha256, st->seg_reranker, rr_err);
        if (rr_ok && st->seg_reranker.in_dim !=
                         (st->seg_reranker.feature_mode == "kbar_katt" ? 2 * k_dim : k_dim)) {
            rr_err = "in_dim " + std::to_string(st->seg_reranker.in_dim) + " does not match the drafter's " +
                     std::to_string(k_dim) + "-wide keys for feature mode " + st->seg_reranker.feature_mode;
            rr_ok = false;
        }
        if (!rr_ok) {
            set_last_error("PFLASH_SEG_RERANKER_GGUF load failed: " + rr_err);
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: segment reranker %s did not load (%s), "
                "refusing to serve without it\n",
                resolved_reranker_path.c_str(), rr_err.c_str());
            std::fflush(stderr);
            free_pflash_seg_reranker(st->seg_reranker);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (st->probe_loaded && st->probe_line_aligned) {
        st->probe_vocab = cached_drafter_tokenizer(gguf_path);
        if (!st->probe_vocab) {
            set_last_error("line-aligned segment probe: drafter vocabulary load failed");
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: the line-aligned probe split needs the "
                "drafter vocabulary, refusing to serve without it\n");
            std::fflush(stderr);
            free_pflash_seg_reranker(st->seg_reranker);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    st->reranker_file_id = st->seg_reranker.loaded_file_identity;
    st->loaded_assets_identity =
        "drafter=" + st->drafter_file_id +
        "|head=" + st->head_file_id +
        "|reranker=" + st->reranker_file_id +
        "|probe=" + st->probe_file_id +
        "|request_probe=" + st->request_probe_file_id;
    out.state = st;
    out.loaded_asset_mode = out.asset_mode;
    out.loaded_assets_identity = st->loaded_assets_identity;
    out.loaded = true;
    std::fprintf(stderr,
        "[drafter] loaded qwen35: n_layer=%d n_head=%d n_head_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d gpu=%d\n",
        st->weights.n_layer, st->weights.n_head, st->weights.n_head_kv,
        st->weights.n_embd, st->weights.n_ff, st->weights.n_embd_head_k,
        st->weights.n_vocab, out.gpu);
    std::fflush(stderr);
    return true;
}

void free_qwen35_drafter_state(DrafterContext & ctx) {
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    for (auto & session : st->sessions) {
        if (session) free_qwen35_scoring_session(*session);
    }
    st->sessions.clear();
    free_pflash_seg_reranker(st->seg_reranker);
    free_qwen35_head(*st);
    free_qwen35_segment_probe(*st);
    free_qwen35_request_probe(*st);
    free_target_weights(st->weights);
    delete st;
    ctx.state = nullptr;
}

} // namespace luce::common
