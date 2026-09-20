// Native GGUF loader for Qwen3.8-Flash-Next (`qwen4exp`).
//
// Ported structure from upstream llama.cpp `src/models/qwen4exp.cpp` and the
// strix-halo lazy PLE reader (`llama-lazy-reader.h`), but this is a hand-rolled
// Luzebox loader: no llama_model, no libllama. Standard split GGUFs (bartowski,
// unsloth, ISTA-DASLab, ...) spread the trunk across N shards named
// "...-00001-of-00003.gguf"; every shard is opened and each wanted tensor is
// resolved from whichever shard holds it. The per_layer_token_embd lookup table
// stays on disk behind a pread pool regardless of which shard owns it.

#include "qwen4exp_internal.h"

#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"

#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

constexpr const char * kArch = "qwen4exp";

uint32_t get_u32_or(const gguf_context * g, const std::string & key,
                    uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return fallback;
        const gguf_type type = gguf_get_arr_type(g, id);
        const void * data = gguf_get_arr_data(g, id);
        if (type == GGUF_TYPE_UINT32) return static_cast<const uint32_t *>(data)[0];
        if (type == GGUF_TYPE_INT32) {
            const int32_t value = static_cast<const int32_t *>(data)[0];
            return value < 0 ? fallback : static_cast<uint32_t>(value);
        }
        return fallback;
    }
    return gguf_get_val_u32(g, id);
}

float get_f32_or(const gguf_context * g, const std::string & key,
                 float fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0 ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) {
            return fallback;
        }
        return static_cast<const float *>(gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

// Read an int-valued array whose element type may be INT32 or UINT32.
std::vector<int32_t> get_i32_array(const gguf_context * g,
                                   const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    const gguf_type type = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    const void * raw = gguf_get_arr_data(g, id);
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        if (type == GGUF_TYPE_UINT32) {
            out[i] = static_cast<int32_t>(static_cast<const uint32_t *>(raw)[i]);
        } else if (type == GGUF_TYPE_INT32) {
            out[i] = static_cast<const int32_t *>(raw)[i];
        } else {
            return {};
        }
    }
    return out;
}

std::vector<float> get_f32_array(const gguf_context * g,
                                 const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    if (gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) return {};
    const size_t n = gguf_get_arr_n(g, id);
    const float * raw = static_cast<const float *>(gguf_get_arr_data(g, id));
    return std::vector<float>(raw, raw + n);
}

// PLE metadata uses uint64 arrays (offsets, vocab sizes, per-ngram multipliers).
std::vector<uint64_t> get_u64_array(const gguf_context * g,
                                    const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return {};
    const gguf_type type = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    if (type != GGUF_TYPE_UINT64 && type != GGUF_TYPE_UINT32 &&
        type != GGUF_TYPE_INT64) {
        return {};
    }
    const void * raw = gguf_get_arr_data(g, id);
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        if (type == GGUF_TYPE_UINT64) {
            out[i] = static_cast<const uint64_t *>(raw)[i];
        } else if (type == GGUF_TYPE_INT64) {
            out[i] = static_cast<uint64_t>(static_cast<const int64_t *>(raw)[i]);
        } else {
            out[i] = static_cast<const uint32_t *>(raw)[i];
        }
    }
    return out;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct TensorAllocation {
    ggml_tensor * tensor = nullptr;
    size_t file_offset = 0;
    size_t file_size = 0;
    size_t buffer_offset = 0;
    size_t shard = 0;
};

// One opened GGUF shard of a (possibly split) model.
struct ShardSource {
    std::string    path;
    gguf_context * gctx = nullptr;
    ggml_context * meta = nullptr;
};

// Discover every sibling shard of a standard split GGUF: a primary named
// "...-00001-of-00003.gguf" yields the full 3-shard list. Anything that is not
// split shard 1 (single-file models included) yields just the primary itself.
std::vector<std::string> discover_shard_paths(const std::string & primary) {
    const std::string ext = ".gguf";
    if (primary.size() <= ext.size() ||
        primary.compare(primary.size() - ext.size(), ext.size(), ext) != 0) {
        return {primary};
    }
    const std::string stem = primary.substr(0, primary.size() - ext.size());

    const size_t of = stem.rfind("-of-");
    if (of == std::string::npos) return {primary};

    const std::string total_str = stem.substr(of + 4);
    if (total_str.empty() || total_str.size() > 9) return {primary};
    for (const char c : total_str) {
        if (c < '0' || c > '9') return {primary};
    }

    size_t idx_start = of;
    while (idx_start > 0 && stem[idx_start - 1] >= '0' && stem[idx_start - 1] <= '9') {
        --idx_start;
    }
    const std::string idx_str = stem.substr(idx_start, of - idx_start);
    if (idx_str.empty() || idx_str.size() > 9) return {primary};

    const int index = std::atoi(idx_str.c_str());
    const int total = std::atoi(total_str.c_str());
    if (index != 1 || total < 1 || total > 99999) return {primary};

    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(total));
    const std::string prefix = stem.substr(0, idx_start);
    for (int i = 1; i <= total; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%0*d", static_cast<int>(idx_str.size()), i);
        paths.push_back(prefix + buf + "-of-" + total_str + ext);
    }
    return paths;
}

}  // namespace

// ─── Qwen4ExpPleReader ──────────────────────────────────────────────────

Qwen4ExpPleReader::~Qwen4ExpPleReader() { close(); }

void Qwen4ExpPleReader::close() {
#if !defined(_WIN32)
    if (fd_ >= 0) {
        ::close(fd_);
    }
#endif
    fd_ = -1;
    to_float_ = nullptr;
}

bool Qwen4ExpPleReader::open(const std::string & path, const char * tensor_name,
                             int n_threads, std::string & out_error) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        out_error = "qwen4exp: shard 2 gguf_init_from_file failed: " + path;
        return false;
    }
    const int64_t tid = gguf_find_tensor(gctx, tensor_name);
    if (tid < 0) {
        out_error = std::string("qwen4exp: shard 2 missing tensor ") + tensor_name;
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    type_ = gguf_get_tensor_type(gctx, tid);
    const size_t data_offset = gguf_get_data_offset(gctx);
    base_ = data_offset + gguf_get_tensor_offset(gctx, tid);

    const ggml_tensor * t = ggml_get_tensor(meta_ctx, tensor_name);
    if (!t) {
        out_error = "qwen4exp: shard 2 tensor descriptor missing";
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    head_dim_ = t->ne[0];
    n_rows_   = t->ne[1];

    const ggml_type_traits * traits = ggml_get_type_traits(type_);
    if (!traits || !traits->to_float) {
        out_error = std::string("qwen4exp: no F32 dequantizer for shard-2 type ") +
                    ggml_type_name(type_);
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    to_float_ = traits->to_float;
    row_size_ = ggml_row_size(type_, head_dim_);

    const size_t expected = static_cast<size_t>(n_rows_) * row_size_;
    const size_t file_size = gguf_get_tensor_size(gctx, tid);
    if (expected != file_size) {
        char message[256];
        std::snprintf(message, sizeof(message),
            "qwen4exp: shard-2 row math mismatch: %lld rows * %zu B = %zu, tensor says %zu",
            static_cast<long long>(n_rows_), row_size_, expected, file_size);
        out_error = message;
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    n_threads_ = std::max(1, n_threads);

#if !defined(_WIN32)
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        out_error = "qwen4exp: open shard 2 failed: " + std::string(std::strerror(errno));
        to_float_ = nullptr;
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
#else
    out_error = "qwen4exp: lazy direct PLE reads are not supported on Windows";
    to_float_ = nullptr;
    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return false;
#endif

    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return true;
}

bool Qwen4ExpPleReader::gather(const int32_t * rows, int64_t n, float * dst) const {
#if defined(_WIN32)
    (void) rows; (void) n; (void) dst;
    return false;
#else
    if (fd_ < 0 || to_float_ == nullptr || n <= 0) return false;

    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || static_cast<int64_t>(rows[i]) >= n_rows_) return false;
        pairs.emplace_back(rows[i], static_cast<int32_t>(i));
    }
    std::sort(pairs.begin(), pairs.end());  // equal/adjacent rows -> sequential file order

    std::vector<uint8_t> scratch(static_cast<size_t>(n) * static_cast<size_t>(row_size_));
    std::vector<std::exception_ptr> errs(static_cast<size_t>(n_threads_));

    auto worker = [&](int w) {
        try {
            const int64_t begin = static_cast<int64_t>(n) * w / n_threads_;
            const int64_t end   = static_cast<int64_t>(n) * (w + 1) / n_threads_;
            uint8_t * buf = scratch.data() + static_cast<size_t>(begin) *
                            static_cast<size_t>(row_size_);
            for (int64_t k = begin; k < end; ++k, buf += row_size_) {
                const off_t at = static_cast<off_t>(base_) +
                                 static_cast<off_t>(pairs[static_cast<size_t>(k)].first) *
                                 static_cast<off_t>(row_size_);
                size_t got = 0;
                while (got < row_size_) {
                    const ssize_t r = ::pread(fd_, buf + got, row_size_ - got,
                                              at + static_cast<off_t>(got));
                    if (r <= 0) throw std::runtime_error("qwen4exp: shard-2 pread failed");
                    got += static_cast<size_t>(r);
                }
                to_float_(reinterpret_cast<const void *>(buf),
                          dst + static_cast<size_t>(pairs[static_cast<size_t>(k)].second) *
                                static_cast<size_t>(head_dim_),
                          head_dim_);
            }
        } catch (...) {
            errs[static_cast<size_t>(w)] = std::current_exception();
        }
    };

    if (n_threads_ == 1 || n < 64) {
        worker(0);
        if (errs[0]) return false;
        return true;
    }

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n_threads_ - 1));
    for (int w = 1; w < n_threads_; ++w) pool.emplace_back(worker, w);
    worker(0);
    for (auto & th : pool) th.join();

    for (const std::exception_ptr & e : errs) {
        if (e) return false;
    }
    return true;
#endif
}

// ─── Loader ─────────────────────────────────────────────────────────────

bool load_qwen4exp_gguf(const std::string & path, ggml_backend_t backend,
                        Qwen4ExpWeights & out) {
    // Open every shard of the model; a single-file GGUF is a one-element list.
    std::vector<ShardSource> shards;
    for (const std::string & shard_path : discover_shard_paths(path)) {
        ShardSource shard;
        shard.path = shard_path;
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &shard.meta;
        shard.gctx = gguf_init_from_file(shard_path.c_str(), params);
        if (!shard.gctx) {
            set_last_error("qwen4exp: gguf_init_from_file failed: " + shard_path);
            for (ShardSource & opened : shards) {
                gguf_free(opened.gctx);
                if (opened.meta) ggml_free(opened.meta);
            }
            return false;
        }
        shards.push_back(std::move(shard));
    }
    gguf_context * gctx = shards.front().gctx;
    ggml_context * meta_ctx = shards.front().meta;

    auto fail = [&](const std::string & message) {
        set_last_error("qwen4exp: " + message);
        for (ShardSource & shard : shards) {
            gguf_free(shard.gctx);
            if (shard.meta) {
                ggml_free(shard.meta);
                if (out.ctx == shard.meta) out.ctx = nullptr;
            }
        }
        out.extra_meta_ctxs.clear();
        return false;
    };

    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    const char * arch = arch_id >= 0 ? gguf_get_val_str(gctx, arch_id) : nullptr;
    if (!arch || std::strcmp(arch, kArch) != 0) {
        return fail(std::string("unexpected architecture '") + (arch ? arch : "") + "'");
    }

    const std::string P = std::string(kArch) + ".";
    const uint32_t n_layer  = get_u32_or(gctx, P + "block_count", 0);
    const uint32_t n_embd   = get_u32_or(gctx, P + "embedding_length", 0);
    const uint32_t n_head   = get_u32_or(gctx, P + "attention.head_count", 0);
    const uint32_t n_head_kv = get_u32_or(gctx, P + "attention.head_count_kv", 0);
    const uint32_t head_k   = get_u32_or(gctx, P + "attention.key_length", 0);
    const uint32_t head_v   = get_u32_or(gctx, P + "attention.value_length", 0);
    const uint32_t fai      = get_u32_or(gctx, P + "full_attention_interval", 0);
    const uint32_t n_ff_exp = get_u32_or(gctx, P + "expert_feed_forward_length", 0);
    const uint32_t n_ff_sh  = get_u32_or(gctx, P + "expert_shared_feed_forward_length", 0);
    const uint32_t n_expert = get_u32_or(gctx, P + "expert_count", 0);
    const uint32_t n_used   = get_u32_or(gctx, P + "expert_used_count", 0);
    const uint32_t n_hc     = get_u32_or(gctx, P + "hyper_connection.count", 0);
    const uint32_t hc_lr    = get_u32_or(gctx, P + "hyper_connection.low_rank", 0);
    const uint32_t ssm_conv = get_u32_or(gctx, P + "ssm.conv_kernel", 0);
    const uint32_t ssm_inner = get_u32_or(gctx, P + "ssm.inner_size", 0);
    const uint32_t ssm_state = get_u32_or(gctx, P + "ssm.state_size", 0);
    const uint32_t ssm_dt   = get_u32_or(gctx, P + "ssm.time_step_rank", 0);
    const uint32_t ssm_grp  = get_u32_or(gctx, P + "ssm.group_count", 0);
    const uint32_t idx_head = get_u32_or(gctx, P + "attention.indexer.head_count", 0);
    const uint32_t idx_dim  = get_u32_or(gctx, P + "attention.indexer.key_length", 0);
    const uint32_t idx_topk = get_u32_or(gctx, P + "attention.indexer.top_k", 0);
    const uint32_t ple_hdim = get_u32_or(gctx, P + "embedding_length_per_layer_input", 0);
    const uint32_t ple_ng   = get_u32_or(gctx, P + "ple.ngram_size", 0);
    const uint32_t ple_hpn  = get_u32_or(gctx, P + "ple.heads_per_ngram", 0);
    const uint32_t ple_conv = get_u32_or(gctx, P + "ple.conv_kernel", 0);

    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0 ||
        head_k != 256 || head_v != 256 || fai == 0 || n_ff_exp == 0 ||
        n_expert == 0 || n_used == 0 || n_used > n_expert || n_hc <= 1 ||
        hc_lr == 0 || ssm_conv < 2 || ssm_inner == 0 || ssm_state == 0 ||
        ssm_dt == 0 || ssm_grp == 0 || idx_head == 0 || idx_dim == 0 ||
        idx_topk == 0 || ple_hdim == 0 || ple_ng < 2 || ple_hpn == 0 ||
        ple_conv < 2 || n_layer % fai != 0) {
        char message[512];
        std::snprintf(message, sizeof(message),
            "invalid hparams: layers=%u embd=%u heads=%u/%u head=%ux%u fai=%u "
            "ff{exp=%u sh=%u} experts=%u/%u hc=%u/%u ssm{%u,%u,%u,%u,%u} "
            "indexer{%u,%u,%u} ple{%u,ng=%u,hpn=%u,conv=%u}",
            n_layer, n_embd, n_head, n_head_kv, head_k, head_v, fai,
            n_ff_exp, n_ff_sh, n_expert, n_used, n_hc, hc_lr,
            ssm_conv, ssm_inner, ssm_state, ssm_dt, ssm_grp,
            idx_head, idx_dim, idx_topk, ple_hdim, ple_ng, ple_hpn, ple_conv);
        return fail(message);
    }

    out.n_layer = static_cast<int>(n_layer);
    out.n_embd = static_cast<int>(n_embd);
    out.n_head = static_cast<int>(n_head);
    out.n_head_kv = static_cast<int>(n_head_kv);
    out.n_embd_head_k = static_cast<int>(head_k);
    out.n_embd_head_v = static_cast<int>(head_v);
    out.full_attention_interval = static_cast<int>(fai);
    out.n_ff_exp = static_cast<int>(n_ff_exp);
    out.n_ff_shexp = static_cast<int>(n_ff_sh);
    out.n_expert = static_cast<int>(n_expert);
    out.n_expert_used = static_cast<int>(n_used);
    out.n_hc = static_cast<int>(n_hc);
    out.hc_lowrank = static_cast<int>(hc_lr);
    out.ssm_d_conv = static_cast<int>(ssm_conv);
    out.ssm_d_inner = static_cast<int>(ssm_inner);
    out.ssm_d_state = static_cast<int>(ssm_state);
    out.ssm_dt_rank = static_cast<int>(ssm_dt);
    out.ssm_n_group = static_cast<int>(ssm_grp);
    out.linear_value_heads = static_cast<int>(ssm_inner / ssm_state);
    out.linear_key_heads = static_cast<int>(ssm_grp);
    out.indexer_n_head = static_cast<int>(idx_head);
    out.indexer_head_size = static_cast<int>(idx_dim);
    out.indexer_top_k = static_cast<int>(idx_topk);
    out.ple_ngram_size = static_cast<int>(ple_ng);
    out.ple_heads_per_ngram = static_cast<int>(ple_hpn);
    out.ple_conv_kernel = static_cast<int>(ple_conv);
    out.ple_head_dim = static_cast<int>(ple_hdim);
    out.ple_n_heads = static_cast<int>((ple_ng - 1) * ple_hpn);
    out.rope_dimension_count = static_cast<int>(get_u32_or(gctx, P + "rope.dimension_count", 64));
    out.rope_theta = get_f32_or(gctx, P + "rope.freq_base", 1e7f);
    out.rms_eps = get_f32_or(gctx, P + "attention.layer_norm_rms_epsilon", 1e-6f);

    const std::vector<int32_t> sections = get_i32_array(gctx, P + "rope.dimension_sections");
    if (sections.size() >= 4) {
        out.rope_sections[0] = sections[0];
        out.rope_sections[1] = sections[1];
        out.rope_sections[2] = sections[2];
        out.rope_sections[3] = sections[3];
    }
    out.compress_ratios = get_i32_array(gctx, P + "attention.compress_ratios");
    if (out.compress_ratios.size() < n_layer) {
        return fail("missing or short attention.compress_ratios");
    }
    out.ple_layer_ids = get_i32_array(gctx, P + "ple.layers");
    {
        const std::vector<uint64_t> offs = get_u64_array(gctx, P + "ple.head_offsets");
        const std::vector<uint64_t> vsz  = get_u64_array(gctx, P + "ple.head_vocab_sizes");
        out.ple_head_offsets.assign(offs.begin(), offs.end());
        out.ple_head_vocab_sizes.assign(vsz.begin(), vsz.end());
    }
    out.ple_layer_multipliers = get_u64_array(gctx, P + "ple.layer_multipliers");
    out.ple_eos_token_id = static_cast<int32_t>(get_u32_or(gctx, P + "ple.eos_token_id", 0xFFFFFFFFu));
    out.ple_image_token_id = static_cast<int32_t>(get_u32_or(gctx, P + "ple.image_token_id", 0xFFFFFFFFu));
    if (out.ple_head_offsets.size() != static_cast<size_t>(out.ple_n_heads) ||
        out.ple_head_vocab_sizes.size() != static_cast<size_t>(out.ple_n_heads)) {
        return fail("ple head offset/vocab arrays do not match (ngram_size-1)*heads_per_ngram");
    }

    const uint32_t missing_token = 0xFFFFFFFFu;
    const uint32_t eos = get_u32_or(gctx, "tokenizer.ggml.eos_token_id", missing_token);
    const uint32_t eot = get_u32_or(gctx, "tokenizer.ggml.eot_token_id", missing_token);
    out.eos_id = eos == missing_token ? -1 : static_cast<int32_t>(eos);
    out.eos_chat_id = eot == missing_token ? -1 : static_cast<int32_t>(eot);

    out.layers.assign(n_layer, Qwen4ExpLayer{});

    // Resolve a tensor by name across every shard's descriptor context.
    auto tensor = [&](const char * name) {
        for (const ShardSource & shard : shards) {
            ggml_tensor * value = ggml_get_tensor(shard.meta, name);
            if (value) return value;
        }
        return static_cast<ggml_tensor *>(nullptr);
    };
    auto layer_tensor = [&](uint32_t il, const char * suffix) {
        char name[160];
        std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
        return tensor(name);
    };
    auto is_ple_layer = [&](uint32_t il) {
        return std::find(out.ple_layer_ids.begin(), out.ple_layer_ids.end(),
                         static_cast<int32_t>(il)) != out.ple_layer_ids.end();
    };

    out.tok_embd = tensor("token_embd.weight");
    out.out_norm = tensor("output_norm.weight");   // qwen4exp folds this into output_hc_*
    out.output   = tensor("output.weight");
    out.output_hc_norm = tensor("output_hc_norm.weight");
    out.output_hc_down = tensor("output_hc_down.weight");
    out.output_hc_up   = tensor("output_hc_up.weight");
    if (!out.tok_embd || !out.output) {
        return fail("missing token_embd/output tensor");
    }
    if (!out.output_hc_norm || !out.output_hc_down || !out.output_hc_up) {
        return fail("missing output_hc_norm/down/up tensor");
    }
    out.n_vocab = static_cast<int>(out.tok_embd->ne[1]);

    for (uint32_t il = 0; il < n_layer; ++il) {
        Qwen4ExpLayer & layer = out.layers[il];
        layer.is_full_attention = out.compress_ratios[il] > 0;
        layer.is_ple = is_ple_layer(il);

        layer.attn_norm = layer_tensor(il, "attn_norm.weight");       // absent in qwen4exp
        layer.attn_post_norm = layer_tensor(il, "attn_post_norm.weight");
        layer.ffn_norm = layer_tensor(il, "ffn_norm.weight");
        layer.hc_attn_norm = layer_tensor(il, "hc_attn_norm.weight");
        layer.hc_attn_down = layer_tensor(il, "hc_attn_down.weight");
        layer.hc_attn_up = layer_tensor(il, "hc_attn_up.weight");
        layer.hc_attn_inject = layer_tensor(il, "hc_attn_inject.weight");
        layer.hc_ffn_norm = layer_tensor(il, "hc_ffn_norm.weight");
        layer.hc_ffn_down = layer_tensor(il, "hc_ffn_down.weight");
        layer.hc_ffn_up = layer_tensor(il, "hc_ffn_up.weight");
        layer.hc_ffn_inject = layer_tensor(il, "hc_ffn_inject.weight");
        if (!layer.hc_attn_norm || !layer.hc_attn_down ||
            !layer.hc_attn_up || !layer.hc_attn_inject || !layer.hc_ffn_norm ||
            !layer.hc_ffn_down || !layer.hc_ffn_up || !layer.hc_ffn_inject) {
            return fail("layer " + std::to_string(il) + " missing HC tensor");
        }

        if (layer.is_full_attention) {
            layer.wq = layer_tensor(il, "attn_q.weight");
            layer.wk = layer_tensor(il, "attn_k.weight");
            layer.wv = layer_tensor(il, "attn_v.weight");
            layer.wo = layer_tensor(il, "attn_output.weight");
            layer.q_norm = layer_tensor(il, "attn_q_norm.weight");
            layer.k_norm = layer_tensor(il, "attn_k_norm.weight");
            layer.indexer_q_proj = layer_tensor(il, "indexer.q_proj.weight");
            layer.indexer_k_proj = layer_tensor(il, "indexer.k_proj.weight");
            layer.indexer_q_norm = layer_tensor(il, "indexer.q_norm.weight");
            layer.indexer_k_norm = layer_tensor(il, "indexer.k_norm.weight");
            if (!layer.wq || !layer.wk || !layer.wv || !layer.wo ||
                !layer.q_norm || !layer.k_norm || !layer.indexer_q_proj ||
                !layer.indexer_k_proj || !layer.indexer_q_norm ||
                !layer.indexer_k_norm) {
                return fail("layer " + std::to_string(il) + " missing full-attention/indexer tensor");
            }
        } else {
            layer.attn_qkv = layer_tensor(il, "attn_qkv.weight");
            layer.attn_gate = layer_tensor(il, "attn_gate.weight");
            layer.ssm_conv1d = layer_tensor(il, "ssm_conv1d.weight");
            layer.ssm_alpha = layer_tensor(il, "ssm_alpha.weight");
            layer.ssm_beta = layer_tensor(il, "ssm_beta.weight");
            layer.ssm_a = layer_tensor(il, "ssm_a");
            layer.ssm_dt_bias = layer_tensor(il, "ssm_dt.bias");
            layer.ssm_norm = layer_tensor(il, "ssm_norm.weight");
            layer.ssm_out = layer_tensor(il, "ssm_out.weight");
            if (!layer.attn_qkv || !layer.attn_gate || !layer.ssm_conv1d ||
                !layer.ssm_alpha || !layer.ssm_beta || !layer.ssm_a ||
                !layer.ssm_dt_bias || !layer.ssm_norm || !layer.ssm_out) {
                return fail("layer " + std::to_string(il) + " missing linear-attention tensor");
            }
        }

        if (layer.is_ple) {
            layer.ple_conv1d = layer_tensor(il, "ple_conv1d.weight");
            layer.ple_key = layer_tensor(il, "ple_key.weight");
            layer.ple_value = layer_tensor(il, "ple_value.weight");
            layer.ple_norm_conv = layer_tensor(il, "ple_norm_conv.weight");
            layer.ple_norm_key = layer_tensor(il, "ple_norm_key.weight");
            layer.ple_norm_query = layer_tensor(il, "ple_norm_query.weight");
            if (!layer.ple_conv1d || !layer.ple_key || !layer.ple_value ||
                !layer.ple_norm_conv || !layer.ple_norm_key ||
                !layer.ple_norm_query) {
                return fail("layer " + std::to_string(il) + " missing PLE tensor");
            }
        }

        layer.ffn_gate_inp = layer_tensor(il, "ffn_gate_inp.weight");
        layer.ffn_exp_probs_b = layer_tensor(il, "exp_probs_b.bias");
        layer.ffn_gate_exps = layer_tensor(il, "ffn_gate_exps.weight");
        layer.ffn_up_exps = layer_tensor(il, "ffn_up_exps.weight");
        layer.ffn_down_exps = layer_tensor(il, "ffn_down_exps.weight");
        layer.ffn_gate_inp_shexp = layer_tensor(il, "ffn_gate_inp_shexp.weight");
        layer.ffn_gate_shexp = layer_tensor(il, "ffn_gate_shexp.weight");
        layer.ffn_up_shexp = layer_tensor(il, "ffn_up_shexp.weight");
        layer.ffn_down_shexp = layer_tensor(il, "ffn_down_shexp.weight");
        if (!layer.ffn_gate_inp || !layer.ffn_gate_exps || !layer.ffn_up_exps ||
            !layer.ffn_down_exps || !layer.ffn_gate_inp_shexp ||
            !layer.ffn_gate_shexp || !layer.ffn_up_shexp || !layer.ffn_down_shexp) {
            return fail("layer " + std::to_string(il) + " missing MoE tensor");
        }
    }

    // Allocate exactly the referenced trunk tensors (MTP, if any, and the
    // shard-2 PLE table are never uploaded).
    std::unordered_set<ggml_tensor *> wanted;
    auto add = [&](ggml_tensor * value) { if (value) wanted.insert(value); };
    add(out.out_norm);
    add(out.output);
    add(out.output_hc_norm);
    add(out.output_hc_down);
    add(out.output_hc_up);
    for (Qwen4ExpLayer & layer : out.layers) {
        add(layer.attn_norm); add(layer.attn_post_norm); add(layer.ffn_norm);
        add(layer.hc_attn_norm); add(layer.hc_attn_down); add(layer.hc_attn_up);
        add(layer.hc_attn_inject); add(layer.hc_ffn_norm); add(layer.hc_ffn_down);
        add(layer.hc_ffn_up); add(layer.hc_ffn_inject);
        add(layer.attn_qkv); add(layer.attn_gate);
        add(layer.ssm_conv1d); add(layer.ssm_alpha); add(layer.ssm_beta);
        add(layer.ssm_a); add(layer.ssm_dt_bias); add(layer.ssm_norm); add(layer.ssm_out);
        add(layer.wq); add(layer.wk); add(layer.wv); add(layer.wo);
        add(layer.q_norm); add(layer.k_norm);
        add(layer.indexer_q_proj); add(layer.indexer_k_proj);
        add(layer.indexer_q_norm); add(layer.indexer_k_norm);
        add(layer.ple_conv1d); add(layer.ple_key); add(layer.ple_value);
        add(layer.ple_norm_conv); add(layer.ple_norm_key); add(layer.ple_norm_query);
        add(layer.ffn_gate_inp); add(layer.ffn_exp_probs_b);
        add(layer.ffn_gate_exps); add(layer.ffn_up_exps); add(layer.ffn_down_exps);
        add(layer.ffn_gate_inp_shexp); add(layer.ffn_gate_shexp);
        add(layer.ffn_up_shexp); add(layer.ffn_down_shexp);
    }

    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
    std::vector<TensorAllocation> allocations;
    allocations.reserve(wanted.size());
    size_t allocation_size = 0;
    for (size_t s = 0; s < shards.size(); ++s) {
        const int64_t n_tensors = gguf_get_n_tensors(shards[s].gctx);
        for (int64_t tid = 0; tid < n_tensors; ++tid) {
            const char * name = gguf_get_tensor_name(shards[s].gctx, tid);
            ggml_tensor * value = ggml_get_tensor(shards[s].meta, name);
            if (!value || wanted.find(value) == wanted.end()) continue;
            allocation_size = align_up(allocation_size, alignment);
            TensorAllocation allocation;
            allocation.tensor = value;
            allocation.file_offset = gguf_get_data_offset(shards[s].gctx) +
                                     gguf_get_tensor_offset(shards[s].gctx, tid);
            allocation.file_size = gguf_get_tensor_size(shards[s].gctx, tid);
            allocation.buffer_offset = allocation_size;
            allocation.shard = s;
            allocation_size += ggml_backend_buft_get_alloc_size(buffer_type, value);
            allocations.push_back(allocation);
        }
    }
    if (allocations.size() != wanted.size()) {
        return fail("failed to resolve every trunk tensor in the GGUF table");
    }

    out.ctx = meta_ctx;
    for (size_t s = 1; s < shards.size(); ++s) {
        out.extra_meta_ctxs.push_back(shards[s].meta);
    }
    out.backend = backend;
    out.buf = ggml_backend_alloc_buffer(backend, allocation_size);
    if (!out.buf) return fail("weight buffer allocation failed");
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(out.buf));
    for (const TensorAllocation & allocation : allocations) {
        if (ggml_backend_tensor_alloc(out.buf, allocation.tensor,
                base + allocation.buffer_offset) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            return fail("weight tensor allocation failed");
        }
    }

    // Every shard is mapped; each tensor reads its bytes from the shard that
    // owns it, at that shard's data offset.
    std::vector<GgufMmap> mmaps(shards.size());
    for (size_t s = 0; s < shards.size(); ++s) {
        std::string mmap_error;
        if (!mmaps[s].open(shards[s].path, mmap_error)) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            return fail(mmap_error);
        }
    }
    for (const TensorAllocation & allocation : allocations) {
        const GgufMmap & mmap = mmaps[allocation.shard];
        if (allocation.file_offset + allocation.file_size < allocation.file_offset ||
            allocation.file_offset + allocation.file_size > mmap.size()) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            return fail("truncated tensor data for " + std::string(allocation.tensor->name));
        }
        ggml_backend_tensor_set(allocation.tensor,
            static_cast<const uint8_t *>(mmap.data()) + allocation.file_offset,
            0, allocation.file_size);
    }

    size_t token_shard = 0;
    int64_t token_tid = -1;
    for (size_t s = 0; s < shards.size(); ++s) {
        token_tid = gguf_find_tensor(shards[s].gctx, "token_embd.weight");
        if (token_tid >= 0) {
            token_shard = s;
            break;
        }
    }
    if (token_tid < 0) {
        ggml_backend_buffer_free(out.buf);
        out.buf = nullptr;
        return fail("token_embd.weight missing from tensor table");
    }
    const size_t token_relative_offset =
        gguf_get_tensor_offset(shards[token_shard].gctx, token_tid);
    const size_t token_size = gguf_get_tensor_size(shards[token_shard].gctx, token_tid);
    const size_t data_offset = gguf_get_data_offset(shards[token_shard].gctx);
    const uint8_t * token_bytes =
        static_cast<const uint8_t *>(mmaps[token_shard].data());
    if (!gguf_tensor_in_file(data_offset, token_relative_offset, token_size,
                             mmaps[token_shard].size())) {
        ggml_backend_buffer_free(out.buf);
        out.buf = nullptr;
        return fail("truncated token_embd.weight");
    }
    out.embedder.tok_embd_owned.resize(token_size);
    std::memcpy(out.embedder.tok_embd_owned.data(),
        token_bytes + data_offset + token_relative_offset, token_size);
    out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
    out.embedder.tok_embd_type = gguf_get_tensor_type(shards[token_shard].gctx, token_tid);
    out.embedder.n_embd = out.n_embd;
    out.embedder.n_vocab = out.n_vocab;
    out.embedder.row_bytes = token_size / static_cast<size_t>(out.n_vocab);

    // The PLE lookup table is served lazily from whichever shard holds it
    // (ISTA-DASLab isolates it in a separate shard; bartowski-style splits keep
    // it in shard 1; single-file models trivially have it in `path`).
    if (!out.ple_layer_ids.empty()) {
        const char * const kPleTensor = "per_layer_token_embd.weight";
        std::string ple_path;
        for (const ShardSource & shard : shards) {
            if (gguf_find_tensor(shard.gctx, kPleTensor) >= 0) {
                ple_path = shard.path;
                break;
            }
        }
        if (ple_path.empty()) {
            ple_path = shards.front().path;
        }
        std::string reader_error;
        if (!out.ple_reader.open(ple_path, kPleTensor, 4, reader_error)) {
            return fail(reader_error);
        }
        out.shard2_path = ple_path;
    }

    for (ShardSource & shard : shards) {
        gguf_free(shard.gctx);
        shard.gctx = nullptr;
    }
    gctx = nullptr;
    meta_ctx = nullptr;  // owned by out.ctx / out.extra_meta_ctxs from here on

    char summary[512];
    std::snprintf(summary, sizeof(summary),
        "qwen4exp trunk loaded: %d layers (%d linear + %d full), %zu tensors %.2f GiB "
        "from %zu shard(s), experts=%d/%d hc=%d/%d ple_layers=%zu ngram=%d "
        "heads_per_ngram=%d table_rows=%lld eos=%d",
        out.n_layer,
        out.n_layer - out.n_layer / out.full_attention_interval,
        out.n_layer / out.full_attention_interval,
        allocations.size(), allocation_size / (1024.0 * 1024.0 * 1024.0),
        shards.size(),
        out.n_expert_used, out.n_expert, out.n_hc, out.hc_lowrank,
        out.ple_layer_ids.size(), out.ple_ngram_size, out.ple_heads_per_ngram,
        static_cast<long long>(out.ple_reader.n_rows()), out.eos_id);
    set_last_error(summary);
    std::fprintf(stderr, "[qwen4exp] %s\n", summary);
    return true;
}

void free_qwen4exp_weights(Qwen4ExpWeights & w) {
    w.ple_reader.close();
    if (w.buf) {
        ggml_backend_buffer_free(w.buf);
        w.buf = nullptr;
    }
    for (ggml_context * extra : w.extra_meta_ctxs) {
        if (extra) ggml_free(extra);
    }
    w.extra_meta_ctxs.clear();
    if (w.ctx) {
        ggml_free(w.ctx);
        w.ctx = nullptr;
    }
    w.backend = nullptr;
}

}  // namespace dflash::common
