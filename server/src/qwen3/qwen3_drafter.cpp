// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.
//
// Wires three pieces:
//   - qwen3_loader.cpp : mmap GGUF + populate ggml tensors on backend
//   - qwen3_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)
//   - chunk-top-K + span merge (this file)
//
// Single-pass forward at full S using a custom Qwen3-0.6B graph with the
// FlashPrefill block-sparse attention kernel (or BSA when enabled). Tail
// attention scoring runs in a separate post-forward graph using saved Q_last
// and K_curr per layer.
//
// Result running_max [n_lookahead, S] f32 is reduced to per-token scores via
// mean-over-lookahead, smoothed with AvgPool, scored per chunk, top-K kept.

#include "qwen3_drafter.h"
#include "qwen3_drafter_model.h"
#include "pflash_selection.h"
#include "qwen3/anchor_params.h"
#include "common/backend_precision.h"
#include "common/gguf_inspect.h"
#include "internal.h"
#include "anchor_scan.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

namespace {

static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up_i(int x, int a) { return ((x + a - 1) / a) * a; }

static void build_causal_mask_f16(std::vector<uint16_t> & out, int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up_i(kv_len, 32);
    const int q_pad = align_up_i(n_tokens, 32);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    static_assert(F16_ZERO == 0, "visible mask entries are zero-filled with memset");
    for (int q = 0; q < n_tokens; ++q) {
        const int visible = std::min(kv_len, kv_start + q + 1);
        if (visible > 0) {
            std::memset(out.data() + (size_t)q * kv_pad, 0, (size_t)visible * sizeof(uint16_t));
        }
    }
}

// Qwen3.5-0.8B LongAttnComp scorer. Features are the residual entering
// full-attention block 15 after the first 15 blocks (twelve GatedDeltaNet and
// three full-attention blocks). Block 15's own Q/K projections score the
// context without RoPE, exactly like the Qwen3-0.6B block-13 head; an
// optional trained head replaces those two projections.
static constexpr int kQwen35HeadBlock = 15;
static constexpr const char * kQwen35HeadSchema = "qwen3_5_0_8b_nope_qk_mass_v1";
static constexpr const char * kQwen35HeadBaseModel = "Qwen/Qwen3.5-0.8B";
static constexpr const char * kQwen35HeadFeatureTap =
    "post_block14_residual_before_block15";

// create_target_cache honours DFLASH27B_KV_TQ3; the drafter cache never wants
// the TurboQuant rotation, so force it off while the cache is created.
struct ScopedKvTq3Off {
    ScopedKvTq3Off() {
#if defined(_WIN32)
        char * raw = nullptr;
        size_t len = 0;
        _dupenv_s(&raw, &len, "DFLASH27B_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        free(raw);
        _putenv_s("DFLASH27B_KV_TQ3", "0");
#else
        const char * raw = std::getenv("DFLASH27B_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        setenv("DFLASH27B_KV_TQ3", "0", 1);
#endif
    }
    ~ScopedKvTq3Off() {
#if defined(_WIN32)
        // _putenv_s with empty value removes the variable on MSVCRT.
        _putenv_s("DFLASH27B_KV_TQ3", had_ ? old_.c_str() : "");
#else
        if (had_) setenv("DFLASH27B_KV_TQ3", old_.c_str(), 1);
        else unsetenv("DFLASH27B_KV_TQ3");
#endif
    }
    bool had_ = false;
    std::string old_;
};

struct Qwen35DrafterState {
    TargetWeights weights;
    std::string gguf_sha256;
    ggml_context *        head_ctx = nullptr;
    ggml_backend_buffer_t head_buf = nullptr;
    ggml_tensor *         head_wq  = nullptr;  // [hidden, n_head * head_dim], query rows only
    ggml_tensor *         head_wk  = nullptr;  // [hidden, n_head_kv * head_dim]
    bool                  head_loaded = false;
};

static void free_qwen35_head(Qwen35DrafterState & st) {
    if (st.head_buf) { ggml_backend_buffer_free(st.head_buf); st.head_buf = nullptr; }
    if (st.head_ctx) { ggml_free(st.head_ctx); st.head_ctx = nullptr; }
    st.head_wq = st.head_wk = nullptr;
    st.head_loaded = false;
}

static bool qwen35_metadata_equals(gguf_context * g, const char * key,
                                   const std::string & expected) {
    const int id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
           expected == gguf_get_val_str(g, id);
}

static bool qwen35_head_block_available(const TargetWeights & w, std::string & error) {
    if (w.n_layer <= kQwen35HeadBlock || (size_t)kQwen35HeadBlock >= w.layers.size()) {
        error = "qwen35 LongAttnComp scorer needs at least 16 blocks";
        return false;
    }
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];
    if (((kQwen35HeadBlock + 1) % w.full_attention_interval) != 0 ||
        !L.wq || !L.wk || !L.attn_norm || !L.q_norm || !L.k_norm) {
        error = "qwen35 LongAttnComp scorer block 15 is not a full-attention block";
        return false;
    }
    return true;
}

// Optional trained head for the block-15 tap. Fails closed on any contract
// mismatch, mirroring the Qwen3-0.6B head loader.
static bool load_qwen35_longattncomp_head(const std::string & path,
                                          Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return false;
    }
    if (st.gguf_sha256.empty()) {
        set_last_error("LongAttnComp head requires the drafter GGUF identity hash");
        return false;
    }
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        set_last_error("LongAttnComp head GGUF could not be opened: " + path);
        return false;
    }
    auto fail = [&](const std::string & message) {
        free_qwen35_head(st);
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "longattncomp") ||
        !qwen35_metadata_equals(g, "longattncomp.schema", kQwen35HeadSchema) ||
        !qwen35_metadata_equals(g, "longattncomp.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "longattncomp.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "longattncomp.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("LongAttnComp head metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    struct Contract {
        const char * name;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"longattncomp.attn_q.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head * w.n_embd_head_k, &st.head_wq},
        {"longattncomp.attn_k.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head_kv * w.n_embd_head_k, &st.head_wk},
    };
    ggml_init_params head_params{};
    head_params.mem_size = 4 * ggml_tensor_overhead();
    head_params.no_alloc = true;
    st.head_ctx = ggml_init(head_params);
    if (!st.head_ctx) return fail("LongAttnComp head context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = data_ctx ? ggml_get_tensor(data_ctx, contract.name) : nullptr;
        if (!source || source->type != GGML_TYPE_F32 || ggml_n_dims(source) != 2 ||
            source->ne[0] != contract.ne0 || source->ne[1] != contract.ne1) {
            return fail(std::string("LongAttnComp head tensor contract mismatch: ") +
                        contract.name);
        }
        *contract.destination =
            ggml_new_tensor_2d(st.head_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    st.head_buf = ggml_backend_alloc_ctx_tensors(st.head_ctx, w.backend);
    if (!st.head_buf) return fail("LongAttnComp head buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.head_loaded = true;
    std::fprintf(stderr, "[qwen35-drafter] loaded LongAttnComp head: %s\n", path.c_str());
    std::fflush(stderr);
    return true;
}

static int env_int(const char * name, int fallback) {
    if (const char * v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x >= 0) return x;
    }
    return fallback;
}

static float env_float(const char * name, float def) {
    if (const char * v = std::getenv(name)) {
        try { return std::stof(v); } catch (...) {}
    }
    return def;
}

static void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                                     int chunk, int radius) {
    int lo = std::max(0, chunk - radius);
    int hi = std::min(n_chunks - 1, chunk + radius);
    for (int c = lo; c <= hi; ++c) forced[(size_t)c] = 1;
}

struct PFlashTraceFields {
    const std::vector<int32_t> * input_ids = nullptr;
    int query_begin = -1;
    int query_end = -1;
    dflash::qwen3::PFlashSelectionMode selector_mode =
        dflash::qwen3::PFlashSelectionMode::Legacy;
    dflash::qwen3::PFlashQueryParser query_parser =
        dflash::qwen3::PFlashQueryParser::SemanticUser;
    int token_budget = 0;
    dflash::qwen3::PFlashSelectionStop stop =
        dflash::qwen3::PFlashSelectionStop::InvalidInput;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    const std::vector<double> * exact_chunk_scores = nullptr;
    const std::vector<PFlashTokenSpan> * required_instruction_spans = nullptr;
};

static void write_compression_trace(
        int input_tokens,
        float keep_ratio,
        int chunk_size,
        int n_lookahead,
        int pool_kernel,
        int n_keep,
        const std::vector<std::pair<float, int>> & chunk_means,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        const std::vector<int32_t> & compressed_ids,
        const PFlashTraceFields * trace_fields = nullptr) {
    const char * path = std::getenv("DFLASH_PFLASH_TRACE_PATH");
    if (!path || !*path) return;

    FILE * file = std::fopen(path, "a");
    if (!file) {
        std::fprintf(stderr, "[pflash-trace] cannot append %s\n", path);
        return;
    }

    std::vector<float> scores(selected.size(), 0.0f);
    for (const auto & chunk : chunk_means) {
        scores[(size_t)chunk.second] = chunk.first;
    }
    const bool has_exact_scores = trace_fields &&
        trace_fields->exact_chunk_scores &&
        trace_fields->exact_chunk_scores->size() == scores.size();
    if (trace_fields &&
        trace_fields->selector_mode !=
            dflash::qwen3::PFlashSelectionMode::Legacy &&
        !has_exact_scores) {
        std::fclose(file);
        std::fprintf(stderr, "[pflash-trace] exact strict scores unavailable\n");
        return;
    }

    std::fprintf(file,
        "{\"schema_version\":%d,\"input_tokens\":%d,\"keep_ratio\":%.9g",
        trace_fields ? 3 : 1, input_tokens, keep_ratio);
    if (trace_fields) {
        std::fputs(",\"input_ids\":[", file);
        for (size_t index = 0; index < trace_fields->input_ids->size(); ++index) {
            if (index) std::fputc(',', file);
            std::fprintf(file, "%d", (*trace_fields->input_ids)[index]);
        }
        std::fprintf(file,
            "],\"query_begin\":%d,\"query_end\":%d,"
            "\"selector_mode\":\"%s\",\"query_parser\":\"%s\","
            "\"token_budget\":%d,"
            "\"retained_tokens\":%d",
            trace_fields->query_begin, trace_fields->query_end,
            dflash::qwen3::pflash_selection_mode_name(
                trace_fields->selector_mode),
            dflash::qwen3::pflash_query_parser_name(trace_fields->query_parser),
            trace_fields->token_budget, trace_fields->retained_tokens);
        std::fputs(",\"required_instruction_spans\":[", file);
        if (trace_fields->required_instruction_spans) {
            for (size_t index = 0;
                 index < trace_fields->required_instruction_spans->size();
                 ++index) {
                if (index) std::fputc(',', file);
                const auto & span =
                    (*trace_fields->required_instruction_spans)[index];
                std::fprintf(file, "[%d,%d]", span.begin, span.end);
            }
        }
        std::fputc(']', file);
        if (trace_fields->selector_mode ==
            dflash::qwen3::PFlashSelectionMode::Legacy) {
            std::fputs(",\"stop_reason\":null,\"retained_mass\":null", file);
        } else {
            std::fprintf(file,
                ",\"stop_reason\":\"%s\",\"retained_mass\":%.17g",
                dflash::qwen3::pflash_selection_stop_name(trace_fields->stop),
                trace_fields->retained_mass);
        }
    }
    std::fprintf(file,
        ",\"chunk_size\":%d,\"n_lookahead\":%d,\"pool_kernel\":%d,"
        "\"n_keep\":%d,\"chunk_scores\":[",
        chunk_size, n_lookahead, pool_kernel, n_keep);
    for (size_t index = 0; index < scores.size(); ++index) {
        if (index) std::fputc(',', file);
        const double score = has_exact_scores
            ? (*trace_fields->exact_chunk_scores)[index]
            : (double) scores[index];
        if (std::isfinite(score)) {
            std::fprintf(file, has_exact_scores ? "%.17g" : "%.9g", score);
        } else {
            std::fputs("null", file);
        }
    }
    std::fputs("],\"selected_chunks\":[", file);
    bool first = true;
    for (size_t index = 0; index < selected.size(); ++index) {
        if (!selected[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"forced_chunks\":[", file);
    first = true;
    for (size_t index = 0; index < forced.size(); ++index) {
        if (!forced[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"compressed_ids\":[", file);
    for (size_t index = 0; index < compressed_ids.size(); ++index) {
        if (index) std::fputc(',', file);
        std::fprintf(file, "%d", compressed_ids[index]);
    }
    std::fputs("]}\n", file);
    std::fclose(file);
}

static std::vector<int32_t> select_longattncomp_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const dflash::qwen3::PFlashLongAttnCompConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace) {
    const int input_tokens = (int) ids.size();
    const int query_end = score_query_end < 0 ? input_tokens : score_query_end;
    const int query_tokens = std::min(n_lookahead, query_end);
    const int query_begin = query_end - query_tokens;
    const int selector_budget = (int) std::floor(
        (double) input_tokens * (double) keep_ratio);
    const int n_chunks =
        (input_tokens + config.chunk_size - 1) / config.chunk_size;

    std::vector<dflash::qwen3::PFlashSelectionCandidate> candidates;
    std::vector<std::pair<float, int>> chunk_means;
    std::vector<double> exact_chunk_scores;
    candidates.reserve((size_t) n_chunks);
    chunk_means.reserve((size_t) n_chunks);
    exact_chunk_scores.reserve((size_t) n_chunks);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int begin = chunk * config.chunk_size;
        const int end = std::min(input_tokens, begin + config.chunk_size);
        double score = 0.0;
        for (int token = begin; token < end; ++token) {
            score += token_scores[(size_t) token];
        }
        if (!direct_mass) {
            score /= (double) std::max(1, end - begin);
        }
        const bool mandatory =
            dflash::qwen3::pflash_chunk_is_structurally_required(
                begin, end, query_begin, query_end, input_tokens,
                required_instruction_spans);
        candidates.push_back({(size_t) chunk, begin, end, score, mandatory});
        chunk_means.push_back({(float) score, chunk});
        exact_chunk_scores.push_back(score);
    }

    const auto selected = dflash::qwen3::select_pflash_candidates(
        candidates,
        dflash::qwen3::PFlashSelectionPolicy{selector_budget, config.top_p},
        config.mode);
    if (!selected.ok) {
        set_last_error("PFlash LongAttnComp selection failed: " + selected.error);
        std::fprintf(stderr,
            "[pflash-longattncomp] ERROR mode=%s budget=%d stop=%s: %s\n",
            dflash::qwen3::pflash_selection_mode_name(config.mode),
            selector_budget,
            dflash::qwen3::pflash_selection_stop_name(selected.stop),
            selected.error.c_str());
        std::fflush(stderr);
        return {};
    }

    std::vector<uint8_t> selected_mask((size_t) n_chunks, 0);
    std::vector<uint8_t> mandatory_mask((size_t) n_chunks, 0);
    for (const auto & candidate : candidates) {
        if (candidate.mandatory) mandatory_mask[candidate.ordinal] = 1;
    }
    for (size_t ordinal : selected.ordinals) {
        if (ordinal >= selected_mask.size()) {
            set_last_error("PFlash LongAttnComp selector returned an invalid ordinal");
            return {};
        }
        selected_mask[ordinal] = 1;
    }

    std::vector<int32_t> output;
    output.reserve((size_t) selected.retained_tokens);
    for (const auto & candidate : candidates) {
        if (!selected_mask[candidate.ordinal]) continue;
        output.insert(output.end(),
                      ids.begin() + candidate.begin,
                      ids.begin() + candidate.end);
    }

    std::fprintf(stderr,
        "[pflash-longattncomp] selected mode=%s chunk=%d query=%d "
        "budget=%d selected_tokens=%zu chunks=%zu/%d stop=%s mass=%.9g\n",
        dflash::qwen3::pflash_selection_mode_name(config.mode),
        config.chunk_size, query_tokens, selector_budget, output.size(),
        selected.ordinals.size(), n_chunks,
        dflash::qwen3::pflash_selection_stop_name(selected.stop),
        selected.retained_mass);
    std::fflush(stderr);

    if (write_trace) {
        const int n_keep_approx = std::max(
            1, (selector_budget + config.chunk_size - 1) / config.chunk_size);
        const PFlashTraceFields strict_fields{
            &ids, query_begin, query_end, config.mode, config.query_parser,
            selector_budget,
            selected.stop, selected.retained_tokens, selected.retained_mass,
            &exact_chunk_scores, &required_instruction_spans};
        write_compression_trace(
            input_tokens, keep_ratio, config.chunk_size, query_tokens,
            pool_kernel, n_keep_approx, chunk_means, selected_mask,
            mandatory_mask, output, &strict_fields);
    }
    return output;
}

#if defined(DFLASH27B_BACKEND_HIP)
bool prewarm_drafter_once(const Qwen3DrafterWeights & w) {
    static bool warmed = false;
    if (warmed || std::getenv("DFLASH_FP_SKIP_PREWARM")) {
        return true;
    }

    const int warm_tokens = 1024;
    const int n_lookahead = 8;
    std::vector<int32_t> ids((size_t)warm_tokens, 0);
    std::vector<float> running_max;

    auto t0 = std::chrono::steady_clock::now();
    bool ok = forward_qwen3_drafter_model(w, ids, n_lookahead, running_max);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) {
        return false;
    }

    std::fprintf(stderr, "[drafter] HIP prewarm %.2fs (%d tokens)\n",
                 std::chrono::duration<double>(t1 - t0).count(), warm_tokens);
    std::fflush(stderr);
    warmed = true;
    return true;
}
#endif

} // namespace

bool parse_drafter_arch(const std::string & name, DrafterArch & out) {
    if (name == "qwen3-0.6b" || name == "qwen3_0p6b" || name == "qwen3") {
        out = DrafterArch::Qwen3_0p6b;
        return true;
    }
    if (name == "qwen35-0.8b" || name == "qwen3.5-0.8b" || name == "qwen35_0p8b" || name == "qwen35") {
        out = DrafterArch::Qwen35_0p8b;
        return true;
    }
    return false;
}

const char * drafter_arch_name(DrafterArch arch) {
    switch (arch) {
        case DrafterArch::Qwen3_0p6b: return "qwen3-0.6b";
        case DrafterArch::Qwen35_0p8b: return "qwen35-0.8b";
    }
    return "unknown";
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, /*gpu=*/0, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  int gpu, DrafterContext & out) {
    DrafterArch arch = DrafterArch::Qwen3_0p6b;
    {
        std::string lower = gguf_path;
        for (auto & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find("qwen3.5") != std::string::npos ||
            lower.find("qwen35")  != std::string::npos) {
            arch = DrafterArch::Qwen35_0p8b;
        }
    }
    return load_drafter(gguf_path, /*gpu_layers=*/999, arch, gpu, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterArch arch, DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, arch, /*gpu=*/0, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterArch arch, int gpu, DrafterContext & out) {
    if (gpu < 0) {
        set_last_error("load_drafter: negative GPU index");
        return false;
    }
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }
    if (out.backend && out.gpu >= 0 && out.gpu != gpu) {
        set_last_error("load_drafter: backend already bound to a different GPU");
        return false;
    }

    // If caller didn't supply a backend, spin up our own GPU backend. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process GPU pools coexist fine; fragmentation is
    // the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        int seen_gpu = 0;
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                if (seen_gpu == gpu) {
                    out.backend = ggml_backend_dev_init(dev, nullptr);
                    break;
                }
                seen_gpu++;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: requested GPU backend unavailable");
            return false;
        }
        out.gpu = gpu;
    } else if (out.gpu < 0) {
        out.gpu = gpu;
    }

    if (arch == DrafterArch::Qwen35_0p8b) {
        auto * st = new Qwen35DrafterState();
        // The scorer never needs logits, and tied-embedding Qwen3.5-0.8B
        // exports omit output.weight, so skip the lm_head entirely.
        TargetLoadPlan plan;
        plan.load_output = false;
        if (!load_target_gguf_partial(gguf_path, out.backend, plan, st->weights)) {
            delete st;
            return false;
        }
        if (const char * head_path = std::getenv("PFLASH_LONGATTNCOMP_HEAD_GGUF")) {
            const auto identity = read_gguf_metadata(gguf_path, /*compute_sha256=*/ true);
            st->gguf_sha256 = identity.ok ? identity.sha256 : std::string();
            if (!*head_path || !load_qwen35_longattncomp_head(head_path, *st)) {
                if (!*head_path) {
                    set_last_error("PFLASH_LONGATTNCOMP_HEAD_GGUF is empty");
                }
                std::fprintf(stderr,
                    "[qwen35-drafter] ERROR: LongAttnComp head load failed, "
                    "refusing to serve without it\n");
                std::fflush(stderr);
                free_target_weights(st->weights);
                delete st;
                return false;
            }
        }
        out.arch_state = st;
        out.loaded = true;
        out.arch = arch;
        std::fprintf(stderr,
            "[drafter] loaded %s qwen35: n_layer=%d n_head=%d n_head_kv=%d "
            "n_embd=%d n_ff=%d head_dim=%d vocab=%d gpu=%d\n",
            drafter_arch_name(arch),
            st->weights.n_layer, st->weights.n_head, st->weights.n_head_kv,
            st->weights.n_embd, st->weights.n_ff, st->weights.n_embd_head_k,
            st->weights.n_vocab, out.gpu);
        std::fflush(stderr);
        return true;
    }

    if (!load_qwen3_drafter_model(gguf_path, out.backend, out.weights)) {
        // last_error already set by loader
        return false;
    }

    out.loaded = true;
    out.arch = arch;
    std::fprintf(stderr,
        "[drafter] loaded %s weights=%s compute=%s: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d gpu=%d\n",
        drafter_arch_name(arch),
        backend_precision_type_name(out.weights.weight_type),
        backend_precision_type_name(out.weights.compute_type),
        out.weights.n_layer, out.weights.n_head, out.weights.n_head_kv,
        out.weights.n_embd, out.weights.n_ff, out.weights.head_dim,
        out.weights.n_vocab, out.gpu);
    std::fflush(stderr);

#if defined(DFLASH27B_BACKEND_HIP)
    if (!prewarm_drafter_once(out.weights)) {
        free_drafter(out);
        return false;
    }
#endif

    return true;
}

void free_drafter(DrafterContext & ctx) {
    free_drafter_weights(ctx);
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.gpu = -1;
}

void free_drafter_weights(DrafterContext & ctx) {
    if (ctx.arch == DrafterArch::Qwen35_0p8b && ctx.arch_state) {
        auto * st = static_cast<Qwen35DrafterState *>(ctx.arch_state);
        free_qwen35_head(*st);
        free_target_weights(st->weights);
        delete st;
        ctx.arch_state = nullptr;
    }
    if (ctx.loaded) {
        if (ctx.arch == DrafterArch::Qwen3_0p6b) {
            free_qwen3_drafter_model(ctx.weights);
        }
    }
    ctx.loaded = false;
}

static std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const dflash::qwen3::PFlashLongAttnCompConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {

    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    if (S < n_lookahead + 1) return ids;
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (n_lookahead < 1 || query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 scorer query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max((size_t)n_lookahead * S, -INFINITY);

    TargetCache cache;
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
            return {};
        }
    }

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1], (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        int fa_idx = 0;
        if (is_attn) {
            for (int k = 0; k < il; ++k) if (((k + 1) % w.full_attention_interval) == 0) ++fa_idx;
        }
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            const int kv_len = start + n;

            ggml_init_params ip{};
            ip.mem_size = 512 * 1024 * 1024;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter layer graph ctx init failed");
                return {};
            }
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1], (size_t)start * act_in->nb[1]);
            ggml_tensor * pos = nullptr;
            ggml_tensor * mask = nullptr;
            if (is_attn) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(pos);
                mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, align_up_i(kv_len, 32), align_up_i(n, 32));
                ggml_set_input(mask);
            }
            ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask, start, n, false, 0);
            ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1], (size_t)start * act_out->nb[1]);
            if (ggml_nelements(out) != ggml_nelements(dst)) {
                std::fprintf(stderr,
                    "[qwen35-drafter] layer output shape mismatch il=%d start=%d out=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                    il, start,
                    (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                    (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 layer output shape mismatch");
                return {};
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
                std::vector<uint16_t> m;
                build_causal_mask_f16(m, kv_len, n, start);
                ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(uint16_t));
            }
            auto st = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }

        if (is_attn) {
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(1024, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph ctx allocation failed");
                return {};
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 1024, false);
            const int K_len = (int) cache.attn_k[(size_t)fa_idx]->ne[1];
            ggml_tensor * mask_tail = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K_len, n_lookahead);
            ggml_tensor * K_f32 = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, w.n_head_kv);
            ggml_tensor * K_cast = ggml_cpy(sctx, cache.attn_k[(size_t)fa_idx], K_f32);
            ggml_tensor * K_score = nullptr;
            if (w.n_head != w.n_head_kv) {
                const int gqa = w.n_head / w.n_head_kv;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K_cast, w.n_embd_head_k, K_len, 1, w.n_head_kv);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, gqa, w.n_head_kv);
                ggml_tensor * K_rep = ggml_repeat(sctx, K_4d, K_tpl);
                K_score = ggml_reshape_3d(sctx, K_rep, w.n_embd_head_k, K_len, w.n_head);
            } else {
                K_score = K_cast;
            }
            const TargetLayer & L = w.layers[il];
            ggml_tensor * inp_tail = ggml_view_2d(sctx, act_in, hidden, n_lookahead,
                act_in->nb[1], (size_t)query_start * act_in->nb[1]);
            ggml_tensor * q_cur = ggml_rms_norm(sctx, inp_tail, w.rms_eps);
            q_cur = ggml_mul(sctx, q_cur, L.attn_norm);
            ggml_tensor * QG = ggml_mul_mat(sctx, L.wq, q_cur);
            QG = ggml_reshape_3d(sctx, QG, w.n_embd_head_k * 2, w.n_head, n_lookahead);
            ggml_tensor * Q = ggml_view_3d(sctx, QG,
                w.n_embd_head_k, w.n_head, n_lookahead,
                ggml_element_size(QG) * w.n_embd_head_k * 2,
                ggml_element_size(QG) * w.n_embd_head_k * 2 * w.n_head,
                0);
            Q = ggml_rms_norm(sctx, Q, w.rms_eps);
            Q = ggml_mul(sctx, Q, L.q_norm);
            ggml_tensor * pos_tail = ggml_new_tensor_1d(sctx, GGML_TYPE_I32, 4 * n_lookahead);
            int sections[4];
            for (int k = 0; k < 4; ++k) sections[k] = w.rope_sections[k];
            Q = ggml_rope_multi(sctx, Q, pos_tail, nullptr,
                                w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE,
                                0, w.rope_theta, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * Q_tail_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));
            ggml_tensor * attn_score = ggml_mul_mat(sctx, K_score, Q_tail_perm);
            ggml_tensor * probs = ggml_soft_max_ext(sctx, attn_score, mask_tail, 1.0f / std::sqrt((float)w.n_embd_head_k), 0.0f);
            ggml_set_output(probs);
            ggml_build_forward_expand(sgf, probs);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph allocation failed");
                return {};
            }
            std::vector<int32_t> pos4((size_t)4 * n_lookahead, 0);
            for (int i = 0; i < n_lookahead; ++i) {
                const int p = query_start + i;
                pos4[(size_t)0 * n_lookahead + i] = p;
                pos4[(size_t)1 * n_lookahead + i] = p;
                pos4[(size_t)2 * n_lookahead + i] = p;
            }
            ggml_backend_tensor_set(pos_tail, pos4.data(), 0, pos4.size() * sizeof(int32_t));
            std::vector<float> mask((size_t)n_lookahead * K_len, 0.0f);
            for (int t = 0; t < n_lookahead; ++t) {
                const int visible_end = query_start + t + 1;
                for (int j = 0; j < K_len; ++j) {
                    mask[(size_t)t * K_len + j] = (j < visible_end) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask_tail, mask.data(), 0, mask.size() * sizeof(float));
            auto st = ggml_backend_graph_compute(w.backend, sgf);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph compute failed");
                return {};
            }
            std::vector<float> tmp((size_t)K_len * n_lookahead * w.n_head);
            ggml_backend_tensor_get(probs, tmp.data(), 0, tmp.size() * sizeof(float));
            const size_t nonfinite =
                count_nonfinite_scores(tmp.data(), tmp.size());
            if (nonfinite != 0) {
                const std::string message =
                    "non-finite Qwen3.5 PFlash scores at layer " +
                    std::to_string(il) + ": " + std::to_string(nonfinite) +
                    "/" + std::to_string(tmp.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                ggml_gallocr_free(salloc); ggml_free(sctx);
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf);
                ggml_free(act_ctx); free_target_cache(cache);
                set_last_error(message);
                return {};
            }
            for (int h = 0; h < w.n_head; ++h) {
                for (int t = 0; t < n_lookahead; ++t) {
                    for (int j = 0; j < S; ++j) {
                        const size_t src = (size_t)h * K_len * n_lookahead + (size_t)t * K_len + j;
                        const size_t dst = (size_t)t * S + j;
                        running_max[dst] = std::max(running_max[dst], tmp[src]);
                    }
                }
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);

    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) s += running_max[(size_t)t * S + j];
        score[(size_t)j] = s / (float)n_lookahead;
    }

    const int n_chunks = (S + chunk_size - 1) / chunk_size;
    const int n_keep = std::max(1, (int)((float)n_chunks * keep_ratio));
    
    std::vector<float> smooth_score = score;
    // Caller pool_kernel takes precedence; if zero/negative, fall back to env or 5.
    const int pk = (pool_kernel > 0)
        ? pool_kernel
        : std::max(3, env_int("DFLASH_COMPRESS_POOL_KERNEL", 5));
    std::vector<float> smoothed((size_t)S, 0.0f);
    int half = pk / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[(size_t)k]; ++n; }
        smoothed[(size_t)j] = (n > 0) ? (s / (float)n) : 0.0f;
    }
    smooth_score.swap(smoothed);

    if (experiment.selection_active) {
        return select_longattncomp_chunks(
            ids, smooth_score, keep_ratio, n_lookahead, score_query_end,
            pk, experiment, required_instruction_spans, false, true);
    }
    
    std::vector<std::pair<float, int>> chunk_means;
    for (int c = 0; c < n_chunks; ++c) {
        int lo = c * chunk_size, hi = std::min(S, lo + chunk_size);
        float s = 0.0f;
        for (int j = lo; j < hi; ++j) s += smooth_score[(size_t)j];
        chunk_means.push_back({s / std::max(1, hi - lo), c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(), [](auto a, auto b) { return a.first > b.first; });
    
    std::vector<uint8_t> selected((size_t)n_chunks, 0);
    int count = 0;
    // Scale head/tail forced chunks so they don't crowd out top-K scoring.
    {
        const int h_raw = env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8);
        const int t_raw = env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24);
        int h_n = h_raw, t_n = t_raw;
        if (h_n + t_n >= n_keep) {
            const int budget = std::max(1, n_keep - 1);
            h_n = std::max(0, h_raw * budget / (h_raw + t_raw));
            t_n = std::max(0, budget - h_n);
        }
        for (int c = 0; c < std::min(n_chunks, h_n); ++c) { selected[(size_t)c] = 1; ++count; }
        for (int c = std::max(0, n_chunks - t_n); c < n_chunks; ++c) if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }

    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
        env_int("DFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", -1));
    const int anchor_radius   = ap.radius;
    const int max_anchor_hits = ap.max_hits;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        std::vector<int> hit_pos(max_anchor_hits);
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < max_anchor_hits) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < max_anchor_hits; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c] && !selected[(size_t)c]) {
            selected[(size_t)c] = 1;
            ++count;
        }
    }

    // Global aggregation tasks often depend on repeated rare tokens that do
    // not appear in the final query. Preserve high-frequency-but-not-filler
    // token chunks before filling with model-score top-K.
    const int repeat_min = env_int("DFLASH_COMPRESS_REPEAT_MIN", 4);
    const int repeat_max = env_int("DFLASH_COMPRESS_REPEAT_MAX", 32);
    const int repeat_limit = env_int("DFLASH_COMPRESS_REPEAT_CHUNKS", n_keep);
    if (repeat_min > 1 && count < repeat_limit) {
        std::unordered_map<int32_t, int> freq;
        freq.reserve((size_t)S);
        const int repeat_scan_end = std::max(0, S - query_tokens);
        for (int j = 0; j < repeat_scan_end; ++j) {
            ++freq[ids[(size_t)j]];
        }
        std::vector<std::pair<int, int32_t>> repeated;
        repeated.reserve(freq.size());
        for (const auto & kv : freq) {
            if (kv.second >= repeat_min && kv.second <= repeat_max) {
                repeated.push_back({kv.second, kv.first});
            }
        }
        std::sort(repeated.begin(), repeated.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (const auto & rp : repeated) {
            if (count >= repeat_limit) break;
            const int32_t tok = rp.second;
            for (int j = 0; j < repeat_scan_end && count < repeat_limit; ++j) {
                if (ids[(size_t)j] != tok) continue;
                const int c = j / chunk_size;
                if (!selected[(size_t)c]) {
                    selected[(size_t)c] = 1;
                    ++count;
                }
            }
        }
    }
    
    for (auto [_, c] : chunk_means) {
        if (count >= n_keep) break;
        if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }
    
    std::vector<int32_t> out_ids;
    std::vector<int> selected_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (selected[(size_t)c]) selected_chunks.push_back(c);
    }
    int span_start = -1, span_end = -1;
    for (int c : selected_chunks) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[qwen35-drafter] forward+compress %.2fs S=%d kept=%zu (%d/%d chunks)\n",
                 std::chrono::duration<double>(t1 - t0).count(), S, out_ids.size(), count, n_chunks);
    std::fflush(stderr);
    return out_ids;
}

// LongAttnComp scoring for the Qwen3.5-0.8B drafter: run blocks 0..14, then
// score every context token against the query window with block 15's NoPE
// Q/K (or a trained replacement) and select chunks by attention mass. This
// is the runtime counterpart of the Python retention screen (trial 0075).
static std::vector<int32_t> qwen35_longattncomp_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const dflash::qwen3::PFlashLongAttnCompConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {

    TargetWeights & w = st.weights;
    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    const int H = w.n_head;
    const int Hk = w.n_head_kv;
    const int D = w.n_embd_head_k;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return {};
    }
    if (n_lookahead < 1 || S < n_lookahead + 1) {
        set_last_error("qwen35 LongAttnComp scorer input is too short");
        return {};
    }
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 LongAttnComp scorer query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];

    auto t0 = std::chrono::steady_clock::now();
    TargetCache cache;
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
            return {};
        }
    }

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }
    auto cleanup = [&]() {
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);
        free_target_cache(cache);
    };

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                cleanup();
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1],
                                    (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    std::vector<uint16_t> mask_bits;
    for (int il = 0; il < kQwen35HeadBlock; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            const int kv_len = start + n;
            ggml_init_params ip{};
            ip.mem_size = 512 * 1024 * 1024;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter layer graph ctx init failed");
                return {};
            }
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1],
                                             (size_t)start * act_in->nb[1]);
            ggml_tensor * pos = nullptr;
            ggml_tensor * mask = nullptr;
            if (is_attn) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(pos);
                mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                          align_up_i(kv_len, 32), align_up_i(n, 32));
                ggml_set_input(mask);
            }
            ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask,
                                                   start, n, false, 0);
            ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1],
                                             (size_t)start * act_out->nb[1]);
            if (ggml_nelements(out) != ggml_nelements(dst)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 layer output shape mismatch");
                return {};
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    const int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
                build_causal_mask_f16(mask_bits, kv_len, n, start);
                ggml_backend_tensor_set(mask, mask_bits.data(), 0,
                                        mask_bits.size() * sizeof(uint16_t));
            }
            const auto status = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    auto t1 = std::chrono::steady_clock::now();

    // Block-15 NoPE Q/K scoring: softmax over keys before the query window,
    // then mean over heads and query tokens. The query never scores itself.
    // Keys are projected in chunks so no intermediate tensor puts the
    // sequence length into a HIP grid y/z dimension (65,535 limit); the
    // logits land in one [S, n_lookahead, H] buffer for a single softmax.
    const int key_chunk = 8192;
    const int n_key_chunks = (S + key_chunk - 1) / key_chunk;
    ggml_init_params lip{};
    lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    lip.no_alloc = true;
    ggml_context * lctx = ggml_init(lip);
    if (!lctx) {
        cleanup();
        set_last_error("qwen35 score buffer ctx allocation failed");
        return {};
    }
    ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, n_lookahead, H);
    ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, n_lookahead);
    ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
    if (!lbuf) {
        ggml_free(lctx); cleanup();
        set_last_error("qwen35 score buffer allocation failed");
        return {};
    }
    {
        std::vector<float> m((size_t)n_lookahead * S, -INFINITY);
        for (int t = 0; t < n_lookahead; ++t) {
            std::fill_n(m.begin() + (size_t)t * S, (size_t)query_start, 0.0f);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    ggml_init_params sip{};
    sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                   ggml_graph_overhead_custom(4096, false) + 64 * 1024;
    sip.no_alloc = true;
    ggml_context * sctx = ggml_init(sip);
    if (!sctx) {
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph ctx allocation failed");
        return {};
    }
    ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
    ggml_tensor * wk_src = st.head_loaded ? st.head_wk : L.wk;
    ggml_tensor * x_q = ggml_view_2d(sctx, act_in, hidden, n_lookahead, act_in->nb[1],
                                     (size_t)query_start * act_in->nb[1]);
    ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
    ggml_tensor * Q = nullptr;
    if (st.head_loaded) {
        Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, n_lookahead);
    } else {
        // Native block 15 packs query and gate rows per head; keep the query half.
        ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                           D * 2, H, n_lookahead);
        Q = ggml_view_3d(sctx, QG, D, H, n_lookahead,
                         ggml_element_size(QG) * D * 2,
                         ggml_element_size(QG) * D * 2 * H, 0);
    }
    Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
    ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, n_lookahead, H]
    for (int b = 0; b < S; b += key_chunk) {
        const int n = std::min(key_chunk, S - b);
        ggml_tensor * x_c = ggml_view_2d(sctx, act_in, hidden, n, act_in->nb[1],
                                         (size_t)b * act_in->nb[1]);
        ggml_tensor * x_norm = ggml_mul(sctx, ggml_rms_norm(sctx, x_c, w.rms_eps), L.attn_norm);
        ggml_tensor * K = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, wk_src, x_norm), D, Hk, n);
        K = ggml_mul(sctx, ggml_rms_norm(sctx, K, w.rms_eps), L.k_norm);
        K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, n, Hk]
        ggml_tensor * K_score = K;
        if (H != Hk) {
            const int gqa = H / Hk;
            ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, n, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, n, gqa, Hk);
            K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, n, H);
        }
        ggml_tensor * part = ggml_mul_mat(sctx, K_score, Q_perm);  // [n, n_lookahead, H]
        ggml_tensor * dst = ggml_view_3d(sctx, logits, n, n_lookahead, H,
                                         logits->nb[1], logits->nb[2],
                                         (size_t)b * logits->nb[0]);
        ggml_build_forward_expand(sgf, ggml_cpy(sctx, part, dst));
    }
    ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                            1.0f / std::sqrt((float)D), 0.0f);
    ggml_set_output(probs);
    ggml_build_forward_expand(sgf, probs);
    ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
        ggml_gallocr_free(salloc); ggml_free(sctx);
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph allocation failed");
        return {};
    }
    const auto score_status = ggml_backend_graph_compute(w.backend, sgf);
    if (score_status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(salloc); ggml_free(sctx);
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph compute failed");
        return {};
    }
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    ggml_backend_tensor_get(probs, probs_h.data(), 0, probs_h.size() * sizeof(float));
    ggml_gallocr_free(salloc);
    ggml_free(sctx);
    ggml_backend_buffer_free(lbuf);
    ggml_free(lctx);
    cleanup();
    const size_t nonfinite = count_nonfinite_scores(probs_h.data(), probs_h.size());
    if (nonfinite != 0) {
        const std::string message =
            "non-finite Qwen3.5 LongAttnComp scores: " + std::to_string(nonfinite) +
            "/" + std::to_string(probs_h.size());
        std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
        std::fflush(stderr);
        set_last_error(message);
        return {};
    }
    std::vector<float> token_mass;
    longattncomp_mean_token_mass(probs_h.data(), S, n_lookahead, H, token_mass);
    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[qwen35-longattncomp] forward %.2fs (blocks 0-%d, S=%d) score %.2fs "
        "total %.2fs head=%s\n",
        std::chrono::duration<double>(t1 - t0).count(), kQwen35HeadBlock - 1, S,
        std::chrono::duration<double>(t2 - t1).count(),
        std::chrono::duration<double>(t2 - t0).count(),
        st.head_loaded ? "trained" : "native-block15");
    std::fflush(stderr);

    return select_longattncomp_chunks(
        ids, token_mass, keep_ratio, n_lookahead, score_query_end,
        /*pool_kernel=*/1, experiment, required_instruction_spans,
        /*direct_mass=*/true, /*write_trace=*/true);
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }

    dflash::qwen3::PFlashLongAttnCompConfig experiment;
    std::string experiment_error;
    if (!dflash::qwen3::resolve_pflash_longattncomp(
            (int) ids.size(), chunk_size, experiment, experiment_error)) {
        set_last_error("invalid PFlash LongAttnComp config: " + experiment_error);
        std::fprintf(stderr, "[pflash-longattncomp] ERROR config: %s\n",
                     experiment_error.c_str());
        std::fflush(stderr);
        return {};
    }
    chunk_size = experiment.chunk_size;
    if (!experiment.selection_active && !required_instruction_spans.empty()) {
        set_last_error(
            "PFlash instruction spans require strict LongAttnComp selection");
        std::fprintf(stderr,
            "[pflash-longattncomp] ERROR instruction spans require strict selection\n");
        std::fflush(stderr);
        return {};
    }
    if (experiment.selection_active) {
        std::string span_error;
        if (!dflash::qwen3::validate_pflash_instruction_spans(
                required_instruction_spans, (int) ids.size(), span_error)) {
            set_last_error("invalid PFlash instruction spans: " + span_error);
            std::fprintf(stderr,
                "[pflash-longattncomp] ERROR instruction spans: %s\n",
                span_error.c_str());
            std::fflush(stderr);
            return {};
        }
    }
    if (experiment.configured) {
        std::fprintf(stderr,
            "[pflash-longattncomp] config mode=%s active=%d chunk=%d "
            "query_parser=%s query_cap=%d query_actual=%d top_p=%.9g "
            "input=%zu\n",
            dflash::qwen3::pflash_selection_mode_name(experiment.mode),
            (int) experiment.selection_active, experiment.chunk_size,
            dflash::qwen3::pflash_query_parser_name(experiment.query_parser),
            experiment.query_tokens, n_lookahead, experiment.top_p, ids.size());
        std::fflush(stderr);
    }
    if (ctx.arch == DrafterArch::Qwen35_0p8b) {
        if (!ctx.arch_state) {
            set_last_error("qwen35 drafter state missing");
            return {};
        }
        auto * st = static_cast<Qwen35DrafterState *>(ctx.arch_state);
        // Strict LongAttnComp selection scores with the block-15 head; the
        // legacy all-layer running-max scorer stays available for legacy
        // selection or when PFLASH_QWEN35_LEGACY_SCORER=1 forces it.
        const char * legacy_scorer = std::getenv("PFLASH_QWEN35_LEGACY_SCORER");
        const bool force_legacy = legacy_scorer && std::string(legacy_scorer) == "1";
        if (experiment.selection_active && !force_legacy) {
            return qwen35_longattncomp_score_and_compress(
                *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                required_instruction_spans);
        }
        if (st->head_loaded) {
            set_last_error("Qwen3.5 LongAttnComp head requires strict selection");
            return {};
        }
        return qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                         n_lookahead, pool_kernel, score_query_end,
                                         experiment,
                                         required_instruction_spans);
    }
    const int S = (int)ids.size();
    if (S < n_lookahead + 1) {
        // Too short to score — return as-is.
        return ids;
    }

    // ── 1. Custom forward + GPU tail-attention scoring ────────────────
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max;
    if (!forward_qwen3_drafter_model(
            ctx.weights, ids, n_lookahead, running_max, score_query_end)) {
        return {};
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[drafter] forward+score in %.2fs S=%d\n",
        std::chrono::duration<double>(t1 - t0).count(), S);
    std::fflush(stderr);

    // ── 2. Mean over lookahead → per-token score [S] ──────────────────
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) {
            s += running_max[(size_t)t * S + j];
        }
        score[j] = s / (float)n_lookahead;
    }

    // ── 3. AvgPool 1D smoothing ───────────────────────────────────────
    std::vector<float> smooth((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[k]; ++n; }
        smooth[j] = (n > 0) ? (s / (float)n) : 0.0f;
    }

    if (experiment.selection_active) {
        return select_longattncomp_chunks(
            ids, ctx.weights.longattncomp_head_loaded ? score : smooth,
            keep_ratio, n_lookahead, score_query_end,
            ctx.weights.longattncomp_head_loaded ? 1 : pool_kernel,
            experiment, required_instruction_spans,
            ctx.weights.longattncomp_head_loaded, true);
    }

    // ── 4. Chunk-top-K + span merge ───────────────────────────────────
    int n_chunks = (S + chunk_size - 1) / chunk_size;
    int n_keep   = std::max(1, (int)((float)n_chunks * keep_ratio));
    std::vector<std::pair<float, int>> chunk_means;
    chunk_means.reserve((size_t)n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        float m = 0.0f;
        for (int j = s_; j < e_; ++j) m += smooth[j];
        m /= std::max(1, e_ - s_);
        chunk_means.push_back({m, c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(),
                      [](auto a, auto b) { return a.first > b.first; });

    // Retrieval tasks often repeat a rare key in the final query and in the
    // needle span. Exact scores alone can keep the query while dropping the
    // neighboring answer chunk, so force a small token-only anchor neighborhood.
    // Head/tail forced chunks scale with n_keep so top-K scoring always gets slots.
    const int h_raw = env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8);
    const int t_raw = env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24);
    int head_chunks = h_raw, tail_chunks = t_raw;
    if (head_chunks + tail_chunks >= n_keep) {
        const int budget = std::max(1, n_keep - 1);
        head_chunks = std::max(0, h_raw * budget / (h_raw + t_raw));
        tail_chunks = std::max(0, budget - head_chunks);
    }
    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
        env_int("DFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", -1));
    const int anchor_radius   = ap.radius;
    const int max_anchor_hits = ap.max_hits;
    std::vector<uint8_t> selected_mask((size_t)n_chunks, 0);
    std::vector<uint8_t> forced((size_t)n_chunks, 0);
    for (int c = 0; c < std::min(n_chunks, head_chunks); ++c) forced[(size_t)c] = 1;
    for (int c = std::max(0, n_chunks - tail_chunks); c < n_chunks; ++c) forced[(size_t)c] = 1;

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        std::vector<int> hit_pos(max_anchor_hits);
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < max_anchor_hits) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < max_anchor_hits; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }

    int selected_count = 0;
    int forced_count = 0;
    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
            ++forced_count;
        }
    }
    for (const auto & cm : chunk_means) {
        if (selected_count >= n_keep) break;
        int c = cm.second;
        if (!selected_mask[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
        }
    }

    std::vector<int> selected;
    selected.reserve((size_t)selected_count);
    for (int c = 0; c < n_chunks; ++c) {
        if (selected_mask[(size_t)c]) selected.push_back(c);
    }

    std::vector<int32_t> out;
    out.reserve((size_t)n_keep * chunk_size + 16);
    int span_start = -1, span_end = -1;
    for (int c : selected) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
    }

    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[drafter] score_and_compress total %.2fs S=%d kept=%zu (%d/%d chunks, forced=%d)\n",
        std::chrono::duration<double>(t2 - t0).count(),
        S, out.size(), (int)selected.size(), n_chunks, forced_count);
    std::fflush(stderr);

    const int query_end = score_query_end < 0 ? S : score_query_end;
    const int query_begin = query_end - n_lookahead;
    const int token_budget = (int) std::floor(
        (double) S * (double) keep_ratio);
    const PFlashTraceFields trace_fields{
        &ids, query_begin, query_end, experiment.mode,
        experiment.query_parser, token_budget,
        dflash::qwen3::PFlashSelectionStop::InvalidInput, (int) out.size(),
        0.0, nullptr};
    write_compression_trace(S, keep_ratio, chunk_size, n_lookahead,
        pool_kernel, n_keep, chunk_means, selected_mask, forced, out,
        &trace_fields);

    return out;
}

} // namespace dflash::common
