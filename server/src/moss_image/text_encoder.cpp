#include "text_encoder.h"

#include "internal.h"
#include "luce.h"
#include "pflash/qwen35_drafter.h"
#include "qwen35/prefill_helpers.h"
#include "server/tokenizer.h"
#include "weights.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime_api.h>

namespace luce::moss_image {

namespace common = luce::common;

namespace {

const char *  kTemplateHead = "<|im_start|>user\n";
const char *  kTemplateTail = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
constexpr int kTextLayers   = 32;
constexpr int kGraphNodes   = 32768;  // a 512-token chunk's graph has about 17k nodes

// The encoder's work runs on its GPU whatever the calling thread had selected, and the thread gets its GPU back: the
// pipeline's work that follows runs on the model's.
struct OnDevice {
    int previous = 0;

    explicit OnDevice(const int gpu) {
        (void) hipGetDevice(&previous);
        (void) hipSetDevice(gpu);
    }

    ~OnDevice() { (void) hipSetDevice(previous); }
};

} // namespace

struct TextEncoder::Impl {
    ggml_backend_t             backend = nullptr;
    common::TargetWeights      weights;
    common::TargetCache        cache;
    common::Tokenizer          tokenizer;
    ggml_context *             out_ctx = nullptr;
    ggml_backend_buffer_t      out_buf = nullptr;
    ggml_tensor *              hidden  = nullptr;  // [4096, kMaxTokens] f32: every token's final normed state
    ggml_gallocr_t             galloc  = nullptr;
    size_t                     arena_bytes = 0;    // graph metadata, rebuilt for every chunk
    std::unique_ptr<uint8_t[]> arena;
    std::vector<float>         embeddings;
    std::vector<int32_t>       positions;
    int                        gpu    = 0;
    bool                       remote = false;  // on another GPU than the model: rows go out through host_rows
    std::vector<float>         host_rows;

    ~Impl() {
        if (galloc != nullptr) {
            ggml_gallocr_free(galloc);
        }
        if (out_buf != nullptr) {
            ggml_backend_buffer_free(out_buf);
        }
        if (out_ctx != nullptr) {
            ggml_free(out_ctx);
        }
        common::free_target_cache(cache);
        common::free_target_weights(weights);
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    // Tokens [kv_start, kv_start + n) through all layers, their normed states into hidden's rows.
    bool chunk(const int32_t * tokens, const int kv_start, const int n) {
        ggml_init_params params{};
        params.mem_size     = arena_bytes;
        params.mem_buffer   = arena.get();
        params.no_alloc     = true;
        ggml_context * ctx  = ggml_init(params);
        const int      embd = weights.n_embd;

        ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, embd, n, 1);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
        ggml_set_input(inp);
        ggml_set_input(pos);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, kGraphNodes, false);

        common::QwenGraphInputs in{};
        in.inp_embed = inp;
        in.positions = pos;
        in.attn_mask = common::build_qwen35_causal_mask(ctx, gf, kv_start, n);
        in.n_tokens  = n;
        in.kv_start  = kv_start;
        common::build_qwen35_graph(ctx, gf, weights, cache, in);
        ggml_tensor * norm = ggml_graph_get_tensor(gf, "result_norm");
        bool          ok   = norm != nullptr;
        if (ok) {
            ggml_tensor * rows =
                ggml_view_2d(ctx, hidden, embd, n, hidden->nb[1], size_t(kv_start) * hidden->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, norm, rows));
            embeddings.resize(size_t(embd) * size_t(n));
            positions.assign(size_t(4) * size_t(n), 0);
            common::fill_qwen35_mrope_positions(positions.data(), kv_start, n);
            ok = ggml_gallocr_alloc_graph(galloc, gf) && weights.embedder.embed(tokens, n, embeddings.data());
        }
        if (ok) {
            ggml_backend_tensor_set(inp, embeddings.data(), 0, ggml_nbytes(inp));
            ggml_backend_tensor_set(pos, positions.data(), 0, ggml_nbytes(pos));
            // Synchronous: the rows are in hidden when it returns.
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        }
        ggml_free(ctx);
        return ok;
    }
};

TextEncoder::TextEncoder() : impl_(std::make_unique<Impl>()) {}

TextEncoder::~TextEncoder() = default;

bool TextEncoder::load(const std::string & gguf_path, const int gpu, const int model_gpu, std::string & error) {
    OnDevice on(gpu);
    Impl &   m = *impl_;
    m.gpu      = gpu;
    m.remote   = gpu != model_gpu;
    m.backend  = ggml_backend_cuda_init(gpu);
    if (m.backend == nullptr) {
        error = "the text encoder cannot open GPU " + std::to_string(gpu);
        return false;
    }
    common::TargetLoadPlan plan;
    plan.load_lm_head = false;
    if (!common::load_target_gguf_partial(gguf_path, m.backend, plan, m.weights)) {
        error = "cannot load the text encoder " + gguf_path + ": " + luce_last_error();
        return false;
    }
    if (m.weights.is_moe || m.weights.n_embd != kTextHidden || m.weights.n_layer != kTextLayers) {
        error = gguf_path + " is not the Qwen3.5-9B text encoder";
        return false;
    }
    // f16 K/V like a plain forward pass; one prompt fills at most kMaxTokens rows.
    if (!common::create_target_cache(m.weights, kMaxTokens, 0, m.backend, m.cache, /*prefill_only=*/true,
                                     /*ctx_alloc=*/0, /*paged_attention=*/false, /*n_seq_slots=*/1,
                                     /*concurrent_tree=*/false, GGML_TYPE_F16, GGML_TYPE_F16)) {
        error = std::string("cannot allocate the text encoder cache: ") + luce_last_error();
        return false;
    }
    if (!m.tokenizer.load_from_gguf(gguf_path.c_str())) {
        error = "cannot read the tokenizer of " + gguf_path;
        return false;
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead();
    params.no_alloc = true;
    m.out_ctx       = ggml_init(params);
    m.hidden        = ggml_new_tensor_2d(m.out_ctx, GGML_TYPE_F32, kTextHidden, kMaxTokens);
    m.out_buf       = ggml_backend_alloc_ctx_tensors(m.out_ctx, m.backend);
    m.galloc        = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (m.out_buf == nullptr || m.galloc == nullptr) {
        error = "cannot allocate the text encoder output";
        return false;
    }
    m.arena_bytes = ggml_graph_overhead_custom(kGraphNodes, false) + size_t(4 * kGraphNodes) * ggml_tensor_overhead();
    m.arena.reset(new uint8_t[m.arena_bytes]);
    return true;
}

std::vector<int32_t> TextEncoder::tokenize(const std::string & prompt) const {
    std::vector<int32_t> tokens = impl_->tokenizer.encode(kTemplateHead + prompt + kTemplateTail);
    if (tokens.size() > size_t(kMaxTokens)) {
        tokens.resize(kMaxTokens);
    }
    return tokens;
}

bool TextEncoder::encode(const std::vector<int32_t> & tokens, TextHidden & out, std::string & error) {
    Impl &    m = *impl_;
    OnDevice  on(m.gpu);
    const int n = int(tokens.size());
    if (n < 1 || n > kMaxTokens) {
        error = "a prompt takes 1 to " + std::to_string(kMaxTokens) + " tokens";
        return false;
    }
    common::reset_recurrent_state(m.cache);
    for (int start = 0; start < n;) {
        const int len = common::qwen35_prefill_chunk_tokens(start, n - start, common::kQwen35PrefillUbatch, {},
                                                            common::kQwen35MinChunkTokens);
        if (!m.chunk(tokens.data() + start, start, len)) {
            error = "the text encoder failed";
            return false;
        }
        start += len;
    }
    out.n_tokens = n;
    out.host     = m.remote;
    if (m.remote) {
        // Through host memory, not GPU to GPU: the two GPUs have no peer access to each other's memory.
        m.host_rows.resize(size_t(n) * kTextHidden);
        ggml_backend_tensor_get(m.hidden, m.host_rows.data(), 0, m.host_rows.size() * sizeof(float));
        out.data = m.host_rows.data();
    } else {
        out.data = static_cast<const float *>(m.hidden->data);
    }
    return true;
}

} // namespace luce::moss_image
