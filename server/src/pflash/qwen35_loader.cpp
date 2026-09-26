// Qwen3.5-0.8B drafter loading: the drafter GGUF, the optional trained
// block-15 scoring head and the optional segment probe.
//
// The scorer is built on the Qwen3.5 target weights
// (load_target_gguf_partial).

#include "qwen35_drafter.h"

#include "common/gguf_inspect.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
    st.head_loaded = false;
}

static void free_qwen35_segment_probe(Qwen35DrafterState & st) {
    if (st.probe_buf) { ggml_backend_buffer_free(st.probe_buf); st.probe_buf = nullptr; }
    if (st.probe_ctx) { ggml_free(st.probe_ctx); st.probe_ctx = nullptr; }
    st.probe_norm_w = st.probe_norm_b = st.probe_fc1_w = st.probe_fc1_b =
        st.probe_fc2_w = st.probe_fc2_b =
        st.probe_sub_fc2_w = st.probe_sub_fc2_b = nullptr;
    st.probe_conv_w.clear();
    st.probe_sub_conv_w.clear();
    st.probe_loaded = false;
}

static bool qwen35_metadata_equals(gguf_context * g, const char * key,
                                   const std::string & expected) {
    const int id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
           expected == gguf_get_val_str(g, id);
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
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
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
    if (!st.head_buf) {
        fail("scoring head buffer allocation failed");
        set_last_oom_error("scoring head buffer allocation failed");
        return false;
    }
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    gguf_free(g);
    ggml_free(data_ctx);
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
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
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
    probe_params.mem_size = 8 * ggml_tensor_overhead();
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
    st.probe_buf = ggml_backend_alloc_ctx_tensors(st.probe_ctx, w.backend);
    if (!st.probe_buf) {
        fail("segment probe buffer allocation failed");
        set_last_oom_error("segment probe buffer allocation failed");
        return false;
    }
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    if (st.probe_sub_fc2_w) {
        ggml_backend_tensor_set(st.probe_sub_fc2_w, sub_src->data, 0, ggml_nbytes(sub_src));
        ggml_backend_tensor_set(st.probe_sub_fc2_b, sub_b_src->data, 0, ggml_nbytes(sub_b_src));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.probe_loaded = true;
    std::fprintf(stderr,
        "[qwen35-drafter] loaded segment probe: %s (width %d, threshold %.3f, "
        "segments %d-%d tokens, %zu conv taps%s)\n",
        path.c_str(), st.probe_width, st.probe_threshold,
        st.probe_min_segment, st.probe_max_segment, st.probe_conv_w.size(),
        st.probe_sub_fc2_w ? ", sub-unit head" : "");
    std::fflush(stderr);
    return true;
}

} // namespace

bool load_qwen35_drafter(const std::string & gguf_path,
                         DrafterContext & out) {
    auto * st = new Qwen35DrafterState();
    // The scorer never needs logits, and tied-embedding Qwen3.5-0.8B
    // exports omit output.weight, so skip the lm_head entirely.
    TargetLoadPlan plan;
    plan.load_output = false;
    if (!load_target_gguf_partial(gguf_path, out.backend, plan, st->weights)) {
        delete st;
        return false;
    }
    const char * head_path = std::getenv("PFLASH_SCORING_HEAD_GGUF");
    const char * probe_path = std::getenv("PFLASH_SEGMENT_PROBE_GGUF");
    if (head_path || probe_path) {
        const auto identity = read_gguf_metadata(gguf_path, /*compute_sha256=*/ true);
        st->gguf_sha256 = identity.ok ? identity.sha256 : std::string();
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
    if (head_path) {
        if (!*head_path || !load_qwen35_scoring_head(head_path, *st)) {
            if (!*head_path) {
                set_last_error("PFLASH_SCORING_HEAD_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: scoring head load failed, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    out.state = st;
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
    free_qwen35_head(*st);
    free_qwen35_segment_probe(*st);
    free_target_weights(st->weights);
    delete st;
    ctx.state = nullptr;
}

} // namespace luce::common
