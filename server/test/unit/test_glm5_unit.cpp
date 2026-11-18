// GLM-5.3-Flash backend tests on tiny random glm5-next GGUFs, CPU backend.
// Optional upstream comparison: GLM5_REF_GGUF + GLM5_REF_PREFIX
// (<prefix>.tokens.txt, .prefill.f32, .decode.f32).

#include "glm5_backend.h"
#include "glm5_graph.h"
#include "glm5_model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace luce::common;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
            return;                                                              \
        }                                                                        \
    } while (0)

struct TinyConfig {
    int n_layer = 4;
    int n_embd = 64;
    int n_head = 2;
    int kda_head = 32;
    int d_conv = 4;
    int q_lora = 32;
    int kv_lora = 64;
    int qk_head = 32;
    int v_head = 32;
    int idx_heads = 2;
    int idx_dim = 16;
    int top_k = 8;
    int kpool = 4;
    int n_ff = 64;
    int n_dense = 1;
    int n_expert = 4;
    int n_used = 2;
    int n_ff_exp = 16;
    int n_vocab = 256;
    uint32_t seed = 7;
};

std::string temp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/test_glm5_" + tag + "_" +
           std::to_string((long) std::random_device{}()) + ".gguf";
}

// Write the tiny model; same seed -> same weights, whatever top_k is.
bool write_tiny_glm5(const std::string & path, const TinyConfig & c) {
    const char * A = "glm5-next";
    auto k = [&](const char * s) { return std::string(A) + "." + s; };
    const int hc = 4, hc_mix = (2 + hc) * hc;
    const int d_inner = c.n_head * c.kda_head;

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", A);
    gguf_set_val_u32(g, k("block_count").c_str(), c.n_layer);
    gguf_set_val_u32(g, k("context_length").c_str(), 4096);
    gguf_set_val_u32(g, k("embedding_length").c_str(), c.n_embd);
    gguf_set_val_u32(g, k("feed_forward_length").c_str(), c.n_ff);
    gguf_set_val_u32(g, k("attention.head_count").c_str(), c.n_head);
    std::vector<int32_t> head_kv(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) head_kv[i] = (i % 4) == 3 ? 1 : 0;
    gguf_set_arr_data(g, k("attention.head_count_kv").c_str(), GGUF_TYPE_INT32, head_kv.data(), head_kv.size());
    gguf_set_val_f32(g, k("attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_f32(g, k("attention.layer_norm_epsilon").c_str(), 1e-6f);
    gguf_set_val_u32(g, k("expert_count").c_str(), c.n_expert);
    gguf_set_val_u32(g, k("expert_used_count").c_str(), c.n_used);
    gguf_set_val_u32(g, k("expert_group_count").c_str(), 1);
    gguf_set_val_u32(g, k("expert_gating_func").c_str(), 2);
    gguf_set_val_u32(g, k("ssm.conv_kernel").c_str(), c.d_conv);
    gguf_set_val_u32(g, k("kda.head_dim").c_str(), c.kda_head);
    gguf_set_val_f32(g, k("kda.gate_lower_bound").c_str(), -5.0f);
    gguf_set_val_u32(g, k("attention.q_lora_rank").c_str(), c.q_lora);
    gguf_set_val_u32(g, k("attention.kv_lora_rank").c_str(), c.kv_lora);
    gguf_set_val_u32(g, k("rope.dimension_count").c_str(), 0);
    gguf_set_val_u32(g, k("attention.key_length_mla").c_str(), c.qk_head);
    gguf_set_val_u32(g, k("attention.value_length_mla").c_str(), c.v_head);
    gguf_set_val_u32(g, k("attention.indexer.head_count").c_str(), c.idx_heads);
    gguf_set_val_u32(g, k("attention.indexer.key_length").c_str(), c.idx_dim);
    gguf_set_val_u32(g, k("attention.indexer.top_k").c_str(), c.top_k);
    gguf_set_val_u32(g, k("attention.indexer.kpool").c_str(), c.kpool);
    gguf_set_val_bool(g, k("attention.indexer.kpool_select_tail").c_str(), true);
    gguf_set_val_u32(g, k("hyper_connection.count").c_str(), hc);
    gguf_set_val_u32(g, k("hyper_connection.sinkhorn_iterations").c_str(), 20);
    gguf_set_val_f32(g, k("hyper_connection.epsilon").c_str(), 1e-6f);
    gguf_set_val_u32(g, k("leading_dense_block_count").c_str(), c.n_dense);
    gguf_set_val_u32(g, k("expert_feed_forward_length").c_str(), c.n_ff_exp);
    gguf_set_val_u32(g, k("expert_shared_count").c_str(), 1);
    gguf_set_val_f32(g, k("expert_weights_scale").c_str(), 2.5f);
    gguf_set_val_bool(g, k("expert_weights_norm").c_str(), true);
    std::vector<float> clamp(c.n_layer, 10.0f);
    gguf_set_arr_data(g, k("swiglu_clamp_exp").c_str(), GGUF_TYPE_FLOAT32, clamp.data(), clamp.size());
    gguf_set_arr_data(g, k("swiglu_clamp_shexp").c_str(), GGUF_TYPE_FLOAT32, clamp.data(), clamp.size());
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", c.n_vocab - 1);

    // Count the tensors to size the data context.
    struct T { std::string name; std::vector<int64_t> ne; float std; float fill; bool neg_exp; };
    std::vector<T> ts;
    auto add = [&](std::string name, std::vector<int64_t> ne, float std, float fill = NAN) {
        ts.push_back({std::move(name), std::move(ne), std, fill, false});
    };
    add("token_embd.weight", {c.n_embd, c.n_vocab}, 1.0f);
    add("output_norm.weight", {c.n_embd}, 0, 1.0f);
    add("output.weight", {c.n_embd, c.n_vocab}, 0.1f);
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add(p + "attn_norm.weight", {c.n_embd}, 0, 1.0f);
        add(p + "ffn_norm.weight", {c.n_embd}, 0, 1.0f);
        for (const char * s : {"attn", "ffn"}) {
            add(p + "hc_" + s + "_fn.weight", {hc * c.n_embd, hc_mix}, 0.1f);
            add(p + "hc_" + s + "_base.weight", {hc_mix}, 0.5f);
            add(p + "hc_" + s + "_scale.weight", {3}, 0.3f);
        }
        if (head_kv[il] == 0) {
            for (const char * x : {"q", "k", "v"}) {
                add(p + "ssm_conv1d_" + x + ".weight", {c.d_conv, 1, d_inner}, 0.3f);
                add(p + "attn_" + x + ".weight", {c.n_embd, d_inner}, 0.1f);
            }
            add(p + "ssm_f_a.weight", {c.n_embd, c.kda_head}, 0.1f);
            add(p + "ssm_f_b.weight", {c.kda_head, d_inner}, 0.1f);
            add(p + "ssm_beta.weight", {c.n_embd, c.n_head}, 0.1f);
            ts.push_back({p + "ssm_a", {c.n_head}, 1.0f, NAN, true});
            add(p + "ssm_dt.bias", {d_inner}, 0.5f);
            add(p + "ssm_g_a.weight", {c.n_embd, c.kda_head}, 0.1f);
            add(p + "ssm_g_b.weight", {c.kda_head, d_inner}, 0.1f);
            add(p + "ssm_norm.weight", {c.kda_head}, 0, 1.0f);
            add(p + "attn_output.weight", {d_inner, c.n_embd}, 0.1f);
        } else {
            add(p + "attn_q_a_norm.weight", {c.q_lora}, 0, 1.0f);
            add(p + "attn_kv_a_norm.weight", {c.kv_lora}, 0, 1.0f);
            add(p + "attn_q_a.weight", {c.n_embd, c.q_lora}, 0.1f);
            add(p + "attn_q_b.weight", {c.q_lora, c.n_head * c.qk_head}, 0.1f);
            add(p + "attn_kv_a_mqa.weight", {c.n_embd, c.kv_lora}, 0.1f);
            add(p + "attn_k_b.weight", {c.qk_head, c.kv_lora, c.n_head}, 0.1f);
            add(p + "attn_v_b.weight", {c.kv_lora, c.v_head, c.n_head}, 0.1f);
            add(p + "attn_output.weight", {c.n_head * c.v_head, c.n_embd}, 0.1f);
            add(p + "indexer.k_norm.weight", {c.idx_dim}, 0, 1.0f);
            add(p + "indexer.k_norm.bias", {c.idx_dim}, 0.05f);
            add(p + "indexer.proj.weight", {c.n_embd, c.idx_heads}, 0.1f);
            add(p + "indexer.attn_k.weight", {c.n_embd, c.idx_dim}, 0.2f);
            add(p + "indexer.attn_q_b.weight", {c.q_lora, c.idx_heads * c.idx_dim}, 0.2f);
            add(p + "indexer_compressor_gate.weight", {c.n_embd, c.idx_dim}, 0.1f);
            add(p + "indexer_compressor_ape.weight", {c.idx_dim, c.kpool}, 0.5f);
        }
        if (il < c.n_dense) {
            add(p + "ffn_gate.weight", {c.n_embd, c.n_ff}, 0.1f);
            add(p + "ffn_up.weight", {c.n_embd, c.n_ff}, 0.1f);
            add(p + "ffn_down.weight", {c.n_ff, c.n_embd}, 0.1f);
        } else {
            add(p + "ffn_gate_inp.weight", {c.n_embd, c.n_expert}, 0.2f);
            add(p + "exp_probs_b.bias", {c.n_expert}, 0.1f);
            add(p + "ffn_gate_exps.weight", {c.n_embd, c.n_ff_exp, c.n_expert}, 0.1f);
            add(p + "ffn_up_exps.weight", {c.n_embd, c.n_ff_exp, c.n_expert}, 0.1f);
            add(p + "ffn_down_exps.weight", {c.n_ff_exp, c.n_embd, c.n_expert}, 0.1f);
            add(p + "ffn_gate_shexp.weight", {c.n_embd, c.n_ff_exp}, 0.1f);
            add(p + "ffn_up_shexp.weight", {c.n_embd, c.n_ff_exp}, 0.1f);
            add(p + "ffn_down_shexp.weight", {c.n_ff_exp, c.n_embd}, 0.1f);
        }
    }

    size_t bytes = 0;
    for (const T & t : ts) {
        int64_t n = 1;
        for (int64_t d : t.ne) n *= d;
        bytes += (size_t) n * sizeof(float) + ggml_tensor_overhead() + 64;
    }
    ggml_init_params ip{};
    ip.mem_size = bytes + 1024 * 1024;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(c.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (const T & t : ts) {
        ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) t.ne.size(), t.ne.data());
        ggml_set_name(x, t.name.c_str());
        float * d = static_cast<float *>(x->data);
        for (int64_t i = 0; i < ggml_nelements(x); ++i) {
            if (t.neg_exp) d[i] = -std::exp(uni(rng));
            else if (!std::isnan(t.fill)) d[i] = t.fill;
            else d[i] = normal(rng) * t.std;
        }
        gguf_add_tensor(g, x);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

struct TinyModel {
    std::string path;
    explicit TinyModel(const TinyConfig & c, const char * tag) : path(temp_path(tag)) {
        if (!write_tiny_glm5(path, c)) path.clear();
    }
    ~TinyModel() {
        if (!path.empty()) std::remove(path.c_str());
    }
};

std::unique_ptr<Glm5Backend> open_cpu(const std::string & path, int max_ctx = 256, int chunk = 512) {
    Glm5BackendConfig cfg;
    cfg.model_path = path;
    cfg.cpu_only = true;
    cfg.device.max_ctx = max_ctx;
    cfg.chunk = chunk;
    auto b = std::make_unique<Glm5Backend>(cfg);
    if (!b->init()) return nullptr;
    return b;
}

std::vector<int32_t> test_tokens(int n, int n_vocab, uint32_t seed = 11) {
    std::mt19937 rng(seed);
    std::vector<int32_t> t(n);
    for (int & x : t) x = (int32_t) (rng() % (uint32_t) (n_vocab - 1));
    return t;
}

float max_abs_diff(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Logits of every position, one forward step per token.
bool incremental_logits(Glm5Backend & b, const std::vector<int32_t> & toks, std::vector<float> & out) {
    b.reset_sequence();
    out.clear();
    for (size_t i = 0; i < toks.size(); ++i) {
        std::vector<float> l;
        if (!b.forward(&toks[i], 1, (int) i, false, l)) return false;
        out.insert(out.end(), l.begin(), l.end());
    }
    return true;
}

// ── tests ────────────────────────────────────────────────────────────────

void test_host_dsa_inputs() {
    // Positions 5..8, kpool 4: visible pools 1,1,2,2; tail sizes 2,3,0,1.
    Glm5DsaHostInputs in;
    const int kv_pad = 256, n = 4, n_pool = 9 / 4;
    glm5_dsa_host_inputs(5, n, kv_pad, n_pool, 4, true, true, in);
    const int n_slot = kv_pad / 4;
    const int vis[4] = {1, 1, 2, 2};
    const int tail[4] = {2, 3, 0, 1};
    for (int t = 0; t < n; ++t) {
        const int p = 5 + t;
        int nv = 0, nt = 0;
        for (int s = 0; s < n_slot; ++s) nv += in.pool_vis[t * n_slot + s] > 0.5f;
        for (int c = 0; c < kv_pad; ++c) {
            if (in.tail[t * kv_pad + c] > 0.5f) {
                ++nt;
                CHECK(c <= p && c > p - 4 && c / 4 == p / 4);
            }
        }
        CHECK(nv == vis[t]);
        CHECK(nt == tail[t]);
        for (int q = 0; q < n_pool; ++q) {
            CHECK((in.pool_bias[t * n_pool + q] == 0.0f) == (q < vis[t]));
        }
        // the causal mask opens exactly [0, p]
        for (int c = 0; c < kv_pad; ++c) {
            const float m = ggml_fp16_to_fp32(in.causal_mask[t * kv_pad + c]);
            CHECK((m == 0.0f) == (c <= p));
        }
    }
}

void test_batch_vs_incremental() {
    TinyConfig c;
    TinyModel m(c, "batch");
    CHECK(!m.path.empty());
    auto b = open_cpu(m.path);
    CHECK(b != nullptr);
    CHECK(b->weights().n_mla_layers() == 1 && b->weights().n_kda_layers() == 3);

    const auto toks = test_tokens(24, c.n_vocab);   // 6 pools > top_k/kpool = 2: sparse
    CHECK(glm5_step_is_sparse(b->weights(), (int) toks.size()));
    std::vector<float> batch, inc;
    b->reset_sequence();
    CHECK(b->forward(toks.data(), (int) toks.size(), 0, true, batch));
    CHECK(incremental_logits(*b, toks, inc));
    CHECK(batch.size() == inc.size());
    const float d = max_abs_diff(batch.data(), inc.data(), batch.size());
    std::printf("  batch vs incremental max|diff| = %.3e\n", d);
    CHECK(d < 1e-3f);
}

void test_chunked_prefill() {
    TinyConfig c;
    TinyModel m(c, "chunk");
    CHECK(!m.path.empty());
    auto b = open_cpu(m.path);
    CHECK(b != nullptr);
    const auto toks = test_tokens(24, c.n_vocab, 3);
    std::vector<float> whole, part;
    b->reset_sequence();
    CHECK(b->forward(toks.data(), 24, 0, false, whole));
    b->reset_sequence();
    int pos = 0;
    for (int n : {5, 7, 12}) {
        CHECK(b->forward(toks.data() + pos, n, pos, false, part));
        pos += n;
    }
    const float d = max_abs_diff(whole.data(), part.data(), whole.size());
    std::printf("  one step vs 5+7+12 max|diff| = %.3e\n", d);
    CHECK(d < 1e-3f);
}

void test_dense_until_top_k() {
    TinyConfig sparse_cfg;              // top_k 8: 2 pools + a 3-token tail = 11 cells
    TinyConfig dense_cfg;
    dense_cfg.top_k = 4096;
    TinyModel ms(sparse_cfg, "sparse"), md(dense_cfg, "dense");
    CHECK(!ms.path.empty() && !md.path.empty());
    auto bs = open_cpu(ms.path), bd = open_cpu(md.path);
    CHECK(bs && bd);
    const auto toks = test_tokens(20, sparse_cfg.n_vocab, 5);
    std::vector<float> ls, ld;
    CHECK(incremental_logits(*bs, toks, ls));
    CHECK(incremental_logits(*bd, toks, ld));
    const size_t V = (size_t) sparse_cfg.n_vocab;
    const int n_sel = 2 * 4 + 3;
    float early = 0.0f, late = 0.0f;
    for (size_t p = 0; p < toks.size(); ++p) {
        const float d = max_abs_diff(&ls[p * V], &ld[p * V], V);
        if ((int) p < n_sel) early = std::max(early, d);
        else late = std::max(late, d);
    }
    std::printf("  sparse vs dense: positions < %d max|diff| = %.3e, later = %.3e\n", n_sel, early, late);
    CHECK(early < 1e-5f);
    CHECK(late > 1e-4f);
}

void test_snapshot_restore() {
    TinyConfig c;
    TinyModel m(c, "snap");
    CHECK(!m.path.empty());
    auto b = open_cpu(m.path, 256, 8);
    CHECK(b != nullptr);

    GenerateRequest req;
    req.prompt = test_tokens(18, c.n_vocab, 9);
    req.n_gen = 6;
    req.ignore_eos = true;
    req.snap_slot = 2;
    req.snap_pos = 13;   // not a chunk boundary
    DaemonIO io;
    GenerateResult cold = b->generate_impl(req, io);
    CHECK(cold.ok());
    CHECK(cold.tokens.size() == 6);
    CHECK(b->snapshot_used(2) && b->snapshot_cur_pos(2) == 13);

    GenerateRequest again = req;
    again.snap_slot = -1;
    again.snap_pos = -1;
    GenerateResult warm = b->restore_and_generate_impl(2, again, io);
    CHECK(warm.ok());
    CHECK(warm.restored_prefix_tokens == 13);
    CHECK(warm.tokens == cold.tokens);

    // A prompt of exactly the snapshot length uses its stored logits.
    GenerateRequest exact = again;
    exact.prompt.resize(13);
    GenerateResult a = b->restore_and_generate_impl(2, exact, io);
    GenerateResult z = b->generate_impl(exact, io);
    CHECK(a.ok() && z.ok());
    CHECK(a.tokens == z.tokens);
}

void test_layer_split() {
    // Layers, embedding and head land on their devices; bad placements fail.
    TinyConfig c;
    TinyModel m(c, "split");
    CHECK(!m.path.empty());
    std::vector<ggml_backend_t> one = {ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> two = {ggml_backend_cpu_init(), ggml_backend_cpu_init()};
    Glm5Weights w1, w2;
    CHECK(load_glm5_model(m.path, one, {0, 0, 0, 0}, w1));
    CHECK(load_glm5_model(m.path, two, {0, 0, 1, 1}, w2));
    CHECK(w2.layers[1].device == 0 && w2.layers[2].device == 1);
    CHECK(w2.embd_device == 0 && w2.head_device == 1);
    CHECK(w2.bufs[0] && w2.bufs[1]);
    // a mismatched placement is rejected
    Glm5Weights w3;
    CHECK(!load_glm5_model(m.path, two, {0, 1}, w3));
    free_glm5_model(w1);
    free_glm5_model(w2);
    for (auto * x : one) ggml_backend_free(x);
    for (auto * x : two) ggml_backend_free(x);
}

bool read_f32(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto n = (size_t) f.tellg();
    out.resize(n / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), (std::streamsize) n);
    return (bool) f;
}

void test_reference() {
    const char * gguf = std::getenv("GLM5_REF_GGUF");
    const char * prefix = std::getenv("GLM5_REF_PREFIX");
    if (!gguf || !prefix) {
        std::printf("  skipped (set GLM5_REF_GGUF and GLM5_REF_PREFIX)\n");
        return;
    }
    std::vector<int32_t> toks;
    {
        std::ifstream tf(std::string(prefix) + ".tokens.txt");
        for (int32_t t; tf >> t;) toks.push_back(t);
    }
    std::vector<float> ref_prefill, ref_decode;
    CHECK(!toks.empty());
    CHECK(read_f32(std::string(prefix) + ".prefill.f32", ref_prefill));
    CHECK(read_f32(std::string(prefix) + ".decode.f32", ref_decode));

    Glm5BackendConfig cfg;
    cfg.model_path = gguf;
    // GLM5_REF_DEVICE: "cpu" (default), a GPU index, or a list like "0,1" to split layers.
    const char * dev = std::getenv("GLM5_REF_DEVICE");
    cfg.cpu_only = !dev || std::string(dev) == "cpu";
    if (!cfg.cpu_only) {
        const std::string list = dev;
        if (list.find(',') == std::string::npos) {
            cfg.device.gpu = std::atoi(dev);
        } else {
            for (size_t at = 0; at <= list.size();) {
                const size_t end = std::min(list.find(',', at), list.size());
                cfg.device.layer_split_gpus.push_back(std::atoi(list.substr(at, end - at).c_str()));
                at = end + 1;
            }
        }
    }
    cfg.device.max_ctx = (int) toks.size() + 16;
    if (const char * th = std::getenv("GLM5_CPU_THREADS")) cfg.cpu_threads = std::atoi(th);
    Glm5Backend b(cfg);
    CHECK(b.init());
    const size_t V = (size_t) b.weights().n_vocab;
    CHECK(ref_prefill.size() == toks.size() * V);

    std::vector<float> batch, inc;
    b.reset_sequence();
    CHECK(b.forward(toks.data(), (int) toks.size(), 0, true, batch));
    CHECK(incremental_logits(b, toks, inc));

    // Near-tie DSA selections may differ across devices; such rows must stay rare.
    auto report = [&](const char * what, const std::vector<float> & mine, const std::vector<float> & ref) {
        float worst = 0.0f, scale = 0.0f;
        for (float r : ref) scale = std::max(scale, std::fabs(r));
        int off = 0, argmax_miss = 0;
        for (size_t p = 0; p < toks.size(); ++p) {
            const float * a = &mine[p * V];
            const float * r = &ref[p * V];
            const float d = max_abs_diff(a, r, V);
            if (d > 1e-2f * scale) {
                ++off;
                argmax_miss += (std::max_element(a, a + V) - a) != (std::max_element(r, r + V) - r);
                continue;
            }
            worst = std::max(worst, d);
        }
        std::printf("  %-22s max|diff| = %.3e (max|ref| = %.2f) on %zu/%zu rows; "
                    "%d row(s) off (selection near-ties), %d of them change the argmax\n",
                    what, worst, scale, toks.size() - off, toks.size(), off, argmax_miss);
        return off;
    };
    if (const char * dump = std::getenv("GLM5_DUMP_PREFIX")) {   // our logits, for offline analysis
        std::ofstream(std::string(dump) + ".prefill.f32", std::ios::binary)
            .write(reinterpret_cast<const char *>(batch.data()), (std::streamsize) (batch.size() * sizeof(float)));
        std::ofstream(std::string(dump) + ".decode.f32", std::ios::binary)
            .write(reinterpret_cast<const char *>(inc.data()), (std::streamsize) (inc.size() * sizeof(float)));
    }
    const int allowed = std::max(1, (int) toks.size() / 100);
    const int off_prefill = report("prefill vs ref prefill", batch, ref_prefill);
    const int off_decode = report("decode  vs ref decode", inc, ref_decode);
    CHECK(off_prefill <= allowed);
    CHECK(off_decode <= allowed);
}

void run(const char * name, const std::function<void()> & fn) {
    const int before = g_failures;
    std::printf("[glm5] %s\n", name);
    fn();
    std::printf("[glm5] %s: %s\n", name, g_failures == before ? "ok" : "FAILED");
}

}  // namespace

int main() {
    run("host_dsa_inputs", test_host_dsa_inputs);
    run("layer_split", test_layer_split);
    run("batch_vs_incremental", test_batch_vs_incremental);
    run("chunked_prefill", test_chunked_prefill);
    run("dense_until_top_k", test_dense_until_top_k);
    run("snapshot_restore", test_snapshot_restore);
    run("reference", test_reference);
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all glm5 tests passed\n");
    return 0;
}
