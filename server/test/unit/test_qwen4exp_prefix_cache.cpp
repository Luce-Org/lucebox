// No arguments: CPU checks. MODEL [VERIFY_WIDTH [CHUNK [CTX]]]: GPU lifecycle.
#include "qwen4exp/qwen4exp_backend.h"
#include "qwen4exp/qwen4exp_chunk.h"
#include "qwen4exp/qwen4exp_graph.h"
#include "server/tokenizer.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

using namespace luce::common;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

static std::vector<char> bytes(ggml_tensor * t) {
    std::vector<char> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}

static void rollback_reset() {
    // Deterministically delay the real rollback's device-copy queue until a
    // backend barrier. Buffer memset remains synchronous on a separate stream.
    using Queue = std::vector<std::pair<const ggml_tensor *, ggml_tensor *>>;
    Queue pending;
    ggml_backend delayed{};
    delayed.context = &pending;
    delayed.iface.cpy_tensor_async = [](ggml_backend_t, ggml_backend_t dst, const ggml_tensor * a, ggml_tensor * b) {
        static_cast<Queue *>(dst->context)->emplace_back(a, b);
        return true;
    };
    delayed.iface.synchronize = [](ggml_backend_t b) {
        auto & queue = *static_cast<Queue *>(b->context);
        for (auto [src, dst] : queue) ggml_backend_tensor_copy(src, dst);
        queue.clear();
    };
    Qwen4ExpCache c;
    c.ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    auto tensor = [&]() { return ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 16); };
    c.ssm_state = {tensor()}; c.conv_state = {tensor()}; c.ple_conv_state = {tensor()};
    c.spec_ssm_rows.resize(1); c.spec_conv_rows.resize(1);
    c.spec_ssm_rows[0][0] = tensor(); c.spec_conv_rows[0][0] = tensor();
    c.spec_ple = c.spec_ple_rows[0] = tensor();
    c.buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_cpu_buffer_type());
    CHECK(c.buf);
    ggml_backend_buffer_clear(c.buf, 0x3f);
    c.spec_pos = 0; c.spec_tokens = 2;
    CHECK(qwen4exp_verify_rollback(&delayed, Qwen4ExpWeights{}, c, 0, 1));
    CHECK(pending.size() == 3); // the final rejected step is still in flight
    reset_qwen4exp_state(&delayed, c);
    ggml_backend_synchronize(&delayed); // any late rollback must not undo reset
    for (auto * t : {c.ssm_state[0], c.conv_state[0], c.ple_conv_state[0]}) {
        CHECK(bytes(t) == std::vector<char>(ggml_nbytes(t), 0));
    }
    free_qwen4exp_cache(c);
    std::puts("PASS pending MTP rollback completes before recurrent/conv/PLE reset");
}

// Test-only readbacks: production checkpoints remain on the device. Report
// every differing state family before stopping, rather than just token inequality.
namespace luce::common {
struct Qwen4ExpPrefixTest {
    static const char * family(const Qwen4ExpCache & c, ggml_tensor * view) {
        const auto * t = view->view_src;
        for (auto entry : {std::make_pair(&c.attn_k, "attention K"), {&c.attn_v, "attention V"},
                {&c.indexer_raw, "QSA raw"}, {&c.indexer_k, "QSA pooled"},
                {&c.ssm_state, "GDN"}, {&c.conv_state, "conv"}, {&c.ple_conv_state, "PLE conv"}}) {
            if (std::find(entry.first->begin(), entry.first->end(), t) != entry.first->end()) return entry.second;
        }
        return t == c.mtp_k ? "MTP K" : t == c.mtp_v ? "MTP V" : "MTP carry";
    }
    static bool equal(Qwen4ExpBackend & b, int expected, const char * label, int actual = -1) {
        const auto & s = b.snapshots_[expected];
        const auto & c = b.cache_;
        const auto & a = actual < 0 ? s : b.snapshots_[actual];
        const int pos = actual < 0 ? c.cur_pos : a.cur_pos;
        const int blocks = actual < 0 ? c.indexer_blocks : a.indexer_blocks;
        const int carry = actual < 0 ? c.mtp_prev_pos : a.mtp_prev_pos;
        const auto bucket = actual < 0 ? c.kv_bucket_base : a.kv_bucket_base;
        std::fprintf(stderr, "[prefix-state] %s pos=%d/%d blocks=%d/%d carry=%d/%d bucket=%lld/%lld live_spec=%d,%d\n",
            label, s.cur_pos, pos, s.indexer_blocks, blocks, s.mtp_prev_pos, carry,
            (long long) s.kv_bucket_base, (long long) bucket, c.spec_pos, c.spec_tokens);
        bool ok = pos == s.cur_pos && blocks == s.indexer_blocks && carry == s.mtp_prev_pos && bucket == s.kv_bucket_base;
        auto host = [&](const auto & lhs, const auto & rhs, const char * name) {
            const bool same = lhs.size() == rhs.size() &&
                (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(lhs[0])) == 0);
            std::fprintf(stderr, "[prefix-state] %s %s=%s\n", label, name, same ? "identical" : "DIFF");
            ok &= same;
        };
        host(s.ple_prev, actual < 0 ? c.ple_prev : a.ple_prev, "PLE history");
        host(s.tokens, actual < 0 ? b.tokens_ : a.tokens, "tokens");
        host(s.logits, actual < 0 ? b.logits_ : a.logits, "logits");
        CHECK(s.strips.size() == a.strips.size());
        std::map<std::string, size_t> checked;
        for (size_t i = 0; i < s.strips.size(); ++i) {
            const auto lhs = bytes(s.strips[i].second);
            const auto rhs = bytes(actual < 0 ? s.strips[i].first : a.strips[i].second);
            if (lhs == rhs) { checked[family(c, s.strips[i].first)] += lhs.size(); continue; }
            const size_t offset = std::mismatch(lhs.begin(), lhs.end(), rhs.begin(), rhs.end()).first - lhs.begin();
            std::fprintf(stderr, "[prefix-state] %s DIFF %s strip=%zu first_byte=%zu sizes=%zu/%zu\n",
                label, family(c, s.strips[i].first), i, offset, lhs.size(), rhs.size());
            ok = false;
        }
        for (const auto & entry : checked) std::fprintf(stderr, "[prefix-state] %s %s identical_bytes=%zu\n",
            label, entry.first.c_str(), entry.second);
        std::fprintf(stderr, "[prefix-state] %s %s\n", label, ok ? "PASS" : "FAIL");
        return ok;
    }

    // Only retain the already-read-back logits (about 7 MB for this fixture).
    // No device reads/barriers here: they would hide the rollback/reset race.
    static void rejection_hook(Qwen4ExpBackend & b, std::vector<float> * rows = nullptr) {
        b.decode_check_ = [rows](bool draft, std::vector<int32_t> & tokens, const std::vector<float> & logits) {
            if (draft) std::fill(tokens.begin(), tokens.end(), 99);
            else if (rows) rows->insert(rows->end(), logits.begin(), logits.end());
        };
    }
    static void unhook(Qwen4ExpBackend & b) { b.decode_check_ = {}; }
    // Split mode: the hot experts and their lookup tables are resident on the target.
    static bool hot_experts_resident(const Qwen4ExpBackend & b) {
        return b.weights_.hot && !b.weights_.hot_lut.empty();
    }
    static size_t limit(Qwen4ExpBackend & b, size_t bytes) {
        return std::exchange(b.snapshot_budget_, bytes);
    }
    static void plan(Qwen4ExpBackend & b, int ctx) {
        if (b.cache_.max_ctx != ctx) {
            // Before any request/snapshot: reuse loaded weights, not two caches.
            free_qwen4exp_cache(b.cache_);
            b.cache_ = {};
            b.cfg_.device.max_ctx = ctx;
            CHECK(create_qwen4exp_cache(b.backend_, b.weights_, ctx, b.cache_, true,
                b.cfg_.verify_width == 0 ? 3 : std::max(1, b.cfg_.verify_width - 1)));
            b.snapshot_budget_ = 3 * b.snapshot_bytes_estimate(ctx);
            b.chunk_ = qwen4exp_select_chunk(b.backend_, b.weights_, b.cache_, 1, 1, &b.snapshot_budget_);
        }
        std::printf("[prefix-policy] ctx=%d chunk=%d allowance=%zu mtp=%d\n",
            ctx, b.chunk_, b.snapshot_budget_, (int) qwen4exp_verify_supported(b.cache_));
        CHECK(b.chunk_ >= 4096);
    }

    static void budget() {
        Qwen4ExpBackend b({});
        b.backend_ = ggml_backend_cpu_init();
        CHECK(b.backend_);
        auto & c = b.cache_;
        c.ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        c.max_ctx = 8; c.cur_pos = 4;
        c.attn_k = {ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 8, 8, 2)};
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, b.backend_);
        CHECK(c.buf);
        b.tokens_ = {1, 2, 3, 4}; b.logits_ = {1}; b.weights_.n_vocab = 1;
        b.snapshot_budget_ = 2 * b.snapshot_bytes_estimate(c.cur_pos);
        CHECK(b.snapshot_save(0) && b.snapshot_save_deferred(1));
        CHECK(!b.snapshot_save(2)); // deferred copies already consume their allowance
        b.snapshot_flush_deferred();
        CHECK(b.snapshot_used(1) && !b.snapshot_save_deferred(2));
        CHECK(b.snapshot_save(0)); // replacement releases its own allowance
        b.snapshot_free(1);
        CHECK(b.snapshot_save_deferred(2));
        std::puts("PASS prefix allowance includes deferred copies and replacement");

        for (int i = 0; i < 3; ++i) b.snapshot_free(i);
        CHECK(b.snapshot_save(0));
        c.cur_pos = 6; b.tokens_ = {1, 2, 3, 4, 5, 6};
        b.snapshot_budget_ = b.snapshot_bytes_estimate(4) + b.snapshot_bytes_estimate(6);
        CHECK(b.snapshot_save_replacing(1, 0) && b.snapshot_used(0)); // keep both when they fit
        b.snapshot_free(1);
        b.snapshot_budget_ = b.snapshot_bytes_estimate(6);
        CHECK(b.snapshot_save_replacing(1, 0));
        CHECK(!b.snapshot_used(0) && b.snapshot_cur_pos(1) == 6);
        c.cur_pos = 7; b.tokens_.push_back(7);
        b.snapshot_budget_ = b.snapshot_bytes_estimate(7) - 1;
        CHECK(!b.snapshot_save_replacing(2, 1) && b.snapshot_used(1)); // successor alone cannot fit
        ++b.snapshot_budget_;
        b.tokens_[0] = 99;
        CHECK(!b.snapshot_save_replacing(2, 1) && b.snapshot_used(1)); // unrelated prefix
        b.tokens_[0] = 1;
        CHECK(b.snapshot_save_replacing(2, 1) && !b.snapshot_used(1));
        CHECK(b.snapshot_cur_pos(2) == 7);
        std::puts("PASS pressure replaces only a strict ancestor; keeps both when they fit");
    }

};
} // namespace luce::common

static bool same_tokens(const GenerateResult & warm, const GenerateResult & cold, const char * label) {
    std::fprintf(stderr, "[prefix-tokens] %s warm_ok=%d cold_ok=%d warm=%zu cold=%zu restored=%d\n",
        label, warm.ok(), cold.ok(), warm.tokens.size(), cold.tokens.size(), warm.restored_prefix_tokens);
    if (warm.tokens != cold.tokens) {
        for (const auto * r : {&warm, &cold}) {
            for (int32_t t : r->tokens) std::fprintf(stderr, "%d ", t);
            std::fputc('\n', stderr);
        }
    }
    return warm.ok() && cold.ok() && warm.tokens == cold.tokens;
}

static void copies(ggml_backend_t backend) {
    CHECK(backend);
    Qwen4ExpCache c;
    c.max_ctx = 63; // packed KV has a padded fourth key beyond the logical limit
    c.ctx = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
    auto kv = [&]() { return ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 8, 64, 2); };
    c.attn_k = {kv(), kv()}; c.attn_v = {kv(), kv()};
    c.mtp_k = kv(); c.mtp_v = kv();
    c.mtp_prev_hidden = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 16);
    for (int i = 0; i < 2; ++i) {
        c.indexer_raw.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 8, c.max_ctx));
        c.indexer_k.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 8, c.max_ctx / 4 + 1));
        c.ssm_state.push_back(ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 4, 4, 2));
        c.conv_state.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 3, 16));
        c.ple_conv_state.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 9, 16));
    }
    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    CHECK(c.buf);
    Qwen4ExpSnapshot s;
    for (int pos : {1, 7, 31, 63}) {
        unsigned char value = 1;
        for (auto * t = ggml_get_first_tensor(c.ctx); t; t = ggml_get_next_tensor(c.ctx, t)) {
            ggml_backend_tensor_memset(t, value++, 0, ggml_nbytes(t));
        }
        c.cur_pos = pos; c.indexer_blocks = pos / 4;
        c.mtp_prev_pos = pos - 1; c.kv_bucket_base = 512; c.ple_prev = {11, 12};
        CHECK(save_qwen4exp_snapshot(backend, c, s));
        size_t host = 0;
        CHECK(ggml_backend_buffer_get_size(s.buf) <= qwen4exp_snapshot_bytes(backend, c, pos, &host));
        CHECK(host >= ggml_get_mem_size(s.ctx) + s.strips.capacity() * sizeof(s.strips[0]));
        for (auto [live, copy] : s.strips) CHECK(bytes(live) == bytes(copy));
        // All states, including QSA raw incomplete blocks and MTP carry, must
        // survive a reset and an unrelated request overwriting the live buffer.
        reset_qwen4exp_state(backend, c);
        ggml_backend_buffer_clear(c.buf, 0xa5);
        restore_qwen4exp_snapshot(backend, s, c);
        CHECK(c.cur_pos == pos && c.indexer_blocks == pos / 4 && c.mtp_prev_pos == pos - 1);
        CHECK(c.kv_bucket_base == 512 && c.ple_prev == std::vector<int32_t>({11, 12}));
        CHECK(c.spec_tokens == 0 && c.spec_pos == -1);
        for (auto [live, copy] : s.strips) CHECK(bytes(live) == bytes(copy));
        // Every byte not serialized (padded KV, incomplete pooled blocks and
        // indexer scratch) must be zero, including the second head's suffix.
        for (auto * t = ggml_get_first_tensor(c.ctx); t; t = ggml_get_next_tensor(c.ctx, t)) {
            std::vector<char> expected(ggml_nbytes(t), 0);
            for (auto [live, copy] : s.strips) if (live->view_src == t) {
                const auto saved = bytes(copy);
                std::copy(saved.begin(), saved.end(), expected.begin() + live->view_offs);
            }
            CHECK(bytes(t) == expected);
        }
    }
    --c.mtp_prev_pos;
    CHECK(!save_qwen4exp_snapshot(backend, c, s)); // incomplete MTP is not publishable
    free_qwen4exp_snapshot(s); free_qwen4exp_snapshot(s);
    free_qwen4exp_cache(c);
    std::puts("PASS prefix snapshot bytes, metadata, overwrite/reset, MTP validity");
}

static void memory(ggml_backend_t backend) {
    // Declare production geometry without allocating its multi-GB payload.
    Qwen4ExpCache c;
    c.max_ctx = 131072;
    c.ctx = ggml_init({256 * ggml_tensor_overhead(), nullptr, true});
    auto kv = [&]() { return ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 256, c.max_ctx, 2); };
    for (int i = 0; i < 12; ++i) {
        c.attn_k.push_back(kv()); c.attn_v.push_back(kv());
        c.indexer_raw.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 128, c.max_ctx));
        c.indexer_k.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 128, c.max_ctx / 4 + 1));
    }
    for (int i = 0; i < 36; ++i) {
        c.ssm_state.push_back(ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 128, 48));
        c.conv_state.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 3, 10240));
    }
    c.ple_conv_state.push_back(ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 9, 10240));
    CHECK(qwen4exp_snapshot_bytes(backend, c, 65536) == 2231967744ULL);
    CHECK(qwen4exp_snapshot_bytes(backend, c, 131072) == 4345896960ULL);
    c.mtp_k = kv(); c.mtp_v = kv();
    c.mtp_prev_hidden = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 10240);
    CHECK(qwen4exp_snapshot_bytes(backend, c, 65536) == 2366224384ULL);
    CHECK(qwen4exp_snapshot_bytes(backend, c, 131072) == 4614371328ULL);
    std::puts("PASS device bytes (MTP): 64Ki=2366224384 128Ki=4614371328");
    free_qwen4exp_cache(c);
}

// expert_gpu >= 0 runs split mode (the routed experts on that GPU), with the hot experts of `placement` if given.
static void model(const char * path, int width, int chunk, int ctx, int expert_gpu, const char * placement) {
    Qwen4ExpBackendConfig cfg;
    cfg.model_path = path; cfg.verify_width = width; cfg.chunk = chunk;
    cfg.device.max_ctx = chunk == 0 && ctx == 262144 ? 131072 : ctx;
    if (expert_gpu >= 0) {
        DevicePlacement expert;
        expert.backend = cfg.device.backend;
        expert.gpu = expert_gpu;
        cfg.expert_device = expert;
        if (placement) cfg.expert_placement_path = placement;
    }
    Qwen4ExpBackend b(cfg);
    CHECK(b.init());
    const bool hot = expert_gpu >= 0 && placement;
    CHECK(Qwen4ExpPrefixTest::hot_experts_resident(b) == hot);
    if (chunk == 0 && ctx == 262144) {
        Qwen4ExpPrefixTest::plan(b, 131072);
        Qwen4ExpPrefixTest::plan(b, 262144);
    }
    Tokenizer tokenizer;
    CHECK(tokenizer.load_from_gguf(path));
    GenerateRequest r;
    std::string source = "<|im_start|>system\nReview this code when asked.\n";
    for (int i = 0; i < 400; ++i) source += "int increment(int value) { return value + 1; }\n";
    const auto tail = tokenizer.encode("\n<|im_end|>\n<|im_start|>user\nPrint integers 1 through 100 separated by spaces."
                                       "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    r.prompt = tokenizer.encode(source);
    CHECK(r.prompt.size() > 2309 && tail.size() < 100);
    r.prompt.resize(2309 - tail.size());
    r.prompt.insert(r.prompt.end(), tail.begin(), tail.end());
    r.n_gen = 32;
    r.restore_points = {1, 129, 257, 2051, 2052, 2200}; // singleton carry + QSA partial blocks
    r.snap_slot = 0; r.snap_pos = 2051;
    auto cold = b.generate(r, {});
    CHECK(cold.ok() && cold.tokens.size() > 3);
    CHECK(b.snapshot_cur_pos(0) == 2051);
    auto warm = b.restore_and_generate(0, r, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2051);
    // A restored request must capture a later cut, including inside its first
    // auto chunk, while the source snapshot remains resident (session 75e).
    r.snap_slot = 1; r.snap_pos = 2200;
    warm = b.restore_and_generate(0, r, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2051);
    CHECK(b.snapshot_cur_pos(0) == 2051 && b.snapshot_cur_pos(1) == 2200);
    warm = b.restore_and_generate(1, r, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2200);
    // Force the same continuation through the single-checkpoint budget path.
    const size_t allowance = Qwen4ExpPrefixTest::limit(b, b.snapshot_bytes_estimate(2200));
    b.snapshot_free(1);
    warm = b.restore_and_generate(0, r, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2051);
    CHECK(!b.snapshot_used(0) && b.snapshot_cur_pos(1) == 2200);
    warm = b.restore_and_generate(1, r, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2200);
    Qwen4ExpPrefixTest::limit(b, allowance);
    b.snapshot_free(1);
    r.snap_slot = 0; r.snap_pos = 2051;
    cold = b.generate(r, {}); // replenish the ancestor for the remaining lifecycle
    CHECK(cold.ok() && b.snapshot_cur_pos(0) == 2051);
    // A changed suffix uses the shallower checkpoint; changing the prefix or
    // shortening it must recompute, never truncate the recurrent state.
    r.prompt[2201] = 42;
    warm = b.restore_and_generate(0, r, {});
    cold = b.generate(r, {});
    CHECK(warm.ok() && cold.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 2051);
    r.prompt[13] = 42;
    warm = b.restore_and_generate(0, r, {});
    CHECK(warm.ok() && warm.restored_prefix_tokens == 0);
    cold = b.generate(r, {});
    CHECK(cold.ok() && warm.tokens == cold.tokens);
    auto shorter = r; shorter.prompt.resize(257); shorter.snap_slot = -1;
    warm = b.restore_and_generate(0, shorter, {});
    cold = b.generate(shorter, {});
    CHECK(warm.ok() && cold.ok() && warm.restored_prefix_tokens == 0 && warm.tokens == cold.tokens);

    // Cancel after capture, while the next chunk's host gather may be pending.
    r.snap_pos = 257; r.snap_slot = 1;
    DaemonIO cancel;
    int polls = 0;
    cancel.should_cancel = [&]() { return ++polls == 6; };
    auto stopped = b.generate(r, cancel);
    CHECK(stopped.error_code() == "cancelled" && b.snapshot_cur_pos(1) == 257);
    CHECK(!b.snapshot_save_deferred(2));
    warm = b.restore_and_generate(1, r, {});
    cold = b.generate(r, {});
    CHECK(warm.ok() && cold.ok() && warm.tokens == cold.tokens && warm.restored_prefix_tokens == 257);

    // Prefill-only saves maintain MTP even though they do not speculate.
    auto pre = r; pre.n_gen = 0; pre.snap_slot = 2; pre.snap_pos = (int) pre.prompt.size();
    CHECK(b.generate(pre, {}).ok());
    CHECK(b.snapshot_cur_pos(2) == (int) pre.prompt.size());
    r.snap_slot = -1;
    warm = b.restore_and_generate(2, r, {});
    cold = b.generate(r, {});
    CHECK(warm.ok() && cold.ok() && warm.tokens == cold.tokens);
    // Deferred prefill: zero-copy continuation, then cold replay at its cut.
    CHECK(b.generate(pre, {}).ok() && b.snapshot_save_deferred(3));
    auto append = r;
    append.restore_points.push_back((int) r.prompt.size());
    append.prompt.insert(append.prompt.end(), 73, 42);
    CHECK(Qwen4ExpPrefixTest::equal(b, 2, "deferred publication"));

    // Isolate the first difference before decode can overwrite its evidence.
    // Cold captures the same 2309-token prefix inside the longer request.
    auto append_pre = append;
    append_pre.n_gen = 0; append_pre.snap_slot = 5; append_pre.snap_pos = (int) r.prompt.size();
    CHECK(b.restore_and_generate(3, append_pre, {}).ok() && b.snapshot_save(6));
    CHECK(b.generate(append_pre, {}).ok());
    const bool prefix_same = Qwen4ExpPrefixTest::equal(b, 2, "cold prefix", 5);
    const bool suffix_same = Qwen4ExpPrefixTest::equal(b, 6, "resident suffix vs cold");
    CHECK(b.restore_and_generate(2, append_pre, {}).ok());
    const bool copied_same = Qwen4ExpPrefixTest::equal(b, 6, "resident suffix vs copied");
    CHECK(prefix_same && suffix_same && copied_same);
    b.snapshot_free(5); b.snapshot_free(6);

    // Retain the original session-75 gate without diagnostic readbacks.
    CHECK(b.generate(pre, {}).ok() && b.snapshot_save_deferred(3));
    warm = b.restore_and_generate(3, append, {});
    CHECK(!b.snapshot_used(3));
    cold = b.generate(append, {});
    const bool untraced_same = same_tokens(warm, cold, "resident continuation");

    auto plain = append; plain.force_ar_decode = true;
    const auto ar = b.generate(plain, {});
    const bool natural_ar_same = same_tokens(warm, ar, "resident MTP vs AR");

    // Every proposed 99 is rejected against forced outputs 42..48. Compare
    // every retained logit row with cold MTP and AR; output substitution alone
    // must not conceal a bad verify/rollback. No readback waits for rollback.
    std::vector<float> warm_rows, cold_rows, ar_rows;
    auto reject = append; reject.n_gen = 8;
    reject.budget_hook.hard_limit_remaining = reject.n_gen;
    for (int i = 0; i < reject.n_gen; ++i) reject.budget_hook.close_token_ids.push_back(42 + i % 7);
    CHECK(b.generate(pre, {}).ok() && b.snapshot_save_deferred(3));
    Qwen4ExpPrefixTest::rejection_hook(b, &warm_rows);
    const auto rejected_warm = b.restore_and_generate(3, reject, {});
    Qwen4ExpPrefixTest::rejection_hook(b, &cold_rows);
    const auto rejected_cold = b.generate(reject, {});
    Qwen4ExpPrefixTest::rejection_hook(b, &ar_rows);
    reject.force_ar_decode = true;
    const auto rejected_ar = b.generate(reject, {});
    Qwen4ExpPrefixTest::unhook(b);
    CHECK(same_tokens(rejected_warm, rejected_cold, "forced reject resident"));
    CHECK(same_tokens(rejected_warm, rejected_ar, "forced reject AR"));
    CHECK(width == 1 || (rejected_warm.spec_decode_ran && rejected_warm.accept_rate == 0.0f));
    CHECK(!warm_rows.empty() && warm_rows.size() == cold_rows.size() && warm_rows.size() == ar_rows.size());
    CHECK(std::memcmp(warm_rows.data(), cold_rows.data(), warm_rows.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(warm_rows.data(), ar_rows.data(), warm_rows.size() * sizeof(float)) == 0);
    CHECK(untraced_same && natural_ar_same);

    // End on the first rejected verify row: no later AR/MTP graph drains the
    // rollback queue. Immediately reset and compare all prefix state bytes.
    auto eos = reject; eos.force_ar_decode = false;
    eos.budget_hook.close_token_ids = {42, tokenizer.eos_id()};
    auto restart = pre; restart.snap_slot = -1;
    CHECK(b.generate(pre, {}).ok() && b.snapshot_save_deferred(3));
    Qwen4ExpPrefixTest::rejection_hook(b);
    const auto ended = b.restore_and_generate(3, eos, {});
    Qwen4ExpPrefixTest::unhook(b);
    const auto restarted = b.generate(restart, {}); // no device readback between rollback and reset
    CHECK(ended.ok() && ended.tokens == eos.budget_hook.close_token_ids && restarted.ok());
    CHECK(width == 1 || (ended.spec_decode_ran && ended.accept_rate == 0.0f));
    CHECK(Qwen4ExpPrefixTest::equal(b, 2, "reset immediately after rejected EOS"));
    // Switching conversation materializes a deferred checkpoint before reset.
    CHECK(b.generate(pre, {}).ok() && b.snapshot_save_deferred(3));
    CHECK(b.generate(shorter, {}).ok() && b.snapshot_used(3));
    warm = b.restore_and_generate(3, append, {});
    CHECK(warm.ok() && warm.tokens == cold.tokens);

    // Force substitutions to exercise partial-accept rollback. The continuation
    // is unforced; compare it to cold T=1 replay of the generated prefix.
    auto gen = r;
    gen.budget_hook.hard_limit_remaining = gen.n_gen;
    for (int i = 0; i < gen.n_gen; ++i) gen.budget_hook.close_token_ids.push_back(42 + i % 7);
    auto turn = b.generate(gen, {});
    CHECK(turn.ok() && (int) turn.tokens.size() == gen.n_gen);
    if (width != 1) CHECK(turn.spec_decode_ran && turn.accept_rate < 1.0f);
    CHECK(b.snapshot_save_deferred(4));
    auto next = r;
    for (int p = (int) r.prompt.size(); p <= (int) r.prompt.size() + gen.n_gen; ++p) next.restore_points.push_back(p);
    next.prompt.insert(next.prompt.end(), turn.tokens.begin(), turn.tokens.end());
    next.prompt.insert(next.prompt.end(), 137, 43);
    warm = b.restore_and_generate(4, next, {});
    CHECK(warm.restored_prefix_tokens == (int) r.prompt.size() + gen.n_gen - 1);
    cold = b.generate(next, {});
    CHECK(warm.ok() && cold.ok() && warm.tokens == cold.tokens);
    // Decode cancellation cannot be exposed as a resident checkpoint.
    auto interrupted = r;
    interrupted.on_token = [](int32_t) { return false; };
    CHECK(b.generate(interrupted, {}).error_code() == "cancelled");
    CHECK(!b.snapshot_save(5));
    const auto before_park = b.generate(r, {});
    CHECK(before_park.ok());
    for (int i = 0; i < 6; ++i) { b.snapshot_free(i); CHECK(!b.snapshot_used(i)); }
    CHECK(b.park(ParkTarget::All));
    CHECK(b.snapshot_bytes_estimate(65536) == 0);
    CHECK(b.unpark(ParkTarget::All));
    // An unpark brings the target back as init did, hot experts included.
    CHECK(Qwen4ExpPrefixTest::hot_experts_resident(b) == hot);
    const auto after_unpark = b.generate(r, {});
    CHECK(after_unpark.ok() && after_unpark.tokens == before_park.tokens);
    std::printf("PASS backend prefix lifecycle, partial match, cancel, resident, MTP rollback width=%d%s\n", width,
                hot ? ", split with hot experts across park/unpark" : expert_gpu >= 0 ? ", split" : "");
}

int main(int argc, char ** argv) {
    rollback_reset();
    Qwen4ExpPrefixTest::budget();
    auto backend = argc > 1 ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    CHECK(backend);
    copies(backend);
    memory(backend);
    ggml_backend_free(backend);
    // test_qwen4exp_prefix_cache MODEL [width] [chunk] [ctx] [expert_gpu] [placement.csv]
    if (argc > 1) model(argv[1], argc > 2 ? std::atoi(argv[2]) : 0,
        argc > 3 ? std::atoi(argv[3]) : 0, argc > 4 ? std::atoi(argv[4]) : 8192,
        argc > 5 ? std::atoi(argv[5]) : -1, argc > 6 ? argv[6] : nullptr);
}
