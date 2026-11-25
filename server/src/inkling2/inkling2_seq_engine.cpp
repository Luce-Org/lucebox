// Inkling2SeqEngine & Inkling2SlotKv — Unified concurrent N-slot SeqEngine
// (Batched Multi-Slot 2DSD + 1D AR/DSpark + SlotKv v2 + Inkling2SlotStateAdapter + 3-Tier KVFlash 2.0).

#include "inkling2_seq_engine.h"
#include "inkling2_2dsd_draft.h"
#include "inkling2_2dsd_graph.h"
#include "inkling2_graph.h"
#include "inkling2_kvflash.h"

#include "common/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace luce::common {
namespace {

static bool sampler_uses_fast_gpu(const SamplerCfg & s) {
    return s.needs_logit_processing() && s.temp > 0.0f &&
           s.rep_pen <= 1.0f && s.freq_pen == 0.0f &&
           s.pres_pen == 0.0f && s.constraint == nullptr;
}

static bool try_build_exact_cands_row(const int32_t * rc,
                                      const SamplerCfg & sampler,
                                      std::vector<std::pair<int32_t, float>> & out_probs,
                                      double & out_cum) {
    const int cnt = rc[0];
    if (cnt <= 0 || cnt > 256) return false;
    float gmax = 0.0f, Z = 0.0f;
    std::memcpy(&gmax, &rc[1], sizeof(float));
    std::memcpy(&Z,    &rc[2], sizeof(float));
    if (!(Z > 0.0f) || !std::isfinite(Z)) return false;
    struct Cand { float v; int32_t id; };
    Cand cands[256];
    for (int j = 0; j < cnt; ++j) {
        std::memcpy(&cands[j].v, &rc[4 + 2 * j], sizeof(float));
        cands[j].id = rc[4 + 2 * j + 1];
        if (!std::isfinite(cands[j].v)) return false;
    }
    std::sort(cands, cands + cnt, [](const Cand & a, const Cand & b) {
        return a.v > b.v || (a.v == b.v && a.id < b.id);
    });
    const int k_lim = (sampler.top_k > 0 && sampler.top_k < cnt) ? sampler.top_k : cnt;
    const float inv_t = 1.0f / std::max(sampler.temp, 1e-6f);
    out_probs.clear();
    out_probs.reserve((size_t) k_lim);
    out_cum = 0.0;
    for (int j = 0; j < k_lim; ++j) {
        const float p = std::exp((cands[j].v - gmax) * inv_t) / Z;
        out_probs.push_back({ cands[j].id, p });
        out_cum += p;
        if (sampler.top_p > 0.0f && sampler.top_p < 1.0f && out_cum >= sampler.top_p) {
            break;
        }
    }
    const bool exact_ok =
        (sampler.top_k > 0 && cnt >= sampler.top_k) ||
        (sampler.top_p > 0.0f && sampler.top_p < 1.0f && out_cum >= sampler.top_p) ||
        (out_cum >= 0.9995);
    return exact_ok && !out_probs.empty() && out_cum > 0.0;
}

static int resolve_draft_window() {
    int win = 2048;
    if (const char * ew = std::getenv("LUCE_INKLING2_2DSD_DRAFT_WINDOW")) {
        win = std::atoi(ew);
    }
    return win;
}

class SlotCommitRunner final : public Inkling2TwoDSDRunner {
public:
    SlotCommitRunner(ggml_backend_t b, const Inkling2Weights & w, Inkling2Cache & c)
        : b_(b), w_(w), c_(c) {}

    int prefix_length() const override { return c_.tree.prefix_len; }
    int max_nodes() const override { return c_.tree.max_nodes; }
    int max_ctx() const override { return c_.max_ctx; }

    bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> &, uint32_t,
                       std::vector<int> &, std::vector<int32_t> &) override {
        return false; // batched advance runs at engine level
    }
    bool commit_nodes(const int * slots, int n) override {
        return inkling2_2dsd_commit_nodes(b_, w_, c_, slots, n);
    }
    bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & reqs,
                     std::vector<Inkling2TwoDSDDraftOut> & out) override {
        return inkling2_2dsd_draft_batch(b_, w_, c_, reqs, out);
    }

private:
    ggml_backend_t b_;
    const Inkling2Weights & w_;
    Inkling2Cache & c_;
};

} // namespace

// ─── Inkling2SlotKv (SlotKv v2) ──────────────────────────────────────────

uint32_t Inkling2SlotKv::compute_pool_blocks_(int max_ctx, size_t n_slots, uint32_t bs) {
    if (max_ctx <= 0 || n_slots == 0 || bs == 0) return 0;
    const uint64_t per_slot = ((uint64_t) max_ctx + bs - 1) / bs;
    const uint64_t total    = per_slot * (uint64_t) n_slots;
    return total <= std::numeric_limits<uint32_t>::max() ? (uint32_t) total : 0;
}

Inkling2SlotKv::Inkling2SlotKv(int max_ctx,
                               const std::vector<Inkling2Cache *> & caches,
                               uint32_t block_size)
    : pool_(compute_pool_blocks_(max_ctx, caches.size(), block_size),
            (uint32_t) caches.size(),
            block_size),
      caches_(caches) {}

SlotKvCaps Inkling2SlotKv::caps() const {
    SlotKvCaps c;
    c.parkable_offload        = true;
    c.sticky_sessions         = false;
    c.supports_prefix_store   = true;
    c.mutates_existing_blocks = false;
    c.dual_tier_tables        = false;
    for (const Inkling2Cache * cache : caches_) {
        if (cache && cache->kvflash && cache->kvflash->active()) {
            c.mutates_existing_blocks = true;
            if (cache->kvflash->tier15_active()) c.dual_tier_tables = true;
        }
    }
    return c;
}

PagedKvStatus Inkling2SlotKv::acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                               PagedKvSequenceHandle & h) {
    return pool_.acquire_reserved(id, cap, h);
}

PagedKvStatus Inkling2SlotKv::acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                               const SlotAcquireOpts & opts,
                                               PagedKvSequenceHandle & h) {
    (void) opts;
    return pool_.acquire_reserved(id, cap, h);
}

PagedKvStatus Inkling2SlotKv::reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) {
    return pool_.reserve_capacity(h, cap);
}

PagedKvAppendResult Inkling2SlotKv::append(PagedKvSequenceHandle h, uint32_t n, bool first_last) {
    return pool_.append(h, n, first_last);
}

PagedKvStatus Inkling2SlotKv::rollback_append(PagedKvSequenceHandle h, uint32_t n) {
    return pool_.rollback_append(h, n);
}

PagedKvStatus Inkling2SlotKv::clear(PagedKvSequenceHandle h) {
    return pool_.clear(h);
}

PagedKvStatus Inkling2SlotKv::release(PagedKvSequenceHandle h) {
    return pool_.release(h);
}

PagedKvStatus Inkling2SlotKv::owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const {
    return pool_.owned_block_count(h, n);
}

bool Inkling2SlotKv::build_table_patch(PagedKvSequenceHandle h, uint32_t token_extent,
                                       uint32_t max_attn_blocks, SlotTablePatch & out,
                                       bool remote) const {
    out = {};
    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(h, snap) != PagedKvStatus::Ok) return false;
    const uint32_t slot_id = h.slot;
    const uint32_t bs = std::max<uint32_t>(1, pool_.block_size());
    const uint32_t need = (token_extent + bs - 1) / bs;
    if (need > max_attn_blocks) return false;

    if (slot_id < caches_.size() && caches_[slot_id] &&
        caches_[slot_id]->kvflash && caches_[slot_id]->kvflash->active()) {
        const auto & kf = *caches_[slot_id]->kvflash;
        if (remote && !kf.tier15_active()) return false;
        const auto & bt = remote ? kf.bt_remote() : kf.bt_local();
        const uint32_t n_copy = std::min<uint32_t>(need, (uint32_t) bt.size());
        out.first_block = 0;
        out.blocks.assign(bt.begin(), bt.begin() + (ptrdiff_t) n_copy);
        return true;
    }

    if (remote) return false;
    if (need > snap.block_table.size()) return false;
    out.first_block = 0;
    out.blocks.resize(need);
    for (uint32_t i = 0; i < need; ++i) out.blocks[i] = (int32_t) snap.block_table[i];
    return true;
}

// ─── Inkling2SeqEngine ───────────────────────────────────────────────────

struct Inkling2SeqEngine::PrefixCheckpoint {
    bool                 valid = false;
    int                  tokens = 0;
    int                  draft_pref_start = 0;
    bool                 has_draft_kv = false;
    bool                 is_kvflash = false;
    std::vector<int32_t> prompt_tokens;
    SlotStateSlab        state_slab;
    std::vector<uint8_t> global_kv;
    size_t               bytes = 0;

    void clear() {
        valid = false;
        tokens = 0;
        draft_pref_start = 0;
        has_draft_kv = false;
        is_kvflash = false;
        std::vector<int32_t>().swap(prompt_tokens);
        std::vector<uint8_t>().swap(state_slab);
        std::vector<uint8_t>().swap(global_kv);
        bytes = 0;
    }
};

struct Inkling2SeqEngine::SlotState {
    std::vector<int32_t> prompt;
    int prefill_pos = 0;
    int rewind_pos  = -1;
    int q_cap_start = 0;
    bool rewind_done = false;
    SlotStateSlab parked_slab;

    // 2DSD per-slot session & sampling bridge
    std::unique_ptr<SlotCommitRunner> runner;
    std::unique_ptr<Inkling2TwoDSDSession> session;
    Inkling2TwoDSDStats stats;
    long long last_reported_draft_slots = 0;

    const Inkling2TwoDSDAdvanceSampleOut * cur_batch_sample = nullptr;
    const std::vector<int> * cur_global_readout_idx = nullptr;
    bool cur_fast_gpu_active = false;
    std::vector<int32_t> cur_penalty_hist;
    std::vector<float> scratch_logits;
    Inkling2TwoDSDSamplerHooks hooks;
};

Inkling2SeqEngine::Inkling2SeqEngine(
        ggml_backend_t target_backend,
        const Inkling2Weights & weights,
        Inkling2Cache & multi_cache,
        int max_concurrency,
        int prefill_chunk)
    : backend_(target_backend),
      weights_(weights),
      multi_cache_(&multi_cache),
      batched_2dsd_(true),
      n_slots_(std::max(1, std::min(max_concurrency, multi_cache.tree.n_seq_slots))),
      prefill_chunk_(std::max(64, prefill_chunk)),
      owned_slot_views_((size_t) n_slots_),
      caches_((size_t) n_slots_, nullptr),
      slot_state_((size_t) n_slots_),
      prefix_checkpoints_((size_t) kPrefixCheckpoints) {
    for (int s = 0; s < n_slots_; ++s) {
        caches_[(size_t) s] = &owned_slot_views_[(size_t) s];
        slot_state_[(size_t) s] = std::make_unique<SlotState>();
    }
    init_slot_views();
    slot_kv_ = std::make_unique<Inkling2SlotKv>(multi_cache.max_ctx, caches_, (uint32_t) kInkling2KvFlashChunk);
    slots_   = std::make_unique<SeqSlotManager>(*slot_kv_, multi_cache.max_ctx);
    state_adapter_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/false);
    state_adapter_with_draft_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/true);
}

Inkling2SeqEngine::Inkling2SeqEngine(
        ggml_backend_t backend,
        const Inkling2Weights & weights,
        std::vector<Inkling2Cache *> caches,
        int max_ctx,
        int prefill_chunk,
        int verify_width,
        Inkling2SeqForwardFn custom_forward)
    : backend_(backend),
      weights_(weights),
      multi_cache_(nullptr),
      batched_2dsd_(false),
      n_slots_(std::max(1, (int) caches.size())),
      prefill_chunk_(std::max(1, prefill_chunk)),
      verify_width_(verify_width),
      custom_forward_(std::move(custom_forward)),
      caches_(std::move(caches)),
      slot_state_((size_t) n_slots_),
      prefix_checkpoints_((size_t) kPrefixCheckpoints) {
    for (int s = 0; s < n_slots_; ++s) {
        slot_state_[(size_t) s] = std::make_unique<SlotState>();
    }
    slot_kv_ = std::make_unique<Inkling2SlotKv>(max_ctx, caches_, (uint32_t) kInkling2KvFlashChunk);
    slots_   = std::make_unique<SeqSlotManager>(*slot_kv_, max_ctx);
    state_adapter_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/false);
    state_adapter_with_draft_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/true);
}

Inkling2SeqEngine::~Inkling2SeqEngine() {
    release_device_state();
}

bool Inkling2SeqEngine::init_slot_views() {
    if (!batched_2dsd_ || !multi_cache_) return true;
    for (int s = 0; s < n_slots_; ++s) {
        if (!make_inkling2_slot_view(*multi_cache_, weights_, s, owned_slot_views_[(size_t) s])) {
            return false;
        }
        caches_[(size_t) s] = &owned_slot_views_[(size_t) s];
        slot_state_[(size_t) s]->runner =
            std::make_unique<SlotCommitRunner>(backend_, weights_, owned_slot_views_[(size_t) s]);
    }
    if (slot_kv_) slot_kv_->set_caches(caches_);
    if (state_adapter_) {
        state_adapter_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/false);
    }
    if (state_adapter_with_draft_) {
        state_adapter_with_draft_ = std::make_unique<Inkling2SlotStateAdapter>(weights_, caches_, /*include_draft_ring=*/true);
    }
    return true;
}

void Inkling2SeqEngine::release_device_state() {
    if (backend_) ggml_backend_synchronize(backend_);
    if (batched_2dsd_) {
        for (int s = 0; s < n_slots_; ++s) {
            if (slot_state_[(size_t) s]) {
                slot_state_[(size_t) s]->session.reset();
                slot_state_[(size_t) s]->runner.reset();
            }
            if ((size_t) s < owned_slot_views_.size()) {
                free_inkling2_cache(owned_slot_views_[(size_t) s]);
            }
        }
    }
}

bool Inkling2SeqEngine::reinstate_device_state(int prefill_chunk) {
    prefill_chunk_ = std::max(64, prefill_chunk);
    return init_slot_views();
}

bool Inkling2SeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

int Inkling2SeqEngine::max_decode_tokens() const {
    if (batched_2dsd_) return kInkling2TwoDSDMaxCommit;
    if (verify_width_ == 1 || !weights_.dspark.loaded) return 1;
    const int cap = std::min(kInkling2MaxVerify, weights_.dspark.block_size);
    return (verify_width_ >= 2 && verify_width_ <= cap) ? verify_width_ : cap;
}

void Inkling2SeqEngine::reset_slot_session(int slot) {
    auto & st = *slot_state_[(size_t) slot];
    st.session.reset();
    st.cur_batch_sample = nullptr;
    st.cur_global_readout_idx = nullptr;
    st.cur_fast_gpu_active = false;
    st.cur_penalty_hist.clear();
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    return admit_bounded(request_id, prompt, sampler, 0, {});
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit_bounded(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int max_new_tokens,
        const AdmitMeta & meta) {
    (void) max_new_tokens; (void) meta;
    if (prompt.empty()) {
        AdmitResult inv;
        inv.status = AdmitResult::Status::failed;
        inv.error  = "inkling2 prompt is empty";
        return inv;
    }
    const int vocab = weights_.n_vocab_unpadded;
    for (int32_t tok : prompt) {
        if (tok < 0 || (vocab > 0 && tok >= vocab)) {
            AdmitResult inv;
            inv.status = AdmitResult::Status::failed;
            inv.error  = "inkling2 prompt contains an invalid token id";
            return inv;
        }
    }

    AdmitResult res = slots_->admit(request_id, prompt, sampler);
    if (res.status != AdmitResult::Status::admitted) return res;

    const int s = res.slot;
    if (s < 0 || s >= n_slots_ || !caches_[(size_t) s] ||
        (batched_2dsd_ && !caches_[(size_t) s]->tree.active)) {
        slots_->retire(s);
        res.status = AdmitResult::Status::failed;
        res.error  = "inkling2 slot cache not initialized";
        return res;
    }
    if (slot_kv_->reserve_capacity(slots_->slot(s).handle, (uint32_t) max_context()) != PagedKvStatus::Ok) {
        slots_->retire(s);
        res.status = AdmitResult::Status::busy;
        res.error  = "inkling2 full-context reservation unavailable";
        return res;
    }

    Inkling2Cache & cache = *caches_[(size_t) s];
    if (!custom_forward_) {
        reset_inkling2_cache(cache);
    }

    auto & st = *slot_state_[(size_t) s];
    reset_slot_session(s);
    slots_->slot(s).pending_capture = {};
    st.prompt = prompt;
    st.prefill_pos = 0;
    st.rewind_pos  = -1;
    st.q_cap_start = (int) prompt.size();
    st.rewind_done = false;
    st.parked_slab.clear();
    st.stats = {};
    st.last_reported_draft_slots = 0;

    if (cache.kvflash && cache.kvflash->active()) {
        st.rewind_pos = cache.kvflash->plan_mode4_rewind(prompt, -1);
        st.q_cap_start = (st.rewind_pos >= 0)
            ? std::max(st.rewind_pos, (int) prompt.size() - cache.kvflash->select_queries())
            : (int) prompt.size();
        cache.kvflash->clear_queries();
    }
    return res;
}

bool Inkling2SeqEngine::supports_prefix_store() const {
    return !custom_forward_ && !caches_.empty() && caches_[0] != nullptr;
}

int Inkling2SeqEngine::checkpoint_index_(PrefixStoreRef checkpoint) const {
    if (!checkpoint.valid() || checkpoint.id > (uint64_t) kPrefixCheckpoints) return -1;
    return (int) checkpoint.id - 1;
}

size_t Inkling2SeqEngine::estimate_prefix_store_bytes(int tokens) const {
    if (!supports_prefix_store() || tokens <= 0) return 0;
    const Inkling2Cache & c0 = *caches_[0];
    const int n_tok = std::clamp(tokens, 1, max_context());
    const bool compact_draft = c0.tree.active && (c0.tree.draft_ring_cap > 0 || c0.tree.draft_cap <= 8192);
    size_t total = compact_draft ? state_adapter_with_draft_->slab_bytes() : state_adapter_->slab_bytes();
    if (c0.kvflash && c0.kvflash->active()) {
        total += c0.kvflash->serialized_bytes_estimate(n_tok);
    } else if (c0.tree.active) {
        const auto & tc = c0.tree;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) continue;
            if (ip < (int) tc.prefix_k.size() && tc.prefix_k[ip] && tc.prefix_v[ip]) {
                const size_t p_st = (size_t) tc.prefix_k[ip]->ne[3];
                total += p_st * (size_t) weights_.n_head_kv * (size_t) n_tok
                       * (tc.prefix_k[ip]->nb[1] + tc.prefix_v[ip]->nb[1]);
            }
        }
    } else {
        for (int il : weights_.global_slots) {
            if (il < (int) c0.k_cache.size() && c0.k_cache[il] && c0.v_cache[il]) {
                total += (size_t) weights_.n_head_kv * (size_t) n_tok
                       * (c0.k_cache[il]->nb[1] + c0.v_cache[il]->nb[1]);
            }
        }
    }
    total += (size_t) n_tok * sizeof(int32_t);
    return total;
}

void Inkling2SeqEngine::discard_prefix_store(PrefixStoreRef checkpoint) {
    const int idx = checkpoint_index_(checkpoint);
    if (idx >= 0 && prefix_checkpoints_[(size_t) idx].valid &&
        prefix_checkpoints_[(size_t) idx].tokens == checkpoint.tokens) {
        prefix_checkpoints_[(size_t) idx].clear();
    }
}

bool Inkling2SeqEngine::save_global_kv_(
        int slot, int tokens, std::vector<uint8_t> & out, bool & is_kvflash) const {
    out.clear();
    if (slot < 0 || slot >= n_slots_ || !caches_[(size_t) slot] || tokens <= 0) return false;
    const Inkling2Cache & cache = *caches_[(size_t) slot];
    if (cache.kvflash && cache.kvflash->active()) {
        is_kvflash = true;
        return cache.kvflash->serialize_state(tokens, out);
    }
    is_kvflash = false;
    if (cache.remote_backend && cache.remote_split_pos >= 0 && tokens > cache.remote_split_pos) {
        return false; // non-KVFlash Split-KV across two GPUs not supported by contiguous prefix store
    }
    if (cache.tree.active) {
        const auto & tc = cache.tree;
        size_t total = 0;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) continue;
            ggml_tensor * pk = tc.prefix_k[ip];
            ggml_tensor * pv = tc.prefix_v[ip];
            if (!pk || !pv || tokens > pk->ne[1] || tokens > pv->ne[1]) return false;
            total += (size_t) pk->ne[3] * (size_t) weights_.n_head_kv * (size_t) tokens * (pk->nb[1] + pv->nb[1]);
        }
        out.resize(total);
        uint8_t * dst = out.data();
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) continue;
            ggml_tensor * pk = tc.prefix_k[ip];
            ggml_tensor * pv = tc.prefix_v[ip];
            const int p_st = (int) pk->ne[3];
            const size_t seg_k = (size_t) tokens * pk->nb[1];
            const size_t seg_v = (size_t) tokens * pv->nb[1];
            for (int p = 0; p < p_st; ++p) {
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_get(pk, dst, (size_t) p * pk->nb[3] + (size_t) h * pk->nb[2], seg_k);
                    dst += seg_k;
                }
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_get(pv, dst, (size_t) p * pv->nb[3] + (size_t) h * pv->nb[2], seg_v);
                    dst += seg_v;
                }
            }
        }
        return (size_t) (dst - out.data()) == total;
    }
    size_t total = 0;
    for (int il : weights_.global_slots) {
        ggml_tensor * k = cache.k_cache[il];
        ggml_tensor * v = cache.v_cache[il];
        if (!k || !v || tokens > k->ne[1] || tokens > v->ne[1]) return false;
        total += (size_t) weights_.n_head_kv * (size_t) tokens * (k->nb[1] + v->nb[1]);
    }
    out.resize(total);
    uint8_t * dst = out.data();
    for (int il : weights_.global_slots) {
        ggml_tensor * k = cache.k_cache[il];
        ggml_tensor * v = cache.v_cache[il];
        const size_t seg_k = (size_t) tokens * k->nb[1];
        const size_t seg_v = (size_t) tokens * v->nb[1];
        for (int h = 0; h < weights_.n_head_kv; ++h) {
            ggml_backend_tensor_get(k, dst, (size_t) h * k->nb[2], seg_k);
            dst += seg_k;
        }
        for (int h = 0; h < weights_.n_head_kv; ++h) {
            ggml_backend_tensor_get(v, dst, (size_t) h * v->nb[2], seg_v);
            dst += seg_v;
        }
    }
    return (size_t) (dst - out.data()) == total;
}

bool Inkling2SeqEngine::restore_global_kv_(
        int slot, int tokens, const std::vector<uint8_t> & in, bool is_kvflash) {
    if (slot < 0 || slot >= n_slots_ || !caches_[(size_t) slot] || tokens <= 0 || in.empty()) return false;
    Inkling2Cache & cache = *caches_[(size_t) slot];
    const bool live_kf = (cache.kvflash && cache.kvflash->active());
    if (is_kvflash != live_kf) return false;
    if (live_kf) {
        return cache.kvflash->restore_state(in, tokens);
    }
    if (cache.tree.active) {
        auto & tc = cache.tree;
        size_t expected = 0;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) continue;
            ggml_tensor * pk = tc.prefix_k[ip];
            ggml_tensor * pv = tc.prefix_v[ip];
            if (!pk || !pv || tokens > pk->ne[1] || tokens > pv->ne[1]) return false;
            expected += (size_t) pk->ne[3] * (size_t) weights_.n_head_kv * (size_t) tokens * (pk->nb[1] + pv->nb[1]);
        }
        if (in.size() != expected) return false;
        const uint8_t * src = in.data();
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) continue;
            ggml_tensor * pk = tc.prefix_k[ip];
            ggml_tensor * pv = tc.prefix_v[ip];
            const int p_st = (int) pk->ne[3];
            const size_t seg_k = (size_t) tokens * pk->nb[1];
            const size_t seg_v = (size_t) tokens * pv->nb[1];
            for (int p = 0; p < p_st; ++p) {
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_set(pk, src, (size_t) p * pk->nb[3] + (size_t) h * pk->nb[2], seg_k);
                    src += seg_k;
                }
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_set(pv, src, (size_t) p * pv->nb[3] + (size_t) h * pv->nb[2], seg_v);
                    src += seg_v;
                }
            }
        }
        if (backend_) ggml_backend_synchronize(backend_);
        return (size_t) (src - in.data()) == expected;
    }
    size_t expected = 0;
    for (int il : weights_.global_slots) {
        ggml_tensor * k = cache.k_cache[il];
        ggml_tensor * v = cache.v_cache[il];
        if (!k || !v || tokens > k->ne[1] || tokens > v->ne[1]) return false;
        expected += (size_t) weights_.n_head_kv * (size_t) tokens * (k->nb[1] + v->nb[1]);
    }
    if (in.size() != expected) return false;
    const uint8_t * src = in.data();
    for (int il : weights_.global_slots) {
        ggml_tensor * k = cache.k_cache[il];
        ggml_tensor * v = cache.v_cache[il];
        const size_t seg_k = (size_t) tokens * k->nb[1];
        const size_t seg_v = (size_t) tokens * v->nb[1];
        for (int h = 0; h < weights_.n_head_kv; ++h) {
            ggml_backend_tensor_set(k, src, (size_t) h * k->nb[2], seg_k);
            src += seg_k;
        }
        for (int h = 0; h < weights_.n_head_kv; ++h) {
            ggml_backend_tensor_set(v, src, (size_t) h * v->nb[2], seg_v);
            src += seg_v;
        }
    }
    if (backend_) ggml_backend_synchronize(backend_);
    return (size_t) (src - in.data()) == expected;
}

bool Inkling2SeqEngine::restore_prefix_(
        int slot, const std::vector<int32_t> & prompt, PrefixStoreRef checkpoint) {
    const int idx = checkpoint_index_(checkpoint);
    if (idx < 0 || checkpoint.tokens >= (int) prompt.size()) return false;
    const PrefixCheckpoint & cp = prefix_checkpoints_[(size_t) idx];
    if (!cp.valid || cp.tokens != checkpoint.tokens ||
        cp.prompt_tokens.size() != (size_t) checkpoint.tokens ||
        !std::equal(cp.prompt_tokens.begin(), cp.prompt_tokens.end(), prompt.begin())) {
        return false;
    }
    Inkling2Cache & cache = *caches_[(size_t) slot];
    if (!state_adapter_->restore_slot_state(slot, cp.state_slab) ||
        !restore_global_kv_(slot, cp.tokens, cp.global_kv, cp.is_kvflash)) {
        reset_inkling2_cache(cache);
        return false;
    }
    if (!slots_->seed_restored_prefix(slot, cp.tokens).ok) {
        reset_inkling2_cache(cache);
        return false;
    }
    auto & st = *slot_state_[(size_t) slot];
    st.prefill_pos = cp.tokens;
    if (cache.tree.active) {
        cache.tree.prefix_len = cp.tokens;
        if (cp.has_draft_kv) {
            cache.tree.draft_pref_start = cp.draft_pref_start;
        } else {
            const int win = resolve_draft_window();
            cache.tree.draft_pref_start = (win > 0 && (int) prompt.size() > win)
                ? std::max(cp.tokens, (int) prompt.size() - win)
                : cp.tokens;
        }
    }
    cache.cur_pos = cp.tokens;
    if (cache.kvflash && cache.kvflash->active()) {
        const int ct = cache.kvflash->block_size();
        st.rewind_pos = cache.kvflash->plan_mode4_rewind(prompt, -1);
        if (st.rewind_pos >= 0 && st.rewind_pos < st.prefill_pos) {
            // Rewind cannot precede the restored prefix; align up to next chunk boundary inside the new suffix.
            const int aligned = ((st.prefill_pos + ct - 1) / ct) * ct;
            if (aligned < (int) prompt.size()) {
                st.rewind_pos = aligned;
                if (aligned == st.prefill_pos) {
                    cache.kvflash->save_rewind_snap(cache, weights_, st.rewind_pos);
                }
            } else {
                st.rewind_pos = -1;
            }
        }
        st.q_cap_start = (st.rewind_pos >= 0)
            ? std::max(st.rewind_pos, (int) prompt.size() - cache.kvflash->select_queries())
            : (int) prompt.size();
        cache.kvflash->clear_queries();
    }
    return true;
}

PrefixStoreEvent Inkling2SeqEngine::capture_prefix_(int slot, const PrefixCaptureTicket & ticket) {
    PrefixStoreEvent event;
    event.ticket = ticket;
    event.status = PrefixStoreEvent::Status::failed;
    const int idx = checkpoint_index_(ticket.checkpoint);
    const SeqSlot & sl = slots_->slot(slot);
    if (!supports_prefix_store() || idx < 0 || !ticket.valid() ||
        ticket.checkpoint.tokens != sl.cur_pos ||
        ticket.checkpoint.tokens > (int) sl.sample_history.size()) {
        event.error = "inkling2 invalid prefix capture boundary";
        return event;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed_us = [&t0] {
        return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    };

    const Inkling2Cache & cache = *caches_[(size_t) slot];
    const bool save_draft = cache.tree.active &&
                            ticket.checkpoint.tokens > cache.tree.draft_pref_start &&
                            (cache.tree.draft_ring_cap > 0 || cache.tree.draft_cap <= 8192);
    PrefixCheckpoint next;
    next.tokens = ticket.checkpoint.tokens;
    next.draft_pref_start = cache.tree.active ? cache.tree.draft_pref_start : 0;
    next.has_draft_kv = save_draft;
    next.prompt_tokens.assign(sl.sample_history.begin(), sl.sample_history.begin() + next.tokens);

    if (backend_) ggml_backend_synchronize(backend_);
    const bool state_ok = save_draft
        ? state_adapter_with_draft_->save_slot_state(slot, next.state_slab)
        : state_adapter_->save_slot_state(slot, next.state_slab);
    if (!state_ok || !save_global_kv_(slot, next.tokens, next.global_kv, next.is_kvflash)) {
        event.elapsed_us = elapsed_us();
        event.error = "inkling2 prefix capture failed";
        return event;
    }
    next.bytes = next.state_slab.size() + next.global_kv.size() + next.prompt_tokens.size() * sizeof(int32_t);
    next.valid = true;
    prefix_checkpoints_[(size_t) idx] = std::move(next);

    event.status = PrefixStoreEvent::Status::saved;
    event.bytes = prefix_checkpoints_[(size_t) idx].bytes;
    event.elapsed_us = elapsed_us();
    std::fprintf(stderr,
        "[inkling2-seq] saved checkpoint=%llu slot=%d tokens=%d bytes=%.1f MiB kvflash=%d draft=%d time_ms=%.1f\n",
        (unsigned long long) ticket.checkpoint.id, slot, ticket.checkpoint.tokens,
        (double) event.bytes / 1048576.0,
        prefix_checkpoints_[(size_t) idx].is_kvflash ? 1 : 0,
        prefix_checkpoints_[(size_t) idx].has_draft_kv ? 1 : 0,
        (double) event.elapsed_us / 1000.0);
    return event;
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit_with_prefix(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        const PrefixStorePlan & plan) {
    AdmitResult res = admit_bounded(request_id, prompt, sampler, 0, {});
    if (res.status != AdmitResult::Status::admitted) return res;
    const int s = res.slot;
    slots_->slot(s).pending_capture = {};

    if (plan.restore.valid()) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool restored = restore_prefix_(s, prompt, plan.restore);
        const uint64_t elapsed_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (restored) {
            res.prefix_store.restored = plan.restore;
            std::fprintf(stderr,
                "[inkling2-seq] restored checkpoint=%llu slot=%d tokens=%d/%zu time_ms=%.1f\n",
                (unsigned long long) plan.restore.id, s, plan.restore.tokens, prompt.size(),
                (double) elapsed_us / 1000.0);
        } else {
            discard_prefix_store(plan.restore);
            res.prefix_store.invalidated = plan.restore;
        }
        res.prefix_store.restore_attempted = true;
        res.prefix_store.restore_elapsed_us = elapsed_us;
    }

    const SeqSlot & seq = slots_->slot(s);
    if (!res.prefix_store.invalidated.valid() && plan.capture.valid() &&
        checkpoint_index_(plan.capture.checkpoint) >= 0 &&
        plan.capture.checkpoint.tokens > seq.cur_pos &&
        plan.capture.checkpoint.tokens <= seq.prompt_len) {
        slots_->slot(s).pending_capture = plan.capture;
        res.prefix_store.capture = plan.capture;
    }
    return res;
}

StepPlanLimits Inkling2SeqEngine::step_plan_limits(int decode_rows) const {
    StepPlanLimits lim;
    if (batched_2dsd_) {
        lim.max_prefill_sequences = 1;
        lim.max_prefill_tokens_per_sequence = prefill_chunk_;
        lim.max_prefill_tokens_total = prefill_chunk_;
        lim.prefill_allocation_quantum = prefill_chunk_;
        lim.prefill_time_share = 1.0;
        lim.time_share_prompt_tokens = 0;
    } else {
        lim.max_prefill_sequences           = std::min(slot_count(), (decode_rows > 0) ? 1 : 2);
        lim.max_prefill_tokens_per_sequence = prefill_chunk_;
        lim.max_prefill_tokens_total        = prefill_chunk_ * lim.max_prefill_sequences;
        lim.prefill_allocation_quantum      = std::min(64, prefill_chunk_);
    }
    return lim;
}

bool Inkling2SeqEngine::reserve_decode(const StepPlan & plan) {
    std::vector<int> growth((size_t) n_slots_, 0);
    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= n_slots_) return false;
        const int rem = std::max(0, max_context() - slots_->slot(in.slot).cur_pos);
        if (rem <= 0) return false;
        growth[(size_t) in.slot] = batched_2dsd_
            ? std::min(rem, in.allow_speculation ? max_decode_tokens() : 1)
            : 1;
    }
    return slots_->reserve_decode(growth);
}

std::string Inkling2SeqEngine::validate_plan_(const StepPlan & plan) const {
    const int n = slot_count();
    std::vector<uint8_t> seen_dec((size_t) n, 0);
    std::vector<uint8_t> seen_pre((size_t) n, 0);

    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= n) return "decode slot out of range";
        if (in.token < 0 || (weights_.n_vocab_unpadded > 0 && in.token >= weights_.n_vocab_unpadded)) {
            return "decode input token out of range";
        }
        if (seen_dec[(size_t) in.slot]) return "duplicate decode slot";
        if (!slots_->slot(in.slot).decoding()) return "decode slot is not in decode phase";
        seen_dec[(size_t) in.slot] = 1;
    }
    for (int s = 0; s < n; ++s) {
        if (slots_->slot(s).decoding() && !seen_dec[(size_t) s]) {
            return "plan omitted an active decoding slot";
        }
    }

    const StepPlanLimits lim = step_plan_limits((int) plan.decode.size());
    if (!batched_2dsd_ && (int) plan.prefills.size() > lim.max_prefill_sequences) {
        return "prefill cohort exceeds max_prefill_sequences";
    }
    int total_pre = 0;
    for (const PrefillSlice & sl : plan.prefills) {
        if (sl.slot < 0 || sl.slot >= n) return "prefill slot out of range";
        if (sl.max_tokens <= 0 || (!batched_2dsd_ && sl.max_tokens > lim.max_prefill_tokens_per_sequence)) {
            return "prefill slice exceeds per-sequence token limit";
        }
        if (seen_pre[(size_t) sl.slot] || seen_dec[(size_t) sl.slot]) {
            return "duplicate or decoding slot in prefill plan";
        }
        if (!slots_->slot(sl.slot).prefilling()) return "prefill slot is not in prefill phase";
        seen_pre[(size_t) sl.slot] = 1;
        total_pre += sl.max_tokens;
    }
    if (!batched_2dsd_ && total_pre > lim.max_prefill_tokens_total) {
        return "prefill slices exceed max_prefill_tokens_total";
    }
    return {};
}

SeqEngine::StepResult Inkling2SeqEngine::step(const StepPlan & plan) {
    if (plan.decode.empty() && plan.prefills.empty()) {
        return {};
    }
    const std::string val_err = validate_plan_(plan);
    if (!val_err.empty()) {
        StepResult res;
        res.error = val_err;
        return res;
    }
    return batched_2dsd_ ? step_batched_2dsd_(plan) : step_1d_(plan);
}

// ─── Batched Multi-Slot 2DSD Step (Shares all 128 MoE Experts + KVFlash!) ─

SeqEngine::StepResult Inkling2SeqEngine::step_batched_2dsd_(const StepPlan & plan) {
    StepResult result;
    const int n_vocab = weights_.n_vocab_unpadded;

    // 1. Prefill selected prompt slices (zero-copy per-slot cache views + per-slot Mode-4!)
    for (const PrefillSlice & slice : plan.prefills) {
        const int s = slice.slot;
        SeqSlot & sl = slots_->slot(s);
        SlotState & st = *slot_state_[(size_t) s];
        Inkling2Cache & sc = *caches_[(size_t) s];
        auto * kf = (sc.kvflash && sc.kvflash->active()) ? sc.kvflash.get() : nullptr;

        const int rem = sl.prompt_len - st.prefill_pos;
        int take = std::min(rem, slice.max_tokens);
        if (take <= 0) {
            PrefillOutput po;
            po.slot = s;
            po.status = PrefillOutput::Status::failed;
            po.error = "inkling2 prefill: no remaining prompt tokens";
            result.prefills.push_back(std::move(po));
            continue;
        }
        if (st.rewind_pos > st.prefill_pos && st.rewind_pos < st.prefill_pos + take) {
            take = st.rewind_pos - st.prefill_pos;
        }
        if (sl.pending_capture.valid() &&
            sl.pending_capture.checkpoint.tokens > st.prefill_pos &&
            sl.pending_capture.checkpoint.tokens < st.prefill_pos + take) {
            take = sl.pending_capture.checkpoint.tokens - st.prefill_pos;
        }
        if (st.prefill_pos == 0) {
            const int win = resolve_draft_window();
            sc.tree.draft_pref_start = (win > 0 && sl.prompt_len > win) ? (sl.prompt_len - win) : 0;
        }

        const int end = st.prefill_pos + take;
        const bool is_last = (end >= sl.prompt_len);
        const bool is_greedy = !sl.sampler.needs_logit_processing();
        const int cap_q = (kf && st.rewind_pos >= 0 && !st.rewind_done && end > st.q_cap_start)
            ? std::min(take, end - st.q_cap_start) : 0;

        int32_t first_tok = -1;
        std::vector<float> last_logits;
        const bool allow_wf = (std::getenv("LUCE_INKLING2_WAVEFRONT_PREFILL") == nullptr ||
                               std::atoi(std::getenv("LUCE_INKLING2_WAVEFRONT_PREFILL")) != 0);
        const bool ok = (allow_wf && take > prefill_chunk_ && weights_.is_loop && weights_.n_passes == 3)
            ? inkling2_2dsd_prefill_wavefront(
                backend_, weights_, sc, st.prompt.data() + st.prefill_pos, take, st.prefill_pos,
                prefill_chunk_, first_tok, (is_last && !is_greedy) ? &last_logits : nullptr, cap_q, is_last)
            : inkling2_2dsd_prefill_chunk(
                backend_, weights_, sc, st.prompt.data() + st.prefill_pos, take, st.prefill_pos,
                is_last, first_tok, (is_last && !is_greedy) ? &last_logits : nullptr, cap_q);
        if (!ok) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2 2dsd prefill chunk failed";
            return result;
        }

        auto chunk_res = slots_->append_prefill(s, take);
        if (!chunk_res.ok) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2 slot manager append_prefill failed";
            return result;
        }
        st.prefill_pos += take;

        if (kf && st.rewind_pos > 0 && st.prefill_pos == st.rewind_pos && !st.rewind_done) {
            kf->save_rewind_snap(sc, weights_, st.rewind_pos);
        }
        if (is_last && kf && st.rewind_pos > 0 && !st.rewind_done && kf->has_rewind_snap()) {
            st.rewind_done = true;
            if (kf->mode4_score_and_recall(st.rewind_pos, sl.prompt_len)) {
                kf->restore_rewind_snap(sc, weights_);
                int rpos = st.rewind_pos;
                while (rpos < sl.prompt_len) {
                    const int rend = std::min(sl.prompt_len, rpos + prefill_chunk_);
                    const int rn = rend - rpos;
                    const bool rlast = (rend == sl.prompt_len);
                    if (!inkling2_2dsd_prefill_chunk(
                            backend_, weights_, sc, st.prompt.data() + rpos, rn, rpos, rlast,
                            first_tok, (rlast && !is_greedy) ? &last_logits : nullptr, 0)) {
                        kf->clear_rewind_snap();
                        result.decode.clear();
                        result.prefills.clear();
                        result.error = "inkling2 2dsd Mode-4 rewind prefill failed";
                        return result;
                    }
                    rpos = rend;
                }
            }
            kf->clear_rewind_snap();
        }

        PrefillOutput po;
        po.slot = s;
        if (sl.pending_capture.valid() && st.prefill_pos == sl.pending_capture.checkpoint.tokens) {
            const PrefixCaptureTicket ticket = sl.pending_capture;
            sl.pending_capture = {};
            po.prefix_store = capture_prefix_(s, ticket);
        }
        if (is_last) {
            if (!is_greedy && !last_logits.empty()) {
                auto ph = sl.penalty_history();
                first_tok = sample_logits(last_logits.data(), n_vocab, sl.sampler, ph, sl.rng);
            }
            slots_->commit_prefill(s);
            po.status = PrefillOutput::Status::completed;
            po.token = first_tok;
        } else {
            po.status = PrefillOutput::Status::advanced;
        }
        result.prefills.push_back(std::move(po));
    }

    if (plan.decode.empty()) return result;

    // 2. Prepare decoding sessions for this cohort
    const int n_dec = (int) plan.decode.size();
    const Inkling2TwoDSDConfig base_cfg = inkling2_2dsd_config_for_concurrency(
        weights_.n_passes, weights_.twodsd.block_size, n_dec);
    const bool allow_split = (weights_.expert_backend != nullptr && weights_.split_exp > 0);

    std::vector<DecodeOutput> dec_outs((size_t) n_dec);
    std::vector<int> pending_idx;
    pending_idx.reserve((size_t) n_dec);

    for (int i = 0; i < n_dec; ++i) {
        const StepInput & in = plan.decode[(size_t) i];
        const int s = in.slot;
        SeqSlot & sl = slots_->slot(s);
        SlotState & st = *slot_state_[(size_t) s];
        Inkling2Cache & sc = *caches_[(size_t) s];

        dec_outs[(size_t) i].slot = s;
        if (sc.tree.prefix_len >= max_context()) {
            dec_outs[(size_t) i].failed = true;
            dec_outs[(size_t) i].error = "inkling2 decode: context limit reached";
            continue;
        }

        Inkling2TwoDSDConfig slot_cfg = base_cfg;
        if (!in.allow_speculation) {
            slot_cfg.k = 1;
            slot_cfg.stage = Inkling2TwoDSDStage::Stage1;
        }

        const bool is_greedy = !sl.sampler.needs_logit_processing();
        const bool warm_match = (st.session != nullptr) &&
                                (st.session->cursor_token() == in.token) &&
                                (st.session->cursor_pos() == sc.tree.prefix_len) &&
                                (st.session->cursor_depth() >= 0) &&
                                (st.session->cursor_depth() < weights_.n_passes);
        if (warm_match) {
            st.session->update_config(slot_cfg);
            if (!in.allow_speculation) {
                st.session->clear_cursor_children();
            }
        } else {
            reset_slot_session(s);
            if (!is_greedy) {
                st.scratch_logits.resize((size_t) n_vocab);
                st.hooks.capture = [&st, &sl, n_vocab](int local_row_idx, Inkling2TwoDSDDepthRSample & out) {
                    out.top_cands.clear();
                    out.logits.clear();
                    if (!st.cur_batch_sample || !st.cur_global_readout_idx) return;
                    if (local_row_idx < 0 || local_row_idx >= (int) st.cur_global_readout_idx->size()) return;
                    const int g_idx = (*st.cur_global_readout_idx)[(size_t) local_row_idx];
                    if (st.cur_fast_gpu_active &&
                        (size_t) (g_idx + 1) * 516 <= st.cur_batch_sample->top_cands.size()) {
                        const int32_t * rc = st.cur_batch_sample->top_cands.data() + (size_t) g_idx * 516;
                        std::vector<std::pair<int32_t, float>> tmp_probs;
                        double tmp_cum = 0.0;
                        if (try_build_exact_cands_row(rc, sl.sampler, tmp_probs, tmp_cum)) {
                            out.top_cands.assign(rc, rc + 516);
                            return;
                        }
                    }
                    if (st.cur_batch_sample->logits_dev) {
                        out.logits.resize((size_t) n_vocab);
                        ggml_backend_tensor_get(
                            st.cur_batch_sample->logits_dev, out.logits.data(),
                            (size_t) g_idx * (size_t) n_vocab * sizeof(float),
                            (size_t) n_vocab * sizeof(float));
                    }
                };
                st.hooks.sample = [&st, &sl, n_vocab](const Inkling2TwoDSDDepthRSample & payload,
                                                      int32_t greedy_tok) -> int32_t {
                    if (payload.top_cands.size() >= 516) {
                        std::vector<std::pair<int32_t, float>> probs;
                        double cum = 0.0;
                        if (try_build_exact_cands_row(payload.top_cands.data(), sl.sampler, probs, cum)) {
                            std::uniform_real_distribution<double> u01(0.0, cum);
                            const double r = u01(sl.rng);
                            double acc = 0.0;
                            for (size_t j = 0; j < probs.size(); ++j) {
                                acc += probs[j].second;
                                if (r <= acc || j + 1 == probs.size()) return probs[j].first;
                            }
                        }
                    }
                    if (!payload.logits.empty()) {
                        std::memcpy(st.scratch_logits.data(), payload.logits.data(),
                                    (size_t) n_vocab * sizeof(float));
                        return sample_logits(st.scratch_logits.data(), n_vocab,
                                             sl.sampler, st.cur_penalty_hist, sl.rng);
                    }
                    return greedy_tok;
                };
            }
            st.session = std::make_unique<Inkling2TwoDSDSession>(
                *st.runner, slot_cfg, st.stats, max_context(),
                is_greedy ? nullptr : &st.hooks);
            if (!sl.sample_history.empty()) {
                st.session->set_prompt_history(sl.sample_history);
            } else if (!st.prompt.empty()) {
                st.session->set_prompt_history(st.prompt);
            }
            std::string serr;
            if (!st.session->seed_cursor(in.token, sc.tree.prefix_len, serr)) {
                result.decode.clear();
                result.prefills.clear();
                result.error = serr;
                return result;
            }
        }
        pending_idx.push_back(i);
    }

    // 3. Batched micro-step loop (1 GPU call in steady state, <= R on cold start)
    std::vector<int> prefix_lens((size_t) n_slots_, 0);
    std::vector<const Inkling2KvFlash *> slot_kfs((size_t) n_slots_, nullptr);
    for (int guard = 0; !pending_idx.empty() && guard < weights_.n_passes + 2; ++guard) {
        for (int idx : pending_idx) {
            const int s = plan.decode[(size_t) idx].slot;
            SlotState & st = *slot_state_[(size_t) s];
            std::vector<Inkling2TwoDSDDraftReq> reqs;
            std::vector<int> plan_heads, plan_widths;
            st.session->collect_draft_reqs(reqs, plan_heads, plan_widths);
            if (!reqs.empty()) {
                std::vector<Inkling2TwoDSDDraftOut> outs;
                if (!inkling2_2dsd_draft_batch(backend_, weights_, *caches_[(size_t) s], reqs, outs) ||
                    outs.size() != reqs.size()) {
                    result.decode.clear();
                    result.prefills.clear();
                    result.error = "inkling2 2dsd draft_batch failed";
                    return result;
                }
                ++st.stats.drafter_batches;
                st.stats.drafter_calls += (long long) reqs.size();
                st.session->apply_draft_outs(plan_heads, plan_widths, outs);
            }
        }

        std::vector<Inkling2TwoDSDNodeDesc> combined_descs;
        std::vector<int> desc_owner_idx;
        std::vector<int> desc_local_pos;
        std::vector<std::vector<int>> slot_active_ids((size_t) n_dec);
        uint32_t combined_mask = 0u;

        bool any_stochastic = false;
        bool all_fast_same_temp = true;
        float shared_fast_temp = 0.0f;

        for (int idx : pending_idx) {
            const int s = plan.decode[(size_t) idx].slot;
            const SeqSlot & sl = slots_->slot(s);
            SlotState & st = *slot_state_[(size_t) s];

            std::vector<Inkling2TwoDSDNodeDesc> descs;
            std::string cerr;
            if (!st.session->collect_active(slot_active_ids[(size_t) idx], descs, cerr)) {
                result.decode.clear();
                result.prefills.clear();
                result.error = cerr;
                return result;
            }
            combined_mask |= st.session->readout_mask();
            for (int r = 0; r < (int) descs.size(); ++r) {
                descs[(size_t) r].seq_slot = s;
                combined_descs.push_back(std::move(descs[(size_t) r]));
                desc_owner_idx.push_back(idx);
                desc_local_pos.push_back(r);
            }
            if (sl.sampler.needs_logit_processing()) {
                any_stochastic = true;
                if (!sampler_uses_fast_gpu(sl.sampler)) {
                    all_fast_same_temp = false;
                } else if (shared_fast_temp == 0.0f) {
                    shared_fast_temp = sl.sampler.temp;
                } else if (shared_fast_temp != sl.sampler.temp) {
                    all_fast_same_temp = false;
                }
            }
        }

        bool any_kf = false;
        for (int s = 0; s < n_slots_; ++s) {
            prefix_lens[(size_t) s] = caches_[(size_t) s]->tree.prefix_len;
            slot_kfs[(size_t) s] = (caches_[(size_t) s]->kvflash && caches_[(size_t) s]->kvflash->active())
                ? caches_[(size_t) s]->kvflash.get() : nullptr;
            if (slot_kfs[(size_t) s]) any_kf = true;
        }

        Inkling2TwoDSDAdvanceSampleOut batch_sample;
        batch_sample.fast_temp = (any_stochastic && all_fast_same_temp) ? shared_fast_temp : 0.0f;

        std::vector<int> global_readout_rows;
        std::vector<int32_t> global_readout_argmax;
        if (!inkling2_2dsd_advance_nodes_batched(
                backend_, weights_, *multi_cache_, combined_descs,
                prefix_lens.data(), n_slots_, combined_mask, allow_split,
                global_readout_rows, global_readout_argmax,
                any_stochastic ? &batch_sample : nullptr,
                any_kf ? slot_kfs.data() : nullptr)) {
            result.decode.clear();
            result.prefills.clear();
            result.error = "inkling2_2dsd_advance_nodes_batched failed";
            return result;
        }

        std::vector<std::vector<int>> per_slot_rrows((size_t) n_dec);
        std::vector<std::vector<int32_t>> per_slot_rargmax((size_t) n_dec);
        std::vector<std::vector<int>> per_slot_gidx((size_t) n_dec);
        for (int g = 0; g < (int) global_readout_rows.size(); ++g) {
            const int comb_r = global_readout_rows[(size_t) g];
            const int idx    = desc_owner_idx[(size_t) comb_r];
            per_slot_rrows[(size_t) idx].push_back(desc_local_pos[(size_t) comb_r]);
            per_slot_rargmax[(size_t) idx].push_back(global_readout_argmax[(size_t) g]);
            per_slot_gidx[(size_t) idx].push_back(g);
        }

        std::vector<int> next_pending;
        next_pending.reserve(pending_idx.size());

        for (int idx : pending_idx) {
            const StepInput & in = plan.decode[(size_t) idx];
            const int s = in.slot;
            SeqSlot & sl = slots_->slot(s);
            SlotState & st = *slot_state_[(size_t) s];
            Inkling2Cache & sc = *caches_[(size_t) s];

            ++st.stats.target_calls;
            st.stats.target_cells += (long long) slot_active_ids[(size_t) idx].size();

            st.cur_batch_sample = any_stochastic ? &batch_sample : nullptr;
            st.cur_global_readout_idx = &per_slot_gidx[(size_t) idx];
            st.cur_fast_gpu_active = (batch_sample.fast_temp > 0.0f) && sampler_uses_fast_gpu(sl.sampler);

            st.cur_penalty_hist.clear();
            if (sl.sampler.needs_logit_processing()) {
                st.cur_penalty_hist = sl.penalty_history();
                st.cur_penalty_hist.push_back(in.token);
            }

            const int rem_ctx = std::max(1, max_context() - sc.tree.prefix_len);
            const int max_emit = in.allow_speculation ? std::min(max_decode_tokens(), rem_ctx) : 1;
            std::vector<int32_t> emitted;
            auto on_commit = [&](int32_t & tok) -> bool {
                emitted.push_back(tok);
                if (sl.sampler.needs_logit_processing()) {
                    st.cur_penalty_hist.push_back(tok);
                }
                if ((int) emitted.size() >= max_emit) return false;
                if (token_is_eos(tok)) return false;
                return true;
            };

            bool stopped = false;
            std::string verr;
            const bool ok = st.session->apply_step_and_verify(
                slot_active_ids[(size_t) idx],
                per_slot_rrows[(size_t) idx],
                per_slot_rargmax[(size_t) idx],
                on_commit, stopped, verr);

            st.cur_batch_sample = nullptr;
            st.cur_global_readout_idx = nullptr;
            st.cur_fast_gpu_active = false;

            if (!ok) {
                result.decode.clear();
                result.prefills.clear();
                result.error = verr;
                return result;
            }

            if (!emitted.empty()) {
                std::vector<int32_t> kv_tokens;
                kv_tokens.reserve(emitted.size());
                kv_tokens.push_back(in.token);
                for (size_t j = 0; j + 1 < emitted.size(); ++j) {
                    kv_tokens.push_back(emitted[j]);
                }
                auto app = slots_->append_tokens(s, kv_tokens.data(), (int) kv_tokens.size());
                if (!app.ok) {
                    result.decode.clear();
                    result.prefills.clear();
                    result.error = "inkling2 slot manager append_tokens failed";
                    return result;
                }
                slots_->commit_step(s);

                DecodeOutput & out = dec_outs[(size_t) idx];
                out.token = emitted.back();
                out.committed_tokens.assign(emitted.begin(), emitted.end() - 1);
                out.spec_accepted = (int) out.committed_tokens.size();
                if (in.allow_speculation) {
                    const long long delta = st.stats.draft_slots - st.last_reported_draft_slots;
                    out.spec_offered = std::max(out.spec_accepted, (int) std::max<long long>(0, delta));
                } else {
                    out.spec_offered = 0;
                }
                st.last_reported_draft_slots = st.stats.draft_slots;
            } else {
                next_pending.push_back(idx);
            }
        }
        pending_idx = std::move(next_pending);
    }

    if (!pending_idx.empty()) {
        result.decode.clear();
        result.prefills.clear();
        result.error = "inkling2 2dsd step failed to produce output within R micro-steps";
        return result;
    }

    result.decode = std::move(dec_outs);
    return result;
}

// ─── 1D / DSpark / Custom-Forward Step ───────────────────────────────────

Inkling2SeqStepResult Inkling2SeqEngine::run_forward_1d_(
        int slot, const int32_t * tokens, int n_tokens, int pos0,
        bool need_logits, int capture_q_rows) {
    if (custom_forward_) {
        return custom_forward_(slot, tokens, n_tokens, pos0, need_logits, capture_q_rows);
    }
    Inkling2SeqStepResult out;
    Inkling2Cache & cache = *caches_[(size_t) slot];
    Inkling2ForwardOut r = inkling2_forward(
        backend_, weights_, cache, tokens, n_tokens, pos0,
        /*verify=*/false, /*need_logits=*/need_logits,
        /*capture_taps_for_dspark=*/weights_.dspark.loaded,
        /*allow_split=*/true, /*fast_sample_temp=*/0.0f,
        /*capture_q_rows=*/capture_q_rows);
    if (!r.ok) {
        out.error = "inkling2_forward failed";
        return out;
    }
    out.ok = true;
    if (!r.greedy.empty()) out.greedy_token = r.greedy.back();
    if (need_logits && !r.logits.empty()) {
        const int nv = weights_.n_vocab_unpadded;
        out.logits.assign(r.logits.end() - nv, r.logits.end());
    }
    return out;
}

int32_t Inkling2SeqEngine::sample_slot_1d_(SeqSlot & seq, const Inkling2SeqStepResult & r, bool is_greedy) {
    if (is_greedy && r.greedy_token >= 0) return r.greedy_token;
    if (!r.logits.empty()) {
        const auto pen_hist = seq.penalty_history();
        return (int32_t) sample_logits(r.logits.data(), (int) r.logits.size(), seq.sampler, pen_hist, seq.rng);
    }
    return r.greedy_token;
}

bool Inkling2SeqEngine::try_solo_dspark_step_(const StepInput & in, DecodeOutput & out) {
    if (custom_forward_ || !weights_.dspark.loaded || verify_width_ == 1 || !in.allow_speculation) {
        return false;
    }
    SeqSlot & seq = slots_->slot(in.slot);
    if (!sample_logits_is_argmax(seq.sampler)) return false;

    Inkling2Cache & cache = *caches_[(size_t) in.slot];
    const int pos = seq.cur_pos;
    const int rem_ctx = max_context() - pos - 1;
    const int max_vw = std::min(kInkling2MaxVerify, weights_.dspark.block_size);
    const int vw = (verify_width_ >= 2 && verify_width_ <= max_vw) ? verify_width_ : max_vw;
    const int k_draft = std::min(vw - 1, rem_ctx);
    if (k_draft < 1) return false;

    std::vector<int32_t> drafts;
    if (!inkling2_dspark_draft(backend_, weights_, cache, in.token, pos, k_draft, drafts, 0) || drafts.empty()) {
        return false;
    }
    const int n_d = (int) drafts.size();
    std::vector<int32_t> verify_ids(1 + (size_t) n_d);
    verify_ids[0] = in.token;
    for (int i = 0; i < n_d; ++i) verify_ids[1 + (size_t) i] = drafts[(size_t) i];

    Inkling2ForwardOut vr = inkling2_forward(
        backend_, weights_, cache, verify_ids.data(), 1 + n_d, pos,
        /*verify=*/true, /*need_logits=*/false, /*capture_taps=*/true,
        /*allow_split=*/(weights_.expert_backend && weights_.split_exp > 0));
    if (!vr.ok || (int) vr.greedy.size() < 1 + n_d) return false;

    int n_acc = 0;
    int32_t final_tok = vr.greedy[0];
    std::vector<int32_t> committed;
    for (int i = 0; i < n_d; ++i) {
        const int32_t tgt = vr.greedy[(size_t) i];
        if (tgt == drafts[(size_t) i] && !token_is_eos(tgt)) {
            committed.push_back(tgt);
            ++n_acc;
            final_tok = vr.greedy[(size_t) (i + 1)];
        } else {
            final_tok = tgt;
            break;
        }
    }
    const int retained = n_acc + 1;
    if (!inkling2_verify_rollback(backend_, weights_, cache, pos, retained)) return false;
    if (ggml_tensor * taps = cache.verify_ws[1 + n_d].taps_out) {
        inkling2_dspark_commit_taps(backend_, weights_, cache, taps, pos, retained);
    }

    auto app = slots_->append_tokens(in.slot, verify_ids.data(), retained);
    if (!app.ok) return false;
    slots_->commit_step(in.slot);

    out.slot             = in.slot;
    out.committed_tokens = std::move(committed);
    out.token            = final_tok;
    out.spec_offered     = n_d;
    out.spec_accepted    = n_acc;
    out.failed           = false;
    return true;
}

SeqEngine::StepResult Inkling2SeqEngine::step_1d_(const StepPlan & plan) {
    StepResult res;

    for (const PrefillSlice & sl : plan.prefills) {
        const int s = sl.slot;
        SeqSlot & seq = slots_->slot(s);
        SlotState & sr = *slot_state_[(size_t) s];
        Inkling2Cache & cache = *caches_[(size_t) s];
        auto * kf = (cache.kvflash && cache.kvflash->active()) ? cache.kvflash.get() : nullptr;

        const int prompt_len = seq.prompt_len;
        const int pos0 = seq.cur_pos;
        int n = std::min(sl.max_tokens, prompt_len - pos0);
        if (sr.rewind_pos > pos0 && sr.rewind_pos < pos0 + n) {
            n = sr.rewind_pos - pos0;
        }
        if (seq.pending_capture.valid() &&
            seq.pending_capture.checkpoint.tokens > pos0 &&
            seq.pending_capture.checkpoint.tokens < pos0 + n) {
            n = seq.pending_capture.checkpoint.tokens - pos0;
        }
        const int end = pos0 + n;
        const bool is_last = (end == prompt_len);
        const bool is_greedy = sample_logits_is_argmax(seq.sampler);
        const int cap_q = (kf && sr.rewind_pos >= 0 && !sr.rewind_done && end > sr.q_cap_start)
            ? std::min(n, end - sr.q_cap_start) : 0;

        auto pc = slots_->append_prefill(s, n);
        if (!pc.ok) {
            res.error = "inkling2 append_prefill failed";
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        Inkling2SeqStepResult fr = run_forward_1d_(
            s, sr.prompt.data() + pos0, n, pos0,
            /*need_logits=*/is_last && !is_greedy, cap_q);
        if (!fr.ok) {
            res.error = fr.error.empty() ? "inkling2 prefill forward failed" : fr.error;
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        if (kf && sr.rewind_pos > 0 && end == sr.rewind_pos && !sr.rewind_done && !custom_forward_) {
            kf->save_rewind_snap(cache, weights_, sr.rewind_pos);
        }

        if (is_last && kf && sr.rewind_pos > 0 && !sr.rewind_done && kf->has_rewind_snap() && !custom_forward_) {
            sr.rewind_done = true;
            if (kf->mode4_score_and_recall(sr.rewind_pos, prompt_len)) {
                kf->restore_rewind_snap(cache, weights_);
                int rpos = sr.rewind_pos;
                while (rpos < prompt_len) {
                    const int rend = std::min(prompt_len, rpos + prefill_chunk_);
                    const int rn = rend - rpos;
                    const bool rlast = (rend == prompt_len);
                    fr = run_forward_1d_(s, sr.prompt.data() + rpos, rn, rpos, rlast && !is_greedy, 0);
                    if (!fr.ok) {
                        kf->clear_rewind_snap();
                        res.error = "inkling2 Mode-4 rewind prefill failed";
                        res.decode.clear();
                        res.prefills.clear();
                        return res;
                    }
                    rpos = rend;
                }
            }
            kf->clear_rewind_snap();
        }

        PrefillOutput po;
        po.slot = s;
        if (seq.pending_capture.valid() && end == seq.pending_capture.checkpoint.tokens) {
            const PrefixCaptureTicket ticket = seq.pending_capture;
            seq.pending_capture = {};
            po.prefix_store = capture_prefix_(s, ticket);
        }
        if (is_last) {
            slots_->commit_prefill(s);
            po.status = PrefillOutput::Status::completed;
            po.token  = sample_slot_1d_(seq, fr, is_greedy);
        } else {
            po.status = PrefillOutput::Status::advanced;
        }
        res.prefills.push_back(std::move(po));
    }

    const bool solo_spec_eligible = plan.prefills.empty() && plan.decode.size() == 1;
    for (const StepInput & in : plan.decode) {
        DecodeOutput dout;
        dout.slot = in.slot;

        if (solo_spec_eligible && try_solo_dspark_step_(in, dout)) {
            res.decode.push_back(std::move(dout));
            continue;
        }

        SeqSlot & seq = slots_->slot(in.slot);
        const int pos = seq.cur_pos;
        const bool is_greedy = sample_logits_is_argmax(seq.sampler);

        auto app = slots_->append_token(in.slot, in.token);
        if (!app.ok) {
            res.error = "inkling2 decode append_token failed";
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        Inkling2SeqStepResult fr = run_forward_1d_(
            in.slot, &in.token, 1, pos, /*need_logits=*/!is_greedy, /*capture_q_rows=*/0);
        if (!fr.ok) {
            slots_->rollback_step(in.slot);
            res.error = fr.error.empty() ? "inkling2 decode forward failed" : fr.error;
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        slots_->commit_step(in.slot);
        dout.token  = sample_slot_1d_(seq, fr, is_greedy);
        dout.failed = false;
        res.decode.push_back(std::move(dout));
    }

    return res;
}

// ─── Slot Offload / Restore / Evict / Retire ─────────────────────────────

size_t Inkling2SeqEngine::kv_offload_capacity() const {
    return state_adapter_->slab_bytes() * (size_t) std::max(0, slot_count() - 1);
}

SeqEngine::KvOffloadState Inkling2SeqEngine::kv_offload_state(int slot) const {
    KvOffloadState st;
    if (slot < 0 || slot >= slot_count()) return st;
    const SeqSlot & seq = slots_->slot(slot);
    st.parked    = seq.parked();
    st.recompute = seq.recomputing();
    st.bytes     = slot_state_[(size_t) slot]->parked_slab.size();
    return st;
}

bool Inkling2SeqEngine::offload_kv(int slot, size_t max_bytes, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    const size_t need = state_adapter_->slab_bytes();
    if (max_bytes > 0 && need > max_bytes) { error = "slab exceeds budget"; return false; }
    SlotStateSlab slab;
    if (!state_adapter_->save_slot_state(slot, slab)) {
        error = "failed to save Inkling2 slot state slab";
        return false;
    }
    if (!slots_->detach_kv(slot)) {
        error = "SeqSlotManager::detach_kv failed";
        return false;
    }
    reset_slot_session(slot);
    slot_state_[(size_t) slot]->parked_slab = std::move(slab);
    return true;
}

bool Inkling2SeqEngine::restore_kv(int slot, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    SeqSlot & seq = slots_->slot(slot);
    auto & st = *slot_state_[(size_t) slot];
    if (seq.recomputing()) {
        if (!slots_->resume_recompute(slot)) return false;
        reset_slot_session(slot);
        if (!custom_forward_) state_adapter_->reset_slot_state(slot);
        st.prompt = seq.sample_history;
        st.prefill_pos = 0;
        st.rewind_done = false;
        return true;
    }
    if (!slots_->attach_kv(slot)) return false;
    if (!st.parked_slab.empty()) {
        if (!state_adapter_->restore_slot_state(slot, st.parked_slab)) {
            error = "failed to restore Inkling2 slot state slab";
            return false;
        }
        st.parked_slab.clear();
    }
    return true;
}

bool Inkling2SeqEngine::evict_kv(int slot, int32_t pending_token, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    if (!slots_->evict_for_recompute(slot, pending_token)) {
        error = "SeqSlotManager::evict_for_recompute failed";
        return false;
    }
    reset_slot_session(slot);
    slot_state_[(size_t) slot]->parked_slab.clear();
    return true;
}

bool Inkling2SeqEngine::kv_restore_feasible(int slot) const {
    return slot >= 0 && slot < slot_count() && slots_->kv_restore_feasible(slot);
}

void Inkling2SeqEngine::retire(int slot) {
    if (slot >= 0 && slot < n_slots_) {
        reset_slot_session(slot);
        auto & st = *slot_state_[(size_t) slot];
        st.prompt.clear();
        st.prefill_pos = 0;
        st.rewind_pos  = -1;
        st.rewind_done = false;
        st.parked_slab.clear();
        if (!custom_forward_ && caches_[(size_t) slot]) {
            reset_inkling2_cache(*caches_[(size_t) slot]);
        }
    }
    slots_->retire(slot);
}

} // namespace luce::common
