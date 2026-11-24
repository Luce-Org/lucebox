// Inkling2SlotStateAdapter implementation.

#include "inkling2_slot_state.h"

#include <cstring>

namespace luce::common {

namespace {

struct Inkling2SlabHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;          // bit 0: is_tree, bit 1: has_draft_ring, bit 2: has_shared_tail
    uint32_t n_layers;       // n_layer_phys (tree) or n_layer_eff (1D)
    int32_t  cur_pos;
    int32_t  prefix_len;
    int32_t  dspark_ctx_pos;
    int32_t  swa_cap;
    int32_t  shared_kv_window;
    int32_t  shared_tail_cap;
    uint64_t payload_bytes;
};

static size_t compute_cache_payload_bytes(const Inkling2Weights & w,
                                          const Inkling2Cache & cache,
                                          bool include_draft_ring) {
    size_t total = 0;
    if (cache.tree.active) {
        const auto & tc = cache.tree;
        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            if (ip < (int) tc.prefix_conv.size() && tc.prefix_conv[ip]) {
                total += ggml_nbytes(tc.prefix_conv[ip]);
            }
            if (w.layers[ip].is_swa) {
                if (ip < (int) tc.prefix_k.size() && tc.prefix_k[ip]) total += ggml_nbytes(tc.prefix_k[ip]);
                if (ip < (int) tc.prefix_v.size() && tc.prefix_v[ip]) total += ggml_nbytes(tc.prefix_v[ip]);
            } else if (tc.shared_kv_window > 0 && tc.planes[ip] > 1 && tc.shared_tail_cap > 0 &&
                       ip < (int) tc.prefix_k.size() && tc.prefix_k[ip] && tc.prefix_v[ip]) {
                const int n_tail = (tc.planes[ip] - 1) * tc.shared_tail_cap;
                total += (size_t) n_tail * tc.prefix_k[ip]->nb[1] * (size_t) w.n_head_kv;
                total += (size_t) n_tail * tc.prefix_v[ip]->nb[1] * (size_t) w.n_head_kv;
            }
        }
        if (include_draft_ring) {
            for (ggml_tensor * dk : tc.draft_k) if (dk) total += ggml_nbytes(dk);
            for (ggml_tensor * dv : tc.draft_v) if (dv) total += ggml_nbytes(dv);
        }
    } else {
        for (int il = 0; il < w.n_layer_eff; ++il) {
            if (il < (int) cache.conv_state.size() && cache.conv_state[il]) {
                total += ggml_nbytes(cache.conv_state[il]);
            }
            const int ip = w.phys_layer_for_slot(il);
            if (w.layers[ip].is_swa) {
                if (il < (int) cache.k_cache.size() && cache.k_cache[il]) total += ggml_nbytes(cache.k_cache[il]);
                if (il < (int) cache.v_cache.size() && cache.v_cache[il]) total += ggml_nbytes(cache.v_cache[il]);
            }
        }
    }
    if (cache.last_taps_dev) {
        total += ggml_nbytes(cache.last_taps_dev);
    }
    return total;
}

} // namespace

Inkling2SlotStateAdapter::Inkling2SlotStateAdapter(
        const Inkling2Weights & weights,
        std::vector<Inkling2Cache *> caches,
        bool include_draft_ring)
    : weights_(weights),
      caches_(std::move(caches)),
      include_draft_ring_(include_draft_ring) {}

bool Inkling2SlotStateAdapter::valid_slot_(int slot) const {
    return slot >= 0 && slot < (int) caches_.size() && caches_[(size_t) slot] != nullptr;
}

size_t Inkling2SlotStateAdapter::slab_bytes() const {
    if (caches_.empty() || !caches_[0]) return 0;
    return sizeof(Inkling2SlabHeader) +
           compute_cache_payload_bytes(weights_, *caches_[0], include_draft_ring_);
}

bool Inkling2SlotStateAdapter::save_slot_state(int slot, SlotStateSlab & out) const {
    if (!valid_slot_(slot)) return false;
    const Inkling2Cache & cache = *caches_[(size_t) slot];
    const size_t payload = compute_cache_payload_bytes(weights_, cache, include_draft_ring_);

    Inkling2SlabHeader hdr{};
    hdr.magic            = kMagic;
    hdr.version          = kVersion;
    hdr.flags            = (cache.tree.active ? 1u : 0u) |
                           (include_draft_ring_ ? 2u : 0u) |
                           ((cache.tree.active && cache.tree.shared_kv_window > 0) ? 4u : 0u);
    hdr.n_layers         = (uint32_t) (cache.tree.active ? weights_.n_layer_phys : weights_.n_layer_eff);
    hdr.cur_pos          = cache.cur_pos;
    hdr.prefix_len       = cache.tree.prefix_len;
    hdr.dspark_ctx_pos   = cache.dspark_ctx_pos;
    hdr.swa_cap          = cache.swa_cap;
    hdr.shared_kv_window = cache.tree.shared_kv_window;
    hdr.shared_tail_cap  = cache.tree.shared_tail_cap;
    hdr.payload_bytes    = (uint64_t) payload;

    out.resize(sizeof(Inkling2SlabHeader) + payload);
    std::memcpy(out.data(), &hdr, sizeof(hdr));
    uint8_t * dst = out.data() + sizeof(hdr);

    auto get_full = [&](ggml_tensor * t) {
        if (!t) return;
        const size_t n = ggml_nbytes(t);
        ggml_backend_tensor_get(t, dst, 0, n);
        dst += n;
    };

    if (cache.tree.active) {
        const auto & tc = cache.tree;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (ip < (int) tc.prefix_conv.size()) get_full(tc.prefix_conv[ip]);
            if (weights_.layers[ip].is_swa) {
                if (ip < (int) tc.prefix_k.size()) get_full(tc.prefix_k[ip]);
                if (ip < (int) tc.prefix_v.size()) get_full(tc.prefix_v[ip]);
            } else if (tc.shared_kv_window > 0 && tc.planes[ip] > 1 && tc.shared_tail_cap > 0 &&
                       ip < (int) tc.prefix_k.size() && tc.prefix_k[ip] && tc.prefix_v[ip]) {
                const int n_tail = (tc.planes[ip] - 1) * tc.shared_tail_cap;
                const size_t seg_k = (size_t) n_tail * tc.prefix_k[ip]->nb[1];
                const size_t seg_v = (size_t) n_tail * tc.prefix_v[ip]->nb[1];
                const size_t off_k = (size_t) tc.global_pool_cap * tc.prefix_k[ip]->nb[1];
                const size_t off_v = (size_t) tc.global_pool_cap * tc.prefix_v[ip]->nb[1];
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_get(tc.prefix_k[ip], dst, (size_t) h * tc.prefix_k[ip]->nb[2] + off_k, seg_k);
                    dst += seg_k;
                }
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_get(tc.prefix_v[ip], dst, (size_t) h * tc.prefix_v[ip]->nb[2] + off_v, seg_v);
                    dst += seg_v;
                }
            }
        }
        if (include_draft_ring_) {
            for (ggml_tensor * dk : tc.draft_k) get_full(dk);
            for (ggml_tensor * dv : tc.draft_v) get_full(dv);
        }
    } else {
        for (int il = 0; il < weights_.n_layer_eff; ++il) {
            if (il < (int) cache.conv_state.size()) get_full(cache.conv_state[il]);
            const int ip = weights_.phys_layer_for_slot(il);
            if (weights_.layers[ip].is_swa) {
                if (il < (int) cache.k_cache.size()) get_full(cache.k_cache[il]);
                if (il < (int) cache.v_cache.size()) get_full(cache.v_cache[il]);
            }
        }
    }
    get_full(cache.last_taps_dev);
    return (size_t) (dst - out.data()) == out.size();
}

bool Inkling2SlotStateAdapter::restore_slot_state(int slot, const SlotStateSlab & slab) {
    if (!valid_slot_(slot) || slab.size() < sizeof(Inkling2SlabHeader)) return false;
    Inkling2Cache & cache = *caches_[(size_t) slot];

    Inkling2SlabHeader hdr{};
    std::memcpy(&hdr, slab.data(), sizeof(hdr));
    if (hdr.magic != kMagic || hdr.version != kVersion) return false;
    const bool is_tree = (hdr.flags & 1u) != 0;
    const bool has_draft = (hdr.flags & 2u) != 0;
    if (is_tree != cache.tree.active) return false;
    if (hdr.swa_cap != cache.swa_cap ||
        hdr.shared_kv_window != cache.tree.shared_kv_window ||
        hdr.shared_tail_cap != cache.tree.shared_tail_cap) {
        return false;
    }

    const size_t expected = compute_cache_payload_bytes(weights_, cache, has_draft);
    if (hdr.payload_bytes != (uint64_t) expected ||
        slab.size() != sizeof(Inkling2SlabHeader) + expected) {
        return false;
    }

    const uint8_t * src = slab.data() + sizeof(hdr);
    auto set_full = [&](ggml_tensor * t) {
        if (!t) return;
        const size_t n = ggml_nbytes(t);
        ggml_backend_tensor_set(t, src, 0, n);
        src += n;
    };

    if (cache.tree.active) {
        auto & tc = cache.tree;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (ip < (int) tc.prefix_conv.size()) set_full(tc.prefix_conv[ip]);
            if (weights_.layers[ip].is_swa) {
                if (ip < (int) tc.prefix_k.size()) set_full(tc.prefix_k[ip]);
                if (ip < (int) tc.prefix_v.size()) set_full(tc.prefix_v[ip]);
            } else if (tc.shared_kv_window > 0 && tc.planes[ip] > 1 && tc.shared_tail_cap > 0 &&
                       ip < (int) tc.prefix_k.size() && tc.prefix_k[ip] && tc.prefix_v[ip]) {
                const int n_tail = (tc.planes[ip] - 1) * tc.shared_tail_cap;
                const size_t seg_k = (size_t) n_tail * tc.prefix_k[ip]->nb[1];
                const size_t seg_v = (size_t) n_tail * tc.prefix_v[ip]->nb[1];
                const size_t off_k = (size_t) tc.global_pool_cap * tc.prefix_k[ip]->nb[1];
                const size_t off_v = (size_t) tc.global_pool_cap * tc.prefix_v[ip]->nb[1];
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_set(tc.prefix_k[ip], src, (size_t) h * tc.prefix_k[ip]->nb[2] + off_k, seg_k);
                    src += seg_k;
                }
                for (int h = 0; h < weights_.n_head_kv; ++h) {
                    ggml_backend_tensor_set(tc.prefix_v[ip], src, (size_t) h * tc.prefix_v[ip]->nb[2] + off_v, seg_v);
                    src += seg_v;
                }
            }
        }
        if (has_draft) {
            for (ggml_tensor * dk : tc.draft_k) set_full(dk);
            for (ggml_tensor * dv : tc.draft_v) set_full(dv);
        }
        tc.prefix_len = hdr.prefix_len;
    } else {
        for (int il = 0; il < weights_.n_layer_eff; ++il) {
            if (il < (int) cache.conv_state.size()) set_full(cache.conv_state[il]);
            const int ip = weights_.phys_layer_for_slot(il);
            if (weights_.layers[ip].is_swa) {
                if (il < (int) cache.k_cache.size()) set_full(cache.k_cache[il]);
                if (il < (int) cache.v_cache.size()) set_full(cache.v_cache[il]);
            }
        }
    }
    set_full(cache.last_taps_dev);

    cache.cur_pos        = hdr.cur_pos;
    cache.dspark_ctx_pos = hdr.dspark_ctx_pos;
    return (size_t) (src - slab.data()) == slab.size();
}

void Inkling2SlotStateAdapter::reset_slot_state(int slot) {
    if (!valid_slot_(slot)) return;
    Inkling2Cache & cache = *caches_[(size_t) slot];
    reset_inkling2_cache(cache);
    // Also zero SWA rings and Shared-KV tail rings for deterministic reset semantics
    if (cache.tree.active) {
        auto & tc = cache.tree;
        for (int ip = 0; ip < weights_.n_layer_phys; ++ip) {
            if (weights_.layers[ip].is_swa) {
                if (ip < (int) tc.prefix_k.size() && tc.prefix_k[ip])
                    ggml_backend_tensor_memset(tc.prefix_k[ip], 0, 0, ggml_nbytes(tc.prefix_k[ip]));
                if (ip < (int) tc.prefix_v.size() && tc.prefix_v[ip])
                    ggml_backend_tensor_memset(tc.prefix_v[ip], 0, 0, ggml_nbytes(tc.prefix_v[ip]));
            }
        }
    } else {
        for (int il = 0; il < weights_.n_layer_eff; ++il) {
            const int ip = weights_.phys_layer_for_slot(il);
            if (weights_.layers[ip].is_swa) {
                if (il < (int) cache.k_cache.size() && cache.k_cache[il])
                    ggml_backend_tensor_memset(cache.k_cache[il], 0, 0, ggml_nbytes(cache.k_cache[il]));
                if (il < (int) cache.v_cache.size() && cache.v_cache[il])
                    ggml_backend_tensor_memset(cache.v_cache[il], 0, 0, ggml_nbytes(cache.v_cache[il]));
            }
        }
    }
}

int Inkling2SlotStateAdapter::preamble_end(const std::vector<int32_t> & prompt) const {
    if (system_msg_id_ < 0 || user_msg_id_ < 0 || prompt.size() < 2) return 0;
    if (prompt[0] != system_msg_id_) return 0;
    for (size_t i = 1; i < prompt.size(); ++i) {
        if (prompt[i] == user_msg_id_) return (int) i;
    }
    return 0;
}

} // namespace luce::common
