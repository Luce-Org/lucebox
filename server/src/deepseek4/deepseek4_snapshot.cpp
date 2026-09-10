// DeepSeek V4 KV snapshots: declare / fill / restore / bind / validate.
// See deepseek4_snapshot.h for the contract.

#include "deepseek4_snapshot.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace dflash::common {

const char * const kDeepSeek4SnapHcName     = "ds4_hc_state_snap";
const char * const kDeepSeek4SnapMetaName   = "ds4_snap_meta";
const char * const kDeepSeek4SnapLogitsName = "ds4_snap_last_logits";
const char * const kDeepSeek4SnapFeatName   = "ds4_snap_spec_feat";

namespace {

// Per-layer tensor name formats, in DeepSeek4Snapshot::LayerSnap order.
enum LayerTensor {
    kRawKv = 0, kCompKv, kIndexKv, kAttnStateKv, kAttnStateScore,
    kIdxStateKv, kIdxStateScore, kLayerTensorCount
};
const char * const kLayerTensorFmt[kLayerTensorCount] = {
    "ds4_snap_raw_kv_%d",
    "ds4_snap_comp_kv_%d",
    "ds4_snap_index_kv_%d",
    "ds4_snap_attn_cs_kv_%d",
    "ds4_snap_attn_cs_score_%d",
    "ds4_snap_idx_cs_kv_%d",
    "ds4_snap_idx_cs_score_%d",
};

ggml_tensor * find_named(ggml_context * ctx, const std::string & name) {
    if (!ctx || name.empty() || name.size() >= (size_t) GGML_MAX_NAME) return nullptr;
    return ggml_get_tensor(ctx, name.c_str());
}

ggml_tensor * new_named(ggml_context * ctx, ggml_tensor * t, const std::string & name) {
    if (!t) return nullptr;
    if (name.size() >= (size_t) GGML_MAX_NAME) return nullptr;
    ggml_set_name(t, name.c_str());
    return t;
}

// Same type/shape as `src` (full-capacity tensors: raw ring, states, HC).
ggml_tensor * declare_like(ggml_context * ctx, const ggml_tensor * src, const std::string & name) {
    if (!ctx || !src) return nullptr;
    return new_named(ctx, ggml_dup_tensor(ctx, const_cast<ggml_tensor *>(src)), name);
}

// Right-sized row prefix of a 2-D tensor. GGML tensors cannot have an empty
// physical dimension, so an empty logical prefix keeps one allocated row and
// copies zero bytes.
ggml_tensor * declare_rows(ggml_context * ctx, const ggml_tensor * src, int live_rows,
                           const std::string & name) {
    if (!ctx || !src || ggml_n_dims(src) > 2 || live_rows < 0 || live_rows > src->ne[1]) {
        return nullptr;
    }
    return new_named(ctx, ggml_new_tensor_2d(ctx, src->type, src->ne[0],
                                             std::max(1, live_rows)), name);
}

size_t prefix_bytes(const ggml_tensor * t, int rows) {
    if (!t || rows <= 0) return 0;
    return (size_t) rows * t->nb[1];
}

bool copy_prefix_from_backend(const ggml_tensor * src, ggml_tensor * dst, int rows) {
    if (!src || !dst || rows < 0) return false;
    const size_t bytes = prefix_bytes(src, rows);
    if (bytes > ggml_nbytes(src) || bytes > ggml_nbytes(dst)) return false;
    if (bytes > 0) ggml_backend_tensor_get(src, dst->data, 0, bytes);
    return true;
}

bool copy_prefix_to_backend(const ggml_tensor * src, ggml_tensor * dst, int rows) {
    if (!src || !dst || rows < 0) return false;
    const size_t bytes = prefix_bytes(src, rows);
    if (bytes > ggml_nbytes(src) || bytes > ggml_nbytes(dst)) return false;
    if (bytes > 0) ggml_backend_tensor_set(dst, src->data, 0, bytes);
    return true;
}

bool copy_from_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src || !dst || ggml_nbytes(src) != ggml_nbytes(dst)) return false;
    ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    return true;
}

bool copy_to_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src || !dst || ggml_nbytes(src) != ggml_nbytes(dst)) return false;
    ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    return true;
}

// Optional pair copy: both null is fine, one null is not.
bool copy_opt_from_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src && !dst) return true;
    return copy_from_backend(src, dst);
}
bool copy_opt_to_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src && !dst) return true;
    return copy_to_backend(src, dst);
}

bool tensors_compatible(const ggml_tensor * a, const ggml_tensor * b) {
    if (!!a != !!b) return false;
    if (!a) return true;
    if (a->type != b->type || ggml_n_dims(a) != ggml_n_dims(b)) return false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

// A right-sized zero/one-row tensor reports one logical GGML dimension while
// its full-capacity cache tensor reports two, so compare the physical row
// layout instead of ggml_n_dims().
bool prefix_tensors_compatible(const ggml_tensor * snap, const ggml_tensor * cache, int live_rows) {
    if (!!snap != !!cache) return false;
    if (!snap) return live_rows == 0;
    if (live_rows < 0 || cache->ne[1] <= 0 || snap->type != cache->type ||
        snap->ne[0] != cache->ne[0] || live_rows > cache->ne[1]) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        if (snap->ne[i] != cache->ne[i]) return false;
    }
    return snap->ne[1] == std::max(1, live_rows);
}

// Shape predicates used by validate(): a 2-D tensor with exact width/rows.
bool is_2d(const ggml_tensor * t, ggml_type type, int64_t ne0, int64_t ne1) {
    return t && t->type == type && t->ne[0] == ne0 && t->ne[1] == ne1 &&
           t->ne[2] == 1 && t->ne[3] == 1;
}

bool layer_snapshot_shape_ok(const DeepSeek4LayerGeometry & g,
                             const DeepSeek4Snapshot::LayerSnap & L,
                             int64_t comp_capacity,   // <= 0: skip capacity checks
                             std::string * why) {
    auto fail = [&](const char * what) { if (why) *why = what; return false; };
    if (L.n_comp < 0 || L.n_index_comp < 0) return fail("negative row count");
    if (!is_2d(L.raw_kv, GGML_TYPE_F16, g.head_dim, g.raw_rows)) return fail("raw window shape");
    if ((!!L.comp_kv) != g.has_comp) return fail("compressed rows presence");
    if ((!!L.attn_compressor.state_kv) != g.has_comp ||
        (!!L.attn_compressor.state_score) != g.has_comp) {
        return fail("attention compressor state presence");
    }
    if ((!!L.index_comp_kv) != g.has_index) return fail("indexer rows presence");
    if ((!!L.indexer_compressor.state_kv) != g.has_index ||
        (!!L.indexer_compressor.state_score) != g.has_index) {
        return fail("indexer compressor state presence");
    }
    if (!g.has_comp && L.n_comp != 0) return fail("compressed rows without capacity");
    if (!g.has_index && L.n_index_comp != 0) return fail("indexer rows without capacity");
    if (g.has_comp) {
        if (!is_2d(L.comp_kv, GGML_TYPE_F16, g.head_dim, std::max(1, L.n_comp))) {
            return fail("compressed rows shape");
        }
        if (!is_2d(L.attn_compressor.state_kv, GGML_TYPE_F32, g.comp_width, g.comp_state_rows) ||
            !is_2d(L.attn_compressor.state_score, GGML_TYPE_F32, g.comp_width, g.comp_state_rows)) {
            return fail("attention compressor state shape");
        }
        if (comp_capacity > 0 && L.n_comp > comp_capacity) return fail("compressed rows exceed capacity");
    }
    if (g.has_index) {
        if (!is_2d(L.index_comp_kv, GGML_TYPE_F16, g.index_dim, std::max(1, L.n_index_comp))) {
            return fail("indexer rows shape");
        }
        if (!is_2d(L.indexer_compressor.state_kv, GGML_TYPE_F32, g.index_state_width, g.index_state_rows) ||
            !is_2d(L.indexer_compressor.state_score, GGML_TYPE_F32, g.index_state_width, g.index_state_rows)) {
            return fail("indexer compressor state shape");
        }
        if (comp_capacity > 0 && L.n_index_comp > comp_capacity) return fail("indexer rows exceed capacity");
    }
    return true;
}

}  // namespace

std::string deepseek4_snapshot_tensor_name(const char * prefix, const char * base) {
    std::string name = prefix ? prefix : "";
    name += base;
    return name;
}

std::string deepseek4_snapshot_layer_tensor_name(const char * prefix, const char * fmt, int layer) {
    char buf[GGML_MAX_NAME];
    std::snprintf(buf, sizeof(buf), fmt, layer);
    return deepseek4_snapshot_tensor_name(prefix, buf);
}

size_t deepseek4_snapshot_tensor_count(int n_layer, bool with_aux) {
    return 1 + (size_t) std::max(0, n_layer) * kLayerTensorCount + (with_aux ? 3 : 0);
}

namespace {

// The aux tensors are sized from the aux lengths; the meta records the
// logical lengths. Variable lengths go in ne[1]: the ondisk layout
// fingerprint normalizes that dimension, so every snapshot shares one layout.
bool declare_aux_payload(ggml_context * ctx, const char * name_prefix,
                         const DeepSeek4SnapshotAux & aux, DeepSeek4Snapshot & out) {
    if (aux.n_logits > (size_t) std::numeric_limits<int>::max() ||
        aux.n_spec_feat > (size_t) std::numeric_limits<int>::max() ||
        (aux.n_logits > 0 && !aux.logits) || (aux.n_spec_feat > 0 && !aux.spec_feat)) {
        return false;
    }
    out.last_logits_snap = new_named(ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) std::max<size_t>(1, aux.n_logits)),
        deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapLogitsName));
    out.spec_feat_snap = new_named(ctx,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, (int64_t) std::max<size_t>(1, aux.n_spec_feat)),
        deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapFeatName));
    return out.last_logits_snap && out.spec_feat_snap;
}

}  // namespace

bool deepseek4_snapshot_declare(ggml_context * ctx,
                                const DeepSeek4Cache & cache,
                                const char * name_prefix,
                                const DeepSeek4SnapshotAux * aux,
                                DeepSeek4Snapshot & out) {
    if (!ctx || !cache.ctx || !cache.buf || !cache.hc_state ||
        cache.layers.size() != (size_t) cache.n_layer || cache.cur_pos < 0 ||
        cache.cur_pos > cache.max_ctx) {
        return false;
    }
    for (const auto & layer : cache.layers) {
        if (layer.n_comp < 0 || layer.n_index_comp < 0 ||
            (layer.comp_kv && layer.n_comp > layer.comp_kv->ne[1]) ||
            (!layer.comp_kv && layer.n_comp != 0) ||
            (layer.index_comp_kv && layer.n_index_comp > layer.index_comp_kv->ne[1]) ||
            (!layer.index_comp_kv && layer.n_index_comp != 0)) {
            return false;
        }
    }

    out.layers.assign((size_t) cache.n_layer, {});
    out.meta_snap = out.last_logits_snap = out.spec_feat_snap = nullptr;
    out.hc_state_snap = declare_like(
        ctx, cache.hc_state, deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapHcName));
    if (!out.hc_state_snap) return false;

    auto lname = [&](int which, int il) {
        return deepseek4_snapshot_layer_tensor_name(name_prefix, kLayerTensorFmt[which], il);
    };
    for (int il = 0; il < cache.n_layer; ++il) {
        const auto & src = cache.layers[(size_t) il];
        auto & dst = out.layers[(size_t) il];
        dst.n_comp = src.n_comp;
        dst.n_index_comp = src.n_index_comp;
        dst.raw_kv = declare_like(ctx, src.raw_kv, lname(kRawKv, il));
        dst.comp_kv = src.comp_kv
            ? declare_rows(ctx, src.comp_kv, src.n_comp, lname(kCompKv, il)) : nullptr;
        dst.index_comp_kv = src.index_comp_kv
            ? declare_rows(ctx, src.index_comp_kv, src.n_index_comp, lname(kIndexKv, il)) : nullptr;
        dst.attn_compressor.state_kv =
            declare_like(ctx, src.attn_compressor.state_kv, lname(kAttnStateKv, il));
        dst.attn_compressor.state_score =
            declare_like(ctx, src.attn_compressor.state_score, lname(kAttnStateScore, il));
        dst.indexer_compressor.state_kv =
            declare_like(ctx, src.indexer_compressor.state_kv, lname(kIdxStateKv, il));
        dst.indexer_compressor.state_score =
            declare_like(ctx, src.indexer_compressor.state_score, lname(kIdxStateScore, il));
        if (!dst.raw_kv ||
            (src.comp_kv && !dst.comp_kv) ||
            (src.index_comp_kv && !dst.index_comp_kv) ||
            (src.attn_compressor.state_kv && !dst.attn_compressor.state_kv) ||
            (src.attn_compressor.state_score && !dst.attn_compressor.state_score) ||
            (src.indexer_compressor.state_kv && !dst.indexer_compressor.state_kv) ||
            (src.indexer_compressor.state_score && !dst.indexer_compressor.state_score)) {
            return false;
        }
    }

    if (aux) {
        out.meta_snap = new_named(ctx,
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                               kDeepSeek4SnapMetaBase + 2 * (int64_t) cache.n_layer),
            deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapMetaName));
        if (!out.meta_snap || !declare_aux_payload(ctx, name_prefix, *aux, out)) return false;
    }
    return true;
}

bool deepseek4_snapshot_fill(const DeepSeek4Cache & cache,
                             const DeepSeek4SnapshotAux * aux,
                             DeepSeek4Snapshot & out) {
    if (!cache.ctx || !cache.hc_state || out.layers.size() != cache.layers.size() ||
        !out.hc_state_snap || !out.hc_state_snap->data) {
        return false;
    }
    if (!copy_from_backend(cache.hc_state, out.hc_state_snap)) return false;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const auto & src = cache.layers[il];
        auto & dst = out.layers[il];
        dst.n_comp = src.n_comp;
        dst.n_index_comp = src.n_index_comp;
        if (!copy_from_backend(src.raw_kv, dst.raw_kv) ||
            (src.comp_kv && !copy_prefix_from_backend(src.comp_kv, dst.comp_kv, src.n_comp)) ||
            (src.index_comp_kv &&
             !copy_prefix_from_backend(src.index_comp_kv, dst.index_comp_kv, src.n_index_comp)) ||
            !copy_opt_from_backend(src.attn_compressor.state_kv, dst.attn_compressor.state_kv) ||
            !copy_opt_from_backend(src.attn_compressor.state_score, dst.attn_compressor.state_score) ||
            !copy_opt_from_backend(src.indexer_compressor.state_kv, dst.indexer_compressor.state_kv) ||
            !copy_opt_from_backend(src.indexer_compressor.state_score, dst.indexer_compressor.state_score)) {
            return false;
        }
    }

    if (out.meta_snap) {
        if (!out.last_logits_snap || !out.spec_feat_snap) return false;
        const size_t n_logits = aux ? aux->n_logits : 0;
        const size_t n_feat = aux ? aux->n_spec_feat : 0;
        if ((int64_t) std::max<size_t>(1, n_logits) != ggml_nelements(out.last_logits_snap) ||
            (int64_t) std::max<size_t>(1, n_feat) != ggml_nelements(out.spec_feat_snap)) {
            return false;  // declared with a different aux
        }
        std::vector<int32_t> meta((size_t) (kDeepSeek4SnapMetaBase + 2 * cache.n_layer), 0);
        meta[0] = kDeepSeek4SnapMetaVersion;
        meta[1] = cache.n_layer;
        meta[2] = (int32_t) n_logits;
        meta[3] = (int32_t) n_feat;
        meta[4] = cache.cur_pos;
        for (int il = 0; il < cache.n_layer; ++il) {
            meta[(size_t) (kDeepSeek4SnapMetaBase + 2 * il)]     = cache.layers[(size_t) il].n_comp;
            meta[(size_t) (kDeepSeek4SnapMetaBase + 2 * il + 1)] = cache.layers[(size_t) il].n_index_comp;
        }
        ggml_backend_tensor_set(out.meta_snap, meta.data(), 0, meta.size() * sizeof(int32_t));
        if (n_logits > 0) {
            ggml_backend_tensor_set(out.last_logits_snap, aux->logits, 0, n_logits * sizeof(float));
        } else {
            const float zero = 0.0f;
            ggml_backend_tensor_set(out.last_logits_snap, &zero, 0, sizeof(zero));
        }
        if (n_feat > 0) {
            ggml_backend_tensor_set(out.spec_feat_snap, aux->spec_feat, 0, n_feat * sizeof(float));
        } else {
            const float zero = 0.0f;
            ggml_backend_tensor_set(out.spec_feat_snap, &zero, 0, sizeof(zero));
        }
    }
    out.cur_pos = cache.cur_pos;
    return true;
}

bool deepseek4_snapshot_save(const DeepSeek4Cache & cache,
                             ggml_backend_t snapshot_backend,
                             DeepSeek4Snapshot & out,
                             const DeepSeek4SnapshotAux * aux) {
    free_deepseek4_snapshot(out);
    if (!snapshot_backend) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() *
                  (deepseek4_snapshot_tensor_count(cache.n_layer, aux != nullptr) + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    if (!deepseek4_snapshot_declare(ctx, cache, nullptr, aux, out)) {
        ggml_free(ctx);
        out = DeepSeek4Snapshot{};
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, snapshot_backend);
    if (!buf) {
        ggml_free(ctx);
        out = DeepSeek4Snapshot{};
        return false;
    }
    // Placeholder rows (empty logical prefixes) are zeroed so serialized
    // bytes are deterministic.
    ggml_backend_buffer_clear(buf, 0);
    out.ctx = ctx;
    out.buf = buf;
    out.owns_storage = true;
    if (!deepseek4_snapshot_fill(cache, aux, out)) {
        free_deepseek4_snapshot(out);
        return false;
    }
    return true;
}

bool deepseek4_snapshot_restore(const DeepSeek4Snapshot & snap,
                                DeepSeek4Cache & cache) {
    if (!snap.ctx || !cache.ctx || !cache.buf || !snap.hc_state_snap ||
        snap.layers.size() != cache.layers.size() || snap.cur_pos < 0 ||
        snap.cur_pos > cache.max_ctx) {
        std::fprintf(stderr,
                     "[deepseek4] snapshot restore: invalid header "
                     "(snap_ctx=%d cache_ctx=%d snap_layers=%zu "
                     "cache_layers=%zu pos=%d max_ctx=%d)\n",
                     snap.ctx != nullptr, cache.ctx != nullptr,
                     snap.layers.size(), cache.layers.size(),
                     snap.cur_pos, cache.max_ctx);
        return false;
    }
    if (!tensors_compatible(snap.hc_state_snap, cache.hc_state)) {
        std::fprintf(stderr, "[deepseek4] snapshot restore: incompatible HC state\n");
        return false;
    }

    // Validate the complete layout before changing the live cache. Compressed
    // tensors are deliberately right-sized to their logical row counts;
    // inactive capacity rows are not part of the snapshot contract.
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const auto & src = snap.layers[il];
        const auto & dst = cache.layers[il];
        const bool raw_ok = tensors_compatible(src.raw_kv, dst.raw_kv);
        const bool comp_ok = prefix_tensors_compatible(src.comp_kv, dst.comp_kv, src.n_comp);
        const bool index_ok = prefix_tensors_compatible(src.index_comp_kv, dst.index_comp_kv, src.n_index_comp);
        const bool attn_kv_ok = tensors_compatible(src.attn_compressor.state_kv, dst.attn_compressor.state_kv);
        const bool attn_score_ok = tensors_compatible(src.attn_compressor.state_score, dst.attn_compressor.state_score);
        const bool index_kv_ok = tensors_compatible(src.indexer_compressor.state_kv, dst.indexer_compressor.state_kv);
        const bool index_score_ok = tensors_compatible(src.indexer_compressor.state_score, dst.indexer_compressor.state_score);
        if (!raw_ok || !comp_ok || !index_ok || !attn_kv_ok ||
            !attn_score_ok || !index_kv_ok || !index_score_ok) {
            std::fprintf(stderr,
                         "[deepseek4] snapshot restore: incompatible layer %zu "
                         "(raw=%d comp=%d[%d/%lld/%lld] "
                         "index=%d[%d/%lld/%lld] states=%d/%d/%d/%d)\n",
                         il, raw_ok, comp_ok, src.n_comp,
                         (long long) (src.comp_kv ? src.comp_kv->ne[1] : 0),
                         (long long) (dst.comp_kv ? dst.comp_kv->ne[1] : 0),
                         index_ok, src.n_index_comp,
                         (long long) (src.index_comp_kv ? src.index_comp_kv->ne[1] : 0),
                         (long long) (dst.index_comp_kv ? dst.index_comp_kv->ne[1] : 0),
                         attn_kv_ok, attn_score_ok, index_kv_ok, index_score_ok);
            return false;
        }
    }

    if (!copy_to_backend(snap.hc_state_snap, cache.hc_state)) {
        std::fprintf(stderr, "[deepseek4] snapshot restore: HC copy failed\n");
        return false;
    }
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const auto & src = snap.layers[il];
        auto & dst = cache.layers[il];
        if (!copy_to_backend(src.raw_kv, dst.raw_kv) ||
            (src.comp_kv && !copy_prefix_to_backend(src.comp_kv, dst.comp_kv, src.n_comp)) ||
            (src.index_comp_kv &&
             !copy_prefix_to_backend(src.index_comp_kv, dst.index_comp_kv, src.n_index_comp)) ||
            !copy_opt_to_backend(src.attn_compressor.state_kv, dst.attn_compressor.state_kv) ||
            !copy_opt_to_backend(src.attn_compressor.state_score, dst.attn_compressor.state_score) ||
            !copy_opt_to_backend(src.indexer_compressor.state_kv, dst.indexer_compressor.state_kv) ||
            !copy_opt_to_backend(src.indexer_compressor.state_score, dst.indexer_compressor.state_score)) {
            std::fprintf(stderr, "[deepseek4] snapshot restore: layer %zu copy failed\n", il);
            return false;
        }
        dst.n_comp = src.n_comp;
        dst.n_index_comp = src.n_index_comp;
    }
    cache.cur_pos = snap.cur_pos;
    return true;
}

void free_deepseek4_snapshot(DeepSeek4Snapshot & s) {
    if (s.owns_storage) {
        if (s.buf) ggml_backend_buffer_free(s.buf);
        if (s.ctx) ggml_free(s.ctx);
    }
    s = DeepSeek4Snapshot{};
}

bool deepseek4_snapshot_bind(ggml_context * ctx,
                             ggml_backend_buffer_t buf,
                             const char * name_prefix,
                             bool take_ownership,
                             DeepSeek4Snapshot & out,
                             DeepSeek4SnapshotBindInfo * info) {
    free_deepseek4_snapshot(out);
    if (!ctx || !buf) return false;

    // Bound the meta length before allocating from it: a corrupt file must
    // not drive the read size.
    constexpr int64_t kMaxLayers = 4096;
    constexpr int64_t kMaxMetaLen = kDeepSeek4SnapMetaBase + 2 * kMaxLayers;
    ggml_tensor * meta = find_named(ctx, deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapMetaName));
    if (!meta || meta->type != GGML_TYPE_I32 || ggml_n_dims(meta) != 1 ||
        meta->ne[0] < kDeepSeek4SnapMetaBase || meta->ne[0] > kMaxMetaLen || !meta->data) {
        return false;
    }
    std::vector<int32_t> m((size_t) meta->ne[0], 0);
    ggml_backend_tensor_get(meta, m.data(), 0, m.size() * sizeof(int32_t));
    if (m[0] != kDeepSeek4SnapMetaVersion) return false;
    const int n_layer = m[1], n_vocab = m[2], n_spec_feat = m[3], cur_pos = m[4];
    if (n_layer <= 0 || n_layer > kMaxLayers || n_vocab < 0 || n_spec_feat < 0 || cur_pos < 0 ||
        meta->ne[0] != (int64_t) (kDeepSeek4SnapMetaBase + 2 * n_layer)) {
        return false;
    }

    DeepSeek4Snapshot tmp;
    tmp.meta_snap = meta;
    tmp.hc_state_snap = find_named(ctx, deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapHcName));
    tmp.last_logits_snap = find_named(ctx, deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapLogitsName));
    tmp.spec_feat_snap = find_named(ctx, deepseek4_snapshot_tensor_name(name_prefix, kDeepSeek4SnapFeatName));
    if (!tmp.hc_state_snap || !tmp.last_logits_snap || !tmp.spec_feat_snap ||
        tmp.last_logits_snap->type != GGML_TYPE_F32 || tmp.spec_feat_snap->type != GGML_TYPE_F32 ||
        ggml_nelements(tmp.last_logits_snap) != (int64_t) std::max(1, n_vocab) ||
        ggml_nelements(tmp.spec_feat_snap) != (int64_t) std::max(1, n_spec_feat)) {
        return false;
    }

    tmp.layers.resize((size_t) n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = tmp.layers[(size_t) il];
        auto lookup = [&](int which) {
            return find_named(ctx, deepseek4_snapshot_layer_tensor_name(name_prefix, kLayerTensorFmt[which], il));
        };
        L.raw_kv = lookup(kRawKv);
        L.comp_kv = lookup(kCompKv);
        L.index_comp_kv = lookup(kIndexKv);
        L.attn_compressor.state_kv = lookup(kAttnStateKv);
        L.attn_compressor.state_score = lookup(kAttnStateScore);
        L.indexer_compressor.state_kv = lookup(kIdxStateKv);
        L.indexer_compressor.state_score = lookup(kIdxStateScore);
        L.n_comp = m[(size_t) (kDeepSeek4SnapMetaBase + 2 * il)];
        L.n_index_comp = m[(size_t) (kDeepSeek4SnapMetaBase + 2 * il + 1)];
        if (!L.raw_kv || L.n_comp < 0 || L.n_index_comp < 0 ||
            (!L.comp_kv && L.n_comp != 0) || (!L.index_comp_kv && L.n_index_comp != 0) ||
            (L.comp_kv && L.comp_kv->ne[1] != std::max(1, L.n_comp)) ||
            (L.index_comp_kv && L.index_comp_kv->ne[1] != std::max(1, L.n_index_comp))) {
            return false;
        }
    }
    // Every tensor must be backed by host memory (CPU snapshot buffer).
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data) return false;
    }

    out = tmp;
    out.cur_pos = cur_pos;
    out.ctx = ctx;
    out.buf = buf;
    out.owns_storage = take_ownership;
    if (info) {
        info->n_layer = n_layer;
        info->n_vocab = n_vocab;
        info->n_spec_feat = n_spec_feat;
        info->cur_pos = cur_pos;
    }
    return true;
}

bool deepseek4_snapshot_validate(const DeepSeek4Weights & w,
                                 int max_ctx,
                                 const DeepSeek4Snapshot & snap,
                                 std::string * why) {
    auto fail = [&](const std::string & what) { if (why) *why = what; return false; };
    if (w.n_layer <= 0 || w.compress_ratios.size() != (size_t) w.n_layer) {
        return fail("weights not loaded");
    }
    if (snap.layers.size() != (size_t) w.n_layer) return fail("layer count mismatch");
    if (snap.cur_pos < 0 || (max_ctx > 0 && snap.cur_pos > max_ctx)) return fail("position out of range");
    if (!snap.hc_state_snap || snap.hc_state_snap->type != GGML_TYPE_F32 ||
        ggml_nelements(snap.hc_state_snap) != deepseek4_hc_state_elements(w)) {
        return fail("HC state shape");
    }
    // The logits sidecar is either empty (one placeholder element, e.g. the
    // layer-split shards, which keep logits at the adapter level) or a full
    // vocabulary row. Callers that need logits check the meta's n_vocab.
    if (snap.last_logits_snap &&
        (snap.last_logits_snap->type != GGML_TYPE_F32 ||
         (ggml_nelements(snap.last_logits_snap) != 1 &&
          ggml_nelements(snap.last_logits_snap) != (int64_t) w.n_vocab))) {
        return fail("logits shape");
    }
    if (snap.spec_feat_snap &&
        (snap.spec_feat_snap->type != GGML_TYPE_F32 || snap.spec_feat_snap->ne[0] != 1)) {
        return fail("feature window shape");
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4LayerGeometry g = deepseek4_layer_geometry(w, il);
        std::string layer_why;
        if (!layer_snapshot_shape_ok(g, snap.layers[(size_t) il],
                                     max_ctx > 0 ? g.comp_capacity(max_ctx) : 0, &layer_why)) {
            return fail("layer " + std::to_string(il) + ": " + layer_why);
        }
    }
    return true;
}

}  // namespace dflash::common
