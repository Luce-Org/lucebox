// DeepSeek V4 KV snapshots.
//
// A snapshot is a right-sized, host-resident copy of a DeepSeek4Cache: the raw
// sliding-window rows, the compressed rows, the indexer rows, the compressor
// states and the HC residual state, plus an optional aux sidecar with the
// host-side decode state (last logits, DSpark feature tail) and an I32 meta
// tensor carrying the per-layer row counts and the cache position.
//
// Every tensor has a stable name, so the ondisk prefix cache can serialize a
// snapshot context and deepseek4_snapshot_bind() can rebind it after a reload.
//
// Two ways to build one:
//   * deepseek4_snapshot_save(): own context + buffer (monolithic backend,
//     DSpark rollback, IPC target shards).
//   * deepseek4_snapshot_declare() + deepseek4_snapshot_fill(): several
//     snapshots share one caller-owned context (layer-split adapter merges
//     all shards into a single serializable context, no second copy).
//
// deepseek4_snapshot_validate() checks every tensor against the geometry the
// weights imply, so a malformed or foreign file is rejected before it is
// adopted.
#pragma once

#include "deepseek4_internal.h"

#include <string>

namespace dflash::common {

// Host-side decode state persisted next to the cache tensors so an adopted
// (deserialized) snapshot resumes exactly like an in-memory one.
struct DeepSeek4SnapshotAux {
    const float * logits = nullptr;
    size_t        n_logits = 0;
    const float * spec_feat = nullptr;
    size_t        n_spec_feat = 0;
};

// Layout of DeepSeek4Snapshot::meta_snap (I32):
//   [0] version, [1] n_layer, [2] n_vocab, [3] n_spec_feat, [4] cur_pos,
//   then per layer: n_comp, n_index_comp.
constexpr int kDeepSeek4SnapMetaVersion = 1;
constexpr int kDeepSeek4SnapMetaBase = 5;

// Metadata recovered by deepseek4_snapshot_bind().
struct DeepSeek4SnapshotBindInfo {
    int n_layer = 0;
    int n_vocab = 0;
    int n_spec_feat = 0;
    int cur_pos = 0;
};

// Tensor names. `prefix` (may be null) is prepended verbatim; the layer-split
// adapter uses "ls<shard>_" so several shards share one context.
std::string deepseek4_snapshot_tensor_name(const char * prefix, const char * base);
std::string deepseek4_snapshot_layer_tensor_name(const char * prefix, const char * fmt, int layer);
extern const char * const kDeepSeek4SnapHcName;
extern const char * const kDeepSeek4SnapMetaName;
extern const char * const kDeepSeek4SnapLogitsName;
extern const char * const kDeepSeek4SnapFeatName;

// Number of ggml tensors one snapshot declares (context sizing).
size_t deepseek4_snapshot_tensor_count(int n_layer, bool with_aux);

// Create the snapshot tensors in `ctx` (a no_alloc context) without copying
// any data. `aux == nullptr` declares no sidecar; otherwise the sidecar is
// sized from the aux lengths (which may be 0). `out` describes the tensors
// afterwards; the caller allocates a buffer for `ctx`, then calls
// deepseek4_snapshot_fill() with the same `aux`. `out.ctx`/`out.buf`/
// `out.owns_storage` are left for the caller to set.
bool deepseek4_snapshot_declare(ggml_context * ctx,
                                const DeepSeek4Cache & cache,
                                const char * name_prefix,
                                const DeepSeek4SnapshotAux * aux,
                                DeepSeek4Snapshot & out);

// Copy the live cache (and the aux the snapshot was declared with) into the
// allocated snapshot tensors and set `out.cur_pos`.
bool deepseek4_snapshot_fill(const DeepSeek4Cache & cache,
                             const DeepSeek4SnapshotAux * aux,
                             DeepSeek4Snapshot & out);

// declare + allocate on `snapshot_backend` + fill, in a context owned by `out`.
// `aux == nullptr` produces a snapshot without the aux sidecar (not
// exportable to disk; used by DSpark rollback and IPC target shards).
bool deepseek4_snapshot_save(const DeepSeek4Cache & cache,
                             ggml_backend_t snapshot_backend,
                             DeepSeek4Snapshot & out,
                             const DeepSeek4SnapshotAux * aux = nullptr);

bool deepseek4_snapshot_restore(const DeepSeek4Snapshot & snap,
                                DeepSeek4Cache & cache);

// Rebind a deserialized context (tensors named as declared above, optionally
// prefixed) into `out` using the meta tensor for row counts and position.
// Structural only; call deepseek4_snapshot_validate() for types and shapes.
// `out` takes ownership of ctx/buf iff `take_ownership`. Returns false and
// leaves `out` empty on any inconsistency.
bool deepseek4_snapshot_bind(ggml_context * ctx,
                             ggml_backend_buffer_t buf,
                             const char * name_prefix,
                             bool take_ownership,
                             DeepSeek4Snapshot & out,
                             DeepSeek4SnapshotBindInfo * info);

// Check every snapshot tensor against the geometry `w` implies (types, widths,
// state rows, HC size, vocab) and, when `max_ctx > 0`, that the row counts fit
// a cache of that size. `why` receives a short reason on failure.
bool deepseek4_snapshot_validate(const DeepSeek4Weights & w,
                                 int max_ctx,
                                 const DeepSeek4Snapshot & snap,
                                 std::string * why);

void free_deepseek4_snapshot(DeepSeek4Snapshot & s);

}  // namespace dflash::common
