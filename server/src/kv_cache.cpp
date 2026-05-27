// Single-sequence KV cache for the target (64 layers) plus a rolling
// bf16 buffer for the concatenated target layer features that feed the draft.
//
// Layout:
//   target_k[layer] : [max_ctx, n_kv_heads, head_dim]   bf16
//   target_v[layer] : [max_ctx, n_kv_heads, head_dim]   bf16
//   target_feat     : [max_ctx, 5*hidden]               bf16
//
// Operations:
//   init(max_ctx)
//   reset()
//   reserve(n)            : ensure capacity, no write
//   commit(pos, n)        : mark positions [pos, pos+n) as valid (just bumps len_)
//   truncate(committed)   : drop everything >= committed (on mis-speculation)
//   length()
//
// The actual writes into target_k/v happen inside the target graph via
// ggml_cpy into the slot at offset = pos*bytes_per_token. Same for target_feat.

#include "internal.h"

namespace dflash::common {

// Placeholder; real impl lives with the spec_loop driver.

} // namespace dflash::common
