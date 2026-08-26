// Exact compact-state commit for Ling/Bailing KDA speculative decoding.

#pragma once

namespace dflash::common {

// Replay the first `accepted` captured KDA recurrence inputs into every
// durable state tensor. `state_ptrs_dev` is a device array containing one
// [state_dim, state_dim, n_heads] F32 pointer per recurrent layer. Factor
// buffers use token-major consolidated layouts:
//   k/v/g: [state_dim, n_heads, n_layers, max_tokens]
//   beta:  [n_heads, n_layers, max_tokens]
//
// The launch is asynchronous on `stream` (nullptr selects the default GPU
// stream). The caller must synchronize after enqueueing any companion conv
// state copies. `launched` distinguishes validation/launch rejection from a
// failure after an in-place kernel was accepted by the runtime.
bool kda_replayssm_commit_async(float * const * state_ptrs_dev,
                                const float * k,
                                const float * v,
                                const float * g,
                                const float * beta,
                                int accepted,
                                int state_dim,
                                int n_heads,
                                int n_layers,
                                int max_tokens,
                                void * stream,
                                bool * launched = nullptr);

}  // namespace dflash::common
