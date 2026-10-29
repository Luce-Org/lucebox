#pragma once

// Admission gate: reject oversized requests with HTTP 400.
// When compression is enabled, a raw prompt up to --max-ctx is let through even if its reply does not fit — the
// post-compress check is the real gate.

#include <algorithm>

// Device memory concurrent serving leaves free beside the paged KV pool for the
// PFlash drafter, which scores on its own thread while the slots decode: its
// weights plus a workspace that grows linearly with the prompt it scores
// (activations, its own cache, the largest ubatch's attention). The
// per-token figure is the Qwen3.5-0.8B drafter's measured peak, 53 KiB per
// token from 33K to 206K tokens, rounded up; its fixed part (0.6 GiB) is
// under the weights.
inline constexpr long long kPFlashScoringBytesPerToken = 56LL * 1024;

inline long long pflash_concurrent_reserve_bytes(long long drafter_weight_bytes, int max_prompt_tokens)
{
    return (std::max)(0LL, drafter_weight_bytes) +
           (long long)(std::max)(0, max_prompt_tokens) * kPFlashScoringBytesPerToken;
}

// The reply a request can get: what it asked for, trimmed to the context its
// prompt leaves (0 when the prompt fills it). Clients ask for their usual cap
// whatever the prompt, Claude Code 32000 tokens, so a long session would
// otherwise be refused once prompt + cap passed the context.
inline int fit_reply_to_context(int prompt_tokens, int max_output, int max_ctx)
{
    return (std::max)(0, (std::min)(max_output, max_ctx - prompt_tokens));
}

// The reply an image request keeps room for while its images expand. The reply
// is trimmed afterwards to what the expanded prompt leaves, as a text request's
// is (fit_reply_to_context), so the images reserve at most the reply budget. The
// whole reply would leave them no room once a card's default reply nears the
// context, and refuse every image request without max_tokens once it passes it
// (DeepSeek V4.1's 262144 against its 204,800-token context).
inline int image_output_reserve(int max_output, int hard_limit_reply_budget)
{
    return (std::max)(0, (std::min)(max_output, hard_limit_reply_budget));
}

// Whether a request's reply is trimmed to the context its prompt leaves. A
// routed request keeps a reply its client sized, so the router can send it
// whole to a model with room for it. A reply the client left to the card's
// default (DeepSeek V4.1's 262144) may fit no model whole, so it is trimmed as
// a direct request's is. PFlash makes its own room.
inline bool trims_reply(bool routed, bool client_sized_reply, bool pflash_will_run)
{
    return !pflash_will_run && (!routed || !client_sized_reply);
}

inline bool should_reject_oversized(int prompt_tokens, int max_output,
                                    int max_ctx, bool compression_enabled)
{
    if (prompt_tokens + max_output <= max_ctx) {
        return false;  // fits — accept regardless of compression
    }
    // Oversized: reject if compression cannot help, or if the raw prompt is
    // longer than --max-ctx, the longest prompt PFlash takes (the drafter
    // scores it whole).
    if (!compression_enabled) return true;
    return prompt_tokens > max_ctx;
}

// Post-compress gate: check whether the effective context overflows max_ctx.
//
// effective_tokens          — raw effective prompt size (post FlowKV / pFlash rewrite, or
//                             req.prompt_tokens when a full-cache hit skipped compression).
// served_from_cache_tokens  — when > 0, a pFlash full-cache hit is serving this request;
//                             use this compressed size for the budget check instead of
//                             effective_tokens, because the cached KV was built from the
//                             compressed form and that is all that will be prefilled.
// max_output                — request's max_tokens.
// max_ctx                   — server context window.
//
// Returns true iff the request should be rejected with 400.
inline bool effective_prompt_overflows(int effective_tokens,
                                       int served_from_cache_tokens,
                                       int max_output,
                                       int max_ctx)
{
    // On a pFlash full-cache hit (sentinel: >=0; -1 = no hit) the KV state was
    // built from the compressed form; budget-check must use that size.
    const int check_tokens = (served_from_cache_tokens >= 0)
        ? served_from_cache_tokens
        : effective_tokens;
    return check_tokens + max_output > max_ctx;
}
