#pragma once
// Admission gate: reject oversized requests with HTTP 400.
// When compression is enabled, lets oversized requests through — post-compress check is the real gate.

inline bool should_reject_oversized(int prompt_tokens, int max_output,
                                    int max_ctx, bool compression_enabled)
{
    if (prompt_tokens + max_output <= max_ctx) {
        return false;  // fits — accept regardless of compression
    }
    // Oversized: only reject if compression cannot help.
    return !compression_enabled;
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
