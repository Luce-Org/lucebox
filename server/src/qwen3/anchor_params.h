// Adaptive anchor_radius / max_anchor_hits resolver — pure, no IO, testable.
#pragma once

namespace dflash::common {

struct AnchorParams { int radius; int max_hits; };

// Resolve anchor params for n_chunks (chunk_size=32).
// Tier: <1024 -> {2,8}; <2048 -> {4,16}; >=2048 -> {8,32}.
// env_* / legacy_* are the parsed env values; pass -1 when unset.
// Precedence: PFLASH env >= legacy DFLASH env >= tier default.
inline AnchorParams resolve_anchor_params(
    int n_chunks,
    int env_radius, int env_hits,
    int legacy_radius, int legacy_hits)
{
    int r, h;
    if (n_chunks >= 2048)      { r = 8; h = 32; }
    else if (n_chunks >= 1024) { r = 4; h = 16; }
    else                       { r = 2; h =  8; }

    if      (env_radius    >= 0) r = env_radius;
    else if (legacy_radius >= 0) r = legacy_radius;

    if      (env_hits    >= 0) h = env_hits;
    else if (legacy_hits >= 0) h = legacy_hits;

    return { r, h };
}

} // namespace dflash::common
