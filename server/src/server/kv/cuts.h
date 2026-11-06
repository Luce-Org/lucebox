// Where a request may checkpoint its prompt and where prefill is cut: the
// capture cut a request takes, and the restore points every request of the
// same prefix shares (GenerateRequest::restore_points, PrefixStorePlan).
// Pure functions of the prompt, its chat boundaries and the backend geometry.

#pragma once

#include "server/kv/chat_boundaries.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace luce::common {

// Pick the inline snapshot boundary for a request: the last marker, the end
// of the prompt's last message and the start of the generation prompt, only
// when it advances past an already-restored prefix. That is the point the
// next turn extends: a person's follow-up, or an agent's next step after its
// tool results. A person who edits that message resumes from the previous
// turn's snapshot instead (PrefixCache::prune_superseded_ancestors keeps it).
// When prefer_head is set (a conversation's first turn, or a tool-heavy agent
// request), prefer the first marker (the system/tools head) until that cut is
// already restored, so new conversations on the same head restore it.
// Returns 0 when there is no useful new boundary.
// Cuts below `reachable_from` (the first position the backend can save at,
// see ModelBackend::snapshot_granularity) are skipped.
int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len = 0,
                                    bool prefer_head = false,
                                    int reachable_from = 0);

// Restore points spaced at least `min_gap` tokens apart: the first boundary
// (the system/tools head), then each boundary at least `min_gap` past the last
// kept one. Prefix-stable: the points of a prompt are the points, up to its
// end, of every prompt that extends it, so a checkpoint saved at one is a
// restore point of every later turn. min_gap <= 0 keeps every boundary.
std::vector<int> spaced_restore_points(const std::vector<int> & boundaries, int min_gap);

// Every position a later request may restore `prompt`'s prefix from: its chat
// boundaries, spaced by `spacing` (spaced_restore_points), plus the extra cuts
// a cache may take (a PPP pin, a fixed disk scope), ascending. See
// GenerateRequest::restore_points. `drop_last_boundary` leaves out the
// generation prompt's own boundary: a request that does not snapshot there
// saves no state past it, so that split would only cost a short extra prefill
// step (a snapshot there comes back through `cuts`).
std::vector<int> prefix_restore_points(const std::vector<int32_t> & prompt,
                                       const ChatMarkers & markers,
                                       std::initializer_list<int> cuts,
                                       bool drop_last_boundary = false,
                                       int spacing = 0);

// Return true when a PPP forced cut should override normal boundary
// selection. Once the tools head is already restored, forcing the pin again
// prevents the cache from deepening into the conversation.
bool should_force_inline_snapshot_boundary(
    const std::vector<int> & boundaries,
    int prompt_len,
    int restored_prefix_len,
    bool prefer_tools_boundary,
    int forced_cut,
    int reachable_from = 0);

}  // namespace luce::common
