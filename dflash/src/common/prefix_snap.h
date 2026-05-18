// prefix_snap.h — Common prefix-cache snapshot helpers.
//
// Single source of truth for the daemon→server ack format that the daemon
// emits when an inline prefix-cache snapshot lands. server.py's bus listens
// for "[snap] inline slot={N} " on stdout — anything else and the slot
// reservation gets dropped ("inline snapshot ack missing").
//
// Today: format helper only. As more backends grow inline-snap call sites
// (DFlash do_prefill, MTP warm_and_decode, future arches), they all funnel
// through here so the format can never diverge.

#pragma once

#include <cstdio>

namespace dflash27b {
namespace common {

inline void emit_inline_snap_ack(int slot, int cur_pos) {
    std::printf("[snap] inline slot=%d cur_pos=%d\n", slot, cur_pos);
    std::fflush(stdout);
}

}  // namespace common
}  // namespace dflash27b
