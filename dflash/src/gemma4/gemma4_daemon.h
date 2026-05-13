// Gemma4 daemon library entry point.
//
// Thin wrapper around the generic daemon_loop.cpp (PR #175) and
// Gemma4Backend (PR #11b). Encapsulates the stdin/stream-fd protocol for
// the Gemma4 31B Dense / 26B-A4B MoE target so dispatch from the
// arch-routing test_dflash binary (or a standalone gemma4_daemon main)
// can hand off without duplicating the daemon loop.
//
// Mirrors run_laguna_daemon() exactly — same wire format, same return
// semantics, same banner protocol. Decode strategy (none / dflash / mtp)
// is configured via Gemma4DaemonArgs::draft_method.

#pragma once

#include <string>

#include "gemma4_backend.h"

namespace dflash27b {

struct Gemma4DaemonArgs {
    std::string target_path;            // path to gemma4-*.gguf (target)
    std::string draft_path;             // path to DFlash draft (kDFlash)
    std::string mtp_path;               // path to MTP assistant GGUF (kMtp)
    Gemma4DraftMethod draft_method = Gemma4DraftMethod::kNone;

    int  max_ctx       = 16384;
    int  chunk         = 2048;
    bool use_pflash    = false;
    float pflash_alpha = 0.12f;

    int  draft_kv_cap_override          = 0;
    int  draft_max_block                = 16;
    bool draft_enable_capture_overrides = false;
    int  mtp_gamma                      = 1;

    int  stream_fd = -1;                // server.py writable pipe end
};

// Boots the gemma4 target on a fresh CUDA backend, prints a
// `[gemma4-daemon] ready ...` banner on stdout, and services stdin
// commands until `quit`, `exit`, or EOF. Returns the process exit code
// (0 on clean shutdown).
int run_gemma4_daemon(const Gemma4DaemonArgs & args);

}  // namespace dflash27b
