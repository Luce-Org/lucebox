// Shared client-facing finish-reason policy for HTTP responses and logs.
#pragma once

#include <string>

namespace dflash::common {

// Resolve the public reason from the emitter's semantic reason and request
// outcome. `generation_cap < 0` disables the cap comparison for callers that
// do not have a generation budget. Disconnect and backend errors retain their
// existing log-only states and take precedence over normal completion.
inline std::string resolve_client_finish_reason(
        const std::string & emitter_reason,
        int completion_tokens,
        int generation_cap,
        bool degenerate_decode_close,
        bool result_ok = true,
        bool client_disconnected = false) {
    if (client_disconnected) return "client_disconnect";
    if (!result_ok) return "error";

    std::string reason = emitter_reason;
    if (reason == "stop" && generation_cap >= 0 &&
        completion_tokens >= generation_cap) {
        reason = "length";
    }
    // Preserve semantic terminal reasons such as tool_calls. A degenerate
    // decode only upgrades an otherwise normal stop to length.
    if (degenerate_decode_close && reason == "stop") reason = "length";
    return reason;
}

}  // namespace dflash::common
