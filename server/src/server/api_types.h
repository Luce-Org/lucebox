// Shared types for the server components.
#pragma once

namespace dflash::common {

enum class ApiFormat { OPENAI_CHAT, ANTHROPIC, RESPONSES, COMPLETIONS };

// Log/status name of a format — shared by the request-tracing logs of the
// classic worker loop and the concurrent scheduler.
inline const char * api_format_name(ApiFormat format) {
    switch (format) {
    case ApiFormat::OPENAI_CHAT: return "chat";
    case ApiFormat::ANTHROPIC:   return "anthropic";
    case ApiFormat::RESPONSES:   return "responses";
    case ApiFormat::COMPLETIONS: return "completions";
    default:                     return "unknown";
    }
}

}  // namespace dflash::common
