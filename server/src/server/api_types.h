// Shared types for the server components.
#pragma once

namespace luce::common {

enum class ApiFormat {
    OPENAI_CHAT,
    ANTHROPIC,
    RESPONSES,
    COMPLETIONS,
    SYSTEMONE,
};

// Log/status name of a format — shared by the request-tracing logs of the
// classic worker loop and the concurrent scheduler.
inline const char * api_format_name(ApiFormat format) {
    switch (format) {
    case ApiFormat::OPENAI_CHAT: return "chat";
    case ApiFormat::ANTHROPIC:   return "anthropic";
    case ApiFormat::RESPONSES:   return "responses";
    case ApiFormat::COMPLETIONS: return "completions";
    case ApiFormat::SYSTEMONE:   return "systemone";
    default:                     return "unknown";
    }
}

}  // namespace luce::common
