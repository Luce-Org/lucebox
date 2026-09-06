#pragma once
#include "server/sse_emitter.h"
#include <string>
#include <vector>

static dflash::common::SseEmitter make_emitter(
    dflash::common::ApiFormat fmt, nlohmann::json tools = nlohmann::json::array(),
    bool started_in_thinking = false) {
    return dflash::common::SseEmitter(fmt, "test_id_001", "test-model", 10,
                      tools, nullptr, {}, started_in_thinking);
}

// Concatenate all SSE chunks into a single string.
static std::string concat(const std::vector<std::string> & chunks) {
    std::string out;
    for (const auto & c : chunks) out += c;
    return out;
}

static dflash::common::SseEmitter make_emitter_with_stops(
    dflash::common::ApiFormat fmt, const std::vector<std::string> & stops) {
    return dflash::common::SseEmitter(fmt, "test_id_001", "test-model", 10,
                      nlohmann::json::array(), nullptr, stops);
}
