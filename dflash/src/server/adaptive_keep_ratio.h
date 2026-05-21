#pragma once
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dflash {

struct AdaptiveKeepRatioState {
    float ema        = 0.0f;
    float last_keep  = 0.10f;
    int   turn_count = 0;
};

constexpr float kBanditEmaAlpha   = 0.7f;
constexpr float kBanditTargetLo   = 0.75f;
constexpr float kBanditTargetHi   = 0.85f;
constexpr float kBanditStepSmall  = 0.005f;
constexpr float kBanditStepLarge  = 0.01f;
constexpr float kBanditKeepMin    = 0.025f;
constexpr float kBanditKeepMax    = 0.20f;
constexpr float kBanditEscalateLo = 0.70f;
constexpr float kBanditEscalateHi = 0.90f;

inline AdaptiveKeepRatioState step_adaptive_keep_ratio(
    const AdaptiveKeepRatioState& state, float observed_accept)
{
    AdaptiveKeepRatioState next = state;

    // First turn: seed EMA directly; later: alpha smoothing
    next.ema = (state.turn_count == 0)
        ? observed_accept
        : kBanditEmaAlpha * state.ema + (1.0f - kBanditEmaAlpha) * observed_accept;

    float delta = 0.0f;
    if (next.ema > kBanditTargetHi) {
        delta = (next.ema > kBanditEscalateHi) ? -kBanditStepLarge : -kBanditStepSmall;
    } else if (next.ema < kBanditTargetLo) {
        delta = (next.ema < kBanditEscalateLo) ? kBanditStepLarge : kBanditStepSmall;
    }
    next.last_keep  = std::clamp(state.last_keep + delta, kBanditKeepMin, kBanditKeepMax);
    next.turn_count = state.turn_count + 1;
    return next;
}

// Thread-safe per-session container
class HttpServerSessions {
public:
    void update(const std::string& session_id, float observed_accept) {
        std::lock_guard<std::mutex> lock(mu_);
        sessions_[session_id] = step_adaptive_keep_ratio(sessions_[session_id], observed_accept);
    }

    float get_keep_ratio(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        return (it == sessions_.end()) ? AdaptiveKeepRatioState{}.last_keep : it->second.last_keep;
    }

    int turn_count(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        return (it == sessions_.end()) ? 0 : it->second.turn_count;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return sessions_.size();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, AdaptiveKeepRatioState> sessions_;
};

}  // namespace dflash
