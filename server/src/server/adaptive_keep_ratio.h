#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dflash::common {

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

constexpr std::size_t kMaxSessions = 1024;

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

// Thread-safe per-session container with LRU eviction.
//
// Bounds memory to at most max_sessions entries (env: DFLASH_BANDIT_MAX_SESSIONS,
// default 1024). When the cap is reached, the least-recently-used session is
// evicted so long-running servers don't accumulate unbounded state.
class HttpServerSessions {
public:
    explicit HttpServerSessions(size_t max_sessions = 0) {
        if (max_sessions != 0) {
            max_sessions_ = max_sessions;
        } else {
            const char* env = std::getenv("DFLASH_BANDIT_MAX_SESSIONS");
            max_sessions_ = (env && *env) ? static_cast<size_t>(std::atol(env)) : 1024;
        }
        if (max_sessions_ == 0) max_sessions_ = 1024;  // guard against env=0
    }

    void update(const std::string& session_id, float observed_accept) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            evict_if_full_locked();
            lru_.push_front(session_id);
            auto [ins, _] = sessions_.emplace(session_id,
                Entry{AdaptiveKeepRatioState{}, lru_.begin()});
            ins->second.state = step_adaptive_keep_ratio(ins->second.state, observed_accept);
        } else {
            touch_locked(it->second.lru_it);
            it->second.state = step_adaptive_keep_ratio(it->second.state, observed_accept);
        }
    }

    float get_keep_ratio(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return AdaptiveKeepRatioState{}.last_keep;
        touch_locked(it->second.lru_it);
        return it->second.state.last_keep;
    }

    float get_ema(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return 0.0f;
        touch_locked(it->second.lru_it);
        return it->second.state.ema;
    }

    int turn_count(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return 0;
        touch_locked(it->second.lru_it);
        return it->second.state.turn_count;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return sessions_.size();
    }

    size_t max_sessions() const { return max_sessions_; }

private:
    struct Entry {
        AdaptiveKeepRatioState       state;
        std::list<std::string>::iterator lru_it;
    };

    // Move an existing LRU entry to the front (most-recently-used).
    // Must be called with mu_ held.
    void touch_locked(std::list<std::string>::iterator it) const {
        lru_.splice(lru_.begin(), lru_, it);
    }

    // Evict LRU entry if the map is at capacity.
    // Must be called with mu_ held.
    void evict_if_full_locked() {
        if (sessions_.size() < max_sessions_) return;
        sessions_.erase(lru_.back());
        lru_.pop_back();
    }

    size_t                                        max_sessions_;
    mutable std::mutex                            mu_;
    mutable std::list<std::string>                lru_;       // front = MRU, back = LRU
    std::unordered_map<std::string, Entry>        sessions_;
};

}  // namespace dflash::common
