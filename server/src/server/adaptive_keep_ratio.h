#pragma once
#include <algorithm>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dflash::common {

struct AdaptiveKeepRatioState {
    float ema              = 0.0f;
    float last_keep        = 0.10f;
    int   turn_count       = 0;
    bool  recover_full_next = false;  // set by compression_failed guard; cleared after one turn
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

// Maximum number of concurrent sessions retained in memory.
// When this limit is reached, the least-recently-used session is evicted.
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

// Thread-safe per-session container with LRU eviction bounded to kMaxSessions.
// Prevents memory exhaustion from unbounded unique-session insertion.
class HttpServerSessions {
public:
    void update(const std::string& session_id, float observed_accept) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) {
            evict_if_full_locked();
            lru_.push_front(session_id);
            map_.emplace(session_id, Entry{step_adaptive_keep_ratio({}, observed_accept), lru_.begin()});
        } else {
            it->second.state = step_adaptive_keep_ratio(it->second.state, observed_accept);
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        }
    }

    float get_keep_ratio(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) return AdaptiveKeepRatioState{}.last_keep;
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        return it->second.state.last_keep;
    }

    float get_ema(const std::string & session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) return 0.0f;
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        return it->second.state.ema;
    }

    int turn_count(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) return 0;
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        return it->second.state.turn_count;
    }

    // Schedule full-keep recovery for the next turn of this session.
    // Called by the compression_failed guard when an agentic compressed turn
    // produced an empty or degenerate response.  Creates the session entry if
    // it does not exist yet (guard may fire before any bandit update).
    void set_recover_full_next(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) {
            evict_if_full_locked();
            lru_.push_front(session_id);
            AdaptiveKeepRatioState s{};
            s.recover_full_next = true;
            map_.emplace(session_id, Entry{s, lru_.begin()});
        } else {
            it->second.state.recover_full_next = true;
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        }
    }

    // Returns true and clears the flag if recovery was scheduled; false otherwise.
    // One-shot: the flag is consumed on read so the next turn runs normally.
    bool consume_recover_full_next(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(session_id);
        if (it == map_.end()) return false;
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        if (!it->second.state.recover_full_next) return false;
        it->second.state.recover_full_next = false;
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

private:
    struct Entry {
        AdaptiveKeepRatioState          state;
        std::list<std::string>::iterator lru_it;
    };

    void evict_if_full_locked() {
        if (map_.size() < kMaxSessions) return;
        const std::string& lru_key = lru_.back();
        map_.erase(lru_key);
        lru_.pop_back();
    }

    mutable std::mutex                              mu_;
    mutable std::list<std::string>                  lru_;
    std::unordered_map<std::string, Entry>          map_;
};

}  // namespace dflash::common
