// What the engine is serving now, and the bound on what waits for it: the
// "load" object of GET /props, and --queue-limit.
//
// A request waits from the admission gate, which runs before it is tokenized,
// until it takes a slot or ends without one: while it is tokenized, queued,
// deferred for KV, waiting for a shared prefix or compressed by PFlash. A
// request with tool checks waits again between its passes. Client threads
// count it in and out; the serving thread moves it to a slot and back, and
// publishes the busy slots and the KV pool. One mutex keeps the gate's check
// and the counts consistent, and nobody holds it past a few loads and stores,
// so /props answers while slots decode.

#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <mutex>

namespace luce::common {

class EngineLoad {
public:
    // A request's place among the waiting. The serving thread gives it up
    // when the request takes a slot (seat), and takes it back when a tool
    // check's pass leaves its slot (vacate); the client thread gives it up
    // when the request ends (leave, or the destructor).
    class Place {
    public:
        explicit Place(EngineLoad & load) : load_(load) {}
        ~Place() { load_.leave(*this); }
        Place(const Place &) = delete;
        Place & operator=(const Place &) = delete;

    private:
        friend class EngineLoad;
        EngineLoad & load_;
        bool held_ = false;  // guarded by load_.mu_
    };

    // queue_limit: the requests that may wait while every slot is taken;
    // negative: no limit.
    explicit EngineLoad(int queue_limit = -1) : queue_limit_(queue_limit) {}

    // Client thread, before tokenizing: counts the request as waiting.
    // False, counting nothing, when `enforce` and the engine already holds a
    // request for every slot and queue_limit more, busy or waiting: a slot
    // left free while the request at the head of the line waits for KV is
    // that request's, so the waiting can pass queue_limit by the free slots.
    bool enter(Place & place, bool enforce) {
        std::lock_guard<std::mutex> lock(mu_);
        if (place.held_) return true;
        if (enforce && queue_limit_ >= 0 &&
            int64_t{busy_} + waiting_ >= int64_t{slots_} + queue_limit_) {
            return false;
        }
        ++waiting_;
        place.held_ = true;
        return true;
    }

    // The request stopped waiting without a slot. Idempotent.
    void leave(Place & place) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!place.held_) return;
        place.held_ = false;
        --waiting_;
    }

    // Serving thread. The slot count and the KV pool's size, in tokens (0:
    // the engine does not report one), set once before the first request.
    void configure(int slots, int64_t kv_tokens) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_ = slots;
        kv_tokens_ = kv_tokens;
        kv_free_tokens_ = kv_tokens;
    }
    // A request took a slot: its place (null for a job no gate counted) stops
    // waiting as `busy` becomes the slots taken.
    void seat(Place * place, int busy, int64_t kv_free_tokens) {
        std::lock_guard<std::mutex> lock(mu_);
        if (place && place->held_) {
            place->held_ = false;
            --waiting_;
        }
        busy_ = busy;
        kv_free_tokens_ = kv_free_tokens;
    }
    // A request left its slot, before its reply goes out. `still_waiting`:
    // the place of a tool check's pass, whose request waits on for its next
    // pass or its reply; it is held again as `busy` drops, so the gate never
    // sees that request in neither count. Null for any other request.
    void vacate(Place * still_waiting, int busy, int64_t kv_free_tokens) {
        std::lock_guard<std::mutex> lock(mu_);
        if (still_waiting && !still_waiting->held_) {
            still_waiting->held_ = true;
            ++waiting_;
        }
        busy_ = busy;
        kv_free_tokens_ = kv_free_tokens;
    }
    // Serving thread. An unchanged step takes no lock: only this thread writes
    // busy_ and kv_free_tokens_, so it may read them unlocked.
    void publish(int busy, int64_t kv_free_tokens) {
        if (busy == busy_ && kv_free_tokens == kv_free_tokens_) return;
        vacate(nullptr, busy, kv_free_tokens);
    }

    // GET /props "load". The KV fields are null when the engine reports no
    // pool; queue_limit is null without a limit.
    nlohmann::json to_json() const {
        int slots, busy, waiting;
        int64_t kv_tokens, kv_free_tokens;
        {
            std::lock_guard<std::mutex> lock(mu_);
            slots = slots_;
            busy = busy_;
            waiting = waiting_;
            kv_tokens = kv_tokens_;
            kv_free_tokens = kv_free_tokens_;
        }
        const auto or_null = [](bool known, int64_t value) {
            return known ? nlohmann::json(value) : nlohmann::json(nullptr);
        };
        return {
            {"slots", slots},
            {"slots_busy", busy},
            {"waiting", waiting},
            {"queue_limit", or_null(queue_limit_ >= 0, queue_limit_)},
            {"kv_tokens", or_null(kv_tokens > 0, kv_tokens)},
            {"kv_free_tokens", or_null(kv_tokens > 0, kv_free_tokens)},
        };
    }

private:
    mutable std::mutex mu_;
    const int queue_limit_;
    int slots_ = 1;
    int busy_ = 0;
    int waiting_ = 0;
    int64_t kv_tokens_ = 0;
    int64_t kv_free_tokens_ = 0;
};

}  // namespace luce::common
