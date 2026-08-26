// Move-only ownership for one continuous-batching prefix capture.
//
// Policy metadata and checkpoint payload have different owners. This object
// keeps their resolution ordered and makes every early exit cancel exactly
// one untouched reservation. A malformed saved outcome discards only this
// transaction's payload and metadata; it never trusts an event-supplied id.

#pragma once

#include "common/concurrency/prefix_store.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace dflash::common {

template <typename Reservation, typename Engine>
class BasicPrefixCaptureTxn {
public:
    enum class Resolution {
        inactive,
        saved,
        failed,
        mismatched,
    };

    BasicPrefixCaptureTxn() = default;

    BasicPrefixCaptureTxn(
            Reservation reservation, Engine & engine,
            PrefixCaptureTicket ticket)
        : reservation_(std::move(reservation)), engine_(&engine),
          ticket_(ticket) {}

    ~BasicPrefixCaptureTxn() { cancel(); }

    BasicPrefixCaptureTxn(const BasicPrefixCaptureTxn &) = delete;
    BasicPrefixCaptureTxn & operator=(
        const BasicPrefixCaptureTxn &) = delete;
    BasicPrefixCaptureTxn(BasicPrefixCaptureTxn &&) noexcept = default;
    BasicPrefixCaptureTxn & operator=(
        BasicPrefixCaptureTxn &&) noexcept = default;

    bool active() const {
        return engine_ && ticket_.valid() && reservation_.active() &&
            ticket_.checkpoint == PrefixStoreRef{
                (uint64_t)reservation_.slot() + 1,
                reservation_.target_cut()};
    }

    Resolution resolve(
            const PrefixStoreEvent & event,
            const std::vector<int32_t> & prompt) {
        if (!active()) return Resolution::inactive;
        if (!event.attempted() || event.ticket != ticket_) {
            if (event.status == PrefixStoreEvent::Status::saved) {
                discard_saved();
            } else {
                cancel();
            }
            return Resolution::mismatched;
        }
        if (event.status == PrefixStoreEvent::Status::saved) {
            const bool committed = reservation_.commit(prompt, event.bytes);
            clear();
            return committed ? Resolution::saved : Resolution::failed;
        }
        if (event.status == PrefixStoreEvent::Status::failed) {
            cancel();
            return Resolution::failed;
        }
        cancel();
        return Resolution::mismatched;
    }

    void cancel() {
        reservation_.cancel();
        clear();
    }

private:
    void discard_saved() {
        engine_->discard_prefix_store(ticket_.checkpoint);
        reservation_.abort();
        clear();
    }

    void clear() {
        engine_ = nullptr;
        ticket_ = {};
    }

    Reservation reservation_;
    Engine * engine_ = nullptr;
    PrefixCaptureTicket ticket_;
};

}  // namespace dflash::common
