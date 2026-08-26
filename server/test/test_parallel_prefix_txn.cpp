#include "server/parallel_prefix_txn.h"
#include "host_check.h"

#include <type_traits>
#include <utility>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

namespace {

struct FakeReservationState {
    std::vector<int> cancelled;
    std::vector<int> aborted;
    std::vector<int> committed;
    size_t committed_bytes = 0;
};

class FakeReservation {
public:
    FakeReservation() = default;
    explicit FakeReservation(
            FakeReservationState & state, int slot = 6, int cut = 16)
        : state_(&state), slot_(slot), cut_(cut) {}
    ~FakeReservation() { cancel(); }

    FakeReservation(const FakeReservation &) = delete;
    FakeReservation & operator=(const FakeReservation &) = delete;
    FakeReservation(FakeReservation && other) noexcept {
        take(std::move(other));
    }
    FakeReservation & operator=(FakeReservation && other) noexcept {
        if (this != &other) {
            cancel();
            take(std::move(other));
        }
        return *this;
    }

    bool active() const { return state_ != nullptr; }
    int slot() const { return slot_; }
    int target_cut() const { return cut_; }

    bool commit(const std::vector<int32_t> &, size_t bytes, bool = false) {
        if (!active()) return false;
        state_->committed.push_back(slot_);
        state_->committed_bytes = bytes;
        clear();
        return true;
    }

    void cancel() {
        if (!active()) return;
        state_->cancelled.push_back(slot_);
        clear();
    }

    void abort() {
        if (!active()) return;
        state_->aborted.push_back(slot_);
        clear();
    }

private:
    void clear() {
        state_ = nullptr;
        slot_ = -1;
        cut_ = 0;
    }

    void take(FakeReservation && other) {
        state_ = other.state_;
        slot_ = other.slot_;
        cut_ = other.cut_;
        other.clear();
    }

    FakeReservationState * state_ = nullptr;
    int slot_ = -1;
    int cut_ = 0;
};

struct FakeEngine {
    std::vector<PrefixStoreRef> discarded;

    void discard_prefix_store(PrefixStoreRef checkpoint) {
        discarded.push_back(checkpoint);
    }
};

using Txn = BasicPrefixCaptureTxn<FakeReservation, FakeEngine>;

PrefixCaptureTicket ticket(uint64_t id = 11) {
    PrefixCaptureTicket value;
    value.id = id;
    value.checkpoint = {7, 16};
    return value;
}

PrefixStoreEvent event(
        PrefixStoreEvent::Status status, PrefixCaptureTicket value) {
    PrefixStoreEvent out;
    out.status = status;
    out.ticket = value;
    if (status == PrefixStoreEvent::Status::saved) {
        out.bytes = 4096;
    }
    if (status == PrefixStoreEvent::Status::failed) {
        out.error = "copy failed";
    }
    return out;
}

}  // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<Txn>);
    static_assert(!std::is_copy_assignable_v<Txn>);
    static_assert(std::is_move_constructible_v<Txn>);
    static_assert(std::is_move_assignable_v<Txn>);

    {
        FakeReservationState state;
        FakeEngine engine;
        {
            Txn first(FakeReservation(state), engine, ticket());
            Txn owner(std::move(first));
            CHECK(!first.active());
            CHECK(owner.active());
        }
        CHECK(state.cancelled == std::vector<int>({6}));
        CHECK(state.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakeReservationState state;
        FakeEngine engine;
        {
            Txn destination(
                FakeReservation(state), engine, ticket(/*id=*/11));
            Txn source(
                FakeReservation(state), engine, ticket(/*id=*/12));
            destination = std::move(source);
            CHECK(!source.active());
            CHECK(destination.active());
            CHECK(state.cancelled == std::vector<int>({6}));
            CHECK(state.aborted.empty());
            CHECK(engine.discarded.empty());
        }
        CHECK(state.cancelled == std::vector<int>({6, 6}));
    }

    {
        FakeReservationState state;
        FakeEngine engine;
        Txn txn(FakeReservation(state), engine, ticket());
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::saved, ticket()),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::saved);
        CHECK(!txn.active());
        CHECK(state.committed_bytes == 4096);
        CHECK(state.committed == std::vector<int>({6}));
        CHECK(state.cancelled.empty());
        CHECK(state.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakeReservationState state;
        FakeEngine engine;
        Txn txn(FakeReservation(state), engine, ticket());
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::failed, ticket()),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::failed);
        CHECK(state.cancelled == std::vector<int>({6}));
        CHECK(state.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakeReservationState state;
        FakeEngine engine;
        Txn txn(FakeReservation(state), engine, ticket());
        PrefixCaptureTicket unrelated = ticket(/*id=*/99);
        unrelated.checkpoint = {63, 16};
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::saved, unrelated),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::mismatched);
        CHECK(state.cancelled.empty());
        CHECK(state.aborted == std::vector<int>({6}));
        CHECK(engine.discarded == std::vector<PrefixStoreRef>({{7, 16}}));
    }

    {
        FakeReservationState state;
        FakeEngine engine;
        Txn txn(FakeReservation(state), engine, ticket());
        txn.cancel();
        CHECK(state.cancelled == std::vector<int>({6}));
        CHECK(state.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    std::printf("OK test_parallel_prefix_txn (%d checks)\n", g_checks);
    return 0;
}
