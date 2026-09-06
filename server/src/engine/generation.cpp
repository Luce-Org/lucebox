#include "generation.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dflash::engine {

namespace detail {

struct GenerationState {
    explicit GenerationState(std::size_t capacity)
        : token_capacity(capacity) {}

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<TokenBatch> tokens;
    std::size_t buffered_tokens = 0;
    std::optional<GenerationProgress> progress;
    std::optional<GenerateCompleted> terminal;
    std::optional<GenerationCancelReason> cancel_reason;
    std::size_t token_capacity;
};

} // namespace detail

Generation::Generation(std::shared_ptr<detail::GenerationState> state)
    : state_(std::move(state)) {}

Generation::Generation(Generation && other) noexcept
    : state_(std::move(other.state_)) {}

Generation & Generation::operator=(Generation && other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
    }
    return *this;
}

Generation::~Generation() {
    reset();
}

Generation::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

GenerateEvent Generation::next() {
    if (!state_) {
        throw std::logic_error("next() called on an empty Generation");
    }

    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->ready.wait(lock, [this] {
        return !state_->tokens.empty() || state_->progress || state_->terminal;
    });

    if (!state_->tokens.empty()) {
        TokenBatch batch = std::move(state_->tokens.front());
        state_->tokens.pop_front();
        state_->buffered_tokens -= batch.tokens.size();
        return batch;
    }
    if (state_->progress) {
        GenerationProgress progress = *state_->progress;
        state_->progress.reset();
        return progress;
    }
    return *state_->terminal;
}

bool Generation::cancel(GenerationCancelReason reason) {
    if (!state_) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->terminal) return false;
    state_->cancel_reason = reason;
    common::GenerateResult result;
    switch (reason) {
    case GenerationCancelReason::OutputBackpressure:
        result.fail(common::GenerateErrorCode::OutputBackpressure);
        break;
    case GenerationCancelReason::Shutdown:
        result.fail(common::GenerateErrorCode::ShuttingDown);
        break;
    case GenerationCancelReason::ConsumerDropped:
    case GenerationCancelReason::ClientDisconnected:
        result.fail(common::GenerateErrorCode::Cancelled);
        break;
    }
    state_->terminal = GenerateCompleted{std::move(result)};
    state_->ready.notify_all();
    return true;
}

void Generation::reset() {
    if (state_) cancel(GenerationCancelReason::ConsumerDropped);
    state_.reset();
}

GenerationSource::GenerationSource(
        std::shared_ptr<detail::GenerationState> state)
    : state_(std::move(state)) {}

GenerationSource & GenerationSource::operator=(
        GenerationSource && other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
    }
    return *this;
}

GenerationSource::~GenerationSource() {
    reset();
}

void GenerationSource::reset() {
    // Losing the producer must wake a surviving consumer with a terminal
    // result; otherwise Generation::next() could wait forever.
    if (!state_) return;
    common::GenerateResult result;
    result.fail(common::GenerateErrorCode::Incomplete,
                "generation ended without a terminal result");
    complete(std::move(result));
    state_.reset();
}

GenerationSource::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

bool GenerationSource::publish(TokenBatch batch) {
    if (!state_) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->terminal) return false;
    if (batch.tokens.empty()) return true;
    if (batch.tokens.size() >
        state_->token_capacity - state_->buffered_tokens) {
        state_->cancel_reason = GenerationCancelReason::OutputBackpressure;
        common::GenerateResult result;
        result.fail(common::GenerateErrorCode::OutputBackpressure);
        state_->terminal = GenerateCompleted{std::move(result)};
        state_->ready.notify_all();
        return false;
    }
    state_->buffered_tokens += batch.tokens.size();
    state_->tokens.push_back(std::move(batch));
    state_->ready.notify_one();
    return true;
}

bool GenerationSource::publish(GenerationProgress progress) {
    if (!state_) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->terminal) return false;
    state_->progress = progress;
    state_->ready.notify_one();
    return true;
}

bool GenerationSource::complete(common::GenerateResult result) {
    if (!state_) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->terminal) return false;
    state_->terminal = GenerateCompleted{std::move(result)};
    state_->ready.notify_all();
    return true;
}

bool GenerationSource::is_cancelled() const {
    if (!state_) return true;
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->cancel_reason.has_value();
}

std::optional<GenerationCancelReason>
GenerationSource::cancellation_reason() const {
    if (!state_) return GenerationCancelReason::ConsumerDropped;
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->cancel_reason;
}

GenerationPair make_generation(std::size_t token_capacity) {
    if (token_capacity == 0) {
        throw std::invalid_argument("Generation token capacity must be positive");
    }
    auto state = std::make_shared<detail::GenerationState>(token_capacity);
    return {Generation(state), GenerationSource(std::move(state))};
}

Generation make_completed_generation(common::GenerateResult result) {
    GenerationPair pair = make_generation(1);
    pair.source.complete(std::move(result));
    return std::move(pair.generation);
}

struct GenerationQueue::Impl {
    struct Queued {
        common::GenerateRequest request;
        std::shared_ptr<detail::GenerationState> state;
    };

    Impl(std::size_t requests, std::size_t tokens)
        : request_capacity(requests), token_capacity(tokens) {}

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Queued> queued;
    // Shutdown can reach every live channel without extending its lifetime.
    std::vector<std::weak_ptr<detail::GenerationState>> live;
    std::size_t request_capacity;
    std::size_t token_capacity;
    bool stopping = false;
};

GenerationQueue::GenerationQueue(std::size_t request_capacity,
                                 std::size_t token_capacity) {
    if (request_capacity == 0) {
        throw std::invalid_argument(
            "Generation request capacity must be positive");
    }
    if (token_capacity == 0) {
        throw std::invalid_argument(
            "Generation token capacity must be positive");
    }
    impl_ = std::make_unique<Impl>(request_capacity, token_capacity);
}

GenerationQueue::~GenerationQueue() {
    shutdown();
}

Generation GenerationQueue::submit(common::GenerateRequest request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->queued.erase(
        std::remove_if(
            impl_->queued.begin(), impl_->queued.end(),
            [](const Impl::Queued & queued) {
                std::lock_guard<std::mutex> state_lock(queued.state->mutex);
                return queued.state->cancel_reason.has_value();
            }),
        impl_->queued.end());
    impl_->live.erase(
        std::remove_if(impl_->live.begin(), impl_->live.end(),
                       [](const auto & state) { return state.expired(); }),
        impl_->live.end());
    if (impl_->stopping) {
        common::GenerateResult result;
        result.fail(common::GenerateErrorCode::ShuttingDown);
        return make_completed_generation(std::move(result));
    }
    if (impl_->queued.size() >= impl_->request_capacity) {
        common::GenerateResult result;
        result.fail(common::GenerateErrorCode::Overloaded);
        return make_completed_generation(std::move(result));
    }

    auto state = std::make_shared<detail::GenerationState>(
        impl_->token_capacity);
    Generation generation(state);
    impl_->live.push_back(state);
    impl_->queued.push_back({std::move(request), std::move(state)});
    impl_->ready.notify_one();
    return generation;
}

std::optional<GenerationQueue::Work> GenerationQueue::next() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    while (true) {
        impl_->ready.wait(lock, [this] {
            return impl_->stopping || !impl_->queued.empty();
        });

        while (!impl_->queued.empty()) {
            Impl::Queued queued = std::move(impl_->queued.front());
            impl_->queued.pop_front();

            {
                std::lock_guard<std::mutex> state_lock(queued.state->mutex);
                if (queued.state->cancel_reason) continue;
            }

            return Work{std::move(queued.request),
                        GenerationSource(std::move(queued.state))};
        }

        if (impl_->stopping) return std::nullopt;
    }
}

void GenerationQueue::shutdown() {
    std::lock_guard<std::mutex> queue_lock(impl_->mutex);
    if (impl_->stopping) return;
    impl_->stopping = true;
    impl_->queued.clear();

    for (const auto & weak : impl_->live) {
        auto state = weak.lock();
        if (!state) continue;
        std::lock_guard<std::mutex> state_lock(state->mutex);
        if (state->terminal) continue;
        state->cancel_reason = GenerationCancelReason::Shutdown;
        common::GenerateResult result;
        result.fail(common::GenerateErrorCode::ShuttingDown);
        state->terminal = GenerateCompleted{std::move(result)};
        state->ready.notify_all();
    }
    impl_->ready.notify_all();
}

} // namespace dflash::engine
