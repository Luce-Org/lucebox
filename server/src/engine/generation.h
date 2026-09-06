#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "common/generation_types.h"

namespace dflash::engine {

struct TokenBatch {
    std::vector<int32_t> tokens;
};

enum class GenerationPhase {
    Queued,
    Prefill,
    Decode,
};

struct GenerationProgress {
    GenerationPhase phase = GenerationPhase::Queued;
    int processed_tokens = 0;
};

struct GenerateCompleted {
    common::GenerateResult result;
};

using GenerateEvent =
    std::variant<TokenBatch, GenerationProgress, GenerateCompleted>;

enum class GenerationCancelReason {
    ConsumerDropped,
    ClientDisconnected,
    OutputBackpressure,
    Shutdown,
};

namespace detail {
struct GenerationState;
}

struct GenerationPair;
GenerationPair make_generation(std::size_t token_capacity);

class Generation {
public:
    Generation() = default;
    Generation(Generation && other) noexcept;
    Generation & operator=(Generation && other) noexcept;
    Generation(const Generation &) = delete;
    Generation & operator=(const Generation &) = delete;
    ~Generation();

    explicit operator bool() const noexcept;
    // Blocks until the next buffered token batch, coalesced progress update,
    // or terminal result. A terminal result remains observable after completion.
    GenerateEvent next();
    // Records the first cancellation before completion. Returns false when a
    // cancellation or terminal result was already present.
    bool cancel(GenerationCancelReason reason);

private:
    explicit Generation(std::shared_ptr<detail::GenerationState> state);
    void reset();

    std::shared_ptr<detail::GenerationState> state_;

    friend struct GenerationPair;
    friend class GenerationQueue;
    friend GenerationPair make_generation(std::size_t token_capacity);
};

class GenerationSource {
public:
    GenerationSource() = default;
    GenerationSource(GenerationSource && other) noexcept = default;
    GenerationSource & operator=(GenerationSource && other) noexcept;
    GenerationSource(const GenerationSource &) = delete;
    GenerationSource & operator=(const GenerationSource &) = delete;
    ~GenerationSource();

    explicit operator bool() const noexcept;
    // Token publication is bounded. Filling the channel terminates this
    // generation with OutputBackpressure instead of blocking the producer.
    bool publish(TokenBatch batch);
    // Only the latest unread progress value is retained.
    bool publish(GenerationProgress progress);
    // Publishes the terminal result exactly once.
    bool complete(common::GenerateResult result);
    bool is_cancelled() const;
    std::optional<GenerationCancelReason> cancellation_reason() const;

private:
    explicit GenerationSource(std::shared_ptr<detail::GenerationState> state);
    void reset();

    std::shared_ptr<detail::GenerationState> state_;

    friend struct GenerationPair;
    friend class GenerationQueue;
    friend GenerationPair make_generation(std::size_t token_capacity);
};

struct GenerationPair {
    Generation generation;
    GenerationSource source;
};

Generation make_completed_generation(common::GenerateResult result);

class GenerationQueue {
public:
    struct Work {
        common::GenerateRequest request;
        GenerationSource source;
    };

    GenerationQueue(std::size_t request_capacity,
                    std::size_t token_capacity);
    GenerationQueue(const GenerationQueue &) = delete;
    GenerationQueue & operator=(const GenerationQueue &) = delete;
    ~GenerationQueue();

    // Returns an already-completed rejection when the queue is full or stopped.
    Generation submit(common::GenerateRequest request);
    // Blocks until FIFO work is available or shutdown begins.
    std::optional<Work> next();
    // Idempotently rejects new work and completes every live generation.
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dflash::engine
