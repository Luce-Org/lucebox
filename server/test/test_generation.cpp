#include "CppUnitTestFramework.hpp"
#include "engine/generation.h"

#include <type_traits>
#include <utility>

using namespace CppUnitTestFramework;
using namespace dflash::common;
using namespace dflash::engine;

namespace {

static_assert(!std::is_copy_constructible_v<Generation>);
static_assert(std::is_move_constructible_v<Generation>);
static_assert(!std::is_copy_constructible_v<GenerationSource>);
static_assert(std::is_move_constructible_v<GenerationSource>);

GenerateRequest owned_request() {
    GenerateRequest request;
    request.prompt = {1, 2};
    request.hint_tokens = {3, 4};
    request.stall_tool_prefix_tokens = {5};
    request.stall_action_suffix_tokens = {6};
    request.stall_skip_tokens = {7};
    return request;
}

struct GenerationFixture : CommonFixture {
    using CommonFixture::CommonFixture;

    void request_payloads_are_owned() {
        GenerateRequest request = owned_request();
        CHECK(request.prompt == std::vector<int32_t>({1, 2}));
        CHECK(request.hint_tokens == std::vector<int32_t>({3, 4}));
        CHECK(request.stall_tool_prefix_tokens == std::vector<int32_t>({5}));
        CHECK(request.stall_action_suffix_tokens == std::vector<int32_t>({6}));
        CHECK(request.stall_skip_tokens == std::vector<int32_t>({7}));
    }

    void tokens_precede_terminal_result() {
        GenerationPair pair = make_generation(2);
        CHECK(pair.source.publish(TokenBatch{{11, 12}}));
        GenerateResult result;
        result.tokens = {11, 12};
        result.succeed();
        CHECK(pair.source.complete(result));
        CHECK(!pair.source.complete(result));

        auto token_event = pair.generation.next();
        CHECK(std::get<TokenBatch>(token_event).tokens ==
              std::vector<int32_t>({11, 12}));
        auto terminal_event = pair.generation.next();
        CHECK(std::get<GenerateCompleted>(terminal_event).result.ok());
    }

    void progress_is_coalesced() {
        GenerationPair pair = make_generation(1);
        CHECK(pair.source.publish(GenerationProgress{GenerationPhase::Prefill, 4}));
        CHECK(pair.source.publish(GenerationProgress{GenerationPhase::Prefill, 9}));
        auto event = pair.generation.next();
        const auto & progress = std::get<GenerationProgress>(event);
        CHECK(progress.phase == GenerationPhase::Prefill);
        CHECK(progress.processed_tokens == 9);
    }

    void full_channel_cancels_only_its_generation() {
        GenerationPair full = make_generation(1);
        GenerationPair healthy = make_generation(1);
        CHECK(full.source.publish(TokenBatch{{1}}));
        CHECK(!full.source.publish(TokenBatch{{2}}));
        CHECK(full.source.cancellation_reason() ==
              GenerationCancelReason::OutputBackpressure);

        GenerateResult ok;
        ok.succeed();
        CHECK(healthy.source.complete(ok));
        CHECK(std::get<GenerateCompleted>(healthy.generation.next()).result.ok());

        CHECK(std::get<TokenBatch>(full.generation.next()).tokens ==
              std::vector<int32_t>({1}));
        const auto terminal =
            std::get<GenerateCompleted>(full.generation.next());
        CHECK(terminal.result.error->code ==
              GenerateErrorCode::OutputBackpressure);
    }

    void token_capacity_counts_tokens_not_batches() {
        GenerationPair pair = make_generation(2);
        CHECK(pair.source.publish(TokenBatch{}));
        CHECK(pair.source.publish(TokenBatch{}));
        CHECK(pair.source.publish(TokenBatch{{1, 2}}));
        CHECK(std::get<TokenBatch>(pair.generation.next()).tokens ==
              std::vector<int32_t>({1, 2}));

        CHECK(pair.source.publish(TokenBatch{{3, 4}}));
        CHECK(!pair.source.publish(TokenBatch{{5}}));
        CHECK(std::get<TokenBatch>(pair.generation.next()).tokens ==
              std::vector<int32_t>({3, 4}));
        const auto terminal =
            std::get<GenerateCompleted>(pair.generation.next());
        CHECK(terminal.result.error->code ==
              GenerateErrorCode::OutputBackpressure);
    }

    void cancellation_is_idempotent() {
        GenerationPair pair = make_generation(1);
        CHECK(pair.generation.cancel(GenerationCancelReason::ClientDisconnected));
        CHECK(!pair.generation.cancel(GenerationCancelReason::Shutdown));
        CHECK(pair.source.cancellation_reason() ==
              GenerationCancelReason::ClientDisconnected);
        const auto terminal =
            std::get<GenerateCompleted>(pair.generation.next());
        CHECK(terminal.result.error->code == GenerateErrorCode::Cancelled);
    }

    void source_destruction_preserves_terminal_result() {
        for (auto reason : {GenerationCancelReason::ConsumerDropped,
                            GenerationCancelReason::ClientDisconnected,
                            GenerationCancelReason::OutputBackpressure,
                            GenerationCancelReason::Shutdown}) {
            GenerationPair pair = make_generation(1);
            CHECK(pair.generation.cancel(reason));
            const auto before = std::get<GenerateCompleted>(pair.generation.next());
            CHECK(!pair.source.publish(TokenBatch{{1}}));
            CHECK(!pair.source.publish(GenerationProgress{}));
            pair.source = GenerationSource{};
            const auto after = std::get<GenerateCompleted>(pair.generation.next());
            CHECK(after.result.error->code == before.result.error->code);
        }

        GenerationPair pair = make_generation(1);
        GenerateResult result;
        result.tokens = {7};
        result.succeed();
        CHECK(pair.source.complete(result));
        CHECK(!pair.generation.cancel(GenerationCancelReason::Shutdown));
        pair.source = GenerationSource{};
        const auto terminal = std::get<GenerateCompleted>(pair.generation.next());
        CHECK(terminal.result.ok());
        CHECK(terminal.result.tokens == result.tokens);
    }

    void dropping_handle_requests_cancellation() {
        GenerationPair pair = make_generation(1);
        pair.generation = Generation{};
        CHECK(pair.source.cancellation_reason() ==
              GenerationCancelReason::ConsumerDropped);
    }

    void source_destruction_wakes_handle() {
        Generation generation;
        {
            GenerationPair pair = make_generation(1);
            generation = std::move(pair.generation);
        }
        const auto terminal = std::get<GenerateCompleted>(generation.next());
        CHECK(terminal.result.error->code == GenerateErrorCode::Incomplete);
    }

    void source_move_assignment_completes_replaced_handle() {
        GenerationPair first = make_generation(1);
        GenerationPair second = make_generation(1);
        first.source = std::move(second.source);

        const auto replaced =
            std::get<GenerateCompleted>(first.generation.next());
        CHECK(replaced.result.error->code == GenerateErrorCode::Incomplete);

        GenerateResult result;
        result.succeed();
        CHECK(first.source.complete(std::move(result)));
        CHECK(std::get<GenerateCompleted>(second.generation.next()).result.ok());
    }

    void immediate_rejection_is_terminal() {
        GenerateResult result;
        result.fail(GenerateErrorCode::Overloaded);
        Generation generation = make_completed_generation(std::move(result));
        const auto terminal = std::get<GenerateCompleted>(generation.next());
        CHECK(terminal.result.error->code == GenerateErrorCode::Overloaded);
    }

    void request_queue_is_bounded_and_fifo() {
        GenerationQueue queue(2, 2);
        GenerateRequest first;
        first.prompt = {1};
        Generation first_generation = queue.submit(std::move(first));

        GenerateRequest second;
        second.prompt = {2};
        Generation second_generation = queue.submit(std::move(second));

        GenerateRequest third;
        third.prompt = {3};
        Generation rejected = queue.submit(std::move(third));
        const auto rejection =
            std::get<GenerateCompleted>(rejected.next());
        CHECK(rejection.result.error->code == GenerateErrorCode::Overloaded);

        auto first_work = queue.next();
        CHECK(first_work.has_value());
        CHECK(first_work->request.prompt == std::vector<int32_t>({1}));
        auto second_work = queue.next();
        CHECK(second_work.has_value());
        CHECK(second_work->request.prompt == std::vector<int32_t>({2}));

        GenerateResult result;
        result.succeed();
        CHECK(first_work->source.complete(result));
        CHECK(second_work->source.complete(std::move(result)));
        CHECK(std::get<GenerateCompleted>(first_generation.next()).result.ok());
        CHECK(std::get<GenerateCompleted>(second_generation.next()).result.ok());
    }

    void request_queue_skips_cancelled_work() {
        GenerationQueue queue(2, 1);
        Generation cancelled = queue.submit(GenerateRequest{});
        GenerateRequest request;
        request.prompt = {8};
        Generation accepted = queue.submit(std::move(request));
        CHECK(cancelled.cancel(GenerationCancelReason::ClientDisconnected));

        auto work = queue.next();
        CHECK(work.has_value());
        CHECK(work->request.prompt == std::vector<int32_t>({8}));
        const auto terminal =
            std::get<GenerateCompleted>(cancelled.next());
        CHECK(terminal.result.error->code == GenerateErrorCode::Cancelled);

        GenerateResult result;
        result.succeed();
        CHECK(work->source.complete(std::move(result)));
        CHECK(std::get<GenerateCompleted>(accepted.next()).result.ok());
    }

    void cancelled_queued_work_releases_capacity() {
        GenerationQueue queue(1, 1);
        Generation cancelled = queue.submit(GenerateRequest{});
        CHECK(cancelled.cancel(GenerationCancelReason::ClientDisconnected));

        Generation replacement = queue.submit(GenerateRequest{});
        queue.shutdown();
        const auto terminal =
            std::get<GenerateCompleted>(replacement.next());
        CHECK(terminal.result.error->code ==
              GenerateErrorCode::ShuttingDown);
    }

    void shutdown_completes_queued_and_active_work() {
        GenerationQueue queue(2, 1);
        Generation active = queue.submit(GenerateRequest{});
        auto work = queue.next();
        CHECK(work.has_value());
        Generation queued = queue.submit(GenerateRequest{});

        queue.shutdown();
        queue.shutdown();

        const auto active_terminal =
            std::get<GenerateCompleted>(active.next());
        const auto queued_terminal =
            std::get<GenerateCompleted>(queued.next());
        CHECK(active_terminal.result.error->code ==
              GenerateErrorCode::ShuttingDown);
        CHECK(queued_terminal.result.error->code ==
              GenerateErrorCode::ShuttingDown);
        CHECK(work->source.cancellation_reason() ==
              GenerationCancelReason::Shutdown);
        CHECK(!work->source.publish(TokenBatch{{1}}));
        CHECK(!queue.next().has_value());

        Generation rejected = queue.submit(GenerateRequest{});
        const auto rejection =
            std::get<GenerateCompleted>(rejected.next());
        CHECK(rejection.result.error->code ==
              GenerateErrorCode::ShuttingDown);
    }

    void handle_outlives_request_queue() {
        Generation generation;
        {
            GenerationQueue queue(1, 1);
            generation = queue.submit(GenerateRequest{});
        }

        const auto terminal =
            std::get<GenerateCompleted>(generation.next());
        CHECK(terminal.result.error->code ==
              GenerateErrorCode::ShuttingDown);
    }
};

} // namespace

TEST_CASE(GenerationFixture, generation_channel_suite) {
    request_payloads_are_owned();
    tokens_precede_terminal_result();
    progress_is_coalesced();
    full_channel_cancels_only_its_generation();
    token_capacity_counts_tokens_not_batches();
    cancellation_is_idempotent();
    source_destruction_preserves_terminal_result();
    dropping_handle_requests_cancellation();
    source_destruction_wakes_handle();
    source_move_assignment_completes_replaced_handle();
    immediate_rejection_is_terminal();
    request_queue_is_bounded_and_fifo();
    request_queue_skips_cancelled_work();
    cancelled_queued_work_releases_capacity();
    shutdown_completes_queued_and_active_work();
    handle_outlives_request_queue();
}
