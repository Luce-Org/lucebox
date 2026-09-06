#include "CppUnitTestFramework.hpp"
#include "common/model_backend.h"
#include "engine/luce_engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

using namespace CppUnitTestFramework;
using namespace dflash::common;
using namespace dflash::engine;

namespace {

static_assert(!std::is_copy_constructible_v<LuceEngine>);
static_assert(!std::is_move_constructible_v<LuceEngine>);

class FakeSeqEngine final : public SeqEngine {
public:
    int slot_count() const override { return 1; }
    int max_context() const override { return 1024; }
    AdmitResult admit(uint64_t, const std::vector<int32_t> &,
                      const SamplerCfg &) override {
        return {};
    }
    StepPlanLimits step_plan_limits(int) const override { return {}; }
    StepResult step(const StepPlan &) override { return {}; }
    void retire(int) override {}
    bool token_is_eos(int32_t) const override { return false; }
};

class FakeBackend final : public ModelBackend {
public:
    FakeBackend(bool concurrent, std::atomic<bool> * destroyed = nullptr,
                std::atomic<int> * shutdown_calls = nullptr)
        : concurrent_(concurrent), destroyed_(destroyed),
          shutdown_calls_(shutdown_calls) {}

    ~FakeBackend() override {
        shutdown();
        if (destroyed_) destroyed_->store(true);
    }

    void print_ready_banner() const override {}
    bool park(ParkTarget) override { return true; }
    bool unpark(ParkTarget) override { return true; }
    bool is_target_parked() const override { return false; }
    GenerateResult generate_impl(const GenerateRequest &,
                                 const DaemonIO &) override {
        return {};
    }
    SeqEngine * seq_engine() override {
        return concurrent_ ? &seq_engine_ : nullptr;
    }
    bool snapshot_save(int) override { return true; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate_impl(
            int, const GenerateRequest &, const DaemonIO &) override {
        return {};
    }
    bool handle_compress(const std::string &, const DaemonIO &) override {
        return false;
    }
    void free_drafter() override {}
    void shutdown() override {
        if (shutdown_calls_) shutdown_calls_->fetch_add(1);
    }

private:
    bool concurrent_;
    std::atomic<bool> * destroyed_;
    std::atomic<int> * shutdown_calls_;
    FakeSeqEngine seq_engine_;
};

struct ServingProbe {
    std::mutex mutex;
    std::condition_variable ready;
    bool started = false;
    bool stopped = false;
    bool ran_concurrent = false;

    LuceEngine::ServingLoops loops() {
        LuceEngine::ServingLoops result;
        result.serial = [this] { run(false); };
        result.concurrent = [this](SeqEngine &) { run(true); };
        result.request_stop = [this] {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
            ready.notify_all();
        };
        return result;
    }

    bool wait_until_started() {
        std::unique_lock<std::mutex> lock(mutex);
        return ready.wait_for(lock, std::chrono::seconds(1),
                              [this] { return started; });
    }

private:
    void run(bool concurrent) {
        std::unique_lock<std::mutex> lock(mutex);
        ran_concurrent = concurrent;
        started = true;
        ready.notify_all();
        ready.wait(lock, [this] { return stopped; });
    }
};

struct LuceEngineFixture : CommonFixture {
    using CommonFixture::CommonFixture;

    void selects_serial_loop_without_sequence_engine() {
        LuceEngine engine(std::make_unique<FakeBackend>(false));
        ServingProbe probe;
        REQUIRE(engine.start_serving(probe.loops(), true));
        REQUIRE(probe.wait_until_started());
        CHECK(!probe.ran_concurrent);
        engine.stop_serving();
    }

    void selects_concurrent_loop_when_allowed() {
        LuceEngine engine(std::make_unique<FakeBackend>(true));
        ServingProbe probe;
        REQUIRE(engine.start_serving(probe.loops(), true));
        REQUIRE(probe.wait_until_started());
        CHECK(probe.ran_concurrent);
        engine.stop_serving();
    }

    void policy_can_force_serial_loop() {
        LuceEngine engine(std::make_unique<FakeBackend>(true));
        ServingProbe probe;
        REQUIRE(engine.start_serving(probe.loops(), false));
        REQUIRE(probe.wait_until_started());
        CHECK(!probe.ran_concurrent);
        engine.stop_serving();
    }

    void owns_backend_lifetime_without_duplicate_shutdown() {
        std::atomic<bool> destroyed{false};
        std::atomic<int> shutdown_calls{0};
        {
            LuceEngine engine(
                std::make_unique<FakeBackend>(false, &destroyed,
                                              &shutdown_calls));
            CHECK(!destroyed.load());
            CHECK(shutdown_calls.load() == 0);
        }
        CHECK(shutdown_calls.load() == 1);
        CHECK(destroyed.load());
    }

    void stop_is_idempotent_and_allows_restart() {
        ServingProbe first;
        ServingProbe second;
        LuceEngine engine(std::make_unique<FakeBackend>(true));
        engine.stop_serving();
        REQUIRE(engine.start_serving(first.loops(), false));
        REQUIRE(first.wait_until_started());
        CHECK(!engine.start_serving(second.loops(), true));
        engine.stop_serving();
        engine.stop_serving();
        REQUIRE(engine.start_serving(second.loops(), true));
        REQUIRE(second.wait_until_started());
        CHECK(second.ran_concurrent);
        engine.stop_serving();
    }

    void finished_loop_must_be_joined_before_restart() {
        std::promise<void> finished;
        auto completion = finished.get_future();
        int stops = 0;
        LuceEngine engine(std::make_unique<FakeBackend>(false));
        LuceEngine::ServingLoops loops;
        loops.serial = [&] { finished.set_value_at_thread_exit(); };
        loops.concurrent = [](SeqEngine &) {};
        loops.request_stop = [&] { ++stops; };
        REQUIRE(engine.start_serving(loops, false));
        REQUIRE(completion.wait_for(std::chrono::seconds(1)) ==
                std::future_status::ready);
        CHECK(!engine.start_serving(loops, false));
        engine.stop_serving();
        engine.stop_serving();
        CHECK(stops == 1);
        loops.serial = [] {};
        REQUIRE(engine.start_serving(loops, false));
        engine.stop_serving();
        CHECK(stops == 2);
    }

    void destruction_joins_before_destroying_backend() {
        ServingProbe probe;
        std::atomic<bool> destroyed{false};
        bool backend_alive_at_loop_exit = false;
        {
            LuceEngine engine(std::make_unique<FakeBackend>(false, &destroyed));
            auto loops = probe.loops();
            loops.serial = [&, run = std::move(loops.serial)] {
                run();
                backend_alive_at_loop_exit = !destroyed.load();
            };
            REQUIRE(engine.start_serving(std::move(loops), false));
            REQUIRE(probe.wait_until_started());
            CHECK(!destroyed.load());
        }
        CHECK(probe.stopped);
        CHECK(backend_alive_at_loop_exit);
        CHECK(destroyed.load());
    }

    void rejects_missing_backend() {
        bool rejected = false;
        try {
            LuceEngine engine(nullptr);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        CHECK(rejected);
    }
};

}  // namespace

TEST_CASE(LuceEngineFixture, luce_engine_lifecycle_suite) {
    selects_serial_loop_without_sequence_engine();
    selects_concurrent_loop_when_allowed();
    policy_can_force_serial_loop();
    owns_backend_lifetime_without_duplicate_shutdown();
    stop_is_idempotent_and_allows_restart();
    finished_loop_must_be_joined_before_restart();
    destruction_joins_before_destroying_backend();
    rejects_missing_backend();
}
