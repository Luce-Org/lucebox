#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace dflash::common {
struct ModelBackend;
class SeqEngine;
} // namespace dflash::common

namespace dflash::engine {

// Owns the model backend and the single execution thread used to serve it.
// Transport-specific request handling remains outside this class; callers
// provide the established serial and concurrent serving loops.
class LuceEngine final {
public:
    struct ServingLoops {
        std::function<void()> serial;
        std::function<void(common::SeqEngine &)> concurrent;
        std::function<void()> request_stop;
    };

    explicit LuceEngine(std::unique_ptr<common::ModelBackend> backend);
    ~LuceEngine();

    LuceEngine(const LuceEngine &) = delete;
    LuceEngine &operator=(const LuceEngine &) = delete;
    LuceEngine(LuceEngine &&) = delete;
    LuceEngine &operator=(LuceEngine &&) = delete;

    common::ModelBackend &backend() noexcept;
    const common::ModelBackend &backend() const noexcept;

    // Starts exactly one serving loop. Concurrent serving is selected only
    // when the caller permits it and the backend exposes a SeqEngine.
    bool start_serving(ServingLoops loops, bool allow_concurrent);
    void stop_serving();

private:
    std::unique_ptr<common::ModelBackend> backend_;
    std::function<void()> request_stop_;
    std::thread worker_;
    std::mutex lifecycle_mu_;
};

} // namespace dflash::engine
