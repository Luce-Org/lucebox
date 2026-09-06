#include "luce_engine.h"

#include "common/model_backend.h"

#include <stdexcept>
#include <utility>

namespace dflash::engine {

LuceEngine::LuceEngine(std::unique_ptr<common::ModelBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("LuceEngine requires a backend");
    }
}

LuceEngine::~LuceEngine() {
    stop_serving();
}

common::ModelBackend &LuceEngine::backend() noexcept {
    return *backend_;
}

const common::ModelBackend &LuceEngine::backend() const noexcept {
    return *backend_;
}

bool LuceEngine::start_serving(ServingLoops loops, bool allow_concurrent) {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    if (worker_.joinable() || !loops.serial || !loops.concurrent ||
        !loops.request_stop) {
        return false;
    }

    common::SeqEngine *seq_engine =
        allow_concurrent ? backend_->seq_engine() : nullptr;
    request_stop_ = std::move(loops.request_stop);

    try {
        if (seq_engine) {
            auto concurrent = std::move(loops.concurrent);
            worker_ = std::thread(
                [concurrent = std::move(concurrent), seq_engine]() mutable {
                    concurrent(*seq_engine);
                });
        } else {
            worker_ = std::thread(std::move(loops.serial));
        }
    } catch (...) {
        request_stop_ = {};
        throw;
    }

    return true;
}

void LuceEngine::stop_serving() {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    if (!worker_.joinable()) {
        return;
    }

    request_stop_();
    worker_.join();
    request_stop_ = {};
}

} // namespace dflash::engine
