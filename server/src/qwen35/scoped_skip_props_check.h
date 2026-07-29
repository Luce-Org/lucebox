#pragma once

#include <utility>

namespace dflash::qwen35 {

// Keeps ggml-cuda's thread-local graph fast path scoped to one decode request.
template <class SetFn>
class ScopedSkipPropsCheck {
public:
    ScopedSkipPropsCheck(SetFn set, bool skip) : set_(std::move(set)) { set_(skip); }
    ~ScopedSkipPropsCheck() { set_(false); }

    ScopedSkipPropsCheck(const ScopedSkipPropsCheck &) = delete;
    ScopedSkipPropsCheck & operator=(const ScopedSkipPropsCheck &) = delete;

    void set(bool skip) { set_(skip); }

private:
    SetFn set_;
};

template <class SetFn>
ScopedSkipPropsCheck(SetFn, bool) -> ScopedSkipPropsCheck<SetFn>;

}  // namespace dflash::qwen35
