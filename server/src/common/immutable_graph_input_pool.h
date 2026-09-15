#pragma once

#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace dflash::common {

// Share identical immutable host inputs within one graph-cache lifetime.
// Equality is byte-exact: floating-point signed zeros and NaN payloads must
// not change when an input is uploaded again. Keep the pool owner-local, not
// global, so releasing a model or replacing its topology releases its inputs.
template <typename T>
class ImmutableGraphInputPool {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    using Values = std::shared_ptr<const std::vector<T>>;

    Values intern(std::vector<T> values) {
        for (const auto & existing : entries_) {
            if (existing->size() == values.size() &&
                (values.empty() || std::memcmp(existing->data(), values.data(),
                                               values.size() * sizeof(T)) == 0)) {
                return existing;
            }
        }
        auto stored = std::make_shared<const std::vector<T>>(std::move(values));
        entries_.push_back(stored);
        return stored;
    }

    void clear() { entries_.clear(); }

private:
    std::vector<Values> entries_;
};

} // namespace dflash::common
