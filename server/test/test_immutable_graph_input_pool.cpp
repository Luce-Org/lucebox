#include "CppUnitTestFramework.hpp"
#include "common/immutable_graph_input_pool.h"

#include <cstdint>
#include <cstring>

using namespace dflash::common;

namespace {
struct ImmutableGraphInputPoolFixture {};

float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
} // namespace

TEST_CASE(ImmutableGraphInputPoolFixture, identical_layer_inputs_share_storage) {
    ImmutableGraphInputPool<float> pool;
    const std::vector<float> mask{0.0f, -1e30f, 0.0f, -1e30f};
    const auto first = pool.intern(mask);
    for (int layer = 1; layer < 43; ++layer) {
        const auto next = pool.intern(mask);
        CHECK(next == first);
        CHECK(std::memcmp(next->data(), mask.data(), mask.size() * sizeof(float)) == 0);
    }
}

TEST_CASE(ImmutableGraphInputPoolFixture, different_inputs_are_not_aliased) {
    ImmutableGraphInputPool<float> pool;
    const auto first = pool.intern({0.0f, -1e30f});
    CHECK(pool.intern({-1e30f, 0.0f}) != first);
    CHECK(pool.intern({0.0f}) != first);
    CHECK(pool.intern({0.0f, -1e30f, 0.0f}) != first);
}

TEST_CASE(ImmutableGraphInputPoolFixture, float_bit_patterns_are_preserved) {
    ImmutableGraphInputPool<float> pool;
    CHECK(pool.intern({0.0f}) != pool.intern({-0.0f}));
    const auto nan = pool.intern({from_bits(0x7fc00001u)});
    CHECK(nan == pool.intern({from_bits(0x7fc00001u)}));
    CHECK(nan != pool.intern({from_bits(0x7fc00002u)}));
}

TEST_CASE(ImmutableGraphInputPoolFixture, empty_inputs_and_owner_lifetime) {
    ImmutableGraphInputPool<float> pool;
    CHECK(pool.intern({}) == pool.intern({}));
    auto held = pool.intern({1.0f, 2.0f});
    std::weak_ptr<const std::vector<float>> weak = held;
    pool.clear();
    CHECK(!weak.expired());
    CHECK((*held)[1] == 2.0f);
    CHECK(pool.intern({1.0f, 2.0f}) != held);
    held.reset();
    CHECK(weak.expired());
}

TEST_CASE(ImmutableGraphInputPoolFixture, integer_inputs_can_reuse_the_same_pool) {
    ImmutableGraphInputPool<int32_t> pool;
    CHECK(pool.intern({1, 2, 3}) == pool.intern({1, 2, 3}));
    CHECK(pool.intern({1, 2, 3}) != pool.intern({3, 2, 1}));
}
