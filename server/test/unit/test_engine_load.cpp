// Unit tests for EngineLoad, the /props "load" counts and the --queue-limit
// gate: no GPU, no model files.

#include "CppUnitTestFramework.hpp"
#include "server/engine_load.h"

#include <limits>

using namespace luce::common;

namespace {
struct EngineLoadFixture {};
}

// A request with tool checks holds its place in the line from the gate to its
// reply: when a pass leaves its slot the place is taken back in the same step,
// so a request arriving between two passes finds the engine full.
TEST_CASE(EngineLoadFixture, a_tool_checked_request_holds_the_line_between_passes) {
    EngineLoad load(0);
    load.configure(1, 0);
    EngineLoad::Place checked(load), other(load);
    REQUIRE(load.enter(checked, true));
    load.seat(&checked, 1, 0);
    load.vacate(&checked, 0, 0);
    CHECK(!load.enter(other, true));
    CHECK(load.to_json()["slots_busy"] == 0 && load.to_json()["waiting"] == 1);
    CHECK(load.enter(checked, false));  // the next pass, in the same place
    load.seat(&checked, 1, 0);
    load.vacate(&checked, 0, 0);
    load.leave(checked);                // the reply goes out
    CHECK(load.enter(other, true));
}

// A limit too large to add to the slots still lets requests in.
TEST_CASE(EngineLoadFixture, the_largest_queue_limit_lets_requests_in) {
    EngineLoad load(std::numeric_limits<int>::max());
    load.configure(4, 0);
    EngineLoad::Place first(load);
    CHECK(load.enter(first, true));
}
