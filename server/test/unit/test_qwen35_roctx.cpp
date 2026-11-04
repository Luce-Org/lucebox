#include "qwen35/qwen35_roctx.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
int failures = 0;
std::vector<std::string> events;

#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
} } while (0)

int push(const char * message) { events.emplace_back(message); return 0; }
int pop() { events.emplace_back("pop"); return 0; }
int failing_push(const char * message) {
    events.emplace_back(std::string("fail:") + message);
    return -1;
}
} // namespace

int main() {
    CHECK(!qwen35_roctx_env_enabled(nullptr));
    CHECK(!qwen35_roctx_env_enabled("0"));
    CHECK(!qwen35_roctx_env_enabled("false-ish"));
    CHECK(qwen35_roctx_env_enabled("1"));
    CHECK(qwen35_roctx_env_enabled("YES"));

    {
        Qwen35RoctxRange range(
            "qwen35.graph_compute", {9, 12, 64, 2, 76, 511}, true,
            {push, pop});
        CHECK(events.size() == 1);
        CHECK(events[0] == "qwen35.graph_compute live=9 bucket=12 "
                           "prefill_tokens=64 prefill_segments=2 total_rows=76 "
                           "max_kv_len=511");
    }
    CHECK(events.size() == 2 && events[1] == "pop");

    events.clear();
    {
        Qwen35RoctxRange service_round(
            "qwen35.service_round", {4, -1, 0, 0, -1, -1}, true,
            {push, pop});
        {
            Qwen35RoctxRange target_step(
                "qwen35.concurrent_step", {4, 4, 0, 0, 32, 128}, true,
                {push, pop});
        }
    }
    CHECK(events.size() == 4);
    CHECK(events[0] == "qwen35.service_round live=4 "
                       "prefill_tokens=0 prefill_segments=0");
    CHECK(events[1] == "qwen35.concurrent_step live=4 bucket=4 "
                       "prefill_tokens=0 prefill_segments=0 total_rows=32 "
                       "max_kv_len=128");
    CHECK(events[2] == "pop");
    CHECK(events[3] == "pop");

    events.clear();
    { Qwen35RoctxRange disabled("qwen35.graph_compute", {}, false, {push, pop}); }
    CHECK(events.empty());

    // Single-request path: unset, the markers are off (one cached check).
    if (!std::getenv("LUCE_QWEN35_ROCTX")) {
        CHECK(!qwen35_roctx_enabled());
        CHECK(qwen35_roctx_callbacks().push == nullptr);
        { Qwen35RoctxScope off(kQwen35RoctxPrefill); }
        Qwen35RoctxPhases off;
        off.open(kQwen35RoctxDraft);
        CHECK(off.open_count() == 0);
        off.close();
    }

    // A scope pushes its name, unadorned, and pops at scope exit.
    events.clear();
    { Qwen35RoctxScope prefill(kQwen35RoctxPrefill, true, {push, pop}); }
    CHECK(events.size() == 2);
    CHECK(events.size() == 2 && events[0] == "qwen35.prefill" && events[1] == "pop");
    events.clear();
    { Qwen35RoctxScope disabled(kQwen35RoctxPrefill, false, {push, pop}); }
    CHECK(events.empty());

    // Phases: start/add pairs inside a step, the step scope around them.
    events.clear();
    {
        Qwen35RoctxPhases phases(true, {push, pop});
        {
            Qwen35RoctxScope step(kQwen35RoctxSpecStep, true, {push, pop});
            phases.open(kQwen35RoctxDraft);
            phases.close();
            phases.open(kQwen35RoctxVerify);
            CHECK(phases.open_count() == 1);
            phases.close();
            phases.close();  // nothing open: no pop
        }
    }
    const std::vector<std::string> expected_step = {
        "qwen35.spec.step", "qwen35.spec.draft", "pop",
        "qwen35.spec.verify", "pop", "pop"};
    CHECK(events == expected_step);

    // An early return between a pair: the destructor closes what is open.
    events.clear();
    {
        Qwen35RoctxPhases phases(true, {push, pop});
        phases.open(kQwen35RoctxRollback);
        phases.open(kQwen35RoctxReplay);
    }
    const std::vector<std::string> expected_drain = {
        "qwen35.spec.rollback", "qwen35.spec.replay", "pop", "pop"};
    CHECK(events == expected_drain);

    // A failed push is not counted, so it is never popped.
    events.clear();
    {
        Qwen35RoctxPhases phases(true, {failing_push, pop});
        phases.open(kQwen35RoctxSnapshot);
        CHECK(phases.open_count() == 0);
        phases.close();
    }
    CHECK(events.size() == 1 && events[0] == "fail:qwen35.spec.snapshot");

    // Disabled phases never call back.
    events.clear();
    {
        Qwen35RoctxPhases phases(false, {push, pop});
        phases.open(kQwen35RoctxFeatureCopy);
        phases.close();
    }
    CHECK(events.empty());
    return failures == 0 ? 0 : 1;
}
