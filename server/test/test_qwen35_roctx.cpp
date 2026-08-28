#include "qwen35/qwen35_roctx.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {
int failures = 0;
std::vector<std::string> events;

#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
} } while (0)

int push(const char * message) { events.emplace_back(message); return 0; }
int pop() { events.emplace_back("pop"); return 0; }
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
    { Qwen35RoctxRange disabled("qwen35.graph_compute", {}, false, {push, pop}); }
    CHECK(events.empty());
    return failures == 0 ? 0 : 1;
}
