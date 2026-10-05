// Standalone host policy check: c++ -std=c++17 -Iserver/src -Iserver/deps/llama.cpp/ggml/include server/test/test_qwen4exp_chunk.cpp -o /tmp/test_qwen4exp_chunk
#include "qwen4exp/qwen4exp_chunk.h"
#include <cassert>
using luce::common::qwen4exp_fit_chunk;
int main() {
    auto workspace = [](int n) { return size_t(n) * 10; };
    assert(qwen4exp_fit_chunk(10000, 100000, 0, workspace) == 8960);
    assert(qwen4exp_fit_chunk(10000, 100000, 20000, workspace) == 6912);
    // Three additional 20K state slots force a smaller chunk on the same GPU.
    assert(qwen4exp_fit_chunk(10000, 100000, 60000, workspace) == 2816);
    assert(qwen4exp_fit_chunk(10000, 100000, 90000, workspace) == 0);
    assert(qwen4exp_fit_chunk(0, 100000, 0, workspace) == 0);
    assert(qwen4exp_fit_chunk(1, 12, 0, workspace) == 1);
    assert(qwen4exp_fit_chunk(1, 9, 0, workspace) == 0);
    assert(qwen4exp_fit_chunk(4096, 50000, 4040, workspace) == 4096); // exact fit
    assert(qwen4exp_fit_chunk(4096, 50000, 4041, workspace) == 3840);
    assert(qwen4exp_fit_chunk(4096, 50000, 0, [](int) { return SIZE_MAX; }) == 0);
}
