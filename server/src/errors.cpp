// Thread-safe last-error string used by loaders and graph builders.
// Consumed by tests and the test_dflash driver via luce_last_error().

#include "luce.h"
#include "internal.h"

#include <mutex>
#include <string>

namespace luce::common {

namespace {
std::mutex g_err_mu;
std::string g_last_error;
bool g_last_error_oom = false;
}

void set_last_error(std::string msg) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_error = std::move(msg);
    g_last_error_oom = false;
}

void set_last_oom_error(std::string msg) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_error = std::move(msg);
    g_last_error_oom = true;
}

bool last_error_is_oom() {
    std::lock_guard<std::mutex> lk(g_err_mu);
    return g_last_error_oom;
}

} // namespace luce::common

extern "C" const char * luce_last_error(void) {
    std::lock_guard<std::mutex> lk(luce::common::g_err_mu);
    return luce::common::g_last_error.c_str();
}
