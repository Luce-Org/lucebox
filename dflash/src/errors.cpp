// Thread-safe last-error string used by loaders and graph builders.
// Consumed by tests and the test_dflash driver via dflash27b_last_error().

#include "dflash27b.h"
#include "gemma4.h"
#include "internal.h"

#include <mutex>
#include <string>

namespace dflash27b {

namespace {
std::mutex g_err_mu;
std::string g_last_error;
thread_local std::string t_err_buf;  // per-thread snapshot for safe c_str return
}

void set_last_error(std::string msg) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_error = std::move(msg);
}

} // namespace dflash27b

extern "C" const char * dflash27b_last_error(void) {
    std::lock_guard<std::mutex> lk(dflash27b::g_err_mu);
    dflash27b::t_err_buf = dflash27b::g_last_error;  // copy under lock
    return dflash27b::t_err_buf.c_str();              // safe: thread-local
}

extern "C" const char * gemma4_last_error(void) {
    std::lock_guard<std::mutex> lk(dflash27b::g_err_mu);
    dflash27b::t_err_buf = dflash27b::g_last_error;
    return dflash27b::t_err_buf.c_str();
}
