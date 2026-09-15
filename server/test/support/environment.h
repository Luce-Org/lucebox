#pragma once
#include <filesystem>
#include <stdexcept>
#include <system_error>

// Temporary path helpers shared by the aggregated test cases. Environment
// variable overrides belong in support/scoped_env.h (ScopedEnvVar), which
// restores the prior process state on scope exit — do not add raw setenv
// wrappers here.

inline std::filesystem::path test_tmp_path(const char * name) {
    std::error_code ec;
    std::filesystem::path root = std::filesystem::temp_directory_path(ec);
    if (ec) {
        throw std::runtime_error("failed to resolve temporary directory: " +
                                 ec.message());
    }
    return root / name;
}

inline void remove_test_path(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        throw std::runtime_error("failed to remove test path " +
                                 path.string() + ": " + ec.message());
    }
}
