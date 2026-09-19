#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#define dflash_setenv(name, value) _putenv_s(name, value)
#define dflash_unsetenv(name) _putenv_s(name, "")
#else
#define dflash_setenv(name, value) setenv(name, value, 1)
#define dflash_unsetenv(name) unsetenv(name)
#endif

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
}
