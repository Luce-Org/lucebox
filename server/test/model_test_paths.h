#pragma once

#include "CppUnitTestFramework.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace luce_test {

inline constexpr std::string_view kQwen35ModelEnv = "LUCE_TEST_MODEL_QWEN35";
inline constexpr std::string_view kDraftModelEnv = "LUCE_TEST_MODEL_DRAFT";
inline constexpr std::string_view kQwen38ModelEnv = "LUCE_TEST_MODEL_QWEN38";
inline constexpr std::string_view kQwen38DraftModelEnv = "LUCE_TEST_MODEL_QWEN38_DRAFT";
inline constexpr std::string_view kDeepSeek4ModelEnv = "LUCE_TEST_MODEL_DEEPSEEK4";
inline constexpr std::string_view kLagunaModelEnv = "LUCE_TEST_MODEL_LAGUNA";

inline std::filesystem::path require_model(const std::string_view environment_variable) {
    const std::string name(environment_variable);
    const char * const value = std::getenv(name.c_str());
    if (!value || !*value) {
        throw CppUnitTestFramework::TestSkippedException(
            "required test asset is not configured: set " + name
        );
    }

    const std::filesystem::path path(value);
    std::error_code error;
    const bool valid = std::filesystem::is_regular_file(path, error);
    if (!valid || error) {
        throw CppUnitTestFramework::TestSkippedException(
            "configured test asset is unavailable for " + name + ": " + path.string()
        );
    }

    return path;
}

} // namespace luce_test
