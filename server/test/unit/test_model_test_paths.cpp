#include "CppUnitTestFramework.hpp"
#include "model_test_paths.h"
#include "scoped_env.h"

#include <filesystem>
using namespace CppUnitTestFramework;

struct ModelTestPaths : CommonFixture {
    using CommonFixture::CommonFixture;
};
TEST_CASE(ModelTestPaths, RejectsMissingRequiredAsset) {
    constexpr char kTestEnvironmentVariable[] = "LUCE_TEST_MODEL_PATHS_TEST";
    luce_test::ScopedEnvVar unset(kTestEnvironmentVariable, nullptr);

    REQUIRE_THROW(
        TestSkippedException,
        luce_test::require_model(kTestEnvironmentVariable)
    );
}

TEST_CASE(ModelTestPaths, ResolvesConfiguredDirectory) {
    constexpr char kTestEnvironmentVariable[] = "LUCE_TEST_MODEL_PATHS_TEST";
    const auto temporary_directory = std::filesystem::temp_directory_path();
    luce_test::ScopedEnvVar set(
        kTestEnvironmentVariable,
        temporary_directory.string().c_str()
    );

    REQUIRE_EQUAL(
        luce_test::require_directory(kTestEnvironmentVariable).string(),
        temporary_directory.string()
    );
}
