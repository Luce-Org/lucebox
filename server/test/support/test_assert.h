#pragma once
#include "CppUnitTestFramework.hpp"
#include <string>

namespace { struct ServerUnitFixture {}; }

#define TEST_ASSERT(expr) REQUIRE_TRUE(expr)
#define TEST_ASSERT_MSG(expr, msg) do { \
    if (!(expr)) { \
        CppUnitTestFramework::CommonFixture::HandleAssert( \
            CppUnitTestFramework::AssertType::Throw, _CPPUTF_ASSERT_LOCATION, \
            CppUnitTestFramework::AssertException(std::string(#expr) + " — " + std::string(msg))); \
    } \
} while (0)
