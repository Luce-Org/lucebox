#include "CppUnitTestFramework.hpp"
#include "common/platform_env.h"
#include "server/socket_handle.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace CppUnitTestFramework;
using namespace dflash::common;

namespace {

struct PlatformCompatFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

}  // namespace

TEST_CASE(PlatformCompatFixture, platform_environment_and_socket_compatibility) {
    constexpr const char * kEnvName = "DFLASH_PLATFORM_COMPAT_TEST";

    if (unset_environment_variable(kEnvName) != 0) {
        CHECK(false);
    }
    if (set_environment_variable(kEnvName, "original", true) != 0) {
        CHECK(false);
    }
    if (set_environment_variable(kEnvName, "replacement", false) != 0) {
        CHECK(false);
    }
    const char * value = std::getenv(kEnvName);
    if (value == nullptr || std::strcmp(value, "original") != 0) {
        CHECK(false);
    }
    if (set_environment_variable(kEnvName, "replacement", true) != 0) {
        CHECK(false);
    }
    value = std::getenv(kEnvName);
    if (value == nullptr || std::strcmp(value, "replacement") != 0) {
        CHECK(false);
    }
    if (unset_environment_variable(kEnvName) != 0 ||
        std::getenv(kEnvName) != nullptr) {
        CHECK(false);
    }

    if (socket_is_valid(kInvalidSocket)) {
        CHECK(false);
    }

#if defined(_WIN32)
    static_assert(sizeof(SocketHandle) == sizeof(void *),
                  "Win64 socket handles must remain pointer-width");
    WSADATA wsa_data{};
    const int wsa_error = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (wsa_error != 0) {
        CHECK(false);
    }
#endif

    const SocketHandle socket_handle = ::socket(AF_INET, SOCK_STREAM, 0);
    if (!socket_is_valid(socket_handle)) {
#if defined(_WIN32)
        WSACleanup();
#endif
        CHECK(false);
        return;
    }

#if defined(_WIN32)
    closesocket(socket_handle);
    WSACleanup();
#else
    ::close(socket_handle);
#endif

}
