#include "server/server_security.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

int main() {
    CHECK(is_loopback_host("127.0.0.1"));
    CHECK(is_loopback_host("127.42.0.9"));
    CHECK(is_loopback_host("localhost"));
    CHECK(!is_loopback_host("0.0.0.0"));
    CHECK(!is_loopback_host("192.168.1.10"));
    CHECK(!is_loopback_host("127."));

    const std::string key = "unit-test-key";
    CHECK(bearer_authorized("", ""));
    CHECK(!bearer_authorized("", key));
    CHECK(!bearer_authorized("Basic unit-test-key", key));
    CHECK(bearer_authorized("Bearer unit-test-key", key));
    CHECK(bearer_authorized("bearer unit-test-key", key));
    CHECK(!bearer_authorized("Bearer unit-test-kez", key));
    CHECK(!bearer_authorized("Bearer " + key + std::string(256, 'x'), key));

    const std::vector<std::string> origins = {
        "https://console.example.test",
        "http://127.0.0.1:3000",
    };
    CHECK(cors_origin_valid("https://console.example.test"));
    CHECK(cors_origin_valid("http://127.0.0.1:3000"));
    CHECK(!cors_origin_valid("*"));
    CHECK(!cors_origin_valid("https://console.example.test/path"));
    CHECK(!cors_origin_valid("https://console.example.test\r\nInjected: yes"));
    CHECK(cors_origin_allowed("", origins));
    CHECK(cors_origin_allowed("https://console.example.test", origins));
    CHECK(!cors_origin_allowed("https://evil.example.test", origins));
    CHECK(!cors_origin_allowed("https://console.example.test/", origins));
    CHECK(!cors_origin_allowed("https://console.example.test", {}));
    return 0;
}
