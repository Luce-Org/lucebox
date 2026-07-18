#include "server_security.h"

#include <cctype>
#include <cstdint>

namespace dflash::common {

bool is_loopback_host(const std::string & host) {
    if (host == "localhost") return true;
    if (host.rfind("127.", 0) != 0) return false;

    // inet_pton() performs the final address validation in HttpServer::run().
    // Here, require only the conventional IPv4 loopback prefix and reject an
    // empty suffix so malformed values cannot bypass the startup boundary.
    return host.size() > 4;
}

static bool ascii_iequal_prefix(const std::string & value, const char * prefix) {
    size_t i = 0;
    for (; prefix[i] != '\0'; ++i) {
        if (i >= value.size()) return false;
        const auto lhs = static_cast<unsigned char>(value[i]);
        const auto rhs = static_cast<unsigned char>(prefix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) return false;
    }
    return true;
}

bool bearer_authorized(const std::string & authorization,
                       const std::string & api_key) {
    if (api_key.empty()) return true;

    static constexpr char kScheme[] = "Bearer ";
    if (!ascii_iequal_prefix(authorization, kScheme)) return false;
    const std::string supplied = authorization.substr(sizeof(kScheme) - 1);

    // Include the length mismatch in the accumulated result and always scan
    // the configured-key length. This avoids an early exit on matching bytes.
    size_t diff = supplied.size() ^ api_key.size();
    for (size_t i = 0; i < api_key.size(); ++i) {
        const uint8_t got = i < supplied.size()
            ? static_cast<uint8_t>(supplied[i])
            : 0;
        diff |= static_cast<size_t>(got ^ static_cast<uint8_t>(api_key[i]));
    }
    return diff == 0;
}

bool cors_origin_valid(const std::string & origin) {
    if (origin == "*" || origin.find('\r') != std::string::npos ||
        origin.find('\n') != std::string::npos) {
        return false;
    }
    size_t authority = std::string::npos;
    if (origin.rfind("http://", 0) == 0) authority = 7;
    if (origin.rfind("https://", 0) == 0) authority = 8;
    return authority != std::string::npos && authority < origin.size() &&
           origin.find_first_of("/?#", authority) == std::string::npos &&
           origin.find_first_of(" \t@", authority) == std::string::npos;
}

bool cors_origin_allowed(const std::string & origin,
                         const std::vector<std::string> & allowlist) {
    if (origin.empty()) return true;
    if (!cors_origin_valid(origin)) return false;
    for (const auto & allowed : allowlist) {
        if (origin == allowed) return true;
    }
    return false;
}

} // namespace dflash::common
