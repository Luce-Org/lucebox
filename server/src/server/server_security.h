#pragma once

#include <string>
#include <vector>

namespace dflash::common {

// The native server is IPv4-only today. Keep the exposure check aligned with
// the addresses accepted by HttpServer::run().
bool is_loopback_host(const std::string & host);

// Authenticate a complete Authorization header against the configured key.
// The comparison is constant-time with respect to matching key bytes.
bool bearer_authorized(const std::string & authorization,
                       const std::string & api_key);

bool cors_origin_valid(const std::string & origin);

// Empty origins are non-browser requests and do not require CORS headers.
bool cors_origin_allowed(const std::string & origin,
                         const std::vector<std::string> & allowlist);

} // namespace dflash::common
