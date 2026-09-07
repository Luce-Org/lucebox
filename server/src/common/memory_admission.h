#pragma once
// Linux host-RAM admission. Swap is deliberately not counted as usable headroom.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace dflash_memory {
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t unknown = std::numeric_limits<uint64_t>::max();
inline uint64_t add(uint64_t a, uint64_t b) { return a > unknown-b ? unknown : a+b; }
inline uint64_t multiply(uint64_t a, uint64_t b) { return b && a > unknown/b ? unknown : a*b; }
inline uint64_t env_bytes(const char *name, uint64_t fallback) {
    const char *s = std::getenv(name);
    if (!s || !*s || *s=='-') return fallback;
    char *end = nullptr;
    const auto n = std::strtoull(s, &end, 10);
    return end && !*end ? n : fallback;
}
inline uint64_t number(const std::filesystem::path &path) {
    std::ifstream f(path); std::string s; f >> s;
    if (s.empty() || s=="max" || s[0]=='-') return unknown;
    try { size_t end; auto n=std::stoull(s,&end); return end==s.size()?n:unknown; }
    catch (...) { return unknown; }
}
inline uint64_t stat_value(const std::filesystem::path &path, const std::string &key) {
    std::ifstream f(path); std::string line;
    while (std::getline(f,line)) {
        std::istringstream row(line); std::string k; uint64_t n;
        if (row >> k >> n && k==key) return n;
    }
    return unknown;
}
inline uint64_t available(const std::filesystem::path &proc="/proc",
                          const std::filesystem::path &cgroups="/sys/fs/cgroup") {
    uint64_t result=stat_value(proc/"meminfo", "MemAvailable:");
    if (result!=unknown) result=multiply(result,1024);
    // Check the namespace root and every visible ancestor of our cgroup.
    // LXC may expose the container limit only through its virtual /proc/meminfo.
    auto check=[&](const std::filesystem::path &p) {
        auto limit=number(p/"memory.max"), used=number(p/"memory.current");
        if (limit==unknown || used==unknown) return;
        auto reclaim=stat_value(p/"memory.stat","inactive_file");
        if (reclaim!=unknown) used-=std::min(used,reclaim);
        result=std::min(result, used>=limit?uint64_t(0):limit-used);
    };
    check(cgroups);
    std::ifstream f(proc/"self/cgroup"); std::string line;
    while (std::getline(f,line)) if (line.rfind("0::",0)==0) {
        auto rel=std::filesystem::path(line.substr(3)).relative_path().lexically_normal();
        bool safe=true; for (const auto &part:rel) if (part=="..") safe=false;
        if (!safe) continue;
        auto p=cgroups;
        for (const auto &part:rel) { p/=part; check(p); }
    }
    return result;
}
inline uint64_t reserve() { return env_bytes("LUCEBOX_RAM_RESERVE_BYTES",2048*MiB); }
inline bool fits(uint64_t headroom,uint64_t required,uint64_t reserved) {
    return headroom!=unknown && reserved<=headroom && required<=headroom-reserved;
}
// Deployment-specific peak-increment estimate, not a universal model formula.
// Current resident caches are already included in the measured headroom.
inline std::string request_error_at(uint64_t tokens, uint64_t headroom, bool with_cache=true) {
    auto profile=env_bytes("LUCEBOX_RAM_BYTES_PER_TOKEN",0);
    if (!profile) return {};
    if (headroom==unknown) return "memory_unavailable: Cannot read host RAM headroom; request was not started.";
    auto per_token=with_cache ? profile : env_bytes("LUCEBOX_RAM_WORK_BYTES_PER_TOKEN",profile);
    auto required=add(multiply(tokens,per_token),env_bytes("LUCEBOX_RAM_FIXED_BYTES",1024*MiB));
    if (fits(headroom,required,reserve())) return {};
    return "memory_unavailable: Insufficient host RAM for this request after cache reclamation. Estimated additional RAM " +
        std::to_string(required/MiB) + " MiB; available " + std::to_string(headroom/MiB) +
        " MiB; safety reserve " + std::to_string(reserve()/MiB) +
        " MiB. Retry later or submit a smaller request.";
}
inline std::string request_error(uint64_t tokens) { return request_error_at(tokens,available()); }
struct Admission {
    std::string error;
    bool without_cache=false;
    unsigned evicted=0;
};
// Only call on the serialized worker, before any live sequence or restore exists.
// Cache is optional. Reclaim it before rejecting; then try uncached execution.
template<class Probe, class Evict>
Admission admit(uint64_t tokens, Probe probe, Evict evict) {
    Admission out;
    out.error=request_error_at(tokens,probe());
    while (!out.error.empty() && out.evicted<128 && evict()) {
        ++out.evicted;
        out.error=request_error_at(tokens,probe());
    }
    if (!out.error.empty()) {
        out.without_cache=true;
        out.error=request_error_at(tokens,probe(),false);
    }
    return out;
}
}
