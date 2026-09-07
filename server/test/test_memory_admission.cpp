#include "memory_admission.h"
#include <cassert>
#include <iostream>
using namespace dflash_memory;
void put(const std::filesystem::path &p,const std::string &s) {
    std::filesystem::create_directories(p.parent_path()); std::ofstream(p)<<s;
}
int main() {
    auto dir=std::filesystem::temp_directory_path()/"lucebox-memory-admission-test";
    std::filesystem::remove_all(dir); auto proc=dir/"proc", cg=dir/"cg";
    put(proc/"meminfo","MemTotal: 999999 kB\nMemAvailable: 8192 kB\nSwapFree: 999999 kB\n");
    put(proc/"self/cgroup","0::/parent/worker\n");
    assert(available(proc,cg)==8*MiB);
    put(cg/"memory.max",std::to_string(20*MiB)); put(cg/"memory.current",std::to_string(19*MiB));
    put(cg/"memory.stat","inactive_file "+std::to_string(3*MiB)+"\n");
    assert(available(proc,cg)==4*MiB);
    put(cg/"parent/memory.max",std::to_string(10*MiB)); put(cg/"parent/memory.current",std::to_string(8*MiB));
    assert(available(proc,cg)==2*MiB);
    put(cg/"parent/memory.current",std::to_string(11*MiB)); assert(available(proc,cg)==0);
    put(cg/"parent/memory.max","max"); assert(available(proc,cg)==4*MiB);
    assert(fits(10,7,3)); assert(!fits(10,8,3)); assert(!fits(unknown,1,1));
    assert(add(unknown-1,2)==unknown); assert(multiply(unknown,2)==unknown);
    setenv("LUCEBOX_RAM_BYTES_PER_TOKEN","0",1); assert(request_error(100).empty());
    setenv("LUCEBOX_RAM_BYTES_PER_TOKEN","18446744073709551615",1);
    auto error=request_error(100); assert(!error.empty());
    setenv("LUCEBOX_RAM_RESERVE_BYTES","18446744073709551615",1);
    assert(request_error(100).rfind("memory_unavailable:",0)==0);
    setenv("LUCEBOX_RAM_BYTES_PER_TOKEN","10",1);
    setenv("LUCEBOX_RAM_WORK_BYTES_PER_TOKEN","1",1);
    setenv("LUCEBOX_RAM_FIXED_BYTES","10",1);
    setenv("LUCEBOX_RAM_RESERVE_BYTES","10",1);
    uint64_t headroom=50;
    unsigned evictions=0;
    auto reclaimed=admit(10,[&]{return headroom;},[&]{++evictions;headroom=150;return true;});
    assert(reclaimed.error.empty() && reclaimed.evicted==1 && !reclaimed.without_cache);
    assert(evictions==1);
    auto uncached=admit(10,[]{return uint64_t(50);},[]{return false;});
    assert(uncached.error.empty() && uncached.without_cache);
    auto impossible=admit(10,[]{return uint64_t(25);},[]{return false;});
    assert(impossible.error.rfind("memory_unavailable:",0)==0);
    auto missing=admit(10,[]{return unknown;},[]{return false;});
    assert(!missing.error.empty());
    std::filesystem::remove_all(dir);
    std::cout<<"PASS: host, cgroup root/ancestors, reclaim, no swap, limits, saturation, admission errors\n";
}
