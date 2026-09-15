#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dflash::common::hybrid {
constexpr uint64_t GiB = 1024ull*1024*1024;
enum class Path { Prefix, Full, Frozen, Dense, Compress };
inline const char * name(Path p) {
    switch(p) {case Path::Prefix:return "prefix";case Path::Full:return "full";
    case Path::Frozen:return "frozen";case Path::Dense:return "dense";case Path::Compress:return "compress";}
    return "invalid";
}
struct Candidate {
    Path path = Path::Dense;
    int id = -1, prefix = 0;
    bool compressed = false, gpu = true, valid = true, feasible = true;
    std::optional<double> seconds;
    bool projected = false; // Cold comparison bound, never an executable reuse challenger.
    bool reuse() const {return path==Path::Prefix || path==Path::Full || path==Path::Frozen;}
};
struct Decision {int index=-1; std::string reason="no_feasible_path";};
inline bool cheaper_reuse(const Candidate & a,const Candidate & b) {
    if(a.seconds && b.seconds) {
        if(std::abs(*a.seconds-*b.seconds)>.25)return *a.seconds<*b.seconds;
        if(a.compressed!=b.compressed)return !a.compressed;
        if(a.gpu!=b.gpu)return a.gpu;
    } else {
        if(a.gpu!=b.gpu)return a.gpu;
        if(a.compressed!=b.compressed)return !a.compressed;
    }
    if(a.prefix!=b.prefix)return a.prefix>b.prefix;
    return a.id<b.id;
}
inline Decision select(const std::vector<Candidate> & c,bool exact) {
    int r=-1,d=-1,p=-1;
    auto prefer_rebuild=[&](int a,int b) {
        if(b<0)return true;
        if(c[a].seconds && c[b].seconds)return *c[a].seconds<*c[b].seconds;
        if(c[a].seconds.has_value()!=c[b].seconds.has_value())return c[a].seconds.has_value();
        if(c[a].gpu!=c[b].gpu)return c[a].gpu;
        return c[a].prefix>c[b].prefix;
    };
    std::vector<int> reusable;
    for(int i=0;i<(int)c.size();++i) {
        const auto & x=c[i];
        if(!x.valid || !x.feasible || (exact && x.compressed))continue;
        if(x.reuse())reusable.push_back(i);
        else if(x.path==Path::Dense)d=i;
        else if(x.path==Path::Compress && !exact && prefer_rebuild(i,p))p=i;
    }
    const bool calibrated=std::all_of(reusable.begin(),reusable.end(),[&](int i){return c[i].seconds.has_value();});
    double minimum=INFINITY;
    if(calibrated)for(int i:reusable)minimum=std::min(minimum,*c[i].seconds);
    for(int i:reusable) {
        if(calibrated && *c[i].seconds>minimum+.25)continue;
        if(r<0){r=i;continue;}
        auto a=c[i],b=c[r];
        // Anchor ties to the global minimum, avoiding order-dependent chains of
        // pairwise 250ms ties. Missing costs use the documented conservative rank.
        if(calibrated){a.seconds=0.;b.seconds=0.;}else{a.seconds.reset();b.seconds.reset();}
        if(cheaper_reuse(a,b))r=i;
    }
    if(r>=0) {
        if(!c[r].seconds)return {r,"reuse_unknown_cost"};
        int best=r;
        for(int i=0;i<(int)c.size();++i)if(c[i].valid && c[i].feasible &&
            !c[i].reuse() && !c[i].projected && !(exact && c[i].compressed) &&
            !(exact && c[i].path==Path::Compress) && c[i].seconds &&
            *c[r].seconds-*c[i].seconds>=std::max(1.,.2**c[r].seconds) &&
            (best==r || *c[i].seconds<*c[best].seconds))best=i;
        return {best,best==r?"reuse_hysteresis":"rebuild_saves_20pct_and_1s"};
    }
    if(d>=0) {
        if(p>=0 && c[d].seconds && c[p].seconds &&
            *c[d].seconds-*c[p].seconds>=std::max(1.,.2**c[d].seconds))return {p,"cold_compression_saves"};
        return {d,"cold_dense"};
    }
    if(p>=0)return {p,"compression_only_feasible"};
    return {};
}
// Observation mode can report the proposed route, but executes only the exact baseline.
inline Decision shadow_baseline(const std::vector<Candidate>& candidates) {
    for (int i=0;i<(int)candidates.size();++i)
        if(candidates[i].path==Path::Dense && candidates[i].valid && candidates[i].feasible)
            return {i,"shadow_exact_baseline"};
    return {-1,"shadow_baseline_infeasible"};
}
struct Resident {
    int id=-1;
    uint64_t bytes=0;
    bool gpu=true,primary=false,in_use=false;
    double saved_seconds=0;
    uint64_t last_used=0;
};
struct Memory {
    uint64_t gpu_free=0,ram_available=0,gpu_budget=9*GiB,ram_budget=8*GiB;
    uint64_t gpu_headroom=2*GiB,ram_headroom=2*GiB,metadata_bytes=0;
    // Required additional allocations at peak, beyond currently live buffers.
    uint64_t work_gpu=0,new_gpu=0,new_ram=0,reclaimable_gpu=0;
    std::vector<Resident> residents;
};
struct ResourcePlan {bool feasible=false;bool release_scratch=false;std::vector<int> evict,spill;std::string reason;};
inline ResourcePlan plan_memory(Memory m) {
    ResourcePlan out;
    uint64_t gpu=0,ram=m.metadata_bytes;
    for(const auto & e:m.residents)(e.gpu?gpu:ram)+=e.bytes;
    auto fits=[&] {return m.gpu_free>=m.work_gpu+m.new_gpu+m.gpu_headroom &&
        gpu+m.new_gpu<=m.gpu_budget && ram+m.new_ram<=m.ram_budget &&
        m.ram_available>=m.new_ram+m.ram_headroom;};
    if(fits()){out.feasible=true;return out;}
    if(m.reclaimable_gpu) {
        out.release_scratch=true;m.gpu_free+=m.reclaimable_gpu;
        if(fits()){out.feasible=true;return out;}
    }
    auto optional=m.residents;
    std::stable_sort(optional.begin(),optional.end(),[](const Resident&a,const Resident&b){
        const double av=a.saved_seconds/std::max<uint64_t>(1,a.bytes),bv=b.saved_seconds/std::max<uint64_t>(1,b.bytes);
        return av==bv?a.last_used<b.last_used:av<bv;});
    for(const auto & e:optional) {
        if(e.primary || e.in_use)continue;
        out.evict.push_back(e.id);
        if(e.gpu){gpu-=e.bytes;m.gpu_free+=e.bytes;}else{ram-=e.bytes;m.ram_available+=e.bytes;}
        if(fits()){out.feasible=true;return out;}
    }
    auto inactive=m.residents;
    std::stable_sort(inactive.begin(),inactive.end(),[](auto&a,auto&b){return a.last_used<b.last_used;});
    for(const auto&e:inactive) {
        if(!e.gpu || !e.primary || e.in_use)continue;
        // Source stays allocated while the destination is copied and verified.
        if(ram+e.bytes+m.new_ram>m.ram_budget ||
           m.ram_available<e.bytes+m.new_ram+m.ram_headroom)continue;
        out.spill.push_back(e.id);gpu-=e.bytes;ram+=e.bytes;
        m.gpu_free+=e.bytes;m.ram_available-=e.bytes;
        if(fits()){out.feasible=true;return out;}
    }
    out.reason="checkpoint_or_working_memory_budget";
    return out;
}
struct MaintenancePlan {ResourcePlan resource;bool capture=false;};
inline MaintenancePlan plan_with_optional_capture(Memory memory,uint64_t capture_bytes) {
    const auto answer=memory;
    memory.new_gpu+=capture_bytes;
    auto plan=plan_memory(memory);
    if(plan.feasible)return {std::move(plan),capture_bytes>0};
    if(capture_bytes)return {plan_memory(answer),false};
    return {std::move(plan),false};
}
// A latency projection is a routing heuristic, never a memory-admission signal.
// For growth beyond a measured cold anchor, compare a discounted prefill-only
// dense estimate with an inflated quadratic compression estimate. Do not project
// reuse paths, shrinking inputs, multiple scoring regions, or more than 50% growth.
struct ColdProjection {bool known=false;double seconds=0,ratio=0;};
inline ColdProjection project_cold(Path path,int input,int scoring,int regions,
        double anchor_input,double anchor_scoring,double anchor_seconds,double anchor_prefill,
        unsigned samples) {
    if(samples<3 || input<32000 || anchor_input<=0 || anchor_seconds<=0 ||
       !std::isfinite(anchor_input) || !std::isfinite(anchor_scoring) ||
       !std::isfinite(anchor_seconds) || !std::isfinite(anchor_prefill))return {};
    const double ratio=input/anchor_input;
    if(ratio<1 || ratio>1.5)return {};
    if(path==Path::Dense && anchor_prefill>0)
        return {true,.9*anchor_prefill*ratio,ratio};
    if(path==Path::Compress && regions==1 && anchor_scoring>0 && scoring>0) {
        const double growth=std::max(ratio,scoring/anchor_scoring);
        if(growth>1.5 || growth<1 || scoring<.85*input || anchor_scoring<.85*anchor_input)return {};
        return {true,1.1*anchor_seconds*growth*growth,growth};
    }
    return {};
}
inline bool prefix_matches(const std::vector<int32_t>& a,const std::vector<int32_t>& b,int position) {
    return position>0 && position<=(int)a.size() && position<=(int)b.size() &&
        std::equal(a.begin(),a.begin()+position,b.begin());
}
struct Estimate {
    double value=0;unsigned observations=0;
    std::vector<double> initial;
    void observe(double x) {
        if(!std::isfinite(x)||x<0)return;
        if(observations<3) {
            initial.push_back(x);++observations;
            if(observations==3) {std::sort(initial.begin(),initial.end());value=initial[1];initial.clear();}
        } else {value=.8*value+.2*x;++observations;}
    }
    std::optional<double> calibrated() const {return observations>=3?std::optional<double>(value):std::nullopt;}
};
inline int length_bucket(int n){if(n<=0)return 0;for(int b:{256,512,1024,2048,4096,8192,16384,32768,65536,81920,131072})if(n<=b)return b;return 262144;}
} // namespace
