#pragma once
#include "hybrid_policy.h"
#include "common/model_backend.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <chrono>
namespace dflash::common::hybrid {
using Json=nlohmann::json;
struct Region {
    int message=0;size_t start=0,end=0;std::string source,replacement;
    Json envelope;
    int part=-1;
    Json part_envelope;
};
struct Representation {
    std::vector<Region> regions;
    std::vector<int32_t> tokens;
    std::string fingerprint,raw_key,generation;
    uint64_t bytes() const {
        uint64_t n=tokens.size()*sizeof(int32_t)+fingerprint.size()+raw_key.size()+generation.size();
        for(auto&r:regions)n+=r.source.size()+r.replacement.size()+r.envelope.dump().size()+r.part_envelope.dump().size()+sizeof(r);
        return n;
    }
};
struct Capture {
    int slot=-1,requested=0,actual=0;
    bool primary=false;
    uint64_t bytes=0;
};
struct Entry {
    int slot=-1,position=0;
    std::string owner;
    bool primary=false,full=false,gpu=true;
    uint64_t bytes=0,last_used=0,hits=0;
    double saved_seconds=0;
    std::shared_ptr<const Representation> representation;
};
inline uint64_t ram_available() {
    std::ifstream f("/proc/meminfo");std::string line;
    while(std::getline(f,line))if(line.rfind("MemAvailable:",0)==0) {
        return std::stoull(line.substr(13))*1024;
    }
    return 0;
}
class Cache {
public:
    std::vector<Entry> entries;
    std::set<std::string> owners;
    std::map<std::string,Estimate> costs;
    struct Anchor {Estimate input,prefill,scoring;};
    std::map<std::string,Anchor> anchors;
    struct Forecast {std::optional<double> seconds;std::string kind="unknown",source;double ratio=0;};
    static std::vector<std::string> key_parts(const std::string& key) {
        std::vector<std::string> parts;size_t start=0,end;
        while((end=key.find(':',start))!=std::string::npos){parts.push_back(key.substr(start,end-start));start=end+1;}
        parts.push_back(key.substr(start));return parts;
    }
    uint64_t tick=0;
    uint64_t gpu_checkpoint_budget=9*GiB,ram_checkpoint_budget=8*GiB;
    std::string calibration_path;
    // End-to-end observations already include movement and capture. Never add movement twice.
    // Existing context and new work have independent buckets, including small tool turns.
    static std::string cost_key(Path path,int length,int prefix=0,int output=0,
                                bool gpu=true,bool compressed=false,const std::string& identity="",
                                int uncached_scoring=0) {
        return "v2:"+identity+":w8192:r20:"+name(path)+":"+
            std::to_string(length_bucket(prefix))+":"+
            std::to_string(length_bucket(std::max(0,length-prefix)))+":"+
            std::to_string(output)+":"+(gpu?"gpu":"ram")+":"+
            (compressed?"lossy":"exact")+":"+std::to_string(length_bucket(uncached_scoring));
    }
    void load_calibration(const std::string &path) {
        calibration_path=path;if(path.empty())return;
        std::ifstream f(path);if(!f)return;
        try {Json j;f>>j;for(auto it=j.begin();it!=j.end();++it) {
            auto &e=costs[it.key()];e.value=it.value().at("seconds");e.observations=it.value().at("samples");
            if(!std::isfinite(e.value)||e.value<0||e.observations<3){e={};continue;}
            if(it.value().contains("anchor")) {
                const auto& values=it.value()["anchor"];auto& a=anchors[it.key()];
                a.input.value=values.at("input_tokens");a.prefill.value=values.at("prefill_seconds");a.scoring.value=values.at("scoring_tokens");
                a.input.observations=a.prefill.observations=a.scoring.observations=values.at("samples").get<unsigned>();
            }
        }}catch(...){costs.clear();anchors.clear();}
    }
    std::optional<double> estimate(const std::string& key) const {
        auto it=costs.find(key);
        return it==costs.end()?std::nullopt:it->second.calibrated();
    }
    Forecast forecast(const std::string& key,Path path,int input,int scoring,int regions,bool allow_projection) const {
        if(auto measured=estimate(key))return {measured,"measured",key,1.};
        Forecast best;if(!allow_projection || (path!=Path::Dense && path!=Path::Compress))return best;
        const auto query=key_parts(key);if(query.size()!=12 || query[5]!="0" || query[8]!="gpu" || query[11]=="capture0")return best;
        for(const auto& [source,anchor]:anchors) {
            const auto parts=key_parts(source);if(parts.size()!=12 || parts[11]=="capture0")continue;
            bool same=true;for(int field:{0,1,2,3,4,5,7,8,9})if(parts[field]!=query[field])same=false;
            const auto measured=estimate(source);
            if(!same || !measured || !anchor.input.calibrated() || !anchor.prefill.calibrated() || !anchor.scoring.calibrated())continue;
            const auto projected=project_cold(path,input,scoring,regions,anchor.input.value,anchor.scoring.value,*measured,anchor.prefill.value,anchor.input.observations);
            if(!projected.known)continue;
            // Prefer the nearest measured input; leave equally near anchors deterministic.
            if(!best.seconds || projected.ratio<best.ratio)best={projected.seconds,"conservative_projection",source,projected.ratio};
        }
        return best;
    }
    void observe(const std::string& key,double seconds,const Json& trace=Json::object()) {
        costs[key].observe(seconds);
        if(trace.value("projection_anchor",false)) {
            auto& a=anchors[key];a.input.observe(trace.at("input_tokens").get<double>());
            a.prefill.observe(trace.at("prefill_seconds").get<double>());
            a.scoring.observe(trace.value("uncached_scoring_tokens",0.));
        }
    }
    uint64_t metadata_bytes() const {
        std::set<const Representation*> seen;uint64_t n=0;
        for(const auto&e:entries)if(e.representation && seen.insert(e.representation.get()).second)n+=e.representation->bytes();
        return n;
    }
    Entry *find(int slot){for(auto &e:entries)if(e.slot==slot)return &e;return nullptr;}
    const Entry *find(int slot) const {for(auto &e:entries)if(e.slot==slot)return &e;return nullptr;}
    bool admitted(const std::string &owner) const {return !owner.empty()&&owners.count(owner);}
    void admit(const std::string &owner){if(!owner.empty()&&owners.size()<3)owners.insert(owner);}
    int free_slot() const {for(int s=0;s<63;++s)if(!find(s))return s;return -1;}
    void erase(ModelBackend &backend,int slot) {
        backend.snapshot_free(slot);
        entries.erase(std::remove_if(entries.begin(),entries.end(),[&](auto&e){return e.slot==slot;}),entries.end());
    }
    void release(ModelBackend &backend,const std::string &owner) {
        std::vector<int> slots;for(auto&e:entries)if(e.owner==owner)slots.push_back(e.slot);
        for(int s:slots)erase(backend,s);owners.erase(owner);
    }
    Memory memory(ModelBackend &backend,int source,uint64_t work,uint64_t allocation=0,uint64_t extra_metadata=0) const {
        Memory m;m.gpu_budget=gpu_checkpoint_budget;m.ram_budget=ram_checkpoint_budget;m.gpu_free=backend.cache_gpu_free_bytes();m.ram_available=ram_available();
        m.reclaimable_gpu=backend.cache_reclaimable_scratch_bytes();
        m.work_gpu=work;m.new_gpu=allocation;m.new_ram=extra_metadata+16*1024*1024;m.metadata_bytes=metadata_bytes();
        for(auto &e:entries)m.residents.push_back({e.slot,e.bytes,e.gpu,e.primary,e.slot==source,e.saved_seconds,e.last_used});
        return m;
    }
    bool execute(ModelBackend &backend,const ResourcePlan &plan) {
        if(!plan.feasible)return false;
        if(plan.release_scratch)backend.release_scratch();
        for(int s:plan.evict){auto*e=find(s);if(!e || e->primary)return false;erase(backend,s);}
        for(int s:plan.spill){auto*e=find(s);if(!e || !backend.snapshot_move(s,false))return false;e->gpu=false;e->bytes=backend.snapshot_bytes(s);}
        return true;
    }
    void commit(ModelBackend &backend,Entry entry) {
        std::vector<int> replaced;
        if(entry.primary)for(auto&e:entries)if(e.primary && e.owner==entry.owner)replaced.push_back(e.slot);
        entry.last_used=++tick;entries.push_back(std::move(entry));
        for(int slot:replaced)erase(backend,slot);
    }
};
} // namespace
