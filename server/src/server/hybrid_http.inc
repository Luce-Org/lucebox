// Included inside dflash::common; all mutation here runs on the worker thread.
namespace {
hybrid::Capture plan_hybrid_capture(ModelBackend& backend,PrefixCache& prefixes,
        const hybrid::Cache& cache,const hybrid::Representation& rep,
        const std::string& owner,bool temporary,int source) {
    hybrid::Capture capture;
    if(temporary || owner.empty())return capture;
    bool primary=cache.admitted(owner)||cache.owners.size()<3;
    int target=std::min<int>(80000,rep.tokens.size());
    if(target==(int)rep.tokens.size()) {
        auto boundaries=find_all_boundaries(rep.tokens,prefixes.chat_markers());
        if(!boundaries.empty())target=std::min(target,boundaries.back());
    }
    int needed=backend.snapshot_capture_position(target,rep.tokens.size(),0);
    bool refresh=primary;
    for(const auto& e:cache.entries)if(e.primary && e.owner==owner &&
        e.representation->fingerprint==rep.fingerprint && e.position>=needed &&
        e.position<=(int)rep.tokens.size() && e.position<=(int)e.representation->tokens.size() &&
        std::equal(rep.tokens.begin(),rep.tokens.begin()+e.position,e.representation->tokens.begin()))refresh=false;
    capture.requested=refresh?target:(int)rep.tokens.size();
    capture.actual=backend.snapshot_capture_position(capture.requested,rep.tokens.size(),
        source>=0?backend.snapshot_cur_pos(source):0);
    if(capture.actual<512)return {};
    capture.slot=cache.free_slot();capture.primary=refresh;
    capture.bytes=backend.snapshot_estimate_bytes(capture.actual);
    if(capture.slot<0 || !capture.bytes)return {};
    return capture;
}
std::string hybrid_fingerprint(const std::string&s) {
    std::vector<int32_t> bytes;bytes.reserve(s.size());for(unsigned char c:s)bytes.push_back(c);
    auto h=hash_prefix(bytes.data(),(int)bytes.size());
    static const char hex[]="0123456789abcdef";std::string out;
    for(auto c:h){out+=hex[c>>4];out+=hex[c&15];}return out;
}
}
void HttpServer::publish_hybrid(const json &trace) {
    json states=json::array();
    for(auto&e:hybrid_cache_.entries)states.push_back({{"slot",e.slot},{"owner",e.owner},
        {"primary",e.primary},{"position",e.position},{"full",e.full},{"gpu",e.gpu},
        {"bytes",e.bytes},{"regions",e.representation->regions.size()},
        {"representation",e.representation->fingerprint},{"hits",e.hits}});
    json costs=json::object();for(auto&[key,e]:hybrid_cache_.costs) {
        costs[key]={{"seconds",e.value},{"samples",e.observations}};
        auto a=hybrid_cache_.anchors.find(key);
        if(a!=hybrid_cache_.anchors.end() && a->second.input.calibrated() && a->second.prefill.calibrated() && a->second.scoring.calibrated())
            costs[key]["anchor"]={{"input_tokens",a->second.input.value},{"prefill_seconds",a->second.prefill.value},{"scoring_tokens",a->second.scoring.value},{"samples",a->second.input.observations}};
    }
    std::lock_guard<std::mutex> lock(hybrid_stats_mutex_);
    hybrid_stats_={{"enabled",hybrid_enabled_},{"shadow",hybrid_shadow_},{"gpu_budget",hybrid_cache_.gpu_checkpoint_budget},{"ram_budget",hybrid_cache_.ram_checkpoint_budget},{"owners",hybrid_cache_.owners},
        {"entries",states},{"metadata_bytes",hybrid_cache_.metadata_bytes()},{"last_request",trace},{"calibration",costs}};
}
HttpServer::PreparedPrompt HttpServer::prepare_hybrid_prompt(const ParsedRequest &req, const std::function<bool()>& cancelled) {
    using namespace hybrid;
    PreparedPrompt out;out.hybrid=true;out.tokens=req.prompt_tokens;
    out.representation=std::make_shared<Representation>();
    auto fail=[&](int code,std::string text){out.error_status=code;out.error=std::move(text);return out;};
    try {
        json meta=json::object();
        if(req.raw_body.contains("extra_body")&&req.raw_body["extra_body"].is_object())
            meta=req.raw_body["extra_body"].value("lucebox_cache",json::object());
        if(!meta.is_object())return fail(400,"lucebox_cache must be an object");
        auto mode=meta.value("mode","auto"),purpose=meta.value("purpose","chat");
        out.hybrid_deadline_ms=meta.value("deadline_unix_ms",int64_t(0));
        if(out.hybrid_deadline_ms<0)return fail(400,"invalid cache request deadline");
        auto interrupted=[&]() {
            const auto now=std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            return (cancelled && cancelled()) || (out.hybrid_deadline_ms && now>=out.hybrid_deadline_ms);
        };
        if(interrupted())return fail(408,"Request cancelled or deadline expired");
        if(mode!="auto"&&mode!="exact")return fail(400,"cache mode must be auto or exact");
        if(purpose!="chat"&&purpose!="compaction")return fail(400,"cache purpose must be chat or compaction");
        out.hybrid_owner=meta.value("owner",req.session_id);
        if(out.hybrid_owner.size()>256||std::any_of(out.hybrid_owner.begin(),out.hybrid_owner.end(),[](unsigned char c){return c<32||c==127;}))
            return fail(400,"invalid cache owner");
        out.hybrid_temporary=purpose=="compaction";
        const bool exact=mode=="exact" || req.format!=ApiFormat::OPENAI_CHAT;
        const auto raw_key=hybrid_fingerprint(req.rendered_prompt);
        const auto salt=compute_disk_cache_salt(config_);
        const auto identity=hybrid_fingerprint(std::string((const char*)salt.data(),salt.size())+req.tools.dump()+std::to_string(req.thinking_enabled)+req.reasoning_effort);
        // Tool definitions belong to checkpoint identity, but their text is already
        // accounted for by rendered token lengths when predicting execution cost.
        const auto cost_identity=hybrid_fingerprint(std::string((const char*)salt.data(),salt.size())+"[]"+std::to_string(req.thinking_enabled)+req.reasoning_effort);
        auto render_messages=[&](const json& messages,std::vector<int32_t>&tokens) {
            auto normalized=normalize_chat_messages(messages,req.format,tool_memory_);
            std::string text,error;
            if(!render_messages_to_text(normalized,req,true,text,error))return false;
            tokens=tokenizer_.encode(text);return true;
        };
        auto render=[&](const std::vector<Region>& regions,std::vector<int32_t>& tokens) {
            if(regions.empty()){tokens=req.prompt_tokens;return true;}
            return render_messages(apply_regions(req.messages,regions),tokens);
        };
        // Compute current protection before accepting any frozen representation.
        const auto permitted_regions = !exact && req.messages.is_array()
            ? eligible_regions(req.messages,meta) : std::vector<Region>{};
        std::vector<Candidate> candidates;
        std::vector<std::shared_ptr<Representation>> representations;
        json rejected=json::array();
        std::vector<std::string> cost_keys;
        std::vector<Capture> captures;
        std::vector<Cache::Forecast> forecasts;
        std::vector<int> scoring_counts,scoring_regions;
        const char* projection_setting=std::getenv("DFLASH_HYBRID_PROJECTIONS");
        const bool projections_enabled=projection_setting && std::string(projection_setting)=="1";
        // Conservative transient allowance on top of already allocated live KV:
        // fixed-window scorer plus decode-drafter loading, never active CPU KV.
        const uint64_t work=4*GiB;
        auto add=[&](Candidate c,std::shared_ptr<Representation> rep,uint64_t extra_work=0) {
            int uncached=0,regions_to_score=0;
            if(c.path==Path::Compress)for(const auto& r:rep->regions)
                if(r.replacement.empty()){uncached+=(int)drafter_tokenizer_->encode(r.source).size();++regions_to_score;}
            auto key=Cache::cost_key(c.path,c.path==Path::Compress?req.prompt_tokens.size():rep->tokens.size(),
                c.prefix,req.max_output,c.gpu,c.compressed,cost_identity,uncached);
            auto capture=plan_hybrid_capture(backend_,prefix_cache_,hybrid_cache_,*rep,
                out.hybrid_owner,out.hybrid_temporary,c.id);
            // Compression is admitted first for bounded scoring. Its exact context size is
            // unknown until rendering; final inference admission happens below.
            c.valid=c.valid && req.max_output>0 && req.max_output<=config_.max_ctx &&
                (c.path==Path::Compress || rep->tokens.size()+(size_t)req.max_output<=(size_t)config_.max_ctx);
            uint64_t promotion=0;
            if(c.id>=0&&!c.gpu){auto*e=hybrid_cache_.find(c.id);if(e)promotion=backend_.snapshot_estimate_bytes(e->position);}
            auto admission=plan_with_optional_capture(
                hybrid_cache_.memory(backend_,c.id,work+extra_work,promotion,rep->bytes()),capture.bytes);
            auto plan=std::move(admission.resource);
            if(!admission.capture)capture={};
            key+=":capture"+std::to_string((capture.bytes+256*1024*1024-1)/(256*1024*1024));
            const bool cold_shape=c.prefix==0 && (c.path==Path::Dense ||
                (c.path==Path::Compress && regions_to_score==1 && rep->regions.size()==1));
            auto forecast=hybrid_cache_.forecast(key,c.path,req.prompt_tokens.size(),uncached,regions_to_score,projections_enabled && cold_shape);
            c.seconds=forecast.seconds;c.projected=forecast.kind=="conservative_projection";forecasts.push_back(std::move(forecast));
            scoring_counts.push_back(uncached);scoring_regions.push_back(regions_to_score);
            cost_keys.push_back(std::move(key));
            captures.push_back(capture);
            c.feasible=plan.feasible;
            candidates.push_back(c);representations.push_back(std::move(rep));
        };
        if(!out.hybrid_owner.empty()&&!out.hybrid_temporary)for(auto&e:hybrid_cache_.entries) {
            auto reject=[&](const char* reason){rejected.push_back({{"slot",e.slot},{"reason",reason}});};
            if(e.owner!=out.hybrid_owner){reject("different_owner");continue;}
            if(e.representation->fingerprint!=identity){reject("model_template_or_settings_changed");continue;}
            if(exact&&!e.representation->regions.empty()){reject("exact_mode_excludes_compressed_history");continue;}
            bool valid=true;for(auto&r:e.representation->regions)if(!permitted(r,req.messages,permitted_regions))valid=false;
            if(!valid){reject("source_or_protection_changed");continue;}
            auto rep=std::make_shared<Representation>(*e.representation);
            if(!render(rep->regions,rep->tokens)){reject("render_failed");continue;}
            int actual=backend_.snapshot_cur_pos(e.slot);
            if(!backend_.snapshot_used(e.slot)||actual!=e.position||actual<=0||actual>(int)rep->tokens.size()||
                actual>(int)e.representation->tokens.size()||!std::equal(rep->tokens.begin(),rep->tokens.begin()+actual,e.representation->tokens.begin())){reject("snapshot_position_or_effective_prefix_changed");continue;}
            rep->raw_key=raw_key;
            bool full=e.full&&actual==(int)rep->tokens.size()&&e.representation->raw_key==raw_key;
            Path path=full?Path::Full:(rep->regions.empty()?Path::Prefix:Path::Frozen);
            add({path,e.slot,actual,!rep->regions.empty(),e.gpu,true,true,
                std::nullopt},rep);
        }
        auto dense=std::make_shared<Representation>();dense->tokens=req.prompt_tokens;dense->fingerprint=identity;dense->raw_key=raw_key;
        add({Path::Dense,-1,0,false,true,true,true,std::nullopt},dense);
        std::vector<Region> eligible;
        int eligible_tokens=0;
        if(!exact&&drafter_tokenizer_&&req.messages.is_array()) {
            eligible=permitted_regions;
            eligible.erase(std::remove_if(eligible.begin(),eligible.end(),[&](auto&r){
                auto count=drafter_tokenizer_->encode(r.source).size();if(count<512)return true;
                eligible_tokens+=(int)count;return false;
            }),eligible.end());
        }
        bool can_compress=!eligible.empty()&&req.prompt_tokens.size()>=32000&&eligible_tokens>=8000;
        if(can_compress) {
            auto rep=std::make_shared<Representation>(*dense);
            rep->regions=rebase_regions(eligible,req.messages,hybrid_cache_.entries,out.hybrid_owner,identity);
            // Own reusable results before any winning memory plan can evict their old entry.
            pin_replacements(rep->regions,hybrid_cache_.entries,out.hybrid_owner,identity);
            add({Path::Compress,-1,0,true,true,true,true,std::nullopt},rep);
            // Reserve a source only before the first newly changed message. A second
            // exact-token check after scoring is mandatory before restoration.
            auto provisional=rep->regions;int changed_message=(int)req.messages.size();
            for(auto& region:provisional)if(region.replacement.empty()) {
                changed_message=std::min(changed_message,region.message);region.replacement=region.source;
            }
            std::vector<int32_t> provisional_tokens,guard_tokens;
            if(render(provisional,provisional_tokens)) {
                int safe=(int)provisional_tokens.size();
                if(changed_message<(int)req.messages.size()) {
                    auto guarded=apply_regions(req.messages,provisional);guarded[changed_message]["content"]="";
                    if(!render_messages(guarded,guard_tokens))safe=0;
                    else {
                        int common=0;while(common<(int)provisional_tokens.size() && common<(int)guard_tokens.size() && provisional_tokens[common]==guard_tokens[common])++common;
                        safe=0;for(int boundary:find_all_boundaries(provisional_tokens,prefix_cache_.chat_markers()))if(boundary<=common)safe=std::max(safe,boundary);
                    }
                }
                if(!out.hybrid_owner.empty() && !out.hybrid_temporary)for(const auto& entry:hybrid_cache_.entries) {
                    if(entry.owner!=out.hybrid_owner || !entry.representation || entry.representation->fingerprint!=identity ||
                       entry.position>safe || !backend_.snapshot_used(entry.slot) || backend_.snapshot_cur_pos(entry.slot)!=entry.position ||
                       !prefix_matches(provisional_tokens,entry.representation->tokens,entry.position))continue;
                    auto rebased=std::make_shared<Representation>(*rep);rebased->tokens=provisional_tokens;
                    add({Path::Compress,entry.slot,entry.position,true,entry.gpu,true,true,std::nullopt},rebased);
                }
            }
        }
        auto decision=select(candidates,exact);
        // Private calibration only; still enforces region/context/memory validity.
        const char* experiment=std::getenv("DFLASH_HYBRID_EXPERIMENT");
        if(experiment&&(std::string(experiment)=="compress" || std::string(experiment)=="rebase")&&!exact &&
           (std::string(experiment)=="rebase" || std::none_of(candidates.begin(),candidates.end(),[](const auto& c){return c.reuse()&&c.valid&&c.feasible;})))
            for(int i=0;i<(int)candidates.size();++i)if(candidates[i].path==Path::Compress&&candidates[i].valid&&candidates[i].feasible)
                if(decision.reason!="private_compression_calibration" || candidates[i].prefix>=candidates[decision.index].prefix)decision={i,"private_compression_calibration"};
        json inspected=json::array();
        for(size_t i=0;i<candidates.size();++i){auto& c=candidates[i];inspected.push_back({{"path",name(c.path)},{"slot",c.id},{"prefix",c.prefix},
            {"valid",c.valid},{"feasible",c.feasible},{"seconds",c.seconds?json(*c.seconds):json(nullptr)},
            {"estimate_kind",forecasts[i].kind},{"estimate_source",forecasts[i].source},{"projection_ratio",forecasts[i].ratio}});}
        out.hybrid_trace={{"owner",out.hybrid_owner},{"mode",mode},{"purpose",purpose},{"reason",decision.reason},
            {"candidates",inspected},{"rejected_checkpoints",rejected},{"generation",meta.value("generation",json())},{"eligible_tokens",eligible_tokens},{"scored_regions",0},{"input_tokens",req.prompt_tokens.size()}};
        if(decision.index<0) {
            if(req.prompt_tokens.size()+(size_t)req.max_output>(size_t)config_.max_ctx)
                return fail(400,"Prompt plus requested output exceeds context; no feasible legal compression path");
            return fail(503,"No cache path fits memory budgets");
        }
        if(hybrid_shadow_) {
            out.hybrid_trace["proposed_path"]=name(candidates[decision.index].path);
            decision=shadow_baseline(candidates);
            if(decision.index<0)return fail(503,"Observation-mode exact baseline exceeds capacity");
        }
        if(interrupted())return fail(408,"Request cancelled or deadline expired");
        bool capture_changed=false;
        auto resource_for=[&](int index) {
            auto& c=candidates[index];const auto* entry=hybrid_cache_.find(c.id);
            auto admission=plan_with_optional_capture(hybrid_cache_.memory(backend_,c.id,work,
                c.id>=0&&!c.gpu&&entry?backend_.snapshot_estimate_bytes(entry->position):0,representations[index]->bytes()),captures[index].bytes);
            if(captures[index].bytes && !admission.capture) {
                captures[index]={};capture_changed=true;
                auto& key=cost_keys[index];key=key.substr(0,key.rfind(":capture"))+":capture0";
                c.seconds=hybrid_cache_.estimate(key);c.projected=false;
            }
            return admission.resource;
        };
        auto plan=resource_for(decision.index);
        if(!plan.feasible || capture_changed) {
            // One fresh selection if free memory changed. Evaluating alternatives
            // still does not mutate snapshots, mappings or scratch.
            for(int i=0;i<(int)candidates.size();++i)candidates[i].feasible=resource_for(i).feasible;
            decision=hybrid_shadow_?shadow_baseline(candidates):select(candidates,exact);
            if(decision.index<0)return fail(503,"Capacity changed before reservation");
            plan=resource_for(decision.index);
        }
        auto chosen=candidates[decision.index];
        uint64_t checkpoint_gpu=0,checkpoint_ram=0;
        for(const auto& entry:hybrid_cache_.entries)
            (entry.gpu?checkpoint_gpu:checkpoint_ram)+=entry.bytes;
        const json admission_trace={{"path",name(chosen.path)},
            {"gpu_free",backend_.cache_gpu_free_bytes()},
            {"reclaimable_graph",backend_.cache_reclaimable_scratch_bytes()},
            {"checkpoint_gpu",checkpoint_gpu},{"checkpoint_ram",checkpoint_ram},
            {"working_allowance",work},{"capture_bytes",captures[decision.index].bytes},
            {"input_tokens",req.prompt_tokens.size()},{"output_capacity",req.max_output},
            {"evict",plan.evict},{"spill",plan.spill},{"feasible",plan.feasible}};
        std::fprintf(stderr,"[hybrid-admit] %s\n",admission_trace.dump().c_str());
        if(!plan.feasible||!hybrid_cache_.execute(backend_,plan))return fail(503,"Cache resource reservation changed or transfer failed");
        out.representation=representations[decision.index];out.hybrid_path=chosen.path;
        out.hybrid_capture=captures[decision.index];
        out.hybrid_trace["cost_key"]=cost_keys[decision.index];
        out.hybrid_trace["uncached_scoring_tokens"]=scoring_counts[decision.index];
        out.hybrid_trace["projection_anchor"]=chosen.prefix==0 && (chosen.path==Path::Dense ||
            (chosen.path==Path::Compress && scoring_regions[decision.index]==1 && out.representation->regions.size()==1));
        out.hybrid_trace["selected_path"]=name(chosen.path);out.hybrid_trace["spills"]=plan.spill;out.hybrid_trace["evictions"]=plan.evict;
        if(chosen.id>=0) {
            if(!chosen.gpu) {
                auto begin=std::chrono::steady_clock::now();
                if(!backend_.snapshot_move(chosen.id,true)){out.hybrid_recoverable=true;return fail(503,"Checkpoint promotion failed; source retained");}
                hybrid_cache_.find(chosen.id)->gpu=true;
                hybrid_cache_.find(chosen.id)->bytes=backend_.snapshot_bytes(chosen.id);
                out.hybrid_trace["promotion_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            }
            out.hybrid_source=chosen.id;
        }
        if(chosen.path==Path::Compress) {
            std::string query;
            for(int i=(int)req.messages.size()-1;i>=0;--i)if(req.messages[i].value("role","")=="user") {
                for(int part:text_parts(req.messages[i])) {
                    auto text=*text_at(req.messages[i],part);
                    for(auto it=eligible.rbegin();it!=eligible.rend();++it)if(it->message==i && it->part==part)text.erase(it->start,it->end-it->start);
                    if(!query.empty())query+="\n";query+=text;
                }
                break;
            }
            auto q=drafter_tokenizer_->encode(query);if(q.size()>64)q.erase(q.begin(),q.end()-64);
            if(q.empty())return fail(400,"Compression requires an unmarked user question");
            for(auto&r:out.representation->regions)if(r.replacement.empty()) {
                if(interrupted())return fail(408,"Request cancelled or deadline expired");
                ModelBackend::CompressRequest cr;cr.should_cancel=interrupted;cr.input_ids=drafter_tokenizer_->encode(r.source);
                if(cr.input_ids.size()<512){r.replacement=r.source;continue;}
                cr.input_ids.insert(cr.input_ids.end(),q.begin(),q.end());cr.keep_ratio=.20f;
                cr.score_query_end=cr.input_ids.size();cr.score_query_tokens=std::min<int>(8,q.size());
                cr.drafter_path=config_.pflash_drafter_path;cr.drafter_gpu=config_.pflash_drafter_gpu;cr.skip_park=true;
                cr.residency_action=DraftResidencyAction::ReleaseAfterUse;
                const auto score_started=std::chrono::steady_clock::now();
                auto result=backend_.compress(cr);
                out.hybrid_trace["scoring_seconds"]=out.hybrid_trace.value("scoring_seconds",0.)+
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-score_started).count();
                if(interrupted())return fail(408,"Request cancelled or deadline expired");
                if(!result.ok||result.compressed_ids.size()<q.size()||
                    !std::equal(q.rbegin(),q.rend(),result.compressed_ids.rbegin())) {
                    out.hybrid_recoverable=true;return fail(500,"Scoring failed or query tail changed");
                }
                result.compressed_ids.resize(result.compressed_ids.size()-q.size());
                r.replacement=drafter_tokenizer_->decode(result.compressed_ids);
                out.hybrid_trace["scored_regions"]=out.hybrid_trace["scored_regions"].get<int>()+1;
            }
            if(!render(out.representation->regions,out.representation->tokens))return fail(400,"Compressed rendering failed");
            if(out.hybrid_source>=0) {
                const auto* source=hybrid_cache_.find(out.hybrid_source);
                if(!source || !backend_.snapshot_used(source->slot) || backend_.snapshot_cur_pos(source->slot)!=source->position ||
                   !prefix_matches(out.representation->tokens,source->representation->tokens,source->position)) {
                    // The rendering boundary was only a conservative reservation hint.
                    // Never restore if the final effective token sequence disagrees.
                    out.hybrid_trace["rebase_prefix_rejected"]=true;out.hybrid_source=-1;
                    out.hybrid_trace["estimated_cost_key"]=out.hybrid_trace["cost_key"];
                    out.hybrid_trace.erase("cost_key"); // Do not train the reserved-prefix bucket.
                } else out.hybrid_trace["rebase_prefix_tokens"]=source->position;
            }
        }
        if(interrupted())return fail(408,"Request cancelled or deadline expired");
        out.representation->generation=meta.value("generation",json()).dump();
        out.tokens=out.representation->tokens;out.compressed=!out.representation->regions.empty();
        out.hybrid_trace["effective_tokens"]=out.tokens.size();out.hybrid_trace["restore_slot"]=out.hybrid_source;
        if(out.tokens.size()+(size_t)req.max_output>(size_t)config_.max_ctx) {
            out.hybrid_recoverable=chosen.path==Path::Compress;
            return fail(400,"Effective prompt plus output exceeds context");
        }
        // Scoring may change the required capture size; final admission uses the real
        // effective history, with replacement storage reserved separately from its source.
        out.hybrid_capture=plan_hybrid_capture(backend_,prefix_cache_,hybrid_cache_,*out.representation,
            out.hybrid_owner,out.hybrid_temporary,out.hybrid_source);
        auto final_admission=plan_with_optional_capture(
            hybrid_cache_.memory(backend_,out.hybrid_source,work,0,out.representation->bytes()),out.hybrid_capture.bytes);
        auto final_plan=std::move(final_admission.resource);
        if(out.hybrid_capture.bytes && !final_admission.capture) {
            out.hybrid_capture={};out.hybrid_trace["maintenance_deferred"]="capture_capacity";
        }
        if(!final_plan.feasible || !hybrid_cache_.execute(backend_,final_plan))
            return fail(503,"Rendered representation exceeds working memory capacity");
        out.hybrid_trace["final_spills"]=final_plan.spill;
        out.hybrid_trace["final_evictions"]=final_plan.evict;
        if(!out.hybrid_temporary) {
            hybrid_cache_.admit(out.hybrid_owner);
            // A newly admitted owner may already have an immutable optional snapshot
            // satisfying the primary cap. Protect it without duplicating its storage.
            bool has_primary=false;
            for(const auto& e:hybrid_cache_.entries)
                if(e.owner==out.hybrid_owner && e.primary)has_primary=true;
            if(!has_primary && hybrid_cache_.admitted(out.hybrid_owner))
                if(auto* e=hybrid_cache_.find(out.hybrid_source))
                    if(e->owner==out.hybrid_owner && e->position<=80000)e->primary=true;
        }
        publish_hybrid(out.hybrid_trace);
        return out;
    }catch(const std::exception&e){return fail(400,std::string("Invalid hybrid cache request: ")+e.what());}
}
