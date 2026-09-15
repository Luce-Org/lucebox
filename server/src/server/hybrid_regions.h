#pragma once
#include "hybrid_cache.h"
#include <stdexcept>
namespace dflash::common::hybrid {
inline Json envelope(Json message){message.erase("content");return message;}
inline Json part_envelope(const Json& message,int part) {
    if(part<0)return Json();
    auto block=message.at("content").at(part);block.erase("text");return block;
}
inline const std::string* text_at(const Json& message,int part) {
    if(!message.contains("content"))return nullptr;
    const auto& content=message["content"];
    if(part<0)return content.is_string()?&content.get_ref<const std::string&>():nullptr;
    if(!content.is_array() || part>=(int)content.size())return nullptr;
    const auto& block=content[part];
    if(!block.is_object() || block.value("type","")!="text" || !block.contains("text") || !block["text"].is_string())return nullptr;
    return &block["text"].get_ref<const std::string&>();
}
inline std::vector<int> text_parts(const Json& message) {
    if(text_at(message,-1))return {-1};
    std::vector<int> parts;
    if(message.contains("content") && message["content"].is_array())
        for(int p=0;p<(int)message["content"].size();++p)if(text_at(message,p))parts.push_back(p);
    return parts;
}
inline bool byte_boundary(const std::string&s,size_t n){return n<=s.size()&&(n==s.size()||((unsigned char)s[n]&0xc0)!=0x80);}
inline std::vector<Region> eligible_regions(const Json& messages,const Json& metadata) {
    std::vector<Region> out;
    if(!messages.is_array())return out;
    std::set<int> summaries,marked_messages;
    if(metadata.contains("summary_messages")) {
        if(!metadata["summary_messages"].is_array())throw std::invalid_argument("summary_messages must be an array");
        for(const auto& index:metadata["summary_messages"]) {
            if(!index.is_number_integer() || index.get<int>()<0 || index.get<int>()>=(int)messages.size())throw std::invalid_argument("invalid summary message index");
            summaries.insert(index.get<int>());
        }
    }
    auto protected_message=[&](int i) {
        auto role=messages[i].value("role","");
        return role=="system" || role=="developer" || summaries.count(i);
    };
    auto append=[&](int i,int part,size_t start,size_t end) {
        const auto* text=text_at(messages[i],part);
        out.push_back({i,start,end,text->substr(start,end-start),{},envelope(messages[i]),part,part_envelope(messages[i],part)});
    };
    if(metadata.contains("bulk_regions")) {
        if(!metadata["bulk_regions"].is_array())throw std::invalid_argument("bulk_regions must be an array");
        for(const auto& region:metadata["bulk_regions"]) {
            int i=region.at("message").get<int>(),part=region.value("part",-1);
            if(i<0 || i>=(int)messages.size() || part<-1 || protected_message(i) || messages[i].value("role","")!="user")
                throw std::invalid_argument("bulk region must be in ordinary user content");
            const auto* text=text_at(messages[i],part);
            if(!text)throw std::invalid_argument("bulk region requires a text content part");
            int64_t start=region.at("start").get<int64_t>(),end=region.at("end").get<int64_t>();
            if(start<0 || end<=start || !byte_boundary(*text,start) || !byte_boundary(*text,end))throw std::invalid_argument("invalid UTF-8 bulk boundaries");
            if(region.at("text").get<std::string>()!=text->substr(start,end-start))throw std::invalid_argument("bulk source does not match request");
            append(i,part,start,end);marked_messages.insert(i);
        }
    }
    auto order=[](const Region&a,const Region&b){return std::tie(a.message,a.part,a.start)<std::tie(b.message,b.part,b.start);};
    std::sort(out.begin(),out.end(),order);
    for(size_t i=1;i<out.size();++i)
        if(out[i].message==out[i-1].message && out[i].part==out[i-1].part && out[i].start<out[i-1].end)
            throw std::invalid_argument("overlapping bulk regions");
    for(int i:marked_messages) {
        std::string outside;
        for(int part:text_parts(messages[i])) {
            auto text=*text_at(messages[i],part);
            for(auto it=out.rbegin();it!=out.rend();++it)if(it->message==i && it->part==part)text.erase(it->start,it->end-it->start);
            outside+=text;
        }
        for(const std::string marker:{"<lucebox_bulk>","</lucebox_bulk>"})
            for(size_t pos;(pos=outside.find(marker))!=std::string::npos;)outside.erase(pos,marker.size());
        if(outside.find_first_not_of(" \t\r\n")==std::string::npos)throw std::invalid_argument("bulk regions must leave the user question outside them");
    }
    std::vector<int> exchanges;bool answered=false;
    std::set<std::string> pending;
    std::map<std::string,int> calls;
    for(int i=0;i<(int)messages.size();++i) {
        const auto& message=messages[i];auto role=message.value("role","");
        if(role=="user" && !summaries.count(i)) {
            if(exchanges.empty() || answered)exchanges.push_back(i);
            answered=false;
        } else if(role=="assistant") {
            if(message.contains("tool_calls") && !message["tool_calls"].empty()) {
                if(!message["tool_calls"].is_array())throw std::invalid_argument("invalid tool call boundary");
                for(const auto& call:message["tool_calls"]) {
                    const auto id=call.at("id").get<std::string>();calls[id]=i;pending.insert(id);
                }
            } else if(!exchanges.empty() && pending.empty() && message.contains("content") && !message["content"].is_null())answered=true;
        } else if(role=="tool")pending.erase(message.value("tool_call_id",""));
    }
    int cutoff=exchanges.size()>3?exchanges[exchanges.size()-3]:0;
    for(int i=0;i<cutoff;++i) {
        const auto& message=messages[i];auto role=message.value("role","");
        if((role!="user" && role!="assistant" && role!="tool") || protected_message(i) || message.contains("tool_calls"))continue;
        if(role=="tool") {
            auto call=calls.find(message.value("tool_call_id",""));
            if(call==calls.end() || call->second>=cutoff || pending.count(call->first))continue;
        }
        for(int part:text_parts(message)) {
            bool marked=false;for(const auto& region:out)if(region.message==i && region.part==part)marked=true;
            const auto* text=text_at(message,part);
            if(!marked && !text->empty())append(i,part,0,text->size());
        }
    }
    std::sort(out.begin(),out.end(),order);return out;
}
inline bool applicable(const Region& region,const Json& messages) {
    if(!messages.is_array() || region.message<0 || region.message>=(int)messages.size())return false;
    const auto& message=messages[region.message];const auto* text=text_at(message,region.part);
    return text && envelope(message)==region.envelope && part_envelope(message,region.part)==region.part_envelope &&
        region.start<=region.end && region.end<=text->size() && text->substr(region.start,region.end-region.start)==region.source;
}
inline bool permitted(const Region& frozen,const Json& messages,const std::vector<Region>& eligible) {
    if(!applicable(frozen,messages))return false;
    for(const auto& region:eligible)if(region.message==frozen.message && region.part==frozen.part && region.start<=frozen.start && region.end>=frozen.end)return true;
    return false;
}
inline void pin_replacements(std::vector<Region>& regions,const std::vector<Entry>& entries,const std::string& owner,const std::string& identity) {
    if(owner.empty())return;
    for(auto& region:regions)for(const auto& entry:entries)
        if(entry.owner==owner && entry.representation && entry.representation->fingerprint==identity)
            for(const auto& old:entry.representation->regions)
                if(old.message==region.message && old.part==region.part && old.start==region.start && old.end==region.end && old.source==region.source && old.envelope==region.envelope && old.part_envelope==region.part_envelope)
                    region.replacement=old.replacement;
}
inline std::vector<Region> rebase_regions(const std::vector<Region>& eligible,const Json& messages,const std::vector<Entry>& entries,const std::string& owner,const std::string& identity) {
    const Entry* base=nullptr;
    if(!owner.empty())for(const auto& entry:entries) {
        if(entry.owner!=owner || !entry.representation || entry.representation->fingerprint!=identity)continue;
        if(std::all_of(entry.representation->regions.begin(),entry.representation->regions.end(),[&](const auto& r){return permitted(r,messages,eligible);}) && (!base || entry.last_used>base->last_used))base=&entry;
    }
    if(!base)return eligible;
    std::vector<Region> result;
    for(const auto& region:eligible) {
        std::vector<Region> frozen;
        for(const auto& old:base->representation->regions)if(old.message==region.message && old.part==region.part && old.start>=region.start && old.end<=region.end)frozen.push_back(old);
        std::sort(frozen.begin(),frozen.end(),[](const auto&a,const auto&b){return a.start<b.start;});
        size_t cursor=region.start;
        auto gap=[&](size_t end) {if(end>cursor)result.push_back({region.message,cursor,end,region.source.substr(cursor-region.start,end-cursor),{},region.envelope,region.part,region.part_envelope});};
        for(const auto& old:frozen){gap(old.start);result.push_back(old);cursor=old.end;}
        gap(region.end);
    }
    return result;
}
inline Json apply_regions(Json messages,const std::vector<Region>& regions) {
    auto sorted=regions;
    std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){return std::tie(a.message,a.part,a.start)>std::tie(b.message,b.part,b.start);});
    for(const auto& region:sorted) {
        if(!applicable(region,messages))throw std::invalid_argument("frozen source changed");
        auto text=*text_at(messages[region.message],region.part);text.replace(region.start,region.end-region.start,region.replacement);
        if(region.part<0)messages[region.message]["content"]=std::move(text);
        else messages[region.message]["content"][region.part]["text"]=std::move(text);
    }
    return messages;
}
} // namespace
