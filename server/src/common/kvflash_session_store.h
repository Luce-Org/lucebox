// KVFlash 2.0 sessions: who a request is and who owns each slot. A session is a parked conversation; the N engine
// slots hold them, and a request resumes the one it owns.
//
//   identity    request headers (lower-cased map) + body session_id -> {client, sid}; a request
//               that sends nothing is anonymous (ownerless: resumable by content alone)
//   owners      the parked sessions and their owners -> the one a request resumes (explicit identity first,
//               ownerless content only past the shared preamble)
//   book        the owner of each slot, in memory; the gate that refuses a bad or running identity
//
// Pure: strings and vectors only, no engine; every decision is testable without a model.

#pragma once

#include "fnv.h"
#include "kvflash_slot_turn.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace luce::common {

// ---- identity ---------------------------------------------------------------------------------

using KvFlashHeaders = std::map<std::string, std::string>;   // names lower-cased, values trimmed

constexpr size_t kKvFlashSidMax = 1024;       // bytes of a session id

namespace detail {
inline std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}
}  // namespace detail

// The header lines of a request (after the request line). A name is stored lower-cased; a
// duplicate keeps its first value; a line without a colon is ignored.
inline KvFlashHeaders kvflash_parse_headers(std::string_view block) {
    KvFlashHeaders out;
    while (!block.empty()) {
        const size_t nl = block.find('\n');
        std::string_view line = block.substr(0, nl);
        block = nl == std::string_view::npos ? std::string_view{} : block.substr(nl + 1);
        const size_t colon = line.find(':');
        if (colon == std::string_view::npos) continue;
        std::string name(detail::trim_ws(line.substr(0, colon)));
        if (name.empty()) continue;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        out.emplace(std::move(name), std::string(detail::trim_ws(line.substr(colon + 1))));
    }
    return out;
}

struct KvFlashSessionId {
    std::string sid, client;   // client: opencode | claude | header | body; both empty: no identity
    bool empty() const { return sid.empty(); }
};

// First hit wins: x-opencode-session-id, x-session-affinity (opencode), X-Claude-Code-Session-Id,
// X-Session-Id, then the body's session_id. Parent ids are a relation, never an identity.
inline KvFlashSessionId kvflash_resolve_session(const KvFlashHeaders & headers, const std::string & body_sid) {
    auto header = [&](const char * name) -> const std::string * {
        const auto it = headers.find(name);
        return it != headers.end() && !it->second.empty() ? &it->second : nullptr;
    };
    if (const std::string * v = header("x-opencode-session-id")) return {*v, "opencode"};
    if (const std::string * v = header("x-session-affinity")) return {*v, "opencode"};
    if (const std::string * v = header("x-claude-code-session-id")) return {*v, "claude"};
    if (const std::string * v = header("x-session-id")) return {*v, "header"};
    if (!body_sid.empty()) return {body_sid, "body"};
    return {};
}

// An identity within the bound on a session id (the HTTP boundary answers 400 otherwise).
inline bool kvflash_identity_valid(const KvFlashSessionId & id) { return id.sid.size() <= kKvFlashSidMax; }

// A client that named its session, as against one that sent nothing (anon): an anonymous
// session is ownerless, resumable by content alone.
inline bool kvflash_client_explicit(const std::string & client) {
    return client == "opencode" || client == "claude" || client == "header" || client == "body";
}

// A parked session and who owns it, for the resume pick.
struct KvFlashOwnedSession {
    KvFlashParkedSession parked;
    std::string client, sid;
};

// The parked session a request resumes. An explicit request takes only its own (client, sid), from
// the first shared token; failing that, an ownerless session its prompt extends beyond the shared
// preamble (lcp > e_role). An anonymous request takes ownerless sessions only, under the same bar.
// Content never overrides a stated identity.
inline KvFlashSessionPick kvflash_pick_parked_owner(const std::vector<KvFlashOwnedSession> & parked, const std::string & req_client,
                                                    const std::string & req_sid, const std::vector<int32_t> & prompt, int e_role) {
    const bool req_explicit = kvflash_client_explicit(req_client);
    std::vector<KvFlashParkedSession> own, free;
    for (const auto & o : parked) {
        if (kvflash_client_explicit(o.client)) { if (req_explicit && o.client == req_client && o.sid == req_sid) own.push_back(o.parked); }
        else free.push_back(o.parked);
    }
    KvFlashSessionPick p = kvflash_pick_session(own, prompt, 1);
    return p.exact() ? p : kvflash_pick_session(free, prompt, std::max(e_role, 0) + 1);
}

// A resume that rewound its parked session to the resume point L (slab restored, pager truncated, history cut to L) and was then
// refused busy leaves the session valid at L but shorter than the prompt that picked it: the retry's pick/plan would see lcp <= L
// and reject the resume (L <= e_role) or cold-admit over the waiting session. The wait remembers that prompt's prefix, so the
// retry picks the same session at the same lcp.
struct KvFlashResumeWait {
    int L = 0, lcp = 0;               // the rewind point and the original shared prefix (L == 0: no wait)
    uint64_t prefix_hash = 0;         // of the original prompt's first lcp tokens
    std::string client, sid;          // who waits
};
inline uint64_t kvflash_prefix_hash(const std::vector<int32_t> & prompt, int n) {
    return fnv1a_bytes(kFnvBasis, (const uint8_t *)prompt.data(), (size_t)std::max(n, 0) * 4);
}
inline KvFlashResumeWait kvflash_resume_wait_make(int L, int lcp, const std::vector<int32_t> & prompt, const std::string & client, const std::string & sid) {
    return {L, lcp, kvflash_prefix_hash(prompt, lcp), client, sid};
}
// The pick for a request when `slot` holds a resume wait (history_size tokens parked): the same session at the original lcp when this
// prompt starts with the waiting prompt's prefix and comes from the same identity, else `pick` unchanged.
inline KvFlashSessionPick kvflash_pick_with_wait(const KvFlashSessionPick & pick, int slot, int history_size, const KvFlashResumeWait & wait,
                                                 const std::vector<int32_t> & prompt, const std::string & client, const std::string & sid) {
    if (wait.L <= 0 || history_size != wait.L || (int)prompt.size() < wait.lcp || wait.client != client || wait.sid != sid ||
        kvflash_prefix_hash(prompt, wait.lcp) != wait.prefix_hash) return pick;
    return {slot, std::max(pick.slot == slot ? pick.lcp : 0, wait.lcp)};
}

// ---- the slot table: who owns each slot ----------------------------------------------------------------

// The engine's per-slot owner bookkeeping, with no GPU in it: every transition the engine makes on ownership (a cold admit, a
// resume that promotes an ownerless session, a drop) is a method here, so the tests drive the same code the engine runs.
class KvFlashSessionBook {
public:
    struct Owner { std::string sid, client; };

    void init(int slots) { slot_.assign((size_t)slots, Owner{}); }
    Owner & operator[](int slot) { return slot_[(size_t)slot]; }
    const Owner & operator[](int slot) const { return slot_[(size_t)slot]; }
    int slots() const { return (int)slot_.size(); }

    // A cold admit: the slot starts the session over.
    void begin_session(int slot, const std::string & client, const std::string & sid) { slot_[(size_t)slot] = Owner{sid, client}; }
    // The slot holds no session any more.
    void reset(int slot) { slot_[(size_t)slot] = Owner{}; }

    // An explicit request resumes an ownerless session (an anonymous one): it becomes that client's.
    // False when nothing changed (the owner is explicit already, or the request is not).
    bool promote_owner(int slot, const std::string & client, const std::string & sid) {
        Owner & o = slot_[(size_t)slot];
        if (kvflash_client_explicit(o.client) || !kvflash_client_explicit(client)) return false;
        o.client = client; o.sid = sid;
        return true;
    }

    // Is (client, sid) running a request in a slot now? One live owner per explicit identity: a second
    // request for it waits.
    bool identity_busy(const std::string & client, const std::string & sid, const std::vector<bool> & active) const {
        if (!kvflash_client_explicit(client)) return false;
        for (size_t i = 0; i < slot_.size() && i < active.size(); ++i)
            if (active[i] && slot_[i].client == client && slot_[i].sid == sid) return true;
        return false;
    }

private:
    std::vector<Owner> slot_;
};

// ---- the engine's admission gate ----------------------------------------------------------------------

enum class KvFlashAdmitVerdict { proceed, invalid_identity, identity_busy };

// The checks that stand between a request and a slot, in the order the engine makes them: an identity that cannot be stored is
// refused; a second request for an identity that is running waits.
inline KvFlashAdmitVerdict kvflash_admit_gate(const KvFlashSessionBook & book, const KvFlashSessionId & who, const std::vector<bool> & active) {
    if (!kvflash_identity_valid(who)) return KvFlashAdmitVerdict::invalid_identity;
    if (book.identity_busy(who.client, who.sid, active)) return KvFlashAdmitVerdict::identity_busy;
    return KvFlashAdmitVerdict::proceed;
}

}  // namespace luce::common
