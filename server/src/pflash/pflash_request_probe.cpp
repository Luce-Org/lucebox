#include "pflash_request_probe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace luce::pflash {

std::vector<PFlashTokenSpan> pflash_request_runs(
        const std::vector<float> & logits,
        PFlashTokenSpan message,
        float tau,
        int min_run) {
    std::vector<PFlashTokenSpan> runs;
    if (message.begin < 0 || message.end <= message.begin ||
        message.end > (int) logits.size()) {
        return runs;
    }
    int begin = -1;
    const auto close = [&](int end) {
        if (begin >= 0 && end - begin >= std::max(1, min_run)) runs.push_back({begin, end});
        begin = -1;
    };
    for (int t = message.begin; t < message.end; ++t) {
        // NaN compares false: it breaks a run like any logit below tau.
        if (logits[(size_t) t] >= tau) {
            if (begin < 0) begin = t;
        } else {
            close(t);
        }
    }
    close(message.end);
    return runs;
}

std::vector<int> pflash_request_run_tokens(const std::vector<PFlashTokenSpan> & runs) {
    std::vector<int> tokens;
    for (const auto & run : runs) {
        for (int t = run.begin; t < run.end; ++t) tokens.push_back(t);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

std::vector<int> pflash_request_cap_tail(const std::vector<int> & tokens, int n) {
    if (n <= 0) return {};
    if ((size_t) n >= tokens.size()) return tokens;
    return std::vector<int>(tokens.end() - n, tokens.end());
}

std::vector<PFlashTokenSpan> pflash_request_token_spans(const std::vector<int> & tokens) {
    std::vector<PFlashTokenSpan> spans;
    for (const int t : tokens) {
        if (!spans.empty() && spans.back().end == t) {
            ++spans.back().end;
        } else if (spans.empty() || t > spans.back().end) {
            spans.push_back({t, t + 1});
        }
    }
    return spans;
}

std::vector<int> pflash_request_turn_tokens(const PFlashRequestMessageRuns & message) {
    return pflash_request_cap_tail(pflash_request_run_tokens(message.runs), kRequestQueryCap);
}

namespace {

PFlashRequestMessageRuns request_message_runs(const std::vector<float> & logits,
                                              PFlashTokenSpan message, float tau) {
    PFlashRequestMessageRuns runs;
    runs.message = message;
    runs.runs = pflash_request_runs(logits, message, tau);
    for (const auto & run : runs.runs) runs.request_tokens += run.end - run.begin;
    runs.message_tokens = message.end - message.begin;
    double sum = 0.0;
    int finite = 0;
    runs.logit_min = std::numeric_limits<float>::infinity();
    runs.logit_max = -std::numeric_limits<float>::infinity();
    for (int t = message.begin; t < message.end; ++t) {
        const float value = logits[(size_t) t];
        if (value >= tau) ++runs.logits_at_or_above_tau;
        if (!std::isfinite(value)) continue;
        ++finite;
        sum += value;
        runs.logit_min = std::min(runs.logit_min, value);
        runs.logit_max = std::max(runs.logit_max, value);
    }
    if (finite == 0) runs.logit_min = runs.logit_max = 0.0f;
    runs.logit_mean = finite > 0 ? sum / finite : 0.0;
    return runs;
}

} // namespace

PFlashRequestQuerySet pflash_request_query_set_v2(
        const std::vector<float> & logits,
        const std::vector<PFlashRequestCandidate> & messages,
        float tau,
        int cap) {
    PFlashRequestQuerySet set;
    set.turns_read = true;
    for (const auto & m : messages) {
        if (m.span.begin < 0 || m.span.end < m.span.begin || m.span.end > (int) logits.size()) {
            set.fallback = "logits_missing";
            return set;
        }
    }
    std::vector<const PFlashRequestCandidate *> humans;
    for (const auto & m : messages) if (m.human) humans.push_back(&m);
    std::sort(humans.begin(), humans.end(), [](auto a, auto b) { return a->message_index < b->message_index; });
    // Candidates: the latest human, the previous one below kRequestPreviousBelow, the recent messages.
    std::vector<const PFlashRequestCandidate *> cands;
    int latest_index = -1;
    int previous_index = -1;
    if (!humans.empty()) {
        const auto * latest = humans.back();
        latest_index = latest->message_index;
        set.messages.push_back(request_message_runs(logits, latest->span, tau));
        cands.push_back(latest);
        if (set.messages[0].request_tokens < kRequestPreviousBelow && humans.size() >= 2) {
            const auto * previous = humans[humans.size() - 2];
            previous_index = previous->message_index;
            set.messages.push_back(request_message_runs(logits, previous->span, tau));
            cands.push_back(previous);
        }
    }
    for (const auto & m : messages) {
        if (!m.recent) continue;
        bool seen = false;
        for (const auto * c : cands) seen = seen || c->message_index == m.message_index;
        if (!seen) cands.push_back(&m);
    }
    if (cands.empty()) {
        set.fallback = "no_human_message";
        return set;
    }
    // previous_used as the trainer mirror reports it: joined by the < 8 rule, not as a recent message.
    bool previous_recent = false;
    for (const auto & m : messages) previous_recent = previous_recent || (m.recent && m.message_index == previous_index);
    set.previous_used = previous_index >= 0 && !previous_recent;
    std::sort(cands.begin(), cands.end(), [](auto a, auto b) { return a->message_index > b->message_index; });
    std::vector<int> taken;
    int budget = cap;
    std::vector<int> all;
    for (const auto * c : cands) {
        const PFlashRequestMessageRuns * runs = nullptr;
        PFlashRequestMessageRuns own;
        if (c->message_index == latest_index) runs = &set.messages[0];
        else if (c->message_index == previous_index) runs = &set.messages[1];
        else {
            own = request_message_runs(logits, c->span, tau);
            set.turns.push_back(own);
            set.turn_messages.push_back(c->message_index);
            runs = &set.turns.back();
        }
        std::vector<int> toks;
        for (const int t : pflash_request_run_tokens(runs->runs)) {
            all.push_back(t);
            if (!std::binary_search(taken.begin(), taken.end(), t)) toks.push_back(t);
        }
        std::sort(toks.begin(), toks.end());
        toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
        const auto take = pflash_request_cap_tail(toks, budget);
        budget -= (int) take.size();
        if (c->message_index == latest_index) set.latest_tokens = (int) take.size();
        else if (c->message_index == previous_index) set.previous_tokens = (int) take.size();
        else set.turn_tokens += (int) take.size();
        taken.insert(taken.end(), take.begin(), take.end());
        std::sort(taken.begin(), taken.end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    set.uncapped_tokens = (int) all.size();
    set.tokens = taken;
    set.capped = (int) set.tokens.size() < set.uncapped_tokens;
    if (set.tokens.empty()) set.fallback = "no_request_runs";
    return set;
}

bool pflash_request_role_from_name(const std::string & name, PFlashRequestRole & role) {
    if (name == "system") role = PFlashRequestRole::System;
    else if (name == "user") role = PFlashRequestRole::User;
    else if (name == "assistant") role = PFlashRequestRole::Assistant;
    else if (name == "tool") role = PFlashRequestRole::Tool;
    else return false;
    return true;
}

const char * pflash_request_role_name(PFlashRequestRole role) noexcept {
    switch (role) {
        case PFlashRequestRole::System: return "system";
        case PFlashRequestRole::User: return "user";
        case PFlashRequestRole::Assistant: return "assistant";
        case PFlashRequestRole::Tool: return "tool";
    }
    return "system";
}

std::vector<int> pflash_request_by_score(const std::vector<int> & tokens, const std::vector<float> & scores) {
    std::vector<int> out = tokens;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    // Run tokens are never NaN (a NaN breaks a run); a NaN still sorts last so the order stays strict.
    const auto key = [&scores](int t) {
        const float v = t >= 0 && t < (int) scores.size() ? scores[(size_t) t] : std::nanf("");
        return std::isnan(v) ? -std::numeric_limits<float>::infinity() : v;
    };
    std::sort(out.begin(), out.end(), [&](int a, int b) {
        const float va = key(a);
        const float vb = key(b);
        if (va != vb) return va > vb;
        return a > b;
    });
    return out;
}

std::vector<int> pflash_request_span_run_tokens(const std::vector<float> & scores, PFlashTokenSpan span, float tau) {
    return pflash_request_run_tokens(pflash_request_runs(scores, span, tau));
}

int pflash_request_latest_user(const std::vector<PFlashRequestMessage> & messages) {
    for (size_t i = messages.size(); i-- > 0;) {
        if (messages[i].role == PFlashRequestRole::User) return (int) i;
    }
    return -1;
}

std::vector<int> pflash_request_seed(const std::vector<float> & s1, float tau,
                                     const std::vector<PFlashRequestMessage> & messages, int latest, int cap,
                                     bool * capped) {
    if (capped) *capped = false;
    if (latest < 0 || latest >= (int) messages.size()) return {};
    std::vector<int> ranked = pflash_request_by_score(
        pflash_request_span_run_tokens(s1, messages[(size_t) latest].span, tau), s1);
    const size_t keep = (size_t) std::max(0, cap);
    if (ranked.size() > keep) {
        if (capped) *capped = true;
        ranked.resize(keep);
    }
    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

std::vector<int> pflash_request_capped_seed(const std::vector<float> & s1, float tau,
                                            const std::vector<PFlashRequestMessage> & messages, int cap, int reserve,
                                            int & reserve_out, std::vector<int> & post_out, bool * capped) {
    const int latest = pflash_request_latest_user(messages);
    post_out.clear();
    if (latest >= 0) {
        for (size_t i = (size_t) latest + 1; i < messages.size(); ++i) {
            if (messages[i].role == PFlashRequestRole::System) continue;
            for (const int t : pflash_request_span_run_tokens(s1, messages[i].span, tau)) post_out.push_back(t);
        }
    }
    std::sort(post_out.begin(), post_out.end());
    post_out.erase(std::unique(post_out.begin(), post_out.end()), post_out.end());
    reserve_out = std::min(std::max(0, reserve), (int) post_out.size());
    return pflash_request_seed(s1, tau, messages, latest, cap - reserve_out, capped);
}

PFlashRequestQuerySet pflash_request_query_set_v5(
        const std::vector<float> & logits,
        const std::vector<PFlashRequestMessage> & messages,
        float tau,
        int cap,
        int reserve) {
    PFlashRequestQuerySet set;
    set.turns_read = true;
    set.rule = kRequestCandidateRuleV5;
    for (const auto & m : messages) {
        if (m.span.end > m.span.begin && (m.span.begin < 0 || m.span.end > (int) logits.size())) {
            set.fallback = "logits_missing";
            return set;
        }
    }
    const int latest = pflash_request_latest_user(messages);
    set.latest_message = latest;
    std::vector<int> post;
    set.seed = pflash_request_capped_seed(logits, tau, messages, cap, reserve, set.reserve, post, &set.seed_capped);
    if (latest >= 0) set.messages.push_back(request_message_runs(logits, messages[(size_t) latest].span, tau));
    std::vector<int> taken = set.seed;
    const auto is_taken = [&taken](int t) { return std::binary_search(taken.begin(), taken.end(), t); };
    // The R best post-latest run tokens.
    std::vector<int> reserved;
    for (const int t : pflash_request_by_score(post, logits)) {
        if ((int) reserved.size() >= set.reserve) break;
        if (!is_taken(t)) reserved.push_back(t);
    }
    taken.insert(taken.end(), reserved.begin(), reserved.end());
    std::sort(taken.begin(), taken.end());
    std::vector<int> others;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        if (m.role == PFlashRequestRole::System || (int) i == latest) continue;
        if (m.span.end <= m.span.begin) continue;
        set.turns.push_back(request_message_runs(logits, m.span, tau));
        set.turn_messages.push_back((int) i);
        for (const int t : pflash_request_run_tokens(set.turns.back().runs)) others.push_back(t);
    }
    std::vector<int> rest;
    for (const int t : pflash_request_by_score(others, logits)) {
        if (!is_taken(t)) rest.push_back(t);
    }
    const size_t room = (size_t) std::max(0, cap - (int) taken.size());
    if (rest.size() > room) rest.resize(room);
    set.latest_tokens = (int) set.seed.size();
    set.turn_tokens = (int) (reserved.size() + rest.size());
    set.tokens = taken;
    set.tokens.insert(set.tokens.end(), rest.begin(), rest.end());
    std::sort(set.tokens.begin(), set.tokens.end());
    std::vector<int> all = others;
    if (latest >= 0) {
        for (const int t : pflash_request_run_tokens(set.messages[0].runs)) all.push_back(t);
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    set.uncapped_tokens = (int) all.size();
    set.capped = (int) set.tokens.size() < set.uncapped_tokens;
    if (set.tokens.empty()) set.fallback = "no_request_runs";
    return set;
}

PFlashRequestQuerySet pflash_request_query_set(
        const std::vector<float> & logits,
        const std::vector<PFlashTokenSpan> & human_messages,
        float tau) {
    PFlashRequestQuerySet set;
    if (human_messages.empty()) {
        set.fallback = "no_human_message";
        return set;
    }
    const size_t n = std::min<size_t>(human_messages.size(), (size_t) kRequestMessages);
    for (size_t index = 0; index < n; ++index) {
        const PFlashTokenSpan message = human_messages[index];
        // An empty message (begin == end) has no request tokens.
        if (message.begin < 0 || message.end < message.begin ||
            message.end > (int) logits.size()) {
            set.messages.clear();
            set.fallback = "logits_missing";
            return set;
        }
        set.messages.push_back(request_message_runs(logits, message, tau));
    }

    const std::vector<int> latest = pflash_request_run_tokens(set.messages[0].runs);
    set.previous_used = set.messages[0].request_tokens < kRequestPreviousBelow &&
        set.messages.size() > 1;
    const std::vector<int> previous = set.previous_used
        ? pflash_request_run_tokens(set.messages[1].runs) : std::vector<int>{};
    set.uncapped_tokens = (int) (latest.size() + previous.size());
    const std::vector<int> latest_capped = pflash_request_cap_tail(latest, kRequestQueryCap);
    const std::vector<int> previous_capped = pflash_request_cap_tail(
        previous, kRequestQueryCap - (int) latest_capped.size());
    set.latest_tokens = (int) latest_capped.size();
    set.previous_tokens = (int) previous_capped.size();
    set.tokens = latest_capped;
    set.tokens.insert(set.tokens.end(), previous_capped.begin(), previous_capped.end());
    std::sort(set.tokens.begin(), set.tokens.end());
    set.tokens.erase(std::unique(set.tokens.begin(), set.tokens.end()), set.tokens.end());
    set.capped = (int) set.tokens.size() < set.uncapped_tokens;
    if (set.tokens.empty()) set.fallback = "no_request_runs";
    return set;
}

bool pflash_request_policy_valid(const PFlashRequestPolicy & p, std::string & error) {
    if (p.min_prompt_tokens < 0 || p.max_prompt_tokens < 0 || p.max_runs < 0 || p.recency_pin_tokens < 0) {
        error = "request-probe policy: negative count"; return false;
    }
    if (p.max_prompt_tokens > 0 && p.min_prompt_tokens > p.max_prompt_tokens) {
        error = "request-probe policy: min_prompt_tokens > max_prompt_tokens"; return false;
    }
    if (!(p.min_latest_share >= 0.0f && p.max_latest_share <= 1.0f && p.min_latest_share <= p.max_latest_share)) {
        error = "request-probe policy: latest-share bounds must satisfy 0 <= min <= max <= 1"; return false;
    }
    if (std::isnan(p.min_mean_margin) || std::isnan(p.min_max_margin)) {
        error = "request-probe policy: NaN margin"; return false;
    }
    return true;
}

PFlashRequestAbstainStats pflash_request_abstain(
        const PFlashRequestQuerySet & set,
        const std::vector<float> & logits,
        float tau,
        int prompt_tokens,
        const PFlashRequestPolicy & policy) {
    PFlashRequestAbstainStats s;
    s.prompt_tokens = prompt_tokens;
    double sum = 0.0;
    float mx = -std::numeric_limits<float>::infinity();
    int n = 0;
    for (const int t : set.tokens) {
        if (t < 0 || t >= (int) logits.size()) continue;
        const float m = logits[(size_t) t] - tau;
        sum += m; mx = std::max(mx, m); ++n;
    }
    s.mean_margin = n > 0 ? (float) (sum / n) : 0.0f;
    s.max_margin = n > 0 ? mx : 0.0f;
    if (!set.messages.empty()) {
        const auto & latest = set.messages[0];
        s.latest_share = latest.message_tokens > 0 ? (float) latest.request_tokens / (float) latest.message_tokens : 0.0f;
        s.runs = (int) latest.runs.size();
        if (set.previous_used && set.messages.size() > 1) s.runs += (int) set.messages[1].runs.size();
    }
    if ((policy.min_prompt_tokens > 0 && prompt_tokens < policy.min_prompt_tokens) ||
        (policy.max_prompt_tokens > 0 && prompt_tokens > policy.max_prompt_tokens)) {
        s.reason = "length";
    } else if (s.mean_margin < policy.min_mean_margin) {
        s.reason = "mean_margin";
    } else if (s.max_margin < policy.min_max_margin) {
        s.reason = "max_margin";
    } else if (s.latest_share < policy.min_latest_share || s.latest_share > policy.max_latest_share) {
        s.reason = "latest_share";
    } else if (policy.max_runs > 0 && s.runs > policy.max_runs) {
        s.reason = "runs";
    }
    return s;
}

PFlashTokenSpan pflash_request_recency_pin(PFlashTokenSpan latest, int k) {
    if (k <= 0 || latest.begin < 0 || latest.end <= latest.begin) return {-1, -1};
    return {std::max(latest.begin, latest.end - k), latest.end};
}

uint16_t pflash_f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((bits >> 16) | 0x0040u);   // quiet NaN
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

float pflash_bf16_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void PFlashRequestRowStore::clear() {
    turns_.clear();
    hidden_ = 0;
}

int PFlashRequestRowStore::invalidate_from(int valid_end) {
    const size_t before = turns_.size();
    turns_.erase(std::remove_if(turns_.begin(), turns_.end(), [valid_end](const Turn & turn) {
        return turn.positions.empty() || turn.positions.back() >= valid_end ||
            turn.message.end > valid_end;
    }), turns_.end());
    return (int) (before - turns_.size());
}

const PFlashRequestRowStore::Turn * PFlashRequestRowStore::find_turn(int position) const {
    for (const auto & turn : turns_) {
        if (std::binary_search(turn.positions.begin(), turn.positions.end(), position)) {
            return &turn;
        }
    }
    return nullptr;
}

bool PFlashRequestRowStore::covers(const std::vector<int> & positions) const {
    for (const int position : positions) {
        if (!find_turn(position)) return false;
    }
    return true;
}

bool PFlashRequestRowStore::lookup(const std::vector<int> & positions, int hidden,
                                   std::vector<float> & out) {
    out.assign((size_t) hidden * positions.size(), 0.0f);
    if (positions.empty()) return true;
    if (hidden != hidden_) return false;
    std::vector<const Turn *> used;
    for (size_t index = 0; index < positions.size(); ++index) {
        const Turn * turn = find_turn(positions[index]);
        if (!turn) return false;
        const size_t row = (size_t) (std::lower_bound(turn->positions.begin(),
            turn->positions.end(), positions[index]) - turn->positions.begin());
        const uint16_t * src = turn->rows.data() + row * (size_t) hidden;
        float * dst = out.data() + index * (size_t) hidden;
        for (int k = 0; k < hidden; ++k) dst[k] = pflash_bf16_to_f32(src[k]);
        if (std::find(used.begin(), used.end(), turn) == used.end()) used.push_back(turn);
    }
    ++clock_;
    for (auto & turn : turns_) {
        if (std::find(used.begin(), used.end(), &turn) != used.end()) turn.last_used = clock_;
    }
    return true;
}

int PFlashRequestRowStore::put(PFlashTokenSpan message, const std::vector<int> & positions,
                               const float * rows, int hidden) {
    if (positions.empty() || hidden <= 0 || cap_tokens_ <= 0) return 0;
    if (hidden != hidden_) {
        turns_.clear();
        hidden_ = hidden;
    }
    // A turn over the cap alone keeps its last cap_tokens rows.
    const size_t skip = positions.size() > (size_t) cap_tokens_
        ? positions.size() - (size_t) cap_tokens_ : 0;
    Turn turn;
    turn.message = message;
    turn.positions.assign(positions.begin() + (long) skip, positions.end());
    turn.rows.resize(turn.positions.size() * (size_t) hidden);
    for (size_t index = 0; index < turn.positions.size(); ++index) {
        const float * src = rows + (index + skip) * (size_t) hidden;
        uint16_t * dst = turn.rows.data() + index * (size_t) hidden;
        for (int k = 0; k < hidden; ++k) dst[k] = pflash_f32_to_bf16(src[k]);
    }
    turn.last_used = ++clock_;
    // Replace a turn of the same message, or one sharing a position.
    turns_.erase(std::remove_if(turns_.begin(), turns_.end(), [&](const Turn & old) {
        if (old.message == message) return true;
        for (const int position : turn.positions) {
            if (std::binary_search(old.positions.begin(), old.positions.end(), position)) {
                return true;
            }
        }
        return false;
    }), turns_.end());
    turns_.push_back(std::move(turn));
    int evicted = 0;
    while (tokens() > cap_tokens_ && turns_.size() > 1) {
        auto oldest = std::min_element(turns_.begin(), turns_.end() - 1,
            [](const Turn & a, const Turn & b) { return a.last_used < b.last_used; });
        turns_.erase(oldest);
        ++evicted;
    }
    return evicted;
}

int PFlashRequestRowStore::tokens() const {
    int total = 0;
    for (const auto & turn : turns_) total += (int) turn.positions.size();
    return total;
}

} // namespace luce::pflash
