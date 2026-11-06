#include "server/pflash_unified.h"

#include "common/sha1.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace pflash_unified {

namespace {

std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

bool json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}


uint64_t fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h == 0 ? 1 : h;
}

}  // namespace

// ── Tool list ──────────────────────────────────────────────────────────

bool ToolList::listed(const std::string & name) const {
    if (all) return true;
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool parse_tool_list(const std::string & spec, ToolList & out, std::string & error) {
    out = ToolList{};
    const std::string s = trim(spec);
    if (s.empty()) return true;
    std::vector<std::string> names;
    size_t at = 0;
    while (true) {
        const size_t comma = s.find(',', at);
        const std::string entry = trim(s.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
        if (entry.empty()) {
            error = "PFLASH_REGION_TOOLS: empty entry in '" + s + "'";
            return false;
        }
        for (unsigned char c : entry) {
            if (c < 0x21 || c == 0x7F || c == '"' || c == '[' || c == ']') {
                error = "PFLASH_REGION_TOOLS: invalid character in tool name '" + entry + "'";
                return false;
            }
        }
        names.push_back(entry);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    const bool star = std::find(names.begin(), names.end(), "*") != names.end();
    if (star && names.size() > 1) {
        error = "PFLASH_REGION_TOOLS: '*' must be the only entry";
        return false;
    }
    if (star) {
        out.all = true;
        out.identity = "*";
        return true;
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    out.names = names;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) out.identity += ',';
        out.identity += names[i];
    }
    return true;
}

std::string call_brief(const std::string & name, const nlohmann::json & arguments, size_t max_args) {
    std::string brief;
    for (unsigned char c : name) brief += (c < 0x20 || c == 0x7F) ? ' ' : (char) c;
    nlohmann::json args = arguments;
    if (args.is_string()) {
        const std::string raw = args.get<std::string>();
        nlohmann::json parsed = nlohmann::json::parse(raw, nullptr, false);
        if (!parsed.is_discarded()) args = std::move(parsed);
    }
    if (args.is_null()) return brief;
    if (args.is_object() && args.empty()) return brief;
    std::string text = args.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace);
    if (text.size() > max_args) text = text.substr(0, max_args) + "\xE2\x80\xA6";
    return brief + " " + text;
}

// ── Line structure ─────────────────────────────────────────────────────

Lines split_lines(const std::string & text, size_t begin, size_t end, bool escaped) {
    Lines out;
    out.escaped = escaped;
    out.begin = begin;
    out.end = std::min(end, text.size());
    if (out.begin >= out.end) return out;
    out.starts.push_back(out.begin);
    size_t i = out.begin;
    while (i < out.end) {
        size_t sep = std::string::npos;
        size_t next = i + 1;
        if (!escaped) {
            if (text[i] == '\n') sep = i;
        } else if (text[i] == '\\' && i + 1 < out.end) {
            if (text[i + 1] == 'n') sep = i;
            next = text[i + 1] == 'u' ? std::min(out.end, i + 6) : i + 2;
        }
        if (sep != std::string::npos) {
            next = sep + out.sep_size();
            out.ends.push_back(sep);
            if (next < out.end) out.starts.push_back(next);
        }
        i = next;
    }
    if (out.ends.size() < out.starts.size()) out.ends.push_back(out.end);
    return out;
}

std::vector<std::pair<size_t, size_t>> json_string_values(const std::string & text, size_t begin, size_t end) {
    std::vector<std::pair<size_t, size_t>> out;
    end = std::min(end, text.size());
    if (begin >= end) return out;
    if (!nlohmann::json::accept(text.begin() + (std::ptrdiff_t) begin, text.begin() + (std::ptrdiff_t) end)) {
        return out;
    }
    size_t i = begin;
    while (i < end) {
        if (text[i] != '"') {
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < end && text[j] != '"') j += text[j] == '\\' ? 2 : 1;
        if (j >= end) break;
        size_t k = j + 1;
        while (k < end && json_ws(text[k])) ++k;
        if (k >= end || text[k] != ':') out.emplace_back(i + 1, j);
        i = j + 1;
    }
    return out;
}

Lines detect_lines(const std::string & text, size_t begin, size_t end) {
    Lines best = split_lines(text, begin, end, false);
    for (const auto & v : json_string_values(text, begin, end)) {
        Lines escaped = split_lines(text, v.first, v.second, true);
        if (escaped.count() > best.count()) best = std::move(escaped);
    }
    return best;
}

// ── Region layout ───────────────────────────────────────────────────────

size_t line_cut(const Lines & lines, const std::vector<size_t> & common, int line) {
    if (line <= 0 || (size_t) line >= lines.count()) return std::string::npos;
    const size_t start = lines.starts[(size_t) line];
    if (std::binary_search(common.begin(), common.end(), start)) return start;
    const size_t sep = lines.ends[(size_t) line - 1];
    if (sep > lines.starts[(size_t) line - 1] && std::binary_search(common.begin(), common.end(), sep)) return sep;
    return std::string::npos;
}

bool layout_region(const Lines & lines, const std::vector<size_t> & common,
                   const std::vector<size_t> & proposed, Layout & out, std::string & why) {
    out = Layout{};
    const int n = (int) lines.count();
    if (n < 5) {
        why = "fewer than five lines";
        return false;
    }
    int head_line = -1;
    size_t head_cut = std::string::npos;
    for (int l = 2; l <= n - 3; ++l) {
        head_cut = line_cut(lines, common, l);
        if (head_cut != std::string::npos) {
            head_line = l;
            break;
        }
    }
    int tail_line = -1;
    size_t tail_cut = std::string::npos;
    for (int l = n - 2; l > std::max(head_line, 2); --l) {
        tail_cut = line_cut(lines, common, l);
        if (tail_cut != std::string::npos) {
            tail_line = l;
            break;
        }
    }
    if (head_line < 0 || tail_line < 0) {
        why = "no clean head or tail cut";
        return false;
    }
    std::vector<int> open;
    for (size_t p : proposed) {
        if (p <= lines.begin || p >= lines.end) continue;
        int li = (int) (std::upper_bound(lines.starts.begin(), lines.starts.end(), p) - lines.starts.begin()) - 1;
        if (li < 0) continue;
        if (p >= lines.ends[(size_t) li] && li + 1 < n) ++li;
        if (li > head_line && li < tail_line) open.push_back(li);
    }
    std::sort(open.begin(), open.end());
    open.erase(std::unique(open.begin(), open.end()), open.end());
    out.head_end = head_cut;
    out.tail_begin = tail_cut;
    out.cuts.push_back(head_cut);
    out.cut_line.push_back(head_line);
    for (int li : open) {
        const size_t c = line_cut(lines, common, li);
        if (c == std::string::npos) continue;
        out.cuts.push_back(c);
        out.cut_line.push_back(li);
    }
    out.cuts.push_back(tail_cut);
    out.cut_line.push_back(tail_line);
    return true;
}

// ── Text ────────────────────────────────────────────────────────────────

std::string json_unescape(const std::string & text, size_t begin, size_t end) {
    std::string out;
    out.reserve(end - begin);
    auto hex4 = [&](size_t at, uint32_t & v) {
        if (at + 4 > end) return false;
        v = 0;
        for (size_t k = at; k < at + 4; ++k) {
            const char c = text[k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return false;
        }
        return true;
    };
    auto put_utf8 = [&](uint32_t cp) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    };
    size_t i = begin;
    while (i < end) {
        if (text[i] != '\\' || i + 1 >= end) {
            out += text[i++];
            continue;
        }
        const char c = text[i + 1];
        switch (c) {
            case 'n': out += '\n'; i += 2; continue;
            case 't': out += '\t'; i += 2; continue;
            case 'r': out += '\r'; i += 2; continue;
            case 'b': out += '\b'; i += 2; continue;
            case 'f': out += '\f'; i += 2; continue;
            case '"': case '\\': case '/': out += c; i += 2; continue;
            case 'u': {
                uint32_t v = 0;
                if (!hex4(i + 2, v)) break;
                size_t used = 6;
                if (v >= 0xD800 && v < 0xDC00 && i + 12 <= end && text[i + 6] == '\\' && text[i + 7] == 'u') {
                    uint32_t lo = 0;
                    if (hex4(i + 8, lo) && lo >= 0xDC00 && lo < 0xE000) {
                        v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        used = 12;
                    }
                }
                put_utf8(v);
                i += used;
                continue;
            }
            default: break;
        }
        out += text[i++];
    }
    return out;
}


std::string json_escape(const std::string & text) {
    static const char * hex = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0xF];
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

bool layout_region_inline(const std::string & text, const Lines & lines, const std::vector<size_t> & common,
                          const std::vector<size_t> & proposed, bool keep_first_line, bool keep_last_line,
                          Layout & out, std::string & why) {
    out = Layout{};
    out.inline_units = true;
    const int n = (int) lines.count();
    // Escaped lines: the bytes after a backslash that belong to its escape.
    std::vector<char> in_escape;
    if (lines.escaped && lines.end > lines.begin) {
        in_escape.assign(lines.end - lines.begin, 0);
        for (size_t i = lines.begin; i < lines.end;) {
            if (text[i] != '\\' || i + 1 >= lines.end) {
                ++i;
                continue;
            }
            const size_t next = text[i + 1] == 'u' ? std::min(lines.end, i + 6) : i + 2;
            for (size_t j = i + 1; j < next; ++j) in_escape[j - lines.begin] = 1;
            i = next;
        }
    }
    const auto usable = [&] (size_t p) {
        return p > lines.begin && p < lines.end && ((unsigned char) text[p] & 0xC0) != 0x80 &&
               (in_escape.empty() || !in_escape[p - lines.begin]);
    };
    std::vector<size_t> cuts;
    for (int l = 1; l < n; ++l) {
        const size_t c = line_cut(lines, common, l);
        if (c != std::string::npos && usable(c)) cuts.push_back(c);
    }
    for (size_t p : proposed) {
        auto it = std::lower_bound(common.begin(), common.end(), p);
        while (it != common.end() && *it < lines.end && !usable(*it)) ++it;
        if (it != common.end() && usable(*it)) cuts.push_back(*it);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    if (n >= 2) {
        // An edge line kept whole: no cut before the first line's separator, none after the last line's start.
        const size_t from = keep_first_line ? lines.ends[0] : lines.begin;
        const size_t to = keep_last_line ? lines.starts[(size_t) n - 1] : lines.end;
        cuts.erase(std::remove_if(cuts.begin(), cuts.end(), [&] (size_t c) { return c < from || c > to; }),
                   cuts.end());
    }
    if (cuts.size() < 3) {
        why = "fewer than three inline cuts";
        return false;
    }
    for (size_t c : cuts) {
        const int li = (int) (std::upper_bound(lines.starts.begin(), lines.starts.end(), c) - lines.starts.begin()) - 1;
        out.cut_line.push_back((std::max)(0, li));
    }
    out.cuts = std::move(cuts);
    out.head_end = out.cuts.front();
    out.tail_begin = out.cuts.back();
    return true;
}

std::pair<int, int> unit_lines(const Lines & lines, const Layout & layout, int first_unit, int last_unit) {
    if (!layout.inline_units) {
        return {layout.cut_line[(size_t) first_unit], layout.cut_line[(size_t) last_unit + 1]};
    }
    const size_t last_byte = layout.cuts[(size_t) last_unit + 1] - 1;
    const int last = (int) (std::upper_bound(lines.starts.begin(), lines.starts.end(), last_byte) -
                            lines.starts.begin()) - 1;
    return {layout.cut_line[(size_t) first_unit], (std::max)(layout.cut_line[(size_t) first_unit], last) + 1};
}

std::string unit_marker_text(const Layout & layout, int first_unit, int last_unit) {
    if (layout.inline_units) return "[\xE2\x80\xA6 text not shown \xE2\x80\xA6]";
    return marker_text(layout.line_count(first_unit, last_unit));
}

std::string region_preamble_text(RegionKind kind, const Layout & layout, int shown_lines, int total_lines) {
    if (!layout.inline_units) return preamble_text(kind, shown_lines, total_lines);
    if (kind == RegionKind::ToolOutput) {
        return "[Server note: this tool result was shortened; each omitted range is marked inline. "
               "Call the tool again if you need an omitted part.]";
    }
    return "[Server note: this message was shortened; each omitted range is marked inline.]";
}

std::string marker_text(int lines_not_shown) {
    return "[\xE2\x80\xA6 " + std::to_string(lines_not_shown) + (lines_not_shown == 1 ? " line" : " lines") +
           " not shown \xE2\x80\xA6]";
}

std::string preamble_text(RegionKind kind, int shown_lines, int total_lines) {
    const std::string counts = std::to_string(shown_lines) + " of " + std::to_string(total_lines) +
                               " lines are shown; each omitted range is marked inline.";
    if (kind == RegionKind::ToolOutput) {
        return "[Server note: this tool result was shortened: " + counts +
               " Call the tool again if you need an omitted part.]";
    }
    return "[Server note: this message was shortened: " + counts + "]";
}

std::string recall_label_text(RegionKind kind, const std::string & brief) {
    if (kind == RegionKind::ToolOutput) return "[From the earlier result of " + brief + "]";
    return "[From an earlier user message]";
}

std::string inserted_line(const std::string & text, const Lines & lines, size_t at, size_t next,
                          const std::string & body) {
    const std::string sep = lines.sep();
    const bool before = at > lines.begin && std::binary_search(lines.starts.begin(), lines.starts.end(), at);
    bool after = false;
    if (next < lines.end && next + sep.size() <= text.size() && text.compare(next, sep.size(), sep) == 0) {
        after = std::binary_search(lines.ends.begin(), lines.ends.end(), next);
    }
    std::string out;
    if (!before) out += sep;
    out += lines.escaped ? json_escape(body) : body;
    if (!after) out += sep;
    return out;
}

std::string render_region_text(const std::string & text, size_t content_begin, size_t content_end,
                               const Lines & lines, const Layout & layout, const std::vector<char> & admitted,
                               RegionKind kind) {
    const int n = layout.units();
    int dropped = 0;
    for (int u = 0; u < n; ++u) {
        if (!admitted[(size_t) u]) dropped += layout.line_count(u, u);
    }
    std::string out = text.substr(content_begin, layout.head_end - content_begin);
    out += inserted_line(text, lines, layout.head_end, layout.head_end,
                         region_preamble_text(kind, layout, (int) lines.count() - dropped, (int) lines.count()));
    for (int u = 0; u < n;) {
        if (admitted[(size_t) u]) {
            out += text.substr(layout.cuts[(size_t) u], layout.cuts[(size_t) u + 1] - layout.cuts[(size_t) u]);
            ++u;
            continue;
        }
        int b = u;
        while (b + 1 < n && !admitted[(size_t) b + 1]) ++b;
        out += inserted_line(text, lines, layout.cuts[(size_t) u], layout.cuts[(size_t) b + 1],
                             unit_marker_text(layout, u, b));
        u = b + 1;
    }
    out += text.substr(layout.tail_begin, content_end - layout.tail_begin);
    return out;
}

// ── Admission ──────────────────────────────────────────────────────────

namespace {

// The maximal run around ``u`` of units whose ``in`` equals ``in[u]``.
std::pair<int, int> run_around(const std::vector<char> & in, int u) {
    int a = u;
    int b = u;
    while (a > 0 && in[(size_t) a - 1] == in[(size_t) u]) --a;
    while (b + 1 < (int) in.size() && in[(size_t) b + 1] == in[(size_t) u]) ++b;
    return {a, b};
}

}  // namespace

Admission admit_paging(const std::vector<int> & unit_cost, const std::vector<char> & locked,
                       const std::vector<int> & order, int fixed, int cap,
                       const std::function<int(int, int)> & marker_cost) {
    Admission out;
    const int n = (int) unit_cost.size();
    out.admitted.assign((size_t) n, 0);
    for (int u = 0; u < n; ++u) out.admitted[(size_t) u] = (u < (int) locked.size() && locked[(size_t) u]) ? 1 : 0;
    out.cost = fixed;
    for (int u = 0; u < n;) {
        if (out.admitted[(size_t) u]) {
            ++u;
            continue;
        }
        const auto run = run_around(out.admitted, u);
        out.cost += marker_cost(run.first, run.second);
        u = run.second + 1;
    }
    if (out.cost > cap) return out;
    out.ok = true;
    for (int u : order) {
        if (u < 0 || u >= n || out.admitted[(size_t) u]) continue;
        const auto gap = run_around(out.admitted, u);
        int delta = unit_cost[(size_t) u] - marker_cost(gap.first, gap.second);
        if (gap.first <= u - 1) delta += marker_cost(gap.first, u - 1);
        if (u + 1 <= gap.second) delta += marker_cost(u + 1, gap.second);
        if (out.cost + delta > cap) continue;
        out.admitted[(size_t) u] = 1;
        out.cost += delta;
        ++out.taken;
    }
    return out;
}

namespace {

// The cost change of taking unit ``u`` into ``admitted``: its tokens, one label for the run it forms, minus the labels
// of the runs it joins.
int recall_delta(const std::vector<char> & admitted, const std::vector<int> & unit_cost, int u,
                 const std::function<int(int, int)> & label_cost) {
    const int n = (int) admitted.size();
    int a = u;
    int b = u;
    int delta = unit_cost[(size_t) u];
    if (u > 0 && admitted[(size_t) u - 1]) {
        const auto run = run_around(admitted, u - 1);
        a = run.first;
        delta -= label_cost(run.first, run.second);
    }
    if (u + 1 < n && admitted[(size_t) u + 1]) {
        const auto run = run_around(admitted, u + 1);
        b = run.second;
        delta -= label_cost(run.first, run.second);
    }
    return delta + label_cost(a, b);
}

}  // namespace

Admission admit_recall(const std::vector<int> & unit_cost, const std::vector<char> & held,
                       const std::vector<int> & order, int cap,
                       const std::function<int(int, int)> & label_cost) {
    Admission out;
    const int n = (int) unit_cost.size();
    out.admitted.assign((size_t) n, 0);
    out.ok = cap >= 0;
    if (!out.ok) return out;
    for (int u : order) {
        if (u < 0 || u >= n || out.admitted[(size_t) u]) continue;
        if (u < (int) held.size() && held[(size_t) u]) continue;
        const int delta = recall_delta(out.admitted, unit_cost, u, label_cost);
        if (out.cost + delta > cap) continue;
        out.admitted[(size_t) u] = 1;
        out.cost += delta;
        ++out.taken;
    }
    return out;
}

GatedRecall admit_recall_gated(const std::vector<RecallRegion> & regions, double context_mass, double theta) {
    GatedRecall out;
    out.regions.resize(regions.size());
    struct Entry {
        size_t region;
        int unit;
        double score;
    };
    std::vector<Entry> order;
    double deficit = 0.0;
    for (size_t r = 0; r < regions.size(); ++r) {
        const RecallRegion & region = regions[r];
        const int n = (int) region.unit_cost.size();
        Admission & a = out.regions[r];
        a.admitted.assign((size_t) n, 0);
        a.ok = region.cap >= 0;
        for (int u = 0; u < n; ++u) {
            if (u < (int) region.held.size() && region.held[(size_t) u]) continue;
            const double mass = u < (int) region.mass.size() ? region.mass[(size_t) u] : 0.0;
            if (std::isfinite(mass) && mass > 0.0) deficit += mass;
            const double score = u < (int) region.score.size() ? region.score[(size_t) u] : NAN;
            order.push_back({r, u, std::isnan(score) ? -INFINITY : score});
        }
    }
    out.deficit_before = deficit / context_mass;
    std::stable_sort(order.begin(), order.end(), [] (const Entry & x, const Entry & y) { return x.score > y.score; });
    for (const Entry & e : order) {
        if (deficit / context_mass < theta) break;
        const RecallRegion & region = regions[e.region];
        Admission & a = out.regions[e.region];
        if (!a.ok) continue;
        const auto label_cost = [&] (int, int) { return region.label_cost; };
        const int delta = recall_delta(a.admitted, region.unit_cost, e.unit, label_cost);
        if (a.cost + delta > region.cap) continue;
        a.admitted[(size_t) e.unit] = 1;
        a.cost += delta;
        ++a.taken;
        const double mass = e.unit < (int) region.mass.size() ? region.mass[(size_t) e.unit] : 0.0;
        if (std::isfinite(mass) && mass > 0.0) deficit -= mass;
    }
    out.deficit_after = (std::max)(0.0, deficit) / context_mass;
    out.gated = out.deficit_before < theta;
    return out;
}

std::vector<int> rank_units(const std::vector<double> & score, const std::vector<char> & skip) {
    std::vector<int> order;
    for (int u = 0; u < (int) score.size(); ++u) {
        if (u < (int) skip.size() && skip[(size_t) u]) continue;
        order.push_back(u);
    }
    auto key = [&](int u) {
        const double s = score[(size_t) u];
        return std::isnan(s) ? -INFINITY : s;
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key(a) > key(b); });
    return order;
}

// ── Re-reads ───────────────────────────────────────────────────────────

std::vector<uint64_t> line_hashes(const std::string & text, const Lines & lines) {
    std::vector<uint64_t> out;
    out.reserve(lines.count());
    for (size_t i = 0; i < lines.count(); ++i) {
        std::string line = lines.escaped ? json_unescape(text, lines.starts[i], lines.ends[i])
                                         : text.substr(lines.starts[i], lines.ends[i] - lines.starts[i]);
        while (!line.empty() && line.back() == '\r') line.pop_back();
        const bool blank = std::all_of(line.begin(), line.end(), [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
        });
        out.push_back(blank ? 0 : fnv1a(line));
    }
    return out;
}

RereadMatch match_reread(const std::vector<uint64_t> & candidate, const std::vector<uint64_t> & region,
                         int min_run, int max_repeat) {
    RereadMatch out;
    out.region_covered.assign(region.size(), 0);
    std::unordered_map<uint64_t, int> freq;
    for (uint64_t h : region) if (h) ++freq[h];
    auto informative = [&](uint64_t h) {
        if (!h) return false;
        const auto it = freq.find(h);
        return it == freq.end() || it->second <= max_repeat;
    };
    std::vector<int> reg;     // informative region lines
    out.region_informative.assign(region.size(), 0);
    for (int i = 0; i < (int) region.size(); ++i) {
        if (!informative(region[(size_t) i])) continue;
        reg.push_back(i);
        out.region_informative[(size_t) i] = 1;
    }
    std::vector<int> cand;    // informative candidate lines
    for (int j = 0; j < (int) candidate.size(); ++j) if (informative(candidate[(size_t) j])) cand.push_back(j);
    out.candidate_informative = (int) cand.size();
    std::unordered_map<uint64_t, std::vector<int>> where;
    for (int p = 0; p < (int) reg.size(); ++p) where[region[(size_t) reg[(size_t) p]]].push_back(p);
    std::vector<char> matched(cand.size(), 0);
    std::unordered_map<int, int> prev;
    for (int j = 0; j < (int) cand.size(); ++j) {
        std::unordered_map<int, int> cur;
        const auto it = where.find(candidate[(size_t) cand[(size_t) j]]);
        if (it != where.end()) {
            for (int p : it->second) {
                const auto pr = prev.find(p - 1);
                const int len = pr == prev.end() ? 1 : pr->second + 1;
                cur[p] = len;
                if (len < min_run) continue;
                const int from = len == min_run ? len - 1 : 0;
                for (int k = from; k >= 0; --k) {
                    matched[(size_t) (j - k)] = 1;
                    out.region_covered[(size_t) reg[(size_t) (p - k)]] = 1;
                }
            }
        }
        prev.swap(cur);
    }
    for (char m : matched) out.matched_candidate += m ? 1 : 0;
    // Skipped region lines between two covered informative lines are covered.
    for (size_t k = 0; k + 1 < reg.size(); ++k) {
        if (!out.region_covered[(size_t) reg[k]] || !out.region_covered[(size_t) reg[k + 1]]) continue;
        for (int i = reg[k] + 1; i < reg[k + 1]; ++i) out.region_covered[(size_t) i] = 1;
    }
    return out;
}

bool lines_covered(const std::vector<char> & covered, const std::vector<char> & informative, int begin, int end) {
    bool any_informative = false;
    bool informative_covered = true;
    bool all_covered = true;
    for (int line = begin; line < end; ++line) {
        const bool is_covered = covered[(size_t) line] != 0;
        all_covered = all_covered && is_covered;
        if (informative[(size_t) line]) {
            any_informative = true;
            informative_covered = informative_covered && is_covered;
        }
    }
    return any_informative ? informative_covered : all_covered;
}

// ── View ───────────────────────────────────────────────────────────────

size_t View::bytes() const {
    size_t n = sizeof(View) + config.size();
    n += (raw_tokens.size() + drafter_ids.size() + view_tokens.size()) * sizeof(int32_t);
    n += pieces.size() * sizeof(Piece);
    for (const auto & p : pieces) n += p.kind.size();
    for (const auto & r : regions) {
        n += sizeof(Region) + r.brief.size() + r.tool_call_id.size() + r.content_sha.size();
        n += (r.lines.starts.size() + r.lines.ends.size() + r.layout.cuts.size()) * sizeof(size_t);
        n += r.layout.cut_line.size() * sizeof(int) + r.hashes.size() * sizeof(uint64_t);
        n += (r.unit_tb.size() + r.unit_te.size() + r.unit_db.size() + r.unit_de.size()) * sizeof(int);
        n += r.mandatory.size() + r.kept.size() + r.held.size() + r.reread.size();
    }
    return n;
}

namespace {

bool prefix_of(const std::vector<int32_t> & stored, int gen_begin, const std::vector<int32_t> & request) {
    const size_t n = gen_begin < 0 ? stored.size() : std::min(stored.size(), (size_t) gen_begin);
    return request.size() >= n && std::equal(stored.begin(), stored.begin() + (std::ptrdiff_t) n, request.begin());
}

}  // namespace

bool ViewStore::find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
                     const std::string & config, View & out) {
    std::lock_guard<std::mutex> lock(mutex_);
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < (int) entries_.size(); ++i) {
        const View & v = entries_[(size_t) i].view;
        if (v.config != config) continue;
        if (!prefix_of(v.raw_tokens, v.raw_gen_begin, raw_tokens)) continue;
        if (!prefix_of(v.drafter_ids, v.drafter_gen_begin, drafter_ids)) continue;
        const size_t len = v.raw_gen_begin < 0 ? v.raw_tokens.size() : (size_t) v.raw_gen_begin;
        if (best < 0 || len > best_len ||
            (len == best_len && entries_[(size_t) i].used > entries_[(size_t) best].used)) {
            best = i;
            best_len = len;
        }
    }
    if (best < 0) return false;
    entries_[(size_t) best].used = ++clock_;
    out = entries_[(size_t) best].view;
    return true;
}

std::vector<View> ViewStore::remember(View view) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<View> evicted;
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const View & v = it->view;
        if (v.config == view.config && v.raw_tokens == view.raw_tokens && v.drafter_ids == view.drafter_ids) {
            entries_.erase(it);
            break;
        }
    }
    entries_.push_back(Entry{std::move(view), ++clock_});
    auto total = [&] {
        size_t n = 0;
        for (const auto & e : entries_) n += e.view.bytes();
        return n;
    };
    while (entries_.size() > 1 && (entries_.size() > capacity_ || total() > byte_cap_)) {
        size_t oldest = 0;
        for (size_t i = 1; i + 1 < entries_.size(); ++i) {
            if (entries_[i].used < entries_[oldest].used) oldest = i;
        }
        evicted.push_back(std::move(entries_[oldest].view));
        entries_.erase(entries_.begin() + (std::ptrdiff_t) oldest);
    }
    return evicted;
}

std::vector<View> ViewStore::all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const Entry *> order;
    for (const auto & e : entries_) order.push_back(&e);
    std::sort(order.begin(), order.end(), [] (const Entry * a, const Entry * b) { return a->used > b->used; });
    std::vector<View> out;
    for (const Entry * e : order) out.push_back(e->view);
    return out;
}

void ViewStore::forget_lineage(uint64_t lineage) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&] (const Entry & e) { return e.view.lineage == lineage; }),
                   entries_.end());
}

std::vector<uint64_t> ViewStore::lineages_continued_by(const std::vector<int32_t> & raw_tokens) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> out;
    for (const auto & e : entries_) {
        if (prefix_of(e.view.raw_tokens, e.view.raw_gen_begin, raw_tokens)) out.push_back(e.view.lineage);
    }
    return out;
}

size_t ViewStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

size_t ViewStore::bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (const auto & e : entries_) n += e.view.bytes();
    return n;
}

namespace {

std::string bits(const std::vector<char> & v) {
    std::string out;
    for (char b : v) out += b ? '1' : '0';
    return out;
}

bool unbits(const nlohmann::json & j, std::vector<char> & out) {
    const std::string s = j.get<std::string>();
    out.clear();
    for (char ch : s) {
        if (ch != '0' && ch != '1') return false;
        out.push_back(ch == '1' ? 1 : 0);
    }
    return true;
}

uint64_t token_hash(const std::vector<int32_t> & tokens, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const auto * p = reinterpret_cast<const unsigned char *>(tokens.data());
    for (size_t i = 0; i < n * sizeof(int32_t); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// token_hash(tokens, n) for every n in ``lengths``, in one pass over the tokens.
std::unordered_map<size_t, uint64_t> prefix_hashes(const std::vector<int32_t> & tokens, std::vector<size_t> lengths) {
    std::sort(lengths.begin(), lengths.end());
    std::unordered_map<size_t, uint64_t> out;
    uint64_t h = 1469598103934665603ULL;
    const auto * p = reinterpret_cast<const unsigned char *>(tokens.data());
    size_t at = 0;
    for (size_t n : lengths) {
        for (; at < n * sizeof(int32_t); ++at) {
            h ^= p[at];
            h *= 1099511628211ULL;
        }
        out[n] = h;
    }
    return out;
}

size_t gen_prefix(const std::vector<int32_t> & tokens, int gen_begin) {
    return gen_begin < 0 ? tokens.size() : std::min(tokens.size(), (size_t) gen_begin);
}

uint64_t now_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Everything serving indexes without a check: lines inside the content and in order, one hash and one re-read flag
// per line, at least one unit, cuts and their lines strictly ascending from the head end to the tail start, and
// per-unit token ranges that tile the content's tokens in order.
bool region_consistent(const Region & r, std::string & error) {
    const auto fail = [&] (const char * what) {
        error = std::string("region ") + what;
        return false;
    };
    const Lines & l = r.lines;
    const size_t n = l.count();
    if (n == 0 || l.begin < r.cb || l.end > r.ce || l.begin > l.end) return fail("lines");
    for (size_t i = 0; i < n; ++i) {
        if (l.starts[i] > l.ends[i] || l.starts[i] < l.begin || l.ends[i] > l.end ||
            (i > 0 && l.starts[i] < l.ends[i - 1])) {
            return fail("lines");
        }
    }
    if (r.hashes.size() != n || r.reread.size() != n) return fail("per-line vectors");
    const Layout & y = r.layout;
    if (y.cuts.size() < 2 || y.cuts.front() != y.head_end || y.cuts.back() != y.tail_begin) return fail("layout");
    // Line units open a later line at every cut; inline units may cut one line several times.
    const int min_line = y.inline_units ? 0 : 1;
    for (size_t i = 0; i < y.cuts.size(); ++i) {
        if (y.cuts[i] < l.begin || y.cuts[i] > l.end || y.cut_line[i] < min_line || (size_t) y.cut_line[i] >= n ||
            (i > 0 && (y.cuts[i] <= y.cuts[i - 1] || y.cut_line[i] < y.cut_line[i - 1] ||
                       (!y.inline_units && y.cut_line[i] == y.cut_line[i - 1])))) {
            return fail("layout");
        }
    }
    const auto tiles = [] (const std::vector<int> & b, const std::vector<int> & e, int lo, int hi) {
        for (size_t u = 0; u < b.size(); ++u) {
            if (b[u] < lo || b[u] > e[u] || e[u] > hi || (u > 0 && b[u] != e[u - 1])) return false;
        }
        return true;
    };
    if (!tiles(r.unit_tb, r.unit_te, r.tb, r.te) || !tiles(r.unit_db, r.unit_de, r.db, r.de)) return fail("units");
    return true;
}

}  // namespace

nlohmann::json view_to_json(const View & view) {
    using nlohmann::json;
    json regions = json::array();
    for (const auto & r : view.regions) {
        regions.push_back({
            {"kind", r.kind == RegionKind::ToolOutput ? "tool_output" : "user_content"},
            {"brief", r.brief}, {"tool_call_id", r.tool_call_id}, {"content_sha", r.content_sha},
            {"cb", r.cb}, {"ce", r.ce}, {"tb", r.tb}, {"te", r.te}, {"db", r.db}, {"de", r.de},
            {"lines", {{"escaped", r.lines.escaped}, {"begin", r.lines.begin}, {"end", r.lines.end},
                       {"starts", r.lines.starts}, {"ends", r.lines.ends}}},
            {"layout", {{"head_end", r.layout.head_end}, {"tail_begin", r.layout.tail_begin},
                        {"cuts", r.layout.cuts}, {"cut_line", r.layout.cut_line},
                        {"inline", r.layout.inline_units}}},
            {"hashes", r.hashes},
            {"unit_tb", r.unit_tb}, {"unit_te", r.unit_te}, {"unit_db", r.unit_db}, {"unit_de", r.unit_de},
            {"mandatory", bits(r.mandatory)}, {"kept", bits(r.kept)}, {"held", bits(r.held)},
            {"reread", bits(r.reread)},
            {"cap", r.cap}, {"droppable", r.droppable}, {"recalled_tokens", r.recalled_tokens},
        });
    }
    json pieces = json::array();
    for (const auto & p : view.pieces) pieces.push_back(json::array({p.begin, p.end, p.kind, p.region, p.turn}));
    return {{"format", 1}, {"lineage", view.lineage}, {"version", view.version}, {"config", view.config},
            {"raw_tokens", view.raw_tokens}, {"raw_gen_begin", view.raw_gen_begin},
            {"drafter_ids", view.drafter_ids}, {"drafter_gen_begin", view.drafter_gen_begin},
            {"view_tokens", view.view_tokens}, {"view_gen_begin", view.view_gen_begin},
            {"regions", regions}, {"pieces", pieces}, {"turns", view.turns}};
}

bool view_from_json(const nlohmann::json & j, View & out, std::string & error) {
    try {
        if (j.at("format").get<int>() != 1) {
            error = "format";
            return false;
        }
        View v;
        v.lineage = j.at("lineage").get<uint64_t>();
        v.version = j.at("version").get<int>();
        v.config = j.at("config").get<std::string>();
        v.raw_tokens = j.at("raw_tokens").get<std::vector<int32_t>>();
        v.raw_gen_begin = j.at("raw_gen_begin").get<int>();
        v.drafter_ids = j.at("drafter_ids").get<std::vector<int32_t>>();
        v.drafter_gen_begin = j.at("drafter_gen_begin").get<int>();
        v.view_tokens = j.at("view_tokens").get<std::vector<int32_t>>();
        v.view_gen_begin = j.at("view_gen_begin").get<int>();
        v.turns = j.at("turns").get<int>();
        const auto within = [] (int at, size_t size) { return at >= 0 && (size_t) at <= size; };
        if (!within(v.raw_gen_begin, v.raw_tokens.size()) || !within(v.drafter_gen_begin, v.drafter_ids.size()) ||
            v.view_gen_begin <= 0 || !within(v.view_gen_begin, v.view_tokens.size())) {
            error = "generation offsets";
            return false;
        }
        for (const auto & jr : j.at("regions")) {
            Region r;
            const std::string kind = jr.at("kind").get<std::string>();
            if (kind != "tool_output" && kind != "user_content") {
                error = "region kind";
                return false;
            }
            r.kind = kind == "tool_output" ? RegionKind::ToolOutput : RegionKind::UserContent;
            r.brief = jr.at("brief").get<std::string>();
            r.tool_call_id = jr.at("tool_call_id").get<std::string>();
            r.content_sha = jr.at("content_sha").get<std::string>();
            r.cb = jr.at("cb").get<size_t>();
            r.ce = jr.at("ce").get<size_t>();
            r.tb = jr.at("tb").get<int>();
            r.te = jr.at("te").get<int>();
            r.db = jr.at("db").get<int>();
            r.de = jr.at("de").get<int>();
            const auto & jl = jr.at("lines");
            r.lines.escaped = jl.at("escaped").get<bool>();
            r.lines.begin = jl.at("begin").get<size_t>();
            r.lines.end = jl.at("end").get<size_t>();
            r.lines.starts = jl.at("starts").get<std::vector<size_t>>();
            r.lines.ends = jl.at("ends").get<std::vector<size_t>>();
            const auto & jy = jr.at("layout");
            r.layout.head_end = jy.at("head_end").get<size_t>();
            r.layout.tail_begin = jy.at("tail_begin").get<size_t>();
            r.layout.cuts = jy.at("cuts").get<std::vector<size_t>>();
            r.layout.cut_line = jy.at("cut_line").get<std::vector<int>>();
            r.layout.inline_units = jy.at("inline").get<bool>();
            r.hashes = jr.at("hashes").get<std::vector<uint64_t>>();
            r.unit_tb = jr.at("unit_tb").get<std::vector<int>>();
            r.unit_te = jr.at("unit_te").get<std::vector<int>>();
            r.unit_db = jr.at("unit_db").get<std::vector<int>>();
            r.unit_de = jr.at("unit_de").get<std::vector<int>>();
            if (!unbits(jr.at("mandatory"), r.mandatory) || !unbits(jr.at("kept"), r.kept) ||
                !unbits(jr.at("held"), r.held) || !unbits(jr.at("reread"), r.reread)) {
                error = "region flags";
                return false;
            }
            r.cap = jr.at("cap").get<int>();
            r.droppable = jr.at("droppable").get<int>();
            r.recalled_tokens = jr.at("recalled_tokens").get<int>();
            const size_t units = (size_t) r.layout.units();
            const size_t lines = r.lines.count();
            if (r.cb > r.ce || r.tb < 0 || r.tb > r.te || (size_t) r.te > v.raw_tokens.size() || r.db < 0 ||
                r.db > r.de || (size_t) r.de > v.drafter_ids.size() || r.lines.ends.size() != lines ||
                r.layout.cut_line.size() != r.layout.cuts.size() || r.unit_tb.size() != units ||
                r.unit_te.size() != units || r.unit_db.size() != units || r.unit_de.size() != units ||
                r.mandatory.size() != units || r.kept.size() != units || r.held.size() != units) {
                error = "region shape";
                return false;
            }
            if (r.te > v.raw_gen_begin || r.de > v.drafter_gen_begin) {
                error = "region after the generation prompt";
                return false;
            }
            if (!region_consistent(r, error)) return false;
            v.regions.push_back(std::move(r));
        }
        for (const auto & jp : j.at("pieces")) {
            Piece p;
            p.begin = jp.at(0).get<int>();
            p.end = jp.at(1).get<int>();
            p.kind = jp.at(2).get<std::string>();
            p.region = jp.at(3).get<int>();
            p.turn = jp.at(4).get<int>();
            if (p.begin < 0 || p.begin > p.end || (size_t) p.end > v.view_tokens.size() || p.region < -1 ||
                p.region >= (int) v.regions.size()) {
                error = "piece";
                return false;
            }
            v.pieces.push_back(std::move(p));
        }
        out = std::move(v);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool ViewDiskStore::init(const std::string & dir, const std::string & identity, uint64_t ttl_seconds, size_t cap) {
    namespace fs = std::filesystem;
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
    metas_.clear();
    last_written_ = 0;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        std::fprintf(stderr, "[pflash-unified] views cannot be kept on disk (%s): RAM only\n", dir.c_str());
        return false;
    }
    dir_ = dir;
    identity_ = identity;
    ttl_ = ttl_seconds;
    cap_ = cap;
    const uint64_t now = this->now();
    size_t foreign = 0;
    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        const fs::path path = entry.path();
        if (path.extension() == ".tmp") {   // a write the previous process did not finish
            fs::remove(path, ec);
            continue;
        }
        if (path.extension() != ".view") continue;
        std::ifstream in(path);
        std::string line;
        Meta m;
        std::string id;
        bool ok = false;
        if (std::getline(in, line)) {
            try {
                const auto h = nlohmann::json::parse(line);
                id = h.at("identity").get<std::string>();
                m.lineage = h.at("lineage").get<uint64_t>();
                m.config = h.at("config").get<std::string>();
                m.raw_len = h.at("raw_len").get<size_t>();
                m.raw_hash = h.at("raw_hash").get<uint64_t>();
                m.drafter_len = h.at("drafter_len").get<size_t>();
                m.drafter_hash = h.at("drafter_hash").get<uint64_t>();
                m.written = h.at("written").get<uint64_t>();
                ok = true;
            } catch (const std::exception &) {
            }
        }
        if (!ok || (ttl_ > 0 && m.written + ttl_ * 1000 < now)) {
            fs::remove(path, ec);
            continue;
        }
        if (id != identity_) {   // another model's or config's views: never read, deleted after the TTL
            ++foreign;
            continue;
        }
        m.path = path.string();
        metas_.push_back(std::move(m));
    }
    // Oldest written first; later saves are stamped after the newest one here, so the cap drops the earliest.
    std::sort(metas_.begin(), metas_.end(), [] (const Meta & a, const Meta & b) { return a.written < b.written; });
    last_written_ = metas_.empty() ? 0 : metas_.back().written;
    enabled_ = true;
    enforce_cap_locked();
    std::fprintf(stderr, "[pflash-unified] views on disk: %zu in %s (%zu of another identity)\n", metas_.size(),
                 dir_.c_str(), foreign);
    return true;
}

bool ViewDiskStore::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

bool ViewDiskStore::save(const View & view) {
    namespace fs = std::filesystem;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return false;
    Meta m;
    m.lineage = view.lineage;
    m.config = view.config;
    m.raw_len = gen_prefix(view.raw_tokens, view.raw_gen_begin);
    m.raw_hash = token_hash(view.raw_tokens, m.raw_len);
    m.drafter_len = gen_prefix(view.drafter_ids, view.drafter_gen_begin);
    m.drafter_hash = token_hash(view.drafter_ids, m.drafter_len);
    m.written = std::max(now(), last_written_ + 1);
    last_written_ = m.written;
    const nlohmann::json header = {
        {"format", 1}, {"identity", identity_}, {"lineage", m.lineage}, {"config", m.config},
        {"raw_len", m.raw_len}, {"raw_hash", m.raw_hash}, {"drafter_len", m.drafter_len},
        {"drafter_hash", m.drafter_hash}, {"written", m.written}};
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.view",
                  (unsigned long long) fnv1a(m.config + ":" + std::to_string(m.raw_len) + ":" +
                                             std::to_string(m.raw_hash) + ":" + std::to_string(m.drafter_len) +
                                             ":" + std::to_string(m.drafter_hash)));
    m.path = dir_ + "/" + name;
    const std::string tmp = m.path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << header.dump() << '\n' << view_to_json(view).dump() << '\n';
        out.flush();
        if (!out) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, m.path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    metas_.erase(std::remove_if(metas_.begin(), metas_.end(), [&] (const Meta & x) { return x.path == m.path; }),
                 metas_.end());
    metas_.push_back(std::move(m));
    enforce_cap_locked();
    return true;
}

bool ViewDiskStore::find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
                         const std::string & config, size_t longer_than, View & out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return false;
    // The TTL counts from a view's last write. A view read here returns to the RAM store and is written again when
    // it leaves it, so a conversation in use stays on disk.
    if (ttl_ > 0) {
        const uint64_t t = now();
        for (size_t i = metas_.size(); i-- > 0;) {
            if (metas_[i].written + ttl_ * 1000 < t) remove_locked(i);
        }
    }
    std::vector<size_t> candidates, raw_lengths, drafter_lengths;
    for (size_t i = 0; i < metas_.size(); ++i) {
        const Meta & m = metas_[i];
        if (m.config != config || m.raw_len <= longer_than || m.raw_len > raw_tokens.size() ||
            m.drafter_len > drafter_ids.size()) {
            continue;
        }
        candidates.push_back(i);
        raw_lengths.push_back(m.raw_len);
        drafter_lengths.push_back(m.drafter_len);
    }
    if (candidates.empty()) return false;
    // Each request prefix is hashed once, however many views are on disk.
    const auto raw_hashes = prefix_hashes(raw_tokens, raw_lengths);
    const auto drafter_hashes = prefix_hashes(drafter_ids, drafter_lengths);
    std::vector<size_t> order;
    for (size_t i : candidates) {
        if (raw_hashes.at(metas_[i].raw_len) == metas_[i].raw_hash &&
            drafter_hashes.at(metas_[i].drafter_len) == metas_[i].drafter_hash) {
            order.push_back(i);
        }
    }
    std::sort(order.begin(), order.end(), [&] (size_t a, size_t b) {
        return metas_[a].raw_len != metas_[b].raw_len ? metas_[a].raw_len > metas_[b].raw_len
                                                      : metas_[a].written > metas_[b].written;
    });
    for (size_t i : order) {
        std::ifstream in(metas_[i].path);
        std::string header, body;
        View v;
        std::string error = "unreadable";
        bool ok = false;
        if (std::getline(in, header) && std::getline(in, body)) {
            try {
                ok = view_from_json(nlohmann::json::parse(body), v, error);
            } catch (const std::exception & e) {
                error = e.what();
            }
        }
        ok = ok && v.config == config && prefix_of(v.raw_tokens, v.raw_gen_begin, raw_tokens) &&
             prefix_of(v.drafter_ids, v.drafter_gen_begin, drafter_ids);
        if (!ok) {
            std::fprintf(stderr, "[pflash-unified] view file %s does not load (%s): deleted\n",
                         metas_[i].path.c_str(), error.c_str());
            remove_locked(i);
            return false;
        }
        out = std::move(v);
        return true;
    }
    return false;
}

void ViewDiskStore::forget_lineage(uint64_t lineage) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = metas_.size(); i-- > 0;) {
        if (metas_[i].lineage == lineage) remove_locked(i);
    }
}

std::vector<uint64_t> ViewDiskStore::lineages_continued_by(const std::vector<int32_t> & raw_tokens) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> out;
    if (!enabled_) return out;
    std::vector<size_t> lengths;
    for (const auto & m : metas_) {
        if (m.raw_len <= raw_tokens.size()) lengths.push_back(m.raw_len);
    }
    const auto hashes = prefix_hashes(raw_tokens, lengths);
    for (const auto & m : metas_) {
        if (m.raw_len <= raw_tokens.size() && hashes.at(m.raw_len) == m.raw_hash) out.push_back(m.lineage);
    }
    return out;
}

size_t ViewDiskStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metas_.size();
}

uint64_t ViewDiskStore::max_lineage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t out = 0;
    for (const auto & m : metas_) out = std::max(out, m.lineage);
    return out;
}

void ViewDiskStore::set_clock(std::function<uint64_t()> clock) {
    std::lock_guard<std::mutex> lock(mutex_);
    clock_ = std::move(clock);
}

uint64_t ViewDiskStore::now() const {
    return clock_ ? clock_() : now_ms();
}

void ViewDiskStore::enforce_cap_locked() {
    while (metas_.size() > cap_) {
        size_t oldest = 0;
        for (size_t i = 1; i < metas_.size(); ++i) {
            if (metas_[i].written < metas_[oldest].written) oldest = i;
        }
        remove_locked(oldest);
    }
}

void ViewDiskStore::remove_locked(size_t index) {
    std::error_code ec;
    std::filesystem::remove(metas_[index].path, ec);
    metas_.erase(metas_.begin() + (std::ptrdiff_t) index);
}

std::string sha1_hex(const std::string & bytes) {
    uint8_t digest[20];
    luce::common::sha1_hash(bytes.data(), bytes.size(), digest);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (uint8_t b : digest) {
        out += hex[b >> 4];
        out += hex[b & 0xF];
    }
    return out;
}

}  // namespace pflash_unified
