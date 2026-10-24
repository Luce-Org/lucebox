#pragma once

// The unified view: one view/recall mechanism for chat and agent sessions
// (server/README.md, PFlash).
// The pure parts: the tool list, line-structure detection, region layout,
// marker and label text, greedy admission of units under a cap, re-read
// matching and the view store. Nothing here knows a harness's output format,
// and nothing here tokenises: callers pass byte offsets of the rendered prompt
// text and token costs.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace pflash_unified {

// ── Tool list (PFLASH_REGION_TOOLS) ───────────────────────────────────

// Which tool outputs may become regions: tools the agent can call again before the next user question (recall
// only runs on a user question). Names only; ``all`` for "*".
struct ToolList {
    bool all = false;
    std::vector<std::string> names;
    std::string identity;      // canonical text (part of the config identity)
    bool listed(const std::string & name) const;
};

// ``spec``: comma-separated tool names, or "*". Empty = none. False with ``error`` on an empty or malformed entry
// (the server refuses to start).
bool parse_tool_list(const std::string & spec, ToolList & out, std::string & error);

// "NAME ARGS" for a recall label: the arguments as compact ASCII JSON (``arguments`` may be an object or a
// JSON-encoded string), cut at ``max_args`` bytes with "…".
std::string call_brief(const std::string & name, const nlohmann::json & arguments, size_t max_args = 120);

// ── Line structure ─────────────────────────────────────────────────────

// The lines of text[begin, end): plain ('\n' separates; a CR stays in its line) or escaped (inside a JSON string:
// the two bytes "\n" separate, other escapes are skipped whole). ``starts[i]`` is where line i begins, ``ends[i]``
// where its separator begins (``end`` for the last line). A separator at ``end`` opens no empty last line.
struct Lines {
    bool escaped = false;
    size_t begin = 0;
    size_t end = 0;
    std::vector<size_t> starts;
    std::vector<size_t> ends;
    size_t count() const { return starts.size(); }
    size_t sep_size() const { return escaped ? 2 : 1; }
    const char * sep() const { return escaped ? "\\n" : "\n"; }
};
Lines split_lines(const std::string & text, size_t begin, size_t end, bool escaped);

// The raw byte ranges (between the quotes) of the string values of text[begin, end) when it is exactly one JSON
// value, surrounding whitespace allowed; keys are not values. Empty when it is not JSON.
std::vector<std::pair<size_t, size_t>> json_string_values(const std::string & text, size_t begin, size_t end);

// The structure to page: the plain lines of the content, or the escaped lines of one of its JSON string
// values, whichever has the most lines; ties go to plain.
Lines detect_lines(const std::string & text, size_t begin, size_t end);

// ── Region layout ───────────────────────────────────────────────────────

// A region's lines split into a pinned head (at least its first two lines), a pinned tail (at least its last two
// lines) and units between cuts. Every cut is a line_cut(): a position that starts a token in both tokenisations
// (``common``), at the start of the line it opens or of the separator before it.
struct Layout {
    size_t head_end = 0;                 // pinned head [content begin, head_end)
    size_t tail_begin = 0;               // pinned tail [tail_begin, content end)
    std::vector<size_t> cuts;            // cuts.front() == head_end, cuts.back() == tail_begin
    std::vector<int> cut_line;           // the line each cut opens (cut_line.back(): the tail's first line)
    int units() const { return cuts.empty() ? 0 : (int) cuts.size() - 1; }
    int first_line(int unit) const { return cut_line[(size_t) unit]; }
    int line_count(int first_unit, int last_unit) const {
        return cut_line[(size_t) last_unit + 1] - cut_line[(size_t) first_unit];
    }
};

// The cut opening ``line`` (> 0), so that kept lines stay whole: the line's start when that is a common boundary,
// else the start of the separator before it when that is one and lies after the previous line's start; npos
// otherwise (the line opens no unit).
size_t line_cut(const Lines & lines, const std::vector<size_t> & common, int line);

// ``common``: ascending byte offsets that start a token in both tokenisations. ``proposed``: candidate cut offsets
// (the drafter's candidate starts: probe segments or fixed chunks); each opens the line it falls in and
// is kept when that line has a cut; a boundary at a separator opens the next line. The head ends at the first line
// >= 2 with a cut, the tail starts at the last line <= count - 2 with one. False with ``why`` when there are fewer
// than five lines, no head/tail cut, or no unit between them.
bool layout_region(const Lines & lines, const std::vector<size_t> & common,
                   const std::vector<size_t> & proposed, Layout & out, std::string & why);

// ── Text ────────────────────────────────────────────────────────────────

enum class RegionKind { UserContent, ToolOutput };

// ``text`` escaped for the inside of a JSON string.
std::string json_escape(const std::string & text);
// The decoded bytes of text[begin, end) read as the inside of a JSON string; malformed escapes stay as written.
std::string json_unescape(const std::string & text, size_t begin, size_t end);

std::string marker_text(int lines_not_shown);
std::string preamble_text(RegionKind kind, int shown_lines, int total_lines);
std::string recall_label_text(RegionKind kind, const std::string & brief);
inline const char * recall_open() { return "[Earlier in this conversation]\n"; }
inline const char * recall_close() { return "\n[End of earlier excerpts]\n\n"; }

// ``body`` as a line of its own at a cut in ``lines``' structure: a separator before it unless the text before
// ``at`` ends with one, a separator after it unless the text from ``next`` starts with one; JSON-escaped in
// escaped mode.
std::string inserted_line(const std::string & text, const Lines & lines, size_t at, size_t next,
                          const std::string & body);

// The served text of a paged region's content text[content_begin, content_end): the pinned head, the preamble,
// each unit raw when ``admitted`` and one marker per run of dropped units (counting their lines), the pinned tail.
// The server serves the same pieces as token slices; this is their text, for parity checks and tests.
std::string render_region_text(const std::string & text, size_t content_begin, size_t content_end,
                               const Lines & lines, const Layout & layout, const std::vector<char> & admitted,
                               RegionKind kind);

// ── Admission ──────────────────────────────────────────────────────────

struct Admission {
    bool ok = false;
    std::vector<char> admitted;          // per unit (locked units included)
    int cost = 0;                        // fixed + optional units + markers (or labels)
    int taken = 0;                       // optional units admitted
};

// Paging: ``locked`` units are kept whatever the cap (mandatory, served on top). The rest start dropped and
// ``fixed`` + Σ marker_cost(first, last) over the gaps is the starting cost. Units are taken in ``order`` (best
// first) while the cost after taking one -- its tokens plus the change in marker cost of the gap it splits -- stays
// within ``cap``. ``ok`` false when even the starting cost exceeds the cap.
Admission admit_paging(const std::vector<int> & unit_cost, const std::vector<char> & locked,
                       const std::vector<int> & order, int fixed, int cap,
                       const std::function<int(int, int)> & marker_cost);

// Recall: ``held`` units are never taken; the others in ``order`` while Σ unit tokens + Σ label_cost(first, last)
// over the runs of consecutive taken units stays within ``cap``.
Admission admit_recall(const std::vector<int> & unit_cost, const std::vector<char> & held,
                       const std::vector<int> & order, int cap,
                       const std::function<int(int, int)> & label_cost);

// PFLASH_RECALL_GATE: recall what the question needs. One region's candidates:
// per unit its token cost, held flag, score (mean lift: the ranking key) and head mass (its share of the head's
// mass on the whole prompt); the region's cap and the label cost of one excerpt run.
struct RecallRegion {
    std::vector<int> unit_cost;
    std::vector<char> held;
    std::vector<double> score;
    std::vector<double> mass;
    int cap = 0;
    int label_cost = 0;
};

struct GatedRecall {
    std::vector<Admission> regions;      // per region, as admit_recall's
    double deficit_before = 0.0;         // D / S before any unit: the head mass on unheld units over S
    double deficit_after = 0.0;          // D / S after the admitted units
    bool gated = false;                  // deficit_before < theta: nothing taken, the present context suffices
};

// ``context_mass`` = S (> 0): the head mass on every candidate that is not pinned (history, held and unheld units).
// D = the mass of the unheld units. When D / S < ``theta`` nothing is taken. Else unheld units are taken best first
// ACROSS regions (one order by score; region then document order on ties), each under its own region's cap with
// admit_recall's run and label accounting (a unit that does not fit is skipped), and the taking stops as soon as the
// mass still unheld is below theta x S. The caps bound it: the worst case is admit_recall's fill.
GatedRecall admit_recall_gated(const std::vector<RecallRegion> & regions, double context_mass, double theta);

// Units best first: higher score first, document order on ties; ``skip`` units are left out.
std::vector<int> rank_units(const std::vector<double> & score, const std::vector<char> & skip);

// ── Re-reads ──────────────────────────────────────────────────────────

// Line fingerprints of a structure (CR stripped; blank lines 0).
std::vector<uint64_t> line_hashes(const std::string & text, const Lines & lines);

// Lines of ``region`` matched by ``candidate``. Informative lines: non-blank and occurring at most ``max_repeat``
// times in the region (blank and over-frequent lines such as "}" are skipped on both sides, never breaking a run).
// A match is a run of >= ``min_run`` consecutive informative candidate lines equal to consecutive informative
// region lines. A skipped region line counts as covered when the informative lines on both sides of it are.
struct RereadMatch {
    std::vector<char> region_covered;    // per region line
    std::vector<char> region_informative;  // per region line
    int matched_candidate = 0;           // informative candidate lines inside a match
    int candidate_informative = 0;
    bool is_reread() const { return candidate_informative > 0 && 2 * matched_candidate >= candidate_informative; }
};
RereadMatch match_reread(const std::vector<uint64_t> & candidate, const std::vector<uint64_t> & region,
                         int min_run = 3, int max_repeat = 32);

// Whether lines [begin, end) are covered as a whole: every informative line is (blank and over-frequent lines at
// the edges do not count), or, when none is informative, every line is.
bool lines_covered(const std::vector<char> & covered, const std::vector<char> & informative, int begin, int end);

// ── View ───────────────────────────────────────────────────────────────

struct Region {
    RegionKind kind = RegionKind::UserContent;
    std::string brief;                   // call_brief() of a tool output ("" for user content)
    std::string tool_call_id;
    std::string content_sha;             // hex digest of the raw content bytes
    size_t cb = 0, ce = 0;               // content bytes in the rendered prompt text
    int tb = 0, te = 0;                  // target tokens of the content
    int db = 0, de = 0;                  // drafter tokens of the content
    Lines lines;
    Layout layout;
    std::vector<uint64_t> hashes;        // line_hashes(lines)
    std::vector<int> unit_tb, unit_te;   // per unit, target tokens
    std::vector<int> unit_db, unit_de;   // per unit, drafter tokens
    std::vector<char> mandatory;         // per unit
    std::vector<char> kept;              // served when paged
    std::vector<char> held;              // kept, recalled or re-read
    std::vector<char> reread;            // per line, covered by a re-read
    int cap = 0;                         // optional served tokens at paging, before stage-2 widening (target)
    int droppable = 0;                   // target tokens of the optional units
    int recalled_tokens = 0;             // cumulative appended by recall (target)
};

struct Piece {
    int begin = 0;                       // served target tokens [begin, end)
    int end = 0;
    std::string kind;                    // raw, paged_text, preamble, marker, recall_wrapper, recall_label,
                                         // recall_text, chat_view, generation
    int region = -1;
    int turn = -1;
};

struct View {
    uint64_t lineage = 0;
    int version = 0;
    std::string config;
    std::vector<int32_t> raw_tokens;     // target, raw prompt
    int raw_gen_begin = -1;
    std::vector<int32_t> drafter_ids;
    int drafter_gen_begin = -1;
    std::vector<int32_t> view_tokens;    // target, served
    int view_gen_begin = -1;
    std::vector<Region> regions;
    std::vector<Piece> pieces;
    int turns = 0;
    size_t bytes() const;
};

// Views by raw prefix. ``find``: the longest stored view whose raw tokens up to its generation prompt prefix this
// request's (target and drafter alike) and whose config identity matches; newest on ties; a copy. ``remember``
// adds a view and evicts the least recently used views over ``capacity`` or ``byte_cap``, never the newest one;
// the view it continued stays, so a fork or an edited request continues it.
class ViewStore {
public:
    ViewStore(size_t capacity = 16, size_t byte_cap = (size_t) 256 << 20)
        : capacity_(capacity), byte_cap_(byte_cap) {}
    bool find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
              const std::string & config, View & out);
    // Returns the views it evicted (the server writes them to disk when the disk session cache is on).
    std::vector<View> remember(View view);
    // Every stored view, most recently used first (the shutdown flush).
    std::vector<View> all() const;
    // Drops every view of ``lineage`` (each turn's view is its own entry, so an ancestor would match next).
    void forget_lineage(uint64_t lineage);
    // The lineages of the views a request with these raw tokens could continue (raw prefix up to the generation
    // prompt, any config).
    std::vector<uint64_t> lineages_continued_by(const std::vector<int32_t> & raw_tokens) const;
    size_t size() const;
    size_t bytes() const;

private:
    struct Entry {
        View view;
        uint64_t used = 0;
    };
    mutable std::mutex mutex_;
    size_t capacity_;
    size_t byte_cap_;
    uint64_t clock_ = 0;
    std::vector<Entry> entries_;
};

// ── Views on disk (the SSD session cache) ─────────────────────────────

// A view as JSON, every field (its disk form), and back. ``view_from_json`` fails closed: false with ``error`` for
// anything that is not a complete view whose offsets and per-unit and per-line vectors agree.
nlohmann::json view_to_json(const View & view);
bool view_from_json(const nlohmann::json & j, View & view, std::string & error);

// The paging rules' version, part of a stored view's identity: a view paged under older rules is never served again.
// 2: a line over its region's cap hands the request to the chat view (it was dropped whole).
inline constexpr int kViewPagingVersion = 2;

// Views of shortened conversations kept on disk next to the KV cache, so a conversation survives a restart or the
// RAM store's eviction: written when a view leaves the RAM store and at shutdown, read on a RAM miss. One file per
// view: a one-line header (identity, config, prefix lengths and hashes) that ``init`` indexes, then the view. A view
// written under another ``identity`` (the disk cache's salt: model, template, context) is never read; any file older
// than ``ttl_seconds`` (0: never) is deleted at ``init``; over ``cap`` views the least recently written go.
class ViewDiskStore {
public:
    bool init(const std::string & dir, const std::string & identity, uint64_t ttl_seconds, size_t cap);
    bool enabled() const;
    // Writes ``view`` (a temporary file, then a rename), replacing the file of the same prefixes and config.
    bool save(const View & view);
    // ViewStore::find over the views on disk whose raw prefix is longer than ``longer_than`` tokens. The file is read
    // and both prefixes compared token by token before it is returned; a file that does not load is deleted.
    bool find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
              const std::string & config, size_t longer_than, View & out);
    void forget_lineage(uint64_t lineage);
    // ViewStore::lineages_continued_by over the views on disk, from their headers (no file is read).
    std::vector<uint64_t> lineages_continued_by(const std::vector<int32_t> & raw_tokens) const;
    size_t size() const;
    // The largest lineage on disk: a restarted server numbers its new conversations after it.
    uint64_t max_lineage() const;
    // Milliseconds since the epoch for write stamps and the TTL (tests inject one).
    void set_clock(std::function<uint64_t()> clock);

private:
    struct Meta {
        std::string path;
        uint64_t lineage = 0;
        std::string config;
        size_t raw_len = 0;
        size_t drafter_len = 0;
        uint64_t raw_hash = 0;
        uint64_t drafter_hash = 0;
        uint64_t written = 0;   // ms since the epoch
    };
    void enforce_cap_locked();
    void remove_locked(size_t index);
    mutable std::mutex mutex_;
    bool enabled_ = false;
    std::string dir_;
    std::string identity_;
    uint64_t ttl_ = 0;
    size_t cap_ = 0;
    uint64_t last_written_ = 0;   // writes are strictly ordered, also within one millisecond
    std::function<uint64_t()> clock_;
    uint64_t now() const;
    std::vector<Meta> metas_;
};

// Hex SHA-1 of ``bytes`` (region identity).
std::string sha1_hex(const std::string & bytes);

}  // namespace pflash_unified
