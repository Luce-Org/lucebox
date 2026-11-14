#include "server/kv/prefix_pins.h"

#include "server/kv/prefix_hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace luce::common {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

json pin_json(const SharedPrefixPin & p) {
    json files = json::array();
    for (const auto & f : p.files) {
        files.push_back({{"layout", f.layout}, {"disk_tokens", f.disk_tokens},
                         {"path", f.path}, {"bytes", f.bytes}, {"last_hit", f.last_hit}});
    }
    return {{"id", p.id}, {"tokens", p.tokens}, {"source", p.source},
            {"client", p.client}, {"hits", p.hits}, {"created", p.created},
            {"last_hit", p.last_hit}, {"files", files}};
}

// Write `text` to `path` durably: a temporary file, fsync, rename, fsync of
// the directory, so a crash leaves the old catalog or the new one.
bool write_durably(const std::string & path, const std::string & text) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    // One temporary per process: servers of one token identity on a directory share the catalog, and the last
    // whole write wins.
#if defined(_WIN32)
    const std::string tmp = path + ".tmp";
#else
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
#endif
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size() &&
              std::fflush(f) == 0;
#if !defined(_WIN32)
    ok = ok && ::fsync(::fileno(f)) == 0;
#endif
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec) return false;
#if !defined(_WIN32)
    const int dir = ::open(fs::path(path).parent_path().c_str(), O_RDONLY | O_CLOEXEC);
    if (dir >= 0) {
        ::fsync(dir);
        ::close(dir);
    }
#endif
    return true;
}

}  // namespace

const SharedPrefixPin::File * SharedPrefixPin::file_for(const std::string & layout) const {
    for (const auto & f : files) {
        if (f.layout == layout) return &f;
    }
    return nullptr;
}

uint64_t SharedPrefixPin::bytes() const {
    uint64_t sum = 0;
    for (const auto & f : files) sum += f.bytes;
    return sum;
}

std::string PrefixPins::id_of(const std::vector<int32_t> & tokens) {
    const PrefixHash hash = hash_prefix(tokens.data(), (int)tokens.size());
    return hex_string(hash.data(), 16);
}

std::shared_ptr<PrefixPins> PrefixPins::open(const std::string & path, Limits limits,
                                             std::function<uint64_t()> clock) {
    if (path.empty()) {
        return std::shared_ptr<PrefixPins>(new PrefixPins(path, limits, std::move(clock)));
    }
    static std::mutex registry_mu;
    static std::map<std::string, std::weak_ptr<PrefixPins>> registry;
    std::error_code ec;
    std::string key = fs::weakly_canonical(fs::absolute(path, ec), ec).string();
    if (key.empty()) key = path;
    std::lock_guard<std::mutex> registry_lock(registry_mu);
    if (auto open = registry[key].lock()) return open;
    std::shared_ptr<PrefixPins> pins(new PrefixPins(path, limits, std::move(clock)));
    {
        std::lock_guard<std::mutex> lock(pins->mu_);
        pins->load_locked();
    }
    registry[key] = pins;
    return pins;
}

PrefixPins::PrefixPins(std::string path, Limits limits, std::function<uint64_t()> clock)
    : path_(std::move(path)), limits_(limits), clock_(std::move(clock)) {}

uint64_t PrefixPins::now() const {
    return clock_ ? clock_() : (uint64_t)std::time(nullptr);
}

void PrefixPins::mark_dirty_locked() {
    if (!dirty_) dirty_since_ = now();
    dirty_ = true;
}

void PrefixPins::load_locked() {
    pins_.clear();
    ring_.clear();
    if (path_.empty()) return;
    std::ifstream in(path_);
    if (!in) return;
    try {
        const json doc = json::parse(in);
        if (doc.contains("ring")) ring_ = doc["ring"].get<std::vector<std::vector<int32_t>>>();
        for (const auto & j : doc.value("pins", json::array())) {
            // A bad entry is skipped, not the catalog.
            try {
                SharedPrefixPin p;
                p.tokens = j.value("tokens", std::vector<int32_t>{});
                if (p.tokens.empty()) continue;
                p.id = id_of(p.tokens);
                p.source = j.value("source", "learned");
                p.client = j.value("client", "");
                p.hits = j.value("hits", (uint64_t)0);
                p.created = j.value("created", (uint64_t)0);
                p.last_hit = j.value("last_hit", (uint64_t)0);
                for (const auto & jf : j.value("files", json::array())) {
                    SharedPrefixPin::File f;
                    f.layout = jf.value("layout", "");
                    f.disk_tokens = std::min<int>(jf.value("disk_tokens", 0), (int)p.tokens.size());
                    f.path = jf.value("path", "");
                    f.bytes = jf.value("bytes", (uint64_t)0);
                    f.last_hit = jf.value("last_hit", (uint64_t)0);
                    if (!f.layout.empty() && f.disk_tokens > 0) p.files.push_back(std::move(f));
                }
                pins_.push_back(std::move(p));
            } catch (const std::exception & e) {
                std::fprintf(stderr, "[shared-prefix] %s: skipping a bad pin: %s\n",
                             path_.c_str(), e.what());
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[shared-prefix] ignoring %s: %s\n", path_.c_str(), e.what());
        pins_.clear();
        ring_.clear();
    }
    dirty_ = false;
}

std::vector<std::vector<int32_t>> PrefixPins::ring() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ring_;
}

void PrefixPins::set_ring(std::vector<std::vector<int32_t>> ring) {
    std::lock_guard<std::mutex> lock(mu_);
    if (ring == ring_) return;
    ring_ = std::move(ring);
    mark_dirty_locked();
}

bool PrefixPins::save(bool force) {
    std::lock_guard<std::mutex> saving(save_mu_);
    std::string text;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (path_.empty() || (!dirty_ && !force)) return true;
        json doc;
        doc["version"] = 2;
        doc["ring"] = ring_;
        doc["pins"] = json::array();
        for (const auto & p : pins_) doc["pins"].push_back(pin_json(p));
        text = doc.dump(-1, ' ', false, json::error_handler_t::replace);
        dirty_ = false;
    }
    if (!write_durably(path_, text)) {
        std::fprintf(stderr, "[shared-prefix] cannot write %s\n", path_.c_str());
        std::lock_guard<std::mutex> lock(mu_);
        mark_dirty_locked();
        return false;
    }
    return true;
}

bool PrefixPins::due(uint64_t seconds) const {
    std::lock_guard<std::mutex> lock(mu_);
    return dirty_ && now() >= dirty_since_ + seconds;
}

std::vector<SharedPrefixPin> PrefixPins::enforce_caps_locked() {
    std::vector<SharedPrefixPin> dropped;
    // Files per pin: the least recently hit layouts go first.
    for (auto & p : pins_) {
        if (p.files.size() <= limits_.max_files) continue;
        std::sort(p.files.begin(), p.files.end(),
                  [](const SharedPrefixPin::File & a, const SharedPrefixPin::File & b) {
                      return a.last_hit > b.last_hit;
                  });
        SharedPrefixPin gone;
        gone.id = p.id;
        gone.tokens = p.tokens;
        gone.files.assign(p.files.begin() + (long)limits_.max_files, p.files.end());
        p.files.resize(limits_.max_files);
        dropped.push_back(std::move(gone));
    }
    const auto total_bytes = [&] {
        uint64_t sum = 0;
        for (const auto & p : pins_) sum += p.bytes();
        return sum;
    };
    while (!pins_.empty() &&
           (pins_.size() > limits_.max_pins || total_bytes() > limits_.max_bytes)) {
        auto lru = std::min_element(pins_.begin(), pins_.end(),
            [](const SharedPrefixPin & a, const SharedPrefixPin & b) {
                return a.last_hit < b.last_hit;
            });
        dropped.push_back(std::move(*lru));
        pins_.erase(lru);
    }
    if (!dropped.empty()) mark_dirty_locked();
    keep_unreferenced_locked(dropped);
    return dropped;
}

void PrefixPins::keep_unreferenced_locked(std::vector<SharedPrefixPin> & dropped) const {
    const auto referenced = [this](const std::string & path) {
        for (const auto & p : pins_) {
            for (const auto & f : p.files) {
                if (f.path == path) return true;
            }
        }
        return false;
    };
    for (auto & d : dropped) {
        d.files.erase(std::remove_if(d.files.begin(), d.files.end(),
                                     [&](const SharedPrefixPin::File & f) {
                                         return referenced(f.path);
                                     }),
                      d.files.end());
    }
}

void PrefixPins::merge_files_locked(SharedPrefixPin & into,
                                    std::vector<SharedPrefixPin::File> files,
                                    std::vector<SharedPrefixPin> & dropped) {
    const uint64_t t = now();
    for (auto & f : files) {
        if (!f.last_hit) f.last_hit = t;
        auto it = std::find_if(into.files.begin(), into.files.end(),
            [&](const SharedPrefixPin::File & g) { return g.layout == f.layout; });
        if (it == into.files.end()) {
            into.files.push_back(std::move(f));
            continue;
        }
        if (it->path != f.path) {
            SharedPrefixPin replaced;
            replaced.id = into.id;
            replaced.tokens = into.tokens;
            replaced.files.push_back(*it);
            dropped.push_back(std::move(replaced));
        }
        *it = std::move(f);
    }
}

std::optional<std::vector<SharedPrefixPin>> PrefixPins::attach(const std::string & id,
                                                               SharedPrefixPin::File file) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto & p : pins_) {
        if (p.id != id) continue;
        std::vector<SharedPrefixPin> dropped;
        merge_files_locked(p, {std::move(file)}, dropped);
        mark_dirty_locked();
        auto capped = enforce_caps_locked();
        keep_unreferenced_locked(dropped);
        dropped.insert(dropped.end(), std::make_move_iterator(capped.begin()),
                       std::make_move_iterator(capped.end()));
        return dropped;
    }
    return std::nullopt;
}

std::vector<SharedPrefixPin> PrefixPins::add(SharedPrefixPin pin) {
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t t = now();
    if (pin.id.empty()) pin.id = id_of(pin.tokens);
    if (!pin.created) pin.created = t;
    if (!pin.last_hit) pin.last_hit = t;
    for (auto & f : pin.files) {
        if (!f.last_hit) f.last_hit = t;
    }
    mark_dirty_locked();
    std::vector<SharedPrefixPin> dropped;
    auto existing = std::find_if(pins_.begin(), pins_.end(),
        [&](const SharedPrefixPin & p) { return p.id == pin.id; });
    if (existing != pins_.end()) {
        existing->hits = std::max(pin.hits, existing->hits);
        existing->last_hit = std::max(pin.last_hit, existing->last_hit);
        if (existing->client.empty()) existing->client = pin.client;
        merge_files_locked(*existing, std::move(pin.files), dropped);
    } else {
        pins_.push_back(std::move(pin));
    }
    auto capped = enforce_caps_locked();
    keep_unreferenced_locked(dropped);
    dropped.insert(dropped.end(), std::make_move_iterator(capped.begin()),
                   std::make_move_iterator(capped.end()));
    return dropped;
}

bool PrefixPins::has(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & p : pins_) {
        if (p.id == id) return true;
    }
    return false;
}

std::optional<SharedPrefixPin> PrefixPins::get(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & p : pins_) {
        if (p.id == id) return p;
    }
    return std::nullopt;
}

std::optional<SharedPrefixPin> PrefixPins::find_by_disk_key(
        const std::vector<int32_t> & key, const std::string & layout) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & p : pins_) {
        const auto * f = p.file_for(layout);
        if (f && f->disk_tokens == (int)key.size() &&
            std::equal(key.begin(), key.end(), p.tokens.begin())) {
            return p;
        }
    }
    return std::nullopt;
}

void PrefixPins::hit(const std::string & id, const std::string & layout) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto & p : pins_) {
        if (p.id != id) continue;
        ++p.hits;
        p.last_hit = now();
        for (auto & f : p.files) {
            if (f.layout == layout) f.last_hit = p.last_hit;
        }
        mark_dirty_locked();
        return;
    }
}

std::optional<SharedPrefixPin> PrefixPins::remove(const std::string & id) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = pins_.begin(); it != pins_.end(); ++it) {
        if (it->id != id) continue;
        std::vector<SharedPrefixPin> out;
        out.push_back(std::move(*it));
        pins_.erase(it);
        mark_dirty_locked();
        keep_unreferenced_locked(out);
        return std::move(out.front());
    }
    return std::nullopt;
}

std::vector<SharedPrefixPin> PrefixPins::expire() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SharedPrefixPin> dropped;
    const uint64_t t = now();
    for (auto it = pins_.begin(); it != pins_.end();) {
        if (limits_.unpin_after_s > 0 && t > it->last_hit &&
            t - it->last_hit > limits_.unpin_after_s) {
            dropped.push_back(std::move(*it));
            it = pins_.erase(it);
        } else {
            ++it;
        }
    }
    if (!dropped.empty()) mark_dirty_locked();
    keep_unreferenced_locked(dropped);
    return dropped;
}

std::vector<SharedPrefixPin> PrefixPins::list() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SharedPrefixPin> out = pins_;
    std::sort(out.begin(), out.end(), [](const SharedPrefixPin & a, const SharedPrefixPin & b) {
        return a.last_hit > b.last_hit;
    });
    return out;
}

size_t PrefixPins::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pins_.size();
}

}  // namespace luce::common
