// Tool memory — exact text replay for assistant tool-calling turns.
//
// Ported from tool_memory.py.
// When the model generates tool calls, we remember the exact raw text.
// On subsequent turns where the client sends those tool_call IDs back,
// we replay the original text instead of re-rendering from JSON,
// preserving tokenization and KV cache alignment.

#pragma once

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

class ToolMemory {
public:
    explicit ToolMemory(size_t max_entries = 50000,
                        size_t max_bytes = 64 * 1024 * 1024);

    // Remember the raw text associated with a set of tool call IDs.
    void remember(const std::vector<std::string> & call_ids,
                  const std::string & raw_text);

    // Look up the raw text for a set of tool call IDs.
    // Returns empty string if not all IDs map to the same block.
    std::string lookup(const std::vector<std::string> & call_ids);

    bool disabled() const { return max_entries_ == 0 || max_bytes_ == 0; }

    // Snapshot for /props. Two successive reads under the same thread,
    // matching the Python implementation's "may tear by one entry" semantics.
    struct Stats {
        size_t max_entries;
        size_t max_bytes;
        size_t current_entries;
        size_t current_bytes;
    };
    Stats stats() const {
        return {max_entries_, max_bytes_, by_id_.size(), total_bytes_};
    }

private:
    struct Block {
        size_t      size_bytes;
        size_t      refs = 0;
    };

    void touch(const std::string & call_id);
    void prune();
    void drop_entry(const std::string & call_id);

    size_t max_entries_;
    size_t max_bytes_;
    size_t total_bytes_ = 0;

    // Block storage: keyed by raw_text content
    std::unordered_map<std::string, Block> blocks_;
    // call_id → raw_text (the key into blocks_)
    std::unordered_map<std::string, std::string> by_id_;
    // LRU order: front = oldest, back = newest
    std::list<std::string> lru_;
    // call_id → iterator in lru_ for O(1) move-to-back
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;
};

}  // namespace dflash::common
