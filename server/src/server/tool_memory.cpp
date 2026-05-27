// Tool memory implementation.

#include "tool_memory.h"

#include <algorithm>

namespace dflash::common {

ToolMemory::ToolMemory(size_t max_entries, size_t max_bytes)
    : max_entries_(max_entries)
    , max_bytes_(max_bytes)
{}

void ToolMemory::remember(const std::vector<std::string> & call_ids,
                          const std::string & raw_text) {
    if (disabled() || raw_text.empty()) return;

    // Deduplicate call_ids
    std::vector<std::string> unique_ids;
    for (const auto & id : call_ids) {
        if (id.empty()) continue;
        bool found = false;
        for (const auto & u : unique_ids) {
            if (u == id) { found = true; break; }
        }
        if (!found) unique_ids.push_back(id);
    }
    if (unique_ids.empty()) return;

    // Find or create block
    auto block_it = blocks_.find(raw_text);
    if (block_it == blocks_.end()) {
        Block block;
        block.size_bytes = raw_text.size();  // accounts for the map key
        block.refs = 0;
        block_it = blocks_.emplace(raw_text, std::move(block)).first;
        total_bytes_ += block_it->second.size_bytes;
    }

    // Associate each call_id with this block
    for (const auto & call_id : unique_ids) {
        auto existing = by_id_.find(call_id);
        if (existing != by_id_.end()) {
            if (existing->second == raw_text) {
                touch(call_id);
                continue;
            }
            // Different block — drop old association
            drop_entry(call_id);
        }
        by_id_[call_id] = raw_text;
        total_bytes_ += raw_text.size();  // account for by_id_ value copy
        block_it->second.refs++;
        touch(call_id);
    }

    prune();
}

std::string ToolMemory::lookup(const std::vector<std::string> & call_ids) {
    std::string result_text;
    bool first = true;

    for (const auto & id : call_ids) {
        if (id.empty()) return "";
        auto it = by_id_.find(id);
        if (it == by_id_.end()) return "";

        if (first) {
            result_text = it->second;
            first = false;
        } else if (result_text != it->second) {
            return "";  // different blocks — can't replay
        }
        touch(id);
    }

    return result_text;
}

void ToolMemory::touch(const std::string & call_id) {
    auto it = lru_map_.find(call_id);
    if (it != lru_map_.end()) {
        lru_.erase(it->second);
    }
    lru_.push_back(call_id);
    lru_map_[call_id] = std::prev(lru_.end());
}

void ToolMemory::drop_entry(const std::string & call_id) {
    auto id_it = by_id_.find(call_id);
    if (id_it == by_id_.end()) return;

    const std::string & text_key = id_it->second;
    size_t id_value_bytes = text_key.size();  // by_id_ value copy size

    auto block_it = blocks_.find(text_key);
    if (block_it != blocks_.end()) {
        if (block_it->second.refs > 0) {
            block_it->second.refs--;
        }
        if (block_it->second.refs == 0) {
            // Subtract the block key size
            if (block_it->second.size_bytes > total_bytes_) {
                total_bytes_ = 0;
            } else {
                total_bytes_ -= block_it->second.size_bytes;
            }
            blocks_.erase(block_it);
        }
    }

    // Subtract the by_id_ value copy size
    if (id_value_bytes > total_bytes_) {
        total_bytes_ = 0;
    } else {
        total_bytes_ -= id_value_bytes;
    }

    by_id_.erase(id_it);

    auto lru_it = lru_map_.find(call_id);
    if (lru_it != lru_map_.end()) {
        lru_.erase(lru_it->second);
        lru_map_.erase(lru_it);
    }
}

void ToolMemory::prune() {
    while (!by_id_.empty() &&
           ((max_entries_ > 0 && by_id_.size() > max_entries_) ||
            (max_bytes_ > 0 && total_bytes_ > max_bytes_))) {
        if (lru_.empty()) break;
        std::string oldest = lru_.front();
        drop_entry(oldest);
    }
}

}  // namespace dflash::common
