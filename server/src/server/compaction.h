#pragma once

#include "chat_template.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace dflash::common {

class Tokenizer;
struct ModelBackend;
struct ServerConfig;

struct CompactionResult {
    bool                     applied = false;
    std::vector<ChatMessage> compacted_messages;
    int                      tokens_saved = 0;
    int                      pre_compaction_tokens = 0;
};

CompactionResult edit_compact(
    const std::vector<ChatMessage> & messages,
    int keep_tool_uses,
    bool strip_thinking);

CompactionResult summarize_compact(
    const std::vector<ChatMessage> & messages,
    float keep_recent_ratio,
    int max_summary_tokens,
    const std::string & compaction_prompt,
    void * backend_ptr,
    void * tokenizer_ptr,
    int chat_format);

CompactionResult hard_truncate(
    const std::vector<ChatMessage> & messages,
    int max_messages_to_keep);

std::string strip_thinking_blocks(const std::string & text);
bool is_tool_result(const ChatMessage & msg);
std::string extract_file_read_path(const ChatMessage & msg);

}  // namespace dflash::common
