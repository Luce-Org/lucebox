// BPE tokenizer for dflash::common native server.
//
// Loads vocabulary (token strings) and merge rules from GGUF metadata,
// then provides encode (text → token IDs) and decode (token IDs → text).
//
// Pre-tokenization uses the Qwen3/3.5 regex pattern (GPT-4-style with
// Unicode Mark class support). This is a self-contained implementation
// with no regex library dependency.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    Tokenizer(const Tokenizer &) = delete;
    Tokenizer & operator=(const Tokenizer &) = delete;

    // Load vocabulary and merges from a GGUF file's metadata.
    // Returns false on failure (prints diagnostic to stderr).
    bool load_from_gguf(const char * model_path);

    // ─── Encode ──────────────────────────────────────────────────────
    // Tokenize a UTF-8 string into token IDs.
    std::vector<int32_t> encode(const std::string & text) const;

    // ─── Decode ──────────────────────────────────────────────────────
    // Convert a single token ID to its text representation.
    // Returns empty string for out-of-range IDs.
    std::string token_text(int32_t id) const;

    // Return the raw BPE-encoded token string (as stored in the GGUF vocab).
    // Useful for checking special tokens without GPT-2 decode overhead.
    const std::string & raw_token(int32_t id) const;

    // Convert a sequence of token IDs to text.
    std::string decode(const std::vector<int32_t> & ids) const;

    // ─── Special tokens ──────────────────────────────────────────────
    int32_t eos_id() const { return eos_id_; }
    int32_t eos_chat_id() const { return eos_chat_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t vocab_size() const { return (int32_t)id_to_token_.size(); }

    // Look up a token by its exact string. Returns -1 if not found.
    int32_t token_to_id(const std::string & token) const;

private:
    // Pre-tokenize text into pieces using Qwen3/3.5 regex pattern.
    std::vector<std::string> pre_tokenize(const std::string & text) const;

    // Apply BPE merges to a single pre-tokenized piece.
    std::vector<int32_t> bpe_encode_piece(const std::string & piece) const;

    // Vocabulary: id → token string
    std::vector<std::string> id_to_token_;

    // Reverse map: token string → id
    std::unordered_map<std::string, int32_t> token_to_id_;

    // BPE merge ranks: "A B" → rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    // Added special tokens (e.g. <think>, </think>) — sorted longest-first
    // for greedy matching during encode.
    std::vector<std::pair<std::string, int32_t>> added_tokens_;

    // Special token IDs
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t eos_chat_id_ = -1;  // <|im_end|> for Qwen3

    // Pre-tokenizer type
    enum class PreTokenizer { QWEN2, QWEN35 };
    PreTokenizer pre_type_ = PreTokenizer::QWEN35;

    // Decode mode: SentencePiece tokens use UTF-8 with ▁ for space;
    // GPT-2/BPE tokens use byte-level Unicode encoding.
    bool is_sentencepiece_ = false;
};

}  // namespace dflash::common
