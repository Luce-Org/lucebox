// BPE tokenizer for luce::common native server.
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

namespace luce::common {

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
    // Tokenize a UTF-8 string into token IDs. Added tokens spelled in the text
    // become their IDs, as Hugging Face tokenizers do. With match_control false,
    // control tokens (<|im_start|>, <|im_end|>, ...) stay ordinary text and only
    // user-defined ones (<think>, <tool_call>, ...) are matched: chat codecs
    // encode message text this way and insert the control tokens by ID.
    std::vector<int32_t> encode(const std::string & text, bool match_control = true) const;
    // As above, with the added tokens in `literal` kept as ordinary text too
    // (a codec's turn markers that the GGUF types user-defined).
    std::vector<int32_t> encode(const std::string & text, bool match_control,
                                const std::vector<int32_t> & literal) const;

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

    // True for an added token (GGUF token type control or user-defined:
    // chat and tool markup such as <|im_start|>, <tool_call>).
    bool is_added_token(int32_t id) const;

    // True for a control token (GGUF token type control).
    bool is_control_token(int32_t id) const;

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
    // The same, without control tokens.
    std::vector<std::pair<std::string, int32_t>> user_added_tokens_;
    std::vector<int32_t> encode_matching(
        const std::string & text,
        const std::vector<std::pair<std::string, int32_t>> & added) const;
    std::vector<int32_t> control_ids_;  // sorted

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

}  // namespace luce::common
