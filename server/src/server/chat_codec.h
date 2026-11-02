// Chat codecs: a model's official chat template, rendered to prompt tokens.
//
// A codec reproduces one official template exactly (the text it renders, and
// the token IDs the model's own tokenizer gives that text), so the prompt is
// the one the model was trained on. test/chat_golden holds the references.
//
// The template's control tokens (<|im_start|>, <|im_end|>, image markers) are
// inserted by ID. The text between two of them is tokenized as one run, as a
// Hugging Face tokenizer does once it has split on special tokens, with
// user-defined tokens (<think>, <tool_call>, ...) matched where they are
// spelled. The one deliberate difference: control-token text inside a message
// (and a codec's other turn markers, RenderedPrompt::literal) stays text, so a
// message cannot open or close a turn.

#pragma once

#include "chat_template.h"
#include "tool_parser.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace luce::common {

class Tokenizer;

// One piece of a rendered prompt: text, control tokens spelled out, or
// token IDs taken as they are (a remembered turn; `text` is their text).
struct PromptPiece {
    std::string text;
    bool control = false;
    std::vector<int32_t> ids;
};

// A byte range of RenderedPrompt::text(); `begin == npos` when there is none.
struct PromptSpan {
    size_t begin = std::string::npos;
    size_t end = std::string::npos;
    bool valid() const { return begin != std::string::npos && end >= begin; }
};

// Where one input message landed in the prompt. `turn` is the message's own
// part, its role envelope included where it has one (a message the template
// merges into a shared turn, like a tool result, is its own part of it);
// `content` is its text as the template rendered it.
struct MessageLayout {
    PromptSpan turn;
    PromptSpan content;
};

struct RenderedPrompt {
    std::vector<PromptPiece> pieces;
    // Added tokens that stay text inside the runs, like control tokens: the
    // codec's turn markers where the GGUF types them user-defined.
    std::vector<std::string> literal;
    // One per input message, and the tool definitions the template renders.
    // PFlash maps the prompt with these instead of rendering variants of it.
    std::vector<MessageLayout> messages;
    PromptSpan tools;
    // With a generation prompt, the turn-memory key of the turn generated
    // after this prompt (TurnMemory::Turn::context): the identity of the
    // prompt before that turn, its text and control tokens.
    std::string context;

    std::string text() const;
    // Control pieces as their token IDs, remembered turns as they were
    // generated, and the text between them as one run.
    std::vector<int32_t> tokens(const Tokenizer & tokenizer) const;
};

// Assistant turns as the model generated them, by the prompt they continued
// and what the client sees of them. A client that sends a turn back unchanged,
// after the same conversation, gets the exact generated tokens in the prompt,
// where the model's own spacing or JSON spelling may differ from the
// template's; any edit, to the turn or before it, is a miss, and the turn
// renders as sent. Keyed by the prompt before the turn, as a prefix cache is
// (text and control tokens apart, so a message that spells a turn marker is
// not that turn marker), so no other conversation can get a turn (or its
// reasoning) back. Turns of conversations with images are not remembered: the
// key holds an image's markers, not the image.
// Content and reasoning compare without their edge whitespace: Qwen's template
// trims them, and an API reply does not carry the model's edges (DeepSeek's
// "\n\n" around content), so an echo differs from the raw turn there.
// Only a cache: when the model kept the template's format, a hit and a miss
// give the same tokens.
class TurnMemory {
public:
    struct Turn {
        std::string context;          // RenderedPrompt::context of the prompt the turn followed
        std::string reasoning;        // stripped
        std::string text;             // the turn from the role line on, without the end of turn
        std::vector<int32_t> tokens;  // the same, as generated
        // The turn opens with its reasoning and the end of thinking (DeepSeek
        // V4.1 puts the opening <think> in the generation prompt): it can
        // stand only where the template shows the turn's thinking.
        bool thinking = false;
    };

    explicit TurnMemory(size_t max_bytes = 64 * 1024 * 1024) : max_bytes_(max_bytes) {}

    void remember(const ChatMessage & turn, Turn generated);
    // The remembered turn the client's `turn` is, after a prompt whose
    // identity is `context`: same content and calls, and the same
    // reasoning unless the client sent no reasoning field (Chat Completions
    // clients rarely send it back; an empty one is sent).
    bool lookup(const ChatMessage & turn, const std::string & context, Turn & out);
    bool disabled() const { return max_bytes_ == 0; }

private:
    static std::string key(const ChatMessage & turn, const std::string & context);
    void prune();

    struct Entry {
        Turn turn;
        std::list<std::string>::iterator lru;
    };
    std::mutex mu_;
    size_t max_bytes_;
    size_t bytes_ = 0;
    std::unordered_map<std::string, Entry> entries_;
    std::list<std::string> lru_;  // front: oldest
};

struct ChatCodecInput {
    const std::vector<ChatMessage> * messages = nullptr;
    // Tool definitions in the client's key order, any API's shape; null or
    // empty for none.
    const nlohmann::ordered_json * tools = nullptr;
    bool add_generation_prompt = true;
    bool enable_thinking = false;
    std::string reasoning_effort;  // Lucebox's normalized effort; empty: the template's default
    int preserve_thinking = -1;    // -1: the template's default
    // The text the server put in a message for each image (control tokens).
    std::string image_marker;
    // Generated turns to put back as they were; null for none.
    TurnMemory * memory = nullptr;
};

class ChatCodec {
public:
    virtual ~ChatCodec() = default;
    // Name and version, e.g. "qwen3.8/1". Part of the prompt-cache keys:
    // bump the version whenever the rendering changes.
    virtual const std::string & id() const = 0;
    // Between the text parts of a message whose content is an array.
    virtual const char * text_part_separator() const { return ""; }
    // How a tool call opens in the template's format: what a grammar that
    // forces a call (tool_choice "required") lets the reply start with.
    virtual std::vector<std::string> tool_call_openers() const = 0;
    // False with `error` set when the template rejects the conversation.
    virtual bool render(const ChatCodecInput & input, RenderedPrompt & out,
                        std::string & error) const = 0;

    // The reasoning controls of the model's own API (its vendor's and
    // vLLM's). They hold with a Jinja template too: the model's family
    // codec reads the request either way.
    //
    // The Lucebox tier ("low", "medium", "high", "x-high", "max") an API
    // effort word selects (lucebox_effort_tier by default).
    virtual std::string effort_tier(const std::string & word) const;
    // The effort the template renders for a tier.
    virtual std::string template_effort(const std::string & tier) const { return tier; }
    // The template's effort when thinking is on and the request names none
    // ("" = none).
    virtual std::string default_effort() const { return {}; }
    // Whether an integer in [1, 100] is an effort (the template's own scale).
    virtual bool integer_effort() const { return false; }
    // Whether chat_template_kwargs.drop_thinking (the encoder's option)
    // sets whether earlier thinking shows.
    virtual bool drop_thinking_kwarg() const { return false; }
    // The tiers the model's API takes, as /props lists them.
    virtual std::vector<std::string> effort_tiers() const {
        return {"low", "medium", "high", "x-high", "max"};
    }

    // The reverse of render() for what the model generates.
    //
    // The tool calls in `block` (generated text from the first call opener to
    // the end of the reply): arguments in the order the model wrote them,
    // typed by the declared tools' schemas. False when the block is not in the
    // template's format; callers then use the heuristic parse_tool_calls.
    virtual bool parse_tool_calls(const std::string & block, const nlohmann::json & tools,
                                  ToolParseResult & out) const = 0;
    // An assistant turn as the template renders it (the text after the role
    // line, without the end-of-turn token), read back into a message. When the
    // model kept the template's format, rendering `out` gives `turn` again.
    virtual bool parse_turn(const std::string & turn, const nlohmann::json & tools,
                            ChatMessage & out) const = 0;
    // The assistant turn a reply ended (`generated`, end of turn removed)
    // after the prompt `rendered` / `prompt_tokens`: the message a client
    // sends back for it, and the turn as generated (but `turn.context`: the
    // caller has the prompt's RenderedPrompt::context). False when the reply
    // does not read as a turn.
    virtual bool generated_turn(const std::string & rendered, const std::vector<int32_t> & prompt_tokens,
                                const std::vector<int32_t> & generated, const Tokenizer & tokenizer,
                                const nlohmann::json & tools, ChatMessage & message,
                                TurnMemory::Turn & turn) const = 0;
};

// Lucebox's five tiers (spec §4.2): "minimal" is low, "xhigh" x-high, and an
// unknown word high.
std::string lucebox_effort_tier(const std::string & word);

// The codec named `name` ("qwen3.8", "qwen3.6", "qwen3.5", "deepseek4.1", "deepseek4"), or null for an
// unknown name.
std::unique_ptr<ChatCodec> make_chat_codec(const std::string & name);
const std::vector<std::string> & chat_codec_names();
// The codec an architecture renders with when neither the model card nor a
// flag names one ("deepseek41": "deepseek4.1", "deepseek4": "deepseek4",
// "qwen4exp": "qwen3.8", "qwen35" and "qwen35moe": "qwen3.5"); empty for the
// others, which keep their built-in renderer.
std::string default_chat_codec_for_arch(const std::string & arch);

// Python's str.strip(): Jinja's |trim in the Hugging Face templates.
std::string python_strip(const std::string & text);

}  // namespace luce::common
