# LLM Automatic Context Compaction: Research & Implementation Guide

## Executive Summary

LLM Auto Context Compaction is the industry's response to **Context Bloat** and **Context Rot** — the accumulation of massive intermediate tokens (tool outputs, reasoning chains, redundant history) that exhausts model context limits, drives up costs quadratically, increases latency, and degrades reasoning quality. This report analyzes how context compaction works across OpenAI, Anthropic, academic research, and major frameworks, then provides a detailed **implementation blueprint for the lucebox-hub inference server** — a C++/CUDA inference engine that currently has **no compaction** and simply returns HTTP 400 when context overflows[^1].

The key finding: **no major inference engine implements server-side self-summarization today**[^2]. All existing compaction happens either client-side (frameworks like LangChain, Inspect AI) or at the API provider level (OpenAI Responses API, Anthropic Beta). Implementing compaction directly in lucebox-hub would make it the **first open-source inference server with native self-summarization**.

---

## Table of Contents

1. [Architectural Taxonomy](#1-architectural-taxonomy)
2. [OpenAI's Implementation (Responses API)](#2-openais-implementation-responses-api)
3. [Anthropic's Implementation](#3-anthropics-implementation)
4. [Academic Advances](#4-academic-advances)
5. [Framework Implementations](#5-framework-implementations)
6. [Inference Engine Landscape](#6-inference-engine-landscape)
7. [lucebox-hub: Current State](#7-lucebox-hub-current-state)
8. [Implementation Blueprint for lucebox-hub](#8-implementation-blueprint-for-lucebox-hub)
9. [Comparative Trade-offs](#9-comparative-trade-offs)
10. [Architectural Recommendations](#10-architectural-recommendations)

---

## 1. Architectural Taxonomy

Modern context compaction splits into three core methodologies:

```
[Full Conversation Window (Messages, Tools, Code, Errors)]
                       │
                       ▼
          (Crosses Threshold: e.g., 90% of Window)
         ┌─────────────┴──────────────┐
         ▼                            ▼
┌─────────────────┐     ┌────────────────────────────────┐
│ Pruning / Edit  │     │     Semantic Compaction         │
│   Compaction    │     └───────────────┬────────────────┘
└────────┬────────┘                     │
         │             ┌────────────────┴───────────────┐
         │             ▼                                ▼
         │    ┌──────────────┐              ┌──────────────┐
         │    │  Text-Based  │              │  Provider-   │
         │    │Summarization │              │ Native Embed │
         ▼    └──────┬───────┘              └──────┬───────┘
[Drop Tool Output,   │                             │
 Extended Thinking]  ▼                             ▼
         ┌────────────────────────────────────────────────────┐
         │  [Compacted Context: Small, High-Density Footprint]│
         └────────────────────────────────────────────────────┘
```

### A. Provider-Native Token-Level Compression

OpenAI and Anthropic both offer **server-side compaction** that delegates optimization to the infrastructure provider. The server monitors tokens in-stream, and when a threshold is crossed, executes an inline compaction pass producing an opaque encrypted state blob[^3][^4].

### B. Dynamic & Incremental Text Summarization

Frameworks like LangChain and Inspect AI use a secondary LLM call to periodically compress older conversation text into high-density summaries. The key pattern is **incremental construction**: `Existing Summary + New Chunk → Updated Summary`[^5][^6].

### C. Edit & Loss-Aware Pruning

Rather than rewriting text, edit compaction selectively alters prompt structure: stripping thinking blocks, truncating tool outputs, or using perplexity-guided token removal[^7][^8].

---

## 2. OpenAI's Implementation (Responses API)

### 2.1 In-Stream Server-Side Compaction

**Endpoint:** `POST /v1/responses`

Developers specify a `context_management` configuration block:

```json
{
  "model": "gpt-5.5",
  "context_management": [
    { "type": "compaction", "compact_threshold": 100000 }
  ],
  "input": [
    { "role": "system", "content": "You are an autonomous engineering agent." },
    { "role": "user", "content": "Analyze the log files and debug the memory leak." }
  ]
}
```

When token usage crosses the `compact_threshold`, the backend produces an encrypted `ResponseCompactionItem`[^9]:

```python
class ResponseCompactionItem(BaseModel):
    id: str                     # Unique ID
    encrypted_content: str      # Opaque compressed state
    type: Literal["compaction"] # Always "compaction"
    created_by: Optional[str]   # Actor identifier
```

**Response during compaction event:**

```json
{
  "id": "resp_92kL8xPqZ2a1",
  "output": [
    {
      "id": "item_comp_7x8y9z",
      "type": "compaction",
      "encrypted_content": "eyJhcmNoaXZl..."
    },
    {
      "id": "item_msg_1a2b3c",
      "type": "message",
      "role": "assistant",
      "content": "I have completed the log analysis..."
    }
  ],
  "usage": {
    "prompt_tokens": 125000,
    "completion_tokens": 450,
    "compacted_tokens_saved": 98000
  }
}
```

### 2.2 Standalone Compact Endpoint

**Endpoint:** `POST /v1/responses/compact`

For explicit, deterministic state minimization[^10]:

```python
# Python SDK
compacted = client.responses.compact(
    model="gpt-5",
    previous_response_id="resp_abc123",  # Or pass full input array
    service_tier="flex",                  # 50% discount for latency-insensitive
)
# Returns CompactedResponse with object="response.compaction"
```

**Response schema:**

```python
class CompactedResponse(BaseModel):
    id: str
    created_at: int
    object: Literal["response.compaction"]
    output: List[ResponseOutputItem]  # User messages + compaction item
    usage: ResponseUsage              # Token accounting
```

### 2.3 State Chaining Mechanics

**Stateless Array Chaining:**

```json
{
  "input": [
    { "type": "compaction", "encrypted_content": "..." },
    { "role": "user", "content": "Great, draft a hotfix for db.py." }
  ]
}
```

**Stateful ID Chaining:**

```json
{
  "model": "gpt-5.5",
  "previous_response_id": "resp_92kL8xPqZ2a1",
  "input": [{ "role": "user", "content": "Great, draft a hotfix." }]
}
```

### 2.4 SDK Details

**Python SDK signature:**

```python
client.responses.create(
    model="gpt-5.2-codex",
    input=conversation,
    store=False,
    context_management=[{"type": "compaction", "compact_threshold": 100000}],
)
```

**TypeScript/Node.js SDK:**

```typescript
const compactedResponse = await client.responses.compact({
  model: 'gpt-5.4',
  previous_response_id: 'resp_abc123',
  service_tier: 'flex',
});
```

### 2.5 Feature Timeline

| Date | SDK Version | Change |
|------|------------|--------|
| Dec 4, 2025 | ~2.9.x | `/responses/compact` endpoint introduced[^11] |
| Dec 10, 2025 | v2.10.0 | `model` parameter made required |
| Jan 9, 2026 | ~2.12.x | `completed_at` property added |
| May 13, 2026 | v2.37.0 | `service_tier` parameter added[^12] |

### 2.6 Key Notes

- `context_management` type currently only supports `"compaction"`
- `truncation: "auto"` is a simpler alternative (drops items from beginning, no summary)
- `service_tier: "flex"` gives 50% discount for latency-insensitive compaction
- The `encrypted_content` field is opaque — cannot be read or audited by humans
- ZDR (Zero Data Retention) compatible via encrypted compaction items

---

## 3. Anthropic's Implementation

### 3.1 Server-Side Compaction API (Beta, Jan 2026)

**Beta header:** `compact-2026-01-12`

```python
class BetaCompact20260112EditParam(TypedDict, total=False):
    type: Required[Literal["compact_20260112"]]
    instructions: Optional[str]           # Custom prompt (REPLACES default)
    pause_after_compaction: bool          # Return early with stop_reason:"compaction"
    trigger: Optional[BetaInputTokensTriggerParam]  # Default: 150,000 tokens (min 50K)
```

**Compaction block (round-tripped in subsequent requests):**

```python
class BetaCompactionBlockParam(TypedDict, total=False):
    type: Required[Literal["compaction"]]
    cache_control: Optional[BetaCacheControlEphemeralParam]  # Cacheable!
    content: Optional[str]           # Human-readable summary
    encrypted_content: Optional[str] # Opaque metadata
```

**Key differences from OpenAI:**

| Feature | Anthropic | OpenAI |
|---------|-----------|--------|
| Summary readable? | ✅ `content` field is human-readable | ❌ Fully opaque blob |
| Pause after compaction? | ✅ `stop_reason: "compaction"` | ❌ No |
| Custom summary prompt? | ✅ Completely replaces default | Limited |
| Use cheaper model? | ❌ Always same model | N/A |
| Cache on compaction block? | ✅ `cache_control` supported | N/A |

### 3.2 Context Editing (Fine-Grained Control)

Beta header: `context-management-2025-06-27`

| Strategy | What It Does |
|----------|-------------|
| `clear_tool_uses_20250919` | Clears oldest tool results, keeps last N (default 3) |
| `clear_thinking_20251015` | Manages `<thinking>` blocks per model defaults |

Key parameters:
- `keep: 3` — preserve most recent 3 tool interactions
- `clear_at_least` — minimum tokens to clear (avoids cache invalidation if not worthwhile)
- `exclude_tools` — tools whose results are never cleared

### 3.3 Claude Code's `/compact` Slash Command

Claude Code's default summary structure:

```
1. Task Overview (core request, success criteria)
2. Current State (completed work, files modified)
3. Important Discoveries (constraints, decisions, errors)
4. Next Steps (specific actions, blockers, priorities)
5. Context to Preserve (user preferences, domain details)
```

When context approaches limits, Claude Code **automatically compacts** using this structure[^14]. Persistent rules belong in `CLAUDE.md` because they're re-injected on every request.

### 3.4 Prompt Caching Interaction

- Add `cache_control` at end of system prompt to cache independently of compaction
- Compaction blocks can be cached too (add `cache_control` to the block)
- Tool result clearing **invalidates** cached prefixes → use `clear_at_least` threshold
- Keeping thinking blocks → preserves cache; clearing → invalidates at that point

---

## 4. Academic Advances

### 4.1 Active Context Compression / "Focus Framework" (Jan 2026)

**Paper:** "Active Context Compression: Autonomous Memory Management in LLM Agents"
**arXiv:** [2601.07190](https://arxiv.org/abs/2601.07190)[^15]

The agent **autonomously decides** when to consolidate learnings into a persistent "Knowledge Block" and prunes raw history — inspired by slime mold exploration strategies.

| Metric | Result |
|--------|--------|
| Token reduction | 22.7% average, up to **57%** |
| Accuracy preservation | 100% (60% → 60% on SWE-bench Lite) |
| Autonomous compressions/task | 6.0 average |

### 4.2 Semantic-Anchor Compression / SAC (ICLR 2026)

**Paper:** "Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors"
**arXiv:** [2510.08907](https://arxiv.org/abs/2510.08907)[^16]
**GitHub:** [lx-Meteors/SAC](https://github.com/lx-Meteors/SAC)

SAC selects **anchor tokens directly from the original context** (no learned compression tokens), modifies bidirectional attention so anchors aggregate surrounding KV representations:

```
Traditional: [COMP_1, COMP_2, ...COMP_N] ← trained autoencoder
SAC:         [token_A*, token_B*, ...token_K*] ← selected anchors with modified attention
```

Consistently outperforms existing methods across compression ratios (Llama-3.2-1B/3B, Llama-3.1-8B).

### 4.3 IC-Former (EMNLP 2024)

**Paper:** "In-Context Former: Lightning-fast Compressing Context for Large Language Model"
**arXiv:** [2406.13618](https://arxiv.org/abs/2406.13618)[^17]
**GitHub:** [wonderful9462/IC-Former](https://github.com/wonderful9462/IC-Former)

A lightweight **cross-attention encoder** (~630M, 9% of target LLM) with learnable "digest tokens" compresses context in **O(n) time**:

| Metric | Result |
|--------|--------|
| Speed improvement | **68–112× faster** than baseline |
| Performance preserved | >90% of downstream accuracy |
| FLOP reduction | 1/32 of self-attention baseline |

### 4.4 ACON Framework (Microsoft, Oct 2025)

**Paper:** "ACON: Optimizing Context Compression for Long-horizon LLM Agents"
**arXiv:** [2510.00615](https://arxiv.org/abs/2510.00615)[^18]
**GitHub:** [microsoft/acon](https://github.com/microsoft/acon)

Gradient-free pipeline: when compressed context causes task failure, an evaluator LLM analyzes what was lost and **refines the compression guideline in natural language**.

| Metric | Result |
|--------|--------|
| Memory reduction | 26–54% (peak tokens) |
| Accuracy preserved | >95% when distilled |
| Small-LM improvement | +46% for smaller agents |

### 4.5 SWE-Pruner (Bytedance, Jan 2026)

**arXiv:** [2601.16746](https://arxiv.org/abs/2601.16746)[^19]
**GitHub:** [Ayanami1314/swe-pruner](https://github.com/Ayanami1314/swe-pruner)

Task-aware pruning for coding agents using a **0.6B neural skimmer** (Qwen3-Reranker-0.6B):

| Metric | Result |
|--------|--------|
| Token reduction (SWE-Bench) | 23–54% while **improving** success rates |
| Compression (LongCodeQA) | **14.84× compression** with minimal impact |
| Cost savings | ~40% on Claude API tokens |
| Training F1 | 0.78 |

### 4.6 LLMLingua Series (Microsoft, 2023–2024)

**GitHub:** [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)

| Paper | Venue | Model | Key Metric |
|-------|-------|-------|-----------|
| LLMLingua | EMNLP 2023 | GPT-2 Small (124M) | Up to **20x compression** |
| LongLLMLingua | ACL 2024 | — | +21.4% RAG with 1/4 tokens |
| LLMLingua-2 | ACL 2024 | XLM-RoBERTa-L (561M) | **3–6x faster** than v1 |

### 4.7 Approach Taxonomy

```
┌─────────────────────────────────────────────────────────────┐
│            Context Compression Approach Taxonomy             │
├─────────────────┬───────────────────────────────────────────┤
│ TRAINING-BASED  │ SAC (anchor tokens + bidirectional attn)  │
│ Model Changes   │ IC-Former (cross-attn digest tokens)       │
│                 │ LLMLingua-2 (BERT classifier distilled)    │
├─────────────────┼───────────────────────────────────────────┤
│ GUIDELINE-BASED │ ACON (gradient-free NL guideline optim.)  │
│ No model change │ SWE-Pruner (goal → neural skimmer)        │
│                 │ LLMLingua v1 (perplexity scoring, no tune) │
├─────────────────┼───────────────────────────────────────────┤
│ AGENT-NATIVE    │ Focus/ACC (self-regulating agent tools)    │
│ Autonomous      │                                           │
├─────────────────┼───────────────────────────────────────────┤
│ KV-CACHE LEVEL  │ SnapKV (attention head eviction)           │
│ Inference Opt.  │ PFlash (speculative prefill — lucebox)    │
└─────────────────┴───────────────────────────────────────────┘
```

---

## 5. Framework Implementations

### 5.1 Inspect AI — Most Sophisticated (5 Strategies)

[UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)[^5]

```python
class CompactionStrategy(abc.ABC):
    def __init__(self, *, threshold: int | float = 0.9, memory: bool = True):
        # threshold: float [0,1] = % of context window; int = absolute token count

    @abc.abstractmethod
    async def compact(self, model, messages, tools):
        ...  # Returns (compacted_input, optional_summary)
```

**5 strategies:**

| Strategy | Approach | What's Preserved |
|----------|----------|-----------------|
| `CompactionAuto` | Try Native → fallback Summary | Depends on sub-strategy |
| `CompactionNative` | Provider API (OpenAI/Anthropic) | Opaque server-side |
| `CompactionSummary` | LLM summarization (incremental) | System + input + summary |
| `CompactionEdit` | Strip thinking + truncate tools | Structure + last N results |
| `CompactionTrim` | Keep fraction of messages | System + recent % |

**Key patterns:**
- Three-tier token counting (baseline reuse → delta → full count)
- Memory pre-warning at 90% of threshold
- Retry loop (up to 3 iterations if first pass insufficient)
- Incremental summarization (only summarize since last summary)

**Integration:**

```python
from inspect_ai.agent import react
from inspect_ai.model import CompactionAuto, CompactionSummary, CompactionEdit

react(tools=[bash(), text_editor()], compaction=CompactionAuto())
react(tools=[bash()], compaction=CompactionSummary(threshold=0.8))
react(tools=[bash()], compaction=CompactionEdit(keep_tool_uses=3))
```

### 5.2 OpenAI Agents SDK

[openai/openai-agents-python](https://github.com/openai/openai-agents-python)[^21]

```python
class OpenAIResponsesCompactionSession:
    DEFAULT_COMPACTION_THRESHOLD = 10  # candidate items (not tokens!)

    def __init__(self, underlying_session, model="gpt-4.1",
                 compaction_mode: Literal["previous_response_id", "input", "auto"] = "auto",
                 should_trigger_compaction: Callable | None = None):
        ...

    async def run_compaction(self):
        compacted = await self.client.responses.compact(**kwargs)
        await self._replace_underlying_session_items(compacted.output)  # atomic with rollback
```

Trigger: fires when ≥10 "candidate items" (assistant messages, tool calls, reasoning) exist.

### 5.3 LangChain v1 — Middleware Pattern

```python
class SummarizationMiddleware(AgentMiddleware):
    def __init__(self, model, trigger=("fraction", 0.8),
                 keep=("messages", 20), summary_prompt=DEFAULT_SUMMARY_PROMPT):
        ...
```

**Classic memory classes (deprecated since 0.3.1):**
- `ConversationSummaryMemory` — rolling summary every turn
- `ConversationTokenBufferMemory` — FIFO truncation with token limit
- `ConversationSummaryBufferMemory` — hybrid: recent raw + older summarized

### 5.4 AutoGen — Pure Truncation

Three context classes:
- `BufferedChatCompletionContext` — N-message sliding window
- `TokenLimitedChatCompletionContext` — drops from **middle** (unique!)
- `HeadAndTailChatCompletionContext` — preserve first + last

### 5.5 Semantic Kernel (C#)

```csharp
// Count-based truncation with hysteresis
new ChatHistoryTruncationReducer(targetCount: 10, thresholdCount: 5)

// LLM summarization with incremental detection
new ChatHistorySummarizationReducer(chatService, targetCount: 2, thresholdCount: 4)
```

### 5.6 Cross-Framework Comparison

| Framework | Approach | Threshold | Tool Handling | Native API |
|-----------|----------|-----------|---------------|------------|
| **Inspect AI** | 5 strategies | % of ctx or absolute | `keep_tool_uses=N` + placeholder | ✅ |
| **OpenAI Agents SDK** | Server-side opaque | ≥10 candidate items | Excluded from candidates | ✅ |
| **LangChain v1** | LLM summarization | Fraction/tokens/messages | Pairs kept together | ❌ |
| **AutoGen** | Pure truncation | Count or tokens | Function pair protection | ❌ |
| **Semantic Kernel** | Truncation or LLM summary | Count + hysteresis | Function pair protection | ❌ |

---

## 6. Inference Engine Landscape

### No Server Implements Self-Summarization

**Critical finding:** No major inference engine (vLLM, llama.cpp, SGLang, TGI) implements server-side self-summarization[^2].

| Engine | Strategy | Mechanism |
|--------|----------|-----------|
| **llama.cpp** | Ring-buffer shift (`--context-shift`) | Drop middle tokens, shift KV positions |
| **vLLM** | Preemption + full re-prefill | Free all KV blocks, recompute entirely |
| **SGLang** | Radix tree LRU eviction | Cross-request prefix sharing |
| **All others** | Hard rejection (HTTP 400) | No action taken |

### llama.cpp's Context-Shift

When enabled, keeps `n_keep` tokens from front (system prompt), discards `n_discard` from middle, shifts remaining positions[^23]:

```cpp
common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);
```

### The Proxy Pattern (agentguard)

[Roboter-Schlafen-Nicht/agentguard](https://github.com/Roboter-Schlafen-Nicht/agentguard)[^24] — 3-phase compaction:

1. **Rule-based truncation**: Stub old tool results, deduplicate file reads
2. **LLM summarization**: Call a separate small model (e.g., Qwen2.5-coder:3b)
3. **Hard cap**: Drop oldest atomic message groups until under budget

---

## 7. lucebox-hub: Current State

### 7.1 Architecture Overview

lucebox-hub is a **C++17/CUDA inference server** implementing DFlash speculative decoding + DDTree verification for 3–5× speedup on Qwen3.5/3.6-27B[^1]. It serves three API formats:

```
Clients (Claude Code, Codex, Open WebUI)
         │
         ▼
┌─────────────────────────────────────────┐
│        lucebox-hub HTTP Server           │
│  ┌───────────┬──────────┬────────────┐  │
│  │/v1/chat/  │/v1/      │/v1/        │  │
│  │completions│messages  │responses   │  │
│  └───────────┴──────────┴────────────┘  │
│         │ (client threads)               │
│         ▼                                │
│  ┌────────────────────────────────────┐  │
│  │      Single Worker Thread           │  │
│  │  [PFlash] → [Prefix Cache] → GPU   │  │
│  └────────────────────────────────────┘  │
│         │                                │
│  ┌──────┴───────────────────────────┐    │
│  │  Prefix Cache (2-tier LRU)        │    │
│  │  Disk Cache (.dkv files)          │    │
│  │  Tool Memory (LRU, 50K entries)   │    │
│  └───────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 7.2 Current Context Overflow Handling

**ZERO compaction exists.** The only overflow handling[^25]:

```cpp
// server/src/server/http_server.cpp:1027-1030
if ((int)req.prompt_tokens.size() + req.max_output > config_.max_ctx) {
    send_error(fd, 400, "prompt + max_tokens exceeds context window");
    return true;
}
```

### 7.3 Existing Infrastructure Relevant to Compaction

| Component | File | Relevance |
|-----------|------|-----------|
| PFlash speculative prefill | `flashprefill.h` | Existing "compression" (structural, not semantic) |
| Prefix Cache (2-tier LRU) | `prefix_cache.h` | Cache invalidation after compaction |
| Tool Memory | `tool_memory.h` | LRU of tool call text for replay |
| Thinking Budget | `http_server.h:64-88` | Precedent for token budget control |
| Token Counting | `/v1/messages/count_tokens` | Pre-flight token measurement |
| Chat Template Rendering | `chat_template.cpp` | Message → token string pipeline |

### 7.4 Threading Model

```
Main Thread: accept() loop → spawn client threads
Client Threads (detached): parse HTTP → route_request() → block on job.cv
Worker Thread (single): dequeue → [pflash] → [prefix cache] → generate() → stream back
```

**Critical constraint:** Only the worker thread calls `backend_.generate()`. Compaction must run in the worker thread[^26].

---

## 8. Implementation Blueprint for lucebox-hub

### 8.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Incoming Request                        │
└───────────────────────────┬─────────────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │    Token Count Check          │
              │    prompt_tokens + max_output │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Under threshold?             │
              │ YES → Normal Generation      │
              │ NO  ↓                        │
              └──────────────┬──────────────┘
                             ▼
     ┌───────────────────────────────────────────────┐
     │ Layer 1: Edit Compaction (CPU-only, <1ms)      │
     │ • Strip <think> blocks from old turns          │
     │ • Truncate old tool results to placeholder     │
     │ • Deduplicate repeated file reads              │
     └────────────────────────┬──────────────────────┘
                              │ Still over?
                              ▼
     ┌───────────────────────────────────────────────┐
     │ Layer 2: Self-Summarization (GPU, 5-30s)       │
     │ • Internal generate() pass with summary prompt │
     │ • Replace old turns with [CONTEXT SUMMARY]     │
     │ • Preserve system prompt + recent 30% turns    │
     └────────────────────────┬──────────────────────┘
                              │ Still over?
                              ▼
     ┌───────────────────────────────────────────────┐
     │ Layer 3: Hard Truncation (CPU, last resort)    │
     │ • Keep system + last N messages only           │
     │ • Drop everything else                         │
     └────────────────────────┬──────────────────────┘
                              │ Still over?
                              ▼
                    ┌──────────────────┐
                    │   HTTP 400 Error  │
                    └──────────────────┘
```

### 8.2 Configuration Additions

**`ServerConfig` additions (`http_server.h`):**

```cpp
// Context compaction configuration
bool        compaction_enabled       = false;
float       compaction_threshold     = 0.9f;    // trigger at 90% of max_ctx
int         compaction_max_tokens    = 2048;    // max tokens for summary output
float       compaction_keep_recent   = 0.3f;    // keep last 30% of turns verbatim
bool        compaction_strip_thinking = true;   // Layer 1: strip old <think> blocks
int         compaction_keep_tool_uses = 3;      // Layer 1: keep last N tool results
std::string compaction_prompt;                  // Custom summarization prompt
```

**CLI flags (`server_main.cpp`):**

```
--compaction              Enable auto context compaction
--compaction-threshold    Trigger ratio (default 0.9)
--compaction-max-tokens   Max summary length (default 2048)
--compaction-keep-recent  Recent turn ratio to preserve (default 0.3)
```

### 8.3 Layer 1: Edit Compaction (CPU-Only)

Inspired by Inspect AI's `CompactionEdit`[^7]:

```cpp
// New file: server/src/server/compaction.h
struct CompactionResult {
    bool                    applied = false;
    std::vector<ChatMessage> compacted_messages;
    int                     tokens_saved = 0;
};

CompactionResult edit_compact(
    const std::vector<ChatMessage>& messages,
    const ServerConfig& config) {

    CompactionResult result;
    result.compacted_messages = messages;

    // Phase 1: Strip thinking blocks from all but last turn
    for (int i = 0; i < (int)result.compacted_messages.size() - 1; i++) {
        auto& msg = result.compacted_messages[i];
        if (msg.role == "assistant") {
            strip_thinking_blocks(msg.content);  // Remove <think>...</think>
        }
    }

    // Phase 2: Truncate old tool results (keep last N)
    int tool_count = 0;
    for (int i = result.compacted_messages.size() - 1; i >= 0; i--) {
        if (is_tool_result(result.compacted_messages[i])) {
            tool_count++;
            if (tool_count > config.compaction_keep_tool_uses) {
                result.compacted_messages[i].content = "(Tool result removed)";
            }
        }
    }

    // Phase 3: Deduplicate repeated file reads
    // Keep only latest read of each file path
    std::unordered_set<std::string> seen_paths;
    for (int i = result.compacted_messages.size() - 1; i >= 0; i--) {
        auto path = extract_file_read_path(result.compacted_messages[i]);
        if (!path.empty()) {
            if (seen_paths.count(path)) {
                result.compacted_messages[i].content =
                    "[dedup: previously read " + path + "]";
            }
            seen_paths.insert(path);
        }
    }

    result.applied = true;
    return result;
}
```

### 8.4 Layer 2: Self-Summarization

The novel pattern — the server uses its own loaded model to summarize older turns:

```cpp
CompactionResult summarize_compact(
    const std::vector<ChatMessage>& messages,
    const ServerConfig& config,
    Tokenizer& tokenizer,
    ModelBackend& backend) {

    // 1. Split messages: keep recent N% verbatim
    int keep_from = messages.size() * (1.0f - config.compaction_keep_recent);
    std::vector<ChatMessage> old_msgs(messages.begin(), messages.begin() + keep_from);
    std::vector<ChatMessage> recent_msgs(messages.begin() + keep_from, messages.end());

    // 2. Construct summarization prompt
    std::vector<ChatMessage> summary_request;
    summary_request.push_back({"system",
        "Summarize the following conversation concisely. "
        "Preserve: key decisions, file paths, current task state, error messages. "
        "Do not reproduce code verbatim. Keep under 500 words."});
    summary_request.push_back({"user", serialize_messages(old_msgs)});

    // 3. Render + tokenize summary request
    std::string rendered = render_chat_template(
        summary_request, chat_format_, true, false, "");
    std::vector<int32_t> prompt_tokens = tokenizer.encode(rendered);

    // 4. Generate summary (internal inference pass)
    GenerateRequest sum_req;
    sum_req.prompt   = prompt_tokens;
    sum_req.n_gen    = config.compaction_max_tokens;
    sum_req.sampler  = {.temp = 0.0f};  // greedy for determinism

    DaemonIO sum_io;
    sum_io.stream_fd = -1;
    std::vector<int32_t> output_tokens;
    sum_io.on_token = [&](int32_t tok) -> bool {
        output_tokens.push_back(tok);
        return true;
    };
    backend.generate(sum_req, sum_io);

    // 5. Decode summary text
    std::string summary_text = tokenizer.decode(output_tokens);

    // 6. Rebuild message array
    CompactionResult result;
    result.compacted_messages.push_back(messages[0]);  // Preserve system prompt
    result.compacted_messages.push_back(
        {"assistant", "[CONTEXT SUMMARY]\n\n" + summary_text});
    result.compacted_messages.insert(
        result.compacted_messages.end(), recent_msgs.begin(), recent_msgs.end());
    result.applied = true;
    return result;
}
```

### 8.5 Integration Point: Worker Thread

```cpp
// In worker_loop() at http_server.cpp, after job dequeue, before generation:
void worker_loop() {
    while (running_) {
        auto job = dequeue();
        auto& req = job.request;

        // === COMPACTION INSERTION POINT ===
        if (req.compaction_needed && config_.compaction_enabled) {
            // Layer 1: Edit compaction (CPU-only, fast)
            auto edit_result = edit_compact(req.chat_messages, config_);
            std::string rendered = render_chat_template(
                edit_result.compacted_messages, ...);
            req.prompt_tokens = tokenizer_.encode(rendered);

            // Check if Layer 1 was sufficient
            if ((int)req.prompt_tokens.size() + req.max_output > config_.max_ctx) {
                // Layer 2: Self-summarization (requires GPU inference)
                auto sum_result = summarize_compact(
                    edit_result.compacted_messages, config_, tokenizer_, backend_);
                rendered = render_chat_template(
                    sum_result.compacted_messages, ...);
                req.prompt_tokens = tokenizer_.encode(rendered);
            }

            // Final check — Layer 3: Hard truncation
            if ((int)req.prompt_tokens.size() + req.max_output > config_.max_ctx) {
                hard_truncate(req, config_);
            }

            req.compaction_applied = true;
        }

        // ... normal generation continues ...
    }
}
```

### 8.6 Prefix Cache Implications

After compaction, the prefix hash changes:

```cpp
// After compaction, only attempt prefix match on system prompt portion
if (req.compaction_applied) {
    int sys_tokens = req.system_prompt_token_count;
    auto cache_hit = prefix_cache_.lookup(
        std::vector<int32_t>(req.prompt_tokens.begin(),
                             req.prompt_tokens.begin() + sys_tokens));
}
```

| Scenario | KV Cache Impact |
|----------|----------------|
| No compaction (current) | Full prefix hash valid |
| Edit only (strip thinking) | Partial miss after stripped regions |
| Summarization | Full miss except system prompt prefix |
| Hard truncation | System prompt still hits |

### 8.7 API Response: Compaction Signal

For the Responses API, signal compaction in the response:

```json
{
  "id": "resp_abc123",
  "object": "response",
  "output": [
    {
      "type": "compaction_state",
      "data": "<base64-encoded summary for client to round-trip>"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": "..."
    }
  ],
  "usage": {
    "prompt_tokens": 15000,
    "completion_tokens": 450,
    "compacted_tokens_saved": 85000
  }
}
```

### 8.8 OpenAI-Compatible `context_management` Parameter

```cpp
// In route_request(), parse context_management from body:
if (body.contains("context_management")) {
    for (auto& cm : body["context_management"]) {
        if (cm["type"] == "compaction") {
            req.compaction_threshold = cm.value("compact_threshold",
                (int)(config_.max_ctx * 0.9));
        }
    }
}
```

### 8.9 Streaming Compaction Notification

```
data: {"type":"compaction","status":"started","original_tokens":125000}\n\n
... (compaction runs) ...
data: {"type":"compaction","status":"completed","saved_tokens":98000}\n\n
data: {"type":"message","content":"..."}\n\n
```

---

## 9. Comparative Trade-offs

| Strategy | Latency | GPU Mem | Quality | Stateless | lucebox-hub Fit |
|----------|:-------:|:-------:|:-------:|:---------:|:---------------:|
| **Hard 400** (current) | None | None | N/A | ✅ | Current |
| **Edit compaction** (L1) | <1ms | None | Medium | ✅ | **Excellent** |
| **Self-summarization** (L2) | 5–30s | Same | Best | ✅ | **Good** |
| **Ring-buffer shift** | <1ms | None | Low | ❌ | Poor |
| **Sidecar 0.6B model** | 1–5s | +1GB | Good | ✅ | Good |
| **Provider-native** | N/A | N/A | Best | ✅ | N/A (IS provider) |

### Key Risks for Self-Summarization

1. **Deadlock**: Single worker busy with compaction → no other requests served. Mitigation: cap `compaction_max_tokens` aggressively (512–1024).
2. **Latency spike**: Users see 15–30s instead of 5s. Mitigation: SSE `compaction_started` event.
3. **Quality**: Q4-quantized 27B summarizing 64K coding session may lose critical details. Mitigation: preserve last 30–50% verbatim.
4. **Cache pollution**: Summary tokens evict existing LRU entries. Mitigation: separate cache namespace.

---

## 10. Architectural Recommendations

### Recommendation 1: Adopt Layered Mitigation

| Layer | Action | Cost | When |
|-------|--------|------|------|
| L1 | Strip thinking + truncate old tools | CPU, <1ms | Always at 80% |
| L2 | Self-summarize with own model | GPU, 5–30s | When L1 insufficient (90%) |
| L3 | Hard truncation (keep system + last N) | CPU | Emergency fallback |

### Recommendation 2: Support OpenAI `context_management` API

Parse the standard parameter in `/v1/responses` requests for drop-in compatibility with OpenAI Agents SDK.

### Recommendation 3: Streaming Compaction Notification

Emit SSE event immediately so clients know compaction is in progress.

### Recommendation 4: Incremental Construction

Mark summary messages with metadata. On next compaction, only summarize content after the last summary — avoid re-summarizing already-summarized content.

### Recommendation 5: Preserve System Prompt for Cache Hits

Never modify the system prompt during compaction. This preserves prefix-cache hits for the highest-value cache entry.

### Recommendation 6: Configurable via Model Cards

```json
{
  "model": "qwen3.5-27b",
  "compaction": {
    "enabled": true,
    "threshold": 0.85,
    "max_summary_tokens": 1024,
    "keep_recent_ratio": 0.4
  }
}
```

---

## Footnotes

[^1]: `server/src/server/http_server.cpp:1027-1030` — Hard 400 error on context overflow
[^2]: Research finding: no inference engine (vLLM, llama.cpp, SGLang) implements self-summarization
[^3]: [openai/openai-python — ResponseCompactionItem](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_compaction_item.py)
[^4]: [anthropics/anthropic-sdk-python — BetaCompact20260112EditParam](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/beta/beta_compact_20260112_edit_param.py)
[^5]: [UKGovernmentBEIS/inspect_ai — CompactionSummary](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/model/_compaction/summary.py)
[^6]: [langchain-ai/langchain — SummarizationMiddleware](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py)
[^7]: [UKGovernmentBEIS/inspect_ai — CompactionEdit](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/model/_compaction/edit.py)
[^8]: [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)
[^9]: [openai/openai-python — response_create_params.py](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py)
[^10]: [openai/openai-python — response_compact_params.py](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_compact_params.py)
[^11]: [openai/openai-python commit 1039d56](https://github.com/openai/openai-python/commit/1039d5637779e035263019a687b562d3ab5d2c1a)
[^12]: [openai/openai-python commit 625827c](https://github.com/openai/openai-python/commit/625827c5509ece3c40e5002be37a9bd9d91b5374)
[^14]: [anthropics/anthropic-sdk-python — _beta_compaction_control.py](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_compaction_control.py)
[^15]: [arXiv:2601.07190](https://arxiv.org/abs/2601.07190) — Active Context Compression
[^16]: [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) — SAC (ICLR 2026)
[^17]: [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) — IC-Former (EMNLP 2024)
[^18]: [arXiv:2510.00615](https://arxiv.org/abs/2510.00615) — ACON Framework
[^19]: [arXiv:2601.16746](https://arxiv.org/abs/2601.16746) — SWE-Pruner
[^21]: [openai/openai-agents-python — compaction session](https://github.com/openai/openai-agents-python/blob/main/src/agents/memory/openai_responses_compaction_session.py)
[^23]: [ggml-org/llama.cpp — server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)
[^24]: [Roboter-Schlafen-Nicht/agentguard](https://github.com/Roboter-Schlafen-Nicht/agentguard)
[^25]: `server/src/server/http_server.cpp:1027-1030` in lucebox-hub
[^26]: `server/src/server/http_server.h:288-302` — Single worker thread architecture
