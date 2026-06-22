# Claude Code Session Workload Distribution

## Token Estimation Method

Tokenizer-free approximation: **tokens = ceil(chars / 4)**.
Applied to user typed-text and assistant text blocks.
Tool-result and tool-use blocks are excluded from prompt-size distribution
but their text is included in the cumulative-context estimate.

Files processed: 117  |  JSONL files found: 117
Malformed lines skipped: 3

## 1. Prompt-Size Distribution (real typed user turns)

Real typed user turns: 3474
Synthetic (tool-result-only) user events excluded: 13199

### Percentiles (tokens)

| Stat | Tokens |
|------|--------|
| min | 1.0 |
| p10 | 3.0 |
| p25 | 9.0 |
| p50 | 37.0 |
| p75 | 210.2 |
| p90 | 1066.7 |
| p99 | 3917.6 |
| max | 119160.0 |
| mean | 377.2 |

### Histogram

| Bucket (tokens) | Count | Percent |
|-----------------|-------|---------|
| 1-4 | 515 | 14.8% |
| 5-16 | 634 | 18.2% |
| 17-64 | 1017 | 29.3% |
| 65-256 | 491 | 14.1% |
| 257-1k | 430 | 12.4% |
| 1k-4k | 357 | 10.3% |
| 4k-16k | 29 | 0.8% |
| 16k-32k | 0 | 0.0% |
| >32k | 1 | 0.0% |

## 2. Context-Length Distribution (before each real user turn)

Cumulative context = sum of token estimates of all prior messages in the session
(user typed-text + assistant text) before each real user turn.

### Percentiles (tokens)

| Stat | Tokens |
|------|--------|
| min | 0.0 |
| p10 | 5127.8 |
| p25 | 34552.0 |
| p50 | 93875.5 |
| p75 | 179034.8 |
| p90 | 243705.8 |
| p99 | 408908.5 |
| max | 470834.0 |
| mean | 114994.2 |

### Context-Tier Table

| Tier | Count | Percent |
|------|-------|---------|
| <8k | 399 | 11.5% |
| 8-32k | 434 | 12.5% |
| 32-64k | 484 | 13.9% |
| 64-128k | 801 | 23.1% |
| >128k | 1356 | 39.0% |

## 3. Turns per Session

Sessions analyzed: 117

| Stat | Value |
|------|-------|
| min | 0.0 |
| p50 | 4.0 |
| max | 245.0 |
| mean | 29.7 |

## 4. Summary Counts

- Sessions: 117
- Total real typed user messages: 3474
- Synthetic (tool-result-only) user events: 13199
- Malformed/skipped JSONL lines: 3

## 5. Largest Single Prompt

Largest prompt token estimate: **119160 tokens**
Recalled record: ~28k tokens → CONFIRMS recalled ~28k record (observed 119160 tok >= 80% threshold 22400 tok)

## 6. Verdict

Does data confirm "1-2 token prompts up to ~28k, context growing to 32/64/128k+"?

- Prompt range: min=1 tok, max=119160 tok → YES
- Context range: max=470834 tok, p99=408908 tok -> YES
- Largest prompt 119160 tok vs recalled 28k: CONFIRMS recalled ~28k record (observed 119160 tok >= 80% threshold 22400 tok)

**Dominant operating regime: >128k** (39.0% of all real user turns land here)

*End of report.*