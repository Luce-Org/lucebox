#!/usr/bin/env python3
"""
Session workload distribution analyzer for Claude Code transcript JSONL files.
PRIVACY: outputs ONLY aggregate statistics — never prints message content.
Token estimation: tokens = ceil(chars / 4)  [tokenizer-free approximation]
"""
import json
import math
import os
import glob
import sys
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────────────

def token_est(chars: int) -> int:
    return math.ceil(chars / 4) if chars > 0 else 0

def text_len_from_content(content) -> tuple[int, bool]:
    """
    Returns (typed_text_char_len, is_synthetic).
    is_synthetic = True when the content is ONLY tool_result blocks (no user text).
    Handles both str and list-of-blocks shapes.
    """
    if content is None:
        return 0, True
    if isinstance(content, str):
        return len(content), False
    if isinstance(content, list):
        text_chars = 0
        has_text_block = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                has_text_block = True
                text_chars += len(block.get("text", ""))
            # tool_result and tool_use are skipped — not user-typed text
        is_synthetic = not has_text_block
        return text_chars, is_synthetic
    return 0, True

def assistant_text_len(content) -> int:
    """Sum text chars in assistant message content (str or list of blocks)."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                total += len(block.get("text", ""))
        return total
    return 0

def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

# ── file discovery ────────────────────────────────────────────────────────────

patterns = [
    "/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/*.jsonl",
    "/home/peppi/.claude/projects/*/*.jsonl",
]

all_files = set()
for pat in patterns:
    for f in glob.glob(pat, recursive=False):
        all_files.add(f)

# Sort for reproducibility
all_files = sorted(all_files)
print(f"[info] Total JSONL files found: {len(all_files)}", file=sys.stderr)

# ── per-session accumulators ──────────────────────────────────────────────────

# A "session" = one JSONL file (each file is one conversation)
session_real_turns = []       # list of int (count of real typed turns per session)
all_prompt_tokens = []        # token estimate for each real typed user turn
all_ctx_before_tokens = []    # context length BEFORE each real typed user turn

total_synthetic_turns = 0
total_real_turns = 0
total_skipped_lines = 0
total_sessions = 0
max_prompt_tok = 0

for fpath in all_files:
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        print(f"[warn] Cannot read {fpath}: {e}", file=sys.stderr)
        continue

    total_sessions += 1
    cumulative_ctx_chars = 0  # running sum of ALL prior message char lengths
    session_real = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            total_skipped_lines += 1
            continue

        # Events may be wrapped in {"type":"...", "message":{...}} or be the message directly
        msg = event.get("message", event)
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content")

        if role == "user":
            char_len, is_synthetic = text_len_from_content(content)
            if is_synthetic:
                total_synthetic_turns += 1
                # Still count tool_result chars toward context growth
                # (tool results are part of the conversation context)
                # Approximate: sum all text in tool_result blocks too for context
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "tool_result":
                                inner = block.get("content", "")
                                if isinstance(inner, str):
                                    cumulative_ctx_chars += len(inner)
                                elif isinstance(inner, list):
                                    for ib in inner:
                                        if isinstance(ib, dict) and ib.get("type") == "text":
                                            cumulative_ctx_chars += len(ib.get("text", ""))
            else:
                # Real typed user turn
                ctx_tok = token_est(cumulative_ctx_chars)
                prompt_tok = token_est(char_len)

                all_ctx_before_tokens.append(ctx_tok)
                all_prompt_tokens.append(prompt_tok)
                if prompt_tok > max_prompt_tok:
                    max_prompt_tok = prompt_tok

                session_real += 1
                total_real_turns += 1
                cumulative_ctx_chars += char_len

        elif role == "assistant":
            a_len = assistant_text_len(content)
            cumulative_ctx_chars += a_len

    session_real_turns.append(session_real)

# ── compute statistics ────────────────────────────────────────────────────────

# Prompt-size distribution (token estimates)
prompt_sorted = sorted(all_prompt_tokens)

def stats(sorted_vals):
    if not sorted_vals:
        return {}
    return {
        "count": len(sorted_vals),
        "min":   sorted_vals[0],
        "p10":   percentile(sorted_vals, 10),
        "p25":   percentile(sorted_vals, 25),
        "p50":   percentile(sorted_vals, 50),
        "p75":   percentile(sorted_vals, 75),
        "p90":   percentile(sorted_vals, 90),
        "p99":   percentile(sorted_vals, 99),
        "max":   sorted_vals[-1],
        "mean":  sum(sorted_vals) / len(sorted_vals),
    }

prompt_stats = stats(prompt_sorted)

# Histogram buckets for prompt sizes (tokens)
buckets = [
    ("1-4",     1,     4),
    ("5-16",    5,    16),
    ("17-64",  17,    64),
    ("65-256", 65,   256),
    ("257-1k", 257, 1000),
    ("1k-4k", 1001, 4000),
    ("4k-16k", 4001, 16000),
    ("16k-32k", 16001, 32000),
    (">32k",   32001, 10**9),
]

hist = {}
for label, lo, hi in buckets:
    cnt = sum(1 for t in prompt_sorted if lo <= t <= hi)
    hist[label] = cnt

total_real = len(prompt_sorted)

# Context-length distribution (tokens) — BEFORE each real user turn
ctx_sorted = sorted(all_ctx_before_tokens)
ctx_stats = stats(ctx_sorted)

ctx_tiers = [
    ("<8k",      0,    7999),
    ("8-32k",  8000,  31999),
    ("32-64k", 32000, 63999),
    ("64-128k",64000, 127999),
    (">128k", 128000, 10**9),
]
ctx_hist = {}
for label, lo, hi in ctx_tiers:
    cnt = sum(1 for t in ctx_sorted if lo <= t <= hi)
    ctx_hist[label] = cnt

# Turns per session
sess_sorted = sorted(session_real_turns)
sess_stats = stats(sess_sorted)

# ── format report ─────────────────────────────────────────────────────────────

lines_out = []
A = lines_out.append

A("# Claude Code Session Workload Distribution")
A("")
A("## Token Estimation Method")
A("")
A("Tokenizer-free approximation: **tokens = ceil(chars / 4)**.")
A("Applied to user typed-text and assistant text blocks.")
A("Tool-result and tool-use blocks are excluded from prompt-size distribution")
A("but their text is included in the cumulative-context estimate.")
A("")
A(f"Files processed: {total_sessions}  |  JSONL files found: {len(all_files)}")
A(f"Malformed lines skipped: {total_skipped_lines}")
A("")

A("## 1. Prompt-Size Distribution (real typed user turns)")
A("")
A(f"Real typed user turns: {total_real_turns}")
A(f"Synthetic (tool-result-only) user events excluded: {total_synthetic_turns}")
A("")
A("### Percentiles (tokens)")
A("")
A("| Stat | Tokens |")
A("|------|--------|")
for k in ["min", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean"]:
    v = prompt_stats.get(k, 0)
    A(f"| {k} | {v:.1f} |")
A("")
A("### Histogram")
A("")
A("| Bucket (tokens) | Count | Percent |")
A("|-----------------|-------|---------|")
for label, lo, hi in buckets:
    cnt = hist[label]
    pct = 100 * cnt / total_real if total_real else 0
    A(f"| {label} | {cnt} | {pct:.1f}% |")
A("")

A("## 2. Context-Length Distribution (before each real user turn)")
A("")
A("Cumulative context = sum of token estimates of all prior messages in the session")
A("(user typed-text + assistant text) before each real user turn.")
A("")
A("### Percentiles (tokens)")
A("")
A("| Stat | Tokens |")
A("|------|--------|")
for k in ["min", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean"]:
    v = ctx_stats.get(k, 0)
    A(f"| {k} | {v:.1f} |")
A("")
A("### Context-Tier Table")
A("")
A("| Tier | Count | Percent |")
A("|------|-------|---------|")
total_ctx_pts = len(all_ctx_before_tokens)
for label, lo, hi in ctx_tiers:
    cnt = ctx_hist[label]
    pct = 100 * cnt / total_ctx_pts if total_ctx_pts else 0
    A(f"| {label} | {cnt} | {pct:.1f}% |")
A("")

A("## 3. Turns per Session")
A("")
A(f"Sessions analyzed: {total_sessions}")
A("")
A("| Stat | Value |")
A("|------|-------|")
for k in ["min", "p50", "max", "mean"]:
    v = sess_stats.get(k, 0)
    A(f"| {k} | {v:.1f} |")
A("")

A("## 4. Summary Counts")
A("")
A(f"- Sessions: {total_sessions}")
A(f"- Total real typed user messages: {total_real_turns}")
A(f"- Synthetic (tool-result-only) user events: {total_synthetic_turns}")
A(f"- Malformed/skipped JSONL lines: {total_skipped_lines}")
A("")

A("## 5. Largest Single Prompt")
A("")
A(f"Largest prompt token estimate: **{max_prompt_tok} tokens**")
recalled = 28000
if max_prompt_tok >= recalled * 0.8:
    verdict_28k = f"CONFIRMS recalled ~28k record (observed {max_prompt_tok} tok >= 80% threshold {int(recalled*0.8)} tok)"
else:
    verdict_28k = f"DOES NOT confirm recalled ~28k record (observed {max_prompt_tok} tok, expected ~{recalled} tok)"
A(f"Recalled record: ~28k tokens → {verdict_28k}")
A("")

A("## 6. Verdict")
A("")
# Determine dominant regime
dominant_ctx = max(ctx_tiers, key=lambda t: ctx_hist[t[0]])[0]
dominant_pct = 100 * ctx_hist[dominant_ctx] / total_ctx_pts if total_ctx_pts else 0

prompt_min = prompt_stats.get("min", 0)
prompt_max = prompt_stats.get("max", 0)
ctx_max = ctx_stats.get("max", 0)

confirm_range = "YES" if (prompt_min <= 2 and prompt_max >= 20000) else "PARTIAL"
confirm_ctx   = "YES" if ctx_max >= 128000 else ("PARTIAL (max ctx below 128k)" if ctx_max >= 32000 else "NO")

A(f'Does data confirm "1-2 token prompts up to ~28k, context growing to 32/64/128k+"?')
A("")
A(f"- Prompt range: min={prompt_min:.0f} tok, max={prompt_max:.0f} tok → {confirm_range}")
ctx_p99 = ctx_stats.get("p99", 0)
A(f"- Context range: max={ctx_max:.0f} tok, p99={ctx_p99:.0f} tok -> {confirm_ctx}")
A(f"- Largest prompt {max_prompt_tok} tok vs recalled 28k: {verdict_28k}")
A("")
A(f"**Dominant operating regime: {dominant_ctx}** ({dominant_pct:.1f}% of all real user turns land here)")
A("")
A("*End of report.*")

report = "\n".join(lines_out)

# Write to file
out_path = "/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep/session_distribution.md"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(report)

print(f"[done] Report written to {out_path}", file=sys.stderr)
# Also print to stdout for direct reading
print(report)
