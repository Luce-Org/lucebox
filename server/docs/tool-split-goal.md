# Tool-split — project goal

## The idea

Agents that use tools (read a file, run a command, search code) should feel **fast after the first turn** — not like every message is starting from scratch.

Today, tool definitions get mixed into the same prompt as the conversation. That hurts **PFlash**, the system that compresses and speeds up long chat history. When tools and chat share one blob of text, PFlash has to fight through tool JSON it was never meant to optimize. You pay for the tools again and again, and the conversation speedups never fully kick in.

**Tool-split** is the fix: **pull tools out of the conversation path.**

- Tool schemas live in their own pinned memory (thin KV slots).
- The chat history stays clean for PFlash and prefix cache.
- Tools no longer drag down the algorithm that makes multi-turn chat fast.

Pay the full cost once. After that — especially after a tool result comes back — the agent should feel snappy.

## What users should feel

- **Turn 1 (cold)** — full cost once (tools + first message). Expected.
- **Later turns with a little new text** — a few seconds, not another cold start.
- **After a tool result** — the common “continue” path must be fast and reliable.

If the cache is “working” but people still wait 15 seconds per message, we have not succeeded.

## How it works (short version)

1. **Pin tools separately** — tool schemas sit in thin snapshot slots (`SNAPSHOT_THIN`). No re-prefill of hundreds of tool tokens every turn.
2. **Cache conversation alone** — chat history uses `RESTORE_CHAIN` + prefix cache so PFlash can focus on what it does best.
3. **Respect VRAM** — on 2×24GB, `DFLASH_PREFIX_CACHE_SLOTS=1` is a conservative default (one conversation slot, updated in place). Thick prefix snapshots live in CPU RAM, so additional slots are viable when `prefix + tool_pins ≤ DAEMON_MAX_SLOTS`. Extra thick snapshots can still OOM and **silently kill every speedup**.

Stack: `model-runner-v4` → lucebox (:8080) → ai-platform proxy (:8000)

## Practical success criteria

| What users care about | Target | How we measure |
|----------------------|--------|----------------|
| Incremental turn latency | **Wall-clock ≪ turn 1** when only a few new tokens are added | `elapsed_s` on benchmark turn 3 / agent-after-tool |
| Time to start generating | **Prefill ≪ cold** on cached turns | `usage.timings.prefill_ms` |
| Reliability | Speedup works every session, not 1-in-3 after OOM | `inline-snap committed` ≥ 1, no `inline snap failed` in logs |
| Agent hot path | **After tool result**, response in **&lt; 4s** typical | `agent_after_tool` benchmark phase |
| Correctness | Multi-turn tools complete, no `bad thick slot` | 3+ turn session completes |

### Validated on ai.local (when cache is active)

| Turn | elapsed | prefill_ms | Notes |
|------|---------|------------|-------|
| 1 cold | ~8s | ~2570 | full prompt + tool pin |
| 2 (bigger delta) | ~6s | ~2240 | still prefills new messages — **not the main win** |
| 3 (tiny delta) | **~3.7s** | **~120** | **21× prefill speedup** — this is the usable win |

**Key insight:** Speedup tracks **how many new prompt tokens** you add. Small follow-ups feel fast; huge new user messages still cost prefill. That is expected — agents usually add short tool results or short replies between turns.

## What “done” is not

- Faster decode tok/s on tool turns (decode is short; **prefill** is what we cache).
- Benchmark-only wins that do not show up as lower `elapsed_s` for incremental turns.
- Production-ready without soak tests, baked image, and CI gate.

## One-line summary

**Split tools out so PFlash can speed up the conversation — pay full prefill once; every small follow-up (especially after tool results) should feel snappy.**

## Infrastructure reference

| Item | Path / detail |
|------|----------------|
| Server | `user@host`, `/path/to/projects/` |
| Patch scripts | `model-runner-v4/lucebox-patch/dflash/scripts/` |
| Daemon binary | `lucebox-hub/build/test_dflash` |
| Benchmark | `model-runner-v4/scripts/benchmark-tool-split.py` |
| Goal doc | `server/docs/tool-split-goal.md` |

## Key env

```bash
DFLASH_TOOL_SPLIT_ENABLED=1
DFLASH_PREFIX_CACHE_SLOTS=1      # conservative default for 2×24GB; not a hard cap
DFLASH_TOOL_SPLIT_PINNED_SLOTS=2
DFLASH_LAYER_SPLIT=0
```

## How to verify practical speed

1. Use **`tools`** in the request (tool-split is off without them).
2. **Restart lucebox** after deploy (clears stale GPU snapshot slots).
3. Run `benchmark-tool-split.py` — check **elapsed_s** and **prefill_ms**, not decode tok/s.
4. Logs must show `thick=0`, `inline-snap committed`, and **no** `inline snap failed`.
5. Compare **turn 3** or **agent_after_tool** to turn 1 — that is the user-visible win.

## Remaining work for production-grade practical speed

- [x] Agent-realistic benchmark phase (user → tool_call → tool result → continue)
- [ ] Bake patch + binary into image (no host-mount drift)
- [ ] CI gate: incremental turn `elapsed_s` &lt; 4s, `prefill_ms` &lt; 500ms
- [ ] Soak test: 50 sessions without daemon death or OOM
- [ ] Merge `lucebox-hub` PR and sync `model-runner-v4` defaults
