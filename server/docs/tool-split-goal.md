# Tool-split on ai.local — project goal

## North star

**Practical, usable speedup** — an agent using tools on `ai.local` should feel meaningfully faster after the first turn, not just score well on internal prefill metrics.

What the user experiences:

- **Turn 1 (cold)** — pays full cost once (large prompt + tool schema + first prefill). Unavoidable.
- **Every later turn with small new content** — should return in **a few seconds wall-clock**, not another ~8s cold start.
- **After a tool result** — the “continue with tool output” turn is the common hot path; it must be fast and reliable.

If cache is working but the user still waits 15s per message, we have not succeeded.

## Primary technical goal

Make **multi-turn tool-calling on Lucebox/DFlash fast and reliable** on:

`model-runner-v4` → lucebox (:8080) → ai-platform proxy (:8000)

Mechanisms:

1. **Tool KV pinned** — reuse tool-schema prefix (`SNAPSHOT_THIN` / thin slots). Saves re-prefilling ~376 tool tokens every turn.
2. **Conversation prefix cached** — reuse conv history (`RESTORE_CHAIN thick=0` + inline snapshots). Saves re-prefilling everything before the latest user turn.
3. **VRAM discipline** — one conv prefix slot, updated in-place (`DFLASH_PREFIX_CACHE_SLOTS=1`). Multiple thick snapshots OOM on 2×24GB and **silently disable all speedups**.

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

**Key insight:** Speedup is proportional to **how many new prompt tokens** you add. Small follow-ups feel fast; large new user messages still cost prefill time. That is expected and usable — agents usually add small tool results or short replies between turns.

## What “done” is not

- Faster decode tok/s on tool turns (decode is short; **prefill** is what we cache).
- Benchmark-only wins that do not show up as lower `elapsed_s` for incremental turns.
- Production-ready without soak tests, baked image, and CI gate.

## One-line summary

**Pay full prefill once per session; every small follow-up (especially after tool results) should feel snappy.**

## Infrastructure reference

| Item | Path / detail |
|------|----------------|
| Server | `david@192.168.87.153`, `/media/data/projects/` |
| Patch scripts | `model-runner-v4/lucebox-patch/dflash/scripts/` |
| Daemon binary | `lucebox-hub-src/dflash/build/test_dflash` |
| Benchmark | `model-runner-v4/scripts/benchmark-tool-split.py` |
| Goal doc | `lucebox-hub/dflash/docs/tool-split-goal.md` |

## Key env

```bash
DFLASH_TOOL_SPLIT_ENABLED=1
DFLASH_PREFIX_CACHE_SLOTS=1      # required for reliable cache on 2×24GB
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

- [ ] Agent-realistic benchmark phase (user → tool_call → tool result → continue)
- [ ] Bake patch + binary into image (no host-mount drift)
- [ ] CI gate: incremental turn `elapsed_s` &lt; 4s, `prefill_ms` &lt; 500ms
- [ ] Soak test: 50 sessions without daemon death or OOM
- [ ] Merge `lucebox-hub` PR and sync `model-runner-v4` defaults
