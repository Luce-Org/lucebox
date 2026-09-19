# Lucebox cache diagnosis — 2026-09-14

## Findings

Inspected linuxmacan over SSH. Gateway configuration is `/opt/lucebox/models.json`; the executable actually used is `/opt/lucebox-concurrent/server/build-hip/dflash_server`. The concurrent checkout is at `f71859885776c42f3cbe50cbe77531e620d52f10` with local modifications; the binary was built September 13 at 23:00 UTC. No configuration, source, or service changes were made.

The concurrent service already keeps copied prefix checkpoints in host RAM. It has 12 slots and an 8192 MiB total resident budget. Disk caching is disabled for this serving mode. Active inference state remains on the GPU.

## Measured evidence

`/opt/lucebox/model-backend.log` records repeated restores of the same 46,013-token checkpoint while prompts increase from 48,634 to 79,616 tokens. The checkpoint itself occupies 1,758,881,792 bytes. At its last successful save, total checkpoint residency was 8,460,922,880 bytes against an 8,589,934,592-byte limit: only about 123 MiB remained.

Representative completed request `chatcmpl_000000000000010d`: 78,827 input tokens, 46,013 cached tokens, 45.7 ms checkpoint restore, 147.0 seconds prefill, 26.1 seconds decode, 590 output tokens. Thus 32,814 input tokens were reprocessed despite the conversation already having progressed beyond the old checkpoint on previous turns. Earlier requests also log resident-budget capture skips.

## Cause in source

`/opt/lucebox-concurrent/server/src/server/prefix_cache.cpp:640-706` checks whether replacing a single eligible entry is sufficient to fit the new checkpoint. It searches alternative individual victims, protects the active restore source, and returns an empty reservation if no individual victim suffices. It does not reclaim multiple old entries. This can freeze cache advancement as conversations grow, despite many obsolete checkpoints remaining resident.

`server/src/server/scheduler.cpp:457` uses this reservation for concurrent requests. `server/src/qwen35/qwen35_target_graph.cpp:3251` estimates host snapshot allocation; the snapshot capture code copies paged attention and recurrent state. Prefix caching depends on an exactly matching token prefix and compatible model state; model switching terminates the old backend, losing its in-memory checkpoints. No time-based expiry was found in the prefix cache implementation.

## Recommended repair

Allow cache admission to reclaim enough older eligible checkpoints to fit the new one, while preserving in-flight restore/capture state and useful shared prefixes. Validate with a bounded-memory regression in which no single eviction suffices but multiple evictions do, then a growing multi-turn serving probe that verifies the restored prefix advances. Increasing RAM allocation alone is a temporary workaround and does not repair the eviction policy.

The host currently has 24 GiB RAM. The previous service invocation reported a 23 GiB memory peak and a 914.4 MiB swap peak. Those peaks do not establish the exact impact of swapping on individual requests, but argue against blindly increasing the cache limit.

## Current service state and limits

The host had recently restarted. Lucebox loaded its default model in approximately 16 seconds and was then explicitly stopped by systemd at 22:37:49 UTC (18:37:49 Toronto). At inspection it was inactive and port 8216 refused connections. The journal does not identify who requested the stop. Its state was left unchanged.

Diagnosis is based on current source/configuration and existing production logs. No new inference benchmark or repair deployment was performed. The cache stall explains the documented long prefill delays; it does not claim to explain every latency fluctuation.
