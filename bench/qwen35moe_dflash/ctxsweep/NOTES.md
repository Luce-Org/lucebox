
### 35B MoE pure-spec verbatim ctx-sweep

| ctx | prefill_s | decode tok/s | accept% | commit | action |
|--:|--:|--:|--:|--:|---|
| 1024 | 0.9 | 230.7 | 80.4 | 13.04 | spec |
| 2048 | 1.2 | 227.9 | 81.5 | 13.04 | spec |
| 4096 | 2.4 | 25.9 | 6.4 | 2.01 | spec |
| 8192 | 5.0 | 23.8 | 6.4 | 2.01 | spec |
| 16384 | 10.6 | 20.6 | 6.4 | 2.01 | spec |
| 24576 | 16.5 | 17.9 | 6.3 | 2.01 | spec |
| 32768 | 23.7 | 16.1 | 6.4 | 2.01 | spec |

### 35B distant-recall verbatim (target = first lines of a long blob)
| ctx | accept% | decode | note |
|--:|--:|--:|---|
| 1024 | 80.4 | 231 | target in SWA window |
| 2048 | 81.5 | 228 | target in SWA window |
| 4096 | 6.4 | 26 | CLIFF: target now beyond 2048 SWA |
| 8192 | 6.4 | 24 | |
| 16384 | 6.4 | 21 | AR slowdown |
| 32768 | 6.4 | 16 | |
FINDING: accept cliff at exactly the drafter SWA window (2048). dFlash blind to content >2048 tokens back.

### 35B recent-target verbatim (target in-window, vary preceding ctx)

| ctx | prefill_s | decode tok/s | accept% | commit | action |
|--:|--:|--:|--:|--:|---|
| 1024 | 0.5 | 299.5 | 92.7 | 14.83 | spec |
| 2048 | 1.0 | 267.5 | 92.7 | 14.83 | spec |
| 4096 | 2.1 | 24.6 | 6.2 | 1.98 | spec |
| 8192 | 4.4 | 26.8 | 8.7 | 2.34 | spec |
| 16384 | 9.7 | 23.9 | 8.7 | 2.34 | spec |
| 24576 | 15.6 | 21.0 | 8.7 | 2.34 | spec |
| 32768 | 22.0 | 19.3 | 8.7 | 2.34 | spec |

### 35B full-attn drafter (draft-swa 40960) — cliff PERSISTS
| ctx | accept% | decode |
|--:|--:|--:|
| 2095 | 92.7 | 226 |
| 4351 | 6.2 | 5.3 |
| 9092 | 8.7 | 1.7 |
FINDING: cliff at ~2048 is NOT the SWA window (full-attn doesnt fix it, makes long-ctx slower). Drafter EFFECTIVE CONTEXT ~2048 despite 40K-claimed training. Likely bf16-reconv conversion/integration bug (same conversion as the rope bug). dFlash works <=2048 ctx only as-is. UNLOCK = fix drafter long-ctx (try official z-lab reconvert / spiritbuun GGUF, or draft-graph position handling).

## KV precision sweep (35B Q3_K_XL, 35K verbatim, cap=2048)
| KV | prefill | decode | accept% | AL |
|---|--:|--:|--:|--:|
| f16 | 18.0s | 174 | 76.8 | 12.86 |
| q4_0 | 18.0s | 167 | 76.8 | 12.86 |
| q8_0 | 18.1s | 143 | 66.4 | 11.25 |
| tq3_0 | 23.6s | 109 | 76.8 | 12.86 |
f16 best; q4_0 EQUAL (free VRAM saver, no accept/AL cost); q8_0 ANOMALOUS (lower accept 66.4
## KVFlash added to the 35B agentic config?
- max_ctx 40960 (KV fits all-hot): --kvflash auto AUTO-DISABLES ("pool not needed"). No-op. decode/caching unchanged (55-80 tok/s, restore=true).
- Forced on (max_ctx 131072, pool 65536): keeps experts HOT (reserves pool KV not max_ctx -> 0 cold = solves spill) BUT breaks prefix-cache restore (restore=false -> warm re-prefill 77.6s vs 0.8s) AND slows decode (20-29 vs 55-67).
- VERDICT: KVFlash trades caching+decode for unbounded context. 34K coding sessions fit in q4_0 KV and NEED caching -> KVFlash net-negative. Only for 128K-256K that wont fit, and even then lose restore on this all-hot MoE path.
