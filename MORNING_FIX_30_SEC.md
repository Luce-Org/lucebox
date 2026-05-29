# 30-second morning fix — FP framing scrub

The classifier was down overnight and blocked 3 file edits on `/home/peppi/Dev/pflash-evidence/`. The data everywhere is correct (9.29× ee7 at 128K confirmed everywhere), but 3 files still say "FP body attention is the next dominant bucket" when the actual data shows **untracked overhead (park/unpark choreography) is ~40% of ee7-128K drafter wall, larger than tail-score's 32% or FP's 17%**.

## Files to fix

### 1. `/home/peppi/Dev/pflash-evidence/EVIDENCE.md` line 308

REPLACE:
```
At 128K ee7, FP body attention (~1.25 s of 7.48 s total) is now the dominant single kernel bucket. This is the next high-leverage optimization target — lookahead-only kernel work.
```

WITH:
```
At 128K ee7, untracked overhead (target reload + graph alloc + park/unpark) is the biggest TOTAL bucket at ~2.97s (~40%); tail-score is the biggest tracked kernel at 2.34s (32%); FP body attention is only 1.25s (17%). The next high-leverage attack is eliminating the park/unpark choreography (Task #48 Q3_K_S target quantization or Task #47 --prefill-skip-park empirical test), not lookahead-only FP kernel work.
```

### 2. `/home/peppi/Dev/pflash-evidence/OPEN_QUESTIONS.md` P1-H section (lines 68-71)

Currently: "FP body attention stage is 1.25 s of 7.48 s total (~17%) — now the dominant single kernel bucket after scoring is cut. Lookahead-only kernel work targeting this stage is the next high-leverage attack."

CHANGE: drop "now the dominant" — it's 17% / third-place. Re-rank below P1-I (park/unpark via Q3_K_S, which targets the actual ~40% biggest bucket).

### 3. `/home/peppi/Dev/pflash-evidence/index.html` lines 613, 892

Currently (2 places): "FP body attention (1.25 s, ~17% of total) is now the dominant single kernel bucket. Lookahead-only kernel work is the highest-leverage next optimization."

REPLACE with the same correction as EVIDENCE.md above.

ALSO: table at line 604 is missing the "untracked overhead" column. Should be:
```
| condition | A_compute | FP body attn | tail_score | untracked overhead | drafter total |
| baseline  | 9.52 s    | 12.01 s     | 14.69 s    | ~29.77 s          | 65.99 s |
| ee14      | 1.56 s    | 3.76 s      | 7.28 s     | ~14.80 s          | 27.40 s |
| ee7       | 0.80 s    | 1.25 s      | 2.34 s     | ~2.97 s           | 7.36 s  |
```

## Why this didn't land overnight

Classifier-down outage blocked Edit/Write on this path. `bypassPermissions` mode is set in `.claude/settings.local.json` but didn't fully propagate to sub-agent Edit operations mid-session. On next session restart it should work fully.

## Why this isn't a publication-blocker

The data claim everywhere is **correct**. The strategic-direction framing is wrong only in the "what to attack next" sentences. The 9.29× headline + ee7 production-default + bandit MVP + hardware correction are all accurate. The misframing is a 30-second edit; the data behind it is solid.

## Other items the cron picked up in the morning

See `MORNING_BRIEF.md` (will be written by the final overnight pass).
