# 3-Seed Day-5 A/B/C Summary - PR #264 Variance Evidence

Run date: 2026-05-23
Branch: feat/pflash-mvp-adaptive-keep (692064f)
GPU: NVIDIA GeForce RTX 3090 24 GB
Model: Qwen3.6-27B Q4_K_M target, Q4_K_M draft, Qwen3-0.6B-BF16 pflash drafter

## Prompts Used Per Seed

| Seed  | Prompt file       | Task                         |
|-------|-------------------|------------------------------|
| seed1 | decode_check.txt  | Python function explanation  |
| seed2 | logic_check.txt   | Logic puzzles (3 items)      |
| seed3 | math_check.txt    | Arithmetic problems (3 items)|

## Per-Run Data

| Seed  | Condition    | keep | wall_s | ok_done | accept_rate% | bandit_fired |
|-------|-------------|------|--------|---------|--------------|--------------|
| seed1 | A_fixed_low  | 0.05 | 14     | YES     | 30.4         | -            |
| seed1 | B_fixed_high | 0.20 | 29     | YES     | 30.1         | -            |
| seed1 | C_bandit     | 0.10 | 15     | YES     | 34.6         | YES          |
| seed2 | A_fixed_low  | 0.05 | 20     | YES     | 32.4         | -            |
| seed2 | B_fixed_high | 0.20 | 23     | YES     | 29.8         | -            |
| seed2 | C_bandit     | 0.10 | 21     | YES     | 30.4         | YES          |
| seed3 | A_fixed_low  | 0.05 | 11     | YES     | 43.8         | -            |
| seed3 | B_fixed_high | 0.20 | 22     | YES     | 38.6         | -            |
| seed3 | C_bandit     | 0.10 | 13     | YES     | 38.9         | YES          |

## Mean +/- Std Across 3 Seeds

| Arm           | keep | wall_s (mean +/- std) | accept_rate% (mean +/- std) |
|---------------|------|------------------------|------------------------------|
| A fixed_low   | 0.05 | 15.0 +/- 3.7           | 35.5 +/- 5.9                 |
| B fixed_high  | 0.20 | 24.7 +/- 3.1           | 32.8 +/- 4.1                 |
| C bandit      | 0.10 | 16.3 +/- 3.4           | 34.6 +/- 3.5                 |

## Pareto Verdict

C (bandit, keep=0.10) vs B (fixed_high, keep=0.20):
- wall_s: C faster by 8.3 s mean (16.3 vs 24.7) = 1.52x speedup, non-overlapping
- accept_rate: C higher by 1.8 pp mean (34.6% vs 32.8%), partially overlapping std bands

PARETO DOMINATES: bandit beats fixed keep=0.20 on both metrics in mean, in all 3 seeds.

## Bandit Log Lines

seed1/C: [pflash-bandit] session=claude_code_day5s1 turn=1 keep=0.1000->0.1100 ema=0.346 accept=0.346
seed2/C: [pflash-bandit] session=claude_code_day5s2 turn=1 keep=0.1000->0.1100 ema=0.304 accept=0.304
seed3/C: [pflash-bandit] session=claude_code_day5s3 turn=1 keep=0.1000->0.1100 ema=0.389 accept=0.389
