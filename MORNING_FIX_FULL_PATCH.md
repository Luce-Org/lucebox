# MORNING FIX — Full Patch Payload (classifier outage blocked auto-apply)

The classifier was down all night for Bash/Edit/Write on the pflash-evidence repo. The frontend-engineer agent produced exact find/replace strings for every needed change. **Apply in ~5 minutes** via `sd` or manual edit. Once these land + commit + push, the site is publication-ready.

Full prepared patches with exact strings in the agent's output file:
`/tmp/claude-1000/-home-peppi-Dev-lucebox-hub/a01e700a-fe7c-4ca7-bfee-467cc291bd25/tasks/a485213eeeae35baa.output`

## Files needing edits + summary

### `/home/peppi/Dev/pflash-evidence/README.md`
- Hero bullet FP framing fix (1 string replace)
- 3 new bullets after the existing claude_code bullet:
  - 5-client validation (claude_code 3.7×, hermes 3.3×, opencode 3.5×, pi 2.1×, codex 3.1×)
  - MVP Day 5 Pareto-dominance (commits 0d40f2f / 1a1a0f6)
  - 67% dead-weight architectural finding

### `/home/peppi/Dev/pflash-evidence/EVIDENCE.md`
- §12 line ~308 FP framing fix (1 string replace)
- 4 NEW sections appended:
  - §16b: 5-client production validation (full table)
  - §16c: MVP Day 5 Pareto-dominance
  - §16d: Bug #42 root cause + downgrade
  - §16e: 67% dead-weight architectural finding

### `/home/peppi/Dev/pflash-evidence/OPEN_QUESTIONS.md`
- P1-H section: de-emphasize FP (it's 17%, NOT dominant; lower priority than P1-I)
- 2 new closed items:
  - P0-E: "Does ee7 work on all 5 agentic clients?" → YES
  - P1-J: "Does bandit Pareto-dominate?" → YES

### `/home/peppi/Dev/pflash-evidence/journey.md`
- 1 NEW dated section "2026-05-22 overnight" with 4 subsections covering all overnight findings

### `/home/peppi/Dev/pflash-evidence/index.html`
- Hero: 2 new chips (5-client, bandit-Pareto)
- FP callout (line ~613): fix wrong "dominant kernel" framing
- Per-stage table (line ~604): add "untracked overhead" column (now 5 cols)
- P1-H table row (line ~892): fix framing
- 3 NEW sections: `#five-client`, `#bandit-pareto`, `#drafter-dead-weight`
- TOC blocks: add 3 new anchor links

### `/home/peppi/Dev/pflash-evidence/share.html`
- OG description: update to lead with 5-client + Pareto + dead-weight
- Pareto callout: change from "inconclusive — Day 5 work" → "VERIFIED (commits 0d40f2f, 1a1a0f6)"

## Total work

- 6 files
- ~15 string replacements + ~6 inserts
- ~5 minutes manual or 1 minute via `sd` find/replace
- Then: `git add -A && git commit && git push`

## After this lands

The pflash-evidence site will be the complete, accurate, publishable record of tonight's work:
- **All 5 named agentic clients validated** with ee7 (2.1×–3.7× drafter speedup)
- **MVP bandit Days 1-5 Pareto-dominance verified**
- **9.29× at 128K** as the headline number
- **Hardware-correct** (RTX 3090, with the RTX 6000 Ada disclosure)
- **Honest constraints** (bug #42 root-caused + downgraded, 64K NIAH cliff distinguished)
- **Architectural moat** (67% of drafter is dead weight in pflash mode — opportunity for purpose-built scoring model)

## If you want the patches inline

The exact find/replace strings are in the agent's output file referenced at the top of this doc. Each replacement is structured as:

```
FIND:
[exact old string]

REPLACE WITH:
[exact new string]
```

So a Python or `sd` script could apply them all programmatically. Example:

```bash
cd /home/peppi/Dev/pflash-evidence/

# Easiest: open each file, paste from the agent output, save.
# Faster: sd 'old' 'new' file.md (for each pair)
```
