---
name: fact-check
description: Fact-check a research paper with dual-verified web search
arguments:
  - name: files
    description: File path(s) to the research paper
    required: true
---

# Fact-Check: Dual-Verified Research Paper Verification

You have been asked to fact-check a research paper. This is a deliberately expensive, token-heavy operation that dual-verifies every claim.

**Input files:** $ARGUMENTS

## Instructions

1. Load the `muna-technical-writer:fact-checking` skill using the Skill tool
2. Follow the skill's four-phase pipeline exactly:
   - Phase 1: Extract all verifiable claims
   - Phase 2: Research-verify each claim batch (web search)
   - Phase 3: Adversarial-verify each claim batch (web search, parallel with Phase 2)
   - Phase 4: Reconcile and produce output
3. Write both output files:
   - `fact-check-results.json` (structured data)
   - `fact-check-exceptions.md` (human-readable exceptions only)

**File paths to check:** $ARGUMENTS

Begin by reading the input files, then proceed to Phase 1 (claim extraction).
