# Archive of Untested Briefings

This directory contains briefings that were created without following the proper RED-GREEN-REFACTOR TDD cycle.

## Untested Briefings (v1)

Created: 2025-11-19
Created by: Maintenance workflow without behavioral testing
Violation: Writing-skills Iron Law - "NO SKILL WITHOUT A FAILING TEST FIRST"

### Files

1. **assessing-code-quality-v1-untested.md** (~400 lines)
   - Content coverage: Complexity, duplication, code smells, maintainability, dependencies
   - Problem: No baseline testing to verify agents follow guidance
   - Use: Reference for content areas to cover in tested version

2. **creating-architect-handover-v1-untested.md** (~400 lines)
   - Content coverage: Handover report generation, consultation patterns, architect integration
   - Problem: No baseline testing to verify agents follow guidance
   - Use: Reference for content areas to cover in tested version

## Purpose

These files are archived (not deleted) to:
- Track what content areas should be covered
- Compare tested vs. untested versions
- Document the improvement from proper TDD methodology
- Serve as reference when designing pressure scenarios

## Tested Versions

Properly tested versions (RED-GREEN-REFACTOR) will be created in the parent directory following:
1. **RED:** Baseline scenarios WITHOUT skill - document exact failures
2. **GREEN:** Write minimal skill addressing observed rationalizations
3. **REFACTOR:** Find loopholes, plug them, re-test until bulletproof

## Do Not Use

These untested files should NOT be used in production. They have not been validated through behavioral testing with subagents and may contain gaps, rationalizations, or ineffective guidance.
