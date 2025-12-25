---
name: using-skillpack-maintenance
description: Use when maintaining or enhancing existing skill packs in the skillpacks repository - systematic pack refresh through domain analysis, structure review, RED-GREEN-REFACTOR gauntlet testing, and automated quality improvements
---

# Skillpack Maintenance

## Overview

Systematic maintenance and enhancement of existing skill packs using investigative domain analysis, RED-GREEN-REFACTOR testing, and automated improvements.

**Core principle:** Maintenance uses behavioral testing (gauntlet with subagents), not syntactic validation. Skills are process documentation - test if they guide agents correctly, not if they parse correctly.

## When to Use

Use when:
- Enhancing an existing skill pack (e.g., "refresh yzmir-deep-rl")
- Improving existing SKILL.md files
- Identifying gaps in pack coverage
- Validating skill quality through testing

**Do NOT use for:**
- Creating new skills from scratch (use superpowers:writing-skills)
- Creating new packs from scratch (design first, then use creation workflow)

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-skillpack-maintenance/SKILL.md`

Reference sheets like `analyzing-pack-domain.md` are at:
  `skills/using-skillpack-maintenance/analyzing-pack-domain.md`

NOT at:
  `skills/analyzing-pack-domain.md` ← WRONG PATH

When you see a link like `[analyzing-pack-domain.md](analyzing-pack-domain.md)`, read the file from the same directory as this SKILL.md.

---

## The Iron Law

**NO SKILL CHANGES WITHOUT BEHAVIORAL TESTING**

Syntactic validation (does it parse?) ≠ Behavioral testing (does it work?)

## Common Rationalizations (from baseline testing)

| Excuse | Reality |
|--------|---------|
| "Syntactic validation is sufficient" | Parsing ≠ effectiveness. Test with subagents. |
| "Quality benchmarking = effectiveness" | Comparing structure ≠ testing behavior. Run gauntlet. |
| "Comprehensive coverage = working skill" | Coverage ≠ guidance quality. Test if agents follow it. |
| "Following patterns = success" | Pattern-matching ≠ validation. Behavioral testing required. |
| "I'll test if issues emerge" | Issues = broken skills in production. Test BEFORE deploying. |

**All of these mean: Run behavioral tests with subagents. No exceptions.**

## Workflow Overview

**Review → Discuss → [Create New Skills if Needed] → Execute**

1. **Investigation & Scorecard** → Load `analyzing-pack-domain.md`
2. **Structure Review (Pass 1)** → Load `reviewing-pack-structure.md`
3. **Content Testing (Pass 2)** → Load `testing-skill-quality.md`
4. **Coherence Check (Pass 3)** → Validate cross-skill consistency
5. **Discussion** → Present findings, get approval
6. **[CONDITIONAL] Create New Skills** → If gaps identified, use `superpowers:writing-skills` for EACH gap (RED-GREEN-REFACTOR)
7. **Execution** → Load `implementing-fixes.md`, enhance existing skills only
8. **Commit** → Single commit with version bump

## Stage 1: Investigation & Scorecard

**Load briefing:** `analyzing-pack-domain.md`

**Purpose:** Establish "what this pack should cover" from first principles.

**Adaptive investigation (D→B→C→A):**
1. **User-guided scope (D)** - Ask user about pack intent and boundaries
2. **LLM knowledge analysis (B)** - Map domain comprehensively, flag if research needed
3. **Existing pack audit (C)** - Compare current state vs. coverage map
4. **Research if needed (A)** - Conditional: only if domain is rapidly evolving

**Output:** Domain coverage map, gap analysis, research currency flag

**Then: Load `reviewing-pack-structure.md` for scorecard**

**Scorecard levels:**
- **Critical** - Pack unusable, recommend rebuild vs. enhance
- **Major** - Significant gaps or duplicates
- **Minor** - Organizational improvements
- **Pass** - Structurally sound

**Decision gate:** Present scorecard → User decides: Proceed / Rebuild / Cancel

## Stage 2: Comprehensive Review

### Pass 1: Structure (from reviewing-pack-structure.md)

**Analyze:**
- Gaps (missing skills based on coverage map)
- Duplicates (overlapping coverage - merge/specialize/remove)
- Organization (router accuracy, faction alignment, metadata sync)

**Output:** Structural issues with priorities (critical/major/minor)

### Pass 2: Content Quality (from testing-skill-quality.md)

**CRITICAL:** This is behavioral testing with subagents, not syntactic validation.

**Gauntlet design (A→C→B priority):**

**A. Pressure scenarios** - Catch rationalizations:
- Time pressure: "This is urgent, just do it quickly"
- Simplicity temptation: "Too simple to need the skill"
- Overkill perception: "Skill is for complex cases, this is straightforward"

**C. Adversarial edge cases** - Test robustness:
- Corner cases where skill principles conflict
- Situations where naive application fails

**B. Real-world complexity** - Validate utility:
- Messy requirements, unclear constraints
- Multiple valid approaches

**Testing process per skill:**
1. Design challenging scenario from gauntlet categories
2. **Run subagent WITH current skill** (behavioral test)
3. Observe: Does it follow? Where does it rationalize/fail?
4. Document failure modes
5. Result: Pass OR Fix needed (with specific issues listed)

**Philosophy:** D as gauntlet to identify issues, B for targeted fixes. If skill passes gauntlet, no changes needed.

**Output:** Per-skill test results (Pass / Fix needed + priorities)

### Pass 3: Coherence

**After structure/content analysis, validate pack-level coherence:**

1. **Cross-skill consistency** - Terminology, examples, cross-references
2. **Router accuracy** - Does using-X router reflect current specialists?
3. **Faction alignment** - Check FACTIONS.md, flag drift, suggest rehoming if needed
4. **Metadata sync** - plugin.json description, skill count
5. **Navigation** - Can users find skills easily?

**CRITICAL:** Update skills to reference new/enhanced skills (post-update hygiene)

**Output:** Coherence issues, faction drift flags

## Stage 3: Interactive Discussion

**Present findings conversationally:**

**Structural category:**
- **Gaps requiring superpowers:writing-skills** (new skills needed - each requires RED-GREEN-REFACTOR)
- Duplicates to remove/merge
- Organization issues

**Content category:**
- Skills needing enhancement (from gauntlet failures)
- Severity levels (critical/major/minor)
- Specific failure modes identified

**Coherence category:**
- Cross-reference updates needed
- Faction alignment issues
- Metadata corrections

**Get user approval for scope of work**

**CRITICAL DECISION POINT:** If gaps (new skills) were identified:
- User approves → **IMMEDIATELY use superpowers:writing-skills for EACH gap**
- Do NOT proceed to Stage 4 until ALL new skills are created and tested
- Each gap = separate RED-GREEN-REFACTOR cycle
- Return to Stage 4 only after ALL gaps are filled

## Stage 4: Autonomous Execution

**Load briefing:** `implementing-fixes.md`

**PREREQUISITE CHECK:**
- ✓ Zero gaps identified, OR
- ✓ All gaps already filled using superpowers:writing-skills (each skill individually tested)

**If gaps exist and you haven't used writing-skills:** STOP. Return to Stage 3.

**Execute approved changes:**

1. **Structural fixes** - Remove/merge duplicate skills, update router
2. **Content enhancements** - Fix gauntlet failures, add missing guidance to existing skills
3. **Coherence improvements** - Cross-references, terminology alignment, faction voice
4. **Version management** - Apply impact-based bump (patch/minor/major)
5. **Git commit** - Single commit with all changes

**Version bump rules (impact-based):**
- **Patch (x.y.Z)** - Low-impact: typos, formatting, minor clarifications
- **Minor (x.Y.0)** - Medium-impact: enhanced guidance, new skills, better examples (DEFAULT)
- **Major (X.0.0)** - High-impact: skills removed, structural changes, philosophy shifts (RARE)

**Commit format:**
```
feat(meta): enhance [pack-name] - [summary]

[Detailed list of changes by category]
- Structure: [changes]
- Content: [changes]
- Coherence: [changes]

Version bump: [reason for patch/minor/major]
```

**Output:** Enhanced pack, commit created, summary report

## Briefing Files Reference

All briefing files are in this skill directory:

- `analyzing-pack-domain.md` - Investigative domain analysis (D→B→C→A)
- `reviewing-pack-structure.md` - Structure review, scorecard, gap/duplicate analysis
- `testing-skill-quality.md` - Gauntlet testing methodology with subagents
- `implementing-fixes.md` - Autonomous execution, version management, git commit

**Load appropriate briefing at each stage.**

## Critical Distinctions

**Behavioral vs. Syntactic Testing:**
- ❌ **Syntactic:** "Does Python code parse?" → ast.parse()
- ✅ **Behavioral:** "Does skill guide agents correctly?" → Subagent gauntlet

**This workflow requires BEHAVIORAL testing.**

**Maintenance vs. Creation:**
- **Maintenance** (this skill): Enhancing existing SKILL.md files
- **Creation** (superpowers:writing-skills): Writing new skills from scratch

**Use the right tool for the task.**

## Red Flags - STOP and Switch Tools

If you catch yourself thinking ANY of these:
- "I'll write the new skills during execution" → NO. Use superpowers:writing-skills for EACH gap
- "implementing-fixes.md says to create skills" → NO. That section was REMOVED. Exit and use writing-skills
- "Token efficiency - I can just write good skills" → NO. Untested skills = broken skills
- "I see the pattern, I can replicate it" → NO. Pattern-matching ≠ behavioral testing
- "User wants this done quickly" → NO. Fast + untested = waste of time fixing later
- "I'm competent, testing is overkill" → NO. Competence = following the process
- "Gaps were approved, so I should fill them" → YES, but using writing-skills, not here
- Validating syntax instead of behavior → Load testing-skill-quality.md
- Skipping gauntlet testing → You're violating the Iron Law
- Making changes without user approval → Follow Review→Discuss→Execute

**All of these mean: STOP. Exit workflow. Use superpowers:writing-skills.**

## The Bottom Line

**Maintaining skills requires behavioral testing, not syntactic validation.**

Same principle as code: you test behavior, not syntax.

Load briefings at each stage. Test with subagents. Get approval. Execute.

No shortcuts. No rationalizations.
