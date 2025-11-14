# Writing-Skills Methodology Analysis

**Date:** 2025-11-14
**Source:** https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md
**Purpose:** Learn from obra's writing-skills approach to improve our own skill creation

---

## Executive Summary

Obra's `writing-skills` skill applies **Test-Driven Development (TDD)** principles to skill creation:
- **RED**: Run baseline scenarios WITHOUT the skill; document exact failures
- **GREEN**: Write minimal skill documentation addressing those violations
- **REFACTOR**: Close loopholes through iterative testing

**The Iron Law**: "If you didn't watch an agent fail without the skill, you don't know if the skill teaches the right thing."

---

## Core Principles from writing-skills

### 1. TDD for Skills (RED-GREEN-REFACTOR)

**RED Phase:**
- Run "pressure scenarios" without the skill
- Document EXACT agent failures (not hypothetical)
- Capture specific violations, workarounds, rationalizations

**GREEN Phase:**
- Write MINIMAL skill documentation addressing those failures
- Include only what's needed to fix observed violations
- No speculative content

**REFACTOR Phase:**
- Iteratively test and refine
- Close loopholes as agents find workarounds
- Capture new rationalizations

**Why this matters:**
- Skills address REAL failures, not imagined ones
- Prevents over-documentation (bloat)
- Ensures skills actually change behavior

### 2. Structure Requirements

**YAML Frontmatter:**
```yaml
---
name: skill-name
description: Use when [concrete trigger] - [symptom] - [situation]
---
```

**Required Sections:**
1. **Overview** - Core principle (1-2 paragraphs)
2. **When to use** - Decision flowchart if not obvious
3. **Quick reference** - Fast lookups during execution
4. **Real examples** - From actual baseline testing
5. **Common mistakes** - Anti-patterns from failures

**Optional Sections:**
- Deep dives for complex skills
- External references (links, not inline content)

### 3. Description Field Rules

**MUST:**
- Start with "Use when..."
- Include concrete triggers (events that signal need)
- Include symptoms (observable problems)
- Include situations (contexts)
- Write in third person
- Be specific, not vague

**Examples:**

❌ **Bad (our current):**
`"Systematic process for fixing lint warnings without disabling them or over-refactoring"`

✅ **Good (writing-skills style):**
`"Use when codebase has 50+ lint warnings and team wants standards-compliant code without disabling rules or risky refactoring"`

❌ **Bad (vague):**
`"Router for systems thinking methodology - patterns, leverage points, archetypes..."`

✅ **Good (concrete):**
`"Use when problems persist despite fixes, solutions create new problems, or system behavior is counter-intuitive - routes to appropriate systems analysis skills"`

### 4. Token Efficiency Targets

| Skill Type | Target Length | Rationale |
|------------|---------------|-----------|
| Getting-started skills | <150 words | Frequently loaded, must be fast |
| Frequently-loaded skills | <200 words | Balance detail vs speed |
| Specialist skills | <500 words | Deep content, infrequent load |
| Reference guides | Exception | Heavy reference material (>100 lines) |

**Why efficiency matters:**
- Faster skill discovery
- Lower context usage
- Agents process faster
- Reduces cognitive load

### 5. Bulletproofing Discipline Skills

**For skills that enforce rules** (like our systematic-delinting):

✅ **Do:**
- Explicitly forbid specific workarounds
- Address spirit-versus-letter arguments
- Capture all rationalizations agents attempt
- Create "Rationalization Resistance Table"

❌ **Don't:**
- Leave loopholes open
- Assume agents will "get the spirit"
- Use vague prohibitions

**Example from writing-skills:**
> "The Iron Law: No skill without a failing test first—no exceptions for 'simple additions' or 'just documentation updates.'"

### 6. File Organization

**Keep inline:**
- Content under 100 lines
- Frequently referenced material
- Core skill content

**Externalize:**
- Heavy reference material (>100 lines)
- Reusable tools/templates
- Supplementary examples

**Why:** Balance completeness with loadability

---

## Our Current Approach vs writing-skills

### Comparison Table

| Aspect | writing-skills | Our Current | Gap Analysis |
|--------|---------------|-------------|--------------|
| **TDD Methodology** | Mandatory RED-GREEN-REFACTOR | Historically used, artifacts removed | ✅ We did this (v1.0.0 mentions TDD validation) |
| **Description Format** | "Use when..." + triggers | Descriptive statements | ❌ Need to reformat |
| **Token Efficiency** | <150/<200/<500 words | 500-2000 lines | ❌ Major gap - our skills are VERBOSE |
| **Structure** | Minimal, focused | Comprehensive, encyclopedic | ⚠️ Trade-off: depth vs speed |
| **Bulletproofing** | Explicit rationalization tables | Some skills have this | ✅ systematic-delinting has this |
| **When-to-use Section** | Required flowchart | Present in most skills | ✅ We have this |
| **Anti-patterns** | Required | Present in many skills | ✅ We have this |
| **Quick Reference** | Required | Present in most skills | ✅ We have this |

### Strengths of Our Approach

✅ **Comprehensive Coverage:**
- Our skills are DEEP (e.g., systematic-delinting is 1525 lines)
- Encyclopedic reference material
- Real implementation examples
- Multiple scenarios covered

✅ **Bulletproofing (in some skills):**
- Rationalization resistance tables (using-systems-thinking)
- Explicit anti-patterns
- Red flags checklists

✅ **Routing Architecture:**
- Meta-router skills guide users to specialists
- Clear dependencies and workflows
- Decision trees for skill selection

✅ **Production-Ready:**
- Not tutorials - actual professional guidance
- Assumes competence, teaches mastery
- Pattern-focused, reusable

### Weaknesses of Our Approach

❌ **Token Inefficiency:**
- systematic-delinting: 1525 lines (~12,000 words) vs target 500 words
- using-systems-thinking: 470 lines (~3,500 words) vs target 200 words
- **10-60x over target length**

❌ **Description Format:**
- Current: "Systematic process for fixing lint warnings..."
- Should be: "Use when codebase has 50+ lint warnings..."
- Not optimized for skill discovery

❌ **Loading Speed:**
- Large skills = slow skill discovery
- High context usage
- Slower agent processing

❌ **Unclear Minimal Viable Content:**
- Hard to tell what's ESSENTIAL vs NICE-TO-HAVE
- Risk of including speculative content (not from failures)

---

## Reconciling the Approaches

### The Tension

**writing-skills philosophy:**
"Minimal documentation addressing observed failures"
→ Fast, focused, efficient

**Our philosophy:**
"Comprehensive professional reference guides"
→ Deep, encyclopedic, production-ready

**Can we have both?**

### Proposed Hybrid Model

**Two-Tier Structure:**

#### Tier 1: Core Skill (SKILL.md)
- **Target:** <500 words (following writing-skills)
- **Content:**
  - Overview (core principle)
  - When to use (flowchart)
  - Quick reference (tables, commands)
  - Common mistakes (anti-patterns)
  - Link to deep dive guide
- **Purpose:** Fast loading, skill discovery, essential guidance

#### Tier 2: Deep Dive Guide (GUIDE.md or docs/)
- **Target:** No limit
- **Content:**
  - Comprehensive examples
  - Extended scenarios
  - Deep technical details
  - Case studies
  - Integration patterns
- **Purpose:** Reference material for deep work

**Example Split:**

**systematic-delinting/SKILL.md** (~400 words):
```yaml
---
name: systematic-delinting
description: Use when codebase has 50+ lint warnings - enables systematic fix without disabling rules or risky refactoring
---

## Overview
Fix warnings systematically, NEVER disable them. Delinting ≠ refactoring.

## When to Use
- Codebase has 50+ lint warnings
- Inheriting legacy code
- Enabling strict linting on existing projects

## Quick Reference

### Essential Commands
ruff check . --statistics        # Baseline
ruff check --select F401 --fix   # Fix specific rule
...

### Fix-Never-Disable Rule
❌ WRONG: `# noqa` comments
✅ CORRECT: Fix code to comply

### Delinting ≠ Refactoring
❌ WRONG: Changing architecture during delinting
✅ CORRECT: Minimal changes to satisfy linter

## Common Mistakes
1. Disabling instead of fixing
2. Over-refactoring during delinting
3. Fixing everything at once

See GUIDE.md for complete workflows, 50+ rule fixes, CI integration.
```

**systematic-delinting/GUIDE.md** (~1500 lines):
- Full Ruff configuration examples
- 50+ specific rule fixes with code examples
- Complete CI integration workflows
- Team adoption strategies
- Progress tracking implementation

**Benefits:**
- Fast initial load (Tier 1)
- Deep reference when needed (Tier 2)
- Clear separation: essentials vs comprehensive
- Maintains professional depth
- Improves skill discovery speed

---

## Applying writing-skills Principles to Our Future Skills

### Skill Creation Workflow

**1. RED Phase (Baseline Testing)**
```bash
# Create pressure scenario WITHOUT skill
# Document exact agent failures:
- What did the agent do wrong?
- What rationalizations did it use?
- What workarounds did it attempt?
- What violations occurred?

# Save to: tests/baseline-failures-[skill-name].md
```

**2. GREEN Phase (Minimal Skill)**
```markdown
# Write ONLY what addresses observed failures
# Target: <500 words for specialists, <200 for routers
# Structure:
## Overview (core principle from failures)
## When to Use (triggers from baseline scenarios)
## Quick Reference (solutions to observed violations)
## Common Mistakes (from baseline failures)
```

**3. REFACTOR Phase (Iterative Testing)**
```bash
# Test skill with agent
# Document new failures, rationalizations, loopholes
# Update skill to close gaps
# Repeat until no violations

# Save iterations to: tests/refactor-log-[skill-name].md
```

**4. DEEP DIVE Phase (Optional)**
```bash
# If skill needs >500 words:
# Create GUIDE.md with comprehensive content
# Keep SKILL.md under 500 words
# Link from SKILL.md to GUIDE.md
```

### Description Writing Template

```yaml
---
name: [skill-name]
description: Use when [TRIGGER] - [SYMPTOM] - [SITUATION]
---
```

**Fill in:**
- **TRIGGER**: Observable event that signals need (e.g., "codebase has 50+ lint warnings")
- **SYMPTOM**: Problem being experienced (e.g., "team pushback on linting standards")
- **SITUATION**: Context/constraints (e.g., "without disabling rules or risky refactoring")

**Examples:**

✅ `"Use when codebase has 50+ lint warnings - team wants standards without disabling rules or risky refactoring"`

✅ `"Use when problems persist despite fixes - solutions create new problems - system behavior counter-intuitive"`

✅ `"Use when training RL agents - policy gradient methods needed - for continuous action spaces or stochastic policies"`

### Bulletproofing Checklist

For discipline skills (those enforcing rules):

- [ ] Identify all rationalizations from baseline testing
- [ ] Create "Rationalization Resistance Table"
- [ ] Explicitly forbid specific workarounds
- [ ] Address spirit-versus-letter arguments
- [ ] Include "Red Flags Checklist"
- [ ] Test against "wriggle attempts"

**Example Rationalization Table:**

| Rationalization | Reality | Counter-Guidance | Red Flag |
|-----------------|---------|------------------|----------|
| "Just add # noqa" | Disabling accumulates as debt | "Fix code, never disable" | Avoiding compliance |
| "This is refactoring" | Delinting ≠ refactoring | "Minimal changes only" | Scope creep |

---

## Recommended Changes to Our Skillpacks

### Immediate Actions (High Priority)

1. **Update All Descriptions** (1-2 hours)
   - Reformat to "Use when..." pattern
   - Add triggers, symptoms, situations
   - Update marketplace.json

2. **Create Two-Tier Structure Template** (2 hours)
   - Define SKILL.md template (<500 words)
   - Define GUIDE.md template (unlimited)
   - Document split criteria

3. **Identify Verbose Skills for Splitting** (1 hour)
   - List skills >500 words
   - Prioritize frequently-loaded skills
   - Create split plan

### Medium-Term Actions (Next Skills)

4. **Apply RED-GREEN-REFACTOR to New Skills** (ongoing)
   - Document baseline failures before writing
   - Keep test artifacts in `tests/` directory
   - Iterate to close loopholes

5. **Split Top 10 Verbose Skills** (10-20 hours)
   - systematic-delinting → SKILL.md (400 words) + GUIDE.md (1500 lines)
   - using-systems-thinking → SKILL.md (200 words) + GUIDE.md (400 lines)
   - [others TBD]

6. **Document TDD Process** (2 hours)
   - Create CONTRIBUTING-SKILLS.md
   - Include RED-GREEN-REFACTOR workflow
   - Add baseline testing templates

### Long-Term Strategy

7. **Measure Skill Performance** (ongoing)
   - Track skill loading times
   - Monitor context usage
   - Collect agent feedback on clarity

8. **Create Skill Quality Metrics** (future)
   - Token efficiency score
   - Bulletproofing completeness
   - TDD validation status

---

## Key Takeaways

### What We're Doing Right

✅ **TDD Validation** - We did this historically (v1.0.0 cleanup)
✅ **Bulletproofing** - Some skills have rationalization resistance
✅ **Anti-patterns** - Present in most skills
✅ **Production-Ready** - Professional depth and quality
✅ **Routing Architecture** - Meta-routers guide to specialists

### What We Should Adopt from writing-skills

🎯 **"Use when..." Description Format** - Improves skill discovery
🎯 **Token Efficiency Targets** - 10-60x reduction needed
🎯 **Two-Tier Structure** - Core skill + deep dive guide
🎯 **Explicit TDD Artifacts** - Keep baseline failures, refactor logs
🎯 **Bulletproofing Tables** - Standardize across all discipline skills

### What We Should Keep from Our Approach

✅ **Comprehensive Depth** - Move to GUIDE.md, not delete
✅ **Professional Quality** - Expert-level, not tutorial
✅ **Real Examples** - Code samples, scenarios
✅ **Faction Organization** - Thematic coherence

---

## Next Steps

1. **Decide:** Accept two-tier SKILL.md + GUIDE.md model?
2. **Prototype:** Split 1-2 skills as proof of concept
3. **Measure:** Compare loading times, context usage
4. **Iterate:** Refine based on results
5. **Document:** Update CONTRIBUTING.md with new standards
6. **Scale:** Apply to all 144 skills over time

---

## Appendix: Example Skill Comparison

### Current: systematic-delinting/SKILL.md (1525 lines)

**Pros:**
- Comprehensive Ruff configuration examples
- 50+ rule-specific fixes
- Complete CI integration workflows
- Team adoption strategies
- Progress tracking implementation

**Cons:**
- 1525 lines (~12,000 words) vs target 500 words
- **24x over target**
- Slow to load in skill discovery
- High context usage
- Hard to extract "essentials"

### Proposed: Two-Tier Split

**systematic-delinting/SKILL.md** (~400 words, ~50 lines):
- Overview: Fix-never-disable, delinting≠refactoring
- When to use: 50+ warnings, legacy code
- Quick reference: 10 essential commands, 5 key rules
- Common mistakes: 5 anti-patterns
- Link: "See GUIDE.md for complete workflows"

**systematic-delinting/GUIDE.md** (~1500 lines):
- Full Ruff/Pylint configuration
- 50+ rule-specific fixes
- Complete CI workflows
- Team adoption playbook
- Progress tracking code

**Result:**
- 12x reduction in core skill size
- Maintains comprehensive depth in GUIDE.md
- Faster skill discovery
- Lower context usage
- Clear essentials vs reference split

---

## Questions for Discussion

1. **Should we adopt the two-tier SKILL.md + GUIDE.md model?**
   - Pro: Dramatically faster skill discovery
   - Con: Effort to split 144 existing skills

2. **What token efficiency target should we use?**
   - writing-skills: 150/200/500 words
   - Our current: 500-2000 lines (4000-16000 words)
   - Proposed: 200/500/1000 words with unlimited GUIDE.md?

3. **Should we keep TDD artifacts in the repo?**
   - Pro: Transparency, reproducibility
   - Con: Clutter (we removed them in v1.0.0)
   - Compromise: Keep in `tests/` directory, not alongside skills?

4. **Priority for splits?**
   - Frequently-loaded routers first?
   - Largest skills first?
   - New skills only?

5. **Should all skills have bulletproofing tables?**
   - Only discipline skills?
   - All skills for consistency?
   - Case-by-case basis?

---

## References

- **obra/superpowers writing-skills**: https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md
- **Our CLAUDE.md**: /home/user/skillpacks/CLAUDE.md
- **Our marketplace**: 17 plugins, 144 skills, v1.7.0
- **TDD history**: v1.0.0 cleanup removed 513 internal files including test artifacts
