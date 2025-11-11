# Approach Summary - generating-architecture-diagrams

## Different Methodology: Codifying Excellence vs Fixing Failures

Unlike previous skills (`using-system-archaeologist`, `analyzing-unknown-codebases`), this skill follows a **POSITIVE baseline** approach.

### Previous Skills (RED-GREEN-REFACTOR)

**Pattern:**
1. RED: Run baselines, identify FAILURES
2. GREEN: Write minimal skill to FIX failures
3. REFACTOR: Close loopholes

**Example:** `analyzing-unknown-codebases`
- Baseline: Contract compliance failures (extra sections, wrong files)
- Skill: Strict contract enforcement, explicit templates
- Result: Transformed violations → perfect compliance

### This Skill (BEST PRACTICES DOCUMENTATION)

**Pattern:**
1. RED: Run baselines, identify STRENGTHS (no failures!)
2. DOCUMENT: Codify observed excellence
3. REFINE: Add conventions and guidance

**Why different?**
- Baselines showed excellent performance (no failures)
- Agent naturally produces high-quality diagrams
- Agent already handles complexity well (abstraction, selection, documentation)

## Baseline Results

### Scenario 1: Complete Catalog (5 subsystems)

**✅ ALL SUCCESS CRITERIA MET:**
- Generated all 3 C4 levels
- Valid Mermaid syntax
- Titles, descriptions, legends present
- Assumptions and limitations documented
- Thoughtful Component diagram selection
- Documented concerns from catalog

**Minor opportunity:** Inferred components (documented in assumptions)

### Scenario 3: Complex System (15 subsystems)

**✅ ADVANCED PERFORMANCE:**
- Faction-based grouping (15 → 6 groups, 60% reduction)
- Metadata enrichment (skill counts)
- Strategic sampling (3/15 for Component diagrams, 20%)
- Architectural diversity in selection
- Clear selection rationale ("Why these, not others")
- Color coding for visual hierarchy
- Notation for relationship types (dotted vs solid)

**No failures identified.**

## Skill Design Decisions

### What to Include

**Codify observed excellence:**
- Abstraction strategies (grouping, metadata, sampling)
- Selection criteria (diversity, scale, critical path)
- Notation conventions (dotted vs solid, color meanings)
- Documentation template (assumptions, limitations, rationale)

**Add lightweight guidance:**
- When to infer vs note as missing
- C4 level selection rules
- Mermaid vs PlantUML choice
- Success criteria

### What to Avoid

**Don't create artificial constraints:**
- Baseline showed flexible, context-appropriate behavior
- Skill reinforces good judgment, doesn't replace it
- No strict "MUST do exactly this" (that's for fixing failures)

**Don't over-specify:**
- Agent already handles complexity well
- Skill provides patterns, not rigid process

## Skill Structure

**288 lines, organized as:**

1. **Purpose and When to Use** (scope setting)
2. **Core Principle** (abstraction over completeness)
3. **Output Contract** (what to produce)
4. **C4 Level Selection** (guidance for each level)
5. **Abstraction Strategies** (grouping, metadata, sampling)
6. **Notation Conventions** (line styles, colors, annotations)
7. **Handling Incomplete Information** (inference guidance)
8. **Documentation Template** (assumptions, limitations)
9. **Mermaid vs PlantUML** (format choice)
10. **Success Criteria** (clear pass/fail)
11. **Best Practices from Baseline** (codified excellence)
12. **Integration with Workflow** (role in process)

## Comparison: Failure-Fixing vs Excellence-Codifying

| Aspect | analyzing-unknown-codebases | generating-architecture-diagrams |
|--------|----------------------------|----------------------------------|
| **Baseline** | Universal contract failures | Universal excellence |
| **Skill length** | 299 lines | 288 lines |
| **Tone** | Strict ("MUST", "DO NOT") | Guidance ("Consider", "Prefer") |
| **Focus** | Contract compliance enforcement | Abstraction strategies documentation |
| **Templates** | Exact contract to copy | Flexible patterns to apply |
| **Rationalizations** | Table of failures to resist | Table of successes to emulate |
| **Self-validation** | 12-item checklist (mandatory) | Success criteria (reference) |

## Testing Approach

**No GREEN phase testing needed** because:
- Baselines already show desired behavior
- Skill codifies what already works
- No failures to verify are fixed

**Instead:**
- Document baseline excellence
- Create skill as best practices guide
- Commit with methodology notes

## Lessons Learned

### When to Use RED-GREEN-REFACTOR

**Use when:** Baselines show consistent failures
- Contract violations
- Missing required elements
- Rationalization patterns
- Systematic errors

### When to Use BEST PRACTICES DOCUMENTATION

**Use when:** Baselines show excellence
- Agent naturally performs well
- Need to codify implicit knowledge
- Add conventions and patterns
- Reinforce good behavior

## Files Created

1. **test-scenarios.md** - 5 test scenarios (same structure as other skills)
2. **baseline-results.md** - Documents excellent performance (not failures)
3. **SKILL.md** - Best practices guide (288 lines)
4. **approach-summary.md** - This file (methodology explanation)

## Commit Strategy

**Different commit message focus:**

Previous commits:
> "feat: Add X skill with TDD validation"
> "Transforms behavior from 'violations' to 'perfect compliance'"

This commit:
> "feat: Add generating-architecture-diagrams skill documenting best practices"
> "Codifies observed excellence in abstraction, selection, and documentation"

**Emphasizes:** Documentation of strengths, not fixing of weaknesses.
