# Reviewing Pack Structure

**Purpose:** Pass 1 - Analyze pack organization, identify structural issues, generate fitness scorecard.

## Inputs

From `analyzing-pack-domain.md`:
- Domain coverage map (what should exist)
- Current skill inventory (what does exist)
- Gap analysis (missing/duplicate/obsolete)
- Research currency flag

## Analysis Tasks

### 1. Fitness Scorecard

Generate scorecard with risk-driven prioritization:

**Critical Issues** - Pack unusable or fundamentally broken:
- Missing core foundational concepts (users can't understand basics)
- Major gaps in essential coverage (50%+ of core techniques missing)
- Router completely inaccurate or missing when needed
- Multiple skills broken or contradictory

**Decision:** Critical issues → Recommend "Rebuild from scratch" vs. "Enhance existing"

Rebuild if:
- More skills missing than exist
- Fundamental philosophy is wrong
- Faction mismatch is severe

Enhance if:
- Core structure is sound, just needs additions/fixes
- Most existing skills are salvageable

**Major Issues** - Significant effectiveness reduction:
- Important gaps in core coverage (20-50% of core techniques missing)
- Multiple duplicate skills causing confusion
- Obsolete skills teaching deprecated patterns
- Faction drift across multiple skills
- Metadata significantly out of sync

**Minor Issues** - Polish and improvements:
- Small gaps in advanced topics
- Minor organizational inconsistencies
- Router descriptions slightly outdated
- Small metadata corrections needed

**Pass** - Structurally sound:
- Comprehensive coverage of foundational and core areas
- No major gaps or duplicates
- Router (if exists) is accurate
- Faction alignment is good
- Metadata is current

**Output:** Scorecard with category and specific issues listed

### 2. Gap Identification

From coverage map, identify missing skills:

**Prioritize by importance:**

**High priority (foundational/core):**
- Foundational concepts users must understand
- Core techniques used frequently
- Common patterns missing from basics

**Medium priority (advanced):**
- Advanced topics for expert users
- Specialized techniques
- Edge case handling

**Low priority (nice-to-have):**
- Rare patterns
- Future-looking topics
- Experimental techniques

**For each gap:**
- Draft skill name (following naming conventions)
- Write description (following CSO guidelines)
- Estimate scope (small/medium/large skill)
- Note dependencies (what skills should be read first)

**Output:** Prioritized list of gaps with draft names/descriptions

### 3. Duplicate Detection

Find skills with overlapping coverage:

**Analysis process:**
1. Read all skill descriptions
2. Identify content overlap (skills covering same concepts)
3. Read overlapping skills to assess actual content
4. Determine relationship

**Duplicate types:**

**Complete duplicates** - Same content, different names:
- **Action:** Remove one, preserve unique value from both

**Partial overlap** - Significant shared content:
- **Action:** Merge into single comprehensive skill

**Specialization** - One general, one specific:
- **Action:** Keep both, clarify relationship via cross-references
- Example: "async-patterns" (general) + "asyncio-taskgroup" (specific)

**Complementary** - Different angles on same topic:
- **Action:** Keep both, strengthen cross-references
- Example: "testing-async-code" + "async-patterns-and-concurrency"

**False positive** - Similar names, different content:
- **Action:** No change, maybe clarify descriptions

**For each duplicate pair:**
- Classification (complete/partial/specialization/complementary/false)
- Recommendation (remove/merge/keep with cross-refs)
- Preserve unique value from each

**Output:** Duplicate analysis with recommendations

### 4. Organization Validation

Check pack-level organization:

**Router skill validation (if exists):**
- Does router list all current specialist skills?
- Are descriptions in router accurate?
- Does routing logic make sense?
- Are there skills NOT mentioned in router?
- Are there router entries for NON-EXISTENT skills?

**Faction alignment:**
- Read FACTIONS.md for this pack's faction principles
- Check 3-5 representative skills for voice/philosophy
- Identify drift patterns
- Severity: Minor (style drift) / Major (wrong philosophy)

**Metadata validation:**
- plugin.json description matches actual content?
- Skill count is accurate?
- Category is appropriate?
- Version reflects current state?

**Navigation experience:**
- Can users find appropriate skills easily?
- Are skill names descriptive?
- Are descriptions helpful for discovery?

**Output:** Organization issues with severity

## Generate Complete Report

Combine all analyses:

```
# Structural Review: [pack-name]

## Scorecard: [Critical / Major / Minor / Pass]

[If Critical]
Recommendation: [Rebuild from scratch / Enhance existing]
Rationale: [Specific reasons]

## Issues by Priority

### Critical Issues ([count])
- [Issue 1] - [Description]
- [Issue 2] - [Description]

### Major Issues ([count])
- [Issue 1] - [Description]
- [Issue 2] - [Description]

### Minor Issues ([count])
- [Issue 1] - [Description]
- [Issue 2] - [Description]

## Gap Analysis

### High Priority Gaps ([count])
- [Gap 1]
  - Skill name: [proposed-name]
  - Description: [draft description]
  - Scope: [small/medium/large]
  - Dependencies: [prerequisites]

### Medium Priority Gaps ([count])
[Same format]

### Low Priority Gaps ([count])
[Same format]

## Duplicate Analysis

- [Skill A] + [Skill B]
  - Type: [complete/partial/specialization/complementary]
  - Recommendation: [remove/merge/keep with cross-refs]
  - Rationale: [why]

## Organization Issues

### Router ([issues count])
- [Issue description]

### Faction Alignment ([severity])
- [Drift pattern]
- Affected skills: [list]

### Metadata ([issues count])
- [Issue description]

## Recommended Actions

Structure fixes:
- Add: [count] new skills
- Remove: [count] duplicate skills
- Merge: [count] partial duplicates
- Update router: [Yes/No]
```

## Decision Gate

Present scorecard and report to user:

**If Critical:**
- Explain rebuild vs. enhance trade-offs
- Get user decision before proceeding

**If Major/Minor/Pass:**
- Present findings
- Confirm user wants to proceed with Pass 2 (content testing)

## Proceeding to Next Stage

After scorecard approval:
- If proceeding → Move to `testing-skill-quality.md` (Pass 2)
- If rebuilding → Stop maintenance workflow, switch to creation workflow
- If canceling → Stop workflow

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Scorecard too lenient | Be honest: missing 50% of core = Critical |
| Vague gap descriptions | Draft actual skill names and descriptions |
| Keeping all duplicates | Duplicates confuse users - merge or remove |
| Ignoring faction drift | Faction identity matters - flag misalignment |
| Skipping metadata check | Inaccurate metadata breaks discovery |
