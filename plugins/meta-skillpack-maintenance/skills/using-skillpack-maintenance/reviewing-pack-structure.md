# Reviewing Plugin Structure

**Purpose:** Analyze plugin organization across all component types and generate fitness scorecard.

## Inputs

From `analyzing-pack-domain.md`:
- Domain coverage map
- Component inventory (skills, commands, agents, hooks)
- Gap analysis
- Research currency flag

---

## Fitness Scorecard

### Critical Issues (Plugin Unusable)

- Missing core foundational concepts (>50% of coverage map gaps)
- Multiple components broken or contradictory
- Router completely inaccurate
- Component types misaligned (commands that should be skills, etc.)

**Decision:** Critical → Recommend "Rebuild" vs. "Enhance"

Rebuild if:
- More components missing than exist
- Fundamental philosophy wrong
- Severe faction mismatch

### Major Issues (Significant Effectiveness Reduction)

- Important gaps in coverage (20-50% missing)
- Multiple duplicate components
- Obsolete components teaching deprecated patterns
- Commands/agents/skills with wrong scope boundaries
- Hooks with incorrect event types

### Minor Issues (Polish)

- Small gaps in advanced topics
- Minor organizational inconsistencies
- Metadata slightly outdated
- Cross-references missing

### Pass (Structurally Sound)

- Comprehensive coverage
- No major gaps or duplicates
- Components appropriately typed
- Metadata current

---

## Analysis by Component Type

### Skills Analysis

**Check for:**
- Description triggers correct activation?
- Gaps in domain coverage?
- Router skill (if exists) references all specialists?
- Reference sheets comprehensive?

**Duplicates:**
| Type | Action |
|------|--------|
| Complete duplicate | Remove one, merge unique value |
| Partial overlap | Merge or clarify boundaries |
| Specialization | Keep both, add cross-references |
| Complementary | Keep both, strengthen links |

### Commands Analysis

**Check for:**
- Clear user-invocable actions?
- Appropriate tool restrictions?
- argument-hint helpful?
- Overlaps with skills? (commands should be explicit actions, not auto-invoked guidance)

**Common issues:**
| Issue | Fix |
|-------|-----|
| Command duplicates skill | Remove command OR convert skill to reference sheet |
| Missing argument-hint | Add for commands with arguments |
| Too permissive tools | Restrict to minimum needed |

### Agents Analysis

**Check for:**
- Clear scope boundaries (does/doesn't do)?
- Appropriate model selection?
- Activation examples (positive AND negative)?
- Overlaps with other agents?

**Model selection guide:**
| Complexity | Model |
|------------|-------|
| Quick, focused tasks | haiku |
| Most agent work | sonnet |
| Complex reasoning | opus |

### Hooks Analysis

**Check for:**
- Correct event type for use case?
- Matcher patterns accurate?
- Scripts executable?
- Conflicts between hooks?

**Event selection:**
| Goal | Event |
|------|-------|
| Validate before action | PreToolUse |
| React after action | PostToolUse |
| Inject context at prompt | UserPromptSubmit |
| Load context at session | SessionStart |
| Cleanup | SessionEnd |

---

## Gap Identification

For each gap from coverage map:

1. **Determine component type:**
   - Model auto-invokes? → Skill
   - User explicitly triggers? → Command
   - Autonomous specialist? → Agent
   - Automated on events? → Hook

2. **Draft specification:**
   ```
   Gap: [description]
   Recommended: [skill/command/agent/hook]
   Name: [proposed-name]
   Description: [draft frontmatter description]
   Scope: [small/medium/large]
   Dependencies: [prerequisites]
   ```

3. **Prioritize:**
   - **High** - Foundational/core, blocks users
   - **Medium** - Advanced, nice to have
   - **Low** - Edge cases, rare patterns

---

## Output Format

```markdown
# Structural Review: [plugin-name]

## Scorecard: [Critical / Major / Minor / Pass]

[If Critical]
**Recommendation:** [Rebuild / Enhance]
**Rationale:** [Specific reasons]

## Issues by Priority

### Critical ([count])
- [Issue] - [Description]

### Major ([count])
- [Issue] - [Description]

### Minor ([count])
- [Issue] - [Description]

## Component Analysis

### Skills ([count] existing, [count] gaps)
| Status | Skill | Issue |
|--------|-------|-------|
| ✓ | [name] | OK |
| ⚠ | [name] | [Issue] |
| ✗ | [gap] | Missing - [recommendation] |

### Commands ([count] existing, [count] gaps)
| Status | Command | Issue |
|--------|---------|-------|
| ✓ | /[name] | OK |
| ⚠ | /[name] | [Issue] |
| ✗ | /[gap] | Missing - [recommendation] |

### Agents ([count] existing, [count] gaps)
| Status | Agent | Issue |
|--------|-------|-------|
| ✓ | [name] | OK |
| ⚠ | [name] | [Issue] |
| ✗ | [gap] | Missing - [recommendation] |

### Hooks
| Status | Event | Matcher | Issue |
|--------|-------|---------|-------|
| ✓ | [event] | [pattern] | OK |
| ⚠ | [event] | [pattern] | [Issue] |

## Duplicates/Overlaps

- [Component A] + [Component B]
  - Type: [complete/partial/specialization]
  - Action: [remove/merge/keep with cross-refs]

## Recommended Actions Summary

**New skills needed:** [count] (each requires superpowers:writing-skills)
**New commands needed:** [count]
**New agents needed:** [count]
**Hook changes needed:** [count]
**Components to remove:** [count]
**Components to enhance:** [count]
```

---

## Red Flags - Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "The gaps are minor" | If coverage map says important, it's not minor |
| "We can add commands later" | Incomplete plugin = frustrated users now |
| "The skills are good enough" | Good enough ≠ behavioral testing passed |
| "This duplicate is fine" | Duplicates confuse users and waste context |
| "The metadata doesn't matter" | Inaccurate metadata breaks discovery |
| "I know this domain, no audit needed" | Expertise ≠ systematic review |

**All of these mean: Follow the process. Don't skip steps.**

---

## Decision Gate

Present scorecard to user:

**If Critical:**
- Explain rebuild vs. enhance trade-offs
- Get explicit user decision

**If Major/Minor/Pass:**
- Present findings
- Confirm proceeding to behavioral testing (Pass 2)

## Proceeding

After scorecard approval → Load `testing-skill-quality.md`
