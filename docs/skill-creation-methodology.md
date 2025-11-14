# Skill Creation Methodology (v2.0)

**Status:** PROPOSED - Based on analysis of obra's writing-skills methodology
**Date:** 2025-11-14
**See Also:** docs/writing-skills-analysis.md (full analysis)

---

## Core Principle

**"If you didn't watch an agent fail without the skill, you don't know if the skill teaches the right thing."**

Skills are created through **Test-Driven Development (TDD)** applied to process documentation:
- **RED**: Test WITHOUT skill → Document EXACT failures
- **GREEN**: Write MINIMAL skill → Fix observed violations
- **REFACTOR**: Test → Close loopholes → Repeat

---

## Two-Tier Skill Structure

### SKILL.md (Core Skill)

**Purpose:** Fast-loading essential guidance for skill discovery
**Target Length:**
- Router skills: <200 words (~25 lines)
- Getting-started skills: <200 words (~25 lines)
- Specialist skills: <500 words (~60 lines)

**Required Sections:**
1. **YAML Frontmatter** - Name + "Use when..." description
2. **Overview** - Core principle (1-2 paragraphs)
3. **When to Use** - Triggers, symptoms, situations
4. **Quick Reference** - Tables, commands, patterns
5. **Common Mistakes** - Top 3-5 anti-patterns
6. **Link to Guide** - "See GUIDE.md for [specifics]"

### GUIDE.md (Deep Dive - Optional)

**Purpose:** Comprehensive reference for deep work
**Target Length:** Unlimited
**Location:** `skills/[skill-name]/GUIDE.md`

**Contains:**
- Extended examples and code samples
- Complete configuration walkthroughs
- Case studies and scenarios
- Integration patterns
- Troubleshooting guides
- Supplementary theory

**When to create:**
- Skill NEEDS >500 words to be effective
- Heavy reference material (configuration, APIs)
- Multiple deep scenarios
- Complex integration workflows

---

## Description Field Format

### Template

```yaml
---
name: skill-name
description: Use when [TRIGGER] - [SYMPTOM] - [SITUATION]
---
```

### Rules

✅ **MUST:**
- Start with "Use when..."
- Include concrete trigger (observable event)
- Include symptom (problem being experienced)
- Include situation (context/constraints)
- Be specific, not vague
- Write in third person

❌ **MUST NOT:**
- Use vague phrases ("helps with", "assists in")
- Be overly technical in description
- List features instead of use cases
- Exceed 200 characters

### Examples

✅ **Good:**
```yaml
description: Use when codebase has 50+ lint warnings - team wants standards without disabling rules or risky refactoring
```

✅ **Good:**
```yaml
description: Use when training RL agents in continuous action spaces - need policy gradient methods - for stochastic policies
```

✅ **Good:**
```yaml
description: Use when problems persist despite fixes - solutions create new problems - system behavior counter-intuitive
```

❌ **Bad (current style):**
```yaml
description: Systematic process for fixing lint warnings without disabling them or over-refactoring
```

❌ **Bad (vague):**
```yaml
description: Helps with systems thinking methodology and analysis
```

---

## RED-GREEN-REFACTOR Workflow

### Phase 1: RED (Baseline Testing)

**Goal:** Document agent failures WITHOUT the skill

**Process:**
1. Create pressure scenario relevant to skill domain
2. Run agent WITHOUT skill loaded
3. Document EXACT failures:
   - What did agent do wrong?
   - What rationalizations did it use?
   - What workarounds did it attempt?
   - What violations occurred?
   - What did it skip or miss?

**Artifacts:**
- Save to `tests/baseline-failures-[skill-name].md`
- Include screenshots/logs if relevant
- Categorize failures by type

**Example (systematic-delinting):**
```markdown
# Baseline Failures: systematic-delinting

## Scenario: Agent asked to fix 100 lint warnings

### Observed Failures:
1. Agent added `# noqa` comments instead of fixing (15 instances)
2. Agent refactored code architecture during delinting (3 files)
3. Agent fixed all rules at once, single commit (impossible to review)
4. Agent skipped testing between fixes (broke 2 tests)
5. Agent ignored line-length, focused only on imports

### Rationalizations Used:
- "Adding # noqa is faster than rewriting"
- "This code needs refactoring anyway"
- "Let's batch all fixes for efficiency"
- "Tests will catch any issues"

### Missing Knowledge:
- Fix-never-disable principle
- Delinting ≠ refactoring distinction
- Rule-by-rule workflow
- Test-after-each-fix discipline
```

**Duration:** 30-60 minutes per scenario

### Phase 2: GREEN (Minimal Skill)

**Goal:** Write MINIMAL skill addressing observed failures

**Process:**
1. Create SKILL.md with required sections
2. Address ONLY failures from baseline testing
3. Target word count limits
4. No speculative content
5. Focus on preventing observed violations

**Template:**
```markdown
---
name: skill-name
description: Use when [TRIGGER] - [SYMPTOM] - [SITUATION]
---

# [Skill Name]

## Overview

[Core principle addressing root cause of failures]

[1-2 paragraphs establishing foundation]

## When to Use

**Use this skill when:**
- [Trigger from baseline]
- [Symptom from baseline]
- [Situation from baseline]

**Don't use when:**
- [Boundary conditions]

## Quick Reference

### [Key Concept 1]

✅ **CORRECT:** [Solution to observed violation]
❌ **WRONG:** [Exact failure from baseline]

### [Key Concept 2]

[Table/command/pattern addressing failures]

## Common Mistakes

1. **[Observed violation 1]** - [Why it fails] - [How to fix]
2. **[Observed violation 2]** - [Why it fails] - [How to fix]
3. **[Observed violation 3]** - [Why it fails] - [How to fix]

## Rationalization Resistance

| Rationalization (from baseline) | Reality | Counter-Guidance | Red Flag |
|--------------------------------|---------|------------------|----------|
| [Agent's excuse] | [Truth] | [How to respond] | [Warning sign] |

## See Also

See GUIDE.md for [extended content].
```

**Word Count Check:**
```bash
# Target: <500 words
wc -w skills/[skill-name]/SKILL.md
```

**Duration:** 1-2 hours

### Phase 3: REFACTOR (Iterative Testing)

**Goal:** Close loopholes, capture new rationalizations

**Process:**
1. Test skill with agent on SAME scenario
2. Document new failures:
   - New workarounds attempted
   - New rationalizations used
   - Loopholes discovered
   - Spirit-vs-letter violations
3. Update skill to close gaps
4. Repeat until NO violations
5. Test on DIFFERENT scenarios
6. Document iteration history

**Artifacts:**
- Save to `tests/refactor-log-[skill-name].md`
- Track iterations with timestamps
- Note what changed and why

**Example Iteration:**
```markdown
# Refactor Log: systematic-delinting

## Iteration 1 (2025-11-14 10:00)
**Test:** Same 100-warning scenario
**Violations:**
- Agent still batched 10 rules in one commit
- Agent skipped baseline statistics

**Changes:**
- Added "Rule-by-Rule Workflow" section
- Added "Essential Commands" with baseline step
- Updated rationalization table

## Iteration 2 (2025-11-14 11:30)
**Test:** Different scenario - 500 warnings
**Violations:**
- Agent got overwhelmed, wanted to disable rules
- Agent didn't triage by effort

**Changes:**
- Added "Triage Methodology" to Quick Reference
- Strengthened fix-never-disable language
- Added decision tree for large codebases

## Iteration 3 (2025-11-14 13:00)
**Test:** Same 500-warning scenario
**Violations:** None
**Status:** PASS - Skill is effective
```

**Duration:** 2-4 hours (multiple iterations)

### Phase 4: DEEP DIVE (Optional)

**Goal:** Create comprehensive GUIDE.md if needed

**Criteria for creating GUIDE.md:**
- Skill needs >500 words to be effective
- Heavy reference material required
- Multiple extended scenarios
- Complex configuration examples
- Detailed troubleshooting

**Process:**
1. Move extended content from SKILL.md to GUIDE.md
2. Keep SKILL.md under word limit
3. Link from SKILL.md to GUIDE.md
4. Organize GUIDE.md by scenario/topic

**GUIDE.md Template:**
```markdown
# [Skill Name] - Deep Dive Guide

> Quick start: See SKILL.md
> This guide: Comprehensive reference for [skill domain]

## Table of Contents

1. [Extended Concept 1]
2. [Extended Concept 2]
3. [Scenarios]
4. [Configuration Examples]
5. [Integration Patterns]
6. [Troubleshooting]

## [Extended Concept 1]

[Unlimited depth, examples, code samples]

## Scenarios

### Scenario 1: [Name]
[Complete walkthrough with code]

### Scenario 2: [Name]
[Complete walkthrough with code]

## Configuration Examples

[Full configuration files, annotated]

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| [...] | [...] | [...] |
```

**Duration:** 2-4 hours

---

## Bulletproofing Discipline Skills

**Applies to:** Skills that enforce rules, standards, or processes

**Required Components:**

### 1. Rationalization Resistance Table

Capture ALL rationalizations from baseline testing:

```markdown
| Rationalization | Reality | Counter-Guidance | Red Flag |
|-----------------|---------|------------------|----------|
| [Agent's excuse] | [Truth] | [Directive] | [Warning] |
```

**Example:**
```markdown
| "Adding # noqa is faster" | Disabling accumulates as debt | "Fix code to comply with rule" | Avoiding compliance |
| "This needs refactoring anyway" | Delinting ≠ refactoring | "Minimal changes only, defer refactoring" | Scope creep |
```

### 2. Explicit Prohibitions

**Use imperative language:**
- "NEVER [workaround]"
- "DO NOT [violation]"
- "ALWAYS [required action]"
- "MUST [requirement]"

**Example:**
```markdown
### The Golden Rule

**NEVER disable warnings** with `# noqa`, `# type: ignore`, or configuration exclusions.

**ALWAYS fix** by changing code to comply with the rule.

**DO NOT refactor** code architecture during delinting.
```

### 3. Red Flags Checklist

Observable warning signs from baseline:

```markdown
## Red Flags Checklist

Watch for these signs of incorrect approach:

- [ ] [Observable behavior indicating violation 1]
- [ ] [Observable behavior indicating violation 2]
- [ ] [Observable behavior indicating violation 3]

**If any red flag triggered → STOP → [corrective action]**
```

### 4. Spirit-vs-Letter Section

Address attempts to follow letter but violate spirit:

```markdown
### Following the Spirit

❌ **Letter but not spirit:**
[Example of technically compliant but wrong approach]

✅ **Both letter and spirit:**
[Example of correctly aligned approach]
```

### 5. Decision Trees

Remove ambiguity with flowcharts:

```markdown
## Should I [Action]?

```
Is [Condition 1]?
└─ YES → [Directive]
└─ NO ↓

Is [Condition 2]?
└─ YES → [Directive]
└─ NO → [Default directive]
```
```

---

## Quality Checklist

Before marking skill as complete:

### Content Quality
- [ ] Addresses ONLY failures from baseline testing
- [ ] No speculative or hypothetical content
- [ ] Core principle clearly stated
- [ ] When-to-use has concrete triggers/symptoms
- [ ] Quick reference is actually quick (not exhaustive)
- [ ] Common mistakes from baseline failures

### Format Quality
- [ ] Description starts with "Use when..."
- [ ] Description has trigger + symptom + situation
- [ ] SKILL.md under word count target
- [ ] YAML frontmatter valid
- [ ] Markdown formatting correct
- [ ] No broken links

### Bulletproofing (if discipline skill)
- [ ] Rationalization resistance table complete
- [ ] Explicit prohibitions for all workarounds
- [ ] Red flags checklist present
- [ ] Spirit-vs-letter addressed
- [ ] Decision trees for ambiguous cases

### Testing Quality
- [ ] Baseline failures documented in `tests/`
- [ ] Refactor iterations logged
- [ ] Final test passed (zero violations)
- [ ] Tested on 2+ different scenarios
- [ ] Edge cases covered

### Integration Quality
- [ ] Links to related skills
- [ ] Positioned in router skill (if applicable)
- [ ] Updated marketplace.json description
- [ ] GUIDE.md created if needed
- [ ] Examples are realistic (not toy code)

---

## Examples

### Minimal Router Skill (<200 words)

```markdown
---
name: using-python-engineering
description: Use when working with Python 3.12+ codebases - routes to modern syntax, testing, async, types, or delinting skills
---

# Using Python-Engineering (Meta-Skill Router)

## Overview

Routes you to the right Python engineering skill for your task.

## When to Use

**Use when:** Working with Python 3.12+ codebases

**Routes to:**
- **modern-syntax-and-types** - Type hints, match statements, union types
- **testing-and-quality** - pytest, fixtures, coverage, mocking
- **async-patterns-and-concurrency** - asyncio, async/await, concurrent.futures
- **systematic-delinting** - Fix 50+ lint warnings systematically
- **resolving-mypy-errors** - mypy type checking errors
- **debugging-and-profiling** - pdb, cProfile, memory profiling
- **project-structure-and-tooling** - pyproject.toml, Poetry, Ruff
- **scientific-computing-foundations** - NumPy, Pandas, Matplotlib
- **ml-engineering-workflows** - PyTorch integration, data pipelines

## Decision Tree

```
What's your goal?

├─ Fix lint warnings (50+) → systematic-delinting
├─ Resolve type errors → resolving-mypy-errors
├─ Set up new project → project-structure-and-tooling
├─ Add async code → async-patterns-and-concurrency
├─ Write tests → testing-and-quality
├─ Modern Python features → modern-syntax-and-types
├─ Performance issues → debugging-and-profiling
├─ Scientific computing → scientific-computing-foundations
└─ ML workflows → ml-engineering-workflows
```

## Quick Reference

| Trigger | Route To |
|---------|----------|
| 50+ lint warnings | systematic-delinting |
| mypy errors | resolving-mypy-errors |
| New project setup | project-structure-and-tooling |
| Need async/await | async-patterns-and-concurrency |

**Total skills:** 10 specialist skills in axiom-python-engineering pack
```

**Word count:** ~180 words ✅

### Minimal Specialist Skill (<500 words)

```markdown
---
name: systematic-delinting
description: Use when codebase has 50+ lint warnings - enables systematic fix without disabling rules or risky refactoring
---

# Systematic Delinting

## Overview

**Core Principle:** Fix warnings systematically, NEVER disable them. Delinting is minimal changes to comply with standards, NOT refactoring.

Lint warnings = measurable technical debt. Systematic delinting: baseline → triage → rule-by-rule fixes → zero warnings. No `# noqa`, no scope creep, no batching.

## When to Use

**Use this skill when:**
- Codebase has 50+ lint warnings
- Inheriting legacy code with no linting
- Enabling strict linting on existing projects
- Team wants standards without disabling rules

**Don't use when:**
- <50 warnings (just fix directly)
- Setting up NEW projects (use project-structure-and-tooling)
- Need to refactor code (separate concern)

**Symptoms:**
- "1000+ warnings, where to start?"
- "Legacy code needs cleanup"
- "How to fix without breaking?"

## Quick Reference

### Essential Commands

```bash
# 1. Baseline
ruff check . --statistics > baseline.txt

# 2. Pick highest-volume auto-fixable rule
# Example: F401 (unused imports) - 423 violations

# 3. Fix that rule only
ruff check . --select F401 --fix

# 4. Test
pytest

# 5. Commit
git commit -m "fix: Remove unused imports (F401)"

# 6. Repeat
```

### The Golden Rule

**NEVER disable:**
- ❌ `# noqa` comments
- ❌ `# type: ignore`
- ❌ `# pylint: disable`
- ❌ Configuration exclusions

**ALWAYS fix:**
- ✅ Change code to comply with rule
- ✅ One rule type at a time
- ✅ Test after each fix
- ✅ Commit per rule type

### Delinting ≠ Refactoring

| Delinting | Refactoring |
|-----------|-------------|
| Minimal changes to comply | Architecture changes |
| Safe, mechanical fixes | Requires design thinking |
| Low risk | Higher risk |
| Break lines, rename vars | Change algorithms, APIs |

❌ **WRONG:** Changing architecture to fix lint warning
✅ **CORRECT:** Minimal change (break line, rename variable)

### Rule-by-Rule Workflow

1. Fix ONE rule type completely
2. Commit
3. Repeat

**Why:** Small reviewable commits, easy revert, clear progress

## Common Mistakes

1. **Disabling Instead of Fixing**
   - ❌ Adding `# noqa` comments
   - ✅ Fix code to comply

2. **Over-Refactoring During Delinting**
   - ❌ Changing algorithms to fix E501 (line length)
   - ✅ Just break the line

3. **Fixing Everything at Once**
   - ❌ `ruff check . --fix` → 825 changes in one commit
   - ✅ One rule type at a time

4. **Skipping Tests Between Fixes**
   - ❌ Fix 5 rules, then test
   - ✅ Fix one rule, test, commit, repeat

5. **Ignoring Baseline**
   - ❌ Start fixing random warnings
   - ✅ Baseline → triage → prioritize

## Rationalization Resistance

| Rationalization | Reality | Counter-Guidance | Red Flag |
|-----------------|---------|------------------|----------|
| "# noqa is faster" | Disabling accumulates as debt | "Fix code, never disable" | Avoiding compliance |
| "This needs refactoring anyway" | Delinting ≠ refactoring | "Minimal changes only" | Scope creep |
| "Let's batch for efficiency" | Batching is unreviewable | "One rule type per commit" | Rushing |
| "Tests will catch issues" | Tests don't catch style | "Test after each rule" | Skipping validation |

## See Also

**GUIDE.md includes:**
- Complete Ruff/Pylint configuration examples
- 50+ rule-specific fixes with code samples
- CI integration (ratcheting, pre-commit hooks)
- Team adoption playbook
- Progress tracking implementation
- File-by-file vs rule-by-rule workflows

**Related skills:**
- @project-structure-and-tooling - Initial linting setup
- @modern-syntax-and-types - Type-related rules
- @testing-and-quality - Test linting standards
```

**Word count:** ~480 words ✅

---

## Migration Strategy

For existing skills:

### Step 1: Audit Current Skills

```bash
# Find verbose skills
find plugins -name "SKILL.md" -exec wc -w {} + | sort -rn | head -20

# Expected: Most skills 500-2000+ words
# Target: <500 words per skill
```

### Step 2: Prioritize for Splitting

**High Priority** (split first):
- Router skills (frequently loaded)
- Getting-started skills (entry points)
- Skills >1000 words

**Medium Priority:**
- Specialist skills 500-1000 words
- Discipline skills (need bulletproofing)

**Low Priority:**
- Reference skills (already concise)
- Skills <500 words (already compliant)

### Step 3: Split Process

For each verbose skill:

1. **Extract Essentials** (1-2 hours)
   - Identify core principle
   - Find key commands/patterns
   - Extract top 5 mistakes
   - Create <500 word SKILL.md

2. **Create GUIDE.md** (1 hour)
   - Move extended content to GUIDE.md
   - Organize by topic/scenario
   - Link from SKILL.md

3. **Update Description** (15 min)
   - Reformat to "Use when..." pattern
   - Add triggers, symptoms, situations
   - Update marketplace.json

4. **Test** (30 min)
   - Verify links work
   - Check word count
   - Validate formatting

**Estimated effort:** 20-40 hours for 144 skills (if all need splitting)

### Step 4: Update Contributing Docs

Create `CONTRIBUTING-SKILLS.md`:
- RED-GREEN-REFACTOR workflow
- Two-tier structure guidelines
- Bulletproofing requirements
- Quality checklist

---

## Tools

### Word Count Check

```bash
#!/bin/bash
# check-word-count.sh

SKILL_FILE=$1
WORD_COUNT=$(grep -v '^---' "$SKILL_FILE" | grep -v '^```' | wc -w)

if [ $WORD_COUNT -lt 200 ]; then
  echo "✅ Router/Getting-started compliant: $WORD_COUNT words"
elif [ $WORD_COUNT -lt 500 ]; then
  echo "✅ Specialist compliant: $WORD_COUNT words"
elif [ $WORD_COUNT -lt 1000 ]; then
  echo "⚠️  Consider splitting: $WORD_COUNT words"
else
  echo "❌ MUST split: $WORD_COUNT words"
fi
```

### Description Validator

```bash
#!/bin/bash
# validate-description.sh

SKILL_FILE=$1
DESC=$(grep '^description:' "$SKILL_FILE" | cut -d':' -f2-)

if [[ $DESC == *"Use when"* ]]; then
  echo "✅ Description format valid"
else
  echo "❌ Description must start with 'Use when...'"
  echo "   Current: $DESC"
fi
```

### Baseline Testing Template

```markdown
# Baseline Failures: [skill-name]

**Date:** YYYY-MM-DD
**Scenario:** [Description]
**Agent Version:** [Claude version]

## Setup

[How to reproduce scenario]

## Observed Failures

1. **[Category]:** [What agent did wrong]
   - Frequency: X instances
   - Severity: High/Medium/Low

2. **[Category]:** [What agent did wrong]
   - Frequency: X instances
   - Severity: High/Medium/Low

## Rationalizations Used

- "[Agent's exact words]"
- "[Another rationalization]"

## Missing Knowledge

- [Concept agent didn't know]
- [Principle agent violated]

## Artifacts

- Screenshot: [link]
- Log file: [link]
- Code sample: [link]
```

---

## FAQ

**Q: Do we delete verbose content when creating GUIDE.md?**
A: NO. Move it to GUIDE.md, don't delete. We maintain professional depth.

**Q: What if a skill CAN'T be under 500 words?**
A: Create GUIDE.md. Keep SKILL.md under 500 with link to GUIDE.md.

**Q: Should we keep TDD artifacts in the repo?**
A: YES, in `tests/` directory. Transparency and reproducibility.

**Q: Do ALL skills need bulletproofing tables?**
A: Only discipline skills (those enforcing rules/processes).

**Q: What about existing skills that work fine but are verbose?**
A: Split over time. Prioritize routers and frequently-loaded skills first.

**Q: How do we know if agent "failed" in RED phase?**
A: Compare to desired behavior. Document ANY deviation from best practice.

**Q: Can we skip RED phase for "simple" skills?**
A: NO. The Iron Law: No skill without failing test first. No exceptions.

**Q: What if refactor phase finds no violations?**
A: Good! Skill is effective. Document test in refactor log, mark PASS.

---

## Summary

**Create skills like we create code:**
1. **Test first** (RED - baseline failures)
2. **Minimal implementation** (GREEN - <500 words)
3. **Iterate** (REFACTOR - close loopholes)
4. **Extend** (GUIDE.md if needed)

**Key principles:**
- "Use when..." descriptions
- Token efficiency targets
- Two-tier structure (SKILL.md + GUIDE.md)
- TDD validation (RED-GREEN-REFACTOR)
- Bulletproofing discipline skills

**Result:**
- Faster skill discovery
- Lower context usage
- Validated effectiveness
- Professional depth maintained
