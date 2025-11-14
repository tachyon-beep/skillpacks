
# Skill Audit

## Overview

**Skill auditing IS Test-Driven Development applied to existing documentation validation.**

You dispatch subagents to test whether skills still work under pressure, verify they match their original scope, and systematically identify issues. Reading a skill is not auditing it. Testing is auditing.

**Core principle:** If you didn't watch an agent try to use the skill and observed the results, you don't know if it still works.

**REQUIRED BACKGROUND:** You MUST understand the TDD methodology for skills (RED-GREEN-REFACTOR cycle applied to documentation) before performing audits. This skill assumes you know how to test skills with subagents under pressure.

## When to Use

**Audit when:**
- Skill was created 3+ months ago (Claude capabilities drift)
- Major Claude version update occurred
- User reports skill "doesn't seem to work anymore"
- Systematic skillpack quality review
- Before publishing or sharing skillpack externally
- Skill has been edited but not re-tested

**Symptoms that trigger audit:**
- "This skill used to work but agents ignore it now"
- "Skill exists but no one uses it"
- "Not sure if this is still relevant"
- Scope creep suspicions
- Tech stack or best practices changed

**Don't audit when:**
- Skill just created and tested (< 1 week old)
- No changes to Claude capabilities or domain
- Pure typo fixes not affecting content

## The Iron Law (Same as TDD)

```
NO APPROVAL WITHOUT TESTING FIRST
```

Reading a skill and saying "looks good" is NOT an audit.
Reviewing code examples for syntax errors is NOT an audit.
Checking if files exist is NOT an audit.

**Auditing means: dispatch subagents with pressure scenarios and observe results.**

## Audit Process Checklist

**IMPORTANT: Use TodoWrite to create todos for EACH phase below.**

### Phase 1: Verify Existence & Scope

- [ ] Locate skill file and confirm it exists
- [ ] Read original frontmatter - what was the intended purpose?
- [ ] Check git history - when created, what was the original scope?
- [ ] Document original intent vs current content

### Phase 2: Test Current Effectiveness

**CRITICAL: This is the audit. Everything else is preparation.**

- [ ] Create 2-3 pressure scenarios based on skill type (see below)
- [ ] Dispatch subagents WITH the skill loaded
- [ ] Document: Do they follow guidance? Where do they fail?
- [ ] Compare to baseline (if baseline tests exist from original creation)

**Pressure scenario requirements:**
- Must include time pressure (10-15 min deadline)
- Must include sunk cost pressure (already did similar work)
- Must include authority pressure ("user trusts your judgment")
- Must be realistic to skill's domain

### Phase 3: Scope Drift & Appropriateness Check

**Part A: Scope Drift**
- [ ] Does current content match original purpose? (±20% is normal)
- [ ] Has scope expanded without updating description?
- [ ] Are there sections unrelated to core purpose?
- [ ] If drifted: Is new scope better, or should it revert?

**Part B: Scope Appropriateness**

**CRITICAL: Check if the skill is scoped correctly. Too broad = gaps and inconsistent depth. Too narrow = fragmentation and duplication.**

- [ ] Check skill size: `wc -l SKILL.md`
  - < 300 lines: May be too focused (could merge with related skill?)
  - 300-1000 lines: Well-scoped for single topic
  - 1000-2000 lines: Large but manageable if coherent
  - > 2000 lines: Likely covering multiple concepts (consider splitting)

- [ ] Analyze topic coherence:
  - Does skill cover ONE core concept or MULTIPLE concepts?
  - Are all sections necessary for understanding the core concept?
  - Could sections be standalone skills?

- [ ] Check depth consistency:
  - Is some content superficial while other parts are deep?
  - Do some topics get 50 lines while others get 500?
  - Depth inconsistency = scope mismatch

- [ ] Identify split indicators:
  - Skill has "Part 1, Part 2, Part 3..." structure covering different algorithms/approaches
  - Description lists 4+ distinct topics
  - Testing reveals one section needs major expansion (would unbalance skill)
  - Skill is > 1500 lines with multiple independent topics

- [ ] Identify merge indicators:
  - Skill is < 300 lines and very specific
  - Multiple sibling skills with overlapping content
  - Skill constantly refers to other skills for context
  - Users need 3+ skills loaded together for typical task

**Scope Recommendations:**

| Size | Topics | Depth | Recommendation |
|------|--------|-------|----------------|
| > 1500 lines | Multiple (4+) | Inconsistent | **SPLIT**: Create focused sub-skills |
| > 1000 lines | Multiple (2-3) | Consistent | **ACCEPTABLE**: Monitor for growth |
| 300-1000 lines | Single | Consistent | **OPTIMAL**: Well-scoped |
| < 300 lines | Single | Superficial | **MERGE**: Combine with related skills |
| < 300 lines | Single | Deep | **ACCEPTABLE**: Specialized topic |

**Report to operator:**
- Current size and line count
- Number of distinct topics/algorithms covered
- Depth consistency assessment
- Split recommendation: "Consider splitting into [skill-A] + [skill-B]"
- Merge recommendation: "Consider merging with [related-skill]"
- Reasoning: Why current scope creates issues or works well

### Phase 4: Cross-Reference Awareness Check

**CRITICAL: Skills operate in an ecosystem. Check if this skill knows about related skills.**

- [ ] Identify skill's domain/category from its plugin and purpose
- [ ] Search for related skills in adjacent domains using pattern matching
- [ ] Check if skill mentions/references those related skills appropriately
- [ ] Determine if missing cross-references create gaps in guidance

**Common cross-reference patterns:**

| If skill is about... | Should be aware of... |
|---------------------|----------------------|
| Backend APIs (axiom-*) | UX design (lyra-*), Security (ordis-*) |
| Frontend/Web (axiom-*, lyra-*) | UX design (lyra-*), Backend patterns |
| ML training (yzmir-training-*) | PyTorch engineering, Production deployment |
| Security patterns (ordis-*) | Relevant tech stacks being secured |
| Documentation (muna-*) | Technical domains being documented |

**Examples:**
- `axiom-web-backend` auditing TypeScript APIs → should mention `lyra-ux-designer` for frontend integration and `ordis-security-architect` for API security
- `yzmir-pytorch-engineering` covering model building → should reference `yzmir-training-optimization` for training and `yzmir-ml-production` for deployment
- `lyra-ux-designer` covering accessibility → should mention `muna-technical-writer` for documentation

**How to check:**
```bash
# Find related skills by domain keywords
grep -r "frontend\|UX\|user interface" plugins/*/skills/*/SKILL.md

# Check if current skill mentions related plugins
grep -i "lyra\|ordis\|axiom" plugins/current-plugin/skills/current-skill/SKILL.md
```

**Categorization:**
- **MINOR:** Add brief mention in "When to Use" or "Related Skills" section
- **MODERATE:** Missing critical workflow connection (e.g., backend→security)
- **MAJOR:** Skill gives advice that contradicts related skill

### Phase 5: Compatibility Check

- [ ] Does skill reference outdated Claude capabilities?
- [ ] Are code examples using deprecated syntax?
- [ ] Do tool references still work? (APIs, CLIs, libraries)
- [ ] Test one code example manually if skill is technique-focused

### Phase 6: Categorize Issues

Sort findings into three categories:

**🟢 MINOR (handle autonomously):**
- Typos, grammar fixes
- Adding missing keywords to description
- Updating version numbers in code examples
- Fixing broken internal links
- Adding clarifying examples
- Adding brief cross-reference mentions to related skills
- Total changes < 20% of content

**🟡 MODERATE (discuss with operator):**
- Scope drift >20%
- Skill needs restructuring
- Major sections need rewriting
- Testing reveals consistent rationalization loopholes
- Missing critical cross-references (e.g., backend skill ignoring security)
- Scope too broad (>1500 lines, multiple topics, consider splitting)
- Scope too narrow (<300 lines, superficial, consider merging)
- Uncertain if skill still needed

**🔴 MAJOR (escalate immediately):**
- Skill completely broken (agents can't use it)
- Fundamental approach is wrong
- Technology/library deprecated and unmaintained
- Testing shows skill causes harm
- Duplicate of another skill
- Skill contradicts guidance in related skills (ecosystem conflict)
- Should be deleted entirely

### Phase 7: Execute or Escalate

**For MINOR issues:**
- Make edits directly
- Document changes in commit message
- Re-test ONE scenario to verify fix works

**For MODERATE/MAJOR issues:**
- Create detailed report with test results
- Include verbatim agent failures
- Propose 2-3 options (fix, merge, delete)
- Wait for operator decision

## Testing by Skill Type

### Discipline-Enforcing Skills

Examples: TDD rules, verification requirements, process enforcement

**Test scenarios:**
1. Time pressure (10 min deadline)
2. Sunk cost pressure (already did similar work wrong way)
3. Authority pressure (user says "I trust you, just fix it")

**Success criteria:** Agent follows rules under ALL THREE pressures combined

### Technique Skills

Examples: Debugging patterns, testing approaches, architecture methods

**Test scenarios:**
1. Application scenario (use the technique on new problem)
2. Missing info scenario (technique description has gaps?)
3. Edge case scenario (technique under unusual constraints)

**Success criteria:** Agent successfully applies technique without errors

### Pattern Skills

Examples: Mental models, design principles, conceptual frameworks

**Test scenarios:**
1. Recognition scenario (identify when pattern applies)
2. Application scenario (apply pattern to new domain)
3. Counter-example scenario (identify when NOT to use)

**Success criteria:** Agent correctly identifies applicability and applies pattern

### Reference Skills

Examples: API docs, command syntax, library guides

**Test scenarios:**
1. Retrieval scenario (find specific information quickly)
2. Application scenario (use info to solve problem)
3. Gap scenario (is common use case documented?)

**Success criteria:** Agent finds info and uses correctly within 5 minutes

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|----------------|-----|
| **Just reading skill** | Looks ≠ works | Dispatch subagent test |
| **Assuming compatibility** | Claude evolves, skills drift | Test with current Claude |
| **Skipping pressure testing** | Skills fail under pressure, not ideal conditions | Add time + sunk cost + authority |
| **Confidence without evidence** | "Seems fine" is not data | Get test results |
| **Batch approvals** | "All 10 look good!" without testing | Test each individually |
| **Fixing before testing** | Making changes then testing proves nothing | Test CURRENT state first |
| **Ignoring ecosystem** | Skills exist in context with other skills | Check cross-references |
| **Ignoring scope issues** | Accepting 2000-line skill or 5 fragmented skills | Check if split/merge needed |

## Rationalization Red Flags

**STOP and dispatch subagent test if you think:**

- "It's obviously still relevant" → Test it
- "Just needs minor updates" → How do you know without testing?
- "I can tell from reading" → Reading shows intent, testing shows reality
- "Too simple to break" → Simple skills break when Claude changes
- "This is tedious" → 3 minutes of testing beats hours debugging bad skill
- "User is waiting" → Shipping broken skill wastes more time
- "I'm confident" → Overconfidence guarantees you'll miss issues

**All of these mean: Stop. Dispatch subagent. Get test data.**

## Minor vs Major Change Boundaries

### 🟢 MINOR - Handle Autonomously

**Criteria:** Does not change meaning or scope

- Typo/grammar fixes
- Formatting improvements
- Adding synonyms to description for searchability
- Fixing broken links
- Adding clarifying examples (< 20 lines)
- Updating version numbers
- Code syntax fixes (not logic changes)

**Verification:** Re-test ONE scenario confirms fix works

### 🟡 MODERATE - Discuss First

**Criteria:** Changes meaning but stays within original scope

- Restructuring sections
- Rewriting 20-50% of content
- Changing core examples
- Adding new subsections
- Updating for new Claude capabilities
- Closing rationalization loopholes

**Verification:** Full re-test with 2-3 scenarios required

### 🔴 MAJOR - Escalate Immediately

**Criteria:** Fundamental changes to purpose, approach, or existence

- Scope changed >50%
- Should merge with another skill
- Should split into multiple skills
- Technique is fundamentally flawed
- Technology deprecated, no replacement
- Should be deleted
- Changes affect other skills

**Verification:** Operator decides, then full RED-GREEN-REFACTOR cycle

## Audit Report Template

```markdown
## Skill Audit: [skill-name]

**Auditor:** [Your identifier]
**Date:** [ISO date]
**Skill location:** [path]

### Original Purpose
[From frontmatter description and git history]

### Test Results
**Scenario 1:** [Description]
- Agent behavior: [What happened]
- Complied with skill? [Yes/No/Partial]
- Issues: [Specific failures]

**Scenario 2:** [Description]
- Agent behavior: [What happened]
- Complied with skill? [Yes/No/Partial]
- Issues: [Specific failures]

### Findings
- ✅ Works as intended: [List what works]
- ⚠️  Issues found: [List issues with severity]
- 📊 Scope drift: [None / Minor / Major]
- 📏 Scope appropriateness: [Optimal / Too broad (split?) / Too narrow (merge?)]
  - Size: [X lines]
  - Topics covered: [List distinct topics]
  - Depth consistency: [Consistent / Inconsistent - describe]
- 🔄 Compatibility: [Current / Needs updates / Broken]
- 🔗 Cross-references: [Appropriate / Missing / Conflicting]

### Recommendation
**Category:** [MINOR / MODERATE / MAJOR]

**Action:** [Fix autonomously / Discuss options / Escalate for decision]

**Proposed changes:** [Specific edits or options]

**Verification plan:** [How to re-test after changes]
```

## Real-World Impact

**Without systematic audit:**
- Skills drift and become useless (agents stop using them)
- Outdated guidance causes failures
- Skillpack quality degrades over time
- Users lose trust in skillpack

**With systematic audit:**
- Skills stay relevant and effective
- Issues caught before causing problems
- Clear escalation prevents unauthorized scope changes
- Maintained quality over time

## The Bottom Line

**Auditing skills IS testing existing documentation.**

Same Iron Law as creating skills: No approval without testing.
Same process: Dispatch subagents, observe under pressure, document results.
Same rigor: Reading ≠ validating.

If you wouldn't ship code without testing, don't approve skills without testing.
