# Contributing to Security Architect & Technical Writer Skills

Thank you for your interest in contributing! These skillpacks welcome new skills, bug fixes, and documentation improvements.

---

## How to Propose a New Skill

**1. Create GitHub issue first** using `.github/ISSUE_TEMPLATE/new_skill.md`

Before implementing, create an issue describing:
- **Skill name:** Which faction and pack (e.g., `ordis/security-architect/api-security-testing`)
- **Purpose:** What does this skill teach? (2-3 sentences)
- **When to use:** What situations trigger using this skill?
- **Related skills:** Which existing skills does this complement?
- **Real-world gap:** What does Claude currently miss that this skill addresses?

**2. Wait for maintainer feedback**

Maintainers will review to ensure:
- Skill addresses real gap (not duplicate of existing skill)
- Fits faction/pack organization
- Scope is appropriate (not too broad or too narrow)

**3. Implement with RED-GREEN-REFACTOR testing** (required)

See "Skill Creation Requirements" below.

---

## Skill Creation Requirements

**All skills MUST follow RED-GREEN-REFACTOR TDD methodology.**

This ensures skills actually work and address real gaps.

### RED Phase: Baseline Test Without Skill

**Goal:** Objectively measure what Claude misses without your skill.

**Process:**
1. Create realistic scenario where your skill should help
2. Test Claude (via subagent or fresh session) WITHOUT loading your skill
3. Document failures objectively:
   - What did Claude miss?
   - What did Claude get wrong?
   - What was inefficient or unclear?

**Example (threat-modeling skill):**
```
Scenario: Design auth API, need threat analysis

WITHOUT skill:
❌ Missed STRIDE methodology (used ad-hoc intuition)
❌ No attack trees (listed threats but no paths)
❌ No risk scoring (couldn't prioritize threats)
❌ No systematic coverage (missed info disclosure threats)
```

### GREEN Phase: Create Skill Addressing Failures

**Goal:** Write skill that addresses every documented baseline failure.

**Required Skill Components:**

1. **YAML Frontmatter:**
```yaml
---
name: skill-name-here
description: Use when [trigger] - covers [what it teaches]
---
```

2. **Overview Section:**
   - Core principle (one sentence)
   - Key insight (what makes this skill valuable)

3. **When to Use Section:**
   - Load this skill when: [situations]
   - Symptoms you need this: [trigger patterns]
   - Don't use for: [anti-patterns]

4. **Core Content:**
   - Teach patterns and methodology (not exhaustive lists)
   - Include decision trees and checklists
   - Provide 3+ real-world examples (concrete, complete)

5. **Cross-References:**
   - Use WITH this skill: [complementary skills]
   - Use AFTER this skill: [follow-up skills]

6. **Real-World Impact Section:**
   - How has this skill been used?
   - What concrete improvements did it deliver?
   - Key lesson (one sentence takeaway)

**Example Structure:**
```markdown
---
name: my-new-skill
description: Use when [situation] - covers [capabilities]
---

# My New Skill

## Overview
[Core principle and key insight]

## When to Use
[Trigger situations, symptoms, anti-patterns]

## [Core Content Sections]
[Teach the methodology with examples]

## Cross-References
**Use WITH:** [related skills]

## Real-World Impact
[Proven effectiveness with concrete examples]
```

### REFACTOR Phase: Verification Test With Skill

**Goal:** Prove skill addresses all baseline failures.

**Process:**
1. Test Claude (via subagent) WITH your skill loaded
2. Apply to same scenario from RED phase
3. Verify every baseline failure is now resolved
4. Test under pressure:
   - Time constraints ("90-minute emergency patch")
   - Edge cases
   - Scope creep scenarios

**Example (threat-modeling skill):**
```
Scenario: Same auth API design

WITH skill:
✅ Applied STRIDE systematically (all 6 categories)
✅ Built attack trees with feasibility marking
✅ Scored risks with Likelihood × Impact matrix
✅ Found ALL threats (including info disclosure)
```

**If any baseline failures remain:** Return to GREEN phase, improve skill, re-test.

### Document Testing in Commit Message

Commit message must include RED-GREEN-REFACTOR testing summary:

```
Implement ordis/security-architect/my-new-skill

RED-GREEN-REFACTOR cycle complete:

RED Phase:
- Tested scenario X without skill
- Documented 5 baseline failures: [list]

GREEN Phase:
- Created skill addressing all failures
- Included [pattern Y] and [methodology Z]

REFACTOR Phase:
- Tested WITH skill - all failures resolved
- Tested under pressure - maintains discipline
- Skill is bulletproof

[Additional context]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Faction Organization

Skills are grouped by Altered TCG-inspired factions:

- **Ordis** (Protectors of Order): Security, governance, compliance
- **Muna** (Weavers of Harmony): Documentation, synthesis, clarity
- **Axiom** (Creators of Marvels): Tooling, infrastructure, automation
- **Bravos** (Champions of Action): Practical tactics, execution
- **Lyra** (Nomadic Artists): UX, creative, design
- **Yzmir** (Magicians of Mind): Theory, algorithms, architecture

See [FACTIONS.md](FACTIONS.md) for details on choosing appropriate faction.

---

## Style Guide

### Markdown Formatting
- Use `#` for main heading, `##` for sections
- Use triple backticks with language tags for code blocks
- Use `**bold**` for emphasis, `*italic*` for secondary emphasis
- Use numbered lists for sequential steps, bullet lists for unordered items

### YAML Frontmatter
```yaml
---
name: faction/pack/skill-name  # Use kebab-case
description: Use when [trigger] - covers [capabilities]  # One line, <120 chars
---
```

### Cross-References
Use relative paths:
```markdown
**Use WITH this skill:**
- `ordis/security-architect/threat-modeling` - [Brief description]
```

### Decision Trees
Use markdown tables or ASCII art:
```markdown
1. **Check X:**
   - If Y → Proceed to step 2
   - If Z → See [section]
```

### Real-World Examples
- Use obviously fake credentials: `fake_api_key_abc123_for_docs_only`
- Use example.com domains (RFC 2606)
- Use complete examples (not placeholders like `YOUR_KEY_HERE`)

---

## Pull Request Process

**1. Fork repository**

```bash
git clone https://github.com/[your-username]/skillpacks
cd skillpacks
```

**2. Create feature branch**

```bash
git checkout -b skill/my-new-skill
```

**3. Implement skill with RED-GREEN-REFACTOR testing**

Follow "Skill Creation Requirements" above.

**4. Commit with detailed message**

Include RED-GREEN-REFACTOR testing summary in commit message.

**5. Push to your fork**

```bash
git push origin skill/my-new-skill
```

**6. Create Pull Request**

Use `.github/PULL_REQUEST_TEMPLATE.md` template.

### PR Checklist

Before submitting PR, verify:

- [ ] RED-GREEN-REFACTOR testing complete (documented in commit message)
- [ ] Real-world examples included (3+ concrete examples)
- [ ] Cross-references updated (complementary skills linked)
- [ ] YAML frontmatter complete (name + description)
- [ ] Follows style guide (markdown, formatting)
- [ ] "Real-World Impact" section included
- [ ] Ready for review

---

## Code Review Process

Maintainers will review:

1. **Skill addresses real gap**
   - Not duplicate of existing skill
   - Solves problem Claude currently struggles with

2. **RED-GREEN-REFACTOR methodology followed**
   - Baseline testing documented
   - Skill addresses baseline failures
   - Verification testing proves effectiveness

3. **Examples are realistic and complete**
   - Concrete scenarios (not abstract)
   - Complete examples (copy-paste-runnable)
   - Obviously fake credentials/data

4. **Writing is clear and actionable**
   - Core principle stated clearly
   - Decision trees and checklists provided
   - No ambiguity

5. **Cross-references are accurate**
   - Related skills linked correctly
   - Relative paths work

---

## Bug Reports & Feature Requests

**Bug in existing skill:**
- Use `.github/ISSUE_TEMPLATE/bug_report.md`
- Include skill name, expected behavior, actual behavior
- Provide reproduction steps

**Feature request:**
- Use GitHub Discussions for initial discussion
- If consensus reached, maintainer creates issue
- Follow "Propose New Skill" process

---

## Development Workflow

For detailed development workflow, testing methodology, and implementation history, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Questions?

- **GitHub Issues:** Propose skills, report bugs
- **GitHub Discussions:** Ask questions, get feedback
- **Pull Requests:** Submit contributions

Thank you for contributing to these skillpacks!
