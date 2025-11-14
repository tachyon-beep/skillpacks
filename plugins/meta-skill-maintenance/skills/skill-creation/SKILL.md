---
name: skill-creation
description: Use when creating new skills for the marketplace using TDD methodology - adapted from obra's writing-skills with marketplace-specific patterns including plugin structure, faction conventions, scope guidelines, and cross-reference integration
---

# Skill Creation

## Attribution & Foundation

**This skill adapts the TDD methodology from obra's writing-skills:**
- **Source:** https://github.com/obra/superpowers/skills/writing-skills
- **Core methodology:** RED-GREEN-REFACTOR for documentation
- **Key concepts:** Rationalization tables, pressure testing, CSO principles, Iron Law
- **Full credit to obra for the foundational TDD approach to documentation**

This skill **extends** obra's methodology with marketplace-specific patterns for creating skills in the tachyon-beep/skillpacks marketplace.

## Overview

**Skill creation IS Test-Driven Development applied to documentation creation.**

Same Iron Law as code: **Write failing test first, watch it fail, write minimal skill, watch it pass, refactor.**

**REQUIRED BACKGROUND:** You MUST understand obra's core TDD methodology before using this skill. The RED-GREEN-REFACTOR cycle, pressure testing, and rationalization table concepts are foundational and documented in obra's writing-skills.

**This skill focuses on:** Marketplace-specific structure, faction selection, scope guidelines, plugin integration, and ecosystem coherence.

## When to Use This Skill

Use when:
- Creating a new skill for this marketplace
- User requests "add a skill for X"
- Identified gap in skillpack coverage
- Need to split existing skill into focused skills

Do NOT use when:
- Editing existing skill (use skill-audit instead)
- Creating skills for other marketplaces (different structure)
- User needs general TDD guidance (see obra's writing-skills)

---

## Marketplace-Specific Requirements

### 1. Plugin Structure

**Every skill belongs to a plugin:**

```
plugins/
  [plugin-name]/
    .claude-plugin/
      plugin.json          # Plugin metadata
    skills/
      [skill-name]/
        SKILL.md           # Your skill content
```

**plugin.json format:**
```json
{
  "name": "plugin-name",
  "version": "1.0.0",
  "description": "Brief description - N skills",
  "category": "development|ai-ml|game-development|user-experience|documentation|security|meta"
}
```

### 2. YAML Frontmatter (Required)

**ONLY two fields allowed:**

```yaml
---
name: skill-name-with-hyphens
description: Use when [specific triggers] - [what it does in third person]
---
```

**Requirements:**
- `name`: Letters, numbers, hyphens only (no parentheses, special chars)
- `description`: < 1024 chars total, start with "Use when...", third person
- Include BOTH triggers (when to use) AND what it does
- Technology-agnostic triggers unless skill is tech-specific

### 3. Faction Selection Guide

**Choose faction based on domain:**

| Faction | Domain | Philosophy |
|---------|--------|------------|
| **Axiom** | Python engineering, architecture analysis | Systematic, analytical, engineering rigor |
| **Yzmir** | AI/ML, neural networks, training, RL | Mathematical, optimization-focused |
| **Bravos** | Game development, simulation, systems | Emergent systems, player-driven design |
| **Lyra** | UX design, accessibility, user research | User-centered, experience-driven |
| **Ordis** | Security, threat modeling, compliance | Defense, threat analysis, compliance |
| **Muna** | Documentation, technical writing | Clarity, structure, maintainability |
| **Meta** | Skills about skills, marketplace tools | Self-referential, meta-level tools |

**Naming convention:** `faction-topic` (e.g., `yzmir-pytorch-engineering`, `lyra-ux-designer`)

### 4. Scope Guidelines (Critical)

**Target sizes per our audits:**
- **< 300 lines:** May be too focused (consider merging with related skill)
- **300-1000 lines:** Optimal - well-scoped for single topic
- **1000-2000 lines:** Large but manageable if coherent
- **> 2000 lines:** Likely covering multiple concepts (split into focused skills)

**Scope appropriateness indicators:**
- **ONE core concept** per skill
- **Consistent depth** (not 50 lines for one topic, 500 for another)
- **Clear boundaries** (what's in scope vs out of scope)

**Split indicators:**
- Skill has "Part 1, Part 2, Part 3..." covering different algorithms
- Description lists 4+ distinct topics
- Testing reveals one section needs major expansion (would unbalance skill)

### 5. Cross-Reference Integration

**Skills operate in an ecosystem. Reference related skills:**

**Dependency patterns:**
| If Your Skill Covers... | Must Reference | Should Reference |
|------------------------|---------------|------------------|
| Loss functions | gradient-management, training-loop | debugging-techniques |
| RL algorithms | pytorch-engineering, training-optimization | ml-production |
| Model architectures | pytorch-engineering | training-optimization, ml-production |
| Backend APIs | security-architect | ux-designer |
| UX design | - | technical-writer (for docs) |

**How to reference:**
- Mention skill name in "Do NOT use" section with routing guidance
- Add to "See also" or "Related Skills" section
- Reference when describing workflow ("After implementing model, see training-optimization for...")

---

## TDD Process for Skills (From obra's methodology)

**For complete TDD guidance, see obra's writing-skills. Summary:**

### RED Phase: Write Failing Test

1. **Create 2-3 pressure scenarios** (time + sunk cost + authority)
2. **Run WITHOUT skill** - Dispatch subagents
3. **Document baseline failures** - What did they do wrong? Exact rationalizations (verbatim)

### GREEN Phase: Write Minimal Skill

1. **Write skill addressing specific baseline failures**
2. **Follow marketplace structure** (frontmatter, faction, scope)
3. **Run WITH skill** - Agents should now comply
4. **Verify improvement** - Compare to baseline

### REFACTOR Phase: Close Loopholes

1. **Identify new rationalizations** from testing
2. **Add explicit counters** (if discipline skill)
3. **Build rationalization table** from all iterations
4. **Re-test until bulletproof**

**For detailed pressure testing techniques, rationalization tables, and CSO principles, see obra's writing-skills.**

---

## Marketplace Integration Checklist

**After creating skill using TDD:**

- [ ] **Skill file:** Created in `plugins/[faction-name]/skills/[skill-name]/SKILL.md`
- [ ] **Frontmatter:** Valid YAML with name and description
- [ ] **Scope:** 300-1000 lines (optimal) or justified if outside range
- [ ] **Cross-references:** Mentions related skills appropriately
- [ ] **Plugin registered:** Entry in plugin's `.claude-plugin/plugin.json`
- [ ] **Marketplace updated:** Entry in `.claude-plugin/marketplace.json` (if new plugin)
- [ ] **Baseline tests preserved:** Save test scenarios for future audits
- [ ] **Slash command:** Created if router skill (run convert-routers-to-commands.sh)

---

## Creating vs Editing Skills

| Task | Use This Skill | Use skill-audit |
|------|---------------|----------------|
| Brand new skill | ✅ Yes | No |
| Fix existing skill | No | ✅ Yes |
| Split existing skill | ✅ Yes (create new) + skill-audit (analyze old) | |
| Merge skills | ✅ Yes (create combined) | |

---

## Example: Creating a New Skill

**Scenario:** User asks for "RL reward shaping skill"

**Step 1: Faction Selection**
- Domain: Reinforcement learning
- Faction: **Yzmir** (AI/ML)
- Plugin: `yzmir-deep-rl` (already exists)

**Step 2: Scope Definition**
- Core concept: Reward function design
- In scope: Shaping techniques, common pitfalls, debugging
- Out of scope: Algorithm selection (other skill), environment design
- Target: 400-600 lines

**Step 3: RED Phase (obra's methodology)**
```
Dispatch subagent WITHOUT skill:
"How do I design reward function for robot navigation?"

Baseline failures:
- Agent suggests dense rewards everywhere (unstable)
- No mention of reward sparsity tradeoffs
- Missing debugging techniques for reward hacking
```

**Step 4: GREEN Phase**
```yaml
---
name: reward-shaping-techniques
description: Use when designing reward functions for RL - covers shaping methods, sparsity tradeoffs, debugging reward hacking, and common pitfalls to avoid unstable training
---

# Reward Shaping Techniques

[... skill content addressing baseline failures ...]

## Related Skills

- **rl-foundations** - For MDP basics and value functions
- **policy-gradient-methods** - For policy optimization with shaped rewards
- **debugging-techniques** (yzmir-pytorch-engineering) - For implementation debugging

[... rest of skill ...]
```

**Step 5: REFACTOR Phase**
- Re-test with skill loaded
- Close rationalization loopholes
- Add to yzmir-deep-rl plugin

**Step 6: Integration**
- Update `plugins/yzmir-deep-rl/.claude-plugin/plugin.json` skill count
- Preserve baseline tests for future audits
- Commit with "feat: Create reward-shaping-techniques skill"

---

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---------|-----------|-----|
| **No TDD** | Untested skills have issues | Follow RED-GREEN-REFACTOR |
| **Wrong faction** | Confuses users, breaks organization | Use faction selection guide |
| **Scope too broad** | 2000+ lines, multiple concepts | Split into focused skills |
| **No cross-references** | Skill operates in isolation | Add references to ecosystem |
| **Invalid frontmatter** | Skill won't load | Only name + description fields |
| **Missing baseline tests** | Can't detect regression in audits | Preserve test scenarios |

---

## Integration with skill-audit

**After creating skill:**
1. Skill will be audited using skill-audit methodology
2. Baseline tests you created will be compared to future audits
3. Regression detection depends on preserved test scenarios
4. Keep test artifacts for auditing

**Baseline test format:**
```markdown
## Baseline Tests for [skill-name]

### Test 1 - [Scenario Type]
**Without skill:** [What agent did wrong]
**With skill:** [What agent does correctly]
**Pressure:** [Time/sunk cost/authority applied]

### Test 2 - [Scenario Type]
...
```

---

## The Bottom Line

**Creating skills IS TDD for documentation.**

Same process as writing code:
1. Write failing test (pressure scenarios without skill)
2. Watch it fail (document baseline)
3. Write minimal code (skill addressing failures)
4. Watch it pass (agents now comply)
5. Refactor (close loopholes)

**Plus marketplace integration:**
- Faction-appropriate structure
- Scope guidelines (300-1000 lines optimal)
- Cross-reference ecosystem
- Plugin registration
- Baseline preservation for audits

**For core TDD methodology, see obra's writing-skills.** This skill adds marketplace-specific structure and integration.
