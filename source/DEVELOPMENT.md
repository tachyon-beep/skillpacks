# Development Guide

This guide documents the implementation history and development workflow for these skillpacks.

---

## Implementation History

### Phase 1: Foundation (4 skills, 10-15 hours)

**Goal:** Prove the concept with meta-skills + initial core skills

**Skills Implemented:**
1. `ordis/security-architect/using-security-architect` (meta-skill, routing) - **COMPLETE**
2. `ordis/security-architect/threat-modeling` (core technique) - **COMPLETE**
3. `muna/technical-writer/using-technical-writer` (meta-skill, routing) - **COMPLETE**
4. `muna/technical-writer/documentation-structure` (core reference) - **COMPLETE**

**Exit Criteria:**
- ✅ All 4 skills pass RED-GREEN-REFACTOR
- ✅ Meta-skills route correctly to core skills
- ✅ Cross-references work (threat-modeling ↔ documentation-structure)
- ✅ Personal use validated on real scenario
- ✅ Committed to git with clear messages

**Achievements:**
- Foundation proven with meta-skills + 2 core skills
- Meta-skill routing demonstrated (using-security-architect → threat-modeling)
- Bidirectional cross-references established (security ↔ documentation)
- Personal validation completed on real project

**Effort:** ~10-15 hours

---

### Phase 2: Core Skills (6 skills, 18-30 hours)

**Goal:** Build universal core skills applicable to any project

**Skills Implemented:**
1. `ordis/security-architect/security-controls-design` (defense-in-depth, boundaries) - **COMPLETE**
2. `muna/technical-writer/clarity-and-style` (active voice, concrete examples) - **COMPLETE**
3. `ordis/security-architect/secure-by-design-patterns` (zero-trust, immutable) - **COMPLETE**
4. `muna/technical-writer/diagram-conventions` (decision tree, semantic labels) - **COMPLETE**
5. `ordis/security-architect/security-architecture-review` (systematic checklists) - **COMPLETE**
6. `muna/technical-writer/documentation-testing` (5 dimensions: completeness, accuracy, findability, examples, walkthrough) - **COMPLETE**

**Exit Criteria:**
- ✅ All 6 core skills pass RED-GREEN-REFACTOR
- ✅ Skills applicable across different project types (APIs, services, pipelines, docs)
- ✅ Security skills complement writing skills (bidirectional cross-references)
- ✅ Each skill tested with baseline → skill → verification cycle
- ✅ Committed to git with detailed RED-GREEN-REFACTOR messages

**Achievements:**
- Universal core skills completed and tested
- Skills proven across diverse project types (web apps, APIs, data pipelines)
- Strong bidirectional cross-references between Ordis and Muna factions
- Every skill validated with rigorous RED-GREEN-REFACTOR methodology

**Effort:** ~18-30 hours

---

### Phase 3: Extensions (8 skills, 24-40 hours)

**Goal:** Add specialized skills for high-security and regulated contexts

**Ordis (Security Architect) - 4 Extension Skills:**
1. `ordis/security-architect/classified-systems-security` (Bell-LaPadula MLS, fail-fast) - **COMPLETE**
2. `ordis/security-architect/compliance-awareness-and-mapping` (framework discovery, ISM/HIPAA/SOC2) - **COMPLETE**
3. `ordis/security-architect/security-authorization-and-accreditation` (ATO/AIS/T&E, SSP/SAR/POA&M) - **COMPLETE**
4. `ordis/security-architect/documenting-threats-and-controls` (security ADRs, threat docs, control docs) - **COMPLETE**

**Muna (Technical Writer) - 4 Extension Skills:**
1. `muna/technical-writer/security-aware-documentation` (sanitizing examples, threat disclosure) - **COMPLETE**
2. `muna/technical-writer/incident-response-documentation` (5-phase template, escalation, runbooks) - **COMPLETE**
3. `muna/technical-writer/itil-and-governance-documentation` (RFC, SLA, DR plans) - **COMPLETE**
4. `muna/technical-writer/operational-acceptance-documentation` (readiness, go-live, handover) - **COMPLETE**

**Exit Criteria:**
- ✅ All 8 extension skills pass RED-GREEN-REFACTOR
- ✅ Specialized skills for government/defense contexts (classified systems, ATO, compliance)
- ✅ Specialized skills for enterprise operations (ITIL, incident response, DR)
- ✅ Security-aware documentation prevents credential/PII leaks
- ✅ Cross-cutting skills bridge Ordis + Muna (documenting-threats-and-controls)
- ✅ Each skill tested and committed with detailed methodology

**Achievements:**
- Complete extension skills for specialized contexts
- Government/defense skills: Bell-LaPadula MLS enforcement, ATO/AIS processes, compliance frameworks
- Enterprise operations skills: ITIL/governance, incident response, disaster recovery, operational acceptance
- Security-aware documentation with sanitized examples (no credential leaks)
- Cross-cutting skill (documenting-threats-and-controls) bridges security and documentation domains

**Effort:** ~17-25 hours (actual), 24-40 hours (estimated)

---

### Total Investment (Phases 1-3)

**Completed:** ~45-50 hours
**Original Estimate:** 52-85 hours

**Deliverables:**
- 18 skills total (2 meta + 8 core + 8 extensions)
- Complete security-architect skillpack (10 skills)
- Complete technical-writer skillpack (8 skills)
- Ready for personal/team use
- Proven methodology with real-world examples

---

## Development Workflow

### RED-GREEN-REFACTOR Methodology

All skills MUST be tested before deployment using the RED-GREEN-REFACTOR cycle.

#### RED Phase: Baseline Test WITHOUT Skill

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

**Critical Rule:** Never skip baseline testing. If you can't identify concrete failures, the skill may not be needed.

---

#### GREEN Phase: Create Skill Addressing Failures

**Goal:** Write skill that addresses every documented baseline failure.

**Process:**
1. Review baseline failures from RED phase
2. Write skill that explicitly teaches missing patterns/methodologies
3. Include real-world examples (3+ concrete examples)
4. Add cross-references to complementary skills
5. Document "When to Use" section with clear triggers

**Skill Structure:**
```markdown
---
name: skill-name-here
description: Use when [trigger] - covers [what it teaches]
---

# Skill Name

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

**Design Principles:**
- Teach patterns and methodology (not exhaustive lists)
- Use concrete examples (no placeholders like `YOUR_KEY_HERE`)
- Include decision trees and checklists
- Cross-reference bidirectionally (skill A → skill B, skill B → skill A)

---

#### REFACTOR Phase: Verification Test WITH Skill

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

**Testing Tools:**
- Use `superpowers:testing-skills-with-subagents` for rigorous testing
- Dispatch fresh subagent for each test (no context pollution)
- Test same scenario WITHOUT skill (RED), then WITH skill (REFACTOR)

---

### Git Commit Pattern

All commits follow a consistent pattern established in Phases 1-3.

**Skill Implementation Commits:**
```
Implement {faction}/{pack}/{skill-name} (Phase X, Skill Y/Z)

RED-GREEN-REFACTOR cycle complete:

RED Phase:
- Tested scenario [X] without skill
- Documented [N] baseline failures: [brief list]

GREEN Phase:
- Created skill addressing all failures
- Included [pattern/methodology Y]
- Added [N] real-world examples

REFACTOR Phase:
- Tested WITH skill - all failures resolved
- Tested under pressure - maintains discipline
- Skill is bulletproof

[Additional context about skill]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Progress Milestone Commits:**
```
Update README: Phase X complete! All Y skills implemented

Phase X Achievement:
- ✅ [Key achievement 1]
- ✅ [Key achievement 2]
- ✅ [Summary statistics]

[Additional details]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Key Principles:**
- One skill per commit (no batching)
- Include RED-GREEN-REFACTOR summary in commit message
- Document baseline failures explicitly
- Note testing under pressure
- Use emoji footer: `🤖 Generated with [Claude Code]...`
- Always co-author with Claude

---

### Faction Organization

Skills are grouped by Altered TCG-inspired factions for thematic coherence.

**The Six Factions:**

1. **Ordis** (Protectors of Order): Security architecture, controls, governance, protection
2. **Muna** (Weavers of Harmony): Documentation, synthesis, cross-discipline patterns
3. **Axiom** (Creators of Marvels): Tooling, infrastructure, automation (future)
4. **Bravos** (Champions of Action): Practical tactics, debugging, reliability (future)
5. **Lyra** (Nomadic Artists): UX, creative design, human-centered (future)
6. **Yzmir** (Magicians of Mind): Theory, algorithms, cryptography (future)

**Current Implementation:**
- **Ordis:** `security-architect` (10 skills) - Complete
- **Muna:** `technical-writer` (8 skills) - Complete

**Naming Convention:**
```
{faction}/{pack-name}/{skill-name}/SKILL.md
```

**Examples:**
- `ordis/security-architect/threat-modeling/SKILL.md`
- `muna/technical-writer/documentation-structure/SKILL.md`

**Cross-Faction References:**
Skills can and should reference skills from other factions:
- `ordis/security-architect/documenting-threats-and-controls` → `muna/technical-writer/documentation-structure`
- `muna/technical-writer/security-aware-documentation` → `ordis/security-architect/threat-modeling`

See [FACTIONS.md](FACTIONS.md) for complete faction organization guide.

---

## Testing Skills

### Testing Methodology

Use `superpowers:testing-skills-with-subagents` for all skill testing.

**Workflow:**
1. **Dispatch subagent** - Fresh Claude instance with no context pollution
2. **RED phase test** - Test scenario WITHOUT skill, document failures
3. **GREEN phase** - Write skill addressing failures
4. **Dispatch new subagent** - Fresh instance for verification test
5. **REFACTOR phase test** - Test scenario WITH skill, verify all failures resolved

**Testing Under Pressure:**
- Time constraints: "You have 90 minutes to ship this emergency patch"
- Edge cases: Test boundary conditions and unusual scenarios
- Scope creep: "Actually, we also need [additional requirement]"

**Success Criteria:**
- All baseline failures resolved
- Skill maintains discipline under pressure
- No rationalization or shortcuts taken
- Concrete, actionable guidance provided

---

## Critical Reminders

### The Iron Law
```
NO SKILL WITHOUT A FAILING TEST FIRST
```

### Required Skills to USE
Before creating new skills, you must use:
- `superpowers:using-superpowers` (mandatory first)
- `superpowers:writing-skills` (skill creation methodology)
- `superpowers:testing-skills-with-subagents` (testing approach)

### Workflow Per Skill
```
RED (baseline test WITHOUT skill)
  ↓
GREEN (write skill addressing failures)
  ↓
REFACTOR (verify under pressure)
  ↓
COMMIT (one skill at a time)
```

### Common Mistakes to Avoid
1. **Skipping RED phase** - Always test baseline first
2. **Batching commits** - One skill per commit
3. **Incomplete examples** - Use complete, runnable examples (not placeholders)
4. **Missing cross-references** - Link to complementary skills bidirectionally
5. **Testing with context pollution** - Always use fresh subagent for verification

---

## Design Principles

1. **Layered:** Core (universal) + Extensions (specialized contexts)
2. **Modular:** Load only what you need
3. **Cross-Referenced:** Bidirectional knowledge graph between skills
4. **TDD for Documentation:** Every skill tested before deployment
5. **Meta-Aware:** Teach patterns and methodology, not exhaustive lists

**Example:** `compliance-awareness-and-mapping` teaches "frameworks vary by jurisdiction" + discovery process, NOT hardcoded list of all frameworks.

---

## Origin Story

Designed alongside **Elspeth** (security-first LLM orchestration platform) but universal and reusable.

**Informed by:**
- Bell-LaPadula MLS enforcement in classified systems
- ADR-quality documentation practices
- TDD-based complexity reduction methodology
- High-security/regulated environment patterns (government, healthcare, finance)

**Applies to any project:** Web apps, mobile, embedded, data pipelines, infrastructure.

---

## What Success Looks Like

**After Phase 1:** ✅ Foundation proven, can load meta-skills to route to core skills
**After Phase 2:** ✅ Universal core skills usable across any project
**After Phase 3:** ✅ Specialized extensions for high-security/regulated environments
**After Phase 4:** 📋 Public-quality skill packs ready for community use

---

## Next Steps

**Phase 4: Public Release Polish (8-14 hours)**
- Wave 1: Repository polish (README restructure, CONTRIBUTING.md, governance files)
- Wave 2: Plugin packaging (plugin.json, marketplace submission)
- Wave 3: Tutorials (5 end-to-end scenarios across diverse domains)

See [docs/plans/2025-10-29-phase-4-public-release-implementation.md](docs/plans/2025-10-29-phase-4-public-release-implementation.md) for Phase 4 implementation plan.

---

## Questions?

**Implementation workflow:** Read this file
**Skill creation methodology:** Use `superpowers:writing-skills`
**Testing approach:** Use `superpowers:testing-skills-with-subagents`
**Faction organization:** See [FACTIONS.md](FACTIONS.md)
**Contribution guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md) (after Phase 4)

**You have everything you need. Trust the process. Ship one skill at a time.**
