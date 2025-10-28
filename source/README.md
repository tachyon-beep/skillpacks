# Security Architect & Technical Writer Skill Packs

**Status**: Phase 3 Complete! All 18 Skills Implemented, Ready for Public Release
**Created**: 2025-10-28
**Packs**: 2 (security-architect, technical-writer)
**Total Skills**: 18 (2 meta + 8 core + 8 extensions)

---

## What This Is

Universal skill packs for Claude Code providing:
- **security-architect**: Threat modeling, security controls, architecture review, secure patterns, classified systems, compliance, authorization
- **technical-writer**: Documentation structure, clarity, diagrams, testing, security-aware docs, incident response, governance, operational acceptance

**Key Innovation**: Layered architecture (universal core + specialized extensions) with bidirectional cross-references creating knowledge graph.

---

## Current Phase: Extensions Complete (Phase 3)

**Phase 1 Skills Progress** (Foundation - Meta + Initial Core):
1. ✅ `ordis/security-architect/using-security-architect` (meta-skill, routing) - **COMPLETE**
2. ✅ `ordis/security-architect/threat-modeling` (core technique) - **COMPLETE**
3. ✅ `muna/technical-writer/using-technical-writer` (meta-skill, routing) - **COMPLETE**
4. ✅ `muna/technical-writer/documentation-structure` (core reference) - **COMPLETE**

**Phase 1 Complete!** Foundation proven with meta-skills + 2 core skills.

---

**Phase 2 Skills Progress** (Core Skills - Universal Applicability):
1. ✅ `ordis/security-architect/security-controls-design` (defense-in-depth, boundaries) - **COMPLETE**
2. ✅ `muna/technical-writer/clarity-and-style` (active voice, concrete examples) - **COMPLETE**
3. ✅ `ordis/security-architect/secure-by-design-patterns` (zero-trust, immutable) - **COMPLETE**
4. ✅ `muna/technical-writer/diagram-conventions` (decision tree, semantic labels) - **COMPLETE**
5. ✅ `ordis/security-architect/security-architecture-review` (systematic checklists) - **COMPLETE**
6. ✅ `muna/technical-writer/documentation-testing` (5 dimensions: completeness, accuracy, findability, examples, walkthrough) - **COMPLETE**

**Phase 2 Complete!** All 6 core skills implemented with rigorous RED-GREEN-REFACTOR testing. Universal skills applicable to any project.

---

**Phase 3 Skills Progress** (Extension Skills - Specialized Contexts):

**Ordis (Security Architect) - 4 Extension Skills:**
1. ✅ `ordis/security-architect/classified-systems-security` (Bell-LaPadula MLS, fail-fast) - **COMPLETE**
2. ✅ `ordis/security-architect/compliance-awareness-and-mapping` (framework discovery, ISM/HIPAA/SOC2) - **COMPLETE**
3. ✅ `ordis/security-architect/security-authorization-and-accreditation` (ATO/AIS/T&E, SSP/SAR/POA&M) - **COMPLETE**
4. ✅ `ordis/security-architect/documenting-threats-and-controls` (security ADRs, threat docs, control docs) - **COMPLETE**

**Muna (Technical Writer) - 4 Extension Skills:**
1. ✅ `muna/technical-writer/security-aware-documentation` (sanitizing examples, threat disclosure) - **COMPLETE**
2. ✅ `muna/technical-writer/incident-response-documentation` (5-phase template, escalation, runbooks) - **COMPLETE**
3. ✅ `muna/technical-writer/itil-and-governance-documentation` (RFC, SLA, DR plans) - **COMPLETE**
4. ✅ `muna/technical-writer/operational-acceptance-documentation` (readiness, go-live, handover) - **COMPLETE**

**Phase 3 Complete!** All 8 extension skills implemented for specialized contexts (government/defense, healthcare, enterprise operations).

**Total Effort (Phases 1-3)**: ~45-50 hours

---

## Quick Start for Claude

**When you arrive in new repo, read in this order:**

1. **This file** (README.md) - Status and quick start
2. **planning/GETTING_STARTED.md** - Implementation workflow and critical reminders
3. **planning/2025-10-28-security-architect-technical-writer-design.md** - Complete design (10,000 words, every skill detailed)

**Then start implementing:**
```
I'm using the writing-skills skill to create ordis/security-architect/using-security-architect.
```

---

## Critical Reminders

### The Iron Law
```
NO SKILL WITHOUT A FAILING TEST FIRST
```

### Required Skills to USE
- `superpowers:using-superpowers` (mandatory first)
- `superpowers:writing-skills` (skill creation methodology)
- `superpowers:testing-skills-with-subagents` (testing approach)

### Workflow Per Skill
```
RED (baseline test WITHOUT skill)
  ↓
GREEN (write skill addressing failures)
  ↓
REFACTOR (close loopholes under pressure)
  ↓
COMMIT (one skill at a time)
```

---

## Files in This Repo

```
skillpacks/source/
├── README.md (this file)
├── FACTIONS.md (faction organization guide)
├── planning/
│   ├── GETTING_STARTED.md (implementation guide, critical context)
│   ├── REAL_WORLD_TEST_SCENARIOS.md (test cases from Elspeth)
│   └── 2025-10-28-security-architect-technical-writer-design.md (complete design)
├── ordis/ (Protectors of Order - security & governance)
│   └── security-architect/ (to be created in Phase 1)
├── muna/ (Weavers of Harmony - documentation & synthesis)
│   └── technical-writer/ (to be created in Phase 1)
├── axiom/ (Creators of Marvels - tooling & infrastructure)
├── bravos/ (Champions of Action - practical tactics)
├── lyra/ (Nomadic Artists - UX & creative)
└── yzmir/ (Magicians of Mind - theory & algorithms)
```

**Faction Organization**: Skills are thematically grouped by Altered TCG-inspired factions. See FACTIONS.md for details.

---

## Repository Independence

This project is **orthogonal to Elspeth** (the project where it was designed). It has:
- ✅ Complete design documentation
- ✅ Implementation workflow
- ✅ Testing methodology
- ✅ Phase-by-phase plan
- ✅ Zero dependencies on Elspeth codebase

**Claude can work independently with ONLY contents of this repo.**

---

## Implementation Phases

| Phase | Skills | Effort | Status |
|-------|--------|--------|--------|
| **Phase 1: Foundation** | 4 (2 meta + 2 core) | 10-15h | ✅ Complete |
| **Phase 2: Core Skills** | 6 core | 18-30h | ✅ Complete |
| **Phase 3: Extensions** | 8 specialized | 24-40h | ✅ Complete |
| **Phase 4: Polish** | Docs + public release | 8-15h | 📋 Next |

**Total**: 60-100 hours over 8-9 weeks
**Completed**: ~45-50 hours (Phases 1-3)
**Remaining**: ~8-15 hours (Phase 4 polish)

---

## Phase 1 Exit Criteria (✅ Complete)

- ✅ All 4 skills pass RED-GREEN-REFACTOR
- ✅ Meta-skills route correctly to core skills
- ✅ Cross-references work (threat-modeling ↔ documentation-structure)
- ✅ Personal use validated on real scenario
- ✅ Committed to git with clear messages

## Phase 2 Exit Criteria (✅ Complete)

- ✅ All 6 core skills pass RED-GREEN-REFACTOR
- ✅ Skills applicable across different project types (APIs, services, pipelines, docs)
- ✅ Security skills complement writing skills (bidirectional cross-references)
- ✅ Each skill tested with baseline → skill → verification cycle
- ✅ Committed to git with detailed RED-GREEN-REFACTOR messages

## Phase 3 Exit Criteria (✅ Complete)

- ✅ All 8 extension skills pass RED-GREEN-REFACTOR
- ✅ Specialized skills for government/defense contexts (classified systems, ATO, compliance)
- ✅ Specialized skills for enterprise operations (ITIL, incident response, DR)
- ✅ Security-aware documentation prevents credential/PII leaks
- ✅ Cross-cutting skills bridge Ordis + Muna (documenting-threats-and-controls)
- ✅ Each skill tested and committed with detailed methodology

---

## What Success Looks Like

**After Phase 1**: ✅ Foundation proven, can load meta-skills to route to core skills
**After Phase 2**: ✅ Universal core skills usable across any project
**After Phase 3**: ✅ Specialized extensions for high-security/regulated environments
**After Phase 4**: 📋 Public-quality skill packs ready for community use

---

## Design Principles

1. **Layered**: Core (universal) + Extensions (specialized)
2. **Modular**: Load only what you need
3. **Cross-Referenced**: Bidirectional knowledge graph
4. **TDD for Documentation**: Every skill tested before deployment
5. **Meta-Aware**: Teach patterns, not exhaustive lists

Example: `compliance-awareness-and-mapping` teaches "frameworks vary by jurisdiction" + discovery process, NOT hardcoded list of all frameworks.

---

## Origin Story

Designed alongside **Elspeth** (security-first LLM orchestration platform) but universal and reusable. Informed by:
- Bell-LaPadula MLS enforcement
- ADR-quality documentation practices
- TDD-based complexity reduction methodology
- High-security/regulated environment patterns

**Applies to any project** - web apps, mobile, embedded, data pipelines, etc.

---

## Future Vision (Not Current Scope)

**Web of complementary skill packs**:
- security-architect + technical-writer (current)
- performance-engineer + reliability-engineer (future)
- data-architect + privacy-engineer (future)

Each pack: universal core + extensions, bidirectional cross-references.

---

## Getting Help

**Lost?** Read in order:
1. This file (status)
2. planning/GETTING_STARTED.md (workflow)
3. planning/2025-10-28-security-architect-technical-writer-design.md (complete context)

**Stuck?** Review:
- `superpowers:writing-skills` for methodology
- `superpowers:testing-skills-with-subagents` for testing approach
- planning/GETTING_STARTED.md "Common Mistakes" section

---

## Ready to Start?

```bash
# In new repo
cat planning/GETTING_STARTED.md  # Read implementation guide
cat planning/2025-10-28-security-architect-technical-writer-design.md  # Read full design

# Announce skill usage
"I'm using the writing-skills skill to create security-architect/using-security-architect."

# Follow RED-GREEN-REFACTOR
# Commit when done
# Move to next skill
```

**You have everything you need. Trust the process. Ship one skill at a time.**

---

## 🎉 Phase 3 Complete! All Core Functionality Implemented 🎉

**Achievements**:
- ✅ 18 skills total (2 meta + 8 core + 8 extensions)
- ✅ Rigorous RED-GREEN-REFACTOR testing for every skill
- ✅ Universal skills (threat modeling, security controls, documentation structure)
- ✅ Specialized skills (classified systems, ATO, compliance, ITIL, incident response)
- ✅ ~45-50 hours invested

**What's Usable Now**:
- Complete security-architect skillpack (10 skills)
- Complete technical-writer skillpack (8 skills)
- Ready for personal/team use
- Proven methodology with real-world examples

**Next Phase**: Phase 4 - Polish for Public Release (README polish, contribution guidelines, public announcement)
