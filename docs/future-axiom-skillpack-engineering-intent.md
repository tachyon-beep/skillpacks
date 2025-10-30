# Axiom/Skillpack-Engineering - Intent & High-Level Design

**Created**: 2025-10-29
**Status**: Intent Document (Not Yet Implemented)
**Faction**: Axiom (Creators of Marvels - Tooling, Infrastructure, Automation)
**Priority**: Build AFTER Yzmir Phase 1 completes

---

## Core Insight

**"Integration is an ongoing body of work, not a one-and-done event."**

After building Ordis (9 skills), Muna (9 skills), and starting Yzmir (~53-65 skills), a pattern emerges: each new faction/pack requires integration work across multiple files and systems. The knowledge of *how to design, maintain, and evolve skillpack collections* is itself a skillable domain.

---

## Why This Matters

### Current Integration Pain Points

Every time a new faction/pack is added:

1. **Plugin manifest updates** - plugin.json needs new skills, categories, keywords
2. **Documentation updates** - README.md faction list, PLUGIN_README.md descriptions
3. **Tutorial creation** - New skills need tutorials showing real-world usage
4. **Version coordination** - When does the system bump from 1.0 → 2.0? How to track?
5. **Cross-reference management** - When/how to add cross-faction references
6. **Testing at scale** - RED-GREEN-REFACTOR works for 18 skills, but what about 80+ skills?
7. **Skill granularity decisions** - Ordis has 9 skills, Yzmir has 53+ - what's right?

### The Pattern

We've done skillpack setup three times now:
1. **Ordis** (security-architect) - learned the basics
2. **Muna** (technical-writer) - learned cross-references
3. **Yzmir** (ai-ml-engineering) - learning 3-level routing and scale

After three iterations, patterns are emerging. That's when you extract the abstraction.

---

## Faction Assignment: Axiom

**Why Axiom (Creators of Marvels)?**
- ✅ Tooling and infrastructure - perfect fit
- ✅ Building tools to build tools (meta-infrastructure)
- ✅ Automation and developer experience
- ✅ Not security (Ordis), not documentation (Muna), not algorithms (Yzmir)

**Axiom's broader scope** includes:
- skillpack-engineering (this pack)
- ci-cd-engineering (build pipelines, deployment automation)
- developer-tooling (CLI tools, IDE integrations)
- infrastructure-as-code (Terraform, Kubernetes patterns)
- monitoring-observability (metrics, tracing, alerting)

All about **creating and maintaining systems that enable other work**.

---

## Proposed Pack: axiom/skillpack-engineering

### Purpose
Teach systematic approaches to designing, testing, integrating, and evolving skillpack collections at scale.

### Target Audience
- Skillpack creators building new factions
- Maintainers managing multi-faction systems
- Contributors adding skills to existing packs

### Meta-skill
**`axiom/skillpack-engineering/using-skillpack-engineering`**
- Routes based on task: design, testing, integration, evolution, versioning

### Core Skills (Proposed ~10 skills)

1. **skill-design-patterns**
   - When to split vs merge skills
   - Granularity decision framework
   - Skill boundary principles
   - When a skill is "too big" or "too small"

2. **faction-organization-principles**
   - Choosing the right faction for a domain
   - Avoiding cross-faction leakage
   - When to create new faction vs extend existing
   - Faction mental models and user expectations

3. **cross-reference-architecture**
   - Building knowledge graphs between skills
   - Managing internal (same-faction) vs external (cross-faction) references
   - Avoiding circular dependencies
   - When cross-refs add value vs create confusion

4. **red-green-refactor-methodology**
   - RED phase: Baseline testing without skill
   - GREEN phase: Writing skill to address failures
   - REFACTOR phase: Pressure testing and loophole closing
   - Scaling testing to 50+ skills

5. **integration-planning**
   - Ongoing integration as process, not event
   - Integration checklist templates
   - Version coordination across factions
   - Regression prevention strategies

6. **skillpack-versioning-strategy**
   - Semantic versioning for skillpacks
   - When to bump major/minor/patch
   - Breaking changes and migration paths
   - System-level version vs pack-level versions

7. **plugin-manifest-management**
   - Keeping plugin.json, PLUGIN_README.md, README.md in sync
   - Automated validation and consistency checks
   - Manifest schema evolution
   - Marketplace submission readiness

8. **tutorial-design-patterns**
   - Creating effective end-to-end scenarios
   - Showcasing multiple skills in one tutorial
   - Balancing comprehensiveness vs length
   - Tutorial testing methodology

9. **routing-architecture-design**
   - 2-level routing (Ordis/Muna: meta-skill → skill)
   - 3-level routing (Yzmir: primary → pack → skill)
   - When each pattern scales
   - Trade-offs and decision framework

10. **skillpack-testing-infrastructure**
    - Automated baseline testing
    - Regression detection across skill updates
    - Testing at scale (50+ skills)
    - Test documentation patterns

---

## Real Problems This Solves

### Problem 1: Integration Inconsistency
**Current**: Each faction integration is slightly different
**With skillpack-engineering**: Standardized integration checklist, automated validation

### Problem 2: Skill Granularity Confusion
**Current**: Ordis has 9 skills, Yzmir proposes 53-65 - what's right?
**With skillpack-engineering**: Decision framework based on domain complexity, user mental models

### Problem 3: Version Coordination Chaos
**Current**: Ad-hoc decisions about when system version bumps
**With skillpack-engineering**: Clear semantic versioning strategy for multi-faction systems

### Problem 4: Cross-Reference Explosion
**Current**: No systematic approach to managing dependencies
**With skillpack-engineering**: Architecture patterns for knowledge graphs, circular dependency prevention

### Problem 5: Testing Doesn't Scale
**Current**: RED-GREEN-REFACTOR tested manually for each skill
**With skillpack-engineering**: Testing infrastructure and automation patterns

---

## When to Build This

### Not Yet - Timing Matters

**Build AFTER**:
- ✅ Ordis complete (9 skills)
- ✅ Muna complete (9 skills)
- ⏳ Yzmir Phase 1 complete (~53-65 skills)

**Why wait?**
1. Need experience integrating 3 diverse factions (security, docs, ML)
2. Need to hit scale problems (routing, versioning, testing)
3. Need to discover patterns from doing integration 3+ times
4. Classic "write code three times before extracting abstraction"

**Trigger point**: When integrating Yzmir Phase 1 reveals repeated patterns and integration pain points.

---

## Preparation: Capture Patterns Now

While waiting to build the skillpack, create:

**`docs/SKILLPACK_ENGINEERING.md`** (living document):
- Skill granularity decision framework (learned from Ordis, Muna, Yzmir differences)
- Integration checklist template (evolved through 3 faction integrations)
- Version bump criteria (learned from system evolution)
- Cross-reference guidelines (learned from Ordis ↔ Muna interactions)
- Testing at scale strategies (learned when testing 80+ skills)

This becomes raw material for **axiom/skillpack-engineering** skills when ready to implement.

---

## Success Criteria (When Built)

### Phase 1 Complete When:
- ✅ Skillpack creators can systematically design new factions
- ✅ Integration is repeatable process with checklists
- ✅ Version coordination follows clear semantic versioning
- ✅ Cross-references are managed architecturally
- ✅ Testing scales to 100+ skills
- ✅ All skills pass RED-GREEN-REFACTOR

### Quality Gates:
- New faction integration takes 50% less time than before
- Version bumps follow consistent criteria
- Cross-reference circular dependencies prevented systematically
- Testing infrastructure catches regressions automatically
- Documentation stays in sync with manifest

---

## Future Evolution

### Phase 2+: Axiom Expansion
After **skillpack-engineering**, Axiom faction grows to include:
- **ci-cd-engineering** - Build automation, deployment pipelines
- **developer-tooling** - CLI tools, IDE plugins, dev experience
- **infrastructure-as-code** - Terraform, Kubernetes, cloud patterns
- **monitoring-observability** - Metrics, logging, tracing

All focused on **building tools and infrastructure** (Creators of Marvels).

---

## Key Design Principles

### 1. Meta-Infrastructure
Building tools to build skillpacks - one level of abstraction up from skillpacks themselves.

### 2. Pattern Recognition
Extract patterns from real experience (Ordis, Muna, Yzmir), not theoretical speculation.

### 3. Systematic Processes
Turn ad-hoc integration into repeatable checklists and frameworks.

### 4. Scale-First
Design for 20+ factions, 200+ skills from the start.

### 5. Living Documentation
Patterns document evolves as skillpack ecosystem grows.

---

## Open Questions (To Answer During Yzmir Integration)

1. **Skill granularity formula**: Is there a mathematical relationship between domain complexity and optimal skill count?
2. **Version coordination**: Should system version track latest faction, or be independent?
3. **Cross-reference timing**: When should cross-faction references be added (immediately, or after both factions stabilize)?
4. **Testing automation**: Can RED phase baselines be captured automatically?
5. **Routing depth**: Is 3-level routing the max, or might 4-level be needed for very large domains?

---

## Next Steps (When Ready)

1. **After Yzmir Phase 1 completes**: Review integration pain points
2. **Create SKILLPACK_ENGINEERING.md**: Document patterns learned
3. **RED phase testing**: Test skillpack design scenarios WITHOUT this pack
4. **Write axiom/skillpack-engineering skills**: Full RED-GREEN-REFACTOR
5. **Self-reference**: Use this pack to evolve itself (meta-meta-engineering)

---

## Related Documents

- `/source/planning/2025-10-29-yzmir-ai-engineering-design.md` - Yzmir design (demonstrates 3-level routing, scale challenges)
- `/source/docs/plans/2025-10-29-phase-4-public-release-design.md` - Phase 4 execution (demonstrates integration work)
- `/source/CONTRIBUTING.md` - Current skill creation guide (will be enhanced by this pack)
- `/source/DEVELOPMENT.md` - Development history (documents Ordis/Muna patterns)

---

## Appendix: Why "Integration as Ongoing Work" Is Right

### Traditional View (Wrong)
Integration is a one-time event:
- Build Ordis → integrate
- Build Muna → integrate
- Build Yzmir → integrate
- Done

### Reality-Based View (Right)
Integration is continuous process:
- Ordis 1.0 + Muna 1.0 → cross-references emerge
- Yzmir 1.0 → routing patterns evolve
- Tutorial 5 → demonstrates Ordis + Muna together
- Future: Axiom references all three
- Ongoing: Skills cross-reference as domains mature

**Integration compounds over time** - each new faction increases integration complexity non-linearly.

This is why **axiom/skillpack-engineering** matters: it systematizes the ongoing work.

---

**End of Intent Document**

*This document captures the insight that skillpack engineering is itself a domain worthy of systematic treatment. Implementation deferred until patterns fully emerge from Yzmir Phase 1 integration.*
