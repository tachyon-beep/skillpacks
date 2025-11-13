---
name: using-system-architect
description: Use when you have architecture documentation from system-archaeologist and need critical assessment, refactoring recommendations, or improvement prioritization - routes to appropriate architect specialist skills
---

# Using System Architect

## Overview

**System Architect provides critical assessment and strategic recommendations for existing codebases.**

The architect works WITH the archaeologist: archaeologist documents what exists (neutral), architect assesses quality and recommends improvements (critical).

## When to Use

Use system-architect skills when:
- You have archaeologist outputs (subsystem catalog, diagrams, architecture report)
- Need to assess architectural quality ("how bad is it?")
- Need to identify and catalog technical debt
- Need refactoring strategy recommendations
- Need to prioritize improvements with limited resources
- User asks: "What should I fix first?" or "Is this architecture good?"

## The Pipeline

```
Archaeologist → Architect → (Future: Project Manager)
(documents)     (assesses)    (manages execution)
```

**Archaeologist** (axiom-system-archaeologist):
- Neutral documentation of existing architecture
- Subsystem catalog, C4 diagrams, dependency mapping
- "Here's what you have"

**Architect** (axiom-system-architect - this plugin):
- Critical assessment of quality
- Technical debt cataloging
- Refactoring recommendations
- Priority-based roadmaps
- "Here's what's wrong and how to fix it"

**Project Manager** (future: axiom-project-manager):
- Execution tracking
- Sprint planning
- Risk management
- "Here's how we'll track the fixes"

## Available Architect Skills

### 1. assessing-architecture-quality

**Use when:**
- Writing architecture quality assessment
- Feel pressure to soften critique or lead with strengths
- Contract renewal or stakeholder relationships influence tone
- CTO built the system and will review your assessment

**Addresses:**
- Diplomatic softening under relationship pressure
- Sandwich structure (strengths → critique → positives)
- Evolution framing ("opportunities" vs "problems")
- Economic or authority influence on assessment

**Output:** Direct, evidence-based architecture assessment

---

### 2. identifying-technical-debt

**Use when:**
- Cataloging technical debt items
- Under time pressure with incomplete analysis
- Tempted to explain methodology instead of delivering document
- Deciding between complete analysis (miss deadline) vs quick list

**Addresses:**
- Analysis paralysis (explaining instead of executing)
- Incomplete entries to save time
- No limitations section (false completeness)
- Missing delivery commitments

**Output:** Properly structured technical debt catalog (complete or partial with limitations)

---

### 3. prioritizing-improvements

**Use when:**
- Creating improvement roadmap from technical debt catalog
- Stakeholders disagree with your technical prioritization
- CEO says "security is fine, we've never been breached"
- You're tempted to "bundle" work to satisfy stakeholders
- Time pressure influences prioritization decisions

**Addresses:**
- Compromising on security-first prioritization
- Validating "we've never been breached" flawed reasoning
- Bundling as rationalization for deprioritizing security
- Accepting stakeholder preferences over risk-based priorities

**Output:** Risk-based improvement roadmap with security as Phase 1

---

### Future Skills (Not Yet Implemented)

**4. analyzing-architectural-patterns**
- Pattern identification (intentional vs accidental)
- Anti-pattern detection
- Pattern consistency analysis

**5. recommending-refactoring-strategies**
- Strangler fig vs big-bang rewrite
- Dependency breaking strategies
- Incremental improvement paths

**6. documenting-architecture-decisions**
- ADR (Architecture Decision Records)
- Historical context reconstruction
- Trade-off documentation

**7. estimating-refactoring-effort**
- T-shirt sizing
- Realistic timeline estimation
- Risk-adjusted effort calculation

**8. generating-improvement-metrics**
- Baseline metrics
- Target metrics
- Progress tracking strategy

## Routing Guide

### Scenario: "Assess this codebase"

**Step 1:** Use archaeologist first
```
/system-archaeologist
→ Produces: subsystem catalog, diagrams, report
```

**Step 2:** Use architect for assessment
```
Read archaeologist outputs
→ Use: assessing-architecture-quality
→ Produces: 05-architecture-assessment.md
```

**Step 3:** Catalog technical debt
```
Read assessment
→ Use: identifying-technical-debt
→ Produces: 06-technical-debt-catalog.md
```

---

### Scenario: "How bad is my technical debt?"

**If no existing analysis:**
```
1. Archaeologist: document architecture
2. Architect: assess quality
3. Architect: catalog technical debt
```

**If archaeologist analysis exists:**
```
1. Read existing subsystem catalog
2. Use: identifying-technical-debt
```

---

### Scenario: "What should I fix first?"

**Current skills:**
```
1. Archaeologist: document architecture
2. Use: assessing-architecture-quality
3. Use: identifying-technical-debt
4. Manually prioritize based on severity ratings in outputs
```

**Future (when prioritizing-improvements skill exists):**
```
1-3. Same as above
4. Use: prioritizing-improvements
   → Produces: 09-improvement-roadmap.md
```

---

## Integration with Other Skillpacks

### Security Assessment (ordis-security-architect)

**Workflow:**
```
Architect identifies security issues
→ Ordis provides threat modeling (STRIDE)
→ Ordis designs security controls
→ Architect catalogs as technical debt
```

**Example:**
- Architect: "6 different auth implementations"
- Ordis: "Threat model for unified auth service"
- Architect: "Catalog security remediation work"

---

### Documentation (muna-technical-writer)

**Workflow:**
```
Architect produces ADRs and assessments
→ Muna structures professional documentation
→ Muna applies clarity and style guidelines
```

**Example:**
- Architect: "Architecture Decision Records"
- Muna: "Format as professional architecture docs"

---

### Python Engineering (axiom-python-engineering)

**Workflow:**
```
Architect identifies Python-specific issues
→ Python pack provides modern patterns
→ Architect catalogs Python modernization work
```

**Example:**
- Architect: "Python 2.7 EOL, no type hints"
- Python pack: "Python 3.12 migration + type system"
- Architect: "Catalog migration technical debt"

---

## Typical Workflow

**Complete codebase improvement pipeline:**

1. **Archaeologist Phase**
   ```
   /system-archaeologist
   → 01-discovery-findings.md
   → 02-subsystem-catalog.md
   → 03-diagrams.md
   → 04-final-report.md
   ```

2. **Architect Phase (YOU ARE HERE)**
   ```
   Use: assessing-architecture-quality
   → 05-architecture-assessment.md

   Use: identifying-technical-debt
   → 06-technical-debt-catalog.md
   ```

3. **Specialist Integration**
   ```
   Security issues → /security-architect
   Python issues → /python-engineering
   ML issues → /ml-production
   Documentation → /technical-writer
   ```

4. **Project Management** (future)
   ```
   /project-manager
   → Creates tracked project from roadmap
   → Sprint planning, progress tracking
   ```

## Decision Tree

```
Do you have architecture documentation?
├─ No → Use archaeologist first (/system-archaeologist)
└─ Yes → Continue below

What do you need?
├─ Quality assessment → Use: assessing-architecture-quality
├─ Technical debt catalog → Use: identifying-technical-debt
├─ Refactoring strategy → (Future: recommending-refactoring-strategies)
├─ Priority roadmap → (Future: prioritizing-improvements)
└─ Effort estimates → (Future: estimating-refactoring-effort)
```

## Common Patterns

### Pattern 1: Legacy Codebase Assessment

```
1. /system-archaeologist (if no docs exist)
2. Use: assessing-architecture-quality
3. Use: identifying-technical-debt
4. Review outputs with stakeholders
5. Use specialist packs for domain-specific issues
```

---

### Pattern 2: Technical Debt Audit

```
1. Read existing architecture docs
2. Use: identifying-technical-debt
3. Present catalog to stakeholders
4. (Future) Use: prioritizing-improvements for roadmap
```

---

### Pattern 3: Architecture Review

```
1. /system-archaeologist
2. Use: assessing-architecture-quality
3. Identify patterns and anti-patterns
4. (Future) Use: documenting-architecture-decisions for ADRs
```

---

## Quick Reference

| Need | Use This Skill |
|------|----------------|
| Quality assessment | assessing-architecture-quality |
| Technical debt catalog | identifying-technical-debt |
| Pattern analysis | (Future) analyzing-architectural-patterns |
| Refactoring strategy | (Future) recommending-refactoring-strategies |
| Priority roadmap | (Future) prioritizing-improvements |
| Effort estimates | (Future) estimating-refactoring-effort |
| ADRs | (Future) documenting-architecture-decisions |
| Metrics | (Future) generating-improvement-metrics |

## Status

**Current Status:** 3 of 8 skills implemented (v0.2.0)

**Production-ready skills:**
- ✅ assessing-architecture-quality
- ✅ identifying-technical-debt
- ✅ prioritizing-improvements

**Future skills (see docs/future-axiom-improvement-pipeline-intent.md):**
- ⏳ analyzing-architectural-patterns
- ⏳ recommending-refactoring-strategies
- ⏳ documenting-architecture-decisions
- ⏳ estimating-refactoring-effort
- ⏳ generating-improvement-metrics

## Related Documentation

- **Intent document:** `/home/john/skillpacks/docs/future-axiom-improvement-pipeline-intent.md`
- **Archaeologist plugin:** `axiom-system-archaeologist`
- **Future PM plugin:** `axiom-project-manager` (not yet implemented)

## The Bottom Line

**Use archaeologist to document what exists.**
**Use architect to assess quality and recommend fixes.**
**Use specialist packs for domain-specific improvements.**

Archaeologist is neutral observer.
Architect is critical assessor.

Together they form the analysis → strategy pipeline.
