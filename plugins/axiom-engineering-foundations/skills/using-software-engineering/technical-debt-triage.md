# Technical Debt Triage

Identify, categorize, and prioritize technical debt. Decide what to fix, what to live with.

## Core Principle

**Not all debt is worth paying.** Some debt slows you down daily; some you'll never touch again. Triage ruthlessly: fix what hurts, ignore what doesn't, and track what might matter later.

## When to Use This

- "Everything needs fixing, where do I start?"
- Sprint planning for maintenance work
- Justifying refactoring to stakeholders
- Deciding whether to fix now or later
- New to codebase, assessing health

**Don't use for**: Actually fixing debt (use [systematic-refactoring.md](systematic-refactoring.md)), understanding unfamiliar code (use [codebase-confidence-building.md](codebase-confidence-building.md) first).

---

## The Triage Process

```
┌─────────────────┐
│ 1. IDENTIFY     │ ← Find the debt
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. CATEGORIZE   │ ← What kind of debt?
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. ASSESS       │ ← Impact and cost
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. PRIORITIZE   │ ← What to fix first
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. DECIDE       │ ← Fix, defer, or accept
└─────────────────┘
```

---

## Phase 1: Identify

**Find where debt lives.**

### Sources of Debt

| Source | How to Find |
|--------|-------------|
| **Code smells** | Linter warnings, complexity metrics |
| **TODO/FIXME comments** | `grep -r "TODO\|FIXME\|HACK\|XXX"` |
| **Test gaps** | Coverage reports, untested paths |
| **Documentation gaps** | Missing/outdated docs |
| **Dependency rot** | Outdated packages, security vulns |
| **Architecture violations** | Wrong layers, circular deps |
| **Performance issues** | Slow queries, N+1 problems |
| **Team friction** | "Everyone hates working in this file" |

### Automated Discovery

```bash
# Code complexity
radon cc src/ -a -s  # Python
npx complexity-report src/  # JavaScript

# TODO/FIXME count
grep -r "TODO\|FIXME\|HACK" --include="*.py" | wc -l

# Outdated dependencies
pip list --outdated  # Python
npm outdated  # Node

# Security vulnerabilities
pip-audit  # Python
npm audit  # Node

# Test coverage gaps
pytest --cov=src --cov-report=term-missing
```

### Team Knowledge

Ask the team:
- "What code do you dread touching?"
- "What slows you down regularly?"
- "What would you fix if you had a week?"
- "What breaks unexpectedly?"

---

## Phase 2: Categorize

**Different debt types need different approaches.**

### Debt Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Reckless/Deliberate** | Knew it was wrong, shipped anyway | "No time for tests" |
| **Reckless/Inadvertent** | Didn't know better | Junior dev anti-patterns |
| **Prudent/Deliberate** | Conscious tradeoff | "Ship now, refactor later" |
| **Prudent/Inadvertent** | Learned better since | "Now we know the right pattern" |

### Technical Categories

| Type | Impact | Typical Fix |
|------|--------|-------------|
| **Code debt** | Hard to read/modify | Refactoring |
| **Test debt** | Gaps in coverage | Add tests |
| **Doc debt** | Missing/wrong docs | Documentation |
| **Dependency debt** | Outdated/vuln packages | Upgrades |
| **Architecture debt** | Wrong structure | Redesign |
| **Infrastructure debt** | Manual/fragile ops | Automation |

---

## Phase 3: Assess

**Measure impact and cost.**

### Impact Assessment

| Factor | Questions | Score 1-5 |
|--------|-----------|-----------|
| **Frequency** | How often does this slow us down? | |
| **Severity** | How much does it slow us down? | |
| **Risk** | Could this cause outages/bugs? | |
| **Spread** | Does it affect many areas? | |
| **Growth** | Is it getting worse over time? | |

**Impact Score** = Sum of factors (5-25)

### Cost Assessment

| Factor | Questions | Score 1-5 |
|--------|-----------|-----------|
| **Size** | How much work to fix? | |
| **Complexity** | How hard to fix correctly? | |
| **Risk** | Could fixing break things? | |
| **Dependencies** | Does fixing require other changes? | |
| **Testing** | How hard to verify the fix? | |

**Cost Score** = Sum of factors (5-25)

### ROI Calculation

```
ROI = Impact Score / Cost Score
```

| ROI | Interpretation |
|-----|----------------|
| > 2.0 | High value - prioritize |
| 1.0-2.0 | Medium value - consider |
| < 1.0 | Low value - defer or accept |

---

## Phase 4: Prioritize

**Order by value, not by annoyance.**

### Prioritization Matrix

```
                    HIGH IMPACT
                         │
         ┌───────────────┼───────────────┐
         │   QUICK WINS  │   MAJOR       │
         │   Do first    │   PROJECTS    │
         │               │   Plan for    │
LOW ─────┼───────────────┼───────────────┼───── HIGH
COST     │               │               │      COST
         │   FILL-INS    │   MONEY PIT   │
         │   Do when     │   Avoid or    │
         │   convenient  │   rethink     │
         └───────────────┼───────────────┘
                         │
                    LOW IMPACT
```

### Priority Tiers

| Tier | Criteria | Action |
|------|----------|--------|
| **P0** | Blocks work NOW | Fix immediately |
| **P1** | High ROI, affects daily work | Plan this sprint |
| **P2** | Medium ROI, occasional friction | Plan this quarter |
| **P3** | Low ROI, minor annoyance | Backlog |
| **Accept** | Not worth fixing | Document and ignore |

### Quick Wins First

Always start with quick wins:
- High impact + low cost
- Builds momentum
- Shows stakeholders progress
- Low risk

---

## Phase 5: Decide

**Fix, defer, or accept.**

### Decision Framework

```
Is it causing active pain?
├── Yes → Is it blocking critical work?
│         ├── Yes → P0: Fix now
│         └── No  → Is ROI > 1.5?
│                   ├── Yes → P1: Plan soon
│                   └── No  → P2: Backlog
└── No  → Is it getting worse?
          ├── Yes → P2: Track and plan
          └── No  → Is it risky?
                    ├── Yes → P3: Monitor
                    └── No  → Accept: Document
```

### When to Accept Debt

Accept debt when:
- Code will be deleted soon
- Rarely touched (< 1x/quarter)
- Cost to fix > value over lifetime
- No risk of spreading
- Team understands and agrees

**Document accepted debt** so future developers understand:

```python
# ACCEPTED DEBT: This function is O(n²) but n is always < 100
# in practice. Optimizing would take 2 days for no user benefit.
# Revisit if data size grows. - Alex, 2024-03
def slow_but_fine(items):
    ...
```

### When NOT to Accept

Never accept debt that:
- Poses security risk
- Could cause data loss
- Affects many developers daily
- Spreads to new code automatically
- Violates compliance requirements

---

## Presenting to Stakeholders

**Speak business, not code.**

### Frame in Business Terms

| Technical | Business |
|-----------|----------|
| "Code is messy" | "Changes take 3x longer in this area" |
| "No tests" | "High risk of regressions, slow releases" |
| "Outdated deps" | "Security vulnerabilities, compliance risk" |
| "Architecture debt" | "Can't add features customers want" |

### The Pitch Structure

```markdown
## Problem
[Business impact in concrete terms]

## Cost of Inaction
[What happens if we don't fix it]

## Proposed Investment
[Time/resources needed]

## Expected Return
[Measurable improvement]

## Risk of Action
[What could go wrong]
```

### Example Pitch

```markdown
## Problem
Our checkout flow takes 3x longer to modify than other features.
Last 3 checkout bugs took 2 weeks each to fix.

## Cost of Inaction
- 6 developer-weeks lost this quarter
- Delayed feature launches
- Higher bug rate in critical revenue path

## Proposed Investment
2 sprints (4 weeks) to refactor checkout module

## Expected Return
- 60% reduction in checkout change time
- Reduced bug rate
- Faster feature delivery

## Risk of Action
- 1 sprint of reduced feature velocity
- Regression risk (mitigated by comprehensive tests)
```

---

## Debt Register

**Track debt systematically.**

### Register Template

| ID | Description | Category | Impact | Cost | ROI | Priority | Status |
|----|-------------|----------|--------|------|-----|----------|--------|
| D001 | Checkout module complexity | Code | 18 | 12 | 1.5 | P1 | Planned |
| D002 | Missing auth tests | Test | 15 | 8 | 1.9 | P1 | In Progress |
| D003 | Outdated React version | Dep | 10 | 15 | 0.7 | P3 | Backlog |
| D004 | Manual deploy process | Infra | 12 | 20 | 0.6 | Accept | Documented |

### Review Cadence

| Frequency | Activity |
|-----------|----------|
| Weekly | Update status of active items |
| Monthly | Review P2/P3 for promotion |
| Quarterly | Full register review, reprioritize |
| Annually | Archive completed/obsolete items |

---

## Integration with Other Skills

### Internal (This Plugin)

| Skill | Relationship |
|-------|--------------|
| [systematic-refactoring.md](systematic-refactoring.md) | HOW to fix debt |
| [codebase-confidence-building.md](codebase-confidence-building.md) | Understanding unfamiliar debt |
| [code-review-methodology.md](code-review-methodology.md) | Catching new debt in review |
| [complex-debugging.md](complex-debugging.md) | When debt causes bugs |

### External (Other Plugins)

| Need | Skill | When |
|------|-------|------|
| Automated debt detection | `ordis-quality-engineering:static-analysis-integration` | Finding code smells automatically |
| Measuring debt | `ordis-quality-engineering:quality-metrics-and-kpis` | Quantifying debt impact |
| Security debt | `ordis-quality-engineering:dependency-scanning` | Finding vulnerable dependencies |
| Architecture debt | `axiom-system-architect:identifying-technical-debt` | Large-scale architecture issues |
| Prioritization framework | `axiom-system-architect:prioritizing-improvements` | Risk-based improvement roadmap |

---

## Red Flags

| Thought | Reality | Action |
|---------|---------|--------|
| "Fix everything" | Impossible and wasteful | Triage ruthlessly |
| "It's not that bad" | Normalizing dysfunction | Measure the impact |
| "We'll fix it later" | Later never comes | Schedule now or accept |
| "Just this once" | Debt compounds | Track and limit |
| "No time for debt" | Debt steals time | Budget ongoing capacity |

---

## Quick Reference

### Triage Checklist

- [ ] **Identify**: Where does debt live?
- [ ] **Categorize**: What type of debt?
- [ ] **Assess**: Impact score? Cost score?
- [ ] **Prioritize**: ROI ranking?
- [ ] **Decide**: Fix, defer, or accept?
- [ ] **Track**: In debt register?

### ROI Quick Guide

| Impact | Cost | ROI | Action |
|--------|------|-----|--------|
| High | Low | >2 | Fix now |
| High | High | ~1 | Plan carefully |
| Low | Low | ~1 | Fill-in work |
| Low | High | <1 | Don't fix |

### Debt Budget

Reserve capacity for debt work:
- **20% rule**: 1 day/week for debt
- **Debt sprints**: Dedicated sprint quarterly
- **Boy Scout rule**: Leave code better than you found it
