# Reference Sheet: Technical Debt Management

## Purpose & Context

This reference sheet provides frameworks for detecting debt crises, classifying debt types, and recovering from debt spirals. It distinguishes normal debt management from CODE RED situations requiring immediate intervention.

**When to apply**: Team spending >40% time on bugs, velocity declining, "every feature breaks something"

**Prerequisites**: Understanding of codebase, ability to measure time spent on bugs vs features

---

## Crisis Detection: When Debt Becomes Emergency

### Thresholds

| Bug Time % | Classification | Action Required |
|------------|---------------|-----------------|
| <30% | Normal | Standard 20% debt allocation per sprint |
| 30-50% | Warning | Increase debt allocation to 40%, investigate trends |
| 50-60% | Serious | 50/50 debt/features, root cause analysis required |
| **>60%** | **CODE RED** | **Feature freeze, architectural audit, recovery plan** |

**At 70% bug time** (Scenario 3 from baseline): This is NOT "technical debt management" - this is a CRISIS requiring emergency intervention.

---

## Debt Classification System

### Architectural Debt

**Definition**: Violates module boundaries, introduces coupling, defers necessary refactoring

**Indicators**:
- Same bug appears in multiple places
- Changes ripple across modules unexpectedly
- Core abstractions wrong for current requirements
- Module dependencies circular or tangled

**Requires**: ADR documenting proper solution, design work

**Example**: Monolith with god objects, microservices with shared database

### Code Quality Debt

**Definition**: Duplication, complexity, no tests, but within correct architecture

**Indicators**:
- Long functions (>50 lines)
- Copy-paste code
- Low test coverage
- High cyclomatic complexity

**Requires**: Refactoring, test writing, cleanup

**Example**: Working code that's hard to maintain

### Unpayable Debt

**Definition**: Fixing costs more than rewriting

**Indicators**:
- >60% of module needs changes to fix properly
- Core assumptions fundamentally wrong
- Technology obsolete or unsupported
- Security issues pervasive

**Requires**: Rewrite decision (requires ADR)

**Example**: PHP 5.6 codebase when PHP 5.6 EOL'd, or entire module built on wrong paradigm

---

## CMMI Maturity Scaling

### Level 2: Managed

**Required Practices**:
- Track debt in issue tracker ("tech-debt" label)
- Allocate 10-20% of sprint capacity to debt
- Document known issues in release notes

**Work Products**:
- Debt items in tracker
- Sprint allocation decision

**Quality Criteria**:
- Team aware of debt
- Debt tracked somewhere

**Audit Trail**:
- Closed debt tickets showing work done

### Level 3: Defined

**Enhanced Practices**:
- Debt classification (architectural/code quality/unpayable)
- Debt register with prioritization matrix
- Retrospective ADRs for debt created by past decisions
- Formal paydown strategy (20% allocation or debt sprints)
- Debt metrics tracked (ratio, trend)

**Additional Work Products**:
- Debt register (spreadsheet or project board)
- Retrospective ADRs
- Debt metrics dashboard

**Quality Criteria**:
- All debt classified by type
- High-impact debt prioritized
- Paydown commitments tracked
- Metrics show debt trend (increasing/stable/decreasing)

**Audit Trail**:
- Debt register updated quarterly
- Retrospective ADRs explaining debt creation
- Metrics showing paydown progress

### Level 4: Quantitatively Managed

**Statistical Practices**:
- Debt metrics: complexity trends, coupling metrics, change amplification
- Statistical thresholds for "unacceptable debt"
- Predictive models (velocity impact, defect density)
- ROI analysis for debt paydown

**Quantitative Work Products**:
- Complexity trend charts (cyclomatic complexity over time)
- Coupling metrics (dependencies between modules)
- Change amplification (files changed per feature, trending down = good)
- Bug clustering (% bugs in top 10% of files)

**Quality Criteria**:
- Debt quantified (not just "high/medium/low")
- Statistical control limits established
- Paydown ROI measured

**Audit Trail**:
- Historical metrics proving debt impact
- ROI calculations for paydown investments

---

## CODE RED Recovery Plan (>60% Bug Time)

**This is NOT normal debt management. This is crisis intervention.**

### Week 1-2: FREEZE FEATURES

**Actions**:
1. **Stop all new feature work** (zero exceptions)
2. **Triage existing debt**:
   - Architectural: Requires design, ADR
   - Code quality: Refactor, test
   - Unpayable: Rewrite candidates
3. **Run architectural audit**:
   - Which modules/files cause most bugs? (hotspots)
   - Are core abstractions wrong? (paradigm mismatch)
   - Is coupling pervasive? (tangled dependencies)
4. **Create retrospective ADRs**:
   - What decisions created this debt?
   - Were ADRs followed or violated?
   - What patterns led to crisis?
5. **Stakeholder communication**:
   - Present data: "70% bugs, 30% features = death spiral"
   - Propose recovery: "2-week freeze → 40% bugs, 60% features long-term"
   - Get executive buy-in

**Deliverables**:
- Debt register (classified by type)
- Architectural audit findings
- Retrospective ADRs
- Stakeholder-approved recovery plan

### Week 3-4: Implement Architectural Fixes

**Actions**:
1. **Fix architectural debt first** (highest impact)
2. **Create ADRs for proper solutions** (not quick fixes)
3. **Implement with tests** (prevent regression)
4. **Monitor bug clustering** (should decrease)

**Goal**: Reduce bug hotspots, break coupling, fix core abstractions

**Success metric**: Bug clustering decreases (fewer bugs in top 10% of files)

### Week 5-8: 50/50 Debt/Features

**Actions**:
1. **Resume feature work at 50% capacity**
2. **Continue debt paydown at 50%**
3. **Monitor bug %**: Should drop from 70% to 40-50%
4. **Adjust if needed**: If bugs don't drop, escalate to rewrite discussion

**Rollback trigger**: If bug % doesn't drop to 50% by Week 4, feature freeze failed. Consider rewrite.

### Week 9+: Stabilize at 20/80 Debt/Features

**Actions**:
1. **Reduce debt allocation to 20% once bugs <30% of time**
2. **Maintain debt discipline** (don't accumulate again)
3. **Quarterly debt audits** (prevent future spirals)

**Long-term success**: Bugs <30% of time, velocity stable or increasing

---

## Normal Debt Management (When <50% Bug Time)

### 20% Rule

**Allocate 20% of sprint capacity to debt paydown:**
- 5-day sprint → 1 day on debt
- 2-week sprint → 2 days on debt

**Prioritization**: High impact + low effort first (quick wins)

### Debt Sprints

**Every 4th sprint is pure debt paydown:**
- 3 sprints features
- 1 sprint debt only

**Use when**: Debt accumulating faster than 20% can handle

### Parallel Tracks

**New features follow new standards, legacy exempt:**
- New code: ADRs, tests, reviews required
- Legacy code: Refactor when touched (boy scout rule)

**Gradually improves quality without stopping delivery**

---

## Debt Metrics

### Debt Ratio

**Formula**: Lines of debt code / Total lines

**How to measure**: Static analysis tools flag complexity, duplication, coverage gaps

**Target**: <10% for healthy codebase

### Cyclomatic Complexity Trend

**Measure**: Average complexity per function, tracked over time

**Target**: Trending downward (refactoring working)

### Change Amplification

**Measure**: How many files change per feature?

**Target**: Decreasing (better module boundaries)

### Bug Clustering

**Measure**: What % of bugs in top 10% of files?

**Target**: <50% (bugs distributed, not hotspots)

**>80% = architectural debt** (core modules wrong)

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Incremental Response to Crisis** | 70% bugs, suggests 20% debt allocation | Treats existential threat as normal problem | >60% bugs = CODE RED. Feature freeze required. |
| **Quick Fix Accumulation** | Every fix adds more debt | Debt compounds, eventual collapse | HOTFIX protocol: retrospective ADR, paydown commitment |
| **Ignoring Debt Metrics** | "We'll pay it down someday" | Someday never comes, spiral continues | Track metrics, enforce 20% allocation, quarterly audits |
| **Rewrite Too Early** | "Let's start over" at first sign of debt | Wastes working code, repeats mistakes | Classify first: Architectural? Code quality? Only rewrite if unpayable. |
| **No Retrospective ADRs** | Don't learn from past | Repeat same mistakes | Document what decisions created debt, learn patterns |

---

## Real-World Example: 70% Bugs → 25% Bugs in 8 Weeks

**Context** (from baseline Scenario 3):
- 2-year-old codebase
- Team spending 70% time on bugs
- Every feature breaks something
- Morale critical

**Actions**:
1. **Week 1-2: Feature freeze**
   - Classified debt: 60% architectural (god objects, tight coupling), 30% code quality (no tests), 10% unpayable (legacy auth module)
   - Architectural audit: 80% of bugs in 3 core modules (`api/models.py`, `frontend/state.ts`, `database/queries.py`)
   - Retrospective ADRs: Identified pattern - "shortcuts under deadline pressure"
2. **Week 3-4: Architectural fixes**
   - Refactored god objects into proper domain models (created ADRs)
   - Broke coupling in state management (extracted store modules)
   - Added tests for core modules (coverage 20% → 60%)
3. **Week 5-8: 50/50 debt/features**
   - Bugs dropped from 70% → 45% of time
   - Velocity increased (less firefighting)
   - Team morale improved
4. **Week 9+: 20/80 debt/features**
   - Bugs stabilized at 25% of time
   - Quarterly debt audits prevent future spirals
   - Velocity 2x higher than crisis period

**Key learning**: CODE RED requires feature freeze. Incremental approach (20% allocation) wouldn't have worked at 70% bug time.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when bug % >40%
