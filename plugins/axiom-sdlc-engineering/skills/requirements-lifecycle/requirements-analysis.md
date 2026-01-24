---
parent-skill: requirements-lifecycle
reference-type: analysis-prioritization
load-when: Conflicting requirements, prioritization needed, scope management, gold plating concerns
---

# Requirements Analysis Reference

**Parent Skill:** requirements-lifecycle
**When to Use:** Prioritizing requirements, resolving conflicts, managing scope, preventing gold plating

This reference provides systematic techniques for analyzing and prioritizing requirements across CMMI Levels 2-4.

---

## Purpose & Context

**What this achieves**: Systematic requirement prioritization, conflict resolution, and scope management using appropriate rigor for maturity level.

**When to apply**:
- Conflicting requirements from multiple stakeholders
- Scope creep pressure ("just one more thing")
- Resource constraints (can't build everything)
- Need to prevent gold plating
- Requirements completeness checking

**Prerequisites**:
- Requirements elicited (see `requirements-elicitation.md`)
- Stakeholders identified
- Project constraints known (budget, timeline, capacity)

---

## CMMI Maturity Scaling

### Level 2: Managed (Analysis)

**Minimal requirements analysis**:

**Prioritization**:
- Simple Must/Should/Could/Won't (MoSCoW) or High/Medium/Low
- Product Owner makes final decisions
- Documented in issue tracker

**Conflict Resolution**:
- Escalate to Product Owner or sponsor
- Document decision rationale
- Informal process

**Effort**:
- 1-2 hours per sprint for prioritization
- Ad-hoc conflict resolution as needed

**Example**:
```
Sprint 15 Backlog Prioritization
Capacity: 30 story points

Requirements Reviewed:
- US-101: OAuth login (8 pts) - MUST HAVE (security requirement)
- US-102: Remember me checkbox (3 pts) - SHOULD HAVE
- US-103: Social login (Facebook) (5 pts) - COULD HAVE
- US-104: Biometric login (13 pts) - WON'T HAVE (defer to Q2)
- US-105: Password strength meter (2 pts) - SHOULD HAVE

Selected for Sprint 15 (30 pts):
✅ US-101 (8 pts) - Must
✅ US-102 (3 pts) - Should
✅ US-105 (2 pts) - Should
✅ US-103 (5 pts) - Could (fills remaining capacity)
❌ US-104 (13 pts) - Deferred to Q2 backlog

Decision: Product Owner (Sarah), 2026-01-20
```

### Level 3: Defined (Analysis)

**Organizational standards for prioritization** (all Level 2 plus):

**Prioritization**:
- Formal prioritization framework (MoSCoW + value/effort matrix)
- Multi-stakeholder prioritization sessions
- Decision criteria documented and reusable
- Peer review of prioritization decisions

**Conflict Resolution**:
- Facilitated negotiation sessions
- Decision Analysis and Resolution (DAR) process for major conflicts
- Documented trade-off analysis
- Cross-reference to ADR process (see `governance-and-risk` skill)

**Completeness Checking**:
- Requirements review checklist
- Traceability to business objectives verified
- Non-functional requirements explicitly identified

**Effort**:
- 4-8 hours per release for prioritization workshop
- 2-4 hours per major conflict resolution

**Example**:
```
Q1 Release Prioritization Workshop

Participants: Product Owner, Tech Lead, 3 stakeholder VPs
Duration: 4 hours
Framework: Value/Effort Matrix + MoSCoW

Step 1: Plot 45 requirements on value/effort matrix
Step 2: Classify into quadrants:
  - High Value, Low Effort (Quick Wins) → 12 requirements
  - High Value, High Effort (Major Projects) → 8 requirements
  - Low Value, Low Effort (Fill-ins) → 15 requirements
  - Low Value, High Effort (Time Sinks) → 10 requirements

Step 3: Apply MoSCoW within quadrants:
  Quick Wins: 12 → 10 Must, 2 Should
  Major Projects: 8 → 3 Must (critical), 5 Should
  Fill-ins: 15 → 0 Must, 5 Should, 10 Could
  Time Sinks: 10 → 0 Must, 0 Should, 10 Won't

Step 4: Capacity allocation (100 story points):
  Must Have: 13 requirements (72 pts) - Committed
  Should Have: 7 requirements (28 pts) - Target (likely 5 will make it)
  Could Have: 10 requirements - Backlog
  Won't Have: 10 requirements - Rejected/Deferred

Output: Prioritized release backlog with decision rationale
Decision Authority: Product Owner (final), Tech Lead (technical feasibility)
Review: Architecture Board (approved 2026-01-22)
```

### Level 4: Quantitatively Managed (Analysis)

**Quantitative prioritization** (all Level 3 plus):

**Prioritization**:
- Quantitative value models (WSJF, weighted scoring)
- Historical data for effort estimation
- ROI calculations with confidence intervals
- Statistical analysis of past prioritization accuracy

**Metrics**:
- Prioritization accuracy: % of "Must Have" requirements actually delivered
- Value realization: Actual business value vs. predicted
- Scope stability: % requirements reprioritized after initial classification

**Example**:
```
Prioritization Metrics Dashboard - Q4 2025

WSJF Scoring Model:
  Score = (Business Value + Time Criticality + Risk Reduction) / Effort

Top 10 Requirements by WSJF:
  REQ-201: API rate limiting (WSJF: 18.5)
    - Business Value: 9/10 (prevent outages)
    - Time Criticality: 8/10 (recent incidents)
    - Risk Reduction: 8/10 (security)
    - Effort: 1.35 sprints (historical avg: 1.2-1.5)
    - ROI: $45k benefit / $15k cost = 3.0x

Historical Accuracy:
  - Last 3 releases: 87% of "Must Have" delivered
  - Baseline: 85% ±5% (in control)
  - Current forecast: 13 Must requirements → 11-12 expected delivery (85-92%)

Scope Stability:
  - Q3: 23% requirements reprioritized mid-quarter
  - Q4 Forecast: 18% (improvement trend, within UCL of 25%)
```

---

## Implementation Guidance

### MoSCoW Prioritization with Enforcement

**MoSCoW Classification**:

**Must Have**:
- Definition: Project fails without it, non-negotiable
- Test: "Can we go to production without this?" If no → Must Have
- Typical: 40-60% of requirements

**Should Have**:
- Definition: Important but not critical, could defer 1 sprint/release
- Test: "Would users be significantly disappointed?" If yes → Should Have
- Typical: 20-30% of requirements

**Could Have**:
- Definition: Nice to have, include if capacity available
- Test: "Would this be a pleasant surprise?" If yes → Could Have
- Typical: 10-20% of requirements

**Won't Have**:
- Definition: Out of scope, explicitly deferred or rejected
- Test: "Does this align with current goals?" If no → Won't Have
- Typical: 20-30% of requirements (important for managing expectations)

**Enforcement Mechanism** (Addresses RED phase gap - baseline lacked enforcement):

**Rule**: Once sprint/release capacity reached, **no new Must Haves** without removing existing Must Haves.

**Example**:
```
Sprint 20 Planning
Team Velocity: 30 story points (historical average)
Current Commitment: 28 points (all Must Have)

Mid-Sprint Addition Requested:
Stakeholder: "We MUST add REQ-305 (critical bug fix, 5 points)"

Enforcement:
Product Owner: "Our capacity is 30 points, committed is 28. Adding REQ-305 (5 pts) puts us at 33 points (110% capacity).

Options:
1. Remove REQ-298 (5 pts) from sprint → Add REQ-305
2. Downgrade REQ-305 to Should Have → Add to Sprint 21
3. Extend sprint deadline by 1 day

Stakeholder must choose. Cannot add without trade-off."

Decision: Option 1 selected (REQ-298 deferred to Sprint 21)
Rationale: REQ-305 blocks production deployment, higher priority
```

**Preventing "Everything is Must Have"** (Common anti-pattern):

If >70% of requirements classified as Must Have:
1. **Re-baseline**: "If everything is critical, nothing is critical"
2. **Force ranking**: Rank Must Haves 1-N, draw line at capacity
3. **Business case**: Require ROI justification for each Must Have

### Value/Effort Matrix (2×2 Prioritization)

**Purpose**: Visual prioritization by business value vs. implementation effort

**Axes**:
- **Y-axis**: Business Value (High/Low)
- **X-axis**: Implementation Effort (Low/High)

**Quadrants**:
```
        High Value
            │
   Quick   │   Major
   Wins    │  Projects
───────────┼─────────── High Effort
   Fill-   │   Time
    ins    │   Sinks
            │
        Low Value
```

**Strategy**:
1. **Quick Wins** (High Value, Low Effort) - Do first
2. **Major Projects** (High Value, High Effort) - Plan carefully, do second
3. **Fill-ins** (Low Value, Low Effort) - Do if capacity available
4. **Time Sinks** (Low Value, High Effort) - Don't do (reject/defer)

**Scoring**:

**Business Value** (0-10 scale):
- Revenue impact (how much $ generated/saved)
- User satisfaction (NPS impact, support ticket reduction)
- Strategic alignment (how well it supports company goals)
- Risk mitigation (what problems does it prevent)

**Effort** (story points or days):
- Use historical data for similar features
- Include: design, implementation, testing, documentation, deployment
- Add 20% buffer for unknowns

**Example Matrix**:
```
Requirements Plotted:

Quick Wins (Do First):
  REQ-201: API rate limiting (Value: 9, Effort: 2) ← START HERE
  REQ-205: Password reset (Value: 8, Effort: 3)
  REQ-210: Email notifications (Value: 7, Effort: 2)

Major Projects (Plan Carefully):
  REQ-302: Payment integration (Value: 10, Effort: 13)
  REQ-304: Mobile app (Value: 9, Effort: 21)

Fill-ins (If Capacity):
  REQ-401: Dark mode (Value: 4, Effort: 3)
  REQ-405: Custom themes (Value: 3, Effort: 2)

Time Sinks (Don't Do):
  REQ-501: Blockchain integration (Value: 2, Effort: 21) ← REJECT
  REQ-505: AI chatbot (Value: 4, Effort: 13) ← DEFER
```

### Conflict Resolution Techniques

**Common Conflict Types**:
1. **Competing requirements** - Two stakeholders want mutually exclusive features
2. **Resource contention** - Multiple high-priority requirements, limited capacity
3. **Technical constraints** - Business wants X, technically infeasible with current architecture

#### Facilitated Negotiation (Level 3)

**Process**:

**Step 1: Objective Documentation**
- Document both requirements without advocacy
- Business objective for each
- Success metrics
- Implementation approach
- Effort/schedule/budget impact

**Step 2: Identify Decision Criteria**
- Business value (revenue, cost savings)
- Strategic alignment
- Risk mitigation
- Time-to-market
- Technical feasibility
- Customer impact

**Step 3: Facilitate Discussion**
- Present both options objectively
- Use decision criteria to score each option
- Explore compromise (can both be achieved differently?)
- If no compromise: escalate to decision authority

**Step 4: Document Decision**
- Which option selected and why
- Rationale for rejection of other option
- Impact on deferred option
- Follow-up actions

**Facilitator role**:
- ✅ Neutral party (no stake in outcome)
- ✅ Ensures both sides heard
- ✅ Keeps discussion on decision criteria (not politics)
- ❌ Does NOT make the business decision

**Example Conflict Resolution**:
```
Conflict: VP Sales vs. VP Operations

VP Sales Requirement:
  REQ-S01: Custom pricing per customer (support negotiated deals)
  Business Value: $500k/year in large contracts
  Effort: 8 weeks (complex pricing engine)

VP Operations Requirement:
  REQ-O01: Automated order fulfillment (reduce manual work)
  Business Value: $200k/year cost savings + faster fulfillment
  Effort: 8 weeks (integration with warehouse system)

Decision Criteria Scoring (1-10):
                        REQ-S01   REQ-O01
Revenue Impact            9         3
Cost Savings              2         8
Strategic Alignment       7         6
Risk Mitigation           4         7
Customer Satisfaction     8         7
Technical Complexity      6         7
──────────────────────────────────────
Total Score              36        38

Compromise Explored:
  - Phased delivery: Custom pricing (Phase 1), Automation (Phase 2)?
  - Result: Not feasible - both needed for Q1 targets

Escalation:
  - Decision Authority: CFO (both VPs report to CFO)
  - Recommendation: REQ-O01 (higher total score, lower risk)
  - CFO Decision: REQ-O01 approved for Q1, REQ-S01 deferred to Q2
  - Rationale: Cost savings more predictable than sales forecasts

Documentation: ADR-023 "Q1 Prioritization: Operations Automation"
Communicated: Both VPs notified, sales team informed of Q2 timeline
```

#### DAR Process for Major Conflicts (Level 3+)

**Decision Analysis and Resolution** (CMMI DAR process)

**When to use**:
- High-stakes decision (>$50k impact or >3 month effort)
- Multiple stakeholders with conflicting interests
- Technical risk requires formal evaluation

**Process**:
1. **Establish criteria** - Define decision factors and weights
2. **Identify alternatives** - Document all options (including "do nothing")
3. **Evaluate alternatives** - Score against criteria using structured method
4. **Select solution** - Choose based on objective evaluation
5. **Document decision** - Create ADR (Architecture Decision Record)

**Cross-reference**: See `governance-and-risk` skill for complete DAR process and ADR templates.

### Completeness Checking

**Purpose**: Verify no critical requirements missing before finalizing

**Completeness Checklist** (Level 3):

**Functional Requirements**:
- ✅ All user workflows covered (happy path + exceptions)
- ✅ All user roles accounted for (admin, user, guest, etc.)
- ✅ All CRUD operations specified (create, read, update, delete)
- ✅ All integrations identified (external systems, APIs)

**Non-Functional Requirements** (Often forgotten):
- ✅ Performance (response time, throughput, scalability)
- ✅ Security (authentication, authorization, data protection)
- ✅ Reliability (uptime, failover, backup/restore)
- ✅ Usability (accessibility, mobile support, i18n)
- ✅ Maintainability (logging, monitoring, troubleshooting)
- ✅ Compliance (GDPR, SOC 2, industry regulations)

**Example Completeness Review**:
```
Feature: User Registration

Functional Requirements Review:
  ✅ User can register with email/password
  ✅ Email verification required
  ✅ Password reset workflow
  ❌ MISSING: What happens if email already registered? (error handling)
  ❌ MISSING: Can users register with social login? (scope clarification needed)

Non-Functional Requirements Review:
  ✅ Performance: Registration completes in <2 seconds
  ✅ Security: Password hashing (bcrypt), HTTPS required
  ❌ MISSING: What's the account lockout policy after failed attempts?
  ❌ MISSING: GDPR compliance - data retention policy for inactive accounts?
  ❌ MISSING: Accessibility - is registration form screen-reader compatible?

Action Items:
  1. Clarify social login scope with Product Owner
  2. Define error handling for duplicate email (REQ-R05)
  3. Define lockout policy (REQ-R06)
  4. Add GDPR data retention requirement (REQ-R07)
  5. Add accessibility requirement (REQ-R08)
```

---

## Common Anti-Patterns

### Everything is Priority 1

**Symptom**: 90% of requirements marked "High Priority" or "Must Have"

**Impact**: No actual prioritization, team paralyzed deciding what to build first

**Example**:
```
Backlog Review:
  78 requirements total
  71 marked "High Priority" (91%)
  5 marked "Medium Priority" (6%)
  2 marked "Low Priority" (3%)

Team reaction: "If everything is high priority, we'll just work on whatever feels urgent"
```

**Solution**:
- **Force ranking**: Rank requirements 1-78, draw line at capacity
- **MoSCoW reset**: Re-classify using strict definitions (40-60% Must maximum)
- **Business case requirement**: Each "High Priority" requires ROI justification

### Analysis Paralysis

**Symptom**: Weeks spent analyzing requirements, no code written

**Impact**: Delayed value delivery, stakeholder frustration

**Example**: 6-week analysis phase produces 200-page requirements spec, project hasn't started implementation

**Solution**:
- **Timebox analysis**: 1-2 weeks maximum for initial requirements
- **MVP approach**: Define minimum viable feature set, defer everything else
- **80/20 rule**: 80% clarity is enough to start, refine during implementation

### Scope Creep Metrics Not Tracked

**Symptom**: Requirements keep growing, no visibility into change rate

**Impact**: Can't identify scope creep until project fails

**Solution - Scope Creep Metrics** (Addresses RED phase gap):

**Track**:
- **Change rate**: (Added + Modified + Deleted) / Total per sprint
- **Baseline growth**: Total requirements over time
- **Scope stability**: % requirements unchanged sprint-to-sprint

**Thresholds**:
- Level 2: <30% change rate per sprint (yellow flag at 30%, red at 40%)
- Level 3: <20% change rate per sprint
- Level 4: <10% change rate per sprint (statistical control)

**Example Tracking**:
```
Sprint | Total Reqs | Added | Modified | Deleted | Change % | Status
-------|-----------|-------|----------|---------|----------|--------
  10   |    45     |   45  |     0    |    0    |   100%   | ✅ (baseline)
  11   |    48     |    3  |     5    |    0    |    18%   | ✅ (normal)
  12   |    52     |    5  |     4    |    1    |    19%   | ✅ (normal)
  13   |    60     |   10  |     6    |    2    |    35%   | ⚠️ (high)
  14   |    65     |    8  |     7    |    3    |    35%   | ⚠️ (high)

Action: Sprint 13-14 show sustained 35% change rate (above 30% threshold)
→ Stakeholder stabilization meeting required
→ Freeze new requirements for Sprint 15, focus on delivery
```

---

## Tool Integration

### GitHub

**Prioritization**:
- Use labels: `priority:must`, `priority:should`, `priority:could`, `priority:wont`
- Use Projects with priority column
- Sort issues by priority + effort estimate

**Conflict tracking**:
- Label: `needs-decision`, `conflicting-requirements`
- Link conflicting issues together
- Document resolution in issue comments

### Azure DevOps

**Prioritization**:
- Use "Priority" field (1-4)
- Stack rank in backlog (drag-and-drop)
- Use queries to filter by priority

**Conflict tracking**:
- Create "Decision" work item type
- Link related requirements
- Document resolution in decision work item

---

## Verification & Validation

**Analysis complete when**:

**Level 2**:
- ✅ All requirements prioritized (MoSCoW or High/Med/Low)
- ✅ Conflicts escalated and resolved
- ✅ Product Owner sign-off on priorities

**Level 3**:
- ✅ Prioritization framework applied (value/effort matrix)
- ✅ Multi-stakeholder prioritization session held
- ✅ Completeness checklist reviewed
- ✅ Decisions documented (meeting minutes or ADRs)

**Level 4**:
- ✅ Quantitative prioritization model applied (WSJF, weighted scoring)
- ✅ Scope metrics within baselines (<20% change rate)
- ✅ Historical accuracy tracked and analyzed

---

## Related Practices

**Before analysis**:
- `requirements-elicitation.md` - Gather requirements before analyzing

**After analysis**:
- `requirements-specification.md` - Document prioritized requirements
- `requirements-change-management.md` - Manage changes to priorities

**Cross-references**:
- `governance-and-risk.md` - DAR process for major conflicts, ADR templates
- `level-scaling.md` - Determine appropriate analysis rigor for project

**Prescription reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.1.1 (Requirements Development - SP 2.1 Establish Requirements, SP 3.1 Analyze Requirements)
