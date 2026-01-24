# Reference Sheet: Level Scaling

## Purpose & Context

Provides criteria for scaling QA rigor based on project maturity level (CMMI 2/3/4). Helps teams avoid over-engineering (Level 4 processes on Level 2 projects) and under-engineering (Level 2 practices on Level 3 projects).

**When to apply**: Starting new projects, auditing QA maturity, deciding when to increase rigor

**Prerequisites**: Understanding of CMMI levels, project risk assessment

---

## CMMI Level Overview

| Level | Name | Characteristic | QA Focus |
|-------|------|----------------|----------|
| **Level 1** | Initial | Ad-hoc, chaotic | Firefighting, no formal QA |
| **Level 2** | Managed | Reactive, project-level | Basic processes, testing exists |
| **Level 3** | Defined | Proactive, organization-level | Standardized processes, prevention |
| **Level 4** | Quantitatively Managed | Measured, statistical control | Data-driven optimization |
| **Level 5** | Optimizing | Continuous improvement | Process innovation |

**This guide covers Levels 2-4** (most teams operate in this range)

---

## Level 2: Managed (Baseline QA)

### Core Principle
"Projects have basic QA processes in place, but practices vary by project"

### Required Practices

**Testing**:
- Unit tests for critical business logic (>50% coverage of critical paths)
- Manual testing before release
- Bug tracking system in use
- Regression testing (manual or automated)

**Code Review**:
- Peer review recommended (not enforced)
- Informal feedback acceptable
- Single reviewer sufficient

**Defect Management**:
- Defects tracked in system
- Severity assigned (P0-P3)
- Basic RCA for P0 defects (optional for others)

**Validation**:
- Product owner approval sufficient
- Demo to stakeholders
- Informal acceptance

### What's NOT Required at Level 2

- Formal test plans
- Mandatory code review
- Statistical metrics
- Organization-wide QA standards
- UAT with end users
- Defect prevention programs

### When Level 2 is Appropriate

- **Team size**: 1-5 developers
- **Project risk**: Low (internal tools, prototypes)
- **Change frequency**: Infrequent releases
- **Customer impact**: Limited (internal users, tolerant of bugs)

**Example**: Internal admin dashboard used by 3 people

---

## Level 3: Defined (Organizational QA Standards)

### Core Principle
"Organization has defined QA processes that all projects follow"

### Required Practices

**Testing**:
- >70% coverage for critical paths, >50% overall
- Test pyramid enforced (many unit, some integration, few E2E)
- Automated regression suite (<1 hour)
- TDD recommended for business logic

**Code Review**:
- **2+ reviewers required** (platform-enforced)
- Review checklist used
- Substantive feedback required (not just "LGTM")
- Finding rate tracked (target: 20-40%)

**Defect Management**:
- RCA mandatory for P0/P1 defects
- RCA mandatory for recurring defects
- Defect prevention actions tracked
- Defect metrics reported (escape rate, density, MTTR)

**Validation**:
- UAT with representative users (not just product owner)
- Acceptance criteria defined (INVEST)
- Stakeholder sign-off required
- Beta rollout for high-risk changes

### Organizational Requirements

- QA standards documented
- Training on QA processes
- QA lead or engineering manager owns metrics
- Monthly/quarterly quality reviews

### What's NOT Required at Level 3

- Statistical process control
- Predictive defect models
- Automated process optimization
- Six Sigma/formal quality programs

### When Level 3 is Appropriate

- **Team size**: 5-50 developers
- **Project risk**: Medium-high (customer-facing, revenue-impacting)
- **Change frequency**: Weekly/bi-weekly releases
- **Customer impact**: Significant (external users, low tolerance for bugs)
- **Regulatory**: Some compliance requirements

**Example**: SaaS product with 10,000 paying customers

---

## Level 4: Quantitatively Managed (Statistical Quality Control)

### Core Principle
"QA processes are in statistical control and optimized using data"

### Required Practices

**Testing**:
- >80% coverage with <5% variation sprint-to-sprint
- Test execution time optimized via profiling
- Flaky test rate <1% (tracked and addressed immediately)
- Mutation testing for critical modules

**Code Review**:
- Review effectiveness in statistical control (finding rate 20-40% ±3σ)
- Review time optimized via historical data
- Review quality predicted by metrics (code complexity, churn)

**Defect Management**:
- Defect density in statistical control (<0.5/KLOC ±3σ)
- Predictive defect models (based on complexity, churn, experience)
- Defects predicted within ±20% accuracy
- Process changes evaluated via control charts

**Metrics & Measurement**:
- Control charts for all key metrics
- Cp/Cpk calculated for critical processes
- Defect prediction models updated monthly
- Process capability analysis

### Organizational Requirements

- Dedicated QA/process improvement team
- Statistical expertise (Six Sigma training)
- Automated data collection and analysis
- Quarterly process audits

### When Level 4 is Appropriate

- **Team size**: 50+ developers (or safety-critical with fewer)
- **Project risk**: High (safety-critical, financial systems)
- **Regulatory**: Strict compliance (FDA, SOC2, ISO)
- **Scale**: Large codebase (>100K LOC), high complexity
- **Customer impact**: Critical (downtime = revenue loss or safety risk)

**Example**: Medical device software, financial trading platform

---

## Escalation Criteria (When to Increase Rigor)

### Level 2 → Level 3 Triggers

**Data-driven indicators**:
- Defect escape rate >20% for 3 consecutive months
- Production incidents >2 per month
- Customer complaints about quality increasing
- Team size growing beyond 5 developers

**Qualitative indicators**:
- Multiple projects with inconsistent QA practices
- New team members confused about QA expectations
- Regulatory requirements emerging

**Action**: Implement Level 3 practices incrementally over 3-6 months

### Level 3 → Level 4 Triggers

**Data-driven indicators**:
- Defect density highly variable (>10% variation month-to-month)
- Unable to predict defects reliably
- Quality costs (rework, incidents) >15% of development effort
- Regulatory audit findings

**Qualitative indicators**:
- Safety-critical system classification
- Financial/compliance penalties for defects
- Organizational mandate for Six Sigma/process maturity

**Action**: Requires 12-18 months transition with dedicated resources

---

## De-escalation Criteria (When to Reduce Rigor)

### Level 4 → Level 3 Appropriate When

**Context changes**:
- Moved from safety-critical to standard product
- Regulatory requirements lifted
- Team size reduced significantly
- Project sunset approaching

**Process overhead indicators**:
- QA activities consuming >30% of development time
- Metrics not actionable (collected but unused)
- Statistical control achieved and stable for 12+ months

**Action**: Maintain core Level 3 practices, drop statistical overhead

### Level 3 → Level 2 Appropriate When

**Context changes**:
- Internal tool with <10 users
- Prototype or proof-of-concept phase
- Transitioning to maintenance-only mode

**Process overhead indicators**:
- QA rigor exceeding business risk
- Team consensus that processes slow delivery without benefit
- Defect escape rate <5% consistently

**Action**: Document decision, maintain automated tests, simplify review/validation

---

## Migration Strategies

### Incremental Level 2 → Level 3 (Recommended)

**Month 1-2**: Foundation
- Document current practices (baseline)
- Define organization-wide QA standards
- Train team on new requirements

**Month 3-4**: Testing & Reviews
- Implement test pyramid (TDD for new code)
- Enforce 2-reviewer rule
- Start tracking finding rate

**Month 5-6**: Validation & Metrics
- Implement UAT process
- Start tracking defect metrics
- Monthly quality dashboard

**Success criteria**: Defect escape rate <10%, finding rate 20-40%

### Incremental Level 3 → Level 4 (Requires Investment)

**Quarter 1**: Measurement Infrastructure
- Implement automated metric collection
- Establish baselines for all key metrics
- Train team on statistical methods

**Quarter 2**: Statistical Control
- Create control charts for key metrics
- Identify processes out of control
- Begin process stabilization

**Quarter 3-4**: Predictive Models
- Build defect prediction models
- Validate models (±20% accuracy target)
- Use models to allocate QA resources

**Quarter 5-6**: Optimization
- Process capability analysis (Cp/Cpk)
- Continuous improvement cycles
- Audit and refine models

**Success criteria**: All key metrics in statistical control, defect prediction within ±20%

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|--------------------|
| **Over-Engineering** | Level 4 rigor on 3-person team | Overhead exceeds benefit, slows delivery | Match rigor to risk and scale |
| **Under-Engineering** | Level 2 practices on safety-critical system | Defects escape, regulatory violations | Escalate to appropriate level |
| **Checklist Mentality** | "We do code reviews, we're Level 3" | Checkbox compliance, missing substance | Measure outcomes (finding rate, escape rate) |
| **Big Bang Migration** | Implement all Level 3 practices at once | Team overwhelmed, poor adoption | Incremental migration over 3-6 months |
| **Level Jumping** | Skip Level 3, go straight to Level 4 | No foundation, metrics meaningless | Build Level 3 base first (12+ months) |

---

## Risk-Based Scaling

**Use highest risk component to determine minimum level:**

| Risk Factor | Low | Medium | High |
|-------------|-----|--------|------|
| **User Impact** | Internal (<10 users) | Department (10-100) | External (>100) |
| **Revenue Impact** | None | <$10K/month | >$10K/month |
| **Data Sensitivity** | Public data | PII | PHI, financial |
| **Regulatory** | None | Industry standards | Legal compliance |
| **Safety** | No harm | Minor inconvenience | Injury/death |

**Minimum QA Level**:
- All Low → Level 2
- Any Medium → Level 3
- Any High → Level 3 (Level 4 if safety-critical or strict regulatory)

---

## Real-World Example: SaaS Startup Scaling

**Year 1 (3 developers, Level 2)**:
- Manual testing, informal reviews
- Product owner approval
- Defect escape rate: 25% (acceptable for early stage)

**Year 2 (12 developers, migrate to Level 3)**:
- Month 1-2: Document standards, train team
- Month 3-4: Enforce 2-reviewer rule, implement test pyramid
- Month 5-6: UAT with beta users, start metrics dashboard
- Result: Defect escape rate 25% → 12% (within Level 3 target)

**Year 3 (15 developers, Level 3 stable)**:
- Defect escape rate: 8-12% (stable)
- Review finding rate: 30% (healthy)
- No need for Level 4 (not safety-critical, metrics stable)

**Decision**: Stay at Level 3, focus on product features

---

**Last Updated**: 2026-01-24
**Review Schedule**: Annually or when project risk changes
