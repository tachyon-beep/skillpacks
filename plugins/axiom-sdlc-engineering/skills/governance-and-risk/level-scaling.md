# Reference Sheet: Governance Level Scaling

## Purpose & Context

Provides criteria for scaling governance rigor based on project risk level (CMMI 2/3/4). Prevents over-engineering (heavyweight process on low-risk projects) and under-engineering (lightweight process on high-risk projects).

**When to apply**: Project start, governance planning, audit preparation

**Prerequisites**: Understanding of project scope, timeline, and risk profile

---

## Risk-Based Governance Principle

**Governance rigor should match project risk, not project size.**

Small high-risk projects need more governance than large low-risk projects.

**Risk factors**:
- User impact (internal vs external, 100 vs 100K users)
- Financial impact (revenue, cost, liability)
- Security/compliance (PII, payments, regulations)
- Longevity (3-month prototype vs 3-year platform)
- Team maturity (experienced vs new)

---

## Level 2: Managed (Baseline Governance)

### Core Principle
"Projects have basic governance - decisions and risks documented, not lost."

### When Level 2 is Appropriate

- Internal tools (<50 users)
- Short-lived projects (<3 months)
- Experienced team with informal communication
- Low financial risk (<$50K impact)
- No regulatory requirements

**Example**: Admin dashboard for support team, 3 developers, 6 weeks

### Level 2 Requirements

**DAR (Decision Analysis)**:
- ADRs for high-risk decisions only (vendor lock-in, major framework choice)
- Lightweight format (context, decision, why not alternatives - 1 page)
- No formal alternatives evaluation required
- Decision log maintained (list of decisions with dates)

**RSKM (Risk Management)**:
- Risk identification required (brainstorming or checklist)
- Basic risk register (risk, probability, impact, owner)
- Mitigation plans for high-probability or high-impact risks
- No formal monitoring cadence (ad-hoc reviews)

**Work Products**:
- Decision log (spreadsheet or wiki page)
- Risk register (simple table)
- Mitigation plans (action items in project tracker)

### Level 2 Example

**Project**: Internal admin dashboard, 6 weeks, 3 developers

**DAR**:
- ADR for database choice (PostgreSQL) - 1-page document
- No ADR for UI framework (React, team standard)
- Decision log: 3 architectural decisions documented

**RSKM**:
- Risk register: 5 risks identified (scope creep, data access, timeline slip)
- Mitigation plans for top 2 risks
- Weekly async check-in on risks (15 min)

**Effort**: ~2% of project time on governance overhead

---

## Level 3: Defined (Organizational Standard)

### Core Principle
"Organization has defined governance processes, all projects follow standard."

### When Level 3 is Appropriate

- Customer-facing applications (external users)
- Medium-long projects (>3 months)
- Revenue-generating or business-critical systems
- Moderate financial risk ($50K-$1M impact)
- Some regulatory requirements (SOC 2, GDPR)
- Team size >5 developers

**Example**: SaaS product, 10 developers, 6 months, 10K paying customers

### Level 3 Requirements

**DAR (Decision Analysis)**:
- **ADRs mandatory for all architectural decisions** (not just high-risk)
- Alternatives analysis with decision criteria
- Decision matrix for major decisions (vendor, framework, architecture pattern)
- Independent analysis before authority/consensus input
- ADR review process (architecture review board or tech leads)

**RSKM (Risk Management)**:
- Risk identification with probability × impact classification
- Risk register with mitigation plans for all medium+ risks
- **Scheduled risk reviews** (bi-weekly or monthly based on timeline)
- Risk triggers defined for ad-hoc reviews
- Risk monitoring with residual risk tracking

**Work Products**:
- ADR repository (numbered, searchable)
- Risk register with complete fields (description, assessment, mitigation, monitoring)
- Decision criteria templates
- Risk review meeting notes

**Governance Processes**:
- Architecture review board (quarterly or per major decision)
- Risk review cadence (monthly or bi-weekly)
- Decision authority matrix (who approves what level of decision)

### Level 3 Example

**Project**: SaaS e-commerce platform, 10 developers, 6 months

**DAR**:
- ADR repository: 15 ADRs over 6 months
- All architectural decisions documented (database, auth, payment, deployment, caching)
- Decision matrices for high-risk choices (auth provider, payment gateway)
- Architecture review board: Monthly meetings

**RSKM**:
- Risk register: 12 risks identified and tracked
- Monthly risk reviews (1 hour)
- Mitigation plans for all high+ risks (7 risks)
- Risk dashboard visible to stakeholders

**Effort**: ~5-8% of project time on governance overhead

---

## Level 4: Quantitatively Managed (Statistical Governance)

### Core Principle
"Governance processes measured and optimized using statistical techniques."

### When Level 4 is Appropriate

- Safety-critical systems (medical, aerospace, automotive)
- High financial risk (>$1M impact)
- Strict regulatory requirements (FDA, FAA, financial)
- Long-lived platforms (>3 years)
- Large teams (>20 developers)

**Example**: Medical device software, 30 developers, 3-year platform, FDA regulated

### Level 4 Requirements

**DAR (Decision Analysis)**:
- All Level 3 requirements PLUS:
- Quantitative decision criteria (performance benchmarks, cost models)
- Statistical analysis of alternatives (Monte Carlo for cost, performance models)
- Historical decision quality tracking (decisions reviewed after 6/12 months)
- Decision effectiveness metrics (% of decisions reversed, decision lead time)

**RSKM (Risk Management)**:
- All Level 3 requirements PLUS:
- Statistical risk models (defect prediction, schedule estimation)
- Quantitative impact analysis (cost models, schedule variance)
- Risk process performance baselines (risk identification rate, mitigation effectiveness)
- Control charts for risk metrics

**Work Products**:
- Statistical analysis reports (decision criteria, risk models)
- Process performance baselines (historical data on governance effectiveness)
- Quantitative risk assessments (Monte Carlo, sensitivity analysis)

**Governance Processes**:
- Quantitative process objectives (decision reversal rate <10%, risk mitigation effectiveness >80%)
- Statistical process control for governance (control charts)
- Root cause analysis for governance failures

### Level 4 Example

**Project**: FDA Class III medical device software, 30 developers, 3 years

**DAR**:
- Quantitative decision criteria (security score >90/100, performance <50ms p99)
- Statistical analysis for vendor decisions (cost models, uptime distributions)
- Decision quality tracking (12-month retrospective on major decisions)
- Decision effectiveness: 5% reversal rate (target <10%)

**RSKM**:
- Statistical defect prediction models (based on complexity, churn, team experience)
- Risk models validated against historical data (±15% accuracy)
- Monthly risk metrics: Risk identification rate (risks/KLOC), mitigation effectiveness (% risks reduced to acceptable levels)

**Effort**: ~10-15% of project time on governance overhead (justified by regulatory and safety requirements)

---

## Escalation Criteria (When to Increase Rigor)

### Level 2 → Level 3 Triggers

**Mandatory escalation when**:
- External users (customer-facing)
- Revenue-generating or business-critical
- Regulatory requirements emerge (SOC 2, GDPR)
- Team size >5 developers
- Timeline >3 months
- Financial impact >$50K

**Data-driven indicators**:
- Decision reversals >20% (major rework due to bad decisions)
- Risks materializing without warning (no proactive mitigation)
- Governance ad-hoc and inconsistent across projects

**Action**: Implement Level 3 governance incrementally over 1-2 months

### Level 3 → Level 4 Triggers

**Mandatory escalation when**:
- Safety-critical classification (medical, aerospace, automotive)
- Strict regulatory requirements (FDA, FAA, financial)
- Financial impact >$1M
- Audit findings requiring statistical rigor

**Data-driven indicators**:
- Governance effectiveness highly variable (decision quality inconsistent)
- Regulatory audit requires quantitative evidence
- Multiple projects with similar governance challenges (organizational maturity needed)

**Action**: Requires 6-12 month transition with dedicated process improvement team

---

## De-Escalation Criteria (When Rigor is Overkill)

### Level 4 → Level 3 Appropriate When

**Context changes**:
- Moved from safety-critical to standard product
- Regulatory requirements lifted
- Project scope reduced significantly

**Process overhead indicators**:
- Statistical governance consuming >15% of project time without ROI
- Metrics collected but not actionable
- Team consensus that rigor exceeds risk

**Action**: Maintain core Level 3 practices, drop statistical analysis

### Level 3 → Level 2 Appropriate When

**Context changes**:
- External product moved to internal tool
- Customer base reduced to <50 users
- Project entering maintenance mode (minimal changes)

**Process overhead indicators**:
- Governance rigor significantly exceeds business risk
- Team consensus that standard process slows delivery without benefit
- No audit or regulatory requirements

**Action**: Document decision to de-escalate, maintain ADRs and risk identification, simplify review processes

---

## Common Mistakes

| Mistake | Why It Fails | Better Approach |
|---------|--------------|-----------------|
| One-size-fits-all | Small projects over-engineered, large projects under-governed | Risk-based scaling (match rigor to risk) |
| Project size = governance level | Large low-risk projects waste effort | Risk factors (impact, users, regulations) determine level |
| Permanent level assignment | Project risk changes over time | Re-assess level at milestones |
| Skipping Level 2 baseline | No baseline governance creates chaos | Level 2 is minimum for professional projects |
| Level jumping (2 → 4) | No foundation for statistical rigor | Build Level 3 first (12+ months before Level 4) |

---

## Quick Decision Tree

```
Is project safety-critical OR highly regulated (FDA, FAA)?
  YES → Level 4
  NO → Continue

Is project customer-facing OR revenue-generating OR >3 months?
  YES → Level 3
  NO → Continue

Is project internal-only AND <3 months AND <50 users?
  YES → Level 2
  NO → Default to Level 3 (when uncertain, higher rigor is safer)
```

---

**Last Updated**: 2026-01-24  
**Review Schedule**: Project start, at major milestones, when context changes
