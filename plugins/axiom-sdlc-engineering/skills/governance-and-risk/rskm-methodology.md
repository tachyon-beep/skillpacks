# Reference Sheet: Risk Management (RSKM)

## Purpose & Context

Provides systematic methodology for identifying, assessing, mitigating, and monitoring risks using the CMMI RSKM process area.

**When to apply**: Project start, architecture changes, external dependencies, ongoing monitoring

**Prerequisites**: Understanding of project scope and timeline

---

## RSKM Process Flow

```
1. Identify Risks → 2. Assess Risks (Probability × Impact) → 
3. Develop Mitigation → 4. Monitor & Review
```

**Critical**: Proactive (identify early) beats reactive (firefight later) by 3-10x cost ratio.

---

## Step 1: Risk Identification

### Risk Categories

**Technical Risks**:
- Technology maturity (new framework, beta software)
- Integration complexity (multiple systems, APIs)
- Performance requirements (latency, scale)
- Technical debt (legacy code, poor architecture)

**Schedule Risks**:
- Optimistic estimates
- Dep

endencies (blocked by other teams)
- Resource availability (PTO, hiring delays)
- Scope creep

**Resource Risks**:
- Team expertise gaps
- Key person dependencies
- Budget constraints
- Infrastructure/tooling

**External Risks**:
- Vendor dependencies (API, SaaS)
- Regulatory changes
- Market conditions
- Third-party delays

### Identification Techniques

**Brainstorming** (30 minutes):
- Gather team: developers, PM, stakeholders
- Ask: "What could prevent us from succeeding?"
- Capture all risks (no filtering yet)
- Group by category

**Checklists** (15 minutes):
- Technical: new tech, integrations, performance, debt
- Schedule: estimates, dependencies, resources, scope
- External: vendors, regulations, market
- Review each category, identify applicable risks

**Historical Data** (if available):
- Review past projects: What went wrong?
- Check post-mortems: What risks materialized?
- Ask experienced team members: "What have you seen before?"

**"What Could Go Wrong?"** (Red Team):
- Assume pessimistic stance
- Identify worst-case scenarios
- Challenge optimistic assumptions

---

## Step 2: Assess Risks (Probability × Impact)

### Probability Scale

| Level | Probability | Description |
|-------|-------------|-------------|
| **5 - Very High** | >70% | Will almost certainly occur |
| **4 - High** | 40-70% | Likely to occur |
| **3 - Medium** | 15-40% | May occur |
| **2 - Low** | 5-15% | Unlikely but possible |
| **1 - Very Low** | <5% | Rare |

### Impact Scale

| Level | Impact | Description |
|-------|--------|-------------|
| **5 - Critical** | Project failure | Cannot deliver without major changes |
| **4 - High** | Major disruption | Significant delay (>1 month), cost overrun (>20%), quality degradation |
| **3 - Medium** | Moderate disruption | Delay (1-4 weeks), cost increase (10-20%), workaround needed |
| **2 - Low** | Minor disruption | Delay (<1 week), cost increase (<10%), easy fix |
| **1 - Negligible** | No impact | Doesn't affect delivery |

### Risk Score Matrix

**Formula**: Risk Score = Probability × Impact (scale: 1-25)

| Risk Score | Priority | Action Required |
|------------|----------|-----------------|
| **20-25** | Critical | Mitigation plan mandatory, weekly monitoring |
| **12-19** | High | Mitigation plan required, bi-weekly monitoring |
| **6-11** | Medium | Mitigation strategy identified, monthly monitoring |
| **3-5** | Low | Accept with awareness, quarterly review |
| **1-2** | Very Low | Accept, document only |

**Example - Third-Party API Risk**:
- Probability: 4 (High, 40-70% - vendor outages happen)
- Impact: 5 (Critical - payment processing down = no revenue)
- **Risk Score**: 4 × 5 = 20 (Critical priority)
- **Action**: Mitigation plan mandatory

---

## Step 3: Develop Mitigation Strategies

### Mitigation Approaches

**Avoid** - Eliminate the risk:
- Example: "Vendor API risk" → Build in-house instead of using vendor
- When: High-risk, low-benefit scenarios
- Cost: Often highest (alternative approach needed)

**Transfer** - Move risk to third party:
- Example: "Data breach risk" → Cyber insurance policy
- When: Financial risks, specialized risks
- Cost: Insurance premiums, contracts

**Mitigate** - Reduce probability or impact:
- Example: "API downtime risk" → Circuit breaker, fallback, queueing
- When: Most common approach for technical risks
- Cost: Development time for safeguards

**Accept** - Acknowledge and plan for consequences:
- Example: "Minor UI bug risk" → Accept, fix in next sprint if reported
- When: Low probability or low impact
- Cost: Minimal (monitoring only)

### Mitigation Plan Template

```markdown
**Risk**: [Description]
**Score**: [Probability × Impact = Total]
**Strategy**: Avoid | Transfer | Mitigate | Accept

**Mitigation Actions**:
1. [Specific action 1] - Owner: [Name], Due: [Date]
2. [Specific action 2] - Owner: [Name], Due: [Date]

**Residual Risk**: [Score after mitigation]
**Contingency**: [If risk materializes despite mitigation, what's Plan B?]
```

**Example - Third-Party API Risk**:
```markdown
**Risk**: Payment API vendor outage
**Score**: 4 (High probability) × 5 (Critical impact) = 20 (Critical)
**Strategy**: Mitigate

**Mitigation Actions**:
1. Implement circuit breaker pattern - Owner: Sarah, Due: Week 2
2. Add fallback to queueing (process payments when API returns) - Owner: Mike, Due: Week 3
3. Set up vendor status monitoring + alerts - Owner: DevOps, Due: Week 1
4. Document manual payment processing procedure - Owner: Support, Due: Week 2

**Residual Risk**: 3 × 3 = 9 (Medium - reduced probability and impact)
**Contingency**: If API down >1 hour, activate manual payment processing procedure
```

---

## Step 4: Monitor & Review

### Review Cadence by Project Length

| Project Length | Review Frequency | Format |
|----------------|------------------|--------|
| **< 1 month** | Weekly | 15-min standup discussion |
| **1-3 months** | Bi-weekly | 30-min dedicated review |
| **3-6 months** | Monthly | 1-hour risk assessment |
| **> 6 months** | Monthly + milestones | 1-hour review, plus at major milestones |

### Risk Triggers (Ad-Hoc Reviews)

**When to conduct unscheduled risk review**:
- Scope change (new features, requirements shift)
- Team change (key person leaves, new hires)
- Technology change (new framework, infrastructure)
- Timeline change (deadline moved, delays)
- External change (vendor announcement, regulation)

### Monitoring Questions

**Three-question async check-in** (15 minutes):
1. **New risks emerged?** (scope, tech, resources, external)
2. **Risk probabilities changed?** (higher or lower than assessed?)
3. **Any risks now higher priority?** (impact increased, deadline closer)

### Risk Register Updates

**Monthly** (or per cadence):
- Mark materialized risks (move to issues log)
- Update probabilities (based on current project state)
- Add new risks (from triggers or monitoring questions)
- Remove obsolete risks (no longer applicable)
- Re-score and re-prioritize

**Example Evolution - Month 1 vs Month 4**:

**Month 1 Risks**:
- Team ramp-up (High probability, Medium impact) → Score: 12
- Requirements churn (Medium probability, High impact) → Score: 12

**Month 4 Risks** (evolved):
- ~~Team ramp-up~~ (Materialized - handled, no longer a risk)
- ~~Requirements churn~~ (Probability reduced, now Low × Medium = 6)
- **Integration testing crunch** (New risk - High probability, High impact = 16)
- **Technical debt** (New risk - Medium probability, Medium impact = 9)

---

## Common Mistakes

| Mistake | Why It Fails | Better Approach |
|---------|--------------|-----------------|
| "Low-risk project" assumption | Optimism bias, scope creep hits all projects | Risk identification mandatory regardless of perceived risk |
| Set-and-forget risk planning | Risks evolve, complacency before crunch | Scheduled reviews based on project length |
| Accept high risks without mitigation | Reactive firefighting costs 3-10x | Mitigation required for scores >12 |
| No contingency plans | When risk materializes, no Plan B | Contingency = "If this happens, we do X" |
| Vague mitigation actions | "Monitor closely" is not a mitigation | Specific actions with owners and deadlines |
| Skip risk monitoring | New risks emerge, probabilities shift | Monthly reviews + ad-hoc triggers |

---

## Real-World Example: SaaS Application Risk Management

**Context**: E-commerce platform, 6-month timeline, 10 developers, 50K users

**Risk Identification** (brainstorming session):
1. Payment API vendor outage
2. Scope creep from stakeholders
3. Database performance at scale
4. Key developer leaving team
5. Third-party shipping API delays
6. Security vulnerability discovery
7. Regulatory compliance (GDPR, PCI-DSS)

**Risk Assessment** (probability × impact):

| Risk | Prob | Impact | Score | Priority |
|------|------|--------|-------|----------|
| Payment API outage | 4 | 5 | 20 | Critical |
| Scope creep | 5 | 4 | 20 | Critical |
| Database performance | 3 | 4 | 12 | High |
| Key developer leaving | 2 | 4 | 8 | Medium |
| Shipping API delays | 3 | 3 | 9 | Medium |
| Security vulnerability | 2 | 5 | 10 | Medium |
| Regulatory compliance | 1 | 5 | 5 | Low |

**Top 3 Mitigation Plans**:

1. **Payment API Outage** (Score: 20):
   - Mitigation: Circuit breaker + queueing
   - Owner: Backend team
   - Due: Week 2-3
   - Residual: 9 (Medium)

2. **Scope Creep** (Score: 20):
   - Mitigation: MVP scope document + change control process
   - Owner: PM
   - Due: Week 1
   - Residual: 12 (High - probability reduced, impact remains)

3. **Database Performance** (Score: 12):
   - Mitigation: Load testing at Week 8, query optimization sprint
   - Owner: Backend + DevOps
   - Due: Week 8
   - Residual: 6 (Medium)

**Monitoring**: Monthly reviews, plus ad-hoc if scope change or team change.

---

**Last Updated**: 2026-01-24  
**Review Schedule**: Project start, monthly reviews, at major milestones
