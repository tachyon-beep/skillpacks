# Reference Sheet: Templates and Examples

## Purpose & Context

Provides concrete templates for ADRs and risk registers with real-world examples.

**When to apply**: Creating documentation for decisions or risks

**Prerequisites**: Decision made or risks identified

---

## ADR Template (Lightweight)

**Use for**: Most architectural decisions at Level 3

```markdown
# ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD  
**Status**: Proposed | Accepted | Deprecated | Superseded  
**Deciders**: [Names]

## Context

[What problem are we solving? What constraints exist? What prompted this decision?]

## Decision

[What did we decide? Be specific and concrete.]

## Alternatives Considered

1. **[Option 1]**: [Brief description, why not chosen]
2. **[Option 2]**: [Brief description, why not chosen]
3. **[Option 3]**: [Brief description, why not chosen]

## Consequences

**Positive**:
- [Benefit 1]
- [Benefit 2]

**Negative** (trade-offs):
- [Trade-off 1]
- [Trade-off 2]

**Neutral**:
- [Impact 1]

## Implementation Notes

[How will this be implemented? Timeline? Validation criteria?]
```

---

## ADR Example: Database Selection

```markdown
# ADR-003: PostgreSQL for User Preferences Service

**Date**: 2026-01-24  
**Status**: Accepted  
**Deciders**: Sarah (Tech Lead), Mike (Backend Team)

## Context

New microservice for user preferences (themes, notifications, settings). Need persistent storage with:
- Read-heavy workload (90% reads, 10% writes)
- <100ms query latency requirement
- ~50K active users, 500K total users
- Team familiar with SQL databases
- Already run PostgreSQL for main application database

Sprint deadline in 2 days. Team aligned on PostgreSQL but documenting decision per Level 3 requirement.

## Decision

Use **PostgreSQL 16** for user preferences service database.

## Alternatives Considered

1. **PostgreSQL** (chose this):
   - Pros: Team expertise, consistency with main app, relational model fits data, proven at our scale
   - Cons: Slight overkill for simple key-value storage
   
2. **Redis**:
   - Pros: Faster for pure key-value, simpler operations
   - Cons: Persistence model less robust, no complex queries if needed later, team less familiar
   
3. **MongoDB**:
   - Pros: Flexible schema, good for JSON documents
   - Cons: Team unfamiliar, adds operational overhead (new database type), overkill for our needs

4. **SQLite**:
   - Pros: Zero operations overhead, embedded
   - Cons: Doesn't scale to distributed system, single-writer limitation

## Consequences

**Positive**:
- Zero new operational overhead (reuse existing PostgreSQL infrastructure)
- Team can implement immediately (no learning curve)
- Future flexibility (can add complex queries, joins if requirements evolve)
- Battle-tested reliability and backup procedures

**Negative**:
- Slight over-engineering (relational DB for simple key-value storage)
- Must manage schema migrations (vs schemaless options)

**Neutral**:
- Consistent technology stack across services (good for ops, but creates PostgreSQL dependency)

## Implementation Notes

- Deploy PostgreSQL instance in same region as service
- Use connection pooling (PgBouncer) for read-heavy workload
- Schema migration: use Alembic (Python) or Flyway (JVM)
- Monitoring: Prometheus + Grafana (same as main DB)
- Validation: Load test at 10K QPS (10x expected peak)

**Timeline**: Week 1 - Setup and initial schema, Week 2 - Migration and testing
```

---

## Risk Register Template

**Use for**: All projects Level 2+

```markdown
# Risk Register - [Project Name]

**Project**: [Name]  
**Timeline**: [Start] to [End]  
**Last Updated**: YYYY-MM-DD

## Risk Summary

| ID | Risk | Prob | Impact | Score | Priority | Status |
|----|------|------|--------|-------|----------|--------|
| R-01 | [Risk 1] | [1-5] | [1-5] | [1-25] | [Critical/High/Med/Low] | [Active/Mitigated/Materialized] |
| R-02 | [Risk 2] | [1-5] | [1-5] | [1-25] | [Critical/High/Med/Low] | [Active/Mitigated/Materialized] |

---

## R-01: [Risk Title]

**Category**: Technical | Schedule | Resource | External  
**Identified Date**: YYYY-MM-DD  
**Status**: Active | Mitigated | Materialized

**Description**:
[What is the risk? What could go wrong?]

**Probability**: [1-5] ([Very Low/Low/Medium/High/Very High])  
**Impact**: [1-5] ([Negligible/Low/Medium/High/Critical])  
**Risk Score**: [Prob × Impact = Total]  
**Priority**: [Critical (20-25) | High (12-19) | Medium (6-11) | Low (3-5)]

**Mitigation Strategy**: Avoid | Transfer | Mitigate | Accept

**Mitigation Actions**:
1. [Action 1] - Owner: [Name], Due: [Date], Status: [Not Started/In Progress/Complete]
2. [Action 2] - Owner: [Name], Due: [Date], Status: [Not Started/In Progress/Complete]

**Residual Risk**: [Score after mitigation]  
**Contingency Plan**: [If risk materializes despite mitigation, Plan B is...]

**Monitoring**:
- Trigger: [What signals this risk is increasing?]
- Frequency: [Weekly/Bi-weekly/Monthly]
- Last Review: [Date]
```

---

## Risk Register Example

```markdown
# Risk Register - E-Commerce Platform v2.0

**Project**: E-commerce platform redesign  
**Timeline**: 2026-01-01 to 2026-06-30 (6 months)  
**Last Updated**: 2026-01-24

## Risk Summary

| ID | Risk | Prob | Impact | Score | Priority | Status |
|----|------|------|--------|-------|----------|--------|
| R-01 | Payment API vendor outage | 4 | 5 | 20 | Critical | Active |
| R-02 | Scope creep from stakeholders | 5 | 4 | 20 | Critical | Mitigated |
| R-03 | Database performance at scale | 3 | 4 | 12 | High | Active |
| R-04 | Third-party shipping API delays | 3 | 3 | 9 | Medium | Active |
| R-05 | Security vulnerability | 2 | 5 | 10 | Medium | Active |

---

## R-01: Payment API Vendor Outage

**Category**: External  
**Identified Date**: 2026-01-15  
**Status**: Active

**Description**:
Third-party payment processor (Stripe) could experience outage. 99.99% SLA = 4.3 min/month downtime. During outage, customers cannot complete purchases (revenue loss, cart abandonment).

**Probability**: 4 (High) - Vendor outages happen monthly based on status page history  
**Impact**: 5 (Critical) - Payment processing down = no revenue, customer frustration  
**Risk Score**: 4 × 5 = 20  
**Priority**: Critical

**Mitigation Strategy**: Mitigate

**Mitigation Actions**:
1. Implement circuit breaker pattern (fail fast if API down) - Owner: Sarah, Due: Week 2, Status: Complete
2. Add payment queueing (retry when API returns) - Owner: Mike, Due: Week 3, Status: In Progress
3. Vendor status monitoring + Slack alerts - Owner: DevOps, Due: Week 1, Status: Complete
4. Document manual payment processing for support team - Owner: Support Lead, Due: Week 2, Status: Complete

**Residual Risk**: 3 × 3 = 9 (Medium)  
After mitigation: Probability reduced (circuit breaker prevents cascading failures), Impact reduced (queueing recovers payments).

**Contingency Plan**:
If API down >1 hour:
1. Activate "maintenance mode" banner with ETA
2. Capture customer payment intents (email, order details)
3. Process queued payments when API returns
4. Support team manually processes high-value orders ($500+) via phone

**Monitoring**:
- Trigger: Stripe status page incident, circuit breaker trips, payment success rate <95%
- Frequency: Real-time monitoring, weekly review in standup
- Last Review: 2026-01-24

---

## R-02: Scope Creep from Stakeholders

**Category**: Schedule  
**Identified Date**: 2026-01-10  
**Status**: Mitigated

**Description**:
Stakeholders (marketing, sales, support) request "just one more feature" during development. Historical pattern: 30% scope growth without timeline extension.

**Probability**: 5 (Very High) - Happens on every project historically  
**Impact**: 4 (High) - Timeline slip, team morale, quality degradation from rushing  
**Risk Score**: 5 × 4 = 20  
**Priority**: Critical

**Mitigation Strategy**: Mitigate

**Mitigation Actions**:
1. MVP scope document with stakeholder sign-off - Owner: PM, Due: Week 1, Status: Complete
2. Change control process (request → impact assessment → approve/defer) - Owner: PM, Due: Week 1, Status: Complete
3. Weekly stakeholder demo with "parking lot" for future features - Owner: PM, Due: Ongoing, Status: In Progress
4. v2.1 roadmap document for deferred features - Owner: PM, Due: Week 2, Status: Complete

**Residual Risk**: 3 × 4 = 12 (High)  
After mitigation: Probability reduced (change control gates new requests), Impact remains (stakeholder pressure still exists).

**Contingency Plan**:
If scope grows >10%:
1. Present trade-off analysis to leadership (scope vs timeline vs quality)
2. Options: Extend timeline, reduce other scope, accept quality trade-off
3. Decision required from executive sponsor (not PM)

**Monitoring**:
- Trigger: New feature request, scope document change, timeline slipping
- Frequency: Weekly in stakeholder demo
- Last Review: 2026-01-24
```

---

**Last Updated**: 2026-01-24  
**Usage**: Copy templates for your decisions and risks
