# Reference Sheet: Architecture & Design

## Purpose & Context

This reference sheet provides systematic frameworks for making architecture and design decisions, preventing resume-driven design, and creating defensible audit trails through Architecture Decision Records (ADRs).

**When to apply**: Technology selection, design pattern choices, system structure decisions

**Prerequisites**: Understanding of project requirements and constraints

---

## CMMI Maturity Scaling

### Level 2: Managed

**Required Practices**:
- Document major architecture decisions (platform choice, deployment strategy)
- Informal discussion and consensus sufficient
- Record decision in wiki, README, or team chat

**Work Products**:
- Architecture overview diagram
- Technology stack list
- Decision notes in accessible location

**Quality Criteria**:
- Key stakeholders participated in decision
- Rationale captured somewhere
- Future team can understand WHY choices were made

**Audit Trail**:
- Meeting notes or wiki page documenting decision
- Who decided, when, and basic rationale

### Level 3: Defined

**Enhanced Practices**:
- **ALL architectural decisions require ADR**
- ADR must document alternatives considered
- Peer review of ADRs before implementation
- ADR repository in version control
- Emergency HOTFIX exception with retrospective ADR within 48 hours

**Additional Work Products**:
- ADR repository (markdown files in version control)
- C4 diagrams (Context, Container, Component)
- Design review checklists
- Architecture standards document

**Quality Criteria**:
- ADR includes alternatives analysis
- At least 2 peer reviewers approve
- Decision criteria explicitly stated
- Consequences (positive and negative) documented

**Audit Trail**:
- All ADRs in git with full history
- Review comments and approvals tracked
- Links from code to relevant ADRs

### Level 4: Quantitatively Managed

**Statistical Practices**:
- ADRs include quantitative justification (performance targets, cost analysis)
- Architecture decisions tracked as metrics
- Statistical analysis of decision outcomes
- Prediction models for architectural impact

**Quantitative Work Products**:
- ADRs with performance models and cost projections
- Architecture decision metrics dashboard
- Post-implementation reviews with actual vs predicted outcomes
- Statistical baselines for architectural complexity

**Quality Criteria**:
- Decisions supported by data, not opinion
- Performance impact quantified
- Cost-benefit analysis included
- Success metrics defined upfront

**Audit Trail**:
- All L3 requirements plus quantitative data
- Post-implementation measurement proving decision quality
- Statistical process control for architecture drift

---

## Implementation Guidance

### Quick Start Checklist

When facing architecture decision:

- [ ] **Clarify requirements**: What problem are you solving? What are success criteria?
- [ ] **Identify constraints**: Team size, timeline, budget, risk tolerance, operational maturity
- [ ] **Generate alternatives**: List 3+ options (don't skip to first idea)
- [ ] **Define decision criteria**: What matters most? (Performance? Cost? Time to market? Maintainability?)
- [ ] **Evaluate systematically**: Score alternatives against criteria
- [ ] **Pick simplest option** that meets requirements
- [ ] **Document in ADR** (Level 3+) or decision notes (Level 2)
- [ ] **Review with peers** before implementing (Level 3+)

---

### ADR Template (Level 3)

```markdown
# ADR-YYYY-MM-DD: [Decision Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
**Date**: YYYY-MM-DD
**Authors**: [Names]
**Reviewers**: [Names]
**CMMI Level**: [2 | 3 | 4]

## Context

What is the issue we're facing? What requirements drive this decision?

**Problem statement**: [Describe the architectural problem or choice]

**Constraints**:
- Team: [size, skills, location]
- Timeline: [deadlines, milestones]
- Budget: [infrastructure costs, operational costs]
- Risk: [acceptable risk level, compliance requirements]

## Decision Drivers

What factors influence this decision?

- [ ] Performance requirements (latency, throughput, scale)
- [ ] Cost constraints (infrastructure, operational, development time)
- [ ] Team expertise (learning curve acceptable?)
- [ ] Maintainability (long-term support burden)
- [ ] Compliance (regulatory, security, audit requirements)
- [ ] Time to market (delivery pressure)

## Considered Options

List all alternatives, including status quo.

1. **Option A**: [Name]
2. **Option B**: [Name]
3. **Option C**: [Name]
4. **Do nothing** (continue current approach)

## Decision Outcome

**Chosen option**: "[Option Name]"

**Rationale**: Why this option over alternatives?

[2-3 paragraphs explaining WHY this is the best choice given constraints and criteria]

### Positive Consequences

- Benefit 1
- Benefit 2
- Benefit 3

### Negative Consequences (Accepted Tradeoffs)

- Tradeoff 1: [What we're giving up]
- Tradeoff 2: [What could go wrong]
- Mitigation: [How we'll address downsides]

## Alternatives Analysis

Detailed comparison of options:

| Criteria | Weight | Option A | Option B | Option C |
|----------|--------|----------|----------|----------|
| Performance | High | 9/10 | 6/10 | 8/10 |
| Cost | High | 5/10 | 9/10 | 7/10 |
| Team Expertise | Medium | 4/10 | 8/10 | 6/10 |
| Maintainability | Medium | 7/10 | 6/10 | 9/10 |
| Time to Market | Low | 6/10 | 8/10 | 5/10 |
| **Total** | | **31/50** | **37/50** | **35/50** |

**Scoring rationale**: [Explain how you scored each option]

## Implementation Notes

How to implement this decision:

1. Step 1
2. Step 2
3. Step 3

**Migration path** (if changing from existing approach):
- Phase 1: [Preparation]
- Phase 2: [Migration]
- Phase 3: [Validation]

**Rollback plan**: If this fails, how do we revert?

## Validation

How we'll verify this was the right choice:

**Success metrics**:
- Metric 1: [Target value, measurement method]
- Metric 2: [Target value, measurement method]

**Review schedule**:
- 30-day retrospective: Did we hit targets?
- 90-day assessment: Any unexpected issues?

**Failure criteria**: If we see [X], we revisit this decision.

## References

- Link 1: [Research, documentation]
- Link 2: [Similar decisions elsewhere]
- Link 3: [Supporting data]

---

## Metadata

**Supersedes**: ADR-XXXX (if applicable)
**Related**: ADR-YYYY, ADR-ZZZZ
**Tags**: #architecture #platform #infrastructure
```

---

### HOTFIX ADR Template (Retrospective)

When you used HOTFIX exception protocol:

```markdown
# ADR-YYYY-MM-DD-HOTFIX: [Emergency Fix]

**Status**: Accepted (Retrospective)
**Incident Date**: YYYY-MM-DD HH:MM UTC
**ADR Created**: YYYY-MM-DD (within 48 hours of incident)
**Authors**: [Names]

## Incident Context

**What broke**: [Description of production issue]
**Impact**: [Customers affected, revenue impact, SLA violation]
**Time pressure**: [Why couldn't we follow normal ADR process]

## Emergency Decision

**Quick fix applied**: [What we did to restore service]

**Why this instead of proper solution**:
- Proper solution would take: [X hours/days]
- Service restoration required: [< 1 hour]
- Chose speed over quality to minimize customer impact

## Technical Debt Introduced

**Debt classification**: [Architectural | Code Quality | Tactical]

**Specific debt items**:
1. [What's now wrong with the code]
2. [What corners were cut]
3. [What assumptions were violated]

**Risk if left unfixed**: [What could break, performance degradation, security issues]

## Paydown Commitment

**Proper solution**: [What we SHOULD have done]

**Implementation plan**:
- Week 1: [Design proper fix]
- Week 2: [Implement and test]
- Maximum 2 weeks from incident

**Assigned to**: [Owner]
**Tracked in**: [Ticket #123]

## Lessons Learned

**Why weren't we prepared**: [Root cause of emergency]
**Prevention**: [How to avoid this in future]
**Process improvement**: [What we'll change]

---

## References

**Incident report**: [Link]
**Follow-up ticket**: [Link]
```

---

### Decision Framework: Technology Selection

**Use this framework to prevent resume-driven design.**

#### Step 1: Requirements First

**Before discussing technology, answer**:
1. What problem are you solving? (Be specific)
2. What are success criteria? (Measurable)
3. What constraints exist? (Team, time, budget, risk)

**If you can't answer these, STOP. You don't have enough information to choose technology.**

#### Step 2: Generate Alternatives

List 3+ options, including:
- Status quo (keep current approach)
- Simplest possible solution
- Industry standard solution
- "Innovative" solution (if justified)

**Don't skip to first idea. Bias toward simplicity.**

#### Step 3: Define Decision Criteria

What matters MOST for this decision?

**Common criteria**:
- Performance (latency, throughput, scale)
- Cost (infrastructure, operational, development)
- Team expertise (learning curve)
- Maintainability (long-term support)
- Time to market (delivery speed)
- Risk (what can go wrong?)

**Assign weights**: High, Medium, Low

#### Step 4: Systematic Evaluation

Score each alternative against criteria (1-10 scale).

**Forces honesty**: Can't hide behind buzzwords when you have to score "team expertise" for new technology you don't know.

#### Step 5: Forcing Function

**Microservices decision framework (objective criteria)**:

| Factor | Objective Threshold | How to Measure | Your Project |
|--------|-------------------|----------------|--------------|
| **Team size** | ≥30 developers | Count active committers in past 3 months: `git shortlog -s -n --since="3 months ago" \| wc -l` | ___ developers |
| **Bounded contexts** | 3+ distinct domains with <10% shared code | Run dependency analysis: `cloc --by-file` + calculate coupling | ___% coupling |
| **Scaling variance** | ≥10x difference in resource needs | Profile system: requests/sec, memory, CPU by module | ___x variance |
| **Operational maturity** | ALL 4: (1) CI/CD <10min build, (2) Distributed tracing, (3) Automated rollback tested quarterly, (4) On-call rotation | Audit checklist: have ALL four? | Yes / No (___/4) |

**Scoring**:
- 0-1 criteria met: Monolith required (microservices will fail)
- 2 criteria met: Modular monolith recommended
- 3-4 criteria met: Microservices justified, proceed with ADR

**Rule**: Microservices allowed ONLY if ≥3 criteria met with objective measurements. Self-assessment rejected - require data.

**For simple CRUD app**: Answer is ALWAYS monolith or serverless. Microservices is ALWAYS wrong.

---

### Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Resume-Driven Design** | "I've heard X is best practice", technology before requirements | Over-engineered, mismatched to needs, long timelines | Requirements first. Justify technology with measurable needs. |
| **Architecture Astronaut** | Microservices for CRUD, event sourcing for simple data | Complexity explosion, team overwhelmed, project stalls | YAGNI. Simplest solution first. Add complexity when demonstrated need exists. |
| **Big Design Up Front** | 6-month design phase before code | Requirements change, design goes stale, analysis paralysis | Evolutionary architecture. Design critical decisions, defer others. Iterate. |
| **Copy-Paste Architecture** | "Company X uses Y, so we should" | Different contexts, different needs, cargo-cult failures | Understand WHY company X chose Y. Does your context match? Likely not. |
| **No Architecture** | "We'll figure it out as we go" | Inconsistent patterns, architectural rot, expensive rewrites | Up-front decisions for critical paths. Defer non-critical. Document in ADRs. |
| **Gold Plating** | "Future-proof" without concrete future | Wasted effort, premature optimization, unused features | Build for NOW. Refactor when requirements actually change. |

---

### Design Patterns Selection Guide

**When to use each pattern**:

| Pattern | Use When | Avoid When | Example |
|---------|----------|-----------|---------|
| **Monolith** | Small team (<10), simple domain, fast iteration | Team >30, clear bounded contexts | Most startups, internal tools |
| **Modular Monolith** | Medium team (10-30), moderate complexity, want refactoring option | Need independent deployment, team >30 | E-commerce site with clear modules |
| **Microservices** | Large team (>30), clear bounded contexts, different scaling needs, mature ops | Small team, simple domain, limited ops maturity | Large platform with independent teams |
| **Serverless** | Variable load, minimal ops, cloud-native, event-driven | Consistent high load, long-running processes, vendor lock-in concerns | Webhooks, batch processing, APIs with spiky traffic |
| **Event-Driven** | Async workflows, decoupled systems, audit trail needed | Simple CRUD, synchronous flows sufficient | Order processing, notifications, audit logs |
| **Layered** | Clear separation of concerns, testability important | Performance critical (extra layers add overhead) | Web apps, enterprise systems |

---

### C4 Model for Architecture Documentation

**Use C4 to document architecture at different abstraction levels.**

#### Level 1: Context Diagram

**Purpose**: Show system in its environment

**Shows**:
- Your system (box)
- Users (people)
- External systems (boxes)
- Relationships

**Audience**: Everyone (non-technical stakeholders)

#### Level 2: Container Diagram

**Purpose**: High-level technology view

**Shows**:
- Web app
- Mobile app
- API
- Database
- Message queue

**Audience**: Technical stakeholders, architects

#### Level 3: Component Diagram

**Purpose**: Components within container

**Shows**:
- Controllers
- Services
- Repositories
- Internal structure

**Audience**: Developers working on system

#### Level 4: Code Diagram

**Purpose**: Class-level detail (rarely needed)

**Shows**:
- Classes
- Interfaces
- Relationships

**Audience**: Developers implementing specific component

**Recommendation**: C4 Level 2 (Container) is sweet spot for ADRs. Enough detail to understand architecture, not so much detail it goes stale.

---

## Verification & Validation

### How to Verify This Practice is Working

**Observable indicators**:
- [ ] All major architecture decisions have ADRs in git
- [ ] ADRs include alternatives analysis (not just "here's what we picked")
- [ ] Peer reviews happen before implementation (comments in ADR PRs)
- [ ] Team can articulate WHY choices were made (ADRs provide rationale)
- [ ] New team members read ADRs to understand system (onboarding)
- [ ] HOTFIX exceptions documented within 48 hours (no lost audit trail)

**Metrics to track**:
- Number of ADRs created per quarter (should correlate with architectural changes)
- % of architectural changes with ADRs (target: 100% for Level 3)
- Average time to create ADR (target: <1 hour for simple, <4 hours for complex)
- HOTFIX retrospective ADR compliance (target: 100% within 48 hours)

### Common Failure Modes

| Failure Mode | Symptoms | Remediation |
|--------------|----------|-------------|
| **ADR Theater** | Many ADRs, but not actually influencing decisions | Review process: Are ADRs created BEFORE implementation? Do they include real alternatives? |
| **Analysis Paralysis** | Weeks spent writing ADR, no code | Timebox ADR creation (4 hours max). Make decision, document, move forward. |
| **Rubber Stamp** | ADRs approved without real review | Require at least 2 reviewers with specific feedback. Track review quality. |
| **Skipping Retrospective** | HOTFIX exceptions without follow-up ADRs | Automated reminder 24 hours after HOTFIX. Escalate if 48 hours missed. |
| **Resume-Driven ADRs** | ADR justifies trendy tech with buzzwords | Enforce "Requirements First" section. Reject ADRs without measurable justification. |

---

## Related Practices

- **Configuration Management**: ADRs for branching strategy, release process
- **Build & Integration**: ADRs for CI/CD platform, deployment strategy
- **Governance & Risk**: DAR process for critical decisions
- **Requirements Lifecycle**: Trace architecture decisions to requirements
- **Platform Integration**: GitHub/Azure DevOps for ADR storage and review

---

## Real-World Examples

### Example 1: Technology Selection (Python Web Framework)

**Context**: Building REST API, team of 5, 3-month timeline

**Alternatives**:
1. FastAPI (async, modern, OpenAPI generation)
2. Django REST Framework (mature, batteries-included, admin UI)
3. Flask + extensions (lightweight, flexible)

**Decision criteria**:
- Performance: Medium (thousands of req/s, not millions)
- Time to market: High (3 months tight)
- Team expertise: 2 know Django, 1 knows FastAPI, 2 new to Python web
- Maintainability: High (5-year horizon)

**Evaluation**:
- FastAPI: Great performance, but learning curve for team
- Django: Overkill (don't need admin), but team knows it
- Flask: Most flexible, but more integration work

**Decision**: Django REST Framework

**Rationale**: Time to market and team expertise outweigh performance. Team can deliver faster with known framework. Performance adequate for requirements (thousands, not millions).

**Tradeoffs accepted**: Heavier framework than needed. Not async-native (but not required).

### Example 2: Branching Strategy Selection

See Configuration Management reference sheet for detailed example.

### Example 3: HOTFIX Decision

**Incident**: Database connection pool exhausted, API returning 500s

**Emergency fix**: Increased pool size from 10 to 100 (env variable change)

**Why not proper solution**: Proper fix requires code change to implement connection retry logic and better error handling. Would take 4 hours to develop and test.

**Retrospective ADR**:
- Documented pool increase as tactical fix
- Identified root cause: No retry logic, connection leaks
- Committed to proper fix within 2 weeks
- Implemented monitoring to detect pool exhaustion earlier

**Outcome**: Service restored in 15 minutes. Proper fix deployed in 10 days. Team learned to monitor connection pools. HOTFIX process validated.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when process failures detected
