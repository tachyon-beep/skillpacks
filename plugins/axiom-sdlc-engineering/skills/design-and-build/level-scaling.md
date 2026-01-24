# Reference Sheet: Level 2→3→4 Scaling

## Purpose & Context

Provides guidance on appropriate rigor for each CMMI maturity level - prevents under-engineering (Level 1 chaos) and over-engineering (Level 4 process for Level 2 project).

**When to apply**: Understanding what practices are required vs optional for your project tier

**Prerequisites**: Understanding of project context (team size, risk, compliance needs)

---

## Level Definitions

### Level 1: Initial (Chaos)

**Characteristics**:
- No documented processes
- Success depends on individual heroics
- Unpredictable results
- "Cowboy coding"

**NOT RECOMMENDED**: This skill assumes at least Level 2 baseline

### Level 2: Managed (Baseline)

**Characteristics**:
- **Projects** are planned and executed per policy
- Basic requirements management
- Configuration management (version control)
- Work products tracked
- **Honor system** (documented but not enforced)

**When appropriate**:
- Small teams (1-8 developers)
- Low-to-medium risk projects
- Internal tools
- No regulatory requirements
- Fast iteration more important than rigor

### Level 3: Defined (Organizational Standard)

**Characteristics**:
- **Organization** has standard processes
- Projects tailor standards to fit
- Processes **enforced by platform** (not honor system)
- Peer reviews required
- Architecture decisions documented (ADRs)
- Comprehensive quality gates

**When appropriate**:
- Medium-to-large teams (8-30 developers)
- Medium-to-high risk projects
- Customer-facing applications
- Audit requirements (SOC 2, ISO)
- Long-term maintenance expected

**DEFAULT for this skill**: If level unspecified, assume Level 3

### Level 4: Quantitatively Managed (Statistical Control)

**Characteristics**:
- Processes controlled using **statistical techniques**
- Performance baselines established
- Quality and process performance measured
- Predictive models used
- Data-driven decision making

**When appropriate**:
- Large teams (30+ developers)
- High-risk, high-stakes projects (financial, healthcare, aerospace)
- Regulatory requirements (FDA, FAA)
- Need predictable quality and performance

---

## Escalation Criteria: When to Move UP Levels

### Level 2 → Level 3

**Trigger ANY of these**:
- Team grows to 10+ developers
- Customer-facing application launched
- Audit/compliance requirements emerge (SOC 2, GDPR, ISO)
- Production incidents increasing (>2 per month)
- Technical debt consuming >40% of time
- Need reproducible processes across teams

**Action**: Implement Level 3 practices (ADRs, required reviews, platform enforcement)

### Level 3 → Level 4

**Trigger ANY of these**:
- Team grows to 30+ developers
- High-risk domain (financial transactions, healthcare data, safety-critical)
- Regulatory requirements demand quantitative evidence (FDA, FAA)
- Need predictable delivery (contractual commitments with penalties)
- Process optimization requires data (don't know what to improve without metrics)

**Action**: Establish measurement program, statistical baselines, predictive models

---

## De-Escalation Criteria: When Level is Overkill

### Level 4 → Level 3

**Trigger ANY of these**:
- Team shrinks below 20 developers (statistical analysis not justified)
- Risk decreases (moved from regulated to non-regulated domain)
- Cost of measurement exceeds value (spending more on metrics than on development)
- Team overwhelmed by process overhead (velocity suffering)

**Action**: Maintain Level 3 practices, drop statistical analysis

### Level 3 → Level 2

**ALLOWED ONLY IF ALL CONDITIONS MET**:

**Startup MVP Exception**:
- [ ] Team ≤5 developers (count active committers in past 30 days)
- [ ] No paying customers yet (beta/alpha/internal only)
- [ ] Time-limited: Max 6 months OR until first paying customer
- [ ] De-escalation documented in ADR with re-escalation triggers

**OR Internal Tool Exception**:
- [ ] Users are internal employees only (not external customers/partners)
- [ ] No compliance/audit requirements (SOC 2, ISO, PCI, HIPAA)
- [ ] Low business risk (failure doesn't cost revenue or reputation)
- [ ] De-escalation documented in ADR

**Automatic Re-Escalation Triggers** (move back to Level 3 immediately when ANY occur):
- First paying customer acquired
- External users >100 (even if free tier)
- 6 months elapsed since de-escalation
- Team grows ≥6 developers
- Compliance requirement emerges (audit, regulatory)

**De-Escalation ADR Required**:
Must document:
- Current project phase and user base
- Which exception applies (MVP or Internal Tool)
- Re-escalation criteria and dates
- Risk acceptance (what could go wrong at Level 2?)
- Who approves re-escalation decision?

**Action After De-Escalation**: Maintain basic CM and reviews, drop ADR requirement and platform enforcement

**CAUTION**: De-escalation is rare. Usually project moves UP levels over time. Don't use MVP exception as permanent escape hatch.

---

## Practice Comparison by Level

### Architecture Decision Records (ADRs)

| Level | Practice |
|-------|----------|
| **Level 2** | Optional. Major decisions documented in wiki or README. Informal discussion OK. |
| **Level 3** | **Required for ALL architectural decisions**. ADR must document alternatives. Peer review required. Exception: HOTFIX with retrospective ADR within 48 hours. |
| **Level 4** | All Level 3 requirements PLUS quantitative justification (performance models, cost analysis). No exceptions. |

### Code Review

| Level | Practice |
|-------|----------|
| **Level 2** | PR workflow encouraged. Review optional or single reviewer. Honor system. |
| **Level 3** | **2+ reviewers required, enforced by branch protection**. Review checklist used. CI must pass before merge. |
| **Level 4** | All Level 3 requirements PLUS review metrics tracked (defect finding rate, review time). Statistical quality gates. |

### Branching Strategy

| Level | Practice |
|-------|----------|
| **Level 2** | Documented strategy exists (GitHub Flow, Git Flow, etc.). Main branch protected. Basic workflow followed. |
| **Level 3** | Strategy documented in ADR with rationale. Platform enforces protection rules. Metrics baseline established (conflicts/week, PR cycle time). |
| **Level 4** | All Level 3 requirements PLUS statistical control (branch lifetime, merge frequency, conflict rate within control limits). |

### Build & Integration (CI/CD)

| Level | Practice |
|-------|----------|
| **Level 2** | Basic CI on PRs (build + unit tests). Manual deployment OK. |
| **Level 3** | Comprehensive CI/CD. Multi-stage pipeline (build/test/integration). Staging auto-deploy, prod manual approval. ADR for platform choice. |
| **Level 4** | All Level 3 requirements PLUS build metrics tracked (time trends, flakiness rates). DORA metrics measured. Statistical process control. |

### Technical Debt

| Level | Practice |
|-------|----------|
| **Level 2** | Debt tracked in issue tracker. 10-20% sprint allocation to debt. Informal prioritization. |
| **Level 3** | Debt classified (architectural/code quality/unpayable). Debt register with prioritization matrix. Retrospective ADRs for debt from past decisions. Metrics tracked (debt ratio, trend). |
| **Level 4** | All Level 3 requirements PLUS quantitative debt metrics (complexity trends, coupling metrics, change amplification, bug clustering). Predictive models for debt impact. |

---

## Example: Same Feature Across Levels

**Feature**: Add user authentication

### Level 2 Implementation

**Planning**:
- Discuss in team meeting: "Let's use JWT for auth"
- Note decision in wiki
- No formal ADR

**Development**:
- Create feature branch
- Implement JWT auth
- Write basic tests
- Open PR
- 1 person reviews (informal), approves
- Merge to main

**Testing**:
- Unit tests run in CI
- Manual testing in staging
- Deploy to prod

**Total time**: ~3 days (fast, minimal overhead)

### Level 3 Implementation

**Planning**:
- Identify as architectural decision (auth mechanism)
- Write ADR:
  - Alternatives: JWT, session-based, OAuth
  - Decision criteria: Stateless (JWT) vs server state (sessions)
  - Chosen: JWT with refresh tokens
  - Rationale: Stateless fits our microservices architecture
- ADR peer reviewed, approved

**Development**:
- Create feature branch
- Implement JWT auth following ADR design
- Write comprehensive tests (unit + integration)
- Security review (check for vulnerabilities)
- Open PR

**Review**:
- **2 reviewers required** (enforced by platform)
- Use code review checklist
- CI must pass (build, test, lint, security scan)
- Reviewers approve

**Testing**:
- Comprehensive CI (unit, integration, security scan)
- Auto-deploy to staging on merge
- Smoke tests in staging
- Manual approval gate for prod
- Deploy to prod

**Documentation**:
- API documentation updated (OpenAPI)
- README updated with auth flow

**Total time**: ~5 days (more rigorous, documented, reviewed)

### Level 4 Implementation

**Planning**:
- All Level 3 planning PLUS
- Performance model: JWT validation <50ms (95th percentile)
- Cost analysis: Refresh token storage vs full re-auth
- Statistical baseline: Current auth failure rate (measure improvement)

**Development** (same as Level 3)

**Review** (same as Level 3 PLUS):
- Review metrics tracked (time spent, defects found)
- Complexity metrics measured (must be within baselines)

**Testing** (same as Level 3 PLUS):
- Performance testing (validate <50ms validation)
- Load testing (ensure scales to baseline + 20%)

**Measurement**:
- Auth latency tracked (statistical control chart)
- Failure rate measured (compare to baseline)
- 30-day review: Did we hit performance targets?

**Total time**: ~7 days (rigorous, quantified, measured)

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Level 4 for Level 2 Project** | 2-person team doing statistical process control | Process overhead exceeds value, velocity suffers | Match level to risk and team size. Level 2 for small low-risk projects. |
| **Level 1 for Level 3 Project** | 15-person team, no reviews, customer-facing app | Inconsistent quality, production bugs, audit failures | Recognize when escalation needed (team size, risk, compliance). |
| **Ignoring De-Escalation** | Startup with 3 people doing Level 3 rigor | Speed-to-market suffers, can't compete | De-escalate temporarily during MVP phase, escalate before launch. |
| **Honor System at Scale** | 20-person team, "we should review code" | Honor system fails under pressure | Level 3: Platform enforcement (required reviews, CI gates). |
| **No Measurement to Justify Level** | "We're Level 3" but no evidence | Can't prove compliance, wasted effort if practices not followed | Track metrics to validate level (review rate, ADR count, CI coverage). |

---

## Checklist: What Level Should We Be?

### Team Size
- [ ] <5 developers → Level 2 sufficient
- [ ] 5-20 developers → Level 3 recommended
- [ ] 20-30 developers → Level 3 required
- [ ] >30 developers → Level 4 consider

### Risk Profile
- [ ] Internal tool, low impact → Level 2 sufficient
- [ ] Customer-facing, medium impact → Level 3 recommended
- [ ] Financial/healthcare/safety-critical → Level 4 consider

### Compliance Requirements
- [ ] None → Level 2 sufficient
- [ ] SOC 2, ISO, GDPR → Level 3 required
- [ ] FDA, FAA, PCI DSS → Level 4 consider

### Current Process Maturity
- [ ] No documented processes → Start at Level 2
- [ ] Documented but not enforced → Move to Level 3
- [ ] Enforced but no metrics → Stay at Level 3 or move to Level 4 if justified

### Technical Debt Level
- [ ] <30% time on bugs → Current level OK
- [ ] 30-50% time on bugs → Consider escalating for more rigor
- [ ] >50% time on bugs → Escalate to Level 3 minimum (enforce reviews, ADRs)

**Action**: Count "yes" answers. More "yes" in higher levels = escalate.

---

## Real-World Example: Level Progression

**Company**: SaaS startup

### Year 1: Level 2 (MVP Phase)
- Team: 3 developers
- Product: Internal MVP, no customers yet
- Process: GitHub Flow, optional reviews, no ADRs
- Result: Fast iteration, launched MVP in 4 months

### Year 2: Level 2→3 (Customer Launch)
- **Trigger**: Launched to customers, team grew to 12
- **Action**: Implemented Level 3
  - ADRs for architectural decisions
  - Required 2+ reviews (platform-enforced)
  - Comprehensive CI/CD
  - Security scanning
- **Result**: Slower iteration (5→7 day feature cycle) but higher quality (production bugs dropped 60%)

### Year 3: Level 3 (Scaling)
- Team: Grew to 25 developers
- Maintained Level 3 practices
- Added compliance (SOC 2 audit)
- Result: SOC 2 certification achieved, Level 3 practices provided required audit trail

### Year 4: Level 3→4 (Enterprise Customers)
- **Trigger**: Enterprise contracts with SLAs, penalties for downtime
- **Action**: Implemented Level 4 metrics
  - DORA metrics (deployment frequency, lead time, MTTR, change failure rate)
  - Statistical baselines for quality
  - Predictive models for capacity planning
- **Result**: Met SLA commitments (99.9% uptime), data-driven optimization (deployment frequency 2x, lead time -40%)

**Key learning**: Level progression follows company growth and risk profile. Don't skip levels, but don't over-engineer early.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Annually or when team size/risk changes
