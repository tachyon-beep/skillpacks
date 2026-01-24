# Reference Sheet: Decision Analysis & Resolution (DAR)

## Purpose & Context

Provides systematic methodology for making and documenting architectural and technical decisions using the CMMI DAR process area.

**When to apply**: Architectural decisions, vendor selection, framework choice, major technical trade-offs

**Prerequisites**: Understanding of project context and technical landscape

---

## DAR Process Flow

```
1. Identify Decision → 2. Establish Criteria → 3. Identify Alternatives → 
4. Evaluate Alternatives → 5. Select Solution → 6. Document Rationale
```

**Critical**: Steps 2-4 happen BEFORE authority input or team consensus. Independent analysis prevents bias.

---

## Step 1: Identify Decisions Requiring Formal DAR

### When ADR is MANDATORY

**Level 3 Projects - Always Required**:
- Architectural patterns (microservices vs monolith, layered vs hexagonal)
- Data storage (database choice, caching strategy, data model)
- Framework/library selection (React vs Vue, FastAPI vs Django)
- Authentication/authorization approach
- Deployment infrastructure (cloud provider, container orchestration)
- Third-party integrations (payment, analytics, auth providers)

**Level 2 Projects - Required for High-Risk**:
- Vendor lock-in risk (3+ year commitment)
- Cost >$10K/year
- Security-sensitive (auth, payments, PII)
- Affects >1K users

### When ADR is OPTIONAL

- Implementation details (algorithm choice for known problem)
- Local optimizations (data structure in single module)
- Temporary solutions (< 3 month lifespan)
- Prototypes and spikes

**If uncertain**: Document it. Cost is low (20 min), benefit is high (context retention).

---

## Step 2: Establish Decision Criteria

**Decision criteria are the factors that matter for YOUR context**, not generic best practices.

### Common Criteria Categories

**Technical**:
- Performance (latency, throughput, scalability)
- Reliability (uptime, fault tolerance, data consistency)
- Security (auth, encryption, compliance)
- Maintainability (code complexity, testability, documentation quality)
- Integration (APIs, existing systems, data migration)

**Non-Technical**:
- Cost (licensing, infrastructure, maintenance)
- Team expertise (learning curve, hiring availability)
- Community/vendor support (documentation, tutorials, bug fixes)
- Longevity (project activity, vendor stability, market adoption)
- Compliance (regulatory requirements, audit needs)

### Weighting Criteria

**Option 1: Simple (High/Medium/Low)**
- High priority: Deal-breakers, must-haves
- Medium priority: Important but negotiable
- Low priority: Nice-to-have

**Option 2: Numeric (1-10 scale)**
- 9-10: Critical
- 6-8: Important
- 3-5: Moderate
- 1-2: Minor

**Example - Database Selection**:
| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Performance (latency <100ms) | 10 | Real-time user experience requirement |
| Cost | 7 | Budget-constrained startup |
| Team expertise | 8 | Small team, can't afford long ramp-up |
| Scalability | 6 | 10K users now, could grow to 100K |
| Vendor support | 5 | Nice-to-have, not critical |

---

## Step 3: Identify Alternatives

**Minimum: 3 alternatives** (including current state if applicable)

### Alternatives Discovery

**Sources**:
- Industry standards (what do most teams use for this problem?)
- Vendor research (Gartner Magic Quadrant, G2 reviews, Stack Overflow surveys)
- Team brainstorming (what have we used before? what do we know?)
- Peer recommendations (ask in Slack communities, Reddit, Discord)

**Range**: Include build vs buy, open-source vs commercial, simple vs sophisticated

**Example - Authentication Decision**:
1. **Auth0** (SaaS, turnkey)
2. **AWS Cognito** (Cloud-native, integrated)
3. **Keycloak** (Open-source, self-hosted)
4. **Build custom** (Full control, high effort)

**Invalid "alternatives"**: Don't just list vendors in same category. Include different approaches (SaaS vs self-hosted, buy vs build).

---

## Step 4: Evaluate Alternatives

### Decision Matrix

**Format**: Alternatives as rows, criteria as columns, scores in cells

**Example - Authentication Decision Matrix**:

| Alternative | Performance (10) | Cost (7) | Team Expertise (8) | Scalability (6) | Vendor Support (5) | **Total** |
|-------------|------------------|----------|--------------------|-----------------|--------------------|-----------|
| **Auth0** | 9 × 10 = 90 | 4 × 7 = 28 | 7 × 8 = 56 | 9 × 6 = 54 | 10 × 5 = 50 | **278** |
| **AWS Cognito** | 8 × 10 = 80 | 8 × 7 = 56 | 5 × 8 = 40 | 9 × 6 = 54 | 7 × 5 = 35 | **265** |
| **Keycloak** | 7 × 10 = 70 | 10 × 7 = 70 | 4 × 8 = 32 | 7 × 6 = 42 | 4 × 5 = 20 | **234** |
| **Build Custom** | 6 × 10 = 60 | 3 × 7 = 21 | 3 × 8 = 24 | 5 × 6 = 30 | 1 × 5 = 5 | **140** |

**Interpretation**: Auth0 scores highest (278), but cost is weak (28). AWS Cognito is close (265) with better cost (56).

**Decision**: Auth0 if budget allows, AWS Cognito if cost-constrained.

### Qualitative Trade-Off Analysis

When quantitative scoring feels forced, use qualitative analysis:

**Option A: Auth0**
- ✅ Fastest time-to-market (turnkey)
- ✅ Best documentation and support
- ❌ Most expensive ($300/month)
- ❌ Vendor lock-in

**Option B: AWS Cognito**
- ✅ Cost-effective ($0-50/month)
- ✅ Integrated with existing AWS infrastructure
- ❌ Learning curve (team unfamiliar)
- ❌ Limited customization

**Option C: Keycloak**
- ✅ Free (open-source)
- ✅ Full control and customization
- ❌ Self-hosting overhead (ops burden)
- ❌ Smaller community than commercial options

**Option D: Build Custom**
- ✅ Perfect fit for requirements
- ✅ No vendor lock-in
- ❌ 3-6 month development time
- ❌ Security risks (homegrown auth is dangerous)

---

## Step 5: Select Solution

**Decision Authority**:
- **Level 2**: Tech lead or senior engineer
- **Level 3**: Documented decision with rationale, reviewed by architecture team
- **Level 4**: Quantitative justification required

**Decision Override**:
If authority (CTO, product) overrides analysis-based recommendation:
- Document override with rationale
- Note: "Analysis recommends X based on criteria. Decision is Y based on [business factor, relationship, strategic alignment]."
- This is valid! Just needs transparency.

**Tie-Breaking**:
When scores are close (within 10%):
- Re-examine highest-weighted criteria
- Consider qualitative factors (team morale, future flexibility)
- Pilot/spike if possible (2-week trial)
- Consult subject matter expert

---

## Step 6: Document Rationale (ADR)

**ADR Template** (Lightweight):
```markdown
# ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD  
**Status**: Proposed | Accepted | Deprecated | Superseded

## Context

[What problem are we solving? What constraints exist?]

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

**Negative**:
- [Trade-off 1]
- [Trade-off 2]

**Neutral**:
- [Impact 1]

## Implementation Notes

[How will this be implemented? Timeline? Who owns it?]
```

**Full ADR template**: See templates.md for comprehensive version with decision criteria, evaluation matrix, and validation plan.

---

## Authority Bias Resistance

**The Problem**: CTO met with vendor, tech lead suggested option, PM wants feature. How do you do independent analysis?

### Process for Independent Analysis

**Step 1: Gather Authority Input LAST**
- Establish criteria first (without authority input)
- Identify alternatives (without authority preference)
- Evaluate options (without knowing authority's choice)
- THEN get authority input

**Step 2: Document Authority Preference Separately**
```markdown
## Analysis Recommendation
Based on criteria [X, Y, Z], analysis recommends **Option A**.

## Stakeholder Preference
CTO prefers **Option B** based on vendor relationship and strategic alignment.

## Final Decision
We chose **Option B** (stakeholder preference).

**Rationale for Override**:
- Vendor relationship provides preferential support
- Strategic alignment with company direction
- CTO willing to accept trade-off on [criterion X]
```

**This is OK!** Authority can override analysis, but it must be transparent.

**Red Flag**: If authority preference is secret until after analysis, analysis is biased.

---

## Common Mistakes

| Mistake | Why It Fails | Better Approach |
|---------|--------------|-----------------|
| "Obvious" choice not documented | Context lost in 6 months | Document even obvious decisions (20 min investment) |
| Analysis after commitment | Becomes validation theater | Evaluate alternatives BEFORE commitment |
| Only 1-2 alternatives | Not enough options to validate choice | Minimum 3 alternatives (including status quo or build vs buy) |
| Generic criteria | Doesn't reflect YOUR context | Criteria specific to project needs and constraints |
| Authority input first | Biases analysis | Establish criteria and evaluate BEFORE authority input |
| No consequences documented | Miss trade-offs and future issues | Document positive, negative, and neutral consequences |

---

## Real-World Example: Microservices vs Monolith

**Context**: E-commerce platform, 10 developers, 50K users, 3-year roadmap

**Decision Criteria** (weighted):
1. Time-to-market (10) - Startup needs to ship fast
2. Team expertise (8) - Team knows monoliths well
3. Scalability (7) - Growth expected to 500K users
4. Operational complexity (6) - Small DevOps team

**Alternatives**:
1. **Monolith** (Rails, Django, Laravel)
2. **Microservices** (Kubernetes, service mesh)
3. **Modular Monolith** (well-bounded modules, single deployment)
4. **Serverless** (AWS Lambda, Firebase)

**Evaluation**:
| Alternative | Time-to-Market (10) | Team Expertise (8) | Scalability (7) | Ops Complexity (6) | **Total** |
|-------------|--------------------|--------------------|-----------------|-------------------|-----------|
| Monolith | 9 × 10 = 90 | 10 × 8 = 80 | 6 × 7 = 42 | 9 × 6 = 54 | **266** |
| Microservices | 4 × 10 = 40 | 3 × 8 = 24 | 10 × 7 = 70 | 2 × 6 = 12 | **146** |
| Modular Monolith | 8 × 10 = 80 | 9 × 8 = 72 | 8 × 7 = 56 | 8 × 6 = 48 | **256** |
| Serverless | 7 × 10 = 70 | 5 × 8 = 40 | 9 × 7 = 63 | 7 × 6 = 42 | **215** |

**Decision**: Modular Monolith (256) - balances fast time-to-market, team expertise, scalability, and operational simplicity.

**Documented in ADR-003**: See templates.md for full example.

---

**Last Updated**: 2026-01-24
**Review Schedule**: When making architectural decisions, before vendor commitments
