# Governance and Risk - RED Phase Rationalization Patterns

## Date
2026-01-24

## Critical Gaps Identified (10)

From baseline testing, agents WITHOUT the skill exhibit these systematic failures:

### 1. No Enforcement of Governance Levels
**What agents do**: Use "I'd recommend" rather than "you must"
**Impact**: All practices treated as optional, shortcuts under pressure
**Skill must provide**: Clear framework for when practices are mandatory based on project risk level

### 2. Validating Rationalizations
**What agents do**: Lead with empathy ("I understand the pressure")
**Impact**: Legitimizes shortcuts before pushing back
**Skill must provide**: Direct confrontation of rationalizations, name cognitive biases

### 3. Making Requirements Palatable
**What agents do**: Offer "lightweight" versions to reduce perceived burden
**Impact**: Signals requirements are negotiable
**Skill must provide**: Non-negotiable baselines, no "lite" versions for mandatory practices

### 4. Accepting Predetermined Outcomes
**What agents do**: Assume outcome ("Likely still Auth0") in alternatives analysis
**Impact**: Analysis becomes validation theater, not genuine evaluation
**Skill must provide**: Requirement that analysis happens BEFORE commitment

### 5. No Risk Classification Framework
**What agents do**: List risks without categorizing probability/impact
**Impact**: All risks seem equal, mitigation feels optional
**Skill must provide**: Risk matrix (probability × impact), mandatory mitigation for high risks

### 6. Documentation Treated as Optional
**What agents do**: Suggest documentation but don't require it
**Impact**: Decisions and risks lost, context erosion
**Skill must provide**: Mandatory documentation templates with clear ownership

### 7. No Ongoing Process for Risk Monitoring
**What agents do**: Offer "lightweight alternatives" like async check-ins
**Impact**: Monitoring doesn't happen consistently
**Skill must provide**: Scheduled review cadence, risk triggers for ad-hoc reviews

### 8. Reactive Stance Despite Proactive Advice
**What agents do**: Acknowledge reactive costs but don't mandate prevention
**Impact**: Mixed signal, prevention still treated as optional
**Skill must provide**: Clear requirement that mitigation plans exist before risk acceptance

### 9. No Decision Analysis Framework
**What agents do**: Suggest ad-hoc criteria and matrices
**Impact**: Systematic evaluation easily abandoned
**Skill must provide**: Step-by-step DAR methodology (CMMI-aligned)

### 10. Authority and Social Pressure Not Confronted
**What agents do**: Accommodate CTO preferences and team consensus
**Impact**: Authority bias and groupthink lead to poor decisions
**Skill must provide**: Requirement for independent analysis, document dissenting views

---

## Rationalization Patterns (10)

These are the excuses agents accept or fail to counter:

| Pattern | Scenarios | Why Tempting | Why It Fails | Skill Must Counter With |
|---------|-----------|--------------|--------------|-------------------------|
| **"It's obvious"** | 1 | Saves time, team aligned | "Obvious" becomes mysterious in 6 months, assumptions not validated | Requirement to document even obvious decisions, context loss examples |
| **"Low-risk project"** | 2 | Small scope, internal only | Scope creep and timeline slips hit simple projects too | Risk identification mandatory regardless of perceived risk |
| **"CTO/authority prefers it"** | 3 | Reduces conflict, speeds decision | Authority bias prevents genuine analysis, vendor lock-in | Independent alternatives analysis before authority input |
| **"We've already invested time"** | 3 | Feels wasteful to go backwards | Sunk cost fallacy - past ≠ future validation | Name the fallacy, calculate future cost vs sunk cost |
| **"Trust the vendor"** | 4 | Vendor reputation, SLA promises | SLAs are probabilistic, all vendors have outages | SLA math (99.9% = 43 min/month), mitigation required |
| **"We'll fix it if it happens"** | 2, 4 | Defers work, focuses on current | Reactive costs 3-10x proactive, incidents hit when capacity lowest | Mitigation cost math, mandate prevention for high risks |
| **"Risks haven't materialized"** | 5 | Past success validates approach | Absence to-date ≠ absence of future, risks evolve | Lifecycle risk evolution, complacency before late-stage crunch |
| **"Process feels like bureaucracy"** | 1, 2, 5 | Team wants to code not document | Lightweight process prevents heavyweight problems | 30 min planning saves hours firefighting, process ≠ bureaucracy |
| **"We're tired/under pressure"** | 2, 4 | Exhaustion and deadlines real | Shortcuts compound into crisis, more pressure later | Skipping governance creates more work, not less |
| **"We'll document later"** | Multiple | Defers effort, focuses on delivery | "Later" never comes, context lost | Documentation = part of "done", not optional follow-up |

**Most dangerous**: "We'll fix it if it happens" (reactive stance)  
**Most insidious**: "It's obvious" (prevents documentation, compounding knowledge loss)

---

## Skill Requirements Derived From Patterns

Based on these gaps and patterns, the governance-and-risk skill MUST provide:

1. **Governance Level Framework**: When ADRs/RSKM are mandatory vs optional (tied to project risk)
2. **Rationalization Counter-Table**: Explicit refutation of each pattern above
3. **Non-Negotiable Templates**: ADR and risk register formats that are mandatory at certain levels
4. **DAR Methodology**: Systematic alternatives analysis process (CMMI SP 1.1-1.6)
5. **Risk Matrix**: Probability × Impact classification with mitigation requirements
6. **Documentation Mandate**: Clear ownership and completeness criteria
7. **Monitoring Schedule**: Risk review cadence based on project duration
8. **Authority Bias Resistance**: Process for independent analysis despite stakeholder preferences
9. **Cost Calculations**: Proactive vs reactive cost comparisons for common scenarios
10. **Red Flags List**: Early warning signs that governance is being skipped

---

**Last Updated**: 2026-01-24
**Status**: Patterns identified, ready for GREEN phase skill writing
