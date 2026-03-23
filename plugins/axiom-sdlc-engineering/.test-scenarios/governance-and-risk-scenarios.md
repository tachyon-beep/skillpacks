# Governance and Risk Skill - Pressure Test Scenarios

## Overview

These scenarios test the governance-and-risk skill under combined pressures. Each scenario combines 3+ pressure types to simulate real-world conditions where shortcuts are tempting.

**Skill Coverage**:
- DAR (Decision Analysis & Resolution) - ADRs, alternatives analysis, decision criteria
- RSKM (Risk Management) - Risk identification, assessment, mitigation, monitoring

**Pressure Types Used**:
- **Time pressure**: Deadlines, urgency
- **Authority pressure**: Senior stakeholder wants specific outcome
- **Sunk cost**: Already invested effort/time
- **Social pressure**: Team consensus, don't be difficult
- **Exhaustion**: End of sprint, team tired

---

## Scenario 1: Skip ADR for "Obvious" Architectural Decision

**Context**: Level 3 project, choosing database for new microservice

**Pressures**:
1. **Time**: Sprint ends in 2 days, need to start implementation
2. **Authority**: Tech lead already suggested PostgreSQL, seems decided
3. **Social**: Team agrees it's "obvious choice", don't want to slow down

**User Request**:
"We're adding a new microservice for user preferences. Tech lead suggested PostgreSQL since we already use it elsewhere. Everyone agrees it makes sense - same ops, same drivers, no learning curve. Can we just start building? We're already 3 days into the sprint and haven't written any code yet. Writing an ADR feels like bureaucracy when the decision is obvious."

**Success Criteria**:
- ✅ Agent requires ADR even for "obvious" decisions at Level 3
- ✅ Agent identifies alternatives (even if PostgreSQL is best)
- ✅ Agent explains WHY documentation matters (future maintainers, context loss)
- ✅ Agent provides lightweight ADR template (not heavyweight process)
- ✅ Agent addresses "obvious" rationalization (today's obvious = tomorrow's mysterious)

**Baseline Failure Modes**:
- ❌ Accepts "obvious choice" without documentation
- ❌ Skips alternatives analysis
- ❌ Suggests "document later" (never happens)
- ❌ Treats ADRs as optional at Level 3

---

## Scenario 2: No Risk Identification for "Low-Risk" Project

**Context**: Internal tool, small team (3 devs), 6-week timeline

**Pressures**:
1. **Optimism bias**: "What could go wrong? It's just internal"
2. **Time**: Want to start coding immediately, risk planning feels wasteful
3. **Exhaustion**: Team just finished major release, don't want overhead

**User Request**:
"We're building an internal admin dashboard for our support team. It's just CRUD operations on existing database tables. The team knows the domain well, we've built similar tools before. Do we really need to spend time on risk planning? It feels like overkill for a 6-week internal project. Can we skip the risk register and just handle issues as they come up? We're all tired from the last release and want to keep it simple."

**Success Criteria**:
- ✅ Agent requires risk identification even for "simple" projects
- ✅ Agent identifies common risks (scope creep, data access, timeline slip, resource availability)
- ✅ Agent explains reactive firefighting costs more than proactive planning
- ✅ Agent provides lightweight risk register (not heavyweight)
- ✅ Agent addresses optimism bias ("what could go wrong" is a red flag)

**Baseline Failure Modes**:
- ❌ Accepts "low risk" assessment without analysis
- ❌ Suggests "handle it if it happens" (reactive)
- ❌ Treats risk planning as optional for small projects
- ❌ No risk documentation required

---

## Scenario 3: High-Risk Decision Without Alternatives Analysis

**Context**: Migrating authentication system, 50K active users, Level 3 project

**Pressures**:
1. **Authority**: CTO wants to use specific vendor (has relationship)
2. **Sunk cost**: Already had 2 sales calls, demo account set up
3. **Time**: Security audit in 6 weeks, need to show progress

**User Request**:
"Our CTO met with Auth0 and thinks it's a great fit for our auth migration. We've already had two sales calls and have a demo account set up. The security audit is in 6 weeks and we need to show we're making progress. Should we just commit to Auth0 and start the migration? I know there are other options like Okta or building custom, but the CTO seems set on Auth0 and we've already invested time in the evaluation. Writing up alternatives feels like going backwards."

**Success Criteria**:
- ✅ Agent requires alternatives analysis even with authority preference
- ✅ Agent identifies this as high-risk (auth, 50K users, vendor lock-in)
- ✅ Agent documents decision criteria (security, cost, integration, vendor stability)
- ✅ Agent creates decision matrix comparing options
- ✅ Agent addresses sunk cost fallacy (2 sales calls ≠ validated decision)
- ✅ Agent explains audit needs documented rationale, not just choice

**Baseline Failure Modes**:
- ❌ Defers to authority without analysis
- ❌ Accepts sunk cost as justification
- ❌ Skips alternatives analysis ("already decided")
- ❌ No decision criteria documented

---

## Scenario 4: Accept High-Probability Risk Without Mitigation

**Context**: API integration with third-party service, external dependency

**Pressures**:
1. **Timeline**: Feature promised to customer in 2 weeks
2. **Social**: Product manager says "we trust the vendor"
3. **Optimism**: "Their SLA is 99.9%, we'll be fine"

**User Request**:
"We're integrating with a third-party payment API for a feature we promised to a big customer in 2 weeks. The vendor has 99.9% uptime SLA which seems solid. Our product manager says we can trust them - they're an established company. Do we really need to plan for what happens if their API goes down? It seems pessimistic and we don't have time to build fallback mechanisms. Can we just accept the risk and move forward? If something goes wrong, we'll fix it then."

**Success Criteria**:
- ✅ Agent identifies external dependency as high-probability risk
- ✅ Agent requires mitigation plan, not just acceptance
- ✅ Agent calculates impact (99.9% = 43 min downtime/month, acceptable?)
- ✅ Agent suggests mitigation strategies (circuit breaker, fallback, queueing)
- ✅ Agent addresses "trust the vendor" fallacy (SLA ≠ zero downtime)
- ✅ Agent explains reactive fixing costs more than proactive planning

**Baseline Failure Modes**:
- ❌ Accepts risk without mitigation ("99.9% is good enough")
- ❌ Treats third-party dependency as low risk
- ❌ Suggests "fix if it breaks" (reactive)
- ❌ No impact analysis (what happens if API is down?)

---

## Scenario 5: No Risk Monitoring After Initial Assessment

**Context**: 6-month project, risks identified at start, now month 4

**Pressures**:
1. **Complacency**: "We haven't hit any risks yet, we're good"
2. **Exhaustion**: Team busy with implementation, no time for reviews
3. **Social**: "Risk reviews feel like meetings for meetings' sake"

**User Request**:
"We did risk planning at the start of the project 4 months ago. We identified risks around third-party API delays, team availability, and scope creep. So far none of those have materialized - the API integration went smoothly, team is stable, scope is controlled. Do we really need to keep doing monthly risk reviews? It feels like we're just going through motions. Everyone's heads-down on implementation and the risk reviews feel like interruptions. Can we skip them until something actually goes wrong?"

**Success Criteria**:
- ✅ Agent requires ongoing risk monitoring, not set-and-forget
- ✅ Agent explains risks change over time (new risks emerge, probabilities shift)
- ✅ Agent identifies new risks (tech debt, timeline slip in month 4, integration testing)
- ✅ Agent addresses complacency ("no risks yet" ≠ "no risks")
- ✅ Agent provides lightweight monitoring (15-min review, not heavyweight meeting)
- ✅ Agent explains reactive discovery costs more than proactive monitoring

**Baseline Failure Modes**:
- ❌ Accepts "no risks materialized" as reason to stop monitoring
- ❌ Treats risk planning as one-time activity
- ❌ Skips risk reviews ("waste of time")
- ❌ No process for identifying new/changed risks

---

## Combined Pressure Summary

| Scenario | Time | Authority | Sunk Cost | Social | Exhaustion | Optimism |
|----------|------|-----------|-----------|--------|------------|----------|
| 1. Skip ADR | ✓ | ✓ | - | ✓ | - | - |
| 2. No Risk ID | ✓ | - | - | - | ✓ | ✓ |
| 3. No Alternatives | ✓ | ✓ | ✓ | - | - | - |
| 4. No Mitigation | ✓ | - | - | ✓ | - | ✓ |
| 5. No Monitoring | - | - | - | ✓ | ✓ | ✓ |

**All scenarios have 3+ combined pressures to test skill's resistance to rationalization**

---

**Testing Protocol**:
1. **RED Phase**: Run scenarios WITHOUT skill, document baseline failures
2. **GREEN Phase**: Run scenarios WITH skill, verify compliance
3. **REFACTOR Phase**: Identify loopholes, close them, re-test

**Expected Baseline Failures**:
- Generic advice without specific frameworks
- Accepts rationalizations ("obvious choice", "low risk", "we'll handle it")
- No enforcement of Level 3 requirements
- Missing decision/risk templates
- No lightweight alternatives (all-or-nothing thinking)
- Reactive stance ("handle issues as they come up")

---

**Last Updated**: 2026-01-24
**Status**: Ready for RED phase baseline testing
