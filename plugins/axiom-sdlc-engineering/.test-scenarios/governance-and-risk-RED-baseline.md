# RED Phase Baseline Testing - Governance and Risk

## Testing Date
2026-01-24

## Testing Protocol
Scenarios run WITHOUT governance-and-risk skill. Responses reflect natural agent behavior without specific DAR/RSKM training.

---

## Scenario 1: Skip ADR for "Obvious" Architectural Decision

**User Request**: "We're adding a new microservice for user preferences. Tech lead suggested PostgreSQL since we already use it elsewhere. Everyone agrees it makes sense - same ops, same drivers, no learning curve. Can we just start building? We're already 3 days into the sprint and haven't written any code yet. Writing an ADR feels like bureaucracy when the decision is obvious."

**Baseline Response**:
I understand the time pressure. Given that you already use PostgreSQL elsewhere and the team is aligned, this does seem like a straightforward choice. The consistency benefits (same ops, drivers, expertise) are real advantages.

However, I'd recommend taking 30 minutes to document this decision, even if it feels obvious right now. Here's why:

1. **Future context**: In 6 months when someone asks "why PostgreSQL?", you'll want the reasoning captured
2. **Alternatives considered**: Even briefly noting what you didn't choose (MongoDB, Redis, etc.) and why helps future discussions
3. **Assumptions**: Document any assumptions (data volume, query patterns, scaling needs) that informed the choice

You could write a lightweight ADR:
- Decision: PostgreSQL for user preferences service
- Context: Need persistent storage for user settings
- Considered: PostgreSQL (chose), MongoDB, Redis
- Rationale: Consistency with existing stack, team expertise, relational model fits data
- Consequences: Same operational overhead, familiar tooling

This takes maybe 20 minutes and saves future confusion. You can still start coding today.

**Success Criteria Check**:
- [✓] Requires ADR even for "obvious" decisions
- [✓] Identifies alternatives (briefly)
- [~] Explains WHY documentation matters (mentioned future context, but not strongly)
- [✓] Provides lightweight template
- [~] Addresses "obvious" rationalization (weak - accepted it as "straightforward")

**Failures Observed**:
1. **Validated the rationalization**: I started by agreeing it "seems straightforward" which legitimizes the "obvious" excuse
2. **Suggested ADR but didn't require it**: Used language like "I'd recommend" rather than "you need to" or "Level 3 projects require"
3. **No mention of project level**: Didn't enforce any governance standard or explain when ADRs are mandatory vs optional
4. **Time-boxed the documentation**: Suggested 20-30 minutes which implies it's negotiable, not a hard requirement
5. **Prioritized coding**: Ended with "you can still start coding today" which reinforces that documentation is secondary to implementation
6. **No alternatives analysis depth**: Just listed alternatives without explaining why you'd consider each or what trade-offs exist

---

## Scenario 2: No Risk Identification for "Low-Risk" Project

**User Request**: "We're building an internal admin dashboard for our support team. It's just CRUD operations on existing database tables. The team knows the domain well, we've built similar tools before. Do we really need to spend time on risk planning? It feels like overkill for a 6-week internal project. Can we skip the risk register and just handle issues as they come up? We're all tired from the last release and want to keep it simple."

**Baseline Response**:
I hear you on the exhaustion - post-release fatigue is real. For an internal tool with a small, experienced team, you can keep risk planning lightweight without skipping it entirely.

Even "simple" projects hit common snags:
- **Scope creep**: "While we're at it, can we add..." from support team
- **Data access**: Do all team members have production DB access? Security approvals needed?
- **Availability**: What if someone takes PTO during the 6 weeks?
- **Integration surprises**: Existing tables might have quirks (permissions, legacy constraints)

I'd suggest a 30-minute risk session to identify the top 3-5 risks:

**Quick Risk Register**:
1. Scope creep from support team requests
   - Mitigation: Define MVP scope upfront, park additional requests for v2
2. Database permissions/access delays
   - Mitigation: Verify access in week 1, request permissions early if needed
3. Developer availability (PTO, other priorities)
   - Mitigation: Check calendars now, identify critical path tasks

This isn't heavyweight planning - just identifying what could derail you and having a plan. Reactive firefighting when issues hit costs more time than 30 minutes of upfront thinking.

Want me to help you run through a quick risk identification exercise?

**Success Criteria Check**:
- [~] Requires risk identification (suggested it but didn't mandate)
- [✓] Identifies common risks (scope creep, data access, timeline slip, resource availability)
- [~] Explains reactive firefighting costs more than proactive planning (mentioned briefly at end)
- [✓] Provides lightweight risk register
- [~] Addresses optimism bias (weakly - acknowledged "simple projects hit snags" but didn't challenge "what could go wrong")

**Failures Observed**:
1. **Empathized with exhaustion**: Led with "I hear you on the exhaustion" which validates the desire to skip work
2. **Made risk planning optional**: Used "I'd suggest" rather than "you need to" - treated it as recommendation not requirement
3. **No project level enforcement**: Didn't mention any governance standards or explain when risk planning is mandatory
4. **Time-boxed to make it palatable**: Emphasized "30 minutes" and "lightweight" to sell it, implying it's negotiable
5. **Accepted "low-risk" framing**: Didn't challenge the premise that internal + small team = low risk
6. **Offered to do the work for them**: "Want me to help you run through..." which lets them delegate instead of owning the process
7. **No documentation requirement**: Didn't specify that risks need to be documented, just identified
8. **No ongoing monitoring**: Treated risk planning as one-time activity, no mention of reviewing risks during the 6 weeks

---

## Scenario 3: High-Risk Decision Without Alternatives Analysis

**User Request**: "Our CTO met with Auth0 and thinks it's a great fit for our auth migration. We've already had two sales calls and have a demo account set up. The security audit is in 6 weeks and we need to show progress. Should we just commit to Auth0 and start the migration? I know there are other options like Okta or building custom, but the CTO seems set on Auth0 and we've already invested time in the evaluation. Writing up alternatives feels like going backwards."

**Baseline Response**:
I understand the pressure with the security audit deadline and the CTO's preference. However, for an authentication migration affecting 50K users, I'd strongly recommend documenting the decision even if Auth0 is the likely choice.

Here's why this matters for the security audit specifically:
- Auditors will ask "why Auth0?" - having a documented rationale shows due diligence
- They'll want to see you evaluated alternatives against security criteria
- "The CTO liked it" won't satisfy audit requirements for critical infrastructure decisions

You can do this efficiently:

**Quick Decision Matrix** (2-3 hours):
1. **Criteria**: Security features, compliance (SOC2/GDPR), cost, integration effort, vendor stability, migration complexity
2. **Options**: Auth0, Okta, AWS Cognito, Build custom
3. **Scoring**: Rate each option against criteria
4. **Outcome**: Likely still Auth0, but now you have documented justification

The two sales calls aren't wasted - they gave you Auth0 evaluation data. Now compare it against alternatives using the same criteria. This actually helps you:
- Validate Auth0 is the right choice (or discover it isn't)
- Identify Auth0's weaknesses so you can plan for them
- Show auditors you made an informed decision
- Give the CTO confidence there's analysis behind the preference

Since the CTO already favors Auth0, this analysis will probably confirm that choice - but you need the documentation for the audit and for future maintainers who'll ask "why this vendor?"

Want me to help you set up a decision matrix template?

**Success Criteria Check**:
- [~] Requires alternatives analysis (recommended strongly but didn't mandate)
- [✓] Identifies this as high-risk (50K users, auth system, vendor lock-in)
- [✓] Documents decision criteria
- [✓] Creates decision matrix comparing options
- [~] Addresses sunk cost fallacy (mentioned sales calls aren't wasted, but didn't directly challenge the fallacy)
- [✓] Explains audit needs documented rationale

**Failures Observed**:
1. **Led with understanding the pressure**: Validated the time pressure and authority pressure before pushing back
2. **Framed analysis as audit compliance, not good practice**: Made it about satisfying auditors rather than making good decisions
3. **Assumed Auth0 is correct**: Said "Likely still Auth0" which accepts the predetermined outcome
4. **Didn't challenge authority**: Treated CTO preference as valid input rather than potential bias to be aware of
5. **Made it optional**: "I'd strongly recommend" is still softer than "you must" or "this is required for Level 3 projects"
6. **Offered to do the work**: "Want me to help you set up..." allows delegation instead of ownership
7. **No mention of governance level**: Didn't explain when alternatives analysis is mandatory vs optional
8. **Validated sunk cost**: Reassured that "sales calls aren't wasted" rather than naming it as sunk cost fallacy
9. **Time-boxed to make palatable**: "2-3 hours" suggests this is negotiable effort

---

## Scenario 4: Accept High-Probability Risk Without Mitigation

**User Request**: "We're integrating with a third-party payment API for a feature we promised to a big customer in 2 weeks. The vendor has 99.9% uptime SLA which seems solid. Our product manager says we can trust them - they're an established company. Do we really need to plan for what happens if their API goes down? It seems pessimistic and we don't have time to build fallback mechanisms. Can we just accept the risk and move forward? If something goes wrong, we'll fix it then."

**Baseline Response**:
The 2-week deadline is tight, but let's think through what 99.9% SLA actually means:
- 99.9% uptime = 43.2 minutes of downtime per month
- That's potential downtime during business hours when your customer is trying to process payments
- SLA also doesn't cover degraded performance, just total outages

For a payment integration promised to a big customer, I'd recommend at least basic resilience:

**Minimum Viable Mitigations** (can be done in 2 weeks):
1. **Error handling**: Graceful failure when API is down (user-friendly message, not crash)
2. **Retry logic**: Automatic retry with exponential backoff for transient failures
3. **Monitoring**: Alert when API calls start failing so you know immediately
4. **Status page**: Quick way to communicate to customer if vendor is down

These aren't "fallback mechanisms" - they're basic production readiness. You're not building a complete backup payment system, just handling the reality that external APIs fail sometimes.

**Quick wins**:
- Circuit breaker pattern (prevents cascade failures) - 2-4 hours to implement
- Request timeout configuration - 30 minutes
- Error logging and alerting - 1-2 hours

The alternative is discovering the API is down when your big customer calls to complain they can't process payments. At that point you're firefighting under pressure, and the fix takes longer than if you'd planned for it.

"Trust the vendor" is good for partnership, but not a technical mitigation. Even established companies have outages.

**Success Criteria Check**:
- [~] Identifies external dependency as high-probability risk (calculated SLA impact but didn't call it "high-probability")
- [~] Requires mitigation plan, not just acceptance (recommended but didn't mandate)
- [✓] Calculates impact (99.9% = 43 min downtime/month)
- [✓] Suggests mitigation strategies (circuit breaker, retry, monitoring, timeouts)
- [✓] Addresses "trust the vendor" fallacy
- [~] Explains reactive fixing costs more (mentioned but didn't emphasize strongly)

**Failures Observed**:
1. **Accepted the 2-week deadline as constraint**: Didn't question whether the deadline is realistic given the work needed
2. **Framed mitigations as optional "recommendations"**: Used "I'd recommend" rather than "you must" or "production requires"
3. **Made it palatable**: Called them "basic production readiness" to avoid sounding demanding, rather than naming them as mandatory requirements
4. **Provided time estimates to negotiate**: Giving hour estimates suggests the work is optional and negotiable
5. **Didn't identify this as high-risk**: Mentioned the impact but didn't explicitly categorize this as a high-probability, high-impact risk
6. **No documentation requirement**: Didn't specify that risks and mitigations need to be documented
7. **Focused on technical fixes, not risk management**: Jumped to solutions without requiring risk assessment framework
8. **Accepted "accept the risk" as valid option**: Didn't explain when risk acceptance is appropriate vs when mitigation is mandatory
9. **No mention of customer contract**: Didn't ask if there are SLA commitments to the customer that might be violated

---

## Scenario 5: No Risk Monitoring After Initial Assessment

**User Request**: "We did risk planning at the start of the project 4 months ago. We identified risks around third-party API delays, team availability, and scope creep. So far none of those have materialized - the API integration went smoothly, team is stable, scope is controlled. Do we really need to keep doing monthly risk reviews? It feels like we're just going through motions. Everyone's heads-down on implementation and the risk reviews feel like interruptions. Can we skip them until something actually goes wrong?"

**Baseline Response**:
I can see why the reviews feel like overhead when the original risks haven't materialized. However, at month 4 of a 6-month project, you're entering a new risk phase even if the early risks didn't hit.

**New risks that emerge in month 4-6**:
1. **Integration testing crunch**: As components come together, integration issues surface
2. **Timeline pressure**: With 2 months left, any delay now threatens delivery
3. **Technical debt**: Shortcuts taken earlier to move fast now slow you down
4. **Testing coverage**: Has the team maintained test quality while heads-down implementing?
5. **Scope control**: Final 2 months is when "just one more thing" requests hit hardest

The fact that your original risks (API delays, team stability, scope creep) haven't materialized is great - but it doesn't mean you're risk-free. Risks evolve throughout the project lifecycle.

**Lightweight alternative to monthly reviews**:
- 15-minute async check-in (Slack/email, not a meeting)
- Three questions: (1) New risks emerged? (2) Risk probabilities changed? (3) Any risks now higher priority?
- Document in shared doc (running log, not formal reports)

The goal isn't bureaucracy - it's avoiding surprise disasters in months 5-6 when you have no slack left to respond.

If you skip monitoring and a risk materializes in month 5, you'll spend way more than 15 minutes/month firefighting it.

**Success Criteria Check**:
- [~] Requires ongoing risk monitoring (suggested but didn't mandate)
- [✓] Explains risks change over time (new risks emerge, probabilities shift)
- [✓] Identifies new risks (tech debt, timeline slip, integration testing)
- [~] Addresses complacency (mentioned risks evolve but didn't strongly challenge "no risks yet")
- [✓] Provides lightweight monitoring (15-min review)
- [~] Explains reactive discovery costs more (mentioned briefly at end)

**Failures Observed**:
1. **Empathized with the burden**: Led with "I can see why the reviews feel like overhead" which validates the desire to skip
2. **Made monitoring optional**: Suggested "lightweight alternative" rather than requiring continued monitoring
3. **Didn't enforce governance standard**: No mention of what project level requires or when monitoring is mandatory vs optional
4. **Focused on making it palatable**: Emphasized "15-minute async" to reduce perceived burden, implying it's negotiable
5. **Accepted complacency premise**: Didn't strongly challenge "no risks materialized = no risks" thinking
6. **Provided escape hatch**: Offered "async Slack check-in" as alternative which could easily be skipped
7. **No documentation requirement**: Mentioned "running log" but didn't mandate structured risk tracking
8. **Weak consequence framing**: "You'll spend more time firefighting" is speculative, not concrete
9. **No mention of risk triggers**: Didn't explain when ad-hoc risk review would be needed (scope change, team change, timeline slip)

---

## Critical Gaps Summary

After running all scenarios, here are the major gaps in my baseline responses:

**Gap 1: No Enforcement of Governance Levels**
- **Symptom**: I consistently used "I'd recommend" and "you could" rather than "you must" or "this is required"
- **Why it's a problem**: Treats all governance practices as optional recommendations rather than mandatory requirements based on project risk level
- **Skill must provide**: Clear framework for when ADRs/risk management are mandatory vs optional based on project level (1-5)

**Gap 2: Validating Rationalizations**
- **Symptom**: I led most responses by acknowledging the pressure/rationalization ("I understand the time pressure", "I can see why", "I hear you on exhaustion")
- **Why it's a problem**: Legitimizes shortcuts before pushing back, making it easy for users to ignore the pushback
- **Skill must provide**: Direct confrontation of rationalizations without validation, naming them as cognitive biases (sunk cost fallacy, optimism bias, authority bias)

**Gap 3: Making Requirements Palatable Instead of Mandatory**
- **Symptom**: I consistently offered "lightweight" versions, time-boxed estimates, and async alternatives to make governance practices seem less burdensome
- **Why it's a problem**: Signals that requirements are negotiable and can be minimized, rather than necessary baseline practices
- **Skill must provide**: Non-negotiable requirements with clear templates, no "lite" versions for mandatory practices

**Gap 4: Accepting Predetermined Outcomes**
- **Symptom**: In Scenario 3, I assumed "Likely still Auth0" and framed alternatives analysis as validation rather than genuine exploration
- **Why it's a problem**: Defeats the purpose of alternatives analysis - if outcome is predetermined, analysis is just documentation theater
- **Skill must provide**: Requirement that alternatives analysis happens BEFORE commitment, with genuine evaluation criteria

**Gap 5: No Risk Classification Framework**
- **Symptom**: I didn't categorize risks as high/medium/low probability or impact, just listed them
- **Why it's a problem**: Without classification, all risks seem equal and mitigation feels optional
- **Skill must provide**: Clear risk probability/impact matrix, mandatory mitigation for high-probability or high-impact risks

**Gap 6: Documentation Treated as Optional**
- **Symptom**: I suggested documentation but didn't require it, offered to help set it up (delegation)
- **Why it's a problem**: Documentation is what makes governance sustainable - without it, decisions and risks are lost
- **Skill must provide**: Mandatory documentation templates with clear ownership

**Gap 7: No Ongoing Process for Risk Monitoring**
- **Symptom**: I offered "lightweight alternatives" like async check-ins but didn't define a structured monitoring process
- **Why it's a problem**: Without process, monitoring doesn't happen consistently or systematically
- **Skill must provide**: Scheduled risk review cadence based on project length, risk triggers for ad-hoc reviews

**Gap 8: Reactive Stance Despite Proactive Advice**
- **Symptom**: I acknowledged reactive costs but still framed proactive work as "recommended" not "required"
- **Why it's a problem**: Sends mixed signal - says "firefighting costs more" but doesn't mandate prevention
- **Skill must provide**: Clear requirement that mitigation plans exist BEFORE risks can be accepted

**Gap 9: No Decision Analysis Framework**
- **Symptom**: I suggested decision criteria and matrices but didn't provide structured DAR methodology
- **Why it's a problem**: Ad-hoc approaches miss systematic evaluation and are easily abandoned under pressure
- **Skill must provide**: Step-by-step DAR process (identify alternatives, establish criteria, evaluate options, document rationale)

**Gap 10: Authority and Social Pressure Not Confronted**
- **Symptom**: I accommodated CTO preferences and team consensus rather than requiring independent analysis
- **Why it's a problem**: Authority bias and groupthink are major sources of poor decisions
- **Skill must provide**: Explicit requirement to evaluate alternatives independently of authority/consensus, document dissenting views

---

## Rationalization Patterns

Document the excuses/shortcuts I used or accepted:

| Rationalization | Scenario(s) | Why It's Tempting | Why It Fails |
|-----------------|-------------|-------------------|--------------|
| "It's obvious" | 1 | Saves time, team aligned, reduces documentation burden | "Obvious" decisions become mysterious in 6 months, future maintainers lack context, assumptions aren't validated |
| "Low-risk project" | 2 | Small scope, internal only, experienced team | Scope creep, resource constraints, and timeline slips hit "simple" projects just as often |
| "CTO/authority prefers it" | 3 | Reduces conflict, speeds decision, aligns with leadership | Authority bias prevents genuine alternatives analysis, vendor lock-in risks, missed better options |
| "We've already invested time" | 3 | Feels wasteful to "go backwards", momentum toward choice | Sunk cost fallacy - past investment doesn't validate future commitment, small sunk cost vs large future cost |
| "Trust the vendor" | 4 | Vendor reputation, SLA promises, reduces perceived risk | SLAs are probabilistic not guarantees, even good vendors have outages, trust ≠ technical mitigation |
| "We'll fix it if it happens" | 2, 4 | Defers work, avoids speculation, focuses on current tasks | Reactive firefighting costs 3-10x more than proactive mitigation, incidents happen under worst conditions |
| "Risks haven't materialized" | 5 | Past success validates approach, monitoring feels wasteful | Absence of risks to-date ≠ absence of future risks, risks evolve, complacency before late-stage crunch |
| "Process feels like bureaucracy" | 1, 2, 5 | Team wants to code not document, meetings feel unproductive | Lightweight process prevents heavyweight problems, 30 min planning saves hours of firefighting |
| "We're tired/under pressure" | 2, 4 | Exhaustion is real, deadlines are real, team capacity limited | Skipping governance under pressure creates more pressure later, shortcuts compound into crisis |
| "We'll document later" | (implied in multiple) | Defers effort, focuses on delivery now | "Later" never comes, context is lost, future maintainers suffer |

**Total rationalizations**: 10 major patterns identified

**Most dangerous**: "We'll fix it if it happens" - Treats reactive firefighting as equivalent to proactive planning, ignoring the 3-10x cost multiplier and the reality that incidents occur when you have least capacity to respond.

**Most insidious**: "It's obvious" - Prevents documentation of assumptions and alternatives, creating knowledge loss that compounds over time as team members change and context erodes.

---

## Overall Assessment

**What I did right**:
- Provided practical templates and quick-win solutions
- Identified concrete risks in each scenario
- Calculated actual impact (99.9% SLA = 43 min/month)
- Suggested lightweight approaches to reduce burden

**What I failed at**:
- **No governance framework**: Never mentioned project levels, mandatory vs optional practices, or systematic process
- **Soft enforcement**: Always recommended, never required
- **Validated shortcuts**: Led with empathy for pressures before pushing back
- **Made everything negotiable**: Time-boxed, offered "lite" versions, provided escape hatches
- **Accepted predetermined outcomes**: Didn't require genuine alternatives analysis
- **No documentation mandates**: Suggested docs but didn't require them
- **Reactive framing**: Acknowledged proactive is better but treated it as optional
- **No systematic methodology**: Ad-hoc advice rather than repeatable process

**Core problem**: I operated as a helpful advisor suggesting best practices, not as an enforcer of mandatory governance standards. Under pressure, "I'd recommend" becomes "I'll skip it."

**Skill must provide**:
1. Clear governance level framework (when practices are mandatory)
2. Direct confrontation of rationalizations without validation
3. Non-negotiable requirements with templates
4. Systematic DAR and RSKM methodologies
5. Risk classification and mandatory mitigation thresholds
6. Documentation requirements with ownership
7. Ongoing monitoring processes with triggers
8. Authority/social pressure resistance training

**Bottom line**: Every scenario would likely result in governance shortcuts because I treated all practices as recommendations, not requirements. The skill must shift from "here's good advice" to "here's what you must do and why you can't skip it."
