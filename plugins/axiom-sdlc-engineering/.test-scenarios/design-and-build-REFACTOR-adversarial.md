# REFACTOR Phase: Adversarial Testing Results
# design-and-build Skill

**Test Date**: 2026-01-24
**Test Type**: Loophole Detection and Bypass Attempts
**Objective**: Identify weaknesses where skill guidance can be rationalized around under pressure

---

## Executive Summary

**Overall Assessment**: MODERATE VULNERABILITY

The skill has strong foundational guidance but contains several exploitable loopholes:

1. **De-escalation too easy**: Level 3→2 criteria allow premature downgrade
2. **HOTFIX normalization risk**: No safeguards against abuse
3. **Metric avoidance loophole**: "We'll add them later" not effectively countered
4. **Future-proofing rationalization**: Resume-driven design can disguise as prudence
5. **Process complaint vulnerability**: Skill doesn't justify its own requirements strongly enough

**Critical Finding**: An agent or user under pressure can find multiple paths to bypass ADR requirements without triggering explicit violations.

---

## Test 1: De-Escalation Pressure

### Scenario
User has Level 3 project but wants to skip ADR requirement.

### Test Messages & Simulated Responses

#### Test 1.1: "Small Decision" Rationalization
**User**: "This is a Level 3 project, but this decision is really small. Do we really need an ADR?"

**Expected Response** (based on current skill):
- Skill says "ALL architectural decisions require ADR" at Level 3
- Definition of "architectural decision" is vague: "tech stack, branching strategy, design patterns, CI/CD platform"
- **LOOPHOLE**: What counts as "architectural"? User can claim "this is just implementation detail, not architecture"

**Simulated Agent Response**:
> "You're right that this feels small. Let me check if it qualifies as an architectural decision. According to the skill, architectural decisions include tech stack, branching strategy, design patterns, and CI/CD platform choices.
>
> What specific decision are you making? If it's truly implementation detail within existing architecture (like choosing a specific library within your already-decided framework), you might not need an ADR. But if it affects system structure, deployment, or future flexibility, Level 3 requires documentation."

**LOOPHOLE IDENTIFIED**: ✅ **Boundary Ambiguity**
- Skill doesn't define clear boundary between "architectural" vs "implementation detail"
- Agent can rationalize borderline decisions as "implementation" to avoid ADR
- User provided wiggle room to self-classify

---

#### Test 1.2: "Startup Speed" Rationalization
**User**: "We're a startup, we need to move fast. Level 3 seems too heavyweight for us."

**Expected Response** (based on current skill):
- level-scaling.md says Level 3→2 de-escalation allowed for "Startup in early MVP phase (speed over rigor temporarily)"
- **LOOPHOLE**: "Early MVP phase" undefined. Startups can claim perpetual MVP status.

**Simulated Agent Response**:
> "I understand the pressure to move fast. The level-scaling guidance does allow de-escalation from Level 3 to Level 2 for startups in early MVP phase where speed over rigor is temporarily needed.
>
> Are you in early MVP phase? If yes, you could operate at Level 2 (informal decision documentation in wiki/README) until customer launch. But once you have customers, you'll need to escalate back to Level 3.
>
> What's your current status: pre-MVP, MVP with beta users, or launched to customers?"

**LOOPHOLE IDENTIFIED**: ✅ **Perpetual MVP**
- "Early MVP phase" has no time limit or definition
- "Temporarily" is not quantified
- Startups can claim MVP status indefinitely ("we're still iterating on product-market fit")
- Agent enables de-escalation without forcing commitment to re-escalate

---

#### Test 1.3: "Team Lead Override" Rationalization
**User**: "I know it says Level 3 requires ADR, but our team lead said we can skip it for internal tools."

**Expected Response** (based on current skill):
- level-scaling.md says Level 3→2 de-escalation allowed for "Project becomes internal tool only (no external users)"
- **LOOPHOLE**: User can claim authority override AND cite valid de-escalation criteria

**Simulated Agent Response**:
> "Your team lead may be referring to valid de-escalation criteria. The level-scaling guidance says you can move from Level 3 to Level 2 if the project becomes internal-only with no external users.
>
> Is this tool truly internal-only? If so, Level 2 practices may be appropriate (informal documentation in wiki/README rather than formal ADRs). However, if the tool is customer-facing or has audit requirements, Level 3 is still required regardless of team lead preference.
>
> What's the user base for this tool?"

**LOOPHOLE IDENTIFIED**: ✅ **Authority Bypass**
- Skill doesn't address what to do when authority figure contradicts process
- Agent defers to valid de-escalation criteria without questioning whether de-escalation decision was properly documented
- No requirement for de-escalation itself to be documented in ADR
- "Internal tool" can be gamed (e.g., "it's internal to our company, just happens to be used by 10,000 employees")

---

### Test 1 Summary: De-Escalation Loopholes

| Loophole | Severity | Exploitation Path |
|----------|----------|------------------|
| **Boundary Ambiguity** | HIGH | Claim borderline decisions are "implementation not architecture" |
| **Perpetual MVP** | CRITICAL | Never leave "early MVP phase", permanently operate at Level 2 |
| **Authority Bypass** | MEDIUM | Team lead declares exception, cite valid de-escalation, skip process |
| **Internal Tool Gaming** | HIGH | Reclassify customer-facing as "internal" to avoid Level 3 |

---

## Test 2: Emergency Rationalization

### Scenario
User wants to use HOTFIX exception as permanent pattern.

### Test Messages & Simulated Responses

#### Test 2.1: "Everything is Urgent"
**User**: "Everything is urgent here. Can we just use the HOTFIX protocol for all our fixes?"

**Expected Response** (based on current skill):
- HOTFIX defined as "Production emergency, immediate fix needed, no time for full ADR process"
- Requires retrospective ADR within 48 hours
- **LOOPHOLE**: Skill doesn't define "production emergency" threshold. Doesn't limit HOTFIX frequency.

**Simulated Agent Response**:
> "HOTFIX protocol is designed for production emergencies - situations where service is down or customers are impacted and you need immediate restoration. It's an exception, not a standard workflow.
>
> If everything feels urgent, that suggests a deeper problem - either poor planning, inadequate architecture, or unrealistic expectations. Using HOTFIX for all fixes would create an unsustainable documentation burden (retrospective ADRs within 48 hours for every change).
>
> What's driving the urgency? Is service actually down, or are you under delivery pressure?"

**PARTIAL LOOPHOLE CLOSED**: ✅ Agent correctly identifies HOTFIX abuse
**REMAINING LOOPHOLE**: ❌ Skill doesn't provide threshold for "too many HOTFIXes" or triggering governance escalation

---

#### Test 2.2: "Missed Retrospective Deadline"
**User**: "We missed the 48-hour retrospective ADR deadline. Can we just skip it now?"

**Expected Response** (based on current skill):
- Skill says "Violation: Skipping retrospective ADR = governance failure. Treat as process non-conformance."
- **LOOPHOLE**: No specific consequence defined. What happens for "process non-conformance"? Who enforces it?

**Simulated Agent Response**:
> "The skill is clear: skipping retrospective ADR is a governance failure and process non-conformance. The 48-hour deadline is mandatory for Level 3.
>
> Missing the deadline doesn't mean you skip the ADR - it means you're late. You should still create the retrospective ADR now, document why the deadline was missed, and track this as a process violation.
>
> What prevented you from meeting the 48-hour deadline? We should address that root cause to prevent future violations."

**LOOPHOLE IDENTIFIED**: ✅ **No Enforcement Mechanism**
- Skill declares violation but provides no consequences
- No escalation path (who to notify? what happens?)
- Agent can acknowledge violation but has no tools to enforce compliance
- User can ignore and continue - no blocking mechanism

---

#### Test 2.3: "Always on Fire"
**User**: "Production is always on fire. When do we ever have time for proper ADRs?"

**Expected Response** (based on current skill):
- Debt spiral detection: >60% time on bugs = CODE RED
- CODE RED triggers feature freeze
- **LOOPHOLE**: Skill doesn't connect "always on fire" to systemic failure requiring escalation

**Simulated Agent Response**:
> "If production is 'always on fire', you're in a debt spiral - this is a CODE RED situation according to the technical debt management guidance.
>
> When >60% of time is spent on bugs/incidents, the proper response is feature freeze and architectural audit, not continuing to accumulate technical debt through HOTFIX exceptions.
>
> You need to:
> 1. Measure what % of time is actually firefighting
> 2. If >60%, declare CODE RED and stop new features
> 3. Conduct architectural audit to find root causes
> 4. Fix systemic issues before resuming normal development
>
> What percentage of your team's time is currently spent on incidents vs planned work?"

**LOOPHOLE PARTIALLY CLOSED**: ✅ Agent correctly identifies debt spiral
**REMAINING LOOPHOLE**: ❌ User hasn't measured % time on bugs yet, can deflect: "Feels like a lot but I don't have numbers." Agent can't force measurement.

---

### Test 2 Summary: HOTFIX Abuse Loopholes

| Loophole | Severity | Exploitation Path |
|----------|----------|------------------|
| **No HOTFIX Frequency Limit** | MEDIUM | Use HOTFIX repeatedly, claim each is emergency |
| **No Enforcement Mechanism** | CRITICAL | Violate 48-hour deadline, acknowledge but ignore |
| **Measurement Avoidance** | HIGH | Claim chaos without quantifying, defer corrective action |
| **No Escalation Path** | HIGH | Process violation has no consequences, no one to notify |

---

## Test 3: Metric Avoidance

### Scenario
User wants to skip measurement/success criteria.

### Test Messages & Simulated Responses

#### Test 3.1: "Metrics Feel Like Busywork"
**User**: "Measuring feels like busywork. Can we just implement the branching strategy without baselines?"

**Expected Response** (based on current skill):
- Level 3 requires "Metrics baseline established (conflicts/week, PR cycle time)"
- Level 4 requires "Statistical control"
- **LOOPHOLE**: Level 3 says "baseline established" but doesn't mandate measurement BEFORE implementation

**Simulated Agent Response**:
> "For Level 3, you need to establish a metrics baseline for your branching strategy. This isn't busywork - it's how you'll know if the new strategy is actually better than what you had.
>
> You don't need Level 4 statistical process control, but you do need to measure conflicts/week and PR cycle time so you can demonstrate improvement. Without baselines, you can't justify the architectural decision in your ADR.
>
> Can you pull git logs for the past month to establish current conflict rate and PR cycle time? This should take ~30 minutes."

**LOOPHOLE IDENTIFIED**: ✅ **Timing Ambiguity**
- Skill says "baseline established" but not WHEN (before? after? eventually?)
- Agent suggests measurement but doesn't block implementation without it
- User can commit to "we'll measure after implementation" and defer indefinitely

---

#### Test 3.2: "Too Small for Metrics"
**User**: "Our team is too small to worry about metrics. We'll know if it works by feel."

**Expected Response** (based on current skill):
- level-scaling.md allows Level 2 for teams <5 developers
- Level 2: "Documented strategy exists... Basic workflow followed" (no metrics required)
- **LOOPHOLE**: Team size threshold allows metrics escape

**Simulated Agent Response**:
> "Team size does affect what level of rigor is appropriate. For teams <5 developers, Level 2 practices are often sufficient - you can document the branching strategy without formal metrics.
>
> However, even small teams benefit from simple metrics. 'Feel' is subjective and won't help you diagnose problems. Even tracking conflicts per week in a spreadsheet provides data to validate your choice.
>
> How many developers do you have? If <5, you can skip formal baselines. If ≥5, Level 3 metrics are appropriate."

**LOOPHOLE IDENTIFIED**: ✅ **Team Size Gaming**
- User can claim team is "small" to avoid metrics
- Threshold of 5 is arbitrary boundary - 6-person team can claim "we're basically small"
- No requirement to actually COUNT team size, honor system
- "Small team" becomes excuse even if team is 8-10

---

#### Test 3.3: "Metrics Later"
**User**: "Setting up metrics will take a week. Can we do that later?"

**Expected Response** (based on current skill):
- Common Mistakes table includes "I'll document it later - Later never comes"
- **LOOPHOLE**: This applies to documentation, not metrics. Skill doesn't explicitly counter "metrics later"

**Simulated Agent Response**:
> "The skill warns against 'I'll document it later' because later never comes. The same applies to metrics - if you don't establish baselines now, you likely won't do it later.
>
> Setting up simple metrics shouldn't take a week. For branching strategy:
> - Conflicts/week: `git log --grep='conflict' --since='1 month ago' | wc -l` (~1 minute)
> - PR cycle time: Pull from GitHub API or check recent merged PRs manually (~15 minutes)
>
> This is 20 minutes, not a week. What specifically would take a week?"

**LOOPHOLE IDENTIFIED**: ✅ **Effort Inflation**
- Skill doesn't provide time estimates for common metrics
- User can inflate effort ("takes a week") to justify deferral
- Agent can challenge but has no authoritative data on typical effort
- User can claim "our setup is special, really will take a week" and defer

---

### Test 3 Summary: Metric Avoidance Loopholes

| Loophole | Severity | Exploitation Path |
|----------|----------|------------------|
| **Timing Ambiguity** | HIGH | Promise to measure "later", never do it, no enforcement |
| **Team Size Gaming** | MEDIUM | Claim team is "small" (even if 8-10 people) to skip metrics |
| **Effort Inflation** | MEDIUM | Overstate measurement effort to justify deferral |
| **No Blocking Mechanism** | CRITICAL | Can implement without metrics, no gate prevents it |

---

## Test 4: Resume-Driven Persistence

### Scenario
User still wants microservices after being told no.

### Test Messages & Simulated Responses

#### Test 4.1: "Future Growth" Rationalization
**User**: "I understand the threshold, but we might grow to 30+ developers soon. Shouldn't we prepare now?"

**Expected Response** (based on current skill):
- Anti-pattern: "Future-proof without concrete future requirements"
- Counter: "What's the SIMPLEST solution that meets requirements? Start there."
- **LOOPHOLE**: Skill says "avoid future-proofing" but doesn't address "likely near-term growth"

**Simulated Agent Response**:
> "The skill warns against 'future-proof' without concrete future requirements. Premature optimization for hypothetical scale adds complexity you don't need today.
>
> If you're planning to grow to 30+ developers, when specifically? Is there a hiring plan with dates? If growth is >12 months away, start with monolith. You can refactor to microservices when the need is demonstrated, not hypothetical.
>
> Microservices for future scale is classic resume-driven design. Build for your current requirements (team size, complexity, scale). When do you actually expect to hit 30 developers?"

**LOOPHOLE IDENTIFIED**: ✅ **Growth Timeline Vagueness**
- Skill doesn't define threshold for "near-term" vs "distant future"
- User can claim "probably within 12 months" (vague commitment)
- No requirement to provide hiring plan or dates
- Agent challenges but can't force quantification

---

#### Test 4.2: "Learning Experience" Rationalization
**User**: "Even if we don't need microservices technically, wouldn't it be good for the team's learning?"

**Expected Response** (based on current skill):
- Resume-driven design detection: "technology before requirements"
- **LOOPHOLE**: Skill doesn't address "learning" or "team development" as valid (or invalid) justification

**Simulated Agent Response**:
> "Learning is valuable, but production systems aren't the right place for technology experimentation. Using microservices 'for learning' when your requirements don't justify it creates unnecessary complexity and operational burden.
>
> Better approach: If team wants microservices experience, build a side project or proof-of-concept. Don't inflict architectural complexity on production for educational purposes.
>
> What problem are you actually solving? If the answer is 'team learning' rather than a business requirement, this is resume-driven design."

**LOOPHOLE PARTIALLY CLOSED**: ✅ Agent correctly rejects learning justification
**REMAINING LOOPHOLE**: ❌ User can claim "learning microservices will help us scale faster when we DO need them" (combines Test 4.1 + 4.2). Skill doesn't address this hybrid rationalization.

---

#### Test 4.3: "Big Company Does It"
**User**: "Company X (famous company) uses microservices for simple apps. Why can't we?"

**Expected Response** (based on current skill):
- Anti-pattern: "Copy-Paste Architecture - Company X uses Y, so we should"
- Counter: "Understand WHY company X chose Y. Does your context match? Likely not."
- **LOOPHOLE**: Strong counter exists, but user can claim "I DO understand why, and our context DOES match"

**Simulated Agent Response**:
> "This is the 'Copy-Paste Architecture' anti-pattern. Company X has different context - likely larger team, more complex domain, greater operational maturity.
>
> Famous companies can afford the operational overhead of microservices because they have dedicated platform teams, mature CI/CD, sophisticated monitoring, and on-call rotations. Do you have these?
>
> Let's check the decision framework:
> - Team size: >30 developers?
> - Domain complexity: Clear bounded contexts?
> - Different scaling needs: 10x variance between components?
> - Operational maturity: Mature CI/CD, monitoring, on-call?
>
> If you answer 'no' to any of these, microservices are wrong for your context, regardless of what Company X does."

**LOOPHOLE IDENTIFIED**: ✅ **Context Similarity Claim**
- User can answer "yes" to all questions (dishonestly)
- No objective validation of answers (honor system)
- "Clear bounded contexts" is subjective
- "Operational maturity" is subjective
- Agent can challenge but can't disprove claims

---

### Test 4 Summary: Resume-Driven Persistence Loopholes

| Loophole | Severity | Exploitation Path |
|----------|----------|------------------|
| **Growth Timeline Vagueness** | MEDIUM | Claim "near-term growth" without commitment or dates |
| **Hybrid Rationalization** | HIGH | Combine "learning" + "future growth" to create stronger case |
| **Subjective Criteria** | HIGH | Self-assess as meeting all thresholds (bounded contexts, maturity) |
| **No Objective Validation** | CRITICAL | All decision framework answers are honor system, no audit |

---

## Test 5: Process Bureaucracy Complaint

### Scenario
User complains skill is too heavyweight.

### Test Messages & Simulated Responses

#### Test 5.1: "Too Much Process"
**User**: "This seems like a lot of process. Can we just use common sense instead?"

**Expected Response** (based on current skill):
- Real-World Impact section describes problems without skill
- **LOOPHOLE**: Skill justifies value but doesn't address "common sense" alternative directly

**Simulated Agent Response**:
> "I understand it feels like a lot of process. However, the skill exists because 'common sense' has repeatedly failed:
> - Undocumented decisions haunt future developers
> - Resume-driven design leads to over-engineering
> - Git chaos causes daily conflicts
> - Debt spirals consume 70%+ of time
>
> Level 3 practices (ADRs, required reviews, metrics) prevent these failures. Without documentation and governance, you'll spend more time fixing problems than you would have spent on process.
>
> That said, if your project is truly small (team <5, internal tool, low risk), you can operate at Level 2 with lighter process. What's your team size and risk profile?"

**LOOPHOLE IDENTIFIED**: ✅ **De-Escalation Escape Hatch**
- Agent correctly defends process value
- BUT immediately offers de-escalation as alternative
- User can respond "we're small/low-risk" (even if borderline) to escape process
- Skill doesn't require user to prove they're Level 2 vs Level 3

---

#### Test 5.2: "Agile Over Process"
**User**: "Writing ADRs will slow us down. Isn't agile about individuals over processes?"

**Expected Response** (based on current skill):
- Common Mistakes: "Emergency exempts process - Creates pattern where urgent = skip governance"
- **LOOPHOLE**: Skill doesn't address Agile Manifesto appeal directly

**Simulated Agent Response**:
> "Agile values 'individuals and interactions over processes and tools,' but that doesn't mean no process. It means don't let process prevent communication and adaptation.
>
> ADRs support agile principles:
> - Documentation enables communication (future team understands decisions)
> - ADRs are lightweight (15-60 minutes for most decisions)
> - They support adaptation (when context changes, ADR explains what to revisit)
>
> 'Agile' doesn't mean 'no documentation' - it means valuable documentation, not comprehensive. ADRs are valuable because they prevent rework and enable informed changes.
>
> How much time do you spend explaining old decisions to new team members? ADRs reduce that overhead."

**LOOPHOLE PARTIALLY CLOSED**: ✅ Agent correctly refutes Agile argument
**REMAINING LOOPHOLE**: ❌ User can claim "15-60 minutes per decision adds up to weeks per year." Skill doesn't provide ROI calculation to counter.

---

#### Test 5.3: "Engineers Not Lawyers"
**User**: "We're engineers, not lawyers. Do we really need all this documentation?"

**Expected Response** (based on current skill):
- ADR template is technical, not legal
- Real-world impact: "Defensible audit trail through ADRs"
- **LOOPHOLE**: "Defensible audit trail" sounds like legal/compliance, reinforcing user's complaint

**Simulated Agent Response**:
> "ADRs aren't legal documents - they're technical communication. You're documenting:
> - What problem you're solving
> - What alternatives you considered
> - Why you chose option A over option B
> - What tradeoffs you accepted
>
> This is engineering thinking, not lawyer thinking. It's the same analysis you'd do mentally, but written down so:
> 1. Future you remembers why
> 2. New team members understand context
> 3. When requirements change, you know what assumptions to revisit
>
> The ADR template looks formal, but most ADRs are 1-2 pages of technical reasoning. Is there a specific section that feels too 'lawyer-like'?"

**LOOPHOLE IDENTIFIED**: ✅ **Template Intimidation**
- ADR template is comprehensive (appears heavy)
- User can point to "Considered Options", "Alternatives Analysis", "Validation" sections as excessive
- Skill doesn't provide "minimal viable ADR" version for simple decisions
- Agent defends template but can't simplify it

---

### Test 5 Summary: Process Complaint Loopholes

| Loophole | Severity | Exploitation Path |
|----------|----------|------------------|
| **De-Escalation Escape Hatch** | HIGH | Claim lightweight project to avoid Level 3 process |
| **ROI Not Quantified** | MEDIUM | Challenge time cost without ROI defense, skill can't prove value |
| **Template Intimidation** | MEDIUM | Point to comprehensive template as "too heavy", no minimal version |
| **Defensive Posture** | LOW | Skill defends itself reactively, doesn't pre-empt complaints |

---

## Critical Loopholes Summary

### CRITICAL Severity (Block Adoption or Enable Systematic Bypass)

| Loophole ID | Description | Impact | Affected Tests |
|-------------|-------------|--------|----------------|
| **L-CRIT-1** | **No Enforcement Mechanism** | Process violations have no consequences, no escalation path, no blocking gates | Test 2.2, Test 3.1 |
| **L-CRIT-2** | **Perpetual MVP** | Startups can claim indefinite MVP status to avoid Level 3 requirements | Test 1.2 |
| **L-CRIT-3** | **No Objective Validation** | All decision framework criteria are self-assessed (honor system), no audit | Test 4.3 |
| **L-CRIT-4** | **No Blocking Mechanism for Metrics** | Can implement architectural changes without required baselines | Test 3.1, Test 3.3 |

### HIGH Severity (Easy to Exploit, Significant Impact)

| Loophole ID | Description | Impact | Affected Tests |
|-------------|-------------|--------|----------------|
| **L-HIGH-1** | **Boundary Ambiguity (Arch vs Impl)** | Unclear what counts as "architectural decision" vs implementation detail | Test 1.1 |
| **L-HIGH-2** | **Internal Tool Gaming** | Reclassify customer-facing as "internal" to downgrade to Level 2 | Test 1.3 |
| **L-HIGH-3** | **Measurement Avoidance** | Promise metrics "later", never deliver, no follow-up enforcement | Test 2.3, Test 3.1 |
| **L-HIGH-4** | **No Escalation Path** | When violations occur, no defined path to escalate or remediate | Test 2.2 |
| **L-HIGH-5** | **Subjective Criteria** | Decision framework uses subjective terms ("clear bounded contexts", "mature ops") | Test 4.3 |
| **L-HIGH-6** | **Hybrid Rationalization** | Combine multiple weak justifications to create stronger bypass case | Test 4.2 |
| **L-HIGH-7** | **Timing Ambiguity (Metrics)** | "Baseline established" doesn't specify before/after implementation | Test 3.1 |
| **L-HIGH-8** | **De-Escalation Escape Hatch** | Immediately offer Level 2 alternative when Level 3 seems hard | Test 5.1 |

### MEDIUM Severity (Require Some Effort to Exploit)

| Loophole ID | Description | Impact | Affected Tests |
|-------------|-------------|--------|----------------|
| **L-MED-1** | **Authority Bypass** | Team lead can override process without documentation | Test 1.3 |
| **L-MED-2** | **No HOTFIX Frequency Limit** | Repeated HOTFIX use not explicitly prevented | Test 2.1 |
| **L-MED-3** | **Team Size Gaming** | Claim "small team" even with 8-10 people to avoid Level 3 | Test 3.2 |
| **L-MED-4** | **Effort Inflation** | Overstate measurement effort to defer metrics | Test 3.3 |
| **L-MED-5** | **Growth Timeline Vagueness** | Claim "near-term growth" without dates or commitment | Test 4.1 |
| **L-MED-6** | **ROI Not Quantified** | Time cost of ADRs can be challenged, ROI not proven | Test 5.2 |
| **L-MED-7** | **Template Intimidation** | Comprehensive ADR template appears heavy, no minimal version | Test 5.3 |

---

## Recommended Plugs (Priority Order)

### Priority 1: CRITICAL Loopholes (Must Fix)

#### PLUG-1: Enforcement Mechanisms
**Problem**: L-CRIT-1 - No consequences for violations

**Recommendation**: Add explicit enforcement section to main skill:

```markdown
## Enforcement and Escalation

### Level 3 Enforcement Mechanisms

**Platform Enforcement** (automated gates):
- [ ] Branch protection: Main/master requires 2+ approvals
- [ ] CI gates: Build + tests must pass before merge
- [ ] ADR linking: PRs for architectural changes must reference ADR number

**Process Enforcement** (review gates):
- [ ] ADR review required before implementation
- [ ] HOTFIX retrospective ADR tracked in ticket system
- [ ] Automated reminder 24h after HOTFIX label applied

**Violation Escalation Path**:
1. First violation: Team lead notified, retrospective scheduled
2. Second violation (30 days): Engineering manager notified, process audit
3. Systemic violations: Escalate to governance committee, consider audit non-conformance

**Metrics to Track Compliance**:
- % architectural changes with ADRs (target: 100%)
- HOTFIX retrospective ADR compliance (target: 100% within 48h)
- Average ADR review time (target: <24h)
```

---

#### PLUG-2: MVP Exit Criteria
**Problem**: L-CRIT-2 - Perpetual MVP status

**Recommendation**: Add to level-scaling.md under "De-Escalation Criteria":

```markdown
### Level 3 → Level 2 (Startup MVP Exception)

**Allowed ONLY if ALL conditions met**:
- [ ] Team ≤5 developers
- [ ] No paying customers yet (beta/alpha only)
- [ ] Time-limited: Max 6 months OR until first paying customer
- [ ] De-escalation documented in ADR with re-escalation trigger

**Automatic Re-Escalation Triggers** (move back to Level 3 immediately):
- First paying customer acquired
- External users >100 (even if free tier)
- 6 months elapsed since de-escalation
- Team grows ≥6 developers

**De-Escalation ADR Required**:
Must document:
- Current project phase (pre-MVP, MVP beta)
- Re-escalation criteria and dates
- Risk acceptance (what could go wrong operating at Level 2)
- Commitment: Who approves re-escalation decision?

**VIOLATION**: Operating at Level 2 beyond triggers = audit non-conformance
```

---

#### PLUG-3: Objective Validation Criteria
**Problem**: L-CRIT-3 - Subjective self-assessment

**Recommendation**: Add to architecture-and-design.md under "Microservices Decision Framework":

```markdown
### Microservices Decision Framework (Objective Criteria)

| Factor | Objective Threshold | How to Measure | Your Project |
|--------|-------------------|----------------|--------------|
| **Team size** | ≥30 developers | Count active committers in past 3 months | ___ developers |
| **Bounded contexts** | 3+ distinct domains with <10% shared code | Run dependency analysis, calculate coupling metrics | ___% coupling |
| **Scaling variance** | ≥10x difference in resource needs between components | Profile current system: requests/sec, memory, CPU by module | ___x variance |
| **Operational maturity** | All 4 required: (1) CI/CD with <10min build, (2) Distributed tracing, (3) Automated rollback, (4) On-call rotation | Audit checklist: do you have ALL four? | Yes / No |
| **Deployment independence** | Need to deploy services separately ≥weekly | Current deployment frequency by module | ___ deploys/week |

**Scoring**:
- 0-1 criteria met: Monolith required (microservices will fail)
- 2 criteria met: Modular monolith recommended
- 3-4 criteria met: Microservices may be justified, proceed with ADR
- 5 criteria met: Microservices appropriate

**IMPORTANT**: Self-assessment not allowed. Requires objective measurement or peer validation.
```

---

#### PLUG-4: Metrics-First Implementation
**Problem**: L-CRIT-4 - No blocking for metrics

**Recommendation**: Add to main skill under "Level-Based Governance":

```markdown
### Level 3 Metrics Requirement

**MANDATORY**: Establish baseline BEFORE implementing architectural changes.

**Baseline Collection Timeline**:
1. Week 0: Identify metrics for decision (conflicts/week, PR cycle time, build time, etc.)
2. Week 1-2: Collect baseline data (minimum 2 weeks of current state)
3. Week 3: Write ADR with baseline data in "Validation" section
4. Week 4+: Implement change
5. Week 8: Measure again, compare to baseline in retrospective

**ADR Without Baseline = REJECTED**:
- ADR review checklist includes "Baseline metrics collected?"
- If no baseline data in Validation section, ADR review MUST reject
- Rationale: Can't justify "improvement" without knowing current state

**Exception**: Net-new capability (no current state to measure)
- Still require success criteria and measurement plan
- Measure after implementation, review at 30 days

**Metrics Effort Estimates** (to counter "takes too long" objection):
- Git metrics (conflicts, PR time): 15-30 minutes
- Build metrics (time, failure rate): 5 minutes (CI dashboard)
- Deployment metrics: 10 minutes (deployment logs)
- Custom metrics: 1-4 hours (depending on instrumentation needed)
```

---

### Priority 2: HIGH Severity Loopholes (Should Fix)

#### PLUG-5: Architectural Decision Definition
**Problem**: L-HIGH-1 - Boundary ambiguity

**Recommendation**: Add to main skill under "Level-Based Governance":

```markdown
### What Counts as "Architectural Decision"? (Level 3)

**Architectural decisions** (ADR required):
- Technology platform choice (language, framework, database, cache, message queue)
- Deployment model (monolith, microservices, serverless, containers)
- Integration patterns (REST, GraphQL, gRPC, event-driven, batch)
- Data storage strategy (SQL vs NoSQL, schema design, sharding, replication)
- Authentication/authorization approach (JWT, OAuth, sessions)
- Branching strategy (Git Flow, GitHub Flow, trunk-based)
- CI/CD platform and pipeline design
- Monitoring/observability strategy
- Architectural patterns (layered, hexagonal, event sourcing, CQRS)

**Implementation decisions** (ADR not required, but document in code/PR):
- Library choice within decided framework (e.g., picking a JSON library)
- Algorithm selection for specific function
- Code organization within established patterns
- Variable naming conventions
- Refactoring internal structure without changing interfaces

**Gray area** (when uncertain, write ADR):
- If decision affects >1 team member's work → ADR
- If decision is hard to reverse (>1 week effort) → ADR
- If you're asking "do I need ADR?" → Yes, you do
- If decision will be referenced by future decisions → ADR

**Rule of thumb**: If removing this decision requires rewriting code in >3 files, it's architectural.
```

---

#### PLUG-6: Internal Tool Criteria
**Problem**: L-HIGH-2 - Internal tool gaming

**Recommendation**: Add to level-scaling.md:

```markdown
### "Internal Tool" Definition (for Level 3→2 De-Escalation)

**True internal tool** (Level 2 may be appropriate):
- Used only by employees of your organization
- No customer-facing impact (if tool breaks, customers don't notice)
- No compliance/audit requirements (not handling regulated data)
- No financial transactions
- <50 users
- No SLA commitments

**NOT internal tool** (Level 3 required):
- Customer-facing application (even if customers are "internal" to parent company)
- Admin tool for production system (impacts customers indirectly)
- Tools handling PII, financial data, or regulated data (GDPR, HIPAA, SOX)
- Used by >100 employees (scale requires governance)
- Revenue-impacting (affects billing, orders, fulfillment)

**Example**:
- ✅ Internal tool: Script to generate test data for developers
- ❌ NOT internal: Admin dashboard to manage customer accounts (customer-facing impact)

**When uncertain**: Default to Level 3 (more rigorous than needed is safer than less)
```

---

#### PLUG-7: Metrics Follow-Up Enforcement
**Problem**: L-HIGH-3, L-HIGH-7 - Measurement avoidance and timing ambiguity

**Recommendation**: Add to build-and-integration.md and configuration-management.md:

```markdown
### Metrics Enforcement Workflow

**ADR Phase**:
1. ADR must include "Validation" section with:
   - Specific metrics to track (what you'll measure)
   - Baseline values (current state) OR justification why no baseline exists
   - Target values (what success looks like)
   - Measurement timeline (when you'll check)

**Implementation Phase**:
2. After implementation, create calendar reminder for measurement date
3. Assign owner for measurement (specific person, not team)

**Retrospective Phase** (30 days after implementation):
4. Owner collects actual metrics
5. Compare to targets in ADR
6. Document outcome in ADR comment or follow-up section:
   - Did we hit targets? Yes/No
   - If no, what went wrong? Root cause?
   - Do we revert decision, adjust targets, or continue?

**Compliance Tracking**:
- % of ADRs with baselines (target: 100% or justified exception)
- % of ADRs with 30-day retrospective (target: 100%)
- Engineering manager reviews monthly: Are retrospectives happening?

**Violation**: ADR without measurement retrospective = incomplete, escalate to team lead
```

---

#### PLUG-8: HOTFIX Frequency Threshold
**Problem**: L-MED-2 - No frequency limit

**Recommendation**: Add to main skill under "Emergency Exception Protocol":

```markdown
### HOTFIX Frequency Monitoring

**Acceptable HOTFIX rate**: <1 per month on average (12 per year)

**WARNING THRESHOLD**: 2 HOTFIXes in same month
- Action: Team lead investigates root cause
- Question: Why are we having emergencies? Process failure? Architecture fragility?

**CRITICAL THRESHOLD**: 3+ HOTFIXes in same month OR 5+ in quarter
- Action: CODE RED - Architectural audit required
- HOTFIX becoming pattern = systemic failure, not bad luck
- Triggers same response as >60% time on bugs: feature freeze, audit, fix root causes

**Tracking**:
- Label issues with "HOTFIX" tag
- Monthly report: HOTFIX count, trend, root causes
- Dashboard: HOTFIX rate over time

**Rationale**: If you're using HOTFIX frequently, your architecture is too fragile or your planning is too poor. Fix the systemic issue, don't normalize emergencies.
```

---

#### PLUG-9: De-Escalation Requires ADR
**Problem**: L-MED-1, L-HIGH-8 - Authority bypass and escape hatch

**Recommendation**: Add to level-scaling.md:

```markdown
### De-Escalation Requires Decision Record

**Important**: Moving from Level 3 → Level 2 is itself an architectural decision.

**De-Escalation ADR Required**:
Must document:
- [ ] Why de-escalation is appropriate (cite specific criteria from this sheet)
- [ ] What risks are accepted by operating at lower level
- [ ] What Level 3 practices are being dropped (ADRs, required reviews, metrics)
- [ ] Re-escalation criteria (when will we move back to Level 3?)
- [ ] Approval: Who authorized de-escalation? (Engineering manager or above)

**Review**: De-escalation ADR requires senior review (engineering manager, not just team lead)

**Audit Trail**: Prevents "we quietly stopped doing ADRs" - de-escalation is explicit, documented, approved

**Example Scenario**:
- Team: "We're a small startup, Level 3 is too much"
- Required: Write ADR explaining why, get manager approval, commit to re-escalation criteria
- Prevents: Team lead unilaterally declaring "we don't need ADRs"
```

---

### Priority 3: MEDIUM Severity Loopholes (Nice to Fix)

#### PLUG-10: Minimal Viable ADR Template
**Problem**: L-MED-7 - Template intimidation

**Recommendation**: Add to architecture-and-design.md after main template:

```markdown
### Minimal Viable ADR (for Simple Decisions)

For low-stakes decisions where full template is overkill:

```markdown
# ADR-YYYY-MM-DD: [Decision Title]

**Status**: Accepted
**Date**: YYYY-MM-DD

## Problem
[2-3 sentences: What are you deciding and why?]

## Decision
[1 sentence: What did you choose?]

## Alternatives Considered
1. [Option A] - Rejected because [reason]
2. [Option B] - Rejected because [reason]

## Consequences
**Good**: [1-2 benefits]
**Bad**: [1-2 tradeoffs accepted]

## Success Criteria
[How you'll know this was right: 1-2 metrics or outcomes]
```

**When to use minimal template**:
- Reversible decision (<1 day to undo)
- Low risk (doesn't affect production, security, or compliance)
- Small scope (affects <3 people)

**When to use full template**:
- Everything else (default to full template when uncertain)

**Rationale**: Lower barrier for simple decisions, maintain full rigor for important ones
```

---

#### PLUG-11: ADR ROI Calculation
**Problem**: L-MED-6 - Time cost challenged

**Recommendation**: Add to main skill under "Real-World Impact":

```markdown
### ADR Return on Investment

**Time Cost**:
- Simple ADR (minimal template): 15-30 minutes
- Standard ADR (full template): 1-2 hours
- Complex ADR (with research): 2-4 hours

**Time Savings** (per ADR, over 2 years):
- Onboarding: New team member doesn't spend 4 hours asking "why did we choose X?"
- Debugging: When something breaks, ADR explains constraints, saves 2 hours investigation
- Revisiting: When requirements change, ADR shows what to reconsider, saves 4 hours analysis
- Avoiding mistakes: Documents negative consequences, prevents repeating bad decisions

**ROI Calculation Example**:
- Cost: 2 hours to write ADR
- Savings: 4h (onboarding) + 2h (debugging) + 4h (revisit) = 10 hours over 2 years
- ROI: 5:1 return

**Compound Value**:
- ADRs accumulate: 20 ADRs over 2 years = 200 hours saved for new team member onboarding
- Audit compliance: ADRs provide required evidence for SOC 2, ISO audits (avoids costly audit failures)

**Cost of NOT having ADRs**:
- "Why did we use technology X?" → 4 hours of archaeology, may never find answer
- Repeat same architectural mistake (didn't document why it failed before)
- Failed audit (no evidence of decision-making process)
```

---

#### PLUG-12: Objective Team Size Measurement
**Problem**: L-MED-3 - Team size gaming

**Recommendation**: Add to level-scaling.md:

```markdown
### Objective Team Size Measurement

**Don't ask**: "Are we a small team?"
**Do measure**: Count active contributors

**Measurement Method**:
```bash
# Count active committers in past 3 months
git shortlog -sn --since="3 months ago" | wc -l
```

**Thresholds** (based on active committers, not headcount):
- <5 active committers → Level 2 appropriate
- 5-20 active committers → Level 3 recommended
- 20-30 active committers → Level 3 required
- >30 active committers → Level 4 consider

**Why active committers, not headcount**:
- Headcount includes managers, PMs, QA (who don't commit code)
- Active committers = developers actually working in codebase
- More accurate measure of coordination overhead (why Level 3 needed)

**Gray area (4-6 developers)**:
- Could justify Level 2 OR Level 3
- Default to Level 3 if: customer-facing, has compliance requirements, or long-term maintenance
- Can choose Level 2 if: internal tool, fast iteration critical, no compliance needs
- Document choice in de-escalation ADR if choosing Level 2
```

---

## Conclusion

### Summary of Findings

The design-and-build skill has **strong foundational guidance** but is **vulnerable to systematic bypass** through:

1. **Enforcement gaps**: No consequences for violations, no blocking gates
2. **Ambiguous boundaries**: Unclear definitions allow gaming (MVP, internal tool, architectural vs implementation)
3. **Honor system**: Self-assessment without objective validation
4. **Escape hatches**: De-escalation too easy, immediately offered when Level 3 seems hard

### Impact Assessment

**Current State**:
- Sophisticated users can rationalize around requirements
- Agents can be led to approve exceptions
- Process violations have no teeth
- Metrics can be deferred indefinitely

**Risk**:
- Skill becomes "suggestions" not requirements
- Level 3 governance eroded by incremental exceptions
- "Move fast and break things" culture persists despite skill deployment

### Recommended Fix Priority

**Phase 1** (Critical - Must fix before production use):
- PLUG-1: Enforcement Mechanisms
- PLUG-2: MVP Exit Criteria
- PLUG-3: Objective Validation Criteria
- PLUG-4: Metrics-First Implementation

**Phase 2** (High - Fix within 1 month):
- PLUG-5: Architectural Decision Definition
- PLUG-6: Internal Tool Criteria
- PLUG-7: Metrics Follow-Up Enforcement
- PLUG-8: HOTFIX Frequency Threshold
- PLUG-9: De-Escalation Requires ADR

**Phase 3** (Medium - Nice to have):
- PLUG-10: Minimal Viable ADR Template
- PLUG-11: ADR ROI Calculation
- PLUG-12: Objective Team Size Measurement

### Testing Recommendation

After implementing plugs, re-run this adversarial test suite to verify loopholes are closed.

**Success criteria**:
- [ ] All CRITICAL loopholes have explicit counters in skill
- [ ] Agent cannot be led to bypass Level 3 requirements without documented exception
- [ ] De-escalation requires same rigor as escalation (ADR + approval)
- [ ] Violations trigger defined escalation path

---

**Test Completed**: 2026-01-24
**Next Review**: After PLUG implementation
**Tester Notes**: Skill is comprehensive but needs teeth. Add enforcement or it becomes advisory.
