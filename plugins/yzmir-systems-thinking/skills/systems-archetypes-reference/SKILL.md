---
name: systems-archetypes-reference
description: Use when recognizing recurring problem patterns, distinguishing between similar system behaviors, or choosing archetype-specific interventions - provides complete catalog of 10 system archetypes with structures, diagnostics, and software engineering examples for rapid pattern matching
---

# Systems Archetypes Reference

## Overview

**System archetypes are recurring structural patterns that produce characteristic behaviors.** Recognizing the archetype reveals the intervention strategy - you don't need to re-solve the problem, you can apply known solutions.

**Core principle:** Systems are governed by archetypal structures. The same 10 patterns appear across domains. Once you recognize the pattern, you know how to intervene.

**Required foundation:** Understanding of feedback loops, stocks/flows, and leverage points. See recognizing-system-patterns and leverage-points-mastery skills.

## The 10 System Archetypes

Quick reference table - detailed explanations follow:

| Archetype | Signature Pattern | Primary Loop | Key Intervention |
|-----------|-------------------|--------------|------------------|
| 1. Fixes that Fail | Solution works temporarily, then problem returns worse | Reinforcing (symptom relief → side effect → worse problem) | Address root cause, not symptom |
| 2. Shifting the Burden | Symptom relief prevents fundamental solution | Balancing (quick fix) overpowers balancing (real fix) | Make quick fix difficult or undesirable |
| 3. Escalation | Two parties each escalate responses to each other | Reinforcing (A→B→A) | Unilateral de-escalation or shared goal |
| 4. Success to the Successful | Winner gets more resources, creates brittleness | Reinforcing (success → resources → more success) | Level the playing field or diversify |
| 5. Tragedy of the Commons | Individual optimization degrades shared resource | Reinforcing (individual gain → commons depletion → less for all) | Regulate commons or create feedback |
| 6. Accidental Adversaries | Well-intentioned actions hurt each other | Reinforcing (A helps self, hurts B; B helps self, hurts A) | Align incentives or coordinate |
| 7. Drifting Goals | Standards erode gradually from complacency | Balancing (gap → lower standard rather than improve) | Make gap visible, fix standards |
| 8. Limits to Growth | Growth slows despite effort, hits ceiling | Balancing (growth → constraint → slow growth) | Remove constraint or shift focus |
| 9. Growth and Underinvestment | Growth creates need for capacity, underfunded | Reinforcing (growth → insufficient capacity → quality drops → growth slows) | Invest ahead of demand |
| 10. Eroding Goals (Pessimistic) | Standards lower in response to performance pressure | Reinforcing (pressure → lower standards → worse performance → more pressure) | Break cycle, re-establish standards |

---

## 1. Fixes that Fail

### Structure

```
Problem Symptom
      ↓
  Quick Fix Applied
      ↓
Symptom Relieved (temporarily)
      ↓
Unintended Side Effect
      ↓
Problem Returns Worse
      ↓
Apply More of Same Fix
      ↓
[REINFORCING LOOP - Gets Worse Over Time]
```

**Causal Loop Diagram:**
```
Problem --+--> Quick Fix --+--> Symptom Relief
   ^                             |
   |                             ↓
   +------o----- Unintended Side Effect (delay)

R: Fix amplifies problem via side effects
```

### Software Engineering Examples

**Database Performance**
- Problem: Slow queries
- Fix: Add indexes
- Works temporarily: Queries faster
- Side effect: Data grows, indexes can't keep up, worse than before
- Root cause unaddressed: Unbounded data growth, no archival

**Alert Fatigue**
- Problem: Missing incidents
- Fix: Add more alerts
- Works temporarily: Catch more issues
- Side effect: Alert fatigue, engineers ignore alerts
- Root cause unaddressed: Incident rate, system reliability

**Hiring for Velocity**
- Problem: Team too slow
- Fix: Hire more engineers
- Works temporarily: More hands
- Side effect: Onboarding burden, communication overhead, slower overall
- Root cause unaddressed: Process inefficiency, tech debt

### Diagnostic Questions

- Does the solution work at first, then stop working?
- Are you applying more of the same solution repeatedly?
- Is there a delay between fix and side effect appearing?
- Are side effects making the original problem worse?

**If YES to these:** Likely Fixes that Fail

### Intervention Strategy

**Level 3 (Goals):** Change goal from "relieve symptom" to "solve root cause"

**Level 6 (Information):** Make side effects visible early (before they dominate)

**Level 5 (Rules):** Prohibit applying the same fix more than twice without root cause analysis

**What NOT to do:**
- ❌ Apply more of the failing fix
- ❌ Ignore the side effects as "unrelated"
- ❌ Speed up the fix (makes side effects appear faster)

**What to DO:**
- ✅ Identify the root cause being masked
- ✅ Trace the path from fix → side effect → worsened problem
- ✅ Solve root cause OR accept living with symptom

---

## 2. Shifting the Burden

### Structure

```
                Problem Symptom
                      ↓
         ┌─────────Quick Fix (Path A)
         │             ↓
         │      Symptom Relieved
         │             ↓
         │      Side Effect: Fundamental
         │      Solution Never Pursued
         │             ↓
         └── Problem Returns → Quick Fix Again

     Fundamental Solution (Path B) ← Never taken
```

**Key difference from Fixes that Fail:** Two pathways compete - symptom relief vs. fundamental solution. Quick fix actively prevents fundamental solution by reducing pressure.

### Software Engineering Examples

**QA Team vs. Quality Culture**
- Symptom: Bugs in production
- Quick fix: Add QA team to catch bugs
- Fundamental: Teams build quality in
- Burden shift: Dev teams never learn quality practices, depend on QA
- Result: QA becomes bottleneck, teams can't ship without them

**Outsourcing vs. Skill Building**
- Symptom: Team lacks skill X
- Quick fix: Outsource or hire contractor
- Fundamental: Train existing team
- Burden shift: Team never gains capability, permanent dependency
- Result: Can't maintain what contractors build

**Framework vs. Understanding**
- Symptom: Complex problem
- Quick fix: Import framework/library
- Fundamental: Understand and solve directly
- Burden shift: Team never learns underlying concepts
- Result: Can't debug framework issues, framework lock-in

### Diagnostic Questions

- Is there a "quick fix" and a "fundamental solution" to the same problem?
- Does the quick fix reduce pressure to pursue fundamental solution?
- Is the team becoming dependent on the quick fix?
- Does the quick fix have ongoing costs (time, money, capability drain)?

**If YES:** Likely Shifting the Burden

### Intervention Strategy

**Level 5 (Rules):** Make quick fix expensive or inconvenient (force fundamental solution)

**Level 6 (Information):** Track total cost of quick fix over time, make dependency visible

**Level 3 (Goals):** Prioritize capability building over symptom relief

**What NOT to do:**
- ❌ Make quick fix easier/cheaper (strengthens burden shift)
- ❌ Remove fundamental solution resources (makes shift permanent)
- ❌ Accept "this is just how we work" (normalization of dependency)

**What to DO:**
- ✅ Simultaneously apply quick fix AND start fundamental solution
- ✅ Set sunset date for quick fix
- ✅ Measure capability growth, not just symptom relief

---

## 3. Escalation

### Structure

```
Party A's Action --+--> Threat to Party B
                            ↓
                    Party B's Response --+--> Threat to Party A
                            ↓                       ↓
                    More Aggressive Response      (loop continues)
                            ↓
                  [REINFORCING LOOP - Accelerating Conflict]
```

**Characteristic:** Both parties think they're being defensive, both are actually escalating.

### Software Engineering Examples

**Tech Debt vs. Feature Pressure**
- Party A (Management): Pressure to ship features faster
- Party B (Engineering): Take shortcuts, accumulate debt
- Escalation: Debt → slower velocity → more pressure → more shortcuts → more debt
- Result: Velocity approaches zero, both sides blame each other

**Security vs. Usability**
- Party A (Security): Add restrictions (2FA, password policies, access controls)
- Party B (Users): Find workarounds (shared passwords, written down, disabled 2FA)
- Escalation: Workarounds → more restrictions → more creative workarounds
- Result: Security theater, actual security compromised

**Performance Team vs. Feature Team**
- Party A (Features): Add features that slow system
- Party B (Performance): Add rules/gates that slow feature delivery
- Escalation: Slower features → pressure to bypass gates → worse performance → stricter gates
- Result: Gridlock, both teams frustrated

### Diagnostic Questions

- Are two parties each making the other's problem worse?
- Does each side think they're being defensive/reasonable?
- Is the conflict intensifying despite both sides "trying harder"?
- Would unilateral de-escalation feel like "giving up"?

**If YES:** Likely Escalation

### Intervention Strategy

**Level 3 (Goals):** Create shared goal that supersedes individual goals

**Level 6 (Information):** Make each party's actions visible to the other (break assumptions)

**Level 2 (Paradigm):** Shift from "zero-sum" to "collaborative" mindset

**What NOT to do:**
- ❌ Escalate further ("fight fire with fire")
- ❌ Blame one party (both are trapped in system)
- ❌ Split the difference (doesn't break the loop)

**What to DO:**
- ✅ Unilateral de-escalation by one party (breaks cycle)
- ✅ Create joint accountability (merge teams, shared metrics)
- ✅ Make escalation cost visible to both parties

---

## 4. Success to the Successful

### Structure

```
Team A's Success --+--> More Resources to Team A
                            ↓
                    Team A More Successful
                            ↓
                    Even More Resources to Team A

Team B Struggles ---o-> Fewer Resources to Team B
                            ↓
                    Team B Struggles More
                            ↓
                    Even Fewer Resources

[REINFORCING LOOP - Rich Get Richer, Poor Get Poorer]
```

**Result:** Concentration risk - over-dependence on "winner"

### Software Engineering Examples

**Enterprise vs. SMB Product**
- Winner: Enterprise team (big deals)
- Gets: Custom features, eng resources, exec attention
- Result: 90% revenue from 5 customers, SMB product dies
- Risk: One enterprise customer leaves = company crisis

**Popular Service Gets Resources**
- Winner: Service A (high traffic)
- Gets: More engineers, better infra, attention
- Result: Service A dominates, Service B atrophies
- Risk: Service B fails, takes down Service A (hidden dependency)

**Star Developer Effect**
- Winner: Senior dev who delivers fast
- Gets: Best projects, promotions, resources
- Result: Junior devs never get growth opportunities, team dependent on one person
- Risk: Star leaves = team collapses

### Diagnostic Questions

- Is one team/project/person getting disproportionate resources?
- Is the gap between "winner" and "loser" widening?
- Is the organization becoming dependent on the winner?
- Would the loser's failure create cascading problems?

**If YES:** Likely Success to the Successful

### Intervention Strategy

**Level 5 (Rules):** Resource allocation must consider portfolio balance, not just current ROI

**Level 6 (Information):** Make concentration risk visible (dependency graphs, customer concentration)

**Level 4 (Self-organization):** Let "losers" experiment with different approaches

**What NOT to do:**
- ❌ "Double down" on winners exclusively
- ❌ Let losers die without understanding systemic value
- ❌ Assume current success predicts future success

**What to DO:**
- ✅ Limit maximum resource allocation to any single entity (diversify)
- ✅ Invest in "losers" as strategic options (portfolio thinking)
- ✅ Rotate resources to prevent permanent advantage

---

## 5. Tragedy of the Commons

### Structure

```
Individual A Optimizes for Self --+--> Uses Shared Resource
Individual B Optimizes for Self --+--> Uses Shared Resource
Individual C Optimizes for Self --+--> Uses Shared Resource
                                            ↓
                                   Shared Resource Degrades
                                            ↓
                                   Less Available for All
                                            ↓
                                   Individuals Use MORE (scarcity response)
                                            ↓
                                   [REINFORCING LOOP - Accelerating Depletion]
```

### Software Engineering Examples

**Database Connection Pool**
- Shared resource: DB connections
- Individual optimization: Each service opens more connections for speed
- Result: Pool exhausted, all services slow
- Commons degraded: Database becomes bottleneck for everyone

**Production Deployment Windows**
- Shared resource: Production stability
- Individual optimization: Each team deploys whenever ready
- Result: Too many changes, hard to debug issues
- Commons degraded: Production unstable for all teams

**Shared Codebase Quality**
- Shared resource: Code maintainability
- Individual optimization: Each team ships fast without refactoring
- Result: Tech debt accumulates, codebase unmaintainable
- Commons degraded: Everyone slowed by poor code quality

### Diagnostic Questions

- Is there a shared resource that multiple parties use?
- Does each party optimize individually without considering others?
- Is the resource degrading over time despite (or because of) individual optimization?
- Would regulation/limits be resisted as "unfair" by individuals?

**If YES:** Likely Tragedy of the Commons

### Intervention Strategy

**Level 5 (Rules):** Regulate access to commons (quotas, rate limits, governance)

**Level 6 (Information):** Make individual impact on commons visible (usage dashboards)

**Level 3 (Goals):** Align individual goals with commons health (team incentives)

**What NOT to do:**
- ❌ Appeal to good behavior without enforcement
- ❌ Wait for commons to collapse before acting
- ❌ Blame individuals (system incentivizes this)

**What to DO:**
- ✅ Create feedback loop: usage → visible cost → self-regulation
- ✅ Privatize commons OR enforce collective management
- ✅ Charge for usage (make externalities internal)

---

## 6. Accidental Adversaries

### Structure

```
Party A Takes Action to Help Self
      ↓
Action Inadvertently Hurts Party B
      ↓
Party B Takes Action to Help Self
      ↓
Action Inadvertently Hurts Party A
      ↓
[REINFORCING LOOP - Mutual Harm Despite Good Intentions]
```

**Key difference from Escalation:** Not intentional conflict - each party solving own problem, unaware they're hurting the other.

### Software Engineering Examples

**API Rate Limiting**
- Party A (Platform): Add rate limits to protect servers
- Hurts B: Users hit limits, break integrations
- Party B (Users): Create multiple accounts to bypass limits
- Hurts A: More load, harder to detect abuse, stricter limits needed
- Result: Arms race, both worse off

**Microservices Boundaries**
- Party A (Team A): Optimizes their service, changes API frequently
- Hurts B: Team B's service breaks from API changes
- Party B (Team B): Adds defensive caching, duplicates data
- Hurts A: Team A can't deploy changes, data consistency issues
- Result: Tight coupling despite microservices

**Oncall Rotation**
- Party A (Oncall eng): Deploys quickly to reduce queue, incomplete testing
- Hurts B: Next oncall gets incidents from rushed deploy
- Party B (Next oncall): Adds deployment gates and approvals
- Hurts A: Original oncall's deploys now blocked, queue grows
- Result: Slower deploys, more incidents

### Diagnostic Questions

- Are two parties pursuing legitimate goals?
- Do their solutions inadvertently harm each other?
- Is neither party trying to cause harm?
- Is the relationship deteriorating despite good intentions?

**If YES:** Likely Accidental Adversaries

### Intervention Strategy

**Level 6 (Information):** Make impact visible - A sees how they hurt B, B sees how they hurt A

**Level 5 (Rules):** Coordinate actions (shared calendar, RFC process, communication protocols)

**Level 3 (Goals):** Create joint success metric that requires cooperation

**What NOT to do:**
- ❌ Blame either party (both acting rationally)
- ❌ Let them "work it out" without structural change
- ❌ Optimize one party at expense of other

**What to DO:**
- ✅ Joint planning sessions, shared visibility
- ✅ Align incentives (both rewarded for cooperation)
- ✅ Create shared ownership or merge teams

---

## 7. Drifting Goals (Complacency-Driven)

### Structure

```
Target Goal: 95%
Actual Performance: 94.8%
      ↓
Small Gap - "Close Enough"
      ↓
Lower Target to 94%
      ↓
Actual Drops to 93%
      ↓
Lower Target to 93% - "Be Realistic"
      ↓
[REINFORCING LOOP - Standards Erode Gradually]
```

**Key characteristic:** Driven by complacency, not necessity. Team CAN achieve target but chooses not to.

### Software Engineering Examples

**Test Coverage Erosion**
- Started: 90% coverage standard
- "Just this once": 70% for urgent feature
- New normal: 75% "is realistic"
- Current: 60%, bugs increasing
- Team accepts: "Given constraints, 60% is good"

**Code Review Standards**
- Started: 2 reviewers, thorough feedback
- Drift: 1 reviewer "to move faster"
- Current: Rubber-stamp reviews
- Result: Quality declined, but normalized

**Deployment Frequency**
- Started: Deploy daily
- Drift: Deploy weekly "to reduce risk"
- Current: Deploy monthly
- Result: Releases become risky big-bang events, confirming fear

### Diagnostic Questions

- Did standards start higher?
- Was there a gradual lowering over time?
- Are current standards justified by "being realistic" rather than necessity?
- Can the team achieve original standards with current resources?

**Critical test:** "If we gave team 2 more weeks, could they hit original target?"
- **If YES:** Drifting Goals (capability exists, will doesn't)
- **If NO:** Different archetype (resource constraint exists)

### Intervention Strategy

**Level 6 (Information - Highest leverage for this archetype):**
- Make drift visible: Historical trend chart, original vs. current standard
- Customer impact metrics tied to lowered standards
- Public commitment to original standard

**Level 3 (Goals):**
- Re-establish non-negotiable minimum standards
- Remove authority to lower standards without explicit approval
- Tie consequences to meeting original target

**Level 5 (Rules):**
- Automatic escalation when standards not met
- Blameless post-mortems for "what would this look like at 95%?"

**What NOT to do:**
- ❌ Accept "constraints" without evidence (often post-hoc justification)
- ❌ Add resources (no resource gap exists)
- ❌ Negotiate standards based on convenience

**What to DO:**
- ✅ Make gap painfully visible
- ✅ Celebrate meeting original standard, don't accept "close enough"
- ✅ Re-commit publicly to original goal

---

## 8. Limits to Growth

### Structure

```
Growth Action --+--> Success/Growth
                      ↓
                Growth Continues
                      ↓
                Hits Limiting Constraint
                      ↓
                Growth Slows Despite Effort
                      ↓
                More Effort → Still Can't Grow
                      ↓
                [BALANCING LOOP - Constraint Dominates]
```

**Characteristic:** Growth works until it doesn't. Constraint kicks in, effort becomes futile.

### Software Engineering Examples

**Traffic Growth Hits Infrastructure**
- Growth: User acquisition working, doubling every 6 months
- Constraint: Infrastructure can't scale fast enough
- Limit: At 180K users, app crashes under load
- Result: Growth stops, users churn, opportunity lost

**Team Growth Hits Communication Overhead**
- Growth: Hiring velocity high, team growing fast
- Constraint: Communication overhead grows exponentially (n² problem)
- Limit: Coordination cost exceeds productivity gain
- Result: Bigger team, slower delivery

**Feature Growth Hits Cognitive Load**
- Growth: Shipping features rapidly
- Constraint: User cognitive overload, can't find anything
- Limit: More features make product HARDER to use
- Result: User satisfaction drops despite more features

### Diagnostic Questions

- Was growth working well, then suddenly stopped?
- Are you applying more effort but seeing diminishing returns?
- Is there an identifiable constraint that wasn't a problem before?
- Does "trying harder" feel increasingly futile?

**If YES:** Likely Limits to Growth

### Intervention Strategy

**Level 10 (Structure - Highest leverage for this archetype):**
- Remove or redesign the constraint
- Examples: Rearchitect for scale, restructure team, simplify product

**Level 3 (Goals):**
- Change growth target to different dimension where constraint doesn't apply
- Example: Growth in user engagement instead of user count

**Level 11 (Buffers):**
- Anticipate constraint, build capacity BEFORE hitting limit

**What NOT to do:**
- ❌ Apply more growth effort (won't work, constraint dominates)
- ❌ Ignore constraint hoping it resolves itself
- ❌ Treat constraint as temporary obstacle

**What to DO:**
- ✅ Identify the limiting constraint explicitly
- ✅ Remove constraint OR pivot to different growth strategy
- ✅ Invest in constraint removal before restarting growth

---

## 9. Growth and Underinvestment

### Structure

```
Growth --+--> Demand Increases
              ↓
        Need for Capacity Investment
              ↓
        Underinvest (short-term thinking)
              ↓
        Quality/Performance Degrades
              ↓
        Growth Slows
              ↓
        "See? Didn't need that investment"
              ↓
        [REINFORCING LOOP - Self-Fulfilling Prophecy]
```

**Key difference from Limits to Growth:** Constraint is CREATED by underinvestment, not inherent.

### Software Engineering Examples

**Infrastructure Underinvestment**
- Growth: Traffic increasing
- Need: Scale infrastructure proactively
- Underinvest: "Wait until we need it"
- Result: Performance degrades → users leave → "see, didn't need more servers"
- Self-fulfilling: Underinvestment killed growth

**Technical Debt Underinvestment**
- Growth: Feature demand high
- Need: Pay down tech debt to maintain velocity
- Underinvest: "Features first, debt later"
- Result: Velocity drops → fewer features shipped → "see, we can ship with this debt"
- Self-fulfilling: Debt accumulation slowed growth

**Team Capability Underinvestment**
- Growth: Business expanding
- Need: Train team on new technologies
- Underinvest: "No time for training, ship features"
- Result: Quality drops → customers churn → "see, training wouldn't have helped"
- Self-fulfilling: Lack of training killed growth

### Diagnostic Questions

- Is there growth potential being unrealized?
- Was there a decision to delay investment?
- Did performance degrade, causing growth to slow?
- Is the slowdown being used to justify the underinvestment?

**Critical tell:** "We didn't need X after all" - but slowdown was CAUSED by not having X

### Intervention Strategy

**Level 3 (Goals):**
- Measure long-term capacity, not just short-term delivery
- Goal: "Sustainable growth" not "maximize short-term growth"

**Level 6 (Information):**
- Model growth scenarios with/without investment
- Make opportunity cost of underinvestment visible

**Level 5 (Rules):**
- Mandatory investment allocation (% of resources to capacity/capability)
- Investment cannot be deferred without explicit growth target reduction

**What NOT to do:**
- ❌ Defer investment "until we're sure we need it"
- ❌ Use growth slowdown to justify underinvestment
- ❌ Optimize for short-term metrics at expense of capacity

**What to DO:**
- ✅ Invest AHEAD of demand (leading indicator)
- ✅ Track capacity utilization, invest before hitting 80%
- ✅ Make investment non-negotiable part of growth strategy

---

## 10. Eroding Goals (Pressure-Driven)

### Structure

```
Performance Gap (can't meet target)
      ↓
Pressure to Improve
      ↓
Can't Improve (resource constrained)
      ↓
Lower Standards to "Be Realistic"
      ↓
Pressure Temporarily Reduced
      ↓
Performance Drops Further (no standards to meet)
      ↓
Lower Standards Again
      ↓
[REINFORCING LOOP - Death Spiral]
```

**Key difference from Drifting Goals (#7):** Driven by necessity, not complacency. Team CANNOT meet target with current resources.

### Software Engineering Examples

**Uptime SLA Erosion**
- Target: 95% uptime
- Reality: Team achieves 92-93%, burning out
- Pressure: Management demands 95%
- Can't achieve: Insufficient resources/tooling
- Lower standards: "92% is realistic given constraints"
- Result: Team delivers 89%, standards lowered again → death spiral

**Velocity Pressure**
- Target: 50 story points/sprint
- Reality: Team delivers 35, working overtime
- Pressure: "Try harder"
- Can't achieve: Structural bottlenecks
- Lower expectations: "35 is the new normal"
- Result: Team delivers 28, morale collapses

**Security Compliance**
- Target: Pass all security audits
- Reality: Team fixes 70% of issues
- Pressure: Must pass audit
- Can't achieve: Not enough security expertise
- Lower standards: Accept "known risks"
- Result: More issues next audit, lower bar again

### Diagnostic Questions

**Critical test:** "If we gave team 2 more weeks, could they hit original target?"
- **If NO:** Eroding Goals (structural constraint)
- **If YES:** Drifting Goals (capability exists)

**Other signs:**
- Is the team burning out trying to meet targets?
- Are resources insufficient for stated goals?
- Is lowering standards framed as "being realistic" given constraints?
- Is performance declining DESPITE effort increase?

### Intervention Strategy

**Level 5 (Rules - Force Honest Choice):**
- "Goals must match resources OR resources must match goals - pick one"
- Cannot demand outcomes without providing means
- Sustainable pace is non-negotiable

**Level 11 (Buffers):**
- Add slack/capacity to stop the death spiral
- Provide recovery time for burned-out team

**Level 2 (Paradigm Shift):**
- From: "Try harder" → "Performance is emergent from system capacity"
- From: "Pressure produces results" → "Burnout produces collapse"

**What NOT to do:**
- ❌ Just lower standards (doesn't address root cause)
- ❌ Add pressure (accelerates death spiral)
- ❌ Accept "try harder" as strategy

**What to DO:**
- ✅ Force explicit choice: Add resources OR lower goals (and own it)
- ✅ Make current gap between goals and resources visible
- ✅ Break the cycle with capacity addition or scope reduction

---

## Distinguishing Similar Archetypes

### Drifting Goals (#7) vs. Eroding Goals (#10)

**Both:** Standards lower over time
**Key difference:** WHY standards are lowered

| Dimension | Drifting Goals | Eroding Goals |
|-----------|----------------|---------------|
| **Driver** | Complacency | Resource pressure |
| **Team capability** | CAN achieve, chooses not to | CANNOT with current resources |
| **Diagnostic test** | "2 more weeks?" → YES | "2 more weeks?" → NO |
| **Pressure level** | Low, comfortable | High, burning out |
| **Justification** | "Close enough" | "Realistic given constraints" |
| **Intervention** | Make gap visible, recommit to standards | Add resources OR lower goals officially |

### Fixes that Fail (#1) vs. Shifting the Burden (#2)

**Both:** Symptomatic solution, problem returns
**Key difference:** Competing pathways

| Dimension | Fixes that Fail | Shifting the Burden |
|-----------|-----------------|---------------------|
| **Structure** | One pathway with side effects | Two pathways (quick vs fundamental) |
| **What happens** | Fix creates side effect that worsens problem | Fix prevents pursuit of real solution |
| **Dependency** | Not necessarily | Creates addiction to quick fix |
| **Example** | Adding alerts creates alert fatigue | QA team prevents quality culture |

### Escalation (#3) vs. Accidental Adversaries (#6)

**Both:** Two parties harming each other
**Key difference:** Intent

| Dimension | Escalation | Accidental Adversaries |
|-----------|-----------|------------------------|
| **Intent** | Deliberate response to threat | Solving own problem, unaware of harm |
| **Awareness** | Both know they're in conflict | Neither realizes they're hurting other |
| **Example** | Tech debt vs feature pressure (both aware) | Rate limits → multi-accounts (unaware impact) |

### Limits to Growth (#8) vs. Growth and Underinvestment (#9)

**Both:** Growth stops
**Key difference:** Source of constraint

| Dimension | Limits to Growth | Growth and Underinvestment |
|-----------|------------------|----------------------------|
| **Constraint** | Inherent limit (user cognitive load) | Created by underinvestment (infrastructure) |
| **Timing** | Hits suddenly | Degradation visible in advance |
| **Prevention** | Hard (inherent to system) | Easy (invest proactively) |

---

## Archetype Combinations

**Systems often exhibit multiple archetypes simultaneously.** Recognize the pattern:

### Example: Feature Factory Disaster

**Primary: Shifting the Burden**
- Quick fix: QA team catches bugs
- Fundamental: Dev teams build quality in
- Burden shifted: Devs never learn quality

**Secondary: Escalation**
- Management: Pressure to ship faster
- Engineering: Cut more corners
- Both escalate: More pressure ↔ worse quality

**Tertiary: Tragedy of the Commons**
- Commons: Codebase quality
- Individual optimization: Each team ships fast
- Commons degraded: Everyone slowed

**Intervention strategy for combinations:**
1. **Identify primary archetype** (drives the system)
2. **Address secondary archetypes** that reinforce primary
3. **Use highest-leverage intervention** that addresses multiple archetypes

Example: Level 2 (Paradigm shift) to "quality is built in, not inspected in" addresses all three.

---

## Quick Recognition Guide

**Start here when analyzing a problem:**

1. **Map the feedback loops** - Reinforcing or balancing?
2. **Identify the parties/stocks** - Who/what is involved?
3. **Check the signature patterns:**
   - Problem returns after fix? → Fixes that Fail (#1)
   - Symptom relief + fundamental solution ignored? → Shifting Burden (#2)
   - Two parties making it worse? → Escalation (#3) or Adversaries (#6)
   - Winner gets more resources? → Success to Successful (#4)
   - Shared resource degrading? → Tragedy of Commons (#5)
   - Standards lowering from complacency? → Drifting Goals (#7)
   - Standards lowering from pressure? → Eroding Goals (#10)
   - Growth stopped suddenly? → Limits to Growth (#8)
   - Growth stopped from underinvestment? → Growth/Underinvestment (#9)

4. **Use diagnostic questions** from each archetype section
5. **Check for archetype combinations** (multiple patterns may apply)

---

## Integration with Leverage Points

**Each archetype has characteristic high-leverage interventions:**

| Archetype | Highest-Leverage Intervention Level |
|-----------|--------------------------------------|
| Fixes that Fail | #3 (Goals) - Focus on root cause not symptom |
| Shifting the Burden | #5 (Rules) - Make quick fix expensive/difficult |
| Escalation | #3 (Goals) - Create shared goal |
| Success to Successful | #5 (Rules) - Regulate resource allocation |
| Tragedy of Commons | #6 (Information) + #5 (Rules) - Feedback + regulation |
| Accidental Adversaries | #6 (Information) - Make impact visible |
| Drifting Goals | #6 (Information) - Make drift visible |
| Limits to Growth | #10 (Structure) - Remove constraint |
| Growth/Underinvestment | #3 (Goals) - Measure long-term capacity |
| Eroding Goals | #5 (Rules) - Force resource/goal alignment |

**Pattern:** Most archetypes respond to Levels 3-6 (Goals, Rules, Information, Feedback)

---

## Red Flags - Rationalizations for Skipping Archetype Analysis

If you catch yourself saying ANY of these, STOP and identify the archetype first:

| Rationalization | Reality | Response |
|-----------------|---------|----------|
| "No time for archetype analysis in crisis" | 10 minutes of pattern matching saves weeks of wrong fixes | Crisis is EXACTLY when archetypes matter most - prevents accelerating the problem |
| "My situation is unique, doesn't fit neat categories" | Uniqueness is in details, not structure - archetypes describe feedback loops | Test archetype predictions - if they match, it's the same structure |
| "This fits multiple archetypes, any intervention works" | Multiple archetypes require identifying PRIMARY one first | Address dominant archetype first, then secondary reinforcing patterns |
| "Archetypes are too academic/theoretical for real engineering" | Every archetype has software examples from production systems | This is pattern recognition, not theory - pragmatic shortcut to solutions |
| "I already know the solution, archetype is overhead" | If solution is obvious, archetype confirms it in 2 minutes | Unknown solutions become obvious once archetype identified |
| "We need action, not analysis" | Wrong action makes crisis worse (see: Fixes that Fail, Escalation) | Archetype analysis IS action - it prevents implementing failed patterns |

**The pattern:** All rationalizations push you toward repeating known failure modes. The archetypes catalog exists because these patterns have been solved before.

**The meta-trap:** "We're unique" is itself predicted by several archetypes (Shifting the Burden creates belief that quick fix is necessary, Drifting Goals creates post-hoc justification for lowered standards).

## The Bottom Line

**Don't reinvent solutions to archetypal problems.**

The same patterns recur across systems. Recognize the archetype, apply the known intervention strategy, save time.

**The skill:** Pattern matching speed. Experienced systems thinkers recognize archetypes in minutes, know immediately where to intervene.

**The discipline:** Don't jump to solutions before identifying the archetype. Taking 15 minutes to recognize the pattern saves hours implementing the wrong fix.