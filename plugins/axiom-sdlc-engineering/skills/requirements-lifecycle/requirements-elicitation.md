---
parent-skill: requirements-lifecycle
reference-type: elicitation-techniques
load-when: Starting requirements gathering, stakeholder workshops, unclear requirements, new project kickoff
---

# Requirements Elicitation Reference

**Parent Skill:** requirements-lifecycle
**When to Use:** Gathering requirements from stakeholders, starting new features, unclear scope, project kickoff

This reference provides systematic techniques for eliciting requirements from stakeholders across CMMI Levels 2-4.

---

## Purpose & Context

**What this achieves**: Systematic requirements gathering from stakeholders using appropriate techniques for project context and maturity level.

**When to apply**:
- Starting new project or major feature
- Stakeholder needs unclear or conflicting
- Requirements volatility high (need to understand root needs)
- Multiple stakeholder groups with different perspectives

**Prerequisites**:
- Stakeholder access available
- Project charter or initial scope defined
- Authority to conduct elicitation activities

---

## CMMI Maturity Scaling

### Level 2: Managed (Elicitation)

**Minimal requirements elicitation**:

**Techniques**:
- Informal interviews (one-on-one, email, meetings)
- Review existing documentation
- Observe current processes (shadowing)

**Documentation**:
- Interview notes (bullet points, email summaries)
- Captured in issue tracker or shared document
- No formal elicitation plan required

**Stakeholder Identification**:
- Primary stakeholder known (product owner, customer rep)
- Ad-hoc identification of others as needed

**Effort**:
- 2-4 hours per feature elicitation
- 1-2 day kickoff for new projects

**Example**:
```
Feature: User password reset
Stakeholder: Product Owner (Sarah)
Method: 30-minute interview + email follow-up
Notes:
  - Users must reset via email link (no SMS)
  - Link expires in 24 hours
  - Send confirmation when reset succeeds
  - Security team already approved email-based approach
```

### Level 3: Defined (Elicitation)

**Organizational standards and formal techniques**:

**Techniques** (all Level 2 plus):
- Structured workshops (JAD sessions, design thinking)
- Prototyping for validation (mockups, wireframes)
- Facilitated stakeholder workshops (multi-party elicitation)
- Use case development with stakeholders

**Documentation**:
- Elicitation plan (who, what, when, techniques)
- Stakeholder register (all identified stakeholders, roles, interests)
- Workshop outputs (affinity diagrams, user story maps)
- Validation records (prototype feedback, sign-offs)

**Stakeholder Identification**:
- Systematic stakeholder analysis (power/interest matrix)
- RACI for requirements (who owns, approves, contributes)
- Documented in stakeholder register

**Effort**:
- 1-2 days workshop facilitation per major feature
- 3-5 day structured elicitation for new projects

**Example**:
```
Elicitation Plan: Payment Processing Feature

Stakeholders Identified:
- Finance VP (high power, high interest) - Owner
- Compliance Lead (high power, medium interest) - Approver
- Customer Service Rep (low power, high interest) - Contributor
- End Users (no power, high interest) - Contributor

Techniques:
- JAD Session #1 (Finance VP, Compliance) - 4 hours
- User Interviews (3 customer service reps) - 3 × 1 hour
- Clickable Prototype (10 end users) - Usability testing

Schedule:
- Week 1: JAD session, prototype design
- Week 2: User interviews, prototype refinement
- Week 3: Usability testing, requirements finalization
```

### Level 4: Quantitatively Managed (Elicitation)

**Metrics-driven elicitation** (all Level 3 plus):

**Techniques**:
- Data-driven elicitation (analyze usage metrics, defect patterns)
- A/B testing for requirement validation
- Quantitative user research (surveys, analytics)

**Metrics**:
- Requirements discovery rate (new requirements per elicitation hour)
- Stakeholder coverage (% of stakeholder groups consulted)
- Elicitation defect density (requirements found defective in later review)
- Validation accuracy (prototype feedback alignment with final implementation)

**Analysis**:
- Historical elicitation effort vs. requirements volatility
- Predictor: "Last 3 projects averaged 25 requirements from 2-day workshop ±5"

**Example**:
```
Elicitation Metrics Dashboard:

Project: Mobile App Redesign
Elicitation Effort: 40 hours across 3 workshops
Requirements Discovered: 63 (discovery rate: 1.58 req/hour)
Baseline: 1.2-1.8 req/hour (in control)

Stakeholder Coverage: 8/9 groups (89%)
Missing: Marketing team (low priority per power/interest)

Prototype Validation:
- 47 requirements validated with users (75%)
- 16 requirements not prototyped (backend-only)
- Validation defect rate: 3 requirements changed after feedback (6.4%)
- Historical baseline: 8% ±3% (in control)

Prediction: Based on discovery rate and historical volatility, expect 10-15 requirements to be added/modified in first sprint (24% change rate, within normal range).
```

---

## Implementation Guidance

### Interview Techniques

#### Structured Interviews (Level 3+)

**Preparation**:
1. Define interview objectives (what questions need answering)
2. Create question guide (open-ended + specific)
3. Schedule 60-90 minutes (45 min interview, 15 min buffer)
4. Share context with interviewee beforehand

**Question Types**:
- **Open-ended**: "Walk me through your current process for X"
- **Probing**: "Why does that step take 30 minutes?"
- **Scenario-based**: "What happens if Y condition occurs?"
- **Prioritization**: "If you could only have one feature, what would it be?"

**During Interview**:
- Record (with permission) or take notes
- Use active listening ("So what I'm hearing is...")
- Avoid leading questions ("You want feature X, right?" → "What features would help?")
- Probe pain points (current workarounds reveal needs)

**Post-Interview**:
- Transcribe notes within 24 hours (while fresh)
- Highlight explicit requirements ("I need X") and implicit needs ("We spend hours on Y" → automation need)
- Send summary to interviewee for validation

**Example Structured Interview Guide**:
```
Feature: Reporting Dashboard
Interviewee: Finance Manager
Duration: 60 minutes

1. Current State (15 min)
   - Walk me through how you generate monthly reports today
   - What tools/systems do you use?
   - How long does it take?

2. Pain Points (15 min)
   - What's most frustrating about the current process?
   - Where do errors occur?
   - What takes the most time?

3. Desired State (15 min)
   - If you could wave a magic wand, what would the ideal workflow look like?
   - What metrics must be included?
   - How often do you need to generate reports?

4. Prioritization (10 min)
   - Rank these potential features (pre-prepared list)
   - Must-haves vs. nice-to-haves

5. Constraints (5 min)
   - Any compliance/regulatory requirements?
   - Integration with existing systems?
   - Performance expectations (report generation time)?
```

### Workshop Facilitation (JAD Sessions)

**JAD (Joint Application Design)** - Multi-stakeholder collaborative requirements session

**When to use**:
- Multiple stakeholders with overlapping interests
- Need rapid requirements definition (2-4 days vs. weeks of interviews)
- Conflicting requirements likely (better to surface early)

**Preparation** (1-2 days):
1. Identify participants (6-12 people ideal, <15 max)
2. Define scope and objectives
3. Send pre-work (context docs, initial questions)
4. Reserve room with whiteboards/sticky notes

**Workshop Structure** (4-8 hours):

**Hour 1: Context Setting**
- Project goals and constraints
- Stakeholder introductions
- Ground rules (one conversation, all ideas valid, focus on needs not solutions)

**Hours 2-3: Requirements Discovery**
- Brainstorming (generate ideas, no filtering)
- Affinity grouping (cluster related ideas)
- Prioritization (dot voting, MoSCoW)

**Hours 3-4: Requirements Specification**
- Detail high-priority requirements
- Define acceptance criteria
- Identify dependencies and constraints

**Hour 4: Validation & Next Steps**
- Read back requirements for confirmation
- Assign owners for follow-up
- Schedule validation activities (prototypes, reviews)

**Facilitator Techniques**:
- **Parking lot**: Capture off-topic items for later
- **Timeboxing**: Limit discussions (5 min per topic, use timer)
- **Silent brainstorming**: Write ideas on sticky notes before discussing (prevents groupthink)
- **Round-robin**: Everyone contributes in turn (prevents dominant voices)

**Output**:
- Prioritized requirements list
- User story map or affinity diagram (photo of whiteboard)
- Action items with owners and dates

**Example JAD Agenda**:
```
Project: Order Management System Redesign
Participants: 8 (Sales VP, Warehouse Manager, 2 Customer Service Reps, IT Lead, UX Designer, 2 Developers)
Duration: 6 hours

9:00-9:30   Context & Goals (Product Owner presents)
9:30-10:00  Stakeholder Pain Points (round-robin, capture on sticky notes)
10:00-10:45 Affinity Grouping & Themes
10:45-11:00 Break
11:00-12:00 Silent Brainstorming: Potential Solutions
12:00-1:00  Lunch
1:00-2:00   Solution Prioritization (MoSCoW + dot voting)
2:00-3:00   Detailing Top 10 Requirements (breakout groups)
3:00-3:15   Break
3:15-4:00   Review & Validation (read back all requirements)
4:00-4:30   Next Steps & Owners

Expected Output: 40-60 requirements prioritized, top 10 detailed with acceptance criteria
```

### Prototyping for Validation

**Purpose**: Validate understanding before building (catch misunderstandings early)

**Level 2**: Optional, informal (hand-drawn sketches, verbal walkthroughs)
**Level 3**: Recommended for complex features (clickable mockups, wireframes)
**Level 4**: Data-driven (A/B testing prototypes, usability metrics)

**Fidelity Levels**:

| Fidelity | When to Use | Tools | Effort |
|----------|-------------|-------|--------|
| **Low** (sketches) | Early exploration, rapid iteration | Paper, whiteboard, Balsamiq | 1-2 hours |
| **Medium** (wireframes) | Workflow validation, layout confirmation | Figma, Sketch, PowerPoint | 4-8 hours |
| **High** (clickable) | User testing, final validation before build | Figma prototypes, InVision | 1-3 days |

**Validation Process**:
1. Create prototype (appropriate fidelity)
2. Define validation questions (Can users complete task X? Do they understand feature Y?)
3. Test with 5-10 users (80% of usability issues found with 5 users)
4. Capture feedback (notes, video recordings)
5. Update requirements based on findings

**Example Prototype Validation**:
```
Feature: Self-Service Password Reset
Prototype: Clickable Figma mockup (medium fidelity)
Validation Questions:
  1. Can users find the reset option? (findability)
  2. Do they understand the email verification step? (workflow clarity)
  3. What do they expect to happen after clicking reset link? (mental model)

Testing: 7 users (5 internal, 2 customers), 15 min each
Findings:
  - 6/7 found reset option within 10 seconds ✅
  - 3/7 confused by "check your email" message (wanted confirmation of email sent)
  - 5/7 expected immediate login after reset (not additional sign-in step)

Requirement Changes:
  - REQ-015: Add "Email sent to [address]" confirmation message
  - REQ-016: Auto-login user after successful password reset (remove redundant sign-in)
```

### Stakeholder Analysis

**Purpose**: Identify all stakeholders, understand their power and interest, determine engagement strategy.

**Power/Interest Matrix** (Level 3):

```
          High Power
              │
    Manage   │   Partner
    Closely  │   Closely
  ───────────┼───────────  High Interest
    Monitor  │   Keep
             │   Informed
              │
          Low Power
```

**Quadrants**:
- **High Power, High Interest** - Partner closely (key decision makers, frequent communication)
- **High Power, Low Interest** - Manage closely (keep satisfied, don't overload)
- **Low Power, High Interest** - Keep informed (contributors, regular updates)
- **Low Power, Low Interest** - Monitor (minimal effort, inform as needed)

**Example Stakeholder Register**:
```
Project: Customer Portal Redesign

| Stakeholder | Role | Power | Interest | Strategy | Engagement |
|-------------|------|-------|----------|----------|------------|
| CEO | Sponsor | High | Medium | Manage Closely | Monthly exec briefing |
| VP Customer Service | Owner | High | High | Partner Closely | Weekly meetings, JAD workshops |
| IT Security Lead | Approver | High | Medium | Manage Closely | Requirements review, security sign-off |
| Customer Service Reps (5) | End Users | Low | High | Keep Informed | User interviews, prototype testing |
| Marketing Manager | Contributor | Medium | Medium | Keep Informed | Bi-weekly updates, content review |
| External Customers (pool) | End Users | Low | High | Keep Informed | Beta testing, surveys |
```

**RACI for Requirements** (Who owns what):
- **Responsible**: Who does the work (requirements analyst, product owner)
- **Accountable**: Who approves (single decision maker - VP, sponsor)
- **Consulted**: Who provides input (subject matter experts, end users)
- **Informed**: Who needs to know (stakeholders, management)

---

## Common Anti-Patterns

### Requirements Telephone Game

**Symptom**: Requirements passed through multiple people, meaning distorted

**Example**:
```
CEO → VP → Manager → Analyst → Developer
"Improve customer satisfaction" → "Faster checkout" → "One-click payment" → "Integrate PayPal" → "PayPal API v2"

Original intent (satisfaction) lost → Built wrong feature (PayPal when customers wanted saved cards)
```

**Impact**: Misunderstood needs, rework, stakeholder dissatisfaction

**Solution**:
- Direct stakeholder access (analyst talks to CEO, not through chain)
- Validate understanding (read back requirements to original source)
- Prototypes to confirm interpretation

### Gold Plating During Elicitation

**Symptom**: Elicitation session generates 200 requirements, team overwhelmed

**Example**: JAD session for "reporting dashboard" produces 47 report types, 23 chart formats, 15 export options

**Impact**: Analysis paralysis, scope creep, never shipping

**Solution**:
- Timebox elicitation (4-hour workshop max, then prioritize)
- Separate "gather" from "prioritize" (first collect all ideas, then MoSCoW separately)
- MVP framing: "What's the minimum to solve the core problem?"

### Interview Leading

**Symptom**: Interviewer suggests answers, stakeholder agrees to avoid conflict

**Bad Question**: "You want the report in PDF format, right?"
**Stakeholder**: "Uh, sure" (actually wanted Excel, but didn't contradict)

**Impact**: Requirements don't match actual needs

**Solution**:
- Open-ended questions: "What format would work best for your workflow?"
- Probe deeper: "Why PDF?" → Stakeholder: "Actually, I need to manipulate data in Excel"

---

## Tool Integration

### GitHub

**Elicitation capture**:
- Create issues during/after interviews
- Label: `requirement`, `needs-elaboration`
- Link to elicitation session notes (in issue description or wiki)

**Example**:
```
Title: [Requirement] Email Confirmation After Password Reset
From: Interview with Customer Service Lead (2026-01-20)

As a user who just reset my password
I want to receive an email confirmation
So that I know the reset was successful

Acceptance Criteria:
- Email sent within 30 seconds of reset
- Email includes timestamp and IP address of reset
- Email provides link to account security settings

Source: Customer Service Lead mentioned users call support asking "did my password actually change?"
```

### Azure DevOps

**Elicitation capture**:
- Create work items (type: Requirement or User Story)
- Tag with elicitation session (e.g., "JAD-2026-01-15")
- Link to stakeholder (assign "Requested By" field)

**Queries**:
- "Show all requirements from JAD session X"
- "Show requirements by stakeholder"

---

## Verification & Validation

**Elicitation complete when**:

**Level 2**:
- ✅ Key stakeholder interviewed (product owner, customer rep)
- ✅ Requirements documented in issue tracker
- ✅ Stakeholder sign-off obtained (email, meeting minutes)

**Level 3**:
- ✅ All identified stakeholders consulted (per stakeholder register)
- ✅ Requirements peer-reviewed for clarity and completeness
- ✅ Prototype validated with end users (for UI-heavy features)
- ✅ Requirements traceability established (linked to elicitation sessions)

**Level 4**:
- ✅ Elicitation metrics within baselines (discovery rate, defect density)
- ✅ Stakeholder coverage >85%
- ✅ Validation defect rate <10% (requirements changed after prototype testing)

---

## Related Practices

**After elicitation, proceed to**:
- `requirements-analysis.md` - Prioritize and analyze elicited requirements
- `requirements-specification.md` - Document in appropriate format (user stories, formal specs)

**Cross-references**:
- `governance-and-risk.md` - Stakeholder management for conflicting requirements
- `level-scaling.md` - Determine appropriate elicitation formality for project

**Prescription reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.1.1 (Requirements Development - SP 1.1 Elicit Needs)
