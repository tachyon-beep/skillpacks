---
parent-skill: requirements-lifecycle
reference-type: change-control
load-when: Requirements changing frequently, volatility management, impact analysis, change approval
---

# Requirements Change Management Reference

**Parent Skill:** requirements-lifecycle
**When to Use:** Managing requirements changes systematically, volatility control, impact analysis

This reference provides change request processes and volatility management across CMMI Levels 2-4.

---

## Purpose & Context

**What this achieves**: Systematic control of requirements changes with impact analysis and approval workflows appropriate for project maturity level.

**When to apply**:
- Requirements changing during development
- Stakeholder requests mid-sprint additions
- Need to track requirements volatility
- Compliance requires change documentation

**Prerequisites**:
- Requirements baseline established (initial set approved)
- Traceability system in place (to identify affected artifacts)
- Approval authorities defined (who can approve changes)

---

## CMMI Maturity Scaling

### Level 2: Managed (Change Management)

**Lightweight change tracking**:

**Process**:
1. Stakeholder submits change request via issue comment or new issue
2. Product Owner reviews impact (1-paragraph assessment)
3. Product Owner approves/rejects based on capacity
4. Change documented in issue history

**Documentation**:
```
Change Request Example (GitHub Issue Comment):

@product-owner requesting change to REQ-AUTH-042 (SSO integration)

Requested Change: Add support for Google OAuth in addition to Microsoft SSO

Business Justification: 30% of users prefer Google accounts over Microsoft

Impact Assessment:
- Effort: +3 story points (Google OAuth SDK integration)
- Schedule: Can fit in current sprint (2 points capacity remaining)
- Testing: +2 test cases (Google auth flow, token validation)
- Documentation: Update user guide with Google login instructions

Approval: @product-owner approved 2026-01-20
Assigned to: Sprint 15
```

**Approval Authority**:
- Product Owner can approve changes within sprint capacity
- Changes exceeding capacity require team/stakeholder discussion

**Volatility Tracking** (minimal):
- Count changes per sprint (manual or query)
- Red flag: >30% requirements changed per sprint → stakeholder meeting required

**Effort**:
- 15-30 minutes per change request (assessment + approval)
- 1-2 changes per sprint is normal, 5+ indicates volatility

### Level 3: Defined (Change Management)

**Formal change request workflow**:

**Change Request Form Template**:
```
Change Request: CR-{YEAR}-{NUMBER}
Date Submitted: [YYYY-MM-DD]
Submitted By: [Name, Role]

Original Requirement:
- ID: REQ-AUTH-042
- Title: SSO Integration via Microsoft Azure AD
- Current Status: In Progress (Sprint 15)

Requested Change:
- Description: Add support for Google OAuth as alternative SSO provider
- Change Type: [Enhancement | Defect Correction | Scope Addition | Deletion]
- Priority: [Critical | High | Medium | Low]

Business Justification:
- Reason: 30% of target users (healthcare professionals) use Google Workspace
- Business Value: Increase user adoption from 70% to 95%
- Cost of Not Changing: Lose 30% of potential users, competitive disadvantage

Technical Impact Analysis:

Affected Requirements:
- REQ-AUTH-042: Modify to support multiple SSO providers (not just Azure AD)
- REQ-AUTH-003: User profile creation (add Google ID field)
- New: REQ-AUTH-047: Google OAuth integration

Affected Design:
- AUTH-DD-02: Authentication module design
  - Impact: Add Google OAuth strategy class
  - Effort: 1 day refactoring

Affected Implementation:
- Files: src/auth/sso-provider.ts, src/auth/google-oauth.ts (new)
- Effort: 3 story points (2 days development)

Affected Tests:
- Add: test/auth/google-oauth.spec.ts
- Modify: test/auth/sso-integration.spec.ts
- Effort: 2 story points (1.5 days)

Affected Documentation:
- User Guide: Add Google SSO instructions (Section 2.3)
- Admin Guide: Update SSO configuration steps
- Effort: 4 hours

Total Effort: 5.5 days
Schedule Impact: Delays Sprint 15 completion by 2 days OR defer to Sprint 16
Budget Impact: $4,400 (1 developer @ $800/day × 5.5 days)

Risk Assessment:
- Technical Risk: Low (Google OAuth SDK is mature, well-documented)
- Schedule Risk: Medium (adds 2 days to current sprint)
- Integration Risk: Low (OAuth standard protocol)
- Security Risk: Low (same security model as Azure AD)

Dependencies:
- Requires Google Cloud Console access for OAuth credentials
- Requires IT approval for third-party integration

Approval Workflow:
[ ] Product Owner: Approved/Rejected, Date: _______________
[ ] Tech Lead: Approved/Rejected, Date: _______________
[ ] QA Lead: Approved/Rejected, Date: _______________

Decision: [Approved | Rejected | Deferred]
Implementation Target: Sprint 16 (deferred from Sprint 15 due to capacity)

Notification Sent To:
- Development Team: [Date]
- QA Team: [Date]
- Documentation Team: [Date]
- Stakeholders: [Date]
```

**Approval Authority Matrix** (Level 3):

| Change Impact | Product Owner | Tech Lead | QA Lead | Stakeholder |
|---------------|---------------|-----------|---------|-------------|
| **Minor** (<1 day, no design change) | Approve | Inform | Inform | - |
| **Moderate** (1-5 days, design change) | Approve | Approve | Consult | Inform |
| **Major** (>5 days, architecture change) | Approve | Approve | Approve | Inform |
| **Scope Change** (new feature area) | Recommend | Recommend | Recommend | **Approve** |

**Formal Workflow**:
1. Stakeholder submits change request using template
2. Requirements analyst performs impact analysis (trace affected artifacts)
3. Multi-party review (Product Owner + Tech Lead + QA Lead)
4. Approval/rejection with documented rationale
5. If approved: Update requirements baseline, notify affected parties
6. If rejected: Document reason, communicate to requestor

**Volatility Tracking** (systematic):
- Track change requests in dedicated tracking category (label, work item type)
- Monthly volatility report: Added, Modified, Deleted requirements
- Red flag threshold: >20% change rate per sprint (Level 3)

**Effort**:
- 1-2 hours per change request (formal impact analysis)
- 30 minutes multi-party review meeting
- 15 minutes notification and communication

### Level 4: Quantitatively Managed (Change Management)

**Metrics-driven change control** (all Level 3 plus):

**Volatility Metrics Dashboard**:
```
Requirements Volatility Dashboard - Sprint 15

Total Requirements Baseline: 85
Changes This Sprint:
- Added: 3 new requirements (REQ-AUTH-047, REQ-NOTIF-023, REQ-REPORT-018)
- Modified: 5 existing requirements
- Deleted: 1 requirement (REQ-PROFILE-009 - obsolete)

Volatility Rate: (3 + 5 + 1) / 85 = 10.6%
Historical Baseline: 12% ± 4% (within control)

Change Category Breakdown:
- Scope additions: 3 (35%)
- Clarifications/refinements: 4 (44%)
- Defect corrections: 1 (11%)
- Deletions: 1 (11%)

Change Source Analysis:
- Customer feedback: 4 changes (44%)
- Internal discovery: 3 changes (33%)
- Compliance requirement: 2 changes (22%)

Impact Metrics:
- Average effort per change: 3.2 days (vs. baseline 2.5-4.0 days, within range)
- Schedule impact: +6 days cumulative (Sprint 15 will miss deadline by 1 day)
- Rework effort: 15% of total sprint capacity (within 20% threshold)

Predictive Analysis:
- Trend: Volatility decreasing (Sprint 12: 18%, Sprint 13: 15%, Sprint 14: 13%, Sprint 15: 10.6%)
- Prediction: If trend continues, expect <8% volatility by Sprint 18 (below L4 target)

Control Chart (Statistical Process Control):
```
Volatility %
   30% |
       |
   20% |                     UCL (Upper Control Limit)
       |    ×
   15% |       ×
       |  ×       ×
   10% |             ×    ◆  ← Sprint 15 (in control)
       | ────────────────── Mean (12%)
    5% |
       |
    0% |________________________
       12   13   14   15   16   Sprint

UCL: 20% (Mean + 2σ)
LCL: 4% (Mean - 2σ)
```
Status: IN CONTROL (Sprint 15 at 10.6%, between control limits)

Action Items:
- Monitor Sprint 16 for continued downward trend
- If volatility exceeds UCL (>20%), trigger root cause analysis
```

**Change Approval with Quantitative Criteria**:

```
Approval Criteria (Level 4):

Minor Change (Auto-approve if):
- Effort <0.5 days AND
- No design change AND
- Volatility <20% threshold

Moderate Change (Approval required if ANY):
- Effort 0.5-5 days OR
- Design change OR
- Volatility 10-20%

Major Change (Escalation required if ANY):
- Effort >5 days OR
- Architecture change OR
- Volatility >20% (exceeds threshold)

Automatic Rejection Triggers:
- Volatility >30% (requires stakeholder stabilization meeting first)
- Budget exceeded (no remaining contingency)
- Schedule critical path affected (release date at risk)
```

---

## Implementation Guidance

### Change Request Process (Step-by-Step)

**Level 2 Lightweight Process**:

```
Step 1: Stakeholder Submits Change
  Where: GitHub issue comment, Azure DevOps discussion, or email
  Format: Informal description of requested change

Step 2: Product Owner Reviews
  - Read request
  - Quick impact assessment (1 paragraph)
  - Check sprint capacity

Step 3: Decision
  If capacity available → Approve (add to sprint backlog)
  If no capacity → Reject OR defer to next sprint

Step 4: Document
  - Log approval/rejection in issue comment
  - If approved: Update issue status, assign to sprint

Step 5: Communicate
  - Notify team via sprint planning or standup
```

**Level 3 Formal Process**:

```
Step 1: Stakeholder Submits Change Request
  Where: Dedicated change request system (GitHub issue with template, Azure DevOps change request work item)
  Format: Formal change request template (see Level 3 section above)
  Required Fields: Original requirement, requested change, justification

Step 2: Impact Analysis
  Owner: Requirements analyst or Product Owner
  Duration: 1-2 hours
  Deliverable: Completed impact analysis section

  Analysis Steps:
  a. Identify affected requirements (use traceability matrix)
  b. Identify affected design documents
  c. Identify affected code modules (via traceability)
  d. Identify affected tests
  e. Estimate effort for each affected artifact
  f. Calculate total effort and schedule impact
  g. Assess risks

Step 3: Multi-Party Review
  Participants: Product Owner, Tech Lead, QA Lead
  Duration: 30 minutes
  Format: Review meeting or async review (comments)

  Review Focus:
  - Is impact analysis complete and accurate?
  - Is effort estimate reasonable?
  - Are risks acceptable?
  - Is justification compelling?

Step 4: Approval Decision
  Authority: Per approval matrix (see Level 3 section)
  Options: Approve, Reject, Defer, Request More Information

  Approval Required From:
  - Minor: Product Owner only
  - Moderate: Product Owner + Tech Lead
  - Major: Product Owner + Tech Lead + QA Lead
  - Scope Change: Stakeholder (final authority)

Step 5: Update Requirements Baseline
  If Approved:
  - Update affected requirements in tracking system
  - Increment version (if using formal version control)
  - Link change request to updated requirements
  - Update traceability matrix

Step 6: Notify Affected Parties
  - Development Team: Affected modules identified
  - QA Team: New/modified test cases required
  - Documentation Team: Updates needed
  - Stakeholders: Change approved, expected delivery date
```

### Impact Analysis Checklist

**Use this checklist to ensure complete impact analysis**:

```
Change Request Impact Analysis Checklist

Change ID: _______________  Analyst: _______________  Date: _______________

Requirements Impact:
[ ] Identified all requirements affected by this change
[ ] Determined if new requirements needed
[ ] Determined if existing requirements will be modified
[ ] Determined if any requirements will be deleted
[ ] Estimated effort for requirements updates

Design Impact:
[ ] Identified affected design documents
[ ] Determined if design changes required (minor refactor vs. major redesign)
[ ] Identified affected architectural components
[ ] Estimated design effort (analysis + documentation)

Implementation Impact:
[ ] Identified affected source code files/modules
[ ] Determined if new code needed or existing code modified
[ ] Estimated development effort (coding + unit testing)
[ ] Assessed technical risk (new technology, unfamiliar area)

Test Impact:
[ ] Identified affected test cases
[ ] Determined if new test cases needed
[ ] Determined if existing test cases will be modified/deleted
[ ] Estimated testing effort (test creation + execution + validation)
[ ] Identified regression testing scope

Documentation Impact:
[ ] Identified affected user documentation
[ ] Identified affected technical documentation
[ ] Estimated documentation effort

Schedule Impact:
[ ] Calculated total effort (requirements + design + implementation + test + docs)
[ ] Determined impact on current sprint/release
[ ] Identified critical path effects
[ ] Proposed implementation timeline

Budget Impact:
[ ] Calculated labor cost (effort × hourly rate)
[ ] Identified any additional costs (tools, licenses, infrastructure)
[ ] Verified budget availability

Risk Assessment:
[ ] Technical risk evaluated (feasibility, complexity)
[ ] Schedule risk evaluated (deadline impact)
[ ] Integration risk evaluated (third-party dependencies)
[ ] Security/compliance risk evaluated

Dependencies:
[ ] External dependencies identified (third-party approvals, resources)
[ ] Internal dependencies identified (other teams, systems)
[ ] Blockers identified and mitigation planned

Traceability Verification:
[ ] Used RTM to identify all downstream artifacts
[ ] Verified no orphaned artifacts will be created
[ ] Confirmed bidirectional links will be maintained

Total Estimated Effort: _______________ (developer-days)
Schedule Impact: _______________ (days delay OR sprint deferral)
Recommendation: [ ] Approve  [ ] Reject  [ ] Defer  [ ] More Info Needed
```

### Volatility Metrics and Thresholds

**Calculating Requirements Volatility**:

```
Volatility Rate = (Added + Modified + Deleted) / Total Baseline × 100%

Example (Sprint 15):
- Baseline: 85 requirements at sprint start
- Added: 3 new requirements
- Modified: 5 existing requirements
- Deleted: 1 requirement

Volatility = (3 + 5 + 1) / 85 = 10.6%
```

**Thresholds by Maturity Level**:

| Level | Green (Healthy) | Yellow (Warning) | Red (Action Required) |
|-------|-----------------|------------------|-----------------------|
| **Level 2** | <20% per sprint | 20-30% per sprint | >30% per sprint |
| **Level 3** | <15% per sprint | 15-20% per sprint | >20% per sprint |
| **Level 4** | <10% per sprint | 10-15% per sprint | >15% per sprint |

**Actions by Threshold**:

**Green Zone**: No action required, normal volatility
**Yellow Zone**:
- Level 2: Monitor closely, discuss at retrospective
- Level 3: Review change sources, identify patterns
- Level 4: Conduct root cause analysis, update control limits if sustained

**Red Zone**:
- Level 2: **Stakeholder stabilization meeting required** (Product Owner + key stakeholders)
- Level 3: **Formal volatility review** with action plan (reduce change rate, improve elicitation)
- Level 4: **Process intervention** (SPC out-of-control signal, systematic investigation)

**Stakeholder Stabilization Meeting Agenda**:
```
Purpose: Address excessive requirements volatility (>30% change rate)
Duration: 1-2 hours
Participants: Product Owner, key stakeholders, Tech Lead, Scrum Master/PM

Agenda:

1. Review Volatility Data (15 min)
   - Show volatility trend over last 3-5 sprints
   - Present change categories (scope additions, clarifications, defects)
   - Demonstrate schedule/budget impact

2. Root Cause Analysis (30 min)
   - Why are requirements changing so frequently?
   - Patterns: Same stakeholder? Same requirement area?
   - Underlying issues: Unclear initial requirements? Evolving understanding? External factors?

3. Stabilization Actions (30 min)
   - Option A: Freeze requirements for 1-2 sprints (only critical defects allowed)
   - Option B: Improved elicitation (conduct additional JAD sessions, prototyping)
   - Option C: Scope reduction (defer nice-to-haves, focus on Must Haves)
   - Option D: Accept volatility and extend timeline/budget

4. Decision & Commitment (15 min)
   - Stakeholders commit to selected action
   - Define success criteria (e.g., "reduce volatility to <20% within 2 sprints")
   - Schedule follow-up review
```

### Change Communication

**Who to notify when requirements change**:

```
Communication Matrix:

Change Type: Minor (cosmetic, <1 day effort)
Notify:
- ✅ Development Team: Assigned developer + team lead
- ⚠️ QA Team: If affects test cases
- ⚠️ Documentation: If user-facing change
- ❌ Stakeholders: Not required

Change Type: Moderate (1-5 days, design impact)
Notify:
- ✅ Development Team: All developers
- ✅ QA Team: QA lead + assigned testers
- ✅ Documentation Team: Tech writer
- ⚠️ Stakeholders: If changes user experience or timeline

Change Type: Major (>5 days, architecture impact)
Notify:
- ✅ Development Team: All developers + architects
- ✅ QA Team: All QA (regression testing required)
- ✅ Documentation Team: All writers
- ✅ Stakeholders: All affected stakeholders
- ✅ Management: Project manager, directors

Change Type: Scope Change (new feature area)
Notify:
- ✅ All teams (development, QA, docs, ops)
- ✅ All stakeholders
- ✅ Management + executives
- ✅ External parties if applicable (vendors, partners)
```

**Notification Template**:

```
Subject: [Change Notification] REQ-AUTH-042 - Google OAuth Support Added

Change Request: CR-2026-015
Original Requirement: REQ-AUTH-042 (SSO Integration via Microsoft Azure AD)
Change Type: Enhancement (Scope Addition)
Approval Date: 2026-01-20

What Changed:
- REQ-AUTH-042 modified to support multiple SSO providers (not just Azure AD)
- New requirement REQ-AUTH-047 added for Google OAuth integration
- REQ-AUTH-003 modified to include Google ID in user profile

Impact:
- Development: +3 story points (2 days effort)
- Testing: +2 test cases (1.5 days effort)
- Documentation: User guide update (4 hours)
- Schedule: Deferred to Sprint 16 (originally Sprint 15)

Why This Change:
- 30% of target users prefer Google accounts over Microsoft
- Competitive advantage (competitors support both)
- Minimal technical risk (OAuth standard protocol)

Action Required:
- Development Team: Review updated design (AUTH-DD-02) by 2026-01-22
- QA Team: Prepare test cases for Google OAuth flow
- Documentation Team: Update user guide Section 2.3

Questions: Contact @product-owner or @tech-lead

Approved By:
- Product Owner: Jane Doe (2026-01-20)
- Tech Lead: Tom Chen (2026-01-20)
- QA Lead: Sarah Johnson (2026-01-20)
```

---

## Common Anti-Patterns

### Uncontrolled Scope Creep

**Symptom**: Product owner verbally adds requirements mid-sprint with no approval or documentation

**Example**:
```
❌ BAD:
Sprint 15 Daily Standup
Product Owner: "Oh, and can you also add email notifications when SSO login fails?"
Developer: "Uh, sure, I'll squeeze it in"

Problem:
- No change request submitted
- No impact analysis (how much effort?)
- No approval workflow
- Not documented anywhere
- Team capacity unknown
```

**Fix**:
```
✅ GOOD:
Sprint 15 Daily Standup
Product Owner: "I'd like to add email notifications for failed SSO attempts"
Scrum Master: "Please submit a change request so we can assess impact"

[Product Owner creates CR-2026-016]
[Team reviews: 2 days effort, exceeds sprint capacity]
Decision: Defer to Sprint 16, add to backlog

Result: Documented, assessed, planned appropriately
```

### Change Request Bureaucracy

**Symptom**: 2-week approval process for fixing a typo in a label

**Example**:
```
❌ BAD (Over-Process):
Change: Fix button label from "Sumbit" to "Submit"
Process:
- Day 1: Submit formal change request (30 min)
- Day 2-3: Requirements analyst performs impact analysis (4 hours)
- Day 4: Multi-party review meeting (1 hour, 5 people)
- Day 5-10: Waiting for VP approval
- Day 11: Change approved
- Day 12: Implement (5 minutes)

Problem: 12 days and 6 hours of labor for a 5-minute fix
```

**Fix**:
```
✅ GOOD (Right-Sized Process):
Change: Fix button label typo "Sumbit" → "Submit"
Classification: Defect correction, trivial impact
Process:
- Developer creates issue, labels as "typo"
- Auto-assigned to Product Owner for review
- Product Owner approves via comment (2 min)
- Developer fixes immediately (5 min)

Total time: <1 hour, properly documented, not bureaucratic
```

**Guideline**: Match process rigor to change magnitude
- Typos, cosmetic fixes: Lightweight approval (Product Owner comment)
- Minor changes (<1 day): Standard process
- Major changes (>5 days): Full formal process with multi-party approval

### Missing Impact Analysis

**Symptom**: Change approved without understanding downstream effects, causes rework

**Example**:
```
❌ BAD:
Change Request: Modify user profile to support international phone numbers
Approval: "Sounds good, go ahead"
Implementation: Developer updates profile form
PROBLEM DISCOVERED:
- Database schema only supports 10-digit US format (VARCHAR(10))
- API validation rejects international numbers
- Test suite assumes US format
- Documentation shows US-only examples
- 3rd-party SMS provider only supports US numbers

Result: 3 days of unplanned rework, missed sprint deadline
```

**Fix**:
```
✅ GOOD:
Change Request: Modify user profile to support international phone numbers
Impact Analysis:
- Database: Modify phone column to VARCHAR(20), add country code field (0.5 days)
- API: Update validation to support E.164 format (1 day)
- Tests: Update 15 test cases with international numbers (0.5 days)
- Documentation: Add international phone format examples (2 hours)
- Third-Party: Research SMS provider international support (BLOCKER)
  - Current provider: US-only
  - Options: Switch to Twilio (supports international) OR restrict feature to US

Decision: DEFER until 3rd-party provider decision made (requires budget approval)
```

---

## Verification & Validation

**Change management process working when**:

**Level 2**:
- ✅ All requirement changes documented in issue history
- ✅ Product owner approval visible for each change
- ✅ Volatility tracked (manual count acceptable)
- ✅ Red flag threshold (<30%) enforced (stakeholder meeting if exceeded)

**Level 3**:
- ✅ Formal change requests submitted using template
- ✅ Impact analysis completed before approval (checklist verified)
- ✅ Multi-party approval documented (per approval matrix)
- ✅ Volatility metrics calculated monthly (<20% threshold)
- ✅ Affected parties notified within 2 business days

**Level 4**:
- ✅ Volatility metrics tracked in control chart (SPC)
- ✅ Trends analyzed (upward/downward, root causes identified)
- ✅ Quantitative approval criteria applied (effort, volatility %)
- ✅ Process intervention triggered when out-of-control
- ✅ Change impact prediction accuracy measured (estimated vs. actual effort)

---

## Related Practices

**Prerequisites**:
- `requirements-traceability.md` - Needed to identify affected artifacts during impact analysis
- `requirements-specification.md` - Baseline requirements must exist before managing changes

**Related**:
- `requirements-analysis.md` - MoSCoW prioritization for change requests
- `level-scaling.md` - Understanding appropriate formality for your maturity level

**Cross-references**:
- `governance-and-risk` skill - ADR process for major architectural changes
- `platform-integration` skill - GitHub/Azure DevOps change tracking workflows

**Prescription reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.1.2 (Requirements Management - SP 1.3 Manage Requirements Changes)
