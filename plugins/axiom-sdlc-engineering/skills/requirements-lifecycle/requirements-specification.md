---
parent-skill: requirements-lifecycle
reference-type: specification-templates
load-when: Writing requirements, creating user stories, defining acceptance criteria, documentation standards
---

# Requirements Specification Reference

**Parent Skill:** requirements-lifecycle
**When to Use:** Documenting requirements, creating user stories, defining done criteria

This reference provides templates and formats for specifying requirements across CMMI Levels 2-4.

---

## Purpose & Context

**What this achieves**: Systematic documentation of requirements in formats appropriate for project maturity level and stakeholder needs.

**When to apply**:
- Elicitation complete, need to document requirements
- Stakeholders need written requirements for approval
- Development team needs clear acceptance criteria
- Audit/compliance requires documented requirements

**Prerequisites**:
- Requirements elicited from stakeholders
- Prioritization complete (MoSCoW or equivalent)
- Basic understanding of user needs and system context

---

## CMMI Maturity Scaling

### Level 2: Managed (Specification)

**User stories with acceptance criteria**:

**Format**:
```
Title: [Actor] [action] [object]

As a [type of user]
I want [goal/desire]
So that [benefit/value]

Acceptance Criteria:
- [Testable condition 1]
- [Testable condition 2]
- [Testable condition 3]

Priority: [Must/Should/Could/Won't Have]
Effort: [Story points or time estimate]
```

**Documentation**:
- Captured in issue tracker (GitHub Issues, Azure DevOps, Jira)
- 3-5 acceptance criteria per story
- Unique ID assigned automatically by platform

**Review**:
- Optional/informal - team reviews during sprint planning
- Product owner approves before sprint commitment

**Effort**:
- 15-30 minutes per user story
- 2-4 hours total for 8-12 stories per sprint

**Example**:
```
Title: Employee can view their vacation balance

As an employee
I want to view my current vacation balance and accrual rate
So that I can plan time off without exceeding my available days

Acceptance Criteria:
- Balance displays in days (not hours) with 1 decimal precision
- Accrual rate shown as "X days per month"
- Historical vacation usage visible for current calendar year
- Balance updates within 24 hours of approved vacation request
- Negative balance shown in red with warning message

Priority: Must Have
Effort: 5 story points
Labels: frontend, reporting, employee-self-service
```

### Level 3: Defined (Specification)

**Organizational templates + peer review**:

**Format Options**:
- User stories (for Agile teams) + organizational template compliance
- Use cases (for complex workflows)
- Formal requirements statements (for regulated industries)

**Organizational Template Additions** (beyond Level 2):
```
User Story Template (Level 3):

ID: REQ-{COMPONENT}-{NUMBER}  (e.g., REQ-AUTH-042)
Title: [Actor] [action] [object]
Epic: [Parent epic/feature]
Dependencies: [Other requirement IDs]
Rationale: [Why this requirement exists]

As a [type of user]
I want [goal/desire]
So that [benefit/value]

Acceptance Criteria:
- [Testable condition 1]
- [Testable condition 2]
- [Testable condition 3]

Non-Functional Requirements:
- Performance: [Response time, throughput]
- Security: [Authentication, authorization, data protection]
- Usability: [Accessibility standards, browser support]

Priority: [Must/Should/Could/Won't Have]
Effort: [Story points]
Estimated Duration: [Developer-days]

Source: [Stakeholder name/elicitation session]
Created By: [Requirements analyst]
Reviewed By: [Peer reviewer, date]
Approved By: [Product owner, date]
```

**Peer Review Required**:
- Requirements reviewed before development starts
- Checklist-based review (completeness, clarity, testability)
- Review findings documented in issue comments

**Effort**:
- 30-45 minutes per user story (due to template fields)
- 1 hour peer review per sprint (batch review of 8-12 stories)

**Example Use Case** (Level 3 alternative to user stories):
```
Use Case: Process Employee Vacation Request

ID: UC-VACATION-001
Actor: Employee, Manager, HR System
Preconditions:
  - Employee is authenticated
  - Employee has positive vacation balance

Main Flow:
1. Employee navigates to vacation request form
2. System displays current balance and accrual rate
3. Employee enters start date, end date, and optional notes
4. System calculates business days (excluding weekends, holidays)
5. System validates sufficient balance available
6. Employee submits request
7. System sends notification to manager for approval
8. Manager reviews request and approves/denies
9. System updates employee vacation balance
10. System sends confirmation to employee

Alternative Flows:
A1: Insufficient balance (step 5)
  - System displays error: "Insufficient vacation balance. Available: X days, Requested: Y days"
  - Employee can modify dates or cancel request

A2: Manager denies request (step 8)
  - Manager enters denial reason
  - System notifies employee with reason
  - Balance remains unchanged

Exception Flows:
E1: System error during submission
  - System displays generic error message
  - Request is not created
  - Employee can retry

Postconditions:
- Vacation request created in system (approved or pending)
- Manager notified via email
- Employee vacation balance reserved (if approved)

Non-Functional Requirements:
- Response time: <2 seconds for balance calculation
- Availability: 99.5% uptime during business hours
- Security: SSL encryption for data transmission
- Audit: Log all approvals/denials with timestamp and user

Source: JAD Session 2026-01-15 (HR VP, Employee Representatives)
Created By: Jane Smith (Business Analyst)
Reviewed By: Tom Chen (Senior Developer), 2026-01-18
Approved By: Sarah Johnson (Product Owner), 2026-01-20
```

### Level 4: Quantitatively Managed (Specification)

**Metrics-tracked quality** (all Level 3 plus):

**Specification Quality Metrics**:
- **Defect density**: Defects found in requirements per review (target: <5%)
- **Completeness**: % requirements with all template fields populated (target: >95%)
- **Clarity score**: % requirements requiring clarification after review (target: <10%)
- **Testability**: % requirements with verifiable acceptance criteria (target: 100%)

**Predictive Metrics**:
- **Specification effort**: Time to write requirements vs. historical baseline
- **Review efficiency**: Defects found per review hour
- **Downstream impact**: Requirements defects that escape to implementation

**Example**:
```
Requirements Quality Dashboard:

Sprint 20 Requirements Specification
Total Requirements: 12 user stories
Specification Effort: 6.5 hours (vs. baseline 5-7 hours, within control)

Completeness Check:
- All template fields populated: 12/12 (100%) ✅
- Acceptance criteria count: 3-5 per story (all within range) ✅
- Non-functional requirements defined: 12/12 (100%) ✅

Peer Review Results:
- Defects found: 3 (defect density: 25% - ABOVE threshold)
  - REQ-VAC-023: Ambiguous acceptance criterion ("balance updates quickly" → specify <24 hours)
  - REQ-VAC-024: Missing dependency on REQ-CAL-015 (holiday calendar)
  - REQ-VAC-028: Incomplete error handling (no specification for negative balance scenario)

Clarity Issues:
- Requirements requiring clarification: 1/12 (8.3%, below 10% threshold) ✅
- REQ-VAC-026: Developer asked "What happens if manager doesn't respond within 5 days?"

Testability:
- Verifiable acceptance criteria: 12/12 (100%) ✅
- All criteria written as "Given-When-Then" or measurable condition

Prediction: Based on defect density (25%), expect 2-3 requirements to require modification during implementation (within normal range for new feature area).

Action Items:
- REQ-VAC-023, REQ-VAC-024, REQ-VAC-028: Updated based on review feedback
- Add organizational checklist item: "Verify all dependencies identified before approval"
```

---

## Implementation Guidance

### User Story Template (Level 2/3)

**INVEST Criteria** (Good user stories are):
- **Independent**: Can be developed in any order
- **Negotiable**: Details can be discussed, not a contract
- **Valuable**: Delivers value to user or business
- **Estimable**: Team can estimate effort
- **Small**: Completable within one sprint
- **Testable**: Clear acceptance criteria

**Writing Effective User Stories**:

**Good Examples**:
```
✅ As a customer, I want to save items to a wishlist so that I can purchase them later
   - Clear actor (customer)
   - Clear goal (save to wishlist)
   - Clear benefit (purchase later)

✅ As a manager, I want to export employee vacation reports to Excel so that I can analyze patterns
   - Specific format (Excel)
   - Clear use case (analyze patterns)
```

**Bad Examples**:
```
❌ As a user, I want the system to work well
   - Too vague ("work well" is not testable)
   - No specific goal or benefit

❌ As a developer, I want to refactor the authentication module
   - Not user-facing (technical task, not user story)
   - No user value stated

❌ As a customer, I want a better checkout experience
   - Not specific enough ("better" is subjective)
   - Missing concrete goal
```

**Template Usage**:

**Minimal (Level 2)**:
```
As a [user type]
I want [goal]
So that [benefit]

Acceptance Criteria:
- [Condition 1]
- [Condition 2]
```

**Standard (Level 3)**:
```
ID: REQ-{AREA}-{NUM}
Title: [Actor] [action] [object]

As a [user type]
I want [goal]
So that [benefit]

Acceptance Criteria:
- [Condition 1 - Given X, When Y, Then Z]
- [Condition 2 - Given X, When Y, Then Z]
- [Condition 3 - Given X, When Y, Then Z]

Non-Functional:
- Performance: [Metric]
- Security: [Requirement]

Priority: [MoSCoW]
Effort: [Points]
Source: [Stakeholder/Session]
Reviewed: [Name, Date]
```

### Acceptance Criteria Best Practices

**Format Options**:

**Checklist Style** (Simple, Level 2):
```
Acceptance Criteria:
- User can enter credit card number (16 digits, no spaces)
- System validates card number using Luhn algorithm
- Invalid card shows error message: "Invalid card number"
- Valid card proceeds to confirmation screen
```

**Given-When-Then Style** (Formal, Level 3):
```
Acceptance Criteria:

Scenario 1: Valid credit card
  Given user is on payment screen
  When user enters valid 16-digit card number
  Then system validates using Luhn algorithm
  And proceeds to confirmation screen

Scenario 2: Invalid credit card
  Given user is on payment screen
  When user enters invalid card number
  Then system displays error: "Invalid card number. Please check and try again"
  And payment screen remains visible

Scenario 3: Incomplete card number
  Given user is on payment screen
  When user enters <16 digits
  Then submit button remains disabled
  And helper text shows "Card number must be 16 digits"
```

**Common Mistakes**:

| Mistake | Example | Better Approach |
|---------|---------|-----------------|
| **Too vague** | "System performs well" | "Page loads in <2 seconds on 3G connection" |
| **Not testable** | "UI looks good" | "Button follows design system color palette (hex #007AFF)" |
| **Missing negative cases** | "User can login" | "User can login with valid credentials AND sees error with invalid credentials" |
| **Implementation-focused** | "Use React hooks for state" | "Form data persists when user navigates away and returns" |
| **No measurable outcome** | "Improve performance" | "Reduce API response time from 800ms to <200ms" |

### Traceability ID Assignment

**Purpose**: Unique, stable identifiers for requirements that persist across tool migrations.

**Level 2**: Platform-assigned IDs sufficient
```
GitHub: Issue #123
Azure DevOps: Work Item 4567
Jira: PROJ-89
```

**Level 3**: Semantic IDs for requirements traceability
```
Format: REQ-{COMPONENT}-{NUMBER}

Examples:
- REQ-AUTH-001: User authentication via SSO
- REQ-AUTH-002: Password reset flow
- REQ-REPORT-015: Export vacation report to Excel
- REQ-NOTIF-032: Email notification on approval

Rationale:
- Component prefix (AUTH, REPORT) groups related requirements
- Sequential numbering within component
- Survives tool migration (unlike #123 which is GitHub-specific)
```

**ID Assignment Process** (Level 3):
1. Define component prefixes in requirements management plan
2. Assign next available number within component
3. Document ID in requirement title and tracking system
4. Never reuse IDs (even if requirement deleted)

**Example Component Prefixes**:
```
Component Prefix Registry:

AUTH   - Authentication and authorization
PROFILE - User profile management
VACATION - Vacation request and tracking
REPORT - Reporting and analytics
NOTIF  - Notifications (email, SMS, in-app)
ADMIN  - System administration
API    - External API integrations
PERF   - Performance requirements
SEC    - Security requirements
```

### Tool Integration

#### GitHub Issue Templates (Level 2/3)

**Create**: `.github/ISSUE_TEMPLATE/user-story.yml`

```yaml
name: User Story
description: Create a new user story requirement
title: "[User Story] "
labels: ["requirement", "user-story"]
body:
  - type: markdown
    attributes:
      value: |
        ## User Story Template

  - type: input
    id: req-id
    attributes:
      label: Requirement ID
      description: Format REQ-{COMPONENT}-{NUMBER} (Level 3 only)
      placeholder: REQ-AUTH-042
    validations:
      required: false

  - type: textarea
    id: user-story
    attributes:
      label: User Story
      description: As a [user type], I want [goal], so that [benefit]
      placeholder: |
        As a customer
        I want to save items to a wishlist
        So that I can purchase them later
    validations:
      required: true

  - type: textarea
    id: acceptance-criteria
    attributes:
      label: Acceptance Criteria
      description: Testable conditions that must be met
      placeholder: |
        - User can add items to wishlist from product page
        - Wishlist persists across sessions
        - User can view wishlist from navigation menu
    validations:
      required: true

  - type: dropdown
    id: priority
    attributes:
      label: Priority (MoSCoW)
      options:
        - Must Have
        - Should Have
        - Could Have
        - Won't Have
    validations:
      required: true

  - type: input
    id: effort
    attributes:
      label: Effort Estimate
      description: Story points or time estimate
      placeholder: "5 points"
    validations:
      required: false
```

**Usage**: Creates structured issues with consistent format, enforces required fields.

#### Azure DevOps Work Item Template (Level 3)

**Create custom work item type**: User Story (Requirements)

**Fields**:
```
System Fields:
- ID (auto-assigned)
- Title
- State (New, Approved, In Progress, Done)
- Assigned To
- Created By, Created Date

Custom Fields:
- Requirement ID (format: REQ-{COMPONENT}-{NUMBER})
- User Story (multi-line text)
- Acceptance Criteria (multi-line text)
- Priority (dropdown: Must/Should/Could/Won't)
- Effort (integer - story points)
- Non-Functional Requirements (multi-line text)
- Source (text - stakeholder/session)
- Reviewed By (text)
- Approved By (text)

Links:
- Parent: Epic/Feature
- Related: Other requirements
- Tests: Test cases
- Implements: Code commits
```

**Query for RTM**:
```
Work Item Type = User Story (Requirements)
AND State = Done
AND Iteration = @CurrentIteration

Show columns: ID, Requirement ID, Title, Linked PRs (count), Linked Tests (count)
```

---

## Common Anti-Patterns

### Vague Acceptance Criteria

**Symptom**: Developers ask "What does this mean?" during implementation

**Example**:
```
❌ BAD:
Acceptance Criteria:
- System is fast
- UI looks professional
- Error handling works correctly

Problem: "Fast" (how fast?), "professional" (what standard?), "works" (what scenarios?)
```

**Fix**:
```
✅ GOOD:
Acceptance Criteria:
- Page load completes in <2 seconds on 3G connection
- UI follows company design system (colors, typography, spacing per Figma specs)
- Error messages display for 3 scenarios: invalid input, network failure, server error
```

### Implementation Leakage

**Symptom**: Requirements specify HOW instead of WHAT

**Example**:
```
❌ BAD:
As a developer, I want to use Redux for state management so that the app is scalable

Problem:
- Actor is "developer" (not user-facing)
- Specifies implementation (Redux)
- Benefit ("scalable") is not measurable user value
```

**Fix**:
```
✅ GOOD:
As a customer, I want my shopping cart to persist across devices so that I can start shopping on mobile and complete purchase on desktop

Technical Note (separate from story): Consider Redux or Context API for cross-device state sync
```

### Missing Rationale

**Symptom**: Team implements requirement without understanding WHY

**Example**:
```
❌ BAD:
REQ-REPORT-023: Export must include middle name field

Problem: Why middle name? Is it legally required? Customer request? Compliance?
```

**Fix**:
```
✅ GOOD:
REQ-REPORT-023: Export must include middle name field

Rationale: Federal reporting regulation 45 CFR §164.514 requires full legal name including middle name for healthcare provider directories. Missing middle name causes compliance violation.

Source: Compliance Lead (Jane Doe), JAD Session 2026-01-10
Priority: Must Have (legal requirement)
```

### Gold-Plated Acceptance Criteria

**Symptom**: 15+ acceptance criteria for a "small" story

**Example**:
```
❌ BAD:
Story: Customer can search products (5 points)

Acceptance Criteria: (20 items)
- Search by product name
- Search by SKU
- Search by category
- Search by price range
- Search by brand
- Search by color
- Search by size
- Autocomplete suggestions
- Fuzzy matching
- Synonym support
- Sorting by relevance
- Sorting by price
- Sorting by date added
- Pagination (20 items per page)
- Export results to CSV
- Save search queries
- Search history
- Recently viewed products
- Recommended products based on search
- Analytics tracking for search queries

Problem: This is 5+ stories masquerading as one
```

**Fix**:
```
✅ GOOD:
Story: Customer can search products by name (3 points)

Acceptance Criteria:
- Customer enters search term in search box
- System returns products matching name (case-insensitive)
- Results display within 2 seconds
- Empty results show "No products found" message

Deferred to Future Stories:
- Advanced filtering (SKU, category, price) → REQ-SEARCH-002
- Autocomplete and suggestions → REQ-SEARCH-003
- Search analytics → REQ-SEARCH-004
```

---

## Verification & Validation

**Specification complete when**:

**Level 2**:
- ✅ User stories have 3-5 testable acceptance criteria
- ✅ Stories follow INVEST principles (independent, negotiable, valuable, estimable, small, testable)
- ✅ Product owner has reviewed and accepted stories
- ✅ Team can estimate effort (no unknowns blocking estimation)

**Level 3**:
- ✅ All template fields populated (ID, rationale, source, etc.)
- ✅ Peer review completed with checklist (see below)
- ✅ Non-functional requirements specified where applicable
- ✅ Traceability IDs assigned and unique

**Level 4**:
- ✅ Specification quality metrics within baselines
  - Defect density <5%
  - Completeness >95%
  - Clarity <10% requiring post-review clarification
- ✅ Testability verified (100% of criteria are verifiable)

**Peer Review Checklist** (Level 3):

```
Requirements Peer Review Checklist

Reviewer: _______________  Date: _______________
Requirement ID: _______________

Completeness:
[ ] User story follows "As a [user], I want [goal], so that [benefit]" format
[ ] 3-5 acceptance criteria defined (not too few, not too many)
[ ] Non-functional requirements specified (performance, security, usability)
[ ] Priority assigned (MoSCoW)
[ ] Source/stakeholder identified
[ ] Traceability ID assigned (Level 3)

Clarity:
[ ] Actor is specific (not "user" - specify type: customer, admin, manager)
[ ] Goal is unambiguous (no words like "fast", "easy", "good")
[ ] Benefit is measurable or observable
[ ] Acceptance criteria are testable (Given-When-Then or measurable conditions)
[ ] No implementation details leaked (focus on WHAT, not HOW)

Correctness:
[ ] Requirement aligns with project scope and goals
[ ] No conflicts with existing requirements
[ ] Dependencies identified (if any)
[ ] Rationale makes sense and is documented

Testability:
[ ] Each acceptance criterion can be verified (manual test or automated)
[ ] Success/failure is binary (not subjective)
[ ] Edge cases and error conditions covered

Issues Found: _______________
[ ] Approved  [ ] Needs Revision

Signature: _______________
```

---

## Related Practices

**Before specification**:
- `requirements-elicitation.md` - Gather requirements from stakeholders
- `requirements-analysis.md` - Prioritize and resolve conflicts

**After specification**:
- `requirements-traceability.md` - Link requirements to design/code/tests
- `requirements-change-management.md` - Handle changes to specifications

**Cross-references**:
- `platform-integration` skill - GitHub/Azure DevOps issue templates
- `governance-and-risk` skill - ADR process for major decisions

**Prescription reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.1.1 (Requirements Development - SP 1.2 Develop Customer Requirements, SP 1.3 Establish Product Requirements)
