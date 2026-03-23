# Platform Integration Test Scenarios

**Skill Type**: Reference (retrieval + application testing)
**Test Approach**: Can agents find and correctly apply platform-specific implementation guidance?

---

## Scenario 1: Requirements Traceability in GitHub

**Context**: Level 3 project, GitHub repository, need bidirectional traceability between requirements and code.

**User Request**: "We need to implement requirements traceability in GitHub. I want to link user stories in issues to code changes in PRs and then to tests. How do I set this up following CMMI REQM practices?"

**Success Criteria**:
1. Agent identifies the correct reference sheet (GitHub Requirements)
2. Agent provides specific implementation pattern (issue references in PR descriptions, labels, projects)
3. Agent covers bidirectional traceability (requirements → code → tests)
4. Agent provides template/example
5. Agent mentions traceability verification approach

**Expected Coverage**:
- Issue templates for requirements with IDs
- PR description linking pattern (`Implements #123`, `Closes #456`)
- Label strategy for requirement types (feature, enhancement, defect)
- Milestone/project organization
- Traceability verification (API query or manual review)

---

## Scenario 2: CI/CD Pipeline with Quality Gates in Azure DevOps

**Context**: Level 3 project, Azure DevOps, need multi-stage pipeline with quality gates for integration testing (PI) and verification (VER).

**User Request**: "Set up a CI/CD pipeline in Azure DevOps with proper quality gates. We need unit tests to pass before integration tests run, and integration tests to pass before deployment to staging. This is for a .NET Core API."

**Success Criteria**:
1. Agent identifies correct reference sheet (Azure DevOps Quality Gates)
2. Agent provides multi-stage pipeline YAML example
3. Agent covers gate dependencies (unit → integration → deployment)
4. Agent includes quality metrics (test coverage, code analysis)
5. Agent mentions approval workflows for production deployment

**Expected Coverage**:
- Azure Pipelines YAML structure (stages, jobs, steps)
- Test execution with result publishing
- Quality gates between stages
- Conditional stage execution (on success)
- Approval gates for production
- Work item linking in commits

---

## Scenario 3: Branch Protection and Code Review in GitHub

**Context**: Level 3 project, GitHub, need to enforce code review and prevent direct pushes to main branch (CM process area).

**User Request**: "Configure GitHub branch protection for our main branch. We need at least 2 reviewers, all tests must pass, and code owners must approve changes to critical files. What's the step-by-step setup?"

**Success Criteria**:
1. Agent identifies correct reference sheet (GitHub Configuration Management)
2. Agent provides step-by-step GitHub UI configuration OR API/settings file
3. Agent covers required reviewers, status checks, CODEOWNERS
4. Agent explains how to set up CODEOWNERS file
5. Agent mentions audit trail (PR review history)

**Expected Coverage**:
- Branch protection rules configuration
- Required status checks setup
- Required reviewers (number + dismissal settings)
- CODEOWNERS file format and patterns
- Merge strategies (squash vs merge vs rebase)
- Branch naming conventions (GitFlow, GitHub Flow, trunk-based)

---

## Scenario 4: Risk Tracking in Azure DevOps Work Items

**Context**: Level 3 project, Azure DevOps, need to implement risk register using work items (RSKM process area).

**User Request**: "We want to track risks in Azure DevOps work items instead of a spreadsheet. How do I set up a risk work item type with probability, impact, mitigation tracking, and a dashboard to visualize high-priority risks?"

**Success Criteria**:
1. Agent identifies correct reference sheet (Azure DevOps risk tracking OR work items)
2. Agent provides work item customization approach (custom type or use Bug/Issue with fields)
3. Agent covers risk fields (probability, impact, score, mitigation, owner)
4. Agent mentions queries for risk prioritization
5. Agent provides dashboard widget configuration

**Expected Coverage**:
- Custom work item type creation (or adaptation of existing type)
- Custom fields for probability (1-5), impact (1-5), risk score (calculated)
- Mitigation plan field
- Risk owner assignment
- Query for high-priority risks (score > 12)
- Dashboard widgets (risk board, risk trend chart)
- State workflow (Identified → Assessed → Mitigating → Closed/Accepted)

---

## Scenario 5: Automated Metrics Collection in GitHub

**Context**: Level 3 project, GitHub, need to collect DORA metrics automatically (MA process area).

**User Request**: "We need to track deployment frequency, lead time, and change failure rate in GitHub. Can you show me how to automate metrics collection using GitHub Actions and display them on a dashboard?"

**Success Criteria**:
1. Agent identifies correct reference sheet (GitHub Measurement)
2. Agent provides GitHub Actions workflow for metrics collection
3. Agent covers DORA metrics calculation logic
4. Agent mentions storage (GitHub API, database, or artifact)
5. Agent provides dashboard/visualization approach

**Expected Coverage**:
- GitHub Actions workflow triggered on deployment events
- API queries for commit → deploy timestamps (lead time)
- Deployment counting (deployment frequency)
- Failed deployment detection (change failure rate)
- Metrics storage options (GitHub Packages artifacts, external DB, API)
- Visualization options (GitHub Insights, Grafana, custom dashboard)
- Scheduled metric calculation (weekly/monthly aggregation)

---

## Baseline Testing Questions

For each scenario, test WITHOUT the skill:

1. **Can the agent find the right information?** (Retrieval)
   - Do they know which CMMI process area applies?
   - Do they know where to look in GitHub/Azure DevOps?

2. **Can they apply the information correctly?** (Application)
   - Do they provide specific, actionable steps?
   - Do they give working examples?
   - Do they cover edge cases?

3. **Are common use cases covered?** (Coverage)
   - Are critical features mentioned?
   - Are integration points explained?
   - Are alternatives discussed?

---

## Expected Baseline Gaps

Without the skill, agents may:
1. Provide generic advice without platform-specific steps
2. Miss CMMI requirements (e.g., bidirectional traceability, audit trail)
3. Not know about platform features (CODEOWNERS, multi-stage pipelines, custom work items)
4. Give partial solutions without end-to-end implementation
5. Not connect CMMI process areas to platform capabilities
6. Skip verification/monitoring setup
7. Provide outdated or deprecated approaches
8. Miss compliance/audit requirements
9. Not explain trade-offs between approaches
10. Forget to mention tool limitations or workarounds
