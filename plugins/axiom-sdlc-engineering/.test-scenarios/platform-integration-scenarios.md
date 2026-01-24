# platform-integration - Test Scenarios

## Purpose

Test that platform-integration skill correctly:
1. Maps CMMI processes to GitHub/Azure DevOps features
2. Provides platform-specific implementation guidance
3. Handles hybrid/multi-platform scenarios
4. Guides platform selection based on needs
5. Supports platform migration strategies

## Scenario 1: GitHub Requirements Traceability

### Context
- Using GitHub Issues for requirements
- Level 3 project needs traceability
- Don't know how to link issues to PRs to tests
- Team of 6 developers

### User Request
"How do I implement requirements traceability in GitHub for CMMI Level 3?"

### Expected Behavior
- Requirements in GitHub reference sheet
- Issue templates for requirements
- Traceability via issue refs in PR descriptions
- Labels for requirement categorization
- GitHub Projects for RTM queries
- Automation via Actions (check for issue refs)

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: Azure DevOps Work Item Hierarchy

### Context
- Using Azure DevOps
- Need Epic → Feature → User Story hierarchy
- Level 3 project
- Confused about work item types

### User Request
"What work item types should I use in Azure DevOps for CMMI requirements?"

### Expected Behavior
- Requirements in Azure DevOps reference sheet
- Epic → Feature → User Story hierarchy
- Custom fields for traceability
- Queries for RTM
- Acceptance criteria fields
- References requirements-lifecycle for WHAT to capture

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: GitHub vs Azure DevOps - Which to Choose?

### Context
- New project, no platform chosen yet
- Team split: developers want GitHub, management wants Azure DevOps
- **Authority conflict**: Different stakeholders
- Level 3 project

### User Request
"Should we use GitHub or Azure DevOps for our CMMI Level 3 project?"

### Expected Behavior
- Platform selection criteria (not subjective preference)
- GitHub strengths: developer experience, ecosystem
- Azure DevOps strengths: work item hierarchy, analytics
- Decision matrix: team size, existing tooling, integration needs
- References governance-and-risk for DAR process
- Counters "platform wars" bias

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: GitHub Actions Quality Gates

### Context
- Using GitHub
- Need automated quality gates for Level 3
- Want to enforce coverage, linting, traceability checks
- Currently manual

### User Request
"How do I set up automated quality gates in GitHub for CMMI Level 3?"

### Expected Behavior
- Quality gates in GitHub reference sheet
- Required status checks
- Coverage gates (fail if below threshold)
- Traceability check (PR must reference issue)
- Branch protection rules
- Example GitHub Actions workflows

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Azure DevOps Metrics Dashboard

### Context
- Using Azure DevOps
- Level 3 requires metrics and baselines
- Want to track DORA metrics
- Don't know where to start

### User Request
"How do I create a metrics dashboard in Azure DevOps for CMMI Level 3?"

### Expected Behavior
- Measurement in Azure DevOps reference sheet
- Analytics views for DORA metrics
- Dashboard widgets
- OData queries for custom metrics
- PowerBI integration (if available)
- References quantitative-management for WHAT to measure

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 6: Hybrid Setup - GitHub + Azure DevOps

### Context
- Code in GitHub (developer preference)
- Work tracking in Azure DevOps (management requirement)
- **Complexity**: Two platforms to integrate
- Level 3 project

### User Request
"Can we use GitHub for code and Azure DevOps for work items? How do we maintain traceability?"

### Expected Behavior
- Multi-platform scenario guidance
- Azure Boards + GitHub integration
- Traceability via AB# references in commits
- Sync mechanisms
- Trade-offs (complexity vs. stakeholder needs)
- Decision: possible but adds overhead

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 7: Platform Migration - GitHub to Azure DevOps

### Context
- 2-year project on GitHub
- Organization standardizing on Azure DevOps
- **Sunk cost**: All history in GitHub
- **Concern**: Lose traceability during migration
- Level 3 project

### User Request
"We're migrating from GitHub to Azure DevOps. How do we preserve our CMMI compliance?"

### Expected Behavior
- Migration strategy guidance
- Export/import patterns (Git history, work items)
- Parallel operation period
- Update traceability links
- Audit trail preservation
- Timeline: 2-4 weeks for migration

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 8: Audit Trail Requirements

### Context
- Financial system (SOX compliance)
- Need 7-year audit trail
- Using GitHub
- Worried about retention policies

### User Request
"How do we ensure our GitHub audit trail meets SOX retention requirements?"

### Expected Behavior
- Audit trail in GitHub reference sheet
- Commit history retention (permanent in Git)
- PR review history (GitHub retention settings)
- Action logs (retention policies)
- Backup strategy for compliance
- References compliance mapping for SOX specifics

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

platform-integration skill is ready when:

- [ ] All 8+ scenarios provide actionable guidance
- [ ] GitHub implementation patterns (5 reference sheets)
- [ ] Azure DevOps implementation patterns (5 reference sheets)
- [ ] Platform selection criteria (decision matrix)
- [ ] Migration strategies (both directions)
- [ ] Hybrid scenarios handled
- [ ] Audit trail guidance for both platforms
- [ ] References to process skills for WHAT (this skill is HOW)
