# Reference Sheet: Requirements Management in GitHub

## Purpose & Context

Implements CMMI **REQM (Requirements Management)** and **RD (Requirements Development)** process areas using GitHub Issues, Projects, and automation.

**When to apply**: Setting up requirements tracking with bidirectional traceability for CMMI compliance

**Prerequisites**: GitHub repository, Issues enabled, Projects access

---

## CMMI Maturity Scaling

### Level 2: Managed

**Required Practices**:
- Requirements documented as GitHub Issues
- Basic traceability (PR links to issue)
- Change tracking via issue comments

**Work Products**:
- Issue template for requirements
- PR description with requirement reference
- Simple traceability matrix (spreadsheet or wiki)

**Quality Criteria**:
- All requirements have unique IDs
- All PRs reference at least one requirement
- Requirements status tracked (Open/Closed)

**Audit Trail**:
- Issue creation/modification history
- PR → Issue links visible
- Comment history on requirement changes

### Level 3: Defined

**Enhanced Practices**:
- Standardized issue templates with acceptance criteria
- Automated traceability verification
- Requirement baselines (milestones)
- Change impact analysis documented

**Additional Work Products**:
- Automated traceability matrix (generated from GitHub data)
- Requirement change log
- Impact analysis template
- Baseline reports

**Quality Criteria**:
- Bidirectional traceability enforced (requirement ↔ code ↔ test)
- All requirements peer-reviewed
- Change requests follow approval workflow
- Coverage metrics tracked

**Audit Trail**:
- Approval history in issue comments
- Baseline tags/releases
- Change request workflow documented
- Traceability verification logs

### Level 4: Quantitatively Managed

**Statistical Practices**:
- Requirements volatility metrics (changes/week)
- Traceability coverage percentage
- Requirements stability trends

**Quantitative Work Products**:
- Requirements churn rate dashboard
- Defect density by requirement
- Lead time from requirement to deployment

**Quality Criteria**:
- Requirements volatility within control limits
- >95% traceability coverage
- Predictive models for requirement stability

**Audit Trail**:
- Metrics collection automation logs
- Statistical baselines documented
- Process performance data retained

---

## Implementation Guidance

### Quick Start Checklist

**Level 2 Setup** (30 minutes):
- [ ] Create requirement issue template
- [ ] Create PR template with requirement reference
- [ ] Add requirement labels
- [ ] Document traceability approach in README

**Level 3 Setup** (2-3 hours):
- [ ] All Level 2 items
- [ ] Add automated traceability verification (GitHub Actions)
- [ ] Create GitHub Project for requirements tracking
- [ ] Set up milestone baselines
- [ ] Configure change request workflow

**Level 4 Setup** (1-2 days):
- [ ] All Level 3 items
- [ ] Implement requirements metrics collection
- [ ] Create metrics dashboard
- [ ] Establish statistical baselines
- [ ] Set up automated alerting

---

## Templates & Examples

### Issue Template for Requirements

**File**: `.github/ISSUE_TEMPLATE/requirement.yml`

```yaml
name: Requirement
description: Document a system requirement or user story
title: "[REQ-YYYY-NNN] "
labels: ["requirement", "needs-review"]
body:
  - type: markdown
    attributes:
      value: |
        ## Requirement Specification
        Complete all fields for CMMI traceability.
        
  - type: input
    id: req-id
    attributes:
      label: Requirement ID
      description: Auto-generated format REQ-YYYY-NNN
      placeholder: REQ-2026-001
    validations:
      required: true
      
  - type: textarea
    id: statement
    attributes:
      label: Requirement Statement
      description: Clear, testable requirement statement
      placeholder: "The system shall..."
    validations:
      required: true
      
  - type: textarea
    id: acceptance-criteria
    attributes:
      label: Acceptance Criteria
      description: Checklist of criteria that must be met
      value: |
        - [ ] Criterion 1
        - [ ] Criterion 2
        - [ ] Criterion 3
    validations:
      required: true
      
  - type: dropdown
    id: priority
    attributes:
      label: Priority
      options:
        - Critical
        - High
        - Medium
        - Low
    validations:
      required: true
      
  - type: input
    id: source
    attributes:
      label: Source
      description: Stakeholder, regulation, or business need
      
  - type: textarea
    id: dependencies
    attributes:
      label: Dependencies
      description: Link to related requirements (use #issue-number)
      
  - type: dropdown
    id: verification
    attributes:
      label: Verification Method
      options:
        - Unit Test
        - Integration Test
        - System Test
        - Manual Review
        - Inspection
    validations:
      required: true
```

### PR Template with Traceability

**File**: `.github/pull_request_template.md`

```markdown
## Description
<!-- Clear description of changes -->

## Requirements Traceability
**Implements**: Closes #<issue-number>
<!-- Use "Closes #123" to auto-close requirement when PR merges -->

**Related Requirements**:
- #<issue-number>
- #<issue-number>

## Acceptance Criteria Verification
<!-- Copy from requirement issue, check off as completed -->
- [ ] Criterion 1 from #<issue-number>
- [ ] Criterion 2 from #<issue-number>
- [ ] Criterion 3 from #<issue-number>

## Testing
### Test Cases Added
- [ ] Unit tests: `test_file.py::test_function`
- [ ] Integration tests: `test_integration.py::test_scenario`
- [ ] Test coverage: XX% (minimum 80% for new code)

### Verification Evidence
- [ ] All tests passing locally
- [ ] CI/CD pipeline passing
- [ ] Code review completed
- [ ] Documentation updated

## CMMI Traceability
- **Requirement**: #<issue-number>
- **Design**: Link to design doc or ADR
- **Tests**: Link to test files or test run results
- **Documentation**: Link to updated docs

## Checklist
- [ ] Code follows project standards
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Requirement acceptance criteria met
- [ ] Backward compatibility maintained
- [ ] No security vulnerabilities introduced
```

### Automated Traceability Verification

**File**: `.github/workflows/traceability-check.yml`

```yaml
name: Requirements Traceability Check

on:
  pull_request:
    types: [opened, edited, synchronize]
  schedule:
    - cron: '0 0 * * 1'  # Weekly verification

jobs:
  verify-pr-traceability:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - name: Check PR links to requirement
        uses: actions/github-script@v6
        with:
          script: |
            const prBody = context.payload.pull_request.body || '';
            
            // Check for requirement reference
            const hasRequirement = /(?:Implements|Closes|Fixes|Resolves):\s*(?:Closes\s+)?#\d+/i.test(prBody);
            if (!hasRequirement) {
              core.setFailed('❌ PR must link to requirement issue using "Implements: Closes #123"');
              return;
            }
            
            // Check for test coverage mention
            const hasTests = /test coverage:\s*\d+%/i.test(prBody);
            if (!hasTests) {
              core.warning('⚠️  PR should specify test coverage percentage');
            }
            
            // Check for acceptance criteria
            const hasAC = /acceptance criteria/i.test(prBody);
            if (!hasAC) {
              core.warning('⚠️  PR should verify acceptance criteria from requirement');
            }
            
            core.info('✅ PR traceability check passed');
  
  verify-repository-traceability:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install PyGithub tabulate
      
      - name: Verify bidirectional traceability
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python << 'PYTHON_SCRIPT'
          from github import Github
          import os
          from tabulate import tabulate
          
          g = Github(os.environ['GITHUB_TOKEN'])
          repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])
          
          # Get all requirement issues
          requirements = list(repo.get_issues(state='all', labels=['requirement']))
          
          orphaned_reqs = []
          unverified_reqs = []
          
          for req in requirements:
              # Check for linked PRs
              events = list(req.get_events())
              has_pr = any(e.event in ['connected', 'cross-referenced'] for e in events)
              
              if req.state == 'closed' and not has_pr:
                  orphaned_reqs.append((req.number, req.title))
              
              # Check for test verification
              if 'verified by' not in req.body.lower():
                  unverified_reqs.append((req.number, req.title))
          
          # Generate report
          print("# Requirements Traceability Report\n")
          print(f"**Total Requirements**: {len(requirements)}")
          print(f"**Open**: {len([r for r in requirements if r.state == 'open'])}")
          print(f"**Closed**: {len([r for r in requirements if r.state == 'closed'])}\n")
          
          if orphaned_reqs:
              print("## ❌ Orphaned Requirements (closed without PRs)")
              print(tabulate(orphaned_reqs, headers=['Issue', 'Title'], tablefmt='github'))
              print()
          
          if unverified_reqs:
              print("## ⚠️  Unverified Requirements (no test verification)")
              print(tabulate(unverified_reqs[:10], headers=['Issue', 'Title'], tablefmt='github'))
              if len(unverified_reqs) > 10:
                  print(f"\n...and {len(unverified_reqs) - 10} more")
              print()
          
          if not orphaned_reqs and not unverified_reqs:
              print("## ✅ All requirements have proper traceability")
          
          # Fail if critical issues found
          if orphaned_reqs:
              exit(1)
          PYTHON_SCRIPT
```

---

## Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Generic issue titles** | "Fix bug" doesn't indicate requirement | Use "[REQ-YYYY-NNN] Specific requirement title" format |
| **Manual traceability matrix** | Spreadsheet becomes stale immediately | Generate matrix from GitHub API (requirements + linked PRs) |
| **No requirement IDs** | Can't reference requirements uniquely | Use REQ-YYYY-NNN format in title and issue body |
| **Mixing requirements with bugs** | Confusion between new functionality and defects | Use separate labels: `requirement` vs `bug` |
| **No acceptance criteria** | "Done" is subjective, leads to rework | Require checklist of testable acceptance criteria |
| **Forgetting test traceability** | Can't prove requirement is verified | Link tests in PR description, use test naming convention |
| **No baseline management** | Requirements change mid-sprint without control | Use milestones for requirement baselines, freeze before sprint |
| **Skipping change impact analysis** | Requirement changes break existing functionality | Document affected PRs/tests when changing requirements |
| **No requirement review** | Bad requirements waste development time | Require `needs-review` label, approval comment before implementation |
| **Orphaned requirements** | Requirements approved but never implemented | Weekly verification workflow finds requirements without PRs |

---

## Tool Integration

### Traceability Matrix Generation

**Script**: `scripts/generate-traceability-matrix.py`

```python
#!/usr/bin/env python3
"""Generate Requirements Traceability Matrix from GitHub."""

from github import Github
import os
import sys
from tabulate import tabulate

def generate_matrix(repo_name, token):
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    requirements = list(repo.get_issues(state='all', labels=['requirement']))
    
    matrix = []
    for req in requirements:
        req_id = req.title.split(']')[0].strip('[') if '[' in req.title else f"#{req.number}"
        status = '✅ Closed' if req.state == 'closed' else '🔄 Open'
        
        # Find linked PRs
        prs = []
        for event in req.get_timeline():
            if event.event == 'cross-referenced' and event.source and event.source.issue and event.source.issue.pull_request:
                prs.append(f"#{event.source.issue.number}")
        
        pr_links = ', '.join(prs[:3]) if prs else '❌ None'
        if len(prs) > 3:
            pr_links += f" (+{len(prs)-3} more)"
        
        matrix.append([req_id, req.title[:50], status, pr_links])
    
    print("# Requirements Traceability Matrix\n")
    print(tabulate(matrix, headers=['Req ID', 'Title', 'Status', 'Implemented By'], tablefmt='github'))
    print(f"\n**Total Requirements**: {len(requirements)}")
    print(f"**With PR Links**: {len([r for r in matrix if '❌' not in r[3]])}")
    
if __name__ == '__main__':
    repo = os.environ.get('GITHUB_REPOSITORY', 'owner/repo')
    token = os.environ.get('GITHUB_TOKEN')
    
    if not token:
        print("ERROR: Set GITHUB_TOKEN environment variable")
        sys.exit(1)
    
    generate_matrix(repo, token)
```

Run weekly and commit to docs:

```yaml
# Add to .github/workflows/traceability-check.yml
- name: Generate traceability matrix
  run: python scripts/generate-traceability-matrix.py > docs/traceability-matrix.md

- name: Commit matrix
  run: |
    git config user.name "GitHub Actions"
    git config user.email "actions@github.com"
    git add docs/traceability-matrix.md
    git diff --quiet && git diff --staged --quiet || git commit -m "docs: update traceability matrix"
    git push
```

---

## Verification & Validation

### How to Verify This Practice is Working

**Observable Indicators**:
- All requirement issues use standardized template
- Every PR references at least one requirement issue
- Traceability matrix generated weekly without errors
- No orphaned requirements (closed issues without PR links)
- Acceptance criteria documented and verified in PRs

**Metrics to Track** (Level 3+):
- Requirements traceability coverage: (PRs with requirement links / Total PRs) × 100%
- Orphaned requirements: Count of closed requirements without PR links
- Requirements churn: Count of changes to closed requirements per week

**Metrics to Track** (Level 4):
- Requirements volatility rate: Changes per requirement per month
- Lead time: Days from requirement opened to closed
- Requirements defect escape rate: Defects found in production / Total defects

### Common Failure Modes

| Failure Mode | Symptom | Remediation |
|--------------|---------|-------------|
| **PRs missing requirement links** | Traceability check workflow fails frequently | Add required status check, train team on PR template |
| **Stale traceability matrix** | Matrix shows requirements without PRs, but PRs exist | Fix PR template to use "Closes #123", not just mentions |
| **Orphaned requirements** | Requirements closed without implementation | Add weekly verification workflow, require PR link before closing |
| **No test verification** | Requirements marked complete but no tests | Require test coverage field in PR template, enforce minimum coverage |
| **Requirement ID conflicts** | Multiple requirements with same ID | Use auto-incrementing IDs (REQ-YYYY-NNN), document ID assignment process |

---

## Related Practices

**Cross-References**:
- `../requirements-lifecycle/SKILL.md` - CMMI REQM/RD process definitions
- `./github-config-mgmt.md` - Branch protection, baselines, release management
- `./github-quality-gates.md` - Testing and verification for requirements
- `../governance-and-risk/dar-methodology.md` - Architecture decisions for requirements

**Integration with Other Process Areas**:
- **VER (Verification)**: Link tests to requirements via test naming convention
- **VAL (Validation)**: Use acceptance criteria for validation
- **CM (Configuration Management)**: Use milestones for requirement baselines
- **MA (Measurement)**: Track requirements volatility, traceability coverage

---

**Last Updated**: 2026-01-25
**Review Schedule**: Update templates when GitHub releases new features, review annually
