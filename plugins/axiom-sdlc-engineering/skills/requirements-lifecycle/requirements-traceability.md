---
parent-skill: requirements-lifecycle
reference-type: traceability-implementation
load-when: Creating RTM, establishing traceability, audit preparation, compliance verification
---

# Requirements Traceability Reference

**Parent Skill:** requirements-lifecycle
**When to Use:** Creating RTM, linking requirements to design/tests, audit trail creation, compliance verification

This reference provides patterns for implementing bidirectional traceability across CMMI Levels 2-4.

---

## Purpose & Context

**What this achieves**: Bidirectional traceability between requirements, design, implementation, and tests using platform-native capabilities.

**When to apply**:
- Level 3 compliance requirement (CMMI REQM SP 1.4)
- Audit preparation (need to demonstrate traceability)
- Impact analysis (which tests affected by requirement change?)
- Coverage verification (are all requirements tested?)

**Prerequisites**:
- Requirements documented in issue tracker
- Version control in use (Git)
- Platform selected (GitHub, Azure DevOps, or other)

**Key principle**: Traceability should be **free** (byproduct of normal workflow), not a separate manual task.

---

## CMMI Maturity Scaling

### Level 2: Managed (Traceability)

**Minimal traceability**:

**Approach**:
- Spreadsheet RTM OR platform-native linking
- Manual updates acceptable
- Forward traceability sufficient (requirements → tests)

**Verification Frequency**:
- At major milestones (release, sprint end)
- Before audit/demo
- Ad-hoc when stakeholder requests

**Coverage Target**:
- >70% requirements traced
- Best-effort basis

**Effort**:
- 1-2 hours per sprint to update RTM
- 4-8 hours for audit preparation

**Example Level 2 RTM (Spreadsheet)**:
```
| Req ID | Description | Design | Implementation | Test | Status |
|--------|-------------|--------|----------------|------|--------|
| REQ-001 | User login | auth-module | PR#45 | TC-001 | Verified |
| REQ-002 | Password reset | auth-module | PR#47 | TC-002 | In Dev |
| REQ-003 | Profile page | profile-mod | [Missing] | [Missing] | Not Started |
```

**Example Level 2 (GitHub Platform Linking)**:
```
Issue #123: User Authentication
Labels: requirement, must-have

Linked PRs:
- PR #145: Implement login endpoint (merged)
- PR #148: Add password hashing (merged)

Test Coverage:
- tests/auth/test_login.py (mentioned in PR #145 description)
- tests/auth/test_password_hash.py (linked in PR #148)

Traceability: REQ → PRs → Tests (via PR descriptions, commit messages)
```

### Level 3: Defined (Traceability)

**Tool-based traceability** (all Level 2 plus):

**Approach**:
- Platform-native traceability (GitHub, Azure DevOps)
- Automated linking via commit messages, PR descriptions
- Bidirectional traceability (forward AND backward)
- Tool-generated RTM reports

**Verification Frequency**:
- Automated weekly verification (detect orphaned tests, missing links)
- Manual review at sprint boundaries
- Continuous monitoring via dashboards

**Coverage Target**:
- >90% requirements traced forward (req → tests)
- >85% tests traced backward (tests → req)
- Orphaned tests flagged for review

**Effort**:
- 30 minutes per sprint (verification only, linking is free)
- 2-4 hours for audit report generation

**Example Level 3 (GitHub with Automation)**:
```
GitHub Issue #123: User Authentication
Labels: requirement, priority:must

Automatic Traceability:
- PRs mentioning "#123" in description (4 PRs)
- Commits with "Implements #123" (12 commits)
- Test files changed in those commits (6 test files)

Generated RTM (via GitHub API query):
  Requirement: #123
  → Design: docs/adr/003-auth-strategy.md (linked in #123)
  → PRs: #145, #148, #151, #153
  → Commits: abc1234, def5678, ... (12 total)
  → Tests: tests/auth/test_login.py, tests/auth/test_password_hash.py, ...
  Status: ✅ Fully traced (req → design → code → tests)

Backward Trace:
  Test: tests/auth/test_2fa.py
  → Commits: ghi9012 (added this test)
  → PR: #160
  → Issue: #125 (2FA requirement)
  Status: ✅ Traced to requirement
```

**Example Level 3 (Azure DevOps)**:
```
Work Item #456: User Authentication
Type: Requirement
State: Done

Built-in Traceability:
- Linked commits (via "AB#456" in commit messages): 15 commits
- Related work items: Design#457, Bug#480
- Test cases: TC-001, TC-002, TC-003 (linked)
- Build runs: Build#890, Build#891 (validated tests)

Query for RTM:
  SELECT [ID], [Title], [State], [Linked Commits], [Test Cases]
  FROM WorkItems
  WHERE [Work Item Type] = 'Requirement'
  AND [Area Path] = 'MyProject\\Authentication'

Output: CSV export for audit
```

### Level 4: Quantitatively Managed (Traceability)

**Metrics-driven traceability** (all Level 3 plus):

**Metrics**:
- **Traceability coverage**: (Traced Requirements / Total Requirements) × 100%
- **Orphaned tests**: Tests not linked to any requirement
- **Missing tests**: Requirements not linked to any test
- **Stale links**: Links to deleted/closed PRs or issues
- **Trace completeness**: % requirements with FULL trace (req → design → code → test)

**Thresholds**:
- Coverage >95% (yellow flag at 90%, red at 85%)
- Orphaned tests <5%
- Missing tests <3%
- Stale links <2%

**Verification Frequency**:
- Automated daily verification
- Dashboard with real-time metrics
- Alerts for threshold violations

**Effort**:
- 15 minutes per sprint (dashboard review)
- Fully automated

**Example Level 4 Dashboard**:
```
Traceability Metrics Dashboard - Sprint 15

Coverage Metrics:
  Forward Traceability: 147/150 requirements (98%) ✅
    - Target: >95%
    - Trend: +2% from last sprint

  Backward Traceability: 243/250 tests (97.2%) ✅
    - Target: >95%
    - Orphaned tests: 7 (2.8%) ✅ (below 5% threshold)

Quality Metrics:
  Full Trace (req → design → code → test): 139/150 (92.7%) ⚠️
    - Target: >95%
    - Missing: 11 requirements lack design traceability

  Stale Links Detected: 3 (2%) ✅
    - PR#102 (deleted), Issue#88 (closed as duplicate), TC-045 (removed)
    - Action: Update RTM, remove stale references

Trend Analysis:
  Sprint  | Coverage | Orphaned | Stale
  --------|----------|----------|-------
    12    |   94%    |   4.2%   |  1.8%
    13    |   96%    |   3.5%   |  1.5%
    14    |   96%    |   3.1%   |  2.1%
    15    |   98%    |   2.8%   |  2.0%  ← Current

  Prediction: Next sprint coverage 98-99% (in control, improving)
```

---

## Implementation Guidance

### GitHub Traceability Patterns

#### Level 2: Manual Linking

**Commit messages**:
```bash
git commit -m "Implement login endpoint

Addresses #123 (User Authentication requirement)
- Add POST /auth/login endpoint
- Validate credentials against user database
- Return JWT token on success"
```

**PR descriptions**:
```markdown
## Description
Implements user authentication as specified in #123

## Requirements Addressed
- Closes #123 (User Authentication)
- Partially addresses #124 (Session Management)

## Test Coverage
- Added tests/auth/test_login.py
- All tests passing (see CI run)

## Traceability
Requirement: #123 → This PR → tests/auth/test_login.py
```

**Test file comments**:
```python
# tests/auth/test_login.py
"""
Test suite for user authentication

Verifies requirements:
- #123: User Authentication (login with email/password)
- #126: Rate limiting (max 5 attempts per minute)
"""

def test_valid_login():
    """Verifies REQ-123: Users can log in with valid credentials"""
    # ...
```

#### Level 3: Automated RTM Generation

**GitHub API Script** (Python example):
```python
#!/usr/bin/env python3
"""Generate Requirements Traceability Matrix from GitHub"""

import requests
import csv

GITHUB_TOKEN = "your_token"
REPO = "org/repo"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

def get_requirements():
    """Get all issues labeled 'requirement'"""
    url = f"https://api.github.com/repos/{REPO}/issues"
    params = {"labels": "requirement", "state": "all"}
    response = requests.get(url, headers=HEADERS, params=params)
    return response.json()

def get_linked_prs(issue_number):
    """Get PRs that reference this issue"""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/timeline"
    response = requests.get(url, headers=HEADERS)
    events = response.json()
    prs = [e for e in events if e.get("event") == "cross-referenced"]
    return [pr["source"]["issue"]["number"] for pr in prs]

def generate_rtm():
    """Generate RTM CSV file"""
    requirements = get_requirements()
    with open("rtm.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Req ID", "Title", "Status", "Linked PRs", "Test Files", "Coverage"])

        for req in requirements:
            req_id = f"#{req['number']}"
            title = req["title"]
            status = req["state"]
            prs = get_linked_prs(req["number"])

            # Get test files from PR changes
            test_files = []
            for pr in prs:
                pr_data = requests.get(f"https://api.github.com/repos/{REPO}/pulls/{pr}/files", headers=HEADERS).json()
                tests = [f["filename"] for f in pr_data if "test" in f["filename"]]
                test_files.extend(tests)

            coverage = "✅ Full" if prs and test_files else "❌ Incomplete"
            writer.writerow([req_id, title, status, len(prs), len(test_files), coverage])

if __name__ == "__main__":
    generate_rtm()
    print("RTM generated: rtm.csv")
```

**GitHub Actions Workflow** (Automated verification):
```yaml
# .github/workflows/traceability-check.yml
name: Traceability Verification

on:
  pull_request:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  verify-traceability:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check PR links requirements
        run: |
          # Verify PR description mentions requirement issue
          if ! grep -q "#[0-9]\+" <<< "${{ github.event.pull_request.body }}"; then
            echo "❌ PR must reference at least one requirement issue"
            exit 1
          fi

      - name: Check test coverage
        run: |
          # Verify test files exist for requirement
          if ! find tests/ -name "*.py" | grep -q .; then
            echo "⚠️  No test files found - requirement may not be tested"
            exit 1
          fi

      - name: Generate RTM
        run: |
          python3 scripts/generate_rtm.py

      - name: Upload RTM artifact
        uses: actions/upload-artifact@v3
        with:
          name: rtm-report
          path: rtm.csv
```

#### Level 3: GitHub Projects for Visual RTM

**Setup**:
1. Create GitHub Project (Projects tab)
2. Add custom fields:
   - `Requirement ID` (Text - auto-populated from issue number)
   - `Design Link` (Text - URL to ADR or design doc)
   - `Implementation PRs` (Text - list of PR numbers)
   - `Test Coverage` (Single select - Full/Partial/None)
   - `Trace Status` (Single select - Complete/Incomplete/Orphaned)

3. Create views:
   - **Coverage View**: Group by Trace Status
   - **Priority View**: Sort by Priority, show only incomplete traces
   - **Audit View**: Show all fields for reporting

**Benefit**: Visual, real-time RTM that's always up-to-date

### Azure DevOps Traceability Patterns

#### Level 2: Manual Linking

**Commit messages**:
```bash
git commit -m "Implement login endpoint

AB#456  # Auto-links to Work Item #456

- Add POST /auth/login endpoint
- Validate credentials
- Return JWT token"
```

**Work Item Linking**:
- In work item #456, click "Add link" → "Existing item"
- Link type: "Tested By" → Test Case TC-001
- Link type: "Implemented By" → Commit abc1234

#### Level 3: Built-in Traceability Views

**Azure DevOps provides built-in traceability**:

1. **Work Item View** → "Links" tab shows:
   - Related requirements
   - Commits (via AB# references)
   - Pull requests
   - Test cases
   - Build runs

2. **Query for RTM**:
```sql
SELECT
  [System.Id],
  [System.Title],
  [System.State],
  [System.AssignedTo],
  [Custom.DesignDocument],
  [Microsoft.VSTS.Common.TestCases]
FROM WorkItems
WHERE
  [System.WorkItemType] = 'Requirement'
  AND [System.AreaPath] = 'MyProject'
ORDER BY [System.Id]
```

3. **Export to Excel**:
   - Run query
   - Click "Export to Excel"
   - RTM ready for audit

#### Level 4: OData Queries for Metrics

**Traceability Coverage Query**:
```odata
https://analytics.dev.azure.com/{org}/{project}/_odata/v3.0-preview/WorkItems?
  $filter=WorkItemType eq 'Requirement'
  &$expand=Links($filter=LinkTypeName eq 'Tested By')
  &$select=WorkItemId,Title,State,Links
  &$compute=
    Links/$count as TestCount,
    iif(Links/$count gt 0, 1, 0) as IsTested
```

**Dashboard Widget** (PowerBI or Azure DevOps):
```
Traceability Coverage:
- Total Requirements: 150
- Requirements with Tests: 147 (98%)
- Requirements without Tests: 3 (2%)

Orphaned Tests:
- Total Tests: 250
- Tests linked to Requirements: 243 (97.2%)
- Orphaned Tests: 7 (2.8%)
```

---

## Common Anti-Patterns

### Traceability Theater

**Symptom**: RTM exists but never updated, becomes stale within days

**Example**:
```
Last Updated: 2025-10-15
Current Date: 2026-01-24 (3 months outdated)

REQ-045: Status shows "In Progress" but feature shipped 2 months ago
REQ-072: Linked to PR#123 which was closed/deleted
```

**Impact**: RTM useless for audit, impact analysis, or decision-making

**Solution**:
- **Automate**: Use platform-native linking, generate RTM on-demand
- **Don't maintain separate artifact**: Query from source of truth (GitHub, Azure DevOps)
- **Weekly verification**: Automated check for stale links, missing traces

### Manual Spreadsheet RTM (Level 3+)

**Symptom**: Using Excel spreadsheet for RTM at Level 3 (should use tool-based)

**Why it's an anti-pattern**:
- Manual updates required (2-4 hours per sprint)
- Becomes stale immediately
- No verification of link validity
- Difficult to maintain as project grows

**Solution**:
- Migrate to GitHub Projects or Azure DevOps Queries
- Use API-generated reports (on-demand, always current)
- Reserve spreadsheets for Level 2 only

### Orphaned Tests Not Detected

**Symptom**: Tests exist but not linked to any requirement

**Example**:
```
tests/legacy/test_old_feature.py - 150 lines
- Feature removed 6 months ago
- Tests still running (wasting CI time)
- No one knows if they're still needed
```

**Impact**: Wasted effort, confusion, false sense of coverage

**Solution - Orphaned Test Detection** (Addresses RED phase gap):

**Level 3: Manual Review**
```
Quarterly Test Audit:
1. List all test files
2. For each test, verify link to requirement
3. If no link found → Flag as orphaned
4. Review orphaned tests:
   - Still relevant? → Add requirement link
   - Obsolete? → Delete test
```

**Level 4: Automated Detection**
```python
# Detect orphaned tests
def find_orphaned_tests():
    all_tests = glob.glob("tests/**/*.py", recursive=True)
    requirements = get_requirements_from_github()
    req_numbers = [r["number"] for r in requirements]

    orphaned = []
    for test_file in all_tests:
        content = open(test_file).read()
        # Check if test references any requirement (via #123 pattern)
        refs = re.findall(r"#(\d+)", content)
        if not any(int(ref) in req_numbers for ref in refs):
            orphaned.append(test_file)

    print(f"Orphaned tests: {len(orphaned)}")
    for test in orphaned:
        print(f"  - {test}")
```

### Stale Link Detection Missing

**Symptom**: RTM links to deleted PRs, closed issues, removed files

**Impact**: RTM shows false traceability, can't verify actual coverage

**Solution - Stale Link Verification** (Addresses RED phase gap):

```python
def verify_links():
    """Check if linked artifacts still exist"""
    rtm = load_rtm()
    stale_links = []

    for req in rtm:
        # Check if PRs exist
        for pr in req["prs"]:
            if not pr_exists(pr):
                stale_links.append(f"REQ-{req['id']}: PR#{pr} deleted")

        # Check if test files exist
        for test in req["tests"]:
            if not os.path.exists(test):
                stale_links.append(f"REQ-{req['id']}: {test} missing")

    if stale_links:
        print(f"⚠️  {len(stale_links)} stale links detected:")
        for link in stale_links:
            print(f"  - {link}")
        return False
    return True
```

---

## Verification & Validation

**Traceability complete when**:

**Level 2**:
- ✅ RTM exists (spreadsheet or platform links)
- ✅ Forward traceability >70% (requirements → tests)
- ✅ Updated at major milestones

**Level 3**:
- ✅ Tool-based traceability (GitHub/Azure DevOps)
- ✅ Forward traceability >90%
- ✅ Backward traceability >85%
- ✅ Automated weekly verification
- ✅ Orphaned tests identified

**Level 4**:
- ✅ Coverage metrics >95% (forward and backward)
- ✅ Orphaned tests <5%
- ✅ Stale links <2%
- ✅ Automated daily verification with dashboards
- ✅ Trend analysis shows improving coverage

---

## Related Practices

**Before traceability**:
- `requirements-specification.md` - Document requirements with unique IDs
- Establish version control and issue tracking

**During implementation**:
- Use linking conventions in commits, PRs, test files
- Review traceability in code reviews

**For advanced implementation**:
- See `platform-integration` skill for GitHub/Azure DevOps advanced patterns
- See `quantitative-management` skill for Level 4 metrics and SPC

**Prescription reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.1.2 (Requirements Management - SP 1.4 Maintain Bidirectional Traceability)
