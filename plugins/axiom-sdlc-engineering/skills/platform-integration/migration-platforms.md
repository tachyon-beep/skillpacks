---
parent-skill: platform-integration
reference-type: migration-guidance
load-when: Migrating between GitHub and Azure DevOps, switching platforms, consolidating toolchains
---

# Platform Migration: GitHub ↔ Azure DevOps

## Purpose & Context

This reference provides practical guidance for migrating CMMI process implementations between GitHub and Azure DevOps. Whether you're switching platforms, running hybrid environments, or consolidating toolchains, this guide helps you preserve traceability, audit trails, and compliance during migration.

**Common migration scenarios**:
- **GitHub → Azure DevOps**: Startup acquired by enterprise, regulatory compliance requirements, Azure cloud standardization
- **Azure DevOps → GitHub**: Enterprise modernization, OSS contributor attraction, platform consolidation
- **Hybrid**: GitHub for code, Azure DevOps for work tracking (common in regulated industries)

**When to use this reference**: Platform migration planning, tool consolidation, maintaining compliance during transition

---

## Migration Strategy Framework

### Pre-Migration Assessment

**Inventory your CMMI artifacts**:

| Process Area | GitHub Artifacts | Azure DevOps Artifacts |
|--------------|------------------|-------------------------|
| **REQM** | Issues, Projects custom fields, PR links | Work Items, Queries, Test Cases |
| **CM** | Branches, tags, releases, branch protection | Branches, tags, policies, gates |
| **VER** | GitHub Actions workflows, status checks | Azure Pipelines, Test Plans |
| **MA** | Actions workflows (DORA), API scripts | Analytics, PowerBI, OData queries |
| **Audit Trail** | Audit log API, webhook history | Audit log, Activity log, Work item history |

**Data to preserve**:
- ✅ Requirements traceability links (critical for audit)
- ✅ Change history with timestamps (compliance requirement)
- ✅ Review approvals and comments (quality evidence)
- ✅ Metrics baselines (before/after comparison)
- ⚠️ Workflow configurations (must be rewritten for target platform)
- ❌ Platform-specific metadata (usually not transferable)

---

## GitHub → Azure DevOps Migration

### Phase 1: Repository Migration

**Git history migration** (preserves all commits, branches, tags):

```bash
# Clone GitHub repo with full history
git clone --mirror https://github.com/org/repo.git
cd repo.git

# Add Azure DevOps remote
git remote add azure https://dev.azure.com/org/project/_git/repo

# Push all branches and tags
git push azure --all
git push azure --tags

# Verify migration
git remote show azure
```

**Preservation checklist**:
- ✅ All commits with authors and timestamps
- ✅ All branches (including protected branches)
- ✅ All tags and releases
- ✅ Git LFS objects (if used)

**Branch protection migration**:

GitHub branch protection → Azure DevOps branch policies mapping:

| GitHub Setting | Azure DevOps Equivalent |
|----------------|-------------------------|
| Require PR before merge | Branch policy: "Require pull requests" |
| Require 2 approvals | Minimum reviewers: 2 |
| Dismiss stale reviews | Reset reviewer votes: On new push |
| Require status checks | Build validation policy |
| Require branches up to date | Not available (manual merge required) |
| Require signed commits | Not available (third-party extension) |

**Azure DevOps branch policy ARM template**:

```json
{
  "isBlocking": true,
  "isEnabled": true,
  "type": {
    "id": "fa4e907d-c16b-4a4c-9dfa-4906e5d171dd"
  },
  "settings": {
    "minimumApproverCount": 2,
    "creatorVoteCounts": false,
    "allowDownvotes": false,
    "resetOnSourcePush": true,
    "requireVoteOnLastIteration": true,
    "resetRejectionsOnSourcePush": false,
    "blockLastPusherVote": true
  }
}
```

### Phase 2: Work Item Migration

**GitHub Issues → Azure DevOps Work Items**:

**Migration approaches**:
1. **Azure DevOps Data Migration Tool** (Microsoft official, recommended for <10,000 items)
2. **Custom API scripts** (full control, better for complex mappings)
3. **Third-party tools** (7pace Timetracker, Unito, Workato - paid)

**Field mapping**:

| GitHub Issue | Azure DevOps Work Item | Notes |
|--------------|------------------------|-------|
| Title | Title | Direct mapping |
| Description | Description | Direct mapping |
| Labels | Tags | Convert to comma-separated |
| Milestone | Iteration Path | Requires iteration structure |
| Assignee | Assigned To | Map GitHub username to Azure DevOps identity |
| State (open/closed) | State (New/Active/Closed) | May need intermediate states |
| Custom fields (Projects) | Custom fields | Requires process customization |

**Traceability link migration** (critical for CMMI compliance):

```python
# Example: Migrate GitHub Issue → Commit links to Azure DevOps Work Item → Commit links
import requests

# Map GitHub issue numbers to Azure DevOps work item IDs
issue_to_wi_map = {
  "123": "456",  # GitHub #123 → Azure DevOps Work Item 456
  # ... populate from migration tool
}

# Get GitHub commits that reference issues
github_commits = get_commits_with_issue_refs(repo)

for commit in github_commits:
    sha = commit['sha']
    issue_num = extract_issue_number(commit['message'])  # e.g., "fixes #123"

    if issue_num in issue_to_wi_map:
        wi_id = issue_to_wi_map[issue_num]

        # Create Azure DevOps work item link
        create_commit_link(wi_id, sha)
        print(f"Linked Work Item {wi_id} → Commit {sha[:7]}")
```

**Verification**: After migration, run traceability report to ensure all issue→commit links preserved.

### Phase 3: CI/CD Pipeline Migration

**GitHub Actions → Azure Pipelines**:

**Syntax differences**:

| GitHub Actions | Azure Pipelines | Notes |
|----------------|-----------------|-------|
| `on: push` | `trigger: - main` | Branch triggers |
| `runs-on: ubuntu-latest` | `pool: vmImage: 'ubuntu-latest'` | Runner specification |
| `steps: - uses: actions/checkout@v3` | `steps: - checkout: self` | Repository checkout |
| `env:` | `variables:` | Environment variables |
| `${{ secrets.TOKEN }}` | `$(TOKEN)` | Secret syntax |
| `if: github.ref == 'refs/heads/main'` | `condition: eq(variables['Build.SourceBranch'], 'refs/heads/main')` | Conditional execution |

**Migration example**:

**GitHub Actions (before)**:
```yaml
name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
      - name: Build
        run: npm run build
        env:
          NODE_ENV: production
```

**Azure Pipelines (after)**:
```yaml
trigger:
  branches:
    include:
      - main

pr:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - checkout: self

  - script: npm test
    displayName: 'Run tests'

  - script: npm run build
    displayName: 'Build'
    env:
      NODE_ENV: production
```

**Tool-specific migrations**:
- **Secrets**: Manually recreate in Azure DevOps Variable Groups (no API export from GitHub)
- **Environments**: Azure DevOps environments with approval gates
- **Artifacts**: GitHub Packages → Azure Artifacts (requires re-publication)

### Phase 4: Metrics Migration

**DORA metrics continuity**:

**Challenge**: Historical data lives in platform-specific APIs

**Solution**: Export before migration, combine in new dashboard

```python
# Export GitHub DORA metrics (Month 1-6)
github_metrics = {
    "deployment_frequency": calculate_from_github_deployments(),
    "lead_time": calculate_from_github_prs(),
    "change_failure_rate": calculate_from_github_issues(),
    "mttr": calculate_from_github_incident_issues()
}

# After migration: Collect Azure DevOps metrics (Month 7+)
azdo_metrics = {
    "deployment_frequency": calculate_from_azdo_pipelines(),
    "lead_time": calculate_from_azdo_prs(),
    "change_failure_rate": calculate_from_azdo_bugs(),
    "mttr": calculate_from_azdo_incidents()
}

# Combine for trend analysis
combined_dashboard = merge_metrics(github_metrics, azdo_metrics, cutover_date="2026-02")
```

**Baseline preservation** (CMMI Level 4 requirement):
- Export GitHub metrics as CSV before migration
- Import into PowerBI or Azure DevOps Analytics
- Document cutover date in metrics dashboard
- Show pre/post migration trend with annotation

---

## Azure DevOps → GitHub Migration

### Phase 1: Repository Migration

**Git history migration** (same process as GitHub → Azure DevOps, reverse direction):

```bash
# Clone Azure DevOps repo with full history
git clone --mirror https://dev.azure.com/org/project/_git/repo
cd repo.git

# Add GitHub remote
git remote add github https://github.com/org/repo.git

# Push all branches and tags
git push github --all
git push github --tags
```

### Phase 2: Work Item Migration

**Azure DevOps Work Items → GitHub Issues**:

**Field mapping**:

| Azure DevOps Work Item | GitHub Issue | Notes |
|------------------------|--------------|-------|
| Title | Title | Direct mapping |
| Description | Description | HTML → Markdown conversion needed |
| Tags | Labels | Direct mapping |
| Iteration Path | Milestone | Flatten iteration hierarchy |
| Assigned To | Assignee | Map Azure DevOps identity to GitHub username |
| State | State (open/closed) | Collapse intermediate states |
| Area Path | NOT MAPPED | GitHub has no equivalent (use labels) |

**Traceability preservation**:

```python
# Export Azure DevOps Work Item → Commit links
azdo_links = get_work_item_commits(wi_id)

# Map to GitHub Issue numbers (from migration tool)
wi_to_issue_map = {"456": "123"}  # Azure DevOps 456 → GitHub #123

for link in azdo_links:
    commit_sha = link['sha']
    issue_num = wi_to_issue_map[link['wi_id']]

    # Add comment to GitHub issue with commit reference
    create_issue_comment(issue_num, f"Related commit: {commit_sha}")
```

**Lossy migration items**:
- ⚠️ Azure DevOps Test Cases → GitHub Issues (test case structure lost)
- ⚠️ Area Path hierarchy → GitHub labels (flattened)
- ❌ Work item parent/child links → NOT PRESERVED (GitHub has no work item hierarchy)

### Phase 3: CI/CD Pipeline Migration

**Azure Pipelines → GitHub Actions** (see syntax table above, reverse direction)

**Migration tips**:
- Azure DevOps variables → GitHub secrets (manual recreation)
- Azure DevOps environments → GitHub environments
- Azure Pipelines service connections → GitHub Actions secrets

### Phase 4: Metrics Migration

Same approach as GitHub → Azure DevOps, reverse direction. Export Azure DevOps OData queries to CSV, import into GitHub dashboards.

---

## Hybrid Platform Strategy

**Common pattern**: GitHub for code, Azure DevOps for work tracking

**When this makes sense**:
- Regulated industry requires Azure DevOps Test Plans
- Open source contributors expect GitHub
- Enterprise standardized on Azure DevOps, team wants GitHub development experience

**Integration approach**:

```yaml
# GitHub Actions workflow that updates Azure DevOps Work Items
name: Update Work Item on PR Merge

on:
  pull_request:
    types: [closed]
    branches: [main]

jobs:
  update-work-item:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest
    steps:
      - name: Extract work item ID from PR
        id: wi
        run: |
          # Extract "AB#123" from PR title or description
          WI_ID=$(echo "${{ github.event.pull_request.title }}" | grep -oP 'AB#\K\d+')
          echo "work_item_id=$WI_ID" >> $GITHUB_OUTPUT

      - name: Update work item state
        run: |
          curl -X PATCH \
            "https://dev.azure.com/org/project/_apis/wit/workitems/${{ steps.wi.outputs.work_item_id }}?api-version=6.0" \
            -H "Content-Type: application/json-patch+json" \
            -H "Authorization: Bearer ${{ secrets.AZDO_PAT }}" \
            -d '[{"op": "add", "path": "/fields/System.State", "value": "Resolved"}]'
```

**Bidirectional linking**:
- GitHub PRs reference Azure DevOps work items: "Implements AB#123"
- Azure DevOps work items link to GitHub commits: Development Links → External Link

---

## Migration Checklist

### Pre-Migration (Week 1-2)

- [ ] Inventory all CMMI artifacts (requirements, tests, ADRs, metrics)
- [ ] Export historical metrics (DORA, defect density, etc.)
- [ ] Document current traceability coverage baseline
- [ ] Create field mapping document (issues ↔ work items)
- [ ] Set up target platform (Azure DevOps project OR GitHub org/repo)
- [ ] Configure branch policies in target platform
- [ ] Test migration with pilot project/repo

### Migration (Week 3)

- [ ] Migrate Git repositories with full history
- [ ] Migrate work items/issues with traceability links
- [ ] Migrate CI/CD pipelines (rewrite for target platform)
- [ ] Migrate secrets and variables (manual recreation)
- [ ] Configure branch protection/policies
- [ ] Set up audit logging

### Post-Migration (Week 4)

- [ ] Verify traceability coverage (target ≥ 95% of pre-migration baseline)
- [ ] Run DORA metrics for Week 1 post-migration (establish new baseline)
- [ ] Train team on new platform workflows
- [ ] Update CMMI process documentation with new tool references
- [ ] Archive old platform (read-only for audit)
- [ ] Schedule 30-day retrospective

---

## Common Pitfalls

| Mistake | Why It Fails | Better Approach |
|---------|--------------|-----------------|
| "Big bang migration Friday night" | Downtime risk, panic if issues occur | Phased migration: repos first, then work items, then pipelines |
| "We'll rebuild traceability later" | Never happens, audit failure | Migrate traceability links as part of work item migration |
| "Just export to CSV and manually re-enter" | Lossy, time-consuming, error-prone | Use API-driven migration scripts for repeatability |
| "One-way migration, shut down old platform immediately" | Lose access to historical data | Keep old platform read-only for 6-12 months |
| "Migrate everything including 5 years of closed issues" | Wastes time, clutters new platform | Migrate active work only, archive old data |

---

## Compliance Preservation

**Audit trail continuity**:

| Requirement | Solution |
|-------------|----------|
| **FDA 21 CFR Part 11** | Maintain read-only access to old platform audit logs for entire retention period (typically 7-21 years) |
| **SOC 2** | Export audit logs before migration, include in compliance evidence package |
| **ISO 27001** | Document migration in change management record, verify traceability integrity |
| **GDPR** | Anonymize or delete personal data if required, document in DPA |

**Evidence package for auditors**:
1. Pre-migration traceability coverage report
2. Post-migration traceability coverage report
3. Migration script logs (proof of automation)
4. Traceability verification test results
5. Old platform archive access procedure

---

## Related Practices

- **requirements-traceability.md**: How to set up RTM in target platform
- **github-audit-trail.md**: GitHub-specific compliance setup
- **azdo-audit-trail.md**: Azure DevOps-specific compliance setup
- **lifecycle-adoption.md**: Managing the transition (Reference Sheet 7)

---

**Last Updated**: 2026-01-25
**Review Schedule**: Before platform migrations, after migration retrospectives
