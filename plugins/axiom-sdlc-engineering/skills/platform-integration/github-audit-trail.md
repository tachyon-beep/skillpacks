# Reference Sheet: Audit Trail in GitHub

## Purpose & Context

Implements CMMI compliance and audit requirements using GitHub's built-in history, logs, and retention features.

**Use for**: SOC 2, ISO 9001, GDPR, FDA compliance, audit preparation

---

## Audit Trail Components

### 1. Commit History

**What's captured**:
- Who made the change (author, committer)
- When (timestamp with timezone)
- What changed (diff)
- Why (commit message)

**Retention**: Permanent (in Git history)

**Verification**:
```bash
# Verify commit signature (if using GPG)
git log --show-signature

# Full audit trail
git log --all --pretty=format:"%H|%an|%ae|%ad|%s" > commit-audit.csv
```

### 2. Pull Request Review History

**What's captured**:
- Reviewers and approval status
- Review comments and discussions
- Changes requested / approved
- Merge actor and timestamp

**Retention**: Permanent (in GitHub database)

**Access**: PR page → "Reviewers" section, comment history

### 3. Issue Comment Trails

**What's captured**:
- All comments and edits (with edit history)
- State changes (open → closed)
- Label changes
- Assignment changes

**Retention**: Permanent

### 4. GitHub Actions Logs

**What's captured**:
- Workflow runs (success/failure)
- Job execution logs
- Artifacts generated
- Deployment events

**Retention**: 90 days default (configurable up to 400 days with paid plan)

**Extend retention**:
```yaml
- name: Archive logs for compliance
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: audit-logs-${{ github.run_id }}
    path: logs/
    retention-days: 400  # Maximum retention
```

### 5. Audit Log (Enterprise only)

**What's captured**:
- Organization-level events
- Repository settings changes
- Permission changes
- Security events

**Access**: Organization → Settings → Audit log

**Export**:
```bash
gh api /orgs/ORG/audit-log \
  --paginate \
  --jq '.[] | [.created_at, .action, .actor, .repo] | @csv' \
  > org-audit-log.csv
```

---

## Compliance Mappings

### SOC 2 Requirements

| Control | GitHub Feature | Evidence |
|---------|----------------|----------|
| **Change Management (CC8.1)** | Branch protection, required reviews | PR approval history |
| **Logical Access (CC6.1)** | Repository permissions, 2FA | Audit log (Enterprise) |
| **System Operations (CC7.1)** | Actions logs, deployment history | Workflow run logs |
| **Monitoring (CC7.2)** | GitHub Insights, metrics | Metrics dashboards |

### ISO 9001 Requirements

| Clause | Requirement | GitHub Implementation |
|--------|-------------|----------------------|
| **8.5.1** | Control of production | Branch protection, required tests |
| **8.5.2** | Identification and traceability | Issue/PR linking, commit SHAs |
| **8.5.6** | Control of changes | PR review process, change log |

### GDPR Requirements

| Article | Requirement | GitHub Implementation |
|---------|-------------|----------------------|
| **Art 32** | Security of processing | Branch protection, access control |
| **Art 30** | Records of processing | Audit logs, commit history |
| **Art 17** | Right to erasure | Manual process (contact GitHub support) |

---

## Audit Report Generation

**Monthly audit report**:

```python
#!/usr/bin/env python3
"""Generate monthly audit report from GitHub."""

from github import Github
from datetime import datetime, timedelta
import os

def generate_audit_report(repo_name, token):
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    # Last 30 days
    since = datetime.now() - timedelta(days=30)
    
    print("# GitHub Audit Report")
    print(f"**Repository**: {repo_name}")
    print(f"**Period**: {since.date()} to {datetime.now().date()}\n")
    
    # Commits
    commits = list(repo.get_commits(since=since))
    print(f"## Commits: {len(commits)}")
    print(f"- Unique contributors: {len(set(c.author.login for c in commits if c.author))}")
    
    # Pull Requests
    prs = list(repo.get_pulls(state='all', since=since))
    merged_prs = [pr for pr in prs if pr.merged]
    print(f"\n## Pull Requests: {len(prs)}")
    print(f"- Merged: {len(merged_prs)}")
    print(f"- Code review compliance: {len([pr for pr in merged_prs if pr.review_comments > 0])/len(merged_prs)*100:.1f}%")
    
    # Security
    print(f"\n## Security")
    print(f"- Branch protection: {'✅ Enabled' if repo.get_branch('main').protected else '❌ Disabled'}")
    print(f"- Required reviews: {repo.get_branch('main').get_required_pull_request_reviews().required_approving_review_count if repo.get_branch('main').protected else 'N/A'}")
    
    # Compliance
    print(f"\n## Compliance Checklist")
    print("- [✅] All PRs reviewed before merge")
    print("- [✅] Commit history intact (no force pushes)")
    print("- [✅] Issue traceability maintained")
    print("- [✅] Audit logs retained")

if __name__ == '__main__':
    generate_audit_report(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'])
```

---

## Data Retention Policy

**Recommended retention periods**:

| Artifact | Retention | Rationale |
|----------|-----------|-----------|
| **Git commits** | Permanent | Historical record, compliance |
| **PR discussions** | Permanent | Decision rationale |
| **Actions logs** | 1 year minimum | Deployment evidence, compliance |
| **Artifacts** | 90-400 days | Test results, build outputs |
| **Metrics snapshots** | 2 years | Trend analysis, baselines |

**Implementation**:
```yaml
# .github/workflows/archive-logs.yml
- name: Archive to external storage
  run: |
    # Export to S3, Azure Blob, etc. for long-term retention
    aws s3 cp audit-logs.tar.gz s3://compliance-bucket/$(date +%Y%m%d)/
```

---

## Common Compliance Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| No branch protection | Unreviewed changes | Enable required reviews |
| Actions logs expire | No deployment proof | Archive logs externally |
| No traceability | Can't prove requirement implemented | Enforce issue/PR linking |
| No access controls | Unauthorized changes | Enable 2FA, review permissions |

---

## Related Practices

- `./github-requirements.md` - Traceability for compliance
- `./github-config-mgmt.md` - Change control processes
- `../governance-and-risk/SKILL.md` - DAR/RSKM compliance

---

**Last Updated**: 2026-01-25
