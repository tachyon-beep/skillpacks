# Reference Sheet: Configuration Management in GitHub

## Purpose & Context

Implements CMMI **CM (Configuration Management)** using GitHub branch protection, CODEOWNERS, and release management.

**CMMI Process Area**: CM (Configuration Management)
**Platforms**: GitHub Repositories

---

## CMMI Maturity Scaling

**Level 2**: Basic branch protection, manual reviews, tag-based releases
**Level 3**: CODEOWNERS enforcement, automated baselines, standardized workflows
**Level 4**: Metrics on branch health, merge frequency, baseline stability

---

## Branch Protection (Level 3 Requirements)

**Settings → Branches → Add rule**:

```yaml
Pattern: main
Require pull request reviews: 2 reviewers
Dismiss stale reviews: Yes
Require status checks: Yes
  - build-and-test
  - code-quality
Require branches up-to-date: Yes
Require conversation resolution: Yes
Include administrators: Yes
```

**As code** (Terraform):

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.repo.node_id
  pattern       = "main"
  
  required_pull_request_reviews {
    required_approving_review_count = 2
    dismiss_stale_reviews          = true
  }
  
  required_status_checks {
    strict = true
    contexts = [
      "build-and-test",
      "code-quality"
    ]
  }
  
  enforce_admins = true
}
```

---

## CODEOWNERS File

**File**: `.github/CODEOWNERS`

```plaintext
# Default owners for everything
* @org/core-team

# Backend code requires backend team review
/src/backend/ @org/backend-team
/api/ @org/backend-team

# Frontend code requires frontend team review
/src/frontend/ @org/frontend-team
/ui/ @org/frontend-team

# Infrastructure changes require DevOps review
/infrastructure/ @org/devops-team
/.github/workflows/ @org/devops-team
/terraform/ @org/devops-team

# Security-sensitive files require security team
/src/auth/ @org/security-team
/config/secrets/ @org/security-team

# Documentation can be reviewed by anyone
/docs/ @org/core-team
```

**Enforcement**: Enable "Require review from Code Owners" in branch protection

---

## Git Workflow Comparison

| Workflow | Best For | Pros | Cons | CMMI Fit |
|----------|----------|------|------|----------|
| **GitFlow** | Release-based products | Clear release process, hotfix support | Complex, many branches | ✅ Level 3/4 |
| **GitHub Flow** | Continuous deployment | Simple, fast | Limited release control | ✅ Level 2/3 |
| **Trunk-Based** | High-velocity teams | Fastest integration | Requires discipline | ✅ Level 3/4 |

**Recommendation for CMMI Level 3**: GitHub Flow with release branches

---

## Baseline Management

**Create baseline** (milestone freeze):

```bash
# Tag release baseline
git tag -a v1.2.0 -m "Release 1.2.0 baseline"
git push origin v1.2.0

# Create GitHub Release
gh release create v1.2.0 \
  --title "Release 1.2.0" \
  --notes "Baseline for Sprint 12" \
  --target main
```

**Lock baseline** (protect tag):

```yaml
# .github/workflows/protect-tags.yml
name: Protect Release Tags

on:
  push:
    tags:
      - 'v*'

jobs:
  protect:
    runs-on: ubuntu-latest
    steps:
      - name: Prevent tag deletion
        run: |
          # Tags starting with 'v' are protected baselines
          echo "Tag created: ${{ github.ref_name }}"
          # Configure tag protection via API (requires Enterprise)
```

---

## Merge Strategies

**Squash** (recommended for Level 3):
- Clean history (one commit per PR)
- Easier to revert features
- Simpler bisect for debugging

**Merge commit**:
- Preserves full PR history
- Better for audit trail
- More complex history

**Rebase**:
- Linear history
- Loses PR merge point
- Can complicate traceability

**Configuration**: Settings → Pull Requests → Allow squash merging (only)

---

## Emergency Hotfix Procedure

**Level 3 hotfix workflow**:

1. Create hotfix branch from production tag
2. Implement fix with expedited review (1 reviewer)
3. Fast-track through CI/CD
4. Deploy to production
5. **Post-facto review** within 24 hours
6. Backport to main

```bash
# Create hotfix
git checkout -b hotfix/critical-bug v1.2.0
# Fix + commit
git tag v1.2.1
git push origin v1.2.1 hotfix/critical-bug

# Deploy (expedited approval)

# Backport to main
git checkout main
git cherry-pick <hotfix-commit>
```

---

## Related Practices

- `./github-requirements.md` - Requirement baselines with milestones
- `./github-quality-gates.md` - Build validation for branch protection
- `../design-and-build/SKILL.md` - CM process definitions

---

**Last Updated**: 2026-01-25
