---
parent-skill: lifecycle-adoption
reference-type: retrofitting-guidance
load-when: Adopting branching strategies mid-project, adding version control discipline
---

# Retrofitting Configuration Management Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Adding CM practices to existing projects, migrating to new branching strategy, enforcing version control

This reference provides guidance for retrofitting configuration management (CM) onto existing codebases.

---

## Reference Sheet 4: Retrofitting Configuration Management

### Purpose & Context

**What this achieves**: Adopt branching workflow and CM practices mid-project without chaos

**When to apply**:
- Git workflow is chaotic (force pushes, merge conflicts, lost work)
- Need to establish branching strategy on active project
- Team frustrated with current CM practices

**Prerequisites**:
- Git repository exists (not migrating version control systems)
- Team willing to learn new workflow
- Management supports brief transition period

### CMMI Maturity Scaling

#### Level 2: Managed (Retrofitting CM)

**Approach**: Minimal viable workflow immediately

**Practices to Adopt**:
- Branch protection (prevent force pushes to main)
- Feature branches (isolate work)
- PR workflow (review before merge)
- Basic release tagging

**Timeline**: 1-2 days to implement, 1-2 weeks to stabilize

**Training**: 1-hour Git workshop

#### Level 3: Defined (Retrofitting CM)

**Approach**: Documented workflow with organizational standard

**Practices to Adopt** (beyond Level 2):
- GitFlow or GitHub Flow (documented strategy)
- Branching naming convention
- Merge policies (squash vs. merge commit)
- Release management process
- CODEOWNERS file

**Timeline**: 1 week to implement, 1 month to fully adopt

**Training**: 2-hour workshop + written docs

#### Level 4: Quantitatively Managed (Retrofitting CM)

**Approach**: Metrics-driven CM with statistical monitoring

**Practices to Adopt** (beyond Level 3):
- Branch metrics (age, size, merge time)
- Merge conflict rate tracking
- Statistical baselines for CM health
- Automated workflow enforcement

**Timeline**: 2-3 weeks to implement metrics

**Training**: Metrics interpretation

### Implementation Guidance

#### Quick Start Checklist

**Step 1: Assess Current CM Chaos** (1 hour)

Identify pain points:
- [ ] Force pushes causing lost work?
- [ ] Merge conflicts daily/weekly?
- [ ] No clear release process?
- [ ] Difficulty tracking what's in production?
- [ ] Team members working on same files without coordination?

**Step 2: Choose Workflow** (30 minutes)

| Workflow | Best For | Complexity |
|----------|----------|------------|
| **GitHub Flow** | Continuous deployment, small teams | Low |
| **GitFlow** | Scheduled releases, larger teams | Medium |
| **Trunk-Based** | Very frequent deploys, mature CI/CD | Low (but requires discipline) |

**Recommendation for retrofitting**: Start with **GitHub Flow** (simplest), evolve to GitFlow if needed.

**Step 3: Enable Branch Protection** (15 minutes)

GitHub:
- Settings → Branches → Add rule for `main`
- ✅ Require pull request before merging
- ✅ Require approvals (1-2 reviewers)
- ✅ Dismiss stale approvals when new commits pushed
- ✅ Require status checks to pass (CI)
- ✅ Do not allow bypassing (even for admins)

Azure DevOps:
- Repos → Branches → `main` → Branch Policies
- ✅ Require minimum 1-2 reviewers
- ✅ Check for linked work items
- ✅ Build validation (CI pipeline must pass)

#### 21. Fork Security Configuration (Prevents: Fork-based bypass attack)

**Problem**: Developer forks repository, disables branch protection in fork, force-pushes from fork/main back to origin/main to bypass branch protection rules.

**Exploit scenario**:
1. Friday 5 PM deadline, PR reviews take too long
2. Developer forks repository to personal account
3. Disables all protection rules in fork (allowed since it's their repo)
4. Implements changes, force-pushes within fork
5. Force-pushes from fork/main to origin/main OR creates fork PR and immediately merges
6. Bypasses branch protection entirely

**Enforcement for GitHub**:

**Repository Settings**:
- Settings → Branches → `main` branch protection
  - ✅ **Restrict who can push to matching branches** (critical setting)
  - Select: Specific teams/users only (not "Everyone")
  - This prevents force-push from forks to protected branch

  - ✅ **Require linear history**
  - Prevents force-push merges (rejects non-fast-forward pushes)

- Settings → Actions → General
  - ✅ **Require approval for all outside collaborators** (for fork PRs)
  - Prevents fork-based auto-merge

**GitHub Organizati settings** (if using organization):
- Settings → Member privileges
  - ✅ **Base permissions: Read** (not Write)
  - Prevents fork-based push by default
  - Grant Write permissions explicitly per repository

**Enforcement for Azure DevOps**:

- Repos → Branches → `main` → Security
  - ✅ **Remove "Force Push" permission for ALL users** (including Project Administrators)
  - ✅ **Remove "Bypass policies when pushing"** (including admins)

- Repos → Security → Permissions
  - ✅ Limit "Contribute" permission to specific teams
  - Deny "Force push" and "Remove Others' Locks"

**Audit & Detection**:

**Weekly automated audit** (GitHub Action):
```yaml
name: Audit Branch Protection & Force Push Attempts
on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday 9 AM

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - name: Check branch protection settings
        run: |
          # Verify "Restrict who can push" is enabled
          PROTECTION=$(gh api repos/${{ github.repository }}/branches/main/protection)

          ENFORCE_ADMINS=$(echo "$PROTECTION" | jq -r '.enforce_admins.enabled')
          if [ "$ENFORCE_ADMINS" != "true" ]; then
            echo "ERROR: Admin bypass is enabled!"
            exit 1
          fi

          RESTRICTIONS=$(echo "$PROTECTION" | jq -r '.restrictions')
          if [ "$RESTRICTIONS" == "null" ]; then
            echo "ERROR: No push restrictions configured - fork bypass possible!"
            exit 1
          fi

      - name: Check for force-push attempts
        run: |
          # Scan git reflog for force-push attempts in last 7 days
          git fetch --all
          FORCE_PUSHES=$(git reflog --all --since="1 week ago" | grep -E 'forced|reset --hard' || true)
          if [ -n "$FORCE_PUSHES" ]; then
            echo "WARNING: Force-push attempts detected in last 7 days:"
            echo "$FORCE_PUSHES"
            # Alert security team
            curl -X POST ${{ secrets.SLACK_WEBHOOK }} -d "{\"text\":\"Force-push attempt detected on main branch\"}"
          fi
```

**Monthly manual audit**:
- Review GitHub audit log: Settings → Security → Audit log
- Filter by: "Branch protection rule" actions
- Check for:
  - Branch protection disabled
  - "Restrict who can push" settings changed
  - Force-push permission grants
- Any changes require documented justification

**Command to check for force-push attempts** (run locally):
```bash
# Check git reflog for force-push attempts in last 30 days
git fetch --all --reflog
git reflog --all --date=iso | grep -E 'forced update|reset --hard' | grep -v '30 days ago'

# Check GitHub audit log (requires gh CLI with admin access)
gh api /repos/{owner}/{repo}/events | jq '.[] | select(.type=="PushEvent" and .payload.forced==true)'
```

**Red flags**:
- Multiple failed force-push attempts from same developer → attempting bypass, investigate
- Force-push attempt from fork → security incident (how did it succeed?)
- Branch protection settings changed → audit log must show justification

**Incident response**:
- If force-push detected: Immediate code review of forced commits
- If malicious: Revert commits, revoke developer access pending investigation
- If accidental: Developer training on proper workflow

#### 22. Emergency Hotfix Enforcement (Prevents: Disabling branch protection)

**Problem**: Production down at 2 AM Friday. Manager temporarily disables branch protection, developer force-pushes hotfix, manager forgets to re-enable protection. Branch protection remains disabled for days/weeks.

**Critical clarification**: Emergency bypass process (lines 443-472) does NOT mean disabling branch protection. It means FASTER review, not NO review.

**What "emergency" means**:
- Production outage affecting users
- Active security vulnerability being exploited
- Data loss in progress

**What "emergency" does NOT allow**:
- ❌ Temporarily disabling branch protection rules
- ❌ Force pushing to protected branches
- ❌ Merging without ANY review
- ❌ Skipping CI/CD pipeline entirely
- ❌ Admin override of protection settings

**Emergency hotfix process (ALL steps required, even at 2 AM)**:

1. **Create hotfix branch** (not push to main):
   ```bash
   git checkout main
   git pull
   git checkout -b hotfix/prod-outage-description
   ```

2. **Implement fix, push hotfix branch**:
   ```bash
   # Make changes
   git add .
   git commit -m "[EMERGENCY] Fix production outage - description"
   git push -u origin hotfix/prod-outage-description
   ```

3. **Create PR with [EMERGENCY] prefix**:
   - Title: `[EMERGENCY] Fix production database connection timeout`
   - Body: Include severity, impact, affected users, root cause

4. **Request oncall reviewer** (4-hour SLA):
   - Slack: `@oncall-reviewer EMERGENCY PR #XXX needs immediate review`
   - PagerDuty: Trigger oncall engineer if reviewer not responding within 1 hour
   - Reviewer can approve from phone via GitHub mobile app

5. **Reviewer approval within 4 hours**:
   - Async acceptable (reviewer in different timezone)
   - Substantive review still required (not rubber-stamp)
   - Check: Does fix address root cause? Any obvious issues?

6. **Merge via PR** (branch protection STAYS ENABLED):
   - Use GitHub's "Squash and merge" or "Merge commit"
   - Branch protection enforces this path
   - NO force push, NO direct commit to main

7. **Post-hotfix documentation** (within 24 hours):
   - Create GitHub Issue documenting incident
   - Write ADR if architectural decision was made under pressure
   - Add tests for the bug (prevent regression)
   - Mark issue with `emergency-bypass` label

8. **Monthly review**: All emergency bypasses reviewed in retrospective
   - >2 emergencies per month = process problem OR abuse
   - Root cause: Are emergencies preventable with better monitoring?

**Branch protection audit alerts**:

**Real-time alert** (GitHub webhook → Slack/PagerDuty):
```python
# GitHub webhook endpoint (hosted on your infrastructure)
@app.route('/github-webhook', methods=['POST'])
def github_webhook():
    event = request.headers.get('X-GitHub-Event')
    payload = request.json

    # Detect branch protection changes
    if event == 'branch_protection_rule' and payload['action'] in ['deleted', 'edited']:
        branch = payload['rule']['name']
        actor = payload['sender']['login']

        if branch == 'main':
            # IMMEDIATE ALERT - branch protection changed on main
            send_alert_to_cto(
                severity='CRITICAL',
                message=f'Branch protection for main was {payload["action"]} by {actor}',
                action='Verify this change was authorized. If not, revert immediately.'
            )

            # If deleted (disabled), this is a security incident
            if payload['action'] == 'deleted':
                create_security_incident(
                    title='Branch protection disabled on main',
                    assignee='security-team'
                )

    return 'OK', 200
```

**Weekly verification** (GitHub Action):
```yaml
name: Verify Branch Protection Never Disabled
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - name: Check branch protection is enabled
        run: |
          PROTECTION=$(gh api repos/${{ github.repository }}/branches/main/protection 2>&1)

          if echo "$PROTECTION" | grep -q "Not Found"; then
            echo "CRITICAL: Branch protection is DISABLED on main!"
            echo "Re-enabling immediately..."

            # Auto-remediate: Re-enable branch protection with standard settings
            gh api -X PUT repos/${{ github.repository }}/branches/main/protection \
              -f "required_pull_request_reviews[required_approving_review_count]=1" \
              -f "required_status_checks[strict]=true" \
              -f "enforce_admins=true" \
              -f "restrictions=null"

            # Alert CTO
            curl -X POST ${{ secrets.SLACK_WEBHOOK_CTO }} \
              -d "{\"text\":\"CRITICAL: Branch protection was disabled on main. Auto-remediated, but investigate who disabled it.\"}"

            exit 1
          fi
```

**Post-incident review**:
- If branch protection was disabled: Mandatory post-mortem
  - Who disabled it?
  - Why? (claimed reason)
  - Was there an actual emergency? (verify)
  - How long was it disabled? (audit log timestamps)
  - What commits were pushed while disabled? (git log review)
  - Consequences: Training, policy update, or access revocation

**Red flags**:
- Branch protection disabled outside of business hours → suspicious (2 AM Friday)
- Branch protection disabled for >1 hour → forgot to re-enable
- Same person disables protection >2 times in 6 months → pattern of bypass abuse
- Force-push immediately after protection disabled → deliberate bypass

**Enforcement**:
- Automated remediation: Protection auto-re-enables within 6 hours (GitHub Action)
- Manual audit: CTO reviews ALL protection changes monthly
- Consequence: Disabling branch protection without documented justification = security incident
- Access revocation: 2nd unauthorized disable = lose admin access

**Cost**: 4 hours to set up webhooks + GitHub Actions (one-time), 30 minutes/month for audit review

---

**Summary: Git Workflow Enforcement**

These 2 mechanisms close CRITICAL security loopholes:

1. **Fork security**: Prevents force-push from fork to protected branch (restrict push permissions, require linear history, audit force-push attempts)
2. **Emergency bypass prevention**: Branch protection NEVER gets disabled (real-time alerts, auto-remediation, post-incident review)

**Impact**: Eliminates the 2 trivial bypasses that undermine all other Git workflow enforcement.

**Step 4: Migrate Existing Work** (2-4 hours)

**Current state**: Team working directly on `main` or chaotic branches

**Migration strategy**:

1. **Announce cutover**: "Starting Monday, all work via feature branches"
2. **Clean up main**:
   - Merge any pending work-in-progress
   - Create release tag for current state: `v1.0-pre-workflow`
3. **Create first feature branch**: Lead by example
   ```bash
   git checkout main
   git pull
   git checkout -b feature/update-readme
   # Make changes
   git push -u origin feature/update-readme
   # Create PR on GitHub/Azure DevOps
   ```
4. **First PR walkthrough**: Team observes, learns process

**Step 5: Document Workflow** (1-2 hours)

Create `docs/git-workflow.md`:

```markdown
# Git Workflow

## Branching Strategy: GitHub Flow

### For New Features/Bug Fixes

1. **Start from latest main**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/short-description
   # OR for bugs:
   git checkout -b fix/bug-short-description
   ```

3. **Make changes, commit frequently**:
   ```bash
   git add .
   git commit -m "Add feature X"
   git push -u origin feature/short-description
   ```

4. **Create Pull Request**:
   - Navigate to GitHub/Azure DevOps
   - Create PR from your branch to `main`
   - Fill in PR template
   - Request 1-2 reviewers

5. **Address review feedback**:
   - Make changes
   - Push to same branch (PR updates automatically)

6. **Merge after approval**:
   - Squash and merge (keeps history clean)
   - Delete branch after merge

### Release Process

- Tag releases from `main`: `git tag -a v1.2.0 -m "Release 1.2.0"`
- Push tags: `git push origin v1.2.0`
- Create GitHub Release with changelog

### Branch Naming Convention

- Features: `feature/short-description`
- Bug fixes: `fix/short-description`
- Hotfixes: `hotfix/critical-issue`
- Experiments: `experiment/what-you-are-trying`
```

**Step 6: Team Training** (1 hour workshop)

Agenda:
- Explain why (Git chaos → structured workflow)
- Demo the workflow (live coding)
- Practice exercise (everyone creates a feature branch, PR, review, merge)
- Q&A

**Step 7: Enforce and Monitor** (Ongoing)

- Week 1: Coach each PR individually
- Week 2: Spot checks (are people following workflow?)
- Week 3-4: Collect feedback, adjust if needed
- Month 2+: Workflow becomes second nature

#### Templates & Examples

**PR Template** (`.github/pull_request_template.md`):

```markdown
## Description

[Brief description of changes]

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work as expected)

## How Has This Been Tested?

[Describe testing: unit tests, manual testing, etc.]

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented hard-to-understand code
- [ ] I have updated documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests covering my changes
- [ ] All tests pass locally

## Related Issues

Closes #[issue number]
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Force Workflow Overnight** | Team revolts, workarounds created | Announce 1 week ahead, provide training |
| **No Branch Protection** | Workflow optional = not followed | Enforce via GitHub/Azure settings |
| **Complex Workflow First** | GitFlow too complex for team new to branching | Start simple (GitHub Flow), evolve later |
| **No Documentation** | Team forgets workflow after training | Written docs + PR template reinforcement |
| **Admin Bypass** | Managers force-push "just this once" → culture broken | No exceptions, not even for admins |

### Tool Integration

**GitHub**:
- **Branch protection**: Settings → Branches → Add rule
- **PR templates**: `.github/pull_request_template.md`
- **CODEOWNERS**: `.github/CODEOWNERS` (auto-assign reviewers)
- **Status checks**: Require CI to pass before merge

**Azure DevOps**:
- **Branch policies**: Repos → Branches → Policies
- **PR templates**: Repo settings → Pull request templates
- **Required reviewers**: Branch policies → Automatically include reviewers
- **Build validation**: Branch policies → Build validation

### Verification & Validation

**How to verify workflow is adopted**:
- Spot check: Last 10 merges, all via PR? (GitHub Insights → Pull Requests)
- Zero force pushes to `main` (check Git reflog)
- 100% of PRs have reviews (GitHub/Azure analytics)
- Team feedback: Is workflow helping or hindering?

**Common failure modes**:
- **Workflow abandoned after 2 weeks** → No enforcement, revert to protecting main
- **Team creates many exceptions** → Workflow too rigid, needs tailoring
- **Merge conflicts increase** → Branches too long-lived, encourage smaller PRs

### Related Practices

- **After CM workflow stable**: See design-and-build skill for deeper CM practices
- **If team resists**: See Reference Sheet 8 (Change Management)
- **If tooling migration needed**: See Reference Sheet 7 (Managing the Transition)

---

