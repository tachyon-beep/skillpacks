# Reference Sheet: Configuration Management

## Purpose & Context

This reference sheet provides systematic frameworks for establishing git workflows, branching strategies, and release management. It prevents git chaos through diagnosis and structured migration from chaos to control.

**When to apply**: Daily merge conflicts, force pushes, lost work, unclear branching strategy, release confusion

**Prerequisites**: Team using git (any platform: GitHub, GitLab, Bitbucket, Azure Repos)

---

## CMMI Maturity Scaling

### Level 2: Managed

**Required Practices**:
- Documented branching strategy (chosen and written down)
- Protected main branch (no force pushes)
- Basic pull request workflow
- Release tagging (semantic versioning)

**Work Products**:
- Branching strategy document (wiki, README)
- Git workflow diagram
- Release notes for each version

**Quality Criteria**:
- Team follows documented workflow
- Main branch deployable at all times
- Releases tagged and tracked

**Audit Trail**:
- Git history shows workflow compliance
- PR discussions captured
- Release tags with notes

### Level 3: Defined

**Enhanced Practices**:
- **ADR documenting branching strategy choice**
- Branch protection rules enforced by platform
- Required reviewers (2+ for main)
- CI required to pass before merge
- Formal release process with approval gates
- CODEOWNERS file for component ownership

**Additional Work Products**:
- ADR for branching strategy
- Documented merge strategy (squash, merge commit, rebase)
- Branch protection rules in platform
- Release checklist
- CODEOWNERS file

**Quality Criteria**:
- ADR explains WHY this branching strategy
- Platform enforces workflow (not honor system)
- 100% of changes through PR with review
- Zero force pushes to protected branches
- Releases follow formal process

**Audit Trail**:
- ADR documenting strategy selection
- Platform audit logs show protection rule compliance
- PR review history
- Release approval trail

### Level 4: Quantitatively Managed

**Statistical Practices**:
- Metrics on merge conflicts (frequency, resolution time)
- PR cycle time tracking (commit → merge)
- Branch lifetime analysis (short-lived preferred)
- Release metrics (frequency, rollback rate)

**Quantitative Work Products**:
- Merge conflict metrics dashboard
- PR cycle time trends
- Branch age reports
- Release quality metrics (defects per release, rollback %)

**Quality Criteria**:
- Merge conflicts <X per week (baseline established)
- PR cycle time <4 hours (statistical control limits)
- Branches live <3 days (trunk-based) or <2 weeks (feature branches)
- Release rollback rate <5%

**Audit Trail**:
- All L3 requirements plus quantitative data
- Statistical process control charts
- Trend analysis showing improvement

---

## Implementation Guidance

### Quick Start Checklist: Diagnose Before Prescribing

**CRITICAL**: Don't jump to branching strategy without diagnosis. Git chaos has root causes.

- [ ] **Analyze conflict patterns**: Run `git log --all --graph --oneline --since='1 month ago'` - what files conflict repeatedly?
- [ ] **Identify force push culprits**: Check who force pushes and why (lack of training? desperation?)
- [ ] **Assess team communication**: Are conflicts due to overlapping work on same modules? Poor coordination?
- [ ] **Check module boundaries**: Frequent conflicts in same files suggests tight coupling or poor architecture
- [ ] **Current state baseline**: Measure conflicts/week, force pushes/week, PR cycle time
- [ ] **Team maturity**: Does team understand git rebase, merge, reset? Or need training first?

**Diagnostic outcomes**:
- **Conflicts in same files repeatedly** → Architectural problem (tight coupling), not git problem
- **Force pushes from same person** → Training problem, not workflow problem
- **Long-lived branches** → CI/integration problem, not branching problem
- **Conflicts across unrelated files** → Communication problem (overlapping work)

**Action**: Fix root cause FIRST, then choose branching strategy that fits team maturity and project needs.

---

### Branching Strategy Decision Framework

**Use this framework to select appropriate strategy for your context.**

| Strategy | Team Size | Release Cadence | Maturity | Level |
|----------|-----------|-----------------|----------|-------|
| **GitHub Flow** | 1-15 | Continuous (daily deploys) | Medium | L2-L3 |
| **Git Flow** | 5-50 | Scheduled releases (weekly/monthly) | Medium-High | L3 |
| **Trunk-Based** | Any | Continuous (multiple daily deploys) | High | L3-L4 |

#### GitHub Flow

**When to use**:
- Continuous deployment (every merge → production)
- Web apps, SaaS products
- Team comfortable with small PRs
- Single environment (production)

**Branch structure**:
```
main (production)
  ↳ feature/user-authentication
  ↳ bugfix/login-validation
  ↳ hotfix/security-patch
```

**Workflow**:
1. Create feature branch from main
2. Develop, commit frequently
3. Open PR when ready
4. Reviews, CI passes
5. Merge to main (triggers deploy)
6. Delete feature branch

**Pros**:
- Simple (one long-lived branch: main)
- Fast iteration
- Always deployable

**Cons**:
- No staging/release branches (must deploy to prod)
- Harder for scheduled releases
- Requires good CI/CD

**Level 3 requirements**:
- ADR documenting why GitHub Flow
- Branch protection on main (required reviews, CI)
- Automated deployment on merge

#### Git Flow

**When to use**:
- Scheduled releases (not continuous)
- Multiple environments (dev, staging, prod)
- Need release branches for stabilization
- Compliance/audit requirements

**Branch structure**:
```
main (production releases)
develop (integration branch)
  ↳ feature/new-dashboard
  ↳ feature/api-v2
release/v2.1.0 (stabilization)
hotfix/critical-bug (from main)
```

**Workflow**:
1. Feature branches from develop
2. Merge features to develop
3. Create release branch from develop
4. Stabilize release (bug fixes only)
5. Merge release to main AND develop
6. Tag main with version
7. Hotfixes from main, merge back to develop

**Pros**:
- Clear release management
- Parallel work on features, current release, and prod hotfixes
- Good for enterprise/scheduled releases

**Cons**:
- More complex (multiple long-lived branches)
- Slower (release branches add overhead)
- Merge conflicts more likely

**Level 3 requirements**:
- ADR documenting why Git Flow
- Protection on main AND develop
- Release branch naming convention
- Formal release approval process

#### Trunk-Based Development

**When to use**:
- High-maturity teams
- Continuous deployment multiple times per day
- Feature flags for incomplete work
- Strong CI/CD culture

**Branch structure**:
```
main (trunk)
  ↳ short-lived-branch-1 (<1 day)
  ↳ short-lived-branch-2 (<1 day)
```

**Workflow**:
1. Create short-lived branch from main (<1 day, <200 lines)
2. Small, frequent commits
3. Merge to main multiple times per day
4. Incomplete features behind feature flags
5. CI runs on every commit to main

**Pros**:
- Minimal merge conflicts (small changes, frequent integration)
- Fast feedback (main always integrated)
- Scales to large teams (Google uses this)

**Cons**:
- Requires discipline (small PRs, feature flags)
- Needs excellent CI/CD (comprehensive tests, fast runs)
- Feature flags add complexity

**Level 3-4 requirements**:
- ADR documenting trunk-based choice
- Branch lifetime policy (<1 day)
- Feature flag management
- Comprehensive CI (all tests <15 min)
- Metrics on branch age, PR size

---

### ADR Template for Branching Strategy

```markdown
# ADR-YYYY-MM-DD: Branching Strategy Selection

**Status**: Accepted
**Date**: YYYY-MM-DD
**Authors**: [Engineering Lead, Team]
**CMMI Level**: 3

## Context

**Current state**: [Describe git chaos - conflicts/week, force pushes, pain points]

**Root cause analysis**:
- Conflict pattern: [Same files? Unrelated files? Hotspots?]
- Force push reasons: [Training gap? Desperation? Lack of workflow?]
- Team maturity: [Git expertise level, CI/CD readiness]

**Requirements**:
- Release cadence: [Continuous? Weekly? Monthly?]
- Environments: [Prod only? Dev/Staging/Prod?]
- Team size: [Current and 6-month projection]
- Compliance: [Any audit/regulatory needs?]

## Considered Options

1. **GitHub Flow** (single main branch, continuous deploy)
2. **Git Flow** (main + develop + release branches)
3. **Trunk-Based Development** (main + short-lived branches)

## Decision Outcome

**Chosen**: [Strategy Name]

**Rationale**: [Why this fits team, release cadence, maturity]

### Positive Consequences

- Benefit 1: [Specific to chosen strategy]
- Benefit 2
- Benefit 3

### Negative Consequences (Accepted Tradeoffs)

- Tradeoff 1: [What we're giving up]
- Mitigation: [How we'll address]

## Alternatives Analysis

| Criteria | GitHub Flow | Git Flow | Trunk-Based |
|----------|-------------|----------|-------------|
| Fits release cadence | [Score] | [Score] | [Score] |
| Team maturity match | [Score] | [Score] | [Score] |
| Reduces conflicts | [Score] | [Score] | [Score] |
| CI/CD readiness | [Score] | [Score] | [Score] |

## Implementation Plan

### Phase 1: Preparation (Week 1)
- [ ] Document strategy in wiki
- [ ] Set up branch protection rules
- [ ] Train team on new workflow
- [ ] Baseline current state (conflicts/week, PR cycle time)

### Phase 2: Parallel Run (Week 2)
- [ ] New PRs follow new workflow
- [ ] In-flight work continues old way
- [ ] Monitor for issues
- [ ] Daily standup check-ins

### Phase 3: Full Adoption (Week 3)
- [ ] All work follows new workflow
- [ ] Old branches cleaned up
- [ ] Measure new conflict rate
- [ ] Team retro on workflow

### Rollback Trigger
If conflicts >2x baseline in first 2 weeks, revert and diagnose further.

## Success Metrics

**Baseline** (before change):
- Conflicts per week: __
- Force pushes per week: __
- PR cycle time: __

**Targets** (30 days after):
- Conflicts: 80% reduction
- Force pushes: Zero
- PR cycle time: <4 hours

**Review schedule**: 30-day retro, 90-day assessment

## References

- Git Flow documentation
- GitHub Flow guide
- Trunk-Based Development principles
```

---

### Branch Protection Rules (Platform Configuration)

**Level 3 baseline protection**:

```yaml
# GitHub branch protection settings for 'main'
required_status_checks:
  strict: true  # Require branches up to date before merging
  contexts:
    - ci/build
    - ci/unit-tests
    - ci/integration-tests

required_pull_request_reviews:
  dismissal_restrictions: {}
  dismiss_stale_reviews: true
  require_code_owner_reviews: true
  required_approving_review_count: 2

enforce_admins: true  # Even admins must follow rules
restrictions: null  # No push restrictions (everyone goes through PR)

required_linear_history: false  # Allow merge commits
allow_force_pushes: false  # NEVER allow force push
allow_deletions: false  # Can't delete protected branch
```

**Level 4 enhancements**:
- Required status checks include complexity analysis, security scanning
- Automated metrics collection on PR
- Statistical quality gates (coverage must meet baseline)

---

### Merge Strategy Selection

**Merge commit** (preserve full history):
```bash
git merge --no-ff feature-branch
```
- **Pros**: Complete history, easy to revert entire feature
- **Cons**: Noisy history with merge commits
- **Use when**: Features are logical units, history matters

**Squash merge** (single commit per feature):
```bash
git merge --squash feature-branch
```
- **Pros**: Clean main history, one commit per feature
- **Cons**: Lose intermediate commits
- **Use when**: Main history should be high-level, intermediate commits don't matter

**Rebase** (linear history):
```bash
git rebase main
```
- **Pros**: Linear history, no merge commits
- **Cons**: Rewrites history, dangerous if misused
- **Use when**: Team experienced with git, linear history preferred

**Recommendation for Level 3**: Squash merge. Clean history, low risk.

---

### CODEOWNERS File

**Purpose**: Automatic reviewer assignment based on file paths

**Example**:
```
# Default owner for everything
* @engineering-lead

# Frontend code
/frontend/** @frontend-team @ux-lead

# Backend API
/api/** @backend-team @tech-lead

# Infrastructure
/terraform/** @devops-team @security-lead
/.github/** @devops-team

# Specific critical files
/api/auth/** @security-lead @tech-lead
/database/migrations/** @backend-lead @dba
```

**Level 3 usage**: Required reviewers from CODEOWNERS must approve before merge.

---

### Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Git Chaos** | Daily conflicts, force pushes, lost work | No documented workflow, honor system | Document strategy, enforce with branch protection, train team |
| **Long-Lived Branches** | Branches live weeks/months, massive merge conflicts | Delayed integration, "integration hell" | Short-lived branches (<3 days), frequent merging, trunk-based if mature |
| **Force Push to Main** | Lost commits, broken builds, team panic | Overwrites others' work, no recovery | **NEVER** allow force push to main. Use revert if needed. |
| **No Branch Protection** | Accidental merges, skipped reviews, broken builds | Honor system fails under pressure | Platform enforcement: required reviews, CI, no force push |
| **Branching Without ADR** | Team doesn't understand WHY strategy chosen | Lack of buy-in, non-compliance | Level 3: ADR explaining strategy selection and tradeoffs |
| **Strategy-First, Diagnosis-Second** | Pick GitHub Flow without understanding root causes | Treats symptom, not disease | Diagnose conflicts FIRST (architecture? communication?), then choose strategy |

---

### Migration Roadmap: Chaos → Structure

**Current state**: No documented workflow, daily conflicts, force pushes, morale low

**Target state**: Documented workflow, 80% reduction in conflicts, zero force pushes

#### Week 1: Stabilize and Diagnose

**Actions**:
1. **Freeze feature work for 2 days** (cleanup time)
2. **Baseline metrics**: Count conflicts last 30 days, force pushes, PR cycle time
3. **Diagnose root causes**:
   - Run `git log --all --graph --stat --since='1 month ago' | grep CONFLICT` → What files?
   - Interview team → Why force pushes? Why conflicts?
   - Check module coupling → Architecture issue?
4. **Protect main branch immediately**: No force push, enable at platform level
5. **Clean up branches**: Close stale PRs, delete dead branches

**Deliverables**:
- Baseline metrics document
- Root cause analysis
- Protected main branch

#### Week 2: Choose and Document

**Actions**:
1. **Select branching strategy**: Use decision framework above
2. **Write ADR**: Document strategy, alternatives, rationale
3. **Update README**: Add workflow diagram and instructions
4. **Train team**: 1-hour session on new workflow, practice with sandbox repo
5. **Set up branch protection**: Implement rules in platform

**Deliverables**:
- ADR for branching strategy
- Documented workflow in README
- Platform configuration (branch protection, CODEOWNERS)

#### Week 3: Parallel Run

**Actions**:
1. **New PRs follow new workflow** (required)
2. **In-flight work continues old way** (grandfathered, must finish within 7 days)
3. **Daily standups**: Check for blockers, answer questions
4. **Monitor metrics**: Conflicts, PR cycle time
5. **Adjust as needed**: Fix pain points quickly

**Deliverables**:
- All new work following new workflow
- Metrics showing early trends

#### Week 4: Full Adoption and Retrospective

**Actions**:
1. **All work follows new workflow** (no exceptions)
2. **Old branches merged or closed** (cleanup)
3. **Measure results**: Compare to baseline (Week 1)
4. **Team retrospective**:
   - What's working?
   - What's painful?
   - What needs adjustment?
5. **Schedule 30-day review**: Re-measure metrics, course-correct

**Deliverables**:
- Clean git history following strategy
- Metrics comparison (before/after)
- Retro notes and action items

**Success criteria**:
- ✅ 80% reduction in merge conflicts
- ✅ Zero force pushes to main
- ✅ PR cycle time <4 hours (median)
- ✅ Team morale improved (subjective, but ask)

**Rollback trigger**:
- If conflicts increase >2x baseline → Revert, diagnose further
- If team strongly resists after 2 weeks → Re-evaluate choice

---

## Verification & Validation

### How to Verify This Practice is Working

**Observable indicators**:
- [ ] Team can articulate branching strategy (not just "we use git")
- [ ] Main branch protected (force push impossible)
- [ ] 100% of changes through PR with reviews
- [ ] Merge conflicts rare (<1 per week)
- [ ] PRs merge quickly (<4 hours median)
- [ ] No force pushes to protected branches (audit log confirms)

**Metrics to track**:
- Merge conflicts per week (trend downward)
- Force pushes to main (target: zero)
- PR cycle time (commit → merge, target: <4 hours median)
- Branch age (target: <3 days for trunk-based, <2 weeks for feature branches)

### Common Failure Modes

| Failure Mode | Symptoms | Remediation |
|--------------|----------|-------------|
| **Compliance Theater** | Strategy documented but not followed | Enforce with platform (branch protection, required reviews). Honor system fails. |
| **Excessive Bureaucracy** | PRs take days, team frustrated | Reduce required reviewers, automate checks, train reviewers to be timely |
| **Training Gap** | Force pushes from confusion, not malice | Git training session, pair programming, sandbox practice repo |
| **Wrong Strategy** | Conflicts increased after adoption | Revert, diagnose root cause (architecture? communication?), re-evaluate |
| **Tool Configuration Drift** | Branch protection rules disabled or weakened | Audit quarterly, enforce as-code (e.g., Terraform for GitHub settings) |

---

## Related Practices

- **Architecture & Design**: ADR for branching strategy selection
- **Build & Integration**: CI/CD depends on branching strategy (when to deploy)
- **Technical Debt Management**: Frequent conflicts suggest architectural coupling
- **Platform Integration**: GitHub vs Azure DevOps specific configuration

---

## Real-World Example: Git Chaos → GitHub Flow

**Context**:
- Team of 8 developers
- Daily merge conflicts (avg 3 per day)
- Force pushes weekly (lost work 2x in last month)
- No documented workflow
- Morale low

**Root cause diagnosis**:
- Analyzed git log: 80% of conflicts in `api/models.py` and `frontend/state.ts`
- Interviewed team: Conflicts due to tight coupling (all features touch same files)
- Force pushes from developers trying to "fix" merge conflicts (lack of training)

**Insight**: Git strategy won't fix architectural coupling. Need parallel effort.

**Actions taken**:
1. **Week 1**: Protected main branch, baseline metrics (3 conflicts/day, 1 force push/week)
2. **Week 2**:
   - Selected **GitHub Flow** (continuous deployment, simple for team)
   - Wrote ADR explaining choice
   - Refactored `models.py` and `state.ts` to reduce coupling (2-day spike)
   - Git training for team (rebase, merge, when to force push: NEVER)
3. **Week 3**: Parallel run, new PRs follow GitHub Flow
4. **Week 4**: Full adoption, retrospective

**Results after 30 days**:
- Conflicts: 3/day → 0.5/day (83% reduction)
- Force pushes: 1/week → 0 (100% reduction)
- PR cycle time: 8 hours → 3 hours (62% faster)
- Team morale: "This is so much better" (qualitative, but measurable in retro)

**Key learning**: Branching strategy + architectural fix + training = success. Strategy alone wouldn't have fixed coupling.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when conflicts spike
