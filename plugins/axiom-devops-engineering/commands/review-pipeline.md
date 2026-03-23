---
description: Review CI/CD pipeline for missing stages, anti-patterns, and production readiness
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[pipeline_file_or_directory]"
---

# Review Pipeline Command

You are reviewing a CI/CD pipeline for completeness, safety, and production readiness.

## Core Principle

**"Deploy to production" is not a single step - it's a sequence of gates, health checks, gradual rollouts, and automated rollback triggers. Skipping these "for speed" causes production incidents.**

## Mandatory Pipeline Stages

Every production pipeline MUST include all 7 stages:

```
1. Build → 2. Test → 3. Deploy Staging → 4. Verify Staging → 5. Deploy Production → 6. Verify Production → 7. Monitor
```

**Missing any stage = production incidents waiting to happen.**

## Review Checklist

### Stage 1: Build

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Immutable artifacts | Tagged with commit SHA | Using "latest" tag |
| Build once | Same artifact to all envs | Rebuild per environment |
| Linting/formatting | Run before build | Skip for speed |
| Container image | Push to registry | Build on deploy server |

### Stage 2: Test

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Unit tests | Fast, isolated | Skipped in CI |
| Integration tests | API contracts, DB | Only run locally |
| Parallel execution | Split across runners | Sequential slow tests |
| Fail fast | Fastest tests first | Random order |

### Stage 3: Deploy to Staging

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Staging exists | Matches production | No staging environment |
| Same infrastructure | Containers, K8s, etc. | Different from production |
| Database migrations | Run with rollback tested | Skip migration testing |

### Stage 4: Verify Staging

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Health checks | Automated, not manual | Manual QA only |
| Smoke tests | Critical paths tested | No verification |
| Gate to production | Blocks on failure | Proceeds anyway |

### Stage 5: Deploy to Production

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Zero-downtime strategy | Blue-green/Canary/Rolling | Direct restart |
| Gradual rollout | Percentage-based | All-at-once |
| Rollback capability | Keep old version running | Terminate immediately |

### Stage 6: Verify Production

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Health endpoint | Returns 200 | No health check |
| Response time check | < baseline + 20% | No latency monitoring |
| Error rate check | < 1% threshold | No error tracking |
| Auto-rollback | On failure triggers | Manual rollback only |

### Stage 7: Monitor

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Post-deploy observation | 1 hour minimum | Deploy and forget |
| Metrics dashboard | Shows deployment version | No visibility |
| Alerting | On anomalies | Silent failures |

## Additional Checks

### Secrets Management

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Secret storage | CI/CD secret manager | Hardcoded in workflow |
| No logging secrets | Masked in output | Secrets in logs |
| Per-environment | Different secrets | Shared across envs |

### Database Migrations

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Backward-compatible | 3-step deployment | Breaking changes |
| Rollback tested | Down migration works | No rollback plan |
| Concurrent-safe | No locks on tables | Blocking migrations |

### Environment Promotion

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| Gates between envs | Tests pass, verification | Auto-promote always |
| Manual approval | For critical systems | No human gate |
| Business hours | Optional but recommended | Deploy anytime |

## Review Process

### Step 1: Identify Pipeline Files

Look for:
- `.github/workflows/*.yml` (GitHub Actions)
- `.gitlab-ci.yml` (GitLab CI)
- `Jenkinsfile` (Jenkins)
- `.circleci/config.yml` (CircleCI)
- `azure-pipelines.yml` (Azure DevOps)

### Step 2: Map Existing Stages

Create inventory of what exists:
- What stages are present?
- What's missing?
- What order do they run?

### Step 3: Check Each Stage

Apply checklist above to each stage.

### Step 4: Identify Anti-Patterns

Flag common mistakes:
- "latest" tags
- No staging environment
- Manual verification
- Direct restart deployments
- Hardcoded secrets
- No rollback capability

### Step 5: Prioritize Findings

- **Critical**: Missing stages, no rollback, hardcoded secrets
- **High**: No staging, manual verification, no health checks
- **Medium**: Slow tests, missing monitoring
- **Low**: Optimization opportunities

## Output Format

```markdown
## Pipeline Review: [Pipeline Name/File]

### Summary

**Production Ready**: Yes/No
**Missing Stages**: [Count]/7
**Critical Issues**: [Count]

### Stage Inventory

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Build | ✓/✗ | [Details] |
| 2. Test | ✓/✗ | [Details] |
| 3. Deploy Staging | ✓/✗ | [Details] |
| 4. Verify Staging | ✓/✗ | [Details] |
| 5. Deploy Production | ✓/✗ | [Details] |
| 6. Verify Production | ✓/✗ | [Details] |
| 7. Monitor | ✓/✗ | [Details] |

### Findings

#### Critical

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| [Issue] | [File:Line] | [Risk] | [Action] |

#### High

[Same format]

#### Medium

[Same format]

### Anti-Patterns Found

| Pattern | Location | Correct Approach |
|---------|----------|------------------|
| [Anti-pattern] | [Where] | [What to do instead] |

### Deployment Strategy Assessment

**Current**: [Direct/Rolling/Blue-Green/Canary/None]
**Recommended**: [Strategy based on context]
**Rollback Capability**: Yes/No

### Secrets Assessment

| Finding | Status |
|---------|--------|
| Secrets in secret manager | ✓/✗ |
| No hardcoded credentials | ✓/✗ |
| Per-environment secrets | ✓/✗ |

### Recommendations

**Immediate Actions**:
1. [Critical fix 1]
2. [Critical fix 2]

**Before Next Deploy**:
1. [High priority fix]

**Improvements**:
1. [Medium/Low priority]
```

## Common Pipeline Anti-Patterns

| Anti-Pattern | Why It's Wrong | Fix |
|--------------|----------------|-----|
| `image: myapp:latest` | Not reproducible | Use commit SHA tag |
| No staging job | Production = your test | Add staging environment |
| `restart: always` | Causes downtime | Blue-green deployment |
| Manual approval only | Slow, error-prone | Automated verification + manual gate |
| Tests after deploy | Too late to catch bugs | Test before deploy |
| `continue-on-error: true` | Hides failures | Fail fast, fix issues |

## Scope Boundaries

**This command covers:**
- Pipeline stage completeness
- Deployment safety
- Secrets handling
- Anti-pattern detection
- Rollback capability

**Not covered:**
- Designing new pipelines (use /design-deployment)
- Infrastructure provisioning
- Application-specific testing
