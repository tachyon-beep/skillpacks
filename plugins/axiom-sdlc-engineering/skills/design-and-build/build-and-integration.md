# Reference Sheet: Build & Integration

## Purpose & Context

Provides frameworks for setting up CI/CD pipelines correctly - requirements gathering BEFORE implementation, platform choice as architectural decision requiring ADR, and phased rollout to avoid "big bang" failures.

**When to apply**: Manual builds, no CI/CD, build optimization needed, deployment automation

**Prerequisites**: Automated tests exist (if not, write tests first - CI amplifies current process)

---

## Requirements Gathering Framework (DO THIS FIRST)

### Build Characteristics

Before choosing CI/CD platform, understand what you're building:

**Questions**:
- What are you building? (Web app, mobile, library, microservices?)
- Current build time breakdown:
  - Compile: __ minutes
  - Unit tests: __ minutes
  - Integration tests: __ minutes
  - Packaging: __ minutes
- Bottleneck? (CPU-bound? I/O-bound? Network-bound?)
- Build frequency? (Every commit? Daily? Weekly?)

**Action**: Profile current build with `time` for each stage

### Deployment Context

**Questions**:
- Where does it run? (AWS/Azure/GCP? Kubernetes? VMs? Serverless?)
- How often deploy? (Multiple daily? Weekly? Monthly?)
- Environments? (Prod only? Dev/Staging/Prod?)
- Downtime tolerance? (Zero downtime required? Maintenance windows OK?)
- Rollback needs? (Automated? Manual? How fast?)

### Risk Profile

**Questions**:
- Project level? (Level 2/3 → determines approval gates)
- Regulatory requirements? (SOC 2, HIPAA, PCI → audit logs needed)
- Security scanning required? (Dependency checks, SAST, container scanning?)
- Blast radius? (How bad if deploy breaks? Financial? Life-safety?)

**Level 2**: Manual approval for prod, basic CI
**Level 3**: Approval gates enforced by platform, comprehensive CI/CD, ADR for platform choice

---

## Platform Selection Decision Framework

| Factor | GitHub Actions | Azure Pipelines | GitLab CI | Jenkins |
|--------|----------------|-----------------|-----------|---------|
| **Integration** | GitHub native | Azure DevOps native | GitLab native | Platform-agnostic |
| **Ease of setup** | Easy (YAML in repo) | Medium (YAML or visual) | Easy (YAML in repo) | Hard (self-hosted) |
| **Cost** | Free tier generous | Free tier good | Free tier very good | Infrastructure cost |
| **Flexibility** | Good (Actions ecosystem) | Excellent (enterprise features) | Excellent (comprehensive) | Maximum (DIY) |
| **Support** | Community | Enterprise support | Community/enterprise | Self-support |

**Requires ADR** (Level 3): Platform choice is architectural - affects team workflow, vendor lock-in, costs

---

## Pipeline Stages (Standard)

```
Commit → Build → Unit Tests → Integration Tests → Package → Deploy Staging → Deploy Prod
```

### Level 2 Minimal Pipeline

```yaml
# GitHub Actions example
name: CI
on: [pull_request]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: make build
      - name: Unit Tests
        run: make test
```

**Required**:
- Build on every PR
- Run tests
- Report status

### Level 3 Comprehensive Pipeline

```yaml
name: CI/CD
on:
  pull_request:  # CI only
  push:
    branches: [main]  # CD to staging
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: make build
      - name: Unit Tests
        run: make test-unit
      - name: Integration Tests
        run: make test-integration
      - name: Security Scan
        run: make security-scan
      - name: Upload Coverage
        run: codecov

  deploy-staging:
    needs: ci
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: make deploy-staging
      - name: Smoke Tests
        run: make smoke-test-staging

  deploy-prod:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production  # Manual approval gate
    steps:
      - name: Deploy to Production
        run: make deploy-prod
```

**Required (Level 3)**:
- All tests on PR
- Security scanning
- Staging deployment automatic
- Production approval gate
- Smoke tests after deploy

---

## Build Optimization (Hours → Minutes)

### Profiling First

```bash
time make build        # Total: 120 minutes
time make compile      #   → 90 minutes (BOTTLENECK)
time make test         #   → 25 minutes
time make package      #   → 5 minutes
```

**Optimization priority**: Fix compile time first (biggest impact)

### Caching Strategy

**Dependencies** (download once, reuse):
```yaml
- uses: actions/cache@v2
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

**Build artifacts** (incremental compilation):
```yaml
- uses: actions/cache@v2
  with:
    path: ./build
    key: ${{ github.sha }}
    restore-keys: ${{ github.base_ref }}
```

**Docker layers** (multi-stage builds):
```dockerfile
FROM node:16 AS build
COPY package.json .
RUN npm install  # Cached if package.json unchanged
COPY . .
RUN npm build

FROM node:16-slim
COPY --from=build /app/dist /app
```

### Parallelization

**Test parallelization**:
```yaml
strategy:
  matrix:
    test-group: [unit, integration, e2e]
steps:
  - run: make test-${{ matrix.test-group }}
```

**Multi-platform builds** (parallel):
```yaml
strategy:
  matrix:
    os: [ubuntu, macos, windows]
runs-on: ${{ matrix.os }}
```

**Target**: <15 minutes CI feedback on PRs

---

## Deployment Strategies

### Blue/Green Deployment

**Pattern**: Two identical environments, switch traffic

```
Blue (current prod) ← 100% traffic
Green (new version) ← 0% traffic

Deploy to Green → Smoke test → Switch traffic (100% to Green) → Keep Blue for rollback
```

**Pros**: Instant rollback (switch back to Blue)
**Cons**: 2x infrastructure cost

### Canary Deployment

**Pattern**: Gradually shift traffic to new version

```
Old version ← 100% traffic
New version ← 0% traffic

→ 10% to new (monitor)
→ 50% to new (monitor)
→ 100% to new (done)
```

**Pros**: Detect issues early with small blast radius
**Cons**: Complex traffic routing, monitoring required

### Rolling Deployment

**Pattern**: Replace instances one by one

```
Instances: [A, B, C, D]
Deploy: A (new), B (old), C (old), D (old)
     → A (new), B (new), C (old), D (old)
     → A (new), B (new), C (new), D (old)
     → A (new), B (new), C (new), D (new)
```

**Pros**: No extra infrastructure
**Cons**: Mixed versions during rollout

**Level 3 requirement**: Choose strategy, document in ADR, test rollback quarterly

---

## Rollback Procedures

### Automated Rollback Triggers

```yaml
deploy-prod:
  steps:
    - name: Deploy
      run: make deploy
    - name: Health Check
      run: make health-check
      timeout-minutes: 5
    - name: Rollback on Failure
      if: failure()
      run: make rollback
```

### Manual Rollback Checklist

1. Identify issue (errors spiking, performance degraded)
2. Decide: Rollback vs fix forward?
3. Execute rollback:
   - Blue/green: Switch traffic back
   - Canary: Reduce new version to 0%
   - Rolling: Re-deploy previous version
4. Verify: Health checks pass, metrics normal
5. Post-mortem: What happened? How prevent?

**Test rollback quarterly** (Level 3): Don't wait for real failure to discover rollback broken

---

## Pre-Flight Checklist (BEFORE Implementing CI/CD)

- [ ] **Tests exist and pass locally**: CI amplifies current process - if tests broken, CI won't help
- [ ] **Build works locally in <15 min**: Optimize local build before automating
- [ ] **Dev/staging/prod environments similar**: Environment parity required for reliable deployments
- [ ] **Branching strategy established**: CI depends on workflow (when to deploy?)
- [ ] **Requirements gathered**: Platform choice needs context (see Requirements Gathering above)
- [ ] **ADR written** (Level 3): Platform and deployment strategy documented

**Common mistake from baseline**: Providing generic CI/CD template without checking prerequisites. CI/CD won't fix broken builds or missing tests.

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **No Requirements Gathering** | Generic template, doesn't fit needs | Wastes time on wrong solution | Answer: What building? Where deploying? Risk profile? |
| **No ADR for Platform** | Informal choice, no rationale | Lose audit trail, can't evaluate later | Level 3: Platform choice requires ADR |
| **Auto-Deploy to Prod** | No approval gate, broken deploys | High risk, no safety net | Staging auto, prod manual approval (Level 3) |
| **Hours-Long Builds** | CI feedback >1 hour | Developers bypass CI, defeats purpose | Profile, cache, parallelize. Target: <15 min |
| **No Rollback Plan** | "We'll fix forward" | Downtime extends, customers angry | Tested rollback procedure, quarterly drills |
| **CI Before Tests** | No tests to run | CI is useless, false confidence | Write tests FIRST, then automate |

---

## Real-World Example (from Baseline Scenario 4)

**Context**:
- Manual builds taking hours
- No automated testing in build
- Level 2 project

**Actions**:
1. **Requirements gathering**:
   - Building: Python web API
   - Bottleneck: `pip install` taking 45 min (slow network, many deps)
   - Deploy target: AWS ECS
   - Risk: Medium (internal tool, not customer-facing)
2. **Pre-flight**:
   - Fixed local build: Used `pip install --cache-dir` + requirements.txt pinning → 45min to 5min
   - Wrote missing tests: Coverage 0% → 60% (week 1)
   - Established GitHub Flow branching strategy (ADR)
3. **Platform choice**:
   - Evaluated: GitHub Actions (integrated), CircleCI (features), Jenkins (complex)
   - Chose: GitHub Actions (team already on GitHub, easy setup, good caching)
   - ADR documented choice, alternatives, rationale
4. **Implementation** (phased):
   - Week 1: Basic CI (build + test on PR)
   - Week 2: Add security scanning, coverage reports
   - Week 3: Deploy to staging on merge to main
   - Week 4: Deploy to prod with manual approval
5. **Results**:
   - Build time: 45min (manual) → 8min (CI with caching)
   - Deployment frequency: Weekly → daily
   - Bugs caught pre-merge: 0 → 12 in first month
   - Team confidence: "We can deploy any time now"

**Key learning**: Fix local build FIRST (45min → 5min), then automate. Requirements gathering prevented generic template mistake.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when build time degrades
