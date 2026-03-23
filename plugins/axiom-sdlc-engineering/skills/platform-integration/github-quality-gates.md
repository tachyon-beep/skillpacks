# Reference Sheet: Quality Gates in GitHub

## Purpose & Context

Implements CMMI **VER (Verification)**, **VAL (Validation)**, and **PI (Product Integration)** process areas using GitHub Actions, status checks, and deployment environments.

**When to apply**: Setting up CI/CD pipelines with quality gates for CMMI compliance

**Prerequisites**: GitHub repository, Actions enabled, deployment environments configured

---

## CMMI Maturity Scaling

### Level 2: Managed
- Basic CI (build + test on PR)
- Manual code review
- Simple pass/fail gates
- Deployment to single environment

### Level 3: Defined
- Multi-stage pipelines (build → test → deploy)
- Required status checks
- Automated quality metrics (coverage, linting)
- Multiple environments (dev, staging, prod)
- Approval workflows

### Level 4: Quantitatively Managed
- Pipeline metrics (success rate, duration)
- Statistical process control on quality
- Predictive failure detection
- Performance benchmarking

---

## Implementation Guidance

### Multi-Stage Pipeline Example

**File**: `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  # Stage 1: Build and Unit Test
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up environment
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=xml --cov-report=term
      
      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
  
  # Stage 2: Integration Tests
  integration-tests:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up environment
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-results
          path: test-results/
  
  # Stage 3: Security Scan
  security-scan:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run security scan
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      
      - name: Check for vulnerabilities
        run: |
          # Fail if high/critical vulnerabilities found
          exit 0  # Customize based on security policy
  
  # Stage 4: Deploy to Staging (on main branch only)
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    needs: [build-and-test, integration-tests, security-scan]
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to staging
        run: |
          # Your deployment script
          echo "Deploying to staging..."
      
      - name: Run smoke tests
        run: |
          # Post-deployment verification
          curl -f https://staging.example.com/health || exit 1
  
  # Stage 5: Deploy to Production (requires approval)
  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          # Your deployment script
          echo "Deploying to production..."
      
      - name: Run smoke tests
        run: |
          curl -f https://example.com/health || exit 1
      
      - name: Notify team
        run: |
          # Send deployment notification
          echo "Production deployment complete"
```

### Branch Protection Configuration

**GitHub UI**: Settings → Branches → Branch protection rules

**Required settings for Level 3**:
- ✅ Require pull request reviews before merging (2 reviewers)
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - ✅ `build-and-test`
  - ✅ `integration-tests`
  - ✅ `security-scan`
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Include administrators (enforce for everyone)

**As code** (using GitHub CLI):

```bash
gh api repos/OWNER/REPO/branches/main/protection \
  --method PUT \
  --field required_status_checks[strict]=true \
  --field required_status_checks[contexts][]=build-and-test \
  --field required_status_checks[contexts][]=integration-tests \
  --field required_status_checks[contexts][]=security-scan \
  --field required_pull_request_reviews[required_approving_review_count]=2 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=true \
  --field required_conversation_resolution=true
```

---

## Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **No required status checks** | PRs can merge with failing tests | Configure branch protection with required checks |
| **Tests only on main** | Defects discovered after merge | Run tests on all PRs before merge |
| **Manual deployment** | Human error, no audit trail | Automate deployment via Actions with approval gates |
| **Single-stage pipeline** | No isolation between unit/integration/deploy | Multi-stage pipeline with dependencies |
| **No coverage enforcement** | Code quality degrades over time | Fail pipeline if coverage drops below threshold |
| **Skipping smoke tests** | Broken deployments reach users | Post-deployment smoke tests as final gate |

---

## Environment Protection Rules

**Staging Environment** (auto-deploy on main):
- No reviewers required
- Deployment branches: `main` only
- Environment secrets: Staging API keys

**Production Environment** (manual approval):
- Required reviewers: 2 from CODEOWNERS
- Wait timer: 5 minutes (allows cancellation)
- Deployment branches: `main` only
- Environment secrets: Production API keys

**Configuration**: Settings → Environments → [Environment] → Protection rules

---

## Verification Metrics (Level 3/4)

**Pipeline metrics to track**:
- Build success rate: (Successful builds / Total builds) × 100%
- Test pass rate: (Passing tests / Total tests) × 100%
- Deployment frequency: Count per day/week
- Lead time: Commit to production deployment time
- Change failure rate: (Failed deployments / Total deployments) × 100%
- Mean time to recovery (MTTR): Time to fix broken pipeline

**Collect via**:
```yaml
- name: Record metrics
  run: |
    # Store metrics in artifact or external service
    echo "build_duration_seconds,${{ job.duration }}" >> metrics.csv
```

---

## Related Practices

- `../quality-assurance/SKILL.md` - VER/VAL process definitions
- `./github-measurement.md` - Pipeline metrics collection
- `./github-config-mgmt.md` - Branch protection, baselines
- `../design-and-build/SKILL.md` - Integration strategies (PI)

---

**Last Updated**: 2026-01-25
