---
parent-skill: lifecycle-adoption
reference-type: retrofitting-guidance
load-when: Adding tests and code reviews to legacy code, implementing quality practices
---

# Retrofitting Quality Practices Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Adding testing and code review to existing projects, implementing VER + VAL practices

This reference provides guidance for retrofitting quality practices (VER + VAL) onto existing codebases.

---

## Reference Sheet 5: Retrofitting Quality Practices

### Purpose & Context

**What this achieves**: Add testing and code review to projects with little/no quality assurance

**When to apply**:
- Legacy code with zero tests
- No code review culture
- Production bugs frequent
- Manual testing only (slow, error-prone)

**Prerequisites**:
- Codebase exists (not greenfield)
- Team willing to write tests (may require convincing)
- CI/CD infrastructure or willingness to set it up

### CMMI Maturity Scaling

#### Level 2: Managed (Retrofitting Quality)

**Approach**: Basic tests + informal code review

**Practices to Adopt**:
- Unit tests for new code only (don't retrofit everything)
- PR review requirement (1 approver)
- Manual testing checklist
- Basic CI (linting + tests)

**Timeline**: 2 weeks to establish baseline

**Baseline Target**: 30-40% test coverage (new code + critical paths)

#### Level 3: Defined (Retrofitting Quality)

**Approach**: Comprehensive testing + formal peer review

**Practices to Adopt** (beyond Level 2):
- Test coverage target (70-80%)
- Code review checklist
- Integration tests for key flows
- Automated regression suite
- Test pyramid enforcement

**Timeline**: 2-3 months to reach 70% coverage

**Organizational Standard**: Templates, checklists, baselines

#### Level 4: Quantitatively Managed (Retrofitting Quality)

**Approach**: Metrics-driven quality with prediction models

**Practices to Adopt** (beyond Level 3):
- Defect density tracking
- Test effectiveness metrics
- Defect prediction models
- Statistical process control for quality

**Timeline**: 6+ months (requires historical data)

**Quantitative Objectives**: Defect density <0.5/KLOC, test escape rate <5%

### Implementation Guidance

#### Quick Start Checklist

**Step 1: Assess Current Quality** (2 hours)

Gather baseline metrics:
- [ ] Current test coverage: `coverage run; coverage report` (likely 0-10%)
- [ ] Manual test time: How long to regression test? (likely days/weeks)
- [ ] Recent bug count: Production bugs in last 3 months
- [ ] Code review rate: % of changes reviewed (likely 0%)

**Step 2: Prioritize What to Test** (1-2 hours)

**DO NOT try to test everything at once**. Prioritize:

| Priority | What to Test | Why |
|----------|--------------|-----|
| **CRITICAL** | Payment, auth, data integrity | Bugs here = business impact |
| **HIGH** | Core user flows (top 10 features) | Used daily, bugs visible |
| **MEDIUM** | Secondary features | Less frequent use |
| **LOW** | Edge cases, experimental features | Defer until critical/high covered |

**Example** (E-commerce system):
- Critical: Payment processing, checkout flow, inventory management
- High: Product search, user registration, order history
- Medium: Wishlist, product reviews, email notifications
- Low: Admin analytics dashboard, experimental recommendation engine

**Step 3: Create Test Infrastructure** (1-2 days)

**For Python**:
```bash
# Install test framework
pip install pytest pytest-cov

# Create test directory structure
mkdir -p tests/unit tests/integration tests/e2e

# Create pytest config
cat > pytest.ini <<EOF
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=src --cov-report=html --cov-report=term
EOF

# Create sample test
cat > tests/unit/test_sample.py <<EOF
def test_placeholder():
    assert True  # Replace with real tests
EOF

# Run tests
pytest
```

**For JavaScript/TypeScript**:
```bash
# Install test framework
npm install --save-dev jest @types/jest

# Create test directory
mkdir -p tests

# Create jest config
npx jest --init

# Run tests
npm test
```

**Step 4: Write Tests for Critical Paths** (2-4 weeks)

**Strategy**: New code + critical existing code

**New code**:
- 100% test coverage requirement (enforced via CI)
- No PR merges without tests

**Critical existing code**:
- Characterization tests (document current behavior)
- Focus on "happy path" first
- Add edge case tests over time

**Example** (retrofitting payment processing):

```python
# tests/unit/test_payment.py
import pytest
from payment import process_payment, PaymentError

def test_process_payment_valid_visa():
    """Test successful Visa payment"""
    result = process_payment(
        card_number="4111111111111111",
        cvv="123",
        amount=100.00
    )
    assert result.status == "success"
    assert result.transaction_id is not None

def test_process_payment_invalid_cvv():
    """Test payment rejection for invalid CVV"""
    with pytest.raises(PaymentError, match="Invalid CVV"):
        process_payment(
            card_number="4111111111111111",
            cvv="12",  # Too short
            amount=100.00
        )

def test_process_payment_zero_amount():
    """Test rejection of zero-amount payments"""
    with pytest.raises(PaymentError, match="Amount must be positive"):
        process_payment(
            card_number="4111111111111111",
            cvv="123",
            amount=0.00
        )
```

**Coverage goal**: 10 critical paths × 5 tests each = 50 tests in 2-4 weeks

**Step 5: Establish Code Review Process** (1 week)

**Minimal viable review process**:

1. **PR requirement**: All changes via PR (enforced via branch protection)
2. **Review checklist**: Create simple checklist (see template below)
3. **1+ approver**: At least one team member must approve
4. **Review training**: 1-hour workshop on effective code review

**Code Review Checklist**:

```markdown
## Code Review Checklist

### Functionality
- [ ] Code does what the PR description says
- [ ] Edge cases handled (null, empty, invalid input)
- [ ] No obvious bugs

### Testing
- [ ] Tests included for new code
- [ ] Tests pass locally and in CI
- [ ] Critical paths have test coverage

### Code Quality
- [ ] Code is readable (clear variable names, simple logic)
- [ ] No commented-out code (unless explained)
- [ ] No obvious performance issues

### Security
- [ ] No hardcoded secrets/passwords
- [ ] User input validated/sanitized
- [ ] No SQL injection or XSS vulnerabilities

### Documentation
- [ ] Complex logic has comments explaining "why"
- [ ] Public APIs have docstrings
- [ ] README updated if needed
```

**Step 6: Automate in CI** (1 day)

**GitHub Actions example**:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run linting
      run: |
        pip install flake8
        flake8 src/ --max-line-length=100

    - name: Run tests with coverage
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term

    - name: Check coverage threshold
      run: |
        coverage report --fail-under=70

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

**Step 7: Gradual Coverage Increase** (2-3 months)

**Month 1**: 30-40% coverage (critical paths)
**Month 2**: 50-60% coverage (add high-priority features)
**Month 3**: 70%+ coverage (comprehensive)

**Strategy**:
- Ratchet enforcement: Coverage must not decrease
- New code: 100% coverage requirement
- Bug fixes: Add test reproducing bug before fixing

#### Templates & Examples

**Test Plan for Retrofitting** (lightweight):

```markdown
# Test Plan: Payment System Retrofit

**Goal**: Achieve 70% test coverage on payment processing module

**Timeline**: 4 weeks

## Week 1: Infrastructure & Critical Happy Paths
- [ ] Set up pytest, coverage reporting
- [ ] Test: Process valid Visa payment
- [ ] Test: Process valid MasterCard payment
- [ ] Test: Process valid Amex payment
- **Target**: 20% coverage

## Week 2: Error Handling
- [ ] Test: Invalid CVV rejection
- [ ] Test: Expired card rejection
- [ ] Test: Insufficient funds handling
- [ ] Test: Network timeout handling
- **Target**: 40% coverage

## Week 3: Edge Cases
- [ ] Test: Zero amount rejection
- [ ] Test: Negative amount rejection
- [ ] Test: Very large amounts (>$10,000)
- [ ] Test: Special characters in cardholder name
- **Target**: 60% coverage

## Week 4: Integration & Refactoring
- [ ] Integration test: End-to-end checkout flow
- [ ] Refactor untested code to be testable
- [ ] Add missing edge case tests
- **Target**: 70% coverage

## Success Criteria
- [ ] 70% coverage achieved
- [ ] All critical paths tested
- [ ] CI enforcing coverage threshold
- [ ] Zero production bugs in payment (30-day window)
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Test Everything** | 0% → 100% is overwhelming, unsustainable | Incremental: 30% → 50% → 70% over 3 months |
| **Tests After Code** | "We'll add tests later" = never | Require tests for all new code NOW |
| **Rubber Stamp Reviews** | "LGTM" without reading = theater | Checklist + spot checks for thoroughness |
| **Testing Theater** | High coverage, low quality tests | Review test quality, not just quantity |
| **No Enforcement** | Coverage target suggestion, not requirement | CI fails if coverage drops |

### Tool Integration

**GitHub**:
- **Required status checks**: CI must pass before merge (branch protection)
- **Codecov integration**: Visual coverage reports on PRs
- **PR templates**: Include testing checklist

**Azure DevOps**:
- **Build validation**: Pipeline must pass for PR approval
- **Code coverage widget**: Dashboard showing trend
- **Quality gates**: Fail build if coverage <70%

### Verification & Validation

**How to verify quality practices adopted**:
- Spot check last 10 PRs: All have tests? All reviewed?
- Coverage trend: Increasing month-over-month?
- Production bugs: Decreasing?
- Team feedback: Are tests helping catch bugs?

**Common failure modes**:
- **Coverage increases, bugs don't decrease** → Low-quality tests (testing implementation, not behavior)
- **Team avoids writing tests** → Too hard to test (architecture issue), need refactoring
- **Code reviews become bottleneck** → Too few reviewers, need to distribute responsibility

### Related Practices

- **For deeper testing strategies**: See quality-assurance skill
- **If team resists testing**: See Reference Sheet 8 (Change Management)
- **If code is hard to test**: See design-and-build skill (refactoring for testability)

---

