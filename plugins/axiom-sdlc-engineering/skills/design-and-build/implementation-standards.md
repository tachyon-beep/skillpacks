# Reference Sheet: Implementation Standards

## Purpose & Context

Provides coding standards, code review practices, and documentation requirements to prevent "cowboy coding" and ensure consistency across team.

**When to apply**: Establishing team standards, code review process, preventing quality degradation

**Prerequisites**: Team agreement on standards (not top-down mandate without buy-in)

---

## CMMI Maturity Scaling

### Level 2: Managed

**Required Practices**:
- Basic coding standards documented (naming, formatting)
- Code review for changes (informal OK)
- Inline comments for complex logic

**Work Products**:
- Style guide (wiki or README)
- Review notes (PR comments)

**Quality Criteria**:
- Standards exist and accessible
- Reviews happen (not rigorous yet)

**Audit Trail**:
- PR comments showing reviews occurred

### Level 3: Defined

**Required Practices**:
- Comprehensive coding standards
- **Required code review (2+ reviewers, enforced by platform)**
- Code review checklist
- Documentation standards (API docs, module docs, inline comments)
- Automated enforcement (linters, formatters in CI)

**Additional Work Products**:
- Code review checklist
- Linter configuration files
- Documentation templates

**Quality Criteria**:
- Platform enforces 2+ reviews before merge
- Linter passes on all PRs
- Public APIs documented
- Review checklist followed

**Audit Trail**:
- PR approval records (2+ reviewers)
- Linter results in CI
- Review checklist completion

### Level 4: Quantitatively Managed

**Statistical Practices**:
- Code review metrics (time spent, defects found)
- Complexity metrics tracked (cyclomatic complexity, maintainability index)
- Code quality trends (technical debt ratio)

**Quantitative Work Products**:
- Review effectiveness metrics (defects found in review vs production)
- Complexity trend charts
- Quality gates based on statistical baselines

**Quality Criteria**:
- Review finding rate within control limits (20-40% of changes)
- Complexity metrics within baselines
- Technical debt ratio decreasing or stable

**Audit Trail**:
- Historical metrics proving quality trends
- Statistical process control for quality gates

---

## Coding Standards Essentials

### Naming Conventions

**Functions/Methods**: Verb phrases, descriptive
```python
# Good
def calculate_monthly_revenue(transactions):
def send_password_reset_email(user):

# Bad
def calc(data):           # Too terse
def doStuff():           # Meaningless
def get_data2():         # Numbered functions
```

**Variables**: Noun phrases, pronounceable
```python
# Good
user_count = 42
is_authenticated = True
customer_emails = []

# Bad
cnt = 42              # Too terse
flag = True           # Meaningless
arr1 = []             # Numbered variables
```

**Classes**: Noun phrases, singular
```python
# Good
class CustomerOrder:
class PaymentProcessor:

# Bad
class Process:        # Too vague
class Utils:          # Utility classes are code smell
```

**Constants**: UPPER_SNAKE_CASE
```python
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30
```

### Formatting

**Automated enforcement** (don't debate in reviews):
- Python: `black` (opinionated formatter)
- JavaScript: `prettier`
- Go: `gofmt`
- Rust: `rustfmt`

**CI enforcement**:
```yaml
- name: Check formatting
  run: black --check .
```

**Reject PRs with formatting failures** (no exceptions)

### Function Length

**Guideline**: Functions <50 lines

**Why**: Long functions indicate multiple responsibilities, hard to test, hard to understand

**Exception**: Sequential logic that's clearer together (e.g., state machine)

### Cyclomatic Complexity

**Guideline**: Complexity <10 per function

**Tool**: `radon cc --min B .` (Python example)

**Why**: High complexity = more paths = more bugs, hard to test

### Code Comments

**When to comment**:
- WHY, not WHAT: Explain rationale, not obvious code
- Complex algorithms: Cite algorithm name, complexity
- Workarounds: Explain why workaround needed, link to proper fix ticket
- Public APIs: Docstrings with params, returns, raises

**When NOT to comment**:
- Restating code: `i = i + 1  # Increment i` (useless)
- Commented-out code: Delete it (git history preserves)
- Obvious code: `if user.is_admin():  # Check if user is admin` (redundant)

**Example - Good comments**:
```python
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number using dynamic programming.

    Args:
        n: Position in Fibonacci sequence (0-indexed)

    Returns:
        int: Fibonacci number at position n

    Time complexity: O(n)
    Space complexity: O(n)
    """
    # Use DP instead of recursion to avoid exponential time
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

---

## Code Review Checklist (Level 3)

### Functionality
- [ ] Code does what PR description says
- [ ] Edge cases handled (null, empty, boundary values)
- [ ] Error handling appropriate (not swallowing exceptions)
- [ ] No obvious bugs

### Tests
- [ ] New code has tests (unit + integration as appropriate)
- [ ] Tests actually test the code (not just pass)
- [ ] Edge cases covered in tests
- [ ] Coverage maintained or improved

### Design
- [ ] Follows existing patterns (consistent with codebase)
- [ ] No code duplication (DRY principle)
- [ ] Appropriate abstraction level (not over-engineered, not under-engineered)
- [ ] Clear separation of concerns

### Readability
- [ ] Naming clear and consistent
- [ ] Functions reasonably sized (<50 lines)
- [ ] Comments explain WHY where needed
- [ ] No commented-out code

### Security
- [ ] No hardcoded secrets (passwords, API keys)
- [ ] Input validation present
- [ ] No SQL injection vectors
- [ ] No XSS vulnerabilities

### Performance
- [ ] No obvious performance issues (N+1 queries, unnecessary loops)
- [ ] Database queries efficient
- [ ] Caching used appropriately

### Documentation
- [ ] Public APIs documented (docstrings)
- [ ] README updated if needed
- [ ] Breaking changes noted

---

## Review Best Practices

### For Reviewers

**Be specific**:
```
# Bad
"This could be better"

# Good
"This function is 120 lines. Consider extracting the validation logic
(lines 45-78) into a separate validate_input() function for clarity."
```

**Distinguish must-fix from suggestions**:
```
MUST FIX: This will cause a null pointer exception if user.email is None.

SUGGESTION: Consider using a dataclass here for type safety, though the
current dict approach works fine.
```

**Provide rationale**:
```
# Bad
"Use a context manager here"

# Good
"Use a context manager here to ensure the file is closed even if an exception
occurs. Current code will leak file handles."
```

**Approve or request changes, don't ghost**:
- Approve if issues are minor suggestions
- Request changes if must-fix issues exist
- Don't leave PR in limbo

### For Authors

**Respond to all comments**:
- Fix and mark "Done"
- Explain if disagree (with reasoning)
- Ask clarifying questions

**Small PRs**:
- Target: <400 lines changed
- Large PRs get shallow reviews (too much cognitive load)
- Split features into smaller PRs

**Self-review first**:
- Read your own diff before requesting review
- Catch obvious issues
- Add comments explaining non-obvious decisions

---

## Documentation Standards

### Module-Level Documentation

**Every module/file should have header**:
```python
"""User authentication and authorization module.

This module provides:
- Password hashing and verification
- JWT token generation and validation
- Role-based access control (RBAC)

Example usage:
    auth = Authenticator()
    token = auth.generate_token(user)
    is_valid = auth.validate_token(token)
"""
```

### Public API Documentation

**All public functions/classes documented**:
```python
def process_payment(amount: Decimal, payment_method: str) -> PaymentResult:
    """Process a payment transaction.

    Args:
        amount: Payment amount in USD, must be positive
        payment_method: Payment method ID (e.g., "card_123")

    Returns:
        PaymentResult with status and transaction ID

    Raises:
        ValueError: If amount is negative or zero
        PaymentError: If payment method invalid or charge fails

    Example:
        result = process_payment(Decimal("99.99"), "card_123")
        if result.success:
            print(f"Transaction ID: {result.transaction_id}")
    """
```

### API Documentation (REST/GraphQL)

**Use standard formats**:
- REST: OpenAPI/Swagger (auto-generated from code)
- GraphQL: Schema documentation (introspection)

**Include**:
- Endpoint/query description
- Parameters (required/optional, types, validation)
- Response format (success and error cases)
- Example requests and responses
- Authentication requirements

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Cowboy Coding** | No reviews, force pushes, "works on my machine" | Bugs reach production, inconsistent quality | Required reviews (platform-enforced), CI on PRs |
| **Rubber Stamp Reviews** | "LGTM" without reading code | Defeats purpose of review | Use checklist, require specific feedback, track review metrics |
| **Bikeshedding** | Hour-long debates about variable names | Wastes time on trivial issues | Automate formatting, defer style debates to team standards doc |
| **Review Backlog** | PRs sit for days | Blocks progress, code goes stale | Review SLA (4 hours), rotate review duty |
| **No Enforcement** | Standards exist but not followed | Honor system fails | Automate: linters, formatters, required reviews in platform |

---

## Automated Enforcement (CI Integration)

**Python example**:
```yaml
# .github/workflows/quality.yml
name: Code Quality
on: [pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Check Formatting
        run: black --check .

      - name: Lint
        run: flake8 .

      - name: Type Check
        run: mypy .

      - name: Complexity Check
        run: radon cc --min B .

      - name: Security Scan
        run: bandit -r .
```

**Fail PR if any check fails** (no exceptions)

---

## Real-World Example

**Context**:
- Team of 8, inconsistent code quality
- No formal standards, reviews optional
- Bugs slipping to production

**Actions**:
1. **Week 1**: Team workshop to agree on standards (not top-down mandate)
2. **Week 2**: Document standards, set up linters, configure CI
3. **Week 3**: Enable branch protection (2 required reviews, CI must pass)
4. **Week 4**: Retrospective, adjust standards based on feedback

**Results**:
- Formatting debates eliminated (automated)
- Review finding rate: 30% (catching bugs pre-merge)
- Production bugs: 40% reduction in first quarter
- Team morale: "Code quality improved, reviews helpful"

**Key learning**: Automate enforcement (linters, CI) to remove friction. Manual enforcement fails.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when quality issues spike
