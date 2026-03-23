---
parent-skill: lifecycle-adoption
reference-type: retrofitting-guidance
load-when: Adding traceability to existing features, requirements tracking for legacy code
---

# Retrofitting Requirements Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Adding requirements traceability to existing features, documenting undocumented features

This reference provides guidance for retrofitting requirements management (RD + REQM) onto existing codebases.

---

## Reference Sheet 3: Retrofitting Requirements

### Purpose & Context

**What this achieves**: Add requirements traceability to features already shipped

**When to apply**:
- Pre-audit compliance (need RTM for existing features)
- Incident analysis (need to trace which requirements led to this code)
- Impact analysis (which features affected by this change?)

**Prerequisites**:
- Existing codebase with shipped features
- Access to stakeholders/product owners (for validation)
- Tool for traceability (GitHub Issues, Azure DevOps Work Items, etc.)

### CMMI Maturity Scaling

#### Level 2: Managed (Retrofitting Requirements)

**Approach**: Manual reverse engineering for critical features only

**Traceability Target**:
- Critical/high-risk features documented (30-40% of features)
- Traceability via issue references in code comments or commit messages
- Spreadsheet RTM (Excel/Google Sheets)

**Effort**: 2-4 hours per feature (10 features = 20-40 hours)

**Audit Sufficiency**: Yes (Level 2 allows selective traceability)

#### Level 3: Defined (Retrofitting Requirements)

**Approach**: Systematic reverse engineering + stakeholder validation

**Traceability Target**:
- All shipped features documented (100%)
- Bidirectional traceability (requirements ↔ code ↔ tests)
- Tool-based RTM (GitHub Projects, Azure DevOps Queries)

**Effort**: 4-8 hours per feature (100 features = 400-800 hours)

**Audit Sufficiency**: Yes (with justification for pre-traceability era)

#### Level 4: Quantitatively Managed (Retrofitting Requirements)

**Approach**: Statistical sampling + risk-based prioritization

**Traceability Target**:
- Statistical sample of features (95% confidence interval)
- Quantitative risk assessment to prioritize which features to retrofit
- Automated traceability checks going forward

**Effort**: Reduced via sampling (50 features instead of 100 with statistical validity)

**Audit Sufficiency**: Yes (statistical sampling is acceptable for Level 4)

### Implementation Guidance

#### Quick Start Checklist

**Step 1: Inventory Features** (2-4 hours)
- [ ] List all shipped features from release notes, changelog, or product backlog
- [ ] Categorize by risk (critical, high, medium, low)
- [ ] Identify "dark matter" (shipped but undocumented features)

**Example** (100 shipped features):
- 10 critical (payment, auth, data integrity)
- 30 high (core user flows)
- 40 medium (secondary features)
- 20 low (nice-to-haves, experimental)

**Step 2: Prioritize for Retrofit** (1 hour)

Use risk-based prioritization:

```
Retrofit Priority = (Regulatory Impact) × (Change Frequency) × (Defect History)
```

| Feature | Regulatory | Change Freq | Defect History | Priority |
|---------|-----------|-------------|----------------|----------|
| Payment Processing | Critical (PCI DSS) | High (monthly) | 5 bugs | **CRITICAL** |
| User Auth | Critical (HIPAA) | Medium (quarterly) | 2 bugs | **HIGH** |
| Settings Page | None | Low (yearly) | 0 bugs | **LOW** |

**Retrofitting Target** (for 6-week audit prep):
- ✅ All critical features (10)
- ✅ Top 20 high-risk features (20)
- ⏸ Defer medium/low (70 features)
- **Total**: 30 features (30% coverage, 70% risk reduction)

**Step 3: Reverse Engineer Requirements** (4-8 hours per feature)

For each feature to retrofit:

**Method 1: Code Analysis**
- Read implementation code
- Identify key behaviors (inputs, outputs, validations)
- Extract implicit requirements

**Example**: Payment processing feature
- Code reveals: "Accept Visa/MC/Amex, validate CVV, encrypt card data, call Stripe API"
- Reverse engineered requirements:
  - REQ-PAY-001: System shall accept Visa, MasterCard, American Express
  - REQ-PAY-002: System shall validate CVV (3 or 4 digits)
  - REQ-PAY-003: System shall encrypt card data before transmission
  - REQ-PAY-004: System shall integrate with Stripe payment API

**Method 2: Test Analysis**
- Read test cases (unit, integration, E2E)
- Tests reveal intended behavior
- Convert test scenarios to requirements

**Example**: Test case `test_payment_rejects_invalid_cvv()`
- Reverse engineered requirement: REQ-PAY-005: System shall reject payments with invalid CVV

**Method 3: Documentation Mining**
- Review design docs, ADRs, meeting notes
- Extract requirements mentioned in decisions
- Consolidate into structured requirements

**Step 4: Stakeholder Validation** (1-2 hours per feature)

- Present reverse-engineered requirements to product owner/stakeholder
- Ask: "Are these the requirements you intended?"
- Capture missing requirements (undocumented features)
- Resolve conflicts (intended behavior vs. actual behavior)

**Step 5: Create Traceability** (1 hour per feature)

**GitHub Approach**:
- Create GitHub Issue for each reverse-engineered requirement
- Label: `requirement`, `retrofitted`, `feature-name`
- Link to code: Add issue number in code comment or commit message
- Link to tests: Reference tests in issue description

**Example**:
```python
# Payment processing implementation
# Requirements: #123, #124, #125 (retrofitted 2026-01-24)
def process_payment(card_number, cvv, amount):
    # REQ-PAY-001: Accept Visa/MC/Amex
    if not validate_card_type(card_number):
        raise InvalidCardError
    # REQ-PAY-002: Validate CVV
    if not validate_cvv(cvv):
        raise InvalidCVVError
    ...
```

**Azure DevOps Approach**:
- Create Work Item (type: Requirement) for each requirement
- Link to code via commit associations
- Link to tests via test plan references
- Query-based RTM: "Show all requirements for Feature X"

**Step 6: Document Retrofit in RTM** (30 minutes total)

Create lightweight RTM showing retrofitted traceability:

```markdown
# Requirements Traceability Matrix (Retrofitted Features)

**Note**: Features shipped before 2026-01-01 have retrospectively documented requirements as part of CMMI adoption. Traceability established via code analysis, test analysis, and stakeholder validation.

| Requirement ID | Feature | Source Code | Tests | Status | Retrofit Date |
|----------------|---------|-------------|-------|--------|---------------|
| REQ-PAY-001 | Payment | payment.py:45 | test_payment.py:12 | Validated | 2026-01-24 |
| REQ-PAY-002 | Payment | payment.py:67 | test_payment.py:34 | Validated | 2026-01-24 |
| REQ-AUTH-001 | Auth | auth.py:23 | test_auth.py:8 | Validated | 2026-01-24 |
...
```

#### Templates & Examples

**Reverse Engineering Template**:

```markdown
# Feature: [Name]

**Retrofit Date**: YYYY-MM-DD
**Analyst**: [Name]
**Stakeholder Validator**: [Name]

## Reverse Engineered Requirements

### From Code Analysis

- REQ-XXX-001: [Requirement extracted from code]
- REQ-XXX-002: [Requirement extracted from code]

### From Test Analysis

- REQ-XXX-003: [Requirement extracted from tests]
- REQ-XXX-004: [Requirement extracted from tests]

### From Documentation

- REQ-XXX-005: [Requirement found in ADR/design doc]

## Stakeholder Validation

- ✅ Validated by [Name] on [Date]
- Missing requirements identified:
  - REQ-XXX-006: [Previously undocumented]
- Conflicts resolved:
  - REQ-XXX-002: Updated per stakeholder feedback

## Traceability

- **Code**: [File paths and line numbers]
- **Tests**: [Test file paths]
- **GitHub Issues**: #123, #124, #125
```

**Filled Example**:

```markdown
# Feature: Two-Factor Authentication (2FA)

**Retrofit Date**: 2026-01-24
**Analyst**: John (Tech Lead)
**Stakeholder Validator**: Alice (Product Owner)

## Reverse Engineered Requirements

### From Code Analysis (auth.py)

- REQ-AUTH-010: System shall support TOTP-based 2FA (RFC 6238)
- REQ-AUTH-011: System shall generate QR code for authenticator app setup
- REQ-AUTH-012: System shall validate 6-digit TOTP codes with 30-second window
- REQ-AUTH-013: System shall provide backup codes (10 codes, single-use)

### From Test Analysis (test_2fa.py)

- REQ-AUTH-014: System shall reject invalid TOTP codes
- REQ-AUTH-015: System shall prevent reuse of backup codes
- REQ-AUTH-016: System shall allow 2FA reset via email for locked accounts

### From Documentation (ADR-042-2fa-implementation.md)

- REQ-AUTH-017: System shall use Google Authenticator-compatible TOTP
- REQ-AUTH-018: System shall store backup codes hashed (bcrypt)

## Stakeholder Validation

- ✅ Validated by Alice on 2026-01-24
- Missing requirements identified:
  - REQ-AUTH-019: System shall allow users to disable 2FA (Alice: "This was implied but not documented")
- Conflicts resolved:
  - REQ-AUTH-012: Window was 60 seconds in code but should be 30 (fixed in PR #456)

## Traceability

- **Code**: `auth.py:234-456`, `totp.py:12-89`
- **Tests**: `test_2fa.py:45-234`
- **GitHub Issues**: #789 (2FA feature), #790 (backup codes), #791 (QR generation)
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Retrofit Everything** | 100 features × 6 hours = 600 hours (4 person-months) | Risk-based: 30 critical features × 6 hours = 180 hours |
| **Fake Requirements** | Writing requirements to match code (not stakeholder needs) | Stakeholder validation required |
| **No Validation** | Reverse-engineered reqs never checked with product owner | 1-hour validation session per feature |
| **Traceability Theater** | Create requirements but don't actually link to code | Bidirectional links enforced |
| **Ignore Dark Matter** | Only document known features, miss undocumented ones | Code analysis reveals undocumented features |

### Tool Integration

**GitHub**:
- **Issues as requirements**: Create issue per requirement, label `requirement` + `retrofitted`
- **Code links**: Use issue refs in comments: `// See #123 for requirement`
- **Test links**: Reference tests in issue description
- **RTM**: GitHub Project board with columns: Requirements | Code | Tests | Validated

**Azure DevOps**:
- **Work Items**: Type = Requirement, State = Retrofitted
- **Code links**: Associated commits feature
- **Test links**: Test plan integration
- **RTM**: Query-based: "SELECT Requirements WHERE Tags CONTAINS 'Retrofitted'"

**Spreadsheet (Minimal)**:
- Excel/Google Sheets with columns: Requirement ID, Description, Code File, Test File, Validated By, Date
- Manual maintenance (update when code changes)

### Verification & Validation

**How to verify retrofit is complete**:
- Sample check: Pick 5 random critical features, verify requirements exist and link to code
- Stakeholder sign-off: Product owner confirms all critical features documented
- Audit dry-run: External reviewer checks RTM for completeness

**Common failure modes**:
- **Requirements don't match code** → Reverse engineering errors, need code review
- **Stakeholder disagrees with requirements** → Conflict between intended vs. actual behavior, may reveal bugs
- **Traceability breaks over time** → No enforcement for new changes, need CI check

### Related Practices

- **After retrofit**: Enforce traceability going forward (see requirements-lifecycle skill)
- **If audit is imminent**: Prioritize ruthlessly (30% coverage may be sufficient for Level 2)
- **If stakeholders unavailable**: Document as "Derived from code analysis, pending validation"

---

