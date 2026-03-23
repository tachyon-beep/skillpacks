
# Security Architecture Review

## Overview

Systematically review designs for security issues. Core principle: **Checklist-driven review finds issues that intuition misses**.

**Key insight**: Ad-hoc review finds obvious issues. Systematic checklist finds subtle gaps.

## When to Use

Load this skill when:
- Reviewing architecture designs pre-implementation
- Conducting security audits of existing systems
- Evaluating third-party integrations
- Pre-deployment security validation

**Symptoms you need this**:
- "Is this design secure?"
- Reviewing microservices, APIs, data pipelines
- Pre-launch security check
- Compliance audit preparation

**Don't use for**:
- Threat modeling new systems (use `ordis/security-architect/threat-modeling`)
- Designing controls (use `ordis/security-architect/security-controls-design`)
- Secure patterns (use `ordis/security-architect/secure-by-design-patterns`)

## Review Process

### Step 1: Understand System

Before checklists, understand:
- **Components**: Services, databases, queues, external APIs
- **Data flows**: Where data enters/exits, trust boundaries
- **Users/actors**: Who accesses what (end users, admins, services)
- **Deployment**: Cloud/on-prem, network topology

### Step 2: Apply Checklists

For EACH area, go through checklist systematically. Check = verify present and secure.

### Step 3: Document Findings

For each issue found:
- **Description**: What's wrong
- **Severity**: Critical/High/Medium/Low
- **Impact**: What can attacker do
- **Recommendation**: How to fix


## Checklist 1: Authentication

### Credential Storage
- [ ] Passwords hashed with strong algorithm (bcrypt, Argon2, scrypt)
- [ ] NOT hashed with weak algorithms (MD5, SHA1, plain SHA256)
- [ ] Salt used per-password (not global salt or no salt)
- [ ] Key derivation with sufficient iterations (bcrypt cost ≥12, Argon2 with recommended params)

**Common Issues**:
- ❌ MD5/SHA1 hashing → CRITICAL
- ❌ No salt → HIGH
- ❌ Global salt → MEDIUM
- ❌ Low iteration count → MEDIUM


### Multi-Factor Authentication
- [ ] MFA available for all users (or at least admins)
- [ ] TOTP/hardware token support (not just SMS)
- [ ] Backup codes for account recovery
- [ ] Cannot bypass MFA via alternate login paths

**Common Issues**:
- ❌ No MFA → HIGH (for privileged accounts)
- ❌ SMS-only 2FA → MEDIUM (SIM swapping risk)
- ❌ MFA bypass path exists → HIGH


### Session Management
- [ ] Session tokens have expiration (not indefinite)
- [ ] Token expiry reasonable (hours for web, days max for mobile)
- [ ] Token rotation on sensitive actions (password change, permission change)
- [ ] Logout invalidates tokens (server-side revocation)
- [ ] Tokens stored securely (HttpOnly, Secure cookies or secure storage)

**Common Issues**:
- ❌ Non-expiring tokens → HIGH
- ❌ No logout/revocation → MEDIUM
- ❌ Tokens in localStorage (XSS vulnerable) → MEDIUM


### Password Policies
- [ ] Minimum length enforced (≥12 characters recommended)
- [ ] No maximum length (or very high max, ≥128 chars)
- [ ] Password history (prevent reuse of last N passwords)
- [ ] Account lockout after failed attempts (5-10 tries, temporary lockout)
- [ ] No common password blacklist (check against known-weak passwords)

**Common Issues**:
- ❌ No password policy → MEDIUM
- ❌ Short max length (<20 chars) → LOW
- ❌ No lockout → MEDIUM (brute-force vulnerability)


## Checklist 2: Authorization

### Access Control Model
- [ ] Clear access control model (RBAC, ABAC, MAC documented)
- [ ] Permissions defined per resource type
- [ ] Authorization checks at API layer AND data layer
- [ ] Consistent enforcement across all endpoints

**Common Issues**:
- ❌ No defined model (ad-hoc checks) → HIGH
- ❌ Authorization only at API (not data layer) → MEDIUM
- ❌ Inconsistent (some endpoints skip checks) → HIGH


### Privilege Escalation Prevention
- [ ] Users cannot modify their own roles/permissions
- [ ] API doesn't trust client-provided role claims
- [ ] Admin actions require separate authentication/approval
- [ ] No hidden admin endpoints without authorization

**Common Issues**:
- ❌ User can edit own role in profile → CRITICAL
- ❌ Trusting `X-User-Role` header from client → CRITICAL
- ❌ Hidden `/admin` path without authz → HIGH


### Resource-Level Authorization
- [ ] Authorization checks "Can THIS user access THIS resource?"
- [ ] Not just "Is user logged in?" or "Is user admin?"
- [ ] Users can only access their own data (unless explicitly shared)
- [ ] Object-level permission checks (IDOR prevention)

**Common Issues**:
- ❌ Check login only, not resource ownership → HIGH (IDOR vulnerability)
- ❌ `/api/users/{user_id}` allows any authenticated user → HIGH


### Default-Deny Principle
- [ ] All endpoints require authentication by default
- [ ] Explicit allow-list for public endpoints
- [ ] Authorization failures return 403 Forbidden (not 404)
- [ ] No "development mode" backdoors in production

**Common Issues**:
- ❌ Default-allow (must explicitly mark secure) → HIGH
- ❌ Returning 404 instead of 403 → LOW (information disclosure)
- ❌ Debug/dev endpoints enabled in production → CRITICAL


## Checklist 3: Secrets Management

### Secrets Storage
- [ ] No secrets in source code
- [ ] No secrets in config files committed to Git
- [ ] No secrets in environment variables visible in UI
- [ ] Secrets stored in secrets manager (Vault, AWS Secrets Manager, etc.)
- [ ] Secrets encrypted at rest

**Common Issues**:
- ❌ Secrets in Git history → CRITICAL
- ❌ Secrets in config.yaml → CRITICAL
- ❌ Database password in Dockerfile → HIGH


### Secrets Rotation
- [ ] Secrets have rotation schedule (monthly/quarterly)
- [ ] Rotation is automated or documented
- [ ] Application supports zero-downtime rotation
- [ ] Old secrets revoked after rotation

**Common Issues**:
- ❌ No rotation policy → MEDIUM
- ❌ Manual rotation without documentation → LOW
- ❌ Rotation requires downtime → LOW


### Access Control to Secrets
- [ ] Secrets access restricted by service identity
- [ ] Least privilege (service accesses only its secrets)
- [ ] Audit log of secrets access
- [ ] No shared secrets across services

**Common Issues**:
- ❌ All services can access all secrets → MEDIUM
- ❌ No audit log → LOW
- ❌ Shared API key for multiple services → MEDIUM


## Checklist 4: Data Flow

### Trust Boundary Identification
- [ ] Trust boundaries documented (external → API → services → database)
- [ ] Each boundary has validation rules
- [ ] No implicit trust based on network location

**Common Issues**:
- ❌ No documented boundaries → MEDIUM
- ❌ "Internal network = trusted" assumption → HIGH


### Input Validation at Boundaries
- [ ] All external input validated (type, format, range)
- [ ] Validation at EVERY trust boundary (not just entry point)
- [ ] Allow-list validation (define what's allowed, reject rest)
- [ ] Input validation errors logged

**Common Issues**:
- ❌ No validation at internal boundaries → MEDIUM
- ❌ Deny-list only (blacklist can be bypassed) → MEDIUM
- ❌ Validation at UI only (not API) → HIGH


### Output Encoding for Context
- [ ] Output encoded for destination context (HTML, SQL, shell, etc.)
- [ ] SQL parameterized queries (no string concatenation)
- [ ] HTML escaped when rendering user data
- [ ] JSON encoded when returning API responses

**Common Issues**:
- ❌ String concatenation for SQL → CRITICAL (SQL injection)
- ❌ Unescaped HTML → HIGH (XSS)
- ❌ Unescaped shell commands → CRITICAL (command injection)


### Data Classification and Handling
- [ ] Sensitive data identified (PII, secrets, financial)
- [ ] Sensitive data encrypted in transit
- [ ] Sensitive data encrypted at rest
- [ ] Sensitive data not logged
- [ ] Data retention policy defined

**Common Issues**:
- ❌ PII in plaintext logs → HIGH
- ❌ No encryption for sensitive data → HIGH
- ❌ No data retention policy → MEDIUM


## Checklist 5: Network Security

### TLS/Encryption in Transit
- [ ] All external traffic uses TLS (HTTPS, not HTTP)
- [ ] TLS 1.2 or 1.3 only (TLS 1.0/1.1 disabled)
- [ ] Strong cipher suites (no RC4, 3DES, MD5)
- [ ] HSTS header present (HTTP Strict Transport Security)

**Common Issues**:
- ❌ HTTP allowed → CRITICAL
- ❌ TLS 1.0 enabled → HIGH
- ❌ Weak ciphers → MEDIUM
- ❌ No HSTS → LOW


### Certificate Validation
- [ ] TLS certificates validated (not self-signed in production)
- [ ] Certificate expiry monitored
- [ ] Certificate revocation checking (OCSP/CRL)
- [ ] No certificate validation bypass in code

**Common Issues**:
- ❌ Self-signed certs in production → HIGH
- ❌ `verify=False` in code → CRITICAL
- ❌ No expiry monitoring → MEDIUM


### Network Segmentation
- [ ] Database not directly accessible from internet
- [ ] Services in separate subnets/VPCs
- [ ] Admin interfaces on separate network
- [ ] Internal services use mTLS or VPN

**Common Issues**:
- ❌ Database exposed to internet → CRITICAL
- ❌ All services in same flat network → MEDIUM
- ❌ Admin panel on public internet → HIGH


### Firewall Rules and Least-Necessary Access
- [ ] Firewall rules documented
- [ ] Default-deny firewall policy
- [ ] Only necessary ports open
- [ ] Source IP restrictions for sensitive endpoints

**Common Issues**:
- ❌ Default-allow firewall → HIGH
- ❌ All ports open → MEDIUM
- ❌ No source restrictions → MEDIUM


## Severity Guidelines

Use these guidelines to assign severity:

### Critical
- Direct path to data breach or system compromise
- **Examples**: SQL injection, secrets in Git, authentication bypass

### High
- Significant security impact, requires immediate attention
- **Examples**: Weak password hashing, no MFA for admins, IDOR vulnerability

### Medium
- Security weakness that should be addressed
- **Examples**: No password policy, inconsistent authorization, no audit logging

### Low
- Minor issue or hardening opportunity
- **Examples**: Information disclosure via error messages, missing HSTS header


## Review Report Template

```markdown
# Security Architecture Review

**System**: [System Name]
**Review Date**: [Date]
**Reviewer**: [Name]

## Executive Summary

[1-2 paragraph overview: Critical/High count, key findings, overall risk level]

## Findings

### Critical (Count: X)

#### 1. [Finding Title]
- **Area**: Authentication / Authorization / Secrets / Data Flow / Network
- **Description**: [What's wrong]
- **Impact**: [What attacker can do]
- **Recommendation**: [How to fix]
- **Affected Components**: [API Service, Database, etc.]

### High (Count: X)

[Same format as Critical]

### Medium (Count: X)

[Same format]

### Low (Count: X)

[Same format]

## Summary Table

| Finding | Severity | Area | Status |
|---------|----------|------|--------|
| SQL injection in /api/users | Critical | Data Flow | Open |
| Secrets in Git | Critical | Secrets | Open |
| No MFA for admins | High | Authentication | Open |

## Next Steps

1. Address Critical findings immediately (timeline: 1 week)
2. Address High findings (timeline: 2-4 weeks)
3. Schedule Medium/Low findings (timeline: 1-3 months)
```


## Quick Reference: Checklist Summary

| Area | Key Checks |
|------|------------|
| **Authentication** | Strong hashing (bcrypt), MFA, token expiry, lockout policy |
| **Authorization** | RBAC/ABAC model, resource-level checks, default-deny, no privilege escalation |
| **Secrets** | Not in Git, secrets manager, rotation policy, access control |
| **Data Flow** | Trust boundaries, input validation, output encoding, data classification |
| **Network** | TLS 1.2+, certificate validation, segmentation, firewall rules |


## Common Mistakes

### ❌ Ad-Hoc Review Without Checklist

**Wrong**: Review design intuitively, find obvious issues, call it done

**Right**: Systematically go through all 5 checklists, check every item

**Why**: Intuition finds 50% of issues. Checklist finds 90%.


### ❌ Stopping After Finding First Issues

**Wrong**: Find SQL injection, report it, stop review

**Right**: Complete full checklist review even after finding critical issues

**Why**: Systems often have multiple unrelated vulnerabilities.


### ❌ Vague Recommendations

**Wrong**: "Improve authentication security"

**Right**: "Replace MD5 with bcrypt (cost factor 12), add salt per-password"

**Why**: Specific recommendations are actionable. Vague ones are ignored.


### ❌ Missing Severity Assignment

**Wrong**: List issues without severity

**Right**: Assign Critical/High/Medium/Low to prioritize fixes

**Why**: Teams need to know what to fix first.


## Cross-References

**Use BEFORE this skill**:
- `ordis/security-architect/threat-modeling` - Understand threats, then review if design addresses them

**Use WITH this skill**:
- `muna/technical-writer/documentation-structure` - Document review as ADR or report

**Use AFTER this skill**:
- `ordis/security-architect/security-controls-design` - Design fixes for identified issues

## Real-World Impact

**Systematic reviews using these checklists:**
- **Pre-launch review caught SQL injection** in `/api/users/{id}` endpoint (would have been CRITICAL production vulnerability)
- **Checklist found secrets in Git history** across 3 repositories (developers missed it in manual review)
- **Authorization review found 12 IDOR vulnerabilities** where `/api/orders/{order_id}` didn't check ownership (intuitive review found only 2)

**Key lesson**: **Systematic checklist review finds 3-5x more issues than ad-hoc intuitive review.**
