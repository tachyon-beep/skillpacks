---
name: security-controls-design
description: Layered security controls at trust boundaries - defense-in-depth, least privilege, fail-secure
---

# Security Controls Design

## Overview

Design security controls as **layered defenses at trust boundaries**. Core principle: Apply systematic checks at every boundary to ensure no single control failure compromises security.

**Key insight**: List specific controls after identifying WHERE to apply them (trust boundaries first, then controls).

## When to Use

Load this skill when:
- Implementing authentication/authorization systems
- Hardening API endpoints, databases, file storage
- Designing data protection mechanisms
- Securing communication channels
- Protecting sensitive operations

**Symptoms you need this**:
- "How do I secure this API/database/upload feature?"
- "What controls should I implement?"
- "How do I prevent unauthorized access to X?"
- "How do I harden this system?"

**Don't use for**:
- Threat modeling (use `ordis/security-architect/threat-modeling` first)
- Code-level security patterns (use `ordis/security-architect/secure-code-patterns`)
- Reviewing existing designs (use `ordis/security-architect/security-architecture-review`)

## Core Methodology: Trust Boundaries First

**DON'T start with**: "What controls should I implement?"

**DO start with**: "Where are the trust boundaries?"

###Step 1: Identify Trust Boundaries

Trust boundaries are **points where data/requests cross from less-trusted to more-trusted zones**.

**Common boundaries:**
- Internet → API Gateway
- API Gateway → Application Server
- Application → Database
- Application → File Storage
- Unauthenticated → Authenticated
- User Role → Admin Role
- External Service → Internal Service

**Example: File Upload System**
```
Trust Boundaries:
1. User Browser → Upload Endpoint (UNTRUSTED → APP)
2. Upload Endpoint → Virus Scanner (APP → SCANNER)
3. Scanner → Storage (SCANNER → S3)
4. Storage → Display (S3 → USER)
5. Storage → Internal Processing (S3 → APP)
```

### Step 2: Apply Defense-in-Depth at Each Boundary

For EACH boundary, apply multiple control layers. If one fails, others provide backup.

## Defense-in-Depth Checklist

Use this checklist at **every trust boundary**:

### Layer 1: Validation (First Line)
- [ ] **Input validation**: Type, format, size, allowed values
- [ ] **Sanitization**: Remove dangerous characters, escape output
- [ ] **Canonicalization**: Resolve to standard form (prevent bypass)

### Layer 2: Authentication (Who Are You?)
- [ ] **Identity verification**: Credentials, tokens, certificates
- [ ] **Multi-factor authentication**: For sensitive boundaries
- [ ] **Session management**: Secure tokens, expiration, rotation

### Layer 3: Authorization (What Can You Do?)
- [ ] **Access control checks**: RBAC, ABAC, resource-level
- [ ] **Least privilege enforcement**: Grant minimum necessary
- [ ] **Privilege escalation prevention**: No path to higher access

### Layer 4: Rate Limiting (Abuse Prevention)
- [ ] **Request rate limits**: Per-IP, per-user, per-endpoint
- [ ] **Resource quotas**: Prevent resource exhaustion
- [ ] **Anomaly detection**: Flag unusual patterns

### Layer 5: Audit Logging (Detective)
- [ ] **Security event logging**: Who, what, when, where, outcome
- [ ] **Tamper-proof logs**: Write-only for applications
- [ ] **Alerting**: Automated detection of suspicious activity

### Layer 6: Encryption (Confidentiality)
- [ ] **Data in transit**: TLS 1.3, certificate validation
- [ ] **Data at rest**: Encryption for sensitive data
- [ ] **Key management**: Secure storage, rotation, separation

**Example Application** (API Authentication Boundary):
```
Internet → API Gateway boundary:

Layer 1 (Validation):
- Validate Authorization header present and well-formed
- Check request size limits (prevent DoS)
- Validate content-type and payload structure

Layer 2 (Authentication):
- Verify JWT signature (RS256, public key validation)
- Check token expiration (exp claim)
- Verify token not revoked (check Redis revocation list)

Layer 3 (Authorization):
- Extract scopes from token
- Verify endpoint requires scope present in token
- Check resource-level permissions (can user access THIS resource?)

Layer 4 (Rate Limiting):
- Per-token: 1000 requests/minute
- Per-IP: 100 requests/minute (catch token sharing)
- Per-endpoint: Stricter limits on write operations

Layer 5 (Audit Logging):
- Log authentication attempts (success/failure)
- Log authorization decisions (allowed/denied)
- Log resource access (who accessed what)

Layer 6 (Encryption):
- Enforce TLS 1.3 only (reject unencrypted)
- Validate certificate chain
- Store tokens encrypted in session store
```

**If ANY layer fails, others provide defense.**

## Fail-Secure Patterns

When a control fails, system should **default to secure state** (deny access, close connection, reject request).

### Fail-Closed (Secure) vs Fail-Open (Insecure)

| Situation | Fail-Open (❌ BAD) | Fail-Closed (✅ GOOD) |
|-----------|-------------------|---------------------|
| **Auth service down** | Allow all requests through | Deny all requests until service recovers |
| **Token validation fails** | Treat as valid | Reject request |
| **Database unreachable** | Skip permission check | Deny access |
| **Rate limit store unavailable** | No rate limiting | Apply strictest default limit |
| **Audit log fails to write** | Continue operation | Reject operation |

### Examples of Fail-Secure Implementation

**Example 1: Authentication Service Failure**
```python
def authenticate_request(request):
    try:
        token = extract_token(request)
        user = auth_service.validate_token(token)  # External service call
        return user
    except AuthServiceUnavailable:
        # ❌ FAIL-OPEN: return AnonymousUser()  # Let them through
        # ✅ FAIL-CLOSED: raise Unauthorized("Authentication service unavailable")
        raise Unauthorized("Authentication service unavailable")
    except InvalidToken:
        raise Unauthorized("Invalid token")
```

**Example 2: Rate Limiter Failure**
```python
def check_rate_limit(user_id):
    try:
        redis.incr(f"rate:{user_id}")
        count = redis.get(f"rate:{user_id}")
        if count > LIMIT:
            raise RateLimitExceeded()
    except RedisConnectionError:
        # ❌ FAIL-OPEN: return  # Let request through
        # ✅ FAIL-CLOSED: Apply strictest default limit
        # If Redis is down, apply aggressive in-memory rate limit
        in_memory_limiter.check(user_id, limit=10)  # Much stricter than normal
```

**Example 3: Database Permission Check**
```python
def can_user_access_resource(user_id, resource_id):
    try:
        permission = db.query(
            "SELECT can_read FROM permissions WHERE user_id = ? AND resource_id = ?",
            user_id, resource_id
        )
        return permission.can_read
    except DatabaseConnectionError:
        # ❌ FAIL-OPEN: return True  # Assume they have access
        # ✅ FAIL-CLOSED: return False  # Deny access if can't verify
        logger.error(f"DB unavailable, denying access for user={user_id} resource={resource_id}")
        return False
```

**Example 4: File Type Validation**
```python
def validate_file_upload(file):
    # Layer 1: Check extension
    if file.extension not in ALLOWED_EXTENSIONS:
        raise ValidationError("Invalid file type")

    # Layer 2: Check magic bytes
    try:
        magic_bytes = file.read(16)
        if not is_valid_magic_bytes(magic_bytes):
            # ✅ FAIL-CLOSED: If magic bytes don't match, reject
            # Even if extension passed, magic bytes take precedence
            raise ValidationError("File content doesn't match extension")
    except Exception as e:
        # ❌ FAIL-OPEN: return True  # Couldn't check, assume valid
        # ✅ FAIL-CLOSED: raise ValidationError("Could not validate file")
        raise ValidationError(f"Could not validate file: {e}")
```

**Principle**: **When in doubt, deny**. It's better to have a false positive (deny legitimate request) than false negative (allow malicious request).

## Least Privilege Principle

Grant **minimum necessary access** for each component to perform its function. No more.

### Application Method

**For each component, ask three questions:**
1. **What does it NEED to do?** (functional requirements)
2. **What's the MINIMUM access to achieve that?** (reduce scope)
3. **What can it NEVER do?** (explicit denials)

### Example: Database Access Roles

**Web Application Role:**
```sql
-- What it NEEDS: Read customers, write audit logs
GRANT SELECT ON customers TO web_app_user;
GRANT INSERT ON audit_logs TO web_app_user;

-- What's MINIMUM: No DELETE, no UPDATE on audit logs (immutable), no admin tables
REVOKE DELETE ON customers FROM web_app_user;
REVOKE ALL ON admin_users FROM web_app_user;

-- Explicit NEVER: Cannot modify audit logs (tamper-proof)
REVOKE UPDATE, DELETE ON audit_logs FROM web_app_user;

-- Row-level security: Only active customers
CREATE POLICY web_app_access ON customers
  FOR SELECT TO web_app_user
  USING (status = 'active');
```

**Analytics Role:**
```sql
-- What it NEEDS: Read non-PII customer data for analytics
-- What's MINIMUM: View with PII columns excluded
CREATE VIEW customers_analytics AS
  SELECT customer_id, country, subscription_tier, created_at
  FROM customers;  -- Excludes: name, email, address

GRANT SELECT ON customers_analytics TO analytics_user;

-- What it can NEVER do: Access PII, modify data, see payment info
REVOKE ALL ON customers FROM analytics_user;
REVOKE ALL ON payment_info FROM analytics_user;
SET default_transaction_read_only = true FOR analytics_user;
```

### File System Permissions

**Application Server:**
```bash
# What it NEEDS: Read config, write logs, read/write uploads
/etc/app/config/          → Read-only (owner: root, chmod 640, group: app)
/var/log/app/             → Write-only (owner: app, chmod 200, append-only)
/var/uploads/             → Read/write (owner: app, chmod 700)

# What it can NEVER do: Write to config, execute from uploads
/etc/app/config/          → No write permissions
/var/uploads/             → Mount with noexec flag (prevent execution)
```

### API Scopes (OAuth2 Pattern)

```python
# User requests minimal scopes
scopes_requested = ["read:profile", "read:posts"]

# DON'T grant admin scopes by default
# DO grant only what was requested and approved
token = create_token(user, scopes=scopes_requested)

# At each endpoint, verify scope
@require_scope("write:posts")
def create_post(request):
    # This endpoint is inaccessible with read:posts scope
    pass
```

**Principle**: **Default deny, explicit allow**. Start with no access, grant only what's needed.

## Separation of Duties

**No single component/person/account should have complete control** over a critical operation.

### Patterns

#### Pattern 1: Multi-Signature Approvals

**Example: Production Deployments**
```yaml
# Require 2 approvals from different teams
approvals:
  required: 2
  teams:
    - engineering-leads
    - security-team

# Cannot approve own PR
prevent_self_approval: true
```

#### Pattern 2: Split Responsibilities

**Example: Payment Processing**
```python
# Component A: Initiates payment (can create, cannot approve)
payment_service.initiate_payment(amount, account)

# Component B: Approves payment (can approve, cannot create)
# Different credentials, different service
approval_service.approve_payment(payment_id)

# Component C: Executes payment (can execute, cannot create/approve)
# Only accepts approved payments
execution_service.execute_payment(approved_payment_id)
```

**No single service can create AND approve AND execute a payment.**

#### Pattern 3: Key Splitting

**Example: Encryption Key Management**
```python
# Master key split into 3 shares using Shamir Secret Sharing
# Require 2 of 3 shares to reconstruct
shares = split_key(master_key, threshold=2, num_shares=3)

# Distribute to different teams/locations
security_team.store(shares[0])
ops_team.store(shares[1])
compliance_team.store(shares[2])

# Reconstruction requires 2 teams to cooperate
reconstructed = reconstruct_key([shares[0], shares[1]])
```

#### Pattern 4: Admin Operations Require Approval

**Example: Database Admin Actions**
```python
# Admin initiates action (creates request, cannot execute)
admin_request = AdminRequest(
    action="DELETE_USER",
    user_id=12345,
    reason="GDPR erasure request",
    requested_by=admin_id
)

# Second admin reviews and approves (cannot initiate)
reviewer.approve(admin_request, reviewer_id=different_admin_id)

# System executes after approval (automated, no single admin control)
if admin_request.is_approved():
    execute_admin_action(admin_request)
```

**Principle**: **Break critical paths into multiple steps requiring different actors.**

## Control Verification Method

For **each control you design**, ask: **"What if this control fails?"**

### Verification Checklist

**For each control:**
1. **What attack does this prevent?** (threat it addresses)
2. **How can this control fail?** (failure modes)
3. **What happens if it fails?** (impact)
4. **What's the next layer of defense?** (backup control)
5. **Is failure logged/detected?** (observability)

### Example: API Token Validation

**Control**: Verify JWT signature before processing request

1. **What attack**: Prevents forged tokens, ensures authenticity
2. **How it can fail**:
   - Public key unavailable (service down)
   - Expired token not caught (clock skew)
   - Token revocation list unavailable (Redis down)
   - Signature algorithm downgrade attack (accept HS256 instead of RS256)
3. **What if it fails**:
   - Public key unavailable → Fail-closed (deny all requests)
   - Expired token → Layer 2: Check expiration explicitly
   - Revocation list down → Layer 3: Apply strict rate limits as fallback
   - Algorithm downgrade → Layer 4: Explicitly require RS256, reject others
4. **Next layer**:
   - Authorization checks (even with valid token, check permissions)
   - Rate limiting (limit damage from compromised token)
   - Audit logging (detect unusual access patterns)
5. **Failure logged**: Yes → Log signature validation failures, alert on spike

**Outcome**: Designed 4 layers of defense against token attacks.

### Example: File Upload Validation

**Control**: Check file extension against allowlist

1. **What attack**: Prevents upload of executable files (.exe, .sh)
2. **How it can fail**:
   - Attacker renames malware.exe → malware.jpg
   - Double extension: malware.jpg.exe
   - Case variation: malware.ExE
3. **What if it fails**: Malicious file stored, potentially executed
4. **Next layers**:
   - Layer 2: Magic byte verification (check file content, not name)
   - Layer 3: Antivirus scanning (detect known malware)
   - Layer 4: File reprocessing (re-encode images, destroying embedded code)
   - Layer 5: noexec mount (storage prevents execution)
   - Layer 6: Separate domain for user content (CSP prevents XSS)
5. **Failure logged**: Yes → Log validation failures, rejected files

**Outcome**: Extension check is Layer 1 of 6. If bypassed, 5 more layers prevent exploitation.

## Quick Reference: Control Selection

**For every trust boundary, apply this checklist:**

| Layer | Control Type | Example |
|-------|--------------|---------|
| **1. Validation** | Input checking | Size limits, type validation, sanitization |
| **2. Authentication** | Identity verification | JWT validation, certificate checks, MFA |
| **3. Authorization** | Permission checks | RBAC, resource-level access, least privilege |
| **4. Rate Limiting** | Abuse prevention | Per-user limits, anomaly detection, quotas |
| **5. Audit Logging** | Detective | Security events, tamper-proof logs, alerting |
| **6. Encryption** | Confidentiality | TLS in transit, encryption at rest, key management |

**For each control:**
- Define fail-secure behavior (what happens if it fails?)
- Apply least privilege (minimum necessary access)
- Verify separation of duties (no single point of complete control)
- Test "what if this fails?" (ensure backup layers exist)

## Common Mistakes

### ❌ Designing Controls Before Identifying Boundaries

**Wrong**: "I need authentication and authorization and rate limiting"

**Right**: "Where are my trust boundaries? → Internet→API, API→Database → At each: apply layered controls"

**Why**: Controls are meaningless without knowing WHERE to apply them.

---

### ❌ Single Layer of Defense

**Wrong**: "Authentication is enough security"

**Right**: "Authentication + Authorization + Rate Limiting + Audit Logging"

**Why**: If authentication is bypassed (bug, misconfiguration), other layers provide defense.

---

### ❌ Fail-Open Defaults

**Wrong**:
```python
try:
    user = auth_service.validate(token)
except ServiceUnavailable:
    user = AnonymousUser()  # Let them through
```

**Right**:
```python
try:
    user = auth_service.validate(token)
except ServiceUnavailable:
    raise Unauthorized("Auth service unavailable")
```

**Why**: Control failure should result in secure state (deny), not insecure state (allow).

---

### ❌ Excessive Privileges

**Wrong**: Grant web application full database access (SELECT, INSERT, UPDATE, DELETE on all tables)

**Right**: Grant only needed operations per table (SELECT on customers, INSERT-only on audit_logs)

**Why**: Minimizes damage from compromised application (SQL injection, stolen credentials).

---

### ❌ Single Point of Control

**Wrong**: One admin account can initiate, approve, and execute critical operations

**Right**: Separate accounts for initiate vs approve, require multi-signature

**Why**: Prevents single compromised account from complete system control.

---

### ❌ No Verification of "What If This Fails?"

**Wrong**: Design controls, assume they work

**Right**: For each control, ask "how can this fail?" and design backup layers

**Why**: Controls fail due to bugs, misconfigurations, attacks. Backup layers provide resilience.

## Cross-References

**Use BEFORE this skill**:
- `ordis/security-architect/threat-modeling` - Identify threats first, then design controls to address them

**Use WITH this skill**:
- `muna/technical-writer/documentation-structure` - Document control architecture as ADR

**Use AFTER this skill**:
- `ordis/security-architect/security-architecture-review` - Review controls for completeness

## Real-World Impact

**Well-designed controls using this methodology:**
- Multi-layered API authentication catching token forgery even when signature validation was bypassed (algorithm confusion attack)
- Database access controls limiting SQL injection damage to read-only operations (least privilege prevented data deletion)
- File upload defenses stopping malware despite extension check bypass (magic bytes + antivirus + reprocessing layers)

**Key lesson**: **Systematic application of defense-in-depth at trust boundaries is more effective than ad-hoc control selection.**
