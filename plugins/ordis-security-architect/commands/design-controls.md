---
description: Design layered security controls at trust boundaries with defense-in-depth
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[boundary_or_component_to_secure]"
---

# Design Controls Command

You are designing security controls as layered defenses at trust boundaries.

## Core Principle

**No single control failure should compromise security. Apply defense-in-depth at every trust boundary.**

Start with WHERE (trust boundaries), then WHAT (controls at each layer).

## Control Design Process

### Step 1: Identify Trust Boundary

**Common boundaries**:
- Internet → API Gateway
- API Gateway → Application
- Application → Database
- Application → File Storage
- Unauthenticated → Authenticated
- User → Admin

### Step 2: Apply Six Defense Layers

For the identified boundary, design controls at each layer:

| Layer | Purpose | Controls |
|-------|---------|----------|
| 1. Validation | First line | Input type, size, format, sanitization |
| 2. Authentication | Who are you? | Credentials, tokens, certificates, MFA |
| 3. Authorization | What can you do? | RBAC, ABAC, resource-level checks |
| 4. Rate Limiting | Abuse prevention | Per-user, per-endpoint, quotas |
| 5. Audit Logging | Detective | Security events, tamper-proof, alerting |
| 6. Encryption | Confidentiality | TLS in transit, encryption at rest |

### Step 3: Design Fail-Secure Behavior

For each control, define what happens on failure:

| Failure Scenario | Fail-Closed (✓) | Fail-Open (✗) |
|-----------------|-----------------|---------------|
| Auth service down | Deny all requests | Allow through |
| Invalid token | Reject request | Treat as valid |
| Rate limit store unavailable | Apply strict default | No rate limiting |
| DB unreachable | Deny access | Skip permission check |

**Always fail-closed for security controls.**

### Step 4: Apply Least Privilege

For each component accessing the boundary:
1. What does it NEED to do?
2. What's MINIMUM access to achieve that?
3. What can it NEVER do?

### Step 5: Design Separation of Duties

For critical operations:
- Split across multiple actors
- Require multi-approval
- Prevent self-approval

## Control Templates

### API Authentication Boundary

```markdown
## Control Design: Internet → API Gateway

### Layer 1: Validation
- Request size limit: 10MB max
- Content-Type: application/json required
- Header validation: Authorization header present
- Parameter sanitization: Escape special characters

### Layer 2: Authentication
- Method: JWT with RS256 signing
- Token source: Authorization: Bearer header
- Expiration: 1 hour max
- Revocation: Check Redis revocation list
- **Fail behavior**: Reject on any validation failure

### Layer 3: Authorization
- Scope extraction: From JWT claims
- Endpoint requirements: Documented per endpoint
- Resource-level: Owner check for user resources
- **Fail behavior**: Deny if scope missing

### Layer 4: Rate Limiting
- Per-token: 1000 requests/minute
- Per-IP: 100 requests/minute (catch sharing)
- Per-endpoint: Stricter on write operations
- **Fail behavior**: Apply 10 req/min fallback

### Layer 5: Audit Logging
- Log: All auth attempts (success/failure)
- Log: All authorization decisions
- Log: Resource access (who accessed what)
- Storage: Write-only for application
- Alert: On failure rate spike

### Layer 6: Encryption
- TLS: 1.3 required, reject older
- Certificates: Validate chain
- Token storage: Encrypted at rest
```

### Database Access Boundary

```markdown
## Control Design: Application → Database

### Layer 1: Validation
- Query parameterization: All queries use prepared statements
- Input validation: Before query construction
- Output encoding: Escape in responses

### Layer 2: Authentication
- Connection: Certificate-based auth
- Rotation: Credentials rotated monthly
- Storage: Secrets manager, not config files

### Layer 3: Authorization (Least Privilege)
| Role | Tables | Operations |
|------|--------|------------|
| web_app | customers | SELECT |
| web_app | audit_logs | INSERT (no UPDATE/DELETE) |
| web_app | admin_users | NONE |
| analytics | customers_view | SELECT (no PII) |

### Layer 4: Rate Limiting
- Connection pooling: Max 20 connections per service
- Query timeout: 30 seconds max
- Rate: Max 100 queries/second per service

### Layer 5: Audit Logging
- Query logging: All queries with user context
- Sensitive access: Alert on admin table access
- Anomaly detection: Flag unusual query patterns

### Layer 6: Encryption
- In transit: TLS required
- At rest: Column encryption for PII
- Key management: HSM-backed keys
```

### File Upload Boundary

```markdown
## Control Design: User → File Storage

### Layer 1: Validation
- Extension allowlist: [.jpg, .png, .pdf, .doc]
- Magic byte verification: Match extension to content
- Size limit: 50MB max
- Filename sanitization: Remove path characters

### Layer 2: Authentication
- Require authenticated session
- Associate file with user identity

### Layer 3: Authorization
- Upload: Only to user's directory
- Download: Only own files (or shared explicitly)
- Delete: Only own files

### Layer 4: Rate Limiting
- Uploads: 10/minute per user
- Total storage: 1GB per user quota

### Layer 5: Audit Logging
- Log: All uploads with hash, size, user
- Log: All downloads with accessor
- Alert: Large file volumes

### Layer 6: Encryption
- Upload: Over TLS only
- Storage: Encrypted at rest
- Virus scan: Before storage

### Additional Controls
- Antivirus: Scan all uploads
- Content reprocessing: Re-encode images (strips malware)
- Execution prevention: noexec mount on storage
- Separate domain: Serve user content from different domain
```

## Output Format

```markdown
# Security Controls: [Boundary Name]

## Trust Boundary

**From**: [Less trusted zone]
**To**: [More trusted zone]
**Risk Level**: [High/Medium/Low]

## Defense Layers

### Layer 1: Validation

**Controls**:
- [Control 1]: [Implementation detail]
- [Control 2]: [Implementation detail]

**Fail behavior**: [What happens on validation failure]

### Layer 2: Authentication

[Same format]

### Layer 3: Authorization

[Same format]

### Layer 4: Rate Limiting

[Same format]

### Layer 5: Audit Logging

[Same format]

### Layer 6: Encryption

[Same format]

## Least Privilege Configuration

| Component | Resource | Access | Rationale |
|-----------|----------|--------|-----------|
| [Component] | [Resource] | [Permissions] | [Why needed] |

## Separation of Duties

| Critical Operation | Actors Required | Approval Flow |
|-------------------|-----------------|---------------|
| [Operation] | [Number/Roles] | [Process] |

## Fail-Secure Summary

| Control | Failure Scenario | Behavior |
|---------|-----------------|----------|
| [Control] | [How it fails] | Deny/Allow |

## Implementation Checklist

- [ ] Layer 1: Validation implemented
- [ ] Layer 2: Authentication implemented
- [ ] Layer 3: Authorization implemented
- [ ] Layer 4: Rate limiting implemented
- [ ] Layer 5: Audit logging implemented
- [ ] Layer 6: Encryption implemented
- [ ] Fail-secure behavior verified
- [ ] Least privilege applied
- [ ] Separation of duties implemented
```

## Control Verification

For each control, verify:
1. **What attack does this prevent?**
2. **How can this control fail?**
3. **What's the next layer of defense?**
4. **Is failure logged/detected?**

## Cross-Pack Discovery

```python
import glob

# For threat modeling (should be done first)
threat_ref = glob.glob("plugins/ordis-security-architect/skills/using-security-architect/threat-modeling.md")
if threat_ref:
    print("Available: threat-modeling.md for identifying threats to control")

# For documentation
doc_pack = glob.glob("plugins/muna-technical-writer/plugin.json")
if doc_pack:
    print("Available: muna-technical-writer for documenting controls")
```

## Scope Boundaries

**This command covers:**
- Trust boundary identification
- Six-layer defense design
- Fail-secure configuration
- Least privilege design
- Separation of duties design

**Not covered:**
- Threat identification (use /threat-model first)
- Architecture review (use /security-review)
- Implementation code
