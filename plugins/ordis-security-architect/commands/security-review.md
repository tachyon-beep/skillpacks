---
description: Review architecture for security gaps, missing controls, and defense-in-depth
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[architecture_or_design_to_review]"
---

# Security Review Command

You are reviewing architecture for security gaps, ensuring defense-in-depth at all trust boundaries.

## Core Principle

**Apply systematic checks at every boundary to ensure no single control failure compromises security.**

Trust boundaries are where data/requests cross from less-trusted to more-trusted zones. Every boundary needs layered defenses.

## Review Scope Determination

### Ask First

1. **What system/design is being reviewed?**
2. **What security requirements exist?** (compliance, data sensitivity)
3. **What threats have been identified?** (if threat model exists)
4. **What's the deployment context?** (cloud, on-prem, hybrid)

## Security Review Framework

### 1. Trust Boundary Analysis

**Identify all boundaries**:
- Internet → API Gateway
- API Gateway → Application
- Application → Database
- Application → External Services
- User → Authenticated User
- User → Admin

**For each boundary, check**:
- What controls exist?
- Is there defense-in-depth (multiple layers)?
- What happens if a control fails?

### 2. Control Layer Assessment

**Six Defense Layers** (check each boundary):

| Layer | Control Type | Check |
|-------|--------------|-------|
| 1. Validation | Input checking | Size, type, format, sanitization |
| 2. Authentication | Identity verification | Strong auth, MFA for sensitive |
| 3. Authorization | Permission checking | RBAC/ABAC, resource-level |
| 4. Rate Limiting | Abuse prevention | Per-user, per-endpoint limits |
| 5. Audit Logging | Detective | Security events, tamper-proof |
| 6. Encryption | Confidentiality | TLS in transit, encryption at rest |

### 3. Fail-Secure Assessment

**For each control, check**:
- What happens if this control fails?
- Does it fail-closed (deny) or fail-open (allow)?
- Is failure logged/detected?

**Fail-Open = Critical Finding**

### 4. Least Privilege Assessment

**Check**:
- Does each component have minimum necessary access?
- Are database roles scoped appropriately?
- Are API scopes minimal?
- Are file permissions restricted?

### 5. Separation of Duties Assessment

**Check**:
- Can single account/component perform entire critical operation?
- Are sensitive operations split across multiple actors?
- Is self-approval prevented?

## Review Process

### Step 1: Map Architecture

Create or review architecture diagram showing:
- Components
- Data flows
- Trust boundaries
- External dependencies

### Step 2: Identify Trust Boundaries

List all points where privilege level changes.

### Step 3: Assess Each Boundary

For each boundary, apply 6-layer defense check.

### Step 4: Check Fail-Secure

For each control, verify fail-secure behavior.

### Step 5: Assess Access Controls

Verify least privilege and separation of duties.

### Step 6: Document Findings

Categorize by severity, provide recommendations.

## Output Format

```markdown
# Security Review: [System Name]

## Scope

**System**: [Description]
**Review Date**: [Date]
**Components Reviewed**:
- [Component 1]
- [Component 2]

## Architecture Overview

[Brief description or diagram reference]

## Trust Boundaries Identified

| Boundary | From | To | Risk Level |
|----------|------|----|------------|
| B1 | Internet | API Gateway | High |
| B2 | API | Database | High |

## Control Assessment by Boundary

### Boundary: Internet → API Gateway

| Layer | Control | Status | Finding |
|-------|---------|--------|---------|
| Validation | Input validation | ✓ Present | Size limits, type checking |
| Authentication | JWT validation | ✓ Present | RS256, expiration checked |
| Authorization | Scope checking | ⚠ Partial | Missing resource-level |
| Rate Limiting | Per-user limits | ✓ Present | 1000/hour |
| Audit Logging | Request logging | ✗ Missing | No auth failure logging |
| Encryption | TLS 1.3 | ✓ Present | Strong cipher suites |

**Findings**:
1. [SEV-HIGH] Missing resource-level authorization
2. [SEV-MEDIUM] No audit logging for auth failures

### Boundary: [Next Boundary]

[Same format]

## Fail-Secure Assessment

| Control | Failure Behavior | Status |
|---------|-----------------|--------|
| Auth service | Fail-closed (deny) | ✓ Secure |
| Rate limiter | Fail-open (allow) | ✗ CRITICAL |
| Database | Fail-closed | ✓ Secure |

**Finding**: Rate limiter fails open when Redis unavailable

## Least Privilege Assessment

| Component | Current Access | Minimum Needed | Status |
|-----------|---------------|----------------|--------|
| Web app | Full DB access | SELECT customers, INSERT logs | ⚠ Excessive |
| Analytics | All tables | Non-PII view only | ⚠ Excessive |

## Separation of Duties Assessment

| Critical Operation | Split? | Finding |
|-------------------|--------|---------|
| Production deploy | ✓ Yes | Requires 2 approvals |
| Admin user delete | ✗ No | Single admin can delete |

## Findings Summary

| ID | Severity | Finding | Location |
|----|----------|---------|----------|
| F-001 | Critical | Rate limiter fails open | API Gateway |
| F-002 | High | Missing resource auth | API Gateway |
| F-003 | High | Excessive DB privileges | Web App |
| F-004 | Medium | No auth failure logging | API Gateway |
| F-005 | Medium | Admin delete not split | Admin Panel |

## Recommendations

### Critical (Fix Immediately)

**F-001: Rate limiter fails open**
- Implement in-memory fallback with strict limits
- Alert on Redis unavailability

### High (Fix Before Launch)

**F-002: Missing resource-level authorization**
- Add resource ownership check before access
- Implement ABAC for cross-tenant protection

**F-003: Excessive database privileges**
- Create scoped database roles
- Apply row-level security

### Medium (Fix Soon)

**F-004: No auth failure logging**
- Log all authentication attempts
- Alert on failure spike

**F-005: Admin delete not split**
- Require second admin approval for deletions
```

## Common Security Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| Single-layer auth | Auth bypass = full access | Add authorization layer |
| Fail-open controls | Degraded = no security | Implement fail-closed |
| Excessive privileges | Breach = max damage | Apply least privilege |
| No audit logging | Undetected attacks | Log security events |
| Missing rate limits | DoS/brute force | Per-user/endpoint limits |

## Cross-Pack Discovery

```python
import glob

# For threat modeling (do before review if not done)
threat_ref = glob.glob("plugins/ordis-security-architect/skills/using-security-architect/threat-modeling.md")
if threat_ref:
    print("Available: threat-modeling.md for systematic threat identification")

# For controls design after review
controls_ref = glob.glob("plugins/ordis-security-architect/skills/using-security-architect/security-controls-design.md")
if controls_ref:
    print("Available: security-controls-design.md for implementing fixes")
```

## Scope Boundaries

**This command covers:**
- Trust boundary analysis
- Defense layer assessment
- Fail-secure verification
- Access control review
- Gap identification

**Not covered:**
- Threat identification (use /threat-model first)
- Controls implementation (use /design-controls)
- Compliance mapping
