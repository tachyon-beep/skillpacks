---
name: secure-by-design-patterns
description: Build security into system foundations - zero-trust, immutable infrastructure, TCB minimization
---

# Secure By Design Patterns

## Overview

Build security into system foundations. Core principle: Design systems that are **secure by default, not secured after the fact**.

**Key insight**: Preventing security issues through architecture is cheaper and more effective than detecting and responding to them.

## When to Use

Load this skill when:
- Designing new systems (greenfield)
- Refactoring existing architecture
- Making fundamental architecture decisions
- Evaluating architecture proposals

**Symptoms you need this**:
- "How do we make this system secure?"
- Designing authentication, secrets management, data access
- Architecting microservices, data pipelines, distributed systems
- Choosing deployment/configuration strategies

**Don't use for**:
- Threat modeling specific attacks (use `ordis/security-architect/threat-modeling`)
- Implementing security controls (use `ordis/security-architect/security-controls-design`)
- Reviewing existing designs (use `ordis/security-architect/security-architecture-review`)

## Core Patterns

### Pattern 1: Zero-Trust Architecture

**Principle**: Never trust, always verify. No implicit trust based on network location.

#### Three Pillars

1. **Verify Explicitly**
   - Authenticate every request (no "internal network = trusted")
   - Authorize based on identity + context (device, location, time, risk)
   - Use strong authentication (mTLS, signed tokens, not IP allowlists alone)

2. **Least Privilege Access**
   - Grant minimum necessary permissions
   - Time-limited access (credentials expire, tokens rotate)
   - Resource-level authorization (not just service-level)

3. **Assume Breach**
   - Design for "when compromised", not "if compromised"
   - Minimize blast radius (segmentation, isolation)
   - Monitor everything (detect lateral movement)

#### Example: Microservices Communication

❌ **Not Zero-Trust**:
```
Service A → Service B (same network, no auth)
# Assumes: Internal network = trusted
# Risk: Compromised service A can access all of service B
```

✅ **Zero-Trust**:
```
Service A → Service B (mTLS + JWT + authz check)
# Every request authenticated + authorized
# Service B validates: Is this service A? Does it have permission for THIS resource?

Implementation:
- Service mesh (Istio/Linkerd) enforces mTLS
- Service B validates JWT from service A
- RBAC policy: service A can only access /resource/{own_resources}
- Audit log: Record all access attempts
```

**Result**: If service A compromised, attacker cannot impersonate other services or access resources outside A's scope.

---

### Pattern 2: Immutable Infrastructure

**Principle**: No runtime modifications. Replace rather than update.

#### Core Concepts

1. **Immutable Artifacts**
   - Container images, VM images, binaries
   - Never modified after creation
   - Versioned and signed

2. **Deployment Replaces Instances**
   - Updates = deploy new version, terminate old
   - No SSH into servers to patch
   - No runtime configuration changes

3. **Configuration as Code**
   - All config in version control
   - Deployments are reproducible
   - Rollback = redeploy previous version

#### Benefits

- **Security**: No drift from known-good state
- **Auditability**: All changes in version control
- **Rollback**: Redeploy previous image
- **Consistency**: Dev/staging/prod identical

#### Example: Application Updates

❌ **Mutable (Insecure)**:
```bash
# SSH into production server
ssh prod-server

# Update code
git pull origin main

# Restart service
systemctl restart app

# Problem: No audit trail, no rollback, config drift
```

✅ **Immutable (Secure)**:
```bash
# Build new image locally
docker build -t app:v2.1.0 .

# Push to registry (signed)
docker push registry/app:v2.1.0

# Deploy new version (Kubernetes)
kubectl set image deployment/app app=registry/app:v2.1.0

# Kubernetes: Creates new pods, terminates old pods
# Rollback if needed:
kubectl rollout undo deployment/app

# Result: Full audit trail, instant rollback, no drift
```

**Configuration Management**:
```yaml
# Configuration in Git, not edited on servers
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  API_TIMEOUT: "30"
  RATE_LIMIT: "1000"

# Changes = commit to Git → CI/CD applies → new deployment
```

---

### Pattern 3: Security Boundaries

**Principle**: Explicit trust zones with validation at every boundary crossing.

#### Boundary Identification

Trust boundaries are points where data/requests cross from lower-trust to higher-trust zones:

```
Internet (UNTRUSTED)
    ↓ BOUNDARY 1
API Gateway (TRUSTED)
    ↓ BOUNDARY 2
Application Services (TRUSTED)
    ↓ BOUNDARY 3
Database (HIGHLY TRUSTED)
```

#### Validation at Boundaries

At EACH boundary:
1. **Authenticate**: Verify identity
2. **Authorize**: Check permissions
3. **Validate**: Check input format/constraints
4. **Sanitize**: Remove dangerous content
5. **Log**: Record crossing

#### Example: Data Pipeline

```
External API (UNTRUSTED)
    ↓ BOUNDARY: API Client
    Validate: JSON schema, required fields
    Authenticate: API key
    Rate limit: 1000 req/hour

Message Queue (SEMI-TRUSTED)
    ↓ BOUNDARY: Consumer
    Validate: Message structure, idempotency key
    Authorize: Consumer has permission for this message type

Processing Service (TRUSTED)
    ↓ BOUNDARY: Database Writer
    Validate: Data types, constraints
    Authorize: Service can write to this table

Database (HIGHLY TRUSTED)
```

**Key insight**: Never trust data just because it came from "internal" service. Validate at every boundary.

#### Minimizing Boundary Surface Area

❌ **Large Surface Area (Insecure)**:
```
# Database accepts connections from all services
firewall: allow 0.0.0.0/0 → database:5432

# Problem: 50 services can connect, huge attack surface
```

✅ **Small Surface Area (Secure)**:
```
# Only specific services connect to database
firewall:
  allow backend-api → database:5432
  allow analytics → database:5432
  deny all others

# Further: Use service mesh with identity-based policies
```

---

### Pattern 4: Trusted Computing Base (TCB) Minimization

**Principle**: Small security-critical core, everything else untrusted.

#### What is TCB?

TCB = Components you MUST trust for security. If TCB is compromised, security fails.

**Goal**: Minimize TCB size (less code = fewer vulnerabilities).

#### Pattern: Small Critical Core

```
┌─────────────────────────────────────┐
│  Untrusted Zone (Applications)      │
│  - Web servers                       │
│  - Application logic                 │
│  - User-facing services              │
└────────────┬────────────────────────┘
             │ API calls (validated)
┌────────────▼────────────────────────┐
│  TRUSTED COMPUTING BASE (TCB)       │
│  - Authentication service (small!)  │
│  - Secrets vault (minimal code)     │
│  - Audit logger (append-only)       │
└─────────────────────────────────────┘
```

#### Example: Secrets Management

❌ **Large TCB (Risky)**:
```
# Every service has secrets management logic
# TCB = All 50+ services (huge attack surface)

each_service:
  - Fetches secrets from vault
  - Decrypts secrets
  - Manages rotation
  - Handles caching

# Problem: Bug in ANY service compromises secrets
```

✅ **Small TCB (Secure)**:
```
# Secrets Vault = TCB (small, auditable, formally verified)
# Applications = Untrusted (use vault API)

Vault (TCB):
  - 10,000 lines of code
  - Formally verified
  - Hardware-backed encryption (HSM)
  - Minimal attack surface (no network egress)

Applications (Untrusted):
  - Call vault API for secrets
  - Vault enforces all access control
  - Apps cannot access secrets they're not authorized for

# Result: Compromise application ≠ compromise vault
```

#### TCB Characteristics

1. **Small**: Minimize code size
2. **Auditable**: Can be formally verified
3. **Isolated**: Runs in separate environment (sandbox, separate machine)
4. **Minimal privileges**: TCB has no unnecessary access
5. **Heavily monitored**: All TCB access logged

---

### Pattern 5: Fail-Fast Security

**Principle**: Validate security properties at construction time. Refuse to operate if misconfigured.

#### Construction-Time vs Runtime

**Construction time**: When system/component is created (startup, initialization)
**Runtime**: When system is processing requests

**Fail-fast**: Validate security at construction, fail immediately if invalid.

#### Example: Security Level Validation

❌ **Runtime Validation (Vulnerable)**:
```python
# Data pipeline starts, processes data, THEN checks security
pipeline = Pipeline()
pipeline.add_source(untrusted_datasource)
pipeline.add_sink(trusted_datasink)
pipeline.start()  # Starts processing!

# Runtime: Check if datasource security level matches sink
for record in pipeline:
    if record.security_level > sink.max_security_level:
        raise SecurityError("Security mismatch!")

# Problem: Exposed data before detecting mismatch
# Exposure window = time until first mismatched record
```

✅ **Fail-Fast (Secure)**:
```python
# Validate security BEFORE processing any data
pipeline = Pipeline()
pipeline.add_source(untrusted_datasource)
pipeline.add_sink(trusted_datasink)

# BEFORE start: Validate security properties
if datasource.security_level > sink.max_security_level:
    raise SecurityError(
        f"Cannot create pipeline: Source {datasource.security_level} "
        f"exceeds sink maximum {sink.max_security_level}"
    )

pipeline.start()  # Only starts if validation passed

# Result: Zero exposure window, fail before processing data
```

#### Startup Validation Checklist

Validate at system startup (fail if any check fails):
- [ ] All required secrets accessible?
- [ ] TLS certificates valid and not expired?
- [ ] Database permissions granted?
- [ ] Security policies loaded?
- [ ] Encryption keys available?

#### Example: Service Startup

```python
class SecureService:
    def __init__(self):
        # Fail-fast validation at construction
        self._validate_security()

    def _validate_security(self):
        # Check TLS certificate
        if not self.tls_cert_valid():
            raise SecurityError("TLS certificate invalid or expired")

        # Check encryption keys accessible
        if not self.can_access_keys():
            raise SecurityError("Cannot access encryption keys")

        # Check database permissions
        if not self.has_required_db_permissions():
            raise SecurityError("Insufficient database permissions")

        # All checks passed
        logger.info("Security validation passed")

    def start(self):
        # Only callable after __init__ validation passed
        self.process_requests()

# Usage:
try:
    service = SecureService()  # Validates security at construction
    service.start()
except SecurityError as e:
    logger.error(f"Service failed security validation: {e}")
    sys.exit(1)  # Refuse to start with invalid security
```

**Benefits**:
- **No exposure window**: Catch misconfigurations before processing data
- **Clear errors**: Fail with specific message ("TLS cert expired")
- **Operational safety**: Misconfigured systems never reach production

---

## Pattern Application Framework

When designing systems, apply patterns in this order:

### 1. Identify Trust Boundaries (Security Boundaries)
Where does data cross trust zones?

### 2. Apply Zero-Trust at Each Boundary
- Authenticate + authorize every crossing
- Never trust based on network location

### 3. Minimize TCB
What MUST be trusted? Can it be smaller?

### 4. Use Immutable Infrastructure
Can deployments replace rather than update?

### 5. Add Fail-Fast Validation
Validate security at construction, refuse to start if invalid

---

## Quick Reference: Pattern Selection

| Situation | Pattern | Key Action |
|-----------|---------|------------|
| **Designing service-to-service communication** | Zero-Trust | mTLS + JWT + authz on every request |
| **Deciding deployment strategy** | Immutable Infrastructure | Container images, replace not update |
| **Architecting multi-tier system** | Security Boundaries | Validate + authenticate at every tier boundary |
| **Building secrets/auth service** | TCB Minimization | Small core, everything else uses API |
| **System startup logic** | Fail-Fast Security | Validate security before processing requests |

---

## Common Mistakes

### ❌ Implicit Trust Based on Network

**Wrong**: "Services in VPC are trusted, no auth needed"

**Right**: Zero-trust - authenticate/authorize every request even within VPC

**Why**: Network boundaries are weak. Compromised service = lateral movement without auth.

---

### ❌ Runtime Security Patches

**Wrong**: SSH into production, apply patch, restart

**Right**: Build new immutable image, deploy via CI/CD

**Why**: Runtime patches create drift, no audit trail, hard to rollback.

---

### ❌ Large Security-Critical Core

**Wrong**: Every service has secrets logic (large TCB)

**Right**: Small secrets vault (TCB), services call API (untrusted)

**Why**: Smaller TCB = fewer vulnerabilities, easier to audit/verify.

---

### ❌ Runtime Security Validation

**Wrong**: Start processing, check security during execution

**Right**: Validate security at construction, refuse to start if invalid

**Why**: Runtime checks have exposure windows. Fail-fast = zero exposure.

---

### ❌ Unclear Trust Boundaries

**Wrong**: No explicit boundaries, assume "internal is safe"

**Right**: Diagram trust zones, validate at every boundary crossing

**Why**: Boundaries are where attacks happen. Explicit validation prevents bypass.

---

## Cross-References

**Use BEFORE this skill**:
- `ordis/security-architect/threat-modeling` - Identify threats, then apply patterns to address them

**Use WITH this skill**:
- `ordis/security-architect/security-controls-design` - Patterns inform control choices

**Use AFTER this skill**:
- `ordis/security-architect/security-architecture-review` - Review architecture against patterns

## Real-World Impact

**Systems using secure-by-design patterns:**
- **Zero-trust + immutable infrastructure**: No successful lateral movement in 2 years despite multiple compromised services (blast radius contained by mTLS + segmentation)
- **Fail-fast validation**: Prevented VULN-004 class (security level overrides) by refusing to start pipelines with mismatched security levels
- **TCB minimization**: Secrets vault with 8,000 lines of code (vs 50+ services with embedded secrets logic) - single formal verification point instead of 50 attack surfaces

**Key lesson**: **Security built into architecture is more effective and cheaper than security added later.**
