
# Mapping Security Surface

## Purpose

Surface security-relevant areas in subsystem catalogs for deeper review. Produces lightweight indicators that flag where security architect analysis is needed, without performing threat modeling or security assessments.

## When to Use

- After subsystem catalog completion (02-subsystem-catalog.md exists)
- User requests security overview during architecture analysis
- Preparing for security architect consultation
- Coordinator delegates security surface mapping
- Deliverable menu option E, F, or G selected

## Core Principle: Surface, Don't Assess

**Archaeologist surfaces security-relevant areas. Architect assesses threats.**

```
ARCHAEOLOGIST: "Found authentication patterns in 3 subsystems, secrets in config files"
ARCHITECT: "Auth uses weak hashing, secrets exposure is Critical severity"

ARCHAEOLOGIST: "Trust boundary crosses between API Gateway and external payment service"
ARCHITECT: "This boundary requires TLS pinning and request signing"
```

**Your role:** Document WHAT you observe. NOT how dangerous it is.

## Division of Labor (MANDATORY)

### What You DO:
- Identify subsystems with security-relevant code (auth, authz, crypto, secrets)
- Document trust boundaries (where external input enters, where sensitive data exits)
- Observe patterns (consistent vs inconsistent security approaches)
- Flag potential red flags for deeper review
- Produce handoff package for ordis-security-architect

### What You DO NOT Do:
- Assess severity ("this is critical")
- Prioritize security fixes ("fix auth before secrets")
- Make security recommendations ("use bcrypt instead of MD5")
- Perform threat modeling (STRIDE, attack trees)
- Calculate risk scores

**If you catch yourself assessing, STOP. Document the observation and move on.**

## Prerequisite Check (MANDATORY)

Before starting security surface mapping:

1. **Verify catalog exists:** `02-subsystem-catalog.md` in workspace
2. **Verify catalog validated:** Validation passed (check temp/validation-catalog.md)
3. **If prerequisites missing:** "Cannot map security surface - subsystem catalog required. Run /analyze-codebase first."

## Process Steps

### Step 1: Identify Security-Critical Subsystems

Scan catalog for subsystems with security-relevant responsibilities:

**Security Indicators by Responsibility:**

| Responsibility Contains | Security Relevance |
|------------------------|-------------------|
| "authentication", "login", "auth" | HIGH - Identity management |
| "authorization", "permission", "access control" | HIGH - Access decisions |
| "payment", "billing", "transaction" | HIGH - Financial data |
| "user", "account", "profile" | MEDIUM - PII handling |
| "API", "gateway", "endpoint" | MEDIUM - Entry points |
| "config", "settings", "environment" | MEDIUM - Secrets management |
| "database", "storage", "cache" | MEDIUM - Data persistence |
| "logging", "audit", "monitoring" | LOW - Security observability |

Mark each subsystem: `Security Relevance: [HIGH/MEDIUM/LOW/NONE]`

### Step 2: Map Trust Boundaries

For each HIGH/MEDIUM relevance subsystem, identify:

**Trust Boundary Types:**

1. **External Input Boundaries**
   - API endpoints receiving user input
   - File upload handlers
   - Configuration file parsers
   - Message queue consumers

2. **Privilege Transition Boundaries**
   - Authentication state changes (anonymous → authenticated)
   - Authorization checks (user → admin)
   - Service-to-service calls with different privileges

3. **Data Classification Boundaries**
   - Where sensitive data enters/exits system
   - PII crossing subsystem boundaries
   - Secrets used in external calls

**Document each boundary:**
```markdown
**Trust Boundary:** [Name]
- Type: [External Input / Privilege Transition / Data Classification]
- Subsystems: [A] <-> [B]
- Mechanism: [HTTP/gRPC/file/queue]
- Evidence: [file:line where boundary observed]
```

### Step 3: Check Security Properties

For each security-critical subsystem, verify presence of:

**Authentication Properties:**
```
[ ] Credential storage mechanism identified
[ ] Session/token management observed
[ ] Password policy enforcement (if applicable)
[ ] MFA indicators (if applicable)
```

**Authorization Properties:**
```
[ ] Access control model identified (RBAC/ABAC/ACL)
[ ] Resource-level authorization checks
[ ] Default-deny principle (explicit allow required)
[ ] Privilege separation between roles
```

**Secrets Management Properties:**
```
[ ] Secrets source identified (env vars/vault/config)
[ ] No hardcoded credentials in source
[ ] Secrets not logged in plaintext
[ ] Rotation mechanism (if observable)
```

**Data Protection Properties:**
```
[ ] Encryption in transit (TLS/HTTPS)
[ ] Encryption at rest (if observable)
[ ] Input validation at boundaries
[ ] Output encoding for user-facing data
```

**For each property:**
- `Present` - Observed with evidence (file:line)
- `Absent` - Looked for, not found
- `Unclear` - Could not determine (document why)

### Step 4: Detect Red Flags

**MANDATORY Red Flag Checklist**

Before writing "No red flags observed," verify you checked for:

| Red Flag | How to Check | Evidence Required |
|----------|--------------|-------------------|
| **Property Override via Config** | Search for security properties that config can modify | File:line where config affects security |
| **Single-Layer Validation** | Check if auth/authz only at API layer, not data layer | Files where validation present/absent |
| **Hardcoded Secrets** | Grep for API keys, passwords, tokens in source | File:line of any findings |
| **Secrets in Logs** | Check logging calls for sensitive data | Logging statements reviewed |
| **Missing Input Validation** | Check entry points for validation | Entry point file:lines |
| **Inconsistent Patterns** | Compare security approach across subsystems | Pattern comparison evidence |
| **Privilege Escalation Paths** | Check if users can modify own roles/permissions | Role modification code |
| **Weak Crypto Indicators** | Search for MD5, SHA1, DES in security contexts | Crypto usage file:lines |

**If ANY red flag found:**
```markdown
**Red Flag:** [Name]
- Location: [file:line]
- Observation: [What you found - factual, not assessed]
- Recommendation: "Refer to ordis-security-architect:threat-modeling"
```

### Step 5: Assess Confidence

**Confidence Levels for Security Surface:**

**High Confidence:**
```markdown
**Confidence:** High - Read auth implementation (auth/handler.py lines 1-200),
verified secrets handling in config/settings.py, checked all 5 entry points
for validation, compared patterns across 4 subsystems.
```

**Medium Confidence:**
```markdown
**Confidence:** Medium - Read auth module structure but not all implementations.
Secrets source identified but rotation not verified. 3/7 entry points checked.
```

**Low Confidence:**
```markdown
**Confidence:** Low - Limited time for security review. Only checked auth
subsystem. Other security-relevant subsystems not examined.
```

## Output Contract (MANDATORY)

Write to `07-security-surface.md` in workspace:

```markdown
# Security Surface Analysis

**Analysis Date:** YYYY-MM-DD
**Scope:** [What was reviewed]
**Confidence:** [High/Medium/Low] - [evidence summary]

## Security-Critical Subsystems

| Subsystem | Relevance | Trust Boundaries | Red Flags |
|-----------|-----------|------------------|-----------|
| [Name] | HIGH | 2 | 1 |
| [Name] | MEDIUM | 1 | 0 |

## Trust Boundary Map

### Boundary 1: [Name]
- **Type:** [External Input / Privilege Transition / Data Classification]
- **Subsystems:** [A] <-> [B]
- **Mechanism:** [HTTP/gRPC/file/etc.]
- **Evidence:** [file:line]

### Boundary 2: [Name]
...

## Security Properties by Subsystem

### [Subsystem Name]

**Authentication:**
- Credential storage: [Present/Absent/Unclear] - [evidence]
- Session management: [Present/Absent/Unclear] - [evidence]

**Authorization:**
- Access control model: [RBAC/ABAC/ACL/None observed] - [evidence]
- Resource-level checks: [Present/Absent/Unclear] - [evidence]

**Secrets Management:**
- Source: [env vars/vault/config/hardcoded] - [evidence]
- Logging safety: [Present/Absent/Unclear] - [evidence]

**Data Protection:**
- Transit encryption: [Present/Absent/Unclear] - [evidence]
- Input validation: [Present/Absent/Unclear] - [evidence]

### [Next Subsystem]
...

## Red Flags Detected

### Red Flag 1: [Name]
- **Location:** [file:line]
- **Observation:** [Factual description of what was found]
- **Checklist Item:** [Which red flag category]

### Red Flag 2: [Name]
...

(Or: "No red flags detected. Verified checklist items: [list what was checked]")

## Patterns Observed

- **Consistent:** [What's done the same way across subsystems]
- **Inconsistent:** [What varies - potential concern]

## Information Gaps

- [What couldn't be analyzed and why]
- [Subsystems not reviewed]
- [Properties that couldn't be verified]

## Security Architect Handoff

**Recommended for deeper review:**
- ordis-security-architect:threat-modeling for:
  - [Specific subsystem 1]: [specific concern]
  - [Specific subsystem 2]: [specific concern]

- ordis-security-architect:security-architecture-review for:
  - [Area needing comprehensive review]

**Priority indicators:**
- [X] red flags requiring assessment
- [Y] HIGH-relevance subsystems needing threat modeling
- [Z] trust boundaries crossing external systems
```

## Common Rationalizations (STOP SIGNALS)

| Rationalization | Reality |
|-----------------|---------|
| "I can see this is insecure" | Your job is to SURFACE, not ASSESS. Document observation, refer to architect. |
| "This is obviously critical" | Severity assessment is architect's job. Document what you found. |
| "I should recommend the fix" | Recommendations are prescriptive. You describe, architect prescribes. |
| "The red flag checklist is overkill for this simple codebase" | Simple codebases have security issues too. Run the checklist. |
| "I already looked at auth during cataloging" | Catalog focus ≠ security focus. Dedicated security pass finds different issues. |
| "Time pressure - I'll skip the checklist" | Partial security surface is worse than none. Scope down, don't skip checklist. |
| "I'll mark confidence as High since it's a small codebase" | Confidence is about verification depth, not codebase size. |

## Anti-Patterns

**DON'T make severity assessments:**
```
WRONG: "Critical vulnerability: MD5 used for password hashing"
RIGHT: "Observed: Password hashing uses MD5 (auth/hash.py:45).
        Refer to ordis-security-architect for assessment."
```

**DON'T prioritize fixes:**
```
WRONG: "Fix secrets exposure before addressing auth"
RIGHT: "Red flags detected: secrets in config (1), auth pattern inconsistency (1).
        Refer to ordis-security-architect:prioritizing-improvements"
```

**DON'T skip the red flag checklist:**
```
WRONG: "No obvious security issues observed"
RIGHT: "Red flag checklist completed. Checked: property override (clear),
        single-layer validation (clear), hardcoded secrets (clear), ..."
```

**DON'T speculate:**
```
WRONG: "This probably has injection vulnerabilities"
RIGHT: "Input validation: Unclear - entry points identified but validation
        logic not reviewed. Recommend deeper review."
```

## Success Criteria

**You succeeded when:**
- All security-critical subsystems identified with relevance level
- Trust boundaries documented with evidence
- Security properties checked for each relevant subsystem
- Red flag checklist completed (all items checked, not skipped)
- Confidence level reflects actual verification depth
- Output follows contract format exactly
- Handoff to security-architect is explicit with specific recommendations
- Written to 07-security-surface.md

**You failed when:**
- Made severity assessments ("this is critical")
- Prioritized security fixes
- Skipped red flag checklist items
- Marked confidence High without evidence of thorough review
- Speculated about vulnerabilities without evidence
- Produced assessment instead of surface mapping
- Output doesn't follow contract

## Integration with Workflow

Security surface mapping is invoked:
1. After subsystem catalog completion and validation
2. When user selects deliverable option E (Full + Security), F, or G
3. Before architect handover if security concerns exist
4. Feeds into 06-architect-handover.md security section

**Pipeline:** Catalog → Validation → Security Surface → Architect Handoff → Security Architect

## Cross-Plugin Handoff

When security surface mapping is complete:

```
Security surface analysis complete. Produced 07-security-surface.md with:
- [X] security-critical subsystems identified
- [Y] trust boundaries mapped
- [Z] red flags detected

For comprehensive security analysis, recommend:
- ordis-security-architect:threat-modeling for [specific areas]
- ordis-security-architect:security-architecture-review for [specific subsystems]
```

**DO NOT attempt threat modeling, attack trees, or risk scoring yourself.**
