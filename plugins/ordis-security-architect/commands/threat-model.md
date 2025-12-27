---
description: Systematic threat modeling using STRIDE methodology, attack trees, and risk scoring
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[system_or_component_to_analyze]"
---

# Threat Model Command

You are conducting systematic threat modeling to identify security threats that intuition misses.

## Core Principle

**Security intuition finds obvious threats. Systematic threat modeling finds subtle, critical threats that lead to real vulnerabilities.**

Do threat modeling BEFORE implementation - threats found after deployment are 10x more expensive to fix.

## STRIDE Framework

Apply STRIDE to EVERY component, interface, and data flow.

### S - Spoofing Identity

**Question**: Can attacker claim a different identity?

**Check for**:
- Authentication bypass
- Token forgery/theft
- Session hijacking
- API key leakage
- Credential stuffing

### T - Tampering with Data

**Question**: Can attacker modify data or code?

**Check for**:
- Data in transit modification (MITM)
- Data at rest tampering
- Configuration override attacks
- Code injection
- Supply chain compromise

**Critical Pattern**: Can configuration override security properties?

### R - Repudiation

**Question**: Can attacker deny performing an action?

**Check for**:
- Missing audit logs
- Log tampering
- Insufficient detail for forensics
- Log injection attacks

### I - Information Disclosure

**Question**: What data could be exposed?

**Check for**:
- Secrets in errors/logs
- Timing attacks
- Enumeration attacks
- Path traversal
- Cache poisoning

### D - Denial of Service

**Question**: Can attacker make system unavailable?

**Check for**:
- Resource exhaustion
- Unbounded operations
- Missing rate limits
- Algorithmic complexity attacks
- Crash via malformed input

### E - Elevation of Privilege

**Question**: Can attacker gain unauthorized capabilities?

**Check for**:
- Missing authorization checks
- Horizontal privilege escalation
- Vertical privilege escalation
- Type system bypasses

## Threat Modeling Process

### Step 1: System Decomposition

Identify:
1. **Entry points**: APIs, file uploads, configuration, user input
2. **Data stores**: Databases, caches, logs, files
3. **External dependencies**: Third-party APIs, libraries
4. **Trust boundaries**: Where privilege level changes
5. **Security-critical components**: Auth, access control, crypto

### Step 2: Apply STRIDE per Component

For EACH component/interface:

| Component | S | T | R | I | D | E |
|-----------|---|---|---|---|---|---|
| API Gateway | ? | ? | ? | ? | ? | ? |
| Database | ? | ? | ? | ? | ? | ? |
| Config Loader | ? | ? | ? | ? | ? | ? |

Fill in threats found.

### Step 3: Build Attack Trees

For high-priority threats, construct attack tree:

```
ROOT: Attacker Goal
├─ Attack Vector 1
│  ├─ Exploit Method A ⭐ (easiest)
│  └─ Exploit Method B
├─ Attack Vector 2
│  └─ Exploit Method C
```

Mark easiest paths with ⭐.

### Step 4: Score Risks

**Likelihood** (1-3):
- 3 (High): Easy to exploit, no special access needed
- 2 (Medium): Requires skill or access
- 1 (Low): Requires expertise, insider access, rare conditions

**Impact** (1-3):
- 3 (High): Complete compromise, data breach, safety risk
- 2 (Medium): Partial compromise, limited exposure
- 1 (Low): Minor leakage, temporary DoS

**Risk = Likelihood × Impact**

| Risk Score | Priority |
|------------|----------|
| 7-9 | Critical - Fix immediately |
| 4-6 | High - Fix before launch |
| 2-3 | Medium - Fix soon |
| 1 | Low - Fix if time permits |

### Step 5: Check Enforcement Gaps

For each security property, verify enforcement at ALL layers:
- Schema/Type layer
- Registration layer
- Construction layer
- Runtime layer
- Post-operation layer

Single-layer enforcement fails. Design defense-in-depth.

## Output Format

```markdown
# Threat Model: [System/Component Name]

## Scope

**Components analyzed**:
- [Component 1]
- [Component 2]

**Entry points**:
- [Entry point 1]
- [Entry point 2]

**Trust boundaries**:
- [Boundary 1]: [Less trusted] → [More trusted]

## STRIDE Analysis

### [Component Name]

| STRIDE | Threat | Priority |
|--------|--------|----------|
| S | [Spoofing threat] | High/Med/Low |
| T | [Tampering threat] | High/Med/Low |
| R | [Repudiation threat] | High/Med/Low |
| I | [Info disclosure threat] | High/Med/Low |
| D | [DoS threat] | High/Med/Low |
| E | [Elevation threat] | High/Med/Low |

## Attack Trees

### THREAT-001: [Goal]

```
ROOT: [Attacker Goal]
├─ [Vector 1]
│  ├─ [Exploit A] ⭐
│  └─ [Exploit B]
└─ [Vector 2]
```

**Easiest path**: [Description]

## Risk Assessment

| Threat ID | Description | L | I | Risk | Priority |
|-----------|-------------|---|---|------|----------|
| THREAT-001 | [Description] | 3 | 3 | 9 | Critical |
| THREAT-002 | [Description] | 2 | 3 | 6 | High |

## Enforcement Gap Analysis

### [Security Property]

| Layer | Check | Gap? |
|-------|-------|------|
| Schema | [What's checked] | Yes/No |
| Runtime | [What's checked] | Yes/No |

## Recommendations

### Critical (Fix Immediately)

1. **THREAT-001**: [Mitigation]

### High (Fix Before Launch)

1. **THREAT-002**: [Mitigation]

### Medium (Fix Soon)

1. **THREAT-003**: [Mitigation]
```

## Common Patterns That Intuition Misses

### Property Override via Configuration

**Check**: Can external config override security properties declared in code?

### Single-Layer Enforcement

**Check**: Is security check at only one layer? Can attacker bypass that layer?

### Type System as Security Boundary

**Check**: Does Protocol/interface provide security? Can duck typing bypass it?

### Trusted Component Assumptions

**Check**: Do trusted components have monitoring? What if compromised?

## Cross-Pack Discovery

```python
import glob

# For documenting threats
doc_pack = glob.glob("plugins/muna-technical-writer/plugin.json")
if doc_pack:
    print("Available: muna-technical-writer for threat documentation")

# For controls design
controls_ref = glob.glob("plugins/ordis-security-architect/skills/using-security-architect/security-controls-design.md")
if controls_ref:
    print("Available: security-controls-design.md for mitigation design")
```

## Scope Boundaries

**This command covers:**
- STRIDE threat analysis
- Attack tree construction
- Risk scoring
- Enforcement gap analysis
- Mitigation recommendations

**Not covered:**
- Security controls design (use /design-controls)
- Architecture review (use /security-review)
- Compliance mapping
