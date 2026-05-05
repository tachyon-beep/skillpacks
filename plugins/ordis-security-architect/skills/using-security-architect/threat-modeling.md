
# Threat Modeling

## Overview

Systematic identification of security threats using proven frameworks. Threat modeling finds threats that intuition misses by applying structured methodologies.

**Core Principle**: Security intuition finds obvious threats. Systematic threat modeling finds subtle, critical threats that lead to real vulnerabilities.

## When to Use

Load this skill when:
- Designing new systems or features (before implementation)
- Adding security-sensitive functionality (auth, data handling, APIs)
- Reviewing existing designs for security gaps
- Investigating security incidents (what else could be exploited?)
- User mentions: "threat model", "security risks", "what could go wrong", "attack surface"

**Use BEFORE implementation** - threats found after deployment are 10x more expensive to fix.

## Don't Use For

- **Implementing specific security controls** (use security-controls-design)
- **Code-level security patterns** (use secure-by-design-patterns)
- **Reviewing existing designs for completeness** (use security-architecture-review)
- **Compliance mapping** (use compliance-awareness-and-mapping)
- **Documenting threats after they're identified** (use documenting-threats-and-controls)

This skill is for IDENTIFYING threats systematically. Once threats are identified, route to appropriate skills for designing controls, implementing patterns, or documenting decisions.

## The STRIDE Framework

**STRIDE** is a systematic threat enumeration framework. Apply to EVERY component, interface, and data flow.

### S - Spoofing Identity

**Definition**: Attacker pretends to be someone/something else

**Questions to Ask**:
- Can attacker claim a different identity?
- Is authentication required? Can it be bypassed?
- Are credentials properly validated?
- Can tokens/sessions be stolen or forged?

**Example Threats**:
- Stolen authentication tokens
- Forged JWT signatures
- Session hijacking via XSS
- API key leakage in logs


### T - Tampering with Data

**Definition**: Unauthorized modification of data or code

**Questions to Ask**:
- Can attacker modify data in transit? (MITM)
- Can attacker modify data at rest? (database, files, config)
- Can attacker modify code? (supply chain, config injection)
- **Can configuration override security properties?** (CRITICAL - often missed)

**Example Threats**:
- Configuration files modifying security_level properties
- YAML/JSON injection overriding access controls
- Database tampering if encryption/MAC missing
- Code injection via deserialization

**⚠️ Property Override Pattern** (VULN-004):
```yaml
# Plugin declares security_level=UNOFFICIAL in code
# Attacker adds to YAML config:
plugins:
  datasource:
    security_level: SECRET  # OVERRIDES code declaration!
```

Always ask: **"Can external configuration override security-critical properties?"**


### R - Repudiation

**Definition**: Attacker denies performing an action (no audit trail)

**Questions to Ask**:
- Are security-relevant actions logged?
- Can logs be tampered with or deleted?
- Is logging sufficient for forensics?
- Can attacker perform reconnaissance without detection?

**Example Threats**:
- No logging of failed authorization attempts
- Logs stored without integrity protection (MAC, signatures)
- Insufficient detail for incident response
- Log injection attacks


### I - Information Disclosure

**Definition**: Exposure of information to unauthorized parties

**Questions to Ask**:
- What data is exposed in responses, logs, errors?
- Can attacker enumerate resources or users?
- Are temporary files, caches, or memory properly cleared?
- Can attacker infer sensitive data from timing/behavior?

**Example Threats**:
- Secrets in error messages or stack traces
- Timing attacks revealing password validity
- Cache poisoning exposing other users' data
- Path traversal reading arbitrary files


### D - Denial of Service

**Definition**: Making system unavailable or degrading performance

**Questions to Ask**:
- Are there resource limits (CPU, memory, connections)?
- Can attacker trigger expensive operations?
- Is rate limiting implemented?
- Can attacker cause crashes or hangs?

**Example Threats**:
- Unbounded recursion or loops
- Memory exhaustion via large payloads
- Algorithmic complexity attacks (e.g., hash collisions)
- Crash via malformed input


### E - Elevation of Privilege

**Definition**: Gaining capabilities beyond what's authorized

**Questions to Ask**:
- Can attacker access admin functions?
- Can attacker escalate from low to high privilege?
- Are privilege checks performed at every layer?
- **Can type system be bypassed?** (ADR-004 pattern)

**Example Threats**:
- Missing authorization checks on sensitive endpoints
- Horizontal privilege escalation (access other users' data)
- Vertical privilege escalation (user → admin)
- Duck typing allowing security bypass


## Attack Tree Construction

**Purpose**: Visual/structured representation of attack paths from goal → exploitation

### Attack Tree Format

```
ROOT: Attacker Goal (e.g., "Access classified data")
├─ BRANCH 1: Attack Vector
│  ├─ LEAF: Specific Exploit (with feasibility)
│  └─ LEAF: Alternative Exploit
├─ BRANCH 2: Alternative Vector
│  └─ LEAF: Exploit Method
```

### Example: Configuration Override Attack (VULN-004)

```
ROOT: Access classified data with insufficient clearance
├─ Override Plugin Security Level
│  ├─ Inject security_level into YAML config ⭐ (VULN-004 - actually happened)
│  ├─ Modify plugin source code (requires code access)
│  └─ Bypass registry to register malicious plugin (ADR-003 gap)
├─ Exploit Trusted Downgrade
│  ├─ Compromise high-clearance component (supply chain)
│  └─ Abuse legitimate downgrade path (ADR-005 gap)
├─ Bypass Type System
│  └─ Duck-type plugin without BasePlugin inheritance (ADR-004 gap)
```

**⭐ = Easiest/highest risk path**

### How to Build Attack Trees

1. **Start with attacker goal**: What does attacker want? (data access, DoS, privilege escalation)
2. **Branch by attack vector**: How could they achieve it? (config, network, code)
3. **Leaf nodes are specific exploits**: Concrete technical steps
4. **Mark feasibility**: Easy, Medium, Hard (or Low/Med/High effort)
5. **Identify easiest path**: This is your highest priority to mitigate


## Enforcement Gap Analysis

**Pattern**: Security properties must be enforced at EVERY layer. Single-layer enforcement fails.

### Layers to Check

**For any security property (e.g., security_level, access control, data classification):**

1. **Schema/Type Layer**: Is property type-safe? Can it be None/invalid?
2. **Registration Layer**: Is component registered? Can attacker bypass registry?
3. **Construction Layer**: Is property immutable after creation? Can it be modified?
4. **Runtime Layer**: Is property checked before sensitive operations?
5. **Post-Operation Layer**: Is result validated against expected property?

### Example: MLS Security Level Enforcement (ADR-002 → 005)

| Layer | Gap Found | Fix Required |
|-------|-----------|--------------|
| **Registry** | Plugin not registered at all (ADR-003) | Central plugin registry with runtime checks |
| **Type System** | Protocol allows duck typing bypass (ADR-004) | ABC with sealed methods, not Protocol |
| **Immutability** | security_level could be mutated (VULN-009) | Frozen dataclass + runtime checks |
| **Trust** | Trusted downgrade assumes no compromise (ADR-005) | Strict mode disables trusted downgrade |

**Key Insight**: Each gap was found AFTER implementation. Systematic enforcement gap analysis would have caught all four upfront.

### How to Apply

For each security property:
1. **List all layers** where property matters
2. **Ask per layer**: "Can attacker bypass this layer?"
3. **Design defense-in-depth**: Redundant checks at multiple layers


## Standard Threat Identifiers (CWE / CVSS / ATT&CK / ATLAS)

Threat-model artifacts that travel outside your team — to security
auditors, blue teams, regulators, or downstream consumers — should be
tagged with **standard identifiers** so others can cross-reference them
against their own tooling.

### CWE — What kind of weakness is this?

**CWE** (Common Weakness Enumeration, MITRE) categorizes the *class* of
defect. Tag every threat with the most specific CWE ID that fits.

Common IDs to know:

| CWE | Name | Use when |
|-----|------|----------|
| CWE-20 | Improper Input Validation | Most input-handling defects (umbrella) |
| CWE-22 | Path Traversal | File access via untrusted path |
| CWE-77 / CWE-78 | Command / OS Command Injection | Shell exec with untrusted input |
| CWE-79 | Cross-Site Scripting | HTML/JS rendered from untrusted input |
| CWE-89 | SQL Injection | SQL constructed from untrusted input |
| CWE-94 | Code Injection | `eval`/dynamic code from untrusted input |
| CWE-200 | Information Exposure | Sensitive data leaked in response/log/error |
| CWE-269 | Improper Privilege Management | Authorization gaps, over-broad grants |
| CWE-287 / CWE-306 | Improper / Missing Authentication | Auth bypass or absent |
| CWE-352 | CSRF | Cross-site state-changing forgery |
| CWE-400 / CWE-770 | Resource Exhaustion / Allocation w/o Limits | DoS |
| CWE-434 | Unrestricted File Upload | Upload of dangerous types |
| CWE-502 | Deserialization of Untrusted Data | Pickle/YAML/Java deserialization |
| CWE-918 | SSRF | Server-side fetch from untrusted URL |
| CWE-1357 / CWE-1395 | Reliance on Vulnerable Component / Dependency | Supply-chain |
| CWE-1426 | Improper Validation of Generative AI Output | LLM output handling |

The full corpus is at `cwe.mitre.org`. CWE has a Top 25 published
annually — **CWE Top 25 (2024)** is the current list at time of writing;
verify the current edition when citing.

### CVSS — How severe is the vulnerability?

**CVSS** (Common Vulnerability Scoring System) is the industry-standard
severity score for *concrete vulnerabilities* (not threats in the
abstract). Useful when a threat has been validated as exploitable and
you need to communicate severity to non-experts or feed into a SOC
prioritization queue.

- **CVSS v3.1**: Still the most widely deployed (FIRST, NVD).
- **CVSS v4.0**: Released Nov 2023, adopted by NVD in 2024. Adds threat
  metrics (exploit maturity), supplemental metrics (safety, automatable),
  and environmental refinements.

Use the calculator at `first.org/cvss/calculator/4.0` (or v3.1).
Severity bands (v3.1 / v4.0): None 0, Low 0.1–3.9, Medium 4.0–6.9,
High 7.0–8.9, Critical 9.0–10.0.

**EPSS** (Exploit Prediction Scoring System) is a complementary score —
*probability of exploitation in the next 30 days* — published by FIRST
at `first.org/epss`. Use CVSS for *severity*, EPSS for *likelihood
under current threat conditions*. The combination beats either alone for
prioritization.

**SSVC** (Stakeholder-Specific Vulnerability Categorization, CISA) is an
alternative decision-tree model that some federal/critical-infrastructure
contexts prefer over scalar scores.

### MITRE ATT&CK — How would the adversary do it?

**ATT&CK** (`attack.mitre.org`) catalogs adversary tactics and
techniques — *what they do once they're in*. Tag attack-tree leaves
with technique IDs so blue teams can map detections.

Tactics are the broad goals (e.g., **TA0001 Initial Access**, **TA0004
Privilege Escalation**, **TA0010 Exfiltration**). Techniques (`T####`)
are the specific methods.

High-frequency techniques to know:

| ID | Technique |
|----|-----------|
| T1190 | Exploit Public-Facing Application |
| T1078 | Valid Accounts (credential abuse) |
| T1059 | Command and Scripting Interpreter |
| T1505 | Server Software Component |
| T1195 | Supply Chain Compromise (.001 dev tools, .002 software, .003 hardware) |
| T1566 | Phishing |
| T1110 | Brute Force |
| T1486 | Data Encrypted for Impact (ransomware) |

ATT&CK has matrices for Enterprise, Mobile, ICS, and Containers. Pick
the matrix that fits the system under analysis.

### MITRE ATLAS — Adversarial threats against AI

**ATLAS** (`atlas.mitre.org`) is the AI-specific equivalent. Tag LLM /
ML threats with `AML.T####` IDs.

Common IDs:

| ID | Technique |
|----|-----------|
| AML.T0051 | LLM Prompt Injection |
| AML.T0054 | LLM Jailbreak |
| AML.T0019 | Publish Poisoned Datasets |
| AML.T0020 | Poison Training Data |
| AML.T0024 | Exfiltration via ML Inference API |
| AML.T0010 | ML Supply Chain Compromise |
| AML.T0053 | LLM Plugin Compromise |

Cross-link to `llm-and-ai-security.md` when modeling LLM-using systems.

### CVE — When the threat is a known vulnerability

If the threat in question is *exploitation of a published CVE*, cite
the CVE ID (`CVE-YYYY-NNNNN`). Verify via `nvd.nist.gov` or `cve.org`.
Do not invent CVE numbers.

### Tagging Convention

Add columns to the threat table:

```markdown
| ID | Description | STRIDE | CWE | ATT&CK / ATLAS | CVSS (v3.1 / v4.0) | EPSS | Risk |
|----|-------------|--------|-----|-----------------|--------------------|------|------|
| T-01 | Config-override of `security_level` | T,E | CWE-1357 | T1565.003 | 9.1 (Critical) | 0.06 | 9 |
| T-02 | Plugin duck-type bypass | E   | CWE-863 | T1078 | 7.5 (High) | 0.02 | 6 |
```

You don't need every column for every threat — but the discipline of
tagging surfaces gaps (e.g., a threat with no plausible CWE is often
too vague to mitigate).

---

## Risk Scoring

**Purpose**: Prioritize threats by (Likelihood × Impact)

The 1–3 × 1–3 matrix below is the recommended **default for
architectural threat modeling** — it's pedagogically simple, quick to
apply, and consistent across a team. Use **CVSS** when the threat is a
concrete vulnerability with a defined attack vector and you need to
communicate severity outside the team. Use **CVSS + EPSS** (or **SSVC**)
when feeding a SOC prioritization queue. The two scoring systems are
complementary: the matrix for design-time prioritization, CVSS for
operational triage.

### Likelihood Scale

| Score | Likelihood | Criteria |
|-------|-----------|----------|
| **3** | High | Easy to exploit, attacker has means and motive, no special access needed |
| **2** | Medium | Requires some skill or access, exploit path exists but not trivial |
| **1** | Low | Requires significant expertise, insider access, or rare conditions |

### Impact Scale

| Score | Impact | Criteria |
|-------|--------|----------|
| **3** | High | Complete system compromise, data breach, financial loss, safety risk |
| **2** | Medium | Partial compromise, limited data exposure, service degradation |
| **1** | Low | Minor information leakage, temporary DoS, limited scope |

### Risk Matrix

```
         IMPACT
         1   2   3
       ┌───┬───┬───┐
     3 │ M │ H │ C │  C = Critical (fix immediately)
L    2 │ L │ M │ H │  H = High (fix before launch)
I    1 │ L │ L │ M │  M = Medium (fix soon)
K      └───┴───┴───┘  L = Low (fix if time permits)
```

### Example: VULN-004 Config Override

- **Likelihood**: 3 (High) - YAML files easily modified with filesystem access
- **Impact**: 3 (High) - Bypass MLS enforcement, access classified data
- **Risk Score**: 9 (Critical) - **Fix immediately**

### Example: ADR-004 Type System Bypass

- **Likelihood**: 2 (Medium) - Requires knowing to create duck-typed plugin
- **Impact**: 3 (High) - Complete security bypass
- **Risk Score**: 6 (High) - **Fix before launch**


## Threat Modeling Workflow

### Step 1: System Decomposition

Break system into components:
1. **Entry points**: APIs, file uploads, configuration, user input
2. **Data stores**: Databases, caches, logs, files
3. **External dependencies**: Third-party APIs, libraries, services
4. **Trust boundaries**: Where privilege level changes, network boundaries
5. **Security-critical components**: Auth, access control, crypto, secrets management


### Step 2: Apply STRIDE per Component

For EACH component/interface, systematically ask STRIDE questions:

**Example: Plugin Configuration Component**

| STRIDE | Threat Found | Priority |
|--------|-------------|----------|
| **S** | None (no identity claims) | - |
| **T** | Config tampering to override security_level (VULN-004) | Critical |
| **R** | Config changes not logged | Medium |
| **I** | Config may contain secrets in plaintext | High |
| **D** | Malformed YAML causes parser crash | Low |
| **E** | Config override elevates plugin privilege | Critical |


### Step 3: Build Attack Trees

For each high-priority threat, construct attack tree:
- Goal: What does attacker want?
- Vectors: How could they get it?
- Exploits: Specific technical steps

Mark easiest paths with ⭐.


### Step 4: Check Enforcement Gaps

For each security property (authentication, authorization, encryption):
1. List enforcement layers (schema, registry, runtime, etc.)
2. Check each layer for gaps
3. Design redundant checks (defense-in-depth)


### Step 5: Score and Prioritize

- Calculate Likelihood × Impact for each threat
- Sort by risk score (highest first)
- Set mitigation deadlines (Critical → immediate, High → before launch)


### Step 6: Document Threats

Create threat model document:
```markdown
# Threat Model: [System Name]

## Scope
[Components, entry points, trust boundaries]

## Threats Identified

### THREAT-001: Configuration Override Attack (CRITICAL)
**STRIDE**: Tampering, Elevation of Privilege
**Attack Tree**: [Include tree diagram or text description]
**Risk Score**: 9 (L:3 × I:3)
**Mitigation**: Forbid security_level in config (schema), runtime verification, frozen dataclass

### THREAT-002: [Next threat...]

## Enforcement Gaps
[List gaps found in defense-in-depth analysis]

## Risk Matrix
[Include prioritized threat list]
```


## Common Patterns That Intuition Misses

### Pattern 1: Property Override via Configuration

**Symptom**: Security property declared in code, but configuration system allows overriding it

**Example**: VULN-004 - Plugin declares security_level in code, YAML config overrides it

**How to Spot**:
- Code declares security property (access_level, security_level, role)
- Configuration system loads external data (YAML, JSON, database)
- No explicit check that config cannot override security properties

**Mitigation**: Schema MUST forbid security properties in config, runtime verification


### Pattern 2: Enforcement at One Layer Only

**Symptom**: Security check at one layer, but attacker can bypass that layer

**Example**: ADR-003 - MLS checks assume plugin is registered, but no check that plugin IS registered

**How to Spot**:
- Security check at schema/type layer but not runtime
- Trust in single source of truth (registry, type system) without verification
- No redundant checks

**Mitigation**: Defense-in-depth - check at schema, registry, runtime, post-operation


### Pattern 3: Type System as Security Boundary

**Symptom**: Relying on type system (Protocol, interface) for security enforcement

**Example**: ADR-004 - Protocol typing allows duck typing to bypass BasePlugin

**How to Spot**:
- Security property defined in Protocol or interface
- No nominal type enforcement (isinstance check, ABC)
- Runtime doesn't verify actual type, just duck typing compatibility

**Mitigation**: Use ABC with sealed methods, runtime isinstance checks


### Pattern 4: Trusted Component Assumptions

**Symptom**: Assuming high-privilege component will never be compromised

**Example**: ADR-005 - Trusted downgrade assumes high-clearance component is always safe

**How to Spot**:
- Component granted special privileges ("trusted")
- No monitoring or verification of trusted component behavior
- Insider threat or supply chain compromise not in threat model

**Mitigation**: Trust but verify - log all actions, anomaly detection, strict mode without trust


### Pattern 5: Immutability Assumption

**Symptom**: Assuming language feature (frozen, const, final) provides security

**Example**: VULN-009 - Frozen dataclass but __dict__ bypass possible

**How to Spot**:
- Security property marked frozen/immutable via language feature
- No runtime check that property hasn't changed
- Language feature has known bypasses (__dict__, __setattr__)

**Mitigation**: Language feature + runtime verification + test all bypass methods


## Quick Reference Checklist

**Use this checklist for every threat modeling session:**

### Pre-Session
- [ ] Identify scope (components, entry points, trust boundaries)
- [ ] Gather architecture diagrams, API specs, data flow diagrams

### STRIDE Application
- [ ] Apply S.T.R.I.D.E to EVERY component/interface
- [ ] Document threats found per category
- [ ] Check for property override patterns
- [ ] Check for enforcement gap patterns

### Attack Trees
- [ ] Build attack tree for each high-priority threat
- [ ] Mark easiest exploitation paths
- [ ] Identify pre-requisites (what attacker needs)

### Risk Scoring
- [ ] Score Likelihood (1-3) for each threat
- [ ] Score Impact (1-3) for each threat
- [ ] Calculate Risk = L × I
- [ ] Prioritize by risk score

### Enforcement Gaps
- [ ] List security properties (auth, authorization, encryption, etc.)
- [ ] For each property, check: Schema? Registry? Runtime? Post-op?
- [ ] Identify gaps in defense-in-depth

### Documentation
- [ ] Create threat model document
- [ ] Include attack trees, risk matrix, mitigation plans
- [ ] Share with team for review


## Common Mistakes

### ❌ Intuitive Threat Finding Only
**Wrong**: "I'll just think about what could go wrong"
**Right**: Systematically apply STRIDE to every component

**Why**: Intuition finds obvious threats. STRIDE finds subtle, critical threats like VULN-004.

### ❌ Threat Modeling After Implementation
**Wrong**: "Let's build it first, then threat model"
**Right**: Threat model BEFORE implementation

**Why**: Threats found post-implementation require expensive re-architecture. Threats found in design are cheap to fix.

### ❌ Single-Layer Validation
**Wrong**: "Schema validates config, so it's secure"
**Right**: Validate at schema, registry, runtime, post-operation

**Why**: Attackers bypass single layers. Defense-in-depth catches them.

### ❌ Trusting Language Features for Security
**Wrong**: "It's frozen=True, so it can't be modified"
**Right**: Language feature + runtime verification + test bypass methods

**Why**: Language features have bypasses (VULN-009). Always verify.

### ❌ Skipping Risk Scoring
**Wrong**: "All threats are important, fix them all"
**Right**: Score L×I, prioritize Critical/High, fix Low only if time permits

**Why**: Resources are limited. Critical threats must be fixed first.


## Real-World Examples

### Example 1: VULN-004 - Configuration Override Attack

**System**: Plugin system with YAML configuration and MLS security levels

**STRIDE Analysis**:
- **T** (Tampering): Config file tampering ✓
- **E** (Elevation): Override security_level property ✓ **← Caught by STRIDE**

**Attack Tree**:
```
Goal: Access classified data
└─ Override security_level to SECRET
   ├─ Inject security_level: SECRET into YAML ⭐ (easiest)
   ├─ Modify source code (harder)
   └─ Compromise plugin registry (harder)
```

**Risk Score**: L:3 × I:3 = 9 (Critical)

**Mitigation**: Forbid security_level in config schema + runtime verification


### Example 2: ADR-002 → 005 - MLS Design Gaps

**System**: Multi-Level Security enforcement for plugins

**Enforcement Gap Analysis**:
1. **Registry Layer**: No check plugin is registered (ADR-003) ✓
2. **Type Layer**: Protocol allows duck typing (ADR-004) ✓
3. **Immutability**: security_level could be mutated (VULN-009) ✓
4. **Trust**: Trusted downgrade assumes no compromise (ADR-005) ✓

**All four gaps found by systematic enforcement analysis** - would have prevented 3 follow-up ADRs.

**Risk Scores**:
- ADR-003 (registry): L:2 × I:3 = 6 (High)
- ADR-004 (type): L:2 × I:3 = 6 (High)
- ADR-005 (trust): L:1 × I:3 = 3 (Medium)


## When NOT to Threat Model

**Don't threat model for**:
- Non-security features (UI styling, analytics dashboards with no sensitive data)
- Changes that don't touch attack surface (refactoring internal code, renaming variables)
- Systems with no sensitive data and no attack value (internal dev tools, prototypes)

**Quick test**: If attacker can't gain anything (data, money, access, disruption), threat modeling may be overkill.


## Other Threat-Modeling Methodologies

STRIDE is the default in this skill because it's well-suited to
architecture-level threat enumeration. Other methodologies cover
different angles — pick the one that matches the question:

| Methodology | Focus | When to use |
|-------------|-------|-------------|
| **STRIDE** | Per-component threat enumeration | Architecture-level (default) |
| **LINDDUN** | Privacy threats (Linkability, Identifiability, Non-repudiation, Detectability, Disclosure, Unawareness, Non-compliance) | Personal-data systems, GDPR/HIPAA contexts |
| **PASTA** | 7-stage process tying business impact to technical threats | Programs needing exec-level risk traceability |
| **OCTAVE Allegro** | Asset-centric organizational risk | Strategic/enterprise risk, not per-system |
| **Attack Trees** | Goal-oriented exploitation paths | Already covered above; pairs with any methodology |
| **Kill chain / Cyber Kill Chain** | Adversary phases | Detection/blue-team alignment; pairs with ATT&CK |

For privacy-heavy systems, run **LINDDUN** alongside STRIDE — STRIDE
finds confidentiality and integrity threats but undercounts linkability,
identifiability, and unawareness threats that GDPR-style regulators
care about.

For LLM/AI systems, the OWASP Top 10 for LLM Applications (2025) is
the right enumeration; see `llm-and-ai-security.md`.

For supply-chain threats, see `supply-chain-security.md`.

---

## Cross-References

### Load These Skills Together

**For comprehensive security**:
- `ordis/security-architect/threat-modeling` (this skill) - Find threats
- `ordis/security-architect/security-controls-design` - Design mitigations
- `ordis/security-architect/secure-by-design-patterns` - Prevent threats at architecture level

**For LLM/AI systems**:
- `ordis/security-architect/llm-and-ai-security` - OWASP LLM Top 10, MITRE ATLAS, agentic threats

**For supply-chain threats**:
- `ordis/security-architect/supply-chain-security` - SLSA, SBOM, Sigstore, dependency threats

**For documentation**:
- `ordis/security-architect/documenting-threats-and-controls` - Document threat model
- `muna/technical-writer/documentation-structure` - Structure threat docs as ADRs


## Summary

**Threat modeling IS systematic threat discovery using STRIDE, attack trees, and risk scoring.**

**Key Principles**:
1. **STRIDE every component** - systematic beats intuition
2. **Build attack trees** - find easiest exploitation paths
3. **Check enforcement gaps** - defense-in-depth at every layer
4. **Score risks** - L × I prioritization
5. **Do it early** - before implementation, when fixes are cheap

**Meta-rule**: If you're designing something security-sensitive and you haven't threat modeled it, you've missed critical threats. Always threat model first.
