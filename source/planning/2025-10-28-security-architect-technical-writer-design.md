# Security Architect & Technical Writer Skill Packs - Design

**Date**: 2025-10-28
**Status**: Design Approved
**Authors**: John (Product Owner), Claude (Design Lead)

## Executive Summary

Two complementary skill packs for Claude Code providing universal security architecture and technical writing capabilities:

- **security-architect**: 8 skills (4 core universal + 4 specialized extensions)
- **technical-writer**: 8 skills (4 core universal + 4 specialized extensions)

**Key Innovation**: Layered architecture with universal core skills applicable to any project, plus specialized extensions for high-security/regulated environments. Bidirectional cross-referencing creates knowledge graph where each pack enhances the other.

**Delivery**: Phased rollout over 8-9 weeks, 80-135 hours total effort, following TDD-for-documentation methodology (RED-GREEN-REFACTOR).

**Use Cases**:
- Primary: Personal use across multiple projects
- Secondary: Public contribution (no existing published skills in these domains)

---

## Table of Contents

1. [Design Overview](#design-overview)
2. [Security-Architect Pack](#security-architect-pack)
3. [Technical-Writer Pack](#technical-writer-pack)
4. [Cross-Referencing Strategy](#cross-referencing-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Implementation Phases](#implementation-phases)
7. [Pick Up Plan](#pick-up-plan)

---

## Design Overview

### Architectural Principles

1. **Layered Design**: Core skills (universal) + Extensions (specialized)
2. **Meta-Skills for Discovery**: Each pack has routing skill (like `using-superpowers`)
3. **Bidirectional Cross-References**: Skills reference each other without hard dependencies
4. **TDD Methodology**: Every skill tested with RED-GREEN-REFACTOR cycle before deployment
5. **Modular Loading**: Load only what you need, when you need it

### Directory Structure

```
~/.claude/skills/
├── security-architect/
│   ├── using-security-architect/              # Meta-skill (routing)
│   ├── threat-modeling/                       # CORE
│   ├── security-controls-design/              # CORE
│   ├── architecture-security-review/          # CORE
│   ├── secure-by-design-patterns/             # CORE
│   ├── classified-systems-security/           # EXTENSION
│   ├── compliance-awareness-and-mapping/      # EXTENSION
│   ├── security-authorization-and-accreditation/  # EXTENSION
│   └── documenting-threats-and-controls/      # CROSS-CUTTING
│
└── technical-writer/
    ├── using-technical-writer/                # Meta-skill (routing)
    ├── documentation-structure/               # CORE
    ├── clarity-and-style/                     # CORE
    ├── diagram-conventions/                   # CORE
    ├── documentation-testing/                 # CORE
    ├── security-aware-documentation/          # EXTENSION
    ├── incident-response-documentation/       # EXTENSION
    ├── itil-and-governance-documentation/     # EXTENSION
    └── operational-acceptance-documentation/  # EXTENSION
```

**Total**: 16 skills across 2 packs (2 meta + 8 core + 8 extensions, with 2 cross-cutting)

---

## Security-Architect Pack

### Pack Purpose

Provide security architects with proven workflows for threat analysis, control design, architecture review, and secure system design. Applicable to any software project with extensions for high-security/regulated environments.

### Skills Catalog

#### Meta-Skill: `using-security-architect`

**Purpose**: Discovery and routing skill
**Content**:
- Symptom mapping (e.g., "Designing new authentication?" → load `architecture-security-review` + `secure-by-design-patterns`)
- Mandatory workflows (always threat model before implementation)
- Quick reference of when to use which skill

**YAML Frontmatter**:
```yaml
---
name: using-security-architect
description: Use when starting security architecture work - routes to specific security skills based on task type (threat modeling, control design, review, secure patterns, classified systems, compliance, authorization)
---
```

---

#### CORE SKILL 1: `threat-modeling`

**When to use**: Analyzing attack surfaces, designing new systems, security reviews

**Techniques**:
- **STRIDE methodology**: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege
- **Attack tree construction**: Root goal → branches of attack vectors → leaf nodes of specific exploits
- **Threat enumeration**: Systematic identification of threats per component/interface
- **Risk scoring**: Likelihood × Impact matrix, prioritization
- **Mitigation mapping**: Threats → Controls relationship

**Deliverable**: Threat model document with prioritized threats and mitigations

**Example Scenarios**:
- "Use when adding OAuth integration to identify authentication threats (credential theft, token replay, authorization bypass)"
- "Use when exposing new API endpoint to enumerate data exposure risks"
- "Use when reviewing third-party integration to assess supply chain threats"

**YAML Frontmatter**:
```yaml
---
name: threat-modeling
description: Use when analyzing attack surfaces, designing new systems, or conducting security reviews - applies STRIDE methodology, attack trees, and risk scoring to identify and prioritize threats systematically
---
```

---

#### CORE SKILL 2: `security-controls-design`

**When to use**: Implementing protective measures, hardening systems, defense planning

**Techniques**:
- **Defense-in-depth**: Layered security (network → host → application → data)
- **Least privilege principle**: Minimum necessary access, deny by default
- **Fail-secure patterns**: System failures result in secure state (closed doors, denied access)
- **Separation of duties**: No single person/component has complete control
- **Trust boundaries**: Explicit trust zones with validation at boundaries
- **Security by design**: Built-in, not bolted-on

**Deliverable**: Control architecture with multiple defensive layers

**Example Scenarios**:
- "Use when designing API authentication to layer controls (rate limiting, token validation, audit logging, anomaly detection)"
- "Use when hardening database access to implement least privilege (role-based access, column-level security, audit triggers)"
- "Use when securing file uploads to layer defenses (type validation, size limits, antivirus scanning, sandboxed storage)"

**YAML Frontmatter**:
```yaml
---
name: security-controls-design
description: Use when implementing protective measures or hardening systems - applies defense-in-depth, least privilege, fail-secure patterns, and separation of duties to create layered security architectures
---
```

---

#### CORE SKILL 3: `architecture-security-review`

**When to use**: Reviewing existing designs, pre-implementation validation, security audits

**Review Checklists**:

**Authentication**:
- Credential storage (hashing algorithm, salting, key derivation)
- Multi-factor authentication support
- Session management (token expiry, rotation, invalidation)
- Password policies (complexity, history, lockout)

**Authorization**:
- Access control model (RBAC, ABAC, MAC)
- Privilege escalation prevention
- Resource-level authorization checks
- Default-deny principle

**Secrets Management**:
- No secrets in code/config/logs
- Secrets rotation policies
- Access control to secrets stores
- Encryption at rest for secrets

**Data Flow**:
- Trust boundary identification
- Input validation at boundaries
- Output encoding for context
- Data classification and handling

**Network Security**:
- TLS/encryption in transit
- Certificate validation
- Network segmentation
- Firewall rules and least-necessary access

**Deliverable**: Security review report with findings categorized by severity (Critical/High/Medium/Low)

**Example Scenarios**:
- "Use when reviewing microservices design to validate inter-service authentication (mutual TLS, service accounts, token validation)"
- "Use when auditing web application to check input validation coverage across all endpoints"
- "Use when reviewing data pipeline to verify encryption at rest and in transit"

**YAML Frontmatter**:
```yaml
---
name: architecture-security-review
description: Use when reviewing existing designs or conducting security audits - provides systematic checklists for authentication, authorization, secrets management, data flow, and network security to identify vulnerabilities before implementation
---
```

---

#### CORE SKILL 4: `secure-by-design-patterns`

**When to use**: Greenfield system design, architecture refactoring, secure defaults

**Patterns**:

**Zero-Trust Architecture**:
- Never trust, always verify
- Verify explicitly (authenticate + authorize every request)
- Least privilege access
- Assume breach (microsegmentation, monitoring)

**Immutable Infrastructure**:
- No runtime modifications
- Deployments replace rather than update
- Audit trail of all changes
- Rollback by redeployment

**Security Boundaries**:
- Explicit trust zones
- Validation at every boundary crossing
- Minimize boundary surface area
- Monitor boundary traffic

**Trusted Computing Base (TCB) Minimization**:
- Small security-critical core
- Everything else is untrusted
- Formal verification for TCB
- Isolation via sandboxing

**Fail-Fast Security**:
- Validate security properties at construction time
- Refuse to operate if misconfigured
- No exposure window for invalid configurations

**Deliverable**: Architecture with security built-in from foundation

**Example Scenarios**:
- "Use when designing data pipeline to enforce security levels at construction time (not runtime checks) - fail-fast if datasource/sink security mismatch"
- "Use when architecting microservices to implement zero-trust (no implicit trust, authenticate every service call, monitor all traffic)"
- "Use when designing secrets management to minimize TCB (small secrets vault, everything else accesses via API, formal verification of vault)"

**YAML Frontmatter**:
```yaml
---
name: secure-by-design-patterns
description: Use when designing greenfield systems or refactoring architecture - applies zero-trust, immutable infrastructure, security boundaries, TCB minimization, and fail-fast security to build security into system foundations
---
```

---

#### EXTENSION SKILL 5: `classified-systems-security`

**When to use**: Handling classified data, multi-level security (MLS), government/defense projects

**Techniques**:

**Bell-LaPadula MLS Model**:
- **No read up**: Subject cannot read data at higher classification
- **No write down**: Subject cannot write data to lower classification
- **Security levels**: UNOFFICIAL → OFFICIAL → OFFICIAL:SENSITIVE → PROTECTED → SECRET → TOP SECRET
- **Clearance validation**: Components declare clearance, system enforces

**Fail-Fast Security Enforcement**:
- Pipeline-wide minimum security level computation
- Construction-time validation (before data retrieval)
- Zero exposure window for misconfigured systems
- Explicit rejection of insufficient clearance

**Trusted Downgrade**:
- High-clearance components operating at lower levels
- Data filtering/sanitization to lower classification
- Explicit downgrade approval and audit
- Immutable classification after assignment

**Immutability Enforcement**:
- Data classification cannot be reduced once set
- Frozen dataclasses prevent runtime modification
- Compile-time and runtime checks

**Example Scenarios**:
- "Use when implementing system processing SECRET and UNOFFICIAL data to enforce no-read-up (UNOFFICIAL components cannot access SECRET) and no-write-down (SECRET data cannot flow to UNOFFICIAL sinks)"
- "Use when designing multi-level pipeline to compute minimum security level across all components and fail-fast if clearance mismatch"
- "Use when handling classified documents to enforce immutable classification (once marked SECRET, cannot be downgraded without formal approval)"

**YAML Frontmatter**:
```yaml
---
name: classified-systems-security
description: Use when handling classified data or implementing multi-level security - applies Bell-LaPadula model, fail-fast enforcement, trusted downgrade patterns, and immutability to prevent unauthorized information flow in government/defense systems
---
```

---

#### EXTENSION SKILL 6: `compliance-awareness-and-mapping`

**When to use**: Any regulated environment, audit preparation, control implementation

**Core Principle**: Different jurisdictions and industries have different frameworks - ALWAYS verify which applies

**Discovery Process**:
1. **Identify applicable frameworks**: Ask "What compliance frameworks/standards apply to this system?"
2. **Understand framework structure**: Control categories, evidence requirements, assessment procedures
3. **Map implemented controls**: Technical controls → Framework requirements
4. **Document traceability**: Control → Requirement → Evidence chains
5. **Gap analysis**: Identify missing controls before audit

**Framework Examples** (not exhaustive):

**By Jurisdiction**:
- **Australia**: ISM (Information Security Manual), IRAP (Infosec Registered Assessors Program), PSPF (Protective Security Policy Framework)
- **United Kingdom**: Cyber Essentials, NCSC Guidance, Official Secrets Act
- **United States**: NIST Cybersecurity Framework, FedRAMP, FISMA
- **European Union**: NIS2 Directive, GDPR (data protection), ISO 27001

**By Industry**:
- **Healthcare**: HIPAA (US), GDPR (EU), Australian Privacy Principles
- **Finance**: PCI-DSS (payment cards), SOX (US), Basel III
- **General**: SOC2 (service organizations), ISO 27001 (information security)

**Universal Control Categories** (common across frameworks):
- Access control (authentication, authorization, least privilege)
- Encryption (data at rest, data in transit)
- Audit logging (who/what/when/where)
- Incident response (detection, containment, recovery)
- Vulnerability management (patching, scanning)
- Configuration management (secure baselines, change control)
- Personnel security (background checks, training)
- Physical security (facility access, environmental controls)

**Deliverable**: Framework-specific compliance mapping with evidence collection plan

**Example Scenarios**:
- "Use when starting regulated healthcare project - first ask 'HIPAA or GDPR or both?' then map technical controls (encryption, access logs, breach notification) to specific requirements"
- "Use when preparing for SOC2 audit to map monitoring controls (log aggregation, alerting, incident response) to Trust Services Criteria"
- "Use when supporting IRAP assessment to create traceability matrix mapping ISM controls to implemented technical measures with evidence artifacts"

**YAML Frontmatter**:
```yaml
---
name: compliance-awareness-and-mapping
description: Use when working in regulated environments or preparing for audits - teaches framework discovery process (ISM/IRAP, HIPAA, PCI-DSS, SOC2, FedRAMP vary by jurisdiction/industry), control mapping patterns, and traceability documentation across diverse compliance landscapes
---
```

---

#### EXTENSION SKILL 7: `security-authorization-and-accreditation`

**When to use**: Preparing systems for production authorization, government/defense deployments, formal security testing

**Processes Covered**:

**ATO (Authority to Operate)**:
- Security authorization package preparation
- Risk acceptance framework
- Authorizing Official (AO) approval process
- Interim ATO vs Full ATO
- Authorization boundary definition

**AIS (Authorization to Interconnect)**:
- System interconnection agreements
- Boundary protection documentation
- Trust relationship establishment
- Data sharing agreements
- Interconnection security requirements

**T&E (Test & Evaluation)**:
- Security test plan development
- Penetration test coordination
- Vulnerability assessment execution
- Security control testing
- Test results documentation and remediation

**System Security Plan (SSP)**:
- System characterization
- Security control implementation descriptions
- Responsible parties identification
- Assessment procedures definition

**Security Assessment Report (SAR)**:
- Control testing evidence
- Finding documentation
- Risk severity classification
- Remediation recommendations

**Plan of Action & Milestones (POA&M)**:
- Risk remediation tracking
- Milestone definitions
- Resource requirements
- Waiver justifications for accepted risks

**Continuous Monitoring**:
- Post-authorization monitoring plans
- Change impact analysis
- Ongoing assessment procedures
- Re-authorization triggers

**Deliverable**: Complete security authorization package ready for authorizing official review

**Example Scenarios**:
- "Use when preparing government system deployment to create ATO package (SSP with system boundary, control implementations, SAR with test results, POA&M for residual risks)"
- "Use when connecting to classified network to document AIS requirements (interconnection agreement, boundary protection, data flow documentation, trust relationships)"
- "Use when coordinating penetration test to develop T&E plan (test objectives, rules of engagement, success criteria, reporting procedures)"

**YAML Frontmatter**:
```yaml
---
name: security-authorization-and-accreditation
description: Use when preparing systems for production authorization in government/defense environments - covers ATO (Authority to Operate), AIS (Authorization to Interconnect), T&E (Test & Evaluation), SSP, SAR, POA&M, and continuous monitoring processes
---
```

---

#### CROSS-CUTTING SKILL 8: `documenting-threats-and-controls`

**When to use**: Writing threat models, security ADRs, incident response procedures, control documentation

**Documentation Patterns**:

**Threat Documentation**:
- Threat description (what/how/who)
- Affected assets
- Attack scenarios
- Likelihood and impact assessment
- References to mitigations

**Security ADRs** (Architecture Decision Records):
```markdown
# ADR-XXX: [Security Decision Title]

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
- What security problem are we solving?
- What are the constraints?
- What's the threat model?

## Decision
- What security approach did we choose?
- What are the security properties?

## Consequences
- Security benefits
- Security trade-offs
- Residual risks
- Ongoing security requirements

## Security Controls
- Control IDs (if compliance-mapped)
- Implementation details
```

**Control Documentation**:
- Control objective (what it protects)
- Implementation description (how it works)
- Responsible parties
- Assessment procedures (how to verify)
- Evidence artifacts

**Security Requirements**:
- Requirement ID and title
- Security property enforced
- Acceptance criteria (testable)
- Traceability to threats/compliance

**Deliverable**: Clear, actionable security documentation

**Example Scenarios**:
- "Use when documenting decision to use JWT tokens (ADR with threat model for session management, chosen mitigation, residual risks like token theft)"
- "Use when writing access control implementation to document control objective (prevent unauthorized access), implementation (RBAC with policy engine), assessment (test coverage)"
- "Use when creating security requirements to trace from threat (credential theft) to requirement (MFA mandatory) to control (TOTP implementation) to test (MFA bypass attempts fail)"

**References**:
- **RECOMMENDED SUB-SKILL**: Use technical-writer:documentation-structure for ADR format patterns
- **RECOMMENDED SUB-SKILL**: Use technical-writer:clarity-and-style for clear security communication

**YAML Frontmatter**:
```yaml
---
name: documenting-threats-and-controls
description: Use when writing threat models, security ADRs, or control documentation - provides patterns for threat descriptions, security decision records, control documentation, and requirements traceability with cross-references to technical-writer skills
---
```

---

## Technical-Writer Pack

### Pack Purpose

Provide technical writers and engineers with proven patterns for creating clear, accurate, findable documentation. Applicable to any technical project with extensions for security-aware, incident response, governance, and operational acceptance documentation.

### Skills Catalog

#### Meta-Skill: `using-technical-writer`

**Purpose**: Discovery and routing skill
**Content**:
- Documentation scenario mapping (e.g., "Documenting system decision?" → load `documentation-structure` ADR pattern)
- Quick reference of documentation types and appropriate skills
- Workflow guidance (structure → write → diagram → test)

**YAML Frontmatter**:
```yaml
---
name: using-technical-writer
description: Use when starting documentation work - routes to specific technical writing skills based on task type (structure, clarity, diagrams, testing, security-aware, incident response, governance, operational acceptance)
---
```

---

#### CORE SKILL 1: `documentation-structure`

**When to use**: Creating new documentation, reorganizing existing docs, choosing documentation patterns

**Patterns Covered**:

**ADR (Architecture Decision Record)**:
```markdown
# Title: [Decision]

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
- What problem are we solving?
- What are the constraints?
- What alternatives did we consider?

## Decision
- What did we choose?
- Why this approach?

## Consequences
- Positive outcomes
- Negative trade-offs
- Ongoing requirements
```

**Architecture Documentation**:
- System overview (purpose, stakeholders, quality attributes)
- Component diagrams (major components and relationships)
- Data flows (how information moves through system)
- Deployment architecture (infrastructure, scaling, DR)
- Security architecture (trust boundaries, authentication flows)

**API Reference**:
- Endpoint patterns (RESTful conventions, versioning)
- Request/response examples (complete, runnable)
- Error codes and handling
- Authentication and authorization
- Rate limiting and pagination

**Runbooks**:
- Purpose and scope
- Prerequisites and access requirements
- Step-by-step procedures
- Troubleshooting guide
- Escalation paths

**README Patterns**:
- Quick start (minimum steps to run)
- Prerequisites (versions, dependencies)
- Configuration (environment variables, config files)
- Examples (common use cases)
- Troubleshooting (common issues)
- Contributing guidelines

**Deliverable**: Correctly structured documentation using appropriate pattern

**Example Scenarios**:
- "Use when documenting 'Why did we choose PostgreSQL over MongoDB?' to create ADR with context (data access patterns), alternatives (document vs relational), decision (PostgreSQL), consequences (schema migrations required)"
- "Use when creating API docs to structure with authentication section, endpoint reference with examples, error code reference, rate limiting policy"
- "Use when writing deployment runbook to include prerequisites (access, tools), step-by-step deployment, rollback procedure, troubleshooting guide"

**YAML Frontmatter**:
```yaml
---
name: documentation-structure
description: Use when creating new documentation or choosing documentation patterns - provides structures for ADRs, architecture docs, API references, runbooks, and READMEs to ensure consistent, complete documentation
---
```

---

#### CORE SKILL 2: `clarity-and-style`

**When to use**: Writing or reviewing any technical content for readability

**Techniques**:

**Active Voice**:
- ✅ "The system validates tokens"
- ❌ "Tokens are validated by the system"
- ✅ "Run the tests with pytest"
- ❌ "Tests should be run with pytest"

**Concrete Examples**:
- Every abstract concept needs runnable example
- ❌ "Configure the system appropriately"
- ✅ "Set `API_TIMEOUT=30` in `.env` for 30-second timeout"

**Progressive Disclosure**:
- Essential information first
- Details on-demand (expandable sections, links to deep dives)
- Don't overwhelm with everything upfront

**Audience Adaptation**:
- **Developer**: How it works (architecture, APIs, data flows)
- **Operator**: How to run it (deployment, configuration, troubleshooting)
- **Executive**: Why it matters (business value, risks, costs)

**Precision Without Jargon**:
- Technical accuracy using accessible language
- Define acronyms on first use
- Avoid unnecessarily complex terminology
- ❌ "Utilize the ingress controller to facilitate external traffic ingress"
- ✅ "Use the ingress controller to route external traffic into the cluster"

**Scannable Structure**:
- Headings, subheadings, bullet points
- Bold for emphasis, not paragraphs of bold
- Code blocks for commands/examples
- Tables for comparing options

**Deliverable**: Clear, scannable, actionable documentation

**Example Scenarios**:
- "Use when reviewing docs that say 'Configure the database properly' to add concrete example: `DATABASE_URL=postgresql://user:pass@host:5432/dbname`"
- "Use when writing for developers to include architecture diagram and API examples; when writing for operators to include deployment steps and troubleshooting"
- "Use when explaining complex concept (like OAuth flow) to start with one-sentence summary, then simple diagram, then detailed steps with examples"

**YAML Frontmatter**:
```yaml
---
name: clarity-and-style
description: Use when writing or reviewing technical content - applies active voice, concrete examples, progressive disclosure, audience adaptation, and scannable structure to create clear, actionable documentation
---
```

---

#### CORE SKILL 3: `diagram-conventions`

**When to use**: Creating visual documentation, choosing diagram types

**Decision Tree**:

```
What are you documenting?
│
├─ Sequence of interactions over time?
│  └─ Use SEQUENCE DIAGRAM (API calls, message flows)
│
├─ System components and relationships?
│  └─ Use COMPONENT/ARCHITECTURE DIAGRAM (services, databases, dependencies)
│
├─ Decision logic with branches?
│  └─ Use FLOWCHART (if small and necessary)
│  └─ WARNING: Flowcharts often anti-pattern for complex logic (use pseudo-code instead)
│
├─ Data movement through system?
│  └─ Use DATA FLOW DIAGRAM (inputs, transformations, outputs)
│
└─ State changes over lifecycle?
   └─ Use STATE DIAGRAM (order states, connection states)
```

**Labeling Standards**:
- **Semantic names**: "AuthService" not "Service1", "ValidateToken" not "Step3"
- **Consistent terminology**: Same terms as code/documentation
- **Meaningful relationships**: "authenticates", "queries", "publishes to" not just arrows
- **Legend when needed**: Especially for custom notation or symbols

**When Flowcharts Become Anti-Patterns**:
- Complex business logic (use pseudo-code or decision tables)
- Long procedures (use numbered lists)
- Code already exists (link to code, don't duplicate in flowchart)

**Deliverable**: Appropriate diagram type with clear, semantic labels

**Example Scenarios**:
- "Use when documenting authentication flow to choose SEQUENCE DIAGRAM showing: User → Frontend → Auth Service → Database → Frontend → User (shows API calls and responses over time)"
- "Use when documenting microservices architecture to choose COMPONENT DIAGRAM showing services, databases, message queues with labeled relationships (queries, publishes to, consumes from)"
- "Use when documenting simple 3-branch decision to use FLOWCHART; when documenting complex 15-branch decision to use DECISION TABLE instead"

**References**:
- See graphviz-conventions.dot for detailed graphviz style rules (if available in skill pack)

**YAML Frontmatter**:
```yaml
---
name: diagram-conventions
description: Use when creating visual documentation - provides decision tree for choosing diagram types (sequence, component, flowchart, data flow, state), labeling standards with semantic names, and guidance on when flowcharts become anti-patterns
---
```

---

#### CORE SKILL 4: `documentation-testing`

**When to use**: Before releasing documentation, during doc reviews, quality gates

**Testing Dimensions**:

**Completeness Testing**:
- Can reader accomplish the task with ONLY this doc?
- Are all prerequisites listed?
- Are all configuration options documented?
- Are error cases covered?
- Is troubleshooting included?

**Accuracy Testing**:
- Do code examples actually run?
- Are commands correct (copy-paste-run test)?
- Are version numbers current?
- Are screenshots up-to-date?
- Do links work?

**Findability Testing**:
- Can users find this via search (keywords present)?
- Is it linked from related pages?
- Is it in navigation/TOC?
- Are cross-references complete?

**Example Verification**:
- Copy every code example
- Paste into clean environment
- Run without modifications
- Verify output matches documented expectations

**Walkthrough Testing**:
- Follow quick start as new user
- Note every point of confusion
- Verify success criteria ("How do I know it worked?")
- Test on clean system (no cached knowledge)

**Deliverable**: Production-ready documentation with verified examples

**Example Scenarios**:
- "Use when finalizing installation guide to: (1) spin up clean VM, (2) copy-paste every command, (3) verify all commands succeed, (4) verify application runs as documented"
- "Use when reviewing API docs to: (1) copy every curl example, (2) run against test environment, (3) verify responses match documentation, (4) test error cases"
- "Use when preparing quick start to have colleague (unfamiliar with project) follow guide and note every point where they get stuck or confused"

**YAML Frontmatter**:
```yaml
---
name: documentation-testing
description: Use when preparing documentation for release - provides testing framework for completeness (can task be accomplished?), accuracy (do examples run?), findability (can users locate?), and walkthrough validation (does quick start work for new users?)
---
```

---

#### EXTENSION SKILL 5: `security-aware-documentation`

**When to use**: Documenting systems handling sensitive data, security features, creating examples with potential security implications

**Techniques**:

**Sanitizing Examples**:
- ❌ Never use real credentials, API keys, tokens (even masked)
- ✅ Generate fake but realistic examples: `api_key_abc123def456` (obviously fake)
- ❌ Don't mask real secrets: `sk_live_***REDACTED***` (implies production secret)
- ✅ Create complete fake examples: `sk_test_fake_key_for_documentation_only`
- ❌ Never use real PII: john.smith@company.com, 123-45-6789
- ✅ Generate fake but valid-looking: jane.doe@example.com, 000-00-0000

**Threat Disclosure Decisions**:
- **Document**: Security features users need to configure (authentication, authorization, encryption settings)
- **Document**: Security best practices and hardening guides
- **Don't document**: Specific vulnerabilities or exploits (coordinate with security team for disclosure)
- **Don't document**: Internal security architecture details that aid attackers (unless necessary for legitimate use)

**Compliance Sensitivity**:
- Document control implementations without revealing weaknesses
- Focus on "what controls exist" not "gaps in controls"
- Coordinate with compliance team for audit-related documentation

**Redaction Patterns**:
- Logs: Redact tokens, passwords, session IDs, PII
- Screenshots: Blur sensitive data, use test accounts
- Diagrams: Anonymize hostnames, IP addresses (use RFC 1918 private ranges)
- Database schemas: Use synthetic data in examples

**Security Feature Documentation**:
- Threat model context ("This feature prevents X attack")
- Configuration requirements ("Must enable MFA for admin accounts")
- Security implications of misconfiguration ("Disabling this exposes Y risk")

**Deliverable**: Documentation that informs without compromising security

**Example Scenarios**:
- "Use when creating API authentication examples to generate complete fake tokens (not mask real ones): `Authorization: Bearer fake_token_abc123_for_docs_only`"
- "Use when documenting OAuth flow to use example.com domains and fake client IDs: `client_id=docs_example_12345`"
- "Use when writing database query examples to use synthetic PII: INSERT INTO users VALUES ('Jane Doe', 'jane@example.com', '000-00-0000')"
- "Use when documenting security feature (rate limiting) to explain threat (brute force attacks) and configuration (max 100 requests/minute per IP)"

**References**:
- **RECOMMENDED SUB-SKILL**: Use security-architect:threat-modeling to understand what threats docs might expose

**YAML Frontmatter**:
```yaml
---
name: security-aware-documentation
description: Use when documenting systems with sensitive data or security features - covers sanitizing examples (never use real credentials/PII), threat disclosure decisions, compliance sensitivity, redaction patterns, and security feature documentation
---
```

---

#### EXTENSION SKILL 6: `incident-response-documentation`

**When to use**: Creating runbooks for security incidents, outages, emergency procedures

**Response Template Structure**:

**1. Detection**:
- Symptoms and alerts
- Monitoring queries
- Severity classification
- Initial triage questions

**2. Containment**:
- Immediate actions (stop the bleeding)
- Isolation procedures
- Communication holds (don't tip off attacker)

**3. Investigation**:
- Log collection commands
- Forensic procedures
- Timeline reconstruction
- Impact assessment

**4. Recovery**:
- Restoration procedures
- Verification steps
- Monitoring for recurrence

**5. Lessons Learned**:
- Post-incident report template
- Blameless retrospective framework
- Action items tracking

**Escalation Paths**:
- Severity-based escalation (P1/P2/P3/P4)
- Contact chains with backup contacts
- Authority delegation (who can make what decisions)
- External notifications (customers, regulators, media)

**Time-Critical Clarity**:
- No ambiguity under stress
- Decision trees for triage: "If X, do Y; else if Z, do W"
- Bold for critical steps
- Numbered steps (not paragraphs)
- Success criteria for each step

**Post-Incident Reports**:
- Timeline (detection → containment → recovery)
- Root cause analysis (without blame)
- Impact summary (affected users, data, services)
- Action items with owners and due dates
- What went well / what could improve

**Deliverable**: Actionable runbook usable during high-stress incidents

**Example Scenarios**:
- "Use when documenting 'What to do if PII exposure detected' to structure: (1) Containment (disable affected endpoint, block IP), (2) Investigation (query logs for scope), (3) Recovery (rotate credentials, notify affected users), (4) Lessons (how did it happen, how to prevent)"
- "Use when creating DDoS response runbook to include: (1) Detection (traffic spike alerts), (2) Containment (enable rate limiting, block malicious IPs), (3) Escalation (when to contact ISP, when to enable DDoS protection service)"
- "Use when writing database outage procedure to use decision tree: If primary down → Promote replica; If replica down → Continue on primary; If both down → Restore from backup"

**References**:
- **RECOMMENDED SUB-SKILL**: Use security-architect:security-controls-design for understanding control failure scenarios

**YAML Frontmatter**:
```yaml
---
name: incident-response-documentation
description: Use when creating runbooks for security incidents or operational emergencies - provides response template (detection, containment, investigation, recovery, lessons), escalation paths, time-critical clarity patterns, and post-incident report structure
---
```

---

#### EXTENSION SKILL 7: `itil-and-governance-documentation`

**When to use**: Formal IT service management environments, enterprise operations, change control processes

**Patterns Covered**:

**Change Requests (RFC - Request for Change)**:
```markdown
# RFC-XXXX: [Change Title]

## Change Type
Standard | Normal | Emergency

## Description
What is changing and why?

## Impact Analysis
- Affected services
- Affected users/teams
- Dependencies
- Risk assessment

## Implementation Plan
Step-by-step with timing

## Rollback Plan
How to undo if fails

## Testing Plan
Pre-production validation

## Approval Chain
Required approvals and status
```

**Service Documentation**:
- Service catalog entries (service name, description, owner, SLA)
- Service definitions (purpose, scope, users)
- SLA documentation (availability, performance, support hours)
- OLA documentation (operational level agreements between internal teams)

**Configuration Management**:
- CMDB (Configuration Management Database) documentation
- Configuration baselines
- Dependency mapping (service → components → infrastructure)
- Change impact analysis

**Release Documentation**:
- Release notes (features, bug fixes, breaking changes)
- Deployment procedures (step-by-step with verification)
- Communication plans (who to notify, when, what to say)
- Rollback procedures

**Operational Handover**:
- Service transition documentation
- Knowledge transfer checklists
- Runbook handoff
- Support model definition

**Business Continuity**:
- DR (Disaster Recovery) plans
- RTO (Recovery Time Objective) documentation
- RPO (Recovery Point Objective) documentation
- Failover procedures with testing schedule

**Capacity Planning**:
- Resource forecasts (CPU, memory, storage, network)
- Scaling thresholds (when to add capacity)
- Performance baselines (normal operating ranges)
- Growth projections

**Problem Management**:
- Known error database (KEDb)
- Workarounds documentation
- Root cause documentation
- Permanent fix tracking

**Deliverable**: Governance-compliant documentation following ITIL/organizational frameworks

**Example Scenarios**:
- "Use when preparing production deployment to document change request with: impact analysis (downtime window, affected users), implementation plan (step-by-step deployment), rollback plan (how to revert), approval chain (dev lead, ops lead, change advisory board)"
- "Use when documenting new service to create service catalog entry: service name, description, owner team, SLA (99.9% uptime, <200ms p95 latency), support hours (24/7 for P1, business hours for P2+)"
- "Use when creating DR plan to document: RTO (4 hours to restore service), RPO (1 hour of data loss acceptable), failover procedure (promote standby region), testing schedule (quarterly DR drills)"

**YAML Frontmatter**:
```yaml
---
name: itil-and-governance-documentation
description: Use when working in formal IT service management environments - covers change requests (RFC), service documentation, configuration management (CMDB), release docs, operational handover, business continuity (DR/RTO/RPO), capacity planning, and problem management
---
```

---

#### EXTENSION SKILL 8: `operational-acceptance-documentation`

**When to use**: Preparing systems for production deployment, formal acceptance gates, operational handover

**Document Patterns Covered**:

**Security Authorization Documentation**:
- **SSP (System Security Plan)**: System characterization, control descriptions, security architecture
- **SAR (Security Assessment Report)**: Test results, findings, risk severity, remediation status
- **POA&M (Plan of Action & Milestones)**: Risk tracking, remediation plans, waiver justifications
- **System boundary definitions**: Network diagrams, data flows, trust boundaries
- **AIS (Authorization to Interconnect)**: Interconnection agreements, boundary protection

**Operational Readiness Documentation**:
- **Production readiness checklist**: Infrastructure, monitoring, logging, alerting, backups complete
- **Capacity validation**: Load testing results, performance baselines, scaling plans
- **Monitoring coverage**: All critical metrics instrumented, alerting configured
- **Backup/recovery validation**: Backup procedures tested, recovery time verified

**Test & Evaluation Documentation**:
- **T&E reports**: Test objectives, methodology, results, defect summary
- **Test completion criteria**: All tests passed or defects dispositioned
- **Defect disposition**: Critical defects fixed, high defects have workarounds or risk acceptance
- **Acceptance test results**: Functional, performance, security tests passed

**Go-Live Approval Documentation**:
- **Executive summary**: System purpose, business value, readiness status
- **Risk acceptance documentation**: Residual risks identified, accepted by stakeholders
- **Launch criteria**: Success metrics, abort criteria, monitoring plan
- **Rollback plan**: How to revert if launch fails

**Transition Planning**:
- **Operational handover**: Knowledge transfer completed, runbooks delivered
- **Support model definition**: On-call rotation, escalation paths, SLA commitments
- **Training completion**: Operators trained, documentation reviewed

**Acceptance Criteria Documentation**:
- **Success metrics**: How to measure successful launch (error rates, latency, user adoption)
- **Service level commitments**: SLA/SLO definitions
- **Monitoring baselines**: Normal operating ranges, alert thresholds

**Residual Risk Documentation**:
- **Known limitations**: What system doesn't do (out of scope)
- **Operational constraints**: Resource limits, scaling boundaries, deprecated features
- **Risk mitigation plans**: How ongoing operations reduce residual risks

**Deliverable**: Complete acceptance package enabling informed go-live decisions

**Example Scenarios**:
- "Use when preparing production launch to document: (1) Security authorization status (ATO approved, POA&M with 2 low-risk items), (2) Operational readiness (monitoring/alerting/backups complete), (3) Test completion (all acceptance tests passed), (4) Residual risks (known performance limit at 1000 concurrent users)"
- "Use when seeking executive go-live approval to create summary: system purpose (customer portal), business value ($2M revenue enabler), readiness (all criteria met), residual risks (graceful degradation above 1000 users), launch criteria (error rate <0.1%, p95 latency <500ms)"
- "Use when completing operational handover to document: runbooks delivered, on-call rotation staffed, operators trained, support SLA defined (P1 15min, P2 4hr, P3 1 day)"

**References**:
- **RECOMMENDED SUB-SKILL**: Use security-architect:security-authorization-and-accreditation for understanding authorization process requirements

**YAML Frontmatter**:
```yaml
---
name: operational-acceptance-documentation
description: Use when preparing systems for production deployment or operational handover - covers security authorization (SSP/SAR/POA&M/ATO), operational readiness, test & evaluation, go-live approval, transition planning, acceptance criteria, and residual risk documentation for multi-stakeholder acceptance gates
---
```

---

## Cross-Referencing Strategy

### Bidirectional Knowledge Graph

**From security-architect → technical-writer:**

| Security Skill | References | Technical Writing Skill |
|---------------|------------|------------------------|
| `security-authorization-and-accreditation` | RECOMMENDED | `operational-acceptance-documentation` (how to write the package) |
| `documenting-threats-and-controls` | RECOMMENDED | `documentation-structure` (ADR patterns) |
| `documenting-threats-and-controls` | RECOMMENDED | `clarity-and-style` (clear security communication) |
| `compliance-awareness-and-mapping` | RECOMMENDED | `clarity-and-style` (writing control descriptions) |
| `threat-modeling` | OPTIONAL | `diagram-conventions` (data flow diagrams for threat models) |

**From technical-writer → security-architect:**

| Technical Writing Skill | References | Security Skill |
|------------------------|------------|----------------|
| `operational-acceptance-documentation` | RECOMMENDED | `security-authorization-and-accreditation` (what ATO process requires) |
| `security-aware-documentation` | RECOMMENDED | `threat-modeling` (understanding threats docs might expose) |
| `incident-response-documentation` | RECOMMENDED | `security-controls-design` (control failure scenarios) |
| `documentation-testing` | OPTIONAL | `architecture-security-review` (security testing of documented procedures) |

### Cross-Reference Syntax

**In skill SKILL.md files:**

```markdown
**RECOMMENDED SUB-SKILL:** Use security-architect:threat-modeling to understand
what threats your documentation might expose.

**OPTIONAL SUB-SKILL:** Consider technical-writer:diagram-conventions for data
flow diagram standards if creating visual threat models.
```

**Benefits**:
- No hard dependencies (skills work standalone)
- Clear enhancement paths (skills reference each other)
- Bidirectional knowledge graph (security ↔ technical writing)
- Gradual adoption (load related skills as needed)

---

## Testing Strategy

### TDD for Documentation: RED-GREEN-REFACTOR

Every skill must be tested BEFORE deployment using the methodology from `superpowers:writing-skills`.

**Iron Law**: NO SKILL WITHOUT FAILING TEST FIRST

### Testing by Skill Type

#### Discipline-Enforcing Skills

**Examples**: `architecture-security-review`, `documentation-testing`

**RED Phase** - Baseline without skill:
- Create pressure scenario: "Quick security review needed, launch tomorrow"
- Run subagent WITHOUT skill
- Document rationalizations: "It's internal only", "We can secure post-launch"
- Identify patterns in violations

**GREEN Phase** - Write skill:
- Address specific rationalizations from baseline
- Add explicit counters: "No exceptions: 'internal only' is not security"
- Run same scenario WITH skill
- Verify agent now complies

**REFACTOR Phase** - Close loopholes:
- Add combined pressures (time + sunk cost + authority)
- Identify new rationalizations
- Add to rationalization table
- Re-test until bulletproof

#### Technique Skills

**Examples**: `threat-modeling`, `documentation-structure`, `security-controls-design`

**RED Phase** - Baseline without skill:
- Create application scenario: "Model threats for new authentication system"
- Run subagent WITHOUT skill
- Document gaps: Missing STRIDE categories, incomplete attack vectors
- Identify missing knowledge

**GREEN Phase** - Write skill:
- Teach technique (STRIDE, ADR format, defense-in-depth)
- Include examples and templates
- Run same scenario WITH skill
- Verify agent applies technique correctly

**REFACTOR Phase** - Handle variations:
- Test edge cases (different system types, unusual constraints)
- Verify instructions handle variations
- Add clarifications for gaps

#### Pattern Skills

**Examples**: `secure-by-design-patterns`, `clarity-and-style`

**RED Phase** - Baseline without skill:
- Create recognition scenario: "Should we use zero-trust here?"
- Run subagent WITHOUT skill
- Document misapplications or missed opportunities
- Identify pattern recognition gaps

**GREEN Phase** - Write skill:
- Teach pattern with mental model
- Include when to apply / when not to apply
- Run scenarios WITH skill
- Verify correct pattern recognition

**REFACTOR Phase** - Counter-examples:
- Test negative cases (when pattern doesn't apply)
- Verify agent doesn't over-apply
- Add "when NOT to use" guidance

#### Reference Skills

**Examples**: `diagram-conventions`, `itil-and-governance-documentation`

**RED Phase** - Baseline without skill:
- Create retrieval scenario: "What diagram type for authentication flow?"
- Run subagent WITHOUT skill
- Document wrong choices or missing information
- Identify reference gaps

**GREEN Phase** - Write skill:
- Provide reference material (decision trees, examples)
- Organize for quick scanning
- Run scenarios WITH skill
- Verify correct retrieval and application

**REFACTOR Phase** - Coverage gaps:
- Test common use cases
- Add missing reference material
- Improve findability (keywords, structure)

### Testing Priorities

**Phase 1 (Foundation)**: Test 4 skills
- `security-architect/using-security-architect` (meta - simpler)
- `security-architect/threat-modeling` (technique - critical)
- `technical-writer/using-technical-writer` (meta - simpler)
- `technical-writer/documentation-structure` (reference - critical)

**Phase 2 (Core)**: Test 6 skills (all technique/pattern)
- All core security-architect skills
- All core technical-writer skills

**Phase 3 (Extensions)**: Test 6 skills (mix of technique/reference)
- All extension skills

**Estimated Effort**: 1-2 hours per skill × 16 skills = 16-32 hours for testing across all phases

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2) - 4 Skills

**Skills**:
1. `security-architect/using-security-architect` (meta-skill, routing)
2. `security-architect/threat-modeling` (core technique)
3. `technical-writer/using-technical-writer` (meta-skill, routing)
4. `technical-writer/documentation-structure` (core reference)

**Goals**:
- Establish pack structure
- Validate TDD methodology for skills
- Test cross-referencing syntax
- Prove phased approach works

**Effort**: 10-15 hours (2-4 hours per skill including testing)

---

### Phase 2: Core Skills (Week 3-5) - 6 Skills

**Skills**:
1. `security-architect/security-controls-design`
2. `security-architect/architecture-security-review`
3. `security-architect/secure-by-design-patterns`
4. `technical-writer/clarity-and-style`
5. `technical-writer/diagram-conventions`
6. `technical-writer/documentation-testing`

**Goals**:
- Complete universal core skills
- Maximize reusability across projects
- Establish quality baseline

**Effort**: 18-30 hours (3-5 hours per skill including testing)

---

### Phase 3: Extensions (Week 6-8) - 8 Skills

**Skills**:
1. `security-architect/classified-systems-security`
2. `security-architect/compliance-awareness-and-mapping`
3. `security-architect/security-authorization-and-accreditation`
4. `security-architect/documenting-threats-and-controls` (cross-cutting)
5. `technical-writer/security-aware-documentation`
6. `technical-writer/incident-response-documentation`
7. `technical-writer/itil-and-governance-documentation`
8. `technical-writer/operational-acceptance-documentation`

**Goals**:
- Add specialized high-security capabilities
- Complete cross-cutting skills
- Validate bidirectional cross-references

**Effort**: 24-40 hours (3-5 hours per skill including testing)

---

### Phase 4: Polish & Public Release (Week 9) - Documentation

**Deliverables**:
1. Pack-level README for each pack
2. CONTRIBUTING.md (how to contribute new skills)
3. Examples document (real-world scenarios)
4. Meta-skill refinement based on usage
5. Public repository setup (if sharing)

**Activities**:
- Cross-reference validation (all links work)
- Example verification (all examples are complete)
- Consistency review (terminology, formatting)
- Public release preparation

**Effort**: 8-15 hours

---

### Total Effort Summary

| Phase | Skills | Hours | Cumulative |
|-------|--------|-------|------------|
| Phase 1 | 4 | 10-15 | 10-15 |
| Phase 2 | 6 | 18-30 | 28-45 |
| Phase 3 | 8 | 24-40 | 52-85 |
| Phase 4 | Docs | 8-15 | 60-100 |

**Note**: Original estimate was 80-135 hours. Refined estimate after detailed design: **60-100 hours** (tighter testing estimates, clearer scope).

---

## Pick Up Plan

### Moving to Independent Repository

When ready to move `skillpacks/` to its own repository, follow this process:

#### Step 1: Repository Setup

```bash
# Create new repository directory
mkdir ~/skillpacks-repo
cd ~/skillpacks-repo

# Initialize git
git init

# Copy contents from Elspeth
cp -r ~/elspeth/skillpacks/* .

# Create initial structure
mkdir -p security-architect
mkdir -p technical-writer
mkdir -p docs
mkdir -p examples
mkdir -p tests

# Move planning docs
mv planning/2025-10-28-security-architect-technical-writer-design.md docs/DESIGN.md
```

#### Step 2: Create Repository Metadata

**README.md**:
```markdown
# Security Architect & Technical Writer Skill Packs

Universal skill packs for Claude Code providing security architecture and technical writing capabilities.

## Status

**Phase 1 In Progress**: Foundation skills (meta-skills + 2 core skills)

## Packs

- **security-architect**: Threat modeling, security controls, architecture review, secure patterns
- **technical-writer**: Documentation structure, clarity, diagrams, testing

See `docs/DESIGN.md` for complete design.

## Installation

Copy to `~/.claude/skills/`:

```bash
cp -r security-architect ~/.claude/skills/
cp -r technical-writer ~/.claude/skills/
```

## Development

See `docs/DESIGN.md` "Pick Up Plan" section for implementation guide.
```

**LICENSE** (choose one):
- MIT (permissive, recommended for wide adoption)
- Apache 2.0 (permissive with patent grant)
- CC BY 4.0 (for documentation-focused work)

**.gitignore**:
```
# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Test artifacts
tests/output/
tests/.pytest_cache/

# Personal notes
scratch/
TODO.md
```

#### Step 3: Initial Commit

```bash
git add .
git commit -m "Initial commit: Security Architect & Technical Writer skill packs design"
```

#### Step 4: GitHub Repository (if public sharing)

```bash
# Create repo on GitHub (via web UI or gh CLI)
gh repo create skillpacks --public --description "Security architecture and technical writing skill packs for Claude Code"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/skillpacks.git
git branch -M main
git push -u origin main
```

---

### Continuing Implementation

#### Phase 1 Work Plan

**Skill 1: `security-architect/using-security-architect`**

1. **RED Phase** - Baseline testing (1 hour):
   - Scenario: "I need to secure this authentication system"
   - Run subagent WITHOUT skill
   - Document: Which skills would they naturally reach for? What do they miss?

2. **GREEN Phase** - Write skill (1 hour):
   ```
   security-architect/using-security-architect/
   └── SKILL.md
   ```
   - Content: Symptom → Skill mapping
   - Decision tree for skill selection
   - Mandatory workflows (always threat model)

3. **REFACTOR Phase** - Validation (30 minutes):
   - Run multiple scenarios
   - Verify correct skill routing

**Skill 2: `security-architect/threat-modeling`**

1. **RED Phase** - Baseline testing (1 hour):
   - Scenario: "Model threats for OAuth integration"
   - Run subagent WITHOUT skill
   - Document: What STRIDE categories do they miss? How incomplete?

2. **GREEN Phase** - Write skill (2 hours):
   ```
   security-architect/threat-modeling/
   ├── SKILL.md
   └── stride-template.md (optional reference)
   ```
   - Content: STRIDE methodology, attack trees, risk scoring
   - Complete example (OAuth threat model)
   - Decision tree for when to use

3. **REFACTOR Phase** - Variation testing (1 hour):
   - Test different system types (API, web app, data pipeline)
   - Verify technique applies across domains
   - Add clarifications for gaps

**Skill 3: `technical-writer/using-technical-writer`**

1. **RED Phase** - Baseline testing (1 hour):
   - Scenario: "I need to document this security decision"
   - Run subagent WITHOUT skill
   - Document: What documentation type do they choose? Appropriate?

2. **GREEN Phase** - Write skill (1 hour):
   ```
   technical-writer/using-technical-writer/
   └── SKILL.md
   ```
   - Content: Documentation type → Skill mapping
   - Quick reference table
   - Workflow guidance (structure → write → diagram → test)

3. **REFACTOR Phase** - Validation (30 minutes):
   - Multiple documentation scenarios
   - Verify correct skill routing

**Skill 4: `technical-writer/documentation-structure`**

1. **RED Phase** - Baseline testing (1 hour):
   - Scenario: "Document decision to use PostgreSQL"
   - Run subagent WITHOUT skill
   - Document: Do they use ADR format? Is it complete?

2. **GREEN Phase** - Write skill (2 hours):
   ```
   technical-writer/documentation-structure/
   ├── SKILL.md
   └── templates/ (optional)
       ├── adr-template.md
       ├── api-reference-template.md
       └── runbook-template.md
   ```
   - Content: ADR, architecture docs, API reference, runbooks, README patterns
   - Complete examples for each
   - Decision tree for pattern selection

3. **REFACTOR Phase** - Coverage testing (1 hour):
   - Test different documentation needs
   - Verify pattern retrieval and application
   - Add missing common cases

**Phase 1 Exit Criteria**:
- ✅ All 4 skills pass RED-GREEN-REFACTOR
- ✅ Meta-skills route correctly to core skills
- ✅ Cross-references work (threat-modeling ↔ documentation-structure)
- ✅ Personal use validation on real scenario
- ✅ Commit Phase 1 skills to git

---

### Working Independently in New Repo

**When Claude picks up work in new repo:**

1. **Context Loading**:
   ```
   Read docs/DESIGN.md to understand full design
   Check current phase status (README.md)
   Review existing skills (what's implemented vs planned)
   ```

2. **Phase Execution**:
   ```
   For each skill in current phase:
   1. Create skill directory
   2. Run RED phase (baseline testing)
   3. Write GREEN phase (implement skill)
   4. Run REFACTOR phase (close loopholes)
   5. Commit skill individually
   6. Update README.md status
   ```

3. **Testing Approach**:
   ```
   Use TodoWrite for each skill's RED-GREEN-REFACTOR checklist
   Dispatch subagents for baseline testing (watch them fail)
   Document rationalizations verbatim
   Write skill to address specific failures
   Re-test with skill loaded
   ```

4. **Quality Gates**:
   ```
   Before moving to next skill:
   - Skill passes all test scenarios
   - YAML frontmatter complete (name, description)
   - Examples are concrete and runnable
   - Cross-references valid (if applicable)
   - Git commit with clear message
   ```

5. **Phase Completion**:
   ```
   All skills in phase implemented ✅
   Cross-phase integration tested ✅
   Phase documentation updated ✅
   README.md updated with progress ✅
   Tag release: git tag v0.1-phase1
   ```

---

### Independence Checklist

**Repository is independent when:**
- ✅ Moved to separate git repository
- ✅ README.md explains project purpose
- ✅ docs/DESIGN.md provides complete context
- ✅ LICENSE file chosen
- ✅ .gitignore appropriate for project
- ✅ No dependencies on Elspeth codebase
- ✅ Claude can work on this repo with ONLY contents of this repo

**Claude can work independently when:**
- ✅ Full design context in docs/DESIGN.md
- ✅ Current phase clearly marked in README.md
- ✅ Each skill has clear RED-GREEN-REFACTOR plan
- ✅ Testing methodology documented
- ✅ Quality gates defined
- ✅ Examples of complete skills available (after Phase 1)

---

### Quick Start Commands for New Repo

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/skillpacks.git
cd skillpacks

# Check status
cat README.md  # What phase are we in?
cat docs/DESIGN.md  # Full context

# Start work on next skill
# Example: Phase 1, Skill 1
mkdir -p security-architect/using-security-architect
cd security-architect/using-security-architect

# Create SKILL.md and implement RED-GREEN-REFACTOR
# (following process in "Phase 1 Work Plan" above)

# Commit when done
git add .
git commit -m "Implement security-architect/using-security-architect (Phase 1)"
git push
```

---

## Success Criteria

**Phase 1 Success**:
- 4 foundational skills implemented and tested
- TDD methodology proven for skills
- Cross-referencing syntax validated
- Personal use case validated

**Phase 2 Success**:
- 6 core skills complete (universal applicability)
- All core skills pass RED-GREEN-REFACTOR
- Personal use across multiple Elspeth scenarios

**Phase 3 Success**:
- 8 extension skills complete (specialized)
- Cross-cutting skills bidirectionally reference
- High-security scenarios validated

**Phase 4 Success**:
- Polish complete (README, CONTRIBUTING, examples)
- Public release ready (if sharing)
- All 16 skills production-quality

**Final Success Criteria**:
- ✅ All skills pass RED-GREEN-REFACTOR testing
- ✅ Meta-skills correctly route to specific skills
- ✅ Cross-references work bidirectionally
- ✅ Examples are concrete and runnable
- ✅ Personal use validated on real projects
- ✅ Public contribution ready (if sharing)

---

## Appendix: Skill Summary Table

| Pack | Skill | Type | Priority | Effort (hrs) |
|------|-------|------|----------|--------------|
| security-architect | using-security-architect | Meta | Phase 1 | 2-3 |
| security-architect | threat-modeling | Core | Phase 1 | 3-4 |
| security-architect | security-controls-design | Core | Phase 2 | 3-5 |
| security-architect | architecture-security-review | Core | Phase 2 | 3-5 |
| security-architect | secure-by-design-patterns | Core | Phase 2 | 3-5 |
| security-architect | classified-systems-security | Extension | Phase 3 | 3-5 |
| security-architect | compliance-awareness-and-mapping | Extension | Phase 3 | 3-5 |
| security-architect | security-authorization-and-accreditation | Extension | Phase 3 | 3-5 |
| security-architect | documenting-threats-and-controls | Cross-cutting | Phase 3 | 3-5 |
| technical-writer | using-technical-writer | Meta | Phase 1 | 2-3 |
| technical-writer | documentation-structure | Core | Phase 1 | 3-4 |
| technical-writer | clarity-and-style | Core | Phase 2 | 3-5 |
| technical-writer | diagram-conventions | Core | Phase 2 | 3-5 |
| technical-writer | documentation-testing | Core | Phase 2 | 3-5 |
| technical-writer | security-aware-documentation | Extension | Phase 3 | 3-5 |
| technical-writer | incident-response-documentation | Extension | Phase 3 | 3-5 |
| technical-writer | itil-and-governance-documentation | Extension | Phase 3 | 3-5 |
| technical-writer | operational-acceptance-documentation | Extension | Phase 3 | 3-5 |

**Total**: 16 skills, 60-100 hours

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-28 | 1.0 | Initial design approved |

---

**Next Steps**: Move to independent repository and begin Phase 1 implementation.
