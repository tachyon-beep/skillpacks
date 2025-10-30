---
name: using-technical-writer
description: Router for documentation tasks - routes to ADRs, APIs, runbooks, security docs, or governance docs
---

# Using Technical Writer

## Overview

This meta-skill routes you to the right technical writing skills based on your documentation task. Load this skill when you need to create, improve, or organize documentation but aren't sure which specific writing skill to use.

**Core Principle**: Different document types and audiences require different skills. Match your situation to the appropriate skill, load only what you need.

## When to Use

Load this skill when:
- Starting any documentation task
- User mentions: "document", "write docs", "README", "API docs", "ADR", "runbook"
- Creating technical content for any audience
- Improving or reorganizing existing documentation

**Don't use for**: Non-technical writing (marketing copy, blog posts, creative content)

## Routing by Document Type

### Architecture Decisions (ADRs)

**Symptoms**: "Document why we chose X", "Record this architectural decision", "Explain technology choice"

**Route to**: `muna/technical-writer/documentation-structure`

**Key Pattern**: ADRs document Context → Decision → Consequences

**Example**: "Document why we chose PostgreSQL over MongoDB" → Load documentation-structure

---

### API Documentation

**Symptoms**: "Document this API", "Create API reference", "Explain endpoints"

**Route to**:
1. `muna/technical-writer/documentation-structure` (API reference pattern)
2. `muna/technical-writer/clarity-and-style` (examples, precision)

**Example**: "Document REST API for user management" → Load both skills

---

### Runbooks & Procedures

**Symptoms**: "Write deployment procedure", "Create incident runbook", "Document how to..."

**Route to**:
1. `muna/technical-writer/documentation-structure` (runbook pattern)
2. `muna/technical-writer/clarity-and-style` (step-by-step clarity)

**Example**: "Create deployment runbook" → Load both skills

---

### README Files

**Symptoms**: "Add a README", "Quick start guide", "Installation instructions"

**For Complex Projects**:
Route to: `muna/technical-writer/documentation-structure` (README pattern)

**For Simple Utilities**:
Route to: **NONE** - basic technical writing sufficient

**Decision Point**: Complex (>100 lines, multiple features, deployment) vs Simple (script, single function)

---

### Security Documentation

**Symptoms**: "Document threat model", "Security controls", "Document security decisions"

**Route to (Cross-Faction)**:
1. `ordis/security-architect/documenting-threats-and-controls` (security content)
2. `muna/technical-writer/documentation-structure` (ADR format, organization)
3. `muna/technical-writer/clarity-and-style` (explain to non-experts)

**Key Insight**: Security docs need BOTH content expertise (Ordis) AND writing skills (Muna)

**Example**: "Document authentication security decisions" → Load all three skills

---

## Routing by Audience

### Developer Audience

**What They Need**: Architecture diagrams, code examples, API references, technical depth

**Route to**:
- `muna/technical-writer/documentation-structure` (architecture docs, API patterns)
- `muna/technical-writer/diagram-conventions` (system diagrams, data flows)
- `muna/technical-writer/clarity-and-style` (concrete examples, precision)

**Example**: "Write docs for internal developers" → Load all three

---

### Operator Audience

**What They Need**: Runbooks, troubleshooting, deployment procedures, configuration guides

**Route to**:
- `muna/technical-writer/documentation-structure` (runbook pattern)
- `muna/technical-writer/clarity-and-style` (step-by-step, scannable)

**Example**: "Create SRE runbook" → Load both skills

---

### Executive Audience

**What They Need**: High-level summaries, business impact, risks, costs (minimal technical detail)

**Route to**:
- `muna/technical-writer/clarity-and-style` (progressive disclosure, audience adaptation)

**Example**: "Executive summary of migration plan" → Load clarity-and-style only

---

### Mixed/Public Audience

**What They Need**: Progressive disclosure (quick start → advanced topics), multiple entry points

**Route to**:
- `muna/technical-writer/documentation-structure` (README pattern, API docs)
- `muna/technical-writer/clarity-and-style` (progressive disclosure, audience adaptation)
- `muna/technical-writer/diagram-conventions` (high-level overviews)

**Example**: "Public documentation for open-source project" → Load all three

---

## Cross-Faction Documentation

### Security + Documentation

**When**: Documenting threat models, security controls, classified systems, compliance

**Route to**:
- `ordis/security-architect/documenting-threats-and-controls`
- `ordis/security-architect/threat-modeling` (for threat content)
- `muna/technical-writer/documentation-structure` (for organization)
- `muna/technical-writer/security-aware-documentation` (if handling sensitive info)

**Example**: "Document MLS security architecture" → Load all four skills

---

### Compliance + Documentation

**When**: Audit documentation, SSP/SAR, compliance mappings

**Route to**:
- `ordis/security-architect/compliance-awareness-and-mapping` (compliance content)
- `muna/technical-writer/operational-acceptance-documentation` (SSP/SAR structure)
- `muna/technical-writer/documentation-structure` (organization)

**Example**: "Create SOC2 compliance documentation" → Load all three skills

---

### Incident Response + Documentation

**When**: Post-mortem reports, incident runbooks, response procedures

**Route to**:
- `muna/technical-writer/incident-response-documentation` (incident-specific patterns)
- `muna/technical-writer/documentation-structure` (runbook pattern)
- `muna/technical-writer/clarity-and-style` (clarity under pressure)

**Example**: "Write post-mortem for outage" → Load all three skills

---

## Documentation Workflow

### Standard Flow

```
1. Determine document type → Route to structure skill
2. Write content → Apply clarity/style skill
3. Add diagrams if needed → Use diagram conventions
4. Test documentation → Use documentation-testing
```

### Quick Reference

| Task | Load |
|------|------|
| "Why did we choose X?" | documentation-structure (ADR) |
| "Document API" | documentation-structure + clarity-and-style |
| "Deployment runbook" | documentation-structure + clarity-and-style |
| "README for utility" | NONE (simple) or documentation-structure (complex) |
| "Security docs" | documenting-threats + documentation-structure + clarity |
| "Developer guide" | documentation-structure + diagram-conventions + clarity |
| "Executive summary" | clarity-and-style only |

---

## When NOT to Load Documentation Skills

**Don't load skills for**:
- Simple utility README (<50 lines, single purpose, obvious usage)
- Code comments (use standard practices)
- Commit messages (use conventional commits)
- Chat/email (conversational writing)
- First drafts where you're exploring (capture ideas first, structure later)

**Example**: "Add README to hello-world script" → No special skills needed

---

## Core vs Extension Skills

### Core Skills (Universal - Any Project)

Use for **any** project:
- `documentation-structure` - ADRs, APIs, runbooks, READMEs, architecture docs
- `clarity-and-style` - Active voice, concrete examples, audience adaptation
- `diagram-conventions` - System diagrams, data flows, architecture visuals
- `documentation-testing` - Verify docs are accurate, complete, findable

### Extension Skills (Specialized Contexts)

Use **only** when context requires:
- `security-aware-documentation` - Sanitizing examples with sensitive data, classification marking
- `incident-response-documentation` - Post-mortems, incident runbooks, RCA templates
- `itil-and-governance-documentation` - ITIL processes, change management, governance frameworks
- `operational-acceptance-documentation` - SSP, SAR, POA&M for government authorization

**Decision**: If unsure whether context is "specialized", start with core skills. Specialized needs will be explicit.

---

## Common Routing Patterns

### Pattern 1: ADR for Architecture Decision

```
User: "We chose to use REST instead of GraphQL. Document this."
You: Loading muna/technical-writer/documentation-structure (ADR pattern)
```

### Pattern 2: API Documentation

```
User: "Document our user management API."
You: Loading muna/technical-writer/documentation-structure + clarity-and-style
```

### Pattern 3: Security Documentation (Cross-Faction)

```
User: "Document the threat model for authentication."
You: Loading ordis/security-architect/documenting-threats-and-controls +
     muna/technical-writer/documentation-structure +
     muna/technical-writer/clarity-and-style
```

### Pattern 4: Simple README

```
User: "Add README to this backup script."
You: [Check script complexity]
     Simple utility → No skills needed
     OR
     Complex tool → Loading documentation-structure
```

### Pattern 5: Operator Runbook

```
User: "Create runbook for database failover."
You: Loading muna/technical-writer/documentation-structure (runbook) +
     clarity-and-style (step-by-step clarity)
```

---

## Decision Tree

```
Starting documentation task?
├─ What type?
│  ├─ Architecture decision → documentation-structure (ADR)
│  ├─ API documentation → documentation-structure + clarity-and-style
│  ├─ Runbook/procedure → documentation-structure + clarity-and-style
│  ├─ README → Complex? documentation-structure : None
│  └─ Security/compliance → Cross-faction (Ordis + Muna)
│
├─ Who's the audience?
│  ├─ Developers → Add diagram-conventions
│  ├─ Operators → Focus on runbook patterns
│  ├─ Executives → clarity-and-style only (progressive disclosure)
│  └─ Mixed → All core skills
│
└─ Specialized context?
   ├─ Sensitive data → ADD: security-aware-documentation
   ├─ Incident response → ADD: incident-response-documentation
   ├─ Government/compliance → ADD: operational-acceptance-documentation
   └─ None → Core skills sufficient
```

---

## Quick Reference Table

| Document Type | Primary Skill | Additional Skills | Cross-Faction? |
|---------------|---------------|-------------------|----------------|
| **ADR** | documentation-structure | clarity-and-style | No |
| **API docs** | documentation-structure | clarity-and-style | No |
| **Runbook** | documentation-structure | clarity-and-style | No |
| **README (complex)** | documentation-structure | clarity-and-style, diagram-conventions | No |
| **README (simple)** | NONE | NONE | No |
| **Security docs** | documenting-threats-and-controls | documentation-structure, clarity-and-style | **Yes (Ordis)** |
| **Compliance** | operational-acceptance-documentation | documentation-structure | **Yes (Ordis)** |
| **Developer guide** | documentation-structure | diagram-conventions, clarity-and-style | No |
| **Operator guide** | documentation-structure | clarity-and-style | No |
| **Executive summary** | clarity-and-style | NONE | No |
| **Post-mortem** | incident-response-documentation | documentation-structure, clarity-and-style | No |

---

## Common Mistakes

### ❌ Loading All Skills for Every Task
**Wrong**: Load all 8 Muna skills for every documentation task
**Right**: Load only skills your situation needs (use decision tree)

### ❌ Missing Cross-Faction Needs
**Wrong**: Document security with only Muna skills (missing security content expertise)
**Right**: Load Ordis skills for content + Muna skills for structure/clarity

### ❌ Over-Engineering Simple Docs
**Wrong**: Load documentation-structure for 10-line README
**Right**: Simple docs don't need special skills (just write clearly)

### ❌ Not Considering Audience
**Wrong**: Same documentation for developers and executives
**Right**: Adapt content and skills based on audience needs

---

## Examples

### Example 1: Documenting Database Choice

```
User: "We decided on PostgreSQL. Document why."

Your routing:
1. Recognize: Architecture decision → ADR format
2. Load: muna/technical-writer/documentation-structure
3. Create: ADR with Context, Decision, Consequences
```

### Example 2: Security Threat Model Documentation

```
User: "Document the threat model for our API gateway."

Your routing:
1. Recognize: Security content (need Ordis) + Documentation (need Muna)
2. Load: ordis/security-architect/documenting-threats-and-controls (threats content)
3. Load: muna/technical-writer/documentation-structure (ADR for security decisions)
4. Load: muna/technical-writer/clarity-and-style (explain to non-security team)
5. Create: Threat model doc with STRIDE analysis + mitigations + clear explanations
```

### Example 3: Simple Utility README

```
User: "Add README to this file-copy script."

Your routing:
1. Recognize: Simple utility (single function, obvious usage)
2. Decision: No special skills needed
3. Create: Basic README with usage example, no complex structure
```

---

## Phase 1 Note

**Currently Available** (Phase 1):
- ✅ `using-technical-writer` (this skill)
- ✅ `documentation-structure` (in progress)

**Coming Soon** (Phases 2-3):
- `clarity-and-style`
- `diagram-conventions`
- `documentation-testing`
- `security-aware-documentation`
- `incident-response-documentation`
- `itil-and-governance-documentation`
- `operational-acceptance-documentation`

**For Phase 1**: Focus on documentation-structure as primary skill. Reference other skills by name even though not implemented yet - this tests routing logic.

---

## Summary

**This skill maps documentation tasks → specific writing skills to load.**

1. Identify document type (ADR, API, runbook, README, security)
2. Use decision tree to find applicable skills
3. Load core skills for universal needs
4. Add extension skills for specialized contexts
5. Cross-reference Ordis for security/compliance content
6. Don't load skills when not needed (simple docs)

**Meta-rule**: When in doubt about document type, start with `documentation-structure` - it covers most common patterns (ADR, API, runbook, README).
