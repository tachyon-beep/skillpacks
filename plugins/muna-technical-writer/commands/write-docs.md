---
description: Write documentation using proven patterns - ADRs, API reference, runbooks, READMEs
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[document_type] [topic_or_system]"
---

# Write Documentation Command

You are creating technical documentation using proven patterns that ensure findability and usability.

## Core Principle

**Structure determines findability. Well-structured docs get used; poorly structured docs get ignored.**

Every abstract concept needs a concrete, runnable example. Every audience needs information in their language.

## Document Type Selection

### Ask First

If document type is unclear, determine:
1. **What kind of document?** ADR, API reference, runbook, README?
2. **Who is the audience?** Developers, operators, executives?
3. **What action should readers take?** Understand, implement, operate?

### Document Patterns

| Document Type | Use When | Key Sections |
|---------------|----------|--------------|
| **ADR** | Recording architecture/technology decisions | Context, Decision, Alternatives, Consequences |
| **API Reference** | Documenting REST/GraphQL APIs | Auth, Rate Limiting, Pagination, Endpoints, Errors |
| **Runbook** | Operational procedures | Prerequisites, Safety, Procedure, Verification, Rollback |
| **README (Simple)** | Small utilities (<100 lines) | Installation, Usage, Options |
| **README (Standard)** | Most projects | Features, Installation, Quick Start, Usage, Config |

## ADR Pattern

**Use for**: Technology choices, architecture patterns, trade-off decisions

```markdown
# ADR-NNN: [Short Title]

**Status**: [Proposed | Accepted | Deprecated]
**Date**: YYYY-MM-DD
**Deciders**: [Names/roles]

## Summary
[One-paragraph summary]

## Context
[Problem being solved, constraints, requirements]

## Decision
We will [decision statement].

## Alternatives Considered

### Alternative 1: [Name]
**Pros**: [List]
**Cons**: [List]
**Why rejected**: [Specific reason]

## Consequences

### Positive
- [Good outcome 1]

### Negative
- [Trade-off 1]

## References
- [Links]
```

## API Reference Pattern

**Use for**: REST APIs, GraphQL APIs, SDK interfaces

**Required Sections**:
1. Overview (Base URL, protocol, format)
2. Authentication (method, obtaining credentials, examples)
3. Rate Limiting (limits, headers, handling 429)
4. Pagination (parameters, response format)
5. Endpoints (per resource: method, parameters, examples, errors)
6. Error Codes (table with code, name, description)

**Per Endpoint**:
```markdown
#### [Action] [Resource]

**Endpoint**: `GET /resource/{id}`

**Parameters**:
- `id` (string, required): Resource identifier

**Example Request**:
```bash
curl -X GET "https://api.example.com/v1/resource/123" \
  -H "Authorization: Bearer token..."
```

**Success Response** (200 OK):
```json
{
  "id": "123",
  "name": "Example"
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid token
- `404 Not Found`: Resource doesn't exist
```

## Runbook Pattern

**Use for**: Deployment, incident response, maintenance, recovery

**Required Sections**:
1. Purpose & Owner
2. Prerequisites (access, tools, knowledge)
3. Safety Checks (stop conditions)
4. Procedure (numbered steps with commands)
5. Verification (success criteria)
6. Rollback (recovery steps)
7. Troubleshooting (common issues)
8. Escalation (who to contact)

**Per Step**:
```markdown
### Step N: [Action]

**Purpose**: [What this achieves]

```bash
# Command to run
```

**Expected Result**: [What you should see]

**If this fails**: [Troubleshooting]
```

## README Pattern

**Simple** (utilities, scripts):
- What it does (one sentence)
- Installation
- Usage example
- Options

**Standard** (applications, libraries):
- Description
- Features
- Installation (prerequisites, steps)
- Quick Start
- Usage (basic, advanced)
- Configuration
- Documentation links
- Contributing
- License

## Writing Process

### Step 1: Determine Type and Audience

Ask:
- What document type fits this content?
- Who will read this?
- What should they do after reading?

### Step 2: Apply Pattern

Use the appropriate pattern structure above.

### Step 3: Apply Clarity Rules

**Active Voice**: "Run tests" not "tests should be run"
**Concrete Examples**: Every instruction has runnable example
**Scannable Structure**: Headings, bullets, tables, code blocks
**Progressive Disclosure**: Essentials first, details expandable

### Step 4: Verify Completeness

**ADR**: Has alternatives? Has consequences (positive AND negative)?
**API**: Has auth? Rate limiting? Pagination? Error codes?
**Runbook**: Has prerequisites? Rollback? Troubleshooting?
**README**: Has installation? Usage example? Is it runnable?

## Output Format

Create documentation in the appropriate pattern, then provide:

```markdown
## Documentation Created

**Type**: [ADR/API/Runbook/README]
**File**: [Path where saved]
**Audience**: [Who this is for]

### Completeness Check
- [x] [Required section 1]
- [x] [Required section 2]
- [ ] [Optional section - not applicable because...]

### Review Recommendations
- [Any follow-up needed]
```

## Cross-Pack Discovery

```python
import glob

# For security documentation
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("Available: ordis-security-architect for threat model documentation")

# For diagram conventions
diagram_ref = glob.glob("plugins/muna-technical-writer/skills/using-technical-writer/diagram-conventions.md")
if diagram_ref:
    print("Available: diagram-conventions.md for architecture diagrams")
```

## Scope Boundaries

**This command covers:**
- ADR creation
- API reference documentation
- Runbook writing
- README creation
- Documentation structure

**Not covered:**
- Documentation review (use /review-docs)
- Security-specific documentation (use ordis-security-architect)
- Diagram creation (separate task)
