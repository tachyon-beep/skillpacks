---
name: documentation-structure
description: Templates for ADRs, API docs, runbooks, READMEs - consistent, complete, findable documentation
---

# Documentation Structure

## Overview

Proven documentation patterns for common technical content. Use these templates to create consistent, complete, findable documentation.

**Core Principle**: Structure determines findability. Well-structured docs get used; poorly structured docs get ignored.

## When to Use

Load this skill when:
- Creating new documentation (ADR, API docs, runbook, README)
- Choosing documentation format
- Organizing existing scattered documentation
- User mentions: "document decision", "API reference", "runbook", "README"

## ADR (Architecture Decision Record)

### When to Use ADRs

**Use ADRs for**:
- Technology choices (database, framework, library)
- Architecture patterns (microservices vs monolith, REST vs GraphQL)
- Design decisions with long-term consequences
- Trade-off decisions (performance vs simplicity)

**Don't use ADRs for**:
- Implementation details (how to write a function)
- Temporary decisions (which bug to fix first)
- Obvious choices (use version control, write tests)

### Complete ADR Template

```markdown
# ADR-NNN: [Short Title of Decision]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
**Date**: YYYY-MM-DD
**Deciders**: [Names or roles of people who made decision]
**Context**: [What prompted this decision]

## Summary

[One-paragraph summary of the decision and its impact]

## Context

[Describe the problem you're solving]

- What constraints exist? (technical, business, time, people)
- What requirements must be met?
- What assumptions are we making?
- What's the current state (if replacing something)?

## Decision

[State the decision clearly and concisely]

We will [decision statement].

## Alternatives Considered

### Alternative 1: [Name]

**Description**: [What this alternative involves]

**Pros**:
- [Advantage 1]
- [Advantage 2]

**Cons**:
- [Disadvantage 1]
- [Disadvantage 2]

**Why rejected**: [Specific reason this wasn't chosen]

### Alternative 2: [Name]

[Same format as Alternative 1]

## Consequences

### Positive

- [Good outcome 1]
- [Good outcome 2]

### Negative

- [Trade-off 1]
- [Trade-off 2]

### Neutral

- [Change that's neither good nor bad, just different]

## Implementation Notes

[Optional: Technical details, migration steps, timeline]

## Related Decisions

- **Supersedes**: ADR-XXX (if applicable)
- **Superseded by**: ADR-YYY (if applicable)
- **Related to**: ADR-ZZZ, ADR-AAA (decisions that interact with this one)

## References

- [Links to relevant documentation, RFCs, blog posts, research papers]

```

### ADR Numbering Convention

- **Sequential numbering**: ADR-001, ADR-002, etc.
- **Never reuse numbers** (even if decision is deprecated)
- **Pad with zeros**: ADR-007 not ADR-7 (sorts correctly)

### ADR Location

```
docs/architecture/decisions/
├── README.md (index of all ADRs)
├── ADR-001-use-postgresql.md
├── ADR-002-mls-enforcement.md
├── ADR-003-plugin-registry.md
└── ADR-004-abc-over-protocol.md
```

### Example: Real ADR (BasePlugin ABC)

```markdown
# ADR-004: Use Abstract Base Class Instead of Protocol for Plugin System

**Status**: Accepted
**Date**: 2025-10-28
**Deciders**: Security Architecture Team
**Context**: Multi-Level Security enforcement requires reliable type checking

## Summary

We will use Abstract Base Class (ABC) instead of Protocol for the BasePlugin interface
to enable runtime type verification critical for security level enforcement.

## Context

The plugin system requires security level validation before plugins can execute. We need
to verify that all plugins inherit from BasePlugin to ensure they implement mandatory
security methods and properties.

Constraints:
- Security level must be immutable and verifiable at runtime
- Plugin registration must confirm plugin type before allowing execution
- Need to prevent duck-typed plugins from bypassing security checks

Python offers two approaches for defining plugin interfaces:
1. Protocol (PEP 544) - structural subtyping (duck typing)
2. Abstract Base Class - nominal typing with inheritance

## Decision

We will use Abstract Base Class (ABC) with @abstractmethod for the BasePlugin interface.

## Alternatives Considered

### Alternative 1: Protocol-based Interface

**Description**: Define BasePlugin as a Protocol, allowing any class implementing
the required methods to be considered a valid plugin.

**Pros**:
- More flexible - no inheritance required
- Easier for third-party plugins
- More "Pythonic" for general use

**Cons**:
- isinstance() checks don't work reliably with Protocol
- Security bypass risk: attacker creates duck-typed plugin without BasePlugin
- Can't seal security-critical methods
- Type checking is structural, not nominal

**Why rejected**: Security level verification requires isinstance() to confirm plugin
inheritance. Protocol duck typing allows security bypasses (see threat model THREAT-003).

### Alternative 2: Manual Registration Without Type Checks

**Description**: Don't enforce type at all - rely on plugin registry and runtime checks.

**Pros**:
- Maximum flexibility
- No inheritance requirements

**Cons**:
- No compile-time safety
- Easy to bypass registration
- Higher runtime overhead for checks

**Why rejected**: Defense-in-depth principle requires type system + registry + runtime.
Single-layer validation is insufficient.

## Consequences

### Positive

- isinstance(plugin, BasePlugin) provides reliable runtime type checking
- Sealed methods prevent subclasses from overriding security-critical code
- Nominal typing makes security boundaries explicit
- Compile-time type safety via mypy

### Negative

- Third-party plugins must inherit from BasePlugin (less flexible)
- Tighter coupling between plugins and framework
- Slightly more boilerplate for plugin authors

### Neutral

- Plugins must be registered AND inherit from BasePlugin (defense-in-depth)

## Implementation Notes

- BasePlugin declared as ABC with frozen dataclass
- security_level property marked as @abstractmethod
- Plugin factory verifies isinstance() before instantiation
- Mypy configured to require nominal types for plugins

## Related Decisions

- **Related to**: ADR-002 (MLS enforcement - requires type checking)
- **Related to**: ADR-003 (Plugin registry - ABC + registry = defense-in-depth)
- **Related to**: ADR-005 (Frozen plugin capability - ABC enables sealed methods)

## References

- PEP 544: Protocols - https://peps.python.org/pep-0544/
- Bell-LaPadula MLS model requirements
- Threat model THREAT-003: Type system bypass via duck typing
```

---

## API Reference Documentation

### When to Use API Reference Pattern

**Use for**:
- REST APIs
- GraphQL APIs
- Library/SDK public interfaces
- Internal service APIs

### Complete API Documentation Structure

```markdown
# [Service/API Name] API Reference

## Overview

**Base URL**: `https://api.example.com/v1`
**Protocol**: HTTPS only
**Format**: JSON

[One-paragraph description of what this API does]

## Authentication

### Method

[OAuth 2.0 | API Key | JWT | etc.]

### Obtaining Credentials

[How to get API key/token]

### Using Authentication

**Header Format**:
```
Authorization: Bearer {token}
```

**Example**:
```bash
curl -H "Authorization: Bearer abc123..." https://api.example.com/v1/users
```

### Token Expiration

- **Access tokens**: 1 hour
- **Refresh tokens**: 30 days

## Rate Limiting

- **Limit**: 1000 requests per hour per API key
- **Headers**:
  - `X-RateLimit-Limit`: Your rate limit ceiling
  - `X-RateLimit-Remaining`: Requests remaining in window
  - `X-RateLimit-Reset`: UTC epoch seconds when limit resets

**Example Response** (429 Too Many Requests):
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 1000 requests per hour exceeded",
  "retry_after": 1800
}
```

## Pagination

### Parameters

- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

### Response Format

```json
{
  "data": [ /* items */ ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 157,
    "pages": 8
  }
}
```

### Navigation Links

```json
{
  "links": {
    "first": "https://api.example.com/v1/users?page=1",
    "prev": null,
    "next": "https://api.example.com/v1/users?page=2",
    "last": "https://api.example.com/v1/users?page=8"
  }
}
```

## Versioning

- **URL-based versioning**: `/v1/`, `/v2/`
- **Current version**: v1
- **Deprecation policy**: 12 months notice before version sunset

## Endpoints

### [Resource Name]

#### List [Resources]

**Endpoint**: `GET /[resource]`

**Description**: [What this endpoint does]

**Authentication**: Required

**Query Parameters**:
- `param1` (string, optional): [Description]
- `param2` (integer, optional): [Description]
- `page` (integer, optional): Page number
- `limit` (integer, optional): Items per page

**Example Request**:
```bash
curl -X GET "https://api.example.com/v1/users?role=admin&page=1&limit=20" \
  -H "Authorization: Bearer abc123..."
```

**Success Response** (200 OK):
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "username": "jdoe",
      "email": "jdoe@example.com",
      "role": "admin",
      "created_at": "2025-10-28T14:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Authenticated but lacks permission
- `429 Too Many Requests`: Rate limit exceeded

---

#### Get [Resource]

**Endpoint**: `GET /[resource]/{id}`

[Similar format as above]

---

#### Create [Resource]

**Endpoint**: `POST /[resource]`

[Similar format as above]

---

#### Update [Resource]

**Endpoint**: `PUT /[resource]/{id}` or `PATCH /[resource]/{id}`

[Similar format as above]

---

#### Delete [Resource]

**Endpoint**: `DELETE /[resource]/{id}`

[Similar format as above]

---

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but lacks permission |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource already exists or version conflict |
| 422 | Unprocessable Entity | Validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error (contact support) |
| 503 | Service Unavailable | Temporary outage or maintenance |

### Error Response Format

```json
{
  "error": "error_code_identifier",
  "message": "Human-readable error message",
  "details": {
    "field": "specific_field_with_error",
    "reason": "why_it_failed"
  },
  "request_id": "req_abc123",
  "timestamp": "2025-10-28T14:30:00Z"
}
```

## SDKs and Client Libraries

[Links to official SDKs for different languages]

## Webhooks

[If applicable - webhook registration, event types, payload formats]

## Changelog

### v1.2.0 (2025-10-15)
- Added: Webhook support for user events
- Changed: Increased rate limit from 500 to 1000 req/hour

### v1.1.0 (2025-09-01)
- Added: PATCH support for partial updates
- Fixed: Pagination links for empty results

```

---

## Runbook Pattern

### When to Use Runbooks

**Use runbooks for**:
- Deployment procedures
- Incident response playbooks
- Maintenance operations
- Recovery procedures
- Regular operational tasks

### Complete Runbook Template

```markdown
# Runbook: [Operation Name]

**Purpose**: [One-sentence description of what this runbook achieves]
**Owner**: [Team or person responsible]
**Last Updated**: YYYY-MM-DD
**Frequency**: [On-demand | Weekly | Monthly | During incidents]

## Overview

[2-3 sentences describing when to use this runbook and what it accomplishes]

## Prerequisites

### Required Access

- [ ] Production database access (role: `db-operator`)
- [ ] Kubernetes cluster access (namespace: `production`)
- [ ] PagerDuty access (for incident updates)
- [ ] VPN connection to production network

### Required Tools

- [ ] `kubectl` v1.28+
- [ ] `psql` PostgreSQL client
- [ ] `aws-cli` configured with production profile
- [ ] SSH key for bastion host

### Required Knowledge

- Basic Kubernetes concepts
- SQL query syntax
- Understanding of [specific system architecture]

### Verification

Run these commands to verify prerequisites:
```bash
# Check kubectl access
kubectl get nodes

# Check database access
psql -h db.production.example.com -U operator -c "SELECT 1"

# Check AWS access
aws sts get-caller-identity
```

## Safety Checks

**STOP if any of these are true**:
- [ ] Active incident in progress (check PagerDuty)
- [ ] Scheduled maintenance window not started
- [ ] Change request not approved
- [ ] Backup not verified (see "Pre-Operation Backup" below)

## Procedure

### Step 1: Create Backup

**Purpose**: Ensure rollback is possible if operation fails

```bash
# Create database backup
pg_dump -h db.production.example.com -U operator \
  -Fc production_db > backup-$(date +%Y%m%d-%H%M%S).dump

# Verify backup
ls -lh backup-*.dump
```

**Expected Result**: Backup file created, size > 0 bytes

**If this fails**: [What to do if backup fails]

---

### Step 2: [Operation Step]

**Purpose**: [What this step does]

```bash
# Commands to run
command1
command2
```

**Expected Result**: [What you should see]

**If this fails**: [Troubleshooting steps]

---

[Repeat for each step]

---

### Final Step: Verify Operation

**Purpose**: Confirm operation succeeded

```bash
# Verification commands
```

**Success Criteria**:
- [ ] Service responds with 200 OK
- [ ] No errors in logs (last 5 minutes)
- [ ] Metrics show normal traffic

## Post-Operation

### Update Tracking

- [ ] Update change request ticket with completion time
- [ ] Update runbook if procedure changed
- [ ] Document any deviations from standard procedure

### Monitoring

Monitor these for 30 minutes after operation:
- Application logs: `kubectl logs -f deployment/app -n production`
- Error rate: [Link to monitoring dashboard]
- Response time: [Link to metrics]

## Rollback Procedure

**When to rollback**:
- Operation failed at any step
- Post-operation verification failed
- Unexpected behavior observed

**Steps**:
```bash
# Restore from backup
pg_restore -h db.production.example.com -U operator \
  -d production_db backup-YYYYMMDD-HHMMSS.dump
```

[Additional rollback steps]

**Verification**:
- [ ] Service restored to pre-operation state
- [ ] No data loss confirmed
- [ ] Application functioning normally

## Troubleshooting

### Problem: [Common Issue 1]

**Symptoms**: [What you see]

**Cause**: [Why this happens]

**Solution**:
```bash
# Commands to fix
```

---

### Problem: [Common Issue 2]

[Same format]

---

## Escalation

**When to escalate**:
- Rollback failed
- Data integrity concerns
- Incident severity increases
- Unsure how to proceed

**Who to contact**:
1. **On-call engineer**: [PagerDuty rotation or phone]
2. **Database team**: [Contact method]
3. **Security team** (if data breach suspected): [Contact method]

## References

- [Link to architecture diagram]
- [Link to related runbooks]
- [Link to incident post-mortems]
- [Link to system documentation]

```

---

## README Patterns

### When to Use Each README Type

**Simple README** (<100 lines):
- Single-purpose utilities
- Scripts
- Small libraries

**Standard README** (100-300 lines):
- Applications
- Multi-feature libraries
- Services

**Comprehensive README** (300+ lines):
- Open-source projects
- Complex systems
- Projects with many contributors

### Simple README Template

For small utilities and scripts:

```markdown
# [Project Name]

[One-sentence description of what it does]

## Installation

```bash
pip install project-name
```

## Usage

```bash
# Basic example
project-name input.txt output.txt

# With options
project-name --verbose input.txt output.txt
```

## Options

- `--verbose`: Print detailed progress
- `--output FILE`: Specify output file

## Requirements

- Python 3.8+
- No external dependencies

## License

MIT
```

### Standard README Template

For most projects:

```markdown
# [Project Name]

[2-3 sentence description of what the project does and why it exists]

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

### Prerequisites

- [Dependency 1] version X.Y+
- [Dependency 2]

### Install from Source

```bash
git clone https://github.com/user/project.git
cd project
pip install -r requirements.txt
```

### Install from Package Manager

```bash
pip install project-name
```

## Quick Start

```bash
# Minimal example to get started
project-name --help
```

## Usage

### Basic Usage

```bash
# Example 1
project-name command arg1 arg2

# Example 2
project-name --option value
```

### Advanced Usage

[More complex examples]

## Configuration

Configuration file location: `~/.project/config.yml`

```yaml
# Example configuration
option1: value1
option2: value2
```

## Documentation

- [User Guide](docs/user-guide.md)
- [API Reference](docs/api-reference.md)
- [Architecture](docs/architecture/README.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

[License name] - see [LICENSE](LICENSE)

## Support

- [Issue tracker](https://github.com/user/project/issues)
- [Discussions](https://github.com/user/project/discussions)
- [Email](support@example.com)
```

### Comprehensive README Template

For open-source and complex projects:

```markdown
# [Project Name]

[![Build Status](badge-url)](build-url)
[![Coverage](badge-url)](coverage-url)
[![License](badge-url)](license-url)

[3-4 sentence description of the project, its purpose, and key benefits]

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

[Rest of content follows standard README template but with more depth]

## Architecture

High-level overview with diagram:

```
[ASCII diagram or link to docs/architecture/]
```

## Performance

- Benchmark results
- Scalability characteristics
- Resource requirements

## Security

See [SECURITY.md](SECURITY.md) for security policy and vulnerability reporting.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Roadmap

- [x] Feature 1 (completed)
- [ ] Feature 2 (in progress)
- [ ] Feature 3 (planned)

See [full roadmap](ROADMAP.md)
```

---

## Architecture Documentation Structure

### Directory Organization

```
docs/
├── README.md (navigation hub)
├── architecture/
│   ├── README.md (system overview)
│   ├── decisions/ (ADRs)
│   │   ├── README.md (ADR index)
│   │   └── ADR-NNN-*.md
│   ├── diagrams/
│   │   ├── system-overview.png
│   │   ├── data-flow.png
│   │   └── deployment.png
│   ├── components/
│   │   ├── authentication.md
│   │   ├── database.md
│   │   └── api-gateway.md
│   └── security/
│       ├── threat-model.md
│       ├── access-control.md
│       └── encryption.md
├── api/
│   └── reference.md
├── guides/
│   ├── getting-started.md
│   ├── contributing.md
│   └── deployment.md
└── runbooks/
    ├── deployment.md
    ├── backup-restore.md
    └── incident-response.md
```

---

## Common Mistakes

### ❌ Incomplete ADRs
**Wrong**: ADR with only "We chose X" and no alternatives/consequences
**Right**: Complete ADR with Context, Alternatives Considered, Consequences, Related Decisions

### ❌ Scattered Documentation
**Wrong**: Decisions in README, code comments, wiki, Slack
**Right**: Single source of truth - decisions in ADRs, linked from other locations

### ❌ Missing API Details
**Wrong**: API docs with only endpoints and examples
**Right**: API docs with auth, rate limiting, pagination, versioning, error codes

### ❌ Incomplete Runbooks
**Wrong**: Runbook with only procedure steps
**Right**: Runbook with prerequisites, safety checks, verification, rollback, troubleshooting

### ❌ Generic README
**Wrong**: README saying "This is a project that does things"
**Right**: README with concrete features, runnable examples, clear installation steps

---

## Quick Reference

| Document Type | Use When | Key Sections |
|---------------|----------|--------------|
| **ADR** | Architecture/technology decisions with long-term impact | Context, Alternatives, Consequences, Related Decisions |
| **API Reference** | Documenting REST/GraphQL APIs | Auth, Rate Limiting, Pagination, Endpoints, Errors |
| **Runbook** | Operational procedures | Prerequisites, Safety, Procedure, Verification, Rollback |
| **README (Simple)** | Small utilities (<100 lines) | Installation, Usage, Options |
| **README (Standard)** | Most projects | Features, Installation, Quick Start, Usage, Config |
| **README (Comprehensive)** | Open-source/complex projects | All standard + Architecture, Performance, Roadmap |

---

## Real-World Example: Elspeth Documentation Evolution

**Before** (Scattered narratives):
- README: 8 sections explaining architecture decisions
- Code comments: "// We chose ABC because..."
- No traceability or findability

**After** (Structured with ADRs):
- 14 ADRs documenting key decisions
- README: Quick start + links to ADRs
- Code comments: `// See ADR-004 for rationale`
- Clear decision trail: ADR-002 (MLS) → ADR-003 (Registry) → ADR-004 (ABC) → ADR-005 (Frozen)

**Key Improvement**: "Can't find why we chose X" → "ADR-004 documents ABC vs Protocol decision with full context"

---

## Summary

**Use the right structure for the content type:**

- **ADRs**: Architecture decisions → Complete template with alternatives and consequences
- **API docs**: REST/GraphQL → Auth, rate limiting, pagination, versioning, errors
- **Runbooks**: Operations → Prerequisites, safety, procedure, verification, rollback
- **READMEs**: Project overview → Match complexity (simple/standard/comprehensive)

**Meta-rule**: Good structure makes docs findable. If readers can't find it, you haven't documented it.
