---
description: Create an Architecture Decision Record documenting a technology or design choice
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[decision_topic]"
---

# Create ADR Command

You are creating an Architecture Decision Record (ADR) to document a technology or design choice.

## Core Principle

**ADRs capture the WHY behind decisions. Without ADRs, future developers don't know why choices were made and may reverse good decisions or repeat bad ones.**

## When to Use ADRs

**Use ADRs for**:
- Technology choices (database, framework, library)
- Architecture patterns (microservices vs monolith, REST vs GraphQL)
- Design decisions with long-term consequences
- Trade-off decisions (performance vs simplicity)

**Don't use ADRs for**:
- Implementation details (how to write a function)
- Temporary decisions (which bug to fix first)
- Obvious choices (use version control, write tests)

## Information Gathering

Before writing, gather:

1. **What is the decision?** Clear statement of what was chosen
2. **Why is this needed?** Problem being solved
3. **What constraints exist?** Technical, business, time, people
4. **What alternatives were considered?** At least 2-3 options
5. **What are the trade-offs?** Pros and cons of chosen option

## ADR Template

```markdown
# ADR-NNN: [Short Descriptive Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
**Date**: YYYY-MM-DD
**Deciders**: [Names or roles]
**Context**: [Brief one-line context]

## Summary

[One paragraph summarizing the decision and its primary impact]

## Context

[Describe the problem you're solving]

- What prompted this decision?
- What constraints exist? (technical, business, time, people)
- What requirements must be met?
- What assumptions are we making?
- What's the current state? (if replacing something)

## Decision

We will [clear decision statement].

[Additional detail about the decision if needed]

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

[Same structure]

### Alternative 3: [Name] (if applicable)

[Same structure]

## Consequences

### Positive

- [Good outcome 1]
- [Good outcome 2]

### Negative

- [Trade-off 1 - what we're giving up]
- [Trade-off 2]

### Neutral

- [Change that's neither good nor bad]

## Implementation Notes

[Optional: Key technical details, migration steps, timeline considerations]

## Related Decisions

- **Supersedes**: ADR-XXX (if replacing previous decision)
- **Superseded by**: ADR-YYY (if this is deprecated)
- **Related to**: ADR-ZZZ (decisions that interact with this one)

## References

- [Links to documentation, RFCs, research papers, discussions]
```

## ADR Conventions

### Numbering

- **Sequential**: ADR-001, ADR-002, etc.
- **Never reuse numbers** (even if deprecated)
- **Pad with zeros**: ADR-007 not ADR-7

### Status Lifecycle

```
Proposed → Accepted → [Deprecated | Superseded]
```

- **Proposed**: Under discussion, not yet decided
- **Accepted**: Decision made, being implemented
- **Deprecated**: No longer applicable (context changed)
- **Superseded**: Replaced by newer ADR (link to it)

### Location

```
docs/architecture/decisions/
├── README.md (index of all ADRs)
├── ADR-001-use-postgresql.md
├── ADR-002-rest-over-graphql.md
└── ADR-003-microservices.md
```

## Writing Process

### Step 1: Identify the Decision

Ask: What specific choice was made (or needs to be made)?

### Step 2: Document Context

Ask: Why did this decision need to be made? What problem were we solving?

### Step 3: List Alternatives

**Minimum 2 alternatives**. For each:
- What is it?
- What are the pros?
- What are the cons?
- Why was it rejected?

### Step 4: Document Consequences

**Both positive AND negative**. If there are no negative consequences, you haven't thought hard enough.

### Step 5: Review for Completeness

- [ ] Title clearly describes decision
- [ ] Context explains why decision was needed
- [ ] At least 2 alternatives with pros/cons
- [ ] Clear reason why alternatives were rejected
- [ ] Consequences include trade-offs (negatives)
- [ ] Related decisions linked

## Example: Database Choice ADR

```markdown
# ADR-001: Use PostgreSQL for Primary Database

**Status**: Accepted
**Date**: 2025-01-15
**Deciders**: Engineering Team
**Context**: Need persistent storage for user data and transactions

## Summary

We will use PostgreSQL as our primary database due to its ACID compliance,
JSON support, and team familiarity, despite higher operational complexity
compared to managed alternatives.

## Context

The application requires reliable storage for user accounts, transactions,
and audit logs. Requirements:
- ACID transactions for financial data
- Flexible schema for evolving data model
- Team has SQL expertise
- Budget constraints favor open-source

## Decision

We will use PostgreSQL 15+ as our primary database.

## Alternatives Considered

### Alternative 1: MongoDB

**Pros**: Flexible schema, easy scaling, JSON-native
**Cons**: Weaker transactions, team unfamiliar, licensing concerns
**Why rejected**: Financial data requires strong ACID guarantees

### Alternative 2: Amazon Aurora

**Pros**: Managed service, PostgreSQL-compatible, auto-scaling
**Cons**: 3x cost, vendor lock-in, less control
**Why rejected**: Budget constraints; can migrate later if needed

## Consequences

### Positive
- Strong ACID compliance for financial data
- Rich JSON support via JSONB
- Team already knows PostgreSQL

### Negative
- Must manage backups, replication ourselves
- Scaling requires more planning than managed services
- Operational overhead for HA setup

## References

- PostgreSQL vs MongoDB comparison: [link]
- Aurora pricing analysis: [internal doc]
```

## Cross-Pack Discovery

```python
import glob

# For security-related ADRs
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("Available: ordis-security-architect for security ADRs")
```

## Scope Boundaries

**This command covers:**
- ADR creation
- Decision documentation
- Alternative analysis
- Consequence documentation

**Not covered:**
- Other documentation types (use /write-docs)
- Documentation review (use /review-docs)
- Technical implementation
