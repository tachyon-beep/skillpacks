---
description: Analyze subsystem dependencies from catalog - detect circular dependencies, layer violations, missing bidirectional references, and generate Mermaid graphs
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write"]
argument-hint: "[workspace_path]"
---

# Analyze Dependencies Command

Extract and analyze dependency relationships from subsystem catalog to identify structural issues and generate visualizations.

## Core Principle

**Dependencies reveal architecture. Circular dependencies and layer violations reveal architecture problems.**

This command transforms the implicit dependency information in subsystem catalogs into explicit analysis with actionable findings.

## Prerequisites

Before running this command:

1. **Subsystem catalog must exist** - `02-subsystem-catalog.md` in workspace
2. **Catalog must be validated** - Validation passed (check `temp/validation-catalog.md`)

If prerequisites missing:
```
Cannot analyze dependencies - subsystem catalog required.
Run /analyze-codebase first.
```

## Mandatory Workflow

### Step 1: Locate Workspace

```bash
# Find most recent analysis workspace
WORKSPACE=$(find docs -name "arch-analysis-*" -type d | sort -r | head -1)

# Verify catalog exists
[ -f "$WORKSPACE/02-subsystem-catalog.md" ] || echo "No catalog found"
```

### Step 2: Extract Dependencies

Parse `02-subsystem-catalog.md` to build dependency graph:

**For each subsystem entry, extract:**
- Subsystem name (from H2 heading)
- Inbound dependencies (from "Inbound:" line)
- Outbound dependencies (from "Outbound:" line)

**Build adjacency lists:**
```markdown
## Extracted Dependencies

| Subsystem | Depends On (Outbound) | Depended By (Inbound) |
|-----------|----------------------|----------------------|
| Auth | Database, Config | API Gateway, User Service |
| Database | Config | Auth, User Service, Payments |
| API Gateway | Auth, Logging | External Clients |
```

### Step 3: Detect Circular Dependencies

**Algorithm:** Depth-first search for back edges

```
For each subsystem S:
  Mark S as "visiting"
  For each dependency D of S:
    If D is "visiting" → CIRCULAR DEPENDENCY: S → ... → D → S
    If D is "unvisited" → recurse
  Mark S as "visited"
```

**Document cycles:**
```markdown
## Circular Dependencies Detected

### Cycle 1: Auth → User → Auth
- **Path:** Auth → User Service → Auth
- **Evidence:**
  - Auth depends on User Service (catalog line 45)
  - User Service depends on Auth (catalog line 89)
- **Severity:** HIGH - bidirectional dependency creates coupling

### Cycle 2: (none additional)
```

### Step 4: Detect Layer Violations

**Common layering patterns:**

| Layer | Typically Contains | Should Depend On |
|-------|-------------------|------------------|
| Presentation | UI, CLI, API handlers | Application, Domain |
| Application | Use cases, orchestration | Domain, Infrastructure |
| Domain | Business logic, entities | Nothing external |
| Infrastructure | Database, external APIs | Domain (interfaces only) |

**Violations to detect:**
- Infrastructure → Application (bypasses domain)
- Presentation → Infrastructure (bypasses application)
- Domain → Infrastructure (inverted dependency)

**Document violations:**
```markdown
## Layer Violations

| Source | Target | Violation Type | Evidence |
|--------|--------|----------------|----------|
| Database | User Service | Infra → Application | catalog:92 |
| API Handler | Database | Presentation → Infra | catalog:34 |

**Assessment:** [count] layer violations detected
```

**Note:** Layer detection requires understanding of codebase organization. If layers unclear from catalog, document:
```markdown
## Layer Analysis

**Layer structure:** Could not determine - subsystems don't map to clear layers
**Recommendation:** Manual review needed to establish layering convention
```

### Step 5: Check Bidirectional Consistency

**Rule:** If A lists B as outbound, B must list A as inbound.

```markdown
## Bidirectional Consistency

| Subsystem A | Claims | Subsystem B | B's Record | Status |
|-------------|--------|-------------|------------|--------|
| Auth | depends on → | Database | shows Auth inbound | OK |
| Payments | depends on → | Auth | missing Auth inbound | MISSING |

**Missing bidirectional references:** [count]
```

### Step 6: Calculate Dependency Metrics

**Afferent Coupling (Ca):** Number of subsystems that depend on this one (inbound count)
**Efferent Coupling (Ce):** Number of subsystems this depends on (outbound count)
**Instability (I):** Ce / (Ca + Ce) - 0 = stable, 1 = unstable

```markdown
## Dependency Metrics

| Subsystem | Ca (Inbound) | Ce (Outbound) | Instability | Assessment |
|-----------|--------------|---------------|-------------|------------|
| Database | 5 | 1 | 0.17 | Stable (foundation) |
| API Gateway | 1 | 4 | 0.80 | Unstable (orchestrator) |
| Auth | 3 | 2 | 0.40 | Balanced |

**High Coupling Alert:** Subsystems with Ca > 5 or Ce > 5 may be coupling hotspots
```

### Step 7: Generate Mermaid Dependency Graph

```markdown
## Dependency Graph

\`\`\`mermaid
graph TD
    subgraph Presentation
        API[API Gateway]
        CLI[CLI Handler]
    end

    subgraph Application
        Auth[Auth Service]
        User[User Service]
        Payments[Payments]
    end

    subgraph Infrastructure
        DB[(Database)]
        Cache[(Cache)]
        External[External API]
    end

    API --> Auth
    API --> User
    Auth --> DB
    Auth --> User
    User --> DB
    Payments --> Auth
    Payments --> External

    %% Circular dependency highlight
    linkStyle 3 stroke:red,stroke-width:2px

    %% Layer violation highlight
    linkStyle 5 stroke:orange,stroke-width:2px
\`\`\`

**Legend:**
- Red edges: Circular dependencies
- Orange edges: Layer violations
- Normal edges: Valid dependencies
```

## Output Contract (MANDATORY)

Write to `10-dependency-analysis.md` in workspace:

```markdown
# Dependency Analysis

**Workspace:** [path]
**Catalog Analyzed:** 02-subsystem-catalog.md
**Analysis Date:** YYYY-MM-DD
**Subsystems Analyzed:** [count]

## Executive Summary

- **Total Dependencies:** [count]
- **Circular Dependencies:** [count] - [HIGH/NONE]
- **Layer Violations:** [count] - [HIGH/MEDIUM/NONE]
- **Missing Bidirectional:** [count]
- **Coupling Hotspots:** [list subsystems with Ca > 5 or Ce > 5]

**Overall Health:** [HEALTHY / CONCERNS / CRITICAL]

## Dependency Matrix

| Subsystem | Depends On | Depended By |
|-----------|------------|-------------|
| [name] | [list] | [list] |

## Circular Dependencies

[Detailed cycle documentation or "None detected"]

## Layer Analysis

[Layer mapping and violations or "Layers not determinable"]

## Bidirectional Consistency

[Missing references or "All references consistent"]

## Coupling Metrics

| Subsystem | Ca | Ce | Instability | Notes |
|-----------|----|----|-------------|-------|

## Dependency Graph

[Mermaid diagram]

## Recommendations

### Immediate Actions (if CRITICAL):
1. [Specific cycle to break]
2. [Layer violation to fix]

### Improvement Opportunities:
1. [Reduce coupling in X]
2. [Add missing interface in Y]

## Limitations

- Analysis based on catalog only (not runtime dependencies)
- Layer detection: [Automated / Manual review needed]
- Dynamic dependencies not captured
```

## Handling Edge Cases

### Catalog Has No Dependencies Listed

```markdown
## Dependency Analysis

**Finding:** Subsystem catalog does not document dependencies.

**Recommendation:** Update catalog entries to include:
- Inbound: [Subsystems that depend on this]
- Outbound: [Subsystems this depends on]

Cannot perform dependency analysis without dependency data.
```

### Only 1-2 Subsystems

```markdown
## Dependency Analysis

**Finding:** Only [N] subsystems in catalog. Dependency analysis is trivial.

**Dependencies:** [list the simple relationships]

**Note:** For small systems, manual review is more appropriate than formal analysis.
```

## Anti-Patterns

**DON'T invent dependencies not in catalog:**
```
WRONG: "Based on common patterns, Auth probably depends on Database"
RIGHT: "Dependency data from catalog only. Auth → Database (catalog line 45)"
```

**DON'T skip cycle detection:**
```
WRONG: "Dependencies look reasonable, skipping cycle detection"
RIGHT: "Cycle detection algorithm applied to all [N] subsystems. Results: [...]"
```

**DON'T assume layers:**
```
WRONG: "Database is infrastructure, so any dependency on it from domain is a violation"
RIGHT: "Layer mapping: [documented reasoning]. Violations based on this mapping: [...]"
```

## Scope Boundaries

**This command covers:**
- Dependency extraction from catalog
- Circular dependency detection
- Layer violation detection (when layers determinable)
- Bidirectional consistency checking
- Coupling metrics calculation
- Mermaid graph generation

**Not covered:**
- Runtime dependency analysis (requires instrumentation)
- Import-level dependency analysis (use language-specific tools)
- Dependency version analysis (use package managers)
- Transitive dependency depth analysis

## Integration with Workflow

This command is invoked:
1. After subsystem catalog validation passes
2. When user selects deliverable option F (Full + Quality) or G (Comprehensive)
3. Manually via `/analyze-dependencies [workspace]`
4. As part of comprehensive architecture analysis

**Produces:** `10-dependency-analysis.md` in workspace

**Feeds into:**
- Architect handover (06-architect-handover.md) - dependency issues section
- Quality assessment (05-quality-assessment.md) - coupling analysis
