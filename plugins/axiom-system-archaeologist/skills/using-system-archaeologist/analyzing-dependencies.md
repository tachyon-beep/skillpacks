
# Analyzing Dependencies

## Purpose

Extract and analyze dependency relationships from subsystem catalogs to identify structural issues: circular dependencies, layer violations, coupling hotspots, and missing bidirectional references. Produces visualizations and metrics for architect handoff.

## When to Use

- After subsystem catalog completion and validation
- User requests dependency analysis during architecture review
- Preparing for architect consultation on coupling issues
- Deliverable menu option F (Full + Quality) or G (Comprehensive) selected
- User mentions: "dependencies", "coupling", "circular", "layer violations"

## Execution

**Run via command:** `/analyze-dependencies [workspace_path]`

The command handles the full workflow. This reference documents the methodology and output contract.

## Core Principle: Extract, Don't Infer

**Archaeologist extracts declared dependencies. Architect assesses improvement strategy.**

```
ARCHAEOLOGIST: "Circular dependency detected: Auth → User → Auth (catalog lines 45, 89)"
ARCHITECT: "Break cycle by extracting shared interface to Core module"

ARCHAEOLOGIST: "Database has Ca=7 (high inbound coupling)"
ARCHITECT: "Introduce repository abstraction to reduce direct coupling"
```

**Your role:** Document WHAT the catalog says. NOT how to fix coupling.

## Prerequisite Check (MANDATORY)

Before running dependency analysis:

1. **Verify catalog exists:** `02-subsystem-catalog.md` in workspace
2. **Verify catalog validated:** Check `temp/validation-catalog.md` shows APPROVED
3. **If prerequisites missing:** "Cannot analyze dependencies - subsystem catalog required. Run /analyze-codebase first."

## Methodology

### Step 1: Extract Dependencies from Catalog

Parse each subsystem entry for:
- Subsystem name (H2 heading)
- Outbound dependencies (from "Outbound:" line)
- Inbound dependencies (from "Inbound:" line)

Build adjacency lists for analysis.

### Step 2: Detect Circular Dependencies

**Algorithm:** Depth-first search for back edges

A circular dependency exists when: A → B → ... → A

**Severity levels:**
- **Direct cycle (A → B → A):** HIGH - tight coupling
- **Transitive cycle (A → B → C → A):** MEDIUM - hidden coupling
- **Large cycle (4+ nodes):** Often indicates architectural confusion

### Step 3: Detect Layer Violations

**Standard layering:**

| Layer | Should Depend On | Should NOT Depend On |
|-------|------------------|---------------------|
| Presentation | Application | Infrastructure directly |
| Application | Domain, Infrastructure interfaces | Nothing above |
| Domain | Nothing (pure) | Infrastructure, Application |
| Infrastructure | Domain interfaces | Application, Presentation |

**Common violations:**
- Infrastructure → Application (bypasses domain)
- Presentation → Infrastructure (bypasses application)
- Domain → Infrastructure (inverted dependency)

**Note:** Layer detection requires clear subsystem categorization. If layers aren't determinable from catalog, document this limitation.

### Step 4: Check Bidirectional Consistency

**Rule:** If A lists B as outbound dependency, B must list A as inbound.

Missing bidirectional references indicate:
- Incomplete catalog entries
- Asymmetric understanding of relationships
- Potential hidden dependencies

### Step 5: Calculate Coupling Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Afferent Coupling (Ca)** | Count of inbound deps | Responsibility (high = many dependents) |
| **Efferent Coupling (Ce)** | Count of outbound deps | Dependency (high = many dependencies) |
| **Instability (I)** | Ce / (Ca + Ce) | 0 = stable, 1 = unstable |

**Hotspot thresholds:**
- Ca > 5: High responsibility - changes affect many
- Ce > 5: High dependency - many things can break this
- Both high: Critical coupling hotspot

### Step 6: Generate Dependency Graph

Produce Mermaid diagram with:
- Subsystem nodes grouped by layer (if determinable)
- Edges showing dependency direction
- Visual highlighting for issues:
  - Red: Circular dependencies
  - Orange: Layer violations

## Output Contract (MANDATORY)

Write to `10-dependency-analysis.md` in workspace:

```markdown
# Dependency Analysis

**Workspace:** [path]
**Catalog Analyzed:** 02-subsystem-catalog.md
**Analysis Date:** YYYY-MM-DD
**Subsystems Analyzed:** [count]
**Confidence:** [High/Medium/Low] - [evidence]

## Executive Summary

- **Total Dependencies:** [count]
- **Circular Dependencies:** [count] - [CRITICAL/HIGH/NONE]
- **Layer Violations:** [count] - [HIGH/MEDIUM/NONE/NOT DETERMINABLE]
- **Missing Bidirectional:** [count]
- **Coupling Hotspots:** [list subsystems with Ca > 5 or Ce > 5]

**Overall Health:** [HEALTHY / CONCERNS / CRITICAL]

## Dependency Matrix

| Subsystem | Depends On (Outbound) | Depended By (Inbound) |
|-----------|----------------------|----------------------|
| [name] | [list] | [list] |

## Circular Dependencies

### Cycle 1: [Name]
- **Path:** [A → B → C → A]
- **Evidence:** [catalog line references]
- **Type:** [Direct / Transitive]

(Or: "None detected - verified via DFS on [N] subsystems")

## Layer Analysis

### Layer Mapping
| Subsystem | Assigned Layer | Reasoning |
|-----------|---------------|-----------|
| [name] | [Presentation/Application/Domain/Infrastructure] | [evidence] |

(Or: "Layers not determinable from catalog - subsystems don't map to clear architectural layers")

### Layer Violations
| Source | Target | Violation | Evidence |
|--------|--------|-----------|----------|
| [A] | [B] | [Type] | [catalog line] |

## Bidirectional Consistency

| Subsystem A | Claims Dependency | Subsystem B | B's Record | Status |
|-------------|------------------|-------------|------------|--------|
| [A] | → | [B] | [inbound list] | [OK/MISSING] |

**Missing references:** [count] - [list]

## Coupling Metrics

| Subsystem | Ca (In) | Ce (Out) | Instability | Assessment |
|-----------|---------|----------|-------------|------------|
| [name] | [N] | [N] | [0.XX] | [Stable/Balanced/Unstable/Hotspot] |

**Hotspots (Ca > 5 or Ce > 5):** [list]

## Dependency Graph

```mermaid
graph TD
    %% Layer grouping
    subgraph Presentation
        ...
    end

    %% Dependencies
    A --> B

    %% Issue highlighting
    linkStyle N stroke:red  %% Circular
    linkStyle M stroke:orange  %% Layer violation
```

**Legend:**
- Red edges: Circular dependencies
- Orange edges: Layer violations

## Architect Handoff

**Recommend axiom-system-architect for:**

| Issue | Recommended Action | Specific Concern |
|-------|-------------------|------------------|
| Circular dependency | Break cycle | [A → B → A path] |
| High coupling | Reduce Ca/Ce | [Subsystem with Ca > 5] |
| Layer violations | Restructure | [Specific violations] |

## Limitations

- **Scope:** Catalog-based analysis only (not runtime/import-level)
- **Layer detection:** [Automated / Manual assignment / Not possible]
- **Dynamic dependencies:** Not captured
- **Transitive depth:** Not analyzed
```

## Common Rationalizations (STOP SIGNALS)

| Rationalization | Reality |
|-----------------|---------|
| "Dependencies look reasonable, skip cycle detection" | Always run DFS. Hidden cycles are common. |
| "Small system doesn't need coupling metrics" | Metrics take 2 minutes. Do them anyway. |
| "I can see the layers" | Document layer mapping explicitly with evidence. |
| "Catalog doesn't have good dependency data" | Document this limitation. Don't guess. |
| "I'll infer the missing dependencies" | Inference is speculation. Extract only what's declared. |

## Anti-Patterns

**DON'T invent dependencies:**
```
WRONG: "Auth probably depends on Database based on common patterns"
RIGHT: "Auth → Database per catalog line 45. Other dependencies not declared."
```

**DON'T skip the algorithm:**
```
WRONG: "Visually inspected for cycles, looks fine"
RIGHT: "DFS cycle detection on 12 subsystems: [results]"
```

**DON'T prescribe fixes:**
```
WRONG: "Should break Auth→User cycle by extracting interface"
RIGHT: "Circular dependency: Auth → User → Auth. Refer to axiom-system-architect."
```

## Integration with Workflow

Dependency analysis is invoked:
1. After subsystem catalog validation passes
2. When deliverable option F or G selected
3. Via `/analyze-dependencies [workspace]` command
4. Before architect handover if coupling concerns exist

**Pipeline:** Catalog → Validation → Dependency Analysis → Architect Handoff

## Cross-Plugin Handoff

When dependency analysis is complete:

```
Dependency analysis complete. Produced 10-dependency-analysis.md with:
- [N] total dependencies mapped
- [N] circular dependencies ([severity])
- [N] layer violations
- [N] coupling hotspots

For architectural improvement planning, recommend:
- axiom-system-architect:assessing-architecture-quality for overall assessment
- axiom-system-architect:prioritizing-improvements for fix sequencing
```

**DO NOT attempt to break cycles or restructure layers yourself.**

## Specialist Subagent Integration

For complex codebases (10+ subsystems), consider spawning specialist analysis:

**Language-specific dependency tools:**
- Python: Import analysis via AST parsing
- JavaScript: Module graph via bundler output
- Go: `go mod graph` output parsing

**When to spawn specialists:**
- Catalog dependencies are incomplete or uncertain
- Need import-level granularity beyond subsystem level
- Verifying catalog accuracy against actual imports

Document any specialist findings that contradict or extend catalog data.
