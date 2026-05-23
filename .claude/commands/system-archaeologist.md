---
description: Deep architectural analysis of existing codebases - coordinated subagent exploration producing subsystem catalogs, C4 diagrams, code-quality assessment, security surface, test-infrastructure analysis, dependency graphs, and incremental delta analysis under mandatory workspace + validation discipline; includes an ultralarge-tier per-module track for 100K+ LOC, >12-subsystem, plugin-registry, or oversized test/doc-corpus repos
---

# System Archaeologist Routing

**Archaeology is reconstruction discipline, not pattern-matching. The router is a coordinator; subagents read code; validators close gates. For forward solution design use `/solution-architect`; for architectural critique and prioritization use `/system-architect`.**

Use the `using-system-archaeologist` skill from the `axiom-system-archaeologist` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-system-archaeologist/skills/using-system-archaeologist/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Generate architecture documentation for an unfamiliar or legacy codebase
- Build a subsystem catalog, C4 diagrams, or a dependency graph
- Map a system's security surface, test infrastructure, or code-quality posture
- Run an incremental analysis on a moving codebase since the last archaeology pass
- Resume prior partial analysis - find existing workspace, decide continue / archive / salvage
- Analyze a codebase too large for single-pass orchestration (use the ultralarge track)

**Don't use** for: greenfield solution design (`/solution-architect`), architectural critique and prioritization (`/system-architect`), language-specific code review (`/python-engineering`, `/rust-engineering`, etc.), or security threat modeling (`/security-architect`).

## Sheets

- **analyzing-unknown-codebases** - subsystem catalog contract (8 required fields per entry, no extras)
- **partitioning-ultralarge-repos** - tier criteria, operator interview, partition manifest for >100K LOC repos
- **module-by-module-with-scribe** - ultralarge per-module review loop (DISPATCH → COLLECT → MERGE → VALIDATE → CHECKPOINT → ADVANCE)
- **findings-schema** - load-bearing YAML schema for ultralarge findings with reviewer/scribe self-validation
- **analyzing-dependencies** - dependency-graph extraction, layer-violation and cycle detection
- **analyzing-test-infrastructure** - test corpus archaeology
- **assessing-code-quality** - quality assessment contract
- **mapping-security-surface** - security boundary mapping (surface, not threat modeling)
- **generating-architecture-diagrams** - C4 diagram patterns at the right abstraction level
- **documenting-system-architecture** - final report contract
- **creating-architect-handover** - handover doc contract for downstream architects
- **validating-architecture-analysis** - validation contract, independence clause, retry limits
- **incremental-analysis** - delta-analysis path for moving codebases
- **deliverable-options** - Options A-G menu offered at Step 1.5
- **language-framework-patterns** - per-language idioms surfaced during discovery
- **specialist-integration** - cross-pack specialist handoff matrix

## Commands

- `/axiom-system-archaeologist:analyze-codebase` - standard small/large-tier flow with workspace + deliverable menu + subagent orchestration
- `/axiom-system-archaeologist:analyze-ultralarge` - ultralarge-tier manual partitioning and per-module review-with-scribe orchestration
- `/axiom-system-archaeologist:analyze-dependencies` - dependency graph from catalog with cycle / layer-violation / bidirectional-reference checks and Mermaid output
- `/axiom-system-archaeologist:generate-diagrams` - C4 diagrams from subsystem catalog at the correct abstraction level
- `/axiom-system-archaeologist:validate-analysis` - validation gate over analysis documents against output contracts

## Agents

- `codebase-explorer` - reads files and produces catalog entries under the subsystem contract; declines assessment-style asks (handed to `axiom-system-architect`)
- `analysis-validator` - independent validation of analysis documents against contracts; non-negotiable independence clause; max 2 re-validation retries
- `module-reviewer` - ultralarge-track per-module per-focus reviewer; deterministic sampling rules; cross-focus discipline
- `subsystem-scribe` - ultralarge-track mechanical-merge specialist; copy-don't-create; MIN-aggregation of confidence; reviewer-self-check + scribe-re-parse defense in depth

All four agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Forward solution design (greenfield) → `/solution-architect`
- Architectural critique, quality judgment, debt prioritization → `/system-architect`
- Language-specific code review and idiom checks → `/python-engineering`, `/rust-engineering`
- Security threat modeling (the surface is mapped here; the threats are modeled there) → `/security-architect`
- Documentation derivation across multi-document architecture sets → `/wiki-manager`
