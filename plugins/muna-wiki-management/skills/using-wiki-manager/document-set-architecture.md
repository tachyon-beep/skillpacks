# Document Set Architecture

Represent canonical sources, audience-specific derivatives and their relationships
explicitly enough to maintain the collection. Reuse existing metadata conventions;
a small edit does not require a new governance system.

## Source roles and tiers

A root owns a claim, decision or model with its evidence. A derivative selects,
distills or translates that material for a reader task. Multiple roots may own
different claims; count alone does not make a collection over-scoped.

Audience/depth tiers are useful local labels, not fixed page budgets. A reader
must have essential task information in their document/path; links may provide
optional depth. Preserve enough rationale/evidence to support the reader's decision.

## Minimal manifest

Store structured metadata in the project's existing format, versioned with content.
For example, a YAML file can express:

```yaml
set_name: migration-suite
set_version: 1.0.0
documents:
  - id: architecture
    path: design/architecture.md
    role: root
    audience: [implementer, reviewer]
  - id: operator-guide
    path: operations/guide.md
    role: derivative
    audience: [operator]
    derives_from: [architecture]
    derivation_mode: translation
```

Optional fields include depth tier, owner, version, source sections, reconciliation
notes and review triggers. Check ID/path uniqueness, actual file existence and
source references. Metadata is not proof that the sources are accurate.

## Section lineage

When document-level lineage is too coarse, record source section → derivative
section edges. Keep the graph acyclic where the model is derivation; links and
citations can have other shapes and should not be confused with derivation edges.
Track repeated consequential claims with canonical source and propagation list.

## Conflicting roots

Identify the exact incompatible claims, source versions and affected derivatives.
Resolve using actual authority, scope and evidence; distinguish temporal/version
changes from genuine contradictions. Record the resolution or unresolved conflict
in source metadata and affected artifacts. Do not silently choose whichever root
makes drafting easier.

## Bootstrap and validate

Inventory documents and actual audience/tasks, identify roots/derivatives and
consequential repeated claims, then record only relationships needed for maintenance.
No minimum number of terms, claims, personas or roots is required.

Check paths/anchors and trace a representative affected task to completion.
Source fidelity can be checked mechanically and by review; human usability needs
actual readers. See [content-derivation.md](content-derivation.md),
[cross-document-consistency.md](cross-document-consistency.md),
[document-evolution.md](document-evolution.md).
