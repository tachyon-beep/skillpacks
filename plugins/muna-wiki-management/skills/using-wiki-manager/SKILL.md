---
name: using-wiki-manager
description: "Use when related documents need explicit source lineage, audience-specific derivatives, claim consistency, reading paths, or controlled change propagation."
---

# Document Set Management

Manage relationships and persistent source contracts across documents. A document
set needs more than individually clear prose, but small collections need not
acquire a complete governance system before a useful edit.

## Establish the source map

Read existing documents, manifests, registries and Git changes. Identify canonical
roots, derivatives, audiences and unresolved source conflicts. Preserve existing
metadata formats where practical. Bootstrap only the relationships needed for the
current decision; grow registries through use, not numerical quotas.

A useful manifest records document ID/path, audience/purpose, source role,
lineage and version. For consequential repeated claims, record canonical wording,
source location and affected derivatives. Section-level edges improve impact
tracing when document-level lineage is too coarse.

## Choose the workflow

| Task | Procedure | Optional reference |
|---|---|---|
| Onboard/restructure | Inventory → source map → audience paths → ownership | [document-set-architecture.md](document-set-architecture.md), [reading-path-design.md](reading-path-design.md) |
| Derive a document | Select relevant source content → preserve meaning → adapt to task → test self-sufficiency | [content-derivation.md](content-derivation.md) |
| Root or standard changed | Classify actual diff → trace affected claims/sections → update → verify | [document-evolution.md](document-evolution.md) |
| Consistency/audit | Check terms/claims/anchors against sources → inspect paths and derivation → triage | [cross-document-consistency.md](cross-document-consistency.md) |
| Ownership/acceptance | Identify actual authority, review triggers and evidence | [document-governance.md](document-governance.md) |

References are beside this file. Load relevant sections only. Commands
`onboard-docset`, `derive-content`, `propagate-change`, and `audit-docset` are optional
entry points to these procedures, not prerequisites.

## Derivation and change contracts

- A derivative contains the information its reader needs to act. Links may offer
  optional depth; they should not replace essential content.
- Preserve qualifiers, counts, conditions and source meaning. Label and source
  any new inference or recommendation separately from the canonical claim.
- Resolve conflicting roots explicitly; never silently choose the convenient one.
- Classify changes by the actual diff: cosmetic, clarification, substantive or
  structural. Trace consequential changes into affected derivatives and links.
- Use existing link/schema validators and inspect the changed claims and paths.
  Record coverage and unresolved conflicts; a registry is a tool, not proof that
  its entries are true or complete.
- Preserve scope and authorization. Human review follows actual publication/risk
  requirements and existing approval, not an unconditional pause at every edit.

## Deliver

Return changed documents/metadata or a prioritized defect/impact list with source
locations, affected readers and verification. Distinguish source fidelity,
synthetic self-sufficiency probes and actual reader/stakeholder acceptance.
Use `muna-technical-writer` for document-level clarity and `muna-panel-review` only
for optional hypothesis generation. Neither a model steward nor simulated
personas can self-approve real organizational acceptance.
