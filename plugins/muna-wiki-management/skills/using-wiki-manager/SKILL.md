---
name: using-wiki-manager
description: Manage complex document sets as wikis - architecture, reading paths, derivation, consistency, evolution, and governance for multi-document collections
---

# Using Wiki Manager

## Overview

This meta-skill routes you to the right document set management skills based on your task. Load this skill when you need to create, organize, maintain, or audit a collection of interrelated documents.

**Core Principle**: Managing a document *set* is a different discipline from writing a document. Individual document quality is necessary but not sufficient — the set must be consistent, navigable, self-sufficient at each depth tier, and maintainable over time.

## When to Use

Load this skill when:
- Managing multiple related documents as a collection (standards suites, policy frameworks, technical documentation sets)
- User mentions: "wiki", "document set", "reading paths", "cross-reference", "consistency audit", "derivative", "propagate changes"
- Writing derivative documents from root/canonical sources
- Checking if documents are self-sufficient for their audience
- Propagating changes across a document collection
- Onboarding an existing collection of documents that lacks formal structure

**Don't use for**: Writing a single standalone document (use muna-technical-writer instead). Panel-testing documents with simulated readers (use muna-panel-review instead).

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-wiki-manager/SKILL.md`

Reference sheets like `content-derivation.md` are at:
  `skills/using-wiki-manager/content-derivation.md`

NOT at:
  `skills/content-derivation.md` ← WRONG PATH

When you see a link like `[content-derivation.md](content-derivation.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Entry Pattern

### Pattern 0: Onboarding an Existing Document Set

**Symptoms**: "I have documents but no structure", "How do I organize these?", "Bootstrap", "Get started with existing docs", "We have a standards suite but no manifest"

**Use command**: `/onboard-docset` if available, or follow this sequence manually.

**Route to**: Skills in order:
1. [document-set-architecture.md](document-set-architecture.md) — inventory documents, classify root/derivative, assign tiers, build manifest
2. [reading-path-design.md](reading-path-design.md) — discover implicit personas and paths
3. [cross-document-consistency.md](cross-document-consistency.md) — bootstrap minimum viable registries (terminology, claims)
4. [document-governance.md](document-governance.md) — assign ownership

The onboarding workflow builds registries *from* the documents rather than requiring them upfront. Start with minimum viable versions and grow through use.

---

### Pattern 1: Creating a New Document Set

**Symptoms**: "Starting from scratch", "New documentation suite", "Greenfield", "Planning a multi-document standards effort"

**Route to**: Skills in sequential order:
1. [document-set-architecture.md](document-set-architecture.md) — define structure, manifest, depth tiers
2. [reading-path-design.md](reading-path-design.md) — define personas and reading paths
3. [content-derivation.md](content-derivation.md) — set up derivation recipes for each derivative section
4. [cross-document-consistency.md](cross-document-consistency.md) — establish terminology and claim registries
5. [document-governance.md](document-governance.md) — assign ownership, define review triggers

[document-evolution.md](document-evolution.md) activates after the first version of the set is published.

---

### Pattern 2: Day-to-Day Maintenance

**Symptoms**: Specific maintenance tasks on an existing, structured document set.

| User intent | Route to |
|---|---|
| Add or restructure a document | [document-set-architecture.md](document-set-architecture.md) |
| Add a persona, check coverage, fix a dead-end path | [reading-path-design.md](reading-path-design.md) |
| Write or update a derivative document | [content-derivation.md](content-derivation.md) |
| Fix terminology, verify claims, check links | [cross-document-consistency.md](cross-document-consistency.md) |
| Respond to upstream change (root document or external standard) | [document-evolution.md](document-evolution.md) |
| Set up review workflows, assign ownership, run quality gates | [document-governance.md](document-governance.md) |

---

### Pattern 3: Change Propagation

**Symptoms**: "I updated the paper", "Root document changed", "Propagate changes", "Section X of the spec was rewritten"

**Use command**: `/propagate-change` if available, or chain manually:

1. **Evolution** ([document-evolution.md](document-evolution.md)) — classify the change, run impact trace → output: work list of affected derivative sections
2. **Derivation** ([content-derivation.md](content-derivation.md)) — for each affected section, re-apply the derivation recipe → output: updated derivative content
3. **Consistency** ([cross-document-consistency.md](cross-document-consistency.md)) — verify claims and terminology match registries → output: defect list
4. **Governance** ([document-governance.md](document-governance.md)) — surface quality gate results to human for review

This is the workhorse workflow — more targeted than a full audit, avoids overkill.

---

### Pattern 4: Full Audit

**Symptoms**: "Check everything", "Audit", "Health check", "Pre-publish review", "Quarterly review"

**Use command**: `/audit-docset` if available, or run in sequence:

1. **Consistency audit** ([cross-document-consistency.md](cross-document-consistency.md)) — terminology, claims, structural conventions, links
2. **Derivation integrity** ([content-derivation.md](content-derivation.md)) — recipe compliance, self-sufficiency, deferral audit
3. **Path completeness** ([reading-path-design.md](reading-path-design.md)) — coverage matrix, stepping stone validation
4. **Link integrity** ([cross-document-consistency.md](cross-document-consistency.md)) — forward links, bidirectional awareness, orphan sections, anchor stability

Output: unified defect list, triaged by priority (P1 wrong info → P2 degraded navigation → P3 friction).

---

## Cross-Faction Integration

### With muna-technical-writer

Use both when you need set-level structure AND individual document quality:
- **Wiki Manager** handles: manifest, derivation recipes, consistency registries, reading paths, governance
- **Technical Writer** handles: document structure (ADRs, APIs, runbooks), clarity and style, diagram conventions

**Example**: "Write a new CISO Assessment for our standards suite" → Load wiki-manager for the derivation recipe and self-sufficiency contract, load technical-writer for the document's internal structure and clarity.

### With muna-panel-review

Wiki Manager ensures the set is structurally sound *before* testing. Panel Review tests how documents land with audiences *after* structure is in place.

The wiki-manager Persona Registry can feed panel-review's persona definitions: export the persona's task, depth tolerance, and domain vocabulary as the scenario framing for a panel-review session.

---

## Reference Sheet Catalog

### Core Skills

| Skill | Covers |
|---|---|
| [document-set-architecture.md](document-set-architecture.md) | Manifest, derivation graph, depth tiers, multi-root reconciliation, conflict resolution |
| [reading-path-design.md](reading-path-design.md) | Persona registry, path completeness, coverage matrix, time-based paths, stepping stone validation |
| [content-derivation.md](content-derivation.md) | Derivation modes, recipes, self-sufficiency test, deferral audit — the anti-laziness discipline |
| [cross-document-consistency.md](cross-document-consistency.md) | Terminology/claim registries, granularity heuristics, link integrity, consistency audit, triage |
| [document-evolution.md](document-evolution.md) | Change classification, impact traces, git integration, external dependencies, deprecation |
| [document-governance.md](document-governance.md) | Ownership model, LLM-as-steward trust model, quality gates, review triggers |

---

## Design Principles

1. **Self-sufficiency over linking** — every document must stand alone for its audience
2. **Explicit over implicit** — derivation lineage, terminology choices, and ownership are declared, not assumed
3. **Auditable** — every quality claim has a concrete, repeatable test
4. **Two-root aware** — handles multiple canonical sources, including conflict resolution when roots disagree
5. **Anti-laziness by design** — derivation recipes make faithful synthesis the path of least resistance for LLMs
6. **Governance ties it together** — the other five skills define what to check; governance defines who checks and when
7. **Bootstrappable** — adoptable incrementally on existing document sets via minimum viable registries
8. **Registries stay lean** — granularity heuristics prevent registries from becoming maintenance burdens
9. **Human-in-the-loop for trust** — when Claude is the steward, quality gate results are surfaced, not silently self-approved
10. **Git-native** — manifests, registries, and documents are versioned together; change classification leverages diffs

---

## Common Mistakes

### Loading All Skills for Every Task
**Wrong**: Load all 6 reference sheets for every document set task.
**Right**: Use the routing table above. A derivative update needs content-derivation.md and maybe governance. A terminology fix needs cross-document-consistency.md only.

### Skipping the Manifest
**Wrong**: Jump straight to writing derivatives without declaring the document set structure.
**Right**: Start with document-set-architecture.md. The manifest is the foundation everything else references.

### Treating Derivation as Summarization
**Wrong**: "Summarize the paper for executives" → write a shorter version.
**Right**: "Derive an executive brief from the paper using distillation mode" → follow the derivation recipe, run the self-sufficiency test, audit deferrals. Derivation is a discipline, not a writing task.

### Auditing Without Registries
**Wrong**: Run a consistency audit by reading all documents and checking for problems.
**Right**: Build the terminology and claim registries first (even minimum viable versions), then audit against them. Without registries, you have no objective standard to audit against.

---

## Decision Tree

```
Starting a document set task?
├─ Do the documents have a manifest?
│  ├─ No → Pattern 0: Onboard (document-set-architecture first)
│  └─ Yes
│     ├─ What are you doing?
│     │  ├─ Writing/updating a derivative → content-derivation
│     │  ├─ Root document changed → Pattern 3: Change propagation
│     │  ├─ Adding a persona/fixing paths → reading-path-design
│     │  ├─ Fixing terminology/claims/links → cross-document-consistency
│     │  ├─ Setting up ownership/review process → document-governance
│     │  └─ Full health check → Pattern 4: Full audit
│     └─ Creating a new document set? → Pattern 1: Sequential setup
```
