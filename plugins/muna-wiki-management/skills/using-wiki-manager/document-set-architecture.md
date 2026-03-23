# Document Set Architecture

## Overview

Document Set Architecture is the structural foundation for managing multiple interrelated documents as a coherent collection. It answers: which documents exist, how they relate to each other, which are authoritative sources and which are derived, and how content flows from roots to derivatives.

Without explicit architecture, a multi-document set degrades predictably: terminology drifts between documents, derivative documents fall out of sync with their sources, readers encounter dead-end links, and nobody can trace which downstream documents are affected when a root changes.

This reference sheet covers five capabilities:

1. **Source-of-truth mapping** — declaring which documents are canonical roots and which are derivatives
2. **Depth tiering** — assigning documents to formal depth levels with self-sufficiency contracts
3. **Document manifest** — the single structural record of the entire set
4. **Multi-root reconciliation** — handling derivatives that draw from more than one root
5. **Conflict resolution** — what to do when root documents contradict each other

## When to Use

Load this guidance when:

- Standing up a new multi-document set (standards suite, policy framework, technical specification with audience-specific derivatives)
- Onboarding an existing collection of related documents that lack formal structure
- Adding or removing a document from an established set
- A derivative document needs to draw from two or more root documents
- You need to trace how a change in one document affects the rest of the set
- A consistency audit has flagged structural problems (orphan documents, missing derivation lineage, circular dependencies)

Do **not** use this for single-document problems. If you are writing or improving one document in isolation, use document-level writing guidance instead. Document Set Architecture activates when N >= 2 and the documents have declared relationships.

## Source-of-Truth Mapping

Every document set contains two kinds of documents:

- **Root documents** (canonical sources): These are authoritative. They originate content. Changes here propagate outward.
- **Derivative documents**: These faithfully represent root content at reduced depth or for different audiences. They do not originate content — they distill, translate, or extract from roots.

### What "Root" Means in Practice

A root document is the place where a claim, model, or recommendation is *first articulated with full evidence*. If someone asks "where does this come from?", the answer points to a root. A document set typically has 1-3 roots. A set with more than 5 roots is likely over-scoped — consider splitting it into multiple sets.

### What "Derivative" Means in Practice

A derivative exists because a specific audience needs the root's content in a different form. Every derivative must declare:

1. **Which root(s) it derives from** — the `derives_from` field
2. **How it transforms the content** — the `derivation_mode` field (distillation, translation, or extraction)
3. **For multi-root derivatives: how content from each root is integrated** — the `reconciliation_note`

### Derivation Modes

| Mode | What It Does | When to Use | Example |
|---|---|---|---|
| **Distillation** | Same argument, fewer words. Meaning preserved, detail reduced. | Executive summaries, briefing documents | A 120-page architecture document distilled into a 2-page executive brief |
| **Translation** | Same content, different vocabulary and frame of reference. | Cross-audience documents where the reader's domain language differs from the root's | A technical breaking change taxonomy translated into coverage-gap language for an SRE Lead audience |
| **Extraction** | A curated subset of the root's content, largely unchanged but selected. | Role-specific reference documents | A developer checklist extracting only the breaking changes relevant to code review from a comprehensive taxonomy |

### Anti-Pattern: Undeclared Derivation

A derivative document that does not declare its derivation lineage (no `derives_from`, no `derivation_mode`) is structurally invisible. When the root changes, nobody knows this derivative needs updating. When a consistency audit runs, this derivative is skipped. The document looks independent but is actually dependent — the worst of both worlds.

**Why it is harmful:** Undeclared derivatives are the primary cause of stale content in document sets. A reader encounters a derivative that contradicts its root and has no way to know which is authoritative. Over time, undeclared derivatives accumulate phantom content (claims not in any root) because there is no recipe constraining what they should contain.

**Fix:** Every derivative gets `derives_from` and `derivation_mode` in the manifest. No exceptions.

## Depth Tiers

Documents are assigned to one of three depth tiers. Each tier has a defined purpose, a target length range, and a self-sufficiency contract.

### Tier Definitions

| Tier | Name | Purpose | Typical Length | Audience Pattern |
|---|---|---|---|---|
| **Tier 0** | Executive | Decision documents. Enough to decide whether to act. | 1-3 pages | Senior leaders, sponsors, governance boards |
| **Tier 1** | Practitioner | Working documents for specific roles. Enough to act. | 8-15 pages | SRE Leads, architects, team leads, reviewers |
| **Tier 2** | Reference | Full technical analysis. Complete evidence and reasoning. | 30-150+ pages | Technical specialists, implementers, auditors |

### The Self-Sufficiency Contract

This is the most important structural rule in document set architecture:

> **A document at Tier N must be actionable without requiring the reader to open any Tier N+1 document.**

Concretely:

- A Tier 0 executive brief must let a senior leader decide whether to invest further — without opening the Tier 2 architecture document.
- A Tier 1 ops assessment must let an SRE Lead evaluate coverage gaps and prioritize remediation — without opening the Tier 2 architecture document.
- Links from Tier 0 or Tier 1 to Tier 2 are acceptable for *optional depth* ("for the full evidence base, see..."). Links to Tier 2 that are *required for comprehension* violate the contract.

### How to Test Self-Sufficiency

1. Give the document to someone matching the target persona.
2. Remove all access to higher-tier documents.
3. Ask: "Can you do your job with just this document?"
4. If the answer is "I need to check the architecture document first" — the self-sufficiency contract is violated.

### Anti-Pattern: Tier 1 That Depends on Tier 2

A Tier 1 practitioner document that contains phrases like "the full list of breaking changes is in the architecture document — see Section 3.2" when the practitioner needs that list to do their job. The document has substituted a link for content.

**Why it is harmful:** The derivative fails its reason for existing. If the reader must open the root anyway, the derivative adds a navigation step without adding value. Readers learn to skip directly to Tier 2, and the Tier 1 document becomes shelf-ware.

**Fix:** The derivative must contain the content the reader needs, synthesized at the appropriate depth. The link to Tier 2 can remain as an optional "for supporting evidence" pointer, but the actionable content must be inline.

## Document Manifest

The manifest is the single source of truth about the *structure* of a document set. It declares every document, its tier, its role (root or derivative), its derivation lineage, and its target audience. All other structural artifacts (registries, reading paths, coverage matrices) depend on the manifest being accurate.

The manifest is a markdown file with YAML frontmatter, stored in the project repository alongside the documents it describes.

### Complete Manifest Example

This is a full manifest for a five-document standards suite. Two root documents (an architecture document and an API reference) and three derivatives at different tiers for different audiences:

```yaml
---
set_name: "API Platform Migration Suite"
set_version: "1.0.0"
---

# Document Manifest

## Root Documents

- id: architecture
  title: "Full Architecture Document"
  path: design/architecture.md
  tier: 2
  role: root
  audience: [technical-lead, reviewer, architect]

- id: api-reference
  title: "API Reference"
  path: reference/api-reference.md
  tier: 2
  role: root
  audience: [sdk-developer, reviewer, migration-lead]

## Derivative Documents

- id: executive-brief
  title: "Executive Brief"
  path: design/executive-brief.md
  tier: 0
  role: derivative
  derives_from: [architecture]
  audience: [vp-eng, executive]
  derivation_mode: distillation

- id: ops-assessment
  title: "Ops Assessment"
  path: operations/ops-assessment.md
  tier: 1
  role: derivative
  derives_from: [architecture]
  audience: [sre-lead, platform-architect]
  derivation_mode: translation

- id: recommendations
  title: "Recommendations"
  path: plan/recommendations.md
  tier: 1
  role: derivative
  derives_from: [architecture, api-reference]
  audience: [sre-lead, engineering-manager]
  derivation_mode: extraction
  reconciliation_note: "Draws migration safeguard proposals from architecture §6 and verification model from api-reference Part I. Where both roots address the same control, architecture provides the impact justification and api-reference provides the technical mechanism."
```

### Manifest Field Reference

**Required for every document:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique short identifier used in cross-references (e.g., `architecture`, `executive-brief`) |
| `title` | string | Human-readable document title |
| `path` | string | Relative file path from the project root |
| `tier` | integer (0, 1, 2) | Depth tier assignment |
| `role` | enum: `root` or `derivative` | Whether this document originates or derives content |
| `audience` | list of strings | Persona IDs that this document serves (see Persona Registry below) |

**Required for derivatives (in addition to the above):**

| Field | Type | Description |
|---|---|---|
| `derives_from` | list of strings | IDs of root documents this derivative draws from |
| `derivation_mode` | enum: `distillation`, `translation`, `extraction` | How content is transformed from the root |

**Required for multi-root derivatives (in addition to the above):**

| Field | Type | Description |
|---|---|---|
| `reconciliation_note` | string | Explains how content from each root is integrated. Must specify which root provides what, and how overlapping content is handled. |

### Anti-Pattern: Manifest Without Reconciliation Notes

A multi-root derivative that lists `derives_from: [architecture, api-reference]` but has no `reconciliation_note`. The manifest knows *that* two roots contribute but not *how*.

**Why it is harmful:** Without a reconciliation note, anyone maintaining the derivative must reverse-engineer the integration strategy every time. When Root A changes, the maintainer cannot tell whether that change affects this derivative's content or only the portion drawn from Root B. The derivative becomes fragile — changes are applied inconsistently because the integration logic is undocumented.

**Fix:** Every multi-root derivative gets a `reconciliation_note` that answers three questions: (1) What does each root contribute? (2) Where do the roots overlap? (3) When both roots address the same topic, which root takes precedence and why?

## Creating a Manifest for a New Set

This is the step-by-step procedure for building a manifest from scratch. Follow it in order.

### Step 1: Inventory All Documents

List every document that will be part of the set. For each, record:

- A short identifier (lowercase, hyphenated: `executive-brief`, not `Executive Brief`)
- The full title
- The file path relative to the project root

**Worked example:** You have a project with these files:

```
docs/
├── design/
│   ├── architecture.md     # 120-page architecture document
│   └── executive-brief.md  # 2-page summary for leadership
├── operations/
│   └── ops-assessment.md   # 12-page assessment for SRE Leads
├── reference/
│   └── api-reference.md    # 80-page API reference
└── plan/
    └── recommendations.md  # 10-page action recommendations
```

Your inventory:

| ID | Title | Path |
|---|---|---|
| architecture | Full Architecture Document | design/architecture.md |
| executive-brief | Executive Brief | design/executive-brief.md |
| ops-assessment | Ops Assessment | operations/ops-assessment.md |
| api-reference | API Reference | reference/api-reference.md |
| recommendations | Recommendations | plan/recommendations.md |

### Step 2: Classify Root vs. Derivative

For each document, ask: "Does this document originate its content, or does it represent content from another document in this set?"

- If it originates content with full evidence and reasoning → **root**
- If it presents another document's content at a different depth or for a different audience → **derivative**

In the worked example:

- `architecture` → root (originates the breaking change taxonomy and impact analysis)
- `api-reference` → root (originates the verification model and technical controls)
- `executive-brief` → derivative (summarizes the architecture document for executives)
- `ops-assessment` → derivative (translates the architecture document for SRE Leads)
- `recommendations` → derivative (extracts action items from both roots)

### Step 3: Assign Depth Tiers

Apply the tier definitions:

- `architecture` → Tier 2 (full technical reference, 120 pages)
- `api-reference` → Tier 2 (full technical specification, 80 pages)
- `executive-brief` → Tier 0 (2-page decision document)
- `ops-assessment` → Tier 1 (12-page working document for a specific role)
- `recommendations` → Tier 1 (10-page working document for specific roles)

### Step 4: Map Derivation Lineage

For each derivative, identify which root(s) it draws from and how:

| Derivative | Derives From | Mode | Rationale |
|---|---|---|---|
| executive-brief | architecture | distillation | Same argument compressed to 2 pages |
| ops-assessment | architecture | translation | Breaking changes re-expressed as coverage gaps |
| recommendations | architecture, api-reference | extraction | Action items curated from both roots |

### Step 5: Write Reconciliation Notes for Multi-Root Derivatives

`recommendations` draws from both `architecture` and `api-reference`. The reconciliation note must explain the integration:

> "Draws migration safeguard proposals from architecture §6 and verification model from api-reference Part I. Where both roots address the same control, architecture provides the impact justification and api-reference provides the technical mechanism."

### Step 6: Identify Target Audiences

A **Persona Registry** is a list of the distinct roles your document set serves. Each persona has an `id` (short identifier), `name` (human-readable role), `task` (what they do with the documents — assess, implement, govern, or understand), `depth_tolerance` (executive, practitioner, or technical), and `domain_vocabulary` (terms they know without definition). The full registry format and construction guidance is in [reading-path-design.md](reading-path-design.md); here you only need the persona IDs for manifest audience fields.

If no Persona Registry exists yet, draft one now — for each role that will read any document in the set, create an ID (e.g., `sre-lead`, `vp-eng`, `technical-lead`):


| Document | Audience |
|---|---|
| architecture | technical-lead, reviewer, architect |
| api-reference | sdk-developer, reviewer, migration-lead |
| executive-brief | vp-eng, executive |
| ops-assessment | sre-lead, platform-architect |
| recommendations | sre-lead, engineering-manager |

### Step 7: Assemble the Manifest

Combine all the above into the manifest format shown in the Complete Manifest Example section. Write the YAML frontmatter (`set_name`, `set_version`), then list root documents, then derivative documents.

### Step 8: Validate

Run the validation checklist (next section) before committing the manifest.

## Validating a Manifest

Use this checklist after creating or modifying a manifest. Every item must pass.

### Structural Completeness

- [ ] Every document in the project's document directory appears in the manifest
- [ ] Every manifest entry has all required fields (`id`, `title`, `path`, `tier`, `role`, `audience`)
- [ ] Every derivative has `derives_from` and `derivation_mode`
- [ ] Every multi-root derivative has a `reconciliation_note`
- [ ] All `id` values are unique within the manifest
- [ ] All `path` values point to files that actually exist

### Tier Consistency

- [ ] No root document is assigned to Tier 0 (roots are Tier 2; Tier 0 and Tier 1 documents are always derivatives or standalone summaries)
- [ ] Derivatives are at a lower tier number than at least one of their roots (a Tier 2 derivative of a Tier 2 root is suspect — why does it exist?)
- [ ] The set has at least one document at Tier 0 or Tier 1 (a set of only Tier 2 documents has no audience accessibility)

### Derivation Integrity

- [ ] Every `derives_from` reference points to an `id` that exists in the manifest
- [ ] No circular derivation (A derives from B derives from A)
- [ ] No derivative derives from another derivative (derivatives derive from roots only; if you need a derivative of a derivative, the intermediate document should probably be a root)
- [ ] Each `derivation_mode` is one of: `distillation`, `translation`, `extraction`

### Audience Coverage

- [ ] Every persona ID referenced in `audience` fields corresponds to a persona in the Persona Registry (or is flagged for addition)
- [ ] No persona appears in zero documents' audience lists (every registered persona has at least one document)
- [ ] No document has an empty audience list

### Graph Integrity

- [ ] The derivation graph is a DAG (directed acyclic graph) — no cycles
- [ ] Every document is reachable from at least one reading path — a sequenced list of documents designed for a specific persona to read in order — (no orphan documents)
- [ ] The `reconciliation_note` for every multi-root derivative specifies which root provides what content and how overlaps are handled

## Multi-Root Reconciliation Patterns

When a derivative draws from two or more root documents, the content integration follows one of three patterns. Each pattern has different complexity and different maintenance implications.

### Pattern 1: Partition

**What it is:** Different sections of the derivative draw from different roots with no overlap. Each section has exactly one source root.

**When to use it:** When the roots cover genuinely separate topics that the derivative brings together for a specific audience.

**Worked example:**

A "Recommendations" document draws from two roots:
- Root A (`architecture`): contains a taxonomy of 15 breaking changes with impact analysis
- Root B (`api-reference`): contains a four-tier verification model with technical controls

The Recommendations document is structured as:

| Derivative Section | Source Root | Content |
|---|---|---|
| §1 Priority Breaking Changes | architecture §3 | The 5 highest-impact breaking changes extracted from the architecture's taxonomy |
| §2 Impact Justification | architecture §4 | The impact analysis supporting why these 5 are prioritized |
| §3 Verification Mechanisms | api-reference Part I | The technical controls from the verification model that address the prioritized breaking changes |
| §4 Implementation Sequence | api-reference Part II | The phased rollout plan from the api-reference |

**Key property:** If Root A changes, only §1 and §2 need review. If Root B changes, only §3 and §4 need review. Maintenance impact is cleanly separated.

**Reconciliation note for this pattern:**

> "Partition reconciliation. §1-2 draw exclusively from architecture (breaking change taxonomy and impact analysis). §3-4 draw exclusively from api-reference (verification model and implementation). No section integrates content from both roots."

### Pattern 2: Primary/Supplementary

**What it is:** One root provides the core narrative structure. The other root provides supporting detail that enriches specific sections. The derivative follows the primary root's organization.

**When to use it:** When one root is clearly the "main source" and the other adds context, evidence, or technical detail that strengthens the derivative's argument.

**Worked example:**

An "SRE Lead Impact Assessment" document uses the architecture as its primary root and the api-reference as supplementary:

| Derivative Section | Primary Source | Supplementary Source | How Supplementary Adds Value |
|---|---|---|---|
| §1 Threat Landscape | architecture §2 (threat overview) | — | No supplementary content needed |
| §2 Breaking Change Analysis | architecture §3 (breaking change taxonomy) | api-reference Part I §3 (detection capabilities) | Architecture provides the breaking changes; api-reference adds which verification tier can detect each change |
| §3 Coverage Gap Assessment | architecture §4 (impact analysis) | api-reference Part I §4 (control mappings) | Architecture provides the impact framing; api-reference provides concrete control identifiers |
| §4 Recommended Actions | architecture §6 (migration safeguards) | api-reference Part II (implementation guidance) | Architecture provides the policy recommendations; api-reference provides the technical implementation steps |

**Key property:** The derivative's section structure mirrors the primary root. The supplementary root never dictates structure — it enriches content within the primary root's framework. When the primary root changes structure, the derivative's structure changes. When the supplementary root changes, only the enrichment content needs updating.

**Reconciliation note for this pattern:**

> "Primary/supplementary reconciliation. Architecture is primary — the derivative follows the architecture's argumentative structure (threat landscape → breaking changes → impact → recommendations). Api-reference is supplementary — it provides verification-tier detection capabilities (§2), control identifiers (§3), and implementation steps (§4). The architecture's structure governs; the api-reference enriches."

### Pattern 3: Synthesis

**What it is:** Content from both roots must be woven together to form an argument that neither root makes independently. The derivative creates new analytical value by combining perspectives.

**When to use it:** When the derivative's purpose requires integrating insights that exist separately in different roots. This is the most complex pattern and requires explicit justification for why the synthesis is faithful to both roots.

**Worked example:**

A "Combined Impact and Verification Assessment" document synthesizes insights from both roots to answer a question neither root answers alone: "For each verification tier, what is the residual impact if only that tier is implemented?"

| Derivative Section | From Architecture | From API Reference | Synthesis |
|---|---|---|---|
| §1 Methodology | §2 (impact scoring model) | Part I §1 (verification tier definitions) | Combines the impact scoring framework with the tier definitions to create an evaluation matrix neither root contains |
| §2 Tier 1 Residual Impact | §3.1-3.4 (breaking changes 1-4) | Part I §2.1 (Tier 1 capabilities) | Maps which of the 15 breaking changes Tier 1 detects, calculates residual impact for undetected changes using the architecture's severity ratings |
| §3 Tier 2 Residual Impact | §3.1-3.8 (breaking changes 1-8) | Part I §2.2 (Tier 2 capabilities) | Same analysis for Tier 2, showing incremental impact reduction |
| §4 Investment Recommendation | §5 (cost-benefit analysis) | Part II §3 (implementation costs) | Merges the architecture's benefit analysis with the api-reference's cost data to produce ROI projections per tier |

**Key property:** No single root can generate this derivative's content alone. The synthesis creates analytical value. This also means maintenance is more complex — a change in *either* root potentially affects *every* section, because the integration logic spans both.

**Reconciliation note for this pattern:**

> "Synthesis reconciliation. Each section weaves content from both roots to answer: 'What is the residual impact at each verification tier?' Architecture provides the breaking change taxonomy and severity ratings; api-reference provides the tier-level detection capabilities and implementation costs. The residual-impact-per-tier analysis does not exist in either root — it is a faithful synthesis of both. Changes to either root's breaking changes, severity ratings, tier definitions, or detection capabilities require re-evaluation of the corresponding residual impact calculation."

### Anti-Pattern: Unlabeled Reconciliation

A multi-root derivative where the author mixed partition, primary/supplementary, and synthesis patterns across different sections without declaring which pattern each section uses.

**Why it is harmful:** During maintenance, when Root A changes, the maintainer cannot tell which sections are affected without reading every section in full. Partition sections are easy — check if the source root matches. Primary/supplementary sections are moderate — check primary root changes, spot-check supplementary enrichments. Synthesis sections are hard — any root change might affect them. Without labels, every section is treated as the hardest case, making maintenance slow and error-prone.

**Fix:** In the reconciliation note or in a section-level derivation map, explicitly label each section's reconciliation pattern.

## Multi-Root Conflict Resolution

The reconciliation patterns above assume roots are complementary — they cover different aspects of the same subject without contradicting each other. When roots *do* contradict each other, a conflict resolution protocol activates.

### Conflict Types

| Type | Definition | Example | Who Resolves |
|---|---|---|---|
| **Terminological** | Same concept, different words in different roots | Root A calls them "verification tiers," Root B calls them "verification levels" | Derivative author chooses one term and notes the alias |
| **Factual** | Different claims about the same thing | Root A says "15 breaking changes," Root B says "the 14 identified compatibility issues" | Human root owner — requires determining which root is correct |
| **Structural** | Incompatible models or frameworks | Root A organizes controls by risk category, Root B organizes by implementation phase | Human root owner — requires deciding how the derivative reconciles the two models |

### The Resolution Protocol

**Step 1: Detect.** During derivation or consistency audit, flag any claim or term where two roots disagree. Record both versions and their exact locations.

**Step 2: Classify.** Determine whether the conflict is terminological, factual, or structural using the definitions above.

**Step 3: Resolve or Escalate.**

- **Terminological conflicts:** The derivative author resolves these. Choose the more precise or audience-appropriate term. Document the alias so that future audits do not re-flag the same conflict.
- **Factual and structural conflicts:** These are *always* escalated to the human root owner. The resolution may require updating one or both roots, not just the derivative. Claude must not resolve these autonomously.

**Step 4: Document.** Record the resolution in two places: (1) the derivative's `reconciliation_note` in the manifest, and (2) the claim registry — a project file that tracks factual claims propagating across documents (see [cross-document-consistency.md](cross-document-consistency.md) for the full format) — with a `conflict_resolution` field.

### Worked Example: Detecting and Resolving a Conflict

**Scenario:** You are building the Recommendations derivative, which draws from both `architecture` and `api-reference`. During derivation, you encounter this conflict:

- **architecture §3.2** states: "The taxonomy identifies 15 distinct breaking changes across four categories."
- **api-reference Part I §1.3** states: "The verification model addresses the 14 identified compatibility issues."

**Step 1 — Detect:** The claim about the count of breaking changes differs: 15 vs. 14. The terminology also differs: "breaking changes" vs. "compatibility issues."

**Step 2 — Classify:**

- The count discrepancy (15 vs. 14) is a **factual conflict**. One root says 15, the other says 14. These cannot both be right about the same set.
- The naming difference ("breaking changes" vs. "compatibility issues") is a **terminological conflict**. Both terms plausibly refer to the same concept.

**Step 3 — Resolve or Escalate:**

- **Terminological:** The derivative author chooses "breaking changes" (the architecture's term) because the architecture is the originating source for the taxonomy. The api-reference's use of "compatibility issues" is noted as an alias.
- **Factual:** This is escalated to the human root owner with the following flag:

```
Conflict detected:
  Type: factual
  Claim: count of breaking changes/compatibility issues
  Root A (architecture §3.2): "15 distinct breaking changes"
  Root B (api-reference Part I §1.3): "14 identified compatibility issues"
  Question for root owner: Is the api-reference excluding one breaking change
    intentionally (e.g., one change is out of scope for verification), or is
    this an error in one of the roots?
  Impact: The Recommendations document (§1) needs to state a specific count.
    Cannot proceed until this is resolved.
```

**Step 4 — Document (after human resolves):**

Suppose the human determines that the api-reference intentionally excludes one breaking change (BC-15: "silent deprecation") because it is not detectable by automated tooling. The resolution is documented:

In the **manifest reconciliation note** for `recommendations`:

> "...Where both roots address breaking change counts, the architecture's count of 15 is canonical for the full taxonomy. The api-reference addresses 14 of 15 because BC-15 (silent deprecation) is excluded from the verification model as non-detectable by automated tooling. The Recommendations document uses the api-reference's count of 14 when discussing verifiable controls and the architecture's count of 15 when discussing the full impact landscape."

In the **claim registry**:

```yaml
- id: breaking-change-count
  claim: "The taxonomy identifies 15 distinct breaking changes"
  canonical_source: architecture §3.2
  conflict_resolution:
    conflicting_source: api-reference Part I §1.3
    conflicting_claim: "14 identified compatibility issues"
    resolution: "Api-reference intentionally excludes BC-15 (silent deprecation) as non-detectable by automated tooling. Architecture count of 15 is canonical for full taxonomy; api-reference count of 14 is canonical for verifiable subset."
    resolved_by: "[root owner name]"
    resolved_date: "2026-03-15"
  propagation:
    - executive-brief §2 (distillation) — uses "15 breaking changes"
    - ops-assessment §1.1 (translation) — uses "15 breaking changes"
    - recommendations §1 (extraction) — uses "14 verifiable" and "15 total" depending on context
```

### Anti-Pattern: Silently Picking a Side

When two roots conflict, the derivative author quietly uses one root's number and ignores the other. No flag, no documentation, no reconciliation note.

**Why it is harmful:** The conflict is invisible. Future auditors checking the derivative against Root B will flag it as a stale claim. The derivative's maintenance history gives no indication that the discrepancy was known and resolved. The same conflict gets re-discovered and re-investigated every audit cycle, wasting effort and eroding trust in the audit process.

**Fix:** Every factual or structural conflict gets documented in the reconciliation note and claim registry, even when the resolution is straightforward. The five minutes spent documenting saves hours of re-investigation later.

### Anti-Pattern: Claude Resolving Factual Conflicts Autonomously

When Claude encounters a factual conflict between roots, it "helpfully" resolves the conflict by choosing the more recent document, the larger number, or the more detailed source — without flagging it to the human.

**Why it is harmful:** Claude lacks the domain context to know *why* roots disagree. The api-reference's count of 14 might be intentional (scoping decision), accidental (typo), or reflective of a later revision. Autonomous resolution risks propagating the wrong answer into the derivative and giving it an authoritative gloss. The human root owner is the only person who can make this call.

**Fix:** Factual and structural conflicts are always escalated. Claude's job is to detect, classify, and present — not to resolve.

## Derivation Graph

The derivation graph is a directed acyclic graph (DAG) that maps section-level derivation lineage. It is more granular than the manifest's document-level `derives_from` — the graph maps which *sections* of a derivative come from which *sections* of a root.

### What the Graph Looks Like

For the standards suite example:

```
architecture §2 (threats)     ──→  executive-brief §1 (threat summary)
architecture §3 (taxonomy)    ──→  executive-brief §2 (key findings)
architecture §3 (taxonomy)    ──→  ops-assessment §1 (breaking change analysis)
architecture §4 (impact)      ──→  ops-assessment §2 (coverage gaps)
architecture §6 (safeguards)  ──→  recommendations §1 (policy proposals)
api-reference Part I          ──→  recommendations §2 (verification mechanisms)
```

### Where to Store the Graph

The derivation graph is a project file stored alongside the manifest. The simplest format is a markdown file (e.g., `derivation-graph.md`) with one line per edge, using the arrow notation shown above. Each entry must specify: source document + section → target document + section. For small sets (under 10 documents), the graph can be a section within the manifest file itself. For larger sets, keep it as a separate file to avoid manifest bloat.

### Why Section-Level Granularity Matters

Document-level lineage (`recommendations derives from architecture and api-reference`) tells you a change to either root might affect the Recommendations document. Section-level lineage tells you a change to `architecture §3` affects `executive-brief §2` and `ops-assessment §1` but not `recommendations §2`. This turns a "review the whole derivative" task into a "review these two sections" task.

### Anti-Pattern: Circular Derivation

Document A derives from Document B, which derives from Document A. Or a longer cycle: A → B → C → A.

**Why it is harmful:** Circular derivation means there is no authoritative source. When a claim changes, the propagation trace loops forever. In practice, circular derivation usually signals that one of the documents should be reclassified as a root, or that two documents are actually one document split artificially.

**Fix:** The derivation graph must be a DAG. If you detect a cycle, identify which document in the cycle is the actual source of truth and reclassify the others as derivatives of it.

### Anti-Pattern: Orphan Documents

A document that exists in the manifest but is not reachable from any reading path and is not the source for any derivative.

**Why it is harmful:** Orphan documents are not maintained because they are not visible. They accumulate stale content. Readers who stumble on them cannot tell if the content is current. The document occupies space in the set without serving any audience.

**Fix:** Either connect the orphan to a reading path (it has an audience — find it), make it a root that feeds a derivative (it has content worth propagating), or remove it from the set.

## Onboarding an Existing Document Set

When documents already exist but lack formal architecture, use this sequence to bootstrap structure without requiring exhaustive up-front work.

### Step 1: Inventory

Read all documents. For each one, record the id, title, path, and classify as root or derivative. Assign depth tiers. Output: a draft manifest with `id`, `title`, `path`, `tier`, `role` — but skip derivation lineage for now.

### Step 2: Discover Implicit Derivation

For each document classified as derivative, identify which root sections its content most likely derives from. Look for:

- Shared claims (the same number or assertion appears in both)
- Parallel structure (sections that cover the same topics in the same order)
- Explicit cross-references ("as discussed in the architecture document")

Output: `derives_from` and `derivation_mode` fields added to the manifest. Draft `reconciliation_note` for any multi-root derivatives.

### Step 3: Bootstrap Minimum Viable Registries

The document set uses three registries as its consistency infrastructure: a **Terminology Registry** (tracks key terms and their permitted variants across depth tiers), a **Claim Registry** (tracks factual claims that propagate across multiple documents), and a **Persona Registry** (tracks the roles the set serves and their reading paths). Full registry formats are defined in [cross-document-consistency.md](cross-document-consistency.md) and [reading-path-design.md](reading-path-design.md). At onboarding, start with minimum viable versions using these granularity heuristics:

- **Terminology Registry:** Start with terms that appear in 3+ documents or that have visible inconsistencies across documents. Target 15-30 terms for a 5-document set. If you pass 60 terms, you are building a glossary, not managing consistency.
- **Claim Registry:** Start with quantitative claims (counts, percentages, named lists with specific lengths) and structural claims (models, taxonomies, frameworks) that propagate across 2+ documents. Target 10-20 claims for a 5-document set.
- **Persona Registry:** Extract from existing "who should read this" statements, navigation guidance, or explicit audience declarations. If none exist, infer from document tiers and stated audiences. Target 8-15 personas for a standards suite.

### Step 4: Discover Implicit Reading Paths

Check for existing navigation aids — tables of contents, "start here" pointers, role-based guides. Formalize what exists. Flag personas that have no clear path through the set.

### Step 5: Baseline Audit

Run a lightweight consistency check: terminology spot-check (not a full sweep), link integrity validation, and a deferral scan on derivatives. This establishes the current quality baseline. Do not demand perfection — the registries will grow through use, and each maintenance cycle adds entries as they become relevant.

## Summary of All Anti-Patterns

| Anti-Pattern | Where It Applies | Core Harm |
|---|---|---|
| Undeclared derivation lineage | Source-of-truth mapping | Makes change propagation invisible; derivatives go stale silently |
| Tier 1 depends on Tier 2 for comprehension | Depth tiers / self-sufficiency | Derivative fails its reason for existing; readers skip it |
| Manifest without reconciliation notes | Multi-root derivatives | Integration logic is undocumented; maintenance is slow and inconsistent |
| Circular derivation | Derivation graph | No authoritative source; change propagation loops |
| Orphan document | Derivation graph | Unmaintained content confuses readers who find it |
| Silently picking a side in a conflict | Conflict resolution | Invisible resolution gets re-investigated every audit cycle |
| Claude resolving factual conflicts autonomously | Conflict resolution | Wrong resolution propagated with authoritative gloss |
| Unlabeled reconciliation patterns | Multi-root reconciliation | Every section treated as worst-case complexity during maintenance |
| Derivative of a derivative | Derivation integrity | Creates a telephone-game chain; content degrades with each hop |

## Cross-References

These reference sheets cover sibling competencies in the wiki management skill set. All files are in the same directory as this document.

- `reading-path-design.md` — Persona Registry, coverage matrices, time-based paths, and stepping-stone validation
- `content-derivation.md` — Derivation recipes, the self-sufficiency test, derivation integrity checks, and the anti-laziness discipline
- `cross-document-consistency.md` — Terminology Registry, Claim Registry, structural conventions, link integrity, and the consistency audit process
- `document-evolution.md` — Change classification, impact tracing, git integration, external dependency tracking, and version coherence
- `document-governance.md` — Ownership model, LLM-as-steward trust model, change workflows, quality gates, and review triggers
