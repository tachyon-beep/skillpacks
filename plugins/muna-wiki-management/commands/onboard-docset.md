---
description: Bootstrap wiki management for an existing document set - inventory, discover derivation, build minimum viable registries
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[directory_containing_documents]"
---

# Onboard Document Set

You are bootstrapping wiki management for an existing collection of documents. These documents exist but lack formal structure — no manifest, no registries, no declared derivation lineage. Your job is to reverse-engineer the implicit structure and create the minimum viable management artifacts.

## Core Principle

**Build registries FROM the documents, not before them.** The chicken-and-egg problem (skills need registries, registries need labor) is solved by bootstrapping minimum viable versions that grow through use.

## Mandatory Workflow

### Step 1: Inventory

Read all documents in the provided directory. For each document, determine:

1. **Title and path** — the document's filename and location
2. **Role** — Is this a **root** (canonical, authoritative, complete analysis) or a **derivative** (summary, translation, or extract for a specific audience)?
3. **Depth tier** — **Tier 0** (executive, 1-5 pages), **Tier 1** (practitioner, 5-20 pages for a specific role), or **Tier 2** (reference, 20+ pages of complete technical analysis)
4. **Audience** — Who is this written for? List role names as short IDs (e.g., `ciso`, `ses`, `developer`)

**Ask the user** if any classification is ambiguous. Root vs. derivative is the most important distinction — get it right.

Output: A table listing each document with its path, role, tier, and audience.

### Step 2: Discover Implicit Derivation

For each derivative document, identify which root(s) it derives from. Look for:

- **Shared claims** — the same number, assertion, or named framework appears in both
- **Parallel structure** — sections covering the same topics in the same order
- **Explicit cross-references** — phrases like "as discussed in the paper," "see the specification"
- **Vocabulary alignment** — a derivative using simplified versions of a root's terminology

For each derivative, determine:
- `derives_from`: which root document(s)
- `derivation_mode`: **distillation** (same argument compressed), **translation** (re-expressed for different vocabulary), or **extraction** (subset curated for a role)
- For multi-root derivatives: draft a `reconciliation_note` explaining how content from each root integrates

Output: Updated inventory table with derivation lineage.

### Step 3: Build the Document Manifest

Assemble the inventory and derivation data into a manifest file. Use this format:

```yaml
---
set_name: "[Name of the document set]"
set_version: "0.1.0"
---

# Document Manifest

## Root Documents

- id: [short-id]
  title: "[Full Title]"
  path: [relative/path/to/file.md]
  tier: 2
  role: root
  audience: [persona-id-1, persona-id-2]

## Derivative Documents

- id: [short-id]
  title: "[Full Title]"
  path: [relative/path/to/file.md]
  tier: [0 or 1]
  role: derivative
  derives_from: [root-id]
  audience: [persona-id-1]
  derivation_mode: [distillation | translation | extraction]
```

Save to the document set's root directory as `manifest.md`.

**Ask the user** to review the manifest before proceeding.

### Step 4: Bootstrap Minimum Viable Registries

Build starter registries using granularity heuristics — do NOT try to be exhaustive.

**Terminology Registry** (target: 15-30 terms):
1. Scan all documents for key terms that appear in 3+ documents
2. Flag terms with visible inconsistencies (different names for the same concept)
3. For each qualifying term: find the canonical definition in the root, note tier variants across documents
4. Save as `terminology-registry.yaml`

**Claim Registry** (target: 10-20 claims):
1. Scan for quantitative claims (counts, percentages, named lists with specific lengths)
2. Scan for structural claims (named models, taxonomies, frameworks referenced across documents)
3. For each qualifying claim: identify canonical source, trace propagation across derivatives
4. Save as `claim-registry.yaml`

**Persona Registry** (target: 8-15 personas):
1. Extract from existing "who should read this" statements, navigation guidance, or explicit audience declarations
2. If none exist, infer from the manifest's audience fields
3. For each persona: assign id, name, task, depth_tolerance, domain_vocabulary, entry_point
4. Save as `persona-registry.yaml`

### Step 5: Discover Implicit Reading Paths

Check for existing navigation aids:
- Tables of contents, "start here" pointers, role-based guides
- "Reading order" or "how to use these documents" sections
- Cross-document links that form a natural sequence

For each persona in the registry:
1. Identify their entry point (usually the document in their audience list at the lowest tier)
2. Trace the most natural reading sequence through their relevant documents
3. Record the path

Flag personas that have no clear path — these are coverage gaps to address.

### Step 6: Baseline Audit

Run a lightweight quality check to establish the current state (not to achieve perfection):

1. **Terminology spot-check**: Pick 5 terms from the registry. Verify they're used consistently across 2-3 documents each.
2. **Link integrity**: Check all cross-document links resolve.
3. **Deferral scan**: In each derivative, count phrases like "see the paper," "refer to the specification." Classify each as acceptable (optional depth) or lazy (substitutes for needed content).

Output: A brief audit report listing defects found, classified by priority (P1/P2/P3).

### Step 7: Present Results

Present to the user:
1. The manifest (already reviewed in Step 3)
2. Registry summaries (counts and notable entries)
3. Reading paths by persona
4. Baseline audit results
5. Recommended next steps (which defects to fix first, which registries to expand)

**Ask the user** if they want to proceed with fixing identified defects or if the baseline is sufficient for now.
