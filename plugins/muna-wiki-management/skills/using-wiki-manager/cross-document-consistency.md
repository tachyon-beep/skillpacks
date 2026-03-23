# Cross-Document Consistency

## Overview

Cross-Document Consistency ensures all documents in a set tell the same story using the same language, even when those documents operate at different depth tiers for different audiences. A Tier 0 executive brief and a Tier 2 reference paper must agree on facts, use compatible terminology, and link to each other reliably — despite being written at vastly different levels of detail.

Without active consistency management, a five-document set drifts within two to three editing cycles. The architecture doc says "15 breaking changes," the executive brief says "over a dozen risk categories," and the ops assessment says "14 identified compatibility issues." All three describe the same taxonomy. None agree. Readers who consult multiple documents lose trust in the entire set.

This reference sheet covers: Terminology Registry, Claim Registry, Granularity Heuristics, Structural Conventions, Link Integrity, the five-step Consistency Audit Process, and Audit Triage and Prioritization.

## When to Use

Load this guidance when:

- Running a consistency audit across a document set (scheduled or triggered by a root change)
- Building or updating a terminology registry for a new or existing document set
- Building or updating a claim registry to track factual assertions across documents
- Investigating a defect flagged by another competency (a derivation integrity check found a stale claim, a reading path audit found conflicting terminology)
- A root document has been updated and you need to verify derivatives still agree
- Establishing structural conventions for a new document set
- Validating link integrity before publishing

Do **not** use this for single-document editing. Cross-Document Consistency activates when you need to verify agreement *between* documents.

## Key Terminology

This discipline references structural concepts from sibling reference sheets. Minimal definitions so you can follow all procedures without leaving this document:

- **Depth Tiers:** Documents are classified into three levels. **Tier 0** = executive-depth (2-page decision documents for senior leaders). **Tier 1** = practitioner-depth (8-15 page working documents for specific roles like SRE Leads or developers). **Tier 2** = reference-depth (full technical analyses, specifications — the root documents that contain the complete argument).
- **Root Document:** A canonical, authoritative document in the set (always Tier 2). Root documents are the source of truth — they originate content. Changes to roots propagate to derivatives.
- **Derivative Document:** A document derived from one or more roots at reduced depth (Tier 0 or Tier 1). Derivatives synthesize, translate, or extract root content for a specific audience. They must be faithful to their roots and self-sufficient for their target persona.
- **Document Manifest:** A structured project file declaring every document in the set — its ID, title, path, tier, role (root or derivative), audience, and derivation lineage. When this document refers to "document IDs," they come from the manifest.
- **Persona Registry:** A project file listing each role the document set serves, with fields for ID, name, task, depth tolerance, domain vocabulary, and entry point. Personas determine who needs what content.
- **Reading Path:** A sequenced list of documents designed for a specific persona to read in order. Each path has an entry point, intermediate steps, and a terminal state where the persona can complete their task.
- **Coverage Matrix:** A personas-by-topics grid showing which document section covers each topic for each persona. Empty cells are explicit gaps.

## Terminology Registry

A maintained YAML file stored in the project repository. It tracks key terms across multiple documents, declares the canonical definition, specifies tier-appropriate simplifications, and lists commonly confused terms.

### Registry Format

```yaml
terms:
  - term: "verification tier"
    canonical_definition: "One of four levels of enforcement granularity in the semantic enforcement model, ranging from Tier 1 (syntax-level checks) through Tier 4 (architectural validation). Each tier subsumes the capabilities of all lower tiers."
    canonical_source: api-reference Part I §2
    tier_variants:
      tier_2: "verification tier"
      tier_1: "verification level"
      tier_0: "level of checking"
    prohibited_conflations:
      - "trust level — trust describes a relationship between parties; verification tiers describe verification capability, not trust"
      - "classification — classification is an information sensitivity label; verification tiers are verification mechanism categories"

  - term: "breaking change"
    canonical_definition: "A specific, catalogued pattern of API compatibility defect. The taxonomy identifies 15 distinct breaking changes across four categories: schema changes, authentication changes, rate limit changes, and deprecation patterns."
    canonical_source: architecture §3
    tier_variants:
      tier_2: "breaking change"
      tier_1: "breaking change" or "compatibility issue"
      tier_0: "risk category"
    prohibited_conflations:
      - "authentication change — an authentication change is one category of breaking change (auth), not a synonym for the full set"
      - "bug — a bug is a specific instance; a breaking change is a catalogued pattern"
      - "threat — a threat is an external actor or event; a breaking change is an internal compatibility issue"

  - term: "self-sufficiency contract"
    canonical_definition: "The structural rule that a document at Tier N must be actionable without requiring the reader to open any Tier N+1 document. Links to higher-tier documents for optional depth are permitted; dependence on higher-tier documents for comprehension is not."
    canonical_source: document-set-architecture (manifest conventions)
    tier_variants:
      tier_2: "self-sufficiency contract"
      tier_1: "self-sufficiency requirement"
      tier_0: "standalone readability"
    prohibited_conflations:
      - "completeness — self-sufficiency does not mean the document contains everything; it means the document contains enough for its audience to act"
      - "independence — documents in a set are not independent; they are self-sufficient at their tier while remaining part of a navigable collection"

```

### Terminology Registry Field Reference

| Field | Required | Description |
|---|---|---|
| `term` | Yes | The canonical name for this concept |
| `canonical_definition` | Yes | The full, precise definition. Must be self-contained — a reader who sees only this definition understands the term. |
| `canonical_source` | Yes | The document and section where this term is authoritatively defined |
| `tier_variants` | Yes | How the term may be expressed at each depth tier. Tier 2 uses the canonical form. Tier 1 may use a simpler synonym. Tier 0 may use a broader, less precise term. |
| `prohibited_conflations` | Yes | Terms that readers commonly confuse with this one, each with an explanation of why the conflation is wrong |

### How Tier Variants Work

Tier variants are controlled simplifications, not free synonyms:

- **Tier 2** always uses the canonical term exactly as defined.
- **Tier 1** may use a simpler term if the audience's domain vocabulary does not include the canonical term, but the simpler term must map unambiguously to the canonical definition.
- **Tier 0** may use a broader category term that loses precision but does not introduce error.

A tier variant fails when it introduces ambiguity or maps to multiple canonical terms. If "risk category" could mean either "breaking change" or "threat category" depending on context, it fails as a Tier 0 variant for either term.

### Anti-Pattern: Unregistered Term Drift

A term appears in different documents with different names, but no registry entry exists. The architecture doc says "verification tier," the ops assessment says "verification level," the executive brief says "control layer." Each document appears internally consistent; the inconsistency is invisible until a reader consults two documents in the same session and cannot tell whether these are the same concept or different ones.

**Fix:** Register the term. Declare "verification tier" as canonical, list "verification level" as the Tier 1 variant, and evaluate "control layer" as either an acceptable Tier 0 variant or a prohibited conflation.

### Anti-Pattern: Registry as Exhaustive Glossary

The terminology registry contains 80+ terms, including obvious domain vocabulary, terms in only one document, and terms with no inconsistency risk. A bloated registry is not maintained — the signal-to-noise ratio drops until the registry is ignored entirely.

**Fix:** Apply the granularity heuristic. Target 15-40 terms for a five-document set. If you exceed 60, strip back to terms with genuine cross-document inconsistency risk.

## Claim Registry

Tracks key factual claims from their canonical source through every document that repeats or adapts them. When a claim changes, the propagation list identifies every derivative section needing updates.

### Registry Format

```yaml
claims:
  - id: breaking-change-count
    claim: "The taxonomy identifies 15 distinct breaking changes across four categories"
    canonical_source: architecture §3.2
    propagation:
      - document: executive-brief
        section: "§2 Key Findings"
        mode: distillation
        adapted_form: "fifteen categories of API compatibility risk"
      - document: ops-assessment
        section: "§1.1 Impact Landscape"
        mode: translation
        adapted_form: "15 catalogued compatibility issues requiring migration coverage"
      - document: recommendations
        section: "§2 Scope"
        mode: extraction
        adapted_form: "15 breaking changes (14 detectable; BC-15 excluded from automated detection)"

  - id: verification-tier-count
    claim: "The enforcement model defines four verification tiers"
    canonical_source: api-reference Part I §2
    propagation:
      - document: recommendations
        section: "§4.1 Enforcement Architecture"
        mode: extraction
        adapted_form: "four enforcement tiers, each subsuming the capabilities of lower tiers"
      - document: architecture
        section: "§7.2 Enforcement Model Integration"
        mode: cross-reference from other root
        adapted_form: "the companion api-reference's four-tier enforcement model"

  - id: safeguard-count
    claim: "Six migration safeguards are proposed to address gaps in API compatibility assurance"
    canonical_source: architecture §6.1
    propagation:
      - document: executive-brief
        section: "§3 Recommendations"
        mode: distillation
        adapted_form: "six new safeguards proposed for the migration framework"
      - document: recommendations
        section: "§3 Migration Safeguards"
        mode: extraction
        adapted_form: "six migration safeguards, each mapped to one or more breaking changes"
      - document: ops-assessment
        section: "§2.4 Coverage Gap Analysis"
        mode: translation
        adapted_form: "six proposed migration safeguards addressing the identified coverage gaps"
```

### Claim Registry Field Reference

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Unique short identifier for the claim, used in defect records and cross-references |
| `claim` | Yes | The canonical phrasing of the claim, exactly as it appears in the canonical source |
| `canonical_source` | Yes | The document and section where this claim originates |
| `propagation` | Yes | List of every location where this claim appears in other documents |
| `propagation[].document` | Yes | The document ID from the manifest |
| `propagation[].section` | Yes | The specific section within that document |
| `propagation[].mode` | Yes | How the claim is adapted: `distillation`, `translation`, `extraction`, or `cross-reference from other root` |
| `propagation[].adapted_form` | Yes | The actual phrasing used in the derivative, so auditors can compare without opening the document |
| `conflict_resolution` | No | Present only when two roots disagree on this claim. Contains `conflicting_source`, `conflicting_claim`, `resolution`, `resolved_by`, and `resolved_date`. |

### How Propagation Tracking Works

When the canonical source changes, the propagation list becomes the work list: (1) identify the changed claim by `id` or diff, (2) check each propagation entry's `adapted_form` against the new canonical claim, (3) update stale adapted forms in both the derivative and the registry, (4) if the claim was removed, flag every propagation entry for review.

### Anti-Pattern: Unregistered Quantitative Claims

A number appears in multiple documents but is not in the claim registry. When the architecture doc's taxonomy is revised from 15 to 16 breaking changes, nobody knows which other documents need updating. Conflicting numbers have no charitable interpretation — a reader who sees "13" in one document and "12" in another immediately questions the reliability of the entire set.

**Fix:** Register every quantitative claim that appears in 2+ documents. Five minutes adding a registry entry saves hours tracking down every instance after a number changes.

### Anti-Pattern: Adapted Forms Not Recorded

The claim registry lists propagation locations but not the actual phrasing used in each derivative. Without recorded adapted forms, auditing takes 5-10x longer because the auditor must open every derivative, find the sentence, and compare manually.

**Fix:** Every propagation entry includes `adapted_form`. Update this field whenever the derivative is updated.

## Granularity Heuristics

Registries are only useful if maintained. Each registry has inclusion criteria and a target size range to prevent unbounded growth.

### Terminology Registry Heuristic

**Register a term if any of these conditions is true:**

- The term appears in 3 or more documents in the set
- The term has visible inconsistency across documents (different names for the same concept in different documents)
- The term has a prohibited conflation that readers commonly make (a term that sounds similar but means something different)

**Target size:** 15-40 terms for a 5-document set. A 10-document set might reach 50-60. If you pass 60 terms for any set size, you are cataloguing a glossary, not managing consistency. Strip the registry back to terms that have actual cross-document inconsistency risk.

**What to exclude:** Terms that are standard domain vocabulary understood identically by all audiences (e.g., "risk" in a security context), terms that appear in only one document (no cross-document consistency risk), and terms that have no plausible conflation (unambiguous technical jargon).

### Claim Registry Heuristic

**Register a claim if any of these conditions is true:**

- The claim is quantitative — it contains a count, a percentage, a named list with a specific length, or a measurable threshold
- The claim is structural — it defines a model, taxonomy, framework, or classification scheme that other documents reference
- The claim appears in 2 or more documents (propagation makes it worth tracking regardless of type)

**What to skip:** Nuanced technical arguments, qualified assertions ("the evidence suggests..."), or claims distributed across multiple paragraphs. These are too fluid to track in a registry — they are better caught during derivation recipe review, where the full context of the source section is available.

**Target size:** 10-30 claims for a 5-document set. If you exceed 40, you are likely registering claims that appear in only one document or claims that are too granular (individual data points rather than headline findings).

### Persona Registry Heuristic

**Register a persona if either condition is true:**

- The persona has a distinct entry point or reading path through the document set (they start at a different document or follow a different sequence than existing personas)
- The persona's depth tolerance or domain vocabulary differs meaningfully from all existing personas (they need content at a different tier or in different language)

**What to skip:** Minor role variations that share the same path. If a "Deputy SRE Lead" follows the exact same reading path as the "SRE Lead" persona, reads the same documents, and has the same depth tolerance, they do not need a separate registry entry. Note them as an alias on the SRE Lead persona instead.

**Target size:** 8-15 personas for a standards suite. A document set serving more than 15 genuinely distinct personas is likely over-scoped — consider whether some personas can be served by the same path with minor annotations rather than separate registry entries.

### Coverage Matrix Heuristic

**A topic qualifies as a column in the coverage matrix if:**

- It represents a decision point or action item for at least one persona. "What are the breaking changes?" is a key topic because an SRE Lead needs to assess migration coverage against them. "How was the taxonomy methodology developed?" is not a key topic — it is supporting detail that belongs in the Tier 2 document but does not drive decisions.

**Target size:** 10-20 topics. If the matrix exceeds 25 columns, the topics are too granular. Merge related topics or promote them to a sub-matrix for a specific competency area.

**What to exclude:** Background context topics (historical motivation, literature review), process topics (how the document set was created), and topics that are relevant to only one persona in only one document (no cross-coverage to track).

## Structural Conventions

Structural conventions are rules that apply uniformly across every document in the set. They govern formatting, cross-referencing, callouts, and external citations. Without declared conventions, each document develops its own patterns, and readers must re-learn navigation conventions every time they switch documents.

### Conventions Template

Copy this template into your project and fill in each section. Store it alongside the manifest and registries. Each convention declares a rule, a worked example, and a rationale.

```yaml
conventions:
  heading_hierarchy:
    rule: "[Heading level rules per tier]"
    example: "Tier 2: H1 title, H2-H4 sections. Tier 1: H1 title, H2-H3 max. Tier 0: H1 title, H2 only."
    rationale: "Heading depth correlates with content depth."

  section_numbering:
    rule: "[Numbering scheme or lack thereof]"
    example: "Implicit numbering via heading hierarchy. Cross-references: 'document-id §Section Name'."
    rationale: "Explicit numbering creates brittle cross-references — adding a section renumbers downstream."

  cross_reference_format:
    rule: "[Exact link format for cross-document references]"
    example: "Relative markdown links: [text](../path/file.md#anchor). Prose: 'see document-title §Section Name for [what the reader will find]'. No bare URLs."
    rationale: "Relative links survive repository moves. The 'for [what]' clause lets readers decide whether to follow."

  callout_patterns:
    rule: "[Admonition/callout format and permitted types]"
    example: "Blockquote callouts with bold labels. Four types: **Note:**, **Warning:**, **Important:**, **Example:**"
    rationale: "Limiting types prevents semantic drift. Four types is enough."

  citation_style:
    rule: "[How external standards are cited]"
    example: "Standard Name (Authority, Version/Date). First use: full citation. Subsequent: short name. All cited standards in External Dependencies registry."
    rationale: "Version-pinned citations enable impact analysis when standards update."

  terminology_enforcement:
    rule: "[How registered terms are used across tiers]"
    example: "Tier 2: canonical term. Tier 1: tier_1 variant with canonical in parentheses on first use. Tier 0: tier_0 variant without parenthetical."
    rationale: "Parenthetical canonical terms in Tier 1 bridge readers who also consult Tier 2."
```

### Anti-Pattern: Undeclared Conventions

Each document follows its own formatting patterns — numbered sections here, unnumbered there, absolute URLs in one, relative paths in another. Inconsistent formatting signals inconsistent content, and mixed cross-reference formats make link integrity validation harder.

**Fix:** Declare conventions in the template above. Apply to existing documents during the next maintenance cycle. New documents conform from creation.

## Link Integrity

Link integrity ensures that cross-document references remain valid and navigable. Four checks cover the full surface area of cross-document linking.

### Check 1: Forward Links Resolve

**What it checks:** Every cross-document link in every document points to a target that exists — the target document exists, and the target section anchor exists within that document.

**How to perform it:**

1. Extract all cross-document links from every document in the set. In markdown, these are relative links matching the pattern `[text](../path/to/file.md#anchor)` or prose references matching "see document-title §Section Name."
2. For each link, verify the target file exists at the declared path.
3. For each link with a section anchor, verify the target file contains a heading that generates that anchor (in markdown, `## Section Name` generates anchor `#section-name`).
4. Record any link where the target file is missing or the anchor does not resolve.

**Defect type when this fails:** `broken-link`

### Check 2: Bidirectional Awareness

**What it checks:** If Document A links to Document B §3.2, then Document B's maintainer knows that §3.2 is a link target — so renaming or removing §3.2 triggers an update to Document A.

**How to perform it:**

1. For each cross-document link, record it in a link inventory: `source: document-A §Section, target: document-B §3.2`.
2. Generate a reverse-link index from the forward-link scan: for each document section, list all documents that link to it. Store this alongside the manifest.
3. For each target section receiving inbound links, verify the maintainer (or manifest) records the inbound dependency.

Without bidirectional awareness, a maintainer who renames a section has no way to know that other documents will break. The broken link is only discovered during the next forward-link audit.

### Check 3: No Orphan Sections

**What it checks:** Every section in every document is reachable from at least one reading path. A section that no reading path traverses and no other document links to is an orphan — it exists but nobody is directed to it.

**How to perform it:**

1. Compile the set of all sections across all documents (every heading at H2 or deeper).
2. Compile the set of all sections that appear in at least one reading path (from the Persona Registry and path definitions).
3. Add all sections that are targets of cross-document links (from the forward-link inventory).
4. The difference — sections in the first set but not in the second or third — are orphan sections.

**Defect type when this fails:** `orphan-section`

### Check 4: Anchor Stability

**What it checks:** Headings that other documents link to have not been renamed without updating all inbound links.

**How to perform it:**

1. From the reverse-link index (Check 2), identify all headings that serve as link targets.
2. Compare these headings against the previous version of each document (using `git diff` if the set is version-controlled, or by comparing against a previously saved snapshot of the document's heading list).
3. Any heading that changed text (and therefore changed its anchor) while still appearing in the reverse-link index is an anchor stability violation.
4. For each violation, identify all documents with inbound links to the old anchor and flag them for update.

**Defect type when this fails:** `broken-link`

**Practical tip:** When renaming a heading that is a link target, search the entire document set for the old anchor before committing the rename. Update all inbound links in the same commit.

## Consistency Audit Process

The consistency audit is a repeatable five-step sweep across the entire document set. It produces a prioritized defect list using a standard defect record format. Run this audit after any substantive change to a root document, before publishing a new version of the set, or on a regular cadence (quarterly for actively maintained sets).

### Step 1: Terminology Sweep

Search all documents for each term in the Terminology Registry (canonical form and all tier variants). Flag: (a) terms not matching the tier's declared variant, (b) prohibited conflations used where the registered term should appear, (c) key concepts in 3+ documents with inconsistent naming but not yet in the registry.

### Step 2: Claim Verification

For each claim in the Claim Registry: (a) verify the canonical source still contains the registered claim, (b) verify each propagation entry's `adapted_form` is still faithful to the (possibly updated) canonical claim, (c) search for unregistered quantitative claims in 2+ documents, (d) verify any `conflict_resolution` entries are still valid.

### Step 3: Link Validation

Run all four link integrity checks: forward links resolve, bidirectional awareness (generate/update reverse-link index), no orphan sections, anchor stability (compare against previous version via `git diff`).

### Step 4: Structural Convention Compliance

For each declared convention, check all documents for compliance: heading depth within tier limits, cross-reference format, callout types, citation format, terminology enforcement at appropriate tiers.

### Step 5: Defect Compilation and Triage

Compile all defects from Steps 1-4 into a single list. Assign each a type (from the 10-type taxonomy below) and a priority (from the Audit Triage framework). Sort by priority (P1 first), then by tier (Tier 0 before Tier 1 before Tier 2), then by document ID.

### Defect Record Format

Every defect found during any audit (consistency, derivation, reading path, link integrity) uses this standard format:

```yaml
defect:
  location: "document-id §section-name"
  type: "one of the 10 defect types listed below"
  severity: "error or warning"
  description: "What is wrong, stated precisely enough that someone unfamiliar with the audit can understand the problem."
  suggested_fix: "A concrete action to resolve the defect, not a restatement of the problem."
  related_registry: "The registry entry that is violated, if applicable (e.g., 'terminology: verification-tier' or 'claim: breaking-change-count'). Omit if no registry entry is involved."
```

### The 10 Defect Types

| Type | Definition | Typical Severity |
|---|---|---|
| `terminology-drift` | A document uses a term inconsistently with the Terminology Registry — wrong tier variant, unregistered synonym, or prohibited conflation | warning |
| `stale-claim` | A derivative contains a claim that no longer matches the canonical source in the root document | error |
| `broken-link` | A cross-document link points to a target that does not exist (missing file or missing anchor) | error |
| `lazy-deferral` | A derivative substitutes a link to the root for content the reader actually needs inline — "see the architecture doc for details" where the details are the core argument | error |
| `orphan-section` | A section exists in a document but is not reachable from any reading path and is not a link target from any other document | warning |
| `convention-violation` | A document deviates from declared structural conventions (heading depth, cross-reference format, callout types, citation style) | warning |
| `coverage-gap` | A persona x topic cell in the coverage matrix is empty — a persona has no document covering a topic relevant to them | warning |
| `path-dead-end` | A persona's reading path terminates before the persona can complete their task — the path does not reach a terminal state | error |
| `phantom-content` | A derivative contains a claim or assertion that does not appear in any root document — the derivative has originated content it should not have | error |
| `self-sufficiency-failure` | A derivative section cannot be understood or acted on without opening the root document — the self-sufficiency contract is violated | error |

### Worked Example: A Defect Record

```yaml
defect:
  location: "ops-assessment §1.1 Impact Landscape"
  type: "stale-claim"
  severity: "error"
  description: "States '14 identified compatibility issues' but the canonical source (architecture §3.2) was updated to '16 distinct breaking changes' in the March revision. The count is stale by two and the terminology ('compatibility issues' vs 'breaking changes') drifts from the Tier 1 variant."
  suggested_fix: "Update to '16 catalogued compatibility issues requiring migration coverage' (preserving the Tier 1 translation while correcting the count). Update the claim registry propagation entry for breaking-change-count."
  related_registry: "claim: breaking-change-count; terminology: breaking-change"
```

## Audit Triage and Prioritization

A full audit on a large document set can produce dozens of defects. This three-priority framework makes the output actionable by distinguishing defects that mislead readers from defects that inconvenience them.

### Priority 1: Fix Immediately

**Criterion:** The defect causes readers to act on wrong information.

**Defect types in this category:**

- **`stale-claim`** — a derivative states something the root no longer says
- **`self-sufficiency-failure`** — a derivative section cannot be understood without the root
- **`phantom-content`** — a derivative introduces claims not in any root

**Service level:** Resolve before anyone reads the affected document — within the current working session if the set is about to be shared, or within 24 hours for a published set.

### Priority 2: Fix Before Next Publish

**Criterion:** The defect degrades navigation or erodes reader trust, but does not directly cause wrong actions.

**Defect types in this category:**

- **`broken-link`** — the reader hits a dead end following a cross-reference
- **`lazy-deferral`** — the reader is sent to the root for content they need inline
- **`path-dead-end`** — a persona's reading path terminates before they can complete their task

**Service level:** Resolve before the next version of the document set is published or shared with a new audience.

### Priority 3: Fix During Maintenance

**Criterion:** The defect causes friction but does not mislead or seriously obstruct the reader.

**Defect types in this category:**

- **`terminology-drift`** — inconsistent naming; reader can usually infer meaning from context
- **`coverage-gap`** — a persona x topic cell is empty but adjacent cells may compensate
- **`convention-violation`** — structural inconsistency (heading depth, cross-reference format, citation style)
- **`orphan-section`** — content exists but no reading path leads to it

**Service level:** Resolve during the next scheduled maintenance cycle or when the affected document is edited for other reasons.

### Within-Priority Ordering Rules

When multiple defects share the same priority level, resolve them in this order:

1. **Root documents before derivatives.** Root defects may cause downstream derivative defects. Fixing the root first may automatically resolve derivative issues.
2. **Higher-audience-reach tiers first.** Fix Tier 0 before Tier 1, Tier 1 before Tier 2. More readers are affected by higher-tier defects.
3. **Registry-linked defects before unregistered defects.** Registry entries provide the canonical reference, making the fix precise. Fix these first to build momentum and reveal additional unregistered defects.

## Building a Terminology Registry from Scratch

Use this procedure when onboarding an existing document set where documents contain terms but no registry tracks them.

### Step 1: Harvest Candidate Terms

Collect terms that are defined with emphasis, used as structural labels, or domain-specific jargon. Do not evaluate yet — collect broadly. A five-document set typically yields 80-150 candidates.

**Worked example — partial harvest from a standards suite:**

| Term (as found) | Document | Section |
|---|---|---|
| verification tier | api-reference | Part I §2 |
| verification level | ops-assessment | §2.1 |
| breaking change | architecture | §3 |
| compatibility issue | ops-assessment | §1.1 |
| risk category | executive-brief | §2 |
| self-sufficiency | document-set-architecture | Depth Tiers |
| migration safeguard | architecture | §6.1 |
| coverage gap | ops-assessment | §2.4 |

### Step 2: Cluster Synonyms

Group candidate terms that refer to the same concept — same argumentative context, interchangeable usage, or explicit "also known as" phrasing.

**Worked example — clusters from the harvest:**

| Cluster | Candidate Terms | Documents |
|---|---|---|
| Cluster A | verification tier, verification level, level of checking | api-reference, ops-assessment, executive-brief |
| Cluster B | breaking change, compatibility issue, risk category | architecture, ops-assessment, executive-brief |
| Cluster C | self-sufficiency contract, self-sufficiency requirement, standalone readability | architecture doc, various |

### Step 3: Apply the Granularity Heuristic

For each cluster, check: appears in 3+ documents? Visible inconsistency? Common conflation risk? Include if any criterion is met; exclude if none.

**Worked example — filtering:**

| Cluster | 3+ docs? | Inconsistency? | Conflation risk? | Include? |
|---|---|---|---|---|
| Cluster A (verification tier) | Yes (3 docs) | Yes (3 different names) | Yes (trust level) | Yes |
| Cluster B (breaking change) | Yes (3 docs) | Yes (3 different names) | Yes (authentication change, bug) | Yes |
| Cluster C (self-sufficiency) | No (2 docs) | Mild | Low | Maybe — include if it becomes contentious |

### Step 4: Choose Canonical Terms

For each included cluster, select the canonical term — the most precise option, used in the root document where the concept is authoritatively defined, and unambiguous (does not collide with other registered terms). Designate remaining terms as tier variants (acceptable simplifications) or prohibited conflations (misleading alternatives).

### Step 5: Write Registry Entries

For each included cluster, write a full registry entry following the format in the Terminology Registry section: canonical definition (self-contained), canonical source, tier variants for all three tiers, and prohibited conflations with explanations.

### Step 6: Validate Coverage

- Is the registry between 15 and 40 terms? Under 15, look for missed clusters. Over 40, re-evaluate whether all entries have genuine inconsistency risk.
- Does every entry have at least one prohibited conflation? Terms with no conflation risk may not warrant registry tracking.
- Are all three tier variants filled in? If a concept does not appear at Tier 0, note "not used at Tier 0" rather than inventing a variant.

## Building a Claim Registry from Scratch

Use this procedure when onboarding an existing document set where claims propagate across documents but no registry tracks them.

### Step 1: Harvest Quantitative Claims

Collect every claim containing a number, percentage, specific count, or named list with a defined length. Numerical inconsistencies are the most visible and damaging consistency failures.

**Worked example — partial harvest:**

| Claim (as found) | Document | Section |
|---|---|---|
| "15 distinct breaking changes" | architecture | §3.2 |
| "over a dozen risk categories" | executive-brief | §2 |
| "15 compatibility issues" | ops-assessment | §1.1 |
| "four verification tiers" | api-reference | Part I §2 |
| "four verification levels" | ops-assessment | §2.1 |
| "six migration safeguards" | architecture | §6.1 |
| "six new safeguards" | executive-brief | §3 |

### Step 2: Harvest Structural Claims

Collect claims that define a model, taxonomy, framework, or classification scheme. Structural claims create dependencies — other documents reference the model by name or structure.

**Worked example — structural claims:**

| Claim (as found) | Document | Section |
|---|---|---|
| "breaking changes across four categories: schema changes, authentication changes, rate limit changes, and deprecation patterns" | architecture | §3.1 |
| "three categories of risk" | executive-brief | §2 |
| "the enforcement model uses a tiered approach with progressive capability" | api-reference | Part I §1 |

### Step 3: Cluster and Deduplicate

Group claims referring to the same underlying fact.

**Worked example — clusters:**

| Cluster ID | Root Claim | Derivatives |
|---|---|---|
| breaking-change-count | "15 distinct breaking changes" (architecture §3.2) | executive-brief §2, ops-assessment §1.1 |
| verification-tier-count | "four verification tiers" (api-reference Part I §2) | ops-assessment §2.1, recommendations §4.1 |
| safeguard-count | "six migration safeguards" (architecture §6.1) | executive-brief §3, recommendations §3 |

### Step 4: Identify the Canonical Source

For each cluster, identify the document and section where the claim originates with full evidence. This is usually a root document (Tier 2). If the claim originates in a derivative, flag it — the derivative may contain phantom content.

### Step 5: Record Adapted Forms

For each propagation entry, record the exact phrasing used in the derivative (`adapted_form`). This turns auditing from a reading-comprehension task into a string-comparison task.

**Worked example — breaking-change-count adapted forms:**

- executive-brief §2: "over a dozen risk categories" (imprecise — flag as a defect)
- ops-assessment §1.1: "15 catalogued compatibility issues requiring migration coverage" (faithful count, translated terminology)

### Step 6: Flag Existing Inconsistencies

You will discover claims already inconsistent across documents. Record these as defects immediately — the bootstrap is the first consistency audit.

**Worked example:**

```yaml
defect:
  location: "executive-brief §2"
  type: "stale-claim"
  severity: "error"
  description: "States 'over a dozen risk categories' where canonical source (architecture §3.2) states '15 distinct breaking changes.' Count is imprecise."
  suggested_fix: "Update to 'fifteen risk categories' (precise count, Tier 0 terminology)."
  related_registry: "claim: breaking-change-count"
```

### Step 7: Validate Coverage

- Is the registry between 10 and 30 claims? Under 10, search for missed structural claims. Over 30, check for entries that appear in only one document.
- Does every entry have at least one propagation entry? Claims with no propagation do not need registry tracking.
- Are adapted forms recorded for every propagation entry? Missing adapted forms negate the registry's audit-acceleration value.

## Anti-Pattern Summary

| Anti-Pattern | Section | Core Harm |
|---|---|---|
| Unregistered term drift | Terminology Registry | Readers cannot tell if differently-named concepts are the same thing or different things |
| Registry as exhaustive glossary | Terminology Registry | Maintenance burden exceeds consistency benefit; registry is ignored |
| Unregistered quantitative claims | Claim Registry | Number changes in roots are not propagated; derivatives contradict each other |
| Adapted forms not recorded | Claim Registry | Auditing requires opening every derivative instead of comparing registry entries |
| Undeclared conventions | Structural Conventions | Each document follows different formatting patterns; readers lose trust in consistency |
| Silently ignoring orphan sections | Link Integrity | Unmaintained content confuses readers who stumble on it |
| Running audits without the registries | Consistency Audit | Audit has no baseline to compare against; findings are ad hoc and non-repeatable |
| Treating all defects as equal priority | Audit Triage | P1 defects (wrong information) wait while P3 defects (formatting) are fixed first |
| Fixing derivatives before roots | Within-Priority Ordering | Root defects may be causing derivative defects; fixing derivatives first means fixing them twice |
| Bootstrapping registries exhaustively | Granularity Heuristics | Registry becomes a maintenance burden before it provides value; team abandons the process |

## Cross-References

These reference sheets cover sibling competencies in the wiki management skill set. All files are in the same directory as this document.

- `document-set-architecture.md` — Document manifest format, source-of-truth mapping, depth tiers, multi-root reconciliation, and conflict resolution protocol
- `reading-path-design.md` — Persona Registry, coverage matrices, time-based paths, and stepping-stone validation
- `content-derivation.md` — Derivation recipes, the self-sufficiency test, derivation integrity checks, and the anti-laziness discipline
- `document-evolution.md` — Change classification, impact tracing, git integration, external dependency tracking, and version coherence
- `document-governance.md` — Ownership model, LLM-as-steward trust model, change workflows, quality gates, and review triggers
