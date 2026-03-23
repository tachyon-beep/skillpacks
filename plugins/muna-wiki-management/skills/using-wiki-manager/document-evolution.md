# Document Evolution

## Overview

Document evolution manages what happens when something changes. A root document gets a new section. An external standard releases an update. A claim that propagates across five derivatives turns out to be wrong. Without a disciplined evolution process, the document set degrades silently: derivatives contradict their roots, cross-references point to reorganized sections, and external dependencies go stale without anyone noticing.

This reference sheet covers seven capabilities:

1. **Change classification** — categorizing a change to determine its propagation scope
2. **Impact tracing** — using the derivation graph and claim registry to produce a concrete work list
3. **Git integration** — diff-based classification, co-commits, branch discipline, and deferral detection
4. **External dependency tracking** — monitoring referenced standards and responding to their updates
5. **Version coherence** — keeping the document set internally consistent across releases
6. **Deprecation workflow** — retiring content without breaking navigation
7. **End-to-end change response procedures** — complete workflows for root changes and external standard updates

## When to Use

Load this discipline when:

- A root document has been modified and you need to determine which derivatives are affected
- An external standard referenced by the document set has been updated
- You are planning a structural change (adding, removing, or reorganizing sections) and need to assess downstream impact
- A consistency audit has flagged stale claims or broken links that resulted from untracked changes
- You need to deprecate a section or document and redirect readers
- You are incrementing the set version and need to verify coherence
- A user says: "the architecture changed," "migration framework updated," "propagate this change," "what does this edit affect," "deprecate this section"

Do NOT use this discipline when:

- You are writing a new derivative from scratch (use content derivation instead)
- You are doing a routine terminology or link audit with no upstream change (use cross-document consistency instead)
- You are designing the document set structure for the first time (use document set architecture instead)

## Key Terminology

This discipline references several structural artefacts from sibling reference sheets. Minimal definitions here so you can follow every procedure without leaving this document:

- **Document Manifest:** A structured project file (markdown with YAML frontmatter) declaring every document in the set. Each entry has `id`, `title`, `path`, `tier` (0 = executive, 1 = practitioner, 2 = reference), `role` (root or derivative), and `audience`. Derivatives additionally have `derives_from` (list of root IDs), `derivation_mode` (distillation, translation, or extraction), and optionally `reconciliation_note` for multi-root derivatives. The manifest's YAML frontmatter contains `set_name` and `set_version`.
- **Derivation Graph:** A directed acyclic graph (DAG) mapping section-level lineage between roots and derivatives. Each edge is: `root-id §section` -> `derivative-id §section`. Stored as a project file. This is more granular than the manifest's document-level `derives_from` — it tells you which *sections* of a derivative come from which *sections* of a root.
- **Derivation Recipe:** A structured template for each derivative section specifying: source section, derivation mode, target persona, self-sufficiency claim, required content (must appear inline), acceptable deferrals (may link to parent), and prohibited deferrals (common laziness traps). Recipes are the operational tool that prevents lazy deferral during re-derivation.
- **Claim Registry:** A project file tracking factual claims that propagate across documents. Each entry has `id`, `claim` (canonical phrasing), `canonical_source` (root section), and `propagation` (list of every derivative section that references it). When a claim changes, the propagation list is the work list.
- **Terminology Registry:** A project file tracking key terms with canonical definitions, tier-appropriate variants (tier 2: "authority tier," tier 1: "enforcement level," tier 0: "level of control"), and prohibited conflations. Consulted during change propagation to ensure updated content uses correct terms.
- **Persona Registry:** A project file listing each role the set serves, with `id`, `name`, `task`, `depth_tolerance`, `domain_vocabulary`, and `entry_point`. Relevant during structural changes that affect reading paths.
- **Depth Tiers:** Tier 0 = executive (1-3 page decision documents), Tier 1 = practitioner (8-15 page working documents), Tier 2 = reference (full technical analysis). A document at Tier N must be actionable without requiring the reader to open a Tier N+1 document (the self-sufficiency contract).
- **Self-Sufficiency Test:** Hide the parent document; read the derivative section as the target persona; determine whether the persona can do their job; audit all deferrals. A derivative that sends the reader to the root for content they need fails the test.
- **Deferral Audit:** Searching a derivative for phrases like "see the architecture," "as described in," "for the complete analysis" and classifying each as acceptable (optional depth) or lazy (substituting a link for required content). New cross-references that replace inline content are the primary signal of laziness during re-derivation.

## Change Classification

Every change to a document in the set falls into one of four categories. The category determines how far the change propagates and how much work is required.

### The Four Categories

| Category | Definition | Propagation Scope | Example |
|---|---|---|---|
| **Cosmetic** | Typo fixes, formatting, whitespace, punctuation | None. No derivative review needed. | Fixing "teh" to "the," adjusting table alignment, correcting a markdown heading level |
| **Clarification** | Same claim, better explanation. The meaning does not change; the expression improves. | Check derivatives for stale phrasing that no longer matches the root's improved wording. | Rewriting a confusing sentence about failure mode G3 to be clearer, adding an example to an existing explanation |
| **Substantive** | A claim, recommendation, count, model, or structural element changes in meaning. | Mandatory review of every derivative section in the changed claim's propagation list. | Changing "13 failure modes" to "14 failure modes," adding a new recommendation, revising a risk rating from High to Critical |
| **Structural** | Sections added, removed, renamed, or reorganized. The document's topology changes. | Reading paths, cross-document links, derivation graph edges, and the document manifest all need review. | Adding a new Section 3.3, merging Sections 5 and 6, removing an appendix, renaming "Authority Tiers" to "Enforcement Levels" |

### Classification Decision Tree

Follow this sequence for any change. Stop at the first match.

```
1. Did any section get added, removed, renamed, merged, or split?
   YES → STRUCTURAL
   NO  → continue

2. Did any factual claim change in meaning?
   (A count changed. A recommendation changed. A risk rating changed.
    A model gained or lost a component. A list gained or lost an item.)
   YES → SUBSTANTIVE
   NO  → continue

3. Did the wording change while preserving exact meaning?
   (Same claim, clearer phrasing. Added example. Rephrased for readability.)
   YES → CLARIFICATION
   NO  → continue

4. Only typos, formatting, or whitespace changed.
   → COSMETIC
```

### Worked Example: Classifying Three Changes in One Commit

Suppose a commit to the root architecture document contains three changes:

**Change A:** In Section 3.2, "The taxonomy identifies 13 distinct failure modes" becomes "The taxonomy identifies 14 distinct failure modes, including the newly catalogued FM-14 (Model Drift)."

Classification: **Substantive.** A count changed (13 to 14) and a new item was added to a named list. The claim registry entry `failure-mode-count` is directly affected. Every derivative section in that claim's propagation list needs review.

**Change B:** In Section 4.1, a paragraph about G3 (Security Blindspot) is rewritten. The old text: "The root cause is that the model optimizes for functional correctness rather than security properties." The new text: "This occurs because code generation models are trained on corpora containing both secure and insecure patterns, and the generation objective rewards functional correctness without a security constraint." Same meaning, better explanation.

Classification: **Clarification.** The claim is unchanged (G3 exists because of training data composition and optimization objective). The expression improved. Check derivatives for phrasing that assumed the old wording.

**Change C:** In Section 3.2, a mistyped reference "see Section 4.12" is corrected to "see Section 4.1.2."

Classification: **Cosmetic.** A typo in a cross-reference. No propagation needed (the derivative sections synthesize their own content and do not replicate this internal cross-reference).

### Anti-Pattern: Treating Everything as Cosmetic

When a root author makes a substantive change but describes it as "minor wording fix" in the commit message. The change propagation workflow never triggers. Derivatives go stale.

**Why it is harmful:** Classification drives propagation scope. Under-classifying a change means affected derivatives are never reviewed. The stale content persists until the next full audit — which might be months away. By then, readers may have acted on outdated information.

**Fix:** Classification is determined by the *content of the diff*, not by the commit message. The git integration procedures below include diff-based classification as a check against author self-reporting.

### Anti-Pattern: Treating Everything as Substantive

Every typo fix triggers a full derivative review. The team drowns in unnecessary propagation work. Eventually, they stop doing propagation reviews because the false positive rate is too high.

**Why it is harmful:** Over-classifying wastes effort and erodes discipline. When every change is treated as urgent, no change is treated as urgent. The propagation workflow becomes a checkbox exercise rather than a quality gate.

**Fix:** Use the decision tree. Cosmetic changes get zero propagation. Clarifications get a phrasing check (fast). Only substantive and structural changes trigger full propagation review.

## Impact Trace

An impact trace takes a classified change and produces a concrete work list: which derivative sections need review, what kind of review each needs, and in what order.

### Prerequisites

You need three artefacts before running an impact trace:

1. **The change itself** — which root section changed, and its classification (from the decision tree above)
2. **The derivation graph** — section-level edges from roots to derivatives
3. **The claim registry** — for substantive changes, the propagation list of the affected claim(s)

### Step-by-Step Procedure

**Step 1: Identify the changed root section(s).**

Read the diff. List every root section that was modified. For each, record the classification.

**Step 2: For each changed section, trace outbound edges in the derivation graph.**

Look up the changed root section in the derivation graph. Every edge leaving that section points to a derivative section that may be affected.

**Step 3: For substantive changes, cross-reference the claim registry.**

If a specific claim changed, look up that claim's `propagation` list. This may surface derivative sections not directly connected in the derivation graph (e.g., a claim quoted in a derivative's introduction that does not have a formal derivation edge).

**Step 4: Merge and deduplicate.**

Combine the derivative sections found in Steps 2 and 3. Remove duplicates. This is your raw work list.

**Step 5: Assign review type based on classification.**

| Change Classification | Review Type for Each Affected Derivative Section |
|---|---|
| Clarification | Phrasing check: does the derivative's wording still align with the improved root phrasing? |
| Substantive | Re-derivation: re-apply the derivation recipe for the affected section. Run self-sufficiency test on the updated section. |
| Structural | Full structural review: check derivation graph edges (do they still point to valid sections?), update reading paths, update cross-document links, update manifest if sections were added or removed. |

**Step 6: Order the work list.**

Process in this order: (1) Tier 0 derivatives first (most readers affected), (2) Tier 1 derivatives second, (3) multi-root derivatives last (these require checking whether the change interacts with content from the other root).

### Worked Example: "Architecture Section 3.2 Changed — Trace It Through"

**The change:** Architecture Section 3.2 now says "14 failure modes" instead of "13 failure modes." A new failure mode FM-14 (Model Drift) has been added. Classification: **substantive**.

**Step 1:** Changed root section: `architecture §3.2`.

**Step 2:** Derivation graph lookup for `architecture §3.2`:

```
architecture §3.2 (taxonomy) ──→ executive-brief §2 (key findings)
architecture §3.2 (taxonomy) ──→ ops-assessment §1.1 (failure mode analysis)
```

Two derivative sections are directly connected.

**Step 3:** Claim registry lookup for `failure-mode-count`:

```yaml
- id: failure-mode-count
  claim: "The taxonomy identifies 13 distinct failure modes"
  canonical_source: architecture §3.2
  propagation:
    - executive-brief §2 (distillation)
    - ops-assessment §1.1 (translation)
    - recommendations §2 (extraction)
```

The claim propagation list adds `recommendations §2`, which was not in the derivation graph edges for `architecture §3.2` (it derives from a different architecture section but quotes the count).

**Step 4:** Merged, deduplicated work list:

1. `executive-brief §2` (Tier 0 — distillation)
2. `ops-assessment §1.1` (Tier 1 — translation)
3. `recommendations §2` (Tier 1 — extraction, multi-root derivative)

**Step 5:** Review type for all three: **re-derivation** (substantive change). For each section:
- Retrieve the derivation recipe — a structured template declaring: `Source` (root section), `Mode` (distillation/translation/extraction), `Target persona`, `Self-sufficiency claim` (what the reader can DO after reading), `Required content` (what MUST appear inline), `Acceptable deferrals` (what can link to the root for optional depth). If no recipe exists, draft one now before re-deriving.
- Update the required content to reflect 14 failure modes and the addition of FM-14
- Re-derive the section following the recipe's mode and required content list
- Run the self-sufficiency test (see below)
- Run a deferral audit on the updated section

**Self-sufficiency test pass/fail criteria:** (1) Hide the root document. (2) Read the derivative section as the target persona. (3) The section **passes** if: the persona can complete their stated task using only this section; all quantitative claims appear verbatim (not approximated); no required content is replaced by a cross-reference to the root. The section **fails** if: any step requires opening the root to understand the argument; any deferral phrase substitutes for content the reader needs; claims are paraphrased in ways that change precision (e.g., "about a dozen" for "13").

**Step 6:** Processing order:
1. `executive-brief §2` — Tier 0, single-root. Update the count from 13 to 14. Add a one-line characterization of FM-14. Verify the three-category structure still holds (does FM-14 fit into an existing category or create a new one?).
2. `ops-assessment §1.1` — Tier 1, single-root. Update the count. Determine if FM-14 represents a control gap (does it affect the SRE Lead's control environment?). If yes, add it to the control gap table. If no, note it in the "out of scope" section.
3. `recommendations §2` — Tier 1, multi-root. Update the count. Check whether FM-14 interacts with any enforcement mechanism from the specification. Update the reconciliation note if the specification's count (previously 12 of 13) needs adjustment.

**After re-derivation:** Update the claim registry entry:

```yaml
- id: failure-mode-count
  claim: "The taxonomy identifies 14 distinct failure modes"
  canonical_source: architecture §3.2
```

### Anti-Pattern: Impact Trace Without the Claim Registry

Running the trace using only the derivation graph. The graph catches direct derivation edges but misses derivative sections that quote a claim without having a formal derivation edge to the source section.

**Why it is harmful:** In the worked example above, `recommendations §2` quotes the failure mode count but derives its primary content from a different architecture section. The derivation graph alone would miss it. The stale count ("13 failure modes") persists in the Recommendations document.

**Fix:** Always cross-reference the claim registry for substantive changes. The derivation graph shows structural lineage; the claim registry shows factual lineage. Both are needed.

## Git Integration

Document sets live in repositories. The evolution discipline leverages git to automate classification, enforce co-commits, maintain branch discipline, and detect deferral regression.

### Diff-Based Change Classification

When a root document is modified, `git diff` partially automates classification. This does not replace human judgment but provides a first-pass signal.

```bash
# Show the diff for a specific root document
git diff HEAD~1 -- understand/architecture.md

# Show only changed lines (no context) for faster scanning
git diff HEAD~1 --no-context -- understand/architecture.md

# Show section-level changes using function context (works with markdown headings)
git diff HEAD~1 -W -- understand/architecture.md
```

**Interpreting the diff for classification:**

| Diff Signal | Likely Classification |
|---|---|
| Only whitespace, punctuation, or formatting changes | Cosmetic |
| Rewording within existing paragraphs, no numbers or claims changed | Clarification |
| Numbers changed, list items added/removed, new assertions appear | Substantive |
| Section headings added, removed, or renamed; large blocks moved | Structural |

**Automatable check for substantive changes — numeric claim scan:**

```bash
# Extract all lines with numbers that changed
git diff HEAD~1 -- understand/architecture.md | grep '^[+-]' | grep -E '[0-9]+'
```

If this output shows changed numbers, the change is at least substantive. A changed number is never cosmetic.

### Manifest and Registry Co-Commits

The manifest and all registries must stay synchronized with the documents they describe. A change to a root document and its registry updates must land in the same commit or the same pull request.

**The co-commit rule:** If a commit modifies a root document AND that modification is substantive or structural, the same commit (or at minimum the same PR) must also update:

1. The claim registry — update any affected claim entries
2. The derivation graph — update edges if sections were added, removed, or renumbered
3. The manifest — update if document-level metadata changed (title, tier, audience)

```bash
# Stage root document change and registry updates together
git add understand/architecture.md
git add registries/claim-registry.yaml
git add derivation-graph.md
git commit -m "feat(architecture): add FM-14 Model Drift to taxonomy

Substantive change: failure mode count 13→14.
Updated claim registry (failure-mode-count).
Derivation graph edges unchanged (§3.2 targets unchanged)."
```

**Why co-commits matter:** If the root change lands in one commit and the registry update lands later (or never), there is a window where the registries are stale. Anyone running an impact trace during that window gets an incomplete work list. Co-commits eliminate the window.

### Branch Discipline

Choose your branching strategy based on change classification:

**Cosmetic or clarification changes:** Commit directly to the working branch. No propagation branch needed.

**Substantive changes affecting 1-2 derivative sections:** Commit the root change and derivative updates on the same branch. The scope is small enough to review in one PR.

```bash
# Small substantive change: root + derivative updates in one branch
git checkout -b update/fm14-addition
# ... make root change, propagate to derivatives ...
git add understand/architecture.md
git add understand/executive-brief.md
git add assess/ops-assessment.md
git add registries/claim-registry.yaml
git commit -m "feat(architecture): add FM-14, propagate to derivatives"
```

**Substantive changes affecting 3+ derivative sections, or any structural change:** Use a dedicated propagation branch. The root change lands first; derivative updates follow as separate, reviewable commits.

```bash
# Large propagation: dedicated branch
git checkout -b propagate/architecture-s3.2-fm14

# Commit 1: root change + registry updates
git add understand/architecture.md registries/claim-registry.yaml
git commit -m "feat(architecture): add FM-14 Model Drift to taxonomy"

# Commit 2: Tier 0 derivative
git add understand/executive-brief.md
git commit -m "propagate(executive-brief): update failure mode count to 14"

# Commit 3: Tier 1 derivatives
git add assess/ops-assessment.md respond/recommendations.md
git commit -m "propagate(tier-1): update failure mode count and control gap analysis"
```

This structure makes code review tractable: each commit has a clear scope, and reviewers can verify each derivative update against its derivation recipe independently.

### Diff-Based Deferral Detection

When reviewing a derivative update (yours or someone else's), diff the derivative against its previous version. This catches deferral regression — the derivative getting *less* self-sufficient after an update.

```bash
# Diff the derivative to check for new deferrals
git diff HEAD~1 -- assess/ops-assessment.md
```

**Red flags in the diff:**

| Diff Pattern | What It Signals |
|---|---|
| New lines containing "see architecture," "refer to," "as described in" | Possible new lazy deferral — verify each one |
| Inline content removed, cross-reference added in its place | Deferral regression: the section lost self-sufficiency |
| Specific numbers or claims replaced with vague language ("several," "multiple," "various") | Possible lazy re-derivation: precision was lost |

```bash
# Search for new deferral phrases introduced in the latest change
git diff HEAD~1 -- assess/ops-assessment.md | grep '^+' | grep -iE '(see (the |)architecture|refer to|as described in|for (the |)(full|complete)|see section)'
```

If this command produces output, inspect each match. A new deferral phrase in a derivative update is a signal — not proof — of laziness. Classify each using the deferral audit table (acceptable vs. lazy) before acting.

### Anti-Pattern: Root Change Without Registry Update

A substantive root change is committed, but the claim registry and derivation graph are not updated in the same commit or PR. The impact trace that runs next week uses stale registry data and misses affected derivatives.

**Why it is harmful:** The registry is the memory of the document set. When the registry is stale, the impact trace is incomplete. Derivatives go unreviewed. The stale content persists until someone notices it by accident.

**Fix:** Enforce the co-commit rule. Substantive and structural root changes must include registry updates in the same commit or PR.

## External Dependency Tracking

Document sets reference external standards, frameworks, and regulations. These change on their own schedule. External dependency tracking maintains awareness of what is referenced, where, and which version the reference assumes.

### The External Dependencies File

Maintain a YAML file in the project repository:

```yaml
external_dependencies:
  - id: pci-dss
    name: "PCI Data Security Standard"
    authority: "PCI Security Standards Council"
    version_referenced: "v4.0.1"
    last_checked: "2026-03-01"
    update_cadence: "annual"
    referencing_sections:
      - architecture §6.1
      - recommendations §3
      - security-assessment §2.4

  - id: stride
    name: "STRIDE Threat Model"
    authority: "Microsoft"
    version_referenced: "canonical (no versioning)"
    last_checked: "2026-03-01"
    update_cadence: "unversioned — check annually"
    referencing_sections:
      - architecture §3.1
      - security-assessment §1.2

  - id: owasp-top10
    name: "OWASP Top 10"
    authority: "OWASP Foundation"
    version_referenced: "2021"
    last_checked: "2026-03-01"
    update_cadence: "every 3-4 years"
    referencing_sections:
      - architecture §4.3
      - ops-assessment §2.1
```

### Field Reference

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Short identifier for cross-referencing |
| `name` | Yes | Full name of the external standard |
| `authority` | Yes | Publishing body |
| `version_referenced` | Yes | The specific version your documents cite. Use "canonical (no versioning)" for standards without formal version numbers. |
| `last_checked` | Yes | Date you last verified the external standard has not been updated |
| `update_cadence` | Yes | How often the standard typically updates (quarterly, annually, etc.). Drives monitoring schedule. |
| `referencing_sections` | Yes | Every section in the document set that references this standard. This is the impact list when the standard updates. |

### Monitoring Approach

External dependencies do not notify you when they change. You must actively monitor.

**Quarterly review (minimum):** For each external dependency, check whether a new version has been published. Update `last_checked` even if no update was found — this proves the check happened.

```bash
# After quarterly review, update last_checked dates
# Edit registries/external-dependencies.yaml, then:
git add registries/external-dependencies.yaml
git commit -m "chore: quarterly external dependency review — no updates found"
```

**Cadence-driven monitoring:** Use the `update_cadence` field to prioritize checks. A quarterly standard (like the migration framework) should be checked every quarter. An infrequently updated standard (like OWASP Top 10) can be checked annually. An unversioned standard should be checked at least once a year.

**When an update is found:** Trigger the "Responding to an External Standard Update" procedure (below).

### Anti-Pattern: Undeclared External Dependencies

A root document references an external standard, but the standard is not listed in the external dependencies file. When the standard updates, nobody knows to check the document set.

**Why it is harmful:** The document set silently goes stale. An SRE Lead reading the assessment assumes the safeguard identifiers are current. If the migration framework updated and the assessment was not revised, the SRE Lead may act on superseded guidance.

**Fix:** Every external standard referenced by name, version, or safeguard identifier in any document gets an entry in the external dependencies file. Audit for undeclared dependencies by searching documents for standard names and safeguard identifiers, then cross-referencing against the dependencies file.

## Version Coherence

Version coherence means that at any point in time, all documents in the set tell the same story. No derivative is stale relative to its root. No claim registry entry is outdated. No external dependency reference is unverified.

### Set Version

The document manifest's `set_version` field (in YAML frontmatter) is the single version identifier for the entire suite. It uses semantic versioning:

- **Patch** (1.0.0 → 1.0.1): Cosmetic or clarification changes only. No claim changes, no structural changes.
- **Minor** (1.0.0 → 1.1.0): Substantive changes. Claims updated, derivatives re-derived, registries updated.
- **Major** (1.0.0 → 2.0.0): Structural changes. Sections added/removed, reading paths revised, manifest restructured.

### Coherence Check

Before incrementing the set version, verify coherence with this checklist:

1. **Claim consistency:** Every claim in the claim registry matches its canonical source. **How to check:** For each claim entry, open the canonical source section and the derivative sections in the propagation list. Verify the claim text matches exactly. For numeric claims, grep all documents for the number (e.g., `grep -rn "13 failure" docs/`) and confirm consistency.
2. **Derivation freshness:** For every derivative section, the content reflects the current state of its source section in the root. **How to check:** Compare the root's last-modified commit date for each source section against the derivative's last-modified date. If the root is newer, the derivative may be stale: `git log --oneline -1 -- <root-file>` vs `git log --oneline -1 -- <derivative-file>`.
3. **Link integrity:** Every cross-document link resolves to an existing section with the correct heading. **How to check:** Extract all markdown links (`grep -oP '\[.*?\]\(.*?\)' <file>`) and verify each target file and anchor exists.
4. **Registry currency:** The claim registry, terminology registry, and external dependencies file all reflect the current state of the documents. **How to check:** Compare each registry entry's `canonical_source` against the actual document section. Flag entries whose source section has been modified since the registry was last updated.
5. **Graph accuracy:** Every edge in the derivation graph points from a section that exists to a section that exists. **How to check:** For each edge, verify both the source section heading and target section heading exist in their respective documents.

### Release Discipline

The release sequence ensures derivatives update *before* the set version increments:

```
1. Root changes are committed (with registry co-commits)
2. Impact trace produces work list
3. All affected derivative sections are re-derived
4. Self-sufficiency test passes on all updated sections
5. Consistency audit passes (no stale claims, no broken links)
6. Coherence check passes (full checklist above)
7. Set version is incremented in manifest frontmatter
8. Version increment committed:
   git add manifest.md
   git commit -m "release: bump set version to 1.1.0"
```

Never increment the version before derivative propagation is complete. A version number is a coherence promise: "everything in the set is consistent as of this version."

### Anti-Pattern: Version Increment Before Propagation

The set version is bumped in the same commit as the root change. Derivative updates follow in later commits. Between those commits, the version number promises coherence that does not exist.

**Why it is harmful:** Anyone who checks out the repository at the version-bump commit gets a set where the root says one thing and the derivatives say another. The version number lies. If the repository is tagged at that commit (for release), the released version is incoherent.

**Fix:** The version increment is always the last commit in the propagation sequence. It lands only after all derivatives are updated and all checks pass.

## Deprecation Workflow

When content is superseded — a section is replaced, a document is retired, or a recommendation is withdrawn — the deprecation workflow ensures readers are redirected and no navigation path hits a dead end.

### The Four-Step Process

**Step 1: Mark as deprecated.**

Add a deprecation notice at the top of the deprecated section. The notice must include: what is deprecated, why, and where the replacement lives.

```markdown
> **DEPRECATED as of set version 1.2.0.**
> This section has been superseded by [Section 4.3: Updated Control Mapping](recommendations.md#section-43-updated-control-mapping).
> This content will be removed in set version 1.3.0.
> Reason: The migration framework v3.0 update reorganized safeguard categories, making this mapping obsolete.
```

**Step 2: Update all inbound links.**

Search the entire document set for links pointing to the deprecated section. Update each link to point to the replacement.

```bash
# Find all references to the deprecated section
grep -rn "recommendations.md#section-32" docs/
# or search for the old heading text
grep -rn "Section 3.2: Legacy Control Mapping" docs/
```

For each inbound link found:
- If the replacement section covers the same content, update the link target
- If the replacement section covers different content, rewrite the surrounding text to reflect the new target
- If no replacement exists for that specific link's context, remove the link and add inline content (do not leave a dead-end)

**Step 3: Update reading paths.**

Check every reading path in the persona registry that traverses the deprecated section. Reroute each path through the replacement. Verify that the rerouted path still satisfies the path completeness contract: entry point exists at the persona's depth tier, no comprehension gaps (every concept introduced is defined in-path or within the persona's known vocabulary), no forced escalation (the path never requires jumping to a higher-depth document to complete the persona's task), and terminal state reached (the persona can do what they came to do at the end of the path).

**Step 4: Remove after one version cycle.**

The deprecated section persists for one version cycle (e.g., deprecated in 1.2.0, removed in 1.3.0). This gives readers who bookmarked the old section time to discover the redirect. After the grace period, remove the deprecated section and its deprecation notice entirely.

### Worked Example: Deprecating a Section

**Scenario:** The migration framework v3.0 update reorganized safeguard categories. The Recommendations document Section 3.2 ("Legacy Control Mapping") maps failure modes to migration framework v2.0 safeguard identifiers. A new Section 4.3 ("Updated Control Mapping") maps to migration framework v3.0 safeguard identifiers.

**Step 1:** Add deprecation notice to Recommendations Section 3.2 (see format above).

**Step 2:** Search for inbound links:
- `ops-assessment §2.4` links to `recommendations §3.2` with text "see the control mapping for framework alignment." Update to point to `recommendations §4.3`.
- `architecture §6.1` links to `recommendations §3.2` with text "the companion recommendations map these failure modes to framework controls." Update to point to `recommendations §4.3`.
- `executive-brief §3` does not link to this section — no change needed.

**Step 3:** Check reading paths:
- The SRE Lead persona's 30-minute path includes `recommendations §3.2` as the third step. Reroute to `recommendations §4.3`. Verify the SRE Lead can still complete their task (identify control augmentation priorities) with the new section.

**Step 4:** In the next version cycle (set version 1.3.0), remove `recommendations §3.2` entirely. Update the derivation graph to remove edges pointing to the old section.

### Anti-Pattern: Removing Without Redirecting

A section is deleted from a document. Links from other documents now point to nothing. Reading paths that included the section now have a gap. No deprecation notice was added; no redirect exists.

**Why it is harmful:** Readers following a cross-reference hit a dead end. Readers following a reading path lose their place. The document set's navigation integrity is broken silently. The harm compounds over time as more links accumulate to the missing section.

**Fix:** Never delete a section that other documents link to without first redirecting all inbound links and updating all reading paths. The four-step process above guarantees this.

## Responding to a Root Document Change

This is the complete end-to-end procedure for propagating a change from a root document through the derivative set. It chains evolution, derivation, consistency, and governance competencies.

### Prerequisites

- The root document change has been made (or is staged to commit)
- The manifest, derivation graph, and claim registry exist and are current
- Derivation recipes exist for the affected derivative sections

### The Procedure

**Step 1: Classify the change.**

Use the decision tree (see Change Classification above). Record the classification.

If cosmetic: commit the change. No further steps. Done.

If clarification, substantive, or structural: continue.

**Step 2: Run impact trace.**

Follow the impact trace procedure (see Impact Trace above). Output: a work list of affected derivative sections with review types.

**Step 3: Update registries.**

For substantive changes:
- Update the claim registry. Change the `claim` field to reflect the new content. Do not change the `propagation` list yet — that list is the work list you are about to process.

For structural changes:
- Update the derivation graph. Add edges for new sections. Remove edges for deleted sections. Update edges for renamed or renumbered sections.
- Update the manifest if document-level metadata changed.

**Step 4: Commit root change with registry co-updates.**

```bash
git add understand/architecture.md
git add registries/claim-registry.yaml
git add derivation-graph.md
git commit -m "feat(architecture): [description of root change]

Change classification: [substantive|structural|clarification].
Impact trace: [N] derivative sections affected.
Registry updates included."
```

**Step 5: Re-derive affected sections.**

For each entry in the work list, in priority order (Tier 0 first, then Tier 1, multi-root last):

1. Retrieve the derivation recipe for the affected section (if none exists, draft one — see the recipe template in the Impact Trace worked example above)
2. Read the updated root section
3. Re-derive the section following the recipe:
   - For distillation: compress the updated content, preserving the new claims exactly
   - For translation: translate the updated content using the vocabulary bridge
   - For extraction: re-evaluate the relevance filter against the updated content
4. Run the self-sufficiency test on the re-derived section
5. Run a deferral audit on the re-derived section
6. If the derivative is multi-root: check whether the change interacts with content from the other root. Update the reconciliation note if needed.

**Step 6: Commit derivative updates.**

For small propagation (1-2 sections):

```bash
git add understand/executive-brief.md assess/ops-assessment.md
git commit -m "propagate: update derivatives for [root change description]"
```

For large propagation (3+ sections), commit per tier or per derivative:

```bash
git add understand/executive-brief.md
git commit -m "propagate(executive-brief): update §2 for FM-14 addition"

git add assess/ops-assessment.md respond/recommendations.md
git commit -m "propagate(tier-1): update failure mode sections for FM-14"
```

**Step 7: Verify consistency.**

Run a targeted consistency check on the updated sections:
- Claim consistency: do updated sections match the updated claim registry?
- Terminology: do updated sections use terms consistent with the terminology registry?
- Link integrity: do all cross-references in updated sections resolve?

**Step 8: Verify coherence.**

Run the coherence check (see Version Coherence above). All items must pass.

**Step 9: Surface results for human review.**

Present to the human reviewer:
- The change classification and rationale
- The impact trace work list
- The deferral audit results for each re-derived section (even if all deferrals are acceptable — the human sees the count)
- Any ambiguities or judgment calls made during re-derivation
- The consistency check results

**Step 10: Increment set version.**

After human approval:

```bash
# Update set_version in manifest frontmatter
git add manifest.md
git commit -m "release: bump set version to [new version]"
```

## Responding to an External Standard Update

This procedure handles the case where an external standard referenced by the document set has been updated.

### Prerequisites

- An external standard update has been detected (during monitoring or reported by a team member)
- The external dependencies file lists which sections reference this standard

### The Procedure

**Step 1: Identify the scope of the external change.**

Determine what changed in the external standard. Classify the external change:

- **Minor update** (errata, clarifications, renumbering): Likely requires cosmetic or clarification updates to referencing sections.
- **Substantive update** (new controls, revised guidance, withdrawn recommendations): Requires substantive updates to referencing sections.
- **Major restructure** (reorganized safeguard categories, new framework edition): May require structural changes in referencing sections.

**Step 2: Look up referencing sections.**

From the external dependencies file, retrieve the `referencing_sections` list for the updated standard. This is your initial work list.

**Step 3: For each referencing section, determine the impact.**

Read each referencing section. Determine how it uses the external standard:

| Usage Pattern | Impact of External Update |
|---|---|
| Cites a specific safeguard identifier or clause | If renumbered: update the reference. If content changed: review the claim. |
| References the standard's version or date | Update the version/date reference. |
| Incorporates the standard's guidance into a recommendation | Re-evaluate the recommendation against the updated standard. |
| Mentions the standard by name without citing specifics | Likely no change needed unless the standard's scope changed. |

**Step 4: Update root documents first.**

External standard references in root documents are the canonical sources. Update roots before derivatives.

For each affected root section:
1. Verify the external reference is still accurate
2. Update safeguard identifiers, version references, or cited guidance as needed
3. Classify the root update using the change classification decision tree

**Step 5: Propagate to derivatives.**

If the root updates are substantive or structural, trigger the "Responding to a Root Document Change" procedure (above) for each updated root section. This automatically handles derivative propagation, registry updates, and coherence checks.

If the root updates are only clarification or cosmetic, check derivatives for stale version references or safeguard identifiers but do not trigger full propagation.

**Step 6: Update the external dependencies file.**

```yaml
  - id: pci-dss
    name: "PCI Data Security Standard"
    authority: "PCI Security Standards Council"
    version_referenced: "v4.1"          # Updated
    last_checked: "2026-03-21"          # Updated
    update_cadence: "annual"
    referencing_sections:
      - architecture §6.1
      - recommendations §3              # May have changed to §4.3
      - security-assessment §2.4
```

```bash
git add registries/external-dependencies.yaml
git commit -m "chore(deps): update PCI-DSS reference to v4.1"
```

**Step 7: Verify no undeclared references remain.**

Search the entire document set for mentions of the external standard that are not captured in `referencing_sections`:

```bash
# Search for migration framework references not yet tracked
grep -rn "migration-framework" docs/ | grep -v "node_modules"
```

Add any newly discovered references to the `referencing_sections` list.

### Worked Example: Framework Version Update

**Scenario:** The migration framework v3.0 update is released. Key change: safeguard category MS-0971 (Schema Validation) has been split into MS-0971 (Manual Schema Validation) and MS-2100 (Automated Schema Validation).

**Step 1:** External change classification: **substantive** (a safeguard category was split — new safeguard identifier, revised scope).

**Step 2:** Referencing sections from external dependencies file: `architecture §6.1`, `recommendations §3`, `ops-assessment §2.4`.

**Step 3:** Impact per section:
- `architecture §6.1`: References MS-0971 by number and cites its scope. Must update to reference both MS-0971 and MS-2100 with their revised scopes.
- `recommendations §3`: Recommends augmenting MS-0971 for AI code risk. The augmentation may now map to MS-2100 instead. Must re-evaluate recommendation.
- `ops-assessment §2.4`: Lists MS-0971 as a control requiring review. Must add MS-2100 and clarify the split.

**Step 4:** Update root document:
- Update `architecture §6.1` to reference both new safeguard identifiers with their scopes.
- This root change is classified as **substantive** (a control mapping changed).

**Step 5:** Trigger "Responding to a Root Document Change" for `architecture §6.1`. This propagates the MS-0971 split through `recommendations §3` and `ops-assessment §2.4`.

**Step 6:** Update external dependencies file with `version_referenced: "March 2026"` and `last_checked: "2026-03-21"`.

**Step 7:** Search for undeclared framework references. Suppose `executive-brief §3` mentions "framework compliance" in passing — verify this general reference does not need updating (it does not cite a specific safeguard identifier, so the mention likely remains valid).

## Anti-Pattern Summary

| Anti-Pattern | Section | Core Harm |
|---|---|---|
| Treating everything as cosmetic | Change Classification | Substantive changes go unpropagated; derivatives go stale |
| Treating everything as substantive | Change Classification | Propagation fatigue; team stops doing reviews |
| Impact trace without the claim registry | Impact Trace | Derivative sections that quote a claim without a derivation edge are missed |
| Root change without registry update | Git Integration | Stale registry produces incomplete impact traces |
| Version increment before propagation | Version Coherence | The version number promises coherence that does not exist |
| Removing without redirecting | Deprecation Workflow | Readers hit dead ends; navigation integrity breaks silently |
| Undeclared external dependencies | External Dependency Tracking | External standard updates go unnoticed; documents cite superseded guidance |
| Updating a root without checking derivative impact | General | Derivatives contradict their roots; readers get conflicting information |
| Updating a derivative without checking if the root changed | General | The derivative diverges from its root; phantom content may be introduced |
| Deferral regression during re-derivation | Git Integration | Re-derived section becomes less self-sufficient than the original; new lazy deferrals replace inline content |

## Cross-References

These reference sheets cover sibling competencies in the wiki management skill set. All files are in the same directory as this document.

- `document-set-architecture.md` — Document manifest format, depth tiers, derivation graph structure, multi-root reconciliation patterns, and conflict resolution protocol
- `reading-path-design.md` — Persona Registry, coverage matrices, time-based paths, stepping-stone validation, and path completeness contract
- `content-derivation.md` — Derivation recipes, the three derivation modes, the self-sufficiency test, deferral audit table, and the anti-laziness discipline
- `cross-document-consistency.md` — Terminology Registry, Claim Registry, structural conventions, link integrity, and the consistency audit process
- `document-governance.md` — Ownership model, LLM-as-steward trust model, change workflows, quality gates, and review triggers
