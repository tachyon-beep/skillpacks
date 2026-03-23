# Document Governance

## Overview

Document governance defines who can change what, what process guards quality, and how to trust LLM-maintained output. The other five competencies in this plugin define what to check — derivation integrity, consistency, reading paths, evolution impact. Governance ties them together by specifying the ownership model, the human-in-the-loop contract for LLM stewardship, the change workflows, the quality gates, and the review triggers.

This is the competency you load when you need to answer: "Who is responsible for this document? What must happen before this change is published? How do I know the LLM's output is trustworthy?"

Without governance, the other competencies are ad hoc — people run audits when they remember, derivatives drift because nobody owns them, and LLM output gets silently self-approved. With governance, every document has an owner, every change follows a workflow, and every quality gate has a defined pass/fail criterion.

## When to Use

Load this discipline when:

- You are setting up ownership and review workflows for a new or existing document set
- You need to run quality gates on LLM-generated derivative content before publishing
- You are assigning or clarifying ownership of root documents and derivatives
- You need to determine what review is required for a specific change
- A user says: "who owns this document," "what's the review process," "run quality gates," "set up governance," "check Claude's output"

Do NOT use this discipline when:

- You are writing or updating a derivative section (use content derivation instead — governance reviews the result)
- You are classifying a change and tracing its impact (use document evolution instead — governance runs after evolution produces the work list)
- You are doing a terminology or link audit (use cross-document consistency instead — governance decides when to trigger audits)

## Key Terminology

These definitions are sufficient to follow every procedure in this document without consulting sibling reference sheets.

- **Depth Tiers:** Documents are classified by depth. **Tier 0** = executive (1-3 page decision documents). **Tier 1** = practitioner (8-15 page working documents for specific roles). **Tier 2** = reference (full technical analysis, typically root documents). A document at Tier N must be actionable without requiring the reader to open a Tier N+1 document.

- **Derivation Recipe:** A structured template for each derivative section specifying: source root section, derivation mode (distillation, translation, or extraction), target persona, self-sufficiency claim (what the reader can DO after reading this section alone), required content (must appear inline), and acceptable deferrals (may link to parent for optional depth). Recipes are the operational tool that prevents lazy deferral.

- **Claim Registry:** A project file tracking factual claims that propagate across documents. Each entry has `id`, `claim` (canonical phrasing), `canonical_source` (root section), and `propagation` (list of every derivative section referencing it). When a claim changes, the propagation list is the work list. Target size: 10-30 claims for a 5-document set.

- **Persona Registry:** A project file listing each role the document set serves, with `id`, `name`, `task` (assess/implement/govern/understand), `depth_tolerance` (executive/practitioner/technical), `domain_vocabulary` (terms they know), and `entry_point`. Target size: 8-15 personas for a standards suite.

- **Change Classification:** Every change falls into one of four categories. **Cosmetic** = typo/formatting (no propagation). **Clarification** = same claim, better wording (check derivative phrasing). **Substantive** = a claim, count, model, or recommendation changes in meaning (mandatory derivative review via propagation list). **Structural** = sections added, removed, renamed, or reorganized (reading paths, links, manifest, and derivation graph all need review).

- **Self-Sufficiency Test:** A four-step check on any derivative section: (1) hide the parent document, (2) read the section as the target persona, (3) determine whether the persona can do their job, (4) run the deferral audit. A derivative that requires the reader to open the root for comprehension fails.

- **Deferral Audit:** Scanning a derivative for phrases like "see the architecture document," "as described in," "for the complete analysis" and classifying each as **acceptable** (pointing to optional depth that supports but is not required) or **lazy** (substituting a link for content the reader actually needs). Any lazy deferral is a defect.

- **External Dependency Tracker:** A project file listing external standards the document set references (e.g., migration framework, STRIDE). Each entry has `id` (standard identifier), `name`, `version_referenced` (the version the documents assume), and `referencing_sections` (every document section that cites or relies on the standard). When an external standard updates, the `referencing_sections` list is the impact scope.

- **Coverage Matrix:** Personas as rows, key topics as columns, cells contain which document/section covers that topic for that persona. Empty cells are explicit gaps. A key topic qualifies for the matrix if it represents a decision point or action item for at least one persona. Target size: 10-20 topic columns.

## Ownership Model

Every document in the set has exactly one owner. Ownership determines authority to change content and responsibility for keeping it accurate.

### Root Owners

A root owner has authority over a canonical document — a Tier 2 reference document that other documents derive from.

**Responsibilities:**

1. Approve all substantive and structural changes to the root
2. Classify changes using the four-category system (cosmetic, clarification, substantive, structural) based on the diff content, not commit messages
3. Notify derivative stewards when a substantive or structural change lands
4. Maintain the root's entries in the claim registry and terminology registry
5. Resolve multi-root conflicts when two roots contradict each other (terminology, factual, or structural conflicts)

**Authority boundaries:**

- Root owners change content in their root document only
- Root owners do NOT directly edit derivatives — they notify stewards who then propagate
- Root owners are the final arbiter when a derivative steward questions whether a root change is substantive or clarification

### Derivative Stewards

A derivative steward is responsible for keeping a derivative document faithful to its root(s). Stewards synthesize and maintain content — they do not originate it. When Claude acts as steward (the primary use case), the LLM-as-Steward Trust Model below governs how its work is reviewed.

**Responsibilities:**

1. Propagate root changes into the derivative using the derivation recipes
2. Run the self-sufficiency test on every updated section
3. Run the deferral audit on every updated section and surface results to the human reviewer
4. Maintain the derivative's derivation recipes (update when root structure changes)
5. Flag any case where a derivation recipe cannot be satisfied without introducing phantom content

**Authority boundaries:**

- Stewards may improve clarity and fix defects within the derivative
- Stewards must NOT introduce substantive claims not present in a root (phantom content)
- Stewards must NOT resolve multi-root conflicts — these escalate to the relevant root owners
- Stewards must NOT silently pass their own quality gate results when Claude is the steward

### Worked Example: Ownership Assignment for a 5-Document Standards Suite

Given this document set:

| Document | Tier | Role | Owner |
|---|---|---|---|
| Full Discussion Paper (`architecture`) | 2 | Root | Architecture Author (human subject-matter expert) |
| Companion Specification (`specification`) | 2 | Root | Specification Author (human technical lead) |
| Executive Brief (`executive-brief`) | 0 | Derivative (from architecture) | Executive Communications Lead or Claude-as-steward |
| SRE Lead Assessment (`ops-assessment`) | 1 | Derivative (from architecture) | Security Advisor or Claude-as-steward |
| Recommendations (`recommendations`) | 1 | Derivative (from architecture + specification) | Governance Lead or Claude-as-steward |

**Why this assignment works:**

- The two root documents have human owners with domain authority. Root ownership cannot be delegated to an LLM because roots originate content — there is no upstream source to derive from.
- The three derivatives have stewards (human or Claude). Claude is a viable steward because derivatives do not originate content; they synthesize, translate, or extract from roots. The derivation recipes constrain what Claude produces, and the trust model ensures human review.
- The Recommendations document derives from both roots. Its steward must handle multi-root reconciliation but escalates conflicts to the relevant root owner (Architecture Author for risk justification conflicts, Specification Author for enforcement mechanism conflicts).

**Ownership changes:** When a steward leaves or is reassigned, the incoming steward must review all derivation recipes for their document and run a full quality gate pass before assuming ownership. This prevents silent inheritance of stale content.

## LLM-as-Steward Trust Model

When Claude acts as the derivative steward, the same LLM is both author and auditor. This creates a trust gap: Claude running quality gates on its own output is necessary (it catches mechanical errors) but not sufficient (it may not catch its own systematic biases, especially the tendency toward lazy deferral). The following contract closes that gap.

### Point 1: Claude Proposes, Human Disposes

Claude drafts derivative updates and runs all quality gates (self-sufficiency test, deferral audit, claim fidelity check, consistency check). But the quality gate *output* is presented to the human for review — Claude does not silently pass its own work.

**What Claude must surface to the human after any derivative update:**

1. The updated derivative section(s)
2. The quality gate results for each section — pass or fail, with specifics
3. The deferral audit results (see Point 2 below)
4. Any ambiguous results or judgment calls Claude made during the update
5. The derivation recipe that governed the update, so the human can verify the recipe was followed

**What Claude can decide autonomously (no human review needed):**

- Cosmetic changes to derivatives (typo fixes, formatting)
- Whether a root change is cosmetic (but if there is any doubt, escalate)
- Ordering of work list items during impact trace

**Worked example — Claude's output after updating a derivative section:**

```
## Update: SRE Lead Assessment Section 1.1

### Change Trigger
Architecture Section 3.2 changed: failure mode count 13 → 14 (FM-14: Model Drift added).
Change classification: SUBSTANTIVE.

### Updated Content
[the updated section text]

### Quality Gate Results
- Derivation integrity: PASS — recipe requires failure mode count and category
  breakdown; both updated to reflect 14 modes and new FM-14.
- Self-sufficiency: PASS — SRE Lead can identify control gaps from this section
  alone without opening the architecture document.
- Claim fidelity: PASS — count matches root (14), category breakdown matches
  (4 specification + 7 generation + 3 integration).
- Consistency: PASS — term "failure mode" used per terminology registry;
  translated to "control gap" per SRE Lead vocabulary bridge.

### Deferral Audit
Total cross-references to root: 2
- "For the complete failure mode taxonomy, see the full discussion architecture §3.2"
  → ACCEPTABLE (optional depth, section is self-sufficient without it)
- "The training data analysis supporting this classification is detailed in
  architecture §3.2.1" → ACCEPTABLE (evidence chain, not required for SRE Lead action)
Lazy deferrals: 0

### Judgment Calls
- FM-14 (Model Drift) classified as relevant to SRE Lead control gap analysis
  because it affects model lifecycle governance. Human should verify this
  relevance classification.
```

### Point 2: Deferral Audit Results Are Always Surfaced

Even if Claude believes every deferral is acceptable, the human sees the deferral count and classification. This is the highest-risk failure mode (LLM laziness producing signpost documents instead of self-sufficient ones), so it gets mandatory human visibility regardless of the audit result.

**The deferral audit output format:**

```
Deferral Audit: [document-id §section]
Total deferrals found: [count]
Acceptable deferrals: [count] — [brief list]
Lazy deferrals: [count] — [brief list with locations]
New deferrals since last version: [count]
Removed inline content replaced by deferrals: [count — THIS IS THE KEY METRIC]
```

**The critical metric: removed inline content.** When reviewing a derivative update, the most informative signal is not the total deferral count but whether inline content was *replaced* by cross-references. A diff showing that a paragraph of synthesized content became "see architecture §X" is a regression — the derivative got less self-sufficient, not more. Claude must report this metric even when the total deferral count looks acceptable.

**Why this cannot be silent:** Claude's systematic bias toward deferral means that Claude rating its own deferrals as "acceptable" is exactly the judgment most likely to be wrong. The human does not need to check every deferral — but they need to see the numbers to decide whether spot-checking is warranted.

### Point 3: Spot-Check Protocol

After Claude updates a derivative, the human spot-checks at least one section against the root by reading both side-by-side. The section to check is not chosen randomly.

**Section selection criteria (in priority order):**

1. **Highest derivation complexity:** Choose the section using synthesis mode (both extraction + translation), multi-root reconciliation, or the most complex derivation recipe. Complex derivations have more opportunities for meaning loss.
2. **Highest deferral count:** If multiple sections have similar complexity, choose the one with the most deferrals. More deferrals means more opportunities for lazy substitution.
3. **Most substantive change:** If the update was triggered by a root change, choose the section most affected by that change — the one where the derivation recipe required the most new content.

**What the human checks during the spot-check:**

- Does the derivative section make the same claims as the root section? (Claim fidelity)
- Are the deferrals truly optional, or does the reader need that content? (Deferral quality)
- Is the depth appropriate for the target persona? (Tier alignment)
- Has any meaning been inverted, qualified differently, or lost during compression/translation? (Meaning preservation)

**Spot-check cadence:** One section per derivative update minimum. For major updates (substantive root changes affecting 3+ derivative sections), spot-check two sections — one from the most complex derivation and one from the highest-traffic section (Tier 0 content or the section most personas traverse).

### Point 4: Escalation Triggers

Claude must flag and defer to the human — not proceed autonomously — when any of the following conditions are detected:

| Trigger | Why It Requires Human Decision | What Claude Surfaces |
|---|---|---|
| Multi-root conflict detected | Two roots contradict each other (different counts, incompatible models, conflicting terminology). The resolution may require updating one or both roots, not just the derivative. | Both versions, their locations, the conflict type (terminological, factual, or structural), and a recommendation — but Claude does not apply the resolution. |
| Derivation recipe unsatisfiable | The recipe's required content cannot be produced from the root without introducing claims not present in the root (phantom content). This usually means the root is missing content the derivative needs. | The recipe, the gap, and whether the fix is to update the root or revise the recipe. |
| Quality gate produces ambiguous results | A section arguably passes or fails the self-sufficiency test depending on how "can do their job" is interpreted for the persona. | The section, the ambiguous criterion, both interpretations, and Claude's tentative assessment. |
| Deferral classification uncertain | A cross-reference might be acceptable (optional depth) or lazy (required content behind a link), and Claude cannot determine which. | The deferral, the context, and both interpretations. |
| Structural change affects 5+ derivative sections | The blast radius is large enough that human prioritization of the work list is warranted. | The full impact trace and a suggested ordering, but the human confirms the plan before Claude executes. |

## Change Workflows

### Root Document Change Workflow (6-Step Checklist)

Use this workflow whenever a root document is modified, regardless of who made the change.

1. **Classify the change.** Read the diff. Apply the classification decision tree: (a) any sections added/removed/renamed/merged/split? STRUCTURAL. (b) any factual claim changed in meaning? SUBSTANTIVE. (c) same meaning, better wording? CLARIFICATION. (d) only typos/formatting? COSMETIC. Classify based on the diff content, not the commit message. If the commit contains changes at multiple classification levels, use the highest level.

2. **If substantive or structural: run impact trace.** Look up the changed root section(s) in the derivation graph. For substantive changes, cross-reference the claim registry propagation lists. Merge and deduplicate to produce the work list. Order: Tier 0 derivatives first, Tier 1 second, multi-root derivatives last.

3. **Notify derivative stewards and execute updates.** Each steward (human or Claude) reviews their affected sections. For substantive changes: re-apply the derivation recipe, update required content. For structural changes: update derivation graph edges, cross-document links, reading paths, and manifest. For clarifications: check derivative phrasing alignment.

4. **Run the self-sufficiency test on every updated derivative section.** Hide the parent, read as the target persona, verify they can do their job, run the deferral audit. Any lazy deferral is a defect — fix before proceeding.

5. **Run the consistency audit across the set.** Check terminology against the terminology registry. Check claims against the claim registry (the canonical source just changed — verify all propagation targets reflect the new value). Validate all cross-document links resolve. Check structural conventions.

6. **Increment the set version.** Update the `set_version` in the manifest frontmatter. Root document changes and their derivative propagation land in the same commit or PR to maintain coherence.

**Anti-pattern: skipping step 2.** A root author makes a "small fix" and does not notify stewards. The fix is actually substantive (a count changed). Derivatives go stale. The stale content persists until the next full audit — which might be months later. Readers act on outdated information in the meantime.

### Derivative-Only Change Workflow (5-Step Checklist)

Use this workflow when a derivative steward improves clarity, fixes a defect, or fills a coverage gap without any upstream root change triggering the update.

1. **Propose the change.** Describe what is changing and why. If Claude is the steward, surface the proposed change to the human before applying it.

2. **Verify no phantom content.** Check every claim in the changed section against the root. If the change introduces a claim not present in any root, it is phantom content — reject it. Phantom content is the derivative originating information rather than synthesizing from the root, which violates the derivation contract.

3. **Verify the derivation recipe is still satisfied.** Check the recipe's required content list. Does the updated section still contain everything the recipe specifies? If the change removed required content, the recipe is violated — restore the content or update the recipe (with human approval if Claude is the steward).

4. **Run the self-sufficiency test on the changed section(s).** Same four-step procedure as the root change workflow: hide parent, persona simulation, claim grounding, deferral audit.

5. **No set version increment needed.** Derivative-only changes do not change the canonical content of the set. They improve the derivative's quality without altering what the set says. Commit normally.

**Anti-pattern: derivative steward introducing a "helpful addition."** The steward adds a recommendation not present in any root document ("organizations should also consider X"). This is phantom content regardless of how helpful it seems. The correct path: propose the addition to the root owner. If accepted, it enters the root first, then propagates to the derivative through the normal workflow.

## Quality Gates

Six gates must pass before any derivative content is published. For each gate: what it checks, what passing looks like, what triggers failure, and the concrete check to run.

### Gate 1: Derivation Integrity

**What it checks:** Every derivative section has a derivation recipe, and the section's content satisfies that recipe.

**Passing looks like:**
- Every derivative section has a recipe with all required fields (source, mode, target persona, self-sufficiency claim, required content, acceptable deferrals)
- Every item in the recipe's "required content" list is present in the section — inline, not behind a link
- The derivation mode matches the section's actual content (distillation sections compress, translation sections restate in persona vocabulary, extraction sections curate)
- The deferral audit finds zero lazy deferrals

**Failure triggers:**
- A derivative section has no recipe (undocumented derivation — you cannot verify what you cannot specify)
- A required content item is missing or replaced by a cross-reference
- The deferral audit finds one or more lazy deferrals
- The section's content does not match its declared mode (e.g., a "translation" that uses the root's vocabulary unchanged)

**Concrete check:** For each derivative section, retrieve its recipe. Compare the recipe's required content list against the section. Run the deferral audit. Output: pass/fail per section with defect details.

### Gate 2: Self-Sufficiency

**What it checks:** Every derivative section can be understood and acted upon by the target persona without consulting the root document.

**Passing looks like:**
- Reading the section as the target persona, the persona can perform their declared task (assess control gaps, make a funding decision, review code, draft a policy, evaluate compliance)
- Every claim in the section is grounded — enough evidence or reasoning for the reader to trust the conclusion
- All deferrals are acceptable (optional depth, not required content)

**Failure triggers:**
- The persona cannot complete their task without opening the root
- A claim is stated without grounding ("several controls need updating" without saying which ones or why)
- A deferral replaces content the persona needs ("see Section 6.1 for the control mapping" when the control mapping IS the section's purpose)

**Concrete check:** For each derivative section, perform the four-step self-sufficiency test: (1) hide the parent, (2) read as the target persona, (3) verify the persona can do their job, (4) run the deferral audit. Output: pass/fail per section with specific failure points.

### Gate 3: Consistency

**What it checks:** Terminology and factual claims match their registry entries across all documents in the set.

**Passing looks like:**
- Every term used in the set matches its terminology registry entry for the appropriate tier (a Tier 0 document uses Tier 0 variants, not Tier 2 technical terms)
- No prohibited conflations appear (e.g., "vulnerability" where the registry says "failure mode")
- Every quantitative or structural claim matches its claim registry entry (the count is "14," not "13" or "about a dozen")
- Claims in derivatives match the canonical source, not a stale version

**Failure triggers:**
- A term in a derivative uses the wrong tier variant (Tier 2 vocabulary in a Tier 0 document)
- A prohibited conflation appears
- A claim in a derivative does not match the current canonical value
- A claim appears in a derivative but has no claim registry entry (untracked propagation)

**Concrete check:** Extract all key terms from all documents, compare against the terminology registry, flag mismatches. Extract all quantitative and structural claims, verify against canonical sources in the claim registry. Output: defect list using the standard defect record format:

```
Defect:
  location: [document-id §section]
  type: terminology-drift | stale-claim | broken-link | lazy-deferral |
        orphan-section | convention-violation | coverage-gap | path-dead-end |
        phantom-content | self-sufficiency-failure | recipe-missing
  severity: error | warning
  description: [what's wrong]
  suggested_fix: [how to fix it]
  related_registry: [which registry entry is violated, if applicable]
```

### Gate 4: Link Integrity

**What it checks:** Every cross-document reference resolves to a valid target, and no section is orphaned.

**Passing looks like:**
- Every cross-document link points to a heading or section that exists in the target document
- No broken links (pointing to renamed, removed, or reorganized sections)
- Every section is reachable from at least one reading path (no orphans)
- Bidirectional awareness is maintained — if document A links to document B Section 3.2, the link target has not been renamed without updating inbound links

**Failure triggers:**
- A link points to a section that does not exist (broken link)
- A section exists but no reading path reaches it (orphan section)
- A heading was renamed but inbound links from other documents still use the old heading

**Concrete check:** Collect all cross-document links from all documents. For each link, verify the target heading exists in the target document. For each section in every document, verify at least one reading path includes it. Output: list of broken links and orphan sections with their locations.

### Gate 5: Path Completeness

**What it checks:** Every persona in the persona registry has a complete, navigable reading path from entry point to terminal state.

**Passing looks like:**
- Every persona has a declared entry point that exists
- The reading path from entry point to terminal state has no gaps — every concept introduced is either defined in-path or within the persona's declared domain vocabulary
- No forced escalation — the path never requires the persona to jump to a higher-depth document to complete their task
- The path reaches terminal state — the persona can do what they came to do (assess, implement, govern, or understand)

**Failure triggers:**
- A persona has no entry point or the entry point document does not exist
- A concept is introduced in the path without definition and is not in the persona's domain vocabulary (comprehension gap)
- The path requires reading a Tier 2 document for a persona with executive or practitioner depth tolerance (forced escalation)
- The path ends without the persona being able to complete their task (incomplete terminal state)

**Concrete check:** For each persona, trace their reading path from entry point through each step. At each step, verify prerequisites are met from previous steps. Verify the final step enables the persona's task. Output: pass/fail per persona with specific gap locations.

### Gate 6: Coverage Matrix

**What it checks:** The persona-by-topic coverage matrix has no new gaps compared to the previous version, and any intentional gaps are documented.

**Passing looks like:**
- Every cell in the matrix either contains a document/section reference or is explicitly marked as "N/A — [reason]"
- No cells that were previously filled are now empty (regression)
- New personas or topics added since the last version have coverage assigned

**Failure triggers:**
- A previously filled cell is now empty (a persona lost coverage of a topic)
- A new persona has no coverage entries
- A cell is empty with no documented justification

**Concrete check:** Compare the current coverage matrix against the previous version (from the last committed version). Flag any cell that went from filled to empty. Flag any new row (persona) or column (topic) with empty cells. Output: list of gaps with their locations and whether they are regressions or new.

## Review Triggers

Four types of events trigger a governance review. Each has a defined scope and cadence.

### Trigger 1: Substantive Root Change

**When it fires:** A root document receives a substantive or structural change (per the classification decision tree).

**Scope:** Run the root document change workflow (6-step checklist above). All six quality gates run on affected derivative sections.

**Cadence:** Event-driven — every time a substantive or structural root change lands.

### Trigger 2: New Persona Added

**When it fires:** A new persona is added to the persona registry.

**Scope:** Verify the new persona has an entry point, a complete reading path, and coverage matrix entries. Run Gate 5 (path completeness) and Gate 6 (coverage matrix) for the new persona. Check whether existing derivatives need new sections or augmented content for this persona.

**Cadence:** Event-driven — each time the persona registry changes.

### Trigger 3: External Standard Update

**When it fires:** An external standard referenced in the external dependency tracker releases a new version (e.g., framework update, STRIDE revision).

**Scope:** Identify all document sections referencing the standard (from the external dependency tracker's `referencing_sections` list). Classify the external change (cosmetic, clarification, substantive, structural) relative to how the document set uses the standard. If substantive: run impact trace and the root document change workflow starting from step 2. Update the `version_referenced` field in the external dependency tracker.

**Cadence:** Event-driven, but also checked during periodic reviews. For actively maintained standards (migration framework: quarterly), check proactively even if no notification was received.

### Trigger 4: Periodic Cadence Review

**When it fires:** On a scheduled cadence, independent of any specific change.

**Scope:** Full audit — consistency audit, derivation integrity check, path completeness verification, link integrity check, coverage matrix review. This catches drift that accumulates from clarification-level changes and gradual terminology migration.

**Cadence:** Quarterly for actively maintained document sets, or aligned with the external standard release cycle if one dominates (e.g., migration framework releases quarterly, so the periodic review follows the same schedule). For stable document sets with infrequent changes, semi-annual is sufficient.

**What the periodic review produces:** A unified defect list, triaged by priority. Priority 1 (fix immediately): stale claims, self-sufficiency failures, phantom content. Priority 2 (fix before next publish): broken links, lazy deferrals, path dead ends. Priority 3 (fix during maintenance): terminology drift, coverage gaps, convention violations, orphan sections.

## Procedure: Setting Up Governance for a New Document Set

Use this procedure when establishing governance over a document set for the first time — either a new set or an existing set being brought under governance.

**Prerequisites:** The document manifest exists (from the document set architecture competency). The persona registry exists (from reading path design). The claim and terminology registries exist, at least in minimum viable form.

**If artifacts are missing — minimum viable forms:** A minimum viable manifest needs: document `id`, `title`, `path`, `tier` (0/1/2), `role` (root/derivative), and `derives_from` for derivatives. A minimum viable persona registry needs: `id`, `name`, `task` (assess/implement/govern/understand), `depth_tolerance` (executive/practitioner/technical). A minimum viable claim registry needs: `id`, `claim` (the assertion text), `canonical_source` (root §section). A minimum viable terminology registry needs: `term`, `canonical_definition`, `canonical_source`. These can be bootstrapped directly — you do not need the full formats from sibling reference sheets to get started.

**Step 1: Assign ownership.**

For each document in the manifest:

- Root documents: assign a human owner. This is the person with domain authority over the content. Root ownership cannot be delegated to an LLM.
- Derivative documents: assign a steward. This can be a human or Claude. If Claude, the LLM-as-Steward Trust Model applies.

Record ownership in the manifest or a separate governance file committed alongside the manifest.

**Step 2: Create derivation recipes for all derivative sections.**

For each section of each derivative document, create a recipe with all required fields:

```
Source: [root-document §section]
Mode: distillation | translation | extraction
Target persona: [persona-id from the persona registry]
Self-sufficiency claim: [what the reader can DO after reading this section alone]
Required content: [list of specific content items that MUST appear inline]
Acceptable deferrals: [what may link to the parent for optional depth]
```

If a section uses hybrid mode (e.g., extraction + translation), declare primary and secondary modes.

**Step 3: Run baseline quality gates.**

Run all six quality gates on the current state of the document set. Record the results. This establishes the quality baseline — you know what passes and what needs fixing before any future changes.

For existing document sets being brought under governance: expect failures. The baseline audit is diagnostic, not gatekeeping. Prioritize fixes using the triage framework (Priority 1 first, then Priority 2, then Priority 3).

**Step 4: Establish review cadence.**

Choose a periodic review cadence:

- Quarterly if the document set is actively maintained or references frequently updated external standards
- Semi-annual if the document set is stable with infrequent changes
- Aligned with external standard release cycles if one standard dominates

Record the cadence decision and next review date.

**Step 5: Document the governance configuration.**

Create or update a governance file that records:

- Ownership assignments (who owns each document)
- The LLM-as-steward trust model applicability (which derivatives have Claude as steward)
- Review cadence and next scheduled review
- Location of all registries (claim, terminology, persona, external dependency)
- Location of derivation recipes

This file is committed alongside the manifest and registries.

**Step 6: Brief all owners and stewards.**

Each root owner should know: which derivatives depend on their root, the change classification system, and their obligation to notify stewards of substantive/structural changes. Each derivative steward should know: which root(s) they derive from, their derivation recipes, and the quality gates their output must pass.

**Anti-pattern: governance without recipes.** Setting up ownership and review cadence but not creating derivation recipes. Without recipes, quality gates cannot verify derivation integrity — gate 1 fails by definition. Recipes are the operational foundation; governance without them is bureaucracy without substance.

## Procedure: Running Quality Gates on LLM Output

This is the most important procedure in this reference sheet. Use it every time Claude produces or updates derivative content.

### When to Run

- After Claude drafts a new derivative section
- After Claude updates a derivative section in response to a root change
- After Claude updates a derivative section as a clarity improvement
- During periodic review, on all derivative sections maintained by Claude

### The Procedure

**Step 1: Collect Claude's output package.**

Claude must provide, for each updated section:

- The updated section text
- The derivation recipe that governed the update
- The quality gate self-assessment (Claude's own pass/fail determination for each gate)
- The deferral audit results (total count, acceptable count, lazy count, new-since-last-version count, removed-inline-content-replaced-by-deferrals count)
- Any judgment calls or ambiguities Claude encountered

If Claude does not provide all five items, request the missing items before proceeding. Incomplete output packages are themselves a governance failure.

**Step 2: Review the deferral audit first.**

This is the highest-risk area. Check:

- **Removed inline content replaced by deferrals:** If this number is greater than zero, the derivative got less self-sufficient. This is a regression regardless of Claude's classification of the deferrals. Investigate each instance.
- **Total deferral count trend:** Compare against the previous version. A rising deferral count across updates is a drift signal even if each individual deferral is acceptable.
- **Lazy deferral count:** If Claude reports zero lazy deferrals, this is the claim most worth verifying. Pick the deferral Claude rated as "most borderline acceptable" and check it yourself.

**Step 3: Select and perform the spot-check.**

Choose one section to check in detail using the selection criteria:

1. Most complex derivation (synthesis mode, multi-root, complex recipe)
2. Highest deferral count
3. Most affected by the triggering change

For the selected section, read both the derivative section and the corresponding root section side-by-side. Verify:

- Claims match (same counts, same categories, same qualifiers)
- Deferrals are genuinely optional (the section works without following them)
- Depth is appropriate for the target persona (not too technical, not too vague)
- No meaning has been lost, inverted, or invented

**Step 4: Verify claim fidelity against the claim registry.**

For every claim in the updated section(s) that has a claim registry entry, verify the derivative's version matches the canonical value. Pay special attention to:

- Quantitative claims (counts, percentages, named list lengths) — these are the easiest to verify and the most consequential when wrong
- Structural claims (models, taxonomies, frameworks) — check that the structure matches, not just the labels

**Step 5: Run the remaining gates.**

- **Consistency gate:** Check terminology against the terminology registry for the appropriate tier. Flag any Tier 2 vocabulary appearing in a Tier 0 or Tier 1 document.
- **Link integrity gate:** Verify all cross-document links in the updated section(s) resolve. Check that any sections linked TO from other documents have not been renamed.
- **Path completeness gate:** If the update added or removed content, verify that all persona reading paths passing through the updated section still work (no comprehension gaps, no forced escalation, terminal state still reachable).
- **Coverage matrix gate:** If the update changed what topics are covered, verify no coverage matrix cells went from filled to empty.

**Step 6: Record the gate results and disposition.**

Record the results of all six gates in the output:

```
Quality Gate Results: [document-id] [date]
Reviewer: [human reviewer name]

Gate 1 (Derivation Integrity): PASS | FAIL — [details]
Gate 2 (Self-Sufficiency): PASS | FAIL — [details]
Gate 3 (Consistency): PASS | FAIL — [details]
Gate 4 (Link Integrity): PASS | FAIL — [details]
Gate 5 (Path Completeness): PASS | FAIL — [details]
Gate 6 (Coverage Matrix): PASS | FAIL — [details]

Deferral Audit Summary:
  Total deferrals: [n]
  Acceptable: [n]
  Lazy: [n]
  Replaced inline content: [n]

Spot-Check Section: [section identifier]
Spot-Check Result: PASS | FAIL — [findings]

Disposition: APPROVED | REVISIONS REQUIRED — [list of required fixes]
```

If any gate fails or the spot-check reveals issues: return to Claude with specific defects. Claude fixes the defects and resubmits. The procedure repeats from Step 1 on the resubmitted content.

**Step 7: Approve or reject.**

- **All gates pass + spot-check clean:** Approve. The derivative update can be committed.
- **Any gate fails:** Reject with specific defects. Claude revises and resubmits.
- **Gates pass but spot-check reveals a concern:** Request targeted revision of the flagged section. Re-run the spot-check on the revised section.

### Worked Example: Running Gates on Claude's Executive Brief Update

**Context:** Architecture Section 3.2 changed from 13 to 14 failure modes. Claude updated executive-brief Section 2 (distillation mode).

**Step 1:** Claude provides the output package:

- Updated section text (85 words, now reflecting 14 failure modes with FM-14 Model Drift)
- Derivation recipe (source: architecture §3.2, mode: distillation, persona: executive sponsor, self-sufficiency claim: "executive can understand the risk landscape and decide on investment")
- Self-assessment: all gates pass
- Deferral audit: 1 deferral ("For the complete taxonomy, see the full discussion architecture §3.2"), classified as acceptable
- Judgment call: FM-14 categorized under generation failures (7 generation modes now, up from 6)

**Step 2:** Deferral audit review.

- Removed inline content replaced by deferrals: 0. Good — no regression.
- Total deferrals: 1 (same as previous version). Stable.
- Lazy deferrals: 0. The single deferral is "for the complete taxonomy" which is optional depth — the section includes the count, categories, and most consequential modes inline. Accept Claude's classification.

**Step 3:** Spot-check.

This section is the only one updated, so it is selected by default. Read the derivative section and architecture §3.2 side-by-side:

- Count matches: "14 distinct failure modes" — correct.
- Category breakdown matches: "specification failures (4 modes), generation failures (7 modes), integration failures (3 modes)" — correct (was 6 generation, now 7).
- FM-14 (Model Drift) correctly categorized as generation failure — verified against root.
- Most consequential modes still highlighted (G3 Security Blindspot, G6 Hallucinated API) — executive framing preserved.
- No meaning inverted or lost. Depth appropriate for Tier 0.

**Step 4:** Claim registry check.

- `failure-mode-count` claim: canonical is now "14 distinct failure modes." Derivative says "14 distinct failure modes." Match.
- Category sub-counts: 4+7+3=14. Arithmetic checks out.

**Steps 5-6:** Remaining gates pass (no terminology issues, links resolve, executive reading path still works, coverage matrix unchanged).

**Step 7:** Disposition: APPROVED.

## Anti-Patterns

### No Clear Ownership

**The pattern:** Documents maintained by whoever touches them. No assigned root owners or derivative stewards.

**Why it fails:** Without ownership, nobody is responsible for propagating root changes to derivatives. Derivatives go stale. When a consistency audit finally catches the drift, nobody knows who should fix it or whether the derivative or the root is the authoritative version.

**The fix:** Assign ownership in the manifest. Every document gets exactly one owner. Ownership is a named person or role, not "the team."

### Quality Gates Treated as Optional

**The pattern:** Gates exist on paper but are routinely skipped for "minor" updates. "We'll catch it in the next full audit."

**Why it fails:** Minor updates accumulate. Each one is individually harmless, but after twenty ungated updates, the derivative has drifted materially from its root. The periodic audit surfaces dozens of defects that each require context reconstruction to fix.

**The fix:** Gates are mandatory for every derivative update, including clarity improvements. The derivative-only change workflow has five steps — none are optional. If the gates are too slow, make them faster (better recipes, fewer false positives), not skippable.

### Silent Self-Approval by Claude

**The pattern:** Claude updates a derivative, runs quality gates, determines everything passes, and commits the change without surfacing results to the human.

**Why it fails:** Claude's most likely failure mode (lazy deferral) is the one Claude is least likely to catch in its own output. Self-approval eliminates the one check — human review — that compensates for Claude's systematic bias.

**The fix:** The LLM-as-steward trust model requires Claude to surface all quality gate results and the deferral audit to the human. The human disposition (approved / revisions required) must precede the commit.

### Governance Without Derivation Recipes

**The pattern:** Ownership is assigned, review cadence is set, but no derivation recipes exist. Quality gates cannot verify derivation integrity because there is no specification of what the derivative should contain.

**Why it fails:** Gate 1 (derivation integrity) checks content against the recipe. Without a recipe, the gate is meaningless — there is nothing to check against. The governance process becomes a review of vibes rather than a verification of requirements.

**The fix:** Create recipes before declaring governance operational. The "Setting Up Governance" procedure makes recipe creation Step 2, immediately after ownership assignment.

### Over-Governing Cosmetic Changes

**The pattern:** Every typo fix in a derivative goes through the full five-step workflow including self-sufficiency test and deferral audit.

**Why it fails:** The overhead erodes discipline. Stewards stop following the workflow because 90% of their changes do not warrant it. When a genuinely substantive change comes, the workflow is treated as another bureaucratic hoop rather than a quality safeguard.

**The fix:** Cosmetic changes to derivatives are exempt from the derivative-only change workflow. Claude can apply cosmetic fixes autonomously. The classification decision tree determines which changes require which workflow — use it.

### Phantom Content Disguised as Helpfulness

**The pattern:** A derivative steward (human or Claude) adds a recommendation, example, or insight not present in any root document. It seems helpful. It is phantom content.

**Why it fails:** The derivative now makes a claim that cannot be traced to a root. When someone questions the claim, there is no authoritative source to verify it against. The derivative has become a partial root document without root-level governance, review, or ownership.

**The fix:** Every claim in a derivative must trace to a root. If new content is genuinely needed, propose it to the root owner. It enters the root first, then propagates to the derivative. No exceptions.

## Cross-References

- **Document Set Architecture** — defines the manifest structure, derivation graph, depth tiers, and multi-root reconciliation patterns that governance references when assigning ownership and tracing change impact
- **Reading Path Design** — defines the persona registry and path completeness contract that governance checks via Gate 5 (path completeness) and Trigger 2 (new persona added)
- **Content Derivation Discipline** — defines derivation modes, the self-sufficiency test, derivation recipes, and the deferral audit that governance uses as the foundation for Gates 1 and 2
- **Cross-Document Consistency** — defines the terminology registry, claim registry, consistency audit process, and defect record format that governance uses for Gate 3 (consistency) and Gate 4 (link integrity)
- **Document Evolution** — defines the change classification system, impact trace procedure, and git integration patterns that governance uses to determine which workflow applies and which derivative sections are affected
