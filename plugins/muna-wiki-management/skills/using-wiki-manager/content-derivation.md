# Content Derivation Discipline

## Overview

Content derivation is the act of producing a new document section from a parent document section at reduced depth, for a different audience, without requiring the reader to consult the parent. This is the single most important discipline in multi-document management because it is the discipline most likely to fail — and when it fails, the entire derivative document becomes a signpost to somewhere else rather than a useful document in its own right.

The breaking change is specific and predictable: you will write "see the full architecture document for details" instead of synthesizing the relevant content. You will do this because it is faster, because it feels more accurate (you are pointing to the authoritative source), and because it satisfies a surface-level reading of the requirement ("the reader can find the information"). But a derivative document that sends the reader to the parent is not a derivative — it is an index. The reader already has an index. They need a document that works without the parent open.

This reference sheet gives you the complete operational toolkit: three derivation modes, the self-sufficiency test, derivation recipes, deferral auditing, and procedures for writing new derivatives and updating existing ones after root changes.

## When to Use

Load this discipline when:

- You are writing a new derivative document section from a root document
- You are updating a derivative section because the root changed
- You are auditing a derivative for lazy deferrals or self-sufficiency failures
- You are reviewing someone else's derivative work (including your own prior output)
- A user says: "summarize this for executives," "translate this into control language," "extract the relevant parts for developers," "update the brief to match the new architecture document"

Do NOT use this discipline when:

- You are writing a root document (root documents do not derive from anything)
- You are doing a terminology or link audit (use cross-document consistency instead)
- You are designing the document set structure (use document set architecture instead)

## Key Terminology

This discipline references several structural artefacts from sibling reference sheets. Minimal definitions here so you can follow procedures without leaving this document:

- **Depth Tiers:** Documents are classified by depth. **Tier 0** = executive (2-page decision documents), **Tier 1** = practitioner (8-15 page working documents for specific roles), **Tier 2** = reference (full technical analysis — root documents). A Tier 1 document must be actionable without requiring the reader to open a Tier 2 document.
- **Document Manifest:** A structured project file declaring every document in the set, its tier, role (root or derivative), `derives_from` field (which root(s) it derives from), and `derivation_mode`. This is where derivation recipes get their structural context.
- **Derivation Graph:** A directed acyclic graph mapping section-to-section lineage between roots and derivatives. Used to trace impact when a root section changes.
- **Persona Registry:** A project file listing each role the document set serves, with fields: `id`, `name`, `task` (assess/implement/govern/understand), `depth_tolerance` (executive/practitioner/technical), `domain_vocabulary` (terms they know), and `time_budget`. Persona entries drive derivation mode selection.
- **Claim Registry:** A project file tracking factual claims that propagate across documents. Each entry has a `claim` (the canonical phrasing), `canonical_source` (root section), and `propagation` list (every derivative section that references it). Used during claim fidelity checks.
- **Terminology Registry:** A project file tracking key terms with canonical definitions, tier-appropriate variants, and prohibited conflations. Consulted during derivation to ensure term usage matches the target tier.

Full formats for all of these are in the sibling reference sheets. The definitions above are sufficient to follow every procedure in this document.

## The Three Derivation Modes

Every derivative section operates in exactly one primary mode. Choosing the wrong mode produces content that is technically derived but functionally useless — a compressed version when the reader needed a translated one, or a curated excerpt when they needed the argument restated for their vocabulary.

### Mode 1: Distillation

**What it does:** Preserves the parent's argument but compresses it. Same meaning, fewer words. The structure of the argument survives; the evidence and elaboration are reduced.

**When to use:** The target persona shares the parent's vocabulary and framing but has less time. They need the conclusion and enough reasoning to trust it, not the full evidentiary chain.

**Typical targets:** Executive briefs, management summaries, decision documents.

**Compression ratio:** A distillation typically reduces content to 15-25% of the parent section's length. If you are above 40%, you are not distilling — you are lightly editing. If you are below 10%, you are probably dropping claims, not compressing them.

**Worked example — before and after:**

Source (root document, architecture Section 3.2, 280 words):

> The taxonomy identifies 13 distinct breaking changes organized into three categories: specification failures (4 modes), generation failures (6 modes), and integration failures (3 modes). Specification failures occur when the AI system receives ambiguous, incomplete, or contradictory instructions. The four specification breaking changes are: S1 (Ambiguous Intent), where natural language prompts permit multiple valid interpretations and the system selects the wrong one; S2 (Missing Constraint), where the prompt omits a requirement that a human developer would infer from context; S3 (Contradictory Requirements), where the prompt contains conflicting instructions and the system silently resolves the conflict rather than flagging it; and S4 (Scope Creep), where the system interprets the prompt as requiring changes beyond the explicitly requested scope.
>
> Generation failures occur during the code production phase. The six generation breaking changes are: G1 (Syntax-Valid Semantic Error), where the generated code compiles but does not do what was requested; G2 (Stale Pattern), where the system uses deprecated APIs or outdated idioms; G3 (Security Blindspot), where the generated code introduces vulnerabilities not present in the prompt; G4 (Performance Anti-Pattern), where the code is functionally correct but unacceptably slow; G5 (Incomplete Implementation), where the system produces partial code without indicating what is missing; and G6 (Hallucinated API), where the system invokes functions or libraries that do not exist.
>
> Integration failures occur when AI-generated code interacts with the existing codebase. The three integration breaking changes are: I1 (Type Mismatch), I2 (State Corruption), and I3 (Dependency Conflict).

Derived (executive brief Section 2, distilled to 85 words — 30% of original):

> The taxonomy identifies 13 distinct breaking changes in three categories: specification failures (4 modes — the AI misunderstands what is being asked), generation failures (6 modes — the AI produces code with defects ranging from semantic errors to hallucinated APIs), and integration failures (3 modes — the AI-generated code conflicts with the existing codebase). The most consequential for organizational risk are G3 (Security Blindspot) and G6 (Hallucinated API), which can introduce vulnerabilities or non-functional dependencies silently.

**What makes this a good distillation:**
- The 13-mode claim is preserved exactly (not "about a dozen" or "numerous")
- The three-category structure survives
- Specific mode codes are retained so the reader can find them later if desired
- The elaboration per mode is compressed to a clause, not dropped entirely
- The executive-relevant insight (which modes matter most for risk) is added — this is acceptable in distillation because it helps the persona act

**What would make this a bad distillation:**
- "The taxonomy identifies multiple breaking changes across several categories" — this drops the count and the structure
- "See architecture Section 3.2 for the complete taxonomy" — this is a deferral, not a distillation
- Listing all 13 modes with full descriptions — this is not compression, it is copying

### Mode 2: Translation

**What it does:** Re-expresses the parent's content for a different vocabulary and frame of reference. The meaning is preserved but the language changes to match how the target persona thinks and talks about the domain.

**When to use:** The target persona has domain expertise but in a different frame. An SRE Lead understands risk, controls, and compliance — not necessarily ML training pipelines or prompt engineering. The content must be restated in their operational language.

**Typical targets:** Role-specific assessments, compliance mappings, operational guides.

**Translation requires a vocabulary bridge.** Before translating, explicitly identify the terms that change:

| Parent term | Target persona term | Justification |
|---|---|---|
| breaking change | coverage gap | SRE Lead frames defects as gaps in the control environment |
| specification failure | requirements risk | SRE Lead maps to risk categories, not failure taxonomies |
| generation failure | output defect | Aligns with defect management vocabulary |
| hallucinated API | phantom dependency | SRE Lead understands dependency risk, not ML hallucination |

**Worked example — before and after:**

Source (root document, architecture Section 4.1, 150 words):

> Breaking change G3 (Security Blindspot) occurs when the AI code generator produces code containing security vulnerabilities that were not present in the prompt and were not requested. Common manifestations include: SQL injection vectors in database queries, hard-coded credentials in configuration handling, path traversal vulnerabilities in file operations, and cross-site scripting openings in web output generation. The root cause is that the model optimizes for functional correctness rather than security properties, and its training data includes both secure and insecure code patterns without reliable differentiation.

Derived (ops-assessment Section 1.3, translated to 120 words):

> **Coverage Gap: Undetected Security Defects in AI-Generated Code.** AI code generators can introduce security vulnerabilities — including injection vectors, credential exposure, path traversal, and XSS — without any indication in the prompt or output that a vulnerability exists. This gap exists because the generation process optimizes for functional correctness, not security properties. Existing code review controls designed for human-authored code may not catch these defects because they follow patterns the model learned from insecure training examples, not patterns a human developer would typically produce. **Residual risk:** High, unless code review workflows are specifically augmented to inspect AI-generated output for vulnerability classes that differ from typical human coding errors. **Recommended control:** Dedicated AI-output security scanning (see Recommendations Section 5.2).

**What makes this a good translation:**
- "Breaking change G3" becomes "Coverage Gap" — SRE Lead language
- The technical detail (SQL injection, XSS, etc.) is preserved because an SRE Lead needs to understand what the vulnerabilities actually are
- "Root cause" becomes an explanation of why existing controls might miss it — operational framing
- A residual risk statement and recommended control are added — this is the SRE Lead's action frame
- The section stands alone: an SRE Lead can read this and act on it without opening the architecture document

**What would make this a bad translation:**
- Keeping the "G3" code as the primary identifier without explaining what it means in control language
- "The architecture document identifies a security-related breaking change; see Section 4.1 for details" — lazy deferral
- Translating so aggressively that the specific vulnerability types are lost ("various security issues")

### Mode 3: Extraction

**What it does:** Selects a subset of the parent's content relevant to a specific role or task. The content is largely unchanged in language but curated — only the parts that matter for this persona are included.

**When to use:** The target persona needs specific items from a large parent document, and the value is in the curation (knowing what to skip), not in compression or restatement.

**Typical targets:** Developer checklists, implementation guides, audit scopes.

**Extraction requires a relevance filter.** Before extracting, explicitly define what the persona needs:

- **Include:** Content directly actionable by this persona in their daily work
- **Exclude:** Content about roles, processes, or decisions outside their scope
- **Edge cases:** Content that is contextually relevant but not directly actionable — include a one-sentence summary with a pointer to the full treatment (this is an acceptable deferral because the content is supplementary, not core)

**Worked example — before and after:**

Source (root document, architecture Sections 3.2 and 4.1-4.3, 1200 words across multiple sections covering all 13 breaking changes, their root causes, organizational impacts, and recommended controls):

Derived (developer inventory, Section 2 — extraction of code-review-relevant breaking changes, 200 words):

> **Breaking Changes Detectable During Code Review**
>
> Of the 13 catalogued breaking changes, 7 are detectable during standard code review if reviewers know what to look for. The remaining 6 require runtime testing, integration testing, or specification review and are outside code review scope.
>
> **Review for these during AI-generated code inspection:**
>
> | Mode | What to look for | Common miss pattern |
> |---|---|---|
> | G1: Semantic Error | Code compiles but logic does not match the stated requirement | Reviewer checks syntax, misses intent |
> | G2: Stale Pattern | Deprecated API calls, outdated idioms | Reviewer unfamiliar with current API version |
> | G3: Security Blindspot | Injection vectors, hardcoded credentials, path traversal, XSS | Reviewer focuses on functionality, skips security |
> | G5: Incomplete Implementation | Missing error handling, partial feature coverage, TODO placeholders | Reviewer sees working happy path, misses gaps |
> | G6: Hallucinated API | Calls to functions or libraries that do not exist | Reviewer does not verify import targets |
> | I1: Type Mismatch | Parameter types incompatible with calling code | Reviewer reads function in isolation |
> | I3: Dependency Conflict | Version conflicts with existing project dependencies | Reviewer does not check dependency tree |
>
> **Out of scope for code review** (handle via other controls): S1-S4 (specification failures — caught during requirements review), G4 (performance — caught during performance testing), I2 (state corruption — caught during integration testing).

**What makes this a good extraction:**
- The 7-of-13 framing gives the developer context about what they are and are not responsible for
- Each mode has a concrete "what to look for" and "common miss pattern" — actionable
- The "out of scope" section prevents the developer from worrying about things that are not their problem
- No content is rewritten or reframed; it is curated and organized for the code review workflow

**What would make this a bad extraction:**
- Including all 13 breaking changes with "not applicable to code review" notes on 6 of them — the curation value is lost
- "See the architecture document for the complete breaking change taxonomy" without any specific modes listed — this is not extraction, it is a redirect

## Hybrid Modes: Primary + Secondary

Some derivative sections genuinely operate in two modes. This is legitimate when a section must both select content and restate it. The rule: if you find yourself needing three modes, the section is doing too much and should be split.

**Declaring hybrid mode:** State the primary mode first (this determines the self-sufficiency standard), then the secondary:

```
Mode: extraction (primary) + translation (secondary)
```

**Worked example — Ops Assessment Section 2: Relevant Breaking Changes**

This section must *extract* the breaking changes relevant to an SRE Lead's control environment (not all 13 are relevant to security governance) and *translate* them into coverage-gap language. Extraction is primary because the first job is curation; translation is secondary because the curated content must be restated.

Source: architecture Sections 3.2 and 4.1-4.3 (all 13 breaking changes).

Derived (180 words):

> **Coverage Gaps Arising from API Migration**
>
> Of the 13 catalogued breaking changes, 5 represent direct coverage gaps in your security governance posture:
>
> | Coverage Gap | Source Breaking Change | Governance Impact |
> |---|---|---|
> | Undetected vulnerability injection | G3: Security Blindspot | Existing secure coding controls do not cover migration-specific compatibility patterns |
> | Phantom dependency introduction | G6: Hallucinated API | Software composition analysis may not flag non-existent libraries that the AI references |
> | Silent contract expansion | S4: Scope Creep | Change management controls may not detect AI-initiated changes outside the requested scope |
> | Incomplete implementation shipped | G5: Incomplete Implementation | QA gates designed for human-authored code may miss partial AI output |
> | Stale security patterns | G2: Stale Pattern | The AI may generate code using deprecated security mechanisms your policy has retired |
>
> The remaining 8 breaking changes affect development efficiency and code quality but do not represent direct gaps in the security control environment. They are addressed in the Developer Inventory and the Recommendations document.

**Why this works as a hybrid:** The extraction (5 of 13 selected) and the translation (breaking change language to coverage gap language) are both visible. An SRE Lead can read this section, understand which gaps affect them, and take action — without opening the architecture document.

## The Self-Sufficiency Test

This is the quality gate that determines whether a derivative section actually works. Run it on every section you write or update.

### The Procedure

**Step 1: Hide the parent.** Pretend the root document does not exist. You have only the derivative section in front of you.

**Step 2: Persona simulation.** Read the section as the target persona. Ask: can this persona do their job based on what is written here? Not "understand the topic exists" — actually perform their task.

Concrete persona-task pairs and what "do their job" means:

| Persona | Task | "Can do their job" means... |
|---|---|---|
| SRE Lead | Assess coverage gaps | Can identify which gaps affect their organization and prioritize remediation |
| Executive sponsor | Make a funding decision | Can decide whether to invest in the recommended controls based on the risk and cost framing |
| Developer | Review AI-generated code | Can inspect a pull request and catch the breaking changes listed |
| Governance lead | Draft a policy update | Can write a policy clause based on the recommendations without needing more context |
| Assessor | Evaluate compliance | Can score an organization against the controls without referencing the root |

**Step 3: Claim grounding check.** For every claim in the section, ask: is there enough evidence or reasoning here for the reader to trust this conclusion? A claim without grounding is either phantom content (you invented it) or a lazy distillation (you compressed away the justification).

**Step 4: Deferral audit.** Flag every phrase that points the reader to another document. Classify each one (see the Deferral Audit Table below).

### Passing vs. Failing — Complete Worked Example

**Context:** Ops Assessment Section 3, derived from architecture Section 6.1. Derivation mode: translation. Target persona: SRE Lead. Self-sufficiency claim: "SRE Lead can identify which migration safeguards need augmentation for API migration risk."

**FAILING version (78 words):**

> **Section 3: Migration Safeguard Augmentation**
>
> The full architecture document identifies several migration safeguards that require augmentation to address API migration risks. The key controls are discussed in architecture Section 6.1, which maps each breaking change to the relevant migration safeguard category. SRE Leads should review this mapping to determine which controls in their organization need updating. For the complete analysis including specific migration safeguard identifiers and recommended augmentation approaches, see architecture Section 6.1.

**Why it fails:**
- The SRE Lead cannot identify which controls need augmentation — the specific controls are not listed
- "Several migration safeguards" is vague — how many? Which ones?
- "See architecture Section 6.1" appears twice — the entire section is a signpost
- The persona cannot do their job (identify and prioritize safeguard augmentation) without opening the architecture document
- Every sentence either defers or summarizes at a level too vague to act on

**PASSING version (210 words):**

> **Section 3: Migration Safeguard Augmentation for API Migration Risk**
>
> Six migration safeguard categories require augmentation to address risks introduced by API migration. The current migration framework (March 2025) assumes manually managed schemas; these augmentations extend existing safeguards to cover migration-specific breaking change patterns.
>
> | Migration Safeguard Category | Current Scope | Required Augmentation | Priority |
> |---|---|---|---|
> | MS-0971: Schema Validation | Manual schema validation | Add automated change inspection protocols covering BC-3 (schema regression) and BC-6 (phantom endpoints) | Critical |
> | MS-1425: Compatibility Review | Peer review of API changes | Extend review checklists with 7 migration-specific compatibility checks (see Developer Inventory §2 for the checklist) | Critical |
> | MS-1467: Dependency Verification | Known dependency tracking | Add verification that referenced API endpoints actually exist (BC-6 detection) | High |
> | MS-0843: Change Management | Manual changes | Add scope verification for automated changes to detect silent contract expansion (BC-4) | High |
> | MS-1228: Compatibility Scanning | Known vulnerability databases | Extend scanning to cover migration-specific compatibility patterns not in standard CVE databases | Medium |
> | MS-0264: Migration Assessment | Periodic assessment scope | Include API migration in assessment scope with dedicated test cases | Medium |
>
> Prioritization is based on likelihood of exploitation and gap between current safeguard coverage and AI-specific risk. The architecture document provides extended justification for each augmentation in Section 6.1 for readers who want the full evidentiary chain.

**Why it passes:**
- The SRE Lead can see exactly which 6 safeguard categories need attention
- Specific migration safeguard identifiers are provided — the SRE Lead can map these to their own environment
- Each augmentation is described concretely, not vaguely
- Priority is assigned — the SRE Lead can triage
- The one deferral ("The architecture document provides extended justification... in Section 6.1") is acceptable: it points to optional depth, not required content. The SRE Lead can act without it.
- 210 words to do the job that 78 words failed at — because the 78 words contained no substance

## The Deferral Audit Table

Every phrase that points the reader elsewhere must be classified. This table provides 12 example phrases with classification and reasoning. Use this as a calibration tool when auditing your own output.

| # | Phrase | Classification | Reasoning |
|---|---|---|---|
| 1 | "See architecture Section 6.1 for the complete analysis" | **LAZY** | The "complete analysis" contains the specific controls the reader needs. This defers core content. |
| 2 | "The architecture document provides extended justification for each augmentation in Section 6.1" | **ACCEPTABLE** | The justification is optional depth. The augmentations themselves are already listed inline. |
| 3 | "As described in the specification" | **LAZY** | "As described" means the description is elsewhere. The reader needs the description here. |
| 4 | "For implementation details, see the Developer Inventory Section 4" | **ACCEPTABLE** | This is a cross-document reference to a different persona's document. The SRE Lead does not need implementation details. |
| 5 | "The full taxonomy is documented in architecture Section 3" | **LAZY** | If the taxonomy is relevant to this section, its relevant parts must appear here. "The full taxonomy" signals that none of it was synthesized. |
| 6 | "The specification defines four authority tiers (see specification Part I Section 2 for the formal definitions)" | **BORDERLINE — fix it.** | The tier count is inline, but "formal definitions" might be needed. If the reader needs to know what each tier means to act, this is lazy. If the names and one-line descriptions are already present and only the formal prose is deferred, it is acceptable. |
| 7 | "Refer to the Terminology Registry for canonical definitions" | **ACCEPTABLE** | The registry is a reference tool, not a content source. Readers consult it when they encounter an unfamiliar term, not as part of a reading path. |
| 8 | "The root cause analysis (architecture Section 4) explains why these patterns emerge" | **ACCEPTABLE** | Root cause analysis is explanatory depth, not actionable content. The derivative section already states what the patterns are and what to do about them. |
| 9 | "Details on implementation are beyond the scope of this assessment" | **ACCEPTABLE** | Explicit scope boundary. Honest about what this document does not cover. Not a deferral — a declaration. |
| 10 | "The architecture document identifies 13 breaking changes (see architecture Section 3.2)" | **LAZY** | The parenthetical "see" replaces listing even the high-level categories. The reader gets a count and a pointer, not content. Compare with the passing example: "13 distinct breaking changes in three categories: specification failures (4), generation failures (6), integration failures (3)." |
| 11 | "Based on the risk framework established in the architecture document" | **LAZY** | "Established in the architecture document" means the framework is not here. If the derivative uses the framework's conclusions, the framework's structure must be at least summarized inline. |
| 12 | "Adapted from the methodology described in architecture Section 5" | **BORDERLINE — probably lazy.** | "Adapted from" suggests the methodology is relevant to understanding this section. If the adaptation is explained here, acceptable. If the reader must consult Section 5 to understand the method, lazy. |

### How to Use This Table

When you audit your own output:

1. Search for these trigger phrases: "see," "refer to," "as described in," "for details," "the full," "for the complete," "as documented in," "established in," "described in," "based on the"
2. For each match, classify it using the reasoning pattern above
3. For every LAZY deferral: replace the phrase with inline content. Do not soften it to ACCEPTABLE by rewording — add the substance that is missing
4. For every BORDERLINE deferral: default to fixing it. Borderline means you are uncertain, which means the reader might need the content

## The Derivation Recipe

A derivation recipe is a structured template that specifies exactly what a derivative section must contain. It is the operational tool that prevents laziness: when you have a recipe, you cannot skip content without violating a concrete, checkable requirement.

### Recipe Template

```
DERIVATION RECIPE
=================
Derivative:     [document-id §section-number: section title]
Source:         [parent-document-id §section-number: section title]
Mode:           [distillation | translation | extraction | hybrid: primary + secondary]
Target persona: [persona-id — the role this section is written for, with their depth tolerance and domain vocabulary]
Vocabulary bridge: [list key term mappings if mode is translation, or "N/A"]

Self-sufficiency claim:
  [One sentence: what the reader can DO after reading this section alone]

Required content (MUST appear inline):
  1. [Specific content item — e.g., "the 6 migration safeguard categories requiring augmentation"]
  2. [Specific content item — e.g., "priority ranking with rationale"]
  3. [Specific content item]

Acceptable deferrals (MAY link to parent for optional depth):
  1. [Specific content item — e.g., "extended justification for each safeguard augmentation"]
  2. [Specific content item]

Prohibited deferrals (MUST NOT defer — these are common laziness traps for this section):
  1. [Specific content item — e.g., "the safeguard category names and identifiers"]
  2. [Specific content item — e.g., "the augmentation descriptions"]
```

### Complete Filled-In Recipe Example

```
DERIVATION RECIPE
=================
Derivative:     ops-assessment §3: Migration Safeguard Augmentation for API Migration Risk
Source:         architecture §6.1: Mapping API Migration Breaking Changes to Migration Safeguards
Mode:           translation
Target persona: sre-lead
Vocabulary bridge:
  - "breaking change" → "coverage gap"
  - "specification failure" → "requirements risk"
  - "generation failure" → "output defect"
  - "hallucinated API" → "phantom dependency"
  - "root cause" → "gap driver"

Self-sufficiency claim:
  The SRE Lead can identify which migration safeguards in their organization need
  augmentation for API migration risk, understand what each augmentation requires,
  and prioritize implementation based on risk severity.

Required content (MUST appear inline):
  1. The 6 migration safeguard categories requiring augmentation, with migration safeguard identifiers
  2. For each family: current scope, required augmentation, and priority level
  3. Priority rationale (what drives Critical vs High vs Medium ranking)
  4. The connection between specific breaking changes and specific controls
  5. Scope statement: what this section covers and what it does not

Acceptable deferrals (MAY link to parent for optional depth):
  1. Extended evidentiary justification for each augmentation mapping
  2. Historical analysis of how each breaking change was discovered
  3. Cross-references to external migration framework documents
  4. Comparison with non-framework standards (OpenAPI Specification, JSON Schema)

Prohibited deferrals (MUST NOT defer — common laziness traps):
  1. The migration safeguard category names and identifiers (these ARE the section's payload)
  2. The augmentation descriptions (the SRE Lead needs these to act)
  3. The breaking change to control mapping (the core translation this section performs)
  4. Priority ranking (the SRE Lead cannot triage without it)
```

### Why Prohibited Deferrals Matter

The "prohibited deferrals" field exists because certain content items are chronic laziness targets. You will be tempted to write "see architecture Section 6.1 for the specific migration safeguard mappings" because listing six safeguard categories with their identifiers, augmentations, and priorities is more work than writing a sentence that points there. The prohibited deferrals list makes this temptation visible and classifies giving in to it as a defect.

When writing a recipe, ask yourself: "If I were tired and taking shortcuts, which content items would I replace with a link?" Those are your prohibited deferrals.

## Procedure: Writing a New Derivative Section

Follow this procedure end-to-end when creating a derivative section for the first time. Do not skip steps. Do not reorder steps.

**Step 1: Write the derivation recipe.**

Before writing any prose, fill in the recipe template completely. This forces you to decide what content is required before you start composing, which prevents the common failure of discovering mid-draft that you are deferring essential content.

**Step 2: Build the vocabulary bridge (if translation mode).**

Create the term-mapping table. For each parent term that appears in the source section, determine the target persona's equivalent. If no equivalent exists, decide whether to introduce the parent term with a parenthetical definition or to coin a descriptive equivalent. Do not silently use the parent's vocabulary — the whole point of translation is that the language changes.

**Step 3: Extract source claims.**

Read the parent section and list every factual claim, quantitative statement, and structural assertion. This is your raw material. Check each claim against the Claim Registry if one exists — use the canonical form, not a paraphrase that might introduce drift.

**Step 4: Filter claims by recipe.**

Using the recipe's "required content" list, identify which source claims map to each required item. Claims that do not map to any required item are candidates for acceptable deferral or omission. Claims that map to a prohibited deferral must appear inline no matter what.

**Step 5: Compose the section.**

Write the prose in the target persona's vocabulary, at the target tier's depth. For each required content item, verify it appears in your draft. Use the self-sufficiency claim as a test: does the draft deliver what it promises?

**Step 6: Run the self-sufficiency test.**

Apply the four-step test (hide the parent, persona simulation, claim grounding, deferral audit). If any step fails, revise before proceeding.

**Step 7: Record the deferral count.**

Count every deferral in the section. Classify each as acceptable or lazy. Report the counts even if all deferrals are acceptable — this gives the human reviewer visibility into deferral density. A section with zero deferrals is ideal. A section with 1-2 acceptable deferrals is normal. A section with 3+ deferrals warrants a second look at whether the section is actually self-sufficient.

**Step 8: Peer the recipe against the output.**

Walk through the recipe line by line:
- Every required content item present? Check.
- Every prohibited deferral avoided? Check.
- Self-sufficiency claim delivered? Check.
- Mode correctly applied (distillation compresses, translation re-vocabularies, extraction curates)? Check.

If any check fails, revise. Do not ship a section that fails its own recipe.

## Procedure: Updating a Derivative After Root Change

When a root document section changes, every derivative section that derives from it must be reviewed and potentially updated. This procedure prevents both stale derivatives (the root changed but the derivative did not) and over-propagation (changing derivatives that are not actually affected).

**Step 1: Classify the root change.**

Determine what kind of change occurred:

| Change type | Definition | Propagation requirement |
|---|---|---|
| Cosmetic | Typo fixes, formatting, whitespace | None — derivative is unaffected |
| Clarification | Same claim, better explanation | Check derivative for stale phrasing — the claim has not changed, but the derivative's phrasing might benefit from the improved clarity |
| Substantive | A claim, count, recommendation, or structural assertion changed | Mandatory update of all derivative sections in the claim's propagation list |
| Structural | Sections added, removed, reorganized | Mandatory review of derivation recipes, reading paths, and cross-document links |

**Step 2: Identify affected derivative sections.**

Use the derivation graph (from the document manifest) and the claim registry to find every derivative section that draws from the changed root section. For substantive changes, also check the claim registry's propagation list.

**Step 3: For each affected section, retrieve the derivation recipe.**

The recipe tells you what content is required. Compare the root change against the recipe's required content items. If a required content item's source changed, the derivative must change.

**Step 4: Update the derivative section.**

Re-derive the section following the "Writing a New Derivative Section" procedure, but starting from Step 3 (extract source claims) since the recipe and vocabulary bridge already exist. Compare your updated draft against the previous version:

- If the update changes a factual claim, verify the new claim matches the root exactly
- If the update is a phrasing improvement only, verify no meaning changed
- If a new claim was added to the root, decide whether it belongs in the derivative (check the recipe's required content and the persona's needs)

**Step 5: Deferral regression check.**

Compare the derivative's deferral count before and after the update. If the count increased, investigate: did you introduce a new lazy deferral? New deferrals during updates are the highest-risk laziness pattern because the pressure to "just point to the updated root section" is strongest when you are making changes to match a root change.

**Step 6: Run the self-sufficiency test.**

Same four-step test as for new sections. If the update broke self-sufficiency (common when root structural changes cascade), revise.

**Step 7: Update registries.**

If the root change altered a term or claim tracked in the registries, update the registry entries. If the root change introduced a new term or claim that meets the registry granularity thresholds (appears in 3+ documents, is quantitative, or has a prohibited conflation), add it to the appropriate registry.

## Anti-Patterns: Vivid Failures You Will Recognize

These are not generic warnings. These are specific patterns that occur repeatedly when LLMs maintain derivative documents. Each one describes what the failure looks like in practice so you can recognize it in your own output.

### Anti-Pattern 1: The Signpost Section

**What it looks like:** A derivative section that is 40-80 words long, consists of 2-3 sentences, and contains 1-2 cross-references to the parent. The sentences are connecting prose ("The architecture document addresses this topic in detail") rather than content.

**Example:**

> **Section 4: Enforcement Model**
>
> The companion specification defines a four-tier enforcement model for AI code assurance. The model establishes graduated levels of automated oversight based on code criticality. For the complete enforcement model including tier definitions, escalation triggers, and implementation guidance, see specification Part I, Section 2.

**Why it happens:** You processed the source section, decided it was complex, and chose to point to it rather than synthesize it. This feels responsible ("I'm not going to oversimplify the enforcement model") but it produces a section that does nothing.

**The fix:** Delete the signpost. Replace it with a section that names the four tiers, gives a one-line description of each, and explains what the persona needs to know about the enforcement model to do their job. If the section is for an SRE Lead, the tiers need to be framed as governance controls. If for a developer, as gating criteria for their code.

### Anti-Pattern 2: The Phantom Elaboration

**What it looks like:** A derivative section that introduces claims, frameworks, or recommendations not present in the parent document. The new content may be accurate, but it is not derived — it is invented.

**Example:** An ops assessment section that recommends "implementing a quarterly AI code audit program with rotating assessors," when the root document says nothing about audit cadence or assessor rotation.

**Why it happens:** You know things about security governance beyond what the architecture document says, and you helpfully add them. The problem is that the derivative's authority comes from the root. Phantom content undermines this authority because the reader cannot trace the recommendation to its source.

**The fix:** If the additional content is genuinely useful, flag it explicitly: "Note: the following recommendation extends beyond the architecture document's scope based on standard security governance practice." Better yet, suggest to the human that the root document should be updated to include this content — then derive from the updated root.

### Anti-Pattern 3: The Meaning-Inverting Compression

**What it looks like:** A distillation that changes the parent's claim by dropping a qualifier, inverting a conditional, or losing a caveat.

**Examples:**
- Parent: "The taxonomy identifies 13 breaking changes, though the authors note this is not exhaustive." Derivative: "There are 13 breaking changes." (The caveat about non-exhaustiveness is load-bearing — it tells the reader the taxonomy may grow.)
- Parent: "Organizations with existing secure coding practices may find that 4 of the 6 generation failure controls are already partially addressed." Derivative: "Existing secure coding practices address most generation failures." (Changed "partially addressed" to "address" and "4 of 6" to "most" — the derivative overstates coverage.)
- Parent: "If the AI system is used for security-critical code, G3 controls should be mandatory." Derivative: "G3 controls should be mandatory." (Dropped the conditional — now it applies to all code, not just security-critical code.)

**Why it happens:** Compression inherently loses detail, and you default to the simpler statement. Qualifiers, conditionals, and caveats are the first casualties because they make sentences longer and more complex.

**The fix:** After compressing, compare every claim in the derivative against the source. For each claim, ask: "Is this still true in exactly the same circumstances as the parent's claim?" If you removed a qualifier, determine whether the qualifier matters for the target persona. If it does, keep it. If it does not (e.g., an executive does not need to know the taxonomy might grow), document the omission in the recipe as a deliberate choice, not an accidental loss.

### Anti-Pattern 4: The Wrong-Depth Derivative

**What it looks like:** A derivative section written at a depth appropriate for a different tier. A Tier 0 executive brief that reads like a Tier 1 practitioner document (too detailed, too much jargon). A Tier 1 assessment that reads like a Tier 0 brief (too vague, not actionable).

**Example:** An executive brief section that includes a table of all 13 breaking changes with technical descriptions, when the executive needs only the three categories and the headline risk.

**Why it happens:** You derived faithfully from the parent but forgot to check the tier contract. The content is accurate and non-deferring, but it is at the wrong altitude for the persona's time budget and depth tolerance.

**The fix:** Check the persona's depth tolerance and time budget from the Persona Registry. An executive with a 5-minute time budget and "executive" depth tolerance should not be reading a 500-word table of breaking changes. Compress to the tier-appropriate level: categories and counts, not individual modes.

### Anti-Pattern 5: The Registry-Stale Derivative

**What it looks like:** A derivative section that uses the correct claims from the root document as it existed when the section was last updated, but the root has since changed. The derivative is internally consistent but externally stale.

**Example:** The ops assessment says "13 breaking changes" but the architecture document was updated to "15 breaking changes" after two new modes were added. The derivative is not wrong about what it says — it is wrong about what the root now says.

**Why it happens:** The root was updated without triggering the propagation procedure. No one checked the claim registry's propagation list.

**The fix:** This is a process failure, not a writing failure. The fix is to run the "Updating a Derivative After Root Change" procedure. The prevention is to run a consistency audit (claim fidelity check) after any root change, using the claim registry's propagation list as the work list.

### Anti-Pattern 6: The Comfortable Extraction

**What it looks like:** An extraction that includes everything from the parent because "it might all be relevant." The extraction adds no curation value — it is a copy of the parent with a different title.

**Why it happens:** You are uncertain about what to exclude, and inclusion feels safer than exclusion. Excluding something that turns out to be relevant feels like an error; including something irrelevant feels like thoroughness.

**The fix:** Apply the relevance filter explicitly. For each content item in the parent, ask: "Can the target persona act on this in their role?" If not, exclude it. The value of extraction is knowing what to skip. An extraction that skips nothing is just a copy.

## Derivation Mode Selection Guide

When you are uncertain which mode to use, this decision tree resolves it:

1. **Does the target persona share the parent's vocabulary?**
   - YES → Not translation. Go to question 2.
   - NO → Translation is primary. If you also need to select a subset, the hybrid is translation (primary) + extraction (secondary).

2. **Does the target persona need all the parent's content or a subset?**
   - ALL (but compressed) → Distillation.
   - SUBSET → Extraction. If the subset also needs compression, the hybrid is extraction (primary) + distillation (secondary).

3. **Sanity check:** Does the mode match the tier transition?
   - Tier 2 → Tier 0: Almost always distillation (massive compression needed)
   - Tier 2 → Tier 1 (same domain): Likely distillation or extraction
   - Tier 2 → Tier 1 (different domain): Likely translation
   - Tier 1 → Tier 0: Distillation

## Quality Metrics for Derivative Sections

Track these metrics to assess derivative quality over time:

| Metric | Good | Acceptable | Failing |
|---|---|---|---|
| Deferral count per section | 0 | 1-2 (all acceptable) | 3+ or any lazy deferral |
| Compression ratio (distillation) | 15-25% | 10-40% | >40% (not compressing) or <10% (losing claims) |
| Vocabulary bridge coverage (translation) | 100% of domain terms mapped | 90%+ mapped | <90% — parent vocabulary leaking through |
| Relevance filter precision (extraction) | 100% of included items are actionable | 90%+ actionable | <90% — including irrelevant content |
| Claim fidelity | 100% of claims match root exactly | Minor phrasing variation, same meaning | Any meaning change, count rounding, or qualifier loss |
| Self-sufficiency test | Pass on all 4 steps | Pass on steps 1-3, borderline on step 4 | Fail on any of steps 1-3 |

## Cross-References

- **Document Set Architecture:** Defines the derivation graph, depth tiers, and document manifest that derivation recipes reference. The manifest's `derivation_mode` and `derives_from` fields are the structural foundation for everything in this discipline.
- **Reading Path Design:** Defines the personas whose needs drive derivation mode selection. The Persona Registry provides the `depth_tolerance`, `domain_vocabulary`, and `task` fields that determine how a derivative section should be written.
- **Cross-Document Consistency:** Provides the Terminology Registry and Claim Registry that derivation must respect. Claims must match the canonical form; terms must use the tier-appropriate variant.
- **Document Evolution:** Provides the change classification and impact trace that trigger the "Updating a Derivative After Root Change" procedure. The propagation list in the Claim Registry is the link between evolution and derivation.
- **Document Governance:** Defines the LLM-as-steward trust model — deferral audit results are always surfaced to the human, even when all deferrals are classified as acceptable. The governance quality gates include derivation integrity as a mandatory check.
