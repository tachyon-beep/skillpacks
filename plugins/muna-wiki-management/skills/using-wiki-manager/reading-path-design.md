# Reading Path Design

## Overview

Reading Path Design ensures every persona in a document set can navigate from their entry point to task completion without hitting dead ends, being forced into documents above their depth tolerance, or encountering concepts with no prior introduction. It is the audience-facing complement to Document Set Architecture: where architecture defines how documents relate to each other structurally, reading path design defines how *people* move through those documents.

This reference sheet covers six capabilities:

1. **Persona Registry** — declaring who the document set serves, with enough detail to design and validate their paths
2. **Path Completeness Contract** — four requirements every reading path must satisfy, each with a concrete pass/fail test
3. **Coverage Matrix** — mapping personas to topics to expose gaps
4. **Time-Based Paths** — estimating reading time and building 5-minute, 30-minute, and deep-dive paths
5. **Stepping Stone Validation** — verifying each step in a path supports the next
6. **Procedures** — step-by-step instructions for building a new path and running a coverage audit

## When to Use

Load this guidance when:

- Adding a new persona to an existing document set
- Checking whether all audiences can accomplish their task with the current documents
- A reader reports confusion, dead ends, or "I had to open the full paper to understand the summary"
- Building navigation aids (reading guides, "start here" pointers, role-based tables of contents)
- Running a coverage audit to find gaps before publishing
- A new document has been added to the set and existing paths need re-evaluation

Do **not** use this for structural problems (missing derivation lineage, circular dependencies, manifest issues). Those belong to Document Set Architecture. Reading Path Design assumes the architecture is sound and asks: "Given this structure, can each persona get what they need?"

## Persona Registry

The Persona Registry is a project file (YAML) that declares every role the document set serves. It is the single source of truth about who reads these documents and what they need. All reading paths, coverage matrices, and time budgets are derived from this registry.

### Full Persona Registry Format

Every persona entry has seven fields. Six are required; one is optional but strongly recommended.

```yaml
personas:
  - id: sre-lead
    name: "SRE Lead / Platform Architect"
    task: assess
    depth_tolerance: practitioner
    domain_vocabulary: [SLA, SLO, incident-response, latency-budget, error-budget, capacity-planning]
    time_budget: "30 minutes"
    entry_point: operations/ops-assessment.md

  - id: vp-eng
    name: "VP Engineering / Executive Sponsor"
    task: govern
    depth_tolerance: executive
    domain_vocabulary: [risk, assurance, compliance, investment]
    time_budget: "5 minutes"
    entry_point: design/executive-brief.md

  - id: technical-lead
    name: "Technical Lead / Senior Developer"
    task: implement
    depth_tolerance: technical
    domain_vocabulary: [SAST, DAST, AST, CI-CD, linting, code-review, false-positive, triage]
    time_budget: "60 minutes"
    entry_point: design/architecture.md

  - id: migration-lead
    name: "Migration Lead"
    task: implement
    depth_tolerance: practitioner
    domain_vocabulary: [rollout, phased-deployment, KPI, baseline, control-group, adoption-metric]
    time_budget: "45 minutes"
    entry_point: reference/api-reference.md

  - id: eng-manager
    name: "Engineering Manager"
    task: govern
    depth_tolerance: practitioner
    domain_vocabulary: [SLA, versioning-policy, deprecation-schedule, backward-compatibility, migration-timeline]
    time_budget: "30 minutes"
    entry_point: plan/recommendations.md
```

### Field Reference

| Field | Required | Type | Description |
|---|---|---|---|
| `id` | Yes | string | Unique short identifier, lowercase hyphenated. Used in manifest `audience` lists and coverage matrices. |
| `name` | Yes | string | Human-readable role title. |
| `task` | Yes | enum: `assess`, `implement`, `govern`, `understand` | The primary job this persona is trying to accomplish with the document set. Determines what "task completion" means for path validation. |
| `depth_tolerance` | Yes | enum: `executive`, `practitioner`, `technical` | The maximum depth tier this persona will engage with. An `executive` persona will not read Tier 2 content. A `practitioner` reads Tier 1 comfortably and may skim Tier 2. A `technical` persona reads all tiers. |
| `domain_vocabulary` | Yes | list of strings | Terms this persona already knows. Any term used in their path documents that is *not* in this list must be defined in-path. |
| `time_budget` | Optional | string | How much time this persona will realistically spend. Drives time-based path design. Omitting it means the persona gets a deep-dive path only. |
| `entry_point` | Yes | string | File path (relative to project root) where this persona starts reading. Must be a document at or below their depth tolerance tier. |

### Granularity Heuristic

Register a persona if:

- They have a distinct entry point or reading path, OR
- Their depth tolerance or domain vocabulary differs meaningfully from existing personas

Skip minor role variations that share the same path. A "Deputy SRE Lead" who follows the exact same path as the SRE Lead does not need a separate entry. A "Platform Architect" who shares the SRE Lead's entry point but has different domain vocabulary (missing SRE-specific terms) does need one, because their path may require additional vocabulary stepping stones.

Target size: 8-15 personas for a document set of 5-15 documents serving multiple audiences; smaller sets (2-4 documents) may need only 3-6 personas. If you exceed 20, audit for duplicates — personas that share the same entry point, task, depth tolerance, and vocabulary are candidates for merging.

### Anti-Pattern: Persona Without a Task

A persona entry that specifies a role name and entry point but omits or vaguely defines the `task` field (e.g., `task: "read the documents"`).

**Why it is harmful:** Without a concrete task, the path has no terminal state. You cannot validate that the path ends with the persona able to do what they came to do, because "what they came to do" is undefined. Path completeness checks degenerate into "did they read something?" rather than "can they act?"

**Fix:** Every persona gets a task from the four-value enum: `assess` (evaluate and judge), `implement` (build or deploy), `govern` (set policy and oversee), `understand` (learn enough to make a decision about engagement). If the task does not fit these categories, the persona may be too granular or the document set's scope needs re-examination.

### Anti-Pattern: Domain Vocabulary as Afterthought

A persona entry where `domain_vocabulary` is either empty or contains a single vague term like `["security"]`.

**Why it is harmful:** The vocabulary list drives the comprehension gap check. If the list is empty, the check assumes the persona knows nothing and flags every technical term as a gap — producing so many false positives that the check becomes useless. If the list is vague, the check misses real gaps because it assumes the persona knows things they do not.

**Fix:** Populate `domain_vocabulary` with 5-15 specific terms the persona is expected to know before entering the document set. These are terms the documents will use *without defining*. Think: "What jargon would this person already use in a meeting?"

## Path Completeness Contract

Every persona's reading path must satisfy four requirements. Each requirement has a concrete test with a binary pass/fail result.

### Requirement 1: Entry Point Exists

The persona's declared `entry_point` must be a document that (a) exists in the manifest, (b) is at or below the persona's depth tolerance tier, and (c) addresses the persona's task within its first two sections.

**Pass test:** Open the entry point document. Read the first two sections (or first 500 words, whichever comes first). Can you identify what the persona's task is and confirm this document addresses it?

**Pass example:** The `sre-lead` persona has `entry_point: operations/ops-assessment.md`. Opening that document, the first section is "Impact Landscape Overview" and the second is "Breaking Change Analysis for Coverage Gap Assessment." The SRE Lead's task is `assess` — the document immediately frames content as an assessment activity. Pass.

**Fail example:** The `eng-manager` persona has `entry_point: design/architecture.md`. Opening that document, the first section is "Abstract" and the second is "Literature Review." The engineering manager's task is `govern` — the document opens with academic framing, not governance framing. The persona has been routed to the wrong document. Fail.

### Requirement 2: No Comprehension Gaps

Every concept introduced in the persona's path is either (a) defined when first encountered, or (b) listed in the persona's `domain_vocabulary`.

**Pass test:** Walk the path document by document. For each section, list every technical term, acronym, or domain concept used. Check each against the persona's `domain_vocabulary` list. For terms not in the list, check whether the current or a prior document in the path defines the term before using it.

**Pass example:** The `vp-eng` persona's path uses the term "residual risk" in the executive brief. Checking `domain_vocabulary: [risk, assurance, compliance, investment]` — "residual risk" is not listed. But the executive brief's opening paragraph defines it: "Residual risk is the risk remaining after controls are applied." The term is defined in-path. Pass.

**Fail example:** The `migration-lead` persona's path uses the term "verification tier" in the api-reference's implementation section. Checking `domain_vocabulary: [rollout, phased-deployment, KPI, baseline, control-group, adoption-metric]` — "verification tier" is not listed. Searching prior documents in the path: the term appears without definition. The migration lead encounters a concept they cannot understand without opening a different document. Fail.

### Requirement 3: No Forced Escalation

The path never requires the persona to jump to a document above their depth tolerance tier to complete their task.

**Pass test:** For each document in the persona's path, check every cross-reference link. If a link points to a higher-tier document, verify that the linking document provides enough content inline to complete the task without following the link. The link must be "for optional depth" not "required for comprehension."

**Pass example:** The `sre-lead` persona's path (Tier 1 depth tolerance) includes a link to `architecture §3.2` (Tier 2) with the text: "For the full evidence base supporting these impact ratings, see the Architecture Document §3.2." The Ops Assessment already contains the impact ratings and coverage gap analysis — the link adds optional evidence. The SRE Lead can complete their assessment without following the link. Pass.

**Fail example:** The `sre-lead` persona's path includes a section that states: "The five priority breaking changes are identified in the Architecture Document §3. Apply the coverage mapping from Appendix B of the api-reference." The SRE Lead must open two Tier 2 documents to identify the breaking changes and find the coverage mapping. The path forces escalation. Fail.

### Requirement 4: Terminal State

The path ends with the persona able to perform their declared task. The final document or section in the path produces a concrete output aligned with the task type.

**Pass test:** At the end of the path, ask: "Given only what this persona has read, can they produce the task output?" Task outputs by type:

| Task | Terminal State Output |
|---|---|
| `assess` | A completed evaluation, risk rating, or gap analysis |
| `implement` | A concrete action plan, configuration, or deployment sequence |
| `govern` | A policy decision, approval/rejection, or governance directive |
| `understand` | A decision about whether to invest further time/resources |

**Pass example:** The `eng-manager` persona's path ends at the Recommendations document §4 "Implementation Priorities." After reading, the engineering manager can produce a prioritized list of policy changes with estimated effort — a concrete governance directive. Pass.

**Fail example:** The `vp-eng` persona's path ends at the executive brief §2 "Key Findings" which lists breaking changes and impact ratings but provides no recommendation, decision framework, or "what should I do next" guidance. The executive has information but no basis for a governance decision. Fail.

## Coverage Matrix

The coverage matrix maps personas (rows) to key topics (columns). Each cell identifies which document and section covers that topic for that persona. Empty cells are explicit gaps that must be classified and addressed.

### What Counts as a Key Topic

A topic earns a column in the coverage matrix if it represents a decision point or action item for at least one persona. "What are the breaking changes?" is a key topic — personas need this to assess impact. "How was the taxonomy methodology developed?" is not — it is supporting detail that belongs inside a Tier 2 document but is not a decision-relevant topic for any persona.

Target: 10-20 topics for a standards suite. If the matrix exceeds 25 columns, the topics are too granular — merge related topics into decision-relevant clusters.

### Worked Example: 4 Personas x 5 Topics

This matrix covers a four-persona, five-document standards suite. Four personas need five key topics.

| | Breaking Changes | Impact Ratings | Coverage Gaps | Verification Model | Implementation Plan |
|---|---|---|---|---|---|
| **VP Eng** (executive, govern) | exec-brief §2: "15 breaking changes across 4 categories" (distilled) | exec-brief §3: "4 critical, 6 high, 5 moderate" (distilled) | -- GAP -- | exec-brief §4: "4-tier verification model overview" (distilled) | -- GAP -- |
| **SRE Lead** (practitioner, assess) | ops-assess §1: full breaking change analysis translated to coverage-gap language | ops-assess §2: impact ratings with coverage mapping | ops-assess §3: gap analysis against current coverage | recommendations §3: verification tier applicability | recommendations §4: prioritized implementation |
| **Technical Lead** (technical, implement) | architecture §3: complete taxonomy with evidence | architecture §4: impact model with methodology | architecture §5: coverage analysis | api-reference Part I: full verification model | api-reference Part II: technical implementation guide |
| **Migration Lead** (practitioner, implement) | ops-assess §1: breaking changes (reused from SRE Lead path) | -- GAP -- | ops-assess §3: gap analysis (reused from SRE Lead path) | api-reference Part I §2: verification tier overview | api-reference Part II §3: phased rollout plan |

### Gap Analysis

The matrix reveals three empty cells:

**Gap 1: VP Eng x Coverage Gaps** — Classification: **acceptable omission**. The VP Eng persona's task is `govern` (make a policy decision about investment). Coverage gap details are operational, not decision-relevant at the executive tier. The executive brief's impact ratings (§3) give the VP Eng enough to decide without knowing specific coverage gaps. No action needed.

**Gap 2: VP Eng x Implementation Plan** — Classification: **needs attention**. The VP Eng persona must decide whether to fund implementation. Without any implementation framing, the executive brief supports "is this a problem?" but not "what would fixing it cost?" Add a 2-3 sentence implementation summary to the executive brief: estimated effort level (not detailed plan), phased approach confirmation, and a pointer to the Recommendations document for the engineering manager to elaborate.

**Gap 3: Migration Lead x Impact Ratings** — Classification: **blocking gap**. The migration lead's task is `implement` — they need to know which breaking changes to prioritize in the migration. Without impact ratings, they cannot sequence their implementation. The migration lead's path currently borrows from the Ops Assessment for breaking changes and coverage gaps, but the impact ratings section (ops-assess §2) uses SRE Lead-specific language ("residual impact tolerance") that is outside the migration lead's vocabulary. Fix: either add an impact-rating summary to the api-reference's migration section using implementation-oriented vocabulary, or add "residual-impact" and "impact-tolerance" to the migration lead's domain vocabulary (if the migration lead is expected to know these terms).

### Gap Classification Framework

When the matrix reveals an empty cell, classify it before acting:

| Classification | Definition | Action |
|---|---|---|
| **Acceptable omission** | The persona does not need this topic to complete their task | Document the rationale in the matrix. No content change needed. |
| **Needs attention** | The persona would benefit from coverage but can complete their task without it | Add lightweight coverage (2-5 sentences, a summary callout, or a cross-reference) at the next maintenance cycle. |
| **Blocking gap** | The persona cannot complete their task without this topic | Fix immediately. Either create new content at the persona's depth tier or extend an existing section. |

## Time-Based Paths

For each persona with a declared `time_budget`, design three paths: a 5-minute path, a 30-minute path, and a deep-dive path. Each path must be self-sufficient at its time budget — the reader can stop at the end of the timed path and have accomplished something meaningful.

### Time Estimation Methodology

Reading time is estimated from word count, adjusted for content density and vocabulary mismatch.

#### Step 1: Baseline Reading Rates

Documents in the set are classified into three **depth tiers**: **Tier 0** (executive — 2-page decision documents), **Tier 1** (practitioner — 8-15 page role-specific working documents), and **Tier 2** (reference — full technical analyses, specifications, root documents). Each tier has a different reading rate:

| Content Type | Words Per Minute | Applies To |
|---|---|---|
| Executive-depth prose | 250 wpm | Tier 0 content: summaries, overviews, recommendations written in plain language |
| Practitioner content with domain vocabulary | 150 wpm | Tier 1 content: role-specific documents using controlled terminology |
| Dense technical content requiring cross-reference | 100 wpm | Tier 2 content: specifications, full analyses, documents with tables and formulas |

These are reading rates for comprehension, not scanning rates. A reader scanning headings and callouts moves faster, but scanning does not produce the comprehension needed for task completion.

#### Step 2: Density Adjustment

Tables, diagrams, code examples, and control matrices take approximately 2x the equivalent word count to process. A 200-word table reads at the effective rate of 400 words of prose.

To apply: count words in dense elements (tables, diagrams, code blocks). Double that count and add it to the prose word count before dividing by the reading rate.

**Density-adjusted word count** = prose words + (2 x dense-element words)

#### Step 3: Vocabulary Adjustment

If the content uses terms outside the persona's declared `domain_vocabulary`, add 20% to the estimated reading time. This accounts for the cognitive overhead of encountering unfamiliar terminology — the reader pauses, re-reads, or mentally translates each unfamiliar term.

To apply: scan the content for terms not in the persona's vocabulary list. If more than 3 unfamiliar terms appear per page (approximately per 300 words), apply the 20% adjustment.

**Vocabulary-adjusted time** = base time x 1.2 (if vocabulary mismatch threshold is exceeded)

### Worked Calculation: SRE Lead Persona, 30-Minute Path

The SRE Lead persona has `depth_tolerance: practitioner`, `domain_vocabulary: [SLA, SLO, incident-response, latency-budget, error-budget, capacity-planning]`, and `time_budget: "30 minutes"`.

The proposed 30-minute path covers:

1. Ops Assessment §1: Breaking Change Analysis (1,800 words prose, 400 words in tables)
2. Ops Assessment §2: Impact Ratings and Coverage Mapping (1,200 words prose, 600 words in tables)
3. Recommendations §3: Verification Tier Applicability (800 words prose, 200 words in tables)

**Section 1 — Breaking Change Analysis:**
- Density-adjusted word count: 1,800 + (2 x 400) = 2,600 effective words
- Reading rate: 150 wpm (practitioner content)
- Base time: 2,600 / 150 = 17.3 minutes
- Vocabulary check: terms used include "verification tier" (not in SRE Lead vocabulary). Only 1 unfamiliar term per ~600 words — below the threshold of 3 per 300 words. No vocabulary adjustment.
- Section 1 time: **17.3 minutes**

**Section 2 — Impact Ratings and Coverage Mapping:**
- Density-adjusted word count: 1,200 + (2 x 600) = 2,400 effective words
- Reading rate: 150 wpm
- Base time: 2,400 / 150 = 16.0 minutes
- Vocabulary check: all terms within SRE Lead vocabulary. No adjustment.
- Section 2 time: **16.0 minutes**

**Section 3 — Verification Tier Applicability:**
- Density-adjusted word count: 800 + (2 x 200) = 1,200 effective words
- Reading rate: 150 wpm
- Base time: 1,200 / 150 = 8.0 minutes
- Vocabulary check: "verification tier" is a reader-friendly alias for the formal term used in the root document. The SRE Lead knows "incident-response" but not "verification tier" specifically. One unfamiliar term per 400 words — below threshold. No adjustment.
- Section 3 time: **8.0 minutes**

**Total estimated time:** 17.3 + 16.0 + 8.0 = **41.3 minutes**

**Result:** The proposed path exceeds the 30-minute budget by 11.3 minutes. To fit the budget, drop Section 3 (Verification Tier Applicability) and include it only in the deep-dive path. The 30-minute path covers breaking changes and impact ratings — enough for the SRE Lead to complete a coverage gap assessment (their `assess` task) without the verification detail.

**Revised 30-minute path:** Ops Assessment §1 + §2 = 33.3 minutes. Still slightly over. Options: (a) accept the slight overage (reading rates are estimates), or (b) trim the path to §1 + §2 first half (impact ratings only, skip coverage mapping detail) for a tighter 25-minute fit.

### The Three Path Tiers

**5-minute path (the "decision gate"):** Enough to decide whether to invest more time. Must answer three questions for the persona:

1. Is this relevant to me?
2. What is the headline finding?
3. What should I do next?

For most personas, the 5-minute path is the entry point document's executive summary or first section. It must be self-sufficient — the reader can stop here and have a valid (if shallow) answer to their task.

**30-minute path (the "working knowledge" path):** Enough to act. The persona can make decisions, assess controls, brief someone else, or begin implementation planning. This is the primary path most personas will follow.

**Deep-dive path (no time constraint):** Everything relevant to this persona across the entire document set. For technical personas, this may include following links into Tier 2 documents. The deep-dive path is the persona's complete reading list, ordered for progressive comprehension.

### Anti-Pattern: Time Budget Without Word Count Basis

A reading path that claims "this is a 30-minute path" without calculating the actual reading time from word counts and adjustments.

**Why it is harmful:** Ungrounded time claims are systematically optimistic. Authors underestimate reading time because they already know the content. A "30-minute path" that actually takes 55 minutes erodes reader trust — the persona abandons the path halfway, missing critical content in later sections.

**Fix:** Every time-budgeted path gets the three-step calculation: baseline word count at the appropriate reading rate, density adjustment for tables and diagrams, vocabulary adjustment if terms exceed the persona's declared vocabulary. The calculated time is the path's time claim.

## Stepping Stone Validation

A stepping stone is one step in a reading path — a document or section that the persona reads in sequence. Stepping stone validation checks that each step in the path prepares the reader for the next step.

### The Three-Question Test

For each step in a reading path, answer these three questions:

1. **Does the reader have enough context from previous steps?** — Can the reader understand this step's content given only what they have read so far in the path (plus their declared `domain_vocabulary`)?
2. **Does this step give enough to proceed to the next?** — After reading this step, does the reader have the concepts, vocabulary, and framing needed to understand the next step?
3. **If this step links to a higher-tier document "for detail," can the reader skip that link and still complete their task?** — Every link to a higher tier must be optional. If skipping the link leaves a comprehension gap, the step fails validation.

A step passes validation only if all three answers are "yes."

### Worked Example: Validating a Path Step

**Persona:** Migration Lead (`depth_tolerance: practitioner`, `domain_vocabulary: [rollout, phased-deployment, KPI, baseline, control-group, adoption-metric]`)

**Path being validated:**
- Step 1: Ops Assessment §1 — Breaking Change Analysis
- **Step 2: api-reference Part I §2 — Verification Tier Overview** (this is the step under validation)
- Step 3: api-reference Part II §3 — Phased Rollout Plan

**Question 1: Does the reader have enough context from previous steps?**

Step 2 (Verification Tier Overview) discusses "four verification tiers mapped to the breaking change categories identified in the impact taxonomy." The reader arrives from Step 1 (Breaking Change Analysis), which introduces the breaking change categories and explains the taxonomy structure. Check: does Step 1 define "breaking change categories" in terms the migration lead understands?

Step 1 uses the terms "breaking change," "category," and "impact rating." The migration lead's vocabulary does not include "breaking change" or "impact rating." Step 1 defines "breaking change" in its opening paragraph: "A breaking change is a specific, catalogued pattern of AI-generated code defect." However, "impact rating" is used without definition in Step 1's table — Step 1 assumes the reader understands impact ratings from the SRE Lead context.

**Result: FAIL on Question 1.** The migration lead arrives at Step 2 without understanding "impact rating," which Step 2 uses to explain how verification tiers map to impact levels.

**Fix options:**
- (a) Add Step 1.5: a brief section that defines "impact rating" in migration-friendly terms (e.g., "Impact ratings indicate how likely a breaking change is to affect production code and how severe the impact")
- (b) Add "impact-rating" to the migration lead's `domain_vocabulary` if the migration lead is expected to know this concept
- (c) Modify Step 1's breaking change analysis to define impact ratings inline when they first appear

**Question 2: Does this step give enough to proceed to the next?**

Step 2 explains the four verification tiers and their capabilities. Step 3 (Phased Rollout Plan) discusses deploying these tiers in sequence. Does Step 2 give the migration lead enough understanding of each tier's capabilities to follow the rollout sequencing rationale in Step 3?

Step 2 describes each tier's detection capabilities and implementation requirements. Step 3 references these by name ("Tier 1 deployment covers breaking changes 1-4 with static analysis"). The migration lead can map tier names to capabilities from Step 2's content.

**Result: PASS on Question 2.**

**Question 3: If this step links to Tier 2 "for detail," can the reader skip it?**

Step 2 contains a link: "For the complete detection-capability matrix, see the Full Architecture Document §3.4." The verification tier overview in Step 2 already lists which breaking changes each tier detects (summary table). The link adds the evidence base (why each detection mapping is believed to be accurate).

Can the migration lead plan a phased rollout without the evidence base? Yes — they need to know which changes each tier detects (provided in Step 2's summary table), not why the detection works.

**Result: PASS on Question 3.**

**Overall step validation: FAIL** (failed Question 1). The path needs modification before the migration lead can traverse it successfully.

## Building a Reading Path for a New Persona

Follow this procedure when adding a persona to the registry and designing their path through the document set.

### Step 1: Define the Persona

Create the persona registry entry with all required fields. Be specific about each field:

- `id`: lowercase, hyphenated, unique
- `name`: the role title people would use in conversation
- `task`: choose exactly one of `assess`, `implement`, `govern`, `understand`
- `depth_tolerance`: choose exactly one of `executive`, `practitioner`, `technical`
- `domain_vocabulary`: list 5-15 specific terms the persona already knows
- `time_budget`: realistic wall-clock time this persona will spend (optional but recommended)
- `entry_point`: the document where this persona starts — must exist in the manifest and must be at or below their depth tolerance tier

### Step 2: Identify the Entry Point

The entry point must satisfy three conditions:

1. It exists in the document manifest
2. Its tier is at or below the persona's depth tolerance (an `executive` persona cannot enter at a Tier 1 document)
3. Its first two sections address the persona's task

If no existing document satisfies all three, flag this as a gap. Either an existing document needs a new opening section, or a new derivative document is needed.

### Step 3: Map the Path

Starting from the entry point, trace the reading sequence the persona needs to reach their terminal state. For each step, record:

- Which document and section
- What the persona learns at this step
- What concepts this step introduces that later steps depend on

Use the Coverage Matrix to check that the path covers all topics the persona needs. Cross-reference the persona's row in the matrix — every cell marked "blocking gap" for this persona must be covered somewhere in the path.

### Step 4: Validate Each Step

Apply the Three-Question Test (see Stepping Stone Validation) to each step in sequence. If any step fails, modify the path before proceeding.

Common fixes for failing steps:
- **Missing context from prior steps:** Insert a bridging section or reorder the path
- **Step does not prepare for the next:** Add a transitional summary at the end of the step
- **Required escalation to higher tier:** Inline the content the persona needs at their depth level

### Step 5: Design Time-Based Variants

If the persona has a `time_budget`, build the three path tiers:

1. **5-minute path:** Entry point's first section only. Verify it answers: Is this relevant? What is the headline? What next?
2. **30-minute path:** Calculate reading time using the estimation methodology (baseline rate, density adjustment, vocabulary adjustment). Add sections until the time budget is reached. Verify the path reaches a meaningful intermediate terminal state.
3. **Deep-dive path:** The full path from Step 3. No time constraint.

### Step 6: Update the Coverage Matrix

Add the new persona as a row. Fill in cells for every topic the persona's path covers, citing the specific document and section. Flag any empty cells using the gap classification framework (acceptable omission, needs attention, blocking gap).

### Step 7: Peer Validation

Have someone unfamiliar with the document set walk the path using only the documents in the path and the persona's declared vocabulary. If they get stuck, the path has a gap that the designer's domain knowledge masked.

## Running a Coverage Audit

Follow this procedure to systematically check that all personas can reach their terminal state through the document set.

### Step 1: Verify the Persona Registry Is Current

- Confirm every persona in the registry still represents an active audience
- Confirm no new audiences have emerged since the last audit (check: has anyone asked "who is this document for?" recently?)
- Confirm all fields are populated — particularly `domain_vocabulary` and `entry_point`

### Step 2: Rebuild the Coverage Matrix

Do not reuse the previous matrix — rebuild it from scratch to catch drift:

1. List all current personas as rows
2. List all current key topics as columns (apply the key topic heuristic: decision points and action items, not supporting detail)
3. For each cell, trace which document and section provides that topic for that persona at their depth tolerance
4. Leave empty cells explicitly empty — do not fill them optimistically

### Step 3: Classify All Gaps

For every empty cell in the matrix, apply the gap classification:

- **Acceptable omission:** Document the rationale. Example: "VP Eng does not need implementation detail to make a funding decision."
- **Needs attention:** Log it for the next maintenance cycle. Example: "Migration lead would benefit from an impact rating summary but can begin planning without it."
- **Blocking gap:** Flag for immediate action. Example: "SRE Lead cannot complete gap assessment without verification tier mapping."

### Step 4: Validate All Entry Points

For each persona, verify:

- The `entry_point` file path still resolves to an existing document
- The entry point document is still at or below the persona's depth tolerance tier
- The entry point's opening content still addresses the persona's task (content may have shifted during document updates)

### Step 5: Run Stepping Stone Validation on All Paths

For each persona's path, apply the Three-Question Test to every step. Record failures using the standard defect format:

```
Defect:
  location: [document-id §section]
  type: path-dead-end | coverage-gap
  severity: error | warning
  description: "Migration lead's path fails Question 1 at api-reference Part I §2 —
    'impact rating' used without definition and not in persona's domain vocabulary"
  suggested_fix: "Add impact rating definition to Ops Assessment §1 opening
    paragraph, or add 'impact-rating' to migration-lead domain vocabulary"
  related_registry: "persona: migration-lead"
```

### Step 6: Verify Time-Based Paths

For each persona with a `time_budget`:

- Recalculate reading time for the 5-minute and 30-minute paths (word counts may have changed)
- Verify the 5-minute path still answers the three decision-gate questions
- Verify the 30-minute path still reaches a meaningful terminal state
- Flag any path where calculated time exceeds the budget by more than 15%

### Step 7: Produce the Audit Report

Compile all findings into a prioritized defect list:

**Priority 1 (fix immediately):**
- Blocking gaps in the coverage matrix
- Entry points that no longer exist or have drifted off-task
- Stepping stone failures that prevent task completion (path-dead-end defects)

**Priority 2 (fix before next publish):**
- "Needs attention" gaps that affect multiple personas
- Time-based paths that exceed budget by more than 15%
- Vocabulary gaps that appear in more than one persona's path

**Priority 3 (fix during maintenance):**
- "Needs attention" gaps affecting a single persona
- Minor vocabulary mismatches below the threshold
- Time estimates slightly over budget (within 15%)

## Anti-Patterns Summary

| Anti-Pattern | Section | Core Harm |
|---|---|---|
| Persona without a task | Persona Registry | No terminal state; path completeness cannot be validated |
| Domain vocabulary as afterthought | Persona Registry | Comprehension gap check produces unusable results (all false positives or all false negatives) |
| Path requiring multiple Tier 2 documents for comprehension | Path Completeness | Forces escalation beyond the persona's depth tolerance; the derivative documents have failed their purpose |
| Path starting at Tier 1 but requiring backtracking to Tier 0 | Path Completeness | Indicates the persona entered at the wrong document; the entry point is misassigned |
| Two personas routed to same document with no section guidance | Coverage Matrix | Both personas read the whole document looking for their content; neither knows which sections matter for them |
| Reading path as a list of links with no narrative thread | Path Design | A list of links is a bibliography, not a path. Without sequencing rationale and stepping-stone connections, the reader does not know what to read first or why one document prepares them for the next. |
| Time budget without word count basis | Time-Based Paths | Ungrounded time estimates are systematically optimistic; readers abandon paths that take longer than claimed |
| Entry point at wrong depth tier | Path Completeness | An executive persona entering at a Tier 2 document will not read it. The path is dead on arrival. |

## Cross-References

These reference sheets cover sibling competencies in the wiki management skill set. All files are in the same directory as this document.

- `document-set-architecture.md` — Document manifest, depth tiers and self-sufficiency contracts, derivation graph, multi-root reconciliation, and conflict resolution
- `content-derivation.md` — Derivation recipes, the self-sufficiency test, derivation integrity checks, and the anti-laziness discipline
- `cross-document-consistency.md` — Terminology Registry, Claim Registry, structural conventions, link integrity, and the consistency audit process
- `document-evolution.md` — Change classification, impact tracing, git integration, external dependency tracking, and version coherence
- `document-governance.md` — Ownership model, LLM-as-steward trust model, change workflows, quality gates, and review triggers
