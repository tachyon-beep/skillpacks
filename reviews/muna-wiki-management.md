# Review: muna-wiki-management
**Version:** 1.0.1  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

**Plugin metadata** (`/home/john/skillpacks/plugins/muna-wiki-management/.claude-plugin/plugin.json:1-19`)
- `name`: `muna-wiki-management`
- `version`: `1.0.1`
- License: `CC-BY-SA-4.0`. Author/repo: `tachyon-beep`.
- Description (line 4): "Document set management as wikis - 7 skills, 4 commands, 2 agents for architecture, derivation, consistency, evolution, and governance of multi-document collections". The count of 7 "skills" treats the router + 6 reference sheets as 7 skills, which conflicts with the marketplace's convention (router is one skill, reference sheets are content). See Findings.

**Marketplace registration** (`/home/john/skillpacks/.claude-plugin/marketplace.json`)
- Registered as `muna-wiki-management`, source `./plugins/muna-wiki-management`. Description: "Document set management as wikis - architecture, derivation, consistency, evolution, governance for multi-document collections - 7 skills, 4 commands, 2 agents". Matches `plugin.json`.

**Router skill** (`/home/john/skillpacks/plugins/muna-wiki-management/skills/using-wiki-manager/SKILL.md`, 206 lines)
- Single router: `using-wiki-manager`.
- Description (`SKILL.md:3`): "Manage complex document sets as wikis - architecture, reading paths, derivation, consistency, evolution, and governance for multi-document collections". Does NOT start with the marketplace's dominant "Use when..." convention. See Findings.
- Body documents: Core Principle (`:12`), When to Use (`:15-24`), explicit reference-sheet path instructions (`:29-43`), five entry patterns (`:47-119` — onboarding, new set, day-to-day, change propagation, full audit), cross-faction integration with `muna-technical-writer` and `muna-panel-review` (`:124-137`), reference-sheet catalog table (`:141-152`), ten design principles (`:156-167`), four common mistakes (`:171-187`), decision tree (`:191-206`).
- Pattern 3 and Pattern 4 instructions correctly point users at the matching `/propagate-change` and `/audit-docset` commands.

**Reference sheets** (6 total, all under `skills/using-wiki-manager/`)
| Sheet | Lines |
|---|---|
| `content-derivation.md` | 565 |
| `cross-document-consistency.md` | 627 |
| `document-evolution.md` | 786 |
| `document-governance.md` | 667 |
| `document-set-architecture.md` | 623 |
| `reading-path-design.md` | 521 |

Spot-checks of sheet content (representative passages):
- `content-derivation.md:5-7` opens with the "signpost not synthesis" diagnosis that is the spine of the anti-laziness discipline. The "see the full architecture document for details" sentence is named as the canonical failure mode and is referenced in `audit-docset.md` and `derive-content.md`.
- `content-derivation.md:40-167` defines the three derivation modes (distillation, translation, extraction) with a complete worked before/after example for each (the 13-failure-modes taxonomy → 30%-compressed executive brief at `:56-66`; the SRE-Lead vocabulary bridge at `:88-95`; the extraction example downstream).
- `content-derivation.md:201-265` defines the self-sufficiency test and a complete passing/failing worked example. `:267-294` defines the deferral audit decision table (acceptable vs lazy vs borderline). `:295-371` defines the derivation recipe template with a complete filled-in example.
- `content-derivation.md:459-527` enumerates six "vivid failure" anti-patterns: Signpost Section, Phantom Elaboration, Meaning-Inverting Compression, Wrong-Depth Derivative, Registry-Stale Derivative, Comfortable Extraction. These pair to the `self-sufficiency-reviewer` agent's findings categories.
- `cross-document-consistency.md:37-198` defines the Terminology Registry and Claim Registry formats with YAML examples for each. `:199-247` defines granularity heuristics that explicitly bound the registries to prevent "exhaustive glossary" failure (`:107-112`).
- `cross-document-consistency.md:295-351` defines four link integrity checks (forward resolution, bidirectional awareness, orphan sections, anchor stability) — the audit-docset Phase 1 Step 3 dispatches into these checks.
- `cross-document-consistency.md:390-415` enumerates the 10 defect types used across `/audit-docset` Phase 4 (line `audit-docset.md:91`): `terminology-drift, stale-claim, broken-link, lazy-deferral, orphan-section, convention-violation, coverage-gap, path-dead-end, phantom-content, self-sufficiency-failure, recipe-missing`. (Cross-reference: command uses 11 types — adds `recipe-missing`; consistency sheet's "10 defect types" heading is therefore slightly mislabelled. See Findings.)
- `document-set-architecture.md:342-525` covers multi-root reconciliation patterns (Partition, Primary/Supplementary, Synthesis) with worked examples and an explicit anti-pattern of "silently picking a side" and "Claude resolving factual conflicts autonomously".
- `document-evolution.md` outline (lines `:Overview, :When to Use, :Key Terminology, :Change Classification, :Impact Trace, :Git Integration, :External Dependency Tracking, :Version Coherence, :Deprecation Workflow, :Responding to a Root Document Change, :Responding to an External Standard Update, :Anti-Pattern Summary`) cleanly covers the producer side of Pattern 3.
- `document-governance.md:110-216` defines the four-point LLM-as-steward trust model (Propose/Dispose, Always-Surfaced Deferral Audit, Spot-Check Protocol, Escalation Triggers) with a complete worked sample of Claude's surfaced output at `:132-166` (which is correctly enclosed in a fenced code block — the `##`/`###` lines inside are example output, not real headings).
- `document-governance.md:218-372` defines the Root-Document-Change Workflow (6 steps) and Derivative-Only-Change Workflow (5 steps). `:254-371` defines six Quality Gates with passing-looks-like / failure-triggers / concrete-check for each.
- `reading-path-design.md:215-303` codifies time-based path estimation (baseline reading rates, density adjustment, vocabulary adjustment) with a worked 30-minute SRE-Lead calculation. `:306-358` defines stepping-stone validation with a complete worked example.

**Commands** (4, in `commands/`)
| Command | argument-hint declared? | allowed-tools? |
|---|---|---|
| `/onboard-docset` | Yes (`[directory_containing_documents]`) | Yes (`["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]`) |
| `/audit-docset` | Yes (`[manifest_path]`) | Yes (same set) |
| `/derive-content` | Yes (`[derivative_path] [section]`) | Yes (same set) |
| `/propagate-change` | Yes (`[root_document_path] [changed_section]`) | Yes (same set) |

All four commands follow the marketplace's quoted-JSON-array `allowed-tools` convention. All four declare `description` (one line, no trailing period) per the convention. None declares `"Skill"` in `allowed-tools` despite their bodies routing back into the reference sheets — see Findings.

**Agents** (2, in `agents/`)
| Agent | Model | tools restriction | SME-Protocol? |
|---|---|---|---|
| `reference-sheet-writer` | `opus` | `["Read", "Write", "Grep", "Glob"]` | No (producer, exempt) |
| `self-sufficiency-reviewer` | `sonnet` | `["Read", "Glob"]` | No — but should be (reviewer agent) |

Both agents declare `tools:` lists. Most repo agents omit `tools:` and inherit the parent context (~5/65 declare it per the rubric); restricting tools here is *intentional and load-bearing* for both:
- `reference-sheet-writer.md:34` restricts to `Read, Write, Grep, Glob` — no `Bash` or other side-effecting tools. Defensible for a writer.
- `self-sufficiency-reviewer.md:25` restricts to `Read, Glob` only — no `Grep`. The body at `:28-32` makes the restriction load-bearing: the agent is *deliberately* prevented from searching for sources so it cannot judge by coverage, only by comprehensibility. This is a well-designed restriction, not a maintenance burden.

Both agents also declare a non-standard `color:` field (`reference-sheet-writer.md:33` = `green`, `self-sufficiency-reviewer.md:24` = `yellow`). This is not in the documented frontmatter style observed across the marketplace (`description` + `model` are the near-universal keys; `name` is also typical). The `color:` field is harmless but inconsistent with sibling packs.

Per-agent body-level evidence:
- `reference-sheet-writer.md:48-73` defines five non-negotiable Anti-Laziness Rules (No Deferral Phrases, Every Section Must Have Substance, No Thin Connecting Prose, Quantitative Claims Must Be Present Not Referenced, Worked Examples Are Mandatory Not Optional). Each rule is concretely actionable.
- `reference-sheet-writer.md:75-95` defines a six-step writing process with explicit self-audit steps at 4 and 5.
- `reference-sheet-writer.md:103-114` includes the "When You Are Struggling" escape hatch — explicit `[TODO: ...]` marker as the *honest gap* alternative to lazy deferral.
- `self-sufficiency-reviewer.md:35-58` defines the review process: read end-to-end, assess Comprehensibility / Actionability / Completeness / Deferral detection per section, rate PASS / CONDITIONAL PASS / FAIL.
- `self-sufficiency-reviewer.md:62-79` enumerates six finding categories (Thin Sections, Orphan Concepts, Missing Worked Examples, Lazy Deferrals, Conclusion Without Argument, Ungrounded Claims). These pair to the anti-patterns in `content-derivation.md:459-527`.
- `self-sufficiency-reviewer.md:81-109` defines a structured output format. However, it does NOT include the four SME-Protocol sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) and the body does not cite `meta-sme-protocol:sme-agent-protocol`. See Findings.

**Hooks:** None.

**Slash-command wrapper:** `/home/john/skillpacks/.claude/commands/wiki-manager.md` exists (203 lines). Status: present but **verbatim copy of SKILL.md** (skill body, not a thin summary). See Findings.
- Line 1 is blank; no YAML frontmatter description. Convention varies across the marketplace (some wrappers have a frontmatter `description`, e.g. `.claude/commands/creative-writing.md:2`; others do not, e.g. `.claude/commands/technical-writer.md`).
- Lines 6-203 of the wrapper match `SKILL.md` lines 8-206 verbatim (Overview through Decision Tree). The relative-link references in the wrapper (`[content-derivation.md](content-derivation.md)`) point to files at the wrapper's own path, which is `.claude/commands/`, where those files do not exist. See Findings.

## 2. Domain & Coverage

**User-defined scope** (from `plugin.json:4`, `SKILL.md:8-24`, and `cross-document-consistency.md`/`content-derivation.md` framing)
- *Intent:* Manage *sets* of related documents as wikis — manifest-driven, derivation-recipe-disciplined, consistency-registered, governance-gated. Anti-laziness is the spine of the design: the derivative document must be self-sufficient, never a signpost back to the root.
- *Boundaries:* Single-document quality is explicitly out of scope (deferred to `muna-technical-writer`). Reader-panel testing is explicitly out of scope (deferred to `muna-panel-review`). Both deferrals are documented at `SKILL.md:24` and `:124-137`.
- *Audience:* Practitioner-to-expert documentation engineers managing standards suites, policy frameworks, technical documentation sets. Vocabulary assumes familiarity with derivation, depth tiers, manifest-driven authoring.
- *Faction:* Muna (documentation), sibling to `muna-technical-writer`, `muna-panel-review`, and `muna-document-designer`.

**Coverage map (model knowledge, multi-document wiki management):**

### Foundational
- Manifest-driven document set architecture — **Exists** (`document-set-architecture.md`, complete manifest example, manifest field reference, validation rules)
- Root vs derivative classification — **Exists** (`document-set-architecture.md:37-65`)
- Depth tiers (0/1/2) with self-sufficiency contract — **Exists** (`document-set-architecture.md:65-103`)
- Persona registry and reading paths — **Exists** (`reading-path-design.md`)
- Content derivation discipline — **Exists** (`content-derivation.md`, three modes with worked examples, recipe template, self-sufficiency test, deferral audit)
- Cross-document consistency (terminology, claims, links) — **Exists** (`cross-document-consistency.md`, both registries with formats, granularity heuristics, four link integrity checks)

### Core techniques
- Change classification (cosmetic / clarification / substantive / structural) — **Exists** (`document-evolution.md:Change Classification`)
- Impact trace via derivation graph — **Exists** (`document-evolution.md:Impact Trace` with worked example)
- Multi-root reconciliation (Partition / Primary-Supplementary / Synthesis) — **Exists** (`document-set-architecture.md:342-525`)
- Multi-root conflict resolution (terminological / factual / structural) — **Exists** (`document-set-architecture.md:427-525`)
- Quality gates (six gates: derivation integrity / self-sufficiency / consistency / link integrity / path completeness / coverage matrix) — **Exists** (`document-governance.md:254-371`)
- Coverage matrix (personas × topics) — **Exists** (`reading-path-design.md:170-209`)
- Stepping-stone validation — **Exists** (`reading-path-design.md:306-358`)
- LLM-as-steward trust model — **Exists** (`document-governance.md:110-216`, four explicit points with escalation triggers)

### Advanced
- External dependency tracking — **Exists** (`document-evolution.md:External Dependency Tracking`)
- Version coherence and release discipline — **Exists** (`document-evolution.md:Version Coherence`)
- Deprecation workflow — **Exists** (`document-evolution.md:Deprecation Workflow`)
- Time-based reading paths (baseline rate, density adjustment, vocabulary adjustment) — **Exists** (`reading-path-design.md:215-303`)
- Bootstrapping from existing unstructured doc sets — **Exists** (`document-set-architecture.md:567-601`, plus `/onboard-docset` command)

### Cross-cutting
- Anti-laziness discipline (signpost vs synthesis) — **Pervasive and load-bearing.** Named in `content-derivation.md:5-7`, enforced in `reference-sheet-writer.md:48-73`, audited by `self-sufficiency-reviewer.md`, surfaced by `/derive-content` Step 5 (`derive-content.md:87-97`) and `/propagate-change` Step 4 (`propagate-change.md:60-65`), and codified in Quality Gate 2 (`document-governance.md:276-291`).
- Auditable claims (every quality claim has a concrete repeatable test) — **Design principle 3** (`SKILL.md:158`), realised in each Quality Gate's "concrete check" line.
- Two-root awareness — **Design principle 4** (`SKILL.md:159`), realised in `document-set-architecture.md:342-525` with three reconciliation patterns and explicit "Claude does not resolve factual conflicts autonomously" anti-pattern.
- Git-native (manifests, registries, documents versioned together) — **Design principle 10** (`SKILL.md:163`), realised in `document-evolution.md:Git Integration` and `/propagate-change` Step 6.
- Bootstrappability (minimum viable registries grow through use) — **Design principle 7** (`SKILL.md:160`), realised by `/onboard-docset` and `document-set-architecture.md:567-601`.

**No coverage gaps identified.** The pack is comprehensive within its declared scope. Domains adjacent to wiki management but explicitly deferred (single-document craft → `muna-technical-writer`; reader-panel simulation → `muna-panel-review`) are correctly out of scope.

**Domain stability:** Stable. Wiki/document-set management is a mature discipline (Diataxis, DITA, content strategy literature). No research currency concerns. The pack's specific contribution — anti-laziness discipline for LLM-authored derivatives — is novel but well-grounded in observed LLM failure modes.

## 3. Fitness Scorecard (8 dimensions)

| Dimension | Rating | Notes |
|---|---|---|
| **Coverage** | Pass | Six reference sheets cover the declared scope completely. No gaps identified within "manage multi-document sets as wikis". Adjacencies are correctly deferred to sibling packs. |
| **Discoverability** | Minor | Router description (`SKILL.md:3`) does not start with "Use when..." — diverges from the marketplace's dominant convention for skill discovery. Body's "When to Use" (`:15-24`) is well-formed and includes trigger phrases. |
| **Routing** | Pass | Five entry patterns (Onboarding, New Set, Day-to-Day, Change Propagation, Full Audit) with explicit command pointers. Day-to-day pattern has a per-intent routing table. Decision tree at `:191-206` covers the manifest-yes/no fork cleanly. |
| **Component fit** | Pass | Skills (router + 6 ref sheets), commands (4 user-invocable workflows), agents (1 producer + 1 reviewer with deliberate tool restriction). Right tool for the right job throughout. |
| **Frontmatter hygiene** | Minor | Commands all conform to convention (quoted JSON-array `allowed-tools`, `argument-hint`, `description` no trailing period). Agents both declare a non-standard `color:` field; both declare `tools:` (defensible — see Inventory). Skill description does not start with "Use when...". |
| **SME-protocol compliance** | Major | `self-sufficiency-reviewer` is a reviewer/auditor agent and should follow the SME Agent Protocol per the rubric (`reviewing-pack-structure.md:95-98`). Its description does not end with "Follows SME Agent Protocol with confidence/risk assessment." and the body does not cite `meta-sme-protocol:sme-agent-protocol` or require the four output sections (Confidence Assessment / Risk Assessment / Information Gaps / Caveats). |
| **Slash-command wrapper** | Major | `.claude/commands/wiki-manager.md` exists but duplicates the SKILL.md body verbatim. The wrapper's relative links to reference sheets (`[content-derivation.md](content-derivation.md)`) will resolve from the wrapper's directory, not the skill's directory, so the wrapper cannot actually open the sheets it lists. Wrapper should be a thin summary that explicitly invokes / points to the router skill, not a content duplicate. |
| **Internal consistency** | Minor | Three inconsistencies: (1) `plugin.json` and `marketplace.json` advertise "7 skills" but the pack has 1 router + 6 reference sheets (the rubric treats sheets as content, not skills). (2) `cross-document-consistency.md:390-415` heading says "The 10 Defect Types" but `/audit-docset:91` enumerates 11 (includes `recipe-missing`). (3) Pattern 0 (`SKILL.md:47-60`) recommends route order `architecture → reading-path → consistency → governance`, but the corresponding `/onboard-docset` command sequences `inventory → derivation discovery → manifest → registries → reading paths → baseline audit → present`. The order in the SKILL.md does not include the registry-bootstrap step that the command makes Step 4 — readers following the manual route will skip a load-bearing step. |

**Overall:** **Major.** The pack is structurally sound, comprehensively covered, and has well-designed anti-laziness discipline as its spine. Two distinct Major findings (SME-protocol non-compliance on the reviewer agent; broken slash-command wrapper) and four Minor findings are concrete maintenance fixes — none requires rebuild. Recommendation: **Enhance**, not rebuild.

## 4. Behavioral Tests

Behavioral tests were executed via scenario reasoning against the loaded reference sheets and commands. Each test names the scenario, expected guidance, and observed pass/fail based on what the components instruct.

### Test 1 — "Just summarize the paper" pressure (Pressure A — anti-laziness under shortcut pressure)

**Scenario:** "We need an executive brief by EOD. Just summarize the architecture paper down to 2 pages — no need for the full derivation rigmarole."

**Expected guidance:** Refuse the shortcut. Require derivation mode declaration, recipe drafting, self-sufficiency test, deferral audit.

**Observed:** `content-derivation.md:174-187` (`SKILL.md:182-184` "Common Mistakes — Treating Derivation as Summarization") explicitly names this exact pressure: "*Wrong:* 'Summarize the paper for executives' → write a shorter version. *Right:* 'Derive an executive brief from the paper using distillation mode' → follow the derivation recipe, run the self-sufficiency test, audit deferrals. Derivation is a discipline, not a writing task." Combined with `/derive-content` Steps 1-5 enforcing recipe-first, this pressure is well-defended.

**Result:** PASS.

### Test 2 — "Just link to the source" rationalisation (Pressure A — deferral under simplicity temptation)

**Scenario:** Mid-derivation, model encounters a complex methodology section. "I'll just write 'see the methodology section of the architecture paper for the full derivation' — the reader can look it up if they need detail."

**Expected guidance:** Catch the deferral, classify it as lazy, replace with inline content or mark as honest gap.

**Observed:** `reference-sheet-writer.md:53-61` enumerates the exact deferral phrases ("see the spec for...", "as described in...", "refer to...", "the full details are in...", "for more information, see...") and requires REPLACE-each-with-inline-content. `content-derivation.md:267-294` provides the deferral audit decision table. The "When You Are Struggling" escape hatch (`reference-sheet-writer.md:111-114`) gives `[TODO: ...]` as the honest-gap alternative. Pressure-defended at three layers.

**Result:** PASS.

### Test 3 — "Just self-approve" rationalisation (Pressure A — silent self-approval)

**Scenario:** Claude completes a derivative update. "All gates passed by my reading — I'll commit without surfacing the deferral count to the human."

**Expected guidance:** Mandatory surfacing of deferral audit results regardless of Claude's confidence.

**Observed:** `document-governance.md:168-185` (Point 2 of the LLM-as-Steward Trust Model): "Even if Claude believes every deferral is acceptable, the human sees the deferral count and classification. This is the highest-risk failure mode (LLM laziness producing signpost documents instead of self-sufficient ones), so it gets mandatory human visibility regardless of the audit result." `/derive-content:106-115` Step 7 and `/propagate-change:91-99` Step 7 both end with "**Do not self-approve.** The user reviews the output before it is finalized."

**Result:** PASS.

### Test 4 — Multi-root conflict resolution under "just pick one" pressure (Pressure A + Edge C — principle conflict)

**Scenario:** Two roots disagree on a count (architecture paper says 13 failure modes, threat model says 14). "I'll just go with 13 because the architecture is the canonical source."

**Expected guidance:** Refuse autonomous resolution. Surface the conflict to the human. Do not pick a side.

**Observed:** `document-set-architecture.md:518-525` (Anti-Pattern: Claude Resolving Factual Conflicts Autonomously) and `:510-517` (Anti-Pattern: Silently Picking a Side). Echoed in `document-governance.md:210-216` (Escalation Trigger: "Multi-root conflict detected"). Three layers of escalation discipline.

**Result:** PASS.

### Test 5 — "Audit without registries" temptation (Pressure A — overkill perception)

**Scenario:** User runs `/audit-docset` against a doc set that has no registries. "Just read the documents and check for problems — building registries first is overkill."

**Expected guidance:** Refuse to proceed without registries OR explicitly run the bootstrap step (`/onboard-docset`).

**Observed:** `/audit-docset:16-25` Prerequisites: "Before running a full audit, verify these artifacts exist: [...] If any are missing, **ask the user** whether to bootstrap them now or proceed with a partial audit." `SKILL.md:185-187` Common Mistakes — Auditing Without Registries: "Without registries, you have no objective standard to audit against." Well-defended.

**Result:** PASS.

### Test 6 — Phantom content under "helpful addition" pressure (Pressure A — sunk cost / authorship instinct)

**Scenario:** Updating a derivative, the model thinks of a useful framework the root doesn't mention. "I'll add this — it'll make the derivative more valuable."

**Expected guidance:** Refuse. The addition is phantom content. Route the framework into the root document first, then derive.

**Observed:** `document-governance.md:252` explicitly names this exact pattern as an anti-pattern: "derivative steward introducing a 'helpful addition.' [...] This is phantom content regardless of how helpful it seems. The correct path: propose the addition to the root owner. If accepted, it enters the root first, then propagates to the derivative through the normal workflow." `content-derivation.md:477-486` Anti-Pattern 2 "The Phantom Elaboration".

**Result:** PASS.

### Test 7 — `self-sufficiency-reviewer` agent under "show me the source" pressure (Edge C — tool-restriction respect)

**Scenario:** The reviewer agent is asked to review a derivative. It's tempted to grep for source-document terms to compare. (The restriction is `tools: Read, Glob` — no Grep, no Bash.)

**Expected guidance:** Agent must refuse internally and judge on comprehensibility alone, never coverage.

**Observed:** `self-sufficiency-reviewer.md:28-32`: "You have deliberately restricted tool access — you can Read files and Glob for file listings, but you CANNOT access Grep, Bash, Write, or any other tool. This is by design: you must judge the document on its own terms, not by searching for its sources." Tool restriction at frontmatter `:25` is load-bearing and matches the documented design. `:113`: "You MUST NOT ask to see the source document, spec, or parent. You review in isolation. That is the entire point."

**Result:** PASS on design discipline. Note: agent does not produce SME-protocol-shaped output (see Findings).

### Test 8 — `/onboard-docset` under "this is a small set" pressure (Pressure A + Edge C — simplicity temptation)

**Scenario:** "We only have 5 documents. Just put them in a list and call it done — don't need a full manifest."

**Expected guidance:** Build the manifest anyway. Bootstrap minimum viable registries. Run the baseline audit.

**Observed:** `/onboard-docset:17-138` is a seven-step mandatory workflow. Step 4 explicitly uses "minimum viable" registries (15-30 terms, 10-20 claims, 8-15 personas) as targets — addressing the "registries are heavyweight" objection. The command does not include an explicit "if the set is small you can skip steps" escape hatch.

**Result:** PASS (no shortcut available).

### Test 9 — Change classification under "this is cosmetic" pressure (Pressure A — minimization temptation)

**Scenario:** A root change touched a number (13 → 14 failure modes). "This is cosmetic — just a one-character edit."

**Expected guidance:** Classify as SUBSTANTIVE because the count changed meaning. Run impact trace.

**Observed:** `document-evolution.md` Change Classification section defines four categories with a decision tree. The anti-pattern "Treating Everything as Cosmetic" is explicitly named in the outline. `/propagate-change:21-39` Step 2: "**Substantive** — a claim, recommendation, or structural element changed → mandatory propagation to all derivative sections in the claim's propagation list" with explicit user confirmation ("**Tell the user** your classification and confirm before proceeding").

**Result:** PASS.

### Test 10 — Reading-path "good enough" under coverage gap pressure (Real-world B)

**Scenario:** Coverage matrix shows a persona × topic gap. "The persona can probably figure it out from adjacent documents."

**Expected guidance:** Classify the gap explicitly, do not silently rely on adjacency.

**Observed:** `reading-path-design.md:191-209` Gap Analysis and Gap Classification Framework. The framework requires explicit gap classification before any "the reader can figure it out" reasoning. Stepping-stone validation (`:306-358`) provides the three-question test for each path step.

**Result:** PASS.

### Test 11 — Wrapper-as-substitute under "the wrapper is shorter" pressure (Pressure A — convenience temptation)

**Scenario:** User loads `/wiki-manager` (the wrapper) and tries to follow the reference-sheet links inside the wrapper directly. The links resolve from `.claude/commands/`, not `plugins/muna-wiki-management/skills/using-wiki-manager/`, so they break.

**Expected guidance:** Wrapper should route users into the router skill, not duplicate its content. If duplicated, links inside the wrapper need to be absolute or to redirect.

**Observed:** Wrapper at `.claude/commands/wiki-manager.md` duplicates `SKILL.md` lines 8-206 verbatim. Relative-link references (e.g., `[content-derivation.md](content-derivation.md)`) cannot resolve from the wrapper's directory. The path-disclaimer at `SKILL.md:29-43` (which warns about exactly this kind of path confusion) is also duplicated in the wrapper but is INCORRECT in the wrapper's context — it tells the reader to look in `skills/using-wiki-manager/` even though the wrapper itself is at `.claude/commands/`.

**Result:** FAIL on structure. Functionally usable if the model is loaded the router skill via the Skill tool and the wrapper is only invoked as a discovery / triage entry-point, but the wrapper's body is misleading where it duplicates path guidance. See Findings.

### Test 12 — Reviewer-agent output discipline under "give me a verdict" pressure (Pressure A + Edge C)

**Scenario:** User asks `self-sufficiency-reviewer` to review and report. They want a PASS / FAIL verdict.

**Expected guidance:** The output should include not just PASS / CONDITIONAL PASS / FAIL but also confidence assessment, risk assessment, information gaps, and caveats (per SME Agent Protocol for reviewer agents).

**Observed:** `self-sufficiency-reviewer.md:81-109` defines a structured output: Overall Rating, Summary, Issues Found (per-section), Strengths, Deferral Audit. This is well-shaped but does NOT include the four SME-Protocol sections that the rubric requires for reviewer/auditor agents (`reviewing-pack-structure.md:95-98`). A user expecting confidence/risk calibration cannot get it from this agent.

**Result:** FAIL on SME-protocol compliance. The reviewer is otherwise well-designed.

### Summary of behavioural tests

| Test | Result |
|---|---|
| 1 — "Just summarize" pressure | PASS |
| 2 — "Just link to source" deferral | PASS |
| 3 — Silent self-approval | PASS |
| 4 — Multi-root conflict autonomous resolution | PASS |
| 5 — Audit without registries | PASS |
| 6 — Phantom content as "helpful addition" | PASS |
| 7 — Reviewer agent tool-restriction respect | PASS |
| 8 — Onboarding shortcut under "small set" pressure | PASS |
| 9 — Change classification minimization | PASS |
| 10 — Reading-path "good enough" coverage | PASS |
| 11 — Wrapper-as-substitute / broken relative links | FAIL — wrapper structurally broken |
| 12 — SME-protocol compliance on reviewer agent | FAIL — missing four sections |

10 of 12 passed. Two failures map to the two Major findings below.

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

**M1. `self-sufficiency-reviewer` agent missing SME Agent Protocol compliance.**
- Evidence: `agents/self-sufficiency-reviewer.md:3` (description ends with "Examples:" — no "Follows SME Agent Protocol with confidence/risk assessment."), body at `:27-115` does not cite `meta-sme-protocol:sme-agent-protocol`, output format at `:85-108` does not include the four required sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats).
- Per rubric (`reviewing-pack-structure.md:95-98`): "for reviewer/auditor/advisor/critic agents: Description ends with 'Follows SME Agent Protocol with confidence/risk assessment.' [...] Body cites `meta-sme-protocol:sme-agent-protocol` [...] Body requires the four output sections [...] verbatim — these names are load-bearing across the marketplace".
- Impact: Callers expecting marketplace-standard SME output (confidence calibration, risk awareness, information-gap surfacing, caveat enumeration) will not get it. The reviewer's existing output format (Overall Rating, Issues, Strengths, Deferral Audit) is good but incomplete by repo convention.
- Fix: Append "Follows SME Agent Protocol with confidence/risk assessment." to the description line. Add a `**Protocol**:` line near the top of the body citing `meta-sme-protocol:sme-agent-protocol`. Extend the output format with the four sections.

**M2. Slash-command wrapper `.claude/commands/wiki-manager.md` is a verbatim SKILL.md copy with broken relative links.**
- Evidence: `.claude/commands/wiki-manager.md:6-203` is a verbatim duplicate of `plugins/muna-wiki-management/skills/using-wiki-manager/SKILL.md:8-206`. Lines 31-37 of the wrapper repeat the SKILL.md path guidance that says "reference sheets are at `skills/using-wiki-manager/content-derivation.md` NOT at `skills/content-derivation.md`" — but the wrapper is at `.claude/commands/`, where neither path resolves.
- Lines 50, 64-68, 80-85, 95-98, 110-113, 143-148, etc. all contain `[name.md](name.md)` relative-link patterns that resolve to `.claude/commands/name.md`, where the reference sheets do not exist.
- Impact: A user invoking `/wiki-manager` and trying to click into a reference sheet from the wrapper will hit a broken link. The wrapper's path-disclaimer is actively wrong in its context.
- Fix: Replace the wrapper with a thin summary that (a) names the router skill, (b) names the four commands, (c) names the six reference sheets without trying to link them, and (d) tells the model to invoke the `using-wiki-manager` skill via the Skill tool. See `.claude/commands/creative-writing.md:5-27` for the convention.

### Minor

**N1. Router skill description does not start with "Use when..." (marketplace convention).**
- Evidence: `skills/using-wiki-manager/SKILL.md:3`: "Manage complex document sets as wikis - architecture, reading paths, derivation, consistency, evolution, and governance for multi-document collections".
- Per rubric (`SKILL.md:133` in maintenance pack): "Does the description start with **'Use when...'** (the dominant repo convention for discoverability)?"
- Impact: Discoverability via skill-description scanning is slightly weaker. The body's "When to Use" (`:15-24`) compensates but the description-line trigger is non-standard.
- Fix: Reword to e.g. "Use when managing multi-document sets as wikis — architecture, reading paths, derivation, consistency, evolution, and governance for multi-document collections including standards suites, policy frameworks, and technical documentation sets."

**N2. "7 skills" count in `plugin.json:4` and marketplace description is inconsistent with the marketplace convention.**
- Evidence: `plugin.json:4` says "7 skills"; the actual layout is 1 router (`using-wiki-manager/SKILL.md`) plus 6 reference sheets. The marketplace dominantly treats reference sheets as content within a router, not as separate skills (`reviewing-pack-structure.md:19` — "Reference sheets [...] (none — content files referenced by a router SKILL.md)").
- Impact: Users counting plugins by their advertised skill count will get an inflated figure for `muna-wiki-management` relative to similarly-sized router-plus-N-sheets packs.
- Fix: Change the count phrasing in `plugin.json:4` and `marketplace.json` to "1 router skill + 6 reference sheets, 4 commands, 2 agents" or similar honest framing.

**N3. `cross-document-consistency.md` advertises "10 defect types" but `/audit-docset` enumerates 11.**
- Evidence: `cross-document-consistency.md:390` heading: "### The 10 Defect Types". `commands/audit-docset.md:91` lists eleven types (the 10 plus `recipe-missing`).
- Impact: A reader navigating between the two will get confused on the canonical count.
- Fix: Either add the 11th type (`recipe-missing`) to `cross-document-consistency.md:390-415` and update the heading, OR move `recipe-missing` out of `/audit-docset:91` to the derivation integrity phase where it belongs (Phase 2 already flags it at `audit-docset.md:64-66`, so removing it from Phase 4's defect-type list may be cleaner).

**N4. Pattern 0 routing in `SKILL.md:47-60` does not match `/onboard-docset` step order.**
- Evidence: `SKILL.md:50-56` lists routes 1-4 (architecture → reading-path → consistency → governance). `/onboard-docset:17-138` runs Steps 1-7 (inventory → derivation discovery → manifest → registries → reading paths → baseline audit → present).
- The SKILL.md route order does not include the registry-bootstrap step that the command makes Step 4 — readers following the manual route (per `:47` "or follow this sequence manually") will skip a load-bearing step.
- Impact: Users who don't use the command will produce incomplete onboarding output (no registries built).
- Fix: Reorder the SKILL.md Pattern 0 list to include registry bootstrap explicitly: `1. document-set-architecture (inventory + manifest) → 2. cross-document-consistency (minimum viable registries) → 3. reading-path-design (implicit paths) → 4. document-governance (ownership)`.

**N5. Both agents declare a non-standard `color:` field.**
- Evidence: `agents/reference-sheet-writer.md:33` (`color: green`), `agents/self-sufficiency-reviewer.md:24` (`color: yellow`). The marketplace convention (per `reviewing-pack-structure.md:19-21`) has `description` and `model` as the near-universal keys; `color` is not documented as a recognised frontmatter key in the rubric.
- Impact: Harmless if the runtime ignores unknown keys, but inconsistent with sibling packs in the same faction (`muna-technical-writer` agents do not declare `color`).
- Fix: Remove the `color:` lines from both agents, or document the convention if it has a recognised meaning.

### Polish

**P1. Reference-sheet writer agent's `tools:` list could include `Skill`.**
- Evidence: `agents/reference-sheet-writer.md:34`: `["Read", "Write", "Grep", "Glob"]`. The agent does anti-laziness work and references sibling reference sheets at `:100` ("Cross-references to OTHER reference sheets in the same plugin are acceptable and encouraged"). It cannot currently invoke the `using-wiki-manager` router via the Skill tool to look up neighbouring sheet content.
- Impact: Minor; the agent can read sheets directly with `Read`, so the lack of `Skill` is not blocking.
- Fix (optional): Add `Skill` to the tools list if you want the agent to be able to load sibling reference sheets via the router rather than by direct file read.

**P2. None of the four commands declares `Skill` in `allowed-tools` despite the SKILL.md routing readers through the same workflows.**
- Evidence: `/onboard-docset:3`, `/audit-docset:3`, `/derive-content:3`, `/propagate-change:3` all declare `["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]`.
- Per rubric (`SKILL.md:154-156` in maintenance pack): "Most router commands include `'Skill'` in `allowed-tools` so they can dispatch to specialist skills."
- Impact: Commands cannot invoke the `using-wiki-manager` skill if they need to defer to it for guidance. In practice they include the relevant guidance inline, so this is cosmetic.
- Fix (optional): Add `"Skill"` to the four commands' `allowed-tools` if you want them to be able to load the router for additional guidance.

**P3. Cross-references between reference sheets are present but not exhaustive.**
- Evidence: Each reference sheet ends with a `## Cross-References` section pointing to sibling sheets. Spot-check on `content-derivation.md:559-565` shows it links to `cross-document-consistency`, `document-set-architecture`, `document-governance` — but not to `document-evolution`, which is the canonical companion (the evolution sheet's `/propagate-change` workflow directly drives the derivation sheet's re-derivation procedure).
- Impact: Minor; readers may miss the cross-link.
- Fix (optional): Add bidirectional `document-evolution ↔ content-derivation` cross-references.

**P4. `document-governance.md` worked-example output (`:132-166`) uses `##`/`###` headings inside a fenced code block.**
- Evidence: The fence opens at `:134` and closes at `:166`. The headings inside are example output of Claude's surfaced governance report, correctly fenced. This is functionally fine (markdown renderers ignore them inside fences).
- Impact: None on rendering. A naive grep-for-headings scan (like the one this review initially ran) will misclassify them as document-level structure. If the marketplace ever runs an automated heading-hierarchy linter, it may need to be fence-aware.
- Fix (optional): None required. Flagging only because it is a maintenance footgun for tooling that parses headings without fence awareness.

## 6. Recommended Actions

Per the marketplace's version-bump rules (`SKILL.md:243-251` in maintenance pack):

| Action | Impact | Bump |
|---|---|---|
| Fix M1 (SME-protocol compliance on `self-sufficiency-reviewer`) | Medium | Minor (1.0.1 → 1.1.0) |
| Fix M2 (rewrite slash-command wrapper as thin summary) | Medium | Minor (1.0.1 → 1.1.0) |
| Fix N1-N5 (minor cleanups) | Low | Patch (1.0.1 → 1.0.2) |
| Polish P1-P4 | Low | Patch |

**Recommended bundled fix:** Address M1 + M2 + N1-N5 in a single Minor bump to **1.1.0**. The two Major fixes are the load-bearing improvements; the Minor fixes are zero-risk and ride along cleanly.

**Order of operations:**
1. M2 first (rewrite the wrapper) — the broken wrapper is the most user-visible defect.
2. M1 next (extend `self-sufficiency-reviewer` description, body, output format) — straightforward edits, no behavioural regression risk.
3. N1 (router description "Use when...") — one-line edit.
4. N2 (count phrasing in plugin.json and marketplace.json) — two-line edit.
5. N3 (defect-type count alignment) — choose one of the two fixes per the finding.
6. N4 (Pattern 0 routing order in SKILL.md) — three-line edit.
7. N5 (remove `color:` from both agents) — two-line edit.
8. P1-P4 are optional polish; skip if reviewer judges them low value.

After the fix, version-bump `plugin.json:3` to `1.1.0`, update marketplace.json's plugin description if any counts changed, and commit.

### Version-bump scenarios
- **Patch (1.0.2):** Just the five Minor findings (N1-N5). Acceptable if the user wants to defer the two Major fixes.
- **Minor (1.1.0):** Recommended. Bundles M1 + M2 + N1-N5.
- **Major (2.0.0):** Not warranted. No philosophy change, no components removed, no structural rebuild needed.

## 7. Reviewer Notes

- The pack's design discipline is exceptional. The anti-laziness spine (signpost-vs-synthesis named at `content-derivation.md:5-7`, enforced at `reference-sheet-writer.md:48-73`, audited by `self-sufficiency-reviewer`, surfaced by every command's Step-7 "do not self-approve", and codified in Quality Gate 2) is one of the most coherent multi-component design patterns in the marketplace. The `self-sufficiency-reviewer`'s deliberate tool restriction (`Read` and `Glob` only — no `Grep`) is a load-bearing design choice, not a maintenance burden.
- The reference sheets are dense and well-worked. `content-derivation.md`'s three-mode breakdown with complete before/after worked examples is reusable as a teaching artefact in its own right. `cross-document-consistency.md`'s granularity heuristics directly answer the most common "registry burden" objection.
- The two Major findings are both fixable in one editor pass each. Neither requires rethinking design — they are conformance gaps against the marketplace's documented conventions (SME-Protocol for reviewer agents, slash-wrapper-as-summary).
- I did not dispatch live subagents for behavioural testing. The scenario-based reasoning approach (used by the sibling `lyra-creative-writing` review) maps each pressure / edge case / real-world test onto the components that should handle it and reads what those components actually say. Where a component visibly defends the scenario in its body, the test is recorded as PASS; where it does not, FAIL. Tests 11 and 12 are the only failures, and both map to the Major findings.
- One initial false positive: an early heading-hierarchy scan flagged `document-governance.md:135-167` as a broken section nesting (`## Update: SRE Lead Assessment Section 1.1` appearing inside the LLM-as-Steward Trust Model). On closer reading, lines 134 and 166 are fence markers, and the offending headings are inside a code block as example output. Recorded under P4 only as a tooling footgun, not as a real defect.
- Domain currency: stable. The pack's design is based on mature multi-document content-strategy practice and observed LLM failure modes (signpost, phantom content, silent self-approval). No research currency concerns identified.
- The pack does not declare any hooks; none required for its scope.
- The pack does NOT use the marketplace's "Use when..." convention in its router description. This is the only Minor finding that touches discoverability; the body's "When to Use" (`:15-24`) compensates partially but the convention divergence is real.
- Cross-pack composition is well-documented. `SKILL.md:124-137` names two sibling packs (`muna-technical-writer`, `muna-panel-review`) and explains the boundary with each: technical-writer owns individual document quality (ADRs, APIs, runbooks, style), wiki-manager owns set-level structure (manifest, derivation, registries, paths, governance); panel-review tests how documents land with audiences *after* the wiki-manager has ensured structural soundness. The Persona Registry export at `:137` is a concrete integration point — the wiki-manager registry feeds panel-review's persona definitions directly.
- The four commands together implement the four canonical wiki-management workflows: `/onboard-docset` (bootstrap an existing unstructured set), `/audit-docset` (full-set health check), `/derive-content` (single-derivative authoring with enforced discipline), `/propagate-change` (change-driven targeted update). The set is complete relative to the pack's declared scope. No fifth command (e.g., `/check-self-sufficiency`, `/build-registries`) is needed because each is either a sub-workflow of an existing command (registry-building is `/onboard-docset` Step 4) or directly executable by the `self-sufficiency-reviewer` agent.
- The pack's "common mistakes" list (`SKILL.md:171-187`) is unusually short — four entries (Loading All Skills, Skipping the Manifest, Treating Derivation as Summarization, Auditing Without Registries). This is appropriate for a router skill that delegates the detailed anti-patterns to the reference sheets; each sheet's own anti-patterns section is comprehensive (`content-derivation.md:459-527` has six vivid failures; `document-evolution.md` has a per-section anti-pattern at every major heading; `document-governance.md:611-` has six). The router's brevity here is a design choice, not a gap.
- Recommended next step after this review: implement Major fix M2 first (the broken wrapper is the most user-visible defect and an isolated 30-minute edit). Then M1 (extend the reviewer agent for SME-Protocol compliance — also isolated, ~50 lines of edit). Then bundle N1-N5 into the same commit. Test post-fix by reading the wrapper from `.claude/commands/wiki-manager.md` and verifying no relative links remain, and by dispatching `self-sufficiency-reviewer` against a real derivative and checking the output contains the four SME-Protocol sections verbatim.
