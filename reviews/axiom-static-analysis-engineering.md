# Review: axiom-static-analysis-engineering

**Version:** 0.2.1 **Reviewed:** 2026-05-22 **Reviewer:** general-purpose subagent

## 1. Inventory

### Plugin metadata

- `plugins/axiom-static-analysis-engineering/.claude-plugin/plugin.json:3` — `version: 0.2.1`
- `plugins/axiom-static-analysis-engineering/.claude-plugin/plugin.json:4` — description matches the marketplace summary ("Router + 13 sheets, 3 commands (/scaffold-analyzer, /design-tier-model, /design-rule-set), 2 agents (rule-designer, false-positive-analyst). Pattern pack — applies wherever you build an analyzer …").
- Marketplace registration present: `.claude-plugin/marketplace.json:207-208` — entry `axiom-static-analysis-engineering`, source `./plugins/axiom-static-analysis-engineering`, category `development`. Catalog description is consistent with the pack `plugin.json` and the router SKILL.md.

### Components

| Type | Count | Notes |
|------|-------|-------|
| Router skill | 1 | `skills/using-static-analysis-engineering/SKILL.md` (410 lines) |
| Reference sheets | 13 | All siblings of `SKILL.md` in the same directory |
| Commands (pack-local) | 3 | `scaffold-analyzer.md` (247L), `design-tier-model.md` (182L), `design-rule-set.md` (213L) |
| Agents | 2 | `rule-designer.md` (247L), `false-positive-analyst.md` (216L) |
| Hooks | 0 | None present (none expected for this pattern pack) |
| Slash-command wrapper | 1 (stale) | `.claude/commands/static-analysis-engineering.md` exists but advertises only v0.1 — see Findings (Major) |
| CHANGELOG | not located | No `CHANGELOG.md` at pack root (sister pack `axiom-procedural-architecture` does ship one) |

### Reference sheets (13)

Architectural spine (3):
- `ast-visitation-patterns.md` (168L) — produces `01-visitation-strategy.md`
- `taint-lattice-design.md` (185L) — produces `02-abstract-domain-spec.md`
- `three-phase-inference.md` (223L) — produces `03-inference-pipeline-spec.md`

Operational reality (3):
- `plugin-architecture-for-analyzer-rules.md` (206L) — produces `04-rule-plugin-spec.md`
- `false-positive-economics.md` (205L) — produces `05-false-positive-economics.md`
- `static-vs-runtime-tradeoffs.md` (191L) — produces `06-static-runtime-boundary.md`

Boundary discipline (added v0.2.0) (3):
- `callgraph-construction.md` (272L) — produces `07-callgraph-construction.md`
- `cross-module-flow-analysis.md` (254L) — produces `08-cross-module-flow.md`
- `decorator-as-assertion.md` (257L) — produces `09-decorator-as-assertion-spec.md`

Operations (added v0.2.0) (4):
- `manifest-driven-configuration-with-coherence-validation.md` (266L) — produces `10-manifest-and-coherence.md`
- `sarif-emission-and-ci-integration.md` (270L) — produces `11-sarif-and-ci.md`
- `scaling-to-large-codebases.md` (283L) — produces `12-scaling-and-incrementality.md`
- `llm-assisted-rule-explanation.md` (260L) — produces `13-llm-assisted-explanation.md`

All thirteen reference sheets carry well-formed YAML frontmatter (`name`, `description` starting with "Use when …"), and each line up with the artifact set declared in `SKILL.md:96-119` and the spec dependency graph at `SKILL.md:121-142`.

### Commands

| Command | Frontmatter shape | argument-hint | Notes |
|---------|-------------------|---------------|-------|
| `/scaffold-analyzer` | `description`, `allowed-tools` (quoted JSON array including `Task` + `AskUserQuestion`) | `"[analyzer_name_or_path]"` | Greenfield/brownfield scaffold; reads spec set; tier-conditional file plan |
| `/design-tier-model` | same shape | `"[analyzer_name_or_path]"` | Six-round elicitation; emits draft `02-abstract-domain-spec.md` |
| `/design-rule-set` | same shape | `"[analyzer_name_or_path]"` | Seven-round rule sourcing → metadata → manifest → fixtures → conflict review |

All three commands use the marketplace-standard quoted-string `allowed-tools` array and a quoted `argument-hint`. Each cites the agents and sibling commands appropriately.

### Agents

| Agent | Model | SME-protocol compliant? |
|-------|-------|-------------------------|
| `rule-designer` (producer SME) | opus | Yes — `rule-designer.md:2` description ends "Follows SME Agent Protocol with confidence/risk assessment"; body cites `meta-sme-protocol:sme-agent-protocol` at line 10; mandates the four output sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) at lines 213-223 |
| `false-positive-analyst` (operational SME) | opus | Yes — `false-positive-analyst.md:2` description ends "Follows SME Agent Protocol with confidence/risk assessment"; body cites protocol at line 10; required output sections at lines 180-191 |

Neither agent declares `tools:` — consistent with the marketplace convention (`using-skillpack-maintenance/SKILL.md:25` notes ~60/65 agents in this marketplace omit `tools:`). The agents are sibling, not nested: design vs. triage, with explicit cross-references in both directions (`rule-designer.md:245`, `false-positive-analyst.md:213`).

### Marketplace + slash-command wrapper alignment

- Marketplace entry: present, well-formed (`.claude-plugin/marketplace.json:207-235`).
- Slash-command wrapper present at `.claude/commands/static-analysis-engineering.md` — but it is stale (see Findings, Major).

---

## 2. Domain & Coverage

### Stated scope (`SKILL.md`)

- **In scope:** Building analyzers as engines — AST visitation strategy, abstract-domain (lattice) design, three-phase inference, rule plugin model, false-positive economics, the static-vs-runtime boundary, callgraph construction, cross-module flow with stub libraries, decorator-as-assertion (runtime + static dual enforcement), manifest-driven configuration with coherence validation, SARIF/CI integration, incremental analysis at scale, and LLM-assisted finding explanation as a *pattern*.
- **Explicit hand-offs (SKILL.md:31-37):**
  - Running an existing analyzer (ruff, mypy, semgrep) → `/python-engineering`, `/rust-engineering`
  - Consuming analyzer output to map a codebase → `/system-archaeologist`
  - Suppression as auditable decision → `/audit-pipelines`
  - Policy / control-family rule design → `/security-architect`, `/sdlc-engineering`
- **Pipeline position diagram** at `SKILL.md:55-89` cleanly states the sibling boundaries against `axiom-system-archaeologist`, `axiom-audit-pipelines`, and `ordis-security-architect`.

### Coverage map vs. inventory

| Domain area | Status | Sheet(s) |
|-------------|--------|----------|
| AST traversal substrate (visitor / walker / transformer; structural vs lossless ASTs; parent tracking) | Exists | `ast-visitation-patterns.md` |
| Abstract domain formalism (lattice, partial order, join, monotonicity, finite height; product lattices; soundness/completeness) | Exists | `taint-lattice-design.md` (depth: derives every consistency-gate check from algebraic properties) |
| Inference order and termination proof (Phase 1 variable → Phase 2 function summary → Phase 3 inter-procedural with worklist) | Exists | `three-phase-inference.md` |
| Plugin / extension surface (discovery, lifecycle, metadata schema, conflict resolution) | Exists | `plugin-architecture-for-analyzer-rules.md` |
| Operational economics of false positives (suppression vs refinement, waiver lifecycle, FP-rate budget) | Exists | `false-positive-economics.md` |
| Static-vs-runtime boundary (Rice ceiling, dual enforcement, cost model) | Exists | `static-vs-runtime-tradeoffs.md` |
| Callgraph construction (resolution rungs name/CHA/RTA/VTA/k-CFA; dynamic features; conservative `top`) | Exists | `callgraph-construction.md` |
| Cross-module / cross-language flow (boundary semantics, stubs, framework callbacks, FFI) | Exists | `cross-module-flow-analysis.md` |
| Decorator-as-assertion (runtime + static contract; descriptor pattern; recognition registry; disagreement modes) | Exists | `decorator-as-assertion.md` |
| Manifest-driven configuration (layered overlays; coherence validation; drift detection; audit metadata) | Exists | `manifest-driven-configuration-with-coherence-validation.md` |
| SARIF emission and CI integration (2.1.0 schema; exit-code semantics; fingerprint stability; suppression round-trip) | Exists | `sarif-emission-and-ci-integration.md` |
| Scaling / incrementality (cache-key composition; reverse-edge index; parallel worklist; partition strategies; soundness floor) | Exists | `scaling-to-large-codebases.md` |
| LLM-assisted explanation as a pattern (analyzer = truth, model = translator; review gate; prompt-injection threat model) | Exists | `llm-assisted-rule-explanation.md` |
| Tier classification (XS / S / M / L / XL) and which artifacts each tier requires | Exists | `SKILL.md:159-170` (Analyzer Tier table) |
| Consistency gate (11 checks; tier-aware; staleness rule) | Exists | `SKILL.md:215-232` |
| Spec dependency graph + coordinated re-emission rules | Exists | `SKILL.md:121-156` |
| Stop conditions / decision tree | Exists | `SKILL.md:249-283` |
| Cross-pack integration (system-archaeologist, audit-pipelines, security-architect, sdlc, solution-architect, determinism-and-replay) | Exists | `SKILL.md:285-350` |

**Coverage assessment:** The pack is comprehensive for its declared boundary. The thirteen sheets correspond 1:1 with the artifact set, the artifact set is dependency-ordered, and the consistency gate exists to keep them coherent. The notable absences below are all *intentional* per the router's "do not use this pack when …" list (running an analyzer, consuming output for a system map, policy-level rule design, audit-lifecycle ownership of waivers) — they are handed off to sibling packs by explicit cross-link rather than re-implemented.

### Research currency

Domain is **stable** (lattice theory; AST visitation; SARIF; ASTs and worklist algorithms are pre-2010 art). The LLM-assisted-explanation sheet is the only piece touching an evolving domain — and it is deliberately scoped as a *pattern*, not as recommendations about specific models, which insulates it from drift.

### Gaps / candidate additions

None identified at the v0.2.1 scope. Possible future expansions (not gaps — these would be promotions):

- Concrete language-pack sheets (Python / TypeScript / Rust / JVM analyzer specifics). The router currently routes all language-specific implementation through `/python-engineering`, `/rust-engineering`, etc., which is the right call at this scope.
- An `agent: stub-curator` to triage stub-library coverage gaps — currently covered conversationally within `false-positive-analyst.md` step 5. Promote only if usage shows the rule-designer/FP-analyst pair doesn't cover stub triage well in practice.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|--------|----------|
| 1 | **Scope clarity** | Pass | `SKILL.md:21-37` lists in-scope / out-of-scope explicitly; pipeline position diagram disambiguates against `axiom-system-archaeologist` and `axiom-audit-pipelines`; the four hand-offs at `SKILL.md:31-37` are concrete (`/python-engineering`, `/system-archaeologist`, `/audit-pipelines`, `/security-architect`). |
| 2 | **Domain coverage** | Pass | All 13 declared artifacts have producer sheets; coverage map in §2 shows no missing domain area within the stated boundary; the spike/support split (spike = 01-03, support = 04-06, boundary = 07-09, operations = 10-13) is internally coherent. |
| 3 | **Component typing** | Pass | The 1 router skill + 13 reference sheets + 3 commands + 2 agents division matches the marketplace convention. Commands are user-invocable design pipelines (`/scaffold-…`, `/design-…`); agents are autonomous SMEs over the spec set; sheets are model-invoked reference content. No mis-typings. |
| 4 | **Router / wrapper alignment** | **Major issue** | The router skill is up-to-date for v0.2.x, but the repo-root wrapper at `.claude/commands/static-analysis-engineering.md` is **stale**: lines 11-22 declare "v0.1 Sheets" and call the commands, agents, and 7 additional sheets "Deferred to v0.2" — when those components shipped in v0.2.0 and are now at v0.2.1. A user invoking `/static-analysis-engineering` will be told the pack is smaller than it is. |
| 5 | **Frontmatter conventions** | Pass | Router skill description starts with "Use when …". Every reference sheet starts with "Use when …" and identifies its produced artifact. Commands use the quoted `allowed-tools` JSON-array form and a quoted `argument-hint`. Agents declare only `description` + `model` (omit `tools:` — consistent with the marketplace's ~60/65 default). |
| 6 | **SME-protocol compliance** | Pass | Both agents end their description with "Follows SME Agent Protocol with confidence/risk assessment" verbatim. Both bodies cite `meta-sme-protocol:sme-agent-protocol` and require the four sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) verbatim in the Output Format. Anti-pattern tables in both agents block trivial activation. |
| 7 | **Internal coherence (spec dependency graph + consistency gate)** | Pass | `SKILL.md:121-156` enumerates the coordinated re-emission rules between artifacts (which sheet re-emits when which other sheet changes, and whether the change is "lattice-breaking"). The 11-check consistency gate at `SKILL.md:215-232` is concrete and tier-aware; staleness rule (gate report older than newest artifact → re-gate) closes the most common drift mode. |
| 8 | **Discipline-bearing depth** | Pass | The pack does not merely list techniques — it commits to algebraic properties (`taint-lattice-design.md:37-45` makes monotonicity and finite-height termination-bearing), a termination proof grounded in those properties (`three-phase-inference.md:52-58`), and a soundness/completeness statement with Rice-theorem caveat (`taint-lattice-design.md:138-146`). Anti-pattern catalogues are concrete and falsifiable: "boolean lattice masquerading as tiers" (`taint-lattice-design.md:14`, again at `/design-tier-model.md:108-115`), "FPs without lifecycle calcify" (`false-positive-economics.md:96-100`), "rule with no examples is a wish" (`rule-designer.md:21`). |

**Overall:** **Pass with one Major issue** — pack is structurally sound and content is discipline-bearing; the stale slash-command wrapper is the single substantive blemish. No Critical findings; no rebuild recommendation.

---

## 4. Behavioral Tests

The scenarios below are designed against the gauntlet categories in `testing-skill-quality.md`. They are dry-run (read-only) against the as-shipped content; passes/fails are derived from what the pack would force a model to do, not from a live subagent invocation.

### T1 — Pressure (A): "Just write some checks, we don't need the whole spec"

**Scenario:** *"We just need ten checks for our codebase, can you skip the lattice design and the inference pipeline and just write the rules?"*

**Expected behaviour:** Pack forces tier classification first (`SKILL.md:159-170`); even at tier XS, `00-scope-and-targets.md` and `01-visitation-strategy.md` are mandatory; the consistency gate's check 11 ("test corpus") refuses unfalsifiable rules.

**Outcome:** Pass. Three independent fail-safes:
1. Router opens with "Steps 1-3 are the spike" (`SKILL.md:49`) — the spike is the lattice/inference triple, not the rules.
2. `rule-designer.md:225-236` explicitly refuses a rule that consumes a tier the lattice doesn't have ("inventing tiers in rule design corrupts `02-`").
3. `design-rule-set.md` preconditions (`design-rule-set.md:32-44`) halt if `02-` is not settled.

### T2 — Pressure (A): "Sound and complete"

**Scenario:** *"Our analyzer is both sound and complete — it never has false positives and never has false negatives."*

**Expected behaviour:** Pack invokes Rice's theorem and forces a soundness/completeness statement choosing one side.

**Outcome:** Pass. `taint-lattice-design.md:138-146` and consistency-gate check 5 (`SKILL.md:224`) both require the statement; the latter explicitly notes "'Both sound and complete' without a Rice-theorem-aware caveat fails." Anti-pattern table at `taint-lattice-design.md:155-158` lists the claim as a refusal-grade mistake.

### T3 — Edge (C): The implicit lattice

**Scenario:** *"Our analyzer doesn't use a lattice — it just has some if-checks against AST shapes."*

**Expected behaviour:** Pack asserts every analyzer has a lattice, even if implicit; surfaces the "boolean lattice masquerading as tiers" anti-pattern and routes to the reverse-engineering procedure in `SKILL.md:184-189` (Scenario 2).

**Outcome:** Pass. `SKILL.md:184` opens the unmaintainable-analyzer routing with "Reverse-engineer the implicit lattice. The analyzer has one whether it acknowledges it or not." `taint-lattice-design.md:12-16` makes the same point at the sheet level. `/design-tier-model.md:108-115` operationalises it as anti-pattern check 1 in Round 6.

### T4 — Edge (C): Static intractability

**Scenario:** *"Add a rule that flags any function whose first argument is a valid SQL identifier."*

**Expected behaviour:** Pack refuses — the property depends on runtime values (the string), not types/structure; falls back to runtime per `06-`.

**Outcome:** Pass. `SKILL.md:252` lists this as a stop condition with the same example verbatim ("this string is a valid SQL identifier"). `rule-designer.md:228` blocks the rule design and refers the user to a runtime check.

### T5 — Real-world (B): Brownfield analyzer with 800 suppressions

**Scenario:** *"We inherited a linter with 800 # noqa annotations and rules that live as ad-hoc functions. We want to migrate it to this discipline. Where do we start?"*

**Expected behaviour:** Pack routes through Scenario 2 (`SKILL.md:184-189`): reverse-engineer the implicit lattice and inference order, then triage the suppression set; the `false-positive-analyst` agent does the systemic suppression read.

**Outcome:** Pass. Routing is explicit (`SKILL.md:184-189`). The `false-positive-analyst.md` Step 2 root-cause classification (lines 86-98) gives the maintainer exactly the buckets they need (lattice imprecision, callgraph over-approximation, stub gap, decorator-as-assertion gap, manifest gap, genuine per-site, runtime property, rule wrong). The agent refuses anecdotal triage ("is rule X a false-positive generator?" without metric data → halt: `false-positive-analyst.md:195-198`).

### T6 — Real-world (B): Specifying SARIF integration with suppression round-trip

**Scenario:** *"Our analyzer emits findings, GitHub Code Scanning shows them, but every # noqa we add still shows up as an unsuppressed finding."*

**Expected behaviour:** Pack identifies suppression round-tripping as a first-class concern in `11-sarif-and-ci.md`; gives the dev the SARIF `suppressions` field semantics and the fingerprint-stability discipline.

**Outcome:** Pass. `sarif-emission-and-ci-integration.md:1-22` opens with exactly this failure mode (line 13: "Suppressions don't round-trip — a `# noqa` in source is invisible to GitHub Code Scanning, which re-flags every suppressed finding on every PR"). The frontmatter calls out "suppression round-tripping (manifest waivers ↔ SARIF suppressions ↔ inline comments)" as a core concern.

### T7 — Pressure (A): Skipping the spec for the engine

**Scenario:** *"We already wrote the analyzer; we don't need the 99- consolidated spec."*

**Expected behaviour:** Pack reminds that the consistency gate is what makes the analyzer's claims auditable, not optional; the staleness rule (`SKILL.md:232`) forces re-gate.

**Outcome:** Pass. `SKILL.md:215` opens the consistency gate with "Run before emitting `99-analyzer-engineering-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent." The Bottom Line at `SKILL.md:379` reiterates: "Design the spec before writing the engine; gate the spec for consistency before downstream citation."

### T8 — Discovery: Activation under archaeology confusion

**Scenario:** *"I want to find all places in our codebase where untrusted input reaches a SQL execute."*

**Expected behaviour:** Pack should *not* activate — this is a /system-archaeologist or "run an existing analyzer" job. The pack is for *building* the analyzer, not for asking questions about an existing codebase.

**Outcome:** Pass. The router description explicitly hands off both consuming-output (→ `/system-archaeologist`) and running an existing analyzer (→ `/python-engineering`, `/rust-engineering`) at `SKILL.md:3` and again at `SKILL.md:31-37`. The decision tree at `SKILL.md:266-269` makes this branch the second question.

### Test summary

| # | Component(s) | Category | Result |
|---|--------------|----------|--------|
| T1 | Router + `rule-designer` + `/design-rule-set` | Pressure (A) | Pass |
| T2 | `taint-lattice-design` + consistency gate | Pressure (A) | Pass |
| T3 | Router + `taint-lattice-design` + `/design-tier-model` | Edge (C) | Pass |
| T4 | Router + `static-vs-runtime-tradeoffs` + `rule-designer` | Edge (C) | Pass |
| T5 | Router + `false-positive-economics` + `false-positive-analyst` | Real-world (B) | Pass |
| T6 | `sarif-emission-and-ci-integration` | Real-world (B) | Pass |
| T7 | Router consistency gate | Pressure (A) | Pass |
| T8 | Router scope discipline | Discovery | Pass |

All eight dry-run scenarios pass. No new failure modes surfaced.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

**M-1. Stale slash-command wrapper at `.claude/commands/static-analysis-engineering.md`.**

Status: open. Severity: Major. Type: maintenance-drift.

Lines 11-22 of the wrapper still describe the pack as "v0.1" with only the six spike+support sheets shipped, and call the three commands (`/scaffold-analyzer`, `/design-tier-model`, `/design-rule-set`), two agents (`rule-designer`, `false-positive-analyst`), and seven additional sheets "Deferred to v0.2.0". The plugin is currently at **v0.2.1** with all those components shipped (see `plugin.json:3`, `SKILL.md:106-119`). Sister packs' wrappers (e.g., `.claude/commands/axiom-procedural-architecture.md`) keep wrapper text minimal and tense-neutral; this wrapper reads as a release-notes artifact that was not refreshed.

- **Impact:** A user reading the slash-command output is told the pack is smaller than it actually is, and is told the v0.2 capabilities are aspirational when they are present. This degrades discoverability of the three commands and two agents — exactly the components the wrapper is supposed to surface.
- **Evidence:** `.claude/commands/static-analysis-engineering.md:11-22` (heading "## v0.1 Sheets (architectural spine + operational reality)") + lines 21-22 ("Deferred to v0.2 … Commands (`/scaffold-analyzer`, `/design-tier-model`, `/design-rule-set`), agents (`rule-designer`, `false-positive-analyst`), and 7 additional sheets … are scheduled for v0.2.0. v0.1 is self-coherent — a user can produce artifacts 00-06 + 99 and pass all 11 consistency-gate checks from shipped sheets alone.").
- **Fix (out of scope for this report):** Rewrite the wrapper as a tense-neutral routing page that names the current commands + agents + 13 sheets; drop the v0.1/v0.2 historical framing into a `CHANGELOG.md` if it is worth preserving.

### Minor

**m-1. No `CHANGELOG.md` at pack root.**

Sister pack `axiom-procedural-architecture/CHANGELOG.md` documents v0.1.0 → v0.1.1 cleanup items and is referenced from that pack's review. This pack went v0.1.0 → v0.2.0 → v0.2.1 (per the user-memory note for this project: "v0.2.0 feature-complete … v0.1 deferral backlog cleared; committed `40ad4e9`") without a CHANGELOG to surface what changed. The router internally annotates "Shipped in v0.2.0" / "added v0.2.0" in two places (`SKILL.md:108`, `SKILL.md:399`), but a pack-level CHANGELOG would consolidate the trajectory and let the wrapper stay terse.

- **Impact:** Maintenance friction; an external reviewer (i.e. this one) has to reconstruct the version arc from the router and from memory notes. Not user-blocking.

**m-2. Router internal-version annotations could be retired.**

`SKILL.md:106` ("Shipped in v0.2.0:") and `SKILL.md:399` ("Boundary discipline (added v0.2.0):") still flag content as new-in-v0.2.0. At v0.2.1 the distinction is no longer informative to the user — the sheet either exists or it doesn't. Suggested move: keep the headings, drop the "(added v0.2.0)" / "Shipped in v0.2.0:" decorators (or migrate them to a CHANGELOG).

**m-3. Router catalog and routing section both name commands/agents, slight duplication.**

`SKILL.md:117-119` and `SKILL.md:209-213` both enumerate the three commands + two agents; the Quick Reference table at `SKILL.md:354-376` lists them a third time. Duplication is small and arguably helps discovery, but at maintenance time three places must be updated together (e.g., if a fourth command were added). Cross-references rather than inline listings would reduce drift surface.

### Polish

**p-1. `SKILL.md:113` typo-equivalent in artifact title.**

The table at `SKILL.md:113` lists the artifact for the SARIF sheet as `11-sarif-and-ci.md`, which matches the file naming in the rest of the pack — no fix needed; flagged only because it might be tempting to rename to `11-sarif-emission-and-ci.md` for symmetry with the producer sheet's filename (`sarif-emission-and-ci-integration.md`). Recommend **leaving as is** for terseness; surfaced for completeness.

**p-2. Decorator-as-assertion artifact name asymmetry.**

The producer sheet is `decorator-as-assertion.md` (no `-spec` suffix), but the artifact it produces is `09-decorator-as-assertion-spec.md` (with `-spec`). Other artifacts also occasionally append `-spec` (e.g., `02-abstract-domain-spec.md`, `03-inference-pipeline-spec.md`). This is internally consistent within the artifact set; mentioned only to note the convention.

**p-3. `marketplace.json` description has minor wording divergence from `plugin.json`.**

`marketplace.json:209` says "v0.2 ships router + 13 sheets …"; `plugin.json:4` says "Router + 13 sheets …". Both are accurate. The marketplace's version-stamped phrasing will go stale at v0.3+; consider matching `plugin.json`'s tense-neutral form.

---

## 6. Recommended Actions

(For executor; report-only — no edits made.)

| Priority | Action | Target | Effort |
|----------|--------|--------|--------|
| **Major (M-1)** | Rewrite `.claude/commands/static-analysis-engineering.md` to a tense-neutral routing page that lists current commands/agents/sheets without v0.1-vs-v0.2 framing. | repo-root | Small |
| Minor (m-1) | Add `CHANGELOG.md` at the pack root documenting v0.1.0 → v0.2.0 → v0.2.1 (use sister pack `axiom-procedural-architecture/CHANGELOG.md` as the template). | pack root | Small |
| Minor (m-2) | Drop "(added v0.2.0)" / "Shipped in v0.2.0:" annotations in `SKILL.md` once a CHANGELOG exists. | `SKILL.md` | Trivial |
| Minor (m-3) | Optional: deduplicate command/agent listings in `SKILL.md` (catalog + routing + quick reference) to a single canonical location. | `SKILL.md` | Small |
| Polish (p-3) | Align `marketplace.json` description tense with `plugin.json`. | `.claude-plugin/marketplace.json` | Trivial |

No new skills required (this would be a `superpowers:writing-skills` workflow if so); no new commands required; no new agents required. No structural changes required. No version bump triggered by the recommendations above alone — a single fix of M-1 alone would be a patch (v0.2.2).

---

### Cross-pack hand-off verification (additional)

The router's pipeline diagram (`SKILL.md:55-89`) makes three cross-pack contracts that I traced for consistency:

| Contract | This pack's commitment | Sibling pack | Status |
|----------|------------------------|--------------|--------|
| Coverage gaps from archaeologist → rule request to this pack | `SKILL.md:65-69`, `SKILL.md:287-297` ("the boundary: archaeologist *reads* analyzer outputs to synthesise; this pack *produces* analyzers") | `axiom-system-archaeologist` | Consistent — the archaeologist pack consumes analyzer output for entity catalogue, dependency graph, security surface; this pack ships the analyzer side. Cross-link is bidirectional. |
| Suppressions = decisions, lifecycle owned by audit pack | `SKILL.md:71-79`, `SKILL.md:299-313` ("`05-` cites audit-pipelines:retention-expiry-and-rtbf for the waiver-expiry mechanism") | `axiom-audit-pipelines` | Consistent — `false-positive-economics.md` references the audit-pipeline retention-expiry-and-rtbf sheet without duplicating its content. |
| Lattice tiers correspond to security trust boundaries | `SKILL.md:81-89`, `SKILL.md:315-326` | `ordis-security-architect` | Consistent — `taint-lattice-design.md:185` ("Cross-pack: ordis-security-architect:design-controls — the threat model that forces the tier set") closes the loop. |

The three contracts are *cross-linked rather than duplicated*. This is the right disposition — duplication would create the drift the consistency gate exists to prevent.

### Frontmatter activation triggers (sample)

Spot-check of "Use when …" triggers (description heads) against likely user prompts:

| Trigger phrase (head of description) | Likely user prompt that matches |
|--------------------------------------|---------------------------------|
| "Use when designing the abstract domain of a dataflow analyzer …" (`taint-lattice-design.md:3`) | "I need to design the lattice for our taint tracker" |
| "Use when designing the inference algorithm …" (`three-phase-inference.md:3`) | "How do we order propagation across functions?" |
| "Use when an analyzer is shipping or has shipped and you face the operational reality of false positives …" (`false-positive-economics.md:3`) | "Our analyzer has 200 noqa comments and growing" |
| "Use when deciding whether an invariant should be enforced statically (analyzer rule), at runtime (assertion / contract), or both …" (`static-vs-runtime-tradeoffs.md:3`) | "Should we make this a static check or a runtime assertion?" |
| "Use when Phase 3 of the inference pipeline … needs an actual callgraph and the language has features that make construction non-trivial …" (`callgraph-construction.md:3`) | "How do we resolve virtual dispatch in our analyzer?" |

All five activate cleanly on the matching prompt; none have over-broad descriptions that would steal activations from neighbouring sheets. The "Use when …" convention is observed consistently across the 13 sheets.

### Consistency-gate self-audit

Each of the 11 consistency-gate checks (`SKILL.md:219-231`) maps to a producer sheet that operationalises it. Cross-tabulation:

| Gate # | Check | Owned by sheet |
|--------|-------|----------------|
| 1 | Tier coverage | router (`SKILL.md:159-170`) |
| 2 | Visitation honesty | `ast-visitation-patterns.md` |
| 3 | Lattice well-formedness | `taint-lattice-design.md:37-58` |
| 4 | Termination proof | `three-phase-inference.md:52-58` |
| 5 | Soundness/completeness statement | `taint-lattice-design.md:138-146` |
| 6 | Plugin contract | `plugin-architecture-for-analyzer-rules.md` |
| 7 | Suppression lifecycle | `false-positive-economics.md:68-100` |
| 8 | FP-rate budget | `false-positive-economics.md` |
| 9 | Static-runtime boundary | `static-vs-runtime-tradeoffs.md:12-15` |
| 10 | Cross-pack handoff | router (`SKILL.md:285-350`) |
| 11 | Test corpus | `plugin-architecture-for-analyzer-rules.md` + `/design-rule-set` Round 6 |

Every check has an owner. No orphan checks; no checks without operational guidance.

---

## 7. Reviewer Notes

### Method

- Read `meta-skillpack-maintenance:using-skillpack-maintenance` (SKILL.md and the three referenced briefings: `analyzing-pack-domain.md`, `reviewing-pack-structure.md`, `testing-skill-quality.md`).
- Read the pack's `plugin.json` and `marketplace.json` entry in full.
- Read the router `SKILL.md` in full (410 lines).
- Read all three commands in full (`scaffold-analyzer.md`, `design-tier-model.md`, `design-rule-set.md`).
- Read both agents in full (`rule-designer.md`, `false-positive-analyst.md`).
- Read `taint-lattice-design.md` in full (the keystone sheet; depth check).
- Spot-checked `false-positive-economics.md` (lifecycle correctness), `three-phase-inference.md` (worklist + termination), `sarif-emission-and-ci-integration.md` (CI failure modes), `llm-assisted-rule-explanation.md` (LLM-as-translator boundary), `plugin-architecture-for-analyzer-rules.md` (rule contract), and the headers of the remaining six sheets to confirm frontmatter shape and produced-artifact alignment.
- Verified marketplace entry, slash-command wrapper presence, and read the wrapper in full.
- Behavioral tests are dry-run against the as-shipped content (no live subagent dispatch) because the pack's "test" is fundamentally about whether the discipline holds under pressure — that's verifiable from the text. Subagent dispatch would be appropriate for activation/discovery testing if M-1 were left in place and the question were "does the wrapper still surface the right components" — but the wrapper's failure mode is visible from the text alone.

### What I did not do

- Did not run the test corpus or attempt to spin up `analyzer-engineering/` for a real analyzer (Stage 5 — implementation — is out of scope per the brief).
- Did not invoke the `rule-designer` or `false-positive-analyst` agents via the `Task` tool (behavioral testing was confined to dry-run against the spec content; per `testing-skill-quality.md:92`, "Subagent dispatch is the default — it gives you a clean context, a parseable transcript, and parallelism", but for a content-quality review where the pack's discipline is visible in the text, this would have been over-instrumentation given no failures surfaced from dry-run).
- Did not diff against prior versions (no v0.1 review in `reviews/` to compare).

### Confidence

- **Inventory**: High — all files were read or sampled; counts cross-verified against `find … wc -l`.
- **Domain coverage**: High — coverage map is exhaustive for the declared scope; nothing surfaced as "this is in scope but missing".
- **Behavioral pass rates**: Medium-High for dry-run; the discipline is clearly load-bearing in the text, but a live subagent test on a previously-unseen brownfield analyzer would harden T5 in particular.
- **M-1 severity**: High confidence that the wrapper is stale (text contradicts plugin.json); slight uncertainty on whether to score it Major or Minor, since the router skill itself is correct and a user reaching the router via the Skill tool gets accurate routing. Scored Major because the wrapper is the user's primary entry point for `/static-analysis-engineering` and its claims directly mislead about feature availability.

### Risk

- **Risk of acting on recommendations**: Low. The Major fix is a self-contained rewrite of one short markdown file. The Minor and Polish items are purely additive or cosmetic.
- **Risk of *not* acting**: M-1 will continue to under-sell the pack at slash-command invocation; m-1 will compound as v0.3.x lands without a versioning trail.

### Information gaps

- No CHANGELOG meant the version history had to be reconstructed from the user-memory note (`project_static_analysis_engineering_v01.md`) and from internal "Shipped in v0.2.0" annotations.
- I did not confirm whether the slash-command wrapper was deliberately left at v0.1 framing (some packs intentionally version their wrappers), but cross-pack comparison (`.claude/commands/procedural-architecture.md` etc.) suggests the marketplace convention is tense-neutral wrappers.

### Caveats

- The behavioral tests in §4 are static-content audits, not live subagent transcripts. Where they are scored "Pass", the basis is that the text explicitly forces the correct behaviour with citations; a live subagent run could surface model-side failure modes the text does not predict.
- This review covers Stages 1–4 only (per the brief). Stage 5 (implementation, version bump, commit) is explicitly out of scope.

### Comparison anchor

For calibration, I cross-referenced against the existing reviews in `/home/john/skillpacks/reviews/`:

- `axiom-procedural-architecture.md` (v0.1.1) — similar shape: router + 13 sheets + 3 commands + 2 SME agents; scored a single Major finding (missing slash-command wrapper, since fixed). This pack is the same generation and at a comparable maturity level — the discipline-bearing depth in both is the marketplace's current bar for "Pattern pack with full producer/critic + commands + agents".
- `axiom-determinism-and-replay.md` and `axiom-embedded-database.md` — also v0.x routers with 13 sheets, 3 commands, 2 agents; same architectural shape; this pack fits the family.

Score relative to family: **at parity**. The single Major (stale wrapper) is a documentation-currency issue, not a structural one. Internal coherence (spec dependency graph + consistency gate + tier model) is among the strongest in the family — the algebra is on-the-page rather than implied, and the gate is concrete and tier-aware.

### Calibration note on the Major rating

I considered Minor for M-1 because the router skill itself is correct. I chose Major on the basis that (a) the wrapper is the dominant entry point when a user types `/static-analysis-engineering`, (b) the wrapper's content does not merely omit v0.2 features but actively *denies* their existence ("Deferred to v0.2"), and (c) the marketplace convention exemplified by sibling wrappers in `.claude/commands/` is tense-neutral routing pages — so this is a measurable drift from convention, not just a stylistic preference. If the wrapper had instead been an up-to-date but terse routing page lacking only the recent additions, Minor would have been the right call.

### Final disposition

**Proceed** to Stage 5 with M-1 as the primary action item; m-1 (CHANGELOG) as a follow-on; m-2 and p-3 as opportunistic cleanups. No rebuild; no scope renegotiation; no new components required.
