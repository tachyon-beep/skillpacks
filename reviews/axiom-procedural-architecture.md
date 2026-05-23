# Review: axiom-procedural-architecture

**Version:** 0.1.1 **Reviewed:** 2026-05-22 **Reviewer:** general-purpose subagent

## 1. Inventory

### Plugin metadata

- `plugins/axiom-procedural-architecture/.claude-plugin/plugin.json:3` — `version: 0.1.1`
- `plugins/axiom-procedural-architecture/.claude-plugin/plugin.json:4` — description matches marketplace summary ("Router + 13 sheets, 3 commands, 2 agents. Pattern pack…")
- Marketplace registration present: `.claude-plugin/marketplace.json` (entry beginning at the `axiom-procedural-architecture` block), category `development`, source path `./plugins/axiom-procedural-architecture` — registration is consistent and complete.

### Components

| Type | Count | Notes |
|------|-------|-------|
| Router skill | 1 | `skills/using-procedural-architecture/SKILL.md` (208 lines) |
| Reference sheets | 13 | All siblings of `SKILL.md` in the same directory |
| Commands | 3 | `decompose-procedure.md`, `review-decomposition.md`, `analyze-procedure.md` |
| Agents | 2 | `decomposition-architect.md`, `decomposition-critic.md` |
| Hooks | 0 | None present (none required for this pack's discipline) |
| Slash-command wrapper | **0** | `.claude/commands/procedural-architecture.md` does **not** exist (see Findings) |

### Reference sheets (13)

Producer cluster (4):
- `decomposition-fundamentals.md` (199L)
- `decision-flow-design.md` (191L)
- `granularity-calibration.md` (243L)
- `audience-modeling-for-procedures.md` (237L)

Critic cluster (4):
- `dependency-and-ordering-audit.md` (219L)
- `branching-and-mece-review.md` (284L)
- `decomposition-smells.md` (162L) — names 9 smells, declared "authoritative" by the router
- `procedural-invariants-and-correctness.md` (121L)

Analyst cluster (4):
- `queueing-theory-for-procedures.md` (172L)
- `discrete-event-simulation-for-procedures.md` (121L)
- `process-algebra-and-workflow-nets.md` (206L)
- `flow-vs-state-vs-decision-modeling.md` (230L)

Boundary (1):
- `procedural-boundary-and-handoffs.md` (324L) — also documents inbound relationships

### Commands

| Command | Frontmatter shape | argument-hint | Notes |
|---------|-------------------|---------------|-------|
| `/decompose-procedure` | description + allowed-tools (quoted JSON-array, includes `AskUserQuestion`) | `"[goal_description_or_file]"` | Producer pipeline; 7 steps |
| `/review-decomposition` | description + allowed-tools (quoted JSON-array) | `"[decomposition_text_or_file]"` | Critic pipeline; 7 steps; emits machine-readable YAML summary |
| `/analyze-procedure` | description + allowed-tools (quoted JSON-array) | `"[procedure_text_or_file] [analysis_question]"` | Analyst pipeline routed on question class |

All three commands restrict tools using the marketplace-standard `["Read","Grep","Glob","Bash","Task","Write","Edit",...]` quoted-string form.

### Agents

| Agent | Model | SME-protocol compliant? |
|-------|-------|-------------------------|
| `decomposition-architect` (producer SME) | opus | Yes — `decomposition-architect.md:14` cites `meta-sme-protocol:sme-agent-protocol`, description ends "Follows SME Agent Protocol with confidence/risk assessment per stage and per decision point." (line 2); four mandated sections enumerated in Output Contract (lines 93–99); Anti-Overconfidence Protocol section explicit (lines 117–133). |
| `decomposition-critic` (critic SME) | opus | Yes — `decomposition-critic.md:14` cites the protocol; description ends with "…per finding" (line 2); four sections required (lines 107–109); Anti-Rubber-Stamp Protocol explicit (lines 123–137). |

Neither agent declares `tools:` — consistent with the marketplace convention.

### Documentation

- `CHANGELOG.md` present; v0.1.0 → v0.1.1 dated 2026-05-12; explains six cleanup items including HTML-comment removal, +2 boundary smells, inbound section, symmetric Anti-Overconfidence, expects-a-critic-finding.

---

## 2. Domain & Coverage

### Stated scope (from `SKILL.md`)

- Owns: structural shape of staged procedures — stage decomposition, dependencies/ordering, decision points, exit artifacts, soundness, capacity, audience parameters as explicit inputs.
- Hands off: code-implementation-plan critique (`/axiom-planning`); system shape (`/system-architect`); continuous-time dynamics (`/simulation-foundations`); emergent game flow (`/simulation-tactics`); rendering as prose/UI/site (`/technical-writer`, `/ux-designer`, `/site-designer`); domain-content judgement (relevant domain pack).
- Three roles (producer / critic / analyst) over one shared 13-sheet corpus; three slash commands map 1:1 to roles.

### Coverage map vs. inventory

**Foundational (structural decomposition):**
- Stage definition / exit artifacts — covered (`decomposition-fundamentals.md`)
- Grain / granularity — covered (`granularity-calibration.md`)
- Audience parameterisation — covered (`audience-modeling-for-procedures.md`)
- Decision-point placement — covered (`decision-flow-design.md`)

**Critic (defect catalog):**
- Dependency / ordering — covered (`dependency-and-ordering-audit.md`)
- MECE / branching — covered (`branching-and-mece-review.md`)
- Smell catalog (9 smells, authoritative) — covered (`decomposition-smells.md`)
- Invariants / soundness — covered (`procedural-invariants-and-correctness.md`)

**Analyst (flow / capacity / soundness):**
- Closed-form queueing — covered (`queueing-theory-for-procedures.md`)
- DES — covered (`discrete-event-simulation-for-procedures.md`)
- Workflow nets / formal soundness — covered (`process-algebra-and-workflow-nets.md`)
- Notation selection — covered (`flow-vs-state-vs-decision-modeling.md`)

**Boundary / handoffs:**
- Single dedicated sheet (`procedural-boundary-and-handoffs.md`) names 8 adjacent territories and a boundary-violation smell catalog; CHANGELOG records that v0.1.1 added Smell 6 (site-IA colonisation) and Smell 7 (emergent-flow colonisation) and an "Inbound Relationships" section for `axiom-system-archaeologist`.

**Gaps identified:** none structural at this scope. The three-cluster design plus boundary sheet exhausts the structural-discipline surface area declared in the SKILL. No coverage map item is missing; no two sheets duplicate.

---

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Router quality** | Pass | `SKILL.md` description (line 3) begins "Use when designing or critiquing the structure of a staged procedure …" — matches marketplace convention. Router clearly distinguishes producer/critic/analyst, names three triggers blocks, names six "Do not use" cases, names three "Start here" entry tracks, ships a 15-row routing symptom→sheet table that covers every reference sheet, and ships an explicit Consistency Gate (lines 192–208) with blocking checks. |
| **Skill descriptions** | Pass | All 13 sheets have YAML frontmatter with `name` + `description`. Descriptions are first-class — most encode the sheet's role (e.g. `decomposition-smells.md:3` declares itself "the definitive source other critic and producer sheets defer to for smell names, false-positive calibration, and fix patterns"). Descriptions are dense and discriminative. |
| **Frontmatter conformance** | Pass | Commands use quoted-array `allowed-tools` (e.g. `decompose-procedure.md:3` `allowed-tools: ["Read", "Grep", ...]`) and quoted `argument-hint`. Agents declare only `description` + `model`, matching the ~60/65 marketplace baseline. No spurious `tools:` keys. Router uses two-key frontmatter (`name`, `description`). |
| **Component cohesion** | Pass | Three roles × one corpus design is internally consistent. Each command maps 1:1 to a role; each cluster of sheets maps 1:1 to a role; the boundary sheet is shared. Cross-references within sheets are reciprocal (e.g. `decomposition-smells.md:158–162` cross-refs three sibling sheets that themselves cross-ref it). Producer agent and critic agent enforce **symmetric** anti-failure protocols (Anti-Overconfidence vs. Anti-Rubber-Stamp). |
| **Slash-command exposure** | **Major fail** | No `.claude/commands/procedural-architecture.md` wrapper at repo root. Every sibling pack with a `using-X` router has the wrapper (`audit-pipelines.md`, `determinism-and-replay.md`, `embedded-database.md`, `static-analysis-engineering.md`, etc., 32 wrappers total in `.claude/commands/`); this pack is the only `using-X`-bearing pack in the marketplace without one. CLAUDE.md `/home/john/skillpacks/CLAUDE.md:46–60` is explicit: "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits." The router is therefore discoverable only by description-match, not by explicit `/procedural-architecture` invocation. |
| **SME agent protocol** | Pass | Both agents cite `meta-sme-protocol:sme-agent-protocol`, end their descriptions with the canonical "Follows SME Agent Protocol …" tagline, and require the four-section output contract (Confidence Assessment / Risk Assessment / Information Gaps / Caveats). Both ship a named anti-failure protocol calibrated to their role: Anti-Overconfidence (producer) and Anti-Rubber-Stamp (critic). |
| **Anti-pattern coverage** | Pass | `decomposition-smells.md` enumerates 9 smells (god-step, mystery-step, decision-without-information, audience-amnesia, ladder-of-trivials, premature-commitment, orphan-state, fake-branch, re-entrancy-blindness). Each entry carries Definition / Diagnostic signal / **False-positive caveat** / Remediation — the false-positive caveat in particular is unusually rigorous for a smell catalog. Boundary sheet ships its own 7-entry boundary-violation smell catalog. Consistency Gate (`SKILL.md:200`) explicitly forbids "no smells found" verdicts that didn't enumerate the catalog. |
| **Cross-skill linkage** | Pass | Every sheet has a closing cross-references block; the router's symptom table covers every sheet; the smells sheet is declared "authoritative" by the router and other sheets cite it back. The producer/critic agents close with explicit slash-command and skill cross-references. |

**Overall:** Pass with one Major issue (missing slash-command wrapper). The pack is structurally sound, internally cohesive, well-documented, and discipline-compliant. The single Major issue is mechanical — adding a 30-line wrapper file at `.claude/commands/procedural-architecture.md` closes it.

---

## 4. Behavioral Tests

Tests are constructed from the gauntlet categories in `testing-skill-quality.md`. Each was traced statically through the router + named sheet content to determine the expected guidance and the rationalization the pack is designed to resist.

### Test 1 — Router pressure scenario (Pressure / Time-pressure + simplicity-temptation)

**Scenario.** "I need a 5-step troubleshooting tree for ourprinter-pairing issue. The PM wants this in an hour. Skip the audience stuff and just give me the stages."

**Expected behavior.** Router triggers (it's a producer task — "troubleshooting tree" is named as a producer trigger at `SKILL.md:23`). Producer agent's **Hard Precondition: Audience Parameter Declaration** (`decomposition-architect.md:16–34`) refuses to emit any decomposition without an audience block: "Without a declared audience, grain calibration is undefined, and any decomposition produced would be an unconstrained guess." Consistency Gate item 1 (`SKILL.md:195`) forbids "the user" or "a developer" as a declaration.

**Result: Pass.** The hard precondition is load-bearing and named "Hard precondition" twice. The Consistency Gate restates it as a blocking failure. Rationalization surface is small — there is no language in the producer command or agent that admits "skip audience for now."

### Test 2 — Critic rubber-stamp pressure (Pressure / Sunk-cost + overkill perception)

**Scenario.** A producer agent has just emitted a 12-stage decomposition. User runs `/review-decomposition` and the critic's first pass finds nothing remarkable. User says: "Looks clean — let's ship it."

**Expected behavior.** Critic agent's **Anti-Rubber-Stamp Protocol** (`decomposition-critic.md:123–137`) is triggered: "Zero findings on a non-trivial decomposition is a smell of the critic, not the decomposition." It mandates five explicit before-emit checks (each of nine smells confirmed checked, four ordering checks ran, four MECE checks ran, invariants checklist ran, non-triviality evidence stated). The router reinforces this at `SKILL.md:109`: "If producer and critic always agree, the pipeline is broken … Treat zero-disagreement runs as a defect of the critic, not as a virtue of the producer." Producer agent's Closing Recommendation (`decomposition-architect.md:101`) further reinforces by **expecting** at least one substantive critic finding.

**Result: Pass.** This is a notably robust paired protocol — the producer pre-emptively declares that a finding-free audit is suspicious, and the critic carries the matching machinery to act on that suspicion. CHANGELOG (`CHANGELOG.md:11`) explicitly cites this symmetric design as a v0.1.1 improvement.

### Test 3 — Analyst boundary discipline (Edge case / Principle conflict)

**Scenario.** User to `/analyze-procedure`: "Our review pipeline is saturating because the upstream Kafka consumer can't keep up — what should the pipeline look like?"

**Expected behavior.** The question conflates two layers: a procedural bottleneck (which the pack owns) and an infrastructure/content-domain question ("is Kafka the right tech for the consumer"). The analyst command (`analyze-procedure.md:96–105`) explicitly handles this: "When the analysis question crosses a boundary this pack does not own, name the correct pack and stop — do not attempt to answer the out-of-scope part" and gives the verbatim hand-off string format. The router's "Do Not Use This Pack When" (`SKILL.md:42–52`) names domain-content judgement as out of scope. The boundary sheet (`procedural-boundary-and-handoffs.md:9–22`) contains an opening warning about exactly this colonisation risk.

**Result: Pass.** Boundary discipline is explicit, repeated across router/command/boundary-sheet, and the analyst command ships a hand-off template. The pack is unusually self-aware about its colonisation surface area.

### Test 4 — Smell-catalog false-positive (Edge case / Naive application failure)

**Scenario.** Critic agent identifies "Provision Kubernetes cluster" as a god-step (smell 1) in an SRE runbook and recommends splitting it.

**Expected behavior.** `decomposition-smells.md:49` carries an explicit false-positive caveat: a stage may have substantial internal complexity that is legitimately invisible to the consumer — "Provision Kubernetes cluster" is named as the canonical false-positive — and lays out a three-part test for when the encapsulation is correct (audience opinionated, single tool call/script, single verifiable exit artifact, failure modes collapsed). The catalog's "How to Use" (lines 16–25) requires false-positive evaluation as step 3 of the four-step pass for every smell. Critic agent process step 5 (`decomposition-critic.md:64`) explicitly: "For each smell that fires, verify the false-positive caveat **before** recording the finding."

**Result: Pass.** The false-positive caveat is operationalised, not decorative. Every smell entry I inspected carries one; the critic protocol enforces evaluation.

### Test 5 — Real-world complexity (Producer / messy audience)

**Scenario.** "We need an onboarding flow that works for a junior dev, an LLM agent, and a senior auditor."

**Expected behavior.** The router's producer-trigger list (`SKILL.md:27`) names exactly this case: "The procedure has to work for a junior, an LLM, and a senior auditor — and I cannot decompose it once and pretend that covers all three." The routing table (line 144) routes "wildly different audience this time" to both `audience-modeling-for-procedures.md` and `granularity-calibration.md`. Producer agent step 2 (`decomposition-architect.md:53`) requires checking audience parameters for internal contradiction and explicitly surfaces a contradiction example (low working-memory + high deliberation requirement) where it must stop and ask the user to resolve.

**Result: Pass with mild caveat.** The pack acknowledges the multi-audience case as a top-of-list trigger but does not appear to ship a worked decomposition example showing how the producer agent would split into multiple audience-specialised renderings or insist on a single conservative grain. This is a minor coverage edge (see Findings — Polish).

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

**M1. Missing slash-command wrapper.** `/home/john/skillpacks/.claude/commands/procedural-architecture.md` does not exist. Every other `using-X` router in the marketplace (32 wrappers in `.claude/commands/`) has one. CLAUDE.md `/home/john/skillpacks/CLAUDE.md:46–60` mandates the pattern explicitly. The pack contains a router skill at `plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md` whose self-described convention is invocation as `/procedural-architecture`, which currently does not resolve. Discoverability is degraded; users following marketplace conventions will type `/procedural-architecture` and receive an "unknown command" response. The pack's CHANGELOG and version (v0.1.1) treat the pack as ready for adoption, but the slash-command surface is incomplete.

  Remediation: add `.claude/commands/procedural-architecture.md` following the pattern in any sibling wrapper (e.g. `.claude/commands/determinism-and-replay.md` or `.claude/commands/embedded-database.md`).

### Minor

**Mi1. Marketplace description lightly diverges from `plugin.json`.** `plugin.json:4` description ends "Pattern pack — applies wherever you build a wizard, troubleshooting tree, …"; the marketplace catalog entry omits the "Pattern pack" framing sentence. Both are otherwise consistent. Polish-grade; either intentional or trivially fixable by aligning the two descriptions.

**Mi2. Marketplace keyword list shorter than `plugin.json`.** `plugin.json` declares 12 keywords (axiom, procedural-architecture, decomposition, workflow, process-design, wizard-design, queueing, discrete-event-simulation, workflow-nets, task-analysis, instructional-design, decision-flow); marketplace entry declares only 6 (axiom, procedural-architecture, decomposition, workflow, process-design, queueing). Probably intentional (marketplace catalog uses a tighter set across packs), but worth noting that `wizard-design`, `discrete-event-simulation`, and `workflow-nets` are dropped from the marketplace-facing keyword surface even though they are first-class triggers in the router.

### Polish

**P1. No worked multi-audience example.** Test 5 above identified the multi-audience case as a top-of-list trigger, but the producer cluster does not appear to ship a worked example showing how a producer agent should approach a goal where audience parameters are mutually contradictory (e.g. junior + LLM + senior auditor on one procedure). The agents have the machinery to detect contradiction; a worked example in `audience-modeling-for-procedures.md` or `granularity-calibration.md` would close the polish gap. Not blocking; the router and agent text correctly direct the user.

**P2. v0.1.x version label.** v0.1.1 is described in CHANGELOG and memory notes (`project_axiom_procedural_architecture_v01.md`) as "feature-complete" with "+2 boundary smells, inbound section, symmetric producer Anti-Overconfidence." The substantive content is at the quality of a v0.2 or v0.3 promotion. Stay-at-v0.1.x is the pack author's call (CHANGELOG line 16: "awaiting external feedback before promotion to v0.2"); this review notes only that on content alone, the pack reads as past the v0.1 scaffold milestone.

**P3. No `analyze-procedure` worked example in repository.** The analyst command names a 3-stage approval flow as its worked example, and `queueing-theory-for-procedures.md:90–101` ships a worked 3-stage M/M/c flow. No similarly worked example exists for the workflow-nets soundness route or the DES route. The reference sheets carry conceptual depth but a worked-end-to-end run of `/analyze-procedure` against a non-queueing question would strengthen the analyst cluster.

---

## 6. Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| 1 (Major) | Create `.claude/commands/procedural-architecture.md` following sibling-pack pattern. Match the marketplace convention; do not invent new content. | Small (~30 lines) |
| 2 (Minor) | Reconcile marketplace catalog description with `plugin.json` description, or document the intentional divergence in a comment. | Trivial |
| 3 (Minor) | Decide whether marketplace catalog should carry the full keyword list from `plugin.json` (notably `wizard-design`, `discrete-event-simulation`, `workflow-nets`). | Trivial |
| 4 (Polish) | Add a worked example in `audience-modeling-for-procedures.md` (or as a new sheet) for the canonical multi-audience case ("junior + LLM + senior auditor"). The trigger is already top-of-router; close the loop with a worked decomposition. | Medium |
| 5 (Polish) | Consider promotion from v0.1.1 → v0.2.0 if external-feedback gate is satisfied. CHANGELOG already signals readiness ("awaiting external feedback before promotion"). | Trivial (version bump + marketplace catalog touch) |
| 6 (Polish) | Add a worked end-to-end `/analyze-procedure` example for the workflow-nets or DES route (the queueing route already has one). | Medium |

No action on agents (both protocol-compliant), the router (already at marketplace standard for `using-X` skills), or the smell catalog (notably rigorous on false-positive caveats).

No skills require behavioral rework. The pack's discipline-enforcing skills (audience preconditions, anti-rubber-stamp, smell false-positive caveats, boundary discipline) all hold under their respective pressure tests on static inspection.

---

## 7. Reviewer Notes

- This is a high-quality pack at v0.1.1. Two design choices stand out as deliberately strong: the **symmetric anti-failure protocols** in producer/critic agents (Anti-Overconfidence ↔ Anti-Rubber-Stamp) and the **operationalised false-positive caveat** in every smell entry. Both raise the floor of what the agents can be talked into.
- The boundary sheet's "Opening Warning" (`procedural-boundary-and-handoffs.md:9–22`) about colonisation risk is unusual for a marketplace pack and reads as authored by a maintainer who had been bitten by the failure mode it warns against. The fact that v0.1.1 adds two more colonisation smells (site-IA, emergent-flow) to that sheet suggests the boundary discipline is being maintained against ongoing pressure.
- The pack's cross-pack relationships are unusually well-declared: `axiom-planning` is explicitly named as the code-implementation-plan specialisation of this discipline; `axiom-system-architect` is named as the sibling that owns *system shape* rather than *procedure shape*; `simulation-foundations` is named as the continuous-time handoff with a testable boundary condition ("if you can draw the procedure as a directed graph of stages, you are in this pack; if you need a vector field, you are not"). These are the kinds of declarations that survive contact with future maintainers.
- The single Major finding (missing slash-command wrapper) is mechanical and trivially fixable. It is the only thing standing between this pack's structural quality and its marketplace-conformant discoverability.
- The pack is read-only against this review; no edits were made.
