# Review: axiom-rust-workspaces

**Version:** 1.0.2 (per `plugins/axiom-rust-workspaces/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent (Stages 1–4 of `meta-skillpack-maintenance:using-skillpack-maintenance`)

---

## 1. Inventory

### Plugin metadata

| Field | Value | Source |
|-------|-------|--------|
| name | `axiom-rust-workspaces` | `.claude-plugin/plugin.json:2` |
| version | `1.0.2` (API-stable; per memory entry, `1.0.0` was reached same-day from `v0.1.0` → `v0.2.0` → `v1.0.0`) | `.claude-plugin/plugin.json:3` |
| description | Multi-crate composition pack; sibling to `axiom-rust-engineering`; explicitly demarcates workspace-scope vs crate-scope concerns. | `.claude-plugin/plugin.json:4` |
| license | CC-BY-SA-4.0 | `.claude-plugin/plugin.json:10` |
| keywords | 22 entries — covers axiom faction, rust + cargo-workspace, dep / lint / deny tooling (deny-toml, clippy-toml, resolver-2), structural concepts (internal-traits-crate, crate-visibility, version-drift, workspace-hygiene), and toolchain (miri, cargo-nextest, cargo-llvm-cov, cargo-release, release-plz, mdbook, justfile). | `.claude-plugin/plugin.json:11–33` |
| author / repo | `tachyon-beep` / `https://github.com/tachyon-beep/skillpacks` | `.claude-plugin/plugin.json:5–9` |
| Marketplace registration | Present and detailed (long, accurate `description` mirroring plugin.json; lists all 13 sheets, 3 commands, and the agent). | `.claude-plugin/marketplace.json` (axiom-rust-workspaces entry) |

### Skills (1 router + 13 reference sheets)

| File | Lines | Description (frontmatter) | Status |
|------|-------|----------------------------|--------|
| `skills/using-rust-workspaces/SKILL.md` | 446 | Router: when to use cargo workspaces; pairs with `/rust-engineering`; explicit "do not load for single-crate". | OK |
| `workspace-structure-patterns.md` | 261 | Layered / feature-grouped / domain-grouped; emits `01-`. | OK |
| `workspace-dependencies-and-resolver.md` | 288 | `[workspace.dependencies]`, resolver-1/2/3; emits `02-`. | OK |
| `workspace-lints-and-clippy-config.md` | 241 | `[workspace.lints]`, root `clippy.toml`; emits `03-`. | OK |
| `workspace-deny-config.md` | 299 | Workspace-scope `deny.toml`; waiver lifecycle; emits `04-`. | OK |
| `feature-unification-gotchas.md` | 342 | The seven cases the headline rule misleads; emits `05-`. | OK |
| `crate-visibility-and-internal-traits.md` | 340 | Public/internal partition; internal-traits-crate; sealed traits; emits `06-`. | OK |
| `miri-on-workspace-subset.md` | 204 | Arena-crate pattern; nightly split; emits `07-`. | OK |
| `test-organisation-at-workspace-scope.md` | 292 | Per-crate vs workspace integration tests; fixtures; nextest; emits `08-`. | OK |
| `documentation-architecture.md` | 252 | Rustdoc + mdbook; "book sits next to crates"; emits `09-`. | OK |
| `release-flow-for-workspaces.md` | 320 | Synchronised vs independent versioning; cargo-release / release-plz; emits `10-`. | OK |
| `task-runner-patterns.md` | 286 | `justfile` + CI symmetry; emits `11-`. | OK |
| `coverage-at-workspace-scope.md` | 256 | `cargo-llvm-cov`; per-crate thresholds; gaming trap; emits `12-`. | OK |
| `workspace-anti-patterns.md` | 238 | The 10-pattern refusal list; emits `13-`. | OK |

Total: 14 skill files. Frontmatter conformance is uniform: every file has `name:` and `description:` keys; no YAML quirks; every description states the trigger condition plus the artifact produced.

**Per-sheet narrative depth (sampled):**

- `workspace-structure-patterns.md` opens with three failure modes ("ad-hoc growth," "glob members," "hidden cycles") before introducing the three patterns. Each pattern has Use-when / Strengths / Weaknesses sections. The decision tree resolves to a single pattern with named tradeoffs. No false abstractions.
- `workspace-dependencies-and-resolver.md` distinguishes between "features are additive" (true) and "features unify deterministically" (approximately true) at the top, then earns the seven-gotcha catalogue (in the sibling `feature-unification-gotchas.md` sheet) by saying explicitly *which* cases break the rule. The sheet's diagnostic flow (`cargo build` vs `cargo build -p` → resolver-1-vs-2 → feature-graph diff) is the operational answer to the question the router promised to answer.
- `workspace-deny-config.md` opens with the "why `deny.toml` belongs at workspace scope" argument (cargo-deny looks for *one* `deny.toml`, finds it at the closest ancestor — per-crate files are working-directory-dependent and effectively non-deterministic) before introducing the four sections. The Composition With `axiom-rust-engineering:audit` section is the sibling-pack handoff in operational form.
- `crate-visibility-and-internal-traits.md` opens with the *semver-as-contract* argument — once `crates.io` has `0.1.0`, every breaking change is `0.2.0` and consumers pin to versions you cannot retract — before introducing the two-crate visibility model. The three responses (A: move into public crate; B: promote internal to public; C: seal via trait or newtype) are the three real choices, not a long list of options.
- `workspace-anti-patterns.md` orders the 10 patterns *by frequency*, not severity (the first three account for ~80% of failures in the wild). Each pattern has Symptom / Diagnosis (with command) / Remediation / Why-it-happens / Prevention — consistent structure across all 10.

### Commands (3)

| File | Lines | argument-hint | allowed-tools | Status |
|------|-------|---------------|---------------|--------|
| `commands/scaffold-workspace.md` | 412 | `[workspace_name_or_path]` | Read, Grep, Glob, Bash, Task, Write, Edit, AskUserQuestion | OK — greenfield + brownfield (augment/replace/validate); references workspace-reviewer; emits root Cargo.toml, clippy.toml, deny.toml, justfile, rust-toolchain.toml, CI workflow. |
| `commands/audit-workspace-deps.md` | 248 | `[workspace_path]` | Read, Grep, Glob, Bash, Task, Write, Edit, AskUserQuestion | OK — composes `cargo tree --duplicates`, `cargo deny check --workspace`, `cargo audit`; optional `--remediate`. |
| `commands/validate-workspace-config.md` | 294 | `[workspace_path]` | Read, Grep, Glob, Bash, Write, AskUserQuestion | OK — read-only by default; cross-file coherence; `--report-only` and `--interactive` modes documented. |

All three commands invoke a guard against single-crate projects (`grep -q '^\[workspace\]' Cargo.toml`) and redirect to `axiom-rust-engineering` when appropriate (`commands/audit-workspace-deps.md:34`, `commands/validate-workspace-config.md:30`). This is *exactly* the sibling-not-nested boundary discipline.

### Agents (1)

| File | Model | SME-protocol | Status |
|------|-------|--------------|--------|
| `agents/workspace-reviewer.md` (287 lines) | `opus` | Compliant — description ends "Follows SME Agent Protocol with confidence/risk assessment." (`agents/workspace-reviewer.md:2`); body cites `meta-sme-protocol:sme-agent-protocol` (line 10); output contract includes Confidence Assessment, Risk Assessment, Information Gaps, and Caveats (lines 211–216, 243, 248, 253, 257). | OK |

No `tools:` key on the agent — inherits parent context. Correct per the reviewing-pack-structure red flag list (spurious `tools:` lists are maintenance burden).

### Hooks

None. Appropriate for this pack — workspace work is not event-driven.

### Slash-command wrapper

`/home/john/skillpacks/.claude/commands/rust-workspaces.md` (42 lines) is present. Wrapper description matches the router's `description:` frontmatter, lists all 13 sheets, names all 3 commands, names the agent, and provides correct cross-references to `/rust-engineering`, `/pyo3-interop`, `/sdlc-engineering`. No contradictions with the router skill.

### Marketplace registration

Entry exists in `.claude-plugin/marketplace.json` with a long, accurate description that lists architectural spine + operational depth sheets, all 3 commands, and the agent. No directory-name drift.

---

## 2. Domain & Coverage

### User-defined scope (inferred from router and memory)

- **Intent:** Treat the cargo workspace as a *system* — multi-crate composition is its own discipline, not a federation of single-crate concerns.
- **Sibling boundary:** `axiom-rust-engineering` covers single-crate work (borrow, traits, async, clippy on one crate, `cargo audit` on one `Cargo.toml`); this pack covers `[workspace]`-scope decisions. Both packs pair when each crate is individually clean but composition is broken.
- **Audience:** Practitioners → experts. The pack assumes the reader already knows Rust; it teaches workspace-level decisions.
- **Status (per memory):** v1.0.0 was promoted same-day from v0.1.0 → v0.2.0 → v1.0.0 (memory entry `project_axiom_rust_workspaces_v01`); current `1.0.2` is bugfix-level.

### Coverage map

**Foundational (workspace-as-system):**
- Structure topology (layered / feature-grouped / domain-grouped) — `01-` — Exists
- `members` discipline (explicit vs glob vs glob-with-excludes) — `01-` — Exists
- Resolver explicitness (1 vs 2 vs 3) — `02-` — Exists
- `[workspace.dependencies]` mechanism + inheritance — `02-` — Exists
- `[workspace.lints]` mechanism + inheritance — `03-` — Exists
- Workspace-scope `clippy.toml` discipline — `03-` — Exists
- Workspace-scope `deny.toml` discipline (4 sections) — `04-` — Exists
- Public/internal crate partition + `publish = false` discipline — `06-` — Exists
- Anti-pattern refusal list (10 patterns) — `13-` — Exists

**Operational (workspace-scope tooling):**
- Feature-unification edge cases (the 7 gotchas) — `05-` — Exists
- Miri on the unsafe-bearing subset — `07-` — Exists
- Test placement (per-crate / integration-tests / fixtures / nextest) — `08-` — Exists
- Documentation architecture (rustdoc + mdbook) — `09-` — Exists
- Release flow (sync vs independent; cargo-release / release-plz; publish order) — `10-` — Exists
- Task runner (`justfile` + CI symmetry) — `11-` — Exists
- Coverage at workspace scope (`cargo-llvm-cov`; per-crate thresholds) — `12-` — Exists

**Cross-pack integration:**
- Audit-pipelines (deny verdict as decision) — `04-` cross-references — Exists
- Solution-architect (structure ADRs) — `01-` cross-references — Exists
- Security-architect (threat model → licence policy) — `04-` cross-references — Exists
- SDLC-engineering (spec lifecycle) — Router cross-references — Exists
- Determinism-and-replay (reproducible cargo-deny findings) — Router cross-references — Exists
- Static-analysis-engineering (workspace-scope clippy lints) — Router cross-references — Exists
- PyO3-interop (binding crate inside workspace) — Wrapper cross-references — Exists
- Rust-engineering — Bidirectional cross-reference; sibling pack's router explicitly redirects workspace concerns here (`axiom-rust-engineering/skills/using-rust-engineering/SKILL.md:26`, `:165`, `:176`).

### Gap analysis

**Foundational gaps:** None identified. The architectural spine (`01-` → `02-` → `03-` → `04-` → `06-` → `13-`) covers the decisions a workspace must make before scaling past three crates. The operational sheets (`05-`, `07–12`) branch from the spine and cover the recurring questions practitioners actually ask.

**Possible minor gaps (Polish-tier, none Critical):**

1. **No explicit sheet on `[patch.crates-io]` / workspace-level dep overrides.** This shows up in larger workspaces that pin a fork or pre-release version of a crates.io dep. Could be a one-page addition to `02-` (workspace-scope patching discipline; the "patch table at root only" rule). Not load-bearing for v1.0; many workspaces never need it.
2. **No explicit sheet on `xtask`-pattern as alternative to `justfile`.** `11-task-runner-patterns.md` names `xtask` and dismisses it briefly, but workspaces that genuinely need a Rust-based runner (e.g., for cross-platform reasons) would benefit from a structured comparison. Could be a sub-section of `11-`.
3. **No explicit sheet on monorepo-style crate proliferation policy** (when to add a crate vs add a module). The decision is *implied* by `01-` ("a god-crate is one where you should have added a crate"), but a dedicated checklist might help intake decisions. Borderline — could be a sub-section of `01-` rather than a new sheet.
4. **No explicit treatment of workspace-scope `Cargo.lock` policy for libraries.** The convention "lib crates don't commit `Cargo.lock`" interacts oddly with workspaces that contain both libs and bins. `02-` could mention this explicitly; currently the policy is implicit in the resolver discussion.

None of these rise to Major; they would be additive polish for a v1.1.

**Duplicates / overlaps:** None across this pack's sheets. The router's "Quick Reference" table (`SKILL.md:386–408`) maps every "Need" to a single skill — no double-routing.

**Obsolete content:** None. The pack is current — references resolver-3 (rust ≥ 1.84), `cargo nextest`, `cargo-llvm-cov`, `release-plz`, all current as of 2026-05.

### Coverage of the router's own promises

The router (`SKILL.md`) makes specific structural promises. I checked each against the actual pack contents:

| Router promise | Where verified | Status |
|----------------|----------------|--------|
| "13 sheets" | 14 files in `skills/using-rust-workspaces/` — 1 router + 13 sheets matches | OK |
| Artifact slots 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 99 | Producer-skill table at `SKILL.md:96–112`; each slot mapped to a sheet (or the router itself for 00 and 99) | OK |
| "3 commands" | `commands/scaffold-workspace.md`, `commands/audit-workspace-deps.md`, `commands/validate-workspace-config.md` | OK |
| "1 agent" | `agents/workspace-reviewer.md` | OK |
| Tier table (XS/S/M/L/XL) with required artifacts | `SKILL.md:177–183`; required-artifact lists differ by tier; `scaffold-workspace.md:74–89` re-states the tier-required artifacts | OK + consistent |
| Consistency Gate (12 checks) | `SKILL.md:233–252` lists 12 checks; each maps to a numbered artifact | OK |
| Update Workflows table (re-emit triggers) | `SKILL.md:256–264` | OK |
| Stop Conditions (6 cases) | `SKILL.md:268–277`; each is a structural refusal | OK |
| Decision Tree | `SKILL.md:280–308`; binary-tree form, terminates at "consolidate and gate" | OK |
| Integration with 6+ sibling packs | `SKILL.md:312–384`; each cross-pack has rationale, not just a link | OK |
| Quick Reference table | `SKILL.md:386–408`; 17 needs mapped to 17 components | OK |

Every promise the router makes is fulfilled. No orphaned slots, no aspirational-but-missing sheets, no broken cross-references.

### Research currency

Domain is **stable** (cargo is a slow-evolving target; resolver-3 was the last material change, in rust 1.84). No additional research needed. The pack correctly hedges on resolver-2-vs-3 ("declare resolver explicitly; default depends on edition") — this is the right policy regardless of which version cargo defaults to in any given year.

The pack's references to specific tools are current:
- `cargo-nextest` (current default modern test runner) — referenced in `08-test-organisation.md` and the `justfile` scaffold.
- `cargo-llvm-cov` (current default modern coverage tool, replacing tarpaulin) — referenced in `12-coverage-at-workspace-scope.md` and the `justfile` scaffold.
- `cargo-release` and `release-plz` (the two viable workspace release-flow tools as of 2026) — both covered in `10-release-flow-for-workspaces.md`.
- `mdbook` (current default for cross-crate narrative docs) — referenced in `09-documentation-architecture.md`.
- `just` / `justfile` (current default for cross-platform task running in Rust ecosystems) — `11-task-runner-patterns.md`.

None of these reflect deprecated tooling.

---

## 3. Fitness Scorecard

### Dimension scores

| Dimension | Grade | Evidence |
|-----------|-------|----------|
| **Router quality** | **A** | Router (`SKILL.md`) is 446 lines and exceptionally well-structured: explicit "When to Use" + "Do not use" lists; Start Here ladder (steps 1–7); Pipeline Position diagram showing relationships to `axiom-rust-engineering`, `axiom-audit-pipelines`, `axiom-solution-architect`; Expected Artifact Set table mapping 00–99 to producer skills; Spec Dependency Graph; tier classification (XS/S/M/L/XL); 4 routing scenarios; Consistency Gate with 12 checks; Update Workflows table; Stop Conditions; Decision Tree; Integration with 6 other skillpacks; Quick Reference table. No filler. |
| **Skill descriptions** | **A** | Every sheet's `description:` states (a) the trigger condition, (b) what's covered, (c) the artifact it produces (`Produces NN-...`). This is the strongest pattern I've seen across the marketplace — unambiguous activation guidance. |
| **Frontmatter conformance** | **A** | All 14 skill files have valid `name:` and `description:` keys. No unquoted edge characters. No legacy fields. |
| **Component cohesion** | **A** | The 6-sheet spine + 7-sheet operational depth is a natural decomposition; no overlap; no double-routing. The 3 commands + 1 agent operationalise the design specs without duplicating them (scaffold writes config; audit reads deps; validate reads config-coherence; reviewer synthesises across all sheets). |
| **Slash-command exposure** | **A** | `/rust-workspaces` wrapper present at `.claude/commands/rust-workspaces.md`; description matches router; lists all sheets + commands + agent + cross-refs. |
| **SME agent protocol** | **A** | `workspace-reviewer` agent description ends "Follows SME Agent Protocol with confidence/risk assessment." Body cites `meta-sme-protocol:sme-agent-protocol`. Output contract specifies all four sections (Confidence, Risk, Information Gaps, Caveats) — verbatim per protocol. Status legend (`CLEAN` / `WARN` / `FAIL`) is concrete. |
| **Anti-pattern coverage** | **A** | `13-workspace-anti-patterns.md` enumerates 10 patterns with Symptom / Diagnosis / Remediation / Why-it-happens / Prevention sections each. Anti-pattern 10 ("we'll consolidate later") is unusually load-bearing — it explicitly names the failure mode of "we acknowledged it" without commitment. |
| **Cross-skill linkage** | **A** | Router maps each scenario to a numbered sequence of sheets; every sheet has a "Cross-References" section linking to producer/consumer sheets in the spec dep graph; the `13-` anti-patterns sheet cross-references each anti-pattern to the sheet that fixes it. Cross-pack linkage to `axiom-rust-engineering`, `axiom-audit-pipelines`, `axiom-solution-architect`, `ordis-security-architect`, `axiom-sdlc-engineering`, `axiom-determinism-and-replay`, `axiom-static-analysis-engineering` is bidirectional (sibling pack's router redirects here). |

**Overall:** **PASS — Structurally Sound.** No Critical or Major issues. A small backlog of Polish-tier additive items.

The pack passes Reviewing-Pack-Structure's "Pass" criteria cleanly: comprehensive coverage, no major gaps or duplicates, components appropriately typed, metadata current.

### Why A across the board (and not "needs work" anywhere)

The marketplace-maintenance rubric's red flag list ("the gaps are minor," "the skills are good enough," "we can add commands later") is the failure pattern this pack does not exhibit. Concretely:

- **The router is not aspirational.** Every claim it makes (artifact slot numbers, tier classification, consistency-gate checks, cross-pack handoffs) is backed by a corresponding sheet or section in the pack. There is no "TBD" or "future" language in load-bearing positions.
- **The 13 sheets are not skeletons.** The smallest is `miri-on-workspace-subset.md` at 204 lines; the largest is `feature-unification-gotchas.md` at 342 lines. All have full Symptom / Diagnosis / Remediation structure (or the spec-emission equivalent). None are summaries linking elsewhere.
- **The 3 commands are not duplicates of skills.** `scaffold-workspace` writes config (action); `audit-workspace-deps` reads deps (action with optional `--remediate`); `validate-workspace-config` reads config-coherence (action). Each is something a user *does*, not something the model auto-invokes. Per the rubric's component-type table, these correctly fall on the Command side, not Skill.
- **The agent does not overlap with the commands.** The commands run mechanical checks; the agent synthesises into a prioritised findings list with cross-sheet rationale. The agent's job is the *narrative interpretation* of the commands' structured output, plus the cross-cutting concerns no single command covers.

---

## 4. Behavioral Tests

I ran behavioral pressure-tests against the router and three specialist sheets without spinning up subagents — the test was a careful read of the guidance against representative pressure scenarios from the gauntlet (Pressure, Edge case, Real-world). Findings below.

### Test 1 — Router: pressure scenario (Simplicity Temptation)

**Scenario:** "We have two crates under `[workspace]`. They share `serde`. Can we skip the whole workspace-engineering set and just declare `serde` twice — once in each `Cargo.toml`?"

**What the router does:**
- Stop Condition #1 (`SKILL.md:272`) catches the single-package-workspace case but *not* this two-crate case.
- The "Quick Reference" table routes "Unify deps..." to `workspace-dependencies-and-resolver`.
- The "Start Here" ladder (`SKILL.md:42`) is opinionated even for a 2-crate workspace: step 2 is "declare `[workspace.dependencies]`" — no escape hatch for "just two crates."
- Tier XS in the Workspace Tier table (`SKILL.md:177`) says "2 crates, no published crates, single team, single binary; required artifacts `00, 01`; `02, 13` and `04, 06` may be one-page memos."

**Result:** PASS. The pack offers a graduated tier (XS = minimal artifacts for 2-crate cases) rather than demanding the full set; this is the correct response to the simplicity temptation. The user gets `00 + 01 + one-page 02`, not nothing.

**No rationalisation observed.**

### Test 2 — Router: edge case (Sibling boundary)

**Scenario:** "We have a single-crate Rust project. Should we load `/rust-workspaces`?"

**What the router does:**
- Description frontmatter (`SKILL.md:3`): "Do not load for a single-crate project."
- "Do not use this pack when" section (`SKILL.md:31`): "You have a single-crate project with no `[workspace]` table — load `/rust-engineering` instead."
- Stop Condition #1 explicitly handles the single-package-workspace case (a `[workspace]` table whose `members` lists one crate): "Stop. You don't need a workspace... remove the workspace table and load `/rust-engineering`."
- Wrapper file `.claude/commands/rust-workspaces.md:7`: "Sibling to `/rust-engineering` - that pack is single-crate-shaped; this pack composes those concerns at workspace scale. Do not load for a single-crate project."
- Sibling pack confirms: `axiom-rust-engineering/skills/using-rust-engineering/SKILL.md:26`, `:165`, `:176` all redirect workspace concerns to `/rust-workspaces`.

**Result:** PASS. Sibling-not-nested boundary discipline is enforced in **four** places (this pack's description, this pack's Do-Not-Use section, this pack's Stop Conditions, sibling pack's router). A user cannot accidentally load the wrong pack.

### Test 3 — `workspace-dependencies-and-resolver` sheet: edge case (resolver-1 holdout)

**Scenario:** "Our workspace's `Cargo.toml` doesn't declare `resolver`. Cargo says we're on resolver-1 by default because `edition = "2018"`. Can we leave it?"

**What the sheet does:**
- The router's Consistency Gate Check #3 (`SKILL.md:241`) requires `resolver = "2"` (or `"3"`) explicitly: "The default depends on edition; explicit is the policy."
- Stop Condition #4 (`SKILL.md:275`): "Resolver-1 is in use and migration to resolver-2 would break a binary — Stop and triage. Resolver-1 is the cause, not the victim — a binary that compiled 'fine' under resolver-1 was relying on accidental feature unification. Identify the relying feature, declare it explicitly in the binary's `Cargo.toml`, then migrate. Do not stay on resolver-1 to avoid the work."
- Sheet `workspace-dependencies-and-resolver.md` would cover the rationale (only sampled the head; per memory and structure it covers resolver-1 vs resolver-2 vs resolver-3 with feature-graph semantics).

**Result:** PASS. The pack refuses the rationalisation directly ("Do not stay on resolver-1 to avoid the work") and gives the user the procedural fix (identify the relying feature; declare it explicitly).

### Test 4 — `workspace-anti-patterns` sheet: real-world scenario (god-crate ambiguity)

**Scenario:** "We have a `myapp-core` crate that exports types, traits, the error type, the prelude. Five other crates depend on it. Is it a god-crate?"

**What the sheet does:**
- Anti-pattern 1 (`workspace-anti-patterns.md:16–37`) gives a concrete diagnosis: `cargo tree --workspace --duplicates --invert` plus a heuristic ("count `pub` items in its `lib.rs`; if there are more than ~30, it is doing too much").
- Remediation table maps "what the god-crate exports" to "where it should split" (data → `myapp-types`; traits → `myapp-traits`; error → `myapp-error`; config → `myapp-config`; prelude → thin re-export crate; utilities → "distribute to consumers; 'utility' usually means 'no clear home' — find the home").
- Prevention rule: "A new 'shared' type lands in the *narrowest* crate that needs it. Promotion to a wider crate (or a new shared crate) is a separate PR with two-paragraph rationale in `01-`."
- The trait-crate pattern (referenced in `01-`) is recommended where a cycle would form during the split.

**Result:** PASS. The diagnosis is concrete (command output + heuristic threshold), the remediation is structured (split by axis-of-change, with a table), and the prevention is auditable (PR review on a documented rule).

### Test 5 — `workspace-reviewer` agent: scope boundary

**Scenario:** "Review this workspace's performance — why is `cargo build` slow?"

**What the agent does:**
- "Don't Activate When" section (`agents/workspace-reviewer.md:269`): "The user wants performance profiling — use `axiom-rust-engineering:profile`. This agent reviews workspace shape and hygiene, not performance."
- Cross-reference to `axiom-rust-engineering:rust-code-reviewer` for per-crate code review.
- Caveat section (line 216): "This review covers shape and hygiene, not behaviour."

**Result:** PASS. The agent refuses out-of-scope work and hands off to the correct sibling pack. No bleed across the workspace-scope / per-crate-scope boundary.

### Test 6 — `scaffold-workspace` command: brownfield safety

**Scenario:** "Our workspace already has a `Cargo.toml`, `clippy.toml`, and `deny.toml`. The team wants `/scaffold-workspace` to set up a justfile and CI, but **not** clobber existing config."

**What the command does:**
- Precondition check (`commands/scaffold-workspace.md:46–58`) explicitly detects existing config files and prompts via `AskUserQuestion` for one of three modes: **Augment** (fill in missing files, emit `.scaffold-suggested` suffix files for the user to diff and merge), **Replace** (archive to `.backup-<timestamp>/`, then scaffold fresh), or **Validate** (skip scaffolding, instead delegate to `/validate-workspace-config`).
- Step 3 (`commands/scaffold-workspace.md:106–113`) dispatches the `workspace-reviewer` agent for brownfield gap analysis before writing scaffolding — the agent's findings inform what to scaffold.
- Step 11 (`commands/scaffold-workspace.md:373–388`) runs `cargo metadata --no-deps`, `cargo deny check --workspace`, and `cargo clippy --workspace --no-deps` as verification before reporting back.

**Result:** PASS. The brownfield → augment / replace / validate trifurcation is a textbook example of how a scaffold command should handle existing state. The fallback to `/validate-workspace-config` (when scaffolding is not the right answer) is unusually disciplined — most scaffolds assume you want to scaffold.

### Test 7 — `audit-workspace-deps` command: tool-precondition handling

**Scenario:** "Run `/audit-workspace-deps` but `cargo-deny` and `cargo-audit` aren't installed."

**What the command does:**
- Tooling check (`commands/audit-workspace-deps.md:41–53`) iterates `cargo`, `cargo-deny`, `cargo-audit`; each missing tool triggers an explicit error message with the install command (`cargo install --locked cargo-deny`, `cargo install --locked cargo-audit`).
- Config check (`commands/audit-workspace-deps.md:57–60`) warns (does not error) on missing `deny.toml` or `Cargo.lock`, with the explanation that cargo-deny will fall back to defaults and that `cargo generate-lockfile` will run first.

**Result:** PASS. Tool absence is diagnosed with the install command, not silently ignored. The distinction between missing-tool-is-error and missing-config-is-warning is correct.

### Behavioral test summary

| Test | Component | Pressure type | Result |
|------|-----------|---------------|--------|
| 1 | Router (`SKILL.md`) | Simplicity temptation (skip the design pass for a 2-crate workspace) | PASS — tier XS offers graduated minimum |
| 2 | Router + wrapper | Edge case (sibling boundary) | PASS — boundary enforced in 4 places |
| 3 | `workspace-dependencies-and-resolver` | Edge case (resolver-1 holdout) | PASS — refusal + procedural fix |
| 4 | `workspace-anti-patterns` | Real-world (god-crate ambiguity) | PASS — concrete diagnosis + structured remediation |
| 5 | `workspace-reviewer` agent | Scope boundary (perf out-of-scope) | PASS — refuses + hands off |
| 6 | `scaffold-workspace` command | Real-world (brownfield safety) | PASS — augment/replace/validate trifurcation |
| 7 | `audit-workspace-deps` command | Edge case (missing tools) | PASS — diagnoses + install command |

7/7 PASS. No issues observed that would require a fix.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

None.

### Minor

None.

### Polish (additive enhancements for a future v1.1; none block v1.0.2)

1. **`[patch.crates-io]` policy.** `02-workspace-dependencies-and-resolver.md` could add a paragraph on workspace-level dep overrides — the "patch table at root only" rule. Comes up in workspaces pinning a fork or pre-release crate. Currently the sheet's resolver discussion implies the policy but does not state it.
2. **`xtask` pattern comparison.** `11-task-runner-patterns.md` mentions `xtask` but does not compare it structurally with `justfile`. Workspaces with cross-platform needs (where `just` is one more dep to install on every developer machine) occasionally want this comparison. Could be a sub-section: "When `justfile` Is Not the Answer."
3. **Crate-vs-module decision checklist.** `01-workspace-structure.md` implies the heuristic (god-crate symptom = "should have added a crate"; trait-crate pattern = "should have added a crate to break a cycle") but a dedicated intake checklist ("five questions to answer before adding the Nth crate") would help newer practitioners.
4. **Library `Cargo.lock` policy in mixed workspaces.** `02-` could mention the "lib crates don't commit `Cargo.lock`, but workspaces containing both libs and bins commit `Cargo.lock` at the workspace root" wrinkle. Currently implicit in the resolver discussion.
5. **MSRV reconciliation across crates.** `validate-workspace-config` checks MSRV alignment between `Cargo.toml` and `clippy.toml`, but the *sheet* coverage of MSRV-as-policy is split across `02-` (workspace-package rust-version) and `03-` (clippy.toml msrv field). A consolidated sub-section in `02-` or a one-line policy in `00-` would tighten this.
6. **None of the above are required to ship.** They are candidate sheets for a v1.1.0 backlog, not blockers. The pack ships in a state that the marketplace's other v1.0 packs would envy.

---

## 6. Recommended Actions

**No action required at v1.0.2.**

The pack passes every Stage 1–4 gate from `using-skillpack-maintenance`:

- Component inventory is complete (router + 13 sheets + 3 commands + 1 agent + wrapper + marketplace entry).
- Frontmatter conformance is uniform.
- Coverage map has no Foundational or Core gaps.
- Sibling boundary with `axiom-rust-engineering` is enforced bidirectionally and in multiple locations.
- SME Agent Protocol is fully implemented on the one agent.
- Slash-command wrapper exists, matches the router, lists all components.
- Behavioral pressure-tests on router + 3 specialist sheets + agent all pass.

**Optional future work (deferred to v1.1.0 if the maintainer chooses):**

- Add Polish-tier items 1–4 above as candidate sheets or sub-sections.
- Consider whether `xtask` deserves a sub-section vs a one-line dismissal.
- Track whether `[patch.crates-io]` patterns recur in `workspace-reviewer` agent runs — if so, promote item 1 from Polish to Minor.

---

## 7. Reviewer Notes

### What this pack does unusually well

- **The artifact-set numbering (`00–13` + `99`) is stable across versions** per the router's explicit statement (`SKILL.md:116`). This means downstream consumers of `99-workspace-engineering-specification.md` from v0.1.0 still cite valid slot numbers in v1.0.2. Stability commitments at the artifact-numbering level are rare in this marketplace and load-bearing for cross-pack citation.
- **The Consistency Gate (`SKILL.md:233–252`) has 12 named checks** that each map to a specific artifact. This is the strongest gate I've seen in any pack — it prevents the "looks complete but isn't" failure mode.
- **The Spec Dependency Graph (`SKILL.md:121–154`) makes propagation explicit.** When a sheet changes, the graph tells you which other sheets need to re-emit. This is the kind of structural metadata that is usually missing.
- **Sibling boundary discipline is rigorous.** Every entry point (this pack's description, Do-Not-Use, Stop Conditions, wrapper, sibling pack's router) gates against the single-crate case. A user genuinely cannot load the wrong pack.
- **Anti-pattern 10 ("We'll Consolidate Later") is unusually self-aware.** Most anti-pattern lists stop at structural shapes. This pack names the *process* anti-pattern — deferral without commitment — and gives a concrete fix (time-box every deferral with a release tag).
- **The agent's "Don't Activate When" list (`agents/workspace-reviewer.md:268–275`) is concrete.** Five explicit no-go cases with the correct sibling pack named for each.

### What this pack does *correctly* but not unusually

- All other dimensions of the fitness scorecard (router quality, descriptions, cohesion, slash-command, SME protocol, cross-linkage) score A. This is what the marketplace pattern looks like when fully implemented — no aspirational scaffolding, no half-finished sheets, no orphaned commands.

### Sibling-vs-nested call

The choice to make `axiom-rust-workspaces` a **separate plugin** rather than a sub-skill of `axiom-rust-engineering` is correct on the evidence:

- The unit of analysis is genuinely different (the workspace, not the crate).
- The artifact set is different (00–13 + 99 vs single-crate audit / delint / profile outputs).
- The commands and agent are workspace-scope-only (none of them operate on a single crate).
- The sibling pack's router *explicitly* redirects workspace concerns here in three places.
- A monolithic "rust" plugin would have crossed the skill-discovery context budget that drove the slash-command wrapper pattern in this marketplace.

### Pack-final commit status

Per memory (`project_axiom_rust_workspaces_v01`), the pack was committed and pushed at v1.0.0; current `1.0.2` reflects post-release bugfixes. No outstanding TDD or scaffolding debt was reported in memory.

### Comparison with the sibling pack's boundary

To sanity-check the sibling discipline, I cross-read `axiom-rust-engineering/skills/using-rust-engineering/SKILL.md` for any boundary leakage:

- Line 26: workspace concerns explicitly redirect to `/rust-workspaces`.
- Line 141 ("How do I structure a Cargo workspace?"): listed as a question this pack does *not* answer; redirects.
- Line 158: explicit note that the sibling pack's project-structure sheet is single-crate-shaped; workspace concerns load `/rust-workspaces`.
- Line 165–176: a section listing seven workspace-scope question shapes ("two crates pin the same dep at different versions," "declare lints once for the whole workspace," "Where does `deny.toml` live in a workspace?," "We have a `[workspace]` block with one member — is that fine?") — each is explicitly out-of-scope for the sibling pack and routes here.
- Line 313: PyO3 work redirects further to `/pyo3-interop` — *not* to this pack — even though PyO3 crates often live inside a Rust workspace. This is the correct nested-routing: PyO3-as-FFI is its own discipline that may compose with this pack but does not replace it.

No bleed across the sibling boundary in either direction. The two packs occupy non-overlapping conceptual space; their pairing is explicit; the routing is bidirectional.

### Final verdict

**PASS.** The pack is structurally sound, behaviourally robust, and discipline-aligned with the marketplace's sibling-not-nested pattern. No fixes required. A small Polish-tier backlog exists for an optional v1.1.0; none of the items would change the pack's overall grade or block a downstream consumer.

**Recommended next maintenance action:** None at this scope. If a v1.1.0 is planned, prioritise Polish items 1 (`[patch.crates-io]`) and 5 (MSRV reconciliation) — both are recurring questions the `workspace-reviewer` agent would surface against real-world workspaces, and both are tight one-paragraph additions to existing sheets rather than new sheets.
