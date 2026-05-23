# Review: axiom-rust-engineering

**Version:** 1.0.2 (per `plugins/axiom-rust-engineering/.claude-plugin/plugin.json`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent
**Methodology:** Stages 1–4 of `meta-skillpack-maintenance:using-skillpack-maintenance` (Stage 5 / fixes skipped — report-only)

---

## 1. Inventory

### Plugin metadata

- `plugins/axiom-rust-engineering/.claude-plugin/plugin.json:1-22` — version `1.0.2`, declares "Router skill + 11 reference sheets, 5 commands, 3 agents". Inventory below matches that count.
- Marketplace registration: present in `.claude-plugin/marketplace.json` (description and keywords mirror the plugin.json blurb).

### Router skill

| Skill | Path | Length |
|-------|------|--------|
| `using-rust-engineering` | `plugins/axiom-rust-engineering/skills/using-rust-engineering/SKILL.md` | 518 lines |

Frontmatter (`SKILL.md:1-4`):

```yaml
---
name: using-rust-engineering
description: Use when working in Rust — routes to specialist sheets for borrow-checker errors (E0502/E0597/E0382), trait bounds (E0277), async Send/Sync, clippy warnings, unsafe/FFI soundness, performance profiling, or PyO3/candle interop
---
```

Conforms to repo convention (`Use when…`, no `allowed-tools`).

### Reference sheets (11)

All siblings of `SKILL.md` in `skills/using-rust-engineering/`:

| Sheet | Lines |
|-------|------:|
| `ai-ml-and-interop.md` | 1290 |
| `async-and-concurrency.md` | 1474 |
| `error-handling-patterns.md` | 1079 |
| `modern-rust-and-editions.md` | 928 |
| `ownership-borrowing-lifetimes.md` | 1087 |
| `performance-and-profiling.md` | 1036 |
| `project-structure-and-tooling.md` | 1190 |
| `systematic-delinting.md` | 828 |
| `testing-and-quality.md` | 1335 |
| `traits-generics-and-dispatch.md` | 1262 |
| `unsafe-ffi-and-low-level.md` | 1307 |

Each sheet uses the SKILL.md → reference-sheet relative-link pattern (`[name.md](name.md)`); router's "How to Access Reference Sheets" section (`SKILL.md:30-44`) calls out the directory convention explicitly.

### Commands (5)

| Command | Path | argument-hint | allowed-tools |
|---------|------|---------------|---------------|
| `audit` | `commands/audit.md` | `"[workspace root] - defaults to current directory"` | `["Read","Edit","Bash","Skill"]` |
| `create-project-scaffold` | `commands/create-project-scaffold.md` | `"[project-name] - name for the new Rust project"` | `["Read","Write","Edit","Bash","Skill"]` |
| `delint` | `commands/delint.md` | `"[package spec] - optional ` + "`-p <name>`" + ` package selector; defaults to current workspace"` | `["Read","Edit","Bash","Glob","Grep","Skill"]` |
| `profile` | `commands/profile.md` | `"[binary or bench target] - what to profile"` | `["Read","Edit","Bash","Glob","Grep","Skill"]` |
| `typecheck` | `commands/typecheck.md` | `"[package spec] - optional ` + "`-p <name>`" + ` package selector; defaults to current workspace"` | `["Read","Edit","Bash","Skill"]` |

All five use quoted JSON-array `allowed-tools` and quoted `argument-hint` strings — conforms to marketplace convention (`reviewing-pack-structure.md` checklist).

### Agents (3)

| Agent | Model | SME description ending | SME `Protocol:` cite | 4 sections present |
|-------|-------|------------------------|----------------------|---------------------|
| `clippy-specialist` | sonnet | `…Follows SME Agent Protocol.` | Yes (`clippy-specialist.md:9`) | Yes (`:122,130,138,146`) |
| `rust-code-reviewer` | sonnet | `…Follows SME Agent Protocol with confidence/risk assessment.` | Yes (`rust-code-reviewer.md:10`) | Yes (`:101,105,110,115`) |
| `unsafe-auditor` | sonnet | `…Follows SME Agent Protocol.` | Yes (`unsafe-auditor.md:10`) | Yes (`:149,153,159,164`) |

No `tools:` restriction declared on any agent (inherits parent context — matches the ~60/65 marketplace baseline).

### Hooks

- `plugins/axiom-rust-engineering/hooks/hooks.json` contains only `{ "hooks": {} }` — an empty placeholder. No matchers, no commands.

### Slash-command wrapper

- `.claude/commands/rust-engineering.md` — **MISSING.** Siblings `.claude/commands/rust-workspaces.md` and `.claude/commands/pyo3-interop.md` both exist; only the foundational router of the rust family is unexposed as a slash command.

---

## 2. Domain & Coverage

### Intended scope (per `plugin.json` and router preamble)

Single-crate-shaped Rust engineering for the 2024 edition: ownership/borrowing/lifetimes, traits/generics/dispatch, modern syntax, async/tokio, error handling, project structure, testing, performance, clippy delinting, unsafe/FFI, ML interop entry point (PyO3/candle/burn). Explicitly excludes:

- Multi-crate workspace topology and policy → defers to `axiom-rust-workspaces`.
- Production PyO3 FFI boundary discipline → defers to `axiom-pyo3-interop`.

### Coverage map vs. inventory

| Domain area | Status | Evidence |
|-------------|--------|----------|
| Editions / 2024 migration | Covered | `modern-rust-and-editions.md` (928 lines) |
| Ownership, borrows, lifetimes | Covered | `ownership-borrowing-lifetimes.md` (1087 lines) |
| Traits, generics, object safety | Covered | `traits-generics-and-dispatch.md` (1262 lines) |
| Error-handling idioms | Covered | `error-handling-patterns.md` (1079 lines) |
| Cargo, lints config, CI, build profiles | Covered | `project-structure-and-tooling.md` (1190 lines) |
| Workspace topology / multi-crate | **Overlaps with sibling pack** (see §5 Major-1) | `project-structure-and-tooling.md:161-289+` redundantly covers `[workspace.dependencies]`, `[workspace.lints]`, resolver 2/3, virtual manifests |
| Test layout, mocks, proptest, llvm-cov | Covered | `testing-and-quality.md` (1335 lines) |
| Clippy delinting methodology | Covered | `systematic-delinting.md` (828 lines) |
| async/tokio, Send/Sync across await | Covered | `async-and-concurrency.md` (1474 lines) |
| Profiling, flamegraphs, criterion | Covered | `performance-and-profiling.md` (1036 lines) |
| Unsafe, raw pointers, FFI, Miri | Covered | `unsafe-ffi-and-low-level.md` (1307 lines) |
| PyO3 onboarding + candle/burn/tch-rs/ndarray | Covered (with redirect to `/pyo3-interop` for production) | `ai-ml-and-interop.md:313-322` |

### Gaps

None at the foundational level. Coverage is complete for the declared scope; the only domain-shaped issue is overlap, not absence (see §5).

### Currency

Stable domain. The pack already pins to current versions: edition 2024, PyO3 0.25 with explicit drift notes for 0.23 / 0.24 (`ai-ml-and-interop.md:20-42`), resolver 3 baseline (`project-structure-and-tooling.md:199`). No research currency flag.

---

## 3. Fitness Scorecard

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Router quality** | Pass | 13 routing tables by symptom, ambiguity-resolution priority rules (`SKILL.md:328-355`), cross-cutting scenarios with execution order (`:361-388`), common-routing-mistakes table (`:391-403`), 6-question self-check (`:434-460`). Among the most disciplined routers in the marketplace. |
| **Skill descriptions** | Pass | Frontmatter `description` starts with `Use when…`, names error codes (E0502/E0597/E0382/E0277), tools (clippy, tokio, PyO3), and concepts — high-precision activation signal. |
| **Frontmatter conformance** | Pass | Router SKILL.md uses `name`+`description` only (correct). Commands use quoted JSON-array `allowed-tools` and quoted `argument-hint`. Agents use `description`+`model` only, no spurious `tools:`. |
| **Component cohesion** | Major-1 | Router and 10 of 11 sheets stay strictly single-crate. `project-structure-and-tooling.md:161-289+` contradicts that scope with ~130 lines of workspace material (manifests, resolver, `[workspace.dependencies]`, `[workspace.lints]`, virtual manifests) — and has *no in-sheet redirect to `/rust-workspaces`*. |
| **Slash-command exposure** | Major-2 | `.claude/commands/rust-engineering.md` is missing. Both siblings (`rust-workspaces.md`, `pyo3-interop.md`) are present; the foundational router of the rust family is uniquely unexposed. |
| **SME agent protocol** | Pass | All three agents declare the protocol in their description, cite `meta-sme-protocol:sme-agent-protocol` in the body, and include all four required section headings (Confidence Assessment / Risk Assessment / Information Gaps / Caveats). |
| **Anti-pattern coverage** | Pass | Router includes "Common Rationalizations" (`SKILL.md:419-431`), "Red Flags — Stop and Route" (`:406-416`), "Red Flags Checklist" (`:434-460`), "Common Routing Mistakes" table (`:393-403`). Specialist sheets (e.g., `systematic-delinting.md:29,515` on `#![allow(clippy::all)]`) repeat the discipline. |
| **Cross-skill linkage** | Pass with one caveat | Inbound links: from sibling packs both reference this one. Outbound: explicit, well-scoped redirect to `/pyo3-interop` at `ai-ml-and-interop.md:313-322` (exemplary), explicit redirect to `/rust-workspaces` at three places in `SKILL.md` (`:26, :158, :176-184`). Caveat: that redirect discipline does not extend into `project-structure-and-tooling.md` itself — see Major-1. |
| **Hooks** | Minor | `hooks/hooks.json` is `{ "hooks": {} }` — an empty placeholder. Either delete the file or document intent. |

**Overall:** **Major** (two Major issues; pack remains usable, but both are squarely within the maintenance remit). Otherwise structurally sound and one of the more disciplined packs in the marketplace.

---

## 4. Behavioral Tests

Tests run as scenario walk-throughs against the router and two specialist sheets. Each scenario quotes the load-bearing text that would (or would not) catch the failure mode.

### Test 4.1 — Router, ambiguity resolution: "PyO3 extension panics with borrow-checker error"

**Scenario:** User reports a PyO3 extension that fails at compile time with E0597 ("borrowed value does not live long enough"). Two possible routes: `ai-ml-and-interop.md` (PyO3 patterns) or `ownership-borrowing-lifetimes.md` (E0597).

**Router text (`SKILL.md:340-342`):**

> **"PyO3 + lifetime errors"**:
> 1. Start with [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — resolve the Rust-side lifetime model first
> 2. Then [ai-ml-and-interop.md](ai-ml-and-interop.md) — apply PyO3-specific patterns (GIL management, `Python<'py>` tokens)

**Result:** **Pass.** The "Ambiguous Symptom Resolution" section catches the exact case and orders the load correctly (Rust-side lifetime first, FFI patterns second). Re-stated in "Common Routing Mistakes" (`:399`): `"PyO3 panic at runtime → ... unsafe-ffi or ownership first"`.

### Test 4.2 — Router, pressure resistance: "Just .clone() it, I'm in a hurry"

**Scenario:** User has a borrow-checker error inside a hot loop and asks: *"this is urgent, can I just `.clone()` the value to make it compile?"*

**Router text (`SKILL.md:425`):**

> | "I'll just add `.clone()`" | That's the anti-pattern the specialist exists to prevent | Route to [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) for the systematic alternative |

Reinforced at `:413`: `"Sprinkle lifetime annotations without a model → Route to ownership-borrowing-lifetimes.md"`. And the 6-step self-check (`:434-460`) closes with `:457-459`:

> 6. **Am I feeling pressure to skip routing?**
>    - Route anyway. Time pressure, apparent complexity, user confidence, and "this is simple" all produce the same bad outcome — a direct answer that misses the specialist's checklist.

**Result:** **Pass.** Pressure pattern is named verbatim ("I'll just add `.clone()`"), the rationalization table catches it, and the self-check explicitly tells the model to route under pressure. Strong rationalization-resistance.

### Test 4.3 — Specialist (`systematic-delinting.md`), pressure resistance: "500 warnings, just add `#![allow(clippy::all)]` to ship"

**Scenario:** User has 500+ clippy warnings before a release deadline and proposes blanket-allowing clippy at crate root to make CI pass.

**Specialist text:**

- Overview (`systematic-delinting.md:5`): `"Fix warnings systematically, NEVER silently suppress them."`
- Trigger section (`:29`) names the rationalization verbatim: `"Team added #![allow(clippy::all)] to shut up CI — that's wrong, right?"` — i.e., the sheet self-identifies as the answer to that exact pressure.
- "When NOT to Use" (`:36`) further refuses blanket-restriction enabling.
- Allow-attribute discipline (`:515-519`):
  > Never add `#[allow]` at the crate level to silence a lint that should be fixed. The allowed scope should be the smallest item where the lint fires: a single function, a single `impl` block, or a single `match` arm.

**Result:** **Pass.** The skill names the shortcut explicitly and refuses it in three places. A test runner under deadline pressure has multiple anchors to push back on.

### Test 4.4 — Specialist (`ai-ml-and-interop.md`), redirect trigger: "Production PyO3 inference server, GIL contention is the bottleneck"

**Scenario:** A production scenario where the FFI surface is the hot path — exactly the case where the pack should hand off to `/pyo3-interop`, not handle the work itself.

**Specialist text (`ai-ml-and-interop.md:313-322`):**

> **For production-grade PyO3 work**, redirect to **`/pyo3-interop`** (the `axiom-pyo3-interop` pack). … Load that pack when the FFI surface is the hot path (RL self-play, simulation engines, training data pipelines, inference servers); this sheet is sufficient for one-off bindings and ML-framework selection.

Followed by five concrete redirect examples (`:318-322`), including:
- `"GIL deadlock in our PyO3 module" → /pyo3-interop:gil-release-patterns and /pyo3-interop:debugging-pyo3.`

**Result:** **Pass.** The redirect criterion is concrete ("FFI surface is the hot path"), names the right consumer scenarios (inference servers, RL self-play), and provides example-level routing to the specific sibling-pack sheets. This is the cleanest sibling-pack handoff in the rust family.

### Test 4.5 — Specialist (`project-structure-and-tooling.md`), boundary discipline: "Set up a Cargo workspace with shared lints"

**Scenario:** User asks how to set up a Cargo workspace with `[workspace.dependencies]` and `[workspace.lints]`.

**Specialist text:** The router (`SKILL.md:158`) says:

> **Note**: This pack's project-structure sheet is **single-crate-shaped**. For workspace-scope concerns … load **`/rust-workspaces`** instead.

But `project-structure-and-tooling.md:161-289+` actually contains ~130 lines of workspace material — canonical workspace layout, full workspace `Cargo.toml` example with `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints.rust]`, `[workspace.lints.clippy]`, member-inheriting Cargo.toml, resolver-2/3 comparison table, and virtual manifests. There is **no in-sheet redirect to `/rust-workspaces`** (verified via `grep` — empty result on the sheet itself).

**Result:** **Fix needed (Major).** A reader who loads the sheet directly (e.g., because routing landed them there from "Project Setup and Tooling") will be handed workspace guidance that the router declared out of scope, with no signal to escalate to `/rust-workspaces`. The boundary discipline that `ai-ml-and-interop.md:313-322` exemplifies is missing here. See §5 Major-1.

---

## 5. Findings

### Critical (0)

None. The pack is usable.

### Major (2)

**Major-1 — Workspace material overlaps with `axiom-rust-workspaces`, with no in-sheet redirect.**

- **Location:** `plugins/axiom-rust-engineering/skills/using-rust-engineering/project-structure-and-tooling.md:161-289+`.
- **Evidence:** The "## Workspaces" section spans ~130 lines and covers exactly the topics the sibling `axiom-rust-workspaces` pack claims as its core sheets per the marketplace description (`[workspace.dependencies]`, `[workspace.lints]`, resolver-2/3, workspace structure). The sibling pack's marketplace entry lists `workspace-structure-patterns`, `workspace-dependencies-and-resolver`, `workspace-lints-and-clippy-config` as top-line sheets.
- **Why this is Major:** The router (`SKILL.md:26, :158, :176-184`) explicitly defers workspace-scope work to `/rust-workspaces` three times, but the actual sheet contradicts that contract. A reader loaded into the sheet (not arriving via the router) gets stale duplicate guidance with no escalation signal. Drift between the two packs is now possible — the sibling can update resolver guidance and this pack will silently disagree.
- **Two readings, severity unchanged:** Either (a) this is leftover from before the pack split and should be trimmed, or (b) it's intentional "minimum workspace literacy" so the single-crate pack stands alone. In case (b) the section still needs an explicit redirect note and a scope statement ("this sheet's workspace coverage is one-page literacy; for workspace-scope decisions load `/rust-workspaces`"). In case (a) it should be cut.
- **Recommended fix:** Add a redirect block at the top of the "## Workspaces" section pointing to `/rust-workspaces`, and either (i) cut the section to a 10–15 line orientation that defers anything beyond that, or (ii) keep the current content but caveat it explicitly as introductory. Mirror the pattern from `ai-ml-and-interop.md:313-322`.

**Major-2 — Missing slash-command wrapper `.claude/commands/rust-engineering.md`.**

- **Location:** Expected at repo root `.claude/commands/rust-engineering.md`. File does not exist.
- **Evidence:** `ls .claude/commands/ | grep -E "rust|pyo3"` returns only `pyo3-interop.md` and `rust-workspaces.md`. Both siblings exist; the foundational router does not.
- **Why this is Major:** Per repo `CLAUDE.md`, router skills are exposed as slash commands to bypass skill-discovery context limits. A user typing `/rust-workspaces` or `/pyo3-interop` is doing so under the assumption that `/rust-engineering` is a sibling — it isn't. The absence is asymmetric with the rest of the rust family.
- **Recommended fix:** Create `.claude/commands/rust-engineering.md` modelled on `.claude/commands/rust-workspaces.md` (1-page sibling-pack-aware wrapper). Cross-link `/rust-workspaces` and `/pyo3-interop`.

### Minor (1)

**Minor-1 — Empty `hooks/hooks.json` placeholder.**

- **Location:** `plugins/axiom-rust-engineering/hooks/hooks.json` contains only `{"hooks": {}}`.
- **Why this is Minor:** Not broken, but it creates the impression of intent without payload. Either delete the file (no hooks needed for this pack — that's fine; most packs do not ship hooks) or document why it's present (e.g., placeholder for a future `cargo fmt`-on-save hook).
- **Recommended fix:** Delete the file unless a hook is planned. If kept, add a one-line comment to plugin.json or a `hooks/README.md` noting the placeholder status.

### Polish (0)

No polish-level findings beyond the Minor above. The router, sheets, commands, and agents are unusually clean for a pack of this scope (13K lines of reference material).

---

## 6. Recommended Actions

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | Create `.claude/commands/rust-engineering.md` modelled on `rust-workspaces.md`. Cross-link both siblings. | Major-2 | Small (~30 min) |
| 2 | Add redirect notice at top of `project-structure-and-tooling.md:161` "## Workspaces" section pointing to `/rust-workspaces`. Trim the section to orientation-only, or caveat the existing content as introductory. | Major-1 | Small–Medium |
| 3 | Resolve `hooks/hooks.json` — either delete the file or annotate the placeholder. | Minor-1 | Trivial |

**Suggested version bump on fix-application:** Minor (1.0.2 → 1.1.0). Major-2 is a new user-facing surface; Major-1 is a meaningful boundary-clarification of existing content. Neither is a breaking change.

**No new skills required.** All gaps are within existing components; no `superpowers:writing-skills` invocations needed.

---

## 7. Reviewer Notes

**Strengths worth preserving:**

1. **The `/pyo3-interop` redirect at `ai-ml-and-interop.md:313-322` is exemplary.** It names the trigger criterion ("FFI surface is the hot path"), enumerates consumer scenarios (RL self-play, inference servers), and provides example-level routing to specific sibling-pack sheets (`/pyo3-interop:gil-release-patterns`). This is the pattern Major-1 should adopt for the workspace boundary.

2. **The router's discipline against direct advice is unusually rigorous.** The 6-question self-check at `SKILL.md:434-460` explicitly addresses the "pressure to skip routing" failure mode, and the rationalization table at `:419-431` names the exact shortcut phrases ("I'll just add `.clone()`", "One clippy `allow` won't hurt"). This is the kind of pressure-resistance most routers handwave.

3. **All three agents are SME-protocol-compliant.** Description endings, `Protocol:` body cite, and the four canonical section headings (Confidence Assessment / Risk Assessment / Information Gaps / Caveats) are all present and verbatim. No `tools:` restrictions — matches the marketplace baseline.

4. **Cross-cutting scenarios with execution order** (`SKILL.md:361-388`) are a feature most routers lack. "Setup before optimization, diagnosis before fixes, soundness before polish" is the right hierarchy for a Rust pack and the load orders are correct.

**Boundary scheme (sibling-not-nested) is sound at the contract level.** Router → `/rust-workspaces` is referenced three times (`:26, :158, :176-184`). Router → `/pyo3-interop` is referenced in the AI/ML routing entry (`:313-322`). The contract holds in `SKILL.md`; what's missing is for the *sheets themselves* to mirror that contract when read in isolation. Major-1 is the one place the sheets fall short of the router's declaration.

**Behavioral testing methodology note.** Tests were run as scenario walk-throughs against the actual sheet text (with verbatim quotes) rather than subagent dispatch. For a Major-finding report this is appropriate; for fix verification post-implementation, prefer subagent dispatch per `testing-skill-quality.md:81-92`.

**Out-of-scope but observed:** The `axiom-rust-engineering` ↔ `axiom-rust-workspaces` ↔ `axiom-pyo3-interop` family is one of the cleaner factional partitions in the marketplace. The two findings here are local to the foundational pack and do not threaten the partition.
