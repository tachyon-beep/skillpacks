# Review: axiom-determinism-and-replay

**Version:** 1.0.3 (from `plugins/axiom-determinism-and-replay/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

## 1. Inventory

### Skills (1 router skill + 13 reference sheets)

The plugin uses a single-skill router pattern: one `SKILL.md` with 13 sibling reference sheets in the same directory. There are no other discrete `SKILL.md` files in the plugin.

Router skill:
- `skills/using-determinism-and-replay/SKILL.md` (438 lines)

Reference sheets (13):
1. `determinism-vs-reproducibility.md` (185 lines) — class choice; produces `01-`
2. `seed-governance.md` (218 lines) — seeds as inputs; produces `02-`
3. `rng-isolation-patterns.md` (243 lines) — per-component RNGs; produces `03-`
4. `snapshot-strategy.md` (198 lines) — capture cadence/encoding; produces `04-`
5. `divergence-detection-and-localisation.md` (227 lines) — compare-points/bisection; produces `05-`
6. `replay-infrastructure-design.md` (250 lines) — read-only/branching machine; produces `06-`
7. `determinism-under-concurrency.md` (165 lines) — A/B/C strategy; produces `07-`
8. `floating-point-determinism.md` (214 lines) — FP policy; produces `08-`
9. `gpu-determinism.md` (220 lines) — cuDNN/cuBLAS/NCCL; produces `09-`
10. `external-effects-substitution.md` (231 lines) — Effects layer; produces `10-`
11. `canonical-state-encoding-for-replay.md` (228 lines) — snapshot bytes; produces `11-`
12. `property-tests-as-determinism-checks.md` (279 lines) — Hypothesis/proptest; produces `12-`
13. `cost-of-determinism.md` (197 lines) — trade record; produces `13-`

### Commands (3)
- `commands/scaffold-replay-system.md` — implementation scaffolding from spec set; `argument-hint: "[component_name_or_path]"`
- `commands/verify-replay.md` — re-runs recorded inputs and asserts equivalence; `argument-hint: "[run_record_path]"`
- `commands/diagnose-divergence.md` — first-differing-op localisation between two runs; `argument-hint: "[run_a_path] [run_b_path]"`

All three declare `allowed-tools` as quoted JSON arrays with `Task` included so they can dispatch the SME agents.

### Agents (2)
- `agents/determinism-reviewer.md` — design-time gap review; `model: opus`
- `agents/replay-debugger.md` — live-divergence localisation; `model: opus`

Both agents declare `description` and `model` only (no `tools:` restriction) — matches the dominant marketplace convention.

### Hooks
None. No `hooks/` directory.

### Slash-Command Wrapper
- `/home/john/skillpacks/.claude/commands/determinism-and-replay.md` (43 lines) — present, well-formed, lists all 13 sheets, 3 commands, 2 agents, and three cross-pack references.

### Marketplace Registration
Registered in `/home/john/skillpacks/.claude-plugin/marketplace.json` with `"source": "./plugins/axiom-determinism-and-replay"` and a complete one-paragraph description that matches the plugin.json description.

## 2. Domain & Coverage

### Stated Domain (from plugin.json + SKILL.md)

"Architecture-level determinism and replay" for systems whose past behaviour must be recoverable as a fact — RL training substrates, multi-agent simulations, multiplayer lockstep engines, replay-debuggable services, and any pipeline where "I cannot reproduce that bug" is unacceptable. Explicitly framed as the architectural counterpart to the verification skill `yzmir-simulation-foundations:check-determinism`.

### Intended Audience

Practitioners-to-experts designing replay-capable substrates. The pack assumes the reader can talk about "determinism class," cuDNN flags, JCS canonical encoding, bit-exact vs logical equivalence, and bisection without explanation. It is *not* a tutorial; it is a discipline spec ("what to write down to claim replay equivalence and not lie about it").

### Coverage Map vs Actual

The pack states its own coverage explicitly via the 14-artifact set (`00-` scope, `01-` through `13-` specialist artifacts, `99-` consolidated spec). Coverage map:

| Channel | Should cover | Sheet | Status |
|---------|--------------|-------|--------|
| Vocabulary / class choice | Bit-exact vs logical vs statistical | `determinism-vs-reproducibility.md` | Present, comprehensive |
| Seeds as inputs | Storage, propagation, audit; `time.time()` ban | `seed-governance.md` | Present |
| RNG partition | Per-component, hierarchical, "one big RNG" anti-pattern | `rng-isolation-patterns.md` | Present |
| Snapshot strategy | Full / delta / event-sourced, what isn't captured | `snapshot-strategy.md` | Present |
| Divergence localisation | Compare-points, hashing, binary-search bisection | `divergence-detection-and-localisation.md` | Present |
| Replay machine | Read-only vs branching; lifecycle | `replay-infrastructure-design.md` | Present |
| Concurrency | Lockstep / recorded-schedule / schedule-independent | `determinism-under-concurrency.md` | Present |
| FP policy | Reduction order, FMA, BLAS pinning, ε | `floating-point-determinism.md` | Present |
| GPU | cuDNN, cuBLAS, TF32, NCCL, atomic-float audit | `gpu-determinism.md` | Present |
| External effects | Effects layer, record-and-replay, deterministic-function | `external-effects-substitution.md` | Present |
| Canonical encoding | JCS, tensor canonicalisation, per-tick hashing | `canonical-state-encoding-for-replay.md` | Present, cross-refs audit pack |
| Property-test verification | Replay equivalence, seed isolation, snapshot round-trip, fork-and-converge | `property-tests-as-determinism-checks.md` | Present |
| Cost / trade record | Perf, library, refactor, ops, cognitive | `cost-of-determinism.md` | Present |

### Gaps

No major gap is visible against the pack's own scope. Minor area-edge gaps observed:

- **No first-class sheet on "determinism for RNGs inside trained NNs at inference time"** (dropout, sampled action selection, beam search) — currently spread across `rng-isolation-patterns.md` and `floating-point-determinism.md`. Reasonable to leave un-split.
- **No sheet on "what to do when an upstream dependency is non-deterministic and cannot be Effects-substituted"** — e.g. cloud APIs whose response is itself nondeterministic. `external-effects-substitution.md` covers substitution, and `cost-of-determinism.md` covers the honest-no pattern, but the specific "third-party stochastic dependency" case is reasoned about across two sheets rather than one.
- **No "test vector exchange" sheet** — the pack mandates a test vector (Consistency Gate Check 10) but doesn't have a sheet on the canonical format / repo location / CI promotion. Likely fine; this is borderline-tooling rather than discipline.

None of these rise to "missing core foundational concept." The pack's coverage of its declared scope is thorough.

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|---|---|---|
| Router quality | Pass | `SKILL.md` (438 lines) is dense but well-organised: Overview, When to Use (with explicit "Do not use") at lines 22-36, Start Here triage (40-48), Pipeline Position diagram (56-75), Expected Artifact Set table (82-98), Spec Dependency Graph + coordinated re-emission rules (102-176), Tier table (182-188), Scenario routing (194-231), Consistency Gate (244-269) with 17 numbered checks, Update Workflows (272-285), Stop Conditions (290-297), Decision Tree (301-333), Integration with three sibling packs (336-379), Quick Reference (382-403), and a Catalog (411-438). All 13 sheets are linked. |
| Skill descriptions | Pass | Every reference sheet has `description: Use when ...` opening (e.g., `seed-governance.md:3`, `gpu-determinism.md:3`, `determinism-vs-reproducibility.md:3`). Each names what produces (`Produces 01-...`, `02-...`, etc.). Conforms to the marketplace's dominant convention. |
| Frontmatter conformance | Pass | Router SKILL.md uses `name` + `description` only. Sheets use `name` + `description` only (sheets are model-readable reference, frontmatter is informational here since the router file is what the runtime indexes). Commands use quoted-array `allowed-tools` (`scaffold-replay-system.md:3` shows `["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]`) and `argument-hint`. Agents use `description` + `model` only — no spurious `tools:` restriction. |
| Component cohesion | Pass | The numbered-artifact contract (00 → 01 → ... → 13 → 99) is consistent across SKILL.md, every sheet's "Produces ##-..." declaration, all three commands' workflow steps, and both agents' input contracts. The `99-` consolidation + Consistency Gate is referenced from every component that needs it. The Spec Dependency Graph (SKILL.md:102-176) is the load-bearing piece — three layers of "if you change X, re-emit Y" tables make cohesion testable, not just claimed. |
| Slash-command exposure | Pass | `/home/john/skillpacks/.claude/commands/determinism-and-replay.md` is present, well-formed, and consistent with router scope. Wrapper description ("Architecture-level determinism and replay...") matches SKILL.md framing. All three commands (`/scaffold-replay-system`, `/verify-replay`, `/diagnose-divergence`) are also slash-exposed via their own `commands/*.md`. |
| SME agent protocol compliance | Pass | Both agents are SME-style (reviewer / debugger). Both: (a) end description with "Follows SME Agent Protocol with confidence/risk assessment" — `determinism-reviewer.md:2`, `replay-debugger.md:2`; (b) include the `**Protocol:**` block citing `meta-sme-protocol:sme-agent-protocol` — `determinism-reviewer.md:10`, `replay-debugger.md:10`; (c) require the four output sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) verbatim — `determinism-reviewer.md:184-205`, `replay-debugger.md:170-192`. All requirements met. |
| Anti-pattern coverage | Pass | Each sheet I sampled has an explicit "Common Mistakes" / "anti-pattern" / "forbidden" section: `determinism-vs-reproducibility.md:141-151` (8 mistakes), `seed-governance.md:44-77` (the `time.time()` family with the variant-that-fools-more-people), `gpu-determinism.md:34-54` (five sources of GPU non-determinism). Commands have `## Common Mistakes` tables: `scaffold-replay-system.md:170-181`, `verify-replay.md:233-244`, `diagnose-divergence.md:208-220`. Agents have `## Common Reviewer/Debugger Mistakes (Self-Discipline)` sections: `determinism-reviewer.md:242-256`, `replay-debugger.md:233-245`. Anti-pattern density is uniformly high. |
| Cross-skill linkage | Pass | Cross-references are dense and bidirectional. Within the pack: every sheet cites the numbered artifact it produces and the consistency-gate check it satisfies. Across packs: `yzmir-simulation-foundations:check-determinism` is cross-linked from SKILL.md:33, the integration section, every relevant sheet, and both agents' Cross-Pack Boundaries tables. `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` is cross-referenced from `canonical-state-encoding-for-replay.md` and `snapshot-strategy.md` per the boundary stated at SKILL.md:352-363. `axiom-solution-architect`, `axiom-static-analysis-engineering`, `yzmir-deep-rl`, and `yzmir-pytorch-engineering:debug-nan` all appear as appropriate cross-pack pointers. |

**Overall: Pass.**

The plugin is structurally sound. No Critical or Major issues; a small number of Minor / Polish items below.

## 4. Behavioral Tests

### Test design notes

This is a stable, expert-tier architectural-discipline pack. Per `testing-skill-quality.md`, technique/reference skills should be pressure-tested for real-world complexity and edge cases; discipline-enforcing skills (which this pack is — the Consistency Gate is the load-bearing discipline) should be pressure-tested for rationalisation resistance. I designed scenarios for both axes.

I did not dispatch live subagents (out-of-scope for a report-only review). Verdicts below are based on close reading of the skill text against each scenario — i.e., "if a model with this skill loaded encountered this scenario, what guidance is present that should pin its behaviour?" Where the text is unambiguous, the verdict is high-confidence; where the model would still have to make judgement calls, that is noted.

### Test 1 — Router: "We need replay; just give us the checklist"

**Scenario:** A senior engineer is on a deadline. They have a half-built RL training loop, no `01-` decided, no seed governance, and ask: "Skip the philosophical stuff. Just tell me which files to add so we get replay." Pressure: time + simplicity-temptation.

**Expected:** The router should refuse to skip class choice (`01-`); it should drive the team back to `determinism-vs-reproducibility.md` before scaffolding anything.

**Likely actual:** Router holds.
- SKILL.md:40-48 "Start Here" mandates reading `determinism-vs-reproducibility.md` *first* and explicitly says "Steps 1-3 are the spike. If those three artifacts hold together — vocabulary fixed, seeds governed, RNGs isolated — the rest is well-defined work. If they don't, no later sheet can save you."
- SKILL.md:293 (Stop Conditions): "The team disagrees on what 'deterministic' means... Stop at `01-`. Resolve the disagreement before any other artifact is written."
- Consistency Gate Check 2 (SKILL.md:250) makes "vague claims like 'deterministic' without a class" a gate failure.
- The Decision Tree (SKILL.md:301-333) starts at class choice, not at scaffolding.

**Verdict:** Pass. The text is unambiguous; rationalisation requires actively ignoring the gate-and-stop machinery.

### Test 2 — `seed-governance.md`: "We already deployed; the seed is `time.time()`, we can't change it now"

**Scenario:** Engineer has shipped a system that seeds RNGs from `time.time()`. They ask for "the minimum viable governance fix that doesn't require redeployment." Pressure: sunk-cost + overkill-perception.

**Expected:** The sheet should refuse the workaround; it should describe `time.time()`-seeding as a determinism bug that voids the run's replay claim, and direct the engineer to either accept "this system is non-replayable" or fix the seed path.

**Likely actual:** Sheet holds.
- Lines 44-52: The forbidden patterns are explicit. `random.seed(time.time())` is flagged "FORBIDDEN: time is not recorded."
- Lines 53-63: The "if seed is None: seed = random.randint(...)" variant is called out as "*less* obvious but equally fatal."
- Lines 65-75: The correct pattern is binary — provide as input or fail fast. No "soft" middle ground.
- Line 41: The framing — "A seed that lives in `__init__` defaults... guarantee that some runs are unreplayable — and which runs are unreplayable is itself non-deterministic" — pre-empts the sunk-cost frame.

**Verdict:** Pass. The sheet explicitly refuses the workaround that the scenario tempts. The model would have to choose to override.

### Test 3 — `gpu-determinism.md`: "Customer wants bit-exact GPU on whatever GPU AWS gives us"

**Scenario:** Real-world complexity. Customer mandates bit-exact replay across whatever GPU SKU AWS provides on a given day. Pressure: messy requirement; the obvious answer ("just turn on deterministic mode") is wrong because cross-SKU bit-exactness is mostly infeasible.

**Expected:** Sheet should distinguish "bit-exact, single-SKU" (achievable) from "bit-exact, cross-SKU" (mostly infeasible), and direct the team to either pin the SKU or downgrade the class to logical-equivalence with ε.

**Likely actual:** Sheet holds.
- Lines 57-63 (Determinism-Class Implications table): "Bit-exact, GPU vs CPU | Almost impossible. Different reduction trees, different libm, different FP behaviour. Reframe to logical-equivalence." Same logic extends to cross-SKU per Channel 8.
- Lines 22-23 ("A CI job on a different GPU SKU produces different results than the dev box") names the exact scenario as a use case for the sheet.
- The cross-SKU case is also discussed in `replay-debugger.md`'s symptom table (line 125: "Diverges across machines but not on dev box | Architecture-specific FP / GPU / library | `08-` or `09-`").
- The decision-feasibility framing — bit-exact-single-SKU is achievable but pinning is a tier-promotion and a cost — is exactly what `cost-of-determinism.md` exists to capture.

**Verdict:** Pass. The text gives the model a clear path: "this is what bit-exact requires; if you can't meet the requirement, the class is wrong; here is the lower class that survives this." The risk is that a less-experienced model might still try to claim bit-exact with a perfunctory `torch.use_deterministic_algorithms(True)` call — but the sheet's lines 65-79 explicitly call this out (it's necessary, not sufficient).

### Test 4 — Skill: `divergence-detection-and-localisation.md` via the `replay-debugger` agent prompt

**Scenario:** Two RL workers diverged at tick 47812. Engineer says "Just diff the logs — I can see they diverged at tick 47812. Patch the FP issue there."

**Expected:** The pack should refuse the "diff logs at T_max" approach and force bisection to T₀ (first-differing), not T_max (where divergence was detected).

**Likely actual:** Sheet and agent both hold.
- `replay-debugger.md` Core Principle (line 19): "Localise before fixing. The first-differing operation is the bug; everything after it is a consequence." Lines 20-21 expand: "A divergence at tick 100 looks like 100 bugs to a casual reader; it is one bug at tick *T₀ ≤ 100*, plus 100−T₀ propagations of that bug."
- Common Debugger Mistakes (line 236): "Reporting the *symptom* tick (T_max) as the bug | The bug is at T₀; everything after is propagation."
- Step 3 (lines 84-95) is the bisection procedure, with cost framed as O(log N) — not an aspirational "do this if you have time."
- The SKILL.md scenario for live divergence (lines 210-215) puts `/diagnose-divergence` and the agent as the first action, not log-diffing.

**Verdict:** Pass. The pack treats "diff the logs" as the named anti-pattern. The `Common Mistakes` table is targeted exactly at this pressure.

### Test 5 — Real-world complexity: `/scaffold-replay-system` invoked against a half-specced system

**Scenario:** Engineer runs `/scaffold-replay-system` against a component that has `00-`, `01-`, `02-` but is missing `03-`, `04-`, `05-`. They want code now and will "fill in the missing specs later."

**Expected:** Command should refuse to scaffold against an incomplete spec at the declared tier; should drive back to producing the missing specs first.

**Likely actual:** Command holds.
- Step 1 (lines 65-92 of `scaffold-replay-system.md`): "For required-but-missing artifacts, run the corresponding sheet from the catalog before scaffolding. Do NOT scaffold against an incomplete spec. The scaffold's choices come from the spec; absent specs make those choices arbitrarily."
- The tier-required artifact table (lines 67-85) is explicit per tier.
- Common Mistakes (line 173): "Scaffolding generated before specs are complete | Halt; complete the spec; come back."

**Verdict:** Pass. The "fill in specs later" rationalisation is pre-empted by the command's own anti-pattern table.

### Test summary

All five probe scenarios resolve in favour of pack guidance over user pressure. The pack is unusually resistant to the "just do it" pressure because the Consistency Gate is concrete (17 numbered checks, each with a question that has a binary answer) rather than being a vibe-based "remember to be careful." That makes shortcuts visible as gate failures, which makes them harder to rationalise. The agents inherit the same discipline because both cite the gate by reference and their output sections (Confidence / Risk / Information Gaps / Caveats) force the same precision on the model.

The only scenario where I'd expect slippage in practice is one I did not test directly: a *novice* trying to choose a determinism class. `determinism-vs-reproducibility.md` is dense; a reader unfamiliar with the underlying concepts (FP non-associativity, ULPs, KL divergence) may pick a class by reading the headlines and miss the predicate work. This is consistent with the pack's stated audience (practitioners-to-experts) and is the natural cost of the depth.

## 5. Findings

### Critical
None.

### Major
None.

### Minor

**M1. "13 specialist sheets" framing vs. router that includes its own consolidation work.**
The router does perform consolidation (`99-determinism-and-replay-specification.md`) and gate-running, which is itself substantial work but is not allocated to a dedicated sheet. This is a deliberate choice (per SKILL.md:98 the consolidation is "router-owned"), and the gate is comprehensive (17 checks, lines 244-269). The minor cost is that the router file is 438 lines and contains both routing logic *and* the load-bearing gate — a future reader may not realise the gate lives in the router rather than in a sheet. **Verdict: leave as-is; it is documented.**

**M2. `replay-infrastructure-design.md` (250 lines) handles both read-only and branching replay.**
Two genuinely different machines living in one sheet. The router (SKILL.md:223-226) does call this out as a separation. A future pack-version might split them, but for v1.0.3 the unified sheet is reasonable — branching replay reuses most of the read-only machinery and the differences are well-scoped.

**M3. Audience pitch is uniformly expert.**
Newcomers to the determinism domain (e.g., a backend engineer suddenly asked to build replay for a SaaS audit feature) will find the first three sheets demanding. There is no "determinism-in-30-minutes" tutorial sheet. Given the pack's stated audience this is intentional, but cross-pack-onboarding cost is real. **Verdict: a "starter" pointer sheet (or a section in the router) would help; not blocking.**

### Polish

**P1. SKILL.md is 438 lines.** Dense. Five-plus tables. The Spec Dependency Graph + three coordinated-re-emission tables (lines 142-175) is itself ~35 lines of dense table content. The text is well-organised but a future maintainer making any structural change has 438 lines to reconcile.

**P2. Plugin description in `.claude-plugin/plugin.json:4` is one very long sentence (1,400+ chars).** It reads as a feature list. The marketplace catalog description (one paragraph) is comparably long. Both are accurate; neither would render gracefully in a marketplace UI list view. **Verdict: cosmetic.**

**P3. `cost-of-determinism.md` covers the trade record but doesn't have a quick "when to refuse" checklist.** The Stop Conditions in the router (lines 290-297) partly fill this, but a quick rule of thumb ("if perf budget < 30% over baseline, target Class 2 not Class 1") would help. The sheet's discipline is already good (it covers per-rule cost, library-trade record, partial-determinism boundary, response-on-budget-breach); this is a polish, not a gap.

**P4. The Consistency Gate has 17 checks but no "minimum-set-for-tier-XS" abridged version.** The gate description (line 269) does say "Checks 11–17 are evaluated *only if their precondition holds*" which addresses this, but a one-line "XS tier runs Checks 1, 2, 3, 4, 10, 17" summary would speed onboarding. The Tier table (lines 182-188) names required artifacts but doesn't pre-map them to gate checks.

**P5. `replay-debugger`'s symptom table (lines 115-127) is excellent for known patterns** but the residual "novel divergence pattern" case is only loosely covered ("Channel attribution confidence: Low when the divergence pattern is novel" at line 173). A short "what to do when the symptom doesn't match the table" workflow would close the loop.

## 6. Recommended Actions

None required for v1.0.3. The plugin is structurally sound, well cross-linked, and the discipline holds under pressure. If a future v1.1 is being planned, the items below are ordered by leverage:

1. **(Polish, low effort)** Add a 3-5 line "XS-tier abridged Consistency Gate" subsection to SKILL.md right after the gate table (after line 269). Map XS, S, M, L, XL each to the subset of checks they actually run. This makes the smallest-system path lighter without changing the discipline.
2. **(Polish, low effort)** Add a one-paragraph "starter" section to SKILL.md aimed at first-time readers — the existing "Start Here" (lines 40-48) is workflow-step-1 framing, not "what to read first if you are new to the concept." A two-link prelude pointing to `determinism-vs-reproducibility.md` and `cost-of-determinism.md` first would help cross-pack onboarding.
3. **(Polish, low effort)** Add a "novel pattern" workflow to `replay-debugger.md` Step 5 covering the case where the divergence symptom is not in the table.
4. **(Polish, low effort)** Shorten the plugin.json description to ~250 chars (one short paragraph) for marketplace-UI rendering; keep the long-form in SKILL.md.
5. **(Optional, medium effort)** Add a "test vector exchange" mini-sheet or section: where the recorded `(seed, code_version) -> state_hash` triples live, how they're updated under code change, how they're consumed by CI. Currently referenced by Consistency Gate Check 10 and by `/verify-replay` Step 7 but not anchored to a single source.
6. **(Optional, larger)** Consider splitting `replay-infrastructure-design.md` into `replay-infrastructure-read-only.md` and `replay-infrastructure-branching.md` if the next major rewrite is targeting v2.0. They are different machines and the unified sheet is on the long side. Defer until there's an editorial reason.

None of these are blocking. The plugin's current state matches its stated v1.0.0 (now v1.0.3) "API stability" promotion.

## 7. Reviewer Notes

This review is read-only. No files in `plugins/axiom-determinism-and-replay/` were modified.

The pack uses a single-router-skill pattern rather than one `SKILL.md` per discipline area. This is unusual relative to some other packs in the marketplace (which have separate `SKILL.md`-per-skill files registered individually), but in this pack the router + sheet approach is intentional and matches what `meta-skillpack-maintenance:using-skillpack-maintenance` describes for router patterns. The 13 sheets are reference content rather than discrete discoverable skills — they are loaded by the router on demand. The structure is internally consistent.

Confidence in this review: high. The pack is well-bounded (architectural discipline for a narrow problem), the discipline-gate (17 numbered consistency checks) makes claims testable, and the SME-agent protocol compliance is clean (both agents have description, protocol block, and required output sections). The behavioral tests are based on close reading rather than live subagent dispatch, which is the noted limit; live testing would marginally tighten verdicts but is unlikely to flip any of them given how explicit the anti-pattern coverage is.

The marketplace context (per the user's memory: `axiom-determinism-and-replay` was promoted from v0.2.0 to v1.0.0 to mark API stability, then v1.0.3 with patches) is consistent with what's on disk: structurally stable, with patch-level polish since the stability promotion. No structural debt visible.
