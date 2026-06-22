# Report Card — muna-panel-review

**Version:** 0.4.0 (plugin.json) · **Track:** S — Soft / Judgment (editorial-intelligence simulation)
**Graded:** 2026-06-22 · **Prior review:** `reviews/muna-panel-review.md` (2026-05-22, v0.3.2 — its three Major packaging findings are now RESOLVED; see Gate analysis)

A structurally unusual pack: 1 orchestration skill (`reader-panel-review`), a ~1130-line methodology authority (`process.md`), 3 agents (persona-reader, panel-synthesiser, persona-designer), 3 in-pack commands, 3 slash wrappers, plus `config-template.md`, a worked `config.md`, and `ARCHITECTURE.md`. There are no numbered reference sheets — the "sheets" are `process.md` phases.

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (track S: judgment defensible, framing right, current practice) | **A** | The methodology is the densest anti-contamination design in the marketplace. The no-read-ahead cardinal rule (`process.md:149-161`) is enforced through four independent mechanisms: structural (hashed chapter filenames + Read/Write-only tool restriction, `SKILL.md:117-132`, `persona-reader.md:3-4,17`), procedural (Step A expectations gate, `SKILL.md:271`, `process.md:949-957`), behavioural (persona operating rules), and detectable (post-hoc contamination flags, `process.md:957`). The epistemic three-tier confidence grading (`process.md:756-762`), the convergent-reasons test (`process.md:764`), the control persona (`process.md:104-122`) and unreliable-narrator (`process.md:125-143`) calibration mechanisms, and the explicit "what this does NOT do" + known-limitations section (`process.md:1070-1132`) are mature, honest, defensible judgment — not platitudes. Held back from S by surface area that exceeds the runnable path (Phase 5 derivative-document development `process.md:789-873` and the delegation-chain variant are documented but not wired into the coordinator skill's lifecycle, `SKILL.md:464-501`). |
| **B — Usefulness** | **A** | Coordinator skill is a precise runtime spec: phase-by-phase, message-type dispatch table for the chapter pump (`SKILL.md:261-336`), explicit state-tracking table (`SKILL.md:412-428`), error handling, and a clean authority split (`SKILL.md:505-511`). The persona-designer affordance means a user with zero config can still run a panel. Cost-confirmation handshake is enforced before spawning (`SKILL.md:17-22`). Commands route cleanly: `/panel-config` (interactive) vs `/panel-designer` (automated) vs `/panel-review` (execute), each cross-referencing the others. Decision support is concrete throughout (panel-sizing guidance, persona-priority archetypes `process.md:91-101`). |
| **C — Discipline** | **A** | All three agents declare `model:` and load-bearing `tools:` restrictions (`persona-reader.md:1-5` Read/Write only; synthesiser/designer Read/Write/Glob). Contamination constraints are applied asymmetrically and correctly: the designer and synthesiser *may* read full content but are explicitly barred from *predicting behaviour* from it (`persona-designer.md:77-91`, `panel-synthesiser.md:96-103`). Named rationalization resistance is verbatim and emphatic ("The urge to optimise by reading everything first will be strong. Resist it," `process.md:159`; "DO NOT read the whole document first," `process.md:157`; "batch-reading-then-writing is theatre," `process.md:1011`). Calibration honesty is exemplary — the pack tells you simulated personas are approximations and findings are hypotheses (`process.md:1070-1081`). Quality gate + process-checksum manifest make the synthesis auditable (`process.md:768-779,1083-1125`). |
| **D — Form** (gates ceiling) | **B** | Conformant and fully wired: 3 commands ship in `commands/`, 3 slash wrappers present and current at `.claude/commands/`, registered in marketplace, marketplace promises "3 agents … 3 slash commands" and all exist. One **Minor count-drift** defect: `config.md` is verifiably a **6-persona** example (Reference Panel table has 6 rows; corroborated by `config-template.md:5,261` and `SKILL.md:26`), but it is described as a **13-persona** example in 5 places — `ARCHITECTURE.md:157`, in-pack `commands/panel-review.md:20,36`, `commands/panel-config.md:32`, and slash `panel-review.md:18`, `panel-config.md:34`. The "13 personas" in `process.md`/synthesiser ("9 of 13 personas") is fine — that is illustrative arithmetic, not a claim about the example file. The slash wrappers also diverge stylistically from the in-pack commands (in-pack have `allowed-tools`/`argument-hint` frontmatter; wrappers are description-only) but both are functional and current. |

## Gate analysis

1. **Discoverability gate (ceiling):** PASSES. The pack installs, the skill loads (`muna-panel-review:reader-panel-review`), all 3 commands ship in-pack, all 3 slash wrappers exist and are current, and it is registered in `marketplace.json` with accurate counts. The prior review's three Major packaging defects (commands not shipping, stale wrappers, empty `using-panel-review/` directory) are all RESOLVED in v0.4.0 — verified: `commands/` contains all three files, no `using-*` directory exists, marketplace description matches reality. No ceiling cap applies.
2. **Substance-dominates gate:** Substance = A, so overall ≤ A+. Not binding below A.
3. **Honor-roll gate (S):** Substance is A not S (Phase 5 / delegation chain are documented-not-runnable surface area), so S is unreachable. Correctly not an S pack.
4. **Honesty override:** Not triggered — nothing is sold as complete that is vapor; the unrunnable phases are presented as documented methodology, and limitations are stated plainly.

Blend (A/A/A/B-, 40/25/20/15) lands at **A−**. The single Minor count-drift across 5 surfaces is the only thing keeping Form off A and is the lone defect of note.

## Layered per-component grades

The pack is uniformly strong; only the count-drift surfaces and the deferred phases are worth flagging.

| Component | Grade | Note |
|---|---|---|
| `process.md` | A (S-exemplar in parts) | Reference-grade anti-contamination + epistemic-tier methodology; §§2-4 (`process.md:149-161,756-764`) are worth copying into any simulation pack. Only weakness: ships Phase 5 + delegation-chain methodology the orchestration skill does not execute. |
| `agents/panel-synthesiser.md` | A | Clean opus analyst; count-don't-summarise, convergent-reasons test, contamination constraint on gap-suggestion (`:96-103`) all present. Exemplar agent. |
| `commands/panel-review.md` + `commands/panel-config.md` | B− | The "13-persona example" label (`panel-review.md:20,36`; `panel-config.md:32`) misdescribes the 6-persona `config.md`. Cosmetic but user-facing. |
| `ARCHITECTURE.md` | B | `:157` repeats the "Worked 13-persona example" mislabel. |
| `.claude/commands/panel-review.md`, `panel-config.md` | B | Current and functional, but inherit the same "13-persona" mislabel (`panel-review.md:18`, `panel-config.md:34`). |

**S-grade exemplar to copy:** `process.md` Phase 2 cardinal rule + the four-mechanism enforcement model, and the Phase 4 epistemic-tier / convergent-reasons synthesis discipline. This is best-in-class judgment-pack methodology.

## Overall: **A−**

Reconciles with existing verdict system: **Pass + 1 Minor** (down from the prior review's Pass + 3 Major — the packaging defects were fixed in v0.4.0).

**Verdict:** A methodologically reference-grade audience-simulation pack, now properly packaged; the only blemish is a 6-vs-13 persona-count mislabel of the worked example echoed across five surfaces.

**Top finding:** The worked `config.md` is a 6-persona example (verified: 6-row Reference Panel; `config-template.md:5,261`; `SKILL.md:26`) but is called a "13-persona example" in `ARCHITECTURE.md:157`, both in-pack commands, and both slash wrappers — a consistent count drift describing the same file.

**Top fix:** Replace "13-persona" with "6-persona" in `ARCHITECTURE.md:157`, `commands/panel-review.md:20,36`, `commands/panel-config.md:32`, and `.claude/commands/{panel-review.md:18,panel-config.md:34}`. (Leave the illustrative "9 of 13 personas" examples in `process.md`/`panel-synthesiser.md` untouched — those are correct.)
