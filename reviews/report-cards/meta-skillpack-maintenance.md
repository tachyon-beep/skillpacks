# Report Card — meta-skillpack-maintenance

**Version:** 2.1.0 · **Track:** P (Process / Hybrid)
**Unit:** pack (router + 4 reference sheets; 0 commands, 0 agents, 0 hooks)
**Graded:** 2026-06-22 · cross-checked against prior review `reviews/meta-skillpack-maintenance.md` (dated 2026-05-22, version matches — not stale)

This pack is the marketplace's self-maintenance methodology, so it is graded by its own kind of criteria. Where it breaks its own documented rule, that is recorded as a Form defect, not excused as meta-irony.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** (P-track: methodology valid, maturity-appropriate, current) | **A−** | Five-stage workflow (Investigate → Scorecard → Test → Discuss → Execute) is valid and well-sequenced (`SKILL.md:54-119`). Methodology is genuinely marketplace-aware, not generic: observed frontmatter conventions with sample counts ("~60/65 agents declare only `description` and `model`", `SKILL.md:25,170-171`), the quoted-array `allowed-tools` convention (`SKILL.md:151`), SME-protocol body convention (`SKILL.md:174-184`), model-selection guide (`reviewing-pack-structure.md:101-109`), and version-bump rules (`SKILL.md:242-250`). Currency is good (references `meta-sme-protocol`, current co-author convention with a caveat to re-check, `implementing-fixes.md:240-245`). Held below A by two real gaps: (1) an **undeclared hard dependency** on `superpowers:writing-skills` — invoked as the mandatory path for every new skill (`SKILL.md:110-117`, `implementing-fixes.md:17-37`) but that skill is external, not in this marketplace, and the pack offers no fallback if it is absent; (2) no bootstrapping/self-application guidance despite being the one pack guaranteed to be maintained by itself. |
| **B — Usefulness / Actionability** | **A** | Consistently concrete. Copy-pasteable bash inventory commands (`analyzing-pack-domain.md:50-110`), full output templates for every stage (scorecard `reviewing-pack-structure.md:174-237`, test report `testing-skill-quality.md:179-232`, completion summary `implementing-fixes.md:268-304`), decision tables (component-type selection `analyzing-pack-domain.md:202-209`, duplicate handling `reviewing-pack-structure.md:66-72`, version bump `SKILL.md:244-249`), and concrete test-runner mechanics — subagent dispatch with an example prompt frame and a fidelity ranking (`testing-skill-quality.md:80-92`). Reading it changes what you do. |
| **C — Discipline / Robustness** | **A** | Three distinct rationalization tables name verbatim excuses and rebut them: structure-review red flags (`reviewing-pack-structure.md:243-250`: "I know this domain, no audit needed" → "Expertise ≠ systematic review"), testing red flags (`testing-skill-quality.md:240-247`), execution red flags (`implementing-fixes.md:253-261`). Anti-pattern table at `implementing-fixes.md:310-316`. Pressure-resistance is the explicit subject of the whole testing sheet (gauntlet categories A/C/B, `testing-skill-quality.md:25-60`). Falsifiability is operationalised via the bundled `.test-fixtures/flawed-plugin/` (vague description, duplicate `data-stuff`/`data-things` skills, command lacking `allowed-tools`) — a self-test corpus most packs lack. SME-protocol enforcement is taught for agents but the pack ships no agents, so that clause is N/A here. |
| **D — Form / Integrity** (gates the ceiling) | **C** | Conformant frontmatter, registered in marketplace (`marketplace.json:445-446`), router skill is model-invocable and discoverable ("Use when…", `SKILL.md:3`). But **one Major**: no slash wrapper. `ls .claude/commands/` (44 wrappers) contains none named `skillpack-maintenance.md`, `meta-skillpack-maintenance.md`, or similar — while the pack's own `reviewing-pack-structure.md:131,138` mandates "Every router skill … has a corresponding `.claude/commands/<name>.md` wrapper" and flags its absence as a fix. The pack violates the exact rule it enforces. Secondary smell: `.test-fixtures/flawed-plugin/` is git-tracked inside the distributable plugin directory (5 files) and is undocumented from the `SKILL.md` entry point — defensible as a test corpus, but it ships to users and is not pointed to. |

---

## Gate analysis

1. **Discoverability gate (ceiling):** Router SKILL.md loads and is model-discoverable, so this is not an F. But a required wiring surface is broken — the slash wrapper is missing — which the rubric explicitly names as a cap: "Loadable but a required wiring surface broken (missing slash wrapper …) → overall capped at **C**." **This is the binding constraint.**
2. **Substance-dominates gate:** overall ≤ Substance (A−) + 1 = S-range. Not binding.
3. **Honor-roll (S):** fails — Form is C and there is a Major defect. Not S.
4. **Honesty override:** not a scaffold; full content delivered. N/A.

The content (A−/A/A) would otherwise land this pack at **A−**. Gate 1 pulls it down to **C+**: the substance and discipline are genuinely strong, but a maintenance pack that fails its own wiring check cannot be invoked the way the marketplace invokes every sibling router.

---

## Layered: per-component grades

| Component | Grade | Note |
|-----------|-------|------|
| `SKILL.md` (router) | A− | Strong, marketplace-specific guidance; undeclared `superpowers:writing-skills` dependency and no bootstrapping note. |
| `analyzing-pack-domain.md` | A | Thorough inventory + coverage-map method; concrete commands; includes the wrapper check it then fails to satisfy (`:104-110`). |
| `reviewing-pack-structure.md` | A | Best sheet for discipline; the router/wrapper-alignment rule (`:128-142`) is exactly what indicts the pack — exemplary self-aware criterion. |
| `testing-skill-quality.md` | A | Concrete test-runner mechanics + gauntlet; small gap — gauntlet labels A/C/B are presented out of order (`:25-60`), cosmetic. |
| `implementing-fixes.md` | A− | Solid execution + versioning; co-author block hardcodes "Opus 4.7" with a caveat to re-check (`:240-245`) — minor staleness risk. |
| **Form surface (wrapper)** | **C** | Missing `.claude/commands/` wrapper — the Major that caps the pack. |
| `.test-fixtures/` packaging | C+ | Ships inside distributable; undocumented from entry point; serves a real testing purpose. |

**S-grade exemplar worth copying:** `reviewing-pack-structure.md:128-142` — the Router/Slash-Command Alignment subsection. A reusable, evidence-checkable wiring audit (router-without-wrapper, wrapper-without-router, missing/renamed marketplace entry) that every router pack should be measured against. The irony that this pack fails it does not diminish the criterion's quality.

---

## Overall: **C+**

Reconciles with the prior review's **Major** verdict and with the existing `reviews/` system's single-Major band.

**Verdict:** Reference-grade maintenance methodology with strong discipline and actionability, capped at C by the one defect it teaches others to catch — its own missing slash-command wrapper.

**Top finding:** The router skill has no `.claude/commands/skillpack-maintenance.md` wrapper, violating the pack's own rule at `reviewing-pack-structure.md:131,138`; every other router pack in the marketplace has one. Per Gate 1 this caps the overall at C regardless of the A-grade content.

**Top fix:** Add `.claude/commands/skillpack-maintenance.md` following the established wrapper pattern (a patch bump). This single change removes the gate cap and lifts the pack to ~A−. Secondary: declare the `superpowers:writing-skills` external dependency (with a fallback note) and either document `.test-fixtures/` from the SKILL.md or exclude it from the distributable.
