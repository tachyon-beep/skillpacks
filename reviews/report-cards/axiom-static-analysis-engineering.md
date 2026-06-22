# Report Card — axiom-static-analysis-engineering

**Version:** 0.3.0 (plugin.json) — see Form finding: a bare version bump; all content/surfaces still say v0.2.0
**Track:** H — Hard / Technical (analyzer-engine construction: lattice algebra, fixed-point inference, callgraph, SARIF)
**Graded:** 2026-06-22 · Layered (pack + components)

Prior review (`reviews/axiom-static-analysis-engineering.md`, 2026-05-22, v0.2.1) flagged a **stale v0.1 slash wrapper** as a Major. That is **FIXED** — the current wrapper at `.claude/commands/static-analysis-engineering.md` covers all 13 sheets, 3 commands, 2 agents. This grading weights the fresh reading; the old Major no longer applies.

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** | **S−** | Genuinely expert across the declared domain. `taint-lattice-design.md:37-45` states the three termination properties (monotonic transfer functions, finite ascending chains / widening, decidable order); `:124-135` builds the **product lattice** for orthogonal properties and names the flat-enum anti-pattern; `:137-145` gives a **Rice-theorem-aware** soundness/completeness statement ("an analyzer claiming both sound and complete on a non-trivial property is wrong"). `:69-98` is a worked three-tier web taint lattice where joining two different sanitisations correctly **downgrades to Untrusted** (sink-class parameterisation). `callgraph-construction.md` ladders resolution rungs (name/CHA/RTA/VTA/k-CFA) with a conservative-`top` floor; `llm-assisted-rule-explanation.md:10-25` correctly fences the LLM as a translator-never-decider with a prompt-injection threat model; `sarif-emission-and-ci-integration.md` pins SARIF 2.1.0 with exit-code semantics. Current, correct, teaches the *why*. Held off straight-S only by depth being concentrated in the spine sheets (168–185 lines) rather than uniformly. |
| **B — Usefulness** | **A** | Router (`SKILL.md`) routes by four named scenarios (`:174-201`), a decision tree (`:259-283`), five stop conditions (`:249-257`), a tier model that *gates which artifacts are required* (`:158-170`), an 11-check consistency gate (`:214-232`), and a quick-reference dispatch table (`:352-376`). The spec dependency graph + coordinated re-emission table (`:121-156`) tell you exactly what to re-emit when the lattice changes. Reading it changes what you build. |
| **C — Discipline** | **A** | Named rationalisations and refusals: "boolean lattice masquerading as types" recurs as the signature anti-pattern; `taint-lattice-design.md:147-158` Common Mistakes table; both agents carry the **full SME Agent Protocol** (`rule-designer.md:10`, `false-positive-analyst.md:10` — Confidence/Risk/Information Gaps/Caveats mandated) with `model: opus` and explicit refusal tables (`rule-designer.md:226-236` refuses tier-inventions, runtime-value rules, example-less rules). FP-rate budget *requires a number* (gate check 8). Honest tier model — promotion not waiver. |
| **D — Form** | **B−** | Conformant frontmatter, clean file layout, registered in `.claude-plugin/marketplace.json`, current slash wrapper, clean sibling boundaries (archaeologist/audit/security cross-refs are crisp, not overlapping). **One real consistency Major**: `plugin.json` says `version: 0.3.0`, but SKILL.md says "Shipped in v0.2.0" / "(v0.2.0)" throughout (`:106-119`), the marketplace description says "v0.2 ships router + 13 sheets", and the git history's last feature commit is `40ad4e9` "v0.2.0". No v0.3 content exists — the version was bumped without a release. Surfaces disagree on the pack's own version. |

## Gate analysis

1. **Discoverability gate (ceiling):** PASS. Router loads, slash wrapper present and current, registered in marketplace, all 13 sheets + 3 commands + 2 agents resolve. No cap.
2. **Substance-dominates gate:** Substance = S− → overall ceiling is S. Not binding.
3. **Honor-roll gate (S):** FAILS — the v0.3.0/v0.2.0 version drift is a Major-class consistency defect, and Form = B−. So overall cannot be S.
4. **Honesty override:** N/A — not a scaffold; content fully delivered against the declared 13-sheet domain.

## Layered — per-component grades

The pack is uniformly strong; only the version drift and the relatively thinner sheets are worth surfacing.

| Component | Grade | Note |
|---|---|---|
| `plugin.json` vs SKILL.md / marketplace.json | **C** | Version drift: plugin.json 0.3.0 vs every other surface (and content) at 0.2.0; bare bump with no v0.3 release. The pack's worst offender. |
| `ast-visitation-patterns.md` (168L) | **B+** | Soundest of the spine but the shortest; visitor/walker/transformer choice criteria are good, depth is adequate not exhaustive. |
| **Exemplar:** `taint-lattice-design.md` | **S** | Reference-grade: lattice algebra, product domains, sink-class-parameterised sanitisers, Rice-aware soundness statement, worked join table, extension-as-lattice-breaking discipline. Copy this as the template for "how a Track-H sheet teaches the why." |

## Overall: **A−**

Substance is reference-grade and the discipline signature (named anti-patterns, SME-compliant agents with refusal tables, numeric FP budget, gated tier model) is fully realised. A clean A is blocked only by the self-inconsistent version (plugin.json 0.3.0 while content, marketplace, wrapper, and git all say 0.2.0) — a single, trivially-closed consistency Major. Reconciles with existing-verdict **Minor**.

**Verdict:** A reference-grade analyzer-construction pack let down only by a version number that lies about itself.

**Top finding:** `plugin.json` declares `version: 0.3.0`, but SKILL.md (`:106-119`), the marketplace description, and the git history all say v0.2.0 with no v0.3 content — a bare version bump with no release behind it.

**Top fix:** Either revert `plugin.json` to `0.2.1`/`0.2.2`, or do a real v0.3.0 — update the SKILL.md "Shipped in v0.2.0" headers and the marketplace.json "v0.2 ships" string to match, and note what 0.3.0 actually adds.
