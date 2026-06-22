# Report Card — axiom-determinism-and-replay

**Version:** 1.0.3 (`plugins/axiom-determinism-and-replay/.claude-plugin/plugin.json:3`)
**Track:** H — Hard / Technical (correctness = claims technically accurate, would run, no wrong APIs; currency = pinned toolchains)
**Date:** 2026-06-22
**Prior evidence:** `reviews/axiom-determinism-and-replay.md` (2026-05-22) graded the *same* version 1.0.3 "Pass." Cross-checked: version unchanged, so the prior review is current — but it scored "Cross-skill linkage = Pass" and missed the stale `planned`/`v0.1.0` sibling references this fresh reading found (see Form). I weight my own reading on that point.

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A. Substance** (40%) | **S−** | All sampled technical claims correct and current. `gpu-determinism.md`: `CUBLAS_WORKSPACE_CONFIG` required since CUDA 10.2 (L184), TF32 Ampere default + per-PyTorch-release drift (L50), `NCCL_PROTO=LL/LL128` use atomics so pin `Simple` (L152), atomic-float audit table with deterministic substitutes (L132-142), `use_deterministic_algorithms(True, warn_only=False)` raise-don't-catch discipline (L99). `floating-point-determinism.md`: six sources (non-associativity, FMA, SIMD horizontal sums, multi-thread reductions, libm divergence glibc/musl/Accelerate, FTZ/DAZ), Kahan summation, quantised-hash policy for logical-equivalence (L160-171). `seed-governance.md`: content-addressable `derive_seed(master, name)` via blake2b, three named sub-seed anti-patterns (counter / `id()`/ASLR / child-from-parent-RNG, L107-130). Full coverage across the 13 declared channels at expert depth; teaches the *why* throughout. Held off clean S only by vestigial "v0.1.0" content phrasing (below). |
| **B. Usefulness** (25%) | **A** | Router routes crisply: 5 named scenarios (`SKILL.md:194-230`), decision tree (L301-333), tier→required-artifact table (L182-188), quick-reference need→sheet table (L383-403). Every sheet ends in a numbered Spec Output checklist bound to a specific gate check ("Without these N items … Check 13 will fail", `gpu-determinism.md:209`; `floating-point-determinism.md:204`; `seed-governance.md:214`). Reading it changes what you do, not just what you know. |
| **C. Discipline** (20%) | **A** | Named rationalizations held verbatim in the 17-check Consistency Gate (`SKILL.md:244-269`): "We just diff the logs" fails (Check 6), "fully replayable without proof of rewind" fails (Check 7), "Mostly deterministic on N cores" fails (Check 11), "mocks-in-production fails" (Check 14). Stop Conditions (L290-297) and the `cost-of-determinism.md` sheet pre-empt "always do all of this." Both agents fully SME-protocol compliant: `model: opus`, `meta-sme-protocol:sme-agent-protocol` cited, and the four required sections (Confidence / Risk / Information Gaps / Caveats) present (`determinism-reviewer.md:184-205`). |
| **D. Form** (15%) | **B−** | Frontmatter conformant; slash wrapper `/home/john/skillpacks/.claude/commands/determinism-and-replay.md` present, current, scope-matched; registered (`marketplace.json:180`); counts match declared 13 sheets / 3 commands / 2 agents. **Defect (D3 consistency):** six sheets describe *already-shipping* siblings as "planned" — `seed-governance.md:150` ("planned `07-concurrency-determinism-spec.md`"), `replay-infrastructure-design.md:173` ("planned `10-...`"), `rng-isolation-patterns.md:193-194,207` ("planned `gpu-determinism.md`", "planned `11-...`"), `determinism-vs-reproducibility.md:48-49,174`, `snapshot-strategy.md:27,43,103` (12 instances total) plus 3 "v0.1.0 spec/tier/systems" content phrases (`cost-of-determinism.md:19`, `snapshot-strategy.md:109`, `seed-governance.md:150`). All cross-refs resolve (the files exist); only the qualifier is false — leftover from the v0.1.0 → v1.0.3 promotion. |

## Gate analysis

1. **Discoverability gate:** PASS — installs, router loads, slash wrapper present and current, registered and marketed accurately. No ceiling.
2. **Substance-dominates gate:** Substance = S− → overall ≤ S. Not binding.
3. **Honor-roll (S) gate:** Requires no subject below A and zero Major+ defects. Form = B− bars S.
4. **Honesty override:** N/A — pack is complete, not a scaffold; marketing matches reality.

Blend: Substance S− (40%), Usefulness A (25%), Discipline A (20%), Form B− (15%) → high-A pack pulled off clean A by the Form consistency defect. **Overall A−.** Reconciles with prior "Pass" plus one previously-unspotted Minor.

## Layered per-component grades

| Component | Grade | Note |
|-----------|-------|------|
| `gpu-determinism.md` | **S** | Exemplar worth copying: correct, current, atomic-float audit table, per-framework knobs (PyTorch/TF/JAX), cross-device honesty, 10-item Spec Output tied to Check 13. |
| `seed-governance.md` | **A−** | Reference-grade content (content-addressable derivation, 3 named anti-patterns) but carries the worst stale line: L150 still calls the shipping concurrency sheet "planned" and says "covers the v0.1.0-relevant patterns." |
| `rng-isolation-patterns.md` | **B+** | Strong content; 3 stale "planned" refs to shipping `gpu-determinism.md` / `11-canonical-state-encoding.md` (L193-194, 207). |
| `snapshot-strategy.md` | **B+** | Solid; 3 stale "planned `external-effects-substitution.md`/`11-`" refs + one "v0.1.0-tier" phrase (L27,43,103,109). |
| `determinism-vs-reproducibility.md` | **B+** | Foundational vocabulary sheet; 3 stale "planned" sibling refs (L48-49,174). |
| `SKILL.md` (router) | **A** | Dense but disciplined: dependency graph + three coordinated re-emission tables, tier model, 17-check gate, clean sibling boundaries (simulation-foundations / audit-pipelines). Correctly lists all 13 sheets as shipping — i.e., the router is *consistent* and the sheets are the ones that drifted. |
| Both agents | **A** | SME-protocol clean; severity rubrics, failure-mode classification tables, self-discipline sections. |

## Overall: **A−**

**Verdict:** A reference-grade hard-technical pack whose only blemish is six sheets still calling already-shipping siblings "planned" — a v0.1.0 → v1.0.3 promotion left vestigial.

**Top finding:** Twelve "planned `<sheet>`" cross-references (plus three "v0.1.0" content phrases) across six sheets describe siblings that now ship in this same pack; the router lists all 13 as present, so the sheets — not the router — carry stale drift the 2026-05-22 review scored as "Cross-skill linkage = Pass."

**Top fix:** Search-and-replace the word "planned " before sheet/artifact names in the six sheets (`seed-governance.md:150`, `replay-infrastructure-design.md:173`, `rng-isolation-patterns.md:193-194,207`, `determinism-vs-reproducibility.md:48-49,174`, `snapshot-strategy.md:27,43,103`) and rephrase the three "v0.1.0" content references — closes the Form defect and lifts the pack to a clean A.
