# Report Card — axiom-rust-workspaces

**Version:** 1.0.2 (`plugins/axiom-rust-workspaces/.claude-plugin/plugin.json:3`)
**Track:** H — Hard / Technical (cargo workspace mechanics; correctness = APIs/semantics accurate, would build; currency = current toolchains)
**Graded:** 2026-06-22
**Prior evidence:** `reviews/axiom-rust-workspaces.md` (2026-05-22, also v1.0.2 — NOT stale; pack unchanged since). Prior verdict: PASS, no fixes. This grade concurs with a tighter Form note.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** | **A** | Technically accurate and current throughout. Resolver semantics correct and current: resolver-1/2/3 distinctions, resolver-3 MSRV-aware selection stabilised Rust 1.84 (`workspace-dependencies-and-resolver.md:120-184`); `dep:` prefix landed cargo 1.60 (`feature-unification-gotchas.md:213`). Feature-unification model is expert-grade, not tutorial: the "default-features is a unanimous vote" gotcha (`feature-unification-gotchas.md:35-70`), same-crate dev-dep contamination that resolver-2 does *not* isolate (`feature-unification-gotchas.md:140-173`), and `-p` being a target-reduction not a graph-reduction (`feature-unification-gotchas.md:175-195`) are the exact corners practitioners get wrong. The split-version `E0308` "two copies of serde" diagnostic (`workspace-dependencies-and-resolver.md:90-98`) is pathognomonic and correct. Coverage complete vs declared domain (13 sheets across spine + operational). Depth gap only: no `[patch.crates-io]` treatment (flagged in prior review's Polish backlog). |
| **B — Usefulness** | **A** | Router routes crisply: four named scenarios with ordered sheet sequences (`SKILL.md:189-219`), a decision tree (`SKILL.md:281-309`), stop conditions (`SKILL.md:270-277`), and a tier model (XS–XL) that makes artifact requirements concrete (`SKILL.md:177-185`). Sheets are command-level actionable — every diagnosis is a copy-pasteable `cargo tree -e features` / `cargo tree --duplicates` / grep invocation with what to look for (e.g. `feature-unification-gotchas.md:284-307`). "Common Mistakes" symptom→fix tables on every sheet. Reading it changes what you do. |
| **C — Discipline** | **A** | Anti-patterns are a full operationalised refusal list (10 patterns, each with symptom/diagnosis/remediation/why/prevention — `workspace-anti-patterns.md`). Pressure-resistance named verbatim: "We'll Consolidate Later" trap (`workspace-anti-patterns.md:205-215`), "Do **not** stay on resolver-1 to avoid the work" (`workspace-dependencies-and-resolver.md:198`). Consistency Gate of 12 checks with explicit-waiver-or-fail discipline (`SKILL.md:233-252`). Agent carries full SME protocol: `model: opus`, Confidence/Risk/Information-Gaps/Caveats mandated, input contract, "necessary not sufficient" honesty caveat (`agents/workspace-reviewer.md:209-216, 257-259`). |
| **D — Form** | **A−** | Conformant and fully wired: slash wrapper `/rust-workspaces` present and current (lists 13 sheets, 3 commands, 1 agent — matches reality); marketplace entry accurate and detailed; all 3 commands carry `description`/`allowed-tools`/`argument-hint`; agent carries `model:`. Description is a strong "Use when…" (`SKILL.md:3`). Two minor internal-drift nits (no surface drift): (1) stale cross-ref `*Planned for v0.2.0:* feature-unification-gotchas.md` in `workspace-dependencies-and-resolver.md:284` — that sheet now ships; (2) router/plugin.json version-label mismatch — router body says "v0.2.0 is feature-complete" (`SKILL.md:417`, also `:113-116`) while plugin.json is `1.0.2`. Cosmetic; does not affect discoverability or routing. |

---

## Gate analysis

1. **Discoverability gate:** PASS — installs, router loads, slash wrapper present + current, registered in marketplace, all commands/agent wired. No ceiling.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+. Not binding below.
3. **Honor-roll gate (S):** Substance is A, not S (small depth gap: no `[patch.crates-io]`; resolver content is reference-grade but the pack as a whole is "complete + correct + current" rather than "teaches the entire discipline at S depth with nothing missing"). Form is A− (two cosmetic drift nits). S not reached.
4. **Honesty override:** N/A — feature-complete, no scaffold.

Blend (40/25/20/15) of A/A/A/A− lands at **A**. No Major defects; the Form nits are below Minor.

---

## Layered per-component grades

Pack is uniformly strong; no weak tail. Exemplars and the only blemish:

| Component | Grade | Note |
|-----------|-------|------|
| `feature-unification-gotchas.md` | **S** | Exemplar worth copying. The 80/20 framing ("`02-` is the 80%, this is the 20%"), seven named gotchas each with the exact `cargo tree` diagnostic and structural fix, and correctly distinguishing what resolver-2 does *not* fix. Reference-grade depth on the single hardest topic in the domain. |
| `agents/workspace-reviewer.md` | **A** | Full SME protocol, `model: opus`, input contract, cost-of-postponing severity model, activate/don't-activate examples, honesty caveats. Template-quality agent. |
| `workspace-dependencies-and-resolver.md` | **A−** | Content reference-grade; only the stale `*Planned for v0.2.0:*` self-cross-ref (`:284`) to a sheet that now exists drops it from S. |

---

## Overall: **A**

## Verdict
A reference-quality multi-crate Rust pack: correct, current, deeply actionable, fully wired — held off S only by a small `[patch.crates-io]` depth gap and two cosmetic version-label drift nits.

## Top finding
Substance and Discipline are both top-tier — the feature-unification sheet is an S-grade exemplar (the corners resolver-2 does *not* fix, each with a runnable `cargo tree -e features` diagnostic), and the SME-compliant `workspace-reviewer` agent is template quality.

## Top fix
Clear the two cosmetic drift nits: delete the obsolete `*Planned for v0.2.0:*` cross-ref at `workspace-dependencies-and-resolver.md:284`, and reconcile the router's "v0.2.0 is feature-complete" labels (`SKILL.md:113-116, 417`) with the actual `1.0.2` version. Optionally add a `[patch.crates-io]` paragraph to `02-` to close the one depth gap.
