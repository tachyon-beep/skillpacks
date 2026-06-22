# Report Card — axiom-rust-engineering

**Version:** 1.1.0 (`plugins/axiom-rust-engineering/.claude-plugin/plugin.json`)
**Track:** H — Hard / Technical
**Graded:** 2026-06-22 (rubric `reviews/RUBRIC.md`)
**Prior evidence:** `reviews/axiom-rust-engineering.md` (2026-05-22, v1.0.2) — STALE. Both Major findings from that review have been fixed in v1.1.0; this grade weights fresh reading. The rubric's own worked example (`RUBRIC.md:134`) predates these fixes.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** (40%) | **S−** | Reference-grade, current, expert-depth. `unsafe-ffi-and-low-level.md` covers the five `unsafe` superpowers, soundness vs safety, the aliasing model (Stacked/Tree Borrows), strict provenance with the **stable 1.84 API** (`addr`/`with_addr`/`expose_provenance`, `:437-496`), `&raw const` (stable 1.82, `:498-526`), edition-2024 `unsafe extern` / `#[unsafe(no_mangle)]` (`:542-672`), allocator-match double-free hazard (`:776-784`), Box ownership round-trips, no_std. `ai-ml-and-interop.md` pins PyO3 **0.25 `Bound<'py,T>`** with a precise 0.23→0.25 migration ledger (`:20-43`), GIL-release discipline, the dangling-borrow-across-`allow_threads` SAFETY rule (`:373-402`), candle/burn/tch decision table, safetensors-vs-pickle RCE. Every sheet carries a version baseline (1.87 / 2024 edition) and honest "verify against crates.io" caveats for fast-moving crates. Why-not-just-S: a single boundary topic (workspaces) is deliberately *delegated* rather than covered, which is correct scoping but means the pack is intentionally not exhaustive of "all Rust." |
| **B — Usefulness** (25%) | **S−** | Router is among the most disciplined in the marketplace: 13 symptom→sheet routing tables with error codes (E0502/E0597/E0382/E0277), ambiguity-resolution priority rules (`SKILL.md:328-355`), cross-cutting scenarios with load-order (`:361-388`), a "Common Routing Mistakes" wrong/right table (`:391-403`), and a 6-question pre-answer self-check (`:434-460`). Sheets are decision-first: candle-vs-burn table, ndarray-vs-nalgebra table, PyO3 binding-approach decision tree. Concrete WRONG/CORRECT pairs throughout. |
| **C — Discipline** (20%) | **S−** | Names rationalizations verbatim — "I'll just `.clone()`", "One clippy `allow` won't hurt", "unsafe is fine here, I checked" (`SKILL.md:419-431`); refuses `#![allow(clippy::all)]` in three places in `systematic-delinting.md` (`:29,:701,:809`); self-check tells the model to route *under pressure* (`SKILL.md:457-459`). All three agents declare the SME Agent Protocol with `model: sonnet` and mandate Confidence/Risk/Information-Gaps/Caveats output (`agents/*.md` frontmatter + body). `unsafe-auditor.md` mandates per-block soundness verdicts. |
| **D — Form** (15%) | **A** | Conformant frontmatter (router `name`+`description`, commands quoted `allowed-tools`/`argument-hint`, agents `description`+`model`). Slash wrapper **present and current** (`.claude/commands/rust-engineering.md`, sibling-aware, accurate 11-sheet/5-command/3-agent list). Registered in marketplace with matching counts. Zero count drift (router + 11 sheets is stated and true). Clean sibling boundaries to `/rust-workspaces` and `/pyo3-interop`. Only nit: empty `hooks/hooks.json` placeholder (`{ "hooks": {} }`). |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, router loads, slash wrapper present and current, registered, no scaffold-as-complete. No cap applied. (This is the gate that capped the pack at the rubric's worked-example time — the missing wrapper is now shipped.)
2. **Substance-dominates gate:** Substance = S−, so overall ≤ A+/S−. Not binding below S−.
3. **Honor-roll (S) gate:** Requires Substance = S, no subject below A, zero Major+. Substance is S− (not full S) and Form is A (one cosmetic nit). Falls just short of straight S.
4. **Honesty override:** N/A — no scaffold; content matches marketing exactly.

**Overall: A** (high A / A+, one notch under S). The v1.1.0 fixes lifted this from the prior **Major** verdict to ship-with-pride.

---

## Layered per-component grades

The pack is uniformly strong; there is no weak tail. Surfacing exemplars and the lone nit:

| Component | Grade | Note |
|-----------|-------|------|
| `unsafe-ffi-and-low-level.md` | **S** | Exemplar worth copying: strict-provenance stable-API treatment, edition-2024 unsafe-attribute migration, allocator-match double-free hazard, miri limitations (no FFI, weak-memory) honestly stated. Best-in-class Track-H safety sheet. |
| `using-rust-engineering/SKILL.md` (router) | **S** | Exemplar router: symptom tables + ambiguity priority + cross-cutting load-order + pressure-resistant self-check + verbatim rationalizations. |
| `ai-ml-and-interop.md` | **A** | Current (PyO3 0.25, candle 0.11, burn 0.17), exemplary `/pyo3-interop` redirect (`:313-322`); fast-moving-crate currency risk acknowledged rather than hidden. |
| `project-structure-and-tooling.md` | **A** | Previously the boundary-leak offender; v1.1.0 removed ~130 lines of duplicated workspace material and replaced with an explicit `/rust-workspaces` boundary block (`:163-181`) plus a changelog note. Clean fix. |
| `hooks/hooks.json` | **(nit)** | Empty `{ "hooks": {} }` placeholder — delete or document. Cosmetic only; does not move Form below A. |

---

## Verdict

Reference-grade single-crate Rust pack: correct, current to 1.87/2024-edition, expert-depth, fully wired, with the cleanest sibling-boundary discipline in the rust family.

**Top finding:** Both Major issues from the May-2026 review (missing `/rust-engineering` slash wrapper; workspace boundary leak in `project-structure-and-tooling.md`) are resolved in v1.1.0 — the slash wrapper now ships and the workspace material was excised in favour of an explicit `/rust-workspaces` redirect. The pack now passes the discoverability gate it previously failed.

**Top fix:** Delete the empty `plugins/axiom-rust-engineering/hooks/hooks.json` placeholder (or document its intent) — the only remaining blemish standing between Form-A and Form-S, and the last thing keeping the pack off straight S.
