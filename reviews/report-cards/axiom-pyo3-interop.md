# Report Card — axiom-pyo3-interop

**Version:** 0.1.2 (`plugins/axiom-pyo3-interop/.claude-plugin/plugin.json:3`)
**Track:** H — Hard / Technical (Python ↔ Rust FFI via PyO3)
**Graded:** 2026-06-22 (rubric `reviews/RUBRIC.md`)
**Structure:** router + 13 reference sheets + 3 commands + 1 agent (matches every declared surface)

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** (H lens) | **A** | Technically accurate and current to PyO3 0.25 / 0.24+ idioms. `pyo3-fundamentals.md` uses the correct modern surface: `Bound<'py,T>`/`Py<T>`/`PyRef` (28–32), `unbind`/`bind` (35–36), `Python::attach` as the 0.24+ split (134–141), `IntoPyObject` as the 0.23+ replacement for `IntoPy`/`ToPyObject` (229–248), `#[pyclass(eq,ord,hash,str)]` since 0.22 (190). `gil-release-patterns.md` gives the canonical phased extract→`allow_threads`→return shape (44–60, 153–181), a correct `Send`-closure rationale (42), and the free-threaded 3.13t no-op caveat (260–264). `numpy-buffer-protocol.md` is expert-depth: the lifetime trap matrix (95–108), the subtle "view is `Send` because the buffer is pinned while the GIL is released" argument (41–43, 104–108), the dynamic borrow-tracker aliasing rules (110–119), strided/non-contiguous handling (132–155), dtype mismatch (158–172). Declared 13-topic domain fully covered; only minor depth gaps (no DLPack/PyTorch tensor-sharing sheet, no nested-`#[pymodule]` submodule pattern) — both correctly cross-referenced or scoped out, not holes. |
| **B — Usefulness** | **A** | Router gives 13 symptom→sheet routes each with **Symptoms / Route to / Why** and verbatim-typed trigger phrases (`SKILL.md:154–257`), 5 named multi-sheet workflows (259–281), and a real refusal channel that redirects one-off bindings to ctypes/cffi (39). Sheets ship copy-pasteable ❌/✅ contrasts (`pyo3-fundamentals.md:345–386`, `gil-release-patterns.md:81–103`) and decision tables (`gil-release-patterns.md:64–77` hold-vs-release; `numpy-buffer-protocol.md:176–197` direction + pitfall tables). Reading it changes what you do. |
| **C — Discipline** | **A** | 13-item anti-pattern refusal list, each tagged to the closing sheet (`SKILL.md:285–299`); 10-item Consistency Gate (303–316); Boundary Gate in Start Here (53). Pressure-resistance is operationalized: `gil-release-patterns.md:10` pre-empts the "fast in benchmarks (single-threaded), starves in production (multi-threaded)" rationalization verbatim. Agent `pyo3-reviewer.md` is fully SME-compliant — `model: opus`, cites `meta-sme-protocol:sme-agent-protocol` (10), output template names all four required sections (197–208), and carries a greenfield-design refusal example (46–49). |
| **D — Form** | **A** | Slash wrapper present and current (`.claude/commands/pyo3-interop.md`, sheets/commands/agent all enumerated, cross-refs intact). Registered and accurately described in `.claude-plugin/marketplace.json:261–263`. Counts consistent across plugin.json, wrapper, marketplace, and router (router+13+3+1). All command frontmatter has `description`+`allowed-tools`+`argument-hint`; agent has `description`+`model`. Clean sibling boundaries vs `axiom-rust-engineering` / `axiom-rust-workspaces` stated and non-overlapping (`SKILL.md:34–42, 84–129`). No drift detected. |

---

## Gate analysis

1. **Discoverability gate:** Loads as a skill; slash wrapper present and current; registered and installable. No cap applied.
2. **Substance-dominates gate:** Overall ≤ Substance + 1 tier = ≤ S. Not binding.
3. **Honor-roll (S) gate:** Substance is **A**, not S — no single sheet is reference-grade-teaches-the-entire-domain, and two minor depth gaps exist. S withheld. No subject below A; zero Major+ defects.
4. **Honesty override:** N/A — fully built, not a scaffold; description claims match delivered content.

**OVERALL: A**

Reconciles with the prior `reviews/axiom-pyo3-interop.md` (2026-05-22) verdict of **Pass** (0 Critical, 0 Major, 2 Minor, 3 Polish). Version unchanged at 0.1.2; fresh reading confirms the prior review's structural conclusions and independently verifies API correctness on the highest-risk sheets.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Surfaced for reference:

| Component | Grade | Note |
|-----------|-------|------|
| `numpy-buffer-protocol.md` | **A+** (exemplar) | Reference-grade boundary teaching: the lifetime trap matrix + `Send`-under-`allow_threads` rationale + borrow-tracker aliasing rules + strided/dtype handling are exactly the use-after-free class PyO3 cannot prevent. Worth copying as the model for "teach the subtle invariant, not the happy path." |
| `gil-release-patterns.md` | **A** | Phased pattern, hold-vs-release table, named rationalization pre-empt, free-threaded caveat. |
| `pyo3-fundamentals.md` | **A−** | Correct and current; the one place a future-currency risk lives (pinned `0.25` examples) — acceptable given an explicit baseline block (16–20). |

**Polish-only backlog (not blocking):** add DLPack/PyTorch tensor-sharing discoverability (cross-ref or small sheet), nested-`#[pymodule]` submodule note in fundamentals, "free-threaded / no-GIL" as a `Routing by Symptom` trigger, and optionally trim `/audit-gil-discipline` `allowed-tools` to drop `Write`/`Edit`.

---

## Verdict

Reference-quality FFI-boundary pack — accurate, current, disciplined, fully wired; ship with pride.

**Top finding:** Substance is genuinely expert across the declared 13-topic domain; the numpy and GIL sheets teach the *why* of the subtle boundary invariants (lifetime pinning under `allow_threads`, phased GIL discipline) rather than the happy path — `numpy-buffer-protocol.md:95–119` is the exemplar.

**Top fix:** Make PyTorch/DLPack tensor-sharing discoverable (expand the `yzmir-pytorch-engineering` cross-ref at `SKILL.md:325` to name `__dlpack__`, or add a small sheet) — the one common production boundary shape not surfaced; everything else is polish.
