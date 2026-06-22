# Skillpack Report Card — F→S Rubric

A grading instrument for the Skillpacks Marketplace. Produces an **F→S** letter grade
per skillpack (and, layered, per reference sheet for the worst offenders).

Designed to **reconcile with** the existing `reviews/` system (Pass / Minor / Major /
Critical) while giving finer granularity, and to flex across the marketplace's full
span of **hard / verifiable** skills (rust, pyo3, determinism, pytorch) and
**soft / judgment** skills (creative-writing, ux, product/program-management).

---

## 1. The scale

Six tiers, best to worst. Optional `+`/`−` modifiers for finer granularity. "E" is folded into D/F.

| Tier | Name | Spirit | Existing-verdict analogue |
|------|------|--------|---------------------------|
| **S** | Reference-grade | Best-in-class. The pack you'd hold up as the *template* others copy. Defect-free **and** actively teaches the discipline. Rare. | Pass (no fixes) + exemplary |
| **A** | Ship with pride | Production-ready, complete, disciplined. Polish-only backlog. | Pass / Pass+Polish |
| **B** | Solid, minor fixes | Fundamentally sound; a few Minor issues or one easily-closed gap. The healthy default. | Minor |
| **C** | Usable but flawed | Works, but a real defect — one Major, a coverage hole, drift, or weak discipline. | Major (single) |
| **D** | Deficient | Multiple Majors or a structural problem; under-delivers against its own promise. | Major (multiple) |
| **F** | Failing | Doesn't deliver its stated purpose: broken, undiscoverable, scaffold-sold-as-complete, or wrong/dangerous content. | Critical / Scaffold |

---

## 2. Structure: 4 graded subjects + a domain track

Every pack gets a letter in **four subjects**, then an overall adjusted by gates.
Only **Subject A (Substance)** changes meaning by domain — read it through one of three
**tracks**. Subjects B–D are domain-neutral.

```
OVERALL = gated blend of:
  A. Substance      40%   ← read through the domain TRACK (H / S / P)
  B. Usefulness     25%   ← domain-neutral
  C. Discipline     20%   ← domain-neutral
  D. Form           15%   ← domain-neutral, but GATES the ceiling
```

---

## 3. Domain tracks (how Subject A is read)

| Track | Typical packs | "Correctness" means | "Currency" means |
|-------|--------------|---------------------|------------------|
| **H — Hard / Technical** | rust-engineering, rust-workspaces, pyo3-interop, python-engineering, web-backend, embedded-database, determinism-and-replay, static-analysis-engineering, mcp-engineering, simulation-tactics, all yzmir AI/ML packs (pytorch, training-optimization, deep-rl, neural-architectures, llm-specialist, ml-production, simulation-foundations, dynamic-architectures, morphogenetic-rl) | Claims technically accurate; code sound; would compile/run; no wrong APIs | Pinned to current toolchains/versions |
| **S — Soft / Judgment** | creative-writing, ux-designer, site-designer, tui-designer, product-management, program-management, systems-thinking, systems-as-experience, technical-writer, wiki-management, panel-review, document-designer | Judgment defensible and not misleading; framing right; no platitudes-as-advice | Reflects current practice/standards |
| **P — Process / Hybrid** | sdlc-engineering, planning, engineering-foundations, procedural-architecture, devops-engineering, audit-pipelines, security-architect, quality-engineering, solution-architect, system-architect, system-archaeologist, ai-engineering-expert (router), meta-sme-protocol, meta-skillpack-maintenance | Methodology valid and maturity-appropriate; gates/checklists actually catch the failure | Current frameworks/standards (CMMI, SLSA v1.1, OWASP LLM 2025, EU AI Act, WCAG 2.2) |

Hybrid packs: grade Substance through the dominant track, borrow the other lens where a sheet calls for it.

---

## 4. The four subjects — per-tier anchors

### Subject A — Substance (read through the track)
*A1 Soundness · A2 Depth & Coverage (expert not tutorial; no holes vs declared domain) · A3 Currency (no rot).*

| Tier | Anchor |
|------|--------|
| **S** | Authoritative across the whole declared domain at expert depth; nothing wrong, nothing stale; teaches the *why*. A practitioner would learn from it. |
| **A** | Complete coverage, correct, current. Minor depth gaps only. |
| **B** | Sound and current, but one real coverage hole or a few shallow spots. |
| **C** | Mostly right, but a notable gap, a dated section, or tutorial-depth where expertise was promised. |
| **D** | Material inaccuracies, large coverage holes, or significant rot (H: wrong/old APIs; S: generic/indefensible judgment; P: invalid or mis-leveled methodology). |
| **F** | Wrong or dangerous content, or scaffold/vapor that doesn't cover the domain it claims. |

### Subject B — Usefulness / Actionability
*B1 Specificity (concrete, usable — not platitudes) · B2 Decision support (router routes well; sheets help you decide — symptom tables, decision trees, checklists where they earn their place).*

| Tier | Anchor |
|------|--------|
| **S** | Reading it changes what you *do*. Routes/decides crisply; concrete examples; a model others should imitate. |
| **A** | Consistently actionable; strong decision scaffolding; few abstract patches. |
| **B** | Useful overall; some sections drift into description rather than guidance. |
| **C** | Helpful in places but noticeably generic, or weak/ambiguous routing. |
| **D** | Mostly tells you *about* the topic instead of helping you act; routing misfires. |
| **F** | Not actionable / actively misroutes. |

### Subject C — Discipline / Robustness
*C1 Anti-pattern coverage · C2 Pressure-resistance (names the rationalizations — "just do it quick", "too simple" — and holds the line) · C3 Calibration & honesty (SME confidence/risk/gaps where required; honest scaffolds; marketing matches reality).*

| Tier | Anchor |
|------|--------|
| **S** | Pre-empts named rationalizations verbatim, catalogs failure modes, emits calibrated confidence/risk where applicable. The discipline signature fully realized. |
| **A** | Solid anti-pattern + pressure coverage; SME protocol present where required. |
| **B** | Present but uneven (one agent missing the protocol, partial rationalization coverage). |
| **C** | Thin discipline — anti-patterns mentioned but not operationalized; some overselling. |
| **D** | Little pressure-resistance; reviewer/SME agents non-compliant; marketing oversells content. |
| **F** | No discipline; misrepresents what it delivers. |

### Subject D — Form / Integrity *(gates the ceiling — see §5)*
*D1 Conformance (frontmatter, file layout, command/agent conventions) · D2 Discoverability & wiring ("Use when…" description, slash wrapper present+current, registered in marketplace, installable) · D3 Consistency (no drift across SKILL/wrapper/plugin.json/marketplace; correct cross-refs; clean sibling boundaries).*

| Tier | Anchor |
|------|--------|
| **S** | Flawless: conformant, fully wired, zero drift across all surfaces, clean boundaries. |
| **A** | Conformant and wired; trivial cosmetic nits only. |
| **B** | One Minor wiring/consistency issue (e.g., a count drift). |
| **C** | One Major: missing/stale slash wrapper, count drift across surfaces, or a boundary leak. |
| **D** | Multiple Majors: drift + missing wrapper + cross-ref bugs. |
| **F** | Doesn't install/route, unregistered-but-marketed, or packaging ships junk (backups/test-scenarios). |

---

## 5. Gates

1. **Discoverability gate (ceiling):** doesn't install or can't be invoked at all → overall **F**. Loadable but a required wiring surface broken (missing slash wrapper, scaffold-sold-as-complete, unregistered-but-marketed) → overall capped at **C**, regardless of content quality.
2. **Substance-dominates gate:** overall ≤ **(Substance grade + 1 tier)**.
3. **Honor-roll gate (S):** S requires Substance = S, no subject below A, and zero Major+ defects.
4. **Honesty override:** an *honest scaffold* that says plainly "sheets deferred to vN" floors at **D**, not F. Vapor that claims completeness is **F**.

---

## 6. Scoring procedure

1. Fix the unit (pack default; or sheet / command / agent).
2. Pick the track (H / S / P).
3. Grade A–D on the anchors (A through the track lens).
4. Apply the four gates.
5. Blend (40/25/20/15); gates override. Emit **overall letter + one-line verdict + top finding + top fix**.

### Layered (pack + per-sheet)
For the pack overall, also identify the **worst-offending sheets/components** and give each a
letter with a one-line note. Strong sheets need no individual grade; surface only the weak tail
that drags the pack down (and any single S-grade exemplar worth copying).

---

## 7. Worked examples

**`axiom-rust-engineering` (Track H):** A=S− (correct, current, exemplary redirect; one boundary overlap) · B=S (disciplined router) · C=S (names "I'll just `.clone()`" / `allow(clippy::all)` verbatim) · D=C (missing `/rust-engineering` wrapper + workspace boundary leak). Gate 1 hits the wrapper but router still loads → **Overall B**. Reconciles with existing **Major**.

**`lyra-creative-writing` (Track S):** A=A (23 sheets, reader-contract discipline) · B=A (mode-gated workflow) · C=C (11 reviewer agents lack SME protocol/`model:`) · D=C (wrapper claims v0.1). **Overall C+**. Reconciles with **Pass + 1 Major**.
