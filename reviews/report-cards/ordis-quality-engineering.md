# Report Card — ordis-quality-engineering

**Version:** 2.4.0 (plugin.json)  ·  **Track:** P (Process / Hybrid)  ·  **Graded:** 2026-06-22

Quality engineering pack: router + 21 reference sheets, 5 commands, 3 SME agents. Spans
test fundamentals → advanced (mutation, property-based, chaos) → cross-cutting
(supply-chain, observability). Counts on disk match the plugin.json claim (21/5/3).

> **Prior-evidence note:** `reviews/ordis-quality-engineering.md` (2026-05-22, v2.3.0)
> graded this **Major**, driven *entirely* by a missing `/quality-engineering` slash
> wrapper ("Without that single dimension, the pack would score Pass"). That wrapper now
> exists at `.claude/commands/quality-engineering.md`, is current, and is richly detailed
> (sheets + commands + agents + cross-refs). The single Major has been **closed**; the
> pack has been remediated since the prior review. This grade weights the fresh reading.

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (Track P) | **A** | 21 sheets, ~9,866 lines total; expert depth not tutorial. Methodology valid and current: `dependency-scanning.md` carries the full 2025 supply-chain stack (Trivy/OSV/Syft/Cosign/SLSA) with a threat→tool mapping table (lines 324–334), correct Cosign 2.x note (`COSIGN_EXPERIMENTAL` "not required on cosign 2.x", line 275), and SHA-pinning-at-point-of-temptation (lines 118–123). `flaky-test-prevention.md` opens with a symptom→root-cause→fix decision tree (lines 16–23). Largest sheets are load-testing (843) and test-isolation (663). Minor shallow spots: `performance-testing-fundamentals.md` / `chaos-engineering-principles.md` at 242 lines each. No material inaccuracy or rot found. |
| **B — Usefulness** | **A** | Router maps every topic to a sheet (SKILL.md lines 76–107), plus 5 worked scenarios with ordered sheet sequences (lines 113–172). Sheets decide, not describe: numeric E2E scoring matrix, vanity-metric tables, remediation-option tables. Routing is crisp and the wrapper adds clean outbound cross-refs (`/security-architect`, `/axiom-devops-engineering`, `/python-engineering`). |
| **C — Discipline** | **A** | All three agents cite `meta-sme-protocol:sme-agent-protocol` and mandate Confidence/Risk/Information-Gaps/Caveats; all declare `model: sonnet`, none declare spurious `tools:`. Dense anti-pattern catalogs across the corpus. Named rationalizations held verbatim: flaky-diagnostician lines 156–161 script refusals for "I'll add a retry / increase the timeout / mark allowed-to-fail / skip it." Router anti-bypass clause (SKILL.md 183). Warnings placed inside code at the moment of temptation (dependency-scanning 118–123). Approaches S; held at A by no single sheet pre-empting rationalizations *verbatim* the way the discipline signature demands top-to-bottom. |
| **D — Form** | **B** | Conformant frontmatter (sheets all "Use when…"; commands quote `allowed-tools` + `argument-hint`; agents are `description`+`model` only). Slash wrapper present + current. **Two Minor nits:** (1) `flaky-test-diagnostician.md` lines 31 & 208 reference the stale namespace `/quality:audit` and `/quality:analyze-pyramid` — actual namespace is `/ordis-quality-engineering:*`; (2) marketplace catalog description ("21 quality engineering skills") is terser/less accurate than plugin.json (omits supply-chain/mutation/observability). No drift in counts. |

## Gate analysis

1. **Discoverability (ceiling):** Pass. Pack installs, router loads, slash wrapper present + current, registered in marketplace. No cap.
2. **Substance-dominates:** Substance = A → overall ≤ A+. Not binding below A.
3. **Honor-roll (S):** Not met — Form = B (stale agent refs + catalog drift), so a subject sits below A. Disqualifies S.
4. **Honesty override:** N/A — not a scaffold; marketing matches delivery.

Blend (A/A/A/B) with no binding cap lands at **A−**.

## Layered per-component grades

Pack is uniformly strong; only the weak tail and the exemplar are surfaced.

| Component | Grade | Note |
|---|---|---|
| `agents/flaky-test-diagnostician.md` | A− | Excellent scripted refusals + conditional cross-pack handoff (lines 210–214), but uses stale `/quality:` command namespace twice (31, 208). |
| `skills/.../performance-testing-fundamentals.md` | B | Shallowest of the substantive sheets (242 lines); benchmarking-methodology depth thinner than peers. |
| marketplace.json entry | B | "21 quality engineering skills" undersells surface area vs plugin.json's richer description. |
| **`skills/.../dependency-scanning.md`** | **S (exemplar)** | Reference-grade currency: layered 2025 supply-chain stack, threat→tool mapping, Cosign 2.x correctness, SHA-pinning warning inside the YAML at point of temptation, clean boundary cross-ref to `ordis-security-architect`. Worth copying as the template for "current-practice" sheets. |

## Overall: **A−**

### Verdict
A current, disciplined, comprehensive quality-engineering pack; the prior Major (missing slash wrapper) is closed, leaving only two cosmetic Form nits.

### Top finding
The pack has been remediated since the v2.3.0 review — the slash wrapper now exists and is current, closing the sole Major; content/discipline/currency are A-grade throughout, with `dependency-scanning.md` an S-grade exemplar.

### Top fix
In `agents/flaky-test-diagnostician.md` replace the stale `/quality:audit` and `/quality:analyze-pyramid` references (lines 31, 208) with the real `/ordis-quality-engineering:*` namespace, and align the marketplace.json description with plugin.json.
