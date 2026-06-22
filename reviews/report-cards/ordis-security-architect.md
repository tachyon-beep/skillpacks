# Report Card — ordis-security-architect

**Version:** 1.3.0 (plugin.json) · **Track:** P — Process / Hybrid (security methodology; borrows H lens for code-level controls)
**Graded:** 2026-06-22 · Layered (pack + per-component)

Structure: 1 router + 10 reference sheets (~6,440 lines), 3 commands, 2 SME agents. Slash wrapper `/security-architect` present and current. Registered in marketplace.json (lines 536-538).

> Prior review `reviews/ordis-security-architect.md` is dated 2026-05-22 against **v1.2.0** and rated the pack **Major**, driven by two router/wrapper drift issues. The pack has since advanced to **v1.3.0** and both Major issues are **resolved** (verified fresh below). This card weights the current reading and notes the divergence.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** (P lens) | **A** | Expert, current methodology across the full declared domain. `llm-and-ai-security.md` frames the LLM as "a second untrusted parser" (line 11), tags every OWASP LLM Top 10:2025 item with CWE + MITRE ATLAS codes (e.g. LLM01 → CWE-1426, AML.T0051/T0054, lines 75-78) and ranks architectural controls by effectiveness with named anti-patterns ("Just tell the model not to follow injected instructions… does not work", line 100). `threat-modeling.md` (740 lines) gives full STRIDE + attack trees + L×I scoring. Currency is strong throughout: SLSA v1.1, SPDX 3.0/CycloneDX 1.6, NIST CSF 2.0, ISO 27001:2022, PCI-DSS v4.0.1, EU AI Act 2024/1689, NIS2, xz-utils CVE-2024-3094. Each framework table carries a version + "always check current versions" caveat. Holds at A not S because of acknowledged gaps: no PQC / crypto-agility sheet (NIST IR 8547, CNSA 2.0) and no dedicated cloud-native / K8s posture sheet — both noted in prior review and still absent. |
| **B — Usefulness** | **A** | Router routes crisply by symptom + worked example (5 patterns, 5 examples, decision tree, quick-reference table). Core-vs-extension split is explicit and load-bearing. Sheets are action-oriented: ordered control lists, discovery-question gates, template scaffolds. Cross-pack handoffs are precise (audit-pipelines for the log itself, llm-specialist for application correctness, web-backend for API-layer impl). Commands carry argument-hints and scoping prompts. |
| **C — Discipline** | **A−** | Strong pressure-resistance: `threat-modeling.md:19` ("threats found after deployment are 10x more expensive to fix"); `compliance-awareness-and-mapping.md` forces a 3-question discovery gate with "Never assume."; `classified-systems-security.md:8` refuses the "sanitize your way out" framing. Both agents cite `meta-sme-protocol:sme-agent-protocol` and mandate Confidence/Risk/Information Gaps/Caveats (threat-analyst.md:10, controls-designer.md:10), declare `model: opus`, and have symmetric positive+negative activation examples with explicit hand-offs. Held below A only by the one residual: agent **Output Format** templates (threat-analyst.md lines ~106-167) scaffold the STRIDE/risk tables but do **not** include inline `### Confidence Assessment` etc. placeholders — a model templating strictly from the block could omit them. The protocol sentence remains load-bearing. |
| **D — Form** | **A** | Conformant frontmatter (router name+description; commands description+allowed-tools+argument-hint; agents description+model). Counts agree across all surfaces: "11 skills, 3 commands, 2 agents" in plugin.json, marketplace.json, and wrapper (10 sheets + router = 11). All 10 router sheet links resolve on disk. Wrapper `.claude/commands/security-architect.md` is current — includes llm + supply-chain extensions, correct plugin-format command names, has frontmatter. **Both prior Majors fixed**: zero occurrences of stale `architecture-security-review` / `secure-code-patterns` anywhere. Trivial nit only: router `description:` doesn't open with "Use when…" (wrapper does). |

---

## Gate analysis

1. **Discoverability gate:** Pack loads; router + wrapper both present and current; registered and installable. No cap. PASS.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+. Not binding.
3. **Honor-roll (S) gate:** Fails — Substance is A not S (PQC/cloud gaps), and C is A−. Correctly not S.
4. **Honesty override:** Not a scaffold; marketing matches delivered content. N/A.

**Blend:** A(40) · A(25) · A−(20) · A(15) → **A**.

---

## Layered per-component grades

Pack is uniformly strong; no weak tail drags it down. Surfacing the one soft spot and one exemplar:

| Component | Grade | Note |
|-----------|-------|------|
| `agents/threat-analyst.md` + `agents/controls-designer.md` | **A−** | SME protocol present and load-bearing in body, but Output Format templates omit inline Confidence/Risk/Gaps/Caveats placeholders — template-only authoring could drop them. Only residual discipline gap. |
| Domain coverage (no PQC / crypto-agility, no cloud-native sheet) | **B+** | Honest, lower-priority gaps for an architecture pack; flagged in prior review, still open. Keeps Substance off S. |
| `llm-and-ai-security.md` | **S (exemplar)** | Copy-worthy: "second untrusted parser" mental model, per-item CWE+ATLAS tagging, effectiveness-ordered controls, named anti-patterns, crisp boundary vs yzmir-llm-specialist. The template other security-domain sheets should imitate. |

---

## Overall: **A**

**Verdict:** Reference-rich, current, well-disciplined security-architecture pack; the two prior Major drift issues are fixed and only polish remains.

**Top finding:** Both v1.2.0 Major issues (stale `architecture-security-review` skill name; wrapper omitting llm/supply-chain extensions) are resolved in v1.3.0 — grep confirms zero stale-name occurrences and the wrapper now mirrors the router. The pack has moved from Major to A.

**Top fix:** Add inline `### Confidence Assessment / Risk Assessment / Information Gaps / Caveats` placeholders to the two agents' Output Format blocks so the SME protocol is reinforced at the point of templating (closes the last A− on Discipline). Stretch: a PQC/crypto-agility sheet would lift Substance toward S.
