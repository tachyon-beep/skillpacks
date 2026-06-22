# Report Card — axiom-mcp-engineering

**Version:** 0.2.0 (feature-complete) · **Track:** H (Hard / Technical)
**Graded:** 2026-06-22 · **Unit:** pack (router + 13 sheets + 3 commands + 2 agents)

> **Divergence from prior review.** `reviews/axiom-mcp-engineering.md` (2026-05-22) graded this
> **Critical (acknowledged scaffold)** at v0.1.0 — router only, 0 sheets / 0 commands / 0 agents,
> absent from the marketplace, no slash wrapper. That review is now **stale**. v0.2.0 ships all 13
> sheets (199–431 lines each), all 3 commands, both agents, marketplace registration, and a current
> slash wrapper. The fresh reading below governs; the scaffold-criticality is fully resolved.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (Track H) | **S−** | Pinned to MCP revision **2025-11-25** throughout (e.g. `idempotency-and-atomicity.md:12`, `transport-reliability.md:14`, `authentication-and-trust.md:54`). Working dual-language code (Python FastMCP + low-level SDK, TypeScript `@modelcontextprotocol/sdk`). Genuinely expert detail: the stdio fd-dup trap — `stdio_server()` dups fd 1 at call time so reassigning `sys.stdout` poisons the protocol channel (`transport-reliability.md:49,89-109`); RFC 8707/9728/8693 token-audience + token-exchange-vs-passthrough (`authentication-and-trust.md:54,135-160`); fencing tokens on TTL leases (`idempotency-and-atomicity.md:223,261`). Coverage matches the declared domain exactly — 13 sheets cover all four primitives, idempotency, errors, schema drift, transport, auth, composition, observability, testing. Not full-S only because two sheets are leaner (`observability-for-tool-calls.md` 199 lines, `tool-api-design.md` 232) relative to the depth set by the discipline sheets. |
| **B — Usefulness** | **A** | Router routing table (`SKILL.md:156-174`) has ≥1 symptom row per sheet, phrased in user voice. Three intent-segmented Start-Here tracks (architect / critic / operator, `SKILL.md:128-150`). Decision tables earn their place: the four idempotency guarantees as a pick-exactly-one table (`idempotency-and-atomicity.md:34-39`), the two reconnect events (`transport-reliability.md:220-224`), the injection-vector table (`authentication-and-trust.md:176-182`). `/audit-mcp-tools` is a five-axis runnable checklist with per-axis severity guides and a JSON output contract. |
| **C — Discipline** | **S−** | Both SME agents set `model: sonnet` and explicitly follow `meta-sme-protocol:sme-agent-protocol` with all four mandatory sections (Confidence / Risk / Information Gaps / Caveats), verified in `mcp-server-critic.md:16,162-168` and `mcp-server-architect.md:1-4`. Critic carries an explicit **Anti-Rubber-Stamp Protocol** (`mcp-server-critic.md:186-201`) and an `intent-context-absent` auto-finding. Rationalization counters are verbatim across sheets ("It's just a debug print, I'll remove it later", "The agent would never do that", "We've never seen a double-execution in testing" — `transport-reliability.md:301-307`, `authentication-and-trust.md:222-228`, `idempotency-and-atomicity.md:300-307`). Per-sheet `Red flags — STOP` + `For the critic` severity rubrics. The architect/critic "if they always agree the pipeline is broken" discipline is operationalized, not just stated. |
| **D — Form / Integrity** | **A** | plugin.json v0.2.0 description matches marketplace entry matches shipped reality (13 sheets / 3 commands / 2 agents — all present). Registered at `marketplace.json:325`. Slash wrapper `.claude/commands/mcp-engineering.md` present and current (lists the right sheets, commands, agents, cross-refs). Frontmatter conformant on router, sheets, agents, commands. Clean sibling boundaries with `/web-backend`, `/llm-specialist`, `/audit-pipelines`, `/determinism-and-replay` (`SKILL.md:36-44`, pipeline diagrams 48-110). Minor nit only: the slash-wrapper description enumerates sheets but not the boundary-sheet absorption note. |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, registered, slash wrapper present + current, router loads and routes. No ceiling.
2. **Substance-dominates gate:** Substance = S− → overall ≤ S. Not binding.
3. **Honor-roll (S) gate:** Requires Substance = S, no subject < A, zero Major+ defects. Substance is S− (not full S) and there are no Major defects — so S is *just* missed on the Substance ceiling. Lands at **A** band.
4. **Honesty override:** N/A — pack is complete, not a scaffold; marketing matches reality.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Surfacing the leanest components and one exemplar.

| Component | Grade | Note |
|---|---|---|
| `idempotency-and-atomicity.md` | **S** (exemplar) | Reference-grade: pick-one guarantee table, dual-language working code committing dedup+side-effect in one txn, fencing tokens, 8 rationalization counters, critic severity rubric. The sheet other Track-H packs should copy. |
| `authentication-and-trust.md` | **S−** | RFC-correct OAuth 2.1 Resource Server, token-exchange-not-passthrough, confused-deputy, tool-poisoning, credentials-never-context. Among the strongest auth sheets in the marketplace. |
| `transport-reliability.md` | **S−** | The stdio fd-dup mechanism is expert-tier; Streamable HTTP + Origin/403 + resumability current. |
| `observability-for-tool-calls.md` | **B+** | Leanest sheet (199 lines); solid but thinner than the discipline cluster around it — a depth gap, not a defect. |
| `tool-api-design.md` | **B+** | Foundational and correct but the shortest content sheet (232 lines); carries heavy cross-ref load from commands, could go deeper on granularity heuristics. |

---

## Overall: **A**

**Verdict:** A complete, current, deeply disciplined Track-H pack — reference-grade discipline sheets and fully SME-compliant agents — held just below S only by two leaner sheets relative to its own high bar.

**Top finding:** v0.2.0 fully discharges the v0.1.0 scaffold debt: all 13 sheets are substantive and pinned to MCP revision 2025-11-25, both agents carry `model:` + the SME protocol + an Anti-Rubber-Stamp gate, and every wiring surface (marketplace, slash wrapper, plugin.json) is consistent.

**Top fix:** Deepen the two leanest sheets — `observability-for-tool-calls.md` (199 lines) and `tool-api-design.md` (232 lines) — to the depth of the idempotency/auth/transport cluster; that closes the only gap between this pack and a straight S.
