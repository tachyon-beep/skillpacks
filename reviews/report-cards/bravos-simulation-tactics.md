# Report Card — bravos-simulation-tactics

**Version:** 1.2.0 (plugin.json)  ·  **Track:** H — Hard / Technical (game-simulation implementation; correctness = sound algorithms + engine-accurate APIs)  ·  **Graded:** 2026-06-22

Prior review (`reviews/bravos-simulation-tactics.md`, 2026-05-22, v1.1.5) is **STALE on its top finding**: its Major (M1, 909-line shadow-copy wrapper) and several Minors (m1 description, m4 cross-refs) have since been remediated. Current wrapper is 63 lines with frontmatter, "Use when", and sibling handoffs. This card weights a fresh reading.

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (H lens) | **A** | 10 specialist sheets, 1,300–2,850 lines each (~23.5k total). Algorithmically sound: `physics-simulation-patterns.md:71-78` correctly identifies explicit Euler energy accumulation / explosion and prescribes semi-implicit + fixed timestep + CCD; engine APIs accurate (Unity `WheelCollider`/`FixedUpdate`, Unreal `WheeledVehicleMovementComponent`, `CollisionDetectionMode.Continuous`). `economic-simulation-patterns.md:25-33` faucet<sink, net-negative chains, bid-ask spread — correct economy-design canon. Seven canonical game-sim domains + perf + debugging + the foundational fake-vs-simulate sheet; no structural holes. Currency is patterns-based (boids, A*, navmesh, GOAP, predator-prey) and stable — not version-rotting. Minor depth note: networking sim (client prediction / rollback) only lives inside the debugging sheet. |
| **B — Usefulness** | **A** | Router gives a 4-step decision tree (`SKILL.md:106-167`), a 15-row Quick Reference routing table (`SKILL.md:195-213`), a flowchart, and 7 named routing mistakes with cost-of-mistake. Sheets open with "When to use / Do NOT use" plus a time-boxed "Quick Start (<4h)" tiered CRITICAL/IMPORTANT/CAN-DEFER ladder (`physics...:26-49`, `economic...:25-44`) — reading it changes what you build first. Three support sheets (20 scenarios, 8 genre workflows, edge-case guide) back the router. |
| **C — Discipline** | **A** | Both agents SME-compliant: `desync-detective.md:2` + `simulation-architect.md:2` end the description with the verbatim SME line, declare `model: sonnet`, cite `meta-sme-protocol:sme-agent-protocol` in body, mandate all four output sections, and carry positive AND negative `<example>` activation clauses. Router names the rationalizations verbatim and holds the line ("I just need a quick traffic system, skip the foundational stuff" → Mistake 1 + "Cost: weeks of wasted work", `SKILL.md:258-269`; premature-optimization and skip-debugging mistakes likewise). Each sheet has explicit anti-pattern / Do-NOT-use fencing. |
| **D — Form** | **B** | Wrapper now exemplary (63 lines, frontmatter, "Use when", "Don't use", sibling handoffs to `/simulation-foundations`, `/determinism-and-replay`, `/systems-as-experience`). Commands carry quoted-JSON `allowed-tools` + `argument-hint`. Remaining drift: (1) router `SKILL.md:3` description is still the terse `Router skill - analyze requirements and direct to appropriate tactics` (no "Use when", no domain enumeration) — the wrapper is fixed but the skill's own discoverability surface is not; (2) `plugin.json:4` and `marketplace.json:348` claim "11 skills" while only 1 SKILL.md exists (router + 13 sheets); marketplace blurb also omits commands/agents. All Minor. |

## Gate analysis

1. **Discoverability ceiling:** Installs, registered (`marketplace.json:346`), wrapper present + current. No ceiling cap.
2. **Substance-dominates:** Substance = A → overall ≤ S−. Not binding.
3. **Honor-roll (S):** Fails — Form is B (router description + count drift), and Substance is A not S. No S.
4. **Honesty override:** N/A — no scaffold; content matches marketing (modulo the "skills" count convention).

Blend (A / A / A / B) lands at **A−**: three A subjects, one B that is Form-only and entirely Minor.

## Layered per-component grades

Pack is uniformly strong; no weak tail drags it down. Notable:

| Component | Grade | Note |
|---|---|---|
| `skills/.../SKILL.md` (router frontmatter) | B | Body is A-grade routing; the `description:` field alone is terse and off-convention — the one real Form drag. |
| `physics-simulation-patterns.md` | A | Exemplar: correct integrator stability reasoning, engine-true APIs, tiered Quick Start. Worth copying as the sheet template. |
| `agents/desync-detective.md` + `simulation-architect.md` | A | Fully SME-compliant with positive+negative activation examples — model for other packs. |
| `plugin.json` / `marketplace.json` counts | B | "11 skills" + marketplace blurb omits commands/agents; repo-wide convention but still a small honesty gap. |

## Overall: **A−**

**Verdict:** A deep, disciplined, engine-accurate game-simulation pack whose only soft spot is the router skill's own one-line description and cosmetic count drift.

**Top finding:** The wrapper was refurbished to best-practice but the router `SKILL.md:3` description was left behind — still `Router skill - analyze requirements...`, with no "Use when" or domain list, weakening skill-level discovery vs `yzmir-simulation-foundations`.

**Top fix:** Rewrite `SKILL.md:3` as a "Use when …" multi-clause description mirroring the wrapper (enumerate the seven domains + perf/debug, with a "for math use /simulation-foundations" do-NOT clause), and harmonise the "11 skills" count phrasing across plugin.json and marketplace.json.
