# Report Card — axiom-web-backend

**Version:** 1.2.1 (plugin.json) · **Track:** H — Hard / Technical
**Graded:** 2026-06-22 · **Unit:** pack (router + 11 sheets + 3 commands + 2 agents)

Prior review (`reviews/axiom-web-backend.md`, 2026-05-22) graded v1.2.0 as "Pass with Minor". Two of its Tier-1 Minors have since landed in v1.2.1 (M1 express cross-ref, M4 rate-limit routing row). This card is a fresh reading at v1.2.1 and weights it over the stale prior.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** (Track H) | **A** | 11 sheets, ~9,855 lines, correct and current. `fastapi-development.md:46-79` modern pooling (`pool_pre_ping`, `pool_recycle`, `expire_on_commit=False`) + lifespan via `asynccontextmanager` (`:180`). `api-authentication.md` is reference-depth: PKCE/RFC 7636 (`:80`), refresh-token rotation with replay detection (`:269`), mTLS + Istio service mesh (`:798`/`:884`), GDPR/PCI-DSS (`:1024`). `graphql-api-design.md` covers DataLoader N+1 (`:87`), Relay connections (`:328`), APQ (`:454`), federation (`:608`), depth-limiting + allowlisting (`:691`/`:703`). No wrong/old APIs spotted. Minor depth gaps: no standalone observability or websocket/SSE sheet (touched only in microservices/django). |
| **B — Usefulness** | **A** | Router routing table (`SKILL.md:43-56`) maps 1:1 to 12 trigger→sheet rows incl. the new rate-limit row (`:53`); catalog (`:130-153`) names all 11. Every sheet leads with a Quick-Reference pattern/decision table and a "When to Use" gate (e.g. fastapi `:23`, auth decision matrix `:40`, graphql vs REST `:29`). Background-task decision matrix (`fastapi:273`), JWT-vs-sessions matrix (`auth:40`). Concrete runnable code throughout, not description. |
| **C — Discipline** | **A** | All 11 sheets carry an explicit anti-pattern section (grep: 11/11). Router rationalization table names verbatim pressures ("I'll just give a quick answer", "specialist not available, I'll answer instead") and holds the line (`SKILL.md:89-98`). Both agents declare `model: sonnet`, cite `meta-sme-protocol:sme-agent-protocol`, and mandate Confidence/Risk/Information-Gaps/Caveats output (`api-architect.md:2,10`; `api-reviewer.md:2,10`) with symmetric scope hand-offs. |
| **D — Form** | **B** | Conformant: router description opens "Use when…"; slash wrapper `.claude/commands/web-backend.md` present, current, lists all 11 + 4 cross-refs; registered in `marketplace.json:79-81`; commands use quoted-JSON `allowed-tools`. One Minor drift: `marketplace.json:81` still advertises "production deployment patterns" but there is no dedicated deployment sheet (folded into per-framework subsections + `/scaffold-api`); that territory now belongs to sibling `axiom-devops-engineering`. plugin.json hardcodes counts (maintenance liability, polish). |

---

## Gate analysis

1. **Discoverability ceiling:** PASS. Installs, registered, router loads, slash wrapper present and current. No cap.
2. **Substance-dominates:** Substance = A → overall ≤ A+. Not binding below A.
3. **Honor-roll (S):** Not met — Substance is A not S (observability/websocket depth gaps; nothing reaches reference-grade-across-the-whole-domain), and D carries one Minor. No S.
4. **Honesty override:** N/A — no scaffold; content matches marketing except the one deployment-claim drift (Minor, not vapor).

Blend (A/A/A/B−) lands at **A−**. The single D-subject Minor (deployment-claim drift) is the only thing keeping it off a clean A.

---

## Layered — per-component grades

The pack is uniformly strong; only the weak tail and one exemplar are surfaced.

| Component | Grade | Note |
|-----------|-------|------|
| `api-authentication.md` | **A/S−** (exemplar) | Reference-grade: PKCE, token-family rotation w/ replay detection, mTLS+service-mesh, GDPR/PCI-DSS, per-tenant limits. The sheet others should copy for depth+discipline. |
| `marketplace.json` entry | **B−** | Description claims "production deployment patterns" with no dedicated deployment sheet — drift; soften, or defer to `axiom-devops-engineering`. |
| `fastapi-development.md` | **B+** | Thinnest sheet (509 lines vs ~1010 median) but coverage is complete (DI, async/sync, lifespan, uploads, bg-tasks, security, anti-patterns, "when NOT to use"). Compact, not deficient. |
| Coverage gaps | **B** | No standalone observability/tracing sheet and no websocket/SSE sheet; both only touched in passing. Minor against declared scope; arguably owned by sibling packs. |

No component grades below B.

---

## Overall: **A−**

Reconciles with prior **Pass with Minor** — and slightly better, since two prior Minors are now fixed in v1.2.1.

**Verdict:** A mature, disciplined, production-grade backend pack — correct, current, fully wired, SME-compliant — held just short of a clean A by a single marketplace description-vs-reality drift on "deployment patterns".

**Top finding:** `marketplace.json:81` advertises "production deployment patterns" but the pack ships no dedicated deployment sheet (only per-framework subsections); deployment is now the sibling `axiom-devops-engineering` pack's territory.

**Top fix:** Soften the marketplace description from "production deployment patterns" to "framework-level production patterns" (or cross-reference `axiom-devops-engineering`), then drop the hardcoded counts from `plugin.json:4`.
