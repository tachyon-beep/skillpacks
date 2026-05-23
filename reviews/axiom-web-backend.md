# Review: axiom-web-backend
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

### Plugin metadata
- `plugins/axiom-web-backend/.claude-plugin/plugin.json` lines 1-22.
- `name`: axiom-web-backend, `version`: 1.2.0, `license`: CC-BY-SA-4.0.
- `description`: "Web backend: FastAPI, Django, Express, REST/GraphQL, microservices. **11 reference sheets, 3 commands, 2 agents.**" — matches what is on disk.
- Marketplace registration confirmed at `.claude-plugin/marketplace.json` (entry `"name": "axiom-web-backend"`, `source` `./plugins/axiom-web-backend`).
- Slash-command wrapper present at `/home/john/skillpacks/.claude/commands/web-backend.md` (28 lines).

### Router skill (1)
| Skill | Path | Notes |
|-------|------|-------|
| `using-web-backend` | `skills/using-web-backend/SKILL.md` (152 lines) | Router — description starts with "Use when …" per repo convention. Routes to 11 specialists. |

### Reference sheets (11) — all in `skills/using-web-backend/`
| Sheet | Lines | Domain |
|-------|-------|--------|
| `fastapi-development.md` | 509 | FastAPI: DI, async, lifespan, background tasks |
| `django-development.md` | 890 | Django ORM, DRF, migrations, caching |
| `express-development.md` | 872 | Express middleware, error handling, validation |
| `rest-api-design.md` | 523 | Resource modeling, HTTP semantics, versioning, pagination |
| `graphql-api-design.md` | 1010 | Schema design, N+1, federation, subscriptions |
| `microservices-architecture.md` | 592 | Boundaries, sagas, service mesh, resilience |
| `message-queues.md` | 993 | RabbitMQ/Kafka/SQS, ordering, schema evolution |
| `api-authentication.md` | 1381 | JWT, OAuth2, mTLS, rotation, multi-tenant |
| `database-integration.md` | 1123 | Pooling, query opt, migrations, transactions |
| `api-testing.md` | 1013 | Integration, performance, security, CI/CD |
| `api-documentation.md` | 949 | OpenAPI, doc-as-code, SDK gen, doc debt |

Total reference-sheet content: ~9855 lines. The shortest (`fastapi-development.md`, 509) is noticeably thinner than the rest; everything else is in the 500–1400 band.

### Commands (3)
| Command | Path | Frontmatter |
|---------|------|-------------|
| `/scaffold-api` | `commands/scaffold-api.md` (268 lines) | `description`, `allowed-tools: ["Read", "Bash", "Glob", "Grep", "Write", "AskUserQuestion"]`, `argument-hint: "[project_name]"` |
| `/debug-api` | `commands/debug-api.md` (264 lines) | `description`, `allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]`, `argument-hint: "[symptom_or_endpoint]"` |
| `/review-api` | `commands/review-api.md` (213 lines) | `description`, `allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]`, `argument-hint: "[api_directory_or_file]"` |

All three follow the marketplace's quoted-JSON-array convention for `allowed-tools` and quoted-string `argument-hint`. ✓

### Agents (2)
| Agent | Path | `model` | SME compliant? |
|-------|------|---------|----------------|
| `api-architect` | `agents/api-architect.md` (296 lines) | `sonnet` | Yes — description ends "Follows SME Agent Protocol with confidence/risk assessment." (line 2); body cites `meta-sme-protocol:sme-agent-protocol` (line 10) and requires the four-section output. |
| `api-reviewer` | `agents/api-reviewer.md` (242 lines) | `sonnet` | Yes — description ends "Follows SME Agent Protocol..." (line 2); body cites protocol (line 10) and requires the four sections. |

Neither agent declares a `tools:` key (consistent with repo norm ~60/65). Both include positive AND negative activation examples (api-architect has 3 positive + 2 negative; api-reviewer has 3 positive + 2 negative). Scope boundaries are explicit and cross-reference each other ("I do NOT … use api-reviewer" / "I do NOT … use api-architect").

### Hooks
None. Not required for this pack type.

### Slash-command wrapper
`.claude/commands/web-backend.md` exists (lines 1-28). Frontmatter: `description: Routes web backend questions to specialist skills (FastAPI, Django, Express, REST, GraphQL, microservices, auth, deployment)`. Lists all 11 specialists by name with one-line descriptions and includes the four cross-pack references. **Passes the missing-wrapper check.**

---

## 2. Domain & Coverage

### User-defined scope (inferred from plugin.json + router SKILL.md)
- **Intent:** Production-grade web backend expertise across three dominant frameworks (FastAPI/Django/Express) and the cross-cutting concerns of API construction (REST, GraphQL, auth, DB, queues, testing, docs, microservices).
- **Boundaries:** Explicitly out-of-scope (deferred via router cross-references): security threat modelling → `ordis-security-architect`; Python language patterns → `axiom-python-engineering`; API UX/ergonomics → `lyra-ux-designer`; doc writing register → `muna-technical-writer`.
- **Audience:** Practitioners and seniors. Sheets assume the reader can read async Python / typed JS, knows what "N+1" and "OAuth2" mean, and is shipping to production.

### Coverage map vs. inventory

**Foundational (must cover):**
- Resource/HTTP semantics — `rest-api-design.md` ✓
- GraphQL fundamentals — `graphql-api-design.md` ✓
- Authentication primitives — `api-authentication.md` ✓ (extensive — 1381 lines)
- Database access patterns — `database-integration.md` ✓
- Testing strategy — `api-testing.md` ✓

**Core (commonly needed):**
- FastAPI — `fastapi-development.md` ✓ (but lightest sheet)
- Django — `django-development.md` ✓
- Express/Node — `express-development.md` ✓
- API docs / OpenAPI — `api-documentation.md` ✓
- Microservices boundaries — `microservices-architecture.md` ✓
- Message-queue patterns — `message-queues.md` ✓

**Advanced / cross-cutting (commonly missing in this kind of pack):**
- **Observability / tracing / structured logging** — Not a standalone sheet. Mentioned in passing in microservices and message-queues. **Gap (minor)** — most production backend packs have a dedicated observability sheet.
- **Rate limiting / throttling as a first-class concern** — Touched in `api-authentication.md` and in the architect/reviewer agents, but no standalone sheet. **Gap (minor).**
- **WebSockets / SSE / streaming APIs** — Not covered. Mentioned only as a one-liner under Django ("async Django, Channels"). **Gap (minor).**
- **Caching strategy (HTTP cache headers, ETags, conditional GET, CDN, Redis)** — Scattered across REST and database sheets. No dedicated treatment of HTTP-layer caching. **Gap (minor).**
- **Deployment / containerization / 12-factor** — The plugin description in `marketplace.json` advertises "production deployment patterns" but there is **no dedicated deployment skill**. Deployment is folded into `/scaffold-api` (Dockerfile snippet) and per-framework sheets. **Description-vs-reality drift (minor).**

### Domain currency
Web-backend frameworks are evolving but the patterns this pack teaches (async DI in FastAPI, DRF serializers, Express middleware, OAuth2 PKCE, JWT short-lived + refresh, OpenAPI 3.x, RabbitMQ vs Kafka selection) are all current as of late 2025. No deprecated guidance spotted in the spot-checks. The `fastapi-development.md` SQLAlchemy pattern (lines 41-72) uses modern `sessionmaker(expire_on_commit=False)` and `pool_pre_ping=True` — current best practice.

---

## 3. Fitness Scorecard (8 dimensions)

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Router activation** | Pass | `SKILL.md` line 3 description begins with "Use when" and enumerates 9+ trigger phrases. Strong discoverability. |
| **Router → specialist alignment** | Pass | All 11 routing-table rows (SKILL.md lines 45-55) map 1:1 to existing files. Catalog at lines 133-152 lists all 11 with one-line descriptions. No dangling links. |
| **Specialist coverage** | Pass with minor gaps | 11 sheets cover all foundational + core territory. Gaps in observability, websockets, caching, dedicated deployment (see Section 2). All Minor. |
| **Command discipline** | Pass | All three commands have correct frontmatter style (quoted JSON-array `allowed-tools`, quoted `argument-hint`). Scope split (scaffold/debug/review) is clean and matches the agent split. |
| **Agent SME compliance** | Pass | Both agents end the description with the canonical "Follows SME Agent Protocol..." phrase. Both cite `meta-sme-protocol:sme-agent-protocol` and require Confidence / Risk / Information Gaps / Caveats. Both have positive AND negative activation examples. Neither declares a spurious `tools:` key. |
| **Cross-pack references** | Pass | Router SKILL.md lines 59-64 cross-reference `ordis-security-architect`, `lyra-ux-designer`, `axiom-python-engineering`, `muna-technical-writer`. Commands `/scaffold-api` (lines 236-255) and `/debug-api` (lines 237-251) include a "Cross-Pack Discovery" snippet pattern. Cross-references are accurate. |
| **Slash-command wrapper** | Pass | `.claude/commands/web-backend.md` exists and accurately enumerates all 11 specialists + 4 cross-pack pointers. No drift vs. router description. |
| **Internal consistency / no copy-paste leakage** | Minor | One copy-paste bug found: `express-development.md` line 24 says *"General TypeScript patterns (use `axiom-python-engineering` equivalents)"* — `axiom-python-engineering` is the wrong pack for TypeScript. No matching TS pack exists in this marketplace; the line should drop the cross-ref or point to a generic resource. |

**Overall:** **Pass with Minor issues.** Structurally sound, comprehensive within declared scope, SME-compliant, slash-wrapper present and accurate. No Critical or Major findings. A handful of Minor polish items (one copy-paste bug, four small coverage gaps, one description-vs-reality drift on "deployment patterns").

---

## 4. Behavioral Tests

Per the rubric (`testing-skill-quality.md`), I prioritise pressure tests on the router and a real-world complexity scenario per agent. Tests are conducted as inline trials within this review session — explicitly the lowest-fidelity option per the rubric, so results are weighted accordingly and flagged where activation/discovery is the thing under test.

### T1 — Router pressure test (rationalisation resistance)
**Scenario:** *"Quick question: I'm building a JWT-authenticated FastAPI service with GraphQL endpoints and a Kafka consumer. Just give me a high-level pattern in one answer, I don't want to load five skills."*

**What the router does:** SKILL.md lines 86-97 ("Rationalization Table") explicitly anticipates this pressure. The "I'll just give a quick answer" excuse is matched by "Quick answers miss edge cases and production patterns." Lines 100-117 ("Example Routing") demonstrate the *correct* multi-skill response — naming each specialist by file. The router would name `fastapi-development.md`, `api-authentication.md`, `graphql-api-design.md`, and `message-queues.md` and address them in sequence rather than collapsing them.

**Verdict:** Pass. Rationalisation resistance is explicit and the multi-skill example matches the pressure scenario.

### T2 — Router edge case (overlap / hand-off)
**Scenario:** *"How should I structure rate-limit logic for a public REST API used by mobile clients?"*

**What happens:** Rate limiting is touched in `api-authentication.md` (security hardening section) and is referenced from `api-architect.md` (line 147-150 "rate_limiting" block). It does not have a dedicated sheet. The router routing-table (SKILL.md lines 45-55) has no "rate limit" row. A literal lookup would land on `api-authentication.md` because of the "auth" hit, or on the architect agent via `/scaffold-api`.

**Verdict:** Acceptable but borderline. The router does not advertise rate-limiting as a routable concern. Either add a row "rate limiting, throttling, quota → api-authentication.md (security hardening section)" or commit to a future standalone sheet. Minor.

### T3 — Agent scope discipline (api-architect)
**Scenario:** *"Review my FastAPI service — I think the resource structure is wrong."*

**What happens:** `api-architect.md` lines 38-42 explicitly include a negative example: *"User: 'Review my API implementation' → Do NOT activate - review task, use api-reviewer or /review-api."* The architect would decline and hand off.

**Verdict:** Pass. Scope boundary is explicit; hand-off target is named.

### T4 — Agent scope discipline (api-reviewer)
**Scenario:** *"Design the auth model for a multi-tenant SaaS API."*

**What happens:** `api-reviewer.md` lines 38-42 negative example: *"User: 'Design an API for this feature' → Do NOT activate - design task, use api-architect."* The reviewer would decline and hand off.

**Verdict:** Pass. Symmetric to T3.

### T5 — Command edge case (`/debug-api` under "intermittent 500s, can't reproduce")
**Scenario:** *"Customer complains about intermittent 500 errors but I can't reproduce locally."*

**What happens:** `debug-api.md` lines 28-40 ("Step 1: Reproduce the Issue") emphasises reproduction as a precondition. Lines 159-173 ("Intermittent Failures") provide `ab -n 1000 -c 100` load tests, deadlock log greps, race-condition hints, and idempotency/retry-with-backoff fixes. The command does not collapse to "add try/except". The "Common Symptoms" table line 21 maps "Intermittent failures" → "Race condition, connection exhaustion" → "Concurrent request count" — directly relevant.

**Verdict:** Pass. The command handles the harder-than-the-default debugging scenario.

### T6 — Discovery / express-development copy-paste bug
**Scenario:** A user reading `express-development.md` follows the "DO NOT use for" cross-reference at line 24.

**What happens:** They are directed to `axiom-python-engineering` for "General TypeScript patterns." That is wrong — `axiom-python-engineering` covers Python, not TypeScript. There is no TypeScript-engineering pack in the marketplace.

**Verdict:** Fail (Minor). Copy-paste residue from a sibling sheet template. Fix: drop the line or rephrase as "General TypeScript / Node patterns — out of scope for this pack."

### Test summary
| Test | Component | Result |
|------|-----------|--------|
| T1 | Router (pressure) | Pass |
| T2 | Router (rate-limit lookup) | Borderline — Minor gap |
| T3 | api-architect (scope) | Pass |
| T4 | api-reviewer (scope) | Pass |
| T5 | /debug-api (intermittent) | Pass |
| T6 | express-development.md xref | Fail — Minor |

**Caveat (per `testing-skill-quality.md` lines 87-92):** These are inline trials. The activation half of router behaviour (does the model discover and load this skill from a fresh user prompt?) is not exercised. To validate discovery, a fresh-context subagent dispatch would be needed; recommended if the pack is being graded on activation reliability rather than guidance quality.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major
None. (The missing slash-command wrapper check — which the task brief flagged as Major if absent — passes: `/home/john/skillpacks/.claude/commands/web-backend.md` exists, is current, and lists all 11 specialists.)

### Minor

**M1. `express-development.md` line 24 — wrong cross-reference.**
- Evidence: `skills/using-web-backend/express-development.md:24` reads `- General TypeScript patterns (use `axiom-python-engineering` equivalents)`.
- Problem: `axiom-python-engineering` does not cover TypeScript.
- Fix: Remove the line, or rewrite as `- General TypeScript / Node language patterns — out of scope for this pack.`

**M2. `marketplace.json` description claims "production deployment patterns" — no dedicated deployment sheet.**
- Evidence: `.claude-plugin/marketplace.json` description for axiom-web-backend includes "production deployment patterns"; the 11 sheets cover deployment only via short subsections inside `fastapi-development.md`, `django-development.md`, `express-development.md`, and `/scaffold-api` (Dockerfile snippet at lines 171-182). No `deployment.md` or `production-readiness.md`.
- Fix options: either add a dedicated sheet (treat as a Stage-5 task using `superpowers:writing-skills`), or soften the marketplace description to "framework-level production patterns" rather than "deployment patterns."

**M3. No standalone observability / tracing / structured-logging sheet.**
- Evidence: `microservices-architecture.md` line 21 mentions "Distributed tracing, logging aggregation, metrics" as in-scope; `message-queues.md` line 19 mentions "Lag tracking, alerting, distributed tracing." Neither produces a focused observability treatment.
- Impact: Practitioners building a single non-microservice API have no skill to load for OTel / structured-logging / metric-emission patterns.
- Fix: New sheet `observability.md` (or fold into a renamed `production-readiness.md` per M2).

**M4. No rate-limiting / throttling row in the router table.**
- Evidence: `SKILL.md` lines 45-55 routing table has no entry for "rate limit", "throttle", "quota". The concern is touched in `api-authentication.md` (security hardening) and in agent output formats.
- Fix: Add a row to the routing table — `| Rate limiting, throttling, quotas | [api-authentication.md](api-authentication.md) | Security hardening section |` — so the router catches the trigger phrase.

**M5. No standalone websocket / SSE / streaming-API sheet.**
- Evidence: `django-development.md` line 20 mentions Channels as in-scope but does not deliver dedicated treatment; no other sheet covers SSE, WebSockets, or HTTP/2 streaming.
- Fix: Either add `realtime-apis.md` or explicitly mark websockets as out-of-scope in the router so users do not expect coverage.

**M6. `fastapi-development.md` is materially shorter than peer sheets (509 lines vs. ~890–1380 for siblings).**
- Evidence: `wc -l` across the 11 sheets — fastapi is the only sheet under 800 lines. Median is ~1010.
- Impact: Possibly missing FastAPI material relative to depth shown in Django/Express. Not necessarily a defect (FastAPI is more compact than Django by nature), but worth a sanity check.
- Fix: Spot-audit fastapi sheet against the topic list it claims (lifespan, file uploads, testing, security, performance) and confirm depth-parity with sibling sheets.

### Polish

**P1.** `plugin.json` description (line 4) hardcodes counts: "11 reference sheets, 3 commands, 2 agents." This is a maintenance liability — any inventory change requires editing plugin metadata. Consider dropping the counts or moving them to a generated README.

**P2.** Router SKILL.md "How to Access Reference Sheets" section (lines 24-37) repeats the same anti-pattern warning across every plugin in the marketplace. Acceptable boilerplate but could be condensed.

**P3.** The Cross-Pack Discovery blocks in commands (`scaffold-api.md` 238-255, `debug-api.md` 239-251, `review-api.md` 188-199) use a `glob.glob(...)` Python snippet to advertise sibling packs. This pattern works but reads as instruction to the model rather than executable code; consider standardising as plain prose ("If the user also needs X, the `axiom-python-engineering` pack covers …").

**P4.** Reference sheets do not declare frontmatter (per repo convention — they are content files referenced by a router SKILL.md, so this is correct). No action needed; noted for completeness.

---

## 6. Recommended Actions

**No immediate action required to keep the pack shipping.** It is structurally sound and behaviourally compliant. The recommended changes are quality improvements.

### Tier 1 — quick fixes (patch bump, x.2.1)
1. Fix `express-development.md:24` (M1) — wrong cross-reference, one-line edit.
2. Add a routing-table row for rate limiting (M4) — one-line edit in `SKILL.md`.
3. Soften the marketplace.json description to remove the "production deployment patterns" claim, OR commit to adding the sheet in Tier 2 (M2).

### Tier 2 — coverage extensions (minor bump, x.3.0)
4. Add `observability.md` reference sheet (M3) — would route Otel, structured logging, metrics, distributed tracing into a focused sheet. Run through `superpowers:writing-skills`, not inline.
5. Either add `realtime-apis.md` (M5) or update the router to declare websockets explicitly out-of-scope.
6. Depth-audit `fastapi-development.md` against sibling sheet topic coverage (M6) — may surface gaps to fill.

### Tier 3 — optional polish
7. Drop the hardcoded counts from `plugin.json` description (P1).
8. Standardise cross-pack discovery pattern across commands (P3).

### What NOT to do
- Do not rebuild. The pack is structurally sound; rebuild would discard working content.
- Do not write new skills inline. Tier 2 additions (M3, M5) must go through `superpowers:writing-skills` with behavioural testing per the maintenance rubric.
- Do not pin the agents' `model` to specific IDs; current `sonnet` declaration is correct.

---

## 7. Reviewer Notes

**Methodology.** Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied. Stage 5 (execution) skipped per task brief ("Report-only — no edits"). All file paths cited use absolute paths from the repo root.

**Coverage of the review.** I read in full: router `SKILL.md`, both agents, all three commands, the slash-command wrapper, `plugin.json`, the marketplace catalog entry. I spot-checked all 11 reference sheets via header reads (first 25–80 lines each) to verify the routing-table claims and the "When to Use" framing. I did not full-read the 9855 lines of specialist sheets — that would be appropriate for a Stage-5 implementation pass, not a Stage-1–4 review.

**Behavioural-test fidelity.** Tests T1–T6 are inline trials, the lowest-fidelity option per `testing-skill-quality.md` lines 87-92. They validate that the *content* of the components would guide a model correctly if loaded. They do *not* validate discovery — i.e., whether a fresh-context model would invoke this router when asked a backend question. For activation/discovery validation, dispatch a fresh subagent (preferred per the rubric); deferred here because the task brief is a report, not an execution.

**Confidence.** High that no Critical or Major issues exist. Medium-high on the Minor findings — M1 is verbatim evidence; M2-M5 are inference from inventory; M6 is statistical (a line-count outlier could be benign).

**Risk.** Low. The pack ships value today. The Tier-1 fixes are one-line edits; Tier-2 additions are pure extensions and won't break existing behaviour. The biggest latent risk is M2 (description-vs-reality drift on "deployment patterns") because it sets user expectations the pack does not fully meet — but this is a discoverability annoyance, not a correctness failure.

**Information gaps.** I did not load the eight unread reference sheets in full, so depth-of-coverage claims for those sheets are taken at face value from their "Overview" and "When to Use" sections. A fresh-context discovery test was not run. The marketplace.json description was inspected for axiom-web-backend specifically; other entries were not cross-referenced.

**Caveats.** Treat the gap analysis (M3, M5) as a function of the pack's *declared* scope rather than an absolute spec. If the author intentionally scoped observability and websockets out, only a router note acknowledging that is required — not a new sheet.
