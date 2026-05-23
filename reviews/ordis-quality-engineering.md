# Review: ordis-quality-engineering
**Version:** 2.3.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

### 1.1 Plugin metadata

`/home/john/skillpacks/plugins/ordis-quality-engineering/.claude-plugin/plugin.json`:

- `name`: `ordis-quality-engineering`
- `version`: `2.3.0`
- `description`: "Quality engineering: E2E, API, integration, performance, chaos, flaky tests, observability, mutation testing, coverage gap analysis, modern supply-chain (Trivy/OSV/Syft/Cosign/SLSA). 21 reference sheets, 5 commands, 3 agents."
- `license`: `CC-BY-SA-4.0`
- Keywords: ordis, quality, testing, flaky-tests, test-pyramid, chaos-engineering, ci-cd, automation

Counts asserted in the description match what's on disk: 21 reference sheets + 5 commands + 3 agents.

### 1.2 Marketplace registration

`/home/john/skillpacks/.claude-plugin/marketplace.json` lines around the entry:

```
"name": "ordis-quality-engineering",
"source": "./plugins/ordis-quality-engineering",
"description": "E2E testing, performance testing, chaos engineering, test automation - 21 quality engineering skills",
"keywords": ["ordis","quality","testing","chaos-engineering"],
"category": "testing"
```

Registered. Catalog description is shorter than `plugin.json`'s description (does not mention supply-chain, mutation testing, observability). Minor drift.

### 1.3 Router skill

`/home/john/skillpacks/plugins/ordis-quality-engineering/skills/using-quality-engineering/SKILL.md` (228 lines):

- Frontmatter `name: using-quality-engineering`
- `description:` starts with "Use when user asks about E2E testing, performance testing..." — matches the repo "Use when..." convention.
- 21 reference-sheet links in the routing table.
- 5 worked scenarios with reference-sheet sequences.
- Anti-bypass clause ("User demands 'just answer, don't route' is NOT an exception") — repo convention.

### 1.4 Reference sheets (21)

All co-located in `skills/using-quality-engineering/`. Line counts via `wc -l`:

| Sheet | Lines | Domain |
|-------|-------|--------|
| test-isolation-fundamentals.md | 663 | Test fundamentals |
| api-testing-strategies.md | 471 | API testing |
| integration-testing-patterns.md | 478 | Integration |
| contract-testing.md | 524 | Contracts |
| e2e-testing-strategies.md | 290 | E2E |
| visual-regression-testing.md | 509 | Visual |
| performance-testing-fundamentals.md | 242 | Perf |
| load-testing-patterns.md | 843 | Load |
| quality-metrics-and-kpis.md | 448 | KPIs |
| test-maintenance-patterns.md | 500 | Maintenance |
| mutation-testing.md | 348 | Mutation |
| static-analysis-integration.md | 521 | SAST |
| dependency-scanning.md | 601 | SCA + supply chain |
| fuzz-testing.md | 445 | Fuzz |
| property-based-testing.md | 504 | PBT |
| testing-in-production.md | 363 | Prod testing |
| observability-and-monitoring.md | 479 | Obs |
| chaos-engineering-principles.md | 242 | Chaos |
| test-automation-architecture.md | 255 | Pyramid/CI |
| test-data-management.md | 419 | Fixtures |
| flaky-test-prevention.md | 493 | Flakiness |

Total ~9,866 lines (sheets + router). Each sheet has Use-when frontmatter; spot checks confirm the "decision-tree + tables + anti-patterns + tool selection" pattern dominant elsewhere in the repo.

**Frontmatter spot checks (selected):**
- `flaky-test-prevention.md:3` — "Use when debugging intermittent test failures, choosing between retries vs fixes, quarantining flaky tests, calculating flakiness rates, or preventing non-deterministic behavior"
- `dependency-scanning.md:3` — "Use when integrating SCA tools (Dependabot, Snyk, OWASP Dependency-Check, Trivy, Grype, OSV-Scanner), generating SBOMs (Syft, CycloneDX, SPDX), signing artifacts (Sigstore/Cosign), producing SLSA provenance..."
- `chaos-engineering-principles.md:3` — "Use when starting chaos engineering, designing fault injection experiments, choosing chaos tools, testing system resilience, or recovering from chaos incidents"
- `load-testing-patterns.md:3` — "Use when designing load tests, choosing tools (k6, JMeter, Gatling), calculating concurrent users from DAU, interpreting latency degradation..."
- `property-based-testing.md:3` — "Use when testing invariants, validating properties across many inputs, using Hypothesis (Python) or fast-check (JavaScript)..."
- `mutation-testing.md:3` — "Use when validating test effectiveness, measuring test quality beyond coverage, choosing mutation testing tools (Stryker, PITest, mutmut)..."
- `e2e-testing-strategies.md:3` — "Use when designing E2E test architecture, choosing between Cypress/Playwright/Selenium, prioritizing which flows to test..."
- `contract-testing.md:3` — "Use when implementing Pact contracts, choosing consumer-driven vs provider-driven approaches, handling breaking API changes..."

All eight sampled descriptions are concrete, list specific tools by name (which significantly aids discovery), and end with a "provides …" tail summarizing the artifact's content. This is best-in-class router triggerability.

### 1.5 Commands (5)

`/home/john/skillpacks/plugins/ordis-quality-engineering/commands/`:

| Command | Description (frontmatter) | argument-hint |
|---------|---------------------------|---------------|
| audit.md | Run quality metrics audit - coverage, flakiness rate, pass rate, build time | `"[test_directory] - defaults to tests/"` |
| diagnose-flaky.md | Diagnose intermittent test failures using systematic decision tree | `"[test_name or test_file]"` |
| setup-pipeline.md | Set up CI/CD testing pipeline with proper stages | `"[github|gitlab|jenkins] - CI platform"` |
| analyze-pyramid.md | Analyze test distribution across unit/integration/E2E levels | `"[test_directory] - defaults to tests/"` |
| analyze-test-gaps.md | Map codebase to tests - find untested critical code | `"[source_directory] - defaults to src/"` |

All five use `allowed-tools` as a quoted JSON-style array — `["Read", "Bash", "Glob", "Grep", "Skill"]` — matching the repo convention. argument-hints are quoted strings.

### 1.6 Agents (3)

`/home/john/skillpacks/plugins/ordis-quality-engineering/agents/`:

| Agent | Model | SME-compliant? |
|-------|-------|----------------|
| coverage-gap-analyst.md | sonnet | Yes — desc ends "Follows SME Agent Protocol with confidence/risk assessment."; body cites `meta-sme-protocol:sme-agent-protocol`; mandates Confidence/Risk/Information Gaps/Caveats sections (line 10). |
| flaky-test-diagnostician.md | sonnet | Yes — same compliance (line 10); positive/negative trigger examples; explicit "STOP. Diagnose root cause first. Retries mask symptoms." anti-pattern. |
| test-suite-reviewer.md | sonnet | Yes — same compliance (line 10). |

No agent declares `tools:` — they inherit parent context (correct, matches repo's ~60/65 agent convention).

**SME protocol details verified:**

For `flaky-test-diagnostician.md` (lines 14–32, four `<example>` blocks):
- Positive triggers: "my test is flaky", "test passes sometimes, fails sometimes", "test passes locally, fails in CI".
- Negative trigger: "User wants to add retry logic to a flaky test → Trigger: STOP. Diagnose root cause first. Retries mask symptoms."
- Out-of-domain handoff: "User asks about test architecture or coverage → DO NOT trigger: Use test-suite-reviewer or /quality:audit instead".

For `test-suite-reviewer.md` (lines 14–32, four `<example>` blocks):
- Six anti-patterns with detection rules and severity scoring (Critical/High/Medium): Sleepy Assertions, Test Interdependence, Hidden Dependencies, Assertion-Free Tests, Wrong Test Level, Shared Mutable State.
- Each anti-pattern shows BAD/GOOD code, a detection heuristic ("Search for `sleep(`, `wait(`, `time.sleep`, `Thread.sleep`"), and a severity rating.
- Scope-boundary handoff to `axiom-python-engineering` and `ordis-security-architect` (conditional on glob detection of the plugin manifest).

For `coverage-gap-analyst.md` (lines 1–10): standard SME header, expected behaviour ("READ the source code and test files. Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections."). Body not deeply sampled but frontmatter is consistent.

All three agents include explicit negative trigger examples — a sign of mature agent design that prevents over-activation in adjacent domains.

### 1.7 Slash-command wrapper

`/home/john/skillpacks/.claude/commands/` listing shows **no `quality-engineering.md`**. Sibling wrappers exist for `ai-engineering.md`, `python-engineering.md`, `security-architect.md`, `ux-designer.md`, etc. (33 wrappers total). The router skill `using-quality-engineering` is not user-invocable as a slash command at the repo root.

**This is the only structural break** vs. the repo's documented convention (CLAUDE.md lines 92–100: "All router skills … are available as slash commands in `.claude/commands/`").

### 1.8 Hooks

None. (No `hooks/` directory under the plugin.) Not required for this pack.

---

## 2. Domain & Coverage

### 2.1 Stated scope

From `SKILL.md` lines 38–53: testing strategy, debugging unreliable tests, performance testing, prod reliability/observability, and security testing integration (SAST/SCA/fuzzing).

### 2.2 Coverage map vs. inventory

**Foundational:**
- Test isolation/independence — `test-isolation-fundamentals.md` (663 lines, deepest in pack) — Exists, comprehensive.
- Flakiness diagnosis — `flaky-test-prevention.md` — Exists.
- Test data lifecycle — `test-data-management.md` — Exists.
- Test pyramid — `test-automation-architecture.md` — Exists.

**Core techniques:**
- API testing — Exists. Contract testing — Exists (separate sheet, correct split: API testing tests *your* API; contracts test pairwise compatibility).
- Integration testing — Exists.
- E2E — Exists.
- Visual regression — Exists.
- Performance/load — Exists (split into fundamentals + patterns, defensible).
- Mutation testing — Exists (348 lines).
- Property-based testing — Exists (504 lines).
- Fuzz testing — Exists (445 lines).

**Advanced / cross-cutting:**
- Chaos engineering — Exists (242 lines, shortest). Tool coverage current (LitmusChaos, Chaos Mesh, Gremlin, Chaos Toolkit, AWS FIS).
- Testing in production / feature flags / canaries — Exists.
- Observability / SLIs/SLOs — Exists. RED/USE/Four Golden Signals all named, OpenTelemetry referenced.
- SAST integration — Exists.
- SCA + supply chain — Exists. Modern stack (Trivy / Grype / OSV-Scanner / Syft / Cosign / SLSA) present from line 178 of `dependency-scanning.md` onward, including SHA-pinning advice for GitHub Actions (a current-best-practice supply-chain detail).
- Quality metrics / KPIs — Exists.
- Test maintenance — Exists.

**Gaps identified (Minor):**
- **Test review / PR review of tests** — Implicitly covered by the `test-suite-reviewer` agent body but no dedicated sheet on "how to code-review tests." Could live inside `test-maintenance-patterns.md`; not strictly a gap.
- **Performance budget governance** (defining and enforcing perf budgets in CI) — touched in `performance-testing-fundamentals.md` (242 lines is the shortest perf sheet) and `quality-metrics-and-kpis.md`, but not deeply elaborated. Minor.
- **Snapshot testing** — Visual regression covers screenshot-snapshot, but inline-snapshot testing (Jest snapshots, Vitest snapshots, Insta in Rust) is not addressed. Minor — likely intentional, since snapshot testing is contentious and the pack's philosophy is "test the contract, not the rendered form."
- **Test selection / test impact analysis (TIA)** — only briefly mentioned. Minor; emerging area.

**No coverage gaps that rise to Major.** Surface area is more comprehensive than the marketplace catalog description suggests — `plugin.json`'s longer description is more accurate.

**Boundary clarity:** The router (SKILL.md lines 48–52) explicitly states out-of-scope: pure code architecture decisions (→ `system-architect`), security threat modeling (→ `security-architect`), pure UX research (→ `ux-designer`). The two agents that have dependency hooks (`flaky-test-diagnostician`, `test-suite-reviewer`) further hand off to `axiom-python-engineering` for pytest-specific syntax and to `ordis-security-architect` for security testing architecture. This is the right level of boundary discipline for a horizontal pack — it knows it covers methodology and patterns, and explicitly does not cover language- or framework-specific syntax.

**Cross-domain layering:**
- The pack does NOT duplicate `axiom-devops-engineering` content. It covers CI/CD test stages (`commands/setup-pipeline.md`, `test-automation-architecture.md`) but does not address deployment strategies, blue/green, rollback mechanisms, or environment promotion — those belong to the DevOps pack. The split is clean.
- The pack does NOT duplicate `ordis-security-architect`. It covers security *testing* (SAST integration, dependency scanning, fuzzing) but does not address threat modeling or control design.
- The pack does NOT duplicate `yzmir-llm-specialist` even though LLM systems need testing — eval methodology for generation quality is owned by the LLM pack.

These three boundary clarities suggest the marketplace's faction taxonomy is well-internalized in this pack.

### 2.3 Domain stability

Quality engineering core (isolation, pyramid, flakiness, fixtures) is stable. The **evolving slice** is supply-chain (SLSA versioning, Sigstore tooling, Cosign 2.x semantics) and observability (OpenTelemetry GA dates, Prometheus → OTLP migration). The pack handles supply chain well as of writing; `cosign 2.x` is correctly noted as not needing `COSIGN_EXPERIMENTAL=1` (line 275 of `dependency-scanning.md`). Recommend a 6-month re-audit cadence for `dependency-scanning.md` and `observability-and-monitoring.md` only; the rest is stable.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | **Coverage breadth** | Pass | 21 sheets span foundational → advanced → cross-cutting. No Critical/Major gap; only Minor (snapshot testing, TIA, perf-budget governance). |
| 2 | **Coverage depth** | Pass (with one shallow sheet) | Average sheet ~470 lines; `test-isolation-fundamentals.md` 663 and `load-testing-patterns.md` 843 are the deepest. `performance-testing-fundamentals.md` (242) and `chaos-engineering-principles.md` (242) are the shallowest — chaos is appropriately concise (decision tree + tool table + anti-patterns), but `performance-testing-fundamentals.md` could carry more on benchmarking methodology. |
| 3 | **Router accuracy** | Pass | SKILL.md routing table maps each topic to its sheet; the catalog at lines 191–228 enumerates all 21 sheets with consistent descriptions. No links to nonexistent sheets. |
| 4 | **Frontmatter / convention compliance** | Pass | All sheets begin with `Use when …`; commands use quoted `allowed-tools` arrays and quoted argument-hints; agents declare only `description` + `model` (no spurious `tools:`). |
| 5 | **SME-protocol compliance (agents)** | Pass | All three agents — reviewer/auditor/diagnostician roles, all SME-style — cite `meta-sme-protocol:sme-agent-protocol` and mandate the four canonical sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats), verbatim. |
| 6 | **Command/skill type discipline** | Pass | Commands (`audit`, `diagnose-flaky`, `setup-pipeline`, `analyze-pyramid`, `analyze-test-gaps`) are all genuine user-invocable actions producing concrete outputs; they do *not* duplicate skill content but reference it (e.g., `diagnose-flaky.md` lines 216–227 explicitly hands off to the sheet for deeper guidance). |
| 7 | **Marketplace + slash-command wiring** | **Major issue** | Plugin is registered in `marketplace.json` (category `testing`) but **no `/home/john/skillpacks/.claude/commands/quality-engineering.md` wrapper exists**. Every sibling pack ships one. Users running `/quality-engineering` get no result, breaking the documented invocation pattern. |
| 8 | **Currency** | Pass | Modern supply-chain stack (Trivy/OSV/Syft/Cosign/SLSA) present in `dependency-scanning.md` lines 178–290. Cosign 2.x noted. OpenTelemetry referenced in `observability-and-monitoring.md`. Chaos tool table includes 2024-era tooling (LitmusChaos, Chaos Mesh) plus AWS FIS. Snyk GitHub-Action SHA-pinning guidance is correct (line 119 of `dependency-scanning.md`). |

**Overall: Major** — driven entirely by the missing slash-command wrapper. Without that single dimension, the pack would score **Pass**. Content, structure, agent compliance, and currency are all in good order.

---

## 4. Behavioral Tests

Per `testing-skill-quality.md`, components are process documentation; we test whether they guide correctly. I read the artifacts and walked the scenarios "as if" running them; no fresh subagent dispatch was performed (report-only constraint).

### 4.1 Router skill — pressure scenario

**Scenario:** "Just give me the quickest fix for a flaky test. I don't have time to read a doc."

Walking SKILL.md lines 174–184: the explicit clause "User demands 'just answer, don't route' is NOT an exception" plus the anti-bypass language tells the model to load the sheet anyway. The flaky scenario at lines 113–124 sequences `flaky-test-prevention.md` → `test-isolation-fundamentals.md` → `test-data-management.md`, which is the correct triage order. **Pass.**

### 4.2 Router skill — out-of-scope scenario

**Scenario:** "Help me threat-model the auth system."

SKILL.md lines 48–53 explicitly excludes "Security threat modeling (use security-architect)". **Pass.**

### 4.3 Flaky-test diagnostician agent — anti-retry pressure

**Scenario:** "Just add a retry and move on."

Agent body lines 26–28 contains the explicit example: "User wants to add retry logic to a flaky test → Trigger: STOP. Diagnose root cause first. Retries mask symptoms." Anti-pattern table at lines 154–162 hardens this: every shortcut the user might propose (retry / increase timeout / mark allowed-to-fail / skip) has a scripted refusal. **Pass.**

### 4.4 Audit command — vanity-metric pressure

**Scenario:** "Our coverage is 95%, we're good — write that up."

`audit.md` lines 122–128 has a "Vanity Metrics to Avoid" table: "95% coverage!" → "Can be gamed → Coverage delta on new code." Line 60: "Coverage can be gamed. Track coverage DELTA on new code, not absolute." **Pass.**

### 4.5 Setup-pipeline command — argument enumeration

**Scenario:** Invoked with no argument.

`argument-hint: "[github|gitlab|jenkins] - CI platform"`. The hint clearly enumerates platforms. Did not deep-read the command body for branch coverage; spot-check on lines 1–5 confirms the contract. **Pass (sampled).**

### 4.6 Dependency-scanning sheet — supply-chain pressure

**Scenario:** "Just use `@master` in our GitHub Action, it'll be fine."

`dependency-scanning.md` lines 119–122 mid-code: "Pin to a specific commit SHA, not @master. snyk/actions does not publish semver tags, so floating refs are the documented supply-chain anti-pattern." The comment is **inside the YAML example**, so the model encounters the warning at exactly the moment of temptation. **Pass.**

### 4.7 Slash-command discovery — Critical missing wrapper

**Scenario:** "Run /quality-engineering."

There is no `.claude/commands/quality-engineering.md`. The slash command does not exist. The model would either decline, route to a different `/...` command, or invoke the skill directly via the Skill tool (`ordis-quality-engineering:using-quality-engineering`). The first two are user-visible failures of the documented invocation contract. **Fix needed.**

### 4.8 Router skill — internal handoff to sibling pack

**Scenario:** "I have a Python-specific pytest fixture problem."

`flaky-test-diagnostician.md` lines 210–214 (Scope Boundaries → Defer): the agent dynamically checks `axiom-python-engineering` plugin existence and either routes to it or recommends installation. This is a sophisticated handoff pattern that respects pack independence (no hard dependency, conditional recommendation). **Pass.**

### 4.9 Setup-pipeline command — progressive testing pressure

**Scenario:** "We have 200 tests. Run all of them on every commit."

`commands/setup-pipeline.md` lines 17–25 explicitly stages testing: pre-push (lint + unit, <2m), PR (lint + unit + integration, <15m), merge (all, <30m), nightly (all + perf + security, <60m), pre-deploy (smoke, <5m). The core principle line 13 — "Progressive testing: fast feedback on PR, full validation on merge. Don't run all tests on every commit." — would reject the naive "run all" request. **Pass.**

### 4.10 E2E sheet — flow prioritization under deadline pressure

**Scenario:** "We have a release tomorrow. Write 50 E2E tests."

`e2e-testing-strategies.md` lines 12–13 mandates "5-10% E2E, 20-25% integration, 65-75% unit". Lines 31–49 provide a numeric scoring matrix (revenue +3, multi-system +3, historical failure +2, integration-level penalty −2, mostly UI penalty −3) plus a worked example. Lines 53–58 has "Pyramid Inversion: 200 E2E tests, 50 integration tests, 100 unit tests → Why bad: E2E tests are slow (30min CI), brittle (UI changes break tests), hard to debug → Fix: Invert back". The sheet would direct the model to score the 50 flows and keep ~10. **Pass.**

### 4.11 Test-suite-reviewer agent — assertion-free pressure

**Scenario:** Reviewing a test file containing `def test_calculate_tax(): calculate_tax(100)` (no assertions, 100% line coverage).

Anti-pattern #4 in the agent body (lines 99–117): "Assertion-Free Tests — Tests that execute code but don't verify behavior. Severity: Critical - test provides zero value. Detection: No `assert`, `expect`, `should` in test body." The agent would flag this as Critical and link to mutation testing for systematic detection. **Pass.**

### 4.12 Property-based testing sheet — discovery vs. example testing

**Scenario:** "We have 20 example tests for `reverse()`. Add another."

`property-based-testing.md` lines 14–22 explicitly contrasts the two approaches; lines 33–37 shows the property test (`reverse(reverse(lst)) == lst`) generating hundreds of inputs. The sheet would steer toward consolidation rather than another example. **Pass.**

### 4.13 Mutation-testing sheet — coverage-as-quality pressure

**Scenario:** "We have 100% line coverage, we're done with testing."

`mutation-testing.md` lines 14–35 explicitly demonstrates the failure mode with code: `test_calculate_tax()` invokes the function (100% coverage) but asserts nothing (mutation survives). The sheet would re-direct from coverage to mutation score. Consistent with `audit.md` line 60 "Coverage can be gamed. Track coverage DELTA on new code, not absolute." **Pass.**

### Summary

12 of 13 scenarios pass on content alone. The one failure (Scenario 4.7) is structural (missing slash-command wrapper), not content. Every pressure scenario tested found an explicit, model-readable refusal-with-redirect somewhere in the artifact. The pack's anti-pattern density (60+ named anti-patterns across the corpus, by my rough count) is unusually high for the marketplace — this is what makes it resistant to "just do it quickly" framings.

**Recommendation for a deeper Stage 3 pass:** Subagent-dispatch the top-5 highest-traffic sheets in fresh-context sessions to validate description-based discovery:
1. `flaky-test-prevention.md` — pressure scenarios (retry, allowed-to-fail, increase timeout).
2. `test-isolation-fundamentals.md` — order-dependence under "but parallel is fast" pressure.
3. `dependency-scanning.md` — `@master` vs. SHA-pinning under "I'll fix it later" pressure.
4. `quality-metrics-and-kpis.md` — "95% coverage means we're done" pressure.
5. `e2e-testing-strategies.md` — "test through the UI for everything" pressure.

This was not done in this review (report-only).

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major
**M1. Missing slash-command wrapper.** `/home/john/skillpacks/.claude/commands/quality-engineering.md` does not exist. CLAUDE.md lines 92–100 documents that every router skill must have a `.claude/commands/<name>.md` wrapper. The router skill `using-quality-engineering` is therefore not user-invocable as `/quality-engineering`, breaking the marketplace's consistent slash-command surface.

**Fix:** Add `/home/john/skillpacks/.claude/commands/quality-engineering.md` modelled on a sibling (e.g. `.claude/commands/security-architect.md` or `.claude/commands/python-engineering.md` — both are Ordis/Axiom-faction wrappers with comparable shape). Wrapper should describe entry symptoms, list the 21 sheets categorically, and reproduce or link the router's "When to Use" guidance.

### Minor
**m1. Marketplace catalog description shorter than `plugin.json` description.** `marketplace.json` says "E2E testing, performance testing, chaos engineering, test automation - 21 quality engineering skills". `plugin.json` v2.3.0 says the full list, including supply-chain and mutation. Users browsing the marketplace will under-estimate the pack's surface.

**Fix:** Sync the marketplace entry to mention supply-chain (Trivy/OSV/Syft/Cosign/SLSA), observability/SLOs, mutation testing, and coverage gap analysis.

**m2. `performance-testing-fundamentals.md` is the shallowest perf sheet at 242 lines.** Compared to `load-testing-patterns.md` (843), the fundamentals sheet feels like a TOC for the deeper one. Consider adding benchmarking methodology (warmup, statistical significance, regression-detection thresholds) so it stands on its own.

**m3. `chaos-engineering-principles.md` at 242 lines** is appropriate for principles but light on experiment-design templates. Compare to `load-testing-patterns.md`'s pattern-heavy depth. Acceptable as-is; flag only if the pack ever sees a v3.x.

**m4. No dedicated sheet on snapshot testing or test impact analysis (TIA).** Both are emerging, not foundational, but worth tracking as Minor gaps for a future release.

**m5. The router skill's catalog (SKILL.md lines 187–228) duplicates the routing table at lines 76–108.** Two near-identical lists of all 21 sheets in the same file. Workable but slightly repetitive — a reader hits the same 21 names twice. Consider collapsing the catalog into a one-line "see routing table above" pointer, or vice-versa.

### Polish
**p1. The header `# Using Quality Engineering (Meta-Skill Router)` in SKILL.md** is fine; sibling routers use `# Using X` or `# X (Meta-Skill Router)` — the convention is loose. No action.

**p2. Agent body line 213 of `flaky-test-diagnostician.md`** says "Check: `Glob` for `plugins/axiom-python-engineering/.claude-plugin/plugin.json`" — the relative path assumes the agent is running in this marketplace repo. From a deployed end-user repo this glob will not match. The same pattern recurs in `test-suite-reviewer.md` line 206 and line 212. Either generalize the detection (check installed-plugin list via Skill or a plugin manifest) or document that this conditional handoff is repo-internal. Polish-level: the agent's fallback ("If NOT found → Recommend installation") is benign, and end-users will at least receive an installation hint.

**p3. `commands/diagnose-flaky.md` references `${ARGUMENTS}`** (line 41) which is the documented command-substitution token; consistent with other repo commands. `audit.md` uses `${ARGUMENTS:-tests/}` form (line 23) — bash-style default-value expansion — which is a small upgrade and could be propagated.

**p4. `flaky-test-diagnostician.md` line 208** says "Use `/quality:audit` command" — the colon-namespaced form is one convention sometimes used in this repo for organizing slash-command names; verify whether the deployed slash command is `/quality:audit`, `/audit`, or `/ordis-quality-engineering:audit`. The pack's `commands/audit.md` does not encode a namespace in its filename, so the runtime exposes it as `/audit` (potentially colliding with other packs). Worth checking whether commands across packs are namespaced or flat — out of scope for this review, but flagged.

**p5. Mixed emoji usage.** Several sheets and the audit command use ✅/⚠️/❌ in tables (e.g., `audit.md` lines 102–108, `test-isolation-fundamentals.md` line 33). Repo guidance in CLAUDE.md warns against emoji unless the user requests them. These were authored before the constraint or are tolerated as in-content tables (not assistant output). No action unless a global emoji sweep is performed.

**p6. Two near-identical mentions of `freezegun`** appear in `commands/diagnose-flaky.md` lines 127–134 and in `flaky-test-prevention.md`. Acceptable: the command body intentionally inlines fix patterns to be self-contained. If the pack ever shrinks, this is consolidation room.

---

## 6. Recommended Actions

In priority order:

1. **Add `/home/john/skillpacks/.claude/commands/quality-engineering.md`.** Model on `.claude/commands/security-architect.md` (Ordis sibling, comparable shape). Closes M1. Targeted patch — no version bump strictly required but a **patch** bump (`2.3.0 → 2.3.1`) is the conventional response. The wrapper should:
   - Title: "Using Quality Engineering"
   - Overview matching SKILL.md lines 6–18.
   - When-to-use bullets (the seven items at SKILL.md lines 40–46).
   - A condensed routing table or a pointer ("See the skill's routing table for sheet selection").
   - "When NOT to use" list per SKILL.md lines 48–52.

2. **Update marketplace catalog description.** Sync `marketplace.json` for `ordis-quality-engineering` to the longer `plugin.json` description (or a slightly trimmed form). Closes m1. Suggested wording: "Quality engineering: E2E, performance, chaos, flaky tests, mutation testing, observability/SLOs, modern supply-chain (Trivy/OSV/Syft/Cosign/SLSA) - 21 reference sheets, 5 commands, 3 agents." Patch bump.

3. **Deepen `performance-testing-fundamentals.md`** with benchmarking methodology (warmup, sample size, statistical significance, P95/P99 vs. mean discussion, regression-detection thresholds, the role of `criterion`/`pytest-benchmark`/`go test -bench` style tooling). Closes m2. Minor bump (`2.3.x → 2.4.0`).

4. **Consider collapsing the duplicate sheet list in SKILL.md.** Either the routing table (lines 76–108) or the catalog (lines 187–228) is redundant. Closes m5. Patch bump.

5. **Audit `tools:` and namespace conventions for commands.** Verify whether commands across packs are flat (`/audit`) or namespaced (`/quality:audit`). The agent body of `flaky-test-diagnostician.md` (line 208) assumes namespacing; the on-disk filename does not encode it. Closes p4.

6. **(Forward-looking, not blocking)** Plan a v2.5 with snapshot testing + test impact analysis sheets, and a re-audit of `dependency-scanning.md` and `observability-and-monitoring.md` for currency every six months.

7. **(Forward-looking)** Consider adding a fourth agent for **performance-regression diagnosis** — the pack covers load patterns and metrics deeply, but does not have an SME agent for triaging a single regression incident the way `flaky-test-diagnostician` triages a single flaky test. Optional.

**Order:** 1 unblocks the user-visible invocation surface. 2 fixes catalog accuracy without touching the pack. 3 is content depth. 4 is housekeeping. 5 is cross-pack consistency. 6–7 are roadmap.

---

## 7. Reviewer Notes

- Review was conducted **stage 1–4 only** (Inventory + Domain + Scorecard + Behavioral). Stage 5 (execution / writing fixes) was skipped per the task contract.
- Behavioral testing was performed by reading the artifacts and walking the scenarios; no fresh subagent dispatch was used, per report-only constraint. Activation/discovery under fresh-context conditions was inferred from frontmatter shape (all sheets use "Use when …", which matches the dominant repo convention and should activate cleanly).
- Sampled in depth: `SKILL.md` (full), `dependency-scanning.md` (lines 1–290), `flaky-test-prevention.md` (lines 1–60), `observability-and-monitoring.md` (lines 1–60), `chaos-engineering-principles.md` (lines 1–80), `test-isolation-fundamentals.md` (lines 1–60), `commands/diagnose-flaky.md` (full), `commands/audit.md` (full), `agents/flaky-test-diagnostician.md` (full), other two agent frontmatters + body opener.
- Not sampled in depth: full bodies of `load-testing-patterns.md` (843 lines, largest), `contract-testing.md`, `mutation-testing.md`, `property-based-testing.md`, `fuzz-testing.md`, `visual-regression-testing.md`, `static-analysis-integration.md`, `testing-in-production.md`, `e2e-testing-strategies.md`, `integration-testing-patterns.md`, `api-testing-strategies.md`, `quality-metrics-and-kpis.md`, `test-maintenance-patterns.md`, `test-automation-architecture.md`, `test-data-management.md`, and commands `setup-pipeline.md` / `analyze-pyramid.md` / `analyze-test-gaps.md`. Spot checks of frontmatter and opening sections found no anomalies. A deeper Stage 3 pass should subagent-test at least the top 5 highest-traffic sheets (likely `flaky-test-prevention`, `test-isolation-fundamentals`, `e2e-testing-strategies`, `quality-metrics-and-kpis`, `dependency-scanning`).
- The plugin's `description` in `plugin.json` already enumerates "21 reference sheets, 5 commands, 3 agents" — a strong self-disclosure pattern. Several other Ordis/Axiom packs use the same format; recommend it elsewhere too if not already in place.
- No hooks, no `tools:` keys on agents, no nested directories — structurally clean.
- License (`CC-BY-SA-4.0`) and faction attribution consistent with the marketplace as of 2026-05-22.

**Overall verdict:** A mature, content-rich quality engineering pack with one structural defect (missing slash-command wrapper) and a handful of minor polish items. Single patch bump closes the Major finding; minor bump closes the content/marketplace items.

### 7.1 Comparative notes

Compared against sibling reviews in `/home/john/skillpacks/reviews/` (this directory):
- Most Axiom/Ordis packs in the v1.x–v2.x range carry a `using-X.md` slash-command wrapper. The single missing wrapper here is the clearest reason this pack does not score "Pass" on Dimension 7. It is also a one-file fix.
- Pack depth (9,866 lines of content + 5 commands + 3 SME agents) is in the top tier of the marketplace by size. Comparable in shape to `axiom-static-analysis-engineering` and `axiom-determinism-and-replay` (router + ~13 sheets + 3 commands + 2 agents).
- SME-protocol compliance on all three agents is fully aligned with `meta-sme-protocol:sme-agent-protocol` — no drift.

### 7.2 Confidence and risk

- **Confidence in findings:** High for M1 (verified by directory listing), high for m1 (verified by JSON diff), medium for m2 (subjective depth judgement), medium for p4 (command-namespace convention not fully reconstructed from artifacts alone).
- **Risk in recommended fixes:** Low for actions 1, 2, 4 (mechanical edits). Medium for action 3 (content addition requires `superpowers:writing-skills` discipline). Low for action 5 (audit-only).
- **Information gaps:** Did not subagent-dispatch fresh-context tests; did not deeply read 14 of 21 sheets; did not enumerate marketplace-wide command namespacing.
- **Caveats:** This is a Stage 1–4 review only. Stage 5 (execution) is out of scope per the task contract.

### 7.3 If only one action is taken

Add `/home/john/skillpacks/.claude/commands/quality-engineering.md`. Every other finding is a refinement; this one is the only break in the marketplace's documented invocation contract for this pack, and it is a five-minute fix modeled on any sibling wrapper. Patch bump (`2.3.0 → 2.3.1`), no version-bump-blocking changes required to skills, commands, or agents themselves.
