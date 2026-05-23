# Review: axiom-engineering-foundations
**Version:** 1.0.1 **Reviewed:** 2026-05-22 **Reviewer:** general-purpose subagent

Source: `/home/john/skillpacks/plugins/axiom-engineering-foundations/`
Rubric: `meta-skillpack-maintenance:using-skillpack-maintenance` (Stages 1-4).
Scope: report-only. No edits performed.

---

## 1. Inventory

### Plugin metadata
- Path: `/home/john/skillpacks/plugins/axiom-engineering-foundations/.claude-plugin/plugin.json`
- `name`: `axiom-engineering-foundations`
- `version`: `1.0.1` (line 3)
- `description`: "Universal software engineering methodology: systematic debugging, safe refactoring, code review, incident response, technical debt triage, and codebase comprehension. Language-agnostic foundations for professional engineering practice." (line 4)
- `author`, `repository`, `license`, `keywords` all present and well-formed.

### Marketplace registration
- Registered in `/home/john/skillpacks/.claude-plugin/marketplace.json` lines 108-123.
- Category: `development`.
- Source path resolves to the existing plugin directory.
- Marketplace description (line 111) is consistent with `plugin.json` (paraphrased + skill count "6 language-agnostic skills"). Minor: the marketplace description does not match `plugin.json` verbatim, but both are accurate. **No action needed.**

### Directory layout
```
axiom-engineering-foundations/
├── .claude-plugin/plugin.json
└── skills/
    └── using-software-engineering/
        ├── SKILL.md                       (router)
        ├── complex-debugging.md           (reference sheet)
        ├── systematic-refactoring.md      (reference sheet)
        ├── code-review-methodology.md     (reference sheet)
        ├── incident-response.md           (reference sheet)
        ├── technical-debt-triage.md       (reference sheet)
        └── codebase-confidence-building.md (reference sheet)
```

No `commands/`, no `agents/`, no `hooks/` — single-skill router-with-reference-sheets pack.

### Skills (1 router + 6 reference sheets)
| File | Lines | Purpose |
|------|-------|---------|
| `SKILL.md` | 270 | Router by situation: bugs / refactor / review / incident / debt / confidence |
| `complex-debugging.md` | 717 | Scientific-method debugging, 6 phases + heisenbug + reset protocol |
| `systematic-refactoring.md` | 398 | Characterize → plan → small steps → verify → commit |
| `code-review-methodology.md` | 388 | Context → high-level → correctness → quality → feedback |
| `incident-response.md` | 467 | Assess → contain → mitigate → diagnose → fix → postmortem |
| `technical-debt-triage.md` | 401 | Identify → categorize → assess → prioritize → decide |
| `codebase-confidence-building.md` | 408 | Orient → trace → interrogate → experiment → document |

### Commands
- None. Pack defines no `commands/*.md`.

### Agents
- None. Pack defines no `agents/*.md`.

### Hooks
- None. No `hooks/hooks.json`.

### Slash-command wrapper
- **Expected:** `/home/john/skillpacks/.claude/commands/software-engineering.md` (following the marketplace convention: slug derived from the `using-X` skill's X — see CLAUDE.md and the `python-engineering.md` template).
- **Actual:** No such file exists. Repo-root `.claude/commands/` contains `ai-engineering.md`, `python-engineering.md`, `sdlc-engineering.md`, etc., but **no** `software-engineering.md` or `engineering-foundations.md`.
- This is the dominant repo convention (CLAUDE.md: "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits."). **Missing — Major finding.**

---

## 2. Domain & Coverage

### User-defined scope (inferred from `plugin.json` + SKILL.md "Don't use for")
- **Intent:** Universal, language-agnostic software-engineering methodology — process and discipline, not language- or domain-specific knowledge.
- **In scope (positive):** Complex debugging, safe refactoring, code review, incident response, technical-debt triage, codebase-confidence building.
- **Out of scope (explicit, SKILL.md line 24):** Language-specific issues (language packs); algorithm design (CS fundamentals); infra/deployment (DevOps packs).
- **Audience:** Practitioners through experts. Tone is "I assume you know how to write code; here is the methodology to apply when the situation is hard."

### Coverage map vs. inventory

| Domain area | Status | Evidence |
|---|---|---|
| Complex / multi-system debugging | **Exists** | `complex-debugging.md` (very thorough — phases 0-6, heisenbug protocol, reset protocol, correlation-ID pattern, cognitive bias warnings) |
| Refactoring (safe transformation) | **Exists** | `systematic-refactoring.md` (5 phases + strangler-fig, parallel change, branch by abstraction) |
| Code review | **Exists** | `code-review-methodology.md` (context → quality → feedback, with comment categories) |
| Incident response | **Exists** | `incident-response.md` (6 phases + severity matrix + postmortem template) |
| Technical-debt triage | **Exists** | `technical-debt-triage.md` (impact/cost scoring, ROI, debt register) |
| Codebase comprehension / ownership | **Exists** | `codebase-confidence-building.md` (orient → trace → interrogate → experiment → document) |

**Adjacent universal-engineering topics that are arguably in-scope but absent (Minor / Polish):**
- **Estimation & sequencing** of a multi-step task (covered by `superpowers:writing-plans` and `axiom-planning:implementation-planning`, so explicit cross-link is acceptable — currently neither is mentioned in this pack).
- **Test-driven development discipline** — `superpowers:test-driven-development` exists at the meta level; complex-debugging.md mentions "Test-Driven Fix" (line 357) and code-review notes "tests must pass", but the pack does not cross-link to `superpowers:test-driven-development` despite mentioning superpowers elsewhere implicitly (it does not — there are no `superpowers:` references in this pack at all, even though it covers identical methodology territory).
- **Verification-before-completion** discipline — `superpowers:verification-before-completion` covers the "claim work is done" failure mode. Refactoring and incident-response both touch the topic but do not link out.
- **Receiving / requesting code review** — `superpowers:receiving-code-review` and `superpowers:requesting-code-review` are the meta-skills; `code-review-methodology.md` has a "Receiving Feedback" section (lines 292-327) that covers similar ground but does not link out.

These four are not gaps in *coverage* — the pack covers the methodology itself competently. They are gaps in *cross-skill linkage*: the pack lives in the same marketplace as `superpowers:*` and does not acknowledge it. (Logged as Minor in §3.)

### Domain currency
- **Stable domain.** Debugging methodology, refactoring catalogs, incident-response process, debt triage, code-review practice are all decades-stable. No research phase needed. Pack content reflects current consensus (e.g., strangler-fig, blameless postmortems, ROI scoring, characterization tests). No deprecated patterns spotted.

---

## 3. Fitness Scorecard

| Dimension | Rating | Notes |
|---|---|---|
| Router quality | **Pass** | Clear "Use when..." description; routing-by-symptom table; ambiguous-query interception with explicit "Ask First"; routing-mistakes table; cross-cutting scenarios (taking over a codebase, inherited bug, etc.) |
| Skill descriptions (frontmatter) | **Pass** | Router description starts with "Use when..." per dominant convention; enumerates trigger symptoms; explicit non-scope |
| Frontmatter conformance | **Pass** | Router has `name` + `description` only (correct — `allowed-tools` rare on skills). Reference sheets correctly have no frontmatter (per rubric `using-skillpack-maintenance:SKILL.md` line 19) |
| Component cohesion | **Pass** | All 6 reference sheets share a consistent structure (Core Principle → When to Use → Process diagram → Phase-by-phase → Anti-patterns/Red Flags → Quick Reference); each cross-references its peers correctly |
| Slash-command exposure | **Major** | No `/home/john/skillpacks/.claude/commands/software-engineering.md` wrapper. Per CLAUDE.md and the marketplace convention, every router skill should have a slash-command wrapper. Pack is invisible via `/` discovery. |
| SME agent protocol | **N/A** | Pack has no agents |
| Anti-pattern coverage | **Pass** | Every reference sheet has a "Red Flags" table with verbatim rationalizations and counter-responses. Heisenbug + Investigation Reset Protocol in complex-debugging is unusually strong. |
| Cross-skill linkage | **Minor** | Intra-pack cross-references are excellent (each sheet has an "Integration with Other Skills" table). Cross-plugin references to `yzmir-*`, `ordis-*`, `bravos-*`, `axiom-system-architect`, `axiom-system-archaeologist`, `muna-technical-writer` verified — referenced sheets exist. **Gap:** no references to `superpowers:*` despite overlapping territory (TDD, verification-before-completion, plan-writing, code-review meta) — see §2. |

### Overall: **Pass with one Major (missing slash-command wrapper) + a few Minor / Polish.**

The pack is structurally sound, internally consistent, and behaviorally well-defended (rationalization tables are unusually robust). The Major is purely a missing repo-root wrapper file; the skill itself is fine and would activate via description-based discovery. No content rebuild required.

---

## 4. Behavioral Tests

The rubric (`testing-skill-quality.md` lines 80-92) prefers subagent dispatch for behavioral tests. **The current session does not dispatch nested subagents** — instead, I executed a paper-based pressure walkthrough (each scenario read against the relevant sheet and judged against the rubric's gauntlet categories A/B/C). This is the "inline trial within the current session" tier of mechanism (rubric line 88), which is acknowledged as lowest-fidelity. **Flagged in Reviewer Notes (§7).** Findings below should be treated as plausible-rather-than-empirical until re-run with subagent dispatch.

### Test 1 — Router: ambiguous query under simplicity pressure
**Scenario:** "Help me with this code, it's a mess. Just figure it out, I'm in a hurry."
**Expected behavior:** Router refuses to guess; asks one clarifying question per the "Ambiguous Queries - Ask First" block.
**Evidence in skill:** SKILL.md lines 192-205 explicitly catch "This is a mess" with the response "Do you need to debug something broken, or clean up working-but-ugly code?" — exact-phrase counter, plus the rule "Never guess. Ask once, route accurately."
**Result:** **Pass.**

### Test 2 — complex-debugging: "I've tried everything, just add a try/except"
**Scenario:** User has been debugging for 2 days, no progress, says "just add error handling and move on."
**Expected behavior:** Skill catches sunk-cost + symptom-fix rationalizations and forces a reset.
**Evidence in skill:** complex-debugging.md lines 570-617 — "Investigation Reset Protocol (I've Tried Everything)". Lists *exactly* the symptom ("wanting to 'just add error handling and move on'", line 581), names it as guessing-not-debugging, prescribes STOP → audit attempts → identify exact error → restart from Phase 1 with discipline. Also Red Flags table lines 707-717: "I'll add a null check" → "STOP. That's a symptom fix."
**Result:** **Pass.** This is the strongest defended scenario in the pack.

### Test 3 — incident-response: "Let me understand the code first before rolling back"
**Scenario:** Production down after a deploy 10 min ago. User wants to read the code to understand the bug before rolling back.
**Expected behavior:** Skill enforces "rollback first, ask questions later."
**Evidence in skill:** incident-response.md lines 49 ("Phases 1-3 are about SPEED. Phases 4-6 are about THOROUGHNESS"), 105 ("Recent deploy + New symptoms = ROLLBACK FIRST, ASK QUESTIONS LATER"), 117 ("Don't wait for root cause to rollback"), and Red Flags table 400-409 with the exact line "Let me understand the code first" → "Rollback first".
**Result:** **Pass.**

### Test 4 — systematic-refactoring: scope creep + sunk-cost
**Scenario:** "While I'm refactoring this function, I noticed three other things to fix. Let me just batch them. I've already invested an hour, no point doing tiny steps now."
**Expected behavior:** Skill enforces "one refactoring operation per step", catches scope creep.
**Evidence in skill:** systematic-refactoring.md lines 144-152 ("Golden Rule: One refactoring operation per step. Never combine: Extract + rename, Move + modify, Delete + add. If tests fail, you must know EXACTLY what caused it."), Red Flags table 343-352 with "While I'm here, I'll also fix..." → "Scope creep. Separate concern. Finish current refactor first."
**Result:** **Pass.**

### Test 5 — code-review-methodology: rubber-stamp pressure
**Scenario:** "I have 30 PRs to review today, just LGTM the small ones."
**Expected behavior:** Skill catches rubber-stamping.
**Evidence in skill:** code-review-methodology.md anti-pattern table lines 332-342 — "**Rubber stamping** | 'LGTM' without reading | Actually review or don't approve." Also "Time-box — 1 hour max per review session" (line 277).
**Result:** **Pass.** Slightly weaker than Tests 1-4: the counter is one table row without a verbatim rationalization phrase. Adequate but not as airtight.

### Summary of behavioral tests
| Test | Component | Pressure type | Result |
|---|---|---|---|
| 1 | Router | Simplicity + time | Pass |
| 2 | complex-debugging | Sunk-cost + symptom-fix | Pass (strongest) |
| 3 | incident-response | "Understand first" | Pass |
| 4 | systematic-refactoring | Scope creep + sunk-cost | Pass |
| 5 | code-review-methodology | Rubber-stamp | Pass (adequate) |

No behavioral failures detected at the lowest-fidelity tier. Subagent re-runs (especially for description-based discovery — does the router activate from a cold context when a user types "help me debug this race condition"?) would strengthen these results.

---

## 5. Findings

### Critical
None.

### Major

**M1. Missing slash-command wrapper.**
- **Path expected:** `/home/john/skillpacks/.claude/commands/software-engineering.md`
- **Status:** does not exist; the marketplace convention (per `CLAUDE.md` lines under "Slash Commands (Router Skills)") is that every `using-X` router skill ships with a wrapper at `.claude/commands/X.md` so users can invoke it as `/X`. All 13 other router skills in the marketplace have wrappers; this is the only one missing.
- **Impact:** users cannot discover or invoke the router via `/`. Skill is still reachable via description-based auto-discovery, so the pack is not broken — but discoverability is reduced and the pack diverges from the marketplace's stated architecture.
- **Fix scope:** small. One wrapper file (~50-200 lines) following the `python-engineering.md` template (see `/home/john/skillpacks/.claude/commands/python-engineering.md`).

### Minor

**m1. Slug ambiguity for the slash-command wrapper.**
- The router skill is `using-software-engineering`, but the pack is `axiom-engineering-foundations`. The convention takes the `X` from `using-X` (so the wrapper would be `software-engineering.md`), but other axiom packs have a `using-X` where `X` matches the pack stem (e.g., `axiom-python-engineering` → `using-python-engineering` → `python-engineering.md`). Here the pack stem (`engineering-foundations`) and the router slug (`software-engineering`) diverge.
- **Implication:** when the wrapper is added (M1), pick `software-engineering.md` (matches the `using-X` slug used in the skill ID — that is what the user types when they `/X`).
- **Fix scope:** decision only, no code.

**m2. No cross-references to `superpowers:*`.**
- Pack covers methodology that heavily overlaps with `superpowers:test-driven-development`, `superpowers:systematic-debugging`, `superpowers:verification-before-completion`, `superpowers:writing-plans`, `superpowers:receiving-code-review`, and `superpowers:requesting-code-review`. None of these are referenced in any of the 6 reference sheets.
- Specifically:
  - `complex-debugging.md` "Domain-Specific Handoffs" (lines 642-674) lists `yzmir-*`, `ordis-quality-engineering:*`, and `yzmir-systems-thinking:*` but not `superpowers:systematic-debugging`. Yet `superpowers:systematic-debugging`'s description (per the system-reminder available-skills list) is "Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes" — overlapping or arguably *upstream of* this entire reference sheet.
  - `code-review-methodology.md` § "Integration with Other Skills" (lines 346-353) has no `superpowers:requesting-code-review` / `superpowers:receiving-code-review` link.
  - `systematic-refactoring.md` does not link to `superpowers:test-driven-development` despite its insistence on characterization tests.
- **Impact:** Minor — pack is internally consistent, but a reader looking for "the next layer up" is not signposted. Could either (a) add reciprocal links, or (b) explicitly declare the relationship (e.g., "`superpowers:systematic-debugging` is the discipline; this sheet is the Claude-adapted methodology"). The decision is a design call, not a correctness call.
- **Fix scope:** small (~5-10 lines across the integration tables).

**m3. Marketplace description and `plugin.json` description differ.**
- `plugin.json` line 4: "Universal software engineering methodology: systematic debugging, safe refactoring, code review, incident response, technical debt triage, and codebase comprehension. Language-agnostic foundations for professional engineering practice."
- `marketplace.json` line 111: "Universal software engineering methodology - systematic debugging, safe refactoring, code review, incident response, technical debt triage, codebase confidence building - 6 language-agnostic skills"
- Differences are cosmetic (colon vs hyphen, "codebase comprehension" vs "codebase confidence building", presence of skill count). Both are accurate. Cleanup is optional.
- **Fix scope:** trivial.

### Polish

**p1. Code-review-methodology rubber-stamp counter lacks a verbatim user-quote.**
- The anti-patterns table (line 333) names "Rubber stamping" with the descriptor "'LGTM' without reading". Compared to the verbatim-rationalization style used in `complex-debugging.md`'s Red Flags table (each row is a literal quoted thought), this row is less behaviorally sharp. Other packs in this marketplace (e.g., `superpowers:receiving-code-review`) tend to phrase counters as a literal user thought + counter. Optional polish.

**p2. The `incident-response.md` "Internal Updates" template (lines 332-350) uses placeholder timestamps from 2024-era examples.**
- No dates appear in the prose, but the implied ISO-week is unspecified. No correctness issue. Optional polish if the pack wants to remain era-neutral.

**p3. `complex-debugging.md` lines 88-100 shows quickstart commands assuming Python or JavaScript stacks.** The pack's premise is *language-agnostic*. The static-check examples are reasonable as illustrative but slightly contradict the "agnostic" framing. Optional: add a one-liner saying "examples shown in Python/JS — substitute your stack's equivalents".

---

## 6. Recommended Actions

Listed in priority order. None are blocking; pack is shippable as-is at v1.0.1.

| # | Priority | Action | Effort | Version impact |
|---|---|---|---|---|
| 1 | **Major** | Create `/home/john/skillpacks/.claude/commands/software-engineering.md` slash-command wrapper following the `python-engineering.md` template. Use slug `software-engineering` (matches `using-X` skill ID). | 30-60 min (one file, prose-only) | Patch bump (1.0.2) since pack content unchanged, or Minor (1.1.0) per repo convention if wrapper-add is treated as exposure expansion |
| 2 | Minor | Add reciprocal cross-references to `superpowers:*` skills in the relevant Integration / Handoffs tables: complex-debugging → `superpowers:systematic-debugging`; systematic-refactoring → `superpowers:test-driven-development`; code-review-methodology → `superpowers:requesting-code-review`, `superpowers:receiving-code-review`; SKILL.md overall → `superpowers:writing-plans` for multi-step engineering work. | 15-30 min | Patch bump |
| 3 | Minor | Reconcile `plugin.json` description with `marketplace.json` description (pick one phrasing). | 5 min | Patch bump |
| 4 | Polish | Sharpen the "Rubber stamping" anti-pattern row in `code-review-methodology.md` to use a verbatim-thought style consistent with other anti-pattern tables in the pack. | 5 min | Patch bump |
| 5 | Polish | Add a one-line "examples shown in Python/JS" disclaimer near the Phase-0 static-check block in `complex-debugging.md` to keep the language-agnostic framing crisp. | 5 min | No bump needed |
| 6 | (Future) | Re-run Stage-3 behavioral tests as actual subagent dispatch (rubber the paper-walkthrough). Especially valuable for description-based discovery testing of the router. | 1-2 hours | No bump — testing only |

### Suggested commit grouping (if all are accepted)
1. **feat(axiom-engineering-foundations): add slash-command wrapper, cross-link superpowers** — actions 1, 2 → Minor bump → 1.1.0.
2. **chore(axiom-engineering-foundations): description and anti-pattern polish** — actions 3, 4, 5 → patch.

(Marketplace catalog version bump rules per CLAUDE.md: this is a "Medium" change → Minor bump.)

---

## 7. Reviewer Notes

- **Behavioral test fidelity (§4) is the lowest tier.** The rubric (`testing-skill-quality.md` lines 80-92) flags subagent dispatch as the default mechanism; in-session paper walkthroughs are explicitly described as "lowest fidelity" because the current session's prior context can mask discovery failures. I did not dispatch a subagent. The findings above (5 passes, 0 fails) are *consistent with the skill content* but should not be treated as empirical activation evidence. A maintenance pass that adds the slash-command wrapper (action 1) is a natural time to also re-run these tests properly.
- **No agents in this pack** — SME-agent-protocol checks (rubric `analyzing-pack-domain.md` lines 73-78, `reviewing-pack-structure.md` lines 95-99) are N/A.
- **No commands in this pack** — command-frontmatter style checks are N/A.
- **No hooks** — event/matcher checks are N/A.
- **Pack matches its `Pass` ceiling.** Per the scorecard rubric (`reviewing-pack-structure.md` lines 46-52), Pass is defined as "comprehensive coverage; no major gaps or duplicates; components appropriately typed; metadata current." The only Major finding is the missing slash-command wrapper, which is an *exposure* gap (the marketplace's discoverability convention), not a *coverage* gap. The pack would arguably score Pass without M1; with M1 it drops one notch to "Pass with one Major" — borderline Minor depending on how strictly one interprets the slash-command convention.
- **Strongest single piece of content in the pack:** `complex-debugging.md`'s "Investigation Reset Protocol" (lines 570-617). It names the exact rationalization patterns ("checked logs", "added retry logic", "increased timeouts", "rolled back", "googled it") and the cognitive failure mode behind each, then prescribes the reset. This is the kind of behavioral defense the rubric prizes most highly.
- **Lowest-confidence claim in this review:** the assertion that the missing slash-command wrapper is "Major" (rather than "Minor"). CLAUDE.md treats wrappers as a stated convention ("All router skills (`using-X` skills) are available as slash commands in `.claude/commands/`") but does not call it a hard requirement. If the maintainer treats it as advisory, the pack is effectively all-Minor and could score a clean Pass.
- **No project-specific data observed in the pack.** All examples are generic (orders, users, checkout, payment) and language-agnostic — no leak of government or proprietary content, consistent with the user feedback note in memory.
