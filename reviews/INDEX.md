# Skillpacks Marketplace — Consolidated Review Index

**Reviewed:** 2026-05-22
**Scope:** 42 plugins (40 domain + 2 meta)
**Method:** `meta-skillpack-maintenance:using-skillpack-maintenance` Stages 1-4 (Investigation / Structure / Behavioral / Synthesis), per-pack subagent, report-only

Reports are one-per-pack at `reviews/<pack>.md`. This index synthesises overall ratings, cross-pack patterns, and prioritized action items.

---

## Verdict Distribution

| Rating | Count | Share |
|--------|------:|------:|
| **Pass** (no fixes required) | 4 | 9.5% |
| **Pass with Minor/Polish** | 12 | 28.6% |
| **Minor overall** | 5 | 11.9% |
| **Major overall** | 20 | 47.6% |
| **Critical / Scaffold-only** | 1 | 2.4% |

Most packs have substantively strong *content*. The bulk of Major ratings come from one recurring class of defect (see "Cross-Pack Patterns") that is mechanically fixable.

---

## Per-Pack Summary

Verdicts are the subagent's overall scorecard rating, distilled to one line. Click through to the per-pack report for full evidence.

### Axiom (engineering disciplines) — 18 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [axiom-audit-pipelines](axiom-audit-pipelines.md) | 1.0.2 | **Pass** | No fixes required; ships as-is. |
| [axiom-determinism-and-replay](axiom-determinism-and-replay.md) | 1.0.3 | **Pass** | Five Polish items for v1.1. |
| [axiom-devops-engineering](axiom-devops-engineering.md) | 1.1.4 | **Minor** | Catalog description oversells; broken `glob.glob` in `commands/design-deployment.md:371-383`. |
| [axiom-embedded-database](axiom-embedded-database.md) | 0.1.0 | **Pass + 1 Major** | Missing `.claude/commands/embedded-database.md` wrapper. |
| [axiom-engineering-foundations](axiom-engineering-foundations.md) | — | **Pass + 1 Major** | Only `using-X` router in marketplace without a slash wrapper. |
| [axiom-mcp-engineering](axiom-mcp-engineering.md) | 0.1.0 | **Scaffold (mechanically Critical)** | Not registered in marketplace.json; no wrapper; sheets deferred. Honest scaffold. |
| [axiom-planning](axiom-planning.md) | 1.1.1 | **Major** | 1 Critical: 5 reviewer/synthesizer agents lack SME-protocol compliance. No router/wrapper. No plan-revision workflow. |
| [axiom-procedural-architecture](axiom-procedural-architecture.md) | 0.1.1 | **Pass + 1 Major** | Missing slash wrapper. |
| [axiom-pyo3-interop](axiom-pyo3-interop.md) | 0.1.2 | **Pass** | 2 Minor + 3 Polish. |
| [axiom-python-engineering](axiom-python-engineering.md) | 1.5.0 | **Minor** | Wrapper omits `textual-tui-development`; router `description` not "Use when…". |
| [axiom-rust-engineering](axiom-rust-engineering.md) | 1.0.2 | **Major** | Scope leak into `axiom-rust-workspaces` territory (~130 lines, no redirect); missing wrapper (asymmetric with siblings). |
| [axiom-rust-workspaces](axiom-rust-workspaces.md) | 1.0.2 | **Pass** | 5 additive Polish items for an optional v1.1.0. |
| [axiom-sdlc-engineering](axiom-sdlc-engineering.md) | 1.1.2 | **Major** | Malformed command (frontmatterless, nested `COMMAND.md`); 228-line bloated wrapper; `.test-scenarios/` + `SKILL.md.backup-*` leak into distribution; broken doc reference cited from most SKILL.md. |
| [axiom-solution-architect](axiom-solution-architect.md) | 1.0.1 | **Pass** | 2 Minor (10 vs 11 failure-mode count drift; "9 skills" claim). |
| [axiom-static-analysis-engineering](axiom-static-analysis-engineering.md) | 0.2.1 | **Pass + 1 Major** | Slash wrapper still advertises v0.1 ("commands deferred to v0.2"). |
| [axiom-system-archaeologist](axiom-system-archaeologist.md) | 1.6.1 | **Pass** | 2 Minor — wrapper drifted from router. |
| [axiom-system-architect](axiom-system-architect.md) | 1.1.4 | **Minor** | Router calls shipped `prioritizing-improvements` skill "Future"; SKILL.md says v1.0.0 while plugin.json says 1.1.4. |
| [axiom-web-backend](axiom-web-backend.md) | 1.2.0 | **Pass + Minor** | Copy-paste cross-ref: `express-development.md:24` points to `axiom-python-engineering`. 4 coverage gaps vs. marketing. |

### Bravos (game dev) — 2 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [bravos-simulation-tactics](bravos-simulation-tactics.md) | 1.1.5 | **Minor** | 909-line wrapper inlines two reference sheets — three-copy maintenance hazard. |
| [bravos-systems-as-experience](bravos-systems-as-experience.md) | 1.1.5 | **Pass + 1 Major** | Skill-count drift (8 vs 9 across 4 surfaces). |

### Lyra (UX/site/writing) — 3 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [lyra-creative-writing](lyra-creative-writing.md) | 0.2.0 | **Pass + 1 Major** | Wrapper still claims v0.1 ("8 agents", "13 sheets", no genre annex). 11 agents lack `model:`. |
| [lyra-site-designer](lyra-site-designer.md) | 1.1.0 | **Major** | Missing `.claude/commands/site-designer.md` wrapper (the only lyra pack without one). |
| [lyra-ux-designer](lyra-ux-designer.md) | 1.3.0 | **Major** | Wrapper omits v1.3 flagship (AI-experience patterns); WCAG version drift (2.1 vs 2.2). |

### Meta — 2 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [meta-skillpack-maintenance](meta-skillpack-maintenance.md) | 2.1.0 | **Major (self-applied)** | Critical: `superpowers:writing-skills` cited 8x but never declared as external dependency — bootstrap/circularity problem. No own wrapper. |
| [meta-sme-protocol](meta-sme-protocol.md) | 1.1.0 | **Pass + 1 Major** | No copy-pasteable canonical citation snippet — 30+ adopters citing by imitation rather than a single source of truth. |

### Muna (documentation) — 4 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [muna-document-designer](muna-document-designer.md) | 1.1.0 | **Major** | Missing wrapper. |
| [muna-panel-review](muna-panel-review.md) | 0.3.2 | **Major** | Distribution defect: no `commands/` dir in plugin; commands at repo root don't travel via `/plugin install`. Two wrappers are informational prose not active dispatchers. Orphan `skills/using-panel-review/`. |
| [muna-technical-writer](muna-technical-writer.md) | 1.4.1 | **Major** | Wrapper + router carry "Phase 1 / Coming Soon" block for already-shipped skills. Marketplace.json count drift. |
| [muna-wiki-management](muna-wiki-management.md) | 1.0.1 | **Major** | `self-sufficiency-reviewer` agent (a reviewer) lacks SME-protocol compliance. Wrapper is verbatim copy with relative-link references that break at the wrapper's path. |

### Ordis (security & quality) — 2 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [ordis-quality-engineering](ordis-quality-engineering.md) | 2.3.0 | **Major** | Mature 21-sheet pack — single defect: missing wrapper. |
| [ordis-security-architect](ordis-security-architect.md) | 1.2.0 | **Major** | Stale skill name `architecture-security-review` referenced 11× (file is `security-architecture-review.md`). Wrapper missing LLM + supply-chain extensions. |

### Yzmir (AI/ML) — 11 packs

| Pack | Version | Overall | Top finding |
|------|--------:|---------|-------------|
| [yzmir-ai-engineering-expert](yzmir-ai-engineering-expert.md) | 1.1.0 | **Major** | Router misses `yzmir-morphogenetic-rl` (10th sibling, invisible to routing). plugin.json says "9 packs". Wrapper drifts from SKILL.md. |
| [yzmir-deep-rl](yzmir-deep-rl.md) | 1.3.0 | **Pass + 2 Major** | Wrapper missing `counterfactual-reasoning` and modern algorithm blurbs. Asymmetric SME template adoption across agents. |
| [yzmir-dynamic-architectures](yzmir-dynamic-architectures.md) | 1.2.1 | **Major** | Missing wrapper — sibling `morphogenetic-rl.md` references `/dynamic-architectures` as a live link. |
| [yzmir-llm-specialist](yzmir-llm-specialist.md) | 1.2.0 | **Major** | Wrapper describes pre-2026 7-skill router (no MCP, no reasoning split, no prompt caching, stale context lengths). |
| [yzmir-ml-production](yzmir-ml-production.md) | 1.2.0 | **Pass + 1 Major** | Wrapper materially out of sync — missing all LLM serving + observability content (vLLM/SGLang/Phoenix/Langfuse). |
| [yzmir-morphogenetic-rl](yzmir-morphogenetic-rl.md) | 1.2.0 | **Pass** | Cleanest review of the session. 1 Minor + 3 Polish. |
| [yzmir-neural-architectures](yzmir-neural-architectures.md) | 1.2.0 | **Major** | 4 metadata-drift items: stale 2021-era router description, 467-line wrapper duplicates SKILL.md, marketplace.json stale, "8" vs "9" architecture count self-contradiction. |
| [yzmir-pytorch-engineering](yzmir-pytorch-engineering.md) | 1.2.0 | **Major** | Wrapper is pre-1.2.0: still routes `torch.cuda.amp`, missing FSDP2/FlexAttention/torch.compile/CUDA Graphs/NVTX triggers, "Phase 1 - Standalone" block. |
| [yzmir-simulation-foundations](yzmir-simulation-foundations.md) | 1.2.0 | **Pass + Minor** | Wrapper exists but near-duplicates router (493 vs 472 LOC). 5 Minor. |
| [yzmir-systems-thinking](yzmir-systems-thinking.md) | 1.1.4 | **Minor** | Vestigial `glob.glob` Python snippets in 2 commands. Missing archetype row. Skill-count inflation. |
| [yzmir-training-optimization](yzmir-training-optimization.md) | 1.2.0 | **Major** | Wrapper is pre-1.2.0: missing Lion/Sophia/Muon, FP8, WSD, muP, ZeRO/FSDP, DPO routing. Marketplace.json stale. |

---

## Cross-Pack Patterns

### Pattern 1 — Slash-command wrapper drift (THE dominant defect)

**Roughly 25 of 42 packs (≈60%) have a wrapper problem.** Three sub-patterns:

1. **Missing wrapper entirely** (≥9 packs)
   `axiom-embedded-database`, `axiom-engineering-foundations`, `axiom-mcp-engineering`, `axiom-planning`, `axiom-procedural-architecture`, `axiom-rust-engineering`, `lyra-site-designer`, `muna-document-designer`, `ordis-quality-engineering`, `yzmir-dynamic-architectures`. CLAUDE.md treats this as a contract; missing wrappers mean the router is not user-invokable as `/<name>`.

2. **Stale wrapper carrying a prior version's content** (≥10 packs)
   `axiom-python-engineering` (omits `textual-tui-development`), `axiom-static-analysis-engineering` ("commands deferred to v0.2"), `axiom-system-archaeologist`, `lyra-creative-writing` (claims v0.1), `lyra-ux-designer` (missing v1.3 AI-experience), `muna-technical-writer` ("Phase 1 / Coming Soon"), `ordis-security-architect`, `yzmir-deep-rl`, `yzmir-llm-specialist`, `yzmir-ml-production`, `yzmir-pytorch-engineering`, `yzmir-training-optimization`.

3. **Bloated/duplicating wrapper** (≥6 packs)
   `axiom-sdlc-engineering` (228 lines), `axiom-system-architect`, `bravos-simulation-tactics` (909 lines, inlines 2 sheets), `bravos-systems-as-experience`, `muna-wiki-management` (broken relative links), `yzmir-neural-architectures` (467 lines), `yzmir-simulation-foundations` (493 vs 472).

**Root cause hypothesis.** No documented canonical-source rule for wrapper vs. SKILL.md, and no test gate that asserts wrapper content is consistent with (or a strict subset of) router content. Wrappers were created by copy-paste at one moment in time and have not tracked subsequent router edits.

**Recommended systemic fix:** add a `tools/check-wrappers.py` (or similar) that runs in CI and asserts (a) every `using-X` router has a wrapper, (b) wrapper claims about skill counts / file names match the actual router state.

### Pattern 2 — "Use when…" description convention not adopted

The meta-skill rubric documents "Use when…" as the dominant discoverability convention. ≥12 packs reviewed have routers whose `description:` describes what the skill *is* rather than when to load it — degrades fresh-context auto-discovery. Notable: `bravos-systems-as-experience`, `muna-technical-writer`, `ordis-security-architect`, `yzmir-deep-rl`, `yzmir-systems-thinking`, both lyra packs, multiple yzmir packs.

### Pattern 3 — Skill-count and metadata drift across surfaces

Counts disagree between `plugin.json`, `marketplace.json`, router SKILL.md header, and the catalog table — at least 9 packs (`axiom-solution-architect`, `bravos-systems-as-experience`, `muna-technical-writer`, `muna-panel-review`, `yzmir-ai-engineering-expert`, `yzmir-deep-rl`, `yzmir-dynamic-architectures`, `yzmir-ml-production`, `yzmir-neural-architectures`, `yzmir-pytorch-engineering`, `yzmir-simulation-foundations`, `yzmir-systems-thinking`, `yzmir-training-optimization`). Routinely the issue is conflating "router + N reference sheets" with "N+1 skills."

**Recommended systemic fix:** a CI script that counts skills/sheets/commands/agents and emits a canonical count, comparing against all four surfaces.

### Pattern 4 — SME-protocol non-adoption in reviewer agents

SME-protocol compliance is the standard for reviewer/auditor agents in this marketplace. Several packs ship reviewer agents that omit it:
- `axiom-planning` — 5 reviewer/critic/synthesizer agents, none declare protocol (Critical)
- `lyra-creative-writing` — 8 reviewer-class agents, undeclared opt-out
- `muna-wiki-management` — `self-sufficiency-reviewer` is a reviewer, not protocol-compliant
- `yzmir-deep-rl` — asymmetric: one agent has the SME template, the other doesn't

### Pattern 5 — Distribution / packaging defects

- **`muna-panel-review`** — no `commands/` directory inside the plugin; the slash commands exist only at repo-root `.claude/commands/` and so do *not* travel via `/plugin install`. Two of those wrappers are informational prose, not dispatchers.
- **`axiom-sdlc-engineering`** — `.test-scenarios/` directory and `SKILL.md.backup-*` files ship inside the plugin distribution.
- **`axiom-mcp-engineering`** — not registered in marketplace.json (intentional during scaffold, but invisible to users).
- **`axiom-rust-engineering`** — empty `hooks/hooks.json` placeholder.

### Pattern 6 — Inter-pack boundary drift / cross-reference bugs

- **`axiom-rust-engineering`** has ~130 lines of workspace material that belongs in `axiom-rust-workspaces` and no in-sheet redirect, despite the router thrice claiming a "single-crate-shaped" contract.
- **`axiom-web-backend`** — `express-development.md:24` cross-references `axiom-python-engineering` for TypeScript patterns (copy-paste from a sibling sheet).
- **`ordis-security-architect`** — references stale skill name `architecture-security-review` 11+ times; file actually named `security-architecture-review.md`.
- **`yzmir-ai-engineering-expert`** — entirely misses `yzmir-morphogenetic-rl` (the 10th yzmir pack).
- **`yzmir-deep-rl`** — broken `/deep-rl:new-experiment` pointer; `offline-rl-methods` vs `offline-rl` filename drift.

### Pattern 7 — Self-application surfaces meta-issues

Reviewing the meta packs against their own rubric produced two findings worth their own thread:
- `meta-skillpack-maintenance` cites `superpowers:writing-skills` 8 times as mandatory without declaring it as an external dependency — a bootstrap problem.
- `meta-sme-protocol` has no copy-pasteable canonical citation snippet, so the ~30 adopter agents are matching by imitation rather than referencing a single source of truth.

---

## Priority Recommendations

### P0 — fix this week (~1-2 hours total)

1. **Create the 9 missing slash-command wrappers** (`axiom-embedded-database`, `axiom-engineering-foundations`, `axiom-planning`, `axiom-procedural-architecture`, `axiom-rust-engineering`, `lyra-site-designer`, `muna-document-designer`, `ordis-quality-engineering`, `yzmir-dynamic-architectures`). Pattern: thin redirect to the router. Use any existing wrapper as the template — `lyra-ux-designer.md` is a reasonable model.
2. **Fix `axiom-sdlc-engineering` packaging defects:** delete `.test-scenarios/` and `SKILL.md.backup-*` from the distribution, fix the malformed `commands/using-sdlc-engineering/COMMAND.md`.
3. **Add `muna-panel-review/commands/` directory** with the three command files so they travel via marketplace install.
4. **Register `axiom-mcp-engineering` in marketplace.json** if it's intended to be discoverable now; otherwise add a `private` marker.

### P1 — fix this month

5. **Wrapper canonicalisation pass** — for each pack with a stale or duplicating wrapper, decide on the canonical source (recommend: SKILL.md is source of truth; wrapper is a thin redirect). Apply across all ≥16 affected packs. Add CI check.
6. **SME-protocol compliance pass** on `axiom-planning` (5 agents), `lyra-creative-writing` (8 agents), `muna-wiki-management`, `yzmir-deep-rl` (1 agent missing template).
7. **Cross-reference bugs:** fix `ordis-security-architect`'s stale skill name (11× rename), `axiom-web-backend`'s `express-development.md:24` cross-pack typo, `yzmir-ai-engineering-expert` to include `yzmir-morphogenetic-rl`.

### P2 — next minor releases

8. **"Use when…" description convention** — sweep ≥12 packs; update routers.
9. **Skill-count drift sweep** — reconcile plugin.json / marketplace.json / SKILL.md header / catalog table on all ≥12 affected packs. Add a CI script that emits canonical counts.
10. **Boundary cleanup for `axiom-rust-engineering` vs `axiom-rust-workspaces`** — move the ~130 lines of workspace material into the workspaces pack with a redirect.
11. **Adopt `meta-skillpack-maintenance` self-fixes:** declare `superpowers:writing-skills` as external dependency; create canonical citation snippet for `meta-sme-protocol`.

### P3 — opportunistic

12. The Polish backlogs across the cleanest packs (`axiom-audit-pipelines`, `axiom-determinism-and-replay`, `axiom-rust-workspaces`, `axiom-pyo3-interop`, `yzmir-morphogenetic-rl`) — bundle into next minor releases.

---

## What Went Well

This is a substantively mature marketplace. Recurring strengths observed across reviews:

- **Content quality and currency.** Multiple packs are explicitly calibrated to 2025-2026 toolchains (WCAG 2.2, Typst 0.14, PyTorch 2.9, GRPO/DreamerV3/REDQ, vLLM/SGLang, EU AI Act 2024/1689, SLSA v1.1, OWASP LLM 2025).
- **Pressure-resistance is operationalised** rather than aspirational. Multiple packs name the rationalisations they expect ("just do it quickly", "too simple for the full process", "team says it's fine") and pre-empt them inside the relevant sheets.
- **SME Agent Protocol works** — where adopted, agents reliably emit the four-section output contract (Confidence / Risk / Information Gaps / Caveats), and the convention is recognised by adopters in spite of the canonical-citation gap noted above.
- **Sibling-not-nested discipline holds** in the rust pair, the morphogenetic/dynamic pair, and the simulation triad (axiom-determinism-and-replay × yzmir-simulation-foundations × bravos-simulation-tactics).
- **Honest scaffolds.** `axiom-mcp-engineering` ships as router-only with sheets deferred and says so plainly. Better than shipping vapor.

---

## Reviewer Notes

- All reviews are behaviorally tested *on paper* (predictive walkthrough of scenarios) rather than executed live. The rubric's lowest-fidelity tier. A future audit pass with live subagent gauntlets would tighten confidence on the Pass-rated packs.
- Reports are read-only. No skill content was edited during this review pass.
- Each per-pack review cites file paths and line numbers as evidence — use them when scoping fixes.
- The CI / drift-detection scripts proposed under P0–P2 would prevent ≥80% of the findings from recurring.
