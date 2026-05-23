---
description: Use when managing multi-document sets as wikis — manifest-driven architecture, derivation discipline, terminology and claim registries, reading paths, change propagation, and governance for standards suites, policy frameworks, and technical documentation collections
---

# Wiki Manager Routing

**Managing a document *set* is a different discipline from writing a document. Individual document quality is necessary but not sufficient — the set must be consistent, navigable, self-sufficient at each depth tier, and maintainable over time. For single-document craft (ADRs, APIs, runbooks, style) use `/technical-writer` instead. For reader-panel simulation use `/panel-review`.**

Use the `using-wiki-manager` skill from the `muna-wiki-management` plugin to route to the right specialist sheet. Content authority lives in `plugins/muna-wiki-management/skills/using-wiki-manager/SKILL.md` — this wrapper is a thin pointer.

## Sheets

- **document-set-architecture** — manifest, derivation graph, depth tiers (0/1/2), root vs derivative classification, multi-root reconciliation patterns (Partition / Primary-Supplementary / Synthesis), conflict resolution
- **reading-path-design** — persona registry, coverage matrix, time-based path estimation, stepping-stone validation, gap classification
- **content-derivation** — three derivation modes (distillation, translation, extraction), derivation recipes, self-sufficiency test, deferral audit — the anti-laziness discipline
- **cross-document-consistency** — terminology and claim registries with granularity heuristics, link integrity checks, the eleven defect types
- **document-evolution** — change classification (cosmetic / clarification / substantive / structural), impact trace via derivation graph, git integration, external dependency tracking, version coherence, deprecation
- **document-governance** — ownership model, four-point LLM-as-steward trust model, six quality gates, review triggers, escalation

## Commands

- `/muna-wiki-management:onboard-docset` — bootstrap an existing unstructured doc set: inventory → derivation discovery → manifest → minimum viable registries → reading paths → baseline audit → present
- `/muna-wiki-management:audit-docset` — full-set health check against the eleven defect types: consistency, derivation integrity, path completeness, link integrity, triaged output
- `/muna-wiki-management:derive-content` — single-derivative authoring with recipe-first discipline, self-sufficiency test, deferral audit, mandatory human surfacing
- `/muna-wiki-management:propagate-change` — change-driven targeted update: classify, impact trace, re-derive affected sections, verify consistency, surface gate results

## Agents

- `reference-sheet-writer` — producer; enforces five non-negotiable anti-laziness rules (no deferral phrases, every section has substance, no thin connecting prose, quantitative claims present not referenced, worked examples mandatory)
- `self-sufficiency-reviewer` — reviewer with deliberately restricted tool access (`Read`, `Glob` only — no `Grep`, no `Bash`); judges by comprehensibility not coverage; follows SME Agent Protocol

## Cross-references

- Individual document quality (ADRs, APIs, runbooks, register translation) → `/technical-writer`
- Panel-testing how documents land with audiences after structural soundness → `/panel-review`
- Pandoc / Typst rendering of the assembled doc set → `/document-designer`
