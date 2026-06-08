# Changelog — axiom-product-management

All notable changes to this plugin are documented here. Versions are independent of the marketplace catalog version.

## [0.1.0] — 2026-06-09

### Added

- Initial release. Product management as **standing ownership** — for a Claude that takes control of a software product and owns it end-to-end across sessions.
- **Router** (`using-product-management`) — intent detection, routing-by-symptom, the seam to sibling packs (program-management / planning / solution-architect / lyra-ux-designer), and pressure-resistance discipline.
- **8 reference sheets:**
  - `product-ownership-operating-model` — the resume → orient → decide → dispatch → accept → checkpoint loop, session protocols, the authority boundary.
  - `product-state-and-continuity` — the git-versioned product workspace (vision, roadmap, Product Decision Records, current-state, metrics), PDR template, resume/checkpoint protocols, tracker-adapter contract.
  - `product-discovery-and-opportunity` — opportunity assessment, JTBD, problem validation, business case; the "is this worth solving" decision.
  - `vision-strategy-and-roadmap` — vision, positioning, north-star, strategic bets, roadmap as intent (sequencing routed to program-management).
  - `prd-and-acceptance-criteria` — problem statements, PRDs, falsifiable acceptance criteria; the seam to planning and solution-architect.
  - `delivery-orchestration-and-acceptance` — decompose → dispatch → verify-it-shipped → accept against criteria.
  - `product-metrics-and-experimentation` — north-star/input/guardrail metrics, instrumentation, A/B and hypothesis design, when to kill a bet.
  - `product-anti-patterns` — build trap, feature factory, vanity metrics, roadmap-as-promise, HiPPO/stakeholder capture, autonomy overreach, acceptance gaps, decision-without-provenance.
- **3 commands:** `/own-product` (bootstrap/resume ownership), `/write-prd` (problem → PRD with falsifiable acceptance), `/product-checkpoint` (write state back, emit status).
- **2 agents:** `product-shaping-architect` (forward-design producer), `product-decision-critic` (red-team critic). Both follow the SME Agent Protocol.
