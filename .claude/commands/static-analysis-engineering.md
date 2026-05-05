---
description: Building static analyzers as engines, not running them as users - AST visitation, taint lattices, three-phase inference, plugin architecture, false-positive economics, static-vs-runtime boundary
---

# Static Analysis Engineering Routing

**Engineering pack: how to build the analyzer. For consuming an existing analyzer's output to map a codebase, use `/system-archaeologist` instead.**

Use the `using-static-analysis-engineering` skill from the `axiom-static-analysis-engineering` plugin to route to the right specialist sheet.

## v0.1 Sheets (architectural spine + operational reality)

- **ast-visitation-patterns** - substrate; how the engine walks the AST
- **taint-lattice-design** - abstract domain; the set of values the analyzer tracks
- **three-phase-inference** - algorithm; variable → function → callgraph propagation with termination
- **plugin-architecture-for-analyzer-rules** - extension surface; how rules compose
- **false-positive-economics** - operational reality; suppression lifecycle as auditable decisions
- **static-vs-runtime-tradeoffs** - scope discipline; when each enforcement layer is right

## Deferred to v0.2

Commands (`/scaffold-analyzer`, `/design-tier-model`, `/design-rule-set`), agents (`rule-designer`, `false-positive-analyst`), and 7 additional sheets (callgraph construction, cross-module flow, decorator-as-assertion, manifest-driven config, SARIF/CI integration, scaling/incremental analysis, LLM-assisted rule explanation) are scheduled for v0.2.0. v0.1 is self-coherent — a user can produce artifacts 00-06 + 99 and pass all 11 consistency-gate checks from shipped sheets alone.

## Cross-references

- Consuming analyzer output to map a codebase → `axiom-system-archaeologist`
- Running tools (ruff, mypy) in CI → `axiom-python-engineering` and `ordis-quality-engineering:static-analysis-integration`
- Suppressions as audited decisions → `axiom-audit-pipelines`
- Static enforcement of policy at trust boundaries → `ordis-security-architect`
