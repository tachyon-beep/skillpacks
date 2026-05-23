---
description: Building static analyzers as engines, not running them as users - AST visitation, taint lattices, three-phase inference, plugin architecture, false-positive economics, static-vs-runtime boundary, callgraph construction, cross-module flow, decorator-as-assertion, manifest-driven configuration, SARIF/CI integration, scaling, LLM-assisted explanation
---

# Static Analysis Engineering Routing

**Engineering pack: how to build the analyzer. For consuming an existing analyzer's output to map a codebase, use `/system-archaeologist` instead. For running existing tools (ruff, mypy) in CI use `/python-engineering` or `/ordis-quality-engineering:static-analysis-integration`.**

Use the `using-static-analysis-engineering` skill from the `axiom-static-analysis-engineering` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-static-analysis-engineering/skills/using-static-analysis-engineering/SKILL.md` - this wrapper is a thin pointer.

## Sheets

### Architectural spine
- **ast-visitation-patterns** - substrate; visitor/walker/transformer, parent tracking, structural vs lossless ASTs
- **taint-lattice-design** - abstract domain; lattice algebra, partial order, monotonicity, finite-height termination
- **three-phase-inference** - algorithm; variable → function summary → inter-procedural worklist with termination proof

### Operational reality
- **plugin-architecture-for-analyzer-rules** - extension surface; discovery, lifecycle, metadata schema, conflict resolution
- **false-positive-economics** - suppression lifecycle as auditable decisions, waiver expiry, FP-rate budget
- **static-vs-runtime-tradeoffs** - Rice ceiling, dual enforcement, cost model for choosing the enforcement layer

### Boundary discipline
- **callgraph-construction** - resolution rungs (name/CHA/RTA/VTA/k-CFA), dynamic-feature handling, conservative `top`
- **cross-module-flow-analysis** - boundary semantics, stub libraries, framework callbacks, FFI
- **decorator-as-assertion** - runtime + static dual contract, descriptor pattern, recognition registry, disagreement modes

### Operations
- **manifest-driven-configuration-with-coherence-validation** - layered overlays, coherence validation, drift detection, audit metadata
- **sarif-emission-and-ci-integration** - SARIF 2.1.0 schema, exit-code semantics, fingerprint stability, suppression round-trip
- **scaling-to-large-codebases** - cache-key composition, reverse-edge index, parallel worklist, partition strategies, soundness floor
- **llm-assisted-rule-explanation** - analyzer = truth, model = translator; review gate; prompt-injection threat model

## Commands

- `/axiom-static-analysis-engineering:scaffold-analyzer` - greenfield/brownfield scaffold; tier-conditional file plan from the spec set
- `/axiom-static-analysis-engineering:design-tier-model` - six-round elicitation producing draft `02-abstract-domain-spec.md`
- `/axiom-static-analysis-engineering:design-rule-set` - seven-round rule sourcing → metadata → manifest → fixtures → conflict review

## Agents

- `rule-designer` - producer SME for analyzer rules; refuses tier inventions and unfalsifiable rules
- `false-positive-analyst` - operational SME for suppression triage; refuses anecdotal triage without metric data

Both agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Consuming analyzer output to map a codebase → `/system-archaeologist`
- Running tools (ruff, mypy) in CI → `/python-engineering`, `/ordis-quality-engineering:static-analysis-integration`
- Suppressions as audited decisions, waiver-expiry mechanism → `/audit-pipelines`
- Static enforcement of policy at trust boundaries → `/security-architect`
- Lifecycle ownership of analyzer build artifacts → `/sdlc-engineering`
