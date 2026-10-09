---
name: using-static-analysis-engineering
description: "Use when building analyzers or rules whose traversal, abstract domain, unknown-call handling, termination, incremental cache or findings output must have explicit guarantees."
---

# Static analysis engineering

Specify the analyzer claim and its limits. A syntactic rule, abstract interpreter and probabilistic code reviewer have different contracts; do not promise soundness from a shared label.

## Work from the affected contract

1. Name the property, accepted language/framework surface, verdict consumers and soundness/completeness boundary. Select the smallest architecture that supports it.
2. For flow inference, define the abstract domain, joins, unknown calls, summaries and termination. Use staged inference where it fits; simple pattern matchers do not require a lattice.
3. Preserve source positions and rule identity. Check third-party boundaries, decorator/runtime agreement and configuration coherence against explicit models.
4. For incrementality, make cache dependencies and invalidation include all meaning-affecting inputs. For CI/SARIF, verify stable identity, exit semantics and suppression lifecycle.
5. Exercise positive, negative, adversarial and known-unknown corpus cases. Record what a passing corpus does and does not establish; measure false-positive costs.

## Scope and completion

Deliver the rule/engine contract, bounded design or patch, test-corpus evidence and unmodeled surface. An LLM explanation may annotate a deterministic finding; do not silently replace the verdict producer. A clean audit is supported by scope and evidence, never a quota of faults.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| AST Visitation Patterns | [ast-visitation-patterns.md](ast-visitation-patterns.md) |
| Callgraph Construction | [callgraph-construction.md](callgraph-construction.md) |
| Cross-Module Flow Analysis | [cross-module-flow-analysis.md](cross-module-flow-analysis.md) |
| Decorator as Assertion | [decorator-as-assertion.md](decorator-as-assertion.md) |
| False-Positive Economics | [false-positive-economics.md](false-positive-economics.md) |
| LLM-Assisted Rule Explanation | [llm-assisted-rule-explanation.md](llm-assisted-rule-explanation.md) |
| Manifest-Driven Configuration with Coherence Validation | [manifest-driven-configuration-with-coherence-validation.md](manifest-driven-configuration-with-coherence-validation.md) |
| Plugin Architecture for Analyzer Rules | [plugin-architecture-for-analyzer-rules.md](plugin-architecture-for-analyzer-rules.md) |
| SARIF Emission and CI Integration | [sarif-emission-and-ci-integration.md](sarif-emission-and-ci-integration.md) |
| Scaling to Large Codebases | [scaling-to-large-codebases.md](scaling-to-large-codebases.md) |
| Static vs Runtime Tradeoffs | [static-vs-runtime-tradeoffs.md](static-vs-runtime-tradeoffs.md) |
| Taint Lattice Design | [taint-lattice-design.md](taint-lattice-design.md) |
| Three-Phase Inference | [three-phase-inference.md](three-phase-inference.md) |

## Optional task entry points

- [design-rule-set](../../commands/design-rule-set.md): Design rule set for the affected static analysis engineering contract, with scoped source and verification evidence.
- [design-tier-model](../../commands/design-tier-model.md): Design tier model for the affected static analysis engineering contract, with scoped source and verification evidence.
- [scaffold-analyzer](../../commands/scaffold-analyzer.md): Scaffold analyzer for the affected static analysis engineering contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [false-positive-analyst](../../agents/false-positive-analyst.md), [rule-designer](../../agents/rule-designer.md). No fixed reviewer count is required.
