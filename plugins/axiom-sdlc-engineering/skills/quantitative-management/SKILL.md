---
name: quantitative-management
description: "Use when a delivery or process decision needs reproducible measurements, baseline comparisons or statistical uncertainty under an explicit policy."
---

# Quantitative Management

Measurements answer a named decision. Do not collect data or impose targets solely to claim maturity.

## Workflow

1. Name the decision, outcome/guardrail, owner and action a result could change.
2. Define the measure, unit, population, numerator/denominator, data source/window, exclusions and missingness. Validate adapters against actual records.
3. Inspect data quality and process comparability before computing baselines or trends. Separate changes in instrumentation or work mix from performance changes.
4. Use an appropriate calculation with explicit assumptions. Distinguish descriptive statistics, forecasts, uncertainty intervals and causal evidence.
5. Publish reproducible inputs/query/calculation and limitations. Historical throughput may support forecast scenarios; illustrative numbers are not live evidence.
6. Review the measure for gaming, burden and unintended effects. Do not grade individuals or enforce universal DORA/coverage/defect ratios absent a selected, justified policy.

## Evidence

Definition and provenance, observed sample/window, calculation, uncertainty, interpretation, decision/action and known omissions. For control charts, justify stability/subgrouping/distribution assumptions; a threshold crossing is a signal to investigate, not proof of cause.

## Optional references

[Planning](measurement-planning.md), [baselines](process-baselines.md), [analysis](statistical-analysis.md), [quantitative decisions](quantitative-management.md), [delivery measures](dora-metrics.md), [domain measures](key-metrics-by-domain.md), [tailoring](level-scaling.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
