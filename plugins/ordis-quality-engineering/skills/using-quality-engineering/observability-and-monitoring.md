# Production Signals for Verification

Delivery/observability implementation belongs to `axiom-devops-engineering`.
Quality engineering defines what signal would support or contradict the release
claim under a stated workload and time window.

Record behavior, baseline, workload/population, environment, instrumentation,
expected bounds and stop/rollback criteria. Examine errors, latency distributions,
throughput/resource saturation and relevant user/business outcomes together.
Correlate with release/configuration changes; do not infer causality from timing
alone. Check missing data, sampling, cardinality and alert delays.

A live check must be authorized and bounded. Preserve privacy and do not pollute
production state simply to generate evidence. Report observed interval, results,
blind spots and response authority. Healthy telemetry during one interval does not
prove future reliability or stakeholder acceptance.
