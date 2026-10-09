# Quantified Nonfunctional Requirements

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

For each consequential NFR record: metric/unit, target/range, workload/data size, environment, observation window, measurement method, owner and failure response. “Fast”, “secure” and “scalable” are intentions until scoped.

Examples of shape, not universal targets: percentile response latency under a named concurrency; recovery point/time with a demonstrated restore; availability over a declared window; memory/compute budget for a defined input distribution.

Map competing constraints (latency/consistency, accuracy/cost, recovery/durability) to explicit tradeoffs and evidence. Identify whether a requirement is mandated, assumed or experimentally provisional. A benchmark outside the target environment may inform an estimate but does not prove acceptance.

Output: requirement → mechanism → verification link, conflicts, assumptions and pending acceptance. Do not widen a target merely to bless a failing implementation without recording the decision.
