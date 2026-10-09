# Service Boundary Decision

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Start with a required isolation/scaling/ownership/failure boundary. Compare keeping a single deployable system before adding distributed coordination.

For each proposed boundary name data/authority ownership, interface/version, delivery/consistency contract, failure/retry behavior, observability and operational owner. Price integration, deployment and migration/exit cost.

Test actual cross-service contracts and recovery where risk warrants it. Use solution architecture for consequential alternatives and distributed-systems guidance for partial failure; no service-count or framework fashion justifies decomposition.
