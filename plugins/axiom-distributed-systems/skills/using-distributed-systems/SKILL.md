---
name: using-distributed-systems
description: "Use when multiple processes or failure domains must remain correct through crashes, partitions, retries, duplicates, clock changes or overload."
---

# Correctness under partial failure

State correctness per operation and test it under the failures the architecture promises to tolerate. Managed services reduce implementation work but do not select the application contract.

## Work from the affected contract

1. Name operations, failure domains, consistency and availability requirements. Decide whether distribution is needed and what is delegated to a managed system.
2. Trace writes, messages, acknowledgments and external effects. Locate crash windows, duplicated effects, lost updates, ordering assumptions and stale owners.
3. Choose idempotency/deduplication, transactions/outbox, fencing/leases and compensation from the actual resources and failure model. Price coordination rather than adding consensus by default.
4. Budget deadlines, retries, queues and backpressure together. Bound overload and specify what is blocked, rejected or shed.
5. Inject applicable crash/partition/duplicate/reorder/clock scenarios and check the declared invariants, recovery and operational visibility.

## Scope and completion

Produce the operation/failure matrix, decisions and evidence at the smallest useful scope. N/A is valid with a reason. Claims such as exactly-once effects or linearizable access require stated boundaries and verification; do not infer them from delivery or quorum terminology.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Backpressure and Flow Control | [backpressure-and-flow-control.md](backpressure-and-flow-control.md) |
| Consensus and Coordination | [consensus-and-coordination.md](consensus-and-coordination.md) |
| Consistency Models and CAP/PACELC | [consistency-models-and-cap.md](consistency-models-and-cap.md) |
| Cost and When NOT to Distribute | [cost-and-when-not-to-distribute.md](cost-and-when-not-to-distribute.md) |
| Delivery and Ordering Semantics | [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) |
| Failure Models and the Fallacies | [failure-models-and-fallacies.md](failure-models-and-fallacies.md) |
| Idempotency and Deduplication | [idempotency-and-deduplication.md](idempotency-and-deduplication.md) |
| Partitioning and Sharding | [partitioning-and-sharding.md](partitioning-and-sharding.md) |
| Replication and Quorums | [replication-and-quorums.md](replication-and-quorums.md) |
| Resilience Patterns | [resilience-patterns.md](resilience-patterns.md) |
| Sagas and Distributed Transactions | [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) |
| Testing Distributed Systems | [testing-distributed-systems.md](testing-distributed-systems.md) |
| Time, Clocks, and Ordering | [time-clocks-and-ordering.md](time-clocks-and-ordering.md) |

## Optional task entry points

- [analyze-failure-modes](../../commands/analyze-failure-modes.md): Analyze failure modes for the affected distributed systems contract, with scoped source and verification evidence.
- [design-distributed-system](../../commands/design-distributed-system.md): Design distributed system for the affected distributed systems contract, with scoped source and verification evidence.
- [review-distributed-design](../../commands/review-distributed-design.md): Review distributed design for the affected distributed systems contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [distributed-design-reviewer](../../agents/distributed-design-reviewer.md), [failure-scenario-analyst](../../agents/failure-scenario-analyst.md). No fixed reviewer count is required.
