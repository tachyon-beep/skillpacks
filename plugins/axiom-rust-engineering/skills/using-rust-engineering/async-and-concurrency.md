# Rust async ownership and shutdown

## Trace ownership

Identify each task's owner, shared state and shutdown path. Dropping a future cancels its local progress; it does not generally undo remote effects, stop detached tasks or halt blocking work. Read the runtime/API cancellation guarantees rather than assuming them.

## Failure checks

- Cancellation between acquiring and committing a resource can lose work. Check queue receive/ack, transaction boundaries and select-loop cancellation safety.
- A spawned task can outlive the caller. Retain handles or use an explicit supervisor; consume failures and join during shutdown where required.
- Blocking or CPU-bound work on an executor thread can starve unrelated tasks. Bound blocking work and its queues; cancellation may not stop it once running.
- Holding a lock across await can deadlock or serialize a service. Examine the full invariant before replacing the lock type.
- Bounded channels provide backpressure only if producers do not create an unbounded task per item first.
- `Send`/`Sync` express type-level safety, not deadlock freedom or atomic multi-step invariants.
- A `select!` loop's fairness and branch cancellation behavior affect starvation and loss. Verify the concrete runtime primitive.
- Async trait erasure/boxing can change dispatch, allocation and Send requirements; use it for the actual API contract.

Validate worker panic/failure, cancellation at critical points, exhausted capacity and shutdown with in-flight work. Report behavior under the supported runtime rather than prescribing a runtime migration.
