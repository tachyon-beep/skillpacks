# Async ownership and cancellation

## Trace the task lifetime

Identify who starts, awaits and cancels each task. Prefer structured task ownership where the supported Python version provides it; a background task needs an explicit owner and shutdown path. Keep references to intentionally detached tasks and consume their exceptions.

Check cancellation at every await boundary, including queue operations, lock acquisition, network calls and cleanup. Use `try/finally` or an async context manager for resources. Do not swallow cancellation as an ordinary retryable error. Shield only an operation whose completion must outlive its caller, and retain ownership of the shielded task.

## Failure checks

- A timeout on an await does not prove the remote operation did not happen. Reconcile side effects or use an idempotency protocol before retrying.
- Blocking I/O or CPU work on the event loop stalls unrelated tasks. Moving work to a thread changes cancellation and context propagation: cancelling the await generally does not stop the thread.
- A semaphore bounds active operations; it does not bound all queued tasks or memory. Use a bounded queue and backpressure for unbounded input.
- An async iterator or generator may hold a resource after early exit. Close it explicitly when the owning context does not do so.
- Task-group failures may arrive as exception groups. Preserve meaningful failures and cleanup rather than reporting only the first exception.
- Shared mutable state can race across await points even on one event-loop thread. Protect the invariant, not just individual assignments.

## Validate

Exercise cancellation during acquisition, use and cleanup; worker failure; exhausted capacity; and shutdown with work in flight. Measure latency under the actual mix of blocking and asynchronous work. Use the project's async test integration and supported runtime rather than introducing a new framework for a local repair.
