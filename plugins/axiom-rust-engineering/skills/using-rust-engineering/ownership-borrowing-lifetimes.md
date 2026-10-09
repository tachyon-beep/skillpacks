# Ownership and lifetime failure checks

## Express the real owner

Trace allocation, borrowing, mutation and drop order through the affected API. Compiler errors often reveal an API that promises a reference longer than its owner exists. Choose owned return data, a caller-owned buffer or a shorter borrow according to the consumer contract.

## Check the proposed repair

- Cloning can fix a lifetime error while changing identity, cost or synchronization. Confirm that a separate owned value is valid.
- `Arc` shares ownership; it does not make the contents thread-safe or remove invariant-level races.
- Interior mutability moves checks or synchronization to runtime. Check borrow panics, lock poisoning and re-entrant calls.
- `Pin` constrains movement through a pointer; it does not automatically pin every field or make arbitrary self-reference sound. Projection and drop behavior need explicit invariants.
- `Send`/`Sync` implementations, especially unsafe ones, must cover all reachable state and callbacks.
- References stored across callbacks/FFI require a contract for retention and teardown. A locally valid borrow does not authorize foreign code to keep it.
- Extending lifetimes with unsafe casts or leaking memory hides the ownership problem unless a justified global lifetime is the actual contract.

Compile affected callers and exercise resource teardown, concurrency and failure paths relevant to the change. For unsafe ownership, write the invariant and use the unsafe/FFI sheet for soundness validation.
