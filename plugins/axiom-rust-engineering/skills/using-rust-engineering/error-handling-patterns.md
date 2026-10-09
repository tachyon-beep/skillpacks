# Rust error boundaries

## Choose the error contract

Read callers and public API compatibility. Distinguish expected domain rejection, environmental failure, programmer invariant violation and cancellation. Typed public errors help callers act; application-level context can preserve an underlying source chain without exposing every internal type.

## Failure checks

- `unwrap`/`expect` require a real invariant. External input, network failure and dependency behavior rarely justify unconditional panic.
- Adding context must preserve the source needed for diagnosis and classification. Avoid converting every error to an opaque string.
- A broad `From` conversion can collapse distinct failures or create ambiguous behavior. Keep meaningful distinctions at the consumer boundary.
- Retrying needs an idempotency/side-effect policy, not just an error category.
- Error messages and logs must not disclose credentials or sensitive payloads.
- FFI boundaries need a deliberate panic/unwind policy; do not let a Rust panic cross an ABI that forbids unwinding.
- Changing enum variants can break exhaustive downstream matches. Check the crate's public API policy before changing representation.

Verify one representative failure per affected class and the consumer's response. Follow existing library/application conventions; adding an error library is not necessary for a local fix.
