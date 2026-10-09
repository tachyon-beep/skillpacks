---
name: using-pyo3-interop
description: "Use when a PyO3 extension needs safe ownership/lifetimes, array exchange, thread attachment, async cancellation, interpreter teardown, wheel compatibility or measured FFI performance."
---

# Python/Rust boundaries

Treat the Python/Rust boundary as an ownership, execution and distribution contract. Check the installed PyO3/Python/numpy versions and GIL-enabled versus free-threaded build first.

## Work from the affected contract

1. Trace ownership of Python handles, Rust data, NumPy views and retained resources. State what can outlive the call or interpreter and who may mutate shared buffers.
2. Separate Python-object access from Rust-only work. Select attach/detach and synchronization rules for the actual build; measure crossing and detach costs rather than using universal timing thresholds.
3. Map typed Rust errors to useful Python exceptions and preserve actionable tracebacks. Specify panic policy rather than swallowing unsafe or corrupted state.
4. For async/resource ownership, trace cancellation, executor/thread affinity, callback access and shutdown order. Exercise import/exit and partial initialization failures.
5. Choose ABI/wheel targets from consumer needs and supported features. Verify import, minimal operation, distribution matrix and measured payoff on representative workloads.

## Scope and completion

Deliver the boundary contract, version/build assumptions, targeted repair/design and observed checks. Keep batching, zero-copy and abi3 as measured choices with ownership/compatibility conditions. Rust acceleration is justified by whole-call cost and required behavior, not language reputation.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| abi3 vs Native Extensions | [abi3-vs-native-extensions.md](abi3-vs-native-extensions.md) |
| Async Across the Boundary: `pyo3-asyncio`, tokio, and asyncio | [async-across-the-boundary.md](async-across-the-boundary.md) |
| Batched Operations Across the FFI Boundary | [batched-ffi-operations.md](batched-ffi-operations.md) |
| Debugging PyO3: Panics, Segfaults, GIL Deadlocks, and Missing Tracebacks | [debugging-pyo3.md](debugging-pyo3.md) |
| Error Mapping and Traceback Fidelity | [error-mapping-and-traceback-fidelity.md](error-mapping-and-traceback-fidelity.md) |
| GIL Release Patterns: `Python::detach` and the Discipline Around It | [gil-release-patterns.md](gil-release-patterns.md) |
| Gymnasium Environments Backed by Rust | [gymnasium-environments-from-rust.md](gymnasium-environments-from-rust.md) |
| Lifecycle and Teardown: Rust-Owned Resources at Interpreter Shutdown | [lifecycle-and-teardown.md](lifecycle-and-teardown.md) |
| Maturin Inside a Cargo Workspace | [maturin-in-cargo-workspace.md](maturin-in-cargo-workspace.md) |
| NumPy Buffer Protocol: Zero-Copy Arrays and Lifetime Traps | [numpy-buffer-protocol.md](numpy-buffer-protocol.md) |
| Packaging and Wheels: cibuildwheel, abi3 Wheels, and the Distribution Matrix | [packaging-and-wheels.md](packaging-and-wheels.md) |
| Performance: When Crossing the FFI Boundary Pays Back | [performance-when-crossing-pays-back.md](performance-when-crossing-pays-back.md) |
| PyO3 Fundamentals: Types, Errors, Lifetime / `'py` Discipline | [pyo3-fundamentals.md](pyo3-fundamentals.md) |

## Optional task entry points

- [audit-gil-discipline](../../commands/audit-gil-discipline.md): Audit gil discipline for the affected pyo3 interop contract, with scoped source and verification evidence.
- [profile-ffi-boundary](../../commands/profile-ffi-boundary.md): Profile ffi boundary for the affected pyo3 interop contract, with scoped source and verification evidence.
- [scaffold-pyo3-crate](../../commands/scaffold-pyo3-crate.md): Scaffold pyo3 crate for the affected pyo3 interop contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [pyo3-reviewer](../../agents/pyo3-reviewer.md). No fixed reviewer count is required.
