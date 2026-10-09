# Rust evidence for behavior and soundness

Choose tests from the affected API, its consumers and plausible failures. Unit, integration, compile-fail, property, fuzz, Miri and benchmark checks establish different claims; none is a universal required suite.

## Match the failure

- Public API/lifetime/trait guarantees may need downstream compile tests as well as runtime tests.
- Unsafe code needs a written invariant, boundary cases and applicable Miri/sanitizer evidence. Passing a finite check cannot prove soundness.
- Independent fixtures/oracles avoid a serializer and parser sharing the same bug. Snapshot approval requires inspecting meaning, not accepting all updates.
- Property/fuzz generators must reach invalid and boundary states; constrain only where the real input contract does.
- Async tests need cancellation, resource cleanup and shutdown paths where relevant. A retry that hides a failure weakens evidence.
- Feature/target cfg can omit the code under test. Record the enabled configuration and check advertised combinations when affected.
- Benchmarks need a representative optimized build and stable workload; correctness and performance gates should remain distinguishable.
- Coverage counts execution; it does not establish assertions or failure-mode coverage.

Run focused checks for a local repair and broader integration checks at the boundary actually changed. Record skipped/unavailable checks, environment, commands and results without equating local success with deployment acceptance.
