# Maintain Tests by Failure Coverage

Inspect why a test exists, what real defect it detects and which dependencies make
it slow or brittle. Preserve externally meaningful assertions when changing setup.

- Consolidate fixtures/helpers that express a stable domain contract, not every
  repeated line. Keep data and scope explicit; avoid hidden shared mutable state.
- Page objects or component helpers can hide selector mechanics while exposing
  user actions. Do not hide waits, retries or assertions that mask real failures.
- Prefer behavior/invariant assertions over incidental internal call sequences.
- Reproduce intermittent failures with order/seed/environment recorded; diagnose
  shared resources and scheduling. A retry is evidence of intermittence, not a fix.
- If temporarily quarantining, retain visibility, owner, reason, exit criterion
  and review trigger. Delete a test when its contract is obsolete or covered more
  effectively, not merely because it is difficult to fix.
- Measure runtime, use focused selection and isolate resources before parallelizing.

For a refactor, run the affected tests and inspect their assertions before/after.
Use [flaky-test-prevention.md](flaky-test-prevention.md),
[test-isolation-fundamentals.md](test-isolation-fundamentals.md), and
[test-data-management.md](test-data-management.md) when those mechanisms matter.
