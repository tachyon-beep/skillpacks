# Python tests that distinguish failures

Choose checks from the behavior changed, its consumers and likely failure modes. Do not add tests that merely repeat the implementation or impose a complete new test architecture for a small repair.

## Build useful evidence

- Reproduce the defect when possible. The test should fail for the relevant wrong behavior, not incidental implementation details.
- Use independent expected values/fixtures for parsers, serializers and transformations. Round-tripping through the same implementation can reproduce the same bug twice.
- Check negative paths: malformed input, missing data, partial failure, cancellation and cleanup when those belong to the affected contract.
- Scope fixtures to resource ownership. Shared mutable fixtures can make tests order-dependent; cleanup must run on failure.
- Patch at the lookup boundary actually used. Excessive mocking can remove the integration that caused the defect.
- Property tests need meaningful generators and invariants, with reproducible failing examples. A vacuous property adds little evidence.
- Coverage indicates executed code, not correct assertions or complete behavior. Mutation/adversarial examples can expose weak oracles where justified.
- Async, clock and random tests need controlled scheduling/state where possible. Retries should not hide a reproducible failure.

## Match the boundary

Use focused tests for a local repair, installed-artifact checks for packaging, and real service/database integration where a mock cannot establish the claim. Record the command, environment and result; distinguish local tests, CI, deployment and user acceptance. Missing dependencies or skipped tests are evidence gaps.
