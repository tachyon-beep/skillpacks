---
name: using-quality-engineering
description: "Use when test reliability, isolation, coverage gaps, performance experiments, resilience, or release evidence needs investigation or a concrete verification strategy."
---

# Quality Engineering

Choose checks for real failure modes and decisions. Test count, instruction
compliance, a fixed pyramid ratio, and a passing local run are not product
acceptance evidence.

## Work from a failure or claim

1. Read the changed behavior, interfaces, existing harnesses, relevant failures,
   environment and acceptance criteria. Preserve unrelated work and test state.
2. State what could fail and the smallest check capable of detecting it. Use
   existing tools and fixtures before proposing another test framework.
3. For intermittent failure, capture the failing seed/order/concurrency/state and
   compare isolated versus combined runs. Diagnose shared resources and scheduling;
   a successful retry does not resolve the defect.
4. For performance/resilience, define workload, baseline, resource limits, metrics,
   stop/rollback conditions and interpretation before running the experiment.
   Load can reveal failures; profiling then helps explain the observed mechanism.
5. Run checks appropriate to the actual change, inspect results and fix material
   failures. Broaden only when changed behavior, failure or unresolved risk warrants
   it. Honor an explicit request to answer directly or skip optional routing.
6. Separate implementation verification, independent review, stakeholder acceptance,
   deployment and live checks. A passing suite does not establish business or human
   acceptance; record the actual approver or missing acceptance evidence.

## Optional references

References are beside this file. Retrieve relevant sections only, not the entire
catalog for every test question. Framework/tool details need installed-version or
official documentation checks before use.

| Problem | Reference |
|---|---|
| Shared state, order/concurrency, fixtures | [test-isolation-fundamentals.md](test-isolation-fundamentals.md), [test-data-management.md](test-data-management.md), [flaky-test-prevention.md](flaky-test-prevention.md) |
| Integration/API/consumer boundaries | [integration-testing-patterns.md](integration-testing-patterns.md), [api-testing-strategies.md](api-testing-strategies.md), [contract-testing.md](contract-testing.md) |
| Browser workflows or visual stability | [e2e-testing-strategies.md](e2e-testing-strategies.md), [visual-regression-testing.md](visual-regression-testing.md) |
| Invariants, generated inputs, test sensitivity | [property-based-testing.md](property-based-testing.md), [fuzz-testing.md](fuzz-testing.md), [mutation-testing.md](mutation-testing.md) |
| Capacity, latency, workload experiments | [performance-testing-fundamentals.md](performance-testing-fundamentals.md), [load-testing-patterns.md](load-testing-patterns.md) |
| Failure injection or controlled rollout | [chaos-engineering-principles.md](chaos-engineering-principles.md), [testing-in-production.md](testing-in-production.md) |
| Strategy, maintenance, decision metrics | [test-automation-architecture.md](test-automation-architecture.md), [test-maintenance-patterns.md](test-maintenance-patterns.md), [quality-metrics-and-kpis.md](quality-metrics-and-kpis.md) |
| Integrating scanners and production signals | [static-analysis-integration.md](static-analysis-integration.md), [dependency-scanning.md](dependency-scanning.md), [observability-and-monitoring.md](observability-and-monitoring.md) |

Canonical language lint/analysis implementation belongs to its engineering pack;
security architecture to `ordis-security-architect`; contract design to
`axiom-contract-engineering`; delivery/observability setup to `axiom-devops-engineering`.
Use native testing/security tools when available. Their absence need not block a
bounded analysis; report missing executable evidence.

## Review and output

Optional `test-suite-reviewer`, `coverage-gap-analyst`, and
`flaky-test-diagnostician` roles provide focused coverage; no automatic panel or
paired review is required. Independent reviewers should inspect the actual diff,
source and results and state their review scope.

Report the conclusion, reproduced conditions, exact checks/results, failed or
unrun coverage, and the next decision. For release gates, name acceptance criteria,
evidence, defect disposition/waiver authority and remaining limits. No arbitrary
metric threshold or synthetic stakeholder sign-off substitutes for that authority.
