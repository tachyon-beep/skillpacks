# Performance Experiment Design

Use measurement to test a concrete latency, throughput, capacity or resource claim.
A reproducible workload can reveal failures that do not occur under a single user;
profiling/tracing then helps explain the mechanism. Neither must always precede
the other.

## Specify the experiment

- Decision and acceptance requirement: user/task consequence, target and rationale.
- Workload: arrival model, concurrency, think time, data distribution, warm/cold
  conditions, external calls and duration.
- Environment: versions, hardware/limits, network/dependencies and differences
  from the target deployment.
- Baseline and comparison: same workload/environment, repeated when variability
  matters; avoid changing several causal factors at once.
- Signals: latency percentiles/distribution, throughput, errors/timeouts, resource
  saturation, queue depth and relevant correctness checks.
- Stop conditions and rollback: bound cost, damage and blast radius; do not load
  production without authorization and suitable controls.

## Choose useful stress

Normal load verifies expected behavior. Increasing stress explores limits; spikes
exercise sudden demand; sustained runs expose accumulation/leaks. Choose duration
from the mechanism being tested rather than a fixed generic minute/hour table.
Use the existing benchmark/load harness and installed-version documentation.

Guard against coordinated omission, unrealistic caching/data, insufficient client
capacity, hidden retries and measurements that omit failures. Averages can hide
tails; arbitrary percentile targets can also mislead without product context.

Report workload and artifacts, before/after results, variability, correctness,
profile/trace evidence and untested extrapolation. See
[load-testing-patterns.md](load-testing-patterns.md) for implementation recipes.
