# Reproducible Forecasting

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Define outcome, remaining scope, unit of work, capacity/dependencies and target window. Use comparable historical throughput/cycle times only when instrumentation/work mix are understood.

For Monte Carlo throughput scenarios: preserve the observed-period throughput sample, remaining-unit assumptions, random seed, sampling method, run count and quantiles. Bootstrap observed periods until remaining scope completes; inspect zero-throughput/stalled cases and independence/stationarity assumptions. Backtest where data permits. The percentile is conditional on those assumptions, not guaranteed confidence.

Without relevant data, provide explicit scenario/expert ranges and uncertainty; do not invent numerical confidence. Publish inputs/calculation and update when scope/capacity/dependencies change. Illustrative dates are not evidence about the live project.

## Optional calculation

[forecast.py](../../scripts/forecast.py) is a stdlib-only conditional bootstrap. Supply a CSV with unique `period` labels and nonnegative integer `completed` counts, including zero-throughput periods; define comparable period length/work units yourself.

```sh
python plugins/axiom-program-management/scripts/forecast.py observed-throughput.csv --remaining 20 --seed 0
```

It emits source/period IDs, observations, scope, seed/trial count, assumptions, p50/p85 completion periods and censored trials. `null` quantiles mean the requested percentile exceeds the simulation horizon. Convert periods to dates only after accounting for the actual calendar/dependency/capacity constraints. This does not manufacture historical data or calibrated confidence; inspect/backtest the assumptions before using a commitment.
