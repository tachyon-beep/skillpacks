# Learning-rate schedule decisions

Use for a schedule/cadence issue or a controlled training experiment; no universal warmup rule applies to all transformers or optimizers.

## Checks

- Record objective, optimizer/parameter groups, initialization/pretraining, effective batch, total update budget and observed loss/gradient behavior.
- Define schedule units explicitly: optimizer updates, batches, epochs or tokens. Accumulation, AMP skipped updates, resumes and changing dataset size can change cadence.
- Verify API semantics for step placement and metric-driven schedules against the installed framework version. Log actual LR per group alongside updates.
- Compare a simple baseline with warmup/constant/decay/cosine/WSD/one-cycle or optimizer-specific schedule-free choices only when the task warrants it.
- Warmup can improve some initial scale/optimizer transients; a successful no-warmup run is counterevidence to a universal requirement. Do not assign generic accuracy gains.
- Continuing a run changes remaining budget and schedule state. Record intended continuation/decay semantics; silently restarting a schedule is a different experiment.
- LR-finder sweeps alter state; restore the baseline checkpoint and validate recommendations on the actual run. Divergence in a sweep can reflect other faults.
- Keep validation/selection separate from report/test data. Compare candidates at common budgets and account for repeated selection.

## Deliverable

Schedule/config diff or experiment plan with cadence, baseline, observed trace and resume checks. See [optimizer choices](optimization-algorithms.md), [gradient health](gradient-management.md) and [batch/precision](batch-size-and-memory-tradeoffs.md).
