# Training update and resume contract

Use when implementing or repairing a loop, callback or checkpoint boundary. Prefer project/framework conventions to a copied full trainer.

## Update invariants

- Establish train/eval modes and data/split identities. Ensure labels, loss and task metric agree before optimizing throughput.
- Specify zero-grad, forward/loss, backward, accumulation, unscale/clip, optimizer/scaler and scheduler cadence. Handle partial accumulation windows and normalization explicitly.
- AMP step skipping can affect scheduler/update counts; choose and test the intended policy. Framework/API-specific mechanics belong to PyTorch engineering.
- Track model/optimizer parameter membership and state when freezing, unfreezing or changing topology. Recreating an optimizer can silently discard useful state.
- Checkpoint all state required by the resume claim, including sampler/data position and callbacks where needed. Use safe-loading-compatible serialized state.
- Test resume by comparing a continued run with checkpoint/restart under the declared equivalence tolerance, not simply by confirming the file loads.
- Preserve cancellation/error cleanup and artifact atomicity. Log a failed or interrupted run truthfully instead of publishing a successful final metric.
- Compile/sharding boundaries and collective ordering can change failure behavior. Pin API assumptions and test the actual distributed target where relevant.

## Bounded verification

Use a tiny deterministic batch/task for update/cadence checks, eval-mode behavior, interruption/resume and a representative shape/precision smoke run. Expand only for relevant unresolved risks.

## Deliverable

Loop change/state diagram, meaningful check results and untested runtime limits. See [gradient management](gradient-management.md), [run tracking](experiment-tracking.md) and PyTorch checkpoint/distributed guidance as needed.
