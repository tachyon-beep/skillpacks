# Training run identity and comparison

Use when a run must be reproduced, compared or resumed. Production registry and artifact release mechanics belong to ML production's tracking/versioning sheet.

## Run record

- Bind run ID to code/config/dependency identity, data/split/preprocessing version, model initialization/checkpoint and randomization policy.
- Log objective/task metrics, steps/examples/tokens, effective batch, optimizer/scheduler/precision, resource consumption and failures. Record actual values and change history, not only launch defaults.
- Keep train/eval and independent-unit identities distinct. A dashboard averaging correlated branches does not correct pseudo-replication.
- Preserve checkpoints and resume contract, including scheduler/scaler/RNG and data position as required. Record resumes/forks with lineage instead of silently extending an unrelated run.
- For BF16/FP8 or other low precision, record backend/toolchain, scales/recipes and numerical failures needed to reproduce the run.
- Bound logging overhead, retention and sensitive payloads; avoid gradient/tensor dumps that alter resource behavior or expose data.
- Compare candidates on common budgets/splits and keep failed/negative trials. Record selection rules and the independent re-evaluation of a winner.

## Tool integration

Use the existing tracker or a minimal structured record that meets this contract. Verify current APIs before adding MLflow/W&B/TensorBoard or another integration. Extra tracker services are not a reproducibility proof.

## Deliverable

Run schema/integration or comparison report with recoverable artifact IDs and known gaps. See [training-loop state](training-loop-architecture.md) and counterfactual statistics for inference after selection.
