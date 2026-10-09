# Sequence architecture tradeoffs

Use when sequence length, state, streaming or memory constraints distinguish feasible architectures.

## Comparison checks

- Define task, token/time resolution, training/inference sequence-length distribution and acceptable lookahead. A causal streaming contract differs from offline bidirectional modeling.
- Include simple/statistical and pretrained baselines where appropriate. Do not call recurrent models obsolete merely because transformers are available.
- Compare attention, convolutional/recurrent and state-space/hybrid candidates using actual supported kernels and checkpoints. Asymptotic scaling does not establish latency or quality.
- Check positional/context extrapolation, state/reset ownership, padding/masking, batching and chunk-boundary behavior.
- For recurrent/SSM state, assess memory/reset/replay semantics and whether fixed state can preserve the required information. For attention, assess KV cache and long-context memory/quality.
- Validate long-tail lengths, changed sampling rates, missing data and out-of-distribution contexts relevant to the task.
- Keep pretraining, parameter/training budget, data/splits and metric comparable; report throughput and tail latency under the deployed shape distribution.

## Deliverable

A feasible candidate/baseline decision with workload and version identity, evidence and a discriminating experiment. Claims that one family dominates a whole modality require current sources and task evidence.

See [attention mechanisms](attention-mechanisms-catalog.md), [transformer details](transformer-architecture-deepdive.md) and [architecture decisions](architecture-design-principles.md) for specific checks.
