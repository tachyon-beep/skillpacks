# Normalization and scale checks

Use for a normalization/residual-scale choice or a demonstrated stability fault. Normalization is not universally required; initialization, residual design, optimizer and architecture can support other strategies. Internal covariate shift is a historical motivation, not a complete universal causal explanation.

## Selection and failure checks

- State which dimensions/statistics are normalized, training versus inference behavior, learned affine terms and epsilon/dtype policy.
- Batch-dependent statistics depend on actual per-device batch, accumulation and distributed synchronization. Accumulated gradients do not make microbatch statistics equal a full batch.
- Layer/RMS/group/instance normalization differ in invariances and state; choose against task and architecture, not a fixed layer-count threshold.
- Check train/eval mode, running-stat updates, frozen/fine-tuned state and serialization. A resumed/exported model may silently use different statistics.
- Trace residual/pre-/post-normalization interfaces and activation/gradient scales; inspect the first invalid operation rather than adding layers blindly.
- Test small/variable batches, constant inputs, extreme magnitude, mixed precision and target inference/export kernels.
- Compare normalization, initialization and scale-control alternatives under common budgets. Report quality/stability/runtime evidence; no generic convergence multiplier applies.

## Deliverable

The normalization/scale contract or focused repair with representative/boundary tests and observed effects. See [architecture decisions](architecture-design-principles.md) and PyTorch engineering for implementation/runtime checks.
