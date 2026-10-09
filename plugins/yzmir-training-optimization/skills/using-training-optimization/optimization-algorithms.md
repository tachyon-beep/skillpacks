# Optimizer and state strategy

Use when selecting an optimizer or diagnosing its updates/state. A named optimizer is a hypothesis to test, not a universal best choice.

## Checks

- Bind objective, model/pretraining, budget, parameter groups and weight-decay semantics. Coupled L2 regularization and decoupled weight decay differ; use the installed implementation's documented options rather than calling all Adam decay broken.
- Inspect gradient flow, zeroing, update cadence, accumulation and scheduler before changing optimizer families.
- Compare a tuned baseline with justified alternatives under common data/splits and resource budgets. AdamW/SGD and newer families such as Lion, Sophia, Muon, Shampoo, AdEMAMix, Prodigy or schedule-free methods have workload/implementation assumptions; consult their primary sources.
- Some optimizers transform only eligible matrix parameters; declare fallback handling for biases/norms/embeddings and mixed parameter groups.
- Fused/foreach paths, low-bit/paged states and distributed sharding affect memory, kernel support and numerical behavior. Measure target-runtime effects and preserve checkpoint portability.
- Estimate parameter/gradient/optimizer/activation memory separately. ZeRO/FSDP choices are state/sharding strategies, not interchangeable optimizer names; framework APIs belong to PyTorch engineering.
- Test freeze/unfreeze, new parameters, resume and state migration. Resetting moments or changing algorithm mid-run changes the experiment.
- Preference objectives such as DPO/GRPO are separate from optimizer family; method choice belongs to LLM specialist.

## Deliverable

Optimizer/state decision or repair with parameter-group policy, actual update evidence, resource/quality comparison and version/assumption limits. See [schedules](learning-rate-scheduling.md), [batch memory](batch-size-and-memory-tradeoffs.md) and [gradient checks](gradient-management.md).

## Existing source pointers

Check current applicability/version before using a recipe. These pointers are optional supporting sources.

- <https://arxiv.org/abs/1412.6980>
- <https://arxiv.org/abs/1711.05101>
- <https://arxiv.org/abs/2302.06675>
- <https://arxiv.org/abs/2305.14342>
- <https://arxiv.org/abs/1802.09568>
- <https://arxiv.org/abs/2002.09018>
- <https://arxiv.org/abs/2409.03137>
- <https://arxiv.org/abs/2502.16982>
