# Architecture decision record

Use for a concrete model-selection or custom-network decision, not a prerequisite lecture.

## Required distinctions

- Task and data modality do not uniquely determine the architecture. Record pretrained assets, invariances, target metric, data quality, interaction scale and deployment budget.
- Compare a simple baseline with feasible candidates under common preprocessing/splits and resource budgets. Separate architecture effects from pretraining, augmentation and optimizer changes.
- Trace shape, receptive/context field, state, residual interfaces, dtype/device and parameter/buffer ownership. Test edge shapes rather than trusting diagrams.
- Estimate parameters, activations, optimizer state, memory traffic and relevant FLOPs; measure actual runtime when latency matters.
- Residual paths, initialization and normalization alter scale/gradient behavior. Verify the particular model; depth alone is not a diagnosis or a mandate for one normalization family.
- Inductive bias can help or constrain the task. Write what observation would show the bias is wrong and what alternative would address it.
- Generalization and quality need held-out empirical evidence. A valid forward/backward pass establishes implementation properties, not task success.

## Deliverable

Constraints, candidates, baseline, assumptions, resource/quality evidence, choice and remaining experiment. Describe uncertainty and current-source versions. The family/component references in [the pack](SKILL.md) are optional support.
