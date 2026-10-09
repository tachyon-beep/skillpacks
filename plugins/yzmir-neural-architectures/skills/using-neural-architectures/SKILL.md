---
name: using-neural-architectures
description: "Use when neural architecture families or components must be compared against data, quality, compute, latency or adaptation constraints."
---

# Neural Architecture Decisions

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Decision contract

Record task/modality, representative data, pretrained assets, target metric, compute/memory/latency budget and deployment constraints. Do not infer the best architecture from dataset size or trend alone. A pretrained model changes the data requirement compared with training from scratch.

- Include a simple or pretrained baseline and explain why a neural approach is needed for the task.
- Compare feasible alternatives using the same data/splits, training budget and evaluation protocol. Distinguish architectural effects from preprocessing, optimization and pretraining.
- Check input/output shape, inductive assumptions, receptive/context requirements, invariances and numerical behavior.
- For attention/sequence choices, measure context length, memory and hardware kernel support; asymptotic complexity alone does not predict runtime.
- For generative models, compare quality, controllability, task-specific speed and available pretrained/distilled models. Neither GAN nor diffusion is a universal real-time recommendation.
- For normalization and residual paths, test the architecture's actual stability/scale behavior. Depth thresholds and normalization family are hypotheses, not diagnoses.
- Record uncertainty and the experiment that would distinguish the leading choices. Explain resource/quality tradeoffs rather than claiming a universally best family.

## Evidence and output

Produce a compact comparison/decision record: constraints, candidates and source versions, baseline, measured or estimated resource use, quality evidence, tradeoff and verification plan. For component repair, give the broken invariant and affected validation. Label unrun comparisons as proposals.

## Fault-specific references

- Vision backbone comparison: `cnn-families-and-selection.md`.
- Sequence/context tradeoff: `sequence-models-comparison.md`, `attention-mechanisms-catalog.md`.
- Generative or multimodal design: the respective family sheet.
- Graph learning: `graph-neural-networks-basics.md`; generated novel graph topology: structure synthesis.
- Normalization/residual or custom transformer questions: component sheets below.

These references are optional background and fault checklists. Resolve current model/tool availability through primary documentation. Training dynamics, PyTorch implementation and serving operations have their own owners; architecture choice alone cannot diagnose every convergence fault.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [architecture design principles](architecture-design-principles.md)
- [attention mechanisms catalog](attention-mechanisms-catalog.md)
- [cnn families and selection](cnn-families-and-selection.md)
- [generative model families](generative-model-families.md)
- [graph neural networks basics](graph-neural-networks-basics.md)
- [multimodal architectures](multimodal-architectures.md)
- [normalization techniques](normalization-techniques.md)
- [sequence models comparison](sequence-models-comparison.md)
- [transformer architecture deepdive](transformer-architecture-deepdive.md)
