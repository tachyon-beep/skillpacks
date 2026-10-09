---
name: using-training-optimization
description: "Use when a training run needs evidence about convergence, objective, gradients, regularization, search budget, batch or precision strategy."
---

# Training Optimization

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Training contract

Inspect the data/split, task metric, objective, loss/gradient traces, update order and environment before changing hyperparameters. Use evidence available in the code/run; ask only for missing facts that affect diagnosis.

- Check that labels, sampling, objective and evaluation metric represent the intended task. Leakage or a broken update is not repaired by a different optimizer.
- Distinguish flat/unstable training, poor generalization and throughput limits. Reproduce on a small batch/task and verify gradient flow before broad sweeps.
- Preserve accumulation normalization, clipping/scaling order, scheduler/optimizer cadence and resume state. Effective batch changes may alter optimization statistics; accumulation is not always equivalent with batch-dependent operations.
- Choose precision and memory/sharding strategy from hardware support, stability and measured memory/throughput. Keep strategy separate from version-specific framework APIs.
- Compare optimizer/schedule/regularization candidates under common splits, budgets and metrics. Familiar defaults are baselines, not universally correct choices.
- Apply augmentation according to task invariances and the evaluation protocol. Evaluation-time transformations are valid when explicitly part of the protocol; never leak fitted augmentation choices through test outcomes.
- Allocate search budget and define selection/early-stopping rules. Log configurations, data/code identity, randomization and failures so the winner can be re-evaluated.

## Evidence and output

Produce a diagnosis and minimal intervention or a budgeted experiment plan. Include baseline, hypothesis, changed factors, acceptance metric, resource budget, run identity and observed result/limitations. Report a proposed optimizer change as a hypothesis until compared.

## Fault-specific references

- NaN/Inf or inconsistent updates: `gradient-management.md`, relevant loss/precision sections.
- Flat loss/plateau: verify updates/data, then schedule/optimizer references if implicated.
- Generalization gap: `overfitting-prevention.md`, `data-augmentation-strategies.md`, with split/label checks first.
- Batch/precision/resource tradeoff: `batch-size-and-memory-tradeoffs.md`.
- Search/scale transfer: `hyperparameter-tuning.md`.
- Logging/resume integration: tracking and training-loop sheets below.

PyTorch API/compiler/distributed faults belong to PyTorch engineering; preference-method choice to LLM specialist; RL-specific algorithm/environment issues to deep RL. Registry/releases belong to ML production. No compulsory questionnaire or multi-sheet sequence is needed for a direct answer.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [batch size and memory tradeoffs](batch-size-and-memory-tradeoffs.md)
- [data augmentation strategies](data-augmentation-strategies.md)
- [experiment tracking](experiment-tracking.md)
- [gradient management](gradient-management.md)
- [hyperparameter tuning](hyperparameter-tuning.md)
- [learning rate scheduling](learning-rate-scheduling.md)
- [loss functions and objectives](loss-functions-and-objectives.md)
- [optimization algorithms](optimization-algorithms.md)
- [overfitting prevention](overfitting-prevention.md)
- [training loop architecture](training-loop-architecture.md)
