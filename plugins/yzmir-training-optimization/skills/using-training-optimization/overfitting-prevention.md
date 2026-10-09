# Generalization diagnosis and interventions

Use for an observed generalization gap, not an automatic request to add dropout.

## Checks

- Verify split independence/temporal policy, label quality, train/eval mode and metric comparability. Leakage, preprocessing skew or a shifted validation population can mimic other problems.
- Compare train/held-out trajectories at common budgets, with uncertainty/slice evidence appropriate to the claim. A fixed percentage gap is not universal severity.
- Test capacity/pretraining, sampling/data coverage, regularization, augmentation, objective and early stopping as specific hypotheses. Change factors deliberately.
- Keep decay versus coupled regularization semantics, dropout train/eval behavior and augmentation invariances explicit. Batch normalization or accumulation can alter the apparent intervention.
- Tune hyperparameters on designated validation data and re-evaluate winners independently. Repeatedly observing test metrics turns the test into a selection set.
- Distinguish fitting noise from failure to learn a relevant signal or distribution shift; more regularization can worsen underfitting.
- For instruction tuning, preserve base capabilities and evaluate held-out prompts/behavioral slices, not merely training loss. Data contamination and template memorization require dedicated checks.

## Deliverable

Evidence-backed diagnosis and a bounded intervention comparison, including unchanged baseline, selection history and remaining uncertainty. See [augmentation](data-augmentation-strategies.md), [search](hyperparameter-tuning.md) and dataset curation for data changes.
