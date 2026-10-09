# Python integration for ML experiments

This sheet covers Python packaging and artifact boundaries around experiments. Model selection, training methodology and serving policy belong with the relevant ML discipline; a Python repair does not require an experiment platform.

## Check the integration boundary

- Resolve the actual environment, lock/configuration files and supported accelerator/library versions. Record the dependency state used for a result.
- Separate configuration from measured results. Preserve the fully resolved configuration, including overrides and defaults that affect behavior.
- Give datasets/splits, code, checkpoints and preprocessing a durable identity. A path called `latest` is not an experiment identity.
- Check checkpoint closure: optimizer, scheduler, RNG, sampler/progress and preprocessing state may be necessary to resume, depending on the equivalence promised.
- Keep train/validation/test decisions separate. Repeated selection on a held-out set changes what its score establishes.
- Log metrics with units, aggregation, independent experimental unit and missing/failed-run status. Do not silently discard failed runs.
- Define who writes artifacts and how interrupted writes become visible. Avoid treating an incomplete directory as a valid checkpoint.
- Keep secrets and sensitive examples out of tracking metadata and logs.

## Choose tools proportionately

Use the existing configuration/tracking system. Add a framework only when the user needs capabilities such as shared lineage or controlled comparison that the current workflow cannot supply. Before adopting a recipe, verify its API against the installed version.

Validate one real run, interrupted resume where claimed, artifact reload, and a downstream consumer. Report local test evidence separately from reproducible training or production acceptance.
