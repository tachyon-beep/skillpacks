# Augmentation as a task assumption

Use when transformed or synthetic examples may improve robustness/generalization. State the invariance/equivariance being taught and the risk that it changes the target.

## Checks

- Validate label/mask/box/sequence alignment under transformations. Crops, flips, token edits, mixup or interpolation can invalidate domain labels.
- Choose strength/composition from representative cases and task evidence; more variation is not automatically useful data.
- Fit augmentation-policy choices on training/validation roles, preserving test independence. Do not leak held-out examples through a learned/synthetic generator.
- Evaluation data should implement the declared deployment/evaluation distribution. Test-time augmentation is legitimate when explicitly part of the protocol; validation transformations must not be tuned on report outcomes.
- Measure rare slices, distribution shift and calibration as well as aggregate quality; augmentation can suppress meaningful minority patterns.
- Track random streams, worker/device behavior, transformations and synthetic provenance/ratio so the experiment can be reproduced.
- Compare real-only, augmented and synthetic mixtures under common budgets; inspect duplication/contamination and label quality. Synthetic volume alone does not establish coverage.
- Profile preprocessing/augmentation when throughput is affected; changing placement/precision can alter transformations or sampling.

## Deliverable

Transformation contract and visual/structural label checks, ablation result and provenance. Dataset release/content quality belongs to ML production's dataset-curation reference; see [generalization](overfitting-prevention.md) for intervention interpretation.
