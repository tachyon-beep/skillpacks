# Vision-backbone comparisons

Use when choosing or comparing a vision backbone for a specified task and target runtime.

## Comparison contract

- Record classification/detection/segmentation needs, resolution/aspect-ratio range, labels/slices, pretraining and adaptation options.
- Compare feasible pretrained convolutional, transformer or hybrid baselines as relevant. Dataset-size thresholds alone ignore pretraining, augmentation and task similarity.
- Check feature stride/receptive field, multi-scale outputs, head/backbone interfaces, localization behavior and normalization under the actual batch regime.
- Measure quality by the task and relevant slices; classification accuracy does not select a detection backbone by itself.
- Benchmark the target resolution/batch/concurrency on intended hardware; parameter count and paper FLOPs are not runtime measurements.
- Include activation memory, latency/power, supported kernels and export/quantization behavior for edge deployments.
- Keep preprocessing, split, augmentation, tuning and evaluation budgets comparable. Report pretrained checkpoint and code identity.

## Deliverable

A candidate comparison and justified baseline, with quality/resource measurements or clearly labeled estimates and the next experiment. Avoid universal ResNet/EfficientNet/ViT recommendations; verify current upstream checkpoints and deployment support.

See [architecture decisions](architecture-design-principles.md), [normalization](normalization-techniques.md) and [multimodal integration](multimodal-architectures.md) for those boundaries.
