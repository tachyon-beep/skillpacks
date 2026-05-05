# Refresh: yzmir-neural-architectures

**Verdict:** MEDIUM / M effort. Strong router/transformer core; CNN/sequence/generative/GNN/multimodal sheets calibrated to ~2022.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-neural-architectures/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-neural-architectures.md`
- Purpose: architecture selection by data modality and constraints.

## Why refresh

Reviewer found strong router/transformer/normalization core but specifics frozen ~2022:
- **No Mamba / state-space models / S4 / RWKV.**
- **No MoE coverage current** — Mixtral, DeepSeek-MoE, OLMoE, post-Switch-Transformer landscape.
- **Generative sheet outdated** — no SDXL, no FLUX, no DiT-family vision diffusion.
- **CNN sheet outdated** — no ConvNeXt v2.
- **Detection/segmentation outdated** — no DETR successors, no SAM/SAM-2.
- **Multimodal outdated** — no LLaVA family, no Flamingo successors, no current CLIP variants.

## Scope — DO

1. **Sequence sheet.** Add Mamba / SSM / S4 / RWKV with comparison to attention.
2. **Transformer sheet.** Add MoE current state (Mixtral, DeepSeek-MoE, fine-grained experts, routing strategies).
3. **Generative sheet.** Add SDXL, FLUX, DiT, video diffusion (Sora-class), discrete-diffusion. Reframe GAN content as legacy + still-useful niches.
4. **CNN sheet.** Add ConvNeXt v2, EfficientNet v2 specifics, Hiera as transformer-CNN hybrid.
5. **Multimodal sheet.** Add LLaVA family, BLIP-2, current CLIP variants, multimodal-from-scratch.
6. **Vision sheet.** Add SAM / SAM-2, DETR successors, DINOv2.

## Scope — DO NOT

- Do not duplicate LLM training methodology (lives in `yzmir-llm-specialist` and `yzmir-training-optimization`).
- Do not duplicate PyTorch implementation patterns (lives in `yzmir-pytorch-engineering`).
- Do not name benchmark numbers that drift — describe regime, not leaderboards.

## Acceptance criteria

1. Mamba / SSM covered.
2. MoE current state covered.
3. SDXL / FLUX / DiT covered in generative sheet.
4. SAM / SAM-2 covered.
5. ConvNeXt v2 covered in CNN sheet.
6. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-neural-architectures.md`.
2. Read every SKILL.md.
3. Each new architecture cited to a paper.
4. Bump version.

## Constraints

- Every architecture has a paper citation.
- No fabricated parameter counts or benchmark scores.
- Stay at architecture level — implementation belongs in PyTorch pack.
