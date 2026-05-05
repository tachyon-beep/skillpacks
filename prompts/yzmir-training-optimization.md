# Refresh: yzmir-training-optimization

**Verdict:** HIGH / L effort. Optimizer/schedule/precision content frozen ~2022.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-training-optimization/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-training-optimization.md`
- Purpose: training configuration / diagnosis / hyperparameter selection / precision / parallelism.

## Why refresh

**Optimizer coverage frozen ~2022:**
- No Lion (Google, 2023)
- No Sophia (Stanford, 2023)
- No Muon (Jordan et al, 2024)
- No Shampoo / distributed Shampoo
- No AdEMAMix
- No 8-bit optimizers (bitsandbytes)
- No Schedule-Free / D-Adaptation / Prodigy

**Schedule coverage stale:**
- No WSD (Warmup-Stable-Decay) — common in modern LLM training.
- No learning-rate-free methods.

**Precision/parallelism stale:**
- No FP8 (H100/H200, transformer-engine)
- No clear BF16 vs FP16 differentiation for current hardware
- ZeRO/FSDP nomenclature outdated (FSDP2 is the current PyTorch path)

**Preference-tuning crossover absent:**
- DPO/GRPO/SimPO training dynamics belong here too (or cross-ref).

## Scope — DO

1. **Optimizer reference.** Add Lion, Sophia, Muon, Shampoo, 8-bit Adam, AdEMAMix. Decision table: when each beats AdamW, with citations.
2. **Schedules.** Add WSD, Schedule-Free / Prodigy / D-Adaptation. Decision criteria.
3. **Precision sheet.** Distinguish FP32 / TF32 / BF16 / FP16 / FP8 (E4M3, E5M2). Hardware-aware (Ampere vs Hopper vs Blackwell). Loss-scaling rules; transformer-engine usage.
4. **Parallelism sheet.** Refresh ZeRO stages (DeepSpeed nomenclature) AND FSDP2 (PyTorch native). Tensor parallel + pipeline parallel + sequence parallel + context parallel. When each kicks in.
5. **Cross-pack pointer.** Add a "preference tuning belongs in `yzmir-llm-specialist`" pointer or own a dedicated sheet — coordinate.
6. **Diagnostician.** Update symptom tables for FSDP2 errors and FP8 numerical issues.

## Scope — DO NOT

- Do not duplicate inference-time precision content (lives in `yzmir-ml-production`).
- Do not duplicate LLM-specific training methodology (DPO/GRPO/SimPO) wholesale; coordinate with `yzmir-llm-specialist`.
- Do not name a single "best" optimizer — every choice is regime-dependent.

## Acceptance criteria

1. Lion / Sophia / Muon / Shampoo / 8-bit Adam / AdEMAMix all covered (at minimum: when, gotchas).
2. WSD schedule covered.
3. FP8 covered with E4M3/E5M2 distinction and hardware constraints.
4. FSDP2 named (not just FSDP1) and contrasted with ZeRO.
5. `plugin.json` version bumped (major).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-training-optimization.md`.
2. Read every SKILL.md in this pack.
3. Confirm `yzmir-pytorch-engineering` and `yzmir-llm-specialist` refresh status — coordinate.
4. List sheets → keep / edit / replace. Plan past user.
5. Each new optimizer/method must cite a paper or official implementation.
6. Bump version.

## Constraints

- Every cited optimizer/method has a paper link.
- No "trust me bro" hyperparameter advice — cite or omit.
- Hardware-specific claims (FP8, sm_90, etc.) must be correct or skipped.
