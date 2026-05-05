# Refresh: yzmir-dynamic-architectures

**Verdict:** MEDIUM / M effort. Continual-learning / MoE citations stop ~2017-18.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-dynamic-architectures/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-dynamic-architectures.md`
- Purpose: networks that grow/prune/adapt — continual learning, modular composition, gradient isolation.

## Why refresh

Solid foundational math and clean routing, but citations stop ~2017-18. Notable gaps:

- **Adapter merging.** TIES-merging, DARE, task-arithmetic, MergeKit ecosystem.
- **Multi-LoRA composition.** Stacking, routing, S-LoRA serving.
- **Post-Mixtral MoE.** Fine-grained experts, shared experts, routing strategies, training dynamics.
- **2024 PEFT.** VeRA, LoRA+, PiSSA, LoftQ, DoRA.
- **Model surgery.** SLERP, model souping current state.

## Scope — DO

1. **Adapter / merging skill.** Add TIES, DARE, task-arithmetic, MergeKit, SLERP.
2. **PEFT sheet.** Add VeRA, LoRA+, PiSSA, LoftQ, DoRA — comparison and use cases.
3. **MoE sheet.** Post-Mixtral MoE — fine-grained, shared experts, routing strategies.
4. **Multi-LoRA serving pointer.** Cross-ref to `yzmir-ml-production` for S-LoRA.

## Scope — DO NOT

- Do not duplicate training-optimization content (parallelism strategies live in `yzmir-training-optimization`).
- Do not duplicate `yzmir-neural-architectures` MoE coverage — focus here on dynamic / adaptive aspects.

## Acceptance criteria

1. TIES/DARE/task-arithmetic covered in merging.
2. VeRA/LoRA+/PiSSA/LoftQ/DoRA covered in PEFT.
3. Post-Mixtral MoE coverage updated.
4. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-dynamic-architectures.md`.
2. Read every SKILL.md.
3. Each technique cites paper.
4. Bump version.

## Constraints

- Every technique has a paper citation.
- No fabricated formulas — verify against papers.
