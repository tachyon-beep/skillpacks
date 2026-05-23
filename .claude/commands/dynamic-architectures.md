---
description: Dynamic/morphogenetic neural networks - grow, prune, adapt topology during training. Continual learning, gradient isolation, modular composition, PEFT (LoRA/QLoRA/DoRA + modern variants), MoE/adapter merging, lifecycle orchestration, progressive training
---

# Dynamic Architectures Routing

**Companion to `/morphogenetic-rl` - this pack covers HOW the growable network trains (gradient isolation, alpha blending, lifecycle FSM training, PEFT mechanics); the morphogenetic-rl pack covers WHEN/HOW the controller decides to grow.**

Use the `using-dynamic-architectures` skill from the `yzmir-dynamic-architectures` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-dynamic-architectures/skills/using-dynamic-architectures/SKILL.md` - this wrapper is a thin pointer.

## Sheets

- **continual-learning-foundations** - EWC / SI / MAS, PackNet, Progressive Networks, rehearsal, BWT/FWT/ACC metrics, stability-plasticity dilemma
- **gradient-isolation-techniques** - freeze strategies, `detach` vs `no_grad` vs `stop_gradient` semantics, hook-based gradient surgery, dual-path training, alpha blending
- **peft-adapter-techniques** - LoRA / QLoRA / DoRA + modern variants (VeRA, PiSSA, LoftQ, LoRA+, rsLoRA, LongLoRA), adapter placement, rank selection
- **dynamic-architecture-patterns** - grow/prune patterns, Net2Net widening, lottery ticket, magnitude / gradient / structured pruning, triggers, slot semantics with cooldown
- **modular-neural-composition** - MoE (Switch / Mixtral / DeepSeek-MoE, Expert Choice, aux-loss-free balancing), grafting semantics, adapter merging (TIES / DARE / SLERP / MergeKit / LoraHub)
- **ml-lifecycle-orchestration** - state machines, transition gates, heuristic / learned / hybrid controllers, rollback and hysteresis, observability
- **progressive-training-strategies** - staged capacity expansion, warmup (zero-init / LR / alpha-ramp / frozen-host), cooldown, multi-stage schedules, transition-shock failure modes

## Commands

- `/yzmir-dynamic-architectures:design-lifecycle` - design a state machine for module growth / training / integration (Requirements → States → Transitions → Gates → Controller → Output)
- `/yzmir-dynamic-architectures:diagnose-growth` - diagnose growth, pruning, gradient-isolation, or lifecycle issues (Symptom → Evidence → Common-failure tables → Report)

## Agents

- `dynamic-architecture-advisor` - SME advisor for dynamic-architecture decisions (opus). Follows SME Agent Protocol with Confidence / Risk / Information Gaps / Caveats sections.

## Cross-references

- RL controller design / governor / rollback shaping for morphogenesis → `/morphogenetic-rl`
- RL algorithm choice (PPO / SAC / DQN) for the controller → `/deep-rl`
- Counterfactual evaluation of architecture changes → `/deep-rl` (counterfactual-reasoning)
- PEFT applied to LLMs in production (instruction tuning, RLHF) → `/llm-specialist`
- Distributed / FP8 / MoE-dispatch kernels, FSDP2 + QLoRA → `/training-optimization`
- Static transformer architecture design → `/neural-architectures`
- Production deployment, multi-tenant LoRA serving (S-LoRA / LoRAX / Punica) → `/ml-production`
- Low-level PyTorch / autograd debugging → `/pytorch-engineering`
