---
description: Use when starting any AI/ML task and unsure which Yzmir specialist pack applies - routes to PyTorch, training, RL, LLM, neural-architecture, ML-production, dynamic-architecture, morphogenetic-RL, simulation-foundations, or systems-thinking specialists. Owns mandatory-clarification triggers, cross-cutting multi-pack routing, and pressure-resistance discipline (time / authority / sunk-cost / keyword / social).
---

# AI Engineering Routing

**This is a router, not a content pack. Its only job is to pick the right Yzmir specialist (or sequence of specialists) for an AI/ML query. Wrong routing wastes more time than the 30-second clarification it skips. For non-ML system architecture use `/solution-architect`; for security threat modeling around AI use `/security-architect`; for Python tooling under PyTorch/Transformers use `/python-engineering`.**

Use the `using-ai-engineering` skill from the `yzmir-ai-engineering-expert` plugin to route to the correct specialist pack. Content authority lives in `plugins/yzmir-ai-engineering-expert/skills/using-ai-engineering/SKILL.md` - this wrapper is a thin pointer. Worked examples (including time/authority/sunk-cost pressure scenarios and the morphogenetic-rl-vs-dynamic-architectures and PEFT-method disambiguations) are in the companion `routing-examples.md`.

## When to Use

- Starting any AI/ML engineering task
- User mentions "neural network", "train a model", "RL agent", "fine-tune LLM", "deploy model", and you're not sure which pack
- Cross-cutting query that spans multiple Yzmir packs (load in dependency order)

**Don't use** for: non-ML data processing, pure system architecture (`/solution-architect`), pure DevOps (`/axiom-devops-engineering`), or AI security threat modeling (`/security-architect`).

## Specialist packs this router dispatches to

- `/pytorch-engineering` - PyTorch framework: CUDA, memory, FSDP, `torch.compile`, distributed, tensor ops
- `/training-optimization` - optimizers, schedules, BF16/FP8, gradients, convergence, hyperparameters
- `/deep-rl` - agents, policies, rewards, environments, MDP, offline RL, MARL
- `/llm-specialist` - prompting, RAG, fine-tuning (SFT/DPO/GRPO), reasoning models, agentic/MCP, multimodal use, prompt caching, evaluation, safety
- `/neural-architectures` - CNN / transformer / Mamba / GNN / diffusion / DiT / multimodal fusion, capacity tradeoffs
- `/ml-production` - vLLM / SGLang / TensorRT-LLM serving, quantization (AWQ/GPTQ/FP8), MLOps, observability, drift
- `/dynamic-architectures` - networks that grow / prune / adapt: continual learning, gradient isolation, MoE, adapter merging, PEFT (LoRA/QLoRA/DoRA/VeRA/PiSSA/LoftQ/LoRA+/rsLoRA/LongLoRA). *Owns HOW the growable network trains.*
- `/morphogenetic-rl` - **RL controllers that decide WHEN and HOW to mutate a network's topology** during training: action/observation/reward design, governor and safety gates, rollback-as-RL-signal, deterministic morphogenesis, growth-aware ablation. *Companion to `/dynamic-architectures` - owns the controller, not the grown net.*
- `/simulation-foundations` - ODEs, integrators, stability, determinism (foundation under many RL envs)
- `/systems-thinking` - causal loops, leverage points, archetypes, stocks-flows

## Routing discipline (load SKILL.md for the full version)

The router owns five hard-earned disciplines that consumers should NOT reinvent:

1. **Mandatory clarification triggers** - ambiguous terms ("model not working", "improve performance", "learning chatbot", missing framework) must trigger ONE clarifying question before routing.
2. **Cross-cutting routing** - queries spanning packs route to ALL relevant packs in dependency order (foundation before domain, training before deployment).
3. **Pressure resistance** - time/emergency, authority, sunk-cost, keyword-anchoring, and social pressures are all rationalizations to mis-route. The skill ships scripted counter-narratives.
4. **The "agent" keyword trap** - in 2026, "agent" overwhelmingly means an LLM tool-loop (`/llm-specialist`), not an RL policy (`/deep-rl`). Disambiguate before routing.
5. **Morphogenetic-RL vs dynamic-architectures vs deep-rl** - "grow a network with an RL controller" is `/morphogenetic-rl` first; `/dynamic-architectures` covers HOW the grown network trains; `/deep-rl` is generic RL machinery underneath.

## Cross-references

- Software-engineering rigor around AI code (debugging, refactoring, code review) → `/software-engineering`
- System design when AI is one component of a larger architecture → `/solution-architect`
- LLM threat modeling, prompt injection, AI supply chain → `/security-architect`
- Python tooling foundations under PyTorch / Transformers → `/python-engineering`
