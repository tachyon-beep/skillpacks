---
name: using-ai-engineering
description: Use when you have an AI/ML task but aren't sure which Yzmir pack handles it - routes frameworks, training, RL, LLMs, architectures, and production to the correct specialist pack and clarifies before guessing
---

# Using AI Engineering

## Overview

This meta-skill routes you to the right AI/ML engineering pack based on your task. Load this skill when you need ML/AI expertise but aren't sure which specific pack to use.

**Core Principle**: Problem type determines routing - clarify before guessing.

## When to Use

Load this skill when:
- Starting any AI/ML engineering task
- User mentions: "neural network", "train a model", "RL agent", "fine-tune LLM", "deploy model"
- You recognize ML/AI work but unsure which pack applies
- Need to combine multiple domains (e.g., train RL + deploy)

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-ai-engineering/SKILL.md`

Reference sheets are at:
  `skills/using-ai-engineering/routing-examples.md`

NOT at:
  `skills/routing-examples.md` ← WRONG PATH

---

## STOP - Mandatory Clarification Triggers

Before routing, if query contains ANY of these ambiguous patterns, ASK ONE clarifying question:

| Ambiguous Term | What to Ask | Why |
|----------------|-------------|-----|
| "Model not working" | "What's not working - architecture, training, or deployment?" | Could be 3+ packs |
| "Improve performance" | "Performance in what sense - training speed, inference speed, or accuracy?" | Different domains |
| "Learning chatbot/agent" | "Fine-tuning language generation or optimizing dialogue policy?" | LLM vs RL vs both |
| "Train/deploy model" | "Both training AND deployment, or just one?" | May need multiple packs |
| Framework not mentioned | "What framework are you using?" | PyTorch-specific vs generic |

**If you catch yourself about to guess the domain, STOP and clarify.**

---

## Routing by Problem Type

| Keywords/Signals | Route To | Why |
|------------------|----------|-----|
| PyTorch, CUDA, memory, distributed, tensor, `torch.compile`, FSDP, GPU | **pytorch-engineering** | Foundation issues |
| NaN loss, converge, unstable, hyperparameters, gradients, LR, FP8, mixed precision | **training-optimization** | Training problems |
| Agent, policy, reward, environment, MDP, game, exploration, MARL | **deep-rl** | RL domain |
| LLM, fine-tune, RLHF, LoRA, prompt, instruction tuning, Claude / o-series / Llama / Mistral / Gemini, RAG, retrieval, embedding, reranker, prompt caching | **llm-specialist** | Language model applications |
| Reasoning models, "thinking tokens", extended thinking, o-series, R1, test-time compute, reasoning eval | **llm-specialist** | Inference-time reasoning is an LLM-application concern |
| Agent loop, tool use, MCP server/client, multi-agent orchestration, autonomous agent | **llm-specialist** | Agentic patterns are LLM-application concerns |
| Multimodal, VLM, vision-language, image+text, audio understanding | **llm-specialist** (application) and/or **neural-architectures** (modality fusion) | Routing depends on whether the question is "use a VLM" vs "design a fusion architecture" |
| Diffusion, flow matching, DiT, Stable Diffusion, image / video / audio generation | **neural-architectures** | Generative-media architecture |
| Which architecture, CNN vs transformer, Mamba vs attention, model selection | **neural-architectures** | Architecture choice |
| Deploy, serve, production, quantize, inference, latency, mobile, vLLM, SGLang, TensorRT-LLM, observability, drift | **ml-production** | Deployment |
| Network grows / prunes during training, continual learning, catastrophic forgetting, modular composition, MoE routing, adapter merging, PEFT (LoRA / QLoRA / DoRA / VeRA / PiSSA / LoftQ / LoRA+ / rsLoRA / LongLoRA) | **dynamic-architectures** | Networks that change topology / adapter composition over time |
| RL controller that decides WHEN / HOW to mutate a network's topology during training, growth actions, governor / safety gates, rollback-as-RL-signal, deterministic morphogenesis, ablation under topology change | **morphogenetic-rl** | The *controller* designing growth actions, not the network being grown (companion to dynamic-architectures) |
| ODE, integrator, physics sim, determinism, stability, replay, time-step, numerical methods | **simulation-foundations** | Simulation mathematics (often underpins RL environments) |
| Causal loop, feedback dynamics, leverage points, system archetypes, stock-flow, behavior-over-time | **systems-thinking** | Whole-system reasoning |

---

## Cross-Cutting Scenarios

When task spans domains, route to ALL relevant packs in execution order:

| Query | Route To | Order |
|-------|----------|-------|
| "Train RL agent and deploy" | deep-rl + ml-production | Train before deploy |
| "Fine-tune LLM with distributed training" | llm-specialist + pytorch-engineering | Domain first, then infrastructure |
| "LLM memory error during fine-tuning" | pytorch-engineering + llm-specialist | Foundation first |
| "RL training unstable" | training-optimization + deep-rl | General training first |
| "RL env determinism / replay broken" | simulation-foundations + deep-rl | Sim correctness before agent |
| "Continual-learning model forgets old tasks" | dynamic-architectures + training-optimization | Lifecycle design first, then training schedule |
| "RL controller decides when to grow my net, but the grown net trains badly" | morphogenetic-rl + dynamic-architectures (+ training-optimization) | Controller design first; HOW the grown net trains second; convergence tuning third |
| "Reasoning model is slow / expensive" | llm-specialist + ml-production | Application strategy first, serving second |
| "Build an agent that uses tools and a vector store" | llm-specialist (agentic + RAG) | Often single-pack; bring in `axiom-engineering-foundations` for system design if scope grows |
| "Diffusion model training diverges" | training-optimization + neural-architectures | General training first, architecture second |
| "Production LLM hallucinations / drift" | ml-production (observability) + llm-specialist (eval, RAG, prompting) | Detect before redesign |

**Principle**: Load in order of dependency. Fix foundation before domain. Complete training before deployment.

---

## Common Routing Mistakes

| Symptom | Wrong Route | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Train agent faster" | deep-rl | training-optimization FIRST | Could be general training issue |
| "LLM memory error" | llm-specialist | pytorch-engineering FIRST | Foundation issue |
| "Deploy RL model" | deep-rl | ml-production | Deployment problem |
| "Transformer for chess" | neural-architectures | deep-rl FIRST | RL problem |
| "Chatbot learning" | llm-specialist | ASK FIRST | Could be LLM OR RL |
| "My model forgets old data" | training-optimization | dynamic-architectures FIRST | Continual-learning lifecycle problem |
| "Replay diverges between machines" | deep-rl | simulation-foundations FIRST | Determinism / numerics problem |
| "o3 / extended thinking gives bad answers" | (guess) | llm-specialist (reasoning models sheet) | Reasoning-model prompting and eval differs from chat |
| "Build an MCP server / tool-using agent" | (none) | llm-specialist (agentic patterns) | Agent design lives with LLM applications |
| "RL agent that decides when to grow a network" | deep-rl | morphogenetic-rl FIRST | This is the controller-design pack; deep-rl alone misses governor/safety-gate/rollback patterns |
| "DoRA vs QLoRA for my 70B fine-tune" | llm-specialist | dynamic-architectures (PEFT comparison) + llm-specialist (fine-tune workflow) | Adapter method choice is the lifecycle pack's domain |

---

## Pressure Resistance - Critical Discipline

### Time/Emergency Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "Emergency means skip diagnostics" | Wrong diagnosis wastes MORE time | Fast systematic diagnosis IS emergency protocol |
| "Quick question means quick answer" | Wrong answer slower than 30-sec clarification | Ask ONE clarifying question |
| "Production down, no time for routing" | Wrong pack = longer outage | Correct routing (60 sec) prevents 20-min detour |

**Emergency Protocol**:
1. Acknowledge urgency
2. Fast clarification (30 sec)
3. Route to correct pack
4. Let pack provide emergency-appropriate approach

### Authority/Hierarchy Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "PM/architect said use X" | Authority can be wrong about routing | Verify task type regardless |
| "Questioning authority is risky" | Professional duty = correct routing | Frame as verification |
| "They have more context" | Context ≠ correct technical routing | Route based on problem type |

**Authority Protocol**: "I see [authority] suggested X - to apply it correctly, let me verify problem type"

### Sunk Cost Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "Already spent N hours in X, continue" | Sunk cost fallacy - wrong direction stays wrong | Cut losses immediately |
| "Redirecting invalidates their effort" | Correct routing validates effort by enabling success | Redirect now |
| "Too invested to change direction" | More investment in wrong direction = more waste | "Stop digging when in hole" |

**Sunk Cost Protocol**: "I see N hours invested - redirecting now prevents more wasted hours"

### Keyword/Anchoring Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "They mentioned transformer" | Keywords mislead; problem type matters | "Transformer for what problem type?" |
| "LLM mentioned, must be llm-specialist" | LLM could have foundation issues | Check problem type first |
| "They asked to 'fix RL'" | User's framing can be wrong | Verify RL is correct approach |
| "They said 'agent', must be deep-rl" | Modern "agent" usually means tool-using LLM, not RL | Distinguish RL agent (policy + reward) from LLM agent (tool loop + planning) |
| "They mentioned diffusion / DiT" | Could be training instability, serving cost, or architecture | Verify problem type before routing to neural-architectures |
| "They mentioned MoE" | Could be training (load balancing), architecture, or serving | Cross-cutting; clarify which slice of MoE they mean |

---

## Red Flags Checklist - STOP Immediately

### Basic Routing Red Flags
- ❌ "I'll guess this domain" → ASK clarifying question
- ❌ "They probably mean X" → Verify, don't assume
- ❌ "Just give generic advice" → Route to specific pack

### Time/Emergency Red Flags
- ❌ "Emergency means skip clarification" → Fast clarification IS emergency protocol
- ❌ "Production issue means guess quickly" → Wrong guess = longer outage
- ❌ "I'll skip asking to save time" → Clarifying (30 sec) faster than wrong route (5+ min)

### Authority/Social Red Flags
- ❌ "Authority figure suggested X, so route to X" → Verify task requirements
- ❌ "PM/senior has more context, trust them" → Route based on problem type
- ❌ "They're frustrated/exhausted, avoid redirect" → Continuing wrong path makes it worse

### Sunk Cost Red Flags
- ❌ "They invested N hours in X, continue there" → Sunk cost fallacy, cut losses
- ❌ "Redirecting invalidates their effort" → Correct routing enables success
- ❌ "Too much sunk cost to change direction" → More investment = more waste

### Keyword/Anchoring Red Flags
- ❌ "They mentioned transformer/CNN, discuss architecture" → Check problem type first
- ❌ "LLM/RL mentioned, route to that domain" → Could be foundation issue
- ❌ "Technical jargon means they know domain" → Vocabulary ≠ correct self-diagnosis

**All of these mean: Either ASK ONE clarifying question, or reconsider your routing logic.**

---

## Comprehensive Rationalization Prevention Table

| Pressure Type | Rationalization | Counter-Narrative | Correct Action |
|---------------|-----------------|-------------------|----------------|
| **Time** | "Emergency means skip diagnostics" | Wrong diagnosis wastes MORE time | "Fast clarification ensures fastest fix" |
| **Time** | "Quick question means quick answer" | Wrong answer slower than clarification | "Quick clarification prevents wrong path" |
| **Time** | "Production down, no time for routing" | Wrong pack = longer outage | "60-second routing prevents 20-minute detour" |
| **Authority** | "PM/architect said use X pack" | Authority can be wrong | "To apply X correctly, let me verify" |
| **Authority** | "Senior colleague suggested X" | Seniority ≠ correct routing | "To use suggestion effectively: [verify]" |
| **Sunk Cost** | "Already spent 6 hours in pack X" | Sunk cost fallacy | "Redirecting now prevents more wasted hours" |
| **Sunk Cost** | "Redirecting invalidates effort" | Correct routing enables success | "Redirect so effort succeeds" |
| **Keywords** | "User mentioned transformers" | Keywords mislead | "Clarifying problem type first" |
| **Keywords** | "They said LLM, route to llm-specialist" | LLM could have foundation issues | "Memory error is foundation issue" |
| **Anchoring** | "They asked to 'fix RL'" | User's framing can be wrong | "Before fixing, verify RL is correct" |
| **Complexity** | "Too many domains, just pick one" | Cross-cutting needs multi-pack | Route to ALL relevant packs |
| **Social** | "They're frustrated, don't redirect" | Continuing wrong path increases frustration | "Redirecting prevents more frustration" |
| **Demanding** | "They said 'just tell me', skip questions" | Tone doesn't change routing needs | "To help effectively, I need: [question]" |

---

## When NOT to Use Yzmir Skills

**Skip AI/ML skills when:**
- Simple data processing without ML.
- Statistical analysis without neural networks.
- Data cleaning / ETL without model training.
- The task is pure system architecture (route to `axiom-solution-architect`), pure DevOps (route to `axiom-devops-engineering`), or pure security threat modeling for the surrounding system (route to `ordis-security-architect`).

**Edge cases worth naming:**
- *"Build a chatbot"* with no ML training and no fine-tuning — usually a Yzmir question (`llm-specialist`: prompting, RAG, agent loop) but bring in `axiom-solution-architect` if the request is really about system design.
- *"Why does my agent's tool call fail?"* — usually `llm-specialist` (agentic patterns), but if it's a tool-runtime / sandbox / IPC issue it's `axiom-engineering-foundations`.
- *"Make my LLM responses cheaper"* — `llm-specialist` (prompt caching, model routing, smaller models) and/or `ml-production` (serving stack). Both are valid.

**Red flag**: If you're not training, deploying, prompting, retrieving for, or evaluating a model, you probably don't need Yzmir.

---

## Routing Summary Flowchart

```
User Query
    ↓
Is query ambiguous? → YES → ASK clarifying question
    ↓ NO
Identify problem type:
    - Framework error / PyTorch API? → pytorch-engineering
    - Training not working / optimizer / precision? → training-optimization
    - RL problem? → deep-rl
    - LLM application (prompt, RAG, fine-tune, reasoning, agent, MCP)? → llm-specialist
    - Architecture choice (CNN/transformer/Mamba/diffusion)? → neural-architectures
    - Production deployment / serving / observability? → ml-production
    - Network grows/prunes / continual learning / PEFT (LoRA/QLoRA/DoRA/...) / MoE composition? → dynamic-architectures
    - RL controller deciding WHEN/HOW to grow a network (governor, safety gates, rollback)? → morphogenetic-rl
    - Simulation math / determinism / ODEs? → simulation-foundations
    - Whole-system feedback / causal loops / leverage? → systems-thinking
    ↓
Cross-cutting? → YES → Route to MULTIPLE packs (order by dependency)
    ↓ NO
Route to single pack
```

---

## Examples

See [routing-examples.md](routing-examples.md) for detailed worked examples:
- Ambiguous queries
- Cross-cutting scenarios
- Misleading keywords
- Time pressure handling
- Foundation issues disguised as domain issues
- Emergency + authority pressure
- Sunk cost + frustration
- Multiple pressures combined

---

## AI Engineering Plugin Router Catalog

This meta-router directs you to the appropriate Yzmir AI/ML plugin. The Yzmir faction ships **10 specialist packs** plus this router:

1. **yzmir-pytorch-engineering** — PyTorch framework: CUDA, memory, FSDP/`torch.compile`, distributed, tensor operations.
2. **yzmir-training-optimization** — Training problems: optimizers, schedules, precision (BF16/FP8), gradients, convergence, hyperparameters.
3. **yzmir-deep-rl** — Reinforcement learning: agents, policies, rewards, environments, MDP, offline RL, MARL.
4. **yzmir-llm-specialist** — LLM applications: prompting, RAG, fine-tuning (SFT/DPO/GRPO), reasoning models, agentic patterns / MCP, multimodal use, prompt caching, evaluation, safety.
5. **yzmir-neural-architectures** — Architecture selection: CNN / transformer / Mamba / GNN / diffusion / DiT / multimodal fusion, capacity and depth-width tradeoffs.
6. **yzmir-ml-production** — Production: vLLM / SGLang / TensorRT-LLM serving, quantization (`torch.ao.quantization`, AWQ, GPTQ, FP8), MLOps, observability (Phoenix / Langfuse / OTel GenAI), drift, scaling.
7. **yzmir-dynamic-architectures** — Networks that grow / prune / adapt: continual learning, gradient isolation, modular composition, MoE routing, adapter merging, PEFT (LoRA / QLoRA / DoRA / VeRA / PiSSA / LoftQ / LoRA+ / rsLoRA / LongLoRA), lifecycle. *Owns the HOW: how the growable / adaptable network trains.*
8. **yzmir-morphogenetic-rl** — RL controllers that decide WHEN and HOW to mutate a network's topology during training: action/observation/reward design for the controller, governor and safety gates, rollback-as-RL-signal, deterministic morphogenesis, growth-aware ablation. *Companion to dynamic-architectures: owns the CONTROLLER that drives growth, not the network being grown.*
9. **yzmir-simulation-foundations** — Simulation mathematics: ODEs, integrators, stability, control theory, determinism — often the foundation under RL environments and physics-based systems.
10. **yzmir-systems-thinking** — Systems thinking methodology: causal loops, leverage points, archetypes, stocks-flows, behavior-over-time graphs.

**Adjacent (non-Yzmir) routers worth knowing about:**
- `axiom-engineering-foundations` — general software-engineering rigor for AI systems (debugging, refactoring, code review).
- `axiom-solution-architect` — when AI is one component of a larger system that needs an architecture document.
- `ordis-security-architect` — LLM threat modeling, prompt injection, exfil, AI supply chain.
- `axiom-python-engineering` — Python tooling foundations underneath PyTorch / Transformers code.

**Remember**: When in doubt, ASK. Clarification takes seconds, wrong routing takes minutes. **Knowledge cutoff awareness**: model IDs and provider features evolve quickly — capability-tier framing in downstream packs (frontier reasoning / frontier general / fast-cheap / on-device) is intentional. Check provider docs for current model IDs before quoting them in user-facing answers.
