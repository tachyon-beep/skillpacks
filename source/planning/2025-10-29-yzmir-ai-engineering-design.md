# Yzmir AI/ML Engineering Skillpacks - Design Document

**Created**: 2025-10-29
**Status**: Phase 4 Complete - Design Validated and Documented
**Faction**: Yzmir (Magicians of the Mind)

---

## Executive Summary

Comprehensive AI/ML engineering skillpack collection for the Yzmir faction, covering the entire deep learning/ML engineering landscape. Phase 1 delivers 6 foundational packs (~53-65 skills) with full RED-GREEN-REFACTOR testing discipline.

**Target Audience**: Beginners and intermediates who want to create expert-level code
**Design Principle**: Channel existing model knowledge rather than teach from scratch - provide decision frameworks, expert patterns, common pitfalls, and systematic methodologies

---

## Project Scope

### Phase 1 (Foundation Release)
**6 core packs:**
1. **pytorch-engineering** - Foundation layer (tensor ops, distributed training, profiling)
2. **neural-architectures** - Component catalog (CNNs, Transformers, GANs, etc.)
3. **training-optimization** - Making training work (optimizers, convergence, debugging)
4. **deep-rl** - Complete RL algorithms (passion project - DQN, PPO, SAC, offline RL)
5. **llm-specialist** - Modern LLM techniques (fine-tuning, RLHF, inference optimization)
6. **ml-production** - Production deployment (serving, quantization, monitoring)

**Estimated effort**: 70-220 hours (2-4 hours per skill × ~53-65 skills)

### Future Phases (Planned)
**Phase 2+** will add:
- computer-vision (object detection, segmentation, image generation)
- nlp-fundamentals (classical NLP, sequence tasks)
- audio-speech (ASR, TTS, audio classification)
- multimodal-ai (CLIP, image captioning, vision-language)
- time-series-forecasting (deep learning + classical approaches)
- tabular-ml (deep learning on structured data, gradient boosting)
- recommendation-systems (collaborative filtering, neural approaches)
- data-engineering-ml (pipelines, feature stores, data versioning)
- classical-ml (for baselines and understanding)
- ai-safety-ethics (bias, fairness, adversarial robustness)
- ml-research-methods (paper reading, experiment design, reproducibility)
- edge-embedded-ai (model compression, on-device deployment)
- graph-ml (expanded GNN coverage)
- probabilistic-ml (Bayesian NNs, uncertainty quantification)
- automl-nas (architecture search, hyperparameter optimization)
- active-learning-hitl (query strategies, weak supervision)
- synthetic-data-simulation (domain randomization, sim-to-real)
- scientific-ml (physics-informed NNs, neural ODEs)

**Total vision**: 20+ packs covering entire AI/ML landscape

---

## Architecture

### 3-Level Nested Routing Hierarchy

**Level 1 - Primary Router**:
- `yzmir/ai-engineering-expert/using-ai-engineering`
- Entry point that analyzes user's task and routes to appropriate domain pack
- Handles cross-cutting concerns and integration questions

**Level 2 - Pack Meta-Skills**:
Each pack has its own meta-skill for internal routing:
- `yzmir/pytorch-engineering/using-pytorch-engineering`
- `yzmir/neural-architectures/using-neural-architectures`
- `yzmir/training-optimization/using-training-optimization`
- `yzmir/deep-rl/using-deep-rl`
- `yzmir/llm-specialist/using-llm-specialist`
- `yzmir/ml-production/using-ml-production`

**Level 3 - Specific Skills**:
Individual techniques and methodologies (e.g., `yzmir/deep-rl/policy-gradient-methods`)

### Why This Architecture?

**Scalability**: Supports 20+ packs without restructuring. Each new domain becomes its own pack.

**Mental model alignment**: Experts think in domains ("working on RL" vs "deploying models"). Users naturally know which pack they need.

**Modularity**: Load only what you need. RL work doesn't need LLM skills loaded.

**Independent evolution**: RL techniques evolve differently than production practices. Separate packs can grow independently.

**Proven at scale**: Matches how PyTorch docs, HuggingFace, Papers with Code organize content.

**Trade-off**: More initial overhead (7 meta-skills in Phase 1) but only architecture that scales to 20+ packs.

---

## Design Principles

### 1. Channel Knowledge, Don't Teach
Model already knows PyTorch and RL algorithms. Skills provide:
- **Decision frameworks** - When PPO vs SAC, when to use distributed training
- **Expert patterns** - How to structure training loops, module design best practices
- **Common pitfalls** - NaN losses, gradient explosions, training instability, what breaks and why
- **Systematic methodology** - Debugging approaches, hyperparameter selection frameworks

### 2. Meta-Awareness Over Exhaustive Lists
Teach patterns and methodology, not comprehensive references. Example: Don't list every optimizer with full math; teach framework for selecting optimizers based on problem characteristics.

### 3. Universal Applicability
Build standalone, universally applicable skills. Avoid project-specific examples (no ELSPETH/Duskmantle references). Skills should work for any ML project.

### 4. Beginner → Expert Code Quality
Target beginners and intermediates who want expert-level output. Provide guardrails and patterns that elevate code quality systematically.

### 5. Full Testing Discipline
RED-GREEN-REFACTOR methodology for EVERY skill:
- **RED**: Test scenarios without skill → document failures
- **GREEN**: Write skill addressing baseline failures
- **REFACTOR**: Test under pressure, close loopholes
Same rigor as Ordis/Muna (2-4 hours per skill)

### 6. Deferred Integration
Phase 1 builds Yzmir standalone. Cross-references to Ordis (security/adversarial robustness) and Muna (documentation/experiment logs) added in Phase 2+ after validation.

---

## Phase 1 Pack Details

### 1. pytorch-engineering (~8-10 skills)

**Purpose**: Foundation layer - everything else builds on this

**Meta-skill**: Routes based on task type (performance, distributed, debugging)

**Core Skills**:
- Tensor operations and memory management patterns
- Module design best practices (nn.Module patterns, hooks, custom layers)
- Distributed training strategies (DDP, FSDP, DeepSpeed when/why)
- Mixed precision and gradient scaling
- Performance profiling and optimization
- Debugging techniques (NaN tracking, gradient inspection)
- Checkpointing and reproducibility
- Custom autograd functions

**Testing focus**: Performance bottlenecks, distributed training edge cases, debugging under pressure

---

### 2. neural-architectures (~8-10 skills)

**Purpose**: Component catalog and architecture selection

**Meta-skill**: Routes based on data type (images, sequences, graphs, etc.)

**Core Skills**:
- CNN families and when to use (ResNet, EfficientNet, ViT)
- Sequence models comparison (RNN/LSTM/GRU vs Transformer)
- Transformer architecture deep-dive
- Attention mechanisms catalog
- Generative model families (VAE, GAN, Diffusion)
- Graph neural networks basics
- Normalization techniques (BatchNorm, LayerNorm, RMSNorm)
- Architecture design principles and patterns

**Testing focus**: Architecture selection decisions, component composition, design trade-offs

---

### 3. training-optimization (~8-10 skills)

**Purpose**: Making training actually work

**Meta-skill**: Routes based on symptom (won't converge, unstable, too slow, etc.)

**Core Skills**:
- Optimizer selection framework (Adam, AdamW, Lion, when each)
- Learning rate schedules and warmup strategies
- Batch size selection and gradient accumulation
- Loss landscape understanding
- Gradient flow debugging (vanishing/exploding)
- Convergence diagnostics and early stopping
- Regularization techniques catalog
- Hyperparameter tuning methodology
- Training instability troubleshooting

**Testing focus**: Debugging training failures, optimization decisions, convergence issues

---

### 4. deep-rl (~10-12 skills)

**Purpose**: Complete RL implementation guide (passion project)

**Meta-skill**: Routes based on problem type (discrete/continuous, on-policy/off-policy, model-based, etc.)

**Core Skills**:
- RL fundamentals (MDPs, Bellman equations, value/policy)
- Value-based methods (DQN, Double DQN, Rainbow)
- Policy gradient methods (REINFORCE, PPO, TRPO)
- Actor-critic algorithms (A3C, SAC, TD3)
- Offline RL approaches (CQL, IQL, BCQ)
- Model-based RL (Dreamer, MuZero patterns)
- Experience replay strategies
- Exploration techniques
- Reward shaping and engineering
- RL debugging methodology (credit assignment, reward hacking)
- Environment design patterns
- RL evaluation and benchmarking

**Testing focus**: Algorithm selection, RL-specific debugging, reward design, evaluation methodology

---

### 5. llm-specialist (~6-8 skills)

**Purpose**: Modern LLM techniques

**Meta-skill**: Routes based on task (pre-training, fine-tuning, inference, alignment)

**Core Skills**:
- Transformer architecture for LLMs (differences from vision transformers)
- Tokenization strategies and vocabulary design
- Fine-tuning methods (full, LoRA, QLoRA, prefix tuning)
- Alignment techniques (RLHF, DPO, constitutional AI)
- Long-context techniques (RoPE, ALiBi, sliding window)
- Inference optimization (KV cache, quantization, speculative decoding)
- Prompt engineering patterns
- LLM evaluation frameworks

**Testing focus**: Fine-tuning decisions, alignment approaches, inference optimization trade-offs

---

### 6. ml-production (~6-8 skills)

**Purpose**: Shipping models to production

**Meta-skill**: Routes based on deployment context (edge, server, batch, real-time)

**Core Skills**:
- Model serving patterns (TorchServe, ONNX, TensorRT)
- Quantization for inference (GPTQ, AWQ, post-training)
- Model compression techniques
- Performance benchmarking methodology
- Production monitoring and observability
- Experiment tracking best practices (Wandb, MLflow)
- Deployment patterns (A/B testing, canary, shadow)
- Hardware optimization strategies

**Testing focus**: Deployment decisions, optimization trade-offs, production debugging

---

## Skill Count Summary

**Phase 1 Total**:
- 6 packs
- 7 meta-skills (1 primary + 6 pack-level)
- ~46-58 specific skills
- **Total: ~53-65 skills**

**Effort estimate**: 2-4 hours per skill × 53-65 skills = **106-260 hours** for Phase 1

---

## Testing Methodology

### RED-GREEN-REFACTOR for Every Skill

**RED Phase** (Baseline Without Skill):
- Create test scenario based on skill type
- Dispatch subagent WITHOUT skill loaded
- Document failures, rationalizations, patterns
- Example: "Implement PPO for CartPole" → Agent uses wrong advantage estimation, unstable training

**GREEN Phase** (Write Skill):
- Write SKILL.md addressing baseline failures
- Test WITH skill loaded
- Verify agent now applies correct patterns
- Example: Agent now uses GAE, proper value clipping, entropy regularization

**REFACTOR Phase** (Close Loopholes):
- Add pressure (time constraints, sunk cost, authority)
- Find new rationalizations
- Add explicit counters
- Re-test until bulletproof

### Skill Types and Testing

**Discipline skills** (routing, methodology): Full pressure testing (time + sunk cost + authority)

**Technique skills** (implementing algorithms): Application scenarios (build X correctly)

**Pattern skills** (design decisions): Recognition scenarios (when to apply)

**Reference skills** (catalogs): Retrieval scenarios (find and use correct info)

---

## Directory Structure

```
source/
└── yzmir/
    ├── ai-engineering-expert/
    │   └── using-ai-engineering/
    │       └── SKILL.md (primary router)
    ├── pytorch-engineering/
    │   ├── using-pytorch-engineering/
    │   │   └── SKILL.md (pack meta-skill)
    │   ├── tensor-operations-and-memory/
    │   │   └── SKILL.md
    │   ├── module-design-patterns/
    │   │   └── SKILL.md
    │   ├── distributed-training-strategies/
    │   │   └── SKILL.md
    │   └── ... (5-7 more skills)
    ├── neural-architectures/
    │   ├── using-neural-architectures/
    │   │   └── SKILL.md
    │   └── ... (8-10 skills)
    ├── training-optimization/
    │   ├── using-training-optimization/
    │   │   └── SKILL.md
    │   └── ... (8-10 skills)
    ├── deep-rl/
    │   ├── using-deep-rl/
    │   │   └── SKILL.md
    │   └── ... (10-12 skills)
    ├── llm-specialist/
    │   ├── using-llm-specialist/
    │   │   └── SKILL.md
    │   └── ... (6-8 skills)
    └── ml-production/
        ├── using-ml-production/
        │   └── SKILL.md
        └── ... (6-8 skills)
```

---

## Implementation Phases

### Phase 1A: Foundation (First)
1. **ai-engineering-expert** (primary router) - 2-3 hours
2. **pytorch-engineering** (entire pack) - 20-40 hours
3. **training-optimization** (entire pack) - 20-40 hours

**Rationale**: Establish infrastructure and core enablers first. Every other pack depends on these.

### Phase 1B: Deep RL (Passion Project)
4. **deep-rl** (entire pack) - 25-50 hours

**Rationale**: Largest pack, highest personal interest, can validate independently.

### Phase 1C: Modern AI
5. **neural-architectures** (entire pack) - 20-40 hours
6. **llm-specialist** (entire pack) - 15-30 hours
7. **ml-production** (entire pack) - 15-30 hours

**Rationale**: Complete Phase 1 with modern techniques and production deployment.

**Phase 1 Total**: 117-233 hours

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All 7 meta-skills pass RED-GREEN-REFACTOR
- ✅ All ~46-58 specific skills pass RED-GREEN-REFACTOR
- ✅ Primary router correctly routes to all 6 packs
- ✅ Pack meta-skills correctly route to specific skills
- ✅ Skills demonstrate expert-level code generation for beginners/intermediates
- ✅ Standalone validation (no dependencies on Ordis/Muna)
- ✅ All skills committed to git with proper testing documentation

### Quality Gates:
- Each skill tested under pressure (time, sunk cost, authority)
- Agents apply patterns correctly without rationalization
- Beginners using skills produce expert-level code
- Common pitfalls systematically avoided
- Decision frameworks lead to correct choices

---

## Future Phases

### Phase 2: Domain Applications (Computer Vision, NLP, Audio)
- computer-vision (~10-12 skills)
- nlp-fundamentals (~8-10 skills)
- audio-speech (~6-8 skills)
- multimodal-ai (~6-8 skills)

### Phase 3: Specialized Techniques
- time-series-forecasting (~6-8 skills)
- tabular-ml (~6-8 skills)
- recommendation-systems (~6-8 skills)
- graph-ml (~6-8 skills)

### Phase 4: Data & Research Infrastructure
- data-engineering-ml (~8-10 skills)
- classical-ml (~6-8 skills)
- ml-research-methods (~6-8 skills)
- ai-safety-ethics (~8-10 skills)

### Phase 5: Advanced Topics
- edge-embedded-ai (~6-8 skills)
- probabilistic-ml (~6-8 skills)
- automl-nas (~6-8 skills)
- active-learning-hitl (~6-8 skills)
- synthetic-data-simulation (~6-8 skills)
- scientific-ml (~6-8 skills)

**Ultimate Vision**: 20-25 packs covering entire AI/ML engineering landscape, all at production quality with full RED-GREEN-REFACTOR testing.

---

## Cross-References (Phase 2+)

### Ordis (Security) Integration:
- Adversarial robustness testing
- Model security and poisoning attacks
- Privacy-preserving ML (differential privacy)
- Secure model deployment

### Muna (Documentation) Integration:
- Model cards and documentation
- Experiment logging best practices
- Technical writing for ML papers
- Documentation for ML systems

**Note**: These integrations deferred to Phase 2+ to validate Yzmir standalone first.

---

---

## Skill Design Pattern

Each specific skill (Level 3) follows this pattern:

### YAML Frontmatter

```yaml
---
name: policy-gradient-methods
description: Use when implementing RL with continuous action spaces or when you need direct policy optimization - covers REINFORCE, PPO, TRPO with implementation patterns and common pitfalls
---
```

### Skill Content Structure

1. **Overview** - Core principle in 1-2 sentences
2. **When to Use** - Symptoms/triggers (bullet list)
3. **Decision Framework** - When X vs Y (table or flowchart)
4. **Expert Patterns** - How to implement correctly
5. **Common Pitfalls** - What breaks and why
6. **Debugging Methodology** - Systematic troubleshooting
7. **Examples** - One excellent example showing pattern

### Key Differences from Reference Docs

- NOT a comprehensive algorithm explanation (model knows this)
- FOCUS on decision points (when PPO vs SAC)
- FOCUS on implementation gotchas (advantage normalization, value clipping)
- FOCUS on debugging (reward not improving → check advantage computation)

### Example Structure

See the detailed `policy-gradient-methods` example in the deep-rl pack breakdown section below for a complete skill implementation following this pattern.

---

## Testing Strategy (RED-GREEN-REFACTOR Applied)

Each of the ~53-65 skills follows RED-GREEN-REFACTOR methodology from superpowers:writing-skills.

### Skill Type Categorization

**Level 1-2 Meta-Skills (7 total) - Routing/Decision Skills**:
- Type: **Discipline-enforcing** (route correctly, don't guess)
- Testing: Pressure scenarios with time constraints, unclear requirements
- Example RED scenario: "Build RL for robotics" → Agent jumps to implementation without loading deep-rl skill
- Example GREEN: Agent loads `using-ai-engineering` → routes to `using-deep-rl` → routes to `continuous-control-algorithms`

**Technique Skills (~35-40)**:
- Type: **Technique** (how to implement correctly)
- Testing: Application scenarios with edge cases
- Example: "Implement PPO" → RED: Missing advantage normalization, wrong clipping → GREEN: Correct GAE, proper value clipping

**Decision Framework Skills (~8-10)**:
- Type: **Pattern** (when to use X vs Y)
- Testing: Recognition scenarios
- Example: "Choose optimizer for transformer" → RED: Defaults to Adam → GREEN: Considers batch size, applies AdamW with proper weight decay

**Reference Skills (~5-8)**:
- Type: **Reference** (architecture catalog, API patterns)
- Testing: Retrieval and application scenarios
- Example: "What attention mechanism for long sequences?" → RED: Generic self-attention → GREEN: Finds and applies sparse attention or FlashAttention

### Testing by Pack

**pytorch-engineering**: Heavy on technique + pitfall avoidance
- RED scenarios: Memory leaks, wrong distributed setup, no gradient checkpointing
- REFACTOR focus: Performance pitfalls under time pressure

**neural-architectures**: Heavy on decision frameworks
- RED scenarios: Wrong architecture for task type, over-engineering
- REFACTOR focus: "This architecture is simpler" rationalizations

**training-optimization**: Heavy on debugging methodology
- RED scenarios: NaN losses, won't converge, too slow
- REFACTOR focus: Systematic debugging vs random hyperparameter changes

**deep-rl**: Mixed technique + debugging
- RED scenarios: Wrong algorithm, unstable training, reward hacking
- REFACTOR focus: RL-specific failure modes under deadline pressure

**llm-specialist**: Mixed technique + decision frameworks
- RED scenarios: Inefficient fine-tuning, wrong alignment approach
- REFACTOR focus: "Just use full fine-tuning" when LoRA appropriate

**ml-production**: Heavy on decision frameworks + pitfalls
- RED scenarios: Wrong serving pattern, inefficient quantization
- REFACTOR focus: Premature optimization vs necessary optimization

### Rationalization Tables to Build

**For routing meta-skills**:
| Excuse | Reality |
|--------|---------|
| "I already know the algorithm" | Skills provide expert patterns, not just algorithms |
| "This is simple, no need for skill" | Simple problems have subtle pitfalls |
| "I'll load skill if I get stuck" | Loading skill prevents getting stuck |

**For training-optimization**:
| Excuse | Reality |
|--------|---------|
| "Let me try random changes" | Systematic debugging finds root cause faster |
| "I'll just increase learning rate" | Could make it worse; diagnose first |

**For deep-rl**:
| Excuse | Reality |
|--------|---------|
| "PPO works for everything" | SAC better for off-policy, continuous |
| "Reward is simple" | Reward shaping critically affects learning |

---

## Detailed Pack Example: Deep RL

Since deep-rl is the passion project and largest pack, here's its complete structure:

### Pack Structure: yzmir/deep-rl/

**Meta-Skill**: `using-deep-rl` (2-3 hours)

**Routes based on**:
- Action space (discrete vs continuous)
- Data availability (on-policy vs off-policy vs offline)
- Sample efficiency needs
- Stability requirements
- Model-based vs model-free

**Example routing**:
- "Robotics arm control" → continuous-control-algorithms → SAC or TD3
- "Atari games" → value-based-methods → DQN or Rainbow
- "Stable policy learning" → policy-gradient-methods → PPO
- "Learning from demonstrations" → offline-rl-methods → IQL or CQL

### Core Skills (10-12 skills)

**1. rl-fundamentals** (3-4 hours)
- MDPs, Bellman equations, value vs policy
- When RL applies (vs supervised/unsupervised)
- Problem formulation patterns
- Testing: Can agent correctly formulate RL problem from description?

**2. value-based-methods** (3-4 hours)
- DQN, Double DQN, Dueling DQN, Rainbow
- When to use (discrete actions, off-policy)
- Target network patterns
- Testing: Correct algorithm selection, proper target network usage

**3. policy-gradient-methods** (3-4 hours)
- REINFORCE, PPO, TRPO
- Advantage estimation (GAE)
- Trust regions and clipping
- Testing: Proper advantage normalization, clipping implementation

**4. actor-critic-algorithms** (3-4 hours)
- A3C, SAC, TD3
- Continuous control patterns
- Entropy regularization
- Testing: Correct twin-Q for TD3, temperature tuning for SAC

**5. offline-rl-methods** (3-4 hours)
- CQL, IQL, BCQ
- When to use (limited interaction, batch data)
- Pessimism and conservatism patterns
- Testing: Correct pessimism application, avoiding overestimation

**6. model-based-rl** (3-4 hours)
- World models (Dreamer patterns)
- Planning approaches
- When model-based helps
- Testing: Model learning validation, planning integration

**7. experience-replay-strategies** (2-3 hours)
- Uniform vs prioritized replay
- Replay buffer sizing
- Off-policy correction
- Testing: Proper buffer management, priority updates

**8. exploration-techniques** (2-3 hours)
- Epsilon-greedy, entropy bonuses, curiosity
- Exploration-exploitation trade-offs
- Testing: Appropriate exploration for problem type

**9. reward-shaping-engineering** (3-4 hours)
- Reward design principles
- Common pitfalls (reward hacking)
- Auxiliary rewards
- Testing: Detecting reward hacking, proper shaping

**10. rl-debugging-methodology** (3-4 hours)
- Systematic debugging (credit assignment, stability)
- Diagnostic plots (returns, Q-values, entropy)
- Common failure modes
- Testing: Systematic debugging vs random changes

**11. environment-design-patterns** (2-3 hours)
- Gym/Gymnasium API patterns
- Observation/action space design
- Environment wrappers
- Testing: Proper environment interface implementation

**12. rl-evaluation-benchmarking** (2-3 hours)
- Evaluation protocols
- Statistical significance
- Benchmark selection
- Testing: Proper evaluation methodology, avoiding cherry-picking

### Example Skill: policy-gradient-methods

```markdown
---
name: policy-gradient-methods
description: Use when implementing RL with continuous action spaces, need stable training with trust regions, or converting from value-based methods - covers REINFORCE, PPO, TRPO with advantage estimation, clipping patterns, and training stability techniques
---

# Policy Gradient Methods

## Overview
Direct policy optimization algorithms that learn stochastic policies through gradient ascent on expected returns. Best for continuous actions and when you need guaranteed monotonic improvement.

## When to Use

**Use policy gradients when:**
- Continuous action spaces (robotics, control problems)
- Need stable, predictable training (PPO)
- Stochastic policies beneficial
- On-policy learning acceptable

**Don't use when:**
- Sample efficiency critical → Use SAC (off-policy)
- Discrete actions, need simplicity → Consider DQN first
- Deterministic policy sufficient → Consider DDPG/TD3

**Symptoms triggering this skill:**
- "DQN doesn't work for continuous actions"
- "Training is unstable, wild policy updates"
- "Need guaranteed improvement per update"

## Algorithm Selection Framework

| Situation | Algorithm | Why |
|-----------|-----------|-----|
| General purpose, stable | **PPO** | Best default - clipping prevents bad updates, works everywhere |
| Maximum stability, research | **TRPO** | Strict KL constraint, very safe but slow |
| Sample efficiency matters | **SAC** | Off-policy, not pure policy gradient but often better |
| Learning from scratch | **PPO** | Most robust to hyperparameters |
| Fine-tuning existing policy | **PPO or TRPO** | Safe small updates |

**Default recommendation**: Start with PPO. Only use TRPO if you need theoretical guarantees or PPO unstable.

## Core Components

### 1. Advantage Estimation (CRITICAL)

**Always use GAE (Generalized Advantage Estimation)**:

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages.

    WHY GAE: Trades off bias vs variance. Lambda=1 is Monte Carlo (high variance),
    lambda=0 is TD(0) (high bias). Lambda=0.95-0.99 works well in practice.
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)
```

**CRITICAL: Normalize advantages**:
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 2. PPO Clipping

**Clipped surrogate objective**:

```python
def ppo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    PPO clipped loss.

    WHY: Prevents destructively large policy updates. If new policy very different
    from old, clipping limits the update magnitude.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
```

**Typical clip_epsilon**: 0.1-0.2 (0.2 is standard)

### 3. Value Function Clipping (Often Missed)

```python
def value_loss(old_values, new_values, returns, clip_epsilon=0.2):
    """
    Clipped value loss - less common but improves stability.

    WHY: Prevents value function from changing too rapidly, which can
    destabilize advantage estimates.
    """
    clipped_values = old_values + torch.clamp(
        new_values - old_values, -clip_epsilon, clip_epsilon
    )

    loss1 = (new_values - returns).pow(2)
    loss2 = (clipped_values - returns).pow(2)
    return torch.max(loss1, loss2).mean()
```

### 4. Entropy Bonus

```python
entropy_coef = 0.01  # Typical value
loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
```

**WHY entropy**: Encourages exploration, prevents premature convergence to deterministic policy.

**Typical range**: 0.01-0.001 (decrease over training if needed)

## Expert Patterns

### Training Loop Structure

```python
# Collect trajectories
for episode in range(num_episodes):
    trajectory = collect_trajectory(env, policy)
    buffer.add(trajectory)

# Multiple epochs per batch (KEY for PPO)
for epoch in range(3, 10):  # 3-10 epochs typical
    for batch in buffer.get_batches():
        advantages = compute_gae(batch)
        advantages = normalize(advantages)  # CRITICAL

        loss = ppo_loss(batch.old_log_probs, policy(batch.states), advantages)
        optimizer.step()
```

**KEY**: Multiple epochs (3-10) over same data. This is what makes PPO sample-efficient.

### Hyperparameter Ranges

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| clip_epsilon | 0.1 - 0.2 | 0.2 standard |
| gamma | 0.99 - 0.999 | Higher for long-horizon |
| lambda (GAE) | 0.95 - 0.99 | 0.95 good default |
| learning_rate | 3e-4 | Standard for PPO |
| epochs | 3 - 10 | More epochs = less bias, more compute |
| batch_size | 64 - 4096 | Larger for complex envs |

## Common Pitfalls

❌ **No advantage normalization**
→ Symptom: Unstable learning, wild swings
→ Fix: Always normalize: `(adv - mean) / (std + 1e-8)`

❌ **Wrong value target**
→ Symptom: Value function doesn't learn well
→ Fix: Use returns or TD(λ) targets, not just rewards

❌ **Single epoch per batch**
→ Symptom: Poor sample efficiency
→ Fix: Use 3-10 epochs per batch (more reuse of data)

❌ **Too large clip_epsilon**
→ Symptom: Training instability, policy collapse
→ Fix: Use 0.1-0.2, don't increase

❌ **No entropy bonus**
→ Symptom: Premature convergence, deterministic policy too early
→ Fix: Add entropy bonus (0.01 typical)

❌ **Forgetting to detach old log probs**
→ Symptom: Gradient flows through old policy
→ Fix: `old_log_probs = old_log_probs.detach()`

## Debugging Methodology

### Symptom: Policy improves then collapses

**Diagnostic steps**:
1. Check value function accuracy (plot predicted vs actual returns)
2. Check if clipping is too loose (reduce clip_epsilon)
3. Check advantage normalization (should be mean≈0, std≈1)
4. Check if updates too large (reduce learning rate)

### Symptom: Policy not improving

**Diagnostic steps**:
1. Check if advantages are all similar (no signal)
2. Check if entropy too high (policy too stochastic)
3. Check if clipping too tight (almost no updates)
4. Check reward scale (very small rewards → scale up)

### Symptom: High variance in returns

**Diagnostic steps**:
1. Increase GAE lambda (reduce variance, increase bias)
2. Increase batch size (more stable gradients)
3. Reduce learning rate
4. Add value function clipping

## When to Switch Algorithms

**Switch to SAC when:**
- Sample efficiency critical (SAC is off-policy)
- PPO unstable despite tuning
- Need maximum entropy policy

**Switch to TRPO when:**
- PPO still too unstable
- Research setting, need theoretical guarantees
- Can afford computational cost

**Switch to value-based when:**
- Discrete actions (DQN simpler)
- Deterministic policy fine

## Red Flags - Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Skip advantage normalization, it's just scaling" | Causes instability. Always normalize. |
| "One epoch is enough" | PPO's sample efficiency comes from multiple epochs. Use 3-10. |
| "Don't need entropy bonus" | Prevents exploration. Use small bonus (0.01). |
| "Clipping seems complicated, skip it" | Clipping is what makes PPO stable. Don't skip. |

## References

- Schulman et al. (2017): PPO paper
- Schulman et al. (2015): TRPO paper, GAE paper
- See also: `yzmir/deep-rl/actor-critic-algorithms` for SAC/TD3 comparison
- See also: `yzmir/training-optimization/debugging-training-failures` for general debugging
```

---

## Future Phases Roadmap

### Phase 2: Domain Applications (80-120 hours)

Focus on applied ML domains that build on Phase 1 foundation.

**Packs**:
- **computer-vision** (10-12 skills): Object detection, segmentation, image generation, video understanding
- **nlp-fundamentals** (8-10 skills): Classical NLP, sequence tasks, text classification, NER, embeddings
- **audio-speech** (6-8 skills): ASR, TTS, audio classification, speaker recognition
- **multimodal-ai** (6-8 skills): Vision-language models, CLIP patterns, image captioning

**Why Phase 2**: These build on neural-architectures and training-optimization. Real-world applications that users frequently need.

**Cross-references added**:
- Link back to Phase 1 packs (architectures, training)
- Begin Ordis integration (adversarial examples in CV)
- Begin Muna integration (model documentation)

---

### Phase 3: Specialized Techniques (60-90 hours)

Focus on domain-specific ML approaches.

**Packs**:
- **time-series-forecasting** (6-8 skills): Classical + deep learning approaches, anomaly detection
- **tabular-ml** (6-8 skills): Deep learning on structured data, gradient boosting, feature engineering
- **recommendation-systems** (6-8 skills): Collaborative filtering, neural CF, ranking
- **graph-ml** (6-8 skills): Expanded GNN coverage, heterogeneous graphs, knowledge graphs

**Why Phase 3**: Common enterprise use cases. Less "cutting edge" than Phase 1-2 but high practical value.

---

### Phase 4: Infrastructure & Research (70-100 hours)

Focus on production infrastructure and research methods.

**Packs**:
- **data-engineering-ml** (8-10 skills): Pipelines, feature stores, data versioning, validation
- **classical-ml** (6-8 skills): Baselines, when to use classical vs deep learning
- **ml-research-methods** (6-8 skills): Paper reading, experiment design, reproducibility
- **ai-safety-ethics** (8-10 skills): Bias detection, fairness, adversarial robustness, privacy

**Why Phase 4**: Complete the professional ML engineer toolkit. Research methods for staying current.

**Major Ordis/Muna integration**: Heavy cross-references for security and documentation.

---

### Phase 5: Advanced & Emerging (70-100 hours)

Focus on cutting-edge and specialized topics.

**Packs**:
- **edge-embedded-ai** (6-8 skills): Model compression for edge, on-device deployment
- **probabilistic-ml** (6-8 skills): Bayesian NNs, uncertainty quantification, Gaussian processes
- **automl-nas** (6-8 skills): Architecture search, hyperparameter optimization at scale
- **active-learning-hitl** (6-8 skills): Query strategies, human-in-the-loop, weak supervision
- **synthetic-data-simulation** (6-8 skills): Domain randomization, sim-to-real transfer
- **scientific-ml** (6-8 skills): Physics-informed NNs, neural ODEs, scientific computing

**Why Phase 5**: Specialized domains, emerging research areas. Completes comprehensive coverage.

---

**Total Vision**: 24 packs, ~200-250 skills across all phases

**Total Effort**: 500-700 hours across all phases (comparable to a large open-source project)

**Timeline**: At 20 hours/week, complete in 6-9 months. At 40 hours/week, complete in 3-4 months.

---

## Success Criteria & Quality Gates

### Phase 1 Complete When:

**Technical Validation**:
- ✅ All 7 meta-skills pass RED-GREEN-REFACTOR testing
- ✅ All ~46-58 specific skills pass RED-GREEN-REFACTOR testing
- ✅ Primary router (`using-ai-engineering`) correctly routes to all 6 packs
- ✅ Pack meta-skills correctly route to specific skills within their domain
- ✅ Cross-pack scenarios work (e.g., "Deploy trained RL model" → deep-rl + ml-production)

**Quality Validation**:
- ✅ Beginners using skills produce expert-level code
- ✅ Common pitfalls systematically avoided (no NaN losses without debugging)
- ✅ Decision frameworks lead to correct algorithm/tool choices
- ✅ Agents apply patterns correctly under time pressure (no rationalization)
- ✅ Skills work standalone (no Ordis/Muna dependencies)

**Documentation**:
- ✅ All skills committed to git with proper YAML frontmatter
- ✅ Testing documentation captured (RED-GREEN-REFACTOR results)
- ✅ FACTIONS.md updated with Yzmir details
- ✅ README.md updated with Yzmir skill catalog

**Real-World Validation**:
- ✅ Test on 3-5 real ML tasks (train RL agent, fine-tune LLM, deploy model)
- ✅ Skills provide value even for intermediate users (not just beginners)
- ✅ Skills channel knowledge effectively (not teaching from scratch)

### Quality Gates (Per Skill)

**RED Phase Gate**:
- [ ] Pressure scenario designed (appropriate for skill type)
- [ ] Baseline run completed WITHOUT skill
- [ ] Failures/rationalizations documented verbatim
- [ ] Patterns identified (what needs to be addressed)

**GREEN Phase Gate**:
- [ ] Skill addresses baseline failures specifically
- [ ] YAML frontmatter complete (name, description with "Use when...")
- [ ] One excellent example included
- [ ] Test WITH skill passes (agent now complies)

**REFACTOR Phase Gate**:
- [ ] New rationalizations tested under pressure
- [ ] Rationalization table built
- [ ] Red flags list created (if discipline skill)
- [ ] Re-tested until bulletproof

**Deployment Gate**:
- [ ] Committed to git
- [ ] README updated
- [ ] Cross-references validated (if applicable)

### Success Metrics

**Quantitative**:
- 53-65 skills in Phase 1
- 100% pass RED-GREEN-REFACTOR
- Average 2-4 hours per skill
- <500 words per skill (token efficiency)

**Qualitative**:
- Code quality elevation (beginner → expert)
- Systematic debugging (vs random changes)
- Correct algorithm selection
- Pitfall avoidance
- Pattern application consistency

---

## Governance & Maintenance

### Yzmir Faction Ownership

- All AI/ML skillpacks live in `yzmir/` directory
- Follow existing Ordis/Muna patterns (meta-skills, core/extension split if needed)
- Integrate with faction system (FACTIONS.md update)

### Skillpack Maintenance (Future)

**Key Insight**: Integration and cross-referencing is an enduring task, like "sweeping the tide into the sea."

**Solution**: Dedicated "skillpack-maintenance" skillpack with agent curator for continuous improvement.

**Phase 1 Integration Scope** (Best Effort):
- Document obvious cross-references during creation
- Note potential Ordis/Muna integration points in comments
- Focus on standalone quality first
- Let dedicated curator handle comprehensive integration later

**Curator Agent Responsibilities** (Future):
- Find cross-reference opportunities
- Update as new skills emerge
- Optimize discovery paths
- Refactor as patterns evolve
- Continuous quality improvement

### Version Control

- Git commits per skill (like Ordis/Muna pattern)
- Proper commit messages with RED-GREEN-REFACTOR status
- Tag Phase 1 completion: `v0.2-phase1-yzmir` (following Ordis/Muna as v0.1)

### README Integration

- Add Yzmir to main README skill catalog
- Update FACTIONS.md with Yzmir completion status
- Document Phase 2+ roadmap

---

## Open Questions

None currently - design validated and approved through Phase 3.

---

## Next Steps

1. ✅ Phase 3: Design Presentation (complete)
2. ✅ Phase 4: Write complete design document (complete)
3. Phase 5: Set up worktree (if implementing immediately)
4. Phase 6: Create detailed implementation plan with bite-sized tasks

---

## Design Decisions Log

**2025-10-29**:
- ✅ Scope: Phased approach - Phase 1 with 6 packs, full planning for 20+ packs
- ✅ Faction: Yzmir (Magicians of the Mind)
- ✅ Structure: Nested routing (3 levels)
- ✅ Testing: Full RED-GREEN-REFACTOR for every skill
- ✅ Integration: Standalone Phase 1, best-effort cross-references, dedicated curator for comprehensive integration
- ✅ Architecture: Modular Domain Packs (vs layered/monolithic/hybrid)
- ✅ Audience: Beginners/intermediates creating expert-level code
- ✅ Principle: Channel knowledge vs teach from scratch
- ✅ Passion project: Deep RL (largest pack, 10-12 skills)
- ✅ Integration philosophy: "Sweeping tide into sea" - enduring task handled by dedicated curator agent
- ✅ Phase 1 effort estimate: 120-150 hours realistic target
- ✅ All sections validated with peer review
