---
name: using-ai-engineering
description: Use when starting AI/ML engineering tasks, implementing neural networks, training models, or deploying ML systems - routes to appropriate Yzmir pack (pytorch-engineering, deep-rl, llm-specialist, neural-architectures, training-optimization, ml-production) based on task type and context
---

# Using AI Engineering

## Overview

This meta-skill routes you to the right AI/ML engineering pack based on your task. Load this skill when you need ML/AI expertise but aren't sure which specific pack to use.

**Core Principle**: Different ML tasks require different packs. Match your situation to the appropriate pack, load only what you need. Problem type determines routing - clarify before guessing.

## When to Use

Load this skill when:
- Starting any AI/ML engineering task
- User mentions: "neural network", "train a model", "RL agent", "fine-tune LLM", "deploy model"
- You recognize ML/AI work but unsure which pack applies
- Need to combine multiple domains (e.g., train RL + deploy)

**Don't use for**: Non-ML tasks, simple data processing without ML, basic statistical analysis

---

## STOP - Mandatory Clarification Triggers

Before routing, if query contains ANY of these ambiguous patterns, ASK ONE clarifying question:

| Ambiguous Term | What to Ask | Why |
|----------------|-------------|-----|
| "Model not working" | "What's not working - architecture choice, training process, or deployment?" | Could be 3+ different packs |
| "Improve performance" | "Performance in what sense - training speed, inference speed, or accuracy?" | Different optimization domains |
| "Learning chatbot/agent" | "What type of learning - fine-tuning language generation or optimizing dialogue policy?" | LLM vs RL vs both |
| "Fix my code" | "What domain and what's breaking?" | Too vague to route |
| "Train/deploy model" | "Both training AND deployment, or just one?" | May need multiple packs |
| Framework not mentioned | "What framework are you using?" | PyTorch-specific vs generic |

**If you catch yourself about to guess the domain, STOP and clarify.**

---

## Routing by Problem Type

### Step 1: Identify Problem Type (BEFORE routing)

| Keywords/Signals | Problem Type | Why This Matters |
|------------------|--------------|------------------|
| "Play game", "policy", "reward", "environment", "MDP", "agent actions" | **Reinforcement Learning** | RL is distinct algorithm class |
| "Fine-tune", "LLM", "transformer", "prompt", "RLHF", "GPT", "language model" | **Large Language Model** | Specialized modern techniques |
| "Deploy", "serve", "production", "inference", "quantize", "optimize latency" | **Production/Deployment** | Different constraints than training |
| "NaN loss", "won't converge", "unstable", "hyperparameters", "optimization" | **Training Issues** | Universal training problems |
| "PyTorch error", "CUDA", "distributed training", "memory", "GPU" | **Framework Foundation** | Infrastructure before algorithms |
| "Which architecture", "CNN vs transformer", "model selection" | **Architecture Choice** | Before training decisions |

**Critical**: Architecture keywords ("transformer", "CNN") can be misleading. Problem type determines algorithm, algorithm constrains architecture.

---

## Routing Decision Tree

### Foundation Layer Issues

**Symptoms**: "PyTorch memory error", "distributed training", "GPU utilization", "tensor operations", "CUDA out of memory", "DataLoader", "torch.*"

**Route to**: `yzmir/pytorch-engineering/using-pytorch-engineering`

**Why**: Foundation issues need foundational solutions. Don't jump to algorithms when infrastructure broken.

**Red Flag**: If you're thinking "This is probably LLM/RL issue, skip PyTorch" but query mentions PyTorch errors → Route to PyTorch first.

---

### Training Not Working

**Symptoms**: "NaN losses", "won't converge", "training unstable", "need to tune hyperparameters", "loss not decreasing", "gradients", "learning rate"

**Route to**: `yzmir/training-optimization/using-training-optimization`

**Why**: Training problems are universal across all model types. Debug training before assuming algorithm issue.

**Example**: "RL training unstable" → training-optimization FIRST (could be general training issue), then deep-rl if needed.

---

### Reinforcement Learning

**Symptoms**: "RL agent", "policy", "reward", "environment", "Atari", "robotics control", "MDP", "Q-learning", "play game", "sequential decisions", "exploration"

**Route to**: `yzmir/deep-rl/using-deep-rl`

**Why**: RL is distinct domain with specialized techniques.

**Red Flag**: User mentions "transformer for chess" - Still RL problem! Transformer is architecture choice WITHIN RL framework. Route to deep-rl first, then neural-architectures for architecture discussion.

---

### Large Language Models

**Symptoms**: "LLM", "language model", "transformer for text", "fine-tune", "RLHF", "LoRA", "prompt engineering", "GPT", "BERT", "instruction tuning", "chatbot fine-tuning"

**Route to**: `yzmir/llm-specialist/using-llm-specialist`

**Why**: Modern LLM techniques are specialized (LoRA, RLHF, quantization, etc.).

**Clarify First**: If query says "learning chatbot" without specifying fine-tuning vs policy learning, ASK. Could be LLM fine-tuning OR RL dialogue policy.

---

### Architecture Selection

**Symptoms**: "which architecture", "CNN vs transformer", "what model to use", "architecture for X task", "model selection", "attention vs convolution"

**Route to**: `yzmir/neural-architectures/using-neural-architectures`

**Why**: Architecture decisions come before training decisions.

**Important**: Route here only AFTER problem type is clear. Don't discuss architecture for RL game-playing before routing to deep-rl for algorithm context.

---

### Production Deployment

**Symptoms**: "deploy model", "serving", "quantization", "production", "inference optimization", "MLOps", "latency", "throughput", "edge device", "mobile deployment"

**Route to**: `yzmir/ml-production/using-ml-production`

**Why**: Production has unique constraints (latency, throughput, hardware).

**Common Cross-Cut**: If query mentions both training and deployment, route to BOTH in order (training first, then production).

---

## Cross-Cutting Scenarios

### Multiple Domains - Route to BOTH

When task spans domains, route to ALL relevant packs in execution order:

| Query | Route To | Order |
|-------|----------|-------|
| "Train RL agent and deploy" | deep-rl + ml-production | Train before deploy |
| "Fine-tune LLM with distributed training" | llm-specialist + pytorch-engineering | Domain first, then infrastructure |
| "Optimize transformer training" | training-optimization + neural-architectures | Training issues before architecture tweaks |
| "Deploy model to mobile, training not finished" | training-optimization + ml-production | Fix training first |
| "LLM memory error during fine-tuning" | pytorch-engineering + llm-specialist | Foundation first |
| "RL training unstable" | training-optimization + deep-rl | General training first |

**Principle**: Load in order of dependency. Fix foundation before domain. Complete training before deployment.

---

## Common Routing Mistakes

| Symptom | Wrong Route | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Train agent faster" | deep-rl | training-optimization FIRST | Could be general training issue, not RL-specific |
| "LLM memory error" | llm-specialist | pytorch-engineering FIRST | Foundation issue, not LLM technique issue |
| "Deploy RL model" | deep-rl | ml-production | Deployment problem, not RL training |
| "Which optimizer for transformer" | neural-architectures | training-optimization | Optimization, not architecture |
| "Transformer for chess" | neural-architectures | deep-rl FIRST | RL problem, architecture secondary |
| "Chatbot learning from users" | llm-specialist | ASK FIRST | Could be LLM OR RL OR both |
| "Model performance bad" | (guess) | ASK: "Training accuracy or inference speed?" | Ambiguous |

---

## Time Pressure - Don't Skip Diagnosis

### Rationalization to Resist

| Excuse | Reality |
|--------|---------|
| "User seems rushed, skip clarifying questions" | Clarifying takes 30 seconds, wrong route wastes 5+ minutes |
| "I'll guess and correct later" | Loading wrong skill wastes time and confuses user |
| "Give quick fix without diagnosis" | Fast systematic diagnosis (2 min) beats trial-and-error (10+ min) |
| "They need speed, skip methodology" | Methodology IS the fast path |

### Time Pressure Protocol

When user says "quick", "urgent", "need this now":

1. Acknowledge: "I understand the time pressure"
2. Clarify if ambiguous (30 seconds): "Quick question to route correctly: [question]"
3. Route to appropriate pack
4. Let that skill provide fast systematic approach

**Never skip clarification under pressure. Wrong routing is slower than asking.**

---

## Red Flags Checklist

If you catch yourself thinking ANY of these, STOP and clarify or reconsider:

- ❌ "I'll guess this domain" → ASK clarifying question
- ❌ "They probably mean X" → Verify, don't assume
- ❌ "I'll skip asking to save time" → Clarifying is faster than wrong route
- ❌ "Authority figure suggested X" → Still verify task requirements
- ❌ "They mentioned transformer/CNN so discuss architecture" → Check problem type first
- ❌ "Just give generic advice" → Route to specific pack if applicable
- ❌ "This is too vague to route" → ASK clarifying question
- ❌ "They tried X so must be Y" → Maybe X was wrong, verify problem

**All of these mean: Either ASK ONE clarifying question, or reconsider your routing logic.**

---

## Common Rationalizations - Don't Do These

| Rationalization | Counter-Narrative | Correct Action |
|-----------------|-------------------|----------------|
| "User mentioned transformers, so must want architecture advice" | Keywords misleading; problem type matters more | "I see you mentioned transformers - let me clarify the problem type first" |
| "User seems rushed, skip questions" | Wrong route wastes more time | "Quick clarification (30 sec) prevents wasted effort - [ask question]" |
| "This is probably just deployment" | Cross-cutting issues common in ML | "Let's check if training approach affects deployment options" |
| "Generic advice is safer" | Domain-specific tools are faster/better | "PyTorch has specific tools for this - routing to pytorch-engineering" |
| "They said chatbot so must be LLM" | Many interpretations | "Are you fine-tuning language generation or optimizing dialogue policy?" |
| "Give quick fix for time pressure" | Diagnosis faster than guessing | "Fast systematic approach: [2-minute diagnostic]" |
| "Already tried pytorch-engineering" | Might have used wrong skill in pack | "Which pytorch skill did you try? Might need different one" |

---

## When NOT to Use Yzmir Skills

**Skip AI/ML skills when:**
- Simple data processing (use Python/Pandas directly)
- Statistical analysis without neural networks (use classical stats)
- Building non-ML features (use appropriate language/framework skills)
- Data cleaning/ETL without model training (use data engineering tools)

**Red flag**: If you're not training/deploying a neural network or implementing ML algorithms, probably don't need Yzmir.

---

## Integration Points (Future)

**Cross-references (Phase 2+)**:
- Security/adversarial testing → ordis/security-architect
- Model documentation → muna/technical-writer
- Compliance/governance → ordis/compliance-awareness-and-mapping

**Phase 1**: These integrations not yet implemented. Focus on Yzmir standalone.

---

## Routing Summary Flowchart

```
User Query
    ↓
Is query ambiguous? → YES → ASK clarifying question → Get answer
    ↓ NO
Identify problem type:
    - Framework error? → pytorch-engineering
    - Training not working? → training-optimization
    - RL problem? → deep-rl
    - LLM fine-tuning? → llm-specialist
    - Architecture choice? → neural-architectures
    - Production deployment? → ml-production
    ↓
Cross-cutting? → YES → Route to MULTIPLE packs (order by dependency)
    ↓ NO
Route to single pack
    ↓
Load appropriate using-[pack] meta-skill
```

---

## Quick Reference: Problem Type → Pack

| Problem Type | Pack | Trigger Keywords |
|--------------|------|------------------|
| Framework issues | pytorch-engineering | PyTorch, CUDA, memory, distributed, tensor |
| Training problems | training-optimization | NaN, converge, unstable, loss, gradients, LR |
| Reinforcement learning | deep-rl | Agent, policy, reward, environment, game, MDP |
| Language models | llm-specialist | LLM, fine-tune, RLHF, LoRA, GPT, prompt |
| Architecture selection | neural-architectures | Which architecture, CNN vs transformer, model selection |
| Production/deployment | ml-production | Deploy, serve, production, quantize, inference, latency |

**Remember**: When in doubt, ASK. Clarification takes seconds, wrong routing takes minutes.

---

## Examples

### Example 1: Ambiguous Query

**User**: "I need to improve my model's performance"

**DON'T**: Guess (training? inference? accuracy?)

**DO**:
```
"To route you correctly, I need to clarify: Do you mean:
- Improve training speed?
- Improve inference/deployment speed?
- Improve model accuracy?

Each requires a different pack."
```

Then route based on answer.

---

### Example 2: Cross-Cutting

**User**: "I've trained an image classifier and need to deploy to mobile devices with limited resources"

**Recognize**: Training + Deployment (cross-cutting)

**Route to**:
1. First: `ml-production` (deployment constraints)
2. Then: `training-optimization` (if training modifications can help)
3. Possibly: `neural-architectures` (if architecture unsuitable for mobile)

**Explain**: "This spans training and deployment. Let's look at deployment constraints first, then see if training approach needs adjustment."

---

### Example 3: Misleading Keywords

**User**: "I'm training a model to play chess. Should I use transformers or CNNs?"

**DON'T**: Jump to neural-architectures (keyword: "transformer", "CNN")

**DO**: Recognize problem type first
```
"This is a reinforcement learning problem (game playing = sequential decision-making).

Let me route to deep-rl first to discuss algorithm choice (MCTS, value-based, policy gradient),
then we'll discuss architecture (transformer vs CNN) in that RL context."
```

**Route**: deep-rl → neural-architectures (in that order)

---

### Example 4: Time Pressure

**User**: "Quick! My training is showing NaN losses. Need to fix in 10 minutes!"

**DON'T**: Give quick hacks

**DO**: Fast systematic approach
```
"Understood - time pressure. Quick systematic diagnosis is faster than guessing.

Routing to training-optimization for 2-minute diagnostic workflow, then targeted fix.

This will be faster and more reliable than trial-and-error."
```

**Route**: training-optimization (which will provide fast systematic debugging)

---

### Example 5: Foundation Issue Disguised as Domain Issue

**User**: "My LLM fine-tuning is running out of GPU memory on a 24GB GPU with batch size 4"

**Recognize**: Memory management = foundation issue

**Route to**: pytorch-engineering FIRST (memory debugging), then llm-specialist (if LLM-specific optimization needed)

**Explain**: "Memory issues are foundation-level. Routing to pytorch-engineering for memory diagnostics, then llm-specialist if we need LLM-specific optimizations like gradient checkpointing."

---

## Testing This Skill

When testing, verify:
- ✅ Clarifies ambiguous queries before routing
- ✅ Routes to multiple packs for cross-cutting concerns
- ✅ Identifies problem type before discussing architecture
- ✅ Resists pressure to skip clarification
- ✅ Routes to foundation (PyTorch) before domain when appropriate
- ✅ Doesn't get hijacked by misleading keywords

---

**Remember: This skill's job is routing, not solving. Route correctly, let domain skills do their job.**
