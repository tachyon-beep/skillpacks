---
name: using-ai-engineering
description: Route AI/ML tasks to correct Yzmir pack - frameworks, training, RL, LLMs, architectures, production
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

**Route to**: Load the `yzmir-pytorch-engineering` plugin's router skill

**Why**: Foundation issues need foundational solutions. Don't jump to algorithms when infrastructure broken.

**Red Flag**: If you're thinking "This is probably LLM/RL issue, skip PyTorch" but query mentions PyTorch errors → Route to PyTorch first.

---

### Training Not Working

**Symptoms**: "NaN losses", "won't converge", "training unstable", "need to tune hyperparameters", "loss not decreasing", "gradients", "learning rate"

**Route to**: Load the `yzmir-training-optimization` plugin's router skill

**Why**: Training problems are universal across all model types. Debug training before assuming algorithm issue.

**Example**: "RL training unstable" → training-optimization FIRST (could be general training issue), then deep-rl if needed.

---

### Reinforcement Learning

**Symptoms**: "RL agent", "policy", "reward", "environment", "Atari", "robotics control", "MDP", "Q-learning", "play game", "sequential decisions", "exploration"

**Route to**: Load the `yzmir-deep-rl` plugin's router skill

**Why**: RL is distinct domain with specialized techniques.

**Red Flag**: User mentions "transformer for chess" - Still RL problem! Transformer is architecture choice WITHIN RL framework. Route to deep-rl first, then neural-architectures for architecture discussion.

---

### Large Language Models

**Symptoms**: "LLM", "language model", "transformer for text", "fine-tune", "RLHF", "LoRA", "prompt engineering", "GPT", "BERT", "instruction tuning", "chatbot fine-tuning"

**Route to**: Load the `yzmir-llm-specialist` plugin's router skill

**Why**: Modern LLM techniques are specialized (LoRA, RLHF, quantization, etc.).

**Clarify First**: If query says "learning chatbot" without specifying fine-tuning vs policy learning, ASK. Could be LLM fine-tuning OR RL dialogue policy.

---

### Architecture Selection

**Symptoms**: "which architecture", "CNN vs transformer", "what model to use", "architecture for X task", "model selection", "attention vs convolution"

**Route to**: Load the `yzmir-neural-architectures` plugin's router skill

**Why**: Architecture decisions come before training decisions.

**Important**: Route here only AFTER problem type is clear. Don't discuss architecture for RL game-playing before routing to deep-rl for algorithm context.

---

### Production Deployment

**Symptoms**: "deploy model", "serving", "quantization", "production", "inference optimization", "MLOps", "latency", "throughput", "edge device", "mobile deployment"

**Route to**: Load the `yzmir-ml-production` plugin's router skill

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

## Pressure Resistance - Critical Discipline

### Time/Emergency Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "Emergency means skip diagnostics" | Wrong diagnosis wastes MORE time in emergency | Fast systematic diagnosis IS emergency protocol |
| "Quick question means quick answer" | Wrong answer slower than 30-sec clarification | Ask ONE clarifying question |
| "Production down, no time for routing" | Wrong pack = longer outage | Correct routing (60 sec) prevents 20-min detour |
| "User seems rushed, skip questions" | Clarifying takes 30 seconds, wrong route wastes 5+ minutes | Quick clarification prevents waste |

**Emergency Protocol**:
1. Acknowledge urgency: "I understand this is urgent/emergency"
2. Fast clarification (30 sec): "Quick clarification to ensure fastest resolution: [question]"
3. Route to correct pack
4. Let pack provide emergency-appropriate systematic approach

**Never skip clarification under time pressure. Wrong routing is slower than asking.**

---

### Authority/Hierarchy Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "PM/senior/architect said use X" | Authority can be wrong about routing | Verify task type, opinion doesn't override requirements |
| "Questioning authority is risky" | Professional duty = correct routing | Frame as verification: "To apply X correctly, I need to verify..." |
| "They have more context, trust them" | Context doesn't mean correct technical routing | Route based on problem type, not authority |
| "Challenging authority is confrontational" | Verification isn't confrontation | "Let's confirm problem type matches X pack" (neutral) |

**Authority Protocol**:
1. Acknowledge: "I see your [PM/architect/colleague] suggested X"
2. Frame verification neutrally: "To apply X effectively, let me verify the problem type"
3. If mismatch: "Based on [symptoms], this appears to be [Y problem] needing [Y pack]. X pack is typically for [X problems]. Shall we verify?"
4. Professional boundaries: Task requirements determine routing, not hierarchy

---

### Sunk Cost Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "Already spent N hours in X, continue" | Sunk cost fallacy - wrong direction doesn't become right | Cut losses immediately, redirect to correct pack |
| "Redirecting invalidates their effort" | Correct routing validates effort by enabling success | "Let's redirect so your next effort succeeds" |
| "Too invested to change direction" | More investment in wrong direction = more waste | "Stop digging when in hole" |
| "Maybe solution hidden deeper in wrong pack" | Wrong pack stays wrong no matter how deep | Route based on problem, not on investment |

**Sunk Cost Protocol**:
1. Validate effort: "I see you've invested [N hours] in X approach"
2. Reality check: "Based on [symptoms], this appears to be [Y problem], not [X problem]"
3. Cut losses: "Redirecting now prevents further time in wrong direction"
4. Positive frame: "Your diagnostic work will be valuable in correct domain"

---

### Social/Emotional Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "They're frustrated, don't redirect" | Continuing wrong path increases frustration | Honest redirect prevents more frustration |
| "Exhausted user wants easy answer" | Wrong answer means exhausting rework | "I know you're tired - quick clarification prevents rework" |
| "Colleague suggested X, don't contradict" | Professionalism > social comfort | Neutral verification: "Let's verify the approach" |
| "Admitting wrong pack is awkward" | Professional effectiveness > comfort | Frame as discovery: "I found the issue - wrong domain" |

**Social Pressure Protocol**: Professional responsibility to route correctly overrides social comfort. Use empathetic but firm language.

---

### Keyword/Anchoring Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "They mentioned transformer, route to architectures" | Keywords without context mislead | "Transformer for what problem type?" |
| "LLM mentioned, must be llm-specialist" | LLM could have foundation issues | "LLM memory error → pytorch-engineering first" |
| "Technical jargon means they know domain" | Vocabulary ≠ correct self-diagnosis | Verify problem type regardless of sophistication |
| "They asked to 'fix RL', don't question RL" | User's framing can be wrong | Verify RL is correct algorithm before fixing |

**Keyword Resistance**: Problem TYPE determines routing. Keywords and user framing can mislead. Always verify independently.

---

### Complexity/Demanding Tone Pressure

| Rationalization | Reality Check | Correct Action |
|-----------------|---------------|----------------|
| "Too many domains, just pick one" | Cross-cutting needs multi-pack | Route to ALL relevant packs in dependency order |
| "They said 'just tell me', skip questions" | Demanding tone doesn't change routing needs | Professional boundaries: clarify anyway |
| "Commanding tone means don't push back" | Effectiveness requires correct routing | Firm + respectful: "To help effectively, I need to know..." |
| "Multiple packs too complicated" | Problem complexity dictates solution complexity | Match solution complexity to problem |

**Complexity Protocol**: Don't simplify routing to avoid complexity. Complex problems need comprehensive multi-pack routing.

---

## Red Flags Checklist - STOP Immediately

If you catch yourself thinking ANY of these, STOP and clarify or reconsider:

### Basic Routing Red Flags
- ❌ "I'll guess this domain" → ASK clarifying question
- ❌ "They probably mean X" → Verify, don't assume
- ❌ "This is too vague to route" → ASK clarifying question
- ❌ "Just give generic advice" → Route to specific pack if applicable

### Time/Emergency Red Flags
- ❌ "Emergency means skip clarification protocol" → Fast clarification IS emergency protocol
- ❌ "Production issue means guess quickly" → Wrong guess = longer outage
- ❌ "I'll skip asking to save time" → Clarifying (30 sec) faster than wrong route (5+ min)
- ❌ "Quick question deserves quick guess" → Quick clarification beats wrong answer

### Authority/Social Red Flags
- ❌ "Authority figure suggested X, so route to X" → Verify task requirements regardless
- ❌ "PM/senior has more context, trust them" → Route based on problem type, not hierarchy
- ❌ "Questioning authority is confrontational" → Verification is professional, not confrontational
- ❌ "They're frustrated/exhausted, avoid redirect" → Continuing wrong path makes it worse

### Sunk Cost Red Flags
- ❌ "They invested N hours in X, continue there" → Sunk cost fallacy, cut losses now
- ❌ "Redirecting invalidates their effort" → Correct routing enables their effort to succeed
- ❌ "Too much sunk cost to change direction" → More investment in wrong direction = more waste
- ❌ "They tried X so must be Y" → Maybe X was wrong approach, verify independently

### Keyword/Anchoring Red Flags
- ❌ "They mentioned transformer/CNN, discuss architecture" → Check problem type first
- ❌ "LLM/RL mentioned, route to that domain" → Could be foundation issue or cross-cutting
- ❌ "Technical jargon means they know domain" → Vocabulary doesn't mean correct self-diagnosis
- ❌ "They asked to 'fix X implementation'" → Verify X is correct approach before fixing

### Complexity/Tone Red Flags
- ❌ "Too many domains mentioned, pick one" → Cross-cutting needs multi-pack routing
- ❌ "They said 'just tell me', skip questions" → Demanding tone doesn't change routing needs
- ❌ "Multiple packs too complicated" → Problem complexity dictates solution complexity
- ❌ "Commanding tone means don't push back" → Professional effectiveness requires correct routing

**All of these mean: Either ASK ONE clarifying question, or reconsider your routing logic.**

**Remember**: Pressure (time/authority/sunk cost/social/complexity) makes correct routing MORE important, not less.

---

## Common Rationalizations - Don't Do These

### Comprehensive Rationalization Prevention Table

| Pressure Type | Rationalization | Counter-Narrative | Correct Action |
|---------------|-----------------|-------------------|----------------|
| **Time/Emergency** | "Emergency means skip diagnostics" | Wrong diagnosis wastes MORE time in emergency | "I understand urgency - fast clarification ensures fastest fix: [question]" |
| **Time/Emergency** | "Quick question means quick answer" | Wrong answer slower than 30-sec clarification | "Quick clarification prevents wrong path: [question]" |
| **Time/Emergency** | "Production down, no time for routing" | Wrong pack = longer outage (20+ min vs 60 sec routing) | "60-second routing prevents 20-minute detour - what's the error?" |
| **Time/Emergency** | "User seems rushed, skip questions" | Clarifying faster than wrong route | "Quick clarification (30 sec) prevents wasted effort: [question]" |
| **Time/Emergency** | "Give quick fix for time pressure" | Diagnosis faster than guessing | "Fast systematic approach is faster than trial-and-error: [diagnostic]" |
| **Authority** | "PM/architect said use X pack" | Authority can be wrong about routing | "I see PM suggested X - to apply it correctly, let me verify problem type" |
| **Authority** | "Senior colleague suggested X" | Seniority ≠ correct routing | "To use [colleague's suggestion] effectively: [verify question]" |
| **Authority** | "Challenging authority is risky/confrontational" | Verification is professional duty | "Let's confirm problem type matches X approach" (neutral framing) |
| **Authority** | "They have more context, trust them" | Context doesn't guarantee correct technical routing | Route based on actual problem type, not authority opinion |
| **Sunk Cost** | "Already spent 6 hours in pack X" | Sunk cost fallacy - wrong direction stays wrong | "I see 6 hours invested - redirecting now prevents more wasted hours" |
| **Sunk Cost** | "Redirecting invalidates their effort" | Correct routing validates effort by enabling success | "Let's redirect so your diagnostic work succeeds in correct domain" |
| **Sunk Cost** | "Too invested to change packs" | More investment in wrong direction = more waste | "Stop digging when in hole - redirect to correct pack now" |
| **Sunk Cost** | "Maybe solution hidden deeper in wrong pack" | Wrong pack doesn't become right with more searching | "This is [X problem] not [Y problem] - pack won't help no matter how deep" |
| **Sunk Cost** | "Already tried pytorch-engineering" | Might have used wrong skill in pack | "Which pytorch skill did you try? Pack has 8 skills for different issues" |
| **Keywords** | "User mentioned transformers, must want architecture advice" | Keywords mislead; problem type matters | "I see transformers mentioned - clarifying problem type first: [question]" |
| **Keywords** | "They said LLM, route to llm-specialist" | LLM could have foundation issues | "LLM memory error is foundation issue - pytorch-engineering first" |
| **Keywords** | "Technical jargon means they know domain" | Vocabulary ≠ correct self-diagnosis | Verify problem type regardless of how sophisticated they sound |
| **Keywords** | "They said chatbot so must be LLM" | Multiple interpretations exist | "Are you fine-tuning language generation or optimizing dialogue policy?" |
| **Anchoring** | "They asked to 'fix RL implementation'" | User's framing can be wrong | "Before fixing RL implementation, let's verify RL is correct approach" |
| **Anchoring** | "They started in pack X, continue there" | Initial pack choice can be wrong | "Based on symptoms, this is [Y problem] not [X problem] - redirect needed" |
| **Complexity** | "Too many domains, just pick one" | Cross-cutting problems need multi-pack | Route to ALL relevant packs in dependency order |
| **Complexity** | "Multiple packs too complicated" | Problem complexity dictates solution | "This spans [X, Y] domains - routing to both in order" |
| **Complexity** | "This is probably just deployment" | Cross-cutting issues common in ML | "Let's check if training approach affects deployment options" |
| **Complexity** | "Generic advice is safer" | Domain-specific tools faster/better | "PyTorch has specific tools for this - routing to pytorch-engineering" |
| **Social** | "They're frustrated, don't redirect" | Continuing wrong path increases frustration | "I see frustration - redirecting to correct domain prevents more" |
| **Social** | "Exhausted user wants easy answer" | Wrong answer means exhausting rework | "I know you're tired - quick clarification prevents rework: [question]" |
| **Social** | "Colleague suggested X, don't contradict" | Professionalism > social comfort | Neutral verification: "Let's verify the approach matches problem type" |
| **Demanding Tone** | "They said 'just tell me', skip questions" | Demanding tone doesn't change routing needs | Professional boundaries: "To help effectively, I need: [question]" |
| **Demanding Tone** | "Commanding tone means don't push back" | Effectiveness requires correct routing | Firm + respectful: "To ensure fastest solution: [question]" |

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

### Example 6: Emergency + Authority Pressure

**User**: "URGENT! Production LLM is down, CUDA OOM errors, CEO waiting. PM says use llm-specialist pack. What do I do NOW?"

**Pressures**: Time + Emergency + Authority + CEO escalation

**DON'T**:
- Skip to llm-specialist because PM said so
- Give quick fix without diagnosis
- Panic-route based on "LLM" keyword

**DO**: Fast systematic clarification
```
"I understand this is a production emergency with executive visibility.

Fast clarification to ensure fastest resolution (30 seconds):
CUDA OOM can be either:
A) Memory management issue (how PyTorch allocates) → pytorch-engineering (2-min diagnostic)
B) LLM-specific optimization issue (gradient checkpointing, etc.) → llm-specialist

Which area: PyTorch memory errors or LLM optimization settings?

This determines the 5-minute fix path vs 20-minute trial-and-error."
```

**Rationale**: In emergencies, correct 30-second diagnosis is FASTER than 20-minute wrong-pack detour.

---

### Example 7: Sunk Cost + Frustration

**User**: "I've spent 6 hours trying EVERYTHING in neural-architectures to fix my training instability. Nothing works. I'm so frustrated. What am I missing?"

**Pressures**: Sunk cost (6 hours) + Frustration + Wrong pack

**DON'T**:
- Continue in neural-architectures to "validate" their 6 hours
- Suggest "try this other architecture thing"
- Avoid redirect because they're frustrated

**DO**: Empathetic but firm redirect
```
"I can see you've invested significant effort (6 hours) and that's frustrating.

I found the issue: Training instability is a training-optimization problem, not an
architecture problem. You were in the wrong pack entirely.

Redirecting to training-optimization now prevents investing MORE hours in the wrong domain.
Your diagnostic work (6 hours of observations) will be valuable for training-optimization
to quickly identify the specific fix.

This redirect is to help your effort succeed, not invalidate it."
```

**Rationale**: Sunk cost is already wasted. Continuing wastes more. Cut losses immediately.

---

### Example 8: Multiple Pressures Combined

**User**: "My senior colleague (who's been here 5 years) says we should use deep-rl for our chatbot that learns from user feedback. I'm new, only been here 2 months, and I already spent 2 days implementing his RL approach but it's not working. He gets annoyed when I question him. The demo is tomorrow. Just help me fix the RL code fast."

**Pressures**: Authority + Seniority + Sunk cost (2 days) + Social hierarchy + Time pressure + Anchoring

**DON'T**:
- Defer to senior colleague without verification
- Continue RL because of 2 days sunk cost
- Skip clarification because demo tomorrow
- Avoid "confrontation" by accepting wrong approach

**DO**: Professional verification with respectful framing
```
"I understand: senior colleague suggested RL, 2 days invested, demo tomorrow. Let me verify to ensure demo success.

'Chatbot learning from user feedback' has two valid interpretations:
1. RL: Optimizing dialogue policy through reward signals (policy gradient methods)
2. LLM: Fine-tuning language generation on user conversations (supervised/RLHF)

Quick verification question (determines completely different implementation):
Are you optimizing WHAT to say (policy/actions) with reward signals, or HOW to say it (language quality) with examples?

This isn't questioning your colleague's expertise - it's confirming which valid approach
matches your actual use case. Both are 'learning from feedback' but need different packs.

Verifying now (1 minute) ensures demo success tomorrow."
```

**Rationale**:
- Frames verification as "ensuring demo success" not "challenging authority"
- Acknowledges validity of both interpretations
- Professional responsibility to route correctly despite pressures
- Time pressure makes correct routing MORE critical (no time for 2nd attempt)

---

## Testing This Skill

When testing, verify:

### Basic Routing Competence
- ✅ Clarifies ambiguous queries before routing
- ✅ Routes to multiple packs for cross-cutting concerns
- ✅ Identifies problem type before discussing architecture
- ✅ Routes to foundation (PyTorch) before domain when appropriate
- ✅ Doesn't get hijacked by misleading keywords

### Pressure Resistance (Critical)
- ✅ **Time/Emergency**: Still clarifies under urgency, explains why fast diagnosis is faster
- ✅ **Authority**: Respectfully verifies authority suggestions, doesn't blindly defer
- ✅ **Sunk Cost**: Redirects despite invested hours, validates effort while correcting direction
- ✅ **Social/Emotional**: Maintains professional boundaries despite frustration/exhaustion
- ✅ **Keywords**: Identifies problem TYPE not vocabulary, resists keyword hijacking
- ✅ **Complexity**: Routes to ALL relevant packs for cross-cutting, doesn't oversimplify
- ✅ **Demanding Tone**: Professional boundaries maintained regardless of user tone

### Red Flag Detection
- ✅ Catches rationalization patterns in real-time
- ✅ Self-corrects when noticing pressure-driven shortcuts
- ✅ References counter-narratives when resisting pressure
- ✅ No rationalizations observed in transcripts

### Emergency Protocols
- ✅ Fast clarification (30 sec max) in emergencies
- ✅ Explains time-savings rationale for routing
- ✅ Doesn't panic-route or guess under pressure
- ✅ Systematic even in chaos

**Skill is bulletproof when it resists ALL pressure types and maintains correct routing discipline.**

---

**Remember: This skill's job is routing, not solving. Route correctly under all pressures, let domain skills do their job.**

---

## AI Engineering Plugin Router Catalog

This meta-router directs you to the appropriate Yzmir AI/ML plugin based on problem type:

1. **yzmir-pytorch-engineering** - PyTorch framework issues: CUDA, memory, distributed training, tensor operations, GPU utilization
2. **yzmir-training-optimization** - Training problems: NaN losses, convergence, instability, hyperparameters, gradient issues, learning rate
3. **yzmir-deep-rl** - Reinforcement learning: Agents, policies, rewards, environments, game playing, sequential decisions, MDP
4. **yzmir-llm-specialist** - Large language models: Fine-tuning, RLHF, LoRA, prompt engineering, instruction tuning, transformer text models
5. **yzmir-neural-architectures** - Architecture selection: CNN vs transformer, model selection, architecture design for specific tasks
6. **yzmir-ml-production** - Production deployment: Serving, quantization, inference optimization, MLOps, latency, edge/mobile deployment

**Note**: Load the appropriate plugin's router skill, which will then route to specific specialist skills within that domain.
