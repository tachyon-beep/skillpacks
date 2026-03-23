# AI Engineering Routing Examples

Detailed examples showing correct routing decisions under various scenarios.

---

## Example 1: Ambiguous Query

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

## Example 2: Cross-Cutting

**User**: "I've trained an image classifier and need to deploy to mobile devices with limited resources"

**Recognize**: Training + Deployment (cross-cutting)

**Route to**:
1. First: `ml-production` (deployment constraints)
2. Then: `training-optimization` (if training modifications can help)
3. Possibly: `neural-architectures` (if architecture unsuitable for mobile)

**Explain**: "This spans training and deployment. Let's look at deployment constraints first, then see if training approach needs adjustment."

---

## Example 3: Misleading Keywords

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

## Example 4: Time Pressure

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

## Example 5: Foundation Issue Disguised as Domain Issue

**User**: "My LLM fine-tuning is running out of GPU memory on a 24GB GPU with batch size 4"

**Recognize**: Memory management = foundation issue

**Route to**: pytorch-engineering FIRST (memory debugging), then llm-specialist (if LLM-specific optimization needed)

**Explain**: "Memory issues are foundation-level. Routing to pytorch-engineering for memory diagnostics, then llm-specialist if we need LLM-specific optimizations like gradient checkpointing."

---

## Example 6: Emergency + Authority Pressure

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

## Example 7: Sunk Cost + Frustration

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

## Example 8: Multiple Pressures Combined

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

## Example 9: New User Wants to Start Training

**User**: "I'm new to ML. I want to train a neural network to classify images. Where do I start?"

**Recognize**: Beginner, classification task, needs architecture guidance first

**Route to**:
1. `neural-architectures` - Select appropriate architecture (CNN for images)
2. `training-optimization` - When training begins
3. `pytorch-engineering` - If framework issues arise

**Explain**: "For image classification, let's start with architecture selection to choose the right model type, then move to training setup."

---

## Example 10: Distributed Training Issues

**User**: "My multi-GPU training keeps hanging after a few epochs"

**Recognize**: Distributed training = framework foundation issue

**Route to**: `pytorch-engineering` (distributed training debugging)

**DON'T**: Route to training-optimization (this is infrastructure, not optimization)

---

## Example 11: Model Accuracy Plateau

**User**: "My model accuracy stuck at 60% and won't improve no matter what I try"

**Clarify first**: "What have you tried so far? Is this during training or after training completed?"

**Possible routes**:
- If training loss still decreasing but validation stuck → `training-optimization` (overfitting)
- If training loss also stuck → `training-optimization` (learning dynamics)
- If model architecture fundamentally limited → `neural-architectures` (capacity)

---

## Example 12: Deploy Trained Model

**User**: "I have a trained PyTorch model and need to deploy it as an API"

**Recognize**: Pure deployment problem (training already complete)

**Route to**: `ml-production` (serving, containerization, API setup)

**DON'T**: Route to pytorch-engineering (this is deployment, not framework issues)
