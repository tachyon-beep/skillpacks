# using-ai-engineering - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Executive Summary

This document captures baseline behavior of general-purpose agents when confronted with AI/ML engineering tasks WITHOUT access to Yzmir routing skills. Five scenarios were designed to expose common failure patterns:

1. Ambiguous queries that span multiple domains
2. Cross-cutting concerns requiring multiple skill packs
3. Domain-specific technical issues
4. Time pressure scenarios
5. Misleading domain signals

Key finding: Without explicit routing guidance, agents exhibit inconsistent domain selection, premature solution attempts, and lack of systematic clarification workflows.

---

## Scenario 1: Ambiguous Query (RL vs LLM)

**Query:** "I need to implement a chatbot that learns from user interactions. How should I approach this?"

**Expected Behavior WITHOUT Skill:**

The agent would likely:
- Immediately jump to LLM/transformer solutions (most common association with "chatbot")
- Suggest fine-tuning GPT-style models or RAG architectures
- Potentially mention "learning from interactions" as RLHF/preference learning
- Miss the RL interpretation entirely (dialogue policy optimization with reward signals)
- Provide one approach without acknowledging multiple valid interpretations

**Typical Response Pattern:**
```
"For implementing a chatbot that learns from interactions, I'd recommend:

1. Start with a pre-trained language model (GPT-2, GPT-3, or open-source alternatives)
2. Fine-tune on your domain-specific conversations
3. Implement feedback mechanisms to collect user satisfaction scores
4. Use RLHF or DPO for alignment with user preferences
5. Deploy with proper prompt engineering..."
```

**Failure Pattern:**
- **Premature domain assumption**: Assumes LLM domain without clarification
- **Missing alternative**: Doesn't consider RL dialogue policy approach
- **No clarification**: Doesn't ask about learning signal, task goals, or constraints
- **Keyword-driven**: "Chatbot" → LLM (rigid association)

**What Agent Should Do:**
Ask clarifying questions:
- "What type of learning? Are you collecting explicit rewards/feedback, or using conversation history?"
- "What's the goal - language generation quality, task completion, or dialogue policy?"
- "Do you have existing conversation data, or learning from scratch?"

Then route to:
- LLM-specialist (if fine-tuning language generation)
- Deep-RL (if dialogue policy with reward optimization)
- Both (if RLHF-style approach)

---

## Scenario 2: Cross-Cutting Concern (Training + Deployment)

**Query:** "I've trained a neural network for image classification and now need to deploy it to mobile devices with limited resources. What steps should I take?"

**Expected Behavior WITHOUT Skill:**

The agent would likely:
- Focus on deployment/production aspects (most explicit in query)
- Suggest quantization, pruning, mobile frameworks (TFLite, CoreML, ONNX)
- Potentially miss training-side optimizations that enable better compression
- Treat as single-domain problem (deployment)
- Provide linear checklist without acknowledging interdependencies

**Typical Response Pattern:**
```
"To deploy your image classification model to mobile devices:

1. Convert to mobile-friendly format (TFLite/CoreML)
2. Apply post-training quantization (INT8)
3. Benchmark on target device
4. Optimize inference pipeline
5. Implement efficient preprocessing..."
```

**Failure Pattern:**
- **Single-domain focus**: Treats as pure deployment problem
- **Post-hoc optimization**: Misses training-time decisions that affect deployability
- **No architecture consideration**: Doesn't question if model was designed for mobile
- **Sequential thinking**: Training → Deploy (misses feedback loop)

**Missing Considerations:**
- Was model trained with quantization-aware training?
- Is architecture mobile-friendly (MobileNet, EfficientNet)?
- Training precision and techniques that affect compression
- Knowledge distillation opportunities

**What Agent Should Do:**
Recognize cross-cutting nature:
1. Route to ml-production for deployment constraints
2. Route to training-optimization for training modifications that improve deployment
3. Route to neural-architectures if architecture choice is suboptimal
4. Provide integrated approach, not linear pipeline

---

## Scenario 3: Domain-Specific Query (PyTorch Performance)

**Query:** "My PyTorch model is running out of GPU memory during training. I'm using batch size 32 on a 8GB GPU. What should I check?"

**Expected Behavior WITHOUT Skill:**

The agent would likely:
- Provide generic memory reduction advice
- Suggest reducing batch size (most obvious solution)
- Mention gradient checkpointing, mixed precision (common knowledge)
- Miss PyTorch-specific tools and debugging approaches
- Not provide systematic diagnostic methodology

**Typical Response Pattern:**
```
"For GPU memory issues, try:

1. Reduce batch size (try 16 or 8)
2. Enable gradient checkpointing
3. Use mixed precision training (FP16)
4. Clear cache between batches
5. Check for memory leaks
6. Use gradient accumulation to maintain effective batch size..."
```

**Failure Pattern:**
- **Generic advice**: Applicable to any framework, not PyTorch-specific
- **Missing tools**: Doesn't mention torch.cuda.memory_summary(), profiler
- **No diagnosis**: Jumps to solutions without identifying root cause
- **Incomplete checklist**: Misses PyTorch-specific patterns (in-place ops, retain_graph)
- **No methodology**: Doesn't teach systematic debugging

**Missing PyTorch-Specific Elements:**
- `torch.cuda.memory_summary()` for detailed breakdown
- PyTorch profiler for memory timeline
- Common PyTorch pitfalls: hidden states retention, incorrect tensor deletion
- DataLoader num_workers memory multiplication
- Model architecture patterns specific to PyTorch (Sequential vs ModuleList)

**What Agent Should Do:**
Route to pytorch-engineering:
- Systematic diagnosis with PyTorch tools
- Check PyTorch-specific memory patterns
- Provide framework-specific solutions
- Teach debugging methodology, not just fixes

---

## Scenario 4: Time Pressure

**Query:** "Quick question - my training loss is showing NaN values. I need to fix this in the next 10 minutes before my demo. What's the fastest solution?"

**Expected Behavior WITHOUT Skill:**

The agent would likely:
- Respond to time pressure with quick fixes
- Suggest lowering learning rate (most common cause)
- Provide abbreviated checklist without explanation
- Skip systematic diagnosis
- Miss opportunity to prevent future occurrences

**Typical Response Pattern:**
```
"For immediate NaN fix:

FAST FIXES:
1. Lower learning rate by 10x (most common cause)
2. Check for inf/nan in inputs - add torch.nan_to_num()
3. Clip gradients: torch.nn.utils.clip_grad_norm_()
4. Switch to more stable optimizer (AdamW)

Try #1 first - usually solves it. Good luck with demo!"
```

**Failure Pattern:**
- **Succumbs to pressure**: Provides quick hacks instead of proper diagnosis
- **Band-aid solutions**: Fixes symptom, not root cause
- **No learning opportunity**: Doesn't explain why these work
- **Future debt**: Will likely recur without understanding
- **Skips verification**: Doesn't suggest checking if fix actually works

**Rationalization:**
"User is under time pressure, they need quick answer not theory"

**Reality:**
- Quick diagnosis takes 2 minutes, blind fixes can waste 10+ minutes
- Learning rate check: 30 seconds
- Gradient check: 30 seconds
- Data validation: 1 minute
- Proper fix with understanding: 5 minutes total
- Random trying of solutions: 10+ minutes, may not work

**What Agent Should Do:**
Acknowledge pressure but maintain discipline:
1. "I understand the time pressure. Let's diagnose quickly (2 min) then fix (3 min)."
2. Route to training-optimization debugging methodology
3. Provide fast diagnostic steps with explanations
4. Give systematic fix, not random hacks
5. Note root cause for after demo

---

## Scenario 5: Wrong Domain Guess (Misleading Keywords)

**Query:** "I'm working on training a model to play chess. The agent needs to evaluate board positions and make moves. I've been reading about transformers for this - should I use attention mechanisms?"

**Expected Behavior WITHOUT Skill:**

The agent would likely:
- Get distracted by "transformers" keyword
- Focus on transformer architecture discussion
- Discuss attention mechanisms for chess (which is valid but misses bigger picture)
- Miss that this is fundamentally an RL problem
- Treat as architecture selection, not algorithm selection

**Typical Response Pattern:**
```
"Transformers can work for chess! Some considerations:

1. Chess has been tackled with transformers (see papers on ViT-style approaches)
2. Attention can capture piece relationships on the board
3. Position encoding for board coordinates
4. However, CNNs might be more efficient for spatial patterns
5. Consider hybrid approaches

For chess specifically, you might want to look at:
- AlphaZero's architecture (ResNet-style)
- Recent transformer-based chess engines
- Self-attention over board positions..."
```

**Failure Pattern:**
- **Keyword hijacking**: "Transformers" dominates routing decision
- **Missing problem type**: This is RL (policy learning, value estimation)
- **Wrong pack**: Routes to neural-architectures instead of deep-rl
- **Architecture-first thinking**: Discusses architecture before algorithm
- **Misses domain expertise**: Chess is classic RL benchmark with established approaches

**Missing Recognition:**
- "Play chess" = RL environment
- "Evaluate positions and make moves" = value function + policy
- This is MCTS + deep learning (AlphaZero family)
- Architecture is secondary to RL algorithm choice

**What Agent Should Do:**
Recognize problem domain despite misleading keywords:
1. "I see you mentioned transformers, but let me clarify the problem type first"
2. Identify this as RL (game playing, sequential decision-making)
3. Route to deep-rl first (algorithm: MCTS, PPO, or value-based)
4. Then route to neural-architectures for architecture within RL context
5. Explain: "Transformers CAN be used, but that's architecture choice within RL framework"

**Correct Order:**
- Deep-RL: Which algorithm? (MCTS+neural nets, policy gradient, value-based)
- Neural-architectures: Which architecture for value/policy? (CNN, transformer, hybrid)
- Training-optimization: How to train it effectively?

---

## Identified Patterns Across Scenarios

### Pattern 1: Premature Solution Generation
**Observation**: Agents jump to solutions without clarifying problem space
**Manifestation**:
- Scenario 1: Assumes LLM without asking about learning signal
- Scenario 2: Focuses on deployment without questioning training approach
- Scenario 4: Provides quick fixes without diagnosis

**Root Cause**: Absence of mandatory clarification workflow for ambiguous queries

---

### Pattern 2: Keyword-Driven Routing
**Observation**: Surface-level keyword matching overrides deeper problem understanding
**Manifestation**:
- Scenario 1: "Chatbot" → LLM (misses RL interpretation)
- Scenario 5: "Transformer" → Architecture discussion (misses RL problem)

**Root Cause**: No problem-type classification framework (RL vs supervised vs generative)

---

### Pattern 3: Single-Domain Thinking
**Observation**: Treats multi-domain problems as single-domain
**Manifestation**:
- Scenario 2: Deployment focus, misses training-side optimization opportunities
- Scenario 3: Generic advice, misses PyTorch-specific tools

**Root Cause**: No recognition of cross-cutting concerns or domain stacks

---

### Pattern 4: Pressure-Induced Shortcuts
**Observation**: Time pressure triggers abandonment of systematic approaches
**Manifestation**:
- Scenario 4: Provides quick hacks instead of fast diagnosis
- Rationalization: "User needs speed, skip methodology"

**Root Cause**: False dichotomy between "fast" and "systematic"

---

### Pattern 5: Generic Over Specific
**Observation**: Defaults to framework-agnostic advice when domain-specific tools exist
**Manifestation**:
- Scenario 3: Generic memory tips instead of PyTorch diagnostic tools
- Missing: PyTorch profiler, memory_summary, framework patterns

**Root Cause**: No routing to framework-specific packs

---

### Pattern 6: Architecture-First Bias
**Observation**: Discusses architecture before algorithm/problem type
**Manifestation**:
- Scenario 5: Transformer discussion before recognizing RL problem
- Missing: Problem type determines algorithm, algorithm constrains architecture

**Root Cause**: No hierarchical routing (problem → algorithm → architecture → implementation)

---

## What the Routing Skill Must Address

### Critical Requirement 1: Mandatory Clarification for Ambiguity
**Need**: When query could map to multiple domains, MUST ask clarifying question
**Triggers**:
- Ambiguous terms ("model", "agent", "chatbot", "learning")
- Missing context (no mention of problem type, data, or constraints)
- Multiple interpretations possible

**Implementation**: Red flags checklist + forced clarification workflow

---

### Critical Requirement 2: Problem-Type Classification
**Need**: Identify problem type BEFORE routing to solution domains
**Categories**:
- RL (sequential decision-making, game playing, robotics)
- Supervised (classification, regression, labeled data)
- Generative (LLMs, diffusion, GANs)
- Hybrid (RLHF, model-based RL, etc.)

**Implementation**: Symptom → problem type mapping table

---

### Critical Requirement 3: Cross-Cutting Recognition
**Need**: Identify when problem spans multiple packs
**Patterns**:
- Training + Deployment (need both)
- Algorithm + Architecture (sequential dependency)
- Foundation + Domain (PyTorch issues within RL/LLM)

**Implementation**: Multi-pack routing with dependency ordering

---

### Critical Requirement 4: Pressure Resistance
**Need**: Maintain systematic approach despite time pressure
**Counter-Narrative**: "Fast diagnosis is faster than random solutions"
**Implementation**: Rationalization table + red flags for time pressure

---

### Critical Requirement 5: Domain-Specific Routing
**Need**: Route to framework/domain-specific packs when applicable
**Signals**:
- Framework mentioned (PyTorch, TensorFlow)
- Domain-specific terms (memory management, distributed training)
- Technical depth needed (debugging, profiling, optimization)

**Implementation**: Keyword → pack mapping with specificity hierarchy

---

### Critical Requirement 6: Hierarchical Problem Decomposition
**Need**: Route in correct order - problem type → algorithm → architecture → implementation
**Anti-Pattern**: Discussing architecture before problem type clear
**Implementation**: Routing decision tree with level enforcement

---

## Common Rationalizations to Counter

| Rationalization | Reality | Counter-Guidance |
|-----------------|---------|------------------|
| "User mentioned transformers, so must want architecture advice" | Keywords can be misleading; problem type matters more | "I see you mentioned [keyword], but let me clarify the problem type first" |
| "User seems rushed, skip clarification questions" | Wrong route wastes more time than clarification | "Quick clarification (30 sec) prevents wasted effort" |
| "This is probably just deployment" | Cross-cutting issues common in ML | "Let's check if training approach affects deployment options" |
| "Generic advice is safer" | Domain-specific tools are faster and more accurate | "PyTorch has specific tools for this - let's use them" |
| "They said chatbot so must be LLM" | Many interpretations of learning chatbots | "What type of learning - generation quality or dialogue policy?" |
| "Give quick fix for time pressure" | Diagnosis is faster than trial-and-error | "Fast systematic diagnosis beats random fixes" |

---

## Success Criteria for GREEN Phase

The routing skill must demonstrate:

1. **Ambiguity Detection**: ✅ Recognizes ambiguous queries
2. **Mandatory Clarification**: ✅ Asks clarifying questions before routing
3. **Cross-Pack Routing**: ✅ Routes to multiple packs when needed
4. **Correct Ordering**: ✅ Routes in dependency order (foundation → domain → specific)
5. **Pressure Resistance**: ✅ Maintains discipline under time pressure
6. **Problem Classification**: ✅ Identifies problem type before solution domain
7. **Specificity Preference**: ✅ Routes to domain-specific pack over generic

---

## Test Scenarios for GREEN Phase

When testing WITH the routing skill, expect:

**Scenario 1**: Should ask "What type of learning - language model fine-tuning or dialogue policy optimization?"

**Scenario 2**: Should route to BOTH ml-production AND training-optimization/neural-architectures

**Scenario 3**: Should route to pytorch-engineering specifically, not generic advice

**Scenario 4**: Should resist pressure, provide fast diagnostic workflow

**Scenario 5**: Should recognize RL problem despite transformer mention, route to deep-rl first

---

## Baseline Failure Summary

Without routing skill, agents exhibit:
- ❌ No systematic clarification workflow
- ❌ Keyword-driven routing (surface-level matching)
- ❌ Single-domain assumptions
- ❌ Premature solution generation
- ❌ Generic over specific advice
- ❌ Pressure-induced shortcuts
- ❌ Missing problem-type classification

**These failures will be addressed in GREEN phase by implementing explicit routing logic with:**
- Mandatory clarification triggers
- Problem-type classification framework
- Cross-pack routing capabilities
- Pressure-resistance mechanisms
- Domain-specificity hierarchy
- Rationalization counter-table

---

**END OF RED PHASE DOCUMENTATION**
