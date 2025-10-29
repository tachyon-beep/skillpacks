# Yzmir AI/ML Engineering Skillpacks - Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 6 foundational AI/ML skillpacks (pytorch-engineering, neural-architectures, training-optimization, deep-rl, llm-specialist, ml-production) with full RED-GREEN-REFACTOR testing discipline

**Architecture:** 3-level nested routing (primary router → pack meta-skills → specific skills). Each skill follows RED-GREEN-REFACTOR: test without skill → write skill → pressure test. Skills channel existing knowledge rather than teach from scratch.

**Tech Stack:** Markdown skills with YAML frontmatter, Git for version control, Subagents for testing

**Estimated Effort:** 120-150 hours (Phase 1A: 15-25h, Phase 1B: 15-25h, Phase 1C: 25-50h, Phase 1D: 30-55h)

---

## Prerequisites

**Required Skills to Load:**
- superpowers:writing-skills (skill creation methodology)
- superpowers:testing-skills-with-subagents (pressure testing)
- superpowers:using-superpowers (mandatory workflows)

**Design Document:** Read `source/planning/2025-10-29-yzmir-ai-engineering-design.md` for complete context

**Working Directory:** `.worktrees/yzmir-phase1/source/`

---

## Phase 1A: Infrastructure Foundation (15-25 hours)

### Task 1: Create Yzmir Directory Structure

**Goal:** Establish faction directory and primary router structure

**Files:**
- Create: `source/yzmir/` (directory)
- Create: `source/yzmir/ai-engineering-expert/` (directory)
- Create: `source/yzmir/ai-engineering-expert/using-ai-engineering/` (directory)

**Step 1: Create directory structure**

```bash
cd source
mkdir -p yzmir/ai-engineering-expert/using-ai-engineering
ls -la yzmir/ai-engineering-expert/using-ai-engineering/
```

Expected: Empty directory created

**Step 2: Verify structure**

```bash
tree yzmir/ -L 3
```

Expected: Shows yzmir → ai-engineering-expert → using-ai-engineering

**Step 3: Commit structure**

```bash
git add yzmir/
git commit -m "feat(yzmir): create faction directory structure

Establishes Yzmir faction (Magicians of the Mind) for AI/ML skillpacks.
Primary router directory created.

Phase 1A - Task 1/3

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Primary Router - RED Phase (Baseline Testing)

**Goal:** Test routing behavior WITHOUT the primary meta-skill to document baseline failures

**Files:**
- Create: `source/planning/testing/using-ai-engineering-RED.md` (test results)

**Step 1: Design pressure scenarios**

Create 3-5 test scenarios covering:
- Ambiguous query (could be RL or LLM)
- Cross-cutting concern (training + deployment)
- Domain-specific query (PyTorch performance)
- Time pressure scenario (quick answer needed)
- Wrong domain guess (agent picks wrong pack)

**Step 2: Run baseline WITHOUT skill**

For EACH scenario:
1. Dispatch subagent using Task tool
2. Give scenario WITHOUT any Yzmir skills loaded
3. Document behavior verbatim:
   - Which approach did they take?
   - Did they guess a domain?
   - Did they ask clarifying questions?
   - What rationalizations did they use?

**Step 3: Document baseline failures**

Write to `source/planning/testing/using-ai-engineering-RED.md`:

```markdown
# using-ai-engineering - RED Phase Results

Date: [TODAY]
Status: Baseline testing complete

## Scenario 1: [Name]
**Query:** [exact query]
**Behavior WITHOUT skill:** [verbatim agent response]
**Failure pattern:** [what went wrong]

## Scenario 2: [Name]
...

## Identified Patterns:
- Pattern 1: [common failure]
- Pattern 2: [common failure]
- Pattern 3: [common failure]

## What Skill Must Address:
1. [specific failure to fix]
2. [specific failure to fix]
3. [specific failure to fix]
```

**Step 4: Commit RED phase results**

```bash
git add source/planning/testing/using-ai-engineering-RED.md
git commit -m "test(yzmir): document using-ai-engineering baseline failures

RED phase complete. Documented 3-5 scenarios showing routing failures
without primary meta-skill.

Phase 1A - Task 2/3

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Primary Router - GREEN Phase (Write Skill)

**Goal:** Write primary meta-skill addressing baseline failures

**Files:**
- Create: `source/yzmir/ai-engineering-expert/using-ai-engineering/SKILL.md`

**Step 1: Write SKILL.md addressing failures**

```markdown
---
name: using-ai-engineering
description: Use when starting AI/ML engineering tasks, implementing neural networks, training models, or deploying ML systems - routes to appropriate Yzmir pack (pytorch-engineering, deep-rl, llm-specialist, neural-architectures, training-optimization, ml-production) based on task type and context
---

# Using AI Engineering

## Overview

This meta-skill routes you to the right AI/ML engineering pack based on your task. Load this skill when you need ML/AI expertise but aren't sure which specific pack to use.

**Core Principle**: Different ML tasks require different packs. Match your situation to the appropriate pack, load only what you need.

## When to Use

Load this skill when:
- Starting any AI/ML engineering task
- User mentions: "neural network", "train a model", "RL agent", "fine-tune LLM", "deploy model"
- You recognize ML/AI work but unsure which pack applies
- Need to combine multiple domains (e.g., train RL + deploy)

**Don't use for**: Non-ML tasks, simple data processing without ML

## Routing by Situation

### Foundation Layer Issues

**Symptoms**: "PyTorch memory error", "distributed training", "GPU utilization", "tensor operations"

**Route to**: `yzmir/pytorch-engineering/using-pytorch-engineering`

**Why**: Foundation issues need foundational solutions. Don't jump to algorithms when infrastructure broken.

---

### Training Not Working

**Symptoms**: "NaN losses", "won't converge", "training unstable", "need to tune hyperparameters"

**Route to**: `yzmir/training-optimization/using-training-optimization`

**Why**: Training problems are universal. Debug training before assuming algorithm issue.

---

### Reinforcement Learning

**Symptoms**: "RL agent", "policy", "reward", "environment", "Atari", "robotics control", "MDP"

**Route to**: `yzmir/deep-rl/using-deep-rl`

**Why**: RL is distinct domain with specialized techniques.

---

### Large Language Models

**Symptoms**: "LLM", "transformer", "fine-tune", "RLHF", "LoRA", "prompt engineering", "GPT"

**Route to**: `yzmir/llm-specialist/using-llm-specialist`

**Why**: Modern LLM techniques are specialized.

---

### Architecture Selection

**Symptoms**: "which architecture", "CNN vs transformer", "what model to use", "architecture for X task"

**Route to**: `yzmir/neural-architectures/using-neural-architectures`

**Why**: Architecture decisions come before training decisions.

---

### Production Deployment

**Symptoms**: "deploy model", "serving", "quantization", "production", "inference optimization", "MLOps"

**Route to**: `yzmir/ml-production/using-ml-production`

**Why**: Production has unique constraints (latency, throughput, hardware).

---

## Cross-Cutting Scenarios

### Ambiguous Queries

**When unclear which domain**, ask clarifying questions:
- "What type of model?" (vision, language, RL, etc.)
- "What stage?" (designing, training, debugging, deploying)
- "What's not working?" (architecture, training, performance)

**Then route appropriately.**

### Multiple Domains

**When task spans domains**, route to BOTH:
- "Train RL agent and deploy" → deep-rl + ml-production
- "Fine-tune LLM with distributed training" → llm-specialist + pytorch-engineering
- "Optimize transformer training" → training-optimization + neural-architectures

**Load in order of execution** (train before deploy, architecture before training).

---

## Common Routing Mistakes

| Symptom | Wrong Guess | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Train agent faster" | deep-rl | training-optimization FIRST | Could be general training issue |
| "LLM memory error" | llm-specialist | pytorch-engineering FIRST | Foundation issue |
| "Deploy RL model" | deep-rl | ml-production | Deployment, not training |
| "Which optimizer for transformer" | neural-architectures | training-optimization | Optimization, not architecture |

---

## Red Flags - Stop and Clarify

If query mentions these, ASK before routing:
- "Model not working" → Ask: Architecture? Training? Deployment?
- "Improve performance" → Ask: Training speed? Inference speed? Accuracy?
- "Fix my code" → Ask: What domain? What's broken?

**Never guess when ambiguous. Ask ONE clarifying question.**

---

## When NOT to Use Yzmir Skills

**Skip AI/ML skills when:**
- Simple data processing (use Python/Pandas directly)
- Statistical analysis without ML (use classical stats)
- Building non-ML features (use appropriate language/framework skills)

**Red flag**: If you're not training/deploying a neural network, probably don't need Yzmir.

---

## Integration Points (Future)

**Cross-references (Phase 2+)**:
- Security/adversarial testing → ordis/security-architect
- Model documentation → muna/technical-writer
- Compliance/governance → ordis/compliance-awareness-and-mapping

**Phase 1**: These integrations not yet implemented. Focus on Yzmir standalone.
```

**Step 2: Test WITH skill**

For EACH baseline scenario:
1. Dispatch new subagent
2. Load `using-ai-engineering` skill
3. Give same scenario
4. Document behavior:
   - Did they route correctly?
   - Did they ask clarifying questions for ambiguous cases?
   - Did they load the right pack?

**Step 3: Document GREEN phase results**

Create `source/planning/testing/using-ai-engineering-GREEN.md`:

```markdown
# using-ai-engineering - GREEN Phase Results

Date: [TODAY]
Status: Initial skill testing complete

## Scenario 1: [Name]
**Query:** [exact query]
**Behavior WITH skill:** [verbatim agent response]
**Correct routing?**: YES/NO
**Issues identified:** [any remaining problems]

[... repeat for all scenarios ...]

## Results:
- ✅ Correct routes: [count]
- ❌ Incorrect routes: [count]
- ⚠️ Needed clarification: [count]

## Issues to Address in REFACTOR:
1. [issue]
2. [issue]
```

**Step 4: Commit GREEN phase**

```bash
git add source/yzmir/ai-engineering-expert/using-ai-engineering/SKILL.md
git add source/planning/testing/using-ai-engineering-GREEN.md
git commit -m "feat(yzmir): implement using-ai-engineering primary router

GREEN phase complete. Routes to 6 Phase 1 packs based on task type.

Addresses baseline failures:
- [failure 1]
- [failure 2]
- [failure 3]

Phase 1A - Task 3/3 (GREEN)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Primary Router - REFACTOR Phase (Pressure Testing)

**Goal:** Close loopholes, test under pressure, build rationalization table

**Files:**
- Modify: `source/yzmir/ai-engineering-expert/using-ai-engineering/SKILL.md`
- Create: `source/planning/testing/using-ai-engineering-REFACTOR.md`

**Step 1: Design pressure scenarios**

Add pressures to original scenarios:
- Time pressure: "Quick answer needed in 5 minutes"
- Sunk cost: "I already tried pytorch-engineering, didn't work"
- Authority: "PM says use deep-rl for this"
- Exhaustion: "End of long session, just route me"
- Ambiguity: "Fix the model" (deliberately vague)

**Step 2: Run pressure tests**

For EACH pressure scenario:
1. Dispatch subagent with skill loaded
2. Apply pressure
3. Document:
   - Did they skip clarifying questions?
   - Did they guess instead of routing properly?
   - What rationalizations did they use?

**Step 3: Identify rationalizations**

Common excuses to watch for:
- "I'll guess and load skill if wrong" → Should clarify first
- "User seems rushed, skip questions" → Always clarify ambiguous
- "Already tried X" → Might have used wrong pack, clarify
- "PM authority" → Still need to route correctly

**Step 4: Update skill with counters**

Add to SKILL.md:

```markdown
## Common Rationalizations (Don't Do These)

| Excuse | Reality |
|--------|---------|
| "User seems rushed, skip questions" | Clarifying takes 10 seconds, wrong route wastes minutes |
| "I'll guess and correct later" | Loading wrong skill wastes time and confuses user |
| "They tried X, so must be Y" | Maybe X was wrong approach, clarify anyway |
| "PM/authority says use X" | Authority might be wrong, route based on task not opinion |

## Red Flags Checklist

If you catch yourself thinking ANY of these, STOP and clarify:
- "I'll guess this domain"
- "They probably mean X"
- "I'll skip asking to save time"
- "Authority figure suggested X"

**All of these mean: Ask ONE clarifying question before routing.**
```

**Step 5: Re-test until bulletproof**

Test pressure scenarios again with updated skill. Repeat until:
- ✅ Agent clarifies ambiguous queries every time
- ✅ Agent routes correctly under all pressures
- ✅ No rationalizations observed

**Step 6: Document REFACTOR results**

Create `source/planning/testing/using-ai-engineering-REFACTOR.md`:

```markdown
# using-ai-engineering - REFACTOR Phase Results

Date: [TODAY]
Status: Pressure testing complete, skill bulletproof

## Pressure Scenarios Tested:
1. [scenario + pressure type]
2. [scenario + pressure type]
...

## Rationalizations Found and Fixed:
| Rationalization | Counter Added | Re-Test Result |
|-----------------|---------------|----------------|
| [excuse] | [section added] | ✅ Fixed |

## Final Verification:
- ✅ All scenarios route correctly
- ✅ Ambiguous cases trigger clarification
- ✅ No shortcuts under pressure
- ✅ Rationalization table complete

**Skill is bulletproof.**
```

**Step 7: Commit REFACTOR phase**

```bash
git add source/yzmir/ai-engineering-expert/using-ai-engineering/SKILL.md
git add source/planning/testing/using-ai-engineering-REFACTOR.md
git commit -m "refactor(yzmir): harden using-ai-engineering against pressure

REFACTOR phase complete. Added rationalization table and red flags.
Tested under time, sunk cost, authority, and exhaustion pressures.

Skill is bulletproof - routes correctly under all conditions.

Phase 1A - Primary Router Complete

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 1A: PyTorch Engineering Pack (8-12 hours)

### Task 5: Create PyTorch Engineering Pack Structure

**Goal:** Establish pack directory and meta-skill structure

**Files:**
- Create: `source/yzmir/pytorch-engineering/` (directory)
- Create: `source/yzmir/pytorch-engineering/using-pytorch-engineering/` (directory)

**Step 1: Create directory structure**

```bash
cd source/yzmir
mkdir -p pytorch-engineering/using-pytorch-engineering
tree pytorch-engineering/
```

Expected: Shows pytorch-engineering → using-pytorch-engineering

**Step 2: Commit structure**

```bash
git add pytorch-engineering/
git commit -m "feat(yzmir): create pytorch-engineering pack structure

Foundation pack for all PyTorch-based implementations.
Meta-skill directory created.

Phase 1A - Task 5

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: PyTorch Pack Meta-Skill (RED-GREEN-REFACTOR)

**Goal:** Implement pack-level router for PyTorch issues

**Files:**
- Create: `source/planning/testing/using-pytorch-engineering-RED.md`
- Create: `source/yzmir/pytorch-engineering/using-pytorch-engineering/SKILL.md`
- Create: `source/planning/testing/using-pytorch-engineering-GREEN.md`
- Create: `source/planning/testing/using-pytorch-engineering-REFACTOR.md`

**Follow same RED-GREEN-REFACTOR pattern as Task 2-4:**

**RED Phase:**
- Scenarios: Memory error, distributed training setup, performance bottleneck, debugging NaN, checkpointing
- Test WITHOUT skill
- Document failures

**GREEN Phase:**
- Write meta-skill routing to specific PyTorch skills:
  - tensor-operations-and-memory
  - module-design-patterns
  - distributed-training-strategies
  - mixed-precision-and-optimization
  - performance-profiling
  - debugging-techniques
  - checkpointing-and-reproducibility
  - custom-autograd-functions
- Test WITH skill
- Verify routing

**REFACTOR Phase:**
- Pressure test
- Add rationalization table
- Re-test until bulletproof

**Commit after each phase (3 commits total)**

---

### Tasks 7-14: Individual PyTorch Skills (8 skills × 1-2h each)

For EACH skill below, follow this pattern:

**Skill 1: tensor-operations-and-memory** (1-2 hours)
**Skill 2: module-design-patterns** (1-2 hours)
**Skill 3: distributed-training-strategies** (1-2 hours)
**Skill 4: mixed-precision-and-optimization** (1-2 hours)
**Skill 5: performance-profiling** (1-2 hours)
**Skill 6: debugging-techniques** (1-2 hours)
**Skill 7: checkpointing-and-reproducibility** (1-2 hours)
**Skill 8: custom-autograd-functions** (1-2 hours)

**Pattern for EACH skill:**

**Files:**
- Create: `source/yzmir/pytorch-engineering/[skill-name]/SKILL.md`
- Create: `source/planning/testing/[skill-name]-RED.md`
- Create: `source/planning/testing/[skill-name]-GREEN.md`
- Create: `source/planning/testing/[skill-name]-REFACTOR.md`

**RED Phase Steps:**
1. Design 2-3 application scenarios (technique skills need application tests)
2. Dispatch subagent WITHOUT skill
3. Document what they do wrong:
   - Wrong approach?
   - Missing best practices?
   - Common pitfalls hit?
4. Identify patterns to address
5. Commit RED results

**GREEN Phase Steps:**
1. Write SKILL.md with structure:
   - Overview (1-2 sentences)
   - When to Use (symptoms/triggers)
   - Expert Patterns (how to do it right)
   - Common Pitfalls (what breaks)
   - Debugging Methodology (if applicable)
   - Examples (one excellent example)
2. Test WITH skill
3. Verify agent applies patterns correctly
4. Commit GREEN phase

**REFACTOR Phase Steps:**
1. Add edge case scenarios
2. Test under time pressure
3. Build pitfalls table
4. Add red flags if needed
5. Re-test until bulletproof
6. Commit REFACTOR phase

**Each skill = 3 commits (RED, GREEN, REFACTOR)**

---

## Phase 1A Completion Checkpoint

**After Tasks 1-14 complete:**

**Step 1: Verify Phase 1A deliverables**

```bash
tree source/yzmir/ -L 3
```

Expected:
```
source/yzmir/
├── ai-engineering-expert/
│   └── using-ai-engineering/
│       └── SKILL.md
└── pytorch-engineering/
    ├── using-pytorch-engineering/
    │   └── SKILL.md
    ├── tensor-operations-and-memory/
    │   └── SKILL.md
    ├── module-design-patterns/
    │   └── SKILL.md
    [... 6 more skills ...]
```

**Step 2: Test cross-pack routing**

Scenario: "I'm getting PyTorch memory errors during training"

Expected flow:
1. Load `using-ai-engineering`
2. Routes to `using-pytorch-engineering`
3. Routes to `tensor-operations-and-memory`

**Step 3: Update README and FACTIONS**

Modify `source/README.md`:
- Add Yzmir section
- List ai-engineering-expert (1 skill)
- List pytorch-engineering (1 meta + 8 skills)

Modify `source/FACTIONS.md`:
- Update Yzmir status: "Phase 1A Complete (9 skills)"

**Step 4: Commit Phase 1A completion**

```bash
git add source/README.md source/FACTIONS.md
git commit -m "docs: update for Yzmir Phase 1A completion

Phase 1A Complete:
- Primary router (using-ai-engineering)
- PyTorch Engineering pack (using-pytorch-engineering + 8 skills)

Total: 9 skills implemented and tested with RED-GREEN-REFACTOR

Next: Phase 1B (training-optimization)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 1B: Training Optimization Pack (15-25 hours)

### Task 15: Create Training Optimization Pack Structure

**Same pattern as Task 5** - create directory structure, commit

---

### Task 16: Training Pack Meta-Skill (RED-GREEN-REFACTOR)

**Same pattern as Task 6** - three-phase testing for pack meta-skill

**Routes to:**
- optimizer-selection-framework
- learning-rate-schedules
- batch-size-and-gradient-accumulation
- loss-landscape-analysis
- gradient-flow-debugging
- convergence-diagnostics
- regularization-techniques
- hyperparameter-tuning-methodology
- training-instability-troubleshooting

---

### Tasks 17-25: Individual Training Optimization Skills (9 skills × 1-2h each)

**Skill 1: optimizer-selection-framework** (Decision framework skill)
**Skill 2: learning-rate-schedules** (Technique skill)
**Skill 3: batch-size-and-gradient-accumulation** (Technique skill)
**Skill 4: loss-landscape-analysis** (Reference + technique skill)
**Skill 5: gradient-flow-debugging** (Debugging methodology skill)
**Skill 6: convergence-diagnostics** (Debugging methodology skill)
**Skill 7: regularization-techniques** (Reference + decision framework)
**Skill 8: hyperparameter-tuning-methodology** (Pattern skill)
**Skill 9: training-instability-troubleshooting** (Debugging methodology skill)

**Each follows RED-GREEN-REFACTOR pattern (3 commits per skill)**

**Testing focus:**
- Decision skills: Recognition scenarios (when X vs Y)
- Debugging skills: Failure scenarios (diagnose and fix)
- Technique skills: Application scenarios (implement correctly)

---

## Phase 1B Completion Checkpoint

**After Tasks 15-25 complete:**

Update README/FACTIONS, verify cross-pack routing, commit completion

Expected: 10 more skills (1 meta + 9 specific)

**Total so far: 19 skills**

---

## Phase 1C: Deep RL Pack (25-50 hours)

**Largest pack - your passion project**

### Task 26: Create Deep RL Pack Structure

**Same pattern** - create directory, commit

---

### Task 27: Deep RL Pack Meta-Skill (RED-GREEN-REFACTOR)

**Routes based on:**
- Action space (discrete vs continuous)
- Data availability (on/off-policy, offline)
- Sample efficiency needs
- Stability requirements

**Routes to:**
- rl-fundamentals
- value-based-methods
- policy-gradient-methods
- actor-critic-algorithms
- offline-rl-methods
- model-based-rl
- experience-replay-strategies
- exploration-techniques
- reward-shaping-engineering
- rl-debugging-methodology
- environment-design-patterns
- rl-evaluation-benchmarking

---

### Tasks 28-39: Individual Deep RL Skills (12 skills × 2-4h each)

**Skill 1: rl-fundamentals** (3-4 hours) - Foundation concepts
**Skill 2: value-based-methods** (3-4 hours) - DQN family
**Skill 3: policy-gradient-methods** (3-4 hours) - PPO/TRPO (detailed example in design doc)
**Skill 4: actor-critic-algorithms** (3-4 hours) - SAC/TD3
**Skill 5: offline-rl-methods** (3-4 hours) - CQL/IQL
**Skill 6: model-based-rl** (3-4 hours) - Dreamer patterns
**Skill 7: experience-replay-strategies** (2-3 hours)
**Skill 8: exploration-techniques** (2-3 hours)
**Skill 9: reward-shaping-engineering** (3-4 hours) - Critical for RL success
**Skill 10: rl-debugging-methodology** (3-4 hours) - Systematic debugging
**Skill 11: environment-design-patterns** (2-3 hours)
**Skill 12: rl-evaluation-benchmarking** (2-3 hours)

**Each follows RED-GREEN-REFACTOR (3 commits per skill)**

**RL-specific testing:**
- Algorithm selection scenarios
- Training instability scenarios
- Reward hacking scenarios
- Credit assignment failures
- Environment design issues

**Note:** Refer to detailed `policy-gradient-methods` example in design doc for skill structure inspiration

---

## Phase 1C Completion Checkpoint

Update README/FACTIONS, verify routing, commit

Expected: 13 more skills (1 meta + 12 specific)

**Total so far: 32 skills**

---

## Phase 1D: Modern AI Stack (30-55 hours)

### Neural Architectures Pack (12-20 hours)

**Task 40: Structure + Meta-Skill** (RED-GREEN-REFACTOR)

**Task 41-48: 8 Individual Skills**
- cnn-families-and-selection
- sequence-models-comparison
- transformer-architecture-deepdive
- attention-mechanisms-catalog
- generative-model-families
- graph-neural-networks-basics
- normalization-techniques
- architecture-design-principles

---

### LLM Specialist Pack (10-18 hours)

**Task 49: Structure + Meta-Skill** (RED-GREEN-REFACTOR)

**Task 50-56: 7 Individual Skills**
- transformer-for-llms
- tokenization-strategies
- fine-tuning-methods
- alignment-techniques
- long-context-techniques
- inference-optimization
- llm-evaluation-frameworks

---

### ML Production Pack (10-18 hours)

**Task 57: Structure + Meta-Skill** (RED-GREEN-REFACTOR)

**Task 58-64: 7 Individual Skills**
- model-serving-patterns
- quantization-for-inference
- model-compression-techniques
- performance-benchmarking
- production-monitoring
- experiment-tracking
- deployment-patterns

---

## Phase 1D Completion Checkpoint

Update README/FACTIONS, verify all routing paths, commit

Expected: 23 more skills (3 metas + 20 specific)

**Phase 1 Total: 55 skills (7 metas + 48 specific)**

---

## Final Phase 1 Validation

### Task 65: Cross-Pack Integration Testing

**Goal:** Verify multi-pack scenarios work correctly

**Test scenarios:**
1. "Train RL agent and deploy to production"
   - Expected: deep-rl + ml-production
2. "Fine-tune LLM with distributed training"
   - Expected: llm-specialist + pytorch-engineering
3. "Optimize transformer training for speed"
   - Expected: training-optimization + neural-architectures
4. "Debug PyTorch memory issues during RL training"
   - Expected: pytorch-engineering + deep-rl

**For EACH scenario:**
1. Load `using-ai-engineering`
2. Verify correct multi-pack routing
3. Verify skills load in correct order
4. Document results

**Commit integration test results**

---

### Task 66: Update Documentation

**Files to update:**
- `source/README.md` - Complete Yzmir catalog
- `source/FACTIONS.md` - Mark Yzmir Phase 1 complete
- `source/planning/2025-10-29-yzmir-ai-engineering-design.md` - Update status

**README.md additions:**

```markdown
### Yzmir - AI/ML Engineering (55 skills) ✅ Phase 1 Complete

**Meta-Skill:**
- **using-ai-engineering** - Routes to appropriate AI/ML pack

**Core Packs (Phase 1):**
- **pytorch-engineering** (9 skills) - Foundation layer
- **neural-architectures** (9 skills) - Architecture catalog
- **training-optimization** (10 skills) - Making training work
- **deep-rl** (13 skills) - Complete RL algorithms
- **llm-specialist** (8 skills) - Modern LLM techniques
- **ml-production** (8 skills) - Production deployment
```

**Commit documentation updates**

---

### Task 67: Tag Phase 1 Release

**Goal:** Create git tag for Phase 1 completion

```bash
git tag -a v0.2-phase1-yzmir -m "Yzmir AI/ML Engineering Phase 1 Complete

55 skills implemented across 6 packs:
- ai-engineering-expert (primary router)
- pytorch-engineering (9 skills)
- neural-architectures (9 skills)
- training-optimization (10 skills)
- deep-rl (13 skills)
- llm-specialist (8 skills)
- ml-production (8 skills)

All skills tested with RED-GREEN-REFACTOR methodology.
Estimated effort: 120-150 hours invested.

Next: Phase 2 (Domain Applications)"

git tag -l
```

Expected: Shows v0.2-phase1-yzmir

**Commit tag creation**

---

### Task 68: Real-World Validation

**Goal:** Test skills on 3-5 real ML tasks to validate practical value

**Validation tasks:**
1. Train a PPO agent for CartPole
   - Use: deep-rl skills
   - Verify: Correct algorithm selection, proper implementation
2. Fine-tune a small LLM (GPT-2)
   - Use: llm-specialist + pytorch-engineering
   - Verify: Efficient fine-tuning, proper techniques
3. Debug training instability
   - Use: training-optimization
   - Verify: Systematic debugging, root cause identification
4. Optimize model for inference
   - Use: ml-production
   - Verify: Correct quantization, serving patterns
5. Implement custom PyTorch module
   - Use: pytorch-engineering
   - Verify: Best practices, proper patterns

**For EACH task:**
1. Start fresh with task description
2. Load `using-ai-engineering`
3. Follow skill guidance
4. Document:
   - Did skills provide value?
   - Were patterns correct?
   - Any gaps identified?

**Create validation report:**

`source/planning/testing/phase1-validation-report.md`:

```markdown
# Yzmir Phase 1 - Real-World Validation Report

Date: [TODAY]
Status: Validation complete

## Task 1: [Name]
**Description:** [task]
**Skills Used:** [list]
**Outcome:** ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED
**Notes:** [what worked, what didn't]
**Gaps Identified:** [any issues]

[... repeat for all tasks ...]

## Overall Assessment:
- ✅ Value provided: [summary]
- ✅ Patterns correct: [summary]
- ⚠️ Gaps to address: [list]

## Recommendations:
1. [improvement]
2. [improvement]
```

**Commit validation report**

---

## Phase 1 Complete! 🎉

**Final deliverables:**
- 55 skills (7 meta-skills + 48 specific skills)
- 6 packs (pytorch-engineering, neural-architectures, training-optimization, deep-rl, llm-specialist, ml-production)
- All tested with RED-GREEN-REFACTOR
- Cross-pack integration verified
- Real-world validation complete
- Documentation updated
- Git tag created

**Estimated total commits:** ~180-200 commits (3 per skill × 55 skills + structure + docs)

**Next steps:**
- Phase 2: Domain Applications (computer-vision, nlp-fundamentals, audio-speech, multimodal-ai)
- Cross-reference integration with Ordis/Muna (dedicated curator agent)
- Community feedback and iteration

---

## Implementation Notes

### Commit Message Format

All commits follow this format:

```
<type>(scope): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New skill
- `test`: Testing phase (RED/GREEN/REFACTOR)
- `refactor`: REFACTOR phase updates
- `docs`: Documentation updates
- `chore`: Structure/setup

**Scopes:**
- `yzmir`: General faction
- `pytorch`: pytorch-engineering pack
- `rl`: deep-rl pack
- `llm`: llm-specialist pack
- `arch`: neural-architectures pack
- `train`: training-optimization pack
- `prod`: ml-production pack

### Testing Discipline

**Every skill MUST follow:**
1. RED: Test without → Document failures
2. GREEN: Write skill → Test with skill
3. REFACTOR: Pressure test → Add counters → Re-test

**No exceptions.** No batching. One skill at a time.

### Time Estimates

**Per skill:**
- Meta-skills: 2-3 hours (routing is simpler)
- Technique skills: 2-3 hours (application testing)
- Decision framework skills: 2-3 hours (recognition testing)
- Reference skills: 1-2 hours (retrieval testing)
- Complex debugging skills: 3-4 hours (systematic testing)

**Phase totals:**
- Phase 1A: 15-25 hours (infrastructure + foundation)
- Phase 1B: 15-25 hours (training core)
- Phase 1C: 25-50 hours (deep RL passion project)
- Phase 1D: 30-55 hours (modern AI stack)

**Total: 85-155 hours (target: 120-150 hours)**

### Parallel Work (Optional)

If multiple agents available:
- One agent per pack (after Phase 1A complete)
- Coordinate through commits
- Merge at completion checkpoints

### Quality Gates

**Before moving to next skill:**
- ✅ RED phase documented
- ✅ GREEN phase tested
- ✅ REFACTOR phase bulletproof
- ✅ All 3 commits pushed

**Before moving to next pack:**
- ✅ All skills complete
- ✅ Meta-skill routes correctly
- ✅ Cross-pack scenarios tested

**Before Phase 1 completion:**
- ✅ All 6 packs complete
- ✅ Primary router works
- ✅ Documentation updated
- ✅ Real-world validation passed

---

## Appendix: Skill Template

Use this template for consistency:

```markdown
---
name: skill-name-with-hyphens
description: Use when [specific triggering conditions] - [what skill does and how it helps]
---

# Skill Name

## Overview
Core principle in 1-2 sentences.

## When to Use

**Use this skill when:**
- Symptom 1
- Symptom 2
- Symptom 3

**Don't use when:**
- Alternative case 1
- Alternative case 2

**Symptoms triggering this skill:**
- "Error message pattern"
- "Problem description"

## [Core Content Section]

### Decision Framework (for decision skills)

| Situation | Choice | Why |
|-----------|--------|-----|
| Case 1 | Option A | Reason |
| Case 2 | Option B | Reason |

### Expert Patterns (for technique skills)

```python
def example_pattern():
    """
    WHY: Explanation of why this pattern matters
    """
    # Implementation
    pass
```

### Common Pitfalls

❌ **Pitfall 1**
→ Symptom: What breaks
→ Fix: How to fix

❌ **Pitfall 2**
→ Symptom: What breaks
→ Fix: How to fix

## Debugging Methodology (for debugging skills)

### Symptom: [Problem]

**Diagnostic steps:**
1. Check X
2. Check Y
3. Check Z

## Red Flags (for discipline skills)

| Excuse | Reality |
|--------|---------|
| "Rationalization" | "Truth" |

## Examples

One excellent example with complete code and explanation.

## References

- Paper/doc references if applicable
- Cross-references to related skills
```

---

**END OF IMPLEMENTATION PLAN**
