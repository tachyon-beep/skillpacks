---
description: Interactive algorithm selection wizard - match problem characteristics to the right RL algorithm
allowed-tools: ["Read", "Skill", "AskUserQuestion"]
argument-hint: ""
---

# Select Algorithm Command

Interactive wizard to select the right RL algorithm based on problem characteristics.

## The Core Questions

You MUST ask these questions before recommending an algorithm:

### Question 1: Action Space

**Ask**: "What actions can your agent take?"

| Answer | Implication |
|--------|-------------|
| Discrete (buttons, menu choices, grid moves) | DQN family OR PPO |
| Continuous (joint angles, forces, steering) | SAC, TD3, or PPO |
| Mixed (some discrete, some continuous) | PPO (handles both) |

**Critical Rule**: NEVER recommend DQN for continuous actions.

### Question 2: Data Regime

**Ask**: "Can your agent interact with the environment during training, or do you have a fixed dataset?"

| Answer | Implication |
|--------|-------------|
| Online (agent interacts, tries actions) | Standard algorithms (DQN, PPO, SAC) |
| Offline (fixed dataset, no interaction) | CQL, IQL (offline-rl) - standard algorithms FAIL |

**Critical Rule**: Offline data requires special algorithms. DQN/PPO/SAC will fail.

### Question 3: Sample Efficiency

**Ask**: "How many environment interactions can you afford?"

| Answer | Implication |
|--------|-------------|
| Unlimited (fast simulator) | PPO (simple, stable) |
| Limited (<100k steps, expensive sim) | SAC (off-policy, sample efficient) |
| Very limited (<10k, real robot) | Model-based RL (MBPO, Dreamer) |

### Question 4: Special Requirements

**Ask**: "Any special requirements?"

| Requirement | Algorithm |
|-------------|-----------|
| Multiple agents | QMIX, MADDPG (multi-agent-rl) |
| Sparse rewards | Add curiosity/RND (exploration-strategies) |
| Need interpretability | Consider simpler (DQN, REINFORCE) |
| Must be deterministic | TD3 (deterministic policy) |

## Decision Tree

```
START
│
├─ Offline data only?
│  └─ YES → CQL or IQL (offline-rl)
│
├─ Continuous actions?
│  ├─ YES + sample efficiency critical → SAC
│  ├─ YES + stability critical → TD3
│  └─ YES + simplicity preferred → PPO
│
├─ Discrete actions?
│  ├─ Small action space (<100) → DQN, Double DQN
│  └─ Large action space → PPO
│
├─ Multi-agent?
│  ├─ Cooperative → QMIX, COMA
│  └─ Competitive/Mixed → MADDPG
│
└─ Extreme sample efficiency needed?
   └─ YES → Model-based (MBPO, Dreamer)
```

## Output Format

After gathering information:

```markdown
## Algorithm Recommendation

### Problem Characteristics
- Action space: [discrete/continuous]
- Data regime: [online/offline]
- Sample budget: [unlimited/limited/very limited]
- Special requirements: [none/multi-agent/sparse rewards/etc.]

### Recommended Algorithm: [NAME]

**Why this algorithm:**
- [Reason 1 based on action space]
- [Reason 2 based on data regime]
- [Reason 3 based on constraints]

**Alternatives to consider:**
- [Alternative 1]: Use if [condition]
- [Alternative 2]: Use if [condition]

### Next Steps
1. Load [algorithm-skill].md for implementation details
2. Use `/deep-rl:new-experiment --algorithm=[name]` to scaffold
3. Follow rl-debugging if training issues arise
```

## Common Mistakes to Prevent

| User Says | Wrong Choice | Correct Choice | Why |
|-----------|--------------|----------------|-----|
| "I'll just use PPO" | PPO for everything | Depends on problem | PPO is good but not optimal everywhere |
| "DQN for my robot arm" | DQN for continuous | SAC or TD3 | DQN requires discrete actions |
| "I have logged data from production" | PPO on offline data | CQL or IQL | Offline needs conservative algorithms |
| "I want the newest algorithm" | Latest paper | Problem-appropriate | Newer ≠ better for your problem |

## Load Detailed Guidance

For complete decision framework:
```
Load skill: yzmir-deep-rl:using-deep-rl
Read: SKILL.md (contains full routing decision tree)
```

For specific algorithm details:
```
- value-based-methods.md (DQN family)
- policy-gradient-methods.md (PPO, REINFORCE)
- actor-critic-methods.md (SAC, TD3)
- offline-rl.md (CQL, IQL)
- multi-agent-rl.md (QMIX, MADDPG)
```
