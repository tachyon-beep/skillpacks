---
description: Initiate systematic systems analysis with pattern recognition, archetype matching, and intervention design
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[problem_description_or_domain]"
---

# Analyze System Command

You are initiating a systematic systems analysis. Follow the layered workflow to understand complex, interconnected problems and design effective interventions.

## Core Principle

**Small shifts at high leverage points beat massive efforts at low leverage points.**

Systems are governed by archetypal structures. Recognize the pattern, apply the known intervention.

## When to Use Systems Thinking

**Use this when:**
- Problems persist despite repeated fixes
- Solutions create new problems (unintended consequences)
- System behavior is counter-intuitive
- Multiple stakeholders with conflicting incentives
- Long delays between action and result
- "The harder we push, the harder the system pushes back"

**Don't use when:**
- Simple, isolated problems with clear cause-effect
- One-time decisions with immediate results
- Pure optimization without feedback dynamics

## Systematic Analysis Workflow

### Phase 1: Pattern Recognition (15-30 min)

**Goal:** Identify the dominant patterns driving behavior

1. **Identify key variables** (states, not actions)
   - What can increase or decrease?
   - What accumulates over time?

2. **Map feedback loops**
   - Reinforcing (R): Amplifies change
   - Balancing (B): Resists change

3. **Find delays**
   - Where does action → result take time?
   - Information delays? Material delays?

4. **Recognize signatures**
   - S-curve growth? Oscillation? Death spiral?

### Phase 2: Archetype Matching (20-30 min)

**Goal:** Match to known patterns with proven solutions

**Check against 10 archetypes:**

| Pattern | Signature |
|---------|-----------|
| Fixes that Fail | Solution works temporarily, returns worse |
| Shifting the Burden | Quick fix prevents fundamental solution |
| Escalation | Two parties making it worse |
| Success to the Successful | Winner gets more resources |
| Tragedy of the Commons | Individual optimization degrades shared resource |
| Accidental Adversaries | Good intentions, mutual harm |
| Drifting Goals | Standards erode from complacency |
| Limits to Growth | Growth stops at constraint |
| Growth and Underinvestment | Growth stops from underinvestment |
| Eroding Goals | Standards erode from pressure |

**Diagnostic questions for each archetype** guide identification.

### Phase 3: Quantitative Analysis (45-60 min)

**Goal:** Calculate concrete predictions

Using stocks-and-flows modeling:
- What is current state?
- What is rate of change?
- When will crisis/opportunity hit?
- What is equilibrium?

**Output:** Specific numbers (e.g., "6.7 weeks to crisis at current rate")

### Phase 4: Leverage Point Identification (20-30 min)

**Goal:** Find high-impact intervention points

**Meadows' hierarchy (weakest to strongest):**
12. Parameters (numbers, budgets)
11. Buffers (reserve capacity)
10. Structure (physical systems)
9. Delays (feedback timing)
8. Balancing loops (error correction)
7. Reinforcing loops (amplification)
6. Information flows (visibility)
5. Rules (incentives, constraints)
4. Self-organization (evolution)
3. Goals (system purpose)
2. Paradigms (mental models)
1. Transcending paradigms

**Rule:** Generate alternatives at multiple levels before choosing.

### Phase 5: Intervention Design (30 min)

**Goal:** Design actionable interventions

For each proposed intervention:
- What leverage level?
- What archetype does it address?
- Prerequisites required?
- Expected resistance?
- Time to impact?

### Phase 6: Communication Prep (30 min)

**Goal:** Create stakeholder-ready outputs

**Behavior-over-time graphs:**
- Current trajectory
- Intervention scenario
- With concrete numbers and dates

## Analysis Output Format

```markdown
# Systems Analysis: [Problem Domain]

## Executive Summary
[2-3 sentences: Pattern identified, key insight, recommended intervention]

## Pattern Recognition

### Key Variables
- [Variable 1]: [Description, measurement]
- [Variable 2]: [Description, measurement]

### Feedback Loops
- R1: [Description of reinforcing loop]
- B1: [Description of balancing loop]

### Delays
- [Delay 1]: [Duration, impact]

## Archetype Match

**Primary:** [Archetype name]
**Evidence:** [Why this pattern matches]
**Known intervention:** [What works for this archetype]

**Secondary (if applicable):** [Archetype name]

## Quantitative Predictions

- Current state: [Value]
- Rate of change: [Value/time]
- Crisis timing: [Date/duration]
- Equilibrium: [Value if no intervention]

## Leverage Analysis

| Level | Intervention | Expected Impact | Risk |
|-------|--------------|-----------------|------|
| [#] | [Description] | [Impact] | [Risk] |

## Recommended Intervention

**Primary action:** [Description]
**Leverage level:** [#] - [Level name]
**Why this level:** [Justification]

**Prerequisites:**
- [What must be in place first]

**Expected resistance:**
- [Where pushback will come from]

**Timeline:**
- [When impact expected]

## Behavior Over Time

[ASCII graph or description showing trajectories]

## Limitations

- [What wasn't analyzed]
- [Confidence gaps]
- [Recommended deeper analysis]
```

## Time-Constrained Analysis

**For 60-minute deadline:**

| Time | Activity |
|------|----------|
| 0-15 min | Pattern recognition (key variables, loops) |
| 15-35 min | Archetype matching (check signatures) |
| 35-50 min | Leverage points (generate 3+ options) |
| 50-60 min | Summary + recommended intervention |

**Output:** Pattern + archetype + recommended intervention
**Trade-off:** No quantitative modeling

## Cross-Pack Discovery

```python
import glob

# For simulation implementation
sim_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if sim_pack:
    print("Available: yzmir-simulation-foundations for implementing models")

# For architecture analysis
arch_pack = glob.glob("plugins/axiom-system-architect/plugin.json")
if arch_pack:
    print("Available: axiom-system-architect for code architecture assessment")
```

## Red Flags - Stop and Reconsider

| Thought | Response |
|---------|----------|
| "Just add more resources" | Resource additions are lowest leverage (Level 12) |
| "This isn't a system, it's simple" | Persistent "simple" problems have hidden loops |
| "We don't have time for analysis" | Wrong action makes crisis worse |
| "Our situation is unique" | 90% match known archetypes |

## Scope Boundaries

**This command covers:**
- Pattern recognition
- Archetype matching
- Leverage point identification
- Intervention design
- Stakeholder communication prep

**Not covered:**
- Detailed stock-flow calculations (use /map-dynamics)
- Implementation planning (use project management)
- Code architecture (use axiom-system-architect)
