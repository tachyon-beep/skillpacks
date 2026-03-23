---
description: Identify high-leverage intervention points using Meadows' 12-level hierarchy
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[proposed_solution_or_problem]"
---

# Find Leverage Points Command

You are analyzing intervention options using Donella Meadows' 12 places to intervene in a system. Your goal is to find the highest-leverage point that's feasible given constraints.

## Core Principle

**Small shifts at high leverage points beat massive efforts at low leverage points.**

Most people intervene at Level 12 (parameters) because it's obvious and easy. Systems thinkers intervene at Levels 6-3 (information, rules, goals) because that's where change actually happens.

## The 12 Places to Intervene (Weakest to Strongest)

### Low Leverage (12-10): Low Risk, Low Reward

**12. Constants, Parameters, Numbers**
- What: Changing quantities (budgets, quotas, targets)
- Example: "Add more servers", "Hire more engineers"
- Why weak: Structure stays intact, system adapts

**11. Buffers (Stabilizing Stocks)**
- What: Reserve capacity that absorbs fluctuation
- Example: Connection pool size, retry queues
- Why stronger: Prevents cascade failures

**10. Stock-and-Flow Structures**
- What: Physical plumbing, topology
- Example: Microservices vs monolith
- Why stronger: Changes what's possible

### Medium Leverage (9-7): Moderate Risk and Reward

**9. Delays**
- What: Time between action and consequence
- Example: CI/CD speed, monitoring latency
- Warning: Shortening delays in reinforcing loops accelerates problems

**8. Balancing Feedback Loops**
- What: Error-correction mechanisms
- Example: Auto-rollback, rate limiters
- Use when: Want stability and self-correction

**7. Reinforcing Feedback Loops**
- What: Amplification mechanisms
- Example: Network effects, technical debt spiral
- Use when: Want to amplify virtuous cycles, dampen vicious ones

### High Leverage (6-5): High Reward, Moderate-High Risk

**6. Information Flows**
- What: Who sees what information when
- Example: Real-time dashboards, transparent incident reports
- Why counterintuitive: Seems passive but often more powerful than mandates

**5. Rules**
- What: Incentives, constraints, permissions
- Example: "You build it, you run it", deployment windows
- Warning: Rules get gamed if goals misaligned

### Highest Leverage (4-1): Highest Reward, Highest Risk

**4. Self-Organization**
- What: System's ability to evolve its own structure
- Example: Engineer-driven RFC process, autonomous teams
- Risk: May optimize locally at expense of global

**3. Goals**
- What: System purpose
- Example: "Prevent incidents" → "Learn from incidents"
- Why powerful: Everything else serves the goal

**2. Paradigms**
- What: Mental models, worldview
- Example: "Engineers as resources" → "Engineers as investors"
- Why powerful: Changes how we see everything

**1. Transcending Paradigms**
- What: Ability to step outside any paradigm
- Example: "Strong opinions, weakly held"
- Why powerful: Not attached to any one way of seeing

## Analysis Process

### Step 1: Identify Current Level

**What level is your proposed solution?**

| If your solution... | Level |
|---------------------|-------|
| Adjusts a number, budget, quantity | 12 |
| Adds capacity, reserves, slack | 11 |
| Redesigns architecture, topology | 10 |
| Speeds up or slows down a process | 9 |
| Adds monitoring, alerts, auto-scaling | 8 |
| Amplifies growth or dampens decline | 7 |
| Makes something visible, adds transparency | 6 |
| Changes policies, mandates, incentives | 5 |
| Enables teams to self-organize | 4 |
| Redefines what success means | 3 |
| Changes fundamental assumptions | 2 |
| Questions whether problem is real | 1 |

### Step 2: Generate Higher-Level Alternatives

**Ask "Why?" three times, then intervene there:**

Example: "We need more servers"
- Why? Response time is slow
- Why? 20 serial service calls
- Why? Designed for sync everywhere
- **Intervention:** Question "sync by default" paradigm (Level 2)

**Move up systematically:**

For solution at Level N, ask:
- Level N+1: What rule would make this parameter self-adjust?
- Level N+2: What information would make people want this?
- Level N+3: What goal would make this rule unnecessary?
- Level N+4: What paradigm shift would make this goal obvious?

### Step 3: Assess Prerequisites

**Higher leverage requires more prerequisites:**

| Level | Prerequisites Required |
|-------|----------------------|
| 12-10 | None, safe to experiment |
| 9-7 | Map system structure first |
| 6-5 | Leadership buy-in, understand power structures |
| 4-1 | Psychological safety, organizational readiness, patience |

### Step 4: Choose Appropriate Level

**Consider:**
- Urgency (lower levels are faster)
- Sustainability (higher levels last longer)
- Prerequisites available
- Resistance expected

**Often best:** Multi-level approach
- Level 12 tactically (buy time)
- Higher level strategically (sustainable change)

## Output Format

```markdown
# Leverage Analysis: [Problem/Solution]

## Current Proposal
**Solution:** [Description]
**Level:** [#] - [Level name]

## Higher-Level Alternatives

### Level [N+1]: [Level name]
**Alternative:** [Description]
**Prerequisite:** [What's needed]
**Expected resistance:** [From whom]

### Level [N+2]: [Level name]
[...]

## Recommendation

**Tactical (immediate):** Level [#] - [Description]
**Strategic (sustainable):** Level [#] - [Description]

**Rationale:** [Why this combination]

## Prerequisites to Address
- [Prerequisite 1]
- [Prerequisite 2]
```

## Red Flags - Rationalizations for Low Leverage

| Thought | Response |
|---------|----------|
| "Too urgent for high-leverage thinking" | Urgency is when leverage matters most |
| "High-leverage is too slow" | Low-leverage that fails is slower |
| "High-leverage is too risky" | Repeating failed low-leverage is riskier |
| "I don't have authority" | You have influence through information (Level 6) |
| "Let's just do what we can control" | Self-limiting your sphere of influence |

## Scope Boundaries

**This command covers:**
- Leverage point identification
- Higher-level alternative generation
- Prerequisite assessment
- Multi-level strategy design

**Not covered:**
- Full systems analysis (use /analyze-system)
- Detailed implementation planning
- Stakeholder communication
