---
description: Analyze existing game systems for interaction depth, feedback loops, and emergence potential
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[game_or_codebase_path]"
---

# Analyze System Interactions Command

You are analyzing existing game systems to map interactions, identify feedback loops, and evaluate emergence potential.

## Core Principle

**Systems ARE the content. The depth of a game comes not from how many mechanics exist, but from how richly they interact. Analysis reveals both designed and emergent connections.**

## Analysis Framework

### What We're Looking For

| Aspect | Good Sign | Warning Sign |
|--------|-----------|--------------|
| **Interaction Density** | Many cross-system effects | Isolated mechanics |
| **Orthogonality** | Each mechanic does ONE thing well | Mechanics overlap/compete |
| **Feedback Clarity** | Clear cause → effect chains | Opaque/arbitrary outcomes |
| **Emergence Ratio** | Players find unintended combos | Only designed interactions work |
| **Loop Balance** | Mix of positive/negative loops | All positive (runaway) or negative (stagnant) |

## Analysis Protocol

### Step 1: Inventory Mechanics

List all player-facing mechanics:

```yaml
mechanics:
  - name: "Fire"
    primary_function: "Damage over time"
    systems_touched: [Combat, Environment, Crafting]

  - name: "Water"
    primary_function: "Extinguish, hydrate"
    systems_touched: [Environment, Farming, Alchemy]
```

### Step 2: Map Interactions

Create interaction matrix:

```
          Fire    Water   Wood    Metal
Fire      -       ✓       ✓       ✓
Water     ✓       -       ✓       ✓
Wood      ✓       ✓       -       ✗
Metal     ✓       ✓       ✗       -

Legend: ✓ = meaningful interaction, ✗ = no interaction, - = self
```

### Step 3: Trace Feedback Loops

Identify cyclic relationships:

```yaml
positive_loops:
  - name: "Expansion Spiral"
    path: "Territory → Resources → Army → Territory"
    risk: "Runaway winner"

negative_loops:
  - name: "Upkeep Brake"
    path: "Army Size → Upkeep Cost → Economic Strain → Army Reduction"
    purpose: "Prevents infinite scaling"
```

### Step 4: Find Cascade Chains

Map multi-step consequences:

```yaml
cascade_chains:
  - trigger: "Fire arrow hits oil barrel"
    step_1: "Oil ignites, spreads"
    step_2: "Fire reaches wooden structures"
    step_3: "Structures collapse"
    step_4: "Blocks path, creates new terrain"
    emergent: true  # Was this designed or discovered?
```

### Step 5: Evaluate Emergence Potential

Test questions:

1. Can players discover interactions not explicitly documented?
2. Do speedrunners/experts use unintended combinations?
3. Does community share "discovered" techniques?
4. Are there multiple valid solutions to challenges?

## Code Analysis Patterns

### Finding Game Systems

```bash
# Look for system managers, controllers
grep -r "class.*System" --include="*.cs" --include="*.cpp" --include="*.py"
grep -r "class.*Manager" --include="*.cs" --include="*.cpp" --include="*.py"

# Look for interaction handlers
grep -r "OnCollision\|OnTrigger\|OnInteract" --include="*.cs"
grep -r "interact\|collision\|trigger" --include="*.py" --include="*.gd"
```

### Finding Interactions

```bash
# Look for cross-system references
grep -r "FireSystem\|WaterSystem\|PhysicsSystem" --include="*.cs"

# Look for event/message passing
grep -r "SendMessage\|EventBus\|Signal" --include="*.cs" --include="*.gd"
```

### Finding Feedback Loops

```bash
# Look for self-referential updates
grep -r "this\.\w*\s*[+\-*]=.*this\.\w*" --include="*.cs" --include="*.py"

# Look for resource cycling
grep -r "Produce\|Consume\|Generate\|Drain" --include="*.cs"
```

## Output Format

```markdown
# System Interaction Analysis: [Game/Project Name]

## Executive Summary

**Interaction Density**: [Low/Medium/High]
**Orthogonality Score**: [X/10]
**Emergence Potential**: [Low/Medium/High]
**Loop Balance**: [Healthy/Positive-Heavy/Negative-Heavy]

## Mechanic Inventory

| Mechanic | Primary Function | Systems Touched | Orthogonal? |
|----------|-----------------|-----------------|-------------|
| [Name] | [What it does] | [List] | Yes/No |

## Interaction Matrix

[Grid showing mechanic × mechanic interactions]

### Interaction Details

#### [Mechanic A] × [Mechanic B]

- **Interaction**: [What happens]
- **Designed**: Yes/No
- **Player Discovery**: Common/Rare/Unknown
- **Depth**: [Shallow/Medium/Deep]

## Feedback Loop Map

### Positive Loops

```
[Loop name]: [A] → [B] → [C] → [A]
Risk: [What could go wrong]
Mitigation: [How it's balanced]
```

### Negative Loops

```
[Loop name]: [A] → [B] → [C] → [A]
Purpose: [What it prevents]
Risk: [Could cause stagnation if...]
```

### Loop Interactions

- [How loops affect each other]

## Cascade Chain Analysis

### Chain 1: [Name]

```
Trigger: [Initial event]
→ Step 1: [First consequence]
→ Step 2: [Second consequence]
→ Final: [End state]
```

**Designed/Emergent**: [Which]
**Player Agency Points**: [Where player can intervene]

## Emergence Evaluation

### Designed Interactions
[Count]: [X]

### Emergent Interactions Found
[Count]: [Y]

### Emergence Ratio
[Y/X] - Target: > 1.5

### Community-Discovered Techniques
- [List known emergent strategies]

## Gaps and Opportunities

### Isolated Mechanics
| Mechanic | Currently Touches | Could Connect To |
|----------|------------------|------------------|
| [Name] | [Systems] | [Opportunities] |

### Missing Loops
- [Feedback loops that could add depth]

### Cascade Opportunities
- [New chain possibilities]

## Recommendations

### High Priority
1. [Most impactful improvement]

### Medium Priority
1. [Worthwhile enhancement]

### Low Priority
1. [Nice-to-have connection]

## Technical Notes

- [Implementation considerations]
- [Performance implications]
- [Testing requirements]
```

## Analysis Anti-Patterns

| Mistake | Problem | Correct Approach |
|---------|---------|------------------|
| Only count mechanics | Misses interaction depth | Map connections, not just count |
| Ignore player behavior | Miss emergent usage | Study speedruns, wikis, community |
| Focus on complexity | More != better | Focus on meaningful interactions |
| Analyze in isolation | Miss context | Consider full player experience |

## Cross-Pack Discovery

```python
import glob

# For simulation analysis
sim_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if sim_pack:
    print("Available: yzmir-simulation-foundations for loop dynamics analysis")

# For codebase exploration
arch_pack = glob.glob("plugins/axiom-system-archaeologist/plugin.json")
if arch_pack:
    print("Available: axiom-system-archaeologist for codebase structure mapping")
```

## Scope Boundaries

**This command covers:**
- Mechanic inventory and mapping
- Interaction matrix creation
- Feedback loop identification
- Cascade chain tracing
- Emergence potential evaluation

**Not covered:**
- Designing new interactions (use /design-emergence)
- Sandbox constraint analysis (use /design-sandbox)
- Code refactoring recommendations
- Balance tuning suggestions
