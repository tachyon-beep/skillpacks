---
description: Design sandbox systems with meaningful constraints and progressive revelation
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[sandbox_context]"
---

# Design Sandbox Command

You are designing sandbox systems where creative freedom emerges from meaningful constraints, and players learn through doing rather than reading.

## Core Principle

**The Constraint Paradox: Creative freedom requires constraints. Unlimited options paralyze; meaningful limits inspire. Sandbox design is constraint curation.**

## The Sandbox Formula

```
Player Agency = Meaningful Choices × Clear Consequences × Reversibility Safety
```

- **Meaningful**: Choices matter, affect the world
- **Clear**: Outcomes are understandable, not arbitrary
- **Safe**: Can experiment without catastrophic failure

## Information Gathering

Before designing, determine:

1. **Player fantasy**: What power/freedom are they seeking?
2. **Core constraints**: What limits create interesting choices?
3. **Consequence scope**: Local, regional, global impact?
4. **Failure tolerance**: How punishing is experimentation?
5. **Discovery layers**: What reveals over time?

## Constraint Design Framework

### Types of Productive Constraints

| Constraint Type | Purpose | Example |
|----------------|---------|---------|
| **Resource** | Forces prioritization | Limited inventory, energy, time |
| **Spatial** | Creates territory value | Build zones, travel costs |
| **Temporal** | Adds urgency/rhythm | Day/night, seasons, cooldowns |
| **Social** | Enables reputation | NPC memory, faction relationships |
| **Physical** | Grounds in reality | Gravity, structural integrity |
| **Knowledge** | Rewards exploration | Hidden recipes, secret areas |

### Constraint Balance

```yaml
too_few_constraints:
  result: "Paralysis of choice, everything feels same"

too_many_constraints:
  result: "Puzzle, not sandbox - one solution"

balanced_constraints:
  result: "Many valid approaches, meaningful tradeoffs"
```

## Progressive Revelation Design

### Layer 1: Immediate (0-30 minutes)

```yaml
player_learns:
  - Basic interactions work
  - World responds to actions
  - Core loop is engaging

how_taught:
  - Environmental affordances
  - Immediate feedback
  - Safe experimentation space
```

### Layer 2: Discovery (1-5 hours)

```yaml
player_learns:
  - System interactions
  - Optimal strategies exist
  - Choices have tradeoffs

how_taught:
  - Natural consequences
  - NPC hints (not lectures)
  - Pattern recognition rewards
```

### Layer 3: Mastery (10+ hours)

```yaml
player_learns:
  - Edge cases and exploits
  - System depth
  - Personal expression through systems

how_taught:
  - Community knowledge
  - Self-directed exploration
  - Emergent discoveries
```

## Onboarding Through Doing

### Anti-Pattern: Tutorial

```
❌ Text box: "Press E to interact"
❌ Forced path through mechanics
❌ Locked content until "learned"
```

### Pattern: Environmental Teaching

```yaml
first_area_design:
  visible_goal: "Door is clearly the exit"
  blocking_element: "Crate blocks path"
  affordance: "Crate looks movable"
  discovery: "Player pushes crate, learns interaction"

key_principles:
  - Goal visible before solution
  - Failure is cheap
  - Success feels discovered
```

## World as Teacher

### Environmental Storytelling

```yaml
instead_of: "NPCs died from poison gas"
show:
  - Bodies near ventilation
  - Gas masks on some
  - Scattered antidotes
  - Player can piece together
```

### Systemic Consequences as Lessons

```yaml
instead_of: "Fire is dangerous (tooltip)"
design:
  - Small fire near wood
  - Fire spreads to wood
  - Player experiences consequence
  - Learns fire + wood interaction
```

## Player Expression Architecture

### Tool vs Toy Spectrum

```
Tool ←――――――――――――――――→ Toy
Efficient              Expressive
One solution           Many solutions
Mastery = speed        Mastery = creativity
```

**Sandbox goal**: Lean toward Toy while enabling Tool mastery

### Expression Dimensions

| Dimension | Enables | Example |
|-----------|---------|---------|
| **Aesthetic** | Visual identity | Building styles, colors |
| **Functional** | Problem-solving | Base layouts, vehicle designs |
| **Social** | Reputation/role | Trader, warrior, explorer |
| **Narrative** | Personal story | Choices that define character |

## Output Format

```markdown
# Sandbox Design: [System Name]

## Context

**Player Fantasy**: [What freedom/power they seek]
**Core Experience**: [Moment-to-moment feeling]
**Scope**: [What can players affect]

## Constraint Architecture

### Primary Constraints

| Constraint | Type | Purpose | Interesting Choices Created |
|------------|------|---------|---------------------------|
| [Name] | [Type] | [Why] | [What decisions it forces] |

### Constraint Interactions

- [How constraints combine to create depth]

## Progressive Revelation

### Layer 1: First 30 Minutes
- **Player learns**: [Core interactions]
- **Through**: [Environmental teaching method]
- **Feels like**: [Discovery, not instruction]

### Layer 2: First 5 Hours
- **Player learns**: [System depth]
- **Through**: [Natural consequences]
- **Feels like**: [Mastery emerging]

### Layer 3: Ongoing
- **Player discovers**: [Edge cases, personal expression]
- **Through**: [Self-directed exploration]
- **Feels like**: [Ownership of knowledge]

## Onboarding Design

### First Area/Experience

```yaml
visible_goal: [What player wants]
blocking_element: [What's in the way]
affordance: [How solution is suggested]
discovery: [What player learns by doing]
```

### Teaching Sequence

1. [First mechanic introduced through...]
2. [Second mechanic builds on first by...]
3. [Combination opportunity reveals...]

## Expression Architecture

### Dimensions Available

| Dimension | Range of Expression | Examples |
|-----------|-------------------|----------|
| [Aesthetic] | [Options] | [Concrete examples] |
| [Functional] | [Options] | [Concrete examples] |
| [Narrative] | [Options] | [Concrete examples] |

### Player Types Supported

- **Builders**: [How they express]
- **Explorers**: [How they express]
- **Optimizers**: [How they express]
- **Socializers**: [How they express]

## Failure Safety

**Experimentation cost**: [Low/Medium/High]
**Recovery options**: [How to undo mistakes]
**Learning from failure**: [What mistakes teach]

## Scope Boundaries

**Player can affect**: [List]
**Player cannot affect**: [List - and why this is good]

## Implementation Notes

- [Technical considerations]
- [Content requirements]
- [Testing priorities]
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Unlimited freedom | Paralysis, no meaning | Add productive constraints |
| Tutorial walls | Kills discovery joy | Teach through environment |
| Permanent failure | Fear of experimentation | Add recovery/reversibility |
| Hidden mechanics | Feels arbitrary | Clear feedback loops |
| Single solution | Not a sandbox | Multiple valid approaches |

## Cross-Pack Discovery

```python
import glob

# For UX onboarding patterns
ux_pack = glob.glob("plugins/lyra-ux-designer/plugin.json")
if ux_pack:
    print("Available: lyra-ux-designer for onboarding flow design")

# For emergent gameplay
emergence_pack = glob.glob("plugins/bravos-systems-as-experience/plugin.json")
if emergence_pack:
    print("Available: /design-emergence for mechanic interaction design")
```

## Scope Boundaries

**This command covers:**
- Constraint architecture
- Progressive revelation
- Environmental teaching
- Player expression dimensions
- Failure safety design

**Not covered:**
- Mechanic interaction matrices (use /design-emergence)
- Technical implementation
- Content creation
- Balance tuning
