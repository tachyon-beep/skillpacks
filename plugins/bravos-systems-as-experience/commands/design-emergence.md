---
description: Design emergent gameplay through orthogonal mechanics and interaction matrices
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[game_or_system_context]"
---

# Design Emergence Command

You are designing emergent gameplay systems where simple, orthogonal mechanics interact to create complex outcomes players discover rather than being told.

## Core Principle

**Emergent gameplay happens when simple orthogonal mechanics interact to create complex outcomes. The game teaches itself through systemic consequences, not tutorials.**

## The Emergence Formula

```
Emergence = Orthogonal Mechanics × Interaction Density × Feedback Clarity
```

- **Orthogonal**: Each mechanic does ONE thing, affects MANY systems
- **Interaction Density**: Number of meaningful system connections
- **Feedback Clarity**: How clearly outcomes communicate cause

## Information Gathering

Before designing, determine:

1. **Core mechanics**: What are the fundamental player actions?
2. **System domains**: Physics, chemistry, AI, economy, ecology?
3. **Player expression goal**: What "stories" should emerge?
4. **Complexity budget**: How many interacting systems?
5. **Teaching strategy**: How will players discover interactions?

## Orthogonality Test

For each mechanic, ask:

| Question | Good Sign | Bad Sign |
|----------|-----------|----------|
| Does it do ONE thing? | "Fire burns" | "Fire burns and heals allies" |
| Does it affect MANY systems? | Burns wood, enemies, food, ice | Only damages enemies |
| Is it predictable? | Always behaves same way | Context-dependent behavior |
| Is it combinable? | Works with other mechanics | Isolated system |

## Interaction Matrix Design

### Step 1: List Core Elements

```
Elements: [Fire, Water, Wood, Metal, Electricity, Ice, Oil, Explosive]
```

### Step 2: Define Pairwise Interactions

```
          Fire    Water   Wood    Metal   Electric
Fire      -       Steam   Burn    Heat    -
Water     Steam   -       Growth  Rust    Conduct
Wood      Burn    Growth  -       -       -
Metal     Heat    Rust    -       -       Conduct
Electric  -       Conduct -       Conduct -
```

### Step 3: Identify Cascade Chains

```
Oil + Fire → Burning Oil → spreads to Wood → Fire spreads → heats Metal → conducts to Water → Steam explosion
```

### Step 4: Verify Emergence

Test: Can players discover interactions you DIDN'T explicitly design?

## Feedback Loop Design

### Positive Loops (Amplifying)

```yaml
loop_type: positive
example: "More territory → More resources → Stronger army → More territory"
risk: Runaway winner, snowballing
balance: Add negative feedback or diminishing returns
```

### Negative Loops (Stabilizing)

```yaml
loop_type: negative
example: "Larger army → Higher upkeep → Economic strain → Smaller army"
purpose: Self-balancing, comeback mechanics
risk: Stagnation if too strong
```

### Loop Interaction

```
Positive loops drive progress
Negative loops create tension
Interacting loops create interesting decisions
```

## Systemic Solution Design

Enable multiple valid approaches:

```yaml
problem: "Enemy fortress"

solutions:
  direct_assault:
    mechanics: [Combat, Siege]
    cost: High casualties

  infiltration:
    mechanics: [Stealth, Lockpicking, Disguise]
    cost: High skill requirement

  economic_siege:
    mechanics: [Trade, Blockade, Diplomacy]
    cost: Time

  environmental:
    mechanics: [Fire, Flood, Earthquake]
    cost: Collateral damage

  social:
    mechanics: [Bribery, Propaganda, Assassination]
    cost: Reputation
```

**Test**: Can players combine solutions? (Infiltrate + Environmental = inside sabotage)

## Output Format

```markdown
# Emergent System Design: [System Name]

## Context

**Game/System Type**: [Description]
**Player Expression Goal**: [What stories should emerge]
**Complexity Budget**: [Number of interacting systems]

## Core Mechanics

| Mechanic | Does ONE Thing | Affects MANY |
|----------|---------------|--------------|
| [Name] | [Single purpose] | [Systems affected] |

## Interaction Matrix

[Element × Element matrix with outcomes]

## Cascade Chains

### Chain 1: [Name]
```
[Element] + [Element] → [Outcome] → [Propagation] → [Final State]
```

### Chain 2: [Name]
[...]

## Feedback Loops

### Positive Loops
- [Loop description with amplification path]

### Negative Loops
- [Loop description with stabilization path]

### Loop Interactions
- [How loops create interesting decisions]

## Systemic Solutions

### Problem: [Challenge]

| Approach | Mechanics Used | Tradeoff |
|----------|---------------|----------|
| [Name] | [Mechanics] | [Cost] |

### Combination Opportunities
- [How approaches can be combined]

## Discovery Teaching

**How players learn**:
1. [Early discovery opportunity]
2. [Mid-game revelation]
3. [Mastery insight]

## Emergence Verification

**Designed interactions**: [Count]
**Emergent interactions found in testing**: [Count]
**Ratio**: [Should be > 1.5x]

## Implementation Notes

- [Technical considerations]
- [Balance concerns]
- [Testing priorities]
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Kitchen sink | Too many mechanics, shallow interactions | Fewer mechanics, deeper connections |
| Isolated systems | Mechanics don't interact | Add interaction points |
| Explained emergence | Tutorials kill discovery | Let systems teach |
| Deterministic chains | No player agency in outcomes | Add decision points |
| Runaway loops | One strategy dominates | Add counterbalancing loops |

## Cross-Pack Discovery

```python
import glob

# For simulation foundations
sim_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if sim_pack:
    print("Available: yzmir-simulation-foundations for feedback loop mathematics")

# For game balance
tactics_pack = glob.glob("plugins/bravos-simulation-tactics/plugin.json")
if tactics_pack:
    print("Available: bravos-simulation-tactics for balance tuning")
```

## Scope Boundaries

**This command covers:**
- Orthogonal mechanic design
- Interaction matrix creation
- Feedback loop architecture
- Systemic solution spaces
- Discovery-based teaching

**Not covered:**
- Sandbox/world design (use /design-sandbox)
- Numeric balance tuning
- UI/UX for system communication
- Technical implementation details
