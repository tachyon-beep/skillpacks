# Strategic Depth from Systems: Creating Emergent Strategy

## Purpose

**This skill teaches how to create strategic depth through system design.** It addresses the catastrophic failure mode where games appear complex but collapse to dominant strategies, leaving players with "false choices" and shallow gameplay.

Strategic depth ≠ complexity. Depth comes from:
- **Orthogonal mechanics** (multiple independent strategic axes)
- **Asymmetric design** (different viable approaches)
- **Synergy matrices** (combinatorial possibility space)
- **Counter-play systems** (rock-paper-scissors dynamics)
- **Branching progression** (meaningful build choices)

Without this skill, you will create games where:
- All choices converge to one optimal strategy (dominant strategy problem)
- Players feel choices don't matter (false depth)
- Factions/classes are reskins with no strategic difference (symmetric problem)
- Progression is linear with no meaningful branching (no build diversity)

---

## When to Use This Skill

Use this skill when:
- Designing strategy games (RTS, 4X, grand strategy, tower defense)
- Creating character builds/classes (RPGs, roguelikes, MOBAs)
- Designing asymmetric PvP (fighting games, card games)
- Building tech trees or skill trees
- Creating faction/race/civilization differences
- Designing card game decks or build variety
- Any system where player choice should create strategic depth
- Playtesting reveals "dominant strategy" or "everyone plays the same way"

**ALWAYS use this skill BEFORE implementing progression systems or factions.**

---

## Core Philosophy: True Depth vs False Complexity

### The Fundamental Truth

> **Depth is measured by the number of viable strategies, not the number of options.**

A game with 100 units and 1 viable strategy is SHALLOW.
A game with 10 units and 10 viable strategies is DEEP.

### The Depth Equation

```
Strategic Depth = Viable Strategies × Meaningful Choices × Skill Ceiling

Where:
  Viable Strategies = Number of approaches that can win
  Meaningful Choices = Decisions that affect outcome
  Skill Ceiling = Mastery headroom
```

### Example: Chess vs Tic-Tac-Toe

**Tic-Tac-Toe**:
- 9 possible moves (complexity = medium)
- 2-3 viable strategies (depth = low)
- Solved game (skill ceiling = low)
- **Result**: SHALLOW

**Chess**:
- ~40 possible moves per turn (complexity = high)
- 100+ viable openings (depth = extreme)
- Unsolved game (skill ceiling = infinite)
- **Result**: DEEP

**Lesson**: Chess is deeper not because it has more moves, but because it has more VIABLE strategies.

---

## CORE CONCEPT #1: Orthogonal Mechanics

**Orthogonal mechanics** are independent strategic axes that don't directly compete.

### What Makes Mechanics Orthogonal?

Two mechanics are orthogonal if:
1. They address different strategic problems
2. Improving one doesn't invalidate the other
3. They create combinatorial depth when mixed

### Example: Non-Orthogonal (One-Dimensional)

```
All mechanics scale DAMAGE:
  • Warrior: 10 damage melee
  • Archer: 8 damage ranged
  • Mage: 12 damage AoE

Problem: All solve same problem (damage), just differently.
Result: Whoever has highest DPS wins. Shallow.
```

### Example: Orthogonal (Multi-Dimensional)

```
Mechanics on different axes:
  • Warrior: High DAMAGE, low MOBILITY (offense axis)
  • Archer: Medium DAMAGE, high RANGE (positioning axis)
  • Healer: Zero DAMAGE, high SUSTAIN (support axis)
  • Scout: Low DAMAGE, high VISION (information axis)

Problem: Each addresses different strategic need.
Result: All viable, combos matter. Deep.
```

### The Strategic Axes Framework

Common orthogonal axes:

| Axis | What It Addresses | Example Units |
|------|------------------|---------------|
| **Offense** | Dealing damage | Warriors, DPS, artillery |
| **Defense** | Surviving damage | Tanks, healers, shields |
| **Mobility** | Positioning | Cavalry, teleporters, fliers |
| **Utility** | Board control | Stuns, walls, slows |
| **Economy** | Resource generation | Miners, farmers, traders |
| **Information** | Vision/scouting | Scouts, radar, spies |
| **Tempo** | Action speed | Fast units, initiative |

### Applying Orthogonal Design

**Step 1**: Identify 3-5 strategic axes for your game

**Step 2**: Distribute faction/unit strengths across axes

**Step 3**: Ensure no axis is "strictly better"

**Example: StarCraft Races**

```
Terran:
  • Offense: Medium
  • Defense: High (bunkers, repair)
  • Mobility: Low (static positioning)
  • Utility: High (scans, detector)
  • Economy: Medium (mules)

Zerg:
  • Offense: High (swarming)
  • Defense: Low (fragile units)
  • Mobility: Very High (creep, burrow)
  • Utility: Medium (infestors)
  • Economy: Very High (larva inject)

Protoss:
  • Offense: Very High (powerful units)
  • Defense: High (shields)
  • Mobility: Medium (warp-in)
  • Utility: High (force fields)
  • Economy: Low (expensive units)
```

**Result**: All races viable because they excel on DIFFERENT axes. No dominant strategy.

### Test: Are Your Mechanics Orthogonal?

Ask:
1. ☐ Can Unit A be strong WITHOUT invalidating Unit B?
2. ☐ Do Units solve DIFFERENT problems?
3. ☐ Is there a situation where Unit A > Unit B AND a situation where Unit B > Unit A?
4. ☐ Do combinations create NEW capabilities?

If all YES → Orthogonal ✅
If any NO → One-dimensional ❌

---

## CORE CONCEPT #2: Asymmetric Design

**Asymmetric design** means factions/classes play DIFFERENTLY, not just cosmetically.

### Symmetric vs Asymmetric

**Symmetric** (mirror match):
```
Faction A: 10 HP, 5 damage, 3 speed
Faction B: 10 HP, 5 damage, 3 speed
  → Same strategy, no depth
```

**Cosmetically Asymmetric** (reskin):
```
Faction A (Warriors): 10 HP, 5 melee damage, 3 speed
Faction B (Archers): 10 HP, 5 ranged damage, 3 speed
  → Different aesthetics, same strategy, shallow
```

**True Asymmetric**:
```
Faction A (Swarm): 5 HP, 2 damage, cheap, fast production
Faction B (Elite): 20 HP, 10 damage, expensive, slow production
  → Different strategies (mass vs quality), deep
```

### The Asymmetry Spectrum

```
MIRROR MATCH (Chess)
│ Both players same pieces
│ Depth from skill, not faction
│
COSMETIC ASYMMETRY (Many RPGs)
│ Different aesthetics
│ Same mechanics
│ Shallow
│
MECHANICAL ASYMMETRY (StarCraft)
│ Different unit capabilities
│ Different optimal strategies
│ Deep
│
RADICAL ASYMMETRY (Root board game)
│ Different rules, win conditions, turn structure
│ Completely different gameplay
│ Very deep (but hard to balance)
│
```

### Designing Asymmetric Factions

**Step 1: Define Core Identity**

Each faction needs a **core strategic identity**:

```
Example: RTS Factions

Faction A - "The Swarm"
  Identity: Overwhelming numbers, fast production, sacrifice units
  Core mechanic: Units cheap, die easily, but respawn quickly
  Playstyle: Aggressive, map control, attrition

Faction B - "The Fortress"
  Identity: Impenetrable defense, slow methodical advance
  Core mechanic: Units expensive, durable, strong defenses
  Playstyle: Defensive, build up, decisive push

Faction C - "The Nomads"
  Identity: Mobility, hit-and-run, map presence
  Core mechanic: Units mobile, moderate cost, weak defenses
  Playstyle: Harassment, multi-pronged attacks, avoid confrontation
```

**Step 2: Distribute Strengths/Weaknesses Asymmetrically**

Make factions strong on DIFFERENT axes:

| Axis | Swarm | Fortress | Nomads |
|------|-------|----------|--------|
| **Production Speed** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Unit Durability** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Mobility** | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Economy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Burst Damage** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Result**: Each faction has unique strengths. No faction dominates all axes.

**Step 3: Define Faction-Specific Mechanics**

Give each faction UNIQUE mechanics, not shared:

```
Swarm:
  ✓ Spawning Pools: Dead units return as larvae
  ✓ Hivemind: Units share vision
  ✓ Evolution: Units level up through combat

Fortress:
  ✓ Engineering: Repair damaged structures/units
  ✓ Fortifications: Buildable walls and turrets
  ✓ Siege Weapons: Long-range artillery

Nomads:
  ✓ Caravan: Mobile base
  ✓ Ambush: Units hide in terrain
  ✓ Raids: Steal enemy resources
```

**Result**: Factions play DIFFERENTLY, not just stronger/weaker at same mechanics.

### Test: Is Your Design Truly Asymmetric?

Ask:
1. ☐ Does each faction have a UNIQUE core mechanic?
2. ☐ Would an expert player use DIFFERENT strategies for each faction?
3. ☐ Can each faction win through DIFFERENT paths?
4. ☐ Do factions excel at DIFFERENT strategic axes?
5. ☐ Is there NO faction that's "strictly better"?

If all YES → Asymmetric ✅
If any NO → Just reskins ❌

---

## CORE CONCEPT #3: Synergy Matrices

**Synergy** = when combining elements creates MORE value than sum of parts.

### Why Synergies Matter

**Without synergies**:
```
5 Warriors = 5 × 10 damage = 50 damage total
  → Linear scaling, predictable
```

**With synergies**:
```
4 Warriors + 1 Banner Bearer = (4 × 10) + (4 × 10 × 0.5 bonus) = 60 damage
  → Combinatorial scaling, encourages mixed armies
```

### Types of Synergies

**1. Multiplicative Synergies** (buffs/debuffs)
```
Tank (defense) + Healer (sustain) = Tank survives 3× longer
Debuffer (reduce armor) + DPS (damage) = DPS deals 2× damage
```

**2. Enabling Synergies** (unlock capabilities)
```
Scout (vision) + Artillery (long range) = Artillery can fire at max range
Builder (walls) + Archer (ranged) = Archers shoot over walls safely
```

**3. Combo Synergies** (sequential actions)
```
Stun unit → High damage unit = Guaranteed hit
Area slow → Area damage = Enemies can't escape
```

**4. Covering Weaknesses** (complementary pairs)
```
Glass cannon (high damage, low HP) + Tank (low damage, high HP) = Balanced
Melee (short range) + Ranged (long range) = Full coverage
```

### Designing Synergy Matrices

**Step 1: Create Synergy Table**

Map which units synergize:

|  | Warrior | Archer | Healer | Tank | Mage |
|--|---------|--------|--------|------|------|
| **Warrior** | ⭐ (banner) | ⭐⭐⭐ (cover fire) | ⭐⭐⭐⭐ (sustain) | ⭐⭐ (frontline) | ⭐⭐⭐ (AoE support) |
| **Archer** | ⭐⭐⭐ | ⭐ (focus fire) | ⭐⭐ | ⭐⭐⭐⭐ (protected) | ⭐⭐ |
| **Healer** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ (chain heal) | ⭐⭐⭐⭐⭐ (enable tank) | ⭐⭐⭐ |
| **Tank** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Mage** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ (spell combo) |

**Analysis**:
- Tank + Healer = strongest synergy (5 stars)
- Pure armies (all Warrior, all Archer) = weak synergy (1-2 stars)
- **Result**: Mixed armies incentivized

**Step 2: Implement Synergy Mechanics**

```python
# Example: Tank + Healer Synergy
class Tank:
    def take_damage(self, damage):
        if nearby_healer:
            damage *= 0.7  # 30% damage reduction when healer nearby
        self.hp -= damage

class Healer:
    def heal_target(self, target):
        heal_amount = 10
        if target.type == "Tank":
            heal_amount *= 1.5  # 50% bonus healing on tanks
        target.hp += heal_amount
```

**Result**: Tank + Healer combo FEELS strong, incentivizes composition diversity.

**Step 3: Test Synergy Balance**

Ensure:
- ☐ Mixed armies > mono armies (in most situations)
- ☐ Multiple viable combos exist (not just one best combo)
- ☐ Synergies discoverable but not obvious
- ☐ Counter-synergies exist (anti-synergy units to break combos)

### Example: Slay the Spire Synergies

**Deck Archetype Synergies**:

```
STRENGTH BUILD:
  • Inflame (gain strength)
  • Heavy Blade (damage scales with strength)
  • Limit Break (double strength)
  Synergy: Multiplicative scaling

BLOCK BUILD:
  • Barricade (block persists)
  • Entrench (double block)
  • Body Slam (damage = block)
  Synergy: Defense becomes offense

EXHAUST BUILD:
  • Feel No Pain (block when card exhausted)
  • Corruption (skills free, auto-exhaust)
  • Dark Embrace (draw when exhaust)
  Synergy: Convert downside into upside
```

**Result**: 15+ viable deck archetypes, all feeling different. Deep.

### Avoiding Synergy Pitfalls

**Pitfall #1: Mandatory Synergies**
```
❌ Unit A useless without Unit B
✅ Unit A functional alone, stronger with Unit B
```

**Pitfall #2: Synergy Power Creep**
```
❌ Synergy combos so strong, non-synergy unplayable
✅ Synergies competitive with solo strategies
```

**Pitfall #3: Hidden Synergies**
```
❌ Synergies undiscoverable (require wiki)
✅ Synergies hinted through tooltips/descriptions
```

---

## CORE CONCEPT #4: Counter-Play Systems

**Counter-play** = rock-paper-scissors dynamics where strategies beat each other cyclically.

### Why Counter-Play Matters

Without counters:
```
Strategy A > Strategy B in ALL situations
  → Strategy A becomes dominant
  → Game solved
```

With counters:
```
Strategy A > Strategy B
Strategy B > Strategy C
Strategy C > Strategy A
  → No dominant strategy
  → Meta-game emerges
```

### Types of Counter Relationships

**1. Hard Counters** (deterministic)
```
Cavalry > Archers (cavalry charges, archers flee)
Pikes > Cavalry (pikes stop charges, cavalry dies)
Archers > Pikes (archers shoot from range, pikes can't reach)

Win rate: 80-90% for counter
```

**2. Soft Counters** (probabilistic)
```
Tank > DPS (tank absorbs damage, DPS struggles)
DPS > Tank (eventually burns through HP, but risky)
Tank > Healer (long time to kill, but inevitable)

Win rate: 55-65% for counter
```

**3. Situational Counters** (context-dependent)
```
Melee > Ranged (in tight corridors)
Ranged > Melee (in open fields)
AoE > Clumped (when enemies grouped)
Single-target > Spread (when enemies dispersed)

Win rate: Variable based on situation
```

### Designing Counter Systems

**Step 1: Map Counter Relationships**

Create counter triangle (or more complex web):

```
     Cavalry
     /    \\
    /      \\
   v        v
Archers ← Pikes
   \\       ^
    \\     /
     v   /
    Infantry
```

**Step 2: Implement Counter Mechanics**

```python
# Example: Unit Type Counter System
class Unit:
    def calculate_damage(self, target):
        base_damage = self.attack

        # Hard counters
        if self.type == "Cavalry" and target.type == "Archer":
            base_damage *= 2.0  # 2× damage (hard counter)
        elif self.type == "Pike" and target.type == "Cavalry":
            base_damage *= 2.5  # 2.5× damage (hard counter)
        elif self.type == "Archer" and target.type == "Pike":
            base_damage *= 1.5  # 1.5× damage (soft counter)

        # Situational modifiers
        if self.type == "Ranged" and self.is_on_high_ground():
            base_damage *= 1.3  # Height advantage

        return base_damage
```

**Step 3: Test Counter Balance**

Ensure:
- ☐ No unit counters EVERYTHING (no silver bullet)
- ☐ Every unit has at least 1 counter (no invincible unit)
- ☐ Counters are discoverable (tooltips/obvious visual cues)
- ☐ Counters incentivize composition diversity

### Example: Pokémon Type Chart

```
Fire > Grass > Water > Fire (triangle)
Electric > Water, Flying
Ground > Electric, Fire, Rock
Flying > Ground (immunity)
...

Result: 18 types × 18 types = 324 counter relationships
Depth: Every team needs type coverage
```

### Counter-Play in Non-Combat Games

**Example: Civilization Victory Conditions**

```
Domination (military) > Science (slow build-up)
Science (tech advantage) > Culture (can't defend)
Culture (tourism pressure) > Domination (cultural conversion)
Diplomacy (city-states) > Culture (votes block)

Result: No single victory path dominates
```

### Avoiding Counter-Play Pitfalls

**Pitfall #1: Counter-Pick Meta**
```
❌ Losing at team select (pre-determined by picks)
✅ Skill expression within counter matchups
```

**Pitfall #2: Dead Matchups**
```
❌ Counter so hard, countered player can't win
✅ Countered player can outplay (70/30, not 95/5)
```

**Pitfall #3: Circular Rock-Paper-Scissors**
```
❌ Only one counter triangle, predictable
✅ Multi-dimensional counters (terrain, timing, economy also matter)
```

---

## DECISION FRAMEWORK #1: Linear vs Branching Progression

### The Linear Trap

**Linear Progression**:
```
Tier 1 Unit → Tier 2 Unit → Tier 3 Unit
10 HP, 5 dmg   20 HP, 10 dmg  30 HP, 15 dmg

Problem: Tier 3 strictly better → no build diversity
```

**Result**: Everyone rushes Tier 3. Path predetermined.

### Branching Progression

**Horizontal Branches** (specializations):
```
       Tier 2A (Damage Focus)
      /  30 HP, 20 dmg, slow
Tier 1
      \\  Tier 2B (Mobility Focus)
       60 HP, 10 dmg, fast

Trade-off: Power vs Speed
Result: Situational optimality
```

**Vertical + Horizontal**:
```
           Berserker (glass cannon)
          /  60 HP, 30 dmg
Tier 1 → Tier 2 → Tier 3
(Basic)  (Fighter)  \\
                     Champion (balanced)
                      120 HP, 20 dmg
```

### Designing Branching Trees

**Step 1: Identify Branch Points**

Every 2-3 tiers, offer meaningful choice:

```
Start
  │
  ├─ Economic Branch (fast economy, weak military)
  │   ├─ Trading (gold focus)
  │   └─ Production (build speed focus)
  │
  ├─ Military Branch (strong units, slow economy)
  │   ├─ Offensive (damage focus)
  │   └─ Defensive (HP focus)
  │
  └─ Tech Branch (advanced units, high cost)
      ├─ Air Units (mobility)
      └─ Naval Units (map control)
```

**Step 2: Ensure Trade-offs**

Each branch must have COST:

```
Economic Branch:
  ✓ Strength: Fast economy, more resources
  ✗ Weakness: Weak military early, vulnerable

Military Branch:
  ✓ Strength: Strong units, early aggression
  ✗ Weakness: Slow economy, fewer resources

Tech Branch:
  ✓ Strength: Advanced units, late game power
  ✗ Weakness: Expensive, slow to scale
```

**Result**: No "correct" choice. Situational optimality.

**Step 3: Test Build Diversity**

Track player choices:
- ☐ Are all branches chosen roughly equally? (30/30/40 is OK, 10/10/80 is bad)
- ☐ Do different branches win? (all should have >40% win rate)
- ☐ Can branches adapt to counters? (flexibility within branch)

### Example: Path of Exile Passive Tree

```
1400+ passive nodes
~123 points to allocate
Multiple viable starting positions
Thousands of possible builds

Result: Extreme build diversity, no dominant path
```

### When to Use Linear Progression

Linear is OK when:
- ✅ Game depth comes from other sources (e.g., Chess has linear piece values but deep gameplay)
- ✅ Players choose WHEN to progress (timing strategy)
- ✅ Resources constrained (can't have everything)

Linear is BAD when:
- ❌ Progression is only source of depth
- ❌ No resource constraints (everyone gets everything)
- ❌ Higher tier always optimal

---

## DECISION FRAMEWORK #2: Symmetric vs Asymmetric Depth

### When to Use Symmetric Design

**Use symmetric when**:
- ✅ Competitive purity important (e-sports, tournaments)
- ✅ Skill expression from execution, not matchup knowledge
- ✅ Easy to balance (mirror matches)
- ✅ Examples: Chess, Go, poker

**Symmetric Depth Sources**:
- Execution skill (mechanics, reflexes)
- Tactical knowledge (openings, gambits)
- Psychological play (reads, bluffs)
- Positioning and timing

### When to Use Asymmetric Design

**Use asymmetric when**:
- ✅ Replayability important (learn multiple factions)
- ✅ Strategic variety desired (different playstyles)
- ✅ Emergent meta-game valued
- ✅ Examples: StarCraft, MOBAs, fighting games

**Asymmetric Depth Sources**:
- Matchup knowledge (counter-play)
- Faction mastery (unique mechanics)
- Composition building (synergies)
- Adaptation (scouting, reads)

### Hybrid Approach

**Example: Magic: The Gathering**
```
Symmetric: Both players use same rules
Asymmetric: Players build different decks

Result: Deep matchup meta, but symmetric rules prevent imbalance
```

### Decision Matrix

| Goal | Symmetric | Asymmetric |
|------|-----------|------------|
| **Easy to learn** | ✅ (one playstyle) | ❌ (multiple playstyles) |
| **Easy to balance** | ✅ (mirror) | ❌ (complex interactions) |
| **High replayability** | ❌ (repetitive) | ✅ (variety) |
| **Deep meta-game** | ⚠️ (possible but hard) | ✅ (natural) |
| **Tournament ready** | ✅ (fair) | ⚠️ (if balanced) |

---

## DECISION FRAMEWORK #3: Cognitive Load Management

### The Complexity Paradox

```
Too Simple: Boring, solved quickly, no depth
  Example: Tic-Tac-Toe

Sweet Spot: Complex enough for depth, simple enough to learn
  Example: Chess, StarCraft

Too Complex: Overwhelming, analysis paralysis, frustrating
  Example: Dwarf Fortress (for many players)
```

### Measuring Cognitive Load

**Factors**:
1. **Decision Count**: How many choices per turn?
2. **Decision Complexity**: How hard to evaluate each choice?
3. **State Space**: How much must player track?
4. **Time Pressure**: How fast must player decide?

**Formula**:
```
Cognitive Load = (Decisions × Complexity × State Space) / Time Available

Target: Keep load under player's capacity
```

### Managing Complexity

**Technique #1: Progressive Disclosure**

Start simple, add complexity over time:

```
Tutorial:
  • Show 1 unit type
  • Teach basic attack

Early Game:
  • Introduce 3 unit types
  • Teach rock-paper-scissors

Mid Game:
  • Introduce synergies
  • Teach combos

Late Game:
  • All mechanics available
  • Player mastery expected
```

**Technique #2: Chunking**

Group related mechanics:

```
❌ 15 individual unit stats (overwhelming)
✅ 3 unit roles: Tank, DPS, Support (manageable)
```

**Technique #3: Automation**

Let system handle micro, player handles macro:

```
Low-level: Auto-attack, auto-move, auto-target
Mid-level: Unit production queues, auto-rally
High-level: Strategic decisions (composition, positioning)

Player focuses on meaningful choices
```

**Technique #4: Information Hierarchy**

Present critical info first:

```
PRIORITY 1: Health, damage (core stats)
PRIORITY 2: Armor, abilities (important but secondary)
PRIORITY 3: Lore, flavor (optional)

Don't bury critical info in walls of text
```

### Test: Is Complexity Justified?

For each mechanic, ask:
1. ☐ Does this add strategic depth? (meaningful choices)
2. ☐ Is this depth worth the cognitive cost? (ROI)
3. ☐ Can this be simplified without losing depth?
4. ☐ Is this discoverable? (can players learn it?)

If any NO → Simplify or cut

### Example: League of Legends Champion Design

```
Early Champions (Annie):
  • Simple kit (4 abilities)
  • Clear role (burst mage)
  • Low skill floor, low skill ceiling

Later Champions (Azir):
  • Complex kit (sand soldiers)
  • Unique role (ranged zone control)
  • High skill floor, extreme skill ceiling

Both viable: Simple for new players, complex for mastery
```

---

## IMPLEMENTATION PATTERN #1: Rock-Paper-Scissors Foundation

### Basic Triangle

```python
class UnitType(Enum):
    WARRIOR = "warrior"  # Beats ARCHER
    ARCHER = "archer"    # Beats MAGE
    MAGE = "mage"        # Beats WARRIOR

class Unit:
    def __init__(self, type: UnitType):
        self.type = type
        self.base_damage = 10

    def calculate_damage(self, target: 'Unit') -> float:
        damage = self.base_damage

        # Counter relationships
        if (self.type == UnitType.WARRIOR and target.type == UnitType.ARCHER) or \\
           (self.type == UnitType.ARCHER and target.type == UnitType.MAGE) or \\
           (self.type == UnitType.MAGE and target.type == UnitType.WARRIOR):
            damage *= 1.5  # 50% bonus vs counter

        return damage
```

### Extended Web

```python
class AdvancedUnit:
    # Multi-dimensional counters
    COUNTER_MATRIX = {
        "cavalry": {"archer": 2.0, "infantry": 1.5, "pike": 0.5},
        "archer": {"pike": 1.5, "cavalry": 0.5, "infantry": 1.0},
        "pike": {"cavalry": 2.5, "infantry": 1.0, "archer": 0.7},
        "infantry": {"archer": 1.2, "pike": 1.2, "cavalry": 0.8},
    }

    def calculate_damage(self, target):
        multiplier = self.COUNTER_MATRIX[self.type].get(target.type, 1.0)
        return self.base_damage * multiplier
```

---

## IMPLEMENTATION PATTERN #2: Synergy Buff Systems

### Aura Buffs

```python
class AuraUnit:
    def __init__(self):
        self.aura_radius = 5.0
        self.aura_bonus_damage = 0.2  # +20% damage

    def update(self):
        # Find nearby allies
        nearby = find_units_in_radius(self.position, self.aura_radius)

        for ally in nearby:
            if ally != self:
                ally.add_buff("damage_bonus", self.aura_bonus_damage)

class Unit:
    def __init__(self):
        self.buffs = {}

    def add_buff(self, buff_type, value):
        self.buffs[buff_type] = value

    def calculate_damage(self):
        damage = self.base_damage
        if "damage_bonus" in self.buffs:
            damage *= (1 + self.buffs["damage_bonus"])
        return damage
```

### Tag-Based Synergies

```python
class Card:
    def __init__(self, name, tags):
        self.name = name
        self.tags = tags  # ["elemental", "fire", "summon"]

    def calculate_power(self, deck):
        power = self.base_power

        # Count synergies
        elemental_count = sum(1 for c in deck if "elemental" in c.tags)
        if "elemental" in self.tags:
            power += elemental_count * 2  # +2 power per elemental

        return power
```

---

## IMPLEMENTATION PATTERN #3: Tech Tree with Branches

### Tree Structure

```python
class TechNode:
    def __init__(self, name, cost, prerequisites):
        self.name = name
        self.cost = cost
        self.prerequisites = prerequisites  # List of required techs
        self.unlocks = []  # Units/buildings this enables

class TechTree:
    def __init__(self):
        self.researched = set()
        self.available = set()

        # Define tree
        self.nodes = {
            "mining": TechNode("Mining", cost=100, prerequisites=[]),
            "smithing": TechNode("Smithing", cost=200, prerequisites=["mining"]),
            "steel": TechNode("Steel", cost=300, prerequisites=["smithing"]),
            "gunpowder": TechNode("Gunpowder", cost=300, prerequisites=["smithing"]),
            # Branches here: steel OR gunpowder
        }

    def can_research(self, tech_name):
        node = self.nodes[tech_name]
        return all(prereq in self.researched for prereq in node.prerequisites)

    def research(self, tech_name):
        if self.can_research(tech_name):
            self.researched.add(tech_name)
            self.update_available()
```

---

## IMPLEMENTATION PATTERN #4: Faction Asymmetry Through Unique Mechanics

### Resource System Variation

```python
class Faction:
    def __init__(self, name):
        self.name = name
        self.resources = {}

class SwarmFaction(Faction):
    # Unique mechanic: Biomass (dead units become resource)
    def __init__(self):
        super().__init__("Swarm")
        self.biomass = 0

    def on_unit_death(self, unit):
        self.biomass += unit.hp_max * 0.5  # Convert HP to biomass

    def spawn_unit(self, unit_type):
        cost = unit_type.biomass_cost
        if self.biomass >= cost:
            self.biomass -= cost
            return unit_type.create()

class FortressFaction(Faction):
    # Unique mechanic: Scrap (repair units)
    def __init__(self):
        super().__init__("Fortress")
        self.scrap = 0

    def repair_unit(self, unit):
        repair_cost = (unit.hp_max - unit.hp) * 0.3
        if self.scrap >= repair_cost:
            self.scrap -= repair_cost
            unit.hp = unit.hp_max
```

---

## IMPLEMENTATION PATTERN #5: Build Diversity Through Mutually Exclusive Choices

### Talent System

```python
class Character:
    def __init__(self):
        self.talent_points = 0
        self.talents = {}

    def choose_talent(self, tree, talent):
        # Mutually exclusive: choosing tree A locks tree B
        if tree == "offensive":
            self.talents["offensive"] = talent
            # Can't choose defensive now
        elif tree == "defensive":
            self.talents["defensive"] = talent
            # Can't choose offensive now

# Example talents
TALENTS = {
    "offensive": {
        "berserker": {"damage": +50%, "defense": -20%},
        "assassin": {"crit": +30%, "hp": -10%},
    },
    "defensive": {
        "tank": {"hp": +100%, "speed": -30%},
        "evasion": {"dodge": +40%, "armor": -50%},
    },
}
```

---

## COMMON PITFALL #1: Dominant Strategy Emergence

### The Mistake

Balancing numbers without testing strategy space:

```
Unit A: 10 HP, 5 damage, 10 cost
Unit B: 8 HP, 6 damage, 10 cost
Unit C: 12 HP, 4 damage, 10 cost

Seems balanced... but:
Unit B has highest DPS per cost → dominant strategy
```

### Why It Happens

- Playtested insufficiently
- Didn't simulate optimal play
- Balanced stats, not strategies

### The Fix

**Test for dominant strategies**:

```python
def test_strategies():
    strategies = [
        "all_unit_a",
        "all_unit_b",
        "all_unit_c",
        "mixed_a_b",
        "mixed_b_c",
        "mixed_a_c",
    ]

    win_rates = {}
    for strat1 in strategies:
        for strat2 in strategies:
            if strat1 != strat2:
                wins = simulate_matches(strat1, strat2, n=1000)
                win_rates[(strat1, strat2)] = wins / 1000

    # Check for dominant strategy
    for strat in strategies:
        avg_win_rate = average([win_rates[(strat, other)]
                                for other in strategies if other != strat])
        if avg_win_rate > 0.65:  # Wins >65% of matchups
            print(f"DOMINANT STRATEGY: {strat}")
```

### Prevention

✅ Simulate optimal play (AI vs AI)
✅ Test all matchups, not just anecdotal
✅ Track win rates by strategy
✅ Nerf dominant, buff underused

---

## COMMON PITFALL #2: False Choices (Illusion of Depth)

### The Mistake

Offering choices that don't matter:

```
Weapon A: 10 damage, 2 speed
Weapon B: 20 damage, 1 speed
Weapon C: 5 damage, 4 speed

DPS: A=20, B=20, C=20 (all identical!)
→ Cosmetic choice, no strategic depth
```

### Why It Happens

- Balanced for "fairness" without considering choice meaningfulness
- Wanted equal power, accidentally made equivalent

### The Fix

**Make choices FEEL different**:

```
Weapon A: 10 damage, 2 speed (balanced, reliable)
Weapon B: 25 damage, 1 speed (high risk/reward, situational)
Weapon C: 4 damage, 5 speed (low burst, high sustained)

Situational optimality:
  • Weapon A: General purpose, always decent
  • Weapon B: Boss fights (high HP enemies)
  • Weapon C: Swarms (many low HP enemies)
```

### Prevention

✅ Ensure each choice has UNIQUE best-case scenario
✅ Avoid perfectly balanced equivalence
✅ Create situational optimality, not universal optimality

---

## COMMON PITFALL #3: Complexity Creep (Adding Without Depth)

### The Mistake

Adding mechanics that don't create strategic choices:

```
Base game: 3 unit types, rock-paper-scissors depth

Expansion adds:
  • 10 more unit types... but all fit same 3 categories
  • Just more content, not more strategy
```

### Why It Happens

- Pressure to add content (DLC, sequels)
- Mistaking quantity for quality
- No depth analysis before adding

### The Fix

**Ask before adding**: Does this create NEW strategies?

```
Example:
  • Adding Unit D (another warrior) → NO new strategies
  • Adding Flying units (ignore terrain) → YES new strategies (air control layer)
```

### Prevention

✅ New mechanics must open new strategic axes
✅ More ≠ better, only add if depth increases
✅ Remove redundant mechanics

---

## COMMON PITFALL #4: Over-Specialization (Rigid Meta)

### The Mistake

Factions so specialized, they're one-dimensional:

```
Faction A: Only has melee units (no ranged)
Faction B: Only has ranged units (no melee)

Problem: Matchups predetermined by maps
  • Tight corridors: A wins always
  • Open fields: B wins always
```

### Why It Happens

- Overcommitting to asymmetry
- No flexibility within factions

### The Fix

**Asymmetry in focus, not exclusivity**:

```
Faction A: MOSTLY melee (80%), some ranged (20%)
Faction B: MOSTLY ranged (80%), some melee (20%)

Result: A favors melee, but can adapt. B favors ranged, but can adapt.
```

### Prevention

✅ Each faction should have SOME access to each strategic axis
✅ Specialization, not exclusivity
✅ Adaptation possible, but faction identity maintained

---

## COMMON PITFALL #5: No Discovery Phase (Solved at Launch)

### The Mistake

Game depth exhausted immediately:

```
All strategies obvious
No hidden synergies
No emergent combos
Meta solved day 1
```

### Why It Happens

- Everything explained in tutorial
- No room for experimentation
- Mechanics too simple

### The Fix

**Design for discovery**:

```
Layer 1 (Obvious): Rock-paper-scissors counters
Layer 2 (Discoverable): Synergy combos
Layer 3 (Hidden): Advanced techniques
Layer 4 (Emergent): Player-discovered exploits (balance later if broken)

Release with layers 1-2 explained, 3-4 for players to discover
```

### Prevention

✅ Don't explain everything in tutorial
✅ Leave room for experimentation
✅ Playtest with fresh players (not devs)
✅ Track strategy evolution over time

---

## REAL-WORLD EXAMPLE #1: StarCraft Brood War

**Challenge**: Create 3 asymmetric factions with deep strategy.

**Solution**: Orthogonal mechanics + asymmetric strengths

### Faction Design

**Terran**:
- Identity: Defensive, flexible, positioning-focused
- Unique mechanics: Bunkers (defensive structures), SCVs repair, scan (detection)
- Strengths: Defense, detection, mid-game timing
- Weaknesses: Immobile, supply blocked easily

**Zerg**:
- Identity: Economic, swarming, map control
- Unique mechanics: Larvae (shared production), creep (vision + speed), burrow
- Strengths: Economy, unit count, harassment
- Weaknesses: Fragile units, relies on momentum

**Protoss**:
- Identity: Powerful units, tech-focused, defensive
- Unique mechanics: Shields (regenerate), warp-in, psionic storm
- Strengths: Unit quality, late-game, splash damage
- Weaknesses: Expensive, vulnerable early

### Why It Works

1. **Orthogonal axes**: Each race strong at different things
2. **Asymmetric mechanics**: Races play differently (not reskins)
3. **Build diversity**: Within each race, multiple viable build orders
4. **Counter-play**: Each race has strategies that beat others
5. **Discovery**: 20+ years, still finding new strategies

**Lesson**: Asymmetry + orthogonality = eternal depth.

---

## REAL-WORLD EXAMPLE #2: Slay the Spire

**Challenge**: Roguelike deckbuilder with build diversity (avoid dominant decks).

**Solution**: Synergy matrices + branching choices

### Deck Archetype Examples

**Ironclad (Warrior)**:
- Strength Scaling: Stack strength stat, deal massive damage
- Block Scaling: Stack block, convert to damage (Body Slam)
- Exhaust Synergy: Exhaust cards for benefits (Feel No Pain)
- Barricade: Block persists between turns

**Silent (Rogue)**:
- Poison: Stack poison, wait for DoT
- Shivs: Generate 0-cost attacks, attack many times
- Discard: Discard cards for benefits
- Card draw: Thin deck, draw entire deck per turn

**Result**: 10+ viable archetypes per character, build diversity extreme.

### Why It Works

1. **Synergy discovery**: Players discover combos through play
2. **Branching choices**: Every card offer creates branching paths
3. **No dominant strategy**: Situational optimality (enemy type matters)
4. **Emergent combos**: New synergies discovered years after launch

**Lesson**: Synergy matrices create combinatorial depth.

---

## REAL-WORLD EXAMPLE #3: Path of Exile Skill Tree

**Challenge**: Create character build diversity in ARPG (avoid dominant builds).

**Solution**: Massive branching skill tree + trade-offs

### Design

- 1400+ passive skill nodes
- ~123 skill points to allocate
- Multiple starting positions (7 classes)
- Mutually exclusive paths (can't reach all areas)

**Example Builds**:
- Life-based melee tank (HP nodes, armor nodes)
- Energy shield caster (ES nodes, spell damage)
- Critical strike assassin (crit nodes, evasion)
- Minion summoner (minion nodes, auras)
- Totem/trap builds (proxy damage)

**Result**: Thousands of viable builds, no dominant path.

### Why It Works

1. **Massive branching**: Every point allocation is a choice
2. **Opportunity cost**: Choosing path A means NOT choosing path B
3. **Specialization required**: Can't have everything, must focus
4. **Trade-offs**: Damage vs defense, offense vs utility

**Lesson**: Branching + opportunity cost = build diversity.

---

## REAL-WORLD EXAMPLE #4: Magic: The Gathering

**Challenge**: Card game with extreme strategy diversity (avoid dominant decks).

**Solution**: Counter-play systems + synergy matrices + asymmetric deck building

### Meta-Game Depth

**Deck Archetypes**:
- Aggro: Fast damage, win before opponent stabilizes
- Control: Slow game, win through inevitability
- Combo: Assemble pieces, win with synergy
- Midrange: Balanced, adapt to opponent

**Counter Relationships**:
- Aggro > Combo (kill before combo assembles)
- Combo > Control (control can't stop combo)
- Control > Aggro (remove threats, stabilize)
- Midrange > varies (adapt based on matchup)

**Result**: No dominant deck, meta evolves constantly.

### Why It Works

1. **Counter-play**: Every deck has bad matchups
2. **Sideboard tech**: Adapt deck between games
3. **Meta-game**: Players adapt to popular decks
4. **New cards**: Constant injection of new synergies

**Lesson**: Counter-play + meta-game evolution = eternal depth.

---

## REAL-WORLD EXAMPLE #5: Civilization VI

**Challenge**: 4X strategy with multiple victory conditions (avoid dominant strategy).

**Solution**: Orthogonal victory paths + asymmetric civilizations

### Victory Conditions

1. **Domination**: Conquer all capitals (military axis)
2. **Science**: Launch spaceship (tech axis)
3. **Culture**: Attract tourists (culture axis)
4. **Religion**: Convert all civilizations (faith axis)
5. **Diplomacy**: Win World Congress votes (diplomacy axis)

**Civilization Asymmetry**:
- Rome: Strong production (good for Domination/Science)
- Greece: Culture bonuses (good for Culture victory)
- Arabia: Faith bonuses (good for Religion victory)
- Korea: Science bonuses (good for Science victory)

### Why It Works

1. **Orthogonal paths**: Each victory addresses different strategic axis
2. **Asymmetric civs**: Each civ favors different victory types
3. **Counter-play**: Military can disrupt Science, Culture can flip cities
4. **Situational optimality**: Map/opponents determine best path

**Lesson**: Multiple victory conditions create strategic diversity.

---

## CROSS-REFERENCE: Related Skills

### Within systems-as-experience Skillpack

1. **emergent-gameplay-design**: How orthogonal mechanics create emergence (this skill teaches WHAT mechanics to make orthogonal)
2. **game-balance**: How to balance asymmetric systems (this skill teaches WHAT to balance)
3. **player-driven-narratives**: How player choices create stories (this skill teaches how to create meaningful choices)

### From Other Skillpacks

1. **simulation-tactics/economic-simulation-patterns**: Economy as strategic axis (synergies with economic depth here)
2. **procedural-generation**: Generating strategic variety (complements build diversity)
3. **difficulty-curves**: Maintaining challenge across strategies (all paths should be challenging)

---

## TESTING CHECKLIST

Before shipping strategy system:

### Depth Validation

- ☐ **Viable strategy count**: Are there 5+ strategies that can win?
- ☐ **Strategy diversity**: Do top players use different strategies?
- ☐ **Build variety**: Within each faction/class, are there 3+ viable builds?
- ☐ **No dominant strategy**: Does any single approach win >65% of matchups?

### Orthogonality Validation

- ☐ **Multiple axes**: Are there 3+ independent strategic axes?
- ☐ **Axis distribution**: Do factions/units excel at DIFFERENT axes?
- ☐ **Situational optimality**: Is there no universally best unit/choice?
- ☐ **Combination depth**: Do unit combinations create new capabilities?

### Asymmetry Validation

- ☐ **Play differently**: Do factions/classes FEEL different to play?
- ☐ **Different strategies**: Do factions require different optimal strategies?
- ☐ **Unique mechanics**: Does each faction have unique mechanics (not shared)?
- ☐ **Adaptation possible**: Can each faction adapt to different situations?

### Counter-Play Validation

- ☐ **Counters exist**: Does every strategy have at least 1 counter?
- ☐ **No silver bullet**: Does any strategy counter EVERYTHING?
- ☐ **Skill expression**: Can skilled players overcome counter matchups?
- ☐ **Meta-game**: Do strategies evolve in response to popularity?

### Complexity Validation

- ☐ **Complexity justified**: Does complexity add depth (not just confusion)?
- ☐ **Learnable**: Can new players understand core mechanics in <1 hour?
- ☐ **Progressive disclosure**: Do mechanics unlock gradually?
- ☐ **Information hierarchy**: Is critical info surfaced, optional info hidden?

### Discovery Validation

- ☐ **Not solved**: Are players still discovering new strategies?
- ☐ **Emergent synergies**: Are there combos players discovered (not dev-intended)?
- ☐ **Meta evolution**: Has meta changed over time?
- ☐ **Room for mastery**: Do experts play differently than novices?

---

## SUMMARY: The Strategic Depth Framework

### Step-by-Step Process

**1. Identify Strategic Axes** (Orthogonal Mechanics)
   - What are the 3-5 core strategic problems players must solve?
   - Offense, defense, mobility, economy, utility, information, tempo
   - Ensure axes are independent (not all scaling damage)

**2. Distribute Across Axes** (Asymmetric Design)
   - Make factions/classes strong at DIFFERENT axes
   - No faction should dominate all axes
   - Create specialization, not exclusivity

**3. Create Synergies** (Combination Depth)
   - Design mechanics that combo/amplify each other
   - Ensure mixed strategies > mono strategies
   - Multiple viable synergy combinations

**4. Implement Counters** (Rock-Paper-Scissors)
   - Every strategy must have at least 1 counter
   - No strategy should counter EVERYTHING
   - Create cyclical counter relationships (no linear dominance)

**5. Branch Progression** (Build Diversity)
   - Offer meaningful choices every 2-3 tiers
   - Ensure choices have trade-offs (opportunity cost)
   - Avoid strictly-better upgrades (no linear power creep)

**6. Test for Dominant Strategies**
   - Simulate optimal play (AI vs AI, 1000+ games)
   - Track win rates by strategy
   - Nerf strategies with >65% win rate, buff <40%

**7. Manage Complexity** (Cognitive Load)
   - Add mechanics ONLY if they create new strategies
   - Remove redundant mechanics
   - Progressive disclosure (tutorial → mastery)

**8. Enable Discovery** (Meta-Game)
   - Don't explain everything (leave room for experimentation)
   - Design emergent synergies (unintended but balanced)
   - Track strategy evolution over time

### The Golden Rules

> **Rule 1**: Depth comes from viable strategies, not number of options.

> **Rule 2**: Orthogonal mechanics prevent dominant strategies.

> **Rule 3**: Asymmetry creates replayability, not just cosmetic variety.

> **Rule 4**: Synergies reward creativity, counters reward adaptation.

> **Rule 5**: Branching creates choice, trade-offs make choices meaningful.

Apply these frameworks rigorously, and your game will have strategic depth that lasts for years, not days.

---

## END OF SKILL

This skill should be used at the START of any strategy game design. It prevents:
1. Dominant strategy emergence (game solved)
2. False choices (illusion of depth)
3. Symmetric designs (reskin problem)
4. Linear progression (no build diversity)
5. Complexity without depth (cognitive load waste)

Master this framework, and you'll create games with emergent, evolving, eternal strategic depth.
