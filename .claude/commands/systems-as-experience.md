
# Using Systems-as-Experience (Meta-Skill Router)

**Your entry point to the systems-as-experience skillpack.** This skill routes you to the right combination of specialized skills for your game design challenge.

## Purpose

This is a **meta-skill** that:
1. ✅ **Routes** you to the correct skills in this pack
2. ✅ **Combines** multiple skills for complex projects
3. ✅ **Provides** quick-start workflows for common game types
4. ✅ **Explains** the pack's philosophy and when to use it

**You should use this skill:** When starting any systems-driven game design project.


## Core Philosophy: Systems AS Experience

### The Central Idea

Traditional game design: **Systems support content**
- Design story → build mechanics to express it
- Design levels → build systems to populate them
- Content is king, systems are servants

Systems-as-experience: **Systems ARE the content**
- Design orthogonal mechanics → emergence creates "content"
- Design constraints → player creativity becomes gameplay
- Design for discovery → hidden depth rewards curiosity
- Systems are king, authored content is optional

### When This Philosophy Applies

**✅ Use systems-as-experience when:**
- Core loop is experimentation, creativity, or optimization
- Replayability through emergent outcomes is desired
- Player agency and expression are paramount
- Systems have inherent depth worth discovering
- Community-driven content extends game life

**❌ Don't use systems-as-experience when:**
- Authored narrative is primary experience
- Linear progression with fixed setpieces
- Cinematic, controlled emotional arcs
- Hand-crafted, bespoke content is the appeal

**Hybrid approach:** Most games blend both (BotW has authored Ganon encounter + emergent physics sandbox)


## Pack Overview: 8 Core Skills

### Wave 1: Foundation (Core Concepts)

#### 1. emergent-gameplay-design (FOUNDATIONAL ⭐)
**When to use:** Every systems-driven game starts here
**Teaches:** Orthogonal mechanics, interaction matrices, emergence from simple rules
**Examples:** BotW physics, Dwarf Fortress simulation, Minecraft crafting
**Time:** 3-4 hours to apply
**Use when:** Designing core systems that interact

#### 2. sandbox-design-patterns
**When to use:** Open-ended creativity games (building, creation, simulation)
**Teaches:** Constraint paradox, progressive complexity, meaningful limitation
**Examples:** Minecraft, Factorio, Kerbal Space Program
**Time:** 2-3 hours to apply
**Use when:** Players create within constraints

#### 3. strategic-depth-from-systems
**When to use:** Competitive games, build diversity, counter-play
**Teaches:** Orthogonal strategic axes, synergy matrices, avoiding dominant strategies
**Examples:** Path of Exile, StarCraft, Dota, deck builders
**Time:** 2-3 hours to apply
**Use when:** Players optimize builds/strategies

### Wave 2: Specific Applications

#### 4. optimization-as-play
**When to use:** Factory games, efficiency puzzles, production chains
**Teaches:** Bottleneck gameplay, multiple optimization dimensions, satisfying feedback
**Examples:** Factorio, Satisfactory, Opus Magnum, SpaceChem
**Time:** 2-4 hours to apply
**Use when:** Optimization IS the core loop

#### 5. discovery-through-experimentation
**When to use:** Exploration games, hidden depth, knowledge-based progression
**Teaches:** Environmental hints, safe experimentation, secrets, community discovery
**Examples:** BotW, Outer Wilds, Noita, The Witness, fighting game tech
**Time:** 2-4 hours to apply
**Use when:** Curiosity drives engagement

#### 6. player-driven-narratives
**When to use:** Simulation games, emergent storytelling, procedural characters
**Teaches:** Systemic drama, relationships, AI storyteller, memorable moments
**Examples:** Dwarf Fortress, Rimworld, Crusader Kings, EVE Online
**Time:** 2-4 hours to apply
**Use when:** Systems generate stories

### Wave 3: Ecosystem

#### 7. modding-and-extensibility
**When to use:** Games designed for long-term community content creation
**Teaches:** Plugin architecture, mod APIs, tools, security, community infrastructure
**Examples:** Skyrim, Minecraft, Factorio, Warcraft 3
**Time:** 2-3 hours to apply
**Use when:** Players extend game beyond initial design

#### 8. community-meta-gaming
**When to use:** Competitive games, theorycrafting communities, speedrunning
**Teaches:** Build diversity, information transparency, competitive tools, persistent worlds
**Examples:** Path of Exile, EVE Online, Dark Souls, Destiny raids
**Time:** 2-3 hours to apply
**Use when:** Community creates external meta-game


## Routing Logic: Which Skills Do I Need?

### Decision Tree

```
START: What type of game are you building?

├─ SANDBOX / CREATIVITY GAME
│  ├─ Players build/create freely?
│  │  └─> emergent-gameplay-design + sandbox-design-patterns
│  ├─ Optimization is core loop?
│  │  └─> + optimization-as-play
│  ├─ Discovery-driven?
│  │  └─> + discovery-through-experimentation
│  └─ Mod support?
│     └─> + modding-and-extensibility
│
├─ COMPETITIVE / STRATEGIC GAME
│  ├─ Build diversity critical?
│  │  └─> emergent-gameplay-design + strategic-depth-from-systems
│  ├─ Competitive scene planned?
│  │  └─> + community-meta-gaming
│  ├─ Theorycrafting encouraged?
│  │  └─> + community-meta-gaming (again)
│  └─ Mod support for custom games?
│     └─> + modding-and-extensibility
│
├─ SIMULATION / MANAGEMENT GAME
│  ├─ Emergent stories important?
│  │  └─> emergent-gameplay-design + player-driven-narratives
│  ├─ Optimization gameplay?
│  │  └─> + optimization-as-play
│  ├─ Discovery-driven depth?
│  │  └─> + discovery-through-experimentation
│  └─ Mod support?
│     └─> + modding-and-extensibility
│
├─ EXPLORATION / DISCOVERY GAME
│  ├─ Physics/systems interactions?
│  │  └─> emergent-gameplay-design + discovery-through-experimentation
│  ├─ Sandbox elements?
│  │  └─> + sandbox-design-patterns
│  ├─ Speedrun community likely?
│  │  └─> + community-meta-gaming
│  └─ Secrets and depth?
│     └─> discovery-through-experimentation (primary)
│
└─ MULTIPLAYER SOCIAL GAME
   ├─ Player-driven politics/drama?
   │  └─> player-driven-narratives + community-meta-gaming
   ├─ Persistent world consequences?
   │  └─> player-driven-narratives (primary)
   ├─ User-generated content?
   │  └─> modding-and-extensibility
   └─ Emergent social dynamics?
      └─> emergent-gameplay-design + player-driven-narratives
```

### Quick Reference Table

| Game Type | Primary Skill | Secondary Skills | Examples |
|-----------|---------------|------------------|----------|
| **Sandbox Builder** | sandbox-design-patterns | emergent, optimization, discovery | Minecraft, Terraria |
| **Factory Game** | optimization-as-play | emergent, sandbox | Factorio, Satisfactory |
| **Physics Sandbox** | emergent-gameplay-design | discovery, sandbox | BotW, Noita |
| **Colony Sim** | player-driven-narratives | emergent, optimization | Rimworld, DF |
| **Competitive Strategy** | strategic-depth-from-systems | emergent, meta-gaming | PoE, Dota |
| **Exploration Game** | discovery-through-experimentation | emergent, sandbox | Outer Wilds, Subnautica |
| **MMO Sandbox** | player-driven-narratives | meta-gaming, modding | EVE Online |
| **Speedrun-Friendly** | discovery-through-experimentation | meta-gaming | Dark Souls, Celeste |


## 20+ Scenarios: Which Skills Apply?

### Scenario 1: "I'm building Minecraft-like voxel sandbox"
**Primary:** sandbox-design-patterns (progressive complexity, constraints)
**Secondary:** emergent-gameplay-design (block interactions), discovery-through-experimentation (crafting recipes)
**Optional:** modding-and-extensibility (Java mods, behavior packs)
**Estimated time:** 6-10 hours

### Scenario 2: "I'm building Factorio-style factory game"
**Primary:** optimization-as-play (bottleneck gameplay, throughput metrics)
**Secondary:** sandbox-design-patterns (constraints drive creativity), emergent-gameplay-design (production chains)
**Optional:** modding-and-extensibility (mod API like Factorio)
**Estimated time:** 6-8 hours

### Scenario 3: "I'm building BotW-style physics playground"
**Primary:** emergent-gameplay-design (orthogonal physics systems)
**Secondary:** discovery-through-experimentation (environmental hints), sandbox-design-patterns (open world)
**Optional:** community-meta-gaming (speedrun support)
**Estimated time:** 7-10 hours

### Scenario 4: "I'm building Rimworld-style colony sim"
**Primary:** player-driven-narratives (AI storyteller, emergent drama)
**Secondary:** emergent-gameplay-design (system interactions), sandbox-design-patterns (base building)
**Optional:** modding-and-extensibility (Rimworld has extensive mods)
**Estimated time:** 6-10 hours

### Scenario 5: "I'm building Path of Exile-style ARPG"
**Primary:** strategic-depth-from-systems (build diversity, synergy matrices)
**Secondary:** community-meta-gaming (theorycrafting, economy), emergent-gameplay-design (skill interactions)
**Optional:** modding-and-extensibility (third-party tools like Path of Building)
**Estimated time:** 8-12 hours

### Scenario 6: "I'm building Outer Wilds-style exploration"
**Primary:** discovery-through-experimentation (knowledge-based progression)
**Secondary:** emergent-gameplay-design (physics puzzles), player-driven-narratives (environmental storytelling)
**Optional:** None (tightly authored experience)
**Estimated time:** 6-8 hours

### Scenario 7: "I'm building EVE Online-style MMO sandbox"
**Primary:** player-driven-narratives (player politics, server-wide events)
**Secondary:** community-meta-gaming (economy, competitive infrastructure), strategic-depth-from-systems (ship builds)
**Optional:** modding-and-extensibility (API for third-party tools)
**Estimated time:** 10-15 hours

### Scenario 8: "I'm building Noita-style alchemy sandbox"
**Primary:** emergent-gameplay-design (elemental interactions)
**Secondary:** discovery-through-experimentation (hidden combos), sandbox-design-patterns (pixel simulation)
**Optional:** community-meta-gaming (speedrun categories)
**Estimated time:** 6-8 hours

### Scenario 9: "I'm building Dwarf Fortress-style simulation"
**Primary:** player-driven-narratives (procedural legends, relationships)
**Secondary:** emergent-gameplay-design (simulation depth), sandbox-design-patterns (fortress building)
**Optional:** modding-and-extensibility (ASCII replacements, mods)
**Estimated time:** 8-12 hours

### Scenario 10: "I'm building fighting game with tech depth"
**Primary:** discovery-through-experimentation (hidden techniques)
**Secondary:** strategic-depth-from-systems (character matchups), community-meta-gaming (competitive scene)
**Optional:** None (tight balance required)
**Estimated time:** 6-8 hours

### Scenario 11: "I'm building Destiny-style raid content"
**Primary:** strategic-depth-from-systems (build optimization)
**Secondary:** community-meta-gaming (world's first races), player-driven-narratives (fireteam dynamics)
**Optional:** None (authored encounters)
**Estimated time:** 5-7 hours

### Scenario 12: "I'm building Kerbal Space Program-style physics sandbox"
**Primary:** sandbox-design-patterns (rocket building constraints)
**Secondary:** optimization-as-play (delta-v calculations), emergent-gameplay-design (physics interactions)
**Optional:** modding-and-extensibility (KSP has massive mod scene)
**Estimated time:** 6-10 hours

### Scenario 13: "I'm building Opus Magnum-style optimization puzzle"
**Primary:** optimization-as-play (cost/cycles/area metrics)
**Secondary:** sandbox-design-patterns (creative solutions), discovery-through-experimentation (optimal techniques)
**Optional:** community-meta-gaming (leaderboards, percentiles)
**Estimated time:** 4-6 hours

### Scenario 14: "I'm building Dark Souls with invasions"
**Primary:** community-meta-gaming (PvP builds, covenants)
**Secondary:** discovery-through-experimentation (secrets, hidden mechanics), strategic-depth-from-systems (build diversity)
**Optional:** None (tight authored experience)
**Estimated time:** 6-8 hours

### Scenario 15: "I'm building Crusader Kings-style dynasty sim"
**Primary:** player-driven-narratives (procedural soap opera)
**Secondary:** strategic-depth-from-systems (dynasty management), sandbox-design-patterns (emergent history)
**Optional:** modding-and-extensibility (CK has massive mod community)
**Estimated time:** 8-10 hours

### Scenario 16: "I'm building Among Us-style social deduction"
**Primary:** community-meta-gaming (social strategies, meta evolution)
**Secondary:** player-driven-narratives (emergent betrayal drama), strategic-depth-from-systems (role balance)
**Optional:** None (tight social dynamics)
**Estimated time:** 4-6 hours

### Scenario 17: "I'm building Terraria-style action-sandbox"
**Primary:** sandbox-design-patterns (progressive content unlocks)
**Secondary:** emergent-gameplay-design (item synergies), discovery-through-experimentation (hidden bosses)
**Optional:** modding-and-extensibility (tModLoader)
**Estimated time:** 7-10 hours

### Scenario 18: "I'm building Stardew Valley-style farming sim"
**Primary:** sandbox-design-patterns (farm optimization)
**Secondary:** optimization-as-play (crop scheduling), player-driven-narratives (NPC relationships)
**Optional:** modding-and-extensibility (SMAPI)
**Estimated time:** 6-8 hours

### Scenario 19: "I'm building Skyrim-style open RPG"
**Primary:** sandbox-design-patterns (open world exploration)
**Secondary:** discovery-through-experimentation (hidden content), modding-and-extensibility (Creation Kit)
**Optional:** player-driven-narratives (emergent quests)
**Estimated time:** 8-12 hours

### Scenario 20: "I'm building Starcraft-style RTS"
**Primary:** strategic-depth-from-systems (unit counters, build orders)
**Secondary:** community-meta-gaming (competitive scene, replays), emergent-gameplay-design (unit interactions)
**Optional:** modding-and-extensibility (custom maps like Arcade)
**Estimated time:** 8-10 hours

### Scenario 21: "I'm building Satisfactory (3D Factorio)"
**Primary:** optimization-as-play (vertical factory planning)
**Secondary:** sandbox-design-patterns (exploration + building), emergent-gameplay-design (conveyor networks)
**Optional:** modding-and-extensibility (mod support growing)
**Estimated time:** 6-8 hours

### Scenario 22: "I'm building Wildermyth-style procedural RPG"
**Primary:** player-driven-narratives (character arcs, relationships)
**Secondary:** emergent-gameplay-design (tactical combat), strategic-depth-from-systems (build variety)
**Optional:** discovery-through-experimentation (hidden events)
**Estimated time:** 6-10 hours


## Multi-Skill Workflows

### Workflow 1: Sandbox Builder (Minecraft, Terraria)
**Skills in sequence:**
1. **emergent-gameplay-design** (2-3h) - Design orthogonal block/item systems
2. **sandbox-design-patterns** (2-3h) - Progressive complexity, constraints
3. **discovery-through-experimentation** (2-3h) - Hidden crafting recipes, secrets
4. **modding-and-extensibility** (2-3h) - Mod API, community tools

**Total time:** 8-12 hours
**Result:** Sandbox with emergent depth, discoverable secrets, mod support

### Workflow 2: Factory Game (Factorio, Dyson Sphere Program)
**Skills in sequence:**
1. **emergent-gameplay-design** (2-3h) - Production chains, interactions
2. **sandbox-design-patterns** (1-2h) - Spatial constraints, layouts
3. **optimization-as-play** (3-4h) - Bottleneck gameplay, metrics, satisfaction
4. **modding-and-extensibility** (2-3h) - Mod API (Factorio-style)

**Total time:** 8-12 hours
**Result:** Factory game with optimization as core loop, mod ecosystem

### Workflow 3: Physics Playground (BotW, Noita)
**Skills in sequence:**
1. **emergent-gameplay-design** (3-4h) - Physics interactions, orthogonal systems
2. **discovery-through-experimentation** (3-4h) - Environmental hints, safe testing
3. **sandbox-design-patterns** (1-2h) - Open world structure
4. **community-meta-gaming** (1-2h) - Speedrun support (optional)

**Total time:** 8-12 hours
**Result:** Physics sandbox with discoverable depth, speedrun-friendly

### Workflow 4: Colony Simulator (Rimworld, Dwarf Fortress)
**Skills in sequence:**
1. **emergent-gameplay-design** (2-3h) - Simulation systems, interactions
2. **player-driven-narratives** (3-4h) - AI storyteller, character arcs, drama
3. **sandbox-design-patterns** (1-2h) - Base building constraints
4. **modding-and-extensibility** (2-3h) - Mod support (DF/Rimworld-style)

**Total time:** 8-12 hours
**Result:** Colony sim with emergent stories, mod ecosystem

### Workflow 5: Competitive ARPG (Path of Exile, Diablo)
**Skills in sequence:**
1. **emergent-gameplay-design** (2-3h) - Skill/item interactions
2. **strategic-depth-from-systems** (3-4h) - Build diversity, synergy matrices
3. **community-meta-gaming** (3-4h) - Theorycrafting, economy, leaderboards
4. **optimization-as-play** (1-2h) - Build optimization tools (optional)

**Total time:** 9-13 hours
**Result:** ARPG with theorycrafting depth, competitive scene

### Workflow 6: Exploration Game (Outer Wilds, Subnautica)
**Skills in sequence:**
1. **discovery-through-experimentation** (3-4h) - Knowledge-based progression
2. **emergent-gameplay-design** (2-3h) - Physics/environmental systems
3. **sandbox-design-patterns** (1-2h) - Open world structure
4. **player-driven-narratives** (1-2h) - Environmental storytelling (optional)

**Total time:** 7-11 hours
**Result:** Exploration game where discovery drives progression

### Workflow 7: MMO Sandbox (EVE Online, Albion)
**Skills in sequence:**
1. **player-driven-narratives** (3-4h) - Player politics, server-wide events
2. **community-meta-gaming** (3-4h) - Economy, competitive infrastructure
3. **strategic-depth-from-systems** (2-3h) - Build variety, counter-play
4. **modding-and-extensibility** (1-2h) - API for third-party tools

**Total time:** 9-13 hours
**Result:** MMO with player-driven content, thriving external meta-game

### Workflow 8: Optimization Puzzle (Opus Magnum, SpaceChem)
**Skills in sequence:**
1. **optimization-as-play** (3-4h) - Multiple metrics, leaderboards
2. **sandbox-design-patterns** (2-3h) - Creative solution space
3. **discovery-through-experimentation** (1-2h) - Optimal technique discovery
4. **community-meta-gaming** (1-2h) - Percentile histograms, sharing

**Total time:** 7-11 hours
**Result:** Optimization puzzle with competitive community


## Quick Start Guides

### Quick Start 1: Minimal Viable Emergence (4 hours)
**Goal:** Sandbox with basic emergent gameplay

**Steps:**
1. Read emergent-gameplay-design Quick Start (1h)
2. Design 3 orthogonal systems with 9 interactions (1h)
3. Prototype interaction matrix (1h)
4. Playtest for emergent moments (1h)

**Result:** Basic sandbox with player-discovered solutions

### Quick Start 2: Optimization Game MVP (4 hours)
**Goal:** Factory game with bottleneck gameplay

**Steps:**
1. Read optimization-as-play Quick Start (1h)
2. Implement production chain with visible metrics (1.5h)
3. Add bottleneck visualization (1h)
4. Playtest optimization loop (0.5h)

**Result:** Basic factory game with satisfying optimization

### Quick Start 3: Discovery-Driven Exploration (4 hours)
**Goal:** Exploration game with secrets

**Steps:**
1. Read discovery-through-experimentation Quick Start (1h)
2. Place environmental hints for 3 mechanics (1h)
3. Create discovery journal system (1h)
4. Playtest hint effectiveness (1h)

**Result:** Exploration game with discoverable depth

### Quick Start 4: Emergent Storytelling (4 hours)
**Goal:** Simulation with memorable stories

**Steps:**
1. Read player-driven-narratives Quick Start (1h)
2. Implement character relationships (1.5h)
3. Add dramatic event generation (1h)
4. Playtest for memorable moments (0.5h)

**Result:** Simulation that generates player stories


## Integration with Other Skillpacks

### Cross-Pack Dependencies

#### From Simulation-Tactics Pack (bravos)
- **When emergence requires simulation:** Use emergent-gameplay-design + simulation-tactics
  - Example: BotW physics requires physics-simulation-patterns
  - Example: Dwarf Fortress requires ai-and-agent-simulation + player-driven-narratives

#### From Yzmir (Theory) Pack
- **When systems need mathematical foundation:** Use strategic-depth-from-systems + yzmir/game-theory-foundations
  - Example: Path of Exile build math requires optimization theory
  - Example: EVE economy requires economic simulation theory

#### From Lyra (Player Experience) Pack
- **When systems need feel/polish:** Use optimization-as-play + lyra/game-feel-and-polish
  - Example: Factorio needs juice on production milestones
  - Example: Satisfactory needs satisfying conveyor animations


## Common Pitfalls

### Pitfall 1: Starting with Wrong Skill
**Problem:** Jumping to optimization-as-play without emergent-gameplay-design foundation

**Symptom:** Optimization game with no depth (one optimal solution)

**Fix:** ALWAYS start with emergent-gameplay-design for orthogonal systems, THEN apply optimization

### Pitfall 2: Over-Applying All Skills
**Problem:** Trying to use all 8 skills on every project

**Symptom:** Scope creep, conflicting design goals, nothing gets finished

**Fix:** Use routing logic above. Most projects need 2-4 skills, not all 8.

### Pitfall 3: Skipping Foundation
**Problem:** Going straight to Wave 2/3 skills without Wave 1

**Symptom:** Surface-level implementation without systemic depth

**Fix:** emergent-gameplay-design is FOUNDATIONAL for most projects. Read it first.

### Pitfall 4: Not Recognizing Hybrid Design
**Problem:** Trying to make EVERYTHING systemic (including authored narrative)

**Symptom:** Losing authored emotional moments, pacing issues

**Fix:** Most games blend systems + authored content. BotW has both. Use routing logic for systems portion only.

### Pitfall 5: Underestimating Time
**Problem:** Thinking "I'll just add emergence" will take 30 minutes

**Symptom:** Rushed implementation, no playtesting, shallow system

**Fix:** Each skill requires 2-4 hours minimum. Budget 8-15 hours for multi-skill projects.


## Success Criteria

### Your project successfully uses systems-as-experience when:

**Emergence (Core):**
- [ ] Players discover solutions you didn't intend
- [ ] System interactions create "content"
- [ ] Replayability through emergent variety

**Depth:**
- [ ] Hidden depth rewards mastery
- [ ] Community discovers new strategies
- [ ] No single dominant strategy

**Discovery:**
- [ ] Curiosity is rewarded
- [ ] Secrets feel earned, not arbitrary
- [ ] "Aha moments" occur frequently

**Community:**
- [ ] Players create external tools (wikis, calculators)
- [ ] Meta-game emerges beyond playing
- [ ] Community extends game through mods/content

**Satisfaction:**
- [ ] Systems feel satisfying to master
- [ ] Optimization provides clear feedback
- [ ] Player expression is visible


## Conclusion

**The Golden Rule:**
> "Design systems that interact, then let players surprise you."

### When You're Done with This Pack

You should be able to:
- ✅ Identify which skills apply to your project
- ✅ Apply 2-4 skills in correct sequence
- ✅ Create games where systems ARE the content
- ✅ Design for emergence, discovery, and community
- ✅ Avoid common pitfalls in systemic design

### Next Steps

1. **Identify your game type** (use routing logic above)
2. **Read foundational skill** (usually emergent-gameplay-design)
3. **Apply skills in sequence** (use workflows above)
4. **Playtest for emergence** (look for unintended solutions)
5. **Iterate on interactions** (orthogonal systems multiply)


## Pack Structure Reference

```
bravos/systems-as-experience/
├── using-systems-as-experience/        (THIS SKILL - router)
├── emergent-gameplay-design/           (FOUNDATIONAL ⭐)
├── sandbox-design-patterns/
├── strategic-depth-from-systems/
├── optimization-as-play/
├── discovery-through-experimentation/
├── player-driven-narratives/
├── modding-and-extensibility/
└── community-meta-gaming/
```

**Total pack time:** 16-28 hours for comprehensive implementation


**Go build systems that surprise you.**
