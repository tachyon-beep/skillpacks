---
name: sandbox-design-patterns
description: Creative tools with constraints - pure, guided, and hybrid sandbox approaches
---

# Sandbox Design Patterns

## Description
Master sandbox game design: provide creative tools without overwhelming players. Apply constraint-based creativity, progressive revelation, optional objectives, and balanced onboarding to create accessible yet deep player-driven experiences. Learn when to use pure sandbox (Minecraft), guided sandbox (Terraria), or hybrid approaches.

## When to Use This Skill
Use this skill when designing or implementing:
- Sandbox building games (block building, construction, city builders)
- Creative mode systems in survival games
- User-generated content platforms
- Level editors and creation tools
- Open-ended simulation games
- Modding and customization systems
- Games where player creativity is core to the experience

Do NOT use this skill for:
- Linear story-driven games with no player creation
- Competitive multiplayer with fixed rules and no customization
- Puzzle games with specific solutions
- Pure narrative experiences without systemic freedom

---

## Quick Start (Time-Constrained Implementation)

If you need working sandbox systems quickly (< 4 hours), follow this priority order:

**CRITICAL (Never Skip)**:
1. **Start with 10-15 core tools/blocks, not 100+**: Less is more for initial experience
2. **Provide 2-3 example builds**: Templates players can copy/modify for inspiration
3. **Tutorial as first project**: Don't explain, guide player through building something
4. **Meaningful constraints**: Limit palette/tools to spark creativity, not unlimited options

**IMPORTANT (Strongly Recommended)**:
5. Optional goal system: Challenges/quests players can ignore if purely creative
6. Progressive tool unlocking: Start simple (5 tools), unlock advanced (20+ tools) through use
7. Gallery/showcase: Display community/pre-made builds for inspiration
8. Clear undo/redo: Creative experimentation requires safe failure

**CAN DEFER** (Optimize Later):
- Multiplayer collaboration features
- Advanced procedural generation tools
- Scripting/logic systems for power users
- Export to external formats

**Example - 4 Hour Sandbox MVP**:
```python
# 1. Core building blocks (10 types, not 200)
STARTER_BLOCKS = [
    "Wood_Floor", "Wood_Wall", "Wood_Roof",
    "Stone_Floor", "Stone_Wall",
    "Glass_Window", "Door",
    "Light", "Decoration", "Stairs"
]

# 2. Essential tools only
STARTER_TOOLS = [
    "Place_Block",
    "Delete_Block",
    "Copy_Area",
    "Undo/Redo"
]

# 3. Tutorial as first build
def tutorial_onboarding():
    # Guide: "Let's build a small house together"
    guide_place_blocks(Wood_Floor, grid_area=(5, 5))
    guide_place_blocks(Wood_Wall, perimeter_only=True)
    guide_place_blocks(Wood_Roof, top_layer=True)
    guide_place_blocks(Door, front_center=True)

    # Unlock message: "Great! Now you have all tools. Want to try building on your own?"
    unlock_all_starter_tools()
    spawn_example_buildings_nearby()  # Inspiration

# 4. Optional goals (can be ignored)
OPTIONAL_CHALLENGES = [
    "Build a 2-story house",
    "Create a garden with decorations",
    "Build a bridge between two hills"
]
```

This gives you a functional sandbox in hours. Expand based on playtesting feedback.

---

## Core Concepts

### 1. The Constraint Paradox

Creative freedom requires constraints. Unlimited options cause analysis paralysis. Meaningful limitations spark innovation.

**The Problem: Blank Canvas Paralysis**
```python
# ❌ TOO MUCH FREEDOM
def start_game():
    player.unlock_all_blocks(200+)
    player.unlock_all_tools(15)
    player.spawn_in_infinite_empty_world()
    # Player: "What do I build? This is overwhelming."
```

**The Solution: Constraint-Based Creativity**
```python
# ✅ MEANINGFUL CONSTRAINTS
def start_game():
    # Limited palette forces creative combinations
    player.unlock_blocks([
        "Wood_Planks", "Stone_Brick", "Glass",
        "Thatch", "Door"
    ])  # 5 blocks, unlimited creativity

    # Start on small island (space constraint)
    world.spawn_player_on_island(size=50x50)

    # Example builds nearby (inspiration)
    world.spawn_example_huts(count=3)

    # Player: "I can work with this! Let me combine these..."
```

**Why Constraints Enable Creativity**:
- **Focused exploration**: 5 blocks = 120 combinations to discover
- **Forced innovation**: Limited tools require creative problem-solving
- **Achievable mastery**: Players can learn full system quickly
- **Pride in solutions**: Overcoming limits feels rewarding

**Real-World Examples**:
- **Minecraft Early Game**: ~40 blocks, vast creativity
- **Factorio**: Limited building types, infinite logistic puzzles
- **LEGO**: Fixed brick sizes/colors, endless creation
- **Haiku poetry**: 5-7-5 syllable constraint, profound expression

**Progressive Constraint Removal**:
```python
class ProgressiveUnlocking:
    def __init__(self):
        self.blocks_unlocked = STARTER_SET  # 5 blocks
        self.mastery_level = 0

    def on_player_uses_tools(self, actions_count):
        if actions_count > 100:
            self.unlock_tier(2)  # +10 blocks
        if actions_count > 500:
            self.unlock_tier(3)  # +20 blocks
        # Never unlock all at once

    def unlock_tier(self, tier):
        new_blocks = TIER_BLOCKS[tier]
        self.blocks_unlocked.extend(new_blocks)
        show_message(f"Unlocked {len(new_blocks)} new materials!")
```

### 2. Tools vs Goals: The Sandbox Spectrum

Sandbox games exist on a spectrum from "pure tools" to "guided objectives". Choose your position intentionally.

**Pure Sandbox (Tool-Focused)**:
```
[Minecraft Creative] ← [Minecraft Survival] ← [Terraria] ← [Tower Defense] → [Story Game]
    ↑                                                                              ↑
Pure tools,                                                              Fixed objectives,
player goals                                                              no creativity
```

**Pure Sandbox** (Minecraft Creative Mode):
- **Philosophy**: "Here are tools. What will you create?"
- **Player-driven goals**: All objectives emerge from player
- **No fail states**: Can't lose, only experiment
- **Best for**: Highly creative players, experienced builders

```python
class PureSandboxMode:
    def __init__(self):
        self.resources = "unlimited"
        self.objectives = None  # Player decides
        self.fail_conditions = None
        self.inspiration = [
            "Example builds",
            "Community gallery",
            "Building challenges (optional)"
        ]
```

**Guided Sandbox** (Terraria, Factorio):
- **Philosophy**: "Here are tools + suggested challenges"
- **Hybrid goals**: Game provides direction, player chooses path
- **Soft fail states**: Can die/lose, but recoverable
- **Best for**: Mixed audience (creative + goal-oriented)

```python
class GuidedSandboxMode:
    def __init__(self):
        self.resources = "gather from world"  # Constraint
        self.objectives = [
            "Defeat first boss (optional)",
            "Build housing for NPCs",
            "Explore underground"
        ]
        self.fail_conditions = "death (respawn with penalty)"
        self.creative_freedom = "how you achieve goals is open-ended"
```

**Structured Sandbox** (Kerbal Space Program Career):
- **Philosophy**: "Master tools through structured challenges"
- **Game-driven goals**: Clear objectives with creative solutions
- **Progressive unlocking**: Earn tools through completion
- **Best for**: Players who need direction and progression

```python
class StructuredSandboxMode:
    def __init__(self):
        self.resources = "limited by career budget"
        self.objectives = [
            "Mission 1: Reach 10km altitude",
            "Mission 2: Orbit planet",
            "Mission 3: Land on moon"
        ]
        self.unlocking = "complete missions → unlock parts"
        self.fail_conditions = "mission failure (retry with learning)"
```

**Decision Framework: Which Sandbox Type?**

| Question | Pure Sandbox | Guided Sandbox | Structured Sandbox |
|----------|-------------|----------------|-------------------|
| Target audience? | Experienced creative players | Mixed (creative + casual) | Goal-oriented players |
| Okay with blank canvas? | Yes, thrives on it | With templates/inspiration | No, needs direction |
| Progression system? | Optional (cosmetic only) | Soft (unlocks + optional goals) | Hard (mission-based) |
| Resource constraints? | None (creative mode) | Yes (survival mode) | Yes (career budget) |
| Fail states? | None | Soft (death/respawn) | Mission failure |
| Implementation time | Fastest (just tools) | Medium (tools + content) | Longest (tools + missions) |

### 3. Progressive Revelation (Not Progressive Unlocking)

Show complexity gradually through use, not through gates. Players should discover depth, not grind for basics.

**The Problem: Feature Overload**
```python
# ❌ OVERWHELMING: Show everything at once
def open_build_menu():
    categories = [
        "Blocks (200 types)",
        "Tools (15 types)",
        "Colors (RGB picker - 16.7M colors)",
        "Advanced Options (30 settings)",
        "Modifiers (12 modes)"
    ]
    display_all_at_once(categories)  # Cognitive overload
```

**The Solution: Progressive Revelation**
```python
# ✅ GRADUAL DISCOVERY: Show as needed
class ProgressiveUI:
    def __init__(self):
        self.visible_blocks = STARTER_BLOCKS  # 5 blocks
        self.visible_tools = ["Place", "Delete"]
        self.advanced_features_hidden = True

    def on_player_uses_starter_tools(self, count):
        if count > 50:  # Player is comfortable with basics
            self.reveal_tool("Copy_Paste")
            show_hint("New tool unlocked: Copy/Paste (Ctrl+C / Ctrl+V)")

        if count > 200:  # Player is proficient
            self.reveal_category("Advanced_Blocks")
            self.advanced_features_hidden = False

    def on_player_opens_menu(self):
        # Simple menu initially
        if len(self.visible_blocks) < 20:
            show_simple_grid(self.visible_blocks)
        else:
            # Expand to categorized view only when needed
            show_categorized_menu(self.visible_blocks)
```

**Progressive Complexity Example (Building Tools)**:

**Phase 1: Starter (First 10 minutes)**
```python
PHASE_1_TOOLS = {
    "Place_Block": "Click to place",
    "Delete_Block": "Right-click to remove",
    "Rotate": "R key to rotate"
}
# Players learn: Basic placement + removal
```

**Phase 2: Intermediate (After 50 actions)**
```python
PHASE_2_TOOLS = {
    **PHASE_1_TOOLS,
    "Copy_Area": "Select area, Ctrl+C to copy",
    "Undo": "Ctrl+Z to undo mistake"
}
# Players learn: Efficient workflows
```

**Phase 3: Advanced (After 200 actions)**
```python
PHASE_3_TOOLS = {
    **PHASE_2_TOOLS,
    "Symmetry_Mode": "Mirror builds automatically",
    "Grid_Snapping": "Precise alignment",
    "Paint_Tool": "Change colors without replacing"
}
# Players learn: Power user techniques
```

**Phase 4: Expert (After 1000 actions)**
```python
PHASE_4_TOOLS = {
    **PHASE_3_TOOLS,
    "Scripting": "Automate repetitive tasks",
    "Boolean_Operations": "Union/subtract shapes",
    "Procedural_Tools": "Generate patterns"
}
# Players learn: Mastery and automation
```

**Key Principle**: Unlocking should feel like "discovering depth" not "grinding to access basics". Player can build complete projects at every phase.

### 4. Onboarding Through Doing (Not Explaining)

Don't teach building mechanics. Guide players through building something, mechanics emerge naturally.

**The Problem: Tutorial Screens**
```python
# ❌ PASSIVE TUTORIAL: Boring and ineffective
def tutorial():
    show_text("Welcome! Use WASD to move")
    wait_for_player_press_ok()

    show_text("Press Q to open block menu")
    wait_for_player_press_ok()

    show_text("Click to place blocks")
    wait_for_player_press_ok()

    # 10 more screens...

    show_text("Now you're ready to build! Good luck!")
    spawn_in_empty_world()
    # Player: "What do I build? I forgot everything."
```

**The Solution: Tutorial as First Build**
```python
# ✅ ACTIVE TUTORIAL: Learning by doing
def tutorial_as_first_build():
    narrator = "Let's build a small house together!"

    # Step 1: Place floor (game auto-selects floor block)
    highlight_ground_area(5, 5)
    hint("Click highlighted area to place floor")
    wait_for_player_place_floor()
    narrator = "Great! You're building!"

    # Step 2: Place walls (game auto-selects walls)
    highlight_perimeter()
    hint("Now add walls around the edge")
    wait_for_player_place_walls()
    narrator = "Looking good! Let's add a roof."

    # Step 3: Roof
    auto_place_roof()  # Game does this part
    narrator = "I added the roof. Now let's add a door."

    # Step 4: Door (introduces block selection)
    show_block_menu_with_door_highlighted()
    hint("Select the door block, then place it")
    wait_for_player_place_door()

    # Step 5: Celebrate
    narrator = "Perfect! You built your first house!"
    camera_zoom_out_to_show_house()

    # Step 6: Freedom with scaffolding
    spawn_example_buildings_nearby()
    narrator = "Here are some other buildings for inspiration. Want to modify yours or build something new?"

    # Player has: Working knowledge, completed project, inspiration, confidence
```

**Tutorial Design Principles**:

1. **Start with outcome, not mechanics**: "Let's build a house" (not "Let me explain controls")
2. **Auto-select tools initially**: Player follows guidance without choosing from menus
3. **Introduce complexity gradually**: First build uses 5 blocks, player learns more exist
4. **Celebrate completion**: Finished tutorial = finished first build (tangible progress)
5. **Provide next steps**: Inspiration nearby, optional challenges, or freeform exploration

### 5. Optional Objectives for Mixed Audiences

Creative players want pure freedom. Goal-oriented players want direction. Provide both.

**The Problem: One-Size-Fits-All**
```python
# ❌ FORCES PLAYSTYLE: Alienates one audience
class SandboxGame:
    def start(self):
        if self.mode == "pure_sandbox":
            # No goals at all
            # Goal-oriented players: "What am I supposed to do?"
            self.spawn_creative_mode()

        elif self.mode == "story_mode":
            # Required objectives
            # Creative players: "Stop telling me what to build!"
            self.force_tutorial()
            self.require_mission_completion()
```

**The Solution: Optional Layered Objectives**
```python
# ✅ SUPPORTS BOTH: Players choose engagement
class LayeredObjectiveSystem:
    def __init__(self):
        self.objectives = {
            "always_visible": None,  # Never force
            "available": [
                # Layer 1: Inspiration (not objectives)
                {"type": "showcase", "content": "Featured community builds"},
                {"type": "theme", "content": "Weekly theme: Medieval castles"},

                # Layer 2: Gentle suggestions (optional)
                {"type": "challenge", "content": "Build a 2-story house", "reward": "cosmetic"},
                {"type": "quest", "content": "Create a village with 5 buildings", "reward": "new blocks"},

                # Layer 3: Structured content (opt-in)
                {"type": "campaign", "content": "Tutorial campaign: 10 build lessons", "reward": "mastery"},
                {"type": "scenario", "content": "Rescue mission: Build bridge in 10 minutes", "reward": "achievement"}
            ]
        }

    def present_to_player(self):
        # Show in separate optional menu, not forced
        menu = ObjectiveMenu()
        menu.add_section("Inspiration", self.get_inspiration_content())
        menu.add_section("Challenges (Optional)", self.get_challenges())
        menu.add_section("Campaigns (Structured Play)", self.get_campaigns())

        # Player can ignore entirely or engage deeply
        menu.show(can_close_immediately=True)
```

**Objective Design Patterns**:

**Pattern 1: Inspiration (Not Objectives)**
- **What**: Showcase builds, themes, community creations
- **Purpose**: Spark ideas without pressure
- **Example**: "This week's theme: Underwater bases (share yours!)"
- **Reward**: None (intrinsic motivation only)

**Pattern 2: Gentle Challenges**
- **What**: Optional build suggestions with themes
- **Purpose**: Direction for directionless players
- **Example**: "Challenge: Build a bridge between two mountains"
- **Reward**: Cosmetic (doesn't gate content)

**Pattern 3: Structured Quests**
- **What**: More specific objectives with progression
- **Purpose**: Mini-campaigns for goal-oriented players
- **Example**: "Quest: Build a village (1. Town hall, 2. Houses, 3. Shops)"
- **Reward**: New blocks/tools (opt-in progression)

**Pattern 4: Time/Resource Challenges**
- **What**: Constraints as interesting problems
- **Purpose**: Different mode for competitive players
- **Example**: "Scenario: Build a shelter with 50 blocks in 5 minutes"
- **Reward**: Leaderboard position

**Key Principle**: Creative players never see objectives unless they want to. Goal-oriented players have clear direction. Both play the same game differently.

---

## Decision Frameworks

### Framework 1: Pure Sandbox vs Guided vs Structured

Choose your sandbox type based on target audience and desired play patterns.

**Pure Sandbox** (Minecraft Creative, Garry's Mod):

Use when:
- Target audience: Experienced creative players
- Primary motivation: Self-expression and experimentation
- No external progression needed
- Community/sharing is core to experience
- Budget: Minimal (just tools, no content)

Don't use when:
- Targeting mass market (too intimidating)
- Players need clear goals to stay engaged
- Monetization requires progression system

Example design:
```python
class PureSandboxGame:
    def __init__(self):
        self.all_tools_unlocked = True
        self.resources_unlimited = True
        self.objectives = None
        self.onboarding = "tutorial build + examples"
        self.engagement = "community showcase + sharing"
```

**Guided Sandbox** (Minecraft Survival, Terraria, Factorio):

Use when:
- Target audience: Mixed (creative + casual)
- Blend of freedom and direction needed
- Resource gathering adds meaning to building
- Progression through world, not just tools
- Budget: Medium (tools + world content)

Don't use when:
- Want pure creative expression (constraints frustrate)
- Want linear story (player freedom conflicts)

Example design:
```python
class GuidedSandboxGame:
    def __init__(self):
        self.tools_unlocked = "progressive (5 → 50)"
        self.resources_gathered = "from world exploration"
        self.objectives = "soft (boss hints, NPC quests)"
        self.onboarding = "survival tutorial + goals"
        self.engagement = "exploration + optional objectives"
```

**Structured Sandbox** (Kerbal Space Program Career, Cities: Skylines scenarios):

Use when:
- Target audience: Goal-oriented players
- Complex systems need teaching through missions
- Progression via mastery is motivating
- Tools overwhelming without structure
- Budget: High (tools + missions + content)

Don't use when:
- Creative freedom is primary appeal
- Players resist being told what to build

Example design:
```python
class StructuredSandboxGame:
    def __init__(self):
        self.tools_unlocked = "mission-based"
        self.resources_budgeted = "per mission/career"
        self.objectives = "required missions + optional challenges"
        self.onboarding = "campaign progression"
        self.engagement = "mission completion + mastery"
```

### Framework 2: Creative Mode vs Survival Mode Integration

Decide if you need one mode or both, and how they relate.

**Creative-Only** (Pure sandbox, no survival):

Use when:
- Building IS the game (not a side feature)
- No combat or resource mechanics needed
- Fastest to implement
- Target: Pure creative players

**Survival-Only** (Guided sandbox, no creative):

Use when:
- Resource constraints are core to design
- Don't want players to "cheat" with creative
- Competitive/challenge-based game
- Target: Goal-oriented players

**Both Modes** (Toggle between):

Use when:
- Want to reach both audiences
- Planning mode (creative) + execution mode (survival)
- Prototyping (creative) + playtesting (survival)

Implementation pattern:
```python
class DualModeGame:
    def __init__(self):
        self.modes = {
            "creative": {
                "resources": "unlimited",
                "damage": "disabled",
                "flight": "enabled",
                "focus": "building without constraints"
            },
            "survival": {
                "resources": "gathered",
                "damage": "enabled",
                "flight": "disabled",
                "focus": "building with meaningful constraints"
            }
        }

    def can_switch_modes(self):
        # Design choice: Can players switch mid-game?

        # Option 1: Never (prevents "cheating")
        return False

        # Option 2: One-way (creative → survival, not reverse)
        return self.current_mode == "creative"

        # Option 3: Freely (planning + execution flow)
        return True  # Most player-friendly
```

**Best Practice**: Offer both modes, allow free switching. Creative players use creative only. Survival players use survival only. Power users use both (plan in creative, execute in survival).

### Framework 3: When to Add Structure to Pure Sandbox

Start with pure sandbox. Add structure based on player behavior and feedback.

**Signals You Need More Structure**:

1. **High abandonment rate**: Players quit within 30 minutes
2. **"What do I do?" feedback**: Players ask for direction
3. **Empty builds**: Players place a few blocks and stop
4. **No sharing**: Players don't showcase their creations (nothing to show)

**Incremental Structure Additions** (in order):

**Level 1: Add Inspiration** (No gameplay change)
```python
# Cost: 1-2 days implementation
def add_inspiration():
    - Featured builds gallery (pre-made examples)
    - Community showcase (player-submitted builds)
    - Random build prompts ("Try building: A treehouse")
```

**Level 2: Add Optional Challenges** (Soft objectives)
```python
# Cost: 3-5 days implementation
def add_challenges():
    - Daily/weekly challenges ("Build a castle")
    - Reward: Cosmetic only (doesn't gate content)
    - Completely optional (can be ignored)
```

**Level 3: Add Progression** (Unlock system)
```python
# Cost: 1-2 weeks implementation
def add_progression():
    - Start with limited blocks (10)
    - Unlock more through play (50+)
    - Still no required objectives
    - Mastery-based, not grind-based
```

**Level 4: Add Structured Content** (Campaigns/scenarios)
```python
# Cost: 2-4 weeks implementation per campaign
def add_campaigns():
    - Tutorial campaign (teaching advanced techniques)
    - Themed scenarios (timed challenges, constraints)
    - Separate mode from main sandbox
```

**Decision Rule**: Add minimum structure needed to retain players. More structure = more development time + less creative freedom. Only add when data shows it's needed.

### Framework 4: Constraint Design (What to Limit and Why)

Choose meaningful constraints that enhance creativity, not arbitrary limits that frustrate.

**Good Constraints** (Spark creativity):

**Space Constraints**:
```python
# Limited build area forces tight designs
island_size = "50x50 blocks"  # Small enough to fill, big enough to be interesting
# Forces: Vertical building, efficient layouts, creative use of space
```

**Palette Constraints**:
```python
# Limited blocks force innovative combinations
starter_blocks = 5  # Wood, stone, glass, door, roof
# Forces: Creative material mixing, simple aesthetic
```

**Resource Constraints** (Survival mode):
```python
# Gathered resources make builds meaningful
rare_materials = ["Diamond", "Gold"]  # Must explore to find
# Forces: Value decisions (what to build with rare materials?)
```

**Time Constraints** (Challenge mode):
```python
# Time pressure forces decisive action
build_challenge = "Create shelter in 10 minutes"
# Forces: Prioritization, simple designs under pressure
```

**Bad Constraints** (Frustrate without purpose):

**Arbitrary Unlocking**:
```python
# ❌ BAD: Basic tools locked behind grind
basic_door = "Unlocks after placing 1000 blocks"
# Frustrates: Artificial gate with no creative benefit
```

**Overcomplicated Rules**:
```python
# ❌ BAD: Realistic structural physics
if not has_foundation:
    building_collapses()
# Frustrates: Punishes experimentation, kills creativity
```

**Resource Tedium**:
```python
# ❌ BAD: Every single block requires gathering
player.inventory.wood = 10  # Need to chop trees for every plank
# Frustrates: Turns building into grinding simulator
```

**Constraint Design Checklist**:
- [ ] Does constraint spark interesting decisions? (Good)
- [ ] Does constraint punish experimentation? (Bad)
- [ ] Can players overcome constraint creatively? (Good)
- [ ] Is constraint just artificial gate? (Bad)
- [ ] Does constraint add meaningful challenge? (Good)
- [ ] Does constraint make building tedious? (Bad)

### Framework 5: Onboarding Length vs Depth

Balance tutorial length with system depth. Deeper systems need longer onboarding.

**Simple Sandbox** (5-10 blocks, 3 tools):
```python
# Onboarding: 5 minutes
Tutorial: "Build a small house (guided), now you're ready!"
# Quick to competence
```

**Medium Sandbox** (50 blocks, 10 tools, categories):
```python
# Onboarding: 15-20 minutes
Tutorial: "Build house → Add advanced features → Explore categories"
# Phased learning
```

**Complex Sandbox** (100+ blocks, 20+ tools, scripting):
```python
# Onboarding: 30-60 minutes (campaign)
Tutorial: "Campaign with 5-10 building lessons"
# Progressive mastery
```

**Decision Table**:

| System Complexity | Onboarding Approach | Time Investment |
|------------------|---------------------|-----------------|
| Simple (5-15 blocks) | Single guided build | 5-10 minutes |
| Medium (20-50 blocks) | Guided build + exploration | 15-20 minutes |
| Complex (50-100 blocks) | Multi-phase tutorial | 20-30 minutes |
| Very Complex (100+) | Campaign-style progression | 30-60 minutes |

**Rule**: Never onboard for longer than 10% of typical play session. If game sessions are 60 minutes, onboarding should be < 6 minutes.

---

## Implementation Patterns

### Pattern 1: Starter Block Set Design

Provide minimum viable palette that allows complete builds.

```python
class StarterBlockSet:
    """
    Design principle: 5-10 blocks that cover all building needs
    Players can create varied, complete builds with this set
    """

    def __init__(self):
        # Essential structure blocks
        self.structural = [
            "Floor_Wood",    # Base foundation
            "Wall_Wood",     # Vertical structure
            "Roof_Thatch"    # Top covering
        ]

        # Essential functional blocks
        self.functional = [
            "Door",          # Entry/exit
            "Window_Glass"   # Light and aesthetics
        ]

        # Optional: One decorative for personality
        self.decorative = [
            "Plant_Pot"      # Player expression
        ]

    def validates_completeness(self):
        """
        Can players build a complete, functional structure?
        """
        checks = {
            "has_floor": "Floor_Wood" in self.structural,
            "has_walls": "Wall_Wood" in self.structural,
            "has_roof": "Roof_Thatch" in self.structural,
            "has_door": "Door" in self.functional,
            "has_window": "Window_Glass" in self.functional
        }
        return all(checks.values())

# Example expansion tiers
TIER_2_BLOCKS = [
    "Floor_Stone", "Wall_Stone",  # Alternative materials
    "Stairs", "Fence",            # Functional variety
    "Torch", "Lantern"            # Lighting options
]  # Now 10+ blocks, still manageable

TIER_3_BLOCKS = [
    "Furniture_Table", "Furniture_Chair",  # Interior design
    "Floor_Brick", "Wall_Brick",           # More materials
    "Decoration_Painting", "Decoration_Rug"  # Aesthetics
]  # Now 20+ blocks, needs categories
```

**Design Validation**:
- [ ] Can player build complete house with starter set? (Yes = good)
- [ ] Does starter set allow variety? (Yes = good)
- [ ] Is any block strictly better than another? (No = good, all have use)
- [ ] Can player express personality? (Yes = good, at least 1 decorative)

### Pattern 2: Progressive Tool Revelation

Start with essential tools, reveal advanced tools through use.

```python
class ToolProgressionSystem:
    def __init__(self):
        self.player_action_count = 0
        self.tools_revealed = set()

        # Tool tiers with unlock thresholds
        self.tool_tiers = {
            "starter": {
                "tools": ["Place", "Delete", "Rotate"],
                "unlock_at": 0  # Available immediately
            },
            "intermediate": {
                "tools": ["Copy", "Paste", "Undo"],
                "unlock_at": 50  # After 50 actions
            },
            "advanced": {
                "tools": ["Symmetry", "Grid_Snap", "Paint"],
                "unlock_at": 200  # After 200 actions
            },
            "expert": {
                "tools": ["Boolean_Operations", "Procedural", "Scripting"],
                "unlock_at": 1000  # After 1000 actions
            }
        }

        # Unlock starter tools
        self.reveal_tier("starter")

    def on_player_action(self, action):
        """Called every time player places/deletes/modifies"""
        self.player_action_count += 1

        # Check for new tier unlocks
        for tier_name, tier_data in self.tool_tiers.items():
            if tier_name not in self.tools_revealed:
                if self.player_action_count >= tier_data["unlock_at"]:
                    self.reveal_tier(tier_name)

    def reveal_tier(self, tier_name):
        """Unlock new tier of tools"""
        self.tools_revealed.add(tier_name)
        tier = self.tool_tiers[tier_name]

        for tool in tier["tools"]:
            self.unlock_tool(tool)

        # Show notification
        if tier_name != "starter":
            self.show_unlock_notification(
                f"New tools unlocked: {', '.join(tier['tools'])}"
            )

    def unlock_tool(self, tool_name):
        """Make tool visible in UI"""
        ui.toolbar.add_tool(tool_name)

    def show_unlock_notification(self, message):
        """Non-intrusive notification"""
        notification = UINotification(message)
        notification.position = "top_right"
        notification.duration = 5  # seconds
        notification.dismissible = True
        notification.show()
```

**Why Action-Based (Not Time-Based)**:
- Rewards engagement (players who experiment unlock faster)
- Respects player pace (slow learners aren't rushed)
- Feels earned (actions → mastery → advanced tools)

### Pattern 3: Tutorial as First Build

Guide player through constructing something, mechanics emerge naturally.

```python
class TutorialFirstBuild:
    def __init__(self):
        self.step = 0
        self.house_position = Vector3(0, 0, 0)
        self.completed_steps = []

    def start_tutorial(self):
        """Begin interactive tutorial"""
        self.narrator("Welcome! Let's build your first house together.")
        self.step_1_foundation()

    def step_1_foundation(self):
        """Step 1: Place floor blocks"""
        # Auto-select floor block (no menu needed yet)
        player.select_block("Floor_Wood")

        # Highlight target area
        self.highlight_grid_area(
            center=self.house_position,
            size=(5, 5),
            color="green_translucent"
        )

        # Instruction
        self.show_hint(
            "Click the highlighted squares to place floor blocks",
            point_to="highlighted_area"
        )

        # Wait for completion
        self.wait_for_player_fill_area(
            area=(5, 5),
            block_type="Floor_Wood",
            on_complete=self.step_2_walls
        )

    def step_2_walls(self):
        """Step 2: Place wall blocks"""
        self.narrator("Great foundation! Now let's add walls.")

        # Auto-select walls
        player.select_block("Wall_Wood")

        # Highlight perimeter
        self.highlight_perimeter(
            area=(5, 5),
            height=3,
            color="green_translucent"
        )

        self.show_hint("Add walls around the edges (3 blocks high)")

        self.wait_for_player_fill_perimeter(
            area=(5, 5),
            height=3,
            block_type="Wall_Wood",
            on_complete=self.step_3_door
        )

    def step_3_door(self):
        """Step 3: Place door (introduces block selection)"""
        self.narrator("Walls look good! Let's add a door so we can get inside.")

        # First time: Teach block selection
        self.show_hint("Press TAB to open block menu, select DOOR")

        # Simplified block menu (only relevant blocks)
        player.open_block_menu(
            blocks=["Door", "Window_Glass"],
            highlight="Door"
        )

        # Wait for door selection
        self.wait_for_player_select_block(
            "Door",
            on_select=lambda: self.continue_door_placement()
        )

    def continue_door_placement(self):
        """Continue door placement after selection"""
        # Highlight door position
        door_pos = self.house_position + Vector3(2, 0, 0)  # Front center
        self.highlight_single_block(door_pos, color="green")

        self.show_hint("Place door in the highlighted spot")

        self.wait_for_player_place_block(
            position=door_pos,
            block_type="Door",
            on_complete=self.step_4_roof
        )

    def step_4_roof(self):
        """Step 4: Auto-place roof (demonstrate what's possible)"""
        self.narrator("Perfect! Let me add a roof for you.")

        # Game places roof automatically (shows advanced result)
        self.auto_place_roof(
            area=(5, 5),
            block_type="Roof_Thatch",
            style="pitched"
        )

        # Camera angle to show completed house
        camera.animate_orbit_around(
            target=self.house_position,
            duration=3,
            on_complete=self.step_5_completion
        )

    def step_5_completion(self):
        """Step 5: Celebrate and transition to freeform"""
        self.narrator("Congratulations! You built your first house!")

        # Show achievement
        show_achievement("First Home", "Built your first structure")

        # Spawn inspiration nearby
        self.spawn_example_buildings(
            count=3,
            distance=20,
            types=["barn", "tower", "cottage"]
        )

        self.narrator(
            "I've placed some other buildings nearby for inspiration. "
            "You can modify your house, copy these examples, or build something completely new!"
        )

        # Unlock all starter tools
        player.unlock_tools(["Copy", "Paste", "Undo"])
        show_hint("New tools available! Check your toolbar.")

        # Tutorial complete - player has freedom
        self.tutorial_complete = True

    # Helper methods
    def narrator(self, text):
        """Show narrator text (non-intrusive)"""
        ui.show_narrator_text(text, duration=5, position="bottom")

    def show_hint(self, text, point_to=None):
        """Show contextual hint"""
        hint = UIHint(text)
        if point_to:
            hint.add_arrow_pointing_to(point_to)
        hint.show()

    def wait_for_player_fill_area(self, area, block_type, on_complete):
        """Wait for player to fill area with blocks"""
        target_blocks = area[0] * area[1]

        def check_completion():
            placed_blocks = count_blocks_in_area(area, block_type)
            progress = placed_blocks / target_blocks

            # Show progress
            ui.show_progress_bar(progress)

            if progress >= 1.0:
                on_complete()

        # Check every time player places block
        event_system.on("block_placed", check_completion)
```

**Key Techniques**:
1. **Auto-select tools/blocks initially**: No menu navigation during first steps
2. **Visual guidance**: Highlight exactly where to build
3. **Introduce complexity gradually**: First no menu, then simplified menu, then full menu
4. **Demonstrate advanced features**: Game places roof to show what's possible
5. **Celebrate completion**: Player finishes tutorial with completed build (not just knowledge)
6. **Immediate inspiration**: Example builds nearby for next steps

### Pattern 4: Optional Challenge System

Provide direction for goal-oriented players without constraining creative players.

```python
class OptionalChallengeSystem:
    def __init__(self):
        self.available_challenges = []
        self.active_challenges = []  # Player opted in
        self.completed_challenges = []

        # Define challenge types
        self.challenge_types = {
            "build": BuildChallenge,
            "theme": ThemeChallenge,
            "timed": TimedChallenge,
            "constraint": ConstraintChallenge
        }

        # Populate initial challenges
        self.generate_daily_challenges()

    def generate_daily_challenges(self):
        """Generate fresh challenges"""
        self.available_challenges = [
            BuildChallenge(
                name="Two-Story House",
                description="Build a house with two floors",
                reward="cosmetic_roof_variant",
                difficulty="easy"
            ),
            ThemeChallenge(
                name="Medieval Village",
                description="Create a medieval-themed building",
                reward="medieval_decoration_pack",
                difficulty="medium"
            ),
            TimedChallenge(
                name="Speed Builder",
                description="Build a shelter in 10 minutes",
                reward="speed_builder_badge",
                difficulty="hard"
            ),
            ConstraintChallenge(
                name="Minimalist",
                description="Build something using only 3 block types",
                reward="minimalist_palette",
                difficulty="medium"
            )
        ]

    def present_challenges_to_player(self):
        """Show optional challenge menu"""
        menu = ChallengeMenu()
        menu.title = "Optional Challenges"
        menu.subtitle = "Ignore these if you prefer freeform building!"

        for challenge in self.available_challenges:
            menu.add_challenge(
                challenge,
                on_accept=lambda c: self.activate_challenge(c),
                on_dismiss=lambda c: None  # Do nothing, player declined
            )

        # Can be closed immediately
        menu.closeable = True
        menu.show()

    def activate_challenge(self, challenge):
        """Player opts into challenge"""
        self.active_challenges.append(challenge)

        # Show non-intrusive reminder
        ui.show_challenge_tracker(challenge)

        # Set up validation
        challenge.start()
        challenge.on_complete = lambda: self.complete_challenge(challenge)

    def complete_challenge(self, challenge):
        """Challenge completed"""
        self.active_challenges.remove(challenge)
        self.completed_challenges.append(challenge)

        # Celebrate
        show_achievement(
            title=f"Challenge Complete: {challenge.name}",
            description=challenge.description
        )

        # Award reward
        self.grant_reward(challenge.reward)

    def grant_reward(self, reward_id):
        """Grant cosmetic or content reward"""
        reward = RewardDatabase.get(reward_id)

        if reward.type == "cosmetic":
            player.unlock_cosmetic(reward)
            show_message(f"Unlocked: {reward.name} (cosmetic)")

        elif reward.type == "blocks":
            player.unlock_blocks(reward.blocks)
            show_message(f"Unlocked: {len(reward.blocks)} new blocks!")

        elif reward.type == "badge":
            player.add_badge(reward)
            show_message(f"Earned badge: {reward.name}")

class BuildChallenge:
    """Open-ended build challenge"""
    def __init__(self, name, description, reward, difficulty):
        self.name = name
        self.description = description
        self.reward = reward
        self.difficulty = difficulty
        self.criteria = None

    def start(self):
        """Begin tracking this challenge"""
        # For "Two-Story House"
        self.criteria = {
            "has_two_floors": False,
            "has_walls": False,
            "has_roof": False
        }

        # Check periodically (not every frame)
        event_system.on_interval(5.0, self.check_completion)

    def check_completion(self):
        """Check if build meets criteria"""
        player_build = get_player_build_area()

        # Detect two floors (floor blocks at y=0 and y=3+)
        has_floor_1 = player_build.count_blocks_at_height(0, "Floor") > 9
        has_floor_2 = player_build.count_blocks_at_height(3, "Floor") > 9
        self.criteria["has_two_floors"] = has_floor_1 and has_floor_2

        # Detect walls
        self.criteria["has_walls"] = player_build.count_blocks("Wall") > 20

        # Detect roof
        self.criteria["has_roof"] = player_build.count_blocks("Roof") > 9

        # All criteria met?
        if all(self.criteria.values()):
            self.on_complete()

class TimedChallenge:
    """Time-limited challenge"""
    def __init__(self, name, description, reward, difficulty):
        self.name = name
        self.description = description
        self.reward = reward
        self.difficulty = difficulty
        self.time_limit = 600  # 10 minutes
        self.start_time = None

    def start(self):
        """Start timer"""
        self.start_time = time.time()

        # Show countdown timer
        ui.show_timer(self.time_limit)

        # Check completion
        event_system.on_interval(1.0, self.check_time)

    def check_time(self):
        """Check timer and completion"""
        elapsed = time.time() - self.start_time
        remaining = self.time_limit - elapsed

        if remaining <= 0:
            # Time's up
            show_message("Time's up! Challenge failed (but you can keep building!)")
            self.on_fail()

        elif self.check_shelter_complete():
            # Completed in time
            self.on_complete()

    def check_shelter_complete(self):
        """Check if player built a shelter"""
        player_build = get_player_build_area()

        # Simple shelter: walls + roof
        has_walls = player_build.count_blocks("Wall") >= 12
        has_roof = player_build.count_blocks("Roof") >= 9

        return has_walls and has_roof
```

**Challenge Design Principles**:
1. **Always optional**: Player can decline or ignore
2. **Non-intrusive tracking**: Small UI element, not full-screen
3. **Cosmetic rewards**: Never gate content behind challenges
4. **Variety**: Different challenge types for different players
5. **Fails gracefully**: Time-up or abandon = no penalty

### Pattern 5: Example Build Spawning

Provide inspiration through pre-made examples players can study and modify.

```python
class ExampleBuildSystem:
    def __init__(self):
        self.example_library = []
        self.spawned_examples = []

        # Load pre-made builds
        self.load_example_builds()

    def load_example_builds(self):
        """Load library of example builds"""
        self.example_library = [
            ExampleBuild(
                name="Starter Cottage",
                file="cottage_5x5.build",
                size=(5, 5, 4),
                blocks_used=["Wood_Floor", "Wood_Wall", "Thatch_Roof", "Door", "Window"],
                difficulty="easy",
                tags=["house", "starter", "simple"]
            ),
            ExampleBuild(
                name="Stone Tower",
                file="tower_3x3x12.build",
                size=(3, 3, 12),
                blocks_used=["Stone_Floor", "Stone_Wall", "Wood_Door", "Glass_Window"],
                difficulty="medium",
                tags=["tower", "vertical", "defense"]
            ),
            ExampleBuild(
                name="Medieval Barn",
                file="barn_8x6.build",
                size=(8, 6, 5),
                blocks_used=["Wood_Floor", "Wood_Wall", "Thatch_Roof", "Large_Door"],
                difficulty="medium",
                tags=["barn", "medieval", "large"]
            ),
            # ... more examples
        ]

    def spawn_examples_near_player(self, count=3, distance=20, filter_tags=None):
        """Spawn example builds around player for inspiration"""
        player_pos = player.get_position()

        # Filter examples by tags if specified
        available = self.example_library
        if filter_tags:
            available = [e for e in available if any(tag in e.tags for tag in filter_tags)]

        # Select random examples
        selected = random.sample(available, min(count, len(available)))

        # Spawn in circle around player
        angle_step = 360 / count
        for i, example in enumerate(selected):
            angle = angle_step * i
            offset = Vector3(
                math.cos(math.radians(angle)) * distance,
                0,
                math.sin(math.radians(angle)) * distance
            )
            spawn_pos = player_pos + offset

            # Spawn the build
            spawned = self.spawn_example_build(example, spawn_pos)
            self.spawned_examples.append(spawned)

            # Add info sign
            self.add_info_sign(spawned, example)

    def spawn_example_build(self, example, position):
        """Spawn a build from file"""
        build_data = load_build_file(example.file)

        # Place blocks
        for block_info in build_data:
            block_pos = position + block_info.relative_position
            world.place_block(
                block_type=block_info.type,
                position=block_pos,
                rotation=block_info.rotation
            )

        return SpawnedExample(example, position)

    def add_info_sign(self, spawned_example, example):
        """Add sign with info about the example"""
        sign_pos = spawned_example.position + Vector3(0, 0, -2)

        sign = world.create_info_sign(sign_pos)
        sign.set_text(f"{example.name}\nClick to copy")
        sign.on_click = lambda: self.offer_copy_example(spawned_example, example)

    def offer_copy_example(self, spawned_example, example):
        """Let player copy example as template"""
        menu = ActionMenu()
        menu.add_action(
            "Study",
            description="Walk around and examine this build",
            action=lambda: camera.focus_on(spawned_example.position)
        )
        menu.add_action(
            "Copy as Template",
            description="Create editable copy to modify",
            action=lambda: self.copy_as_template(spawned_example, example)
        )
        menu.add_action(
            "Remove",
            description="Remove this example to clear space",
            action=lambda: self.remove_example(spawned_example)
        )
        menu.show()

    def copy_as_template(self, spawned_example, example):
        """Create editable copy player can modify"""
        # Copy build to new location
        copy_offset = Vector3(10, 0, 0)  # Offset from original
        new_pos = spawned_example.position + copy_offset

        self.spawn_example_build(example, new_pos)

        # Give player copy tool and focus camera
        player.select_tool("Copy")
        camera.focus_on(new_pos)

        show_hint(
            "Template copied! Modify it to make it your own. "
            "Use Copy tool to duplicate parts you like."
        )

class ExampleBuild:
    """Data for a pre-made example build"""
    def __init__(self, name, file, size, blocks_used, difficulty, tags):
        self.name = name
        self.file = file  # Path to build file
        self.size = size  # (width, depth, height)
        self.blocks_used = blocks_used
        self.difficulty = difficulty
        self.tags = tags
```

**Example Build Principles**:
1. **Variety**: Different sizes, styles, difficulties
2. **Accessible**: Use starter blocks primarily
3. **Modifiable**: Players can copy and modify, not just view
4. **Removable**: Players can delete examples to clear space
5. **Educational**: Examples teach techniques (vertical building, symmetry, etc.)

---

## Common Pitfalls

### Pitfall 1: Too Much Freedom (Analysis Paralysis)

**The Mistake**:
```python
# ❌ Overwhelming player with infinite options
def start_game():
    player.unlock_all_blocks(200+)
    player.unlock_all_tools(15)
    player.unlock_all_colors(16_777_216)  # Full RGB spectrum
    world.spawn_player_in_infinite_empty_world()
```

**Why It Fails**:
- Blank canvas + infinite options = paralysis
- No constraints = no creative direction
- Players don't know where to start
- "What should I build?" becomes insurmountable question

**Real-World Example**:
Game launches with "ultimate creative freedom" - 500+ blocks, infinite world, all tools unlocked. 70% of players quit within 10 minutes. Exit surveys: "Too overwhelming", "Didn't know what to do", "Too many options".

**The Fix**:
```python
# ✅ Meaningful constraints spark creativity
def start_game():
    # Start with minimal palette
    player.unlock_blocks(STARTER_SET)  # 5-10 blocks

    # Start in constrained space
    world.spawn_player_on_small_island(size=50x50)

    # Provide examples for inspiration
    world.spawn_example_builds(count=3, nearby=True)

    # Progressive expansion
    player.on_mastery_milestone(unlock_more_blocks)
```

**Detection**:
- Player places <10 blocks before quitting
- High abandonment rate in first 30 minutes
- Feedback: "Didn't know what to build"

### Pitfall 2: Tutorial Screens (Passive Learning)

**The Mistake**:
```python
# ❌ Text-heavy tutorial that teaches nothing
def tutorial():
    screens = [
        "Welcome! Press WASD to move",
        "Press Q to open menu",
        "Press E to place blocks",
        "Press R to rotate",
        "Press T to open tools",
        # ... 10 more screens
        "Now go build something!"
    ]

    for screen in screens:
        show_text_screen(screen)
        wait_for_player_click_ok()

    # Player: "I already forgot everything"
```

**Why It Fails**:
- Passive reading doesn't create muscle memory
- Players forget information immediately
- No context for why mechanics matter
- Boring and demotivating

**Real-World Example**:
Tutorial is 15 text screens explaining controls. Players skip through quickly to "get to the game". Then they're dropped into empty world and don't remember any controls. Ask for tutorial again or quit.

**The Fix**:
```python
# ✅ Tutorial through guided building
def tutorial():
    narrator("Let's build a house together!")

    # Players learn by doing
    guide_player_place_floor()      # Learns: Placement
    guide_player_place_walls()      # Learns: Selection
    guide_player_place_door()       # Learns: Block menu
    auto_place_roof()               # Shows: What's possible

    # Result: Finished house + working knowledge
    show_example_builds_for_inspiration()
```

**Detection**:
- Players skip tutorial screens rapidly
- Players don't use taught mechanics after tutorial
- Request for "how do I...?" after tutorial completes

### Pitfall 3: No Onboarding for Directionless Players

**The Mistake**:
```python
# ❌ Assumes all players have ideas ready
def start_game():
    tutorial()  # Just explains controls
    spawn_player_in_empty_world()
    # No inspiration, no examples, no prompts
```

**Why It Fails**:
- Not everyone is naturally creative
- Empty canvas is terrifying for many players
- "Build anything!" is paralyzing without direction
- Goal-oriented players need objectives

**Real-World Example**:
Sandbox game with no guided content. Creative players (20% of audience) thrive and create amazing things. Other 80% of players build a few blocks, get bored, quit. Game succeeds with niche audience but fails to reach mainstream.

**The Fix**:
```python
# ✅ Provide inspiration and optional direction
def start_game():
    # Guided tutorial
    tutorial_as_first_build()

    # Inspiration nearby
    spawn_example_builds(count=5, variety=True)

    # Optional challenges
    present_daily_challenges(skippable=True)

    # Community showcase
    show_featured_builds(from_players=True)

    # Players have: Examples to copy, challenges to try, or freeform if desired
```

**Detection**:
- Players build <100 blocks before quitting
- Feedback: "Didn't know what to build"
- Low engagement despite good tools

### Pitfall 4: Forced Objectives (Constraining Creatives)

**The Mistake**:
```python
# ❌ Forces players to complete objectives before creativity
def start_game():
    # Must complete tutorial missions
    force_mission("Build a house (exactly as shown)")
    force_mission("Build a barn (exactly as shown)")
    force_mission("Build a tower (exactly as shown)")

    # Only after 10 forced missions:
    unlock_creative_mode()
```

**Why It Fails**:
- Creative players want freedom immediately
- Forced objectives feel like chores
- "Build exactly as shown" kills creativity
- Delays the core appeal (creative expression)

**Real-World Example**:
Building game requires 2 hours of tutorial missions before unlocking "freeform mode". Creative players (the target audience) quit in frustration. Reviews: "Just let me build!", "Why can't I skip this?", "Tutorial is longer than the actual game".

**The Fix**:
```python
# ✅ Make objectives optional
def start_game():
    # Quick tutorial
    tutorial_as_first_build()  # 5-10 minutes

    # Then: Full freedom
    unlock_creative_mode()

    # But also: Optional objectives for those who want them
    optional_challenges = [
        "Try building a 2-story house (optional)",
        "This week's theme: Medieval castles (optional)",
        "Campaign mode: 10 building lessons (optional separate mode)"
    ]

    present_optional_menu(optional_challenges, can_ignore=True)
```

**Detection**:
- Creative players quit during forced tutorial
- Feedback: "Too restrictive", "Just let me build"
- High abandonment before reaching freeform mode

### Pitfall 5: No Progressive Revelation (Feature Overload)

**The Mistake**:
```python
# ❌ Show all features at once
def open_build_menu():
    menu = Menu()

    # 200 blocks in one flat list
    menu.add_section("All Blocks", all_blocks_alphabetically(200))

    # 15 tools with no categories
    menu.add_section("All Tools", all_tools_in_one_list(15))

    # Advanced options visible immediately
    menu.add_section("Advanced", [
        "Boolean operations",
        "Procedural generation",
        "Scripting interface",
        "Custom shaders"
    ])

    # Player: "This is overwhelming"
```

**Why It Fails**:
- Cognitive overload from too many options
- Can't find basic tools in sea of advanced features
- Interface complexity scares away newcomers
- Advanced users also suffer (can't find anything)

**Real-World Example**:
Block menu has 300+ blocks in alphabetical list. Players can't find "Door" because they're looking for "Wood Door" but it's listed as "Door_Wood_Oak". Takes 5 minutes to find basic items. Players use only 5-10 blocks they can find quickly, rest are lost in menu hell.

**The Fix**:
```python
# ✅ Progressive revelation through use
class ProgressiveUI:
    def __init__(self):
        self.visible_blocks = STARTER_BLOCKS  # 5-10 initially
        self.visible_tools = STARTER_TOOLS    # 3 initially
        self.show_advanced = False

    def open_build_menu(self):
        menu = Menu()

        # Simple initially
        if len(self.visible_blocks) < 20:
            menu.show_simple_grid(self.visible_blocks)
        else:
            # Expand to categories only when needed
            menu.show_categorized(self.visible_blocks)

        # Hide advanced features until player is ready
        if self.show_advanced:
            menu.add_section("Advanced", advanced_tools)

    def on_player_action_count(self, count):
        if count > 100:
            self.reveal_more_blocks(TIER_2_BLOCKS)
        if count > 500:
            self.show_advanced = True
```

**Detection**:
- Players use only small subset of available blocks
- Feedback: "Can't find anything", "Too complicated"
- Time spent navigating menus > time spent building

### Pitfall 6: No Undo/Redo (Punishing Experimentation)

**The Mistake**:
```python
# ❌ No undo - mistakes are permanent
def place_block(position, block_type):
    world.set_block(position, block_type)
    # That's it - no undo buffer

# Player places wrong block → must manually delete → frustrating
# Player experiments with design → can't revert → stops experimenting
```

**Why It Fails**:
- Creativity requires safe failure
- Fear of permanent mistakes reduces experimentation
- Players are conservative (don't try new things)
- Accidental mistakes are rage-inducing

**Real-World Example**:
Player spends 2 hours on detailed build. Accidentally selects delete tool instead of place tool. One click deletes 50 blocks. No undo. Player quits game in frustration. Lost customer + bad review.

**The Fix**:
```python
# ✅ Robust undo/redo system
class UndoSystem:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_depth = 100

    def record_action(self, action):
        """Record action to undo stack"""
        self.undo_stack.append(action)

        # Clear redo stack (new action invalidates redo)
        self.redo_stack.clear()

        # Limit stack size
        if len(self.undo_stack) > self.max_undo_depth:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            return

        action = self.undo_stack.pop()
        action.undo()

        # Move to redo stack
        self.redo_stack.append(action)

    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            return

        action = self.redo_stack.pop()
        action.redo()

        # Move back to undo stack
        self.undo_stack.append(action)

# Usage:
def place_block(position, block_type):
    action = PlaceBlockAction(position, block_type)
    action.execute()
    undo_system.record_action(action)

# Hotkeys
input.bind("Ctrl+Z", undo_system.undo)
input.bind("Ctrl+Y", undo_system.redo)
```

**Critical**: Undo/redo is not optional for creative tools. It's as essential as save functionality.

### Pitfall 7: No Example Builds (Blank Canvas Problem)

**The Mistake**:
```python
# ❌ No inspiration provided
def start_game():
    tutorial()  # Explains controls only
    spawn_player_in_empty_world()
    # No examples, no templates, no inspiration
    # Player: "Now what?"
```

**Why It Fails**:
- Most players need inspiration to start creating
- Empty world = no reference for what's possible
- Players don't know quality standards or style options
- Underestimate what can be built with available tools

**Real-World Example**:
Sandbox game tutorial ends with "Now build something amazing!". Players look around empty world, place a few random blocks, think "this looks nothing like a house", give up. They never saw what's possible, so they don't know how to start.

**The Fix**:
```python
# ✅ Provide varied examples
def start_game():
    tutorial_build_house()  # Player builds guided house

    # Spawn varied examples nearby
    spawn_examples = [
        "cottage_small",     # Similar to tutorial (confidence)
        "tower_vertical",    # Different style (vertical vs horizontal)
        "barn_large",        # Larger scale (aspiration)
        "bridge_creative",   # Creative use of tools (inspiration)
        "decoration_cozy"    # Interior design (different focus)
    ]

    for example in spawn_examples:
        spawn_example_build(example, near_player=True, with_sign=True)

    narrator("Here are some examples. Study them, copy them, or build something totally new!")
```

**Example Build Checklist**:
- [ ] Variety of sizes (small, medium, large)
- [ ] Variety of styles (simple, detailed, creative)
- [ ] Variety of purposes (house, tower, decoration, functional)
- [ ] Use primarily starter blocks (so player can replicate)
- [ ] Some advanced builds (show what's possible)

### Pitfall 8: Survival Mode Without Creative Mode

**The Mistake**:
```python
# ❌ Only survival mode available
def start_game():
    mode = "survival"  # Only option

    # Players must gather every resource
    # Takes 5 hours to gather materials
    # Experimentation is expensive (wasted resources)
    # Creative players frustrated
```

**Why It Fails**:
- Resource gathering delays building (the core appeal)
- Experimentation wastes limited resources
- Creative players want to build, not grind
- Can't prototype designs before committing resources

**Real-World Example**:
Building game is survival-only. Takes 10 hours of mining to gather resources for large build. Player builds it, realizes design doesn't work, needs to tear down and rebuild. Can't afford to do this (would take another 10 hours mining). Player settles for mediocre build they don't like. Frustrated and quits.

**The Fix**:
```python
# ✅ Offer both modes
def start_game():
    menu = ModeSelectionMenu()

    menu.add_mode(
        "Creative Mode",
        description="Unlimited resources, focus on building",
        audience="Creative players, designers, planners"
    )

    menu.add_mode(
        "Survival Mode",
        description="Gather resources, meaningful constraints",
        audience="Goal-oriented players, challenge seekers"
    )

    menu.add_mode(
        "Hybrid",
        description="Switch between modes (plan in creative, build in survival)",
        audience="Both playstyles"
    )

    selected_mode = menu.show()

    # Let players switch later if they want
    allow_mode_switching = True
```

**Best Practice**: Offer both modes. Let players choose their preferred playstyle. Allow switching (creative for planning, survival for execution).

### Pitfall 9: Everything Unlocked vs Grindy Unlocking

**The Mistake (Option A: Everything Unlocked)**:
```python
# ❌ Everything available immediately
def start_game():
    player.unlock_all_blocks(200+)
    player.unlock_all_tools(15)
    # No progression, no sense of growth
```

**The Mistake (Option B: Grindy Unlocking)**:
```python
# ❌ Tedious grind to unlock basics
def start_game():
    player.unlock_blocks(["Dirt"])  # Only dirt block initially

    # Must place 1000 blocks to unlock Wood
    # Must place 5000 blocks to unlock Stone
    # Must place 10000 blocks to unlock Door
    # Basics locked behind grind
```

**Why Both Fail**:
- Everything unlocked: No progression, overwhelming, nothing to work toward
- Grindy unlocking: Frustrating, gates basic functionality, feels like mobile game

**Real-World Example**:
Game unlocks 1 new block per 1000 blocks placed. "Door" unlocks after 15,000 blocks placed. Players build huge amounts of placeholder blocks just to unlock door. Ruins builds, wastes time, feels like grind. Reviews: "Artificial progression", "Just unlock building tools".

**The Fix**:
```python
# ✅ Mastery-based progressive unlocking
class MasteryUnlocking:
    def __init__(self):
        # Tier 1: Starter set (complete buildings possible)
        self.tier_1 = {
            "blocks": ["Wood_Floor", "Wood_Wall", "Wood_Roof", "Door", "Window"],
            "unlock_at": 0  # Immediate
        }

        # Tier 2: Variety (alternative materials)
        self.tier_2 = {
            "blocks": ["Stone_Floor", "Stone_Wall", "Stairs", "Fence"],
            "unlock_at": 50  # After 50 actions (15 minutes play)
        }

        # Tier 3: Advanced (decorative and functional)
        self.tier_3 = {
            "blocks": ["Furniture", "Decorations", "Lights", "Advanced_Materials"],
            "unlock_at": 200  # After 200 actions (1 hour play)
        }

        # Tier 4: Expert (rare and special)
        self.tier_4 = {
            "blocks": ["Special_Effects", "Logic_Gates", "Rare_Materials"],
            "unlock_at": 1000  # After mastery (5+ hours play)
        }
```

**Principles**:
- Starter set allows complete builds (never lock essentials)
- Progressive unlocking adds variety (not core functionality)
- Mastery-based (through play, not grind)
- Fast initially (tier 2 in 15 minutes), slower later (tier 4 in 5+ hours)

### Pitfall 10: No Community Showcase (Isolation)

**The Mistake**:
```python
# ❌ No way to share creations
def game_features():
    features = [
        "Build anything you want",
        "Hundreds of blocks",
        "Creative tools"
    ]
    # No sharing, no gallery, no community
    # Players build in isolation
```

**Why It Fails**:
- Creating without audience reduces motivation
- Players don't see what others built (no inspiration)
- Can't learn from community (techniques, styles)
- No social aspect (multiplayer/async)

**Real-World Example**:
Amazing building game with no sharing features. Players create incredible builds... then close the game and that's it. No one sees their work. After initial projects, motivation drops. Players quit because "no one will see what I make anyway".

**The Fix**:
```python
# ✅ Community showcase and sharing
class CommunityFeatures:
    def __init__(self):
        self.sharing_enabled = True
        self.gallery_visible = True

    def enable_sharing(self):
        # In-game screenshot
        button_screenshot = "F12: Screenshot with metadata"

        # Share to gallery
        button_share = "Share to community gallery"

        # Export build file
        button_export = "Export as file (share anywhere)"

    def show_community_gallery(self):
        gallery = CommunityGallery()

        # Featured builds (curated)
        gallery.add_section("Featured This Week", featured_builds)

        # Recent builds (chronological)
        gallery.add_section("Recent Creations", recent_builds)

        # Popular builds (liked/rated)
        gallery.add_section("Most Popular", popular_builds)

        # Search/filter
        gallery.add_filter("Style", ["Medieval", "Modern", "Fantasy", "Sci-fi"])
        gallery.add_filter("Size", ["Small", "Medium", "Large", "Massive"])
        gallery.add_filter("Blocks Used", block_types)

        # For each build:
        # - View screenshot/video
        # - Download build file
        # - Like/favorite
        # - Comment

        gallery.show()
```

**Community Features Priority**:
1. **Must have**: Screenshot + share to built-in gallery
2. **Should have**: Download other players' builds to study/modify
3. **Nice to have**: Multiplayer collaborative building
4. **Optional**: External sharing (social media integration)

---

## Real-World Examples

### Example 1: Minecraft Creative Mode

**What It Does Right**:

```python
class MinecraftCreativeMode:
    """Pure sandbox with progressive revelation"""

    def start_game(self):
        # Start with ~40 basic blocks
        self.blocks_available = 40  # Not 400+ immediately

        # No resource constraints
        self.inventory = "unlimited"
        self.damage = "disabled"
        self.flight = "enabled"

        # Natural progression through exploration
        self.block_discovery = "explore world to see possibilities"

        # Community showcase
        self.inspiration = [
            "Realm featured builds",
            "Marketplace community creations",
            "YouTube/Twitch creator builds"
        ]

    def design_principles(self):
        # ✅ Constraint-based creativity
        # Limited block palette (compared to what's theoretically possible)
        # Block-based grid (constraint that enables creativity)

        # ✅ Progressive revelation
        # Blocks organized by categories
        # Creative inventory shows all, but organized intelligently

        # ✅ Strong community
        # Built-in Realms sharing
        # Massive community content ecosystem

        # ✅ Optional structure
        # Creative mode = pure freedom
        # Survival mode = guided by resource gathering
        # Adventure maps = structured challenges
```

**Why It Works**:
- Started simple (Alpha: <100 blocks), expanded gradually
- Grid-based building = constraint that sparks creativity
- Strong community ecosystem provides inspiration
- Modes for different playstyles (creative, survival, adventure)

### Example 2: Factorio (Guided Sandbox)

**What It Does Right**:

```python
class FactorioGuidedSandbox:
    """Sandbox with meaningful constraints and goals"""

    def start_game(self):
        # Tutorial through missions
        self.tutorial = "campaign missions teaching mechanics"

        # Start with minimal buildings
        self.available_buildings = 5  # Expands to 100+

        # Technology tree provides structure
        self.progression = "research unlocks new buildings"

        # Constraints spark creativity
        self.constraints = [
            "Resource scarcity (must find and mine)",
            "Space constraints (factory layout matters)",
            "Logistics challenges (how to move items)",
            "Power management (energy requirements)"
        ]

    def optional_objectives(self):
        # Optional goal (can be ignored)
        self.primary_goal = "Launch rocket (optional)"

        # Player-created goals emerge
        self.emergent_goals = [
            "Design efficient factory layout",
            "Optimize production chains",
            "Create aesthetic designs",
            "Automate everything"
        ]

    def design_principles(self):
        # ✅ Meaningful constraints
        # Resources must be gathered and transported
        # Space is limited (forces layout optimization)
        # Power management adds complexity

        # ✅ Progressive unlocking makes sense
        # Research tree teaches systems gradually
        # Each unlock adds complexity thoughtfully

        # ✅ Tools vs goals balance
        # Primary goal (rocket) is optional
        # Most goals emerge from player (efficiency, aesthetics)

        # ✅ Guided freedom
        # Technology tree provides direction
        # How you achieve goals is completely open-ended
```

**Why It Works**:
- Resource/space/power constraints create interesting problems
- Technology tree provides gentle direction without forcing path
- Goals are optional (can play purely creatively)
- Extremely deep system mastery curve

### Example 3: Terraria (Balanced Guided Sandbox)

**What It Does Right**:

```python
class TerrariaGuidedSandbox:
    """Blend of sandbox freedom and structured content"""

    def start_game(self):
        # Very brief tutorial
        self.tutorial = "Guide NPC gives hints"

        # Start with minimal tools
        self.starting_tools = ["Pickaxe", "Axe", "Sword"]

        # Progressive through exploration
        self.progression = [
            "Explore world → Find materials",
            "Craft better tools → Access new areas",
            "Defeat bosses → Unlock new content"
        ]

        # Mix of freedom and structure
        self.goals = {
            "soft_goals": [
                "Build houses for NPCs (NPC moves in when conditions met)",
                "Explore underground (hints about what's below)",
                "Boss hints (Eye of Cthulhu looks at you from distance)"
            ],
            "optional_progression": [
                "Can fight bosses in any order (some harder than others)",
                "Can explore anywhere (danger level varies)",
                "Can build anywhere (complete freedom)"
            ]
        }

    def design_principles(self):
        # ✅ Gentle guidance through NPCs
        # Guide NPC gives contextual hints
        # Not forced, just helpful suggestions

        # ✅ Building has purpose
        # Build houses → NPCs move in → NPCs sell items
        # Building is integrated with progression

        # ✅ Exploration drives discovery
        # Find materials → Craft new items
        # Find bosses → Optional challenges

        # ✅ Freedom within structure
        # Boss order is flexible
        # Building style is completely free
        # Progression has multiple paths
```

**Why It Works**:
- Survival mode adds meaning to building (shelter from monsters)
- NPC system rewards building (functional purpose)
- Boss hints provide direction without forcing
- Multiple progression paths (mining, building, combat, exploration)

### Example 4: Cities: Skylines (Structured Sandbox)

**What It Does Right**:

```python
class CitiesSkylinesStructuredSandbox:
    """Progressive unlocking through city growth"""

    def start_game(self):
        # Tutorial through first city
        self.tutorial = "Build first neighborhood (guided)"

        # Start with basics
        self.available_buildings = [
            "Residential_Zone",
            "Commercial_Zone",
            "Industrial_Zone",
            "Roads",
            "Power_Plant",
            "Water_Pump"
        ]  # ~10 building types initially

        # Progressive unlocking through city growth
        self.progression = "city population → unlock buildings"

        # Milestones provide structure
        self.milestones = {
            500: "Unlock schools and fire stations",
            1000: "Unlock healthcare and police",
            5000: "Unlock parks and tourism",
            # ... up to 100,000+
        }

    def design_principles(self):
        # ✅ Progression teaches complexity
        # Start simple (just zones + roads)
        # Gradually add services (police, fire, health)
        # Then advanced systems (tourism, specialization)

        # ✅ Natural unlocking
        # City needs drive unlocks (traffic → need metros)
        # Population milestones feel earned

        # ✅ Sandbox within structure
        # Must reach milestones (structure)
        # How you design city is free (sandbox)

        # ✅ Scenario mode
        # Pre-built cities with challenges
        # Adds variety for goal-oriented players
```

**Why It Works**:
- Population-based unlocking feels natural (bigger city needs more services)
- Complexity scales with player understanding
- Scenarios add structured challenges for variety
- Unlimited creative freedom in how you design city layout

### Example 5: Kerbal Space Program (Complex Structured Sandbox)

**What It Does Right**:

```python
class KerbalSpaceProgramStructuredSandbox:
    """Multiple modes for different playstyles"""

    def game_modes(self):
        self.modes = {
            "sandbox": {
                "description": "All parts unlocked, unlimited budget",
                "audience": "Experienced players, creative designers",
                "progression": None
            },
            "science": {
                "description": "Unlock parts through science collection",
                "audience": "Mixed (exploration + unlocking)",
                "progression": "exploration → science → unlocks"
            },
            "career": {
                "description": "Budget constraints + mission contracts",
                "audience": "Goal-oriented, challenge seekers",
                "progression": "missions → money → parts → capabilities"
            }
        }

    def career_mode(self):
        # Structured progression through missions
        self.missions = [
            "Mission 1: Reach 10km altitude (basic rocket)",
            "Mission 2: Orbit Kerbin (orbital mechanics)",
            "Mission 3: Mun flyby (interplanetary travel)",
            "Mission 4: Mun landing (precision control)",
            # ... 50+ missions
        ]

        # Each mission teaches core concepts
        # Unlocks parts relevant to next challenges
        # Provides clear goals for goal-oriented players

    def design_principles(self):
        # ✅ Multiple modes for multiple audiences
        # Sandbox: Creative/experienced players
        # Science: Exploration-driven
        # Career: Mission-driven

        # ✅ Complex systems taught gradually
        # Physics-based gameplay needs teaching
        # Career missions teach through doing

        # ✅ Meaningful constraints in career
        # Budget constraints force creative solutions
        # Part limitations require innovation

        # ✅ Sandbox available for prototyping
        # Test designs in sandbox
        # Execute in career/science
```

**Why It Works**:
- Three modes serve three audiences perfectly
- Complex physics system needs structured teaching (career mode)
- Creative freedom available (sandbox mode)
- Constraints in career mode spark innovation (limited budget/parts)

---

## Cross-References

### Use This Skill WITH:
- **emergent-gameplay-design**: Sandbox tools create emergent player stories
- **modding-and-extensibility**: Community content extends sandbox possibilities
- **player-progression-systems**: If adding optional unlocking/achievements
- **tutorial-design-patterns**: Onboarding is critical for sandbox accessibility

### Use This Skill AFTER:
- **game-core-loops**: Understand what makes building loop engaging
- **ui-ux-patterns**: Menu design critical for tool accessibility
- **player-motivation**: Different players engage with sandbox differently

### Related Skills:
- **procedural-generation**: Generate worlds for players to build in
- **multiplayer-sandbox-coordination**: Collaborative building systems
- **level-editor-design**: Similar principles for user-generated content
- **creative-constraints**: How limitations enhance creativity

---

## Testing Checklist

### Onboarding Testing
- [ ] New player completes tutorial without confusion
- [ ] Tutorial ends with completed build (not just knowledge)
- [ ] Player knows what to do next after tutorial
- [ ] Tutorial is <10% of typical play session length
- [ ] Tutorial teaches by doing (not reading text)

### Constraint Testing
- [ ] Starter block set allows complete builds (5-10 blocks sufficient)
- [ ] Constraints spark creativity (players find innovative combinations)
- [ ] No arbitrary gates (all essential tools unlocked early)
- [ ] Progressive unlocking feels like discovery (not grind)
- [ ] Can create varied builds with starter set

### Objective Testing
- [ ] Creative players can ignore objectives completely
- [ ] Goal-oriented players have clear direction
- [ ] Challenges are optional (can be declined)
- [ ] Rewards don't gate content (cosmetic only)
- [ ] Mixed playstyles supported (can switch between modes)

### UI/UX Testing
- [ ] Can find basic tools within 5 seconds
- [ ] Block menu not overwhelming (categories, progressive revelation)
- [ ] Undo/redo works reliably (never lose work)
- [ ] Keyboard shortcuts for common actions
- [ ] Tools grouped logically (not alphabetically)

### Inspiration Testing
- [ ] Example builds visible from tutorial end
- [ ] Examples show variety (size, style, purpose)
- [ ] Can copy/modify examples as templates
- [ ] Community showcase of player builds
- [ ] Daily/weekly challenges for ideas

### Freedom Testing
- [ ] No forced objectives after tutorial
- [ ] Can build anything without restrictions
- [ ] No "you can't do that" messages
- [ ] Creative mode available (unlimited resources)
- [ ] Can switch between modes freely

### Engagement Testing
- [ ] Players build >100 blocks in first session
- [ ] <30% abandonment rate in first hour
- [ ] Players share creations (if sharing available)
- [ ] Players return for second session (retention)
- [ ] Average session length >30 minutes

### Edge Case Testing
- [ ] Empty world doesn't paralyze players (examples nearby)
- [ ] Directionless players get guidance (optional challenges)
- [ ] Overwhelmed players get simplified UI (progressive revelation)
- [ ] Advanced players access power tools (unlocked through mastery)
- [ ] Mistakes are reversible (undo/redo)

---

## Summary

Sandbox game design is about balancing freedom with accessibility. The core principles are:

1. **Constraints enable creativity** - Limited palette sparks innovation
2. **Progressive revelation** - Show complexity gradually through use
3. **Tutorial through doing** - Guide first build, don't explain controls
4. **Optional objectives** - Support both creative and goal-oriented players
5. **Inspiration over blank canvas** - Provide examples and community showcase
6. **Meaningful constraints** - Resource/space/time limits create interesting problems
7. **Safe experimentation** - Undo/redo enables creative risk-taking
8. **Multiple modes** - Pure sandbox, guided sandbox, or structured progression

**Common Pitfall Pattern**: The "Maximum Freedom Fallacy" - believing unlimited options maximize creativity. Reality: meaningful constraints spark creativity, unlimited options cause paralysis.

**Testing Red Flag**: High abandonment rate (>50%) in first 30 minutes = onboarding or blank canvas problem.

**Quick Win**: Add 3-5 example builds near player spawn = instant inspiration, dramatic retention increase.

Master these patterns and your sandbox will be accessible to newcomers while providing depth for experienced creators.
