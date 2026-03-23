
# Game UI Design

## Overview

This skill provides **The Game UI Integration Framework**: a systematic 4-dimension methodology for designing game interfaces that balance player immersion with information delivery, optimize for input devices, maintain aesthetic coherence, and preserve game performance.

**Core Principle**: Game UI faces unique constraints not present in productivity software. Players need critical information without breaking immersion, controls must work across multiple input methods (gamepad, keyboard+mouse, touch), visual design must match the game's genre and era, and UI rendering must never compromise frame rates. Success means players get the information they need while staying immersed in the game world.

**Platform Focus**: Multi-platform games (console, PC, mobile) with guidance on input method optimization and performance constraints.

## When to Use This Skill

**Use this skill when:**
- Designing game interfaces (HUDs, menus, pause screens, inventory systems)
- Building game UI systems (Unity UI, Unreal UMG, custom engines)
- Balancing immersion with usability in games
- Optimizing UI for different input methods (gamepad, keyboard+mouse, touch)
- Creating genre-appropriate UI (RPG stats, FPS minimalism, strategy overlays)
- Ensuring UI doesn't impact frame rate performance
- User mentions: "game UI", "HUD", "game menu", "inventory screen", "gamepad navigation"

**Don't use this skill for:**
- Non-game applications (use appropriate platform skill: web, mobile, desktop)
- Game engines themselves (use `lyra/ux-designer/desktop-software-design`)
- Simple 2D casual games with minimal UI (framework may be overkill)
- Game design mechanics (use game design resources)


## The Game UI Integration Framework

A systematic 4-dimension framework for evaluating and designing game interfaces:

1. **VISIBILITY VS IMMERSION** - Diegetic vs non-diegetic UI, minimal HUD, contextual display
2. **INPUT METHOD OPTIMIZATION** - Gamepad, keyboard+mouse, touch controls
3. **AESTHETIC COHERENCE** - UI matches game's visual style and genre
4. **PERFORMANCE IMPACT** - GPU acceleration, minimize draw calls, optimize assets

Evaluate game UI designs by examining each dimension systematically, identifying genre conventions, and ensuring designs balance information delivery with immersion.


## Dimension 1: VISIBILITY VS IMMERSION

**Purpose:** Balance the need for player information with immersion in the game world

Players need critical information (health, resources, objectives) but constant UI elements can break immersion and clutter the screen. The challenge is showing exactly what players need, exactly when they need it, without disrupting the game experience.

### Evaluation Questions

1. **Does the HUD show only essential information?**
   - Health in action games, stamina in combat
   - Resource counts in strategy/survival
   - Objective markers in open-world games
   - Avoid: Everything all the time

2. **Can players toggle UI visibility?**
   - Button to hide/show HUD (screenshot mode)
   - Allow players to customize HUD elements
   - Optional: Auto-hide UI after inactivity

3. **Is critical information always visible when needed?**
   - Health bar when taking damage
   - Ammo count when weapon drawn
   - Quest markers when near objectives
   - Never hide critical info when player needs it

4. **Does the UI design match the game's immersion level?**
   - Horror games: Minimal to no HUD (maximum fear)
   - Arcade games: Bold, prominent HUD (score focus)
   - Simulation games: Realistic, diegetic UI (cockpit displays)
   - RPGs: Stats-heavy UI (character progression focus)

### UI Integration Levels

**Diegetic UI (In-World)**

UI elements that exist within the game world - other characters can theoretically see them.

**Examples:**
- Dead Space: Health bar on Isaac's suit spine (glowing segments)
- Splinter Cell: Radar projection on Sam's wrist device
- Halo: Spartans see HUD inside their helmet visor
- Racing games: Speedometer on car dashboard
- Sci-fi games: Holographic computer interfaces

**Advantages:**
- Maximum immersion (feels part of world)
- Natural aesthetic fit (designed as game objects)
- Reinforces setting (futuristic tech, medieval scrolls)

**Challenges:**
- Harder to read (perspective, distance, lighting)
- Limited by world constraints (must fit in 3D space)
- Can be obscured by game elements (enemies, effects)
- Higher art production cost (3D models, animations)

**When to Use:**
- Horror/immersive sims (maximum immersion)
- VR games (UI must exist in 3D space)
- Strong artistic vision (The Division's AR overlays)

**Non-Diegetic UI (Overlay)**

Traditional HUD overlays - flat 2D elements on screen that don't exist in the game world.

**Examples:**
- Call of Duty: Health bar, minimap, ammo counter (corners)
- Assassin's Creed: Objective markers, health, minimap
- Mario: Coin count, lives, score (top of screen)
- Most games: Pause menu, settings, inventory screens

**Advantages:**
- Always readable (fixed position, no occlusion)
- Flexible positioning (corners, edges, center)
- Clear information hierarchy (consistent layout)
- Lower production cost (2D UI elements)

**Challenges:**
- Breaks immersion (reminds player it's a game)
- Can clutter screen (overlays on gameplay)
- Generic appearance if not styled well

**When to Use:**
- Most games (practical default)
- Competitive games (clarity over immersion)
- When information density is high (strategy, RPG)

**Spatial UI (World-Space)**

UI anchored to 3D space but not physically diegetic - floating elements in the world.

**Examples:**
- Floating health bars above enemies (MMORPGs)
- Objective markers floating at quest locations
- Damage numbers floating off enemies
- Waypoint beacons in the distance
- Username tags over players (multiplayer)

**Advantages:**
- Shows location context (where enemy is)
- Less screen clutter than full overlay
- Scalable with distance (far = smaller)

**Challenges:**
- Can clutter 3D space (many enemies = many bars)
- May obscure world details
- Depth sorting issues (render order)

**When to Use:**
- Open-world navigation (waypoints)
- Multiplayer identification (player names)
- Enemy status in action games

**Meta UI (Player-Aware)**

UI that acknowledges it's a game, breaking the fourth wall intentionally.

**Examples:**
- Tutorial hints: "Press X to jump"
- Achievement notifications: "Achievement Unlocked!"
- Loading screens with tips
- Fourth-wall-breaking games (Deadpool, Stanley Parable)

**Advantages:**
- Clear communication (no in-world justification needed)
- Good for teaching mechanics
- Can be humorous or stylistically interesting

**Challenges:**
- Breaks immersion completely
- Can feel intrusive if overused

**When to Use:**
- Tutorials and onboarding
- Achievements and progression systems
- Games with comedic or meta tone

### HUD Visibility Patterns

**Minimal HUD**

Show only absolutely essential information, hide everything else.

**Implementation:**
- **FPS:** Health (bottom-left), ammo (bottom-right), crosshair (center)
- **Horror:** No HUD, or only health when low
- **Racing:** Speed, position, lap time (minimal text)
- **Exploration:** Nothing, or only compass/objective marker

**Best For:**
- Immersive games (horror, narrative adventures)
- Games with simple mechanics (few systems to track)
- Cinematic experiences

**Avoid:**
- Complex RPGs (too many stats to hide)
- Strategy games (need resource/unit information)

**Contextual Display**

Show UI elements only when relevant, hide when not needed.

**Patterns:**
- **Health bar:** Appears when taking damage, fades after 3-5 seconds
- **Stamina bar:** Visible when sprinting/dodging, hidden when full
- **Ammo counter:** Shows when weapon drawn, hidden when holstered
- **Interaction prompts:** "Press E to open" appears near interactive objects
- **Quest notifications:** Appear when near objectives, hidden otherwise

**Timing Guidelines:**
- Appear: Immediate (0ms when condition met)
- Stay: 3-5 seconds after last update
- Fade: 500-1000ms smooth fade-out
- Reappear: Instant if condition triggers again

**Best For:**
- Most modern games (good balance)
- Reduces clutter without hiding critical info
- Keeps player focused on action

**Auto-Hide UI**

UI fades after period of no input, returns on interaction.

**Implementation:**
- Full HUD visible during gameplay
- After 3-5 seconds of no input → fade out (1s transition)
- Any input (movement, camera, action) → fade in (500ms)
- Critical info (low health) overrides auto-hide

**Best For:**
- Screenshot enthusiasts
- Cutscene transitions
- Exploration phases (walking, sightseeing)

**Configuration:**
- Allow players to adjust timing (3s, 5s, 10s, never)
- Allow players to exclude elements (always show health)

**HUD Toggle**

Button press completely hides/shows all HUD elements.

**Standard Mapping:**
- PC: F1 or H key (common convention)
- Console: D-pad down (or hold)
- Mobile: Two-finger tap (uncommon, conflicts with zoom)

**Behavior:**
- Instant toggle (no animation)
- Or smooth fade (500ms)
- Remember state across sessions (save preference)
- Show critical warnings even when hidden (very low health)

**Best For:**
- Screenshot mode
- Replay/spectator mode
- Accessibility (players with sensory issues)

### Genre Conventions

Understanding genre-specific HUD expectations helps meet player expectations.

**FPS (First-Person Shooter):**
- **Minimal HUD:** Health, ammo, crosshair, minimap (optional)
- **Position:** Health bottom-left, ammo bottom-right, minimap top-corner
- **Style:** Sleek, minimal, doesn't obscure center view
- **Examples:** Call of Duty, Halo, Titanfall
- **Reasoning:** Players need clear sightlines for shooting

**RPG (Role-Playing Game):**
- **Stats-Heavy:** Health, mana/stamina, XP bar, level, quest log, minimap
- **Position:** Multiple bars (top or bottom), minimap (top-right)
- **Style:** Ornate, fantasy or sci-fi themed borders
- **Examples:** Skyrim, World of Warcraft, The Witcher 3
- **Reasoning:** Character progression is core, players want stats

**Survival:**
- **Resource Tracking:** Health, hunger, thirst, temperature, inventory weight
- **Position:** Multiple gauges (bottom or side), icon-based status effects
- **Style:** Gritty, realistic, gauge-based
- **Examples:** Rust, ARK, Subnautica
- **Reasoning:** Resource management is core gameplay

**Racing:**
- **Speed & Position:** Speedometer, current position, lap time, minimap
- **Position:** Bottom center (speed), top center (position/time)
- **Style:** Sleek, digital, racing-aesthetic
- **Examples:** Forza, Gran Turismo, Mario Kart
- **Reasoning:** Need speed awareness, competitive position

**Strategy (RTS/4X):**
- **Resource Counts:** Gold, wood, food, population, minimap
- **Position:** Top bar (resources), bottom (unit selection), corner (minimap)
- **Style:** Information-dense, clear icons
- **Examples:** StarCraft, Civilization, Age of Empires
- **Reasoning:** Managing many systems simultaneously

**Horror:**
- **Minimal to None:** No HUD, or only health/ammo when critical
- **Style:** Diegetic (flashlight, wrist device), minimal overlay
- **Examples:** Resident Evil, Dead Space, Alien: Isolation
- **Reasoning:** UI breaks fear and tension

**Fighting Games:**
- **Health Bars:** Large horizontal bars top of screen, character portraits
- **Position:** Top (health), bottom (super meter, combo counter)
- **Style:** Bold, high-contrast, instantly readable
- **Examples:** Street Fighter, Mortal Kombat, Tekken
- **Reasoning:** Frame-perfect decisions need instant readability

**MOBA (Multiplayer Online Battle Arena):**
- **Complex HUD:** Health/mana, abilities (cooldowns), minimap, inventory, gold, team status
- **Position:** Bottom center (abilities), corners (minimap, score), side (inventory)
- **Style:** Dense, icon-heavy, color-coded teams
- **Examples:** League of Legends, Dota 2, Smite
- **Reasoning:** High information density for competitive play


## Dimension 2: INPUT METHOD OPTIMIZATION

**Purpose:** Ensure UI works seamlessly with the player's input device

Games support multiple input methods - gamepad (console), keyboard+mouse (PC), touch (mobile) - and UI navigation must feel native to each. Radial menus excel with analog sticks but fail with mouse. Keyboard hotkeys are fast for PC but impossible on gamepad. Touch requires large targets unavailable on small phone screens.

### Evaluation Questions

1. **Is the UI navigable with all supported input methods?**
   - Gamepad: D-pad or analog stick navigation
   - Keyboard+Mouse: Click + hotkeys
   - Touch: Tap targets 60px+ minimum

2. **Do input prompts show the correct button/key?**
   - Xbox controller: "Press A to continue"
   - PlayStation: "Press X to continue"
   - PC: "Press E to interact"
   - Dynamic switching when controller changes

3. **Are radial menus used effectively for gamepad?**
   - Analog stick selects from circular menu
   - 4-8 items maximum (more = hard to select)
   - Visual feedback (highlight selection)

4. **Are touch targets appropriately sized for mobile?**
   - Minimum 60x60px (larger than standard mobile UI)
   - Bottom half of screen (thumb zone)
   - Semi-transparent controls (don't obscure gameplay)

### Input Methods

**Gamepad Navigation**

Console players expect smooth navigation with controller, using analog sticks, D-pad, and face buttons.

**Radial Menus (Analog Stick)**

Circular menus ideal for analog stick selection - fast, intuitive, visually interesting.

**Specifications:**
- **Item Count:** 4-8 items maximum (8 = cardinal/diagonal directions)
- **Selection:** Analog stick direction (or flick gesture)
- **Deadzone:** Center 20% diameter (no selection)
- **Visual Feedback:** Highlighted slice, icon enlarges
- **Confirmation:** Release stick (auto-select) or press button
- **Center Option:** Cancel/back action

**Examples:**
- Mass Effect: Power wheel (8 abilities mapped to analog directions)
- Grand Theft Auto V: Weapon wheel (8 weapon categories)
- Red Dead Redemption: Item wheel (8 item slots)

**Best For:**
- Weapon/ability selection (fast, mid-combat)
- Item quick-access
- Context-sensitive actions (multiple options)

**Avoid:**
- More than 8 items (too precise, hard to select)
- Nested radial menus (confusing)
- Text-heavy options (labels should be icons)

**D-pad Menus (Cardinal Navigation)**

Traditional up/down/left/right navigation through menus and lists.

**List Navigation (Vertical):**
- **Up/Down:** Navigate items in list
- **A/X (confirm):** Select item
- **B/Circle (cancel):** Go back
- **Wrap:** Bottom → top (optional, helps long lists)

**Grid Navigation (2D):**
- **All Directions:** Navigate grid cells
- **Visual Indicator:** Highlight current cell (border, background)
- **Wrap:** Edges wrap to opposite side (optional)

**Tab Switching:**
- **L1/R1 (shoulder buttons):** Previous/next tab
- **Visual:** Tab bar at top, current tab highlighted
- **Examples:** Inventory tabs (Weapons, Armor, Items, Quest)

**Best For:**
- Menus, settings, inventory lists
- Turn-based games (no time pressure)
- Text-heavy interfaces

**Face Button Mapping (Standard Conventions)**

**Xbox (and Nintendo Switch positions):**
- **A (bottom):** Confirm/Select
- **B (right):** Cancel/Back
- **X (left):** Secondary action (reload, use item)
- **Y (top):** Alternative action (switch weapon, jump)

**PlayStation (different positions, same function):**
- **X (bottom):** Confirm/Select
- **Circle (right):** Cancel/Back (or confirm in Japan)
- **Square (left):** Secondary action
- **Triangle (top):** Alternative action

**Note:** Japan has opposite confirm/cancel (Circle=confirm, X=cancel). Respect regional settings.

**Shoulder Buttons:**
- **L1/R1 (bumpers):** Tab switching, weapon cycling
- **L2/R2 (triggers):** Aim, fire, brake (in-game actions, not UI)
- **L3/R3 (stick clicks):** Rarely used for menus (accessibility issue)

**Special Buttons:**
- **Start:** Pause menu, main menu
- **Select/Share/View:** Map, inventory, photo mode

**Quick Access Patterns**

Access frequently-used actions without pausing - critical for action games.

**D-pad Quick Actions:**
- **D-pad Up:** Heal/use item
- **D-pad Down:** Change stance/crouch
- **D-pad Left/Right:** Quick weapon switch
- **Hold D-pad:** Open radial menu for more options

**Bumper Cycling:**
- **Tap L1/R1:** Cycle weapons/abilities
- **Visual:** Weapon icon flashes briefly (bottom corner)
- **No Menu:** Instant, doesn't pause game

**Keyboard + Mouse**

PC players expect precision with mouse and speed with keyboard shortcuts.

**Mouse Click Interactions**

**Left-Click:**
- Primary action (attack, select, confirm)
- Drag to rotate camera
- Click UI buttons

**Right-Click:**
- Secondary action (aim, block, context menu)
- Hold for continuous action (aim down sights)

**Middle-Click (Scroll Wheel Click):**
- Special action (grenade, ping, mark)
- Less common (not all mice have reliable middle-click)

**Scroll Wheel:**
- **Scroll Up/Down:** Weapon switch, zoom in/out
- **Fast Switch:** Common in FPS (scroll = cycle weapons)

**Mouse Hover:**
- Tooltips appear after 300ms delay
- Highlight interactive objects
- Preview information (item stats)

**Keyboard Shortcuts**

Fast access to actions without mouse movement - critical for competitive play.

**Number Keys (1-9):**
- **Hotbar:** Weapons, abilities, items mapped to keys
- **Instant Access:** Press 3 → switch to weapon in slot 3
- **Visual:** Hotbar UI at bottom showing 9 slots

**Common Keyboard Mappings:**
- **WASD:** Movement (never use for UI navigation - conflicts)
- **E:** Interact/use
- **F:** Melee/special action
- **R:** Reload
- **Q:** Quick ability/gadget
- **G:** Grenade/throwable
- **Tab:** Scoreboard, map, overlay
- **ESC:** Pause menu
- **I:** Inventory
- **M:** Map
- **J:** Journal/quests
- **C:** Character stats
- **H:** Toggle HUD
- **F1-F12:** Less common (farther reach)

**Modifier Keys:**
- **Shift+Key:** Alternative action (Shift+1 = ability variant)
- **Ctrl+Key:** Rarely used (hand position conflicts with WASD)
- **Alt+Key:** Possible but less common

**Menu Navigation:**
- **Arrow Keys:** Navigate menus (alternative to mouse)
- **Enter:** Confirm
- **ESC:** Cancel/back
- **Tab:** Next field (forms)

**Hybrid Navigation (Mouse + Keyboard)**

Fastest method: mouse for selection, keyboard for confirmation/common actions.

**Pattern:**
- Hover item with mouse (instant)
- Press E to use (faster than clicking "Use" button)
- Or press 1-9 to quick-slot item

**Best For:**
- Inventory management (fast sorting)
- Crafting systems
- PC-first game design

**Touch (Mobile Games)**

Mobile games require larger touch targets, thumb-zone positioning, and gesture controls.

**Touch Target Sizing**

**Minimum Sizes:**
- **Critical Actions:** 60x60px minimum (larger than mobile UI standard)
- **Secondary Actions:** 50x50px acceptable
- **Spacing:** 16px minimum between targets (prevent accidental taps)

**Reasoning:**
- Gameplay is fast, precision is lower
- Fingers obscure screen (larger = easier to hit)
- "Fat finger" problem exacerbated by stress

**Thumb Zones**

Design for one-handed play when possible - most mobile gamers use thumbs.

**Most Reachable (Right Hand):**
- **Bottom 50%:** Easiest reach
- **Right side:** Natural thumb arc
- **Center-bottom:** Both hands reachable

**Least Reachable:**
- **Top corners:** Require hand repositioning
- **Top center:** Requires thumb stretch

**Layout Strategy:**
- **Left Side:** Virtual joystick (movement)
- **Right Side:** Action buttons (fire, jump, crouch)
- **Top:** Non-critical info (score, minimap)
- **Bottom Center:** Pause button (reachable by both thumbs)

**On-Screen Controls**

**Virtual Joystick:**
- **Position:** Bottom-left corner
- **Size:** 120-150px diameter
- **Appearance:** Semi-transparent circle (50-70% opacity)
- **Behavior:** Stick appears on touch, follows thumb (floating joystick)
- **Deadzone:** Center 20% (no movement)

**Action Buttons:**
- **Position:** Bottom-right, clustered arrangement
- **Size:** 60-80px diameter each
- **Layout:** Primary button largest, secondary smaller, arranged in arc
- **Labels:** Icons (not text) for clarity
- **Spacing:** 10-16px between buttons

**Transparency:**
- **Semi-Transparent:** 60-70% opacity (see gameplay underneath)
- **Solid on Press:** 100% opacity when finger touching (clear feedback)
- **Customizable:** Let players adjust opacity (30-100%)

**Gestures**

Touch-specific interactions for mobile games.

**Swipe:**
- **Use:** Camera control (swipe to rotate view)
- **Direction:** Anywhere on empty screen space
- **Conflicts:** Don't conflict with joystick area

**Pinch:**
- **Use:** Zoom map (strategy games)
- **Not Common:** In gameplay (hard to pinch while controlling character)

**Long-Press:**
- **Use:** Context menu, alternate action
- **Duration:** 500ms hold
- **Feedback:** Circle fills around finger (progress indicator)

**Tap:**
- **Use:** Primary action (fire, select)
- **Double-Tap:** Less common (timing window issues)

**Two-Finger Tap:**
- **Use:** Rare (HUD toggle, screenshot)
- **Not Recommended:** Awkward, conflicts with other gestures

**Input Prompts (Dynamic)**

Show correct button/key based on current input device - critical for multi-platform games.

**Implementation:**
- **Detect Input:** Game detects controller type (Xbox, PlayStation, keyboard)
- **Update Prompts:** All on-screen prompts update instantly
- **Switch On Input:** If player picks up controller, prompts change from keyboard to gamepad

**Examples:**
- **Xbox Controller:** "Press A to continue"
- **PlayStation Controller:** "Press X to continue"
- **Keyboard:** "Press E to interact"

**Icon Libraries:**
- Use official button icons (Microsoft, Sony, Nintendo)
- Or generic icons (acceptable if clean)
- Never: Wrong platform icons (Xbox icons on PlayStation)

**Text Fallback:**
- If no icon available: "[A Button]" or "Confirm Button"

**Consistent Mapping**

Buttons should have consistent meaning across all menus and screens.

**Standard Mapping:**
- **A/X:** Always confirm (never cancel)
- **B/Circle:** Always cancel (never confirm)
- **L1/R1:** Always tab switching (not random actions)

**Why This Matters:**
- Muscle memory (players don't read prompts after first few times)
- Consistency reduces errors (accidentally backing out of menu)

**Exceptions:**
- Regional differences (Japan swaps confirm/cancel)
- Game respects system settings


## Dimension 3: AESTHETIC COHERENCE

**Purpose:** Ensure UI visually integrates with the game's art style and genre

Game UI should feel like it belongs in the game world, not a generic overlay. A medieval fantasy game with sleek modern UI feels wrong. A cyberpunk game with ornate serif fonts breaks the aesthetic. Typography, color palette, animations, and visual effects must all reinforce the game's setting and tone.

### Evaluation Questions

1. **Does the UI typography match the game's era/genre?**
   - Medieval fantasy: Ornate serif, parchment textures
   - Sci-fi: Futuristic sans-serif, glowing effects
   - Modern military: Stencil fonts, tactical styling

2. **Do UI colors complement the game's palette?**
   - Extract colors from game's environment art
   - Use game's accent colors for highlights
   - Avoid generic blue/gray if game uses warm tones

3. **Do animations match the game's pacing?**
   - Fast-paced shooter: Snappy 100-150ms transitions
   - Slow RPG: Smooth 200-300ms transitions
   - Horror: Delayed, unsettling animations

4. **Does the UI evolve with player progression?**
   - Basic UI early game (tutorial-friendly)
   - Enhanced UI as player gains skills/gear
   - Visual upgrades reinforce progression

### Visual Style Matching

**Typography**

Font choice must match game setting, era, and tone.

**Fantasy RPG:**
- **Fonts:** Ornate serif (Cinzel, Trajan), blackletter for titles
- **Textures:** Parchment, leather, weathered paper
- **Colors:** Warm (gold, brown, burgundy)
- **Examples:** Skyrim, Diablo, World of Warcraft

**Sci-Fi/Futuristic:**
- **Fonts:** Clean sans-serif (Orbitron, Exo, Rajdhani), monospace for data
- **Effects:** Glowing edges, scan lines, holographic flicker
- **Colors:** Cool (cyan, blue, green accents), high contrast
- **Examples:** Halo, Mass Effect, Cyberpunk 2077

**Horror:**
- **Fonts:** Distressed, handwritten, scratchy
- **Textures:** Blood stains, torn paper, smudges
- **Colors:** Desaturated, red accents
- **Examples:** Resident Evil, Silent Hill, Dead Space

**Military/Tactical:**
- **Fonts:** Stencil (Eurostile, Quantico), military sans-serif
- **Style:** Grid overlays, tactical markers, NATO symbols
- **Colors:** Olive drab, black, orange/yellow highlights
- **Examples:** Call of Duty, Battlefield, Ghost Recon

**Cartoon/Stylized:**
- **Fonts:** Rounded, playful, bold (Fredoka, Baloo, Comic Sans actually works here)
- **Style:** Thick outlines, bright colors, bouncy animations
- **Colors:** Saturated, high contrast, cheerful
- **Examples:** Mario, Overwatch, Fall Guys

**Minimalist/Modern:**
- **Fonts:** Clean sans-serif (Helvetica, Roboto, Inter)
- **Style:** Flat design, subtle shadows, geometric
- **Colors:** Monochrome with single accent color
- **Examples:** Mirror's Edge, Monument Valley, Inside

**Historical/Period:**
- **Medieval:** Gothic fonts, illuminated manuscript style
- **Western:** Slab serif, wanted poster aesthetic
- **1920s-40s:** Art deco, vintage film noir
- **Match Historical Context:** Research typography of era

**Readability Requirements:**

Despite theme, text must always be readable:
- **Minimum Font Size:** 14px for body text, 18px+ for critical info
- **Contrast Ratio:** 4.5:1 minimum (WCAG AA) even in themed UI
- **Test in Game Lighting:** Dark areas, bright areas, motion blur
- **Fallback:** Provide accessibility option for plain readable fonts

**Color Palette**

UI colors should be extracted from and complement the game's art direction.

**Extract from Game:**
- Sample dominant colors from environment art
- Use game's lighting mood (warm desert, cool snow)
- Match color temperature (warm vs cool)

**Accent Colors:**
- Use game's existing accent colors (character glows, magic effects)
- Maintain brand consistency (if game has signature color)
- Color-code UI elements (red=danger, green=safe, blue=info)

**Examples:**
- **The Legend of Zelda: Breath of the Wild:** Warm golds, sky blues (matches Hyrule landscapes)
- **Bloodborne:** Dark grays, deep reds, muted purples (Gothic horror)
- **Ori and the Blind Forest:** Ethereal blues, vibrant greens (magical forest)

**Avoid:**
- Generic game UI blue (#0066CC) if game has warm palette
- Pure black backgrounds if game uses warm lighting
- Overly saturated colors in realistic games

**Visual Effects**

Effects should match game's visual language and genre conventions.

**Fantasy:**
- **Effects:** Magic particles, glowing runes, sparkles
- **Transitions:** Fade with magical shimmer
- **Textures:** Aged parchment, leather stitching, metal clasps

**Sci-Fi:**
- **Effects:** Hologram flicker, scan lines, lens flare
- **Transitions:** Digital wipes, hexagon patterns
- **Textures:** Brushed metal, circuit patterns, glass panels

**Modern Military:**
- **Effects:** Minimal, tactical overlays, grid lines
- **Transitions:** Fast, efficient (no decoration)
- **Textures:** Matte surfaces, stenciled text

**Cyberpunk:**
- **Effects:** Glitch effects, chromatic aberration, neon glows
- **Transitions:** Digital corruption, VHS distortion
- **Textures:** Neon signs, rain-wet surfaces, CRT scanlines

**Horror:**
- **Effects:** Blood drips, scratches, static noise
- **Transitions:** Slow, unnerving fades
- **Textures:** Rust, decay, organic textures

**Performance Note:** Effects are expensive - use sparingly, optimize particle counts.

**Animations and Timing**

Animation speed and easing should match game's overall pacing.

**Fast-Paced Games (FPS, Racing, Fighting):**
- **Menu Transitions:** 100-150ms (snappy, get out of player's way)
- **Button Feedback:** Instant (<50ms)
- **Modal Open/Close:** 150ms (quick)
- **Reasoning:** Players want to get back to action fast

**Medium-Paced Games (Action-RPG, Third-Person Shooter):**
- **Menu Transitions:** 200-250ms (smooth but not slow)
- **Animations:** Ease-out curves (natural settling)
- **Modal Open/Close:** 200ms
- **Reasoning:** Balance between speed and polish

**Slow-Paced Games (Turn-Based RPG, Strategy, Simulation):**
- **Menu Transitions:** 250-300ms (luxurious, smooth)
- **Animations:** Elaborate transitions (page turns, unfurls)
- **Modal Open/Close:** 300ms
- **Reasoning:** Players aren't rushed, appreciate polish

**Horror Games:**
- **Menu Transitions:** 300-500ms (deliberately slow, unsettling)
- **Animations:** Delayed, stuttering, unpredictable timing
- **Modal Open/Close:** Slow fade (200-400ms)
- **Reasoning:** Build tension even in menus

**Puzzle Games:**
- **Menu Transitions:** 200ms (smooth, not distracting)
- **Animations:** Satisfying completion animations (300ms)
- **Reasoning:** Calm, thoughtful pace

**Consistency Within Game:**

All UI animations should use same timing/easing family:
- Don't mix 100ms and 500ms transitions randomly
- Use consistent easing curves (all ease-out, or all ease-in-out)
- Faster animations for frequently-accessed menus, slower for rare screens

### Patterns

**Themed Menus**

Menus that visually belong to the game world, not generic overlays.

**Medieval Fantasy:**
- Menus appear as parchment pages with wax seals
- Item icons on weathered paper background
- Navigation tabs as leather-bound book sections
- Sound effects: Paper rustling, quill scratching

**Sci-Fi/Space:**
- Menus as holographic projections with flicker
- Grid overlays and hexagonal patterns
- Blue/cyan glows, scan line effects
- Sound effects: Computer beeps, digital chirps

**Post-Apocalyptic:**
- Menus as worn, salvaged technology (cracked screens)
- Hand-drawn maps, graffiti-style markers
- Rusty metal textures, duct-tape repairs
- Sound effects: Static, mechanical clicks

**Pirate/Naval:**
- Menus as nautical charts, ship logs
- Rope borders, compass rose decorations
- Ink stains, water damage on paper
- Sound effects: Creaking wood, waves

**Diegetic Justification**

Explain UI through game world - gives immersion and context.

**Examples:**
- **Dead Space:** "This is your RIG suit's holographic interface"
- **Fallout:** "This is your Pip-Boy wrist computer"
- **Tom Clancy Games:** "This is your tactical AR headset display"
- **Hacking Games:** "This is the terminal you're using"

**Benefits:**
- Justifies UI existence in-world
- Excuses UI limitations ("Old tech, low resolution")
- Allows UI to break/glitch when narrative appropriate

**Progression Reflection**

UI evolves as player progresses - visual reward for advancement.

**Patterns:**
- **Early Game:** Basic, minimal UI (few features unlocked)
- **Mid Game:** Enhanced UI elements (skill tree unlocked, shows new options)
- **Late Game:** Polished, feature-rich UI (all systems available)

**Examples:**
- **RPG:** Character portrait upgrades with better gear (armor shown in UI)
- **Crafting Game:** Inventory UI upgrades from bag → chest → warehouse
- **Sci-Fi:** HUD upgrades from basic to advanced (new scan modes, data overlays)

**Implementation:**
- Tie UI unlocks to player progression milestones
- Tutorial: Introduce UI elements gradually, not all at once
- Visual polish increases (more effects, better icons) as player advances

**Consistency Across Screens**

All menus, HUD, and UI should feel cohesive - unified visual language.

**Checklist:**
- **Font Family:** Same 1-2 fonts across all UI
- **Color Palette:** Same accent colors, backgrounds, borders
- **Button Style:** All buttons same shape, size, effects
- **Icon Style:** Consistent line weight, detail level, colors
- **Animation Timing:** Same durations, same easing curves
- **Spacing:** Consistent padding, margins, alignment grid

**Why This Matters:**
- Professional appearance (sloppy if inconsistent)
- Easier to learn (patterns repeat)
- Lower production cost (reuse components)

**Testing:**
- Screenshot all menu screens side-by-side
- Check for visual outliers (one menu looks different)
- Ensure UI feels like single design system

### Readability vs Theme

Aesthetic theming must never sacrifice legibility - players need to read text.

**Non-Negotiable Requirements:**

**Contrast Ratio:**
- **WCAG AA:** 4.5:1 for normal text, 3:1 for large text (18pt+)
- **Game Recommendation:** 7:1 for critical info (easier to read in motion)
- **Test:** Use contrast checker tools, test in-game with real lighting

**Font Sizing:**
- **Body Text:** 14px minimum on PC, 18px+ on console (TV distance)
- **Critical Info:** 16px+ minimum
- **Headings:** 24px+ for clear hierarchy

**Lighting Conditions:**
- **Dark Scenes:** UI must be visible (add background, increase contrast)
- **Bright Scenes:** UI must not wash out (darken backgrounds)
- **Motion Blur:** Text must remain readable even when camera moving

**TV Distance (Console Games):**
- Players sit 6-10 feet from TV (much farther than PC monitor)
- Text must be larger, higher contrast
- Test on real TV at typical living room distance

**Accessibility Options:**

Provide fallbacks for players with readability issues:
- **High Contrast Mode:** Black/white UI, maximum contrast
- **Large Text Mode:** 1.5x or 2x text size
- **Colorblind Modes:** Alternative color palettes (red/green, blue/yellow)
- **Simplified UI:** Remove decorative elements, plain backgrounds

**When to Prioritize Readability:**
- Critical information (health, ammo, quest objectives)
- Text-heavy games (RPGs, visual novels, strategy)
- Accessibility compliance (legal requirement in some regions)

**When Theme Can Bend Rules Slightly:**
- Decorative elements (menu backgrounds, borders)
- Non-critical information (lore text, optional codex)
- With accessibility toggle available


## Dimension 4: PERFORMANCE IMPACT

**Purpose:** Ensure UI doesn't negatively impact game frame rate or responsiveness

Game UI competes with gameplay for GPU/CPU resources. A 60fps game that drops to 45fps when opening inventory feels broken. UI must be optimized through GPU acceleration, draw call reduction, lazy rendering, and asset optimization. Poor UI performance is especially visible when contrasted with smooth gameplay.

### Evaluation Questions

1. **Does UI maintain target frame rate (60fps or 30fps)?**
   - Measure frame time with UI open vs closed
   - UI should add <10% to frame time
   - Critical UI (HUD) should add <5%

2. **Are draw calls minimized through batching?**
   - Combine UI elements into single draw call
   - Use texture atlases (sprite sheets)
   - Minimize material changes

3. **Are only visible UI elements rendered?**
   - Disable hidden menus (not just set invisible)
   - Cull off-screen elements
   - Use object pooling (reuse UI objects)

4. **Are UI assets optimized?**
   - Compressed textures (DXT, ASTC)
   - Appropriate texture sizes (256x256 for icons, not 2048x2048)
   - Minimal overdraw (transparent overlays)

### Performance Testing

**Measure Frame Impact:**

**Baseline:**
- Run gameplay without UI (or minimal HUD)
- Measure frame time (16.67ms at 60fps, 33.33ms at 30fps)
- Record average, min, max frame times

**With UI:**
- Open complex menu (inventory, map, skill tree)
- Measure frame time again
- Calculate UI overhead (difference from baseline)

**Target Overhead:**
- **HUD (Always Visible):** <5% overhead (0.8ms at 60fps)
- **Menus (Paused):** <10% overhead (1.67ms at 60fps)
- **Acceptable:** <15% overhead (2.5ms at 60fps)
- **Problematic:** >20% overhead (3.3ms+ at 60fps)

**Tools:**
- Unity: Profiler (CPU/GPU breakdown)
- Unreal: Unreal Insights, stat commands (stat fps, stat unit)
- Custom: Frame time overlays, PIX (Windows), RenderDoc

**Test on Minimum Spec:**
- Test on lowest-end supported hardware (base console, min-spec PC)
- UI that runs fine on high-end may tank on low-end
- Optimize for worst-case hardware

**Measure Input Latency:**

**Target:** <50ms from input to UI response (button press to visual feedback)

**Test:**
- Press button, measure time until UI updates
- Use high-speed camera (240fps) to measure lag
- Competitive games: <30ms critical

**Common Causes of Lag:**
- UI framework overhead (Unity UI, Unreal UMG can add 1-2 frames)
- Complex layout recalculations
- Synchronous asset loading (blocking)

### Optimization Strategies

**GPU Acceleration**

Use hardware rendering, never CPU-based UI rendering.

**Implementation:**
- **Unity:** Canvas set to Screen Space - Overlay or Camera, GPU mode
- **Unreal:** UMG uses native GPU rendering by default
- **Custom Engines:** Use GPU texture quads, not CPU-drawn pixels

**Indicators of CPU Rendering:**
- UI updates cause CPU spikes (not GPU)
- Frame rate drops on UI updates but GPU utilization doesn't increase
- Profiler shows UI code on CPU thread, not render thread

**Draw Call Optimization**

Reduce number of draw calls - single biggest performance factor.

**Batching:**
- Combine multiple UI elements into single draw call
- **Static Batching:** Elements that don't move/change (backgrounds)
- **Dynamic Batching:** Elements that update (health bars)

**Texture Atlases (Sprite Sheets):**
- Pack all UI icons/sprites into single texture (2048x2048 or 4096x4096)
- Reduces texture switches (expensive)
- **Tools:** TexturePacker, Unity Sprite Packer, Unreal Paper2D

**Example:**
- 100 icons as separate textures = 100 draw calls
- 100 icons in atlas = 1-5 draw calls (huge savings)

**Minimize Material Changes:**
- Use same material for all UI elements (just different textures/colors)
- Each material change breaks batching
- **Unity:** Use same shader, shared material

**Canvas Optimization (Unity-Specific):**
- Split UI into multiple canvases (static vs dynamic)
- Updating one element doesn't rebuild entire canvas
- Static canvas (background) separate from dynamic canvas (health bar)

**Lazy Rendering**

Only render visible UI - disable hidden elements completely.

**Disable Hidden Menus:**
- Don't just set invisible (UI still calculated, rendered)
- Disable GameObject entirely (Unity: SetActive(false))
- Save state and restore when re-enabled

**Cull Off-Screen Elements:**
- Scrollable lists: Only render visible items (virtualization)
- **Example:** Inventory with 1000 items - only render 20 visible at once
- **Unity:** Scroll Rect with pooling
- **Unreal:** List View with virtualization

**Object Pooling:**
- Reuse UI objects instead of creating/destroying
- Create pool of 20 list items, reuse as player scrolls
- Avoids garbage collection spikes (memory allocation lag)

**Level of Detail (LOD) for UI**

Simplify UI at low frame rates - graceful degradation.

**Patterns:**
- **<60fps:** Reduce particle effects (skill tree sparkles, menu animations)
- **<30fps:** Disable non-critical animations (bouncing buttons, fades)
- **<20fps:** Remove decorative elements (background effects, shadows)
- **Dynamic Adjustment:** Automatically reduce quality when frame rate drops

**Implementation:**
- Monitor frame rate continuously
- Disable effects when FPS drops below threshold
- Re-enable when FPS stable above threshold for 3+ seconds

**Example:**
- Inventory menu has animated background (particle effects)
- If FPS <50, disable particle system
- Menu still functional, just less decorative

**Asset Optimization**

UI textures are often unoptimized - easy performance wins.

**Texture Compression:**
- **PC/Console:** DXT1/DXT5 (up to 6x compression)
- **Mobile:** ASTC, ETC2, PVRTC
- **Lossless:** PNG for UI (but larger file size)
- **Quality:** High compression acceptable for backgrounds, low for icons (readability)

**Texture Sizes:**
- **Icons:** 64x64 to 256x256 (rarely need larger)
- **Backgrounds:** 1024x1024 to 2048x2048
- **Buttons:** 128x64 to 512x256
- **Never:** 4096x4096 for small UI element (huge waste)

**Mipmaps:**
- Generate mipmaps for scaled UI elements
- Prevents aliasing, improves performance when downscaled
- Not needed for pixel-perfect UI (native resolution)

**Vector UI (Scalable):**
- Use vector graphics where possible (crisp at any resolution)
- **Unity:** TextMeshPro (vector fonts), SVG plugins
- **Unreal:** Slate vectors, SVG support
- **Benefits:** Small file size, no resolution limitations

**Overdraw Reduction:**

Overdraw = pixels drawn multiple times (transparency layers).

**Problem:**
- 10 transparent UI panels stacked = 10x overdraw (GPU draws same pixel 10 times)
- Mobile GPUs especially sensitive to overdraw

**Solutions:**
- Reduce transparent overlays (use opaque backgrounds when possible)
- Minimize UI depth (fewer stacked elements)
- **Tools:** Unity Frame Debugger (overdraw mode), Unreal shader complexity view

**Example:**
- Modal with transparent darkened background (50% black overlay)
- Overlay covers entire screen = full-screen overdraw
- **Optimization:** Use opaque dark background (80% black, 20% transparency) - still looks good, less overdraw

**Frame Budget**

Allocate specific time budget to UI - enforce performance discipline.

**60fps Game (16.67ms per frame):**
- **Gameplay:** 13-14ms (80-85%)
- **UI (HUD):** 0.8-1.5ms (5-10%)
- **Other:** 1-2ms (audio, physics, etc.)

**30fps Game (33.33ms per frame):**
- **Gameplay:** 26-28ms (80-85%)
- **UI (HUD):** 1.5-3ms (5-10%)
- **Other:** 3-5ms

**Menu Screen (Paused):**
- UI can use more budget (gameplay paused)
- Still target <20% frame time (6ms at 30fps)
- Maintain responsive feel

**Enforcement:**
- Profile UI regularly (every sprint/milestone)
- Flag any UI element exceeding budget
- Optimize or simplify until within budget


## Game-Specific UI Patterns

Common UI elements with production-ready specifications.

### HUD Elements

**Health/Stamina Bars**

**Position:**
- **Bottom-Left:** Common in FPS, action games (near character)
- **Top-Left:** Alternative, used in some RPGs
- **Top-Center:** Fighting games (both players side-by-side)

**Style:**
- **Bar Fill:** Horizontal bar, colored fill left-to-right
- **Segmented:** Individual chunks (3-5 hits per segment)
- **Numeric:** "100/100" or "100%" text
- **Hybrid:** Bar + number

**Colors:**
- **Health:** Red (universal), green when full (some games)
- **Stamina:** Yellow, green, or blue (depends on game)
- **Mana/Energy:** Blue (magic), purple (energy)

**Specifications:**
```
Health Bar Example:
- Size: 200px width, 24px height
- Position: Bottom-left, 32px margin
- Background: Dark gray/black (80% opacity)
- Fill: Red gradient (bright to dark)
- Border: 2px white/gray outline
- Text: White, 16px, centered (optional)
- Animation: Smooth decrease (200ms), instant increase
- Flash: Red flash when taking damage (100ms)
```

**Minimap**

**Position:**
- **Top-Right:** Most common (doesn't obscure center view)
- **Bottom-Right:** Alternative
- **Top-Left:** Rare (can conflict with health)

**Size:**
- **Small:** 10-12% of screen (FPS, action)
- **Medium:** 15-20% of screen (RPG, exploration)
- **Large:** 25%+ (strategy, when map is critical)

**Rotation:**
- **World-Locked:** North always up (easier orientation, used in strategy)
- **Player-Locked:** Player always center, map rotates (intuitive movement, used in FPS)

**Elements:**
- **Player Icon:** Distinct, centered (arrow showing facing direction)
- **Enemies:** Red dots/icons
- **Allies:** Blue/green dots
- **Objectives:** Yellow markers
- **Fog of War:** Gray unexplored areas (strategy games)

**Toggle Zoom:**
- **Scroll Wheel:** Zoom in/out (PC)
- **Button Press:** Cycle zoom levels (console)
- **Levels:** Close (detailed), Medium, Far (overview)

**Objective Markers**

**World-Space Markers:**
- Icon floating at objective location (3D space)
- Arrow pointing to objective if off-screen
- Distance indicator: "350m" text below icon

**Off-Screen Indicators:**
- Arrow at edge of screen pointing toward objective
- Fades when close (on-screen)
- Multiple colors for multiple objectives (red=main, yellow=side)

**Specifications:**
```
Objective Marker:
- Icon: 32x32px (on-screen), 48x48px (close-up)
- Distance Text: 14px, white with black outline
- Arrow: 64px, at screen edge, points to objective
- Color: Yellow (neutral), changes to green when near
- Fade: Opacity 100% far, 50% close, 0% when reached
```

**Crosshairs/Reticles**

**Position:** Center screen, exactly (critical for aiming accuracy)

**Types:**
- **Dot:** Single pixel or small circle (precision rifles)
- **Cross:** Four lines forming +, with center gap (assault rifles)
- **Circle:** Circular outline (shotguns, spread weapons)
- **Dynamic:** Expands when moving, contracts when still (shows accuracy)

**Customization:**
- Color (player preference)
- Size (1x, 1.5x, 2x)
- Opacity (50-100%)
- Enable/disable (some players prefer no crosshair)

**Hit Feedback:**
- **On Hit:** Crosshair changes color (white → red, 100ms)
- **Damage Number:** Damage dealt appears briefly (100ms fade-in, 500ms fade-out)
- **Audio:** Hit sound (essential feedback)

**Ammo Counter**

**Position:**
- **Bottom-Right:** Most common (near weapon, doesn't obscure aim)
- **Bottom-Center:** Alternative (minimalist FPS)

**Format:**
- **Current / Reserve:** "30 / 120" (current magazine / reserve ammo)
- **Current Only:** "30" (reserve shown separately or not at all)
- **Icon + Number:** Bullet icon + number (visual clarity)

**Low Ammo Warning:**
- **Color Change:** White → red when <30% remaining
- **Pulsing:** Gentle pulse animation (every 1s)
- **Audio:** Click sound when magazine empty

**Specifications:**
```
Ammo Counter:
- Size: 48px height text
- Position: Bottom-right, 32px margin
- Font: Bold sans-serif, high contrast
- Color: White (normal), red (low)
- Format: "30 / 120" (large / small text)
- Reload Indicator: Circular progress bar (2s animation)
```

### Menus

**Pause Menu**

**Trigger:**
- **PC:** ESC key
- **Console:** Start button
- **Mobile:** Pause button (top corner)

**Background:**
- Dim/blur gameplay (50-70% dark overlay)
- Or pause frame (freeze last frame, apply blur)

**Options (Typical):**
- Resume (highlighted by default)
- Settings (audio, video, controls)
- Save/Load (if applicable)
- Map (if applicable)
- Quit to Menu
- Quit to Desktop (PC only)

**Layout:**
- Vertical list, centered
- Large touch targets (64px height minimum)
- Clear visual hierarchy (Resume most prominent)

**Inventory System**

**Grid Layout:**
- Items arranged in grid cells (8x8, 10x10, etc.)
- Each cell: 64x64px to 128x128px
- Drag-and-drop to move/equip items
- Hover for tooltip (item stats)

**Sorting Options:**
- Sort by: Name, Type, Rarity, Weight, Value
- Dropdown or button strip at top
- Filter tabs: All, Weapons, Armor, Consumables, Quest

**Equipment Slots:**
- Visual representation of character (paper doll)
- Slots: Head, Chest, Legs, Hands, Feet, Weapon, Shield, etc.
- Drag item to slot to equip
- Right-click for quick equip (PC)

**Weight/Capacity:**
- Current weight / max weight: "45 / 100 kg"
- Progress bar (fill = closer to max)
- Warning: Red text/bar when overencumbered

**Specifications:**
```
Inventory Grid:
- Cell Size: 96x96px
- Spacing: 8px between cells
- Background: Dark semi-transparent
- Selected: Yellow border (4px)
- Hover: White border (2px)
- Tooltip: Appears after 300ms hover
- Drag: Item follows cursor at 90% opacity
```

**Skill Trees**

**Visual Progression:**
- Nodes connected by lines (show prerequisites)
- Locked nodes: Grayed out (requires previous skill)
- Unlocked nodes: Bright, colored (can be selected)
- Purchased nodes: Filled, checkmark icon

**Navigation:**
- **PC:** Click node to view, click again to purchase
- **Console:** D-pad or analog stick to navigate, A/X to view, hold to confirm purchase
- **Zoom:** Scroll wheel (PC) or triggers (console)

**Node Information:**
- **Title:** Skill name (16px bold)
- **Icon:** 64x64px visual representation
- **Description:** What skill does (14px)
- **Cost:** Skill points required (yellow text, coin icon)
- **Requirements:** Level or prerequisite skills

**Unlock Animation:**
- Node glows, particles burst outward (500ms)
- Line animates connecting to next nodes (300ms)
- Sound effect (success chime)

**Settings Menu**

**Tabbed Categories:**
- **Video:** Resolution, quality, VSync, brightness
- **Audio:** Master, music, SFX, voice volume sliders
- **Controls:** Key bindings, sensitivity, invert Y-axis
- **Gameplay:** Difficulty, subtitles, HUD options, language

**Tab Navigation:**
- **PC:** Click tab or Q/E to cycle
- **Console:** L1/R1 shoulder buttons to cycle tabs

**Option Types:**
- **Sliders:** Volume (0-100%), brightness, mouse sensitivity
- **Dropdowns:** Resolution, quality preset, language
- **Toggles:** VSync on/off, subtitles on/off
- **Key Binding:** Click field, press key to rebind (PC)

**Buttons:**
- **Apply:** Save and apply changes immediately
- **Save:** Save changes, return to game
- **Revert:** Undo changes since last save
- **Default:** Reset all to default values

**Visual Preview:**
- Resolution change: Show example (1920x1080 vs 2560x1440)
- Brightness: Show gradient bar with dark → light
- Audio: Play test sound when adjusting volume

### Feedback Systems

**Hit Markers**

**Visual:**
- Crosshair briefly changes (white → red, or + to X)
- Duration: 100ms
- Position: Center screen (overlays crosshair)

**Critical Hit:**
- Different visual (larger X, yellow color)
- Damage number larger/different color (yellow vs white)
- Sound effect distinct (heavy thunk vs light hit)

**Kill Confirmation:**
- Special icon (skull, X, etc.) flashes (200ms)
- Kill feed notification (top-right): "You eliminated PlayerName"
- Audio: Satisfying impact sound

**Specifications:**
```
Hit Marker:
- Size: 32x32px
- Color: White (normal), yellow (critical), red (kill)
- Duration: 100ms (normal), 200ms (critical/kill)
- Animation: Scale from 0% to 100% (50ms), hold, fade (50ms)
- Position: Centered on crosshair
- Audio: Sync with visual (essential)
```

**XP/Level-Up Notifications**

**XP Gain:**
- **Position:** Bottom-right corner (doesn't obscure gameplay)
- **Format:** "+50 XP" (green text, fades in/out)
- **Duration:** 2 seconds visible
- **Stacking:** Multiple XP gains add to counter ("+50 XP" → "+75 XP")

**Level-Up:**
- **Visual:** Full-screen flash (200ms), or corner burst animation
- **Notification:** "Level Up! Level 15" (large text, 3s duration)
- **Sound:** Triumphant fanfare (reward audio)
- **Reward:** Show unlocked skills/items (modal or banner)

**Progress Bar:**
- Horizontal bar showing XP progress to next level
- Position: Top of screen (under level number) or bottom corner
- Fill animation smooth (500ms when gaining XP)

**Loot Notifications**

**Item Acquired:**
- **Position:** Bottom-right (or top-right, consistent)
- **Format:** Icon + item name (64px icon, 16px text)
- **Duration:** 2-3 seconds visible
- **Queue:** Multiple items stack vertically (up to 5, then scroll)

**Rarity Color-Coding:**
- **Common:** White/gray
- **Uncommon:** Green
- **Rare:** Blue
- **Epic:** Purple
- **Legendary:** Orange/gold
- **Border/glow matches rarity color**

**Loot Explosion (On-Screen):**
- When opening chest/killing enemy
- Items shoot out (particle effect)
- Icons briefly visible in 3D space
- Collect automatically or by pressing button

**Specifications:**
```
Loot Notification:
- Size: 64x64px icon, 200px width card
- Background: Semi-transparent dark (80% opacity)
- Border: Rarity color (4px, glowing)
- Text: White, 16px, item name
- Animation: Slide in from right (300ms ease-out)
- Duration: 2s visible, then fade out (500ms)
- Audio: Pickup sound (different for rare items)
```

**Status Effects**

**Icon Display:**
- Position: Near health bar (clustered)
- Size: 32x32px icon per effect
- Max Visible: 5-8 icons (more = scroll or grid)

**Timer:**
- Circular progress bar around icon (clockwise fill)
- Or numeric countdown: "5s" below icon
- Flash/pulse when about to expire (last 3 seconds)

**Color Coding:**
- **Buffs (Positive):** Green border/glow
- **Debuffs (Negative):** Red border/glow
- **Neutral:** Blue or white border

**Tooltip:**
- Hover (PC) or select (console) to see details
- Name: "Poison" (16px bold)
- Description: "Lose 5 HP per second" (14px)
- Duration: "8 seconds remaining" (14px)

**Stacking:**
- Multiple stacks of same effect: Show number badge "x3"
- Or separate icons for each stack (if important)


## Anti-Patterns

Common mistakes that harm game UI usability, immersion, or performance.

### Priority 0 (Critical - Never Do)

**1. UI Blocking Gameplay Input:**
- **Problem:** Menu animation plays, player can't move/shoot during transition (200ms+ input lock)
- **Impact:** Player feels unresponsive, frustration in action games
- **Example:** Opening inventory freezes character for 500ms
- **Fix:** Input should remain active (or gameplay pauses if menu is modal)

**2. No Way to Rebind Controls:**
- **Problem:** Hardcoded controls, player can't change bindings
- **Impact:** Accessibility failure, left-handed players, alternative keyboard layouts
- **Fix:** Full key/button remapping in settings

**3. UI Consuming >20% Frame Time:**
- **Problem:** Menu open = frame rate drops from 60fps to 40fps (12ms overhead)
- **Impact:** Feels broken, especially contrasted with smooth gameplay
- **Fix:** Profile and optimize (batching, disable hidden elements, reduce particles)

**4. Critical Information Hidden:**
- **Problem:** Health bar auto-hides, player dies without knowing they're low
- **Impact:** Unfair deaths, frustration
- **Fix:** Critical info always visible, or reappears when critical (low health)

**5. No Input Prompts for Controller:**
- **Problem:** Tutorial says "Press E" but player using gamepad
- **Impact:** Player confused, doesn't know which button to press
- **Fix:** Dynamic prompts based on current input device

### Priority 1 (High - Avoid)

**6. Cluttered HUD (Too Much Information):**
- **Problem:** 20+ UI elements on screen simultaneously
- **Impact:** Visual clutter, hard to focus on gameplay
- **Fix:** Minimal HUD, contextual display, let players customize

**7. Generic UI (Doesn't Match Game):**
- **Problem:** Medieval fantasy game with modern blue/gray UI
- **Impact:** Breaks immersion, feels cheap/lazy
- **Fix:** Themed UI matching game's aesthetic and era

**8. No Gamepad Support (PC-Only):**
- **Problem:** Game on Steam but only supports keyboard+mouse
- **Impact:** Excludes controller players, poor couch gaming
- **Fix:** Support Xbox/PlayStation controllers, radial menus, proper navigation

**9. Tiny Text on TV (Console Games):**
- **Problem:** 14px text unreadable from 8 feet away on TV
- **Impact:** Player can't read quest text, item descriptions
- **Fix:** 18px+ minimum for console, test on real TV at distance

**10. No Loading Feedback (>2s Loads):**
- **Problem:** Screen freezes for 5 seconds, no spinner/progress bar
- **Impact:** Player thinks game crashed
- **Fix:** Loading spinner, progress bar, or tips/art during loading

### Priority 2 (Medium - Be Cautious)

**11. Over-Animated UI (Distracting):**
- **Problem:** Every button bounces, sparkles constantly animate, particles everywhere
- **Impact:** Distracting from gameplay, sensory overload
- **Fix:** Subtle animations, purposeful not decorative

**12. Diegetic UI Sacrificing Readability:**
- **Problem:** Health bar on character's armor, but dark lighting makes it invisible
- **Impact:** Can't see critical info, defeats purpose
- **Fix:** Increase contrast, add glow/outline, or fallback to overlay

**13. No HUD Toggle Option:**
- **Problem:** Can't hide UI for screenshots/video
- **Impact:** Content creators frustrated, limits community engagement
- **Fix:** Simple button to toggle HUD visibility

**14. Inconsistent Button Mapping:**
- **Problem:** A=confirm in main menu, B=confirm in sub-menu
- **Impact:** Muscle memory breaks, accidental actions
- **Fix:** Consistent A=confirm, B=cancel across all menus

**15. Hover-Only Actions (No Touch Alternative):**
- **Problem:** Critical button only appears on mouse hover
- **Impact:** Unusable on mobile/touch devices
- **Fix:** Always-visible buttons, or tap-to-reveal on touch


## Practical Application

Step-by-step workflows for common game UI design scenarios.

### Workflow 1: FPS HUD Design

**Scenario:** Designing minimal HUD for fast-paced first-person shooter.

**Step 1: Identify Essential Information**
- Health (critical - need to know when low)
- Ammo (critical - need to know when to reload)
- Objective marker (important - where to go)
- Minimap (optional - depends on game)
- Crosshair (essential - aiming)

**Step 2: Minimal Layout**
- **Crosshair:** Center screen (dot or small cross)
- **Health Bar:** Bottom-left, 200x24px, red fill
- **Ammo Counter:** Bottom-right, large numbers (48px), "30 / 120" format
- **Objective Marker:** World-space arrow, top of screen when off-screen
- **Minimap:** Top-right, 12% screen size (optional toggle)

**Step 3: Contextual Display**
- Health bar appears when taking damage, fades after 5s
- Ammo counter always visible when weapon drawn
- Objective marker fades when within 10m (too close)

**Step 4: Hit Markers & Feedback**
- Hit marker: White X flashes on crosshair (100ms)
- Critical hit: Yellow X, larger (200ms)
- Damage numbers: Float upward from hit location (500ms fade)

**Step 5: Performance Test**
- Profile HUD frame time (should be <1ms)
- Test at 60fps minimum
- Optimize: Batch UI elements, use texture atlas for icons

**Step 6: Playtesting**
- Test with fast movement (can player track info while moving?)
- Test in dark/bright environments (is UI always readable?)
- Test with multiple players (feedback needed?)

### Workflow 2: RPG Menu System

**Scenario:** Designing inventory, character stats, and skill tree for fantasy RPG.

**Step 1: Information Architecture**
- **Tabs:** Inventory, Equipment, Skills, Map, Journal, Settings
- **Primary:** Inventory (most accessed)
- **Secondary:** Equipment, Skills
- **Tertiary:** Map, Journal, Settings

**Step 2: Inventory Grid Design**
- 10x8 grid (80 slots)
- Cell size: 96x96px (large enough for detailed icons)
- Drag-and-drop to rearrange
- Sorting: Name, Type, Rarity, Value
- Filter tabs: All, Weapons, Armor, Consumables, Quest Items

**Step 3: Character Equipment (Paper Doll)**
- Visual character in center (shows equipped gear)
- Equipment slots around character: Head, Chest, Legs, Hands, Feet, Weapon, Shield, Ring1, Ring2, Amulet
- Drag item from inventory to slot to equip
- Tooltip on hover: Item stats, compare with equipped

**Step 4: Skill Tree**
- Branching tree showing skill progression
- Nodes: Locked (gray), Unlocked (colored), Purchased (filled)
- Lines connecting prerequisites
- Zoom: Scroll wheel to zoom in/out
- Info panel: Selected skill details (right side)

**Step 5: Themed Aesthetic**
- **Font:** Ornate serif (Cinzel) for titles, readable sans-serif (Lato) for body
- **Textures:** Parchment background, leather borders, metal corner decorations
- **Colors:** Warm browns, golds, burgundy accents
- **Animations:** Page-turn transitions (300ms), skill unlock glow (500ms)

**Step 6: Gamepad Navigation**
- D-pad: Navigate grid cells
- L1/R1: Switch tabs
- A/X: Select item
- B/Circle: Cancel
- Y/Triangle: Quick-use item
- Radial menu (hold LB): Quick-access to consumables

**Step 7: Performance Optimization**
- Disable inventory UI when not open (SetActive false)
- Use object pooling for grid cells (reuse 80 cells)
- Texture atlas for all item icons (single draw call)
- Target: <3ms frame time for menu

### Workflow 3: Mobile Game UI

**Scenario:** Designing touch controls and HUD for mobile action game.

**Step 1: Touch Controls Layout**
- **Left Side:** Virtual joystick (120px diameter, bottom-left corner)
- **Right Side:** Action buttons (3 buttons, 70px diameter each)
  - Jump (bottom-right, largest)
  - Attack (above jump, slightly smaller)
  - Special (top-right, smallest)
- **Top:** Pause button (50px, top-right)

**Step 2: Thumb Zone Optimization**
- All controls in bottom 50% of screen (thumb reachable)
- Virtual joystick: Semi-transparent (60% opacity), solid on touch
- Action buttons: Clustered in arc (easy thumb sweep)

**Step 3: HUD Placement**
- **Top-Left:** Health bar (horizontal, 150x20px)
- **Top-Right:** Score/coins (next to pause button)
- **Top-Center:** Timer (if applicable)
- **Bottom:** Space reserved for controls (no UI here)

**Step 4: Touch Target Sizing**
- All buttons: 60px+ minimum (larger than standard mobile UI)
- Spacing: 16px between buttons (prevent accidental taps)
- Test on smallest supported phone (iPhone SE, small Android)

**Step 5: Performance Optimization (Critical for Mobile)**
- Use ASTC texture compression (smallest file size)
- Icon textures: 256x256 max (not 1024x1024)
- Disable all particle effects on low-end devices
- Target: 30fps minimum on oldest supported device (2-3 year old phone)

**Step 6: Landscape vs Portrait**
- **Landscape:** Action games (better control layout)
- **Portrait:** Casual games (one-handed play)
- Test UI in chosen orientation, ensure controls reachable

### Workflow 4: Horror Game UI

**Scenario:** Designing minimal, immersive UI for survival horror game.

**Step 1: Minimal HUD Philosophy**
- **No HUD** during normal gameplay (maximum immersion)
- **Diegetic Elements Only:** Flashlight for visibility, no health bar
- **Contextual Display:** UI appears only when absolutely necessary

**Step 2: Health Indication (No Bar)**
- Visual: Screen desaturation when low health (color fades to gray)
- Audio: Heartbeat sound increases when injured
- Breathing: Heavy breathing audio when very low health
- No numeric value (player infers from visual/audio cues)

**Step 3: Inventory System (Diegetic)**
- **Pause to Open:** Pause game, open backpack (freeze-frame background)
- **Grid Inventory:** Limited slots (8-12 items, resource scarcity)
- **Inspection:** Pick up item, examine in 3D (rotate with mouse)
- **Combine:** Drag items together to craft (bullets, health kits)

**Step 4: Interaction Prompts (Minimal)**
- **Prompt Appearance:** "Press E to open" appears only when very close to object (<1m)
- **Style:** Small, faded text (not bold, not intrusive)
- **Duration:** Fades in (500ms), stays while in range

**Step 5: Menu Aesthetic (Unsettling)**
- **Font:** Distressed, handwritten (Creepster, or custom scratchy font)
- **Colors:** Desaturated, brown/gray tones, red accents (blood)
- **Animations:** Slow fades (300-500ms), occasional glitch effect
- **Sound:** Subtle unsettling ambience in menus (creaking, distant sounds)

**Step 6: Performance**
- Minimal UI = minimal performance impact (<0.5ms)
- Focus performance budget on atmosphere (lighting, shadows, audio)


## Related Skills

**Core Lyra UX Skills:**
- **`lyra/ux-designer/visual-design-foundations`**: Visual hierarchy, contrast, color theory, typography (apply to game UI, but games have unique constraints like genre aesthetics and TV viewing distance)
- **`lyra/ux-designer/interaction-design-patterns`**: Button states, touch targets (44x44pt iOS, 48x48dp Android, 60x60px games), feedback timing, animations (games require faster feedback <100ms for responsive feel)
- **`lyra/ux-designer/accessibility-and-inclusive-design`**: Colorblind modes, text sizing, control remapping, subtitle support (critical for games, often legally required)
- **`lyra/ux-designer/information-architecture`**: Menu structure, navigation flow (games have complex nested menus - inventory, skills, settings - need clear hierarchy)

**Platform Skills:**
- **`lyra/ux-designer/mobile-design-patterns`**: Touch controls, thumb zones, on-screen controls (mobile games overlap with mobile app patterns but with game-specific needs like virtual joysticks)
- **`lyra/ux-designer/desktop-software-design`**: Keyboard shortcuts, hotkeys (PC games use extensive keyboard shortcuts like MMO hotbars 1-9, modifier keys)

**Cross-Faction:**
- **`muna/technical-writer/clarity-and-style`**: UI copy, error messages, tutorial text (games need concise, flavorful text that fits genre and doesn't break immersion)
- **`ordis/security-architect/secure-authentication-patterns`**: Account systems, login flows (multiplayer games need secure authentication without disrupting game experience)


## Additional Resources

**Game UI Design References:**
- **GDC Talks:** Game Developers Conference UI/UX talks (YouTube: search "GDC UI design")
- **Gamasutra:** Articles on game UI patterns, case studies
- **Game UI Database:** gameuidatabase.com (screenshots of game UIs by genre)

**Engine Documentation:**
- **Unity UI (uGUI):** docs.unity3d.com (Canvas, anchors, event system)
- **Unity TextMeshPro:** Vector fonts, better than legacy text
- **Unreal UMG:** docs.unrealengine.com (Widget blueprints, animations)
- **Unreal Slate:** Low-level UI framework (for editor/engine UI)

**Performance Profiling:**
- **Unity Profiler:** Analyze CPU/GPU, UI draw calls, frame time
- **Unreal Insights:** Frame analysis, rendering breakdown
- **RenderDoc:** Frame capture, inspect draw calls, shaders
- **PIX (Windows/Xbox):** Microsoft's GPU profiler

**Input Standards:**
- **Xbox Controller:** docs.microsoft.com (button mapping, guidelines)
- **PlayStation Controller:** developer.playstation.com (button conventions)
- **Steam Input:** docs.steampowered.com (controller API, supports all controllers)
- **Touch Standards:** 60x60px minimum for game controls (larger than 44-48px mobile UI)

**Accessibility Standards:**
- **Game Accessibility Guidelines:** gameaccessibilityguidelines.com (free, comprehensive)
- **Xbox Accessibility:** xbox.com/accessibility (guidelines, best practices)
- **CVAA (US Law):** Advanced communications features must be accessible (multiplayer chat, menus)
- **WCAG 2.1:** Apply to game UI text (contrast 4.5:1, resizable text)

**Books:**
- **"The Design of Everyday Things"** by Don Norman (general design principles, affordances)
- **"Game UI Discoveries"** by Heck Media (game UI screenshots, pattern library)
- **"Game Feel"** by Steve Swink (feedback, responsiveness, juice in games)
- **"Don't Make Me Think"** by Steve Krug (usability principles, apply to game menus)

**Example Games to Study:**
- **HUD Design:** Halo (minimalist FPS), The Division (diegetic AR), God of War (minimal one-take camera)
- **Menu Systems:** The Witcher 3 (RPG inventory), Destiny 2 (live-service menus), Civilization VI (strategy UI)
- **Mobile Controls:** PUBG Mobile (touch controls), Call of Duty Mobile (layout customization)
- **Accessibility:** The Last of Us Part II (extensive accessibility options, industry-leading)
