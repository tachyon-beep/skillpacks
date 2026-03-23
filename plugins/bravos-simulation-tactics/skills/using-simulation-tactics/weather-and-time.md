
# Weather and Time Systems

**When to use this skill**: When implementing day/night cycles, weather systems, seasonal changes, or time-based gameplay mechanics in games. Critical for survival games, open-world games, farming simulators, and any game where time and weather affect gameplay.

**What this skill provides**: Comprehensive understanding of time-of-day systems (sun angle calculation, twilight), dynamic weather (rain, snow, fog), performance-optimized particle systems, gameplay integration patterns, seasonal simulation, time acceleration, and visibility management to maintain 60 FPS while keeping the game playable.


## Core Concepts

### Time of Day Systems

**Solar Angle Calculation**
- **Real-world sun path**: Sun rises in east, peaks at south (northern hemisphere), sets in west
- **Solar elevation angle**: Varies with latitude and season (0° at horizon, 90° at zenith)
- **Azimuth angle**: Horizontal direction (0° = north, 90° = east, 180° = south, 270° = west)
- **Use case**: Realistic sun movement, shadow direction, day length variation

**Twilight Phases**
- **Civil twilight**: Sun 0° to -6° below horizon (still visible light)
- **Nautical twilight**: Sun -6° to -12° below horizon (horizon barely visible)
- **Astronomical twilight**: Sun -12° to -18° below horizon (darkest before true night)
- **Gameplay impact**: Smooth transition from day to night (no jarring darkness)

**Time Acceleration**
- **Real-time**: 1 second game = 1 second real-time (Animal Crossing)
- **Accelerated**: 1 minute game = 1 day in-game (Minecraft: 20 minutes = 24 hours)
- **Player-controlled**: Variable speed (Stardew Valley: speed up while sleeping)
- **Paused time**: Strategic games pause time during combat (XCOM)

### Weather Systems

**Weather Types and Properties**
- **Clear**: High visibility (1000m+), no movement penalty, minimal particles
- **Rain**: Reduced visibility (500m), slight movement penalty, moderate particles
- **Heavy Rain**: Low visibility (200m), movement penalty, high particles
- **Snow**: Moderate visibility (300m), movement penalty, moderate particles
- **Blizzard**: Very low visibility (100m), major movement penalty, high particles
- **Fog**: Very low visibility (50m), no movement penalty, zero particles (post-process)
- **Thunderstorm**: Periodic flashes, audio cues, lightning strikes

**Weather Simulation Models**

**Static Weather**: Pre-scripted weather patterns
- **Use case**: Scripted story moments, performance-critical scenarios
- **Pros**: Predictable, performant, no simulation cost
- **Cons**: Repetitive, not dynamic

**Procedural Weather**: Dynamic simulation based on rules
- **Use case**: Open-world games, long play sessions
- **Pros**: Varied, emergent patterns, replayability
- **Cons**: Can be unpredictable, requires tuning

**Markov Chain Weather**: Probability-based state transitions
- **Method**: Each weather type has transition probabilities to other types
- **Example**: Clear 70% → Clear, 20% → Cloudy, 10% → Rain
- **Pros**: Realistic patterns, controllable, prevents rapid oscillation
- **Cons**: Requires weather state machine

**Seasonal Variation**
- **Spring**: High rain probability, moderate temperatures
- **Summer**: High clear probability, hot temperatures
- **Fall**: Moderate rain, temperature drops
- **Winter**: High snow probability, cold temperatures, shorter days

### Particle System Optimization

**Performance Budgets**
- **Target frame time**: 16.67ms for 60 FPS
- **Particle budget**: 1-2ms per frame (5-10% of frame time)
- **Maximum particles**: 1,000-5,000 depending on platform (PC higher, mobile lower)
- **Particle update cost**: ~0.001ms per particle (CPU), ~0.0002ms (GPU)

**Level of Detail (LOD)**
- **Near camera (0-50m)**: Full particle density, full physics
- **Medium distance (50-200m)**: 50% particle density, simplified physics
- **Far distance (200m+)**: 25% particle density, no physics (billboards)
- **Out of view**: No particles (frustum culling)

**Object Pooling**
- **Problem**: Creating/destroying particles every frame causes GC pressure
- **Solution**: Pre-allocate pool, recycle particles
- **Pool size**: 2× maximum active particles (allows burst without allocation)

**Spatial Culling**
- **Frustum culling**: Don't render particles outside camera view
- **Distance culling**: Don't simulate particles beyond visibility range
- **Occlusion culling**: Don't render particles behind solid objects

### Visibility Management

**Gameplay Visibility vs Atmospheric Realism**
- **Problem**: Realistic night is pitch black (unplayable)
- **Solution**: Minimum ambient light for gameplay (0.15-0.25 even at midnight)
- **Trick**: Blue tint simulates moonlight without true darkness

**Fog Distance Curve**
- **Linear fog**: Visibility fades linearly with distance
- **Exponential fog**: Realistic, visibility drops off exponentially
- **Exponential squared**: Most realistic, dense fog feel
- **Gameplay consideration**: Too much fog frustrates players, balance with gameplay needs

**Dynamic Visibility**
- **Weather-based**: Rain/snow/fog reduce visibility distance
- **Time-based**: Night reduces visibility (but not below minimum)
- **Additive**: Multiple factors combine (night + fog = very limited visibility)


## Decision Frameworks

### Framework 1: Real-Time vs Accelerated Time

```
START: What type of game am I building?

├─ REAL-TIME EXPERIENCE?
│  ├─ Social/multiplayer game? → Use REAL-TIME CLOCK
│  │  - Animal Crossing: Real-world time = game time
│  │  - MMOs: Synchronized time across players
│  │  - Pros: Shared experience, event scheduling
│  │  - Cons: Players can't skip boring parts
│  │
│  └─ Single-player atmospheric? → Use REAL-TIME with PAUSE
│     - Survival horror: Real-time tension
│     - Flight simulators: Real-time weather
│     - Add pause/fast-forward for player convenience
│
├─ GAMEPLAY PACING REQUIRES FAST TIME?
│  ├─ Farming/crafting mechanics? → Use ACCELERATED TIME
│  │  - Stardew Valley: 1 second real = 1 minute game
│  │  - Minecraft: 20 minutes real = 1 day game
│  │  - Allows crops to grow in reasonable playtime
│  │  - Players experience full day/night in one session
│  │
│  └─ Long-term progression? → Use ACCELERATED with SLEEP
│     - Skip to next day when sleeping
│     - Time passes while player away (mobile games)
│
├─ STRATEGIC/TURN-BASED?
│  └─ Use PAUSED TIME during decisions
│     - XCOM: Time stops during combat turns
│     - Civilization: Discrete turns (day/night cosmetic only)
│     - RTS: Can pause and issue orders
│
└─ PLAYER CHOICE?
   └─ Provide TIME CONTROL UI
      - Speed slider: 1x, 2x, 5x, 10x
      - "Wait" button: Skip to morning/night
      - Strategic layer: Pause/slow/fast
```

**Example Decision**: Survival game with crafting
- **Chosen**: Accelerated time (30 min real = 24 hours game)
- **Reasoning**: Players need full day/night cycle per session, crafting takes minutes not hours
- **Time control**: Let players sleep to skip night, speed up to 3x when safe

### Framework 2: Static vs Dynamic Weather

```
START: How important is weather to my gameplay?

├─ WEATHER IS COSMETIC ONLY?
│  └─ Use STATIC WEATHER with scripted changes
│     - Pre-defined weather for each level/mission
│     - Lighter on performance
│     - Predictable for testing
│     - Example: Linear FPS games (scripted rain in mission 3)
│
├─ WEATHER AFFECTS GAMEPLAY SIGNIFICANTLY?
│  ├─ Short sessions (< 1 hour)? → Use PROCEDURAL with SHORT CYCLE
│  │  - Ensure players experience weather variety
│  │  - Faster transitions (10-15 minute cycles)
│  │  - Example: PUBG (random weather per match)
│  │
│  └─ Long sessions (1+ hours)? → Use PROCEDURAL with REALISTIC CYCLE
│     - Slower transitions (30-60 minute cycles)
│     - Weather affects strategy (take cover in storm)
│     - Example: Zelda BOTW (dynamic weather with gameplay effects)
│
├─ SEASONAL PROGRESSION IMPORTANT?
│  └─ Use SEASONAL SYSTEM with weather probability curves
│     - Spring: 60% rain, 30% clear, 10% cloudy
│     - Summer: 70% clear, 20% cloudy, 10% rain
│     - Fall: 40% clear, 40% cloudy, 20% rain
│     - Winter: 50% snow, 30% clear, 20% overcast
│
└─ MULTIPLAYER SYNCHRONIZATION NEEDED?
   └─ Use SERVER-AUTHORITATIVE weather
      - Server decides weather, clients render
      - All players see same weather
      - Important for competitive games (fair visibility)
```

**Example Decision**: Open-world survival game
- **Chosen**: Dynamic procedural weather with seasonal variation
- **Reasoning**: Long sessions, weather affects gameplay (shelter needed in rain)
- **Implementation**: Markov chain transitions, 45-min average weather duration

### Framework 3: Cosmetic vs Gameplay-Affecting Weather

```
START: Should weather affect gameplay?

├─ WEATHER AS ATMOSPHERE ONLY?
│  └─ Cosmetic effects
│     - Visual particles (rain, snow)
│     - Audio (thunder, wind)
│     - No mechanical effects
│     - Use when: Story-driven games where consistency > variance
│
├─ WEATHER AS MINOR MODIFIER?
│  └─ Subtle gameplay effects
│     - Visibility slightly reduced (800m → 600m)
│     - Movement speed -5% in rain
│     - Audio masking (harder to hear enemies)
│     - Use when: Competitive games (small effects for fairness)
│
├─ WEATHER AS MAJOR MECHANIC?
│  └─ Significant gameplay effects
│     - Visibility heavily reduced (fog: 1000m → 100m)
│     - Movement penalty (snow: -25% speed)
│     - Health effects (hypothermia in blizzard)
│     - Resource requirements (shelter from rain)
│     - Use when: Survival games, tactical advantages
│
└─ WEATHER AS CORE SYSTEM?
   └─ Central gameplay pillar
      - Plan activities around weather
      - Specific gear for weather types
      - Weather-dependent quests
      - Example: Death Stranding (rain damages cargo)
      - Example: Rain World (rain cycle is core mechanic)
```

**Decision Matrix**:
| Game Type | Weather Role | Visibility Impact | Movement Impact | Example |
|-----------|-------------|------------------|----------------|---------|
| Arena FPS | Cosmetic | None | None | Quake |
| Battle Royale | Minor | -20% range | None | PUBG |
| Open-world | Major | -50% in fog | -25% in snow | Zelda BOTW |
| Survival | Core | -70% in blizzard | -40% | The Long Dark |

**Example Decision**: Survival game
- **Chosen**: Weather as major mechanic
- **Effects**: Rain requires shelter, snow reduces movement, fog limits visibility
- **Balance**: Weather gives strategic depth but doesn't feel unfair


## Implementation Patterns

### Pattern 1: Smooth Time-of-Day System with Solar Math

**Problem**: Hardcoded hour-to-angle mapping looks unnatural. Binary day/night creates jarring transitions.

**Solution**: Calculate sun position using realistic solar math, implement twilight phases.

```python
import math

class TimeOfDaySystem:
    def __init__(self, day_length_seconds=1200, latitude=45.0):
        """
        day_length_seconds: Real-time seconds for full 24-hour cycle
        latitude: Degrees north (0-90) affects day length and sun angle
        """
        self.day_length_seconds = day_length_seconds
        self.latitude_rad = math.radians(latitude)
        self.current_time = 12.0  # Hours (0-24), start at noon
        self.day_of_year = 172  # Day 172 = summer solstice (longest day)
        self.time_scale = 1.0  # Multiplier for acceleration

    def update(self, delta_time):
        """Update time, wrapping at 24 hours"""
        hours_per_second = 24.0 / self.day_length_seconds
        self.current_time += delta_time * hours_per_second * self.time_scale

        if self.current_time >= 24.0:
            self.current_time -= 24.0
            self.day_of_year = (self.day_of_year + 1) % 365

    def get_sun_position(self):
        """Calculate sun elevation and azimuth angles"""
        # Solar declination (tilt of Earth's axis)
        # Ranges from -23.44° (winter) to +23.44° (summer)
        day_angle = 2 * math.pi * (self.day_of_year - 81) / 365
        declination = math.radians(23.44) * math.sin(day_angle)

        # Hour angle: Sun's position relative to solar noon
        # -180° at midnight, 0° at noon, +180° at next midnight
        hour_angle = math.radians(15.0 * (self.current_time - 12.0))

        # Solar elevation angle (altitude)
        sin_elevation = (math.sin(self.latitude_rad) * math.sin(declination) +
                        math.cos(self.latitude_rad) * math.cos(declination) *
                        math.cos(hour_angle))
        elevation = math.asin(max(-1, min(1, sin_elevation)))

        # Solar azimuth angle
        cos_azimuth = ((math.sin(declination) -
                       math.sin(self.latitude_rad) * sin_elevation) /
                      (math.cos(self.latitude_rad) * math.cos(elevation)))
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.acos(cos_azimuth)

        # Correct azimuth for afternoon (hour angle > 0)
        if hour_angle > 0:
            azimuth = 2 * math.pi - azimuth

        return math.degrees(elevation), math.degrees(azimuth)

    def get_ambient_light_factor(self):
        """Calculate ambient light with twilight transitions"""
        elevation, _ = self.get_sun_position()

        if elevation > 0:
            # Daytime: Full light
            return 1.0
        elif elevation > -6:
            # Civil twilight: Smooth transition
            t = (elevation + 6) / 6  # 0 to 1
            return 0.25 + (0.75 * t)  # 0.25 to 1.0
        elif elevation > -12:
            # Nautical twilight
            t = (elevation + 12) / 6
            return 0.15 + (0.10 * t)  # 0.15 to 0.25
        else:
            # Night: Minimum light for gameplay
            return 0.15  # Never truly black (moonlight simulation)

    def get_sun_color(self):
        """Calculate sun color based on elevation"""
        elevation, _ = self.get_sun_position()

        if elevation > 30:
            # High sun: White
            return (1.0, 1.0, 0.95)
        elif elevation > 0:
            # Low sun: Orange-red
            t = elevation / 30
            return (1.0, 0.6 + 0.4*t, 0.3 + 0.65*t)
        elif elevation > -6:
            # Sunset: Deep red
            t = (elevation + 6) / 6
            return (0.9 + 0.1*t, 0.3*t, 0.1*t)
        else:
            # Night: Dark blue (moonlight)
            return (0.2, 0.3, 0.5)

    def set_time_scale(self, scale):
        """Change time acceleration: 0=paused, 1=normal, 2+=faster"""
        self.time_scale = max(0, scale)

    def skip_to_time(self, target_hour):
        """Instantly jump to specific time (for 'wait' or 'sleep' actions)"""
        self.current_time = target_hour % 24.0
```

**Benefits**:
- Realistic sun movement (rises east, sets west)
- Smooth twilight transitions (no jarring darkness)
- Day length varies with season (longer summer days)
- Minimum ambient light (0.15) keeps game playable at night
- Configurable latitude and day length

**When to use**:
- Open-world games with visual realism
- Games where sun position matters (shadows, solar panels)
- Long play sessions where players experience full day

**Real-world example**: Red Dead Redemption 2 uses accurate solar calculations for realistic lighting throughout the day.

### Pattern 2: Budget-Constrained Particle System with LOD

**Problem**: Naive particle systems spawn unlimited particles, causing FPS death.

**Solution**: Enforce hard particle limit, use LOD based on distance, pool objects.

```python
import random
from collections import deque

class Particle:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.lifetime = 0.0
        self.max_lifetime = 1.0
        self.active = False

class WeatherParticleSystem:
    def __init__(self, max_particles=5000):
        self.max_particles = max_particles

        # Pre-allocate particle pool (NO allocations during gameplay)
        self.particle_pool = [Particle() for _ in range(max_particles)]
        self.active_particles = []
        self.free_particles = deque(self.particle_pool)

        # LOD settings
        self.lod_near = 50.0  # Full density
        self.lod_medium = 150.0  # 50% density
        self.lod_far = 300.0  # 25% density

        # Performance tracking
        self.time_budget_ms = 2.0  # Max 2ms per frame
        self.spawn_this_frame = 0

    def spawn_rain(self, camera_pos, intensity, delta_time):
        """Spawn rain particles with LOD and budget constraints"""
        # Calculate spawn budget based on frame time
        base_spawn_rate = intensity * 1000  # Particles per second
        desired_spawn = int(base_spawn_rate * delta_time)

        # LOD: Reduce spawn in layers by distance
        near_spawn = desired_spawn  # Full density near camera
        medium_spawn = desired_spawn // 2  # 50% at medium distance
        far_spawn = desired_spawn // 4  # 25% at far distance

        # Enforce particle cap
        available_slots = self.max_particles - len(self.active_particles)
        total_spawn = min(near_spawn + medium_spawn + far_spawn, available_slots)

        if total_spawn == 0:
            return  # At capacity or no budget

        spawned = 0

        # Spawn near particles (0-50m)
        for _ in range(min(near_spawn, available_slots)):
            if not self.free_particles:
                break

            particle = self.free_particles.popleft()
            self._initialize_rain_particle(particle, camera_pos, self.lod_near)
            self.active_particles.append(particle)
            spawned += 1

        # Spawn medium particles (50-150m)
        for _ in range(min(medium_spawn, available_slots - spawned)):
            if not self.free_particles:
                break

            particle = self.free_particles.popleft()
            self._initialize_rain_particle(particle, camera_pos, self.lod_medium,
                                          offset_min=self.lod_near)
            self.active_particles.append(particle)
            spawned += 1

        # Spawn far particles (150-300m) - only if budget allows
        remaining_budget = available_slots - spawned
        for _ in range(min(far_spawn, remaining_budget)):
            if not self.free_particles:
                break

            particle = self.free_particles.popleft()
            self._initialize_rain_particle(particle, camera_pos, self.lod_far,
                                          offset_min=self.lod_medium)
            self.active_particles.append(particle)

    def _initialize_rain_particle(self, particle, camera_pos, max_distance,
                                  offset_min=0.0):
        """Initialize a rain particle at random position"""
        # Random position in cylindrical volume around camera
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(offset_min, max_distance)

        particle.x = camera_pos[0] + distance * math.cos(angle)
        particle.z = camera_pos[2] + distance * math.sin(angle)
        particle.y = camera_pos[1] + random.uniform(30, 50)  # High in sky

        # Rain falls straight down with slight wind
        particle.vx = random.uniform(-0.5, 0.5)
        particle.vy = -10.0  # Fall speed
        particle.vz = random.uniform(-0.5, 0.5)

        particle.lifetime = 0.0
        particle.max_lifetime = random.uniform(3.0, 5.0)
        particle.active = True

    def update(self, delta_time, camera_pos):
        """Update all active particles with time budget"""
        import time
        start_time = time.perf_counter()

        particles_to_remove = []

        for particle in self.active_particles:
            # Update position
            particle.x += particle.vx * delta_time
            particle.y += particle.vy * delta_time
            particle.z += particle.vz * delta_time

            particle.lifetime += delta_time

            # Remove if below ground or lifetime expired
            if particle.y < 0 or particle.lifetime > particle.max_lifetime:
                particle.active = False
                particles_to_remove.append(particle)
                continue

            # Distance culling: Remove particles far from camera
            dx = particle.x - camera_pos[0]
            dz = particle.z - camera_pos[2]
            dist_sq = dx*dx + dz*dz

            if dist_sq > self.lod_far * self.lod_far:
                particle.active = False
                particles_to_remove.append(particle)

            # Budget check: Stop updating if over time budget
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.time_budget_ms:
                break  # Update remaining particles next frame

        # Return particles to pool
        for particle in particles_to_remove:
            self.active_particles.remove(particle)
            self.free_particles.append(particle)

    def get_particle_count(self):
        return len(self.active_particles)
```

**Benefits**:
- Hard cap on particles (5,000 max) prevents unbounded growth
- Object pooling eliminates GC pressure (zero allocations during gameplay)
- LOD reduces particles by distance (75% at 300m)
- Time budget prevents frame time spikes (stops at 2ms)
- Distance culling removes particles outside range

**Performance**:
- CPU cost: ~1-2ms per frame (5000 particles)
- Memory: Fixed (pre-allocated pool)
- FPS: Stable 60 FPS

**When to use**:
- Any game with weather particles (rain, snow)
- Performance-critical scenarios (mobile, VR)
- Large open worlds (distance culling essential)

**Real-world example**: Zelda BOTW limits rain particles to ~3,000, uses heavy LOD culling beyond 100m.

### Pattern 3: Gameplay-Integrated Weather System

**Problem**: Weather is cosmetic-only, doesn't affect player strategy or tactics.

**Solution**: Weather modifies visibility, movement, audio, and environmental hazards.

```python
from enum import Enum
from dataclasses import dataclass

class WeatherType(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    BLIZZARD = "blizzard"
    FOG = "fog"
    THUNDERSTORM = "thunderstorm"

@dataclass
class WeatherProperties:
    """Gameplay properties for each weather type"""
    visibility_range: float  # Meters
    movement_modifier: float  # 1.0 = normal, 0.75 = 25% slower
    audio_masking: float  # 0-1, higher = harder to hear
    particle_count_multiplier: float  # Relative to base
    ambient_light_modifier: float  # 1.0 = normal, 0.7 = 30% darker

# Weather property database
WEATHER_PROPERTIES = {
    WeatherType.CLEAR: WeatherProperties(
        visibility_range=1000.0,
        movement_modifier=1.0,
        audio_masking=0.0,
        particle_count_multiplier=0.0,
        ambient_light_modifier=1.0
    ),
    WeatherType.RAIN: WeatherProperties(
        visibility_range=500.0,
        movement_modifier=0.95,
        audio_masking=0.3,
        particle_count_multiplier=1.0,
        ambient_light_modifier=0.85
    ),
    WeatherType.HEAVY_RAIN: WeatherProperties(
        visibility_range=200.0,
        movement_modifier=0.85,
        audio_masking=0.6,
        particle_count_multiplier=2.0,
        ambient_light_modifier=0.7
    ),
    WeatherType.SNOW: WeatherProperties(
        visibility_range=300.0,
        movement_modifier=0.75,  # Slow in snow
        audio_masking=0.4,
        particle_count_multiplier=1.5,
        ambient_light_modifier=1.1  # Snow reflects light
    ),
    WeatherType.BLIZZARD: WeatherProperties(
        visibility_range=100.0,
        movement_modifier=0.6,  # Very slow
        audio_masking=0.7,
        particle_count_multiplier=3.0,
        ambient_light_modifier=0.8
    ),
    WeatherType.FOG: WeatherProperties(
        visibility_range=50.0,  # Very limited
        movement_modifier=1.0,  # Fog doesn't slow
        audio_masking=0.2,
        particle_count_multiplier=0.0,  # Fog is post-process, no particles
        ambient_light_modifier=0.9
    ),
}

class GameplayWeatherSystem:
    def __init__(self):
        self.current_weather = WeatherType.CLEAR
        self.transition_progress = 1.0  # 0-1, 1=fully transitioned
        self.transition_duration = 5.0  # Seconds
        self.target_weather = WeatherType.CLEAR

    def change_weather(self, new_weather, transition_time=5.0):
        """Smoothly transition to new weather"""
        if new_weather == self.current_weather:
            return

        self.target_weather = new_weather
        self.transition_duration = transition_time
        self.transition_progress = 0.0

    def update(self, delta_time):
        """Update weather transition"""
        if self.transition_progress < 1.0:
            self.transition_progress += delta_time / self.transition_duration
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.current_weather = self.target_weather

    def get_visibility_range(self):
        """Get current visibility distance in meters"""
        if self.transition_progress >= 1.0:
            return WEATHER_PROPERTIES[self.current_weather].visibility_range

        # Interpolate during transition
        current_props = WEATHER_PROPERTIES[self.current_weather]
        target_props = WEATHER_PROPERTIES[self.target_weather]

        t = self.transition_progress
        return current_props.visibility_range * (1-t) + target_props.visibility_range * t

    def get_movement_modifier(self):
        """Get movement speed multiplier (1.0 = normal)"""
        if self.transition_progress >= 1.0:
            return WEATHER_PROPERTIES[self.current_weather].movement_modifier

        current_props = WEATHER_PROPERTIES[self.current_weather]
        target_props = WEATHER_PROPERTIES[self.target_weather]

        t = self.transition_progress
        return current_props.movement_modifier * (1-t) + target_props.movement_modifier * t

    def get_audio_masking(self):
        """Get audio masking factor (0=clear, 1=fully masked)"""
        if self.transition_progress >= 1.0:
            return WEATHER_PROPERTIES[self.current_weather].audio_masking

        current_props = WEATHER_PROPERTIES[self.current_weather]
        target_props = WEATHER_PROPERTIES[self.target_weather]

        t = self.transition_progress
        return current_props.audio_masking * (1-t) + target_props.audio_masking * t

    def apply_to_player(self, player):
        """Apply weather effects to player"""
        # Movement speed
        weather_speed = self.get_movement_modifier()
        player.movement_speed = player.base_movement_speed * weather_speed

        # Visibility (for AI detection, fog of war)
        player.visibility_range = self.get_visibility_range()

        # Audio (for enemy hearing player)
        player.audio_masking = self.get_audio_masking()

    def apply_to_camera(self, camera):
        """Apply weather effects to camera/rendering"""
        # Fog distance for rendering
        visibility = self.get_visibility_range()
        camera.fog_start = visibility * 0.5
        camera.fog_end = visibility

        # Ambient light modifier
        current_props = WEATHER_PROPERTIES[self.current_weather]
        target_props = WEATHER_PROPERTIES[self.target_weather]
        t = self.transition_progress

        light_mod = (current_props.ambient_light_modifier * (1-t) +
                    target_props.ambient_light_modifier * t)
        camera.ambient_light_scale = light_mod
```

**Benefits**:
- Weather directly affects gameplay (movement, visibility, audio)
- Smooth transitions prevent jarring changes
- Easy to balance (modify property values)
- AI can react to weather (seek shelter, change tactics)

**Gameplay Applications**:
- **Stealth**: Use rain to mask footsteps, fog to avoid detection
- **Combat**: Heavy rain reduces visibility, favors close-range
- **Survival**: Blizzard forces player to find shelter (hypothermia risk)
- **Strategy**: Plan attacks during favorable weather

**When to use**:
- Survival games (weather is hazard)
- Stealth games (weather affects detection)
- Open-world games (weather adds variety)

**Real-world example**: Metal Gear Solid V uses rain to mask noise, sandstorms to reduce visibility.

### Pattern 4: Markov Chain Weather Transitions with Seasonal Variation

**Problem**: Completely random weather feels unnatural and lacks patterns.

**Solution**: Use Markov chain with season-dependent transition probabilities.

```python
import random

class Season(Enum):
    SPRING = 0
    SUMMER = 1
    FALL = 2
    WINTER = 3

class WeatherSimulation:
    def __init__(self):
        self.current_weather = WeatherType.CLEAR
        self.current_season = Season.SPRING
        self.time_in_current_weather = 0.0
        self.min_weather_duration = 300.0  # 5 minutes minimum

        # Transition probability matrices (current → next)
        # Rows: current weather, Columns: next weather
        # Order: CLEAR, CLOUDY, RAIN, HEAVY_RAIN, SNOW, BLIZZARD, FOG

        self.transition_probabilities = {
            Season.SPRING: {
                WeatherType.CLEAR: [0.4, 0.4, 0.15, 0.05, 0.0, 0.0, 0.0],
                WeatherType.CLOUDY: [0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0],
                WeatherType.RAIN: [0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 0.0],
                WeatherType.HEAVY_RAIN: [0.1, 0.2, 0.5, 0.2, 0.0, 0.0, 0.0],
                WeatherType.FOG: [0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.2],
            },
            Season.SUMMER: {
                WeatherType.CLEAR: [0.7, 0.2, 0.05, 0.05, 0.0, 0.0, 0.0],
                WeatherType.CLOUDY: [0.5, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0],
                WeatherType.RAIN: [0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0],
                WeatherType.HEAVY_RAIN: [0.2, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0],  # Storms
                WeatherType.THUNDERSTORM: [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
            },
            Season.FALL: {
                WeatherType.CLEAR: [0.4, 0.4, 0.1, 0.05, 0.0, 0.0, 0.05],
                WeatherType.CLOUDY: [0.3, 0.4, 0.2, 0.05, 0.0, 0.0, 0.05],
                WeatherType.RAIN: [0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 0.0],
                WeatherType.FOG: [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.4],
            },
            Season.WINTER: {
                WeatherType.CLEAR: [0.5, 0.2, 0.0, 0.0, 0.25, 0.05, 0.0],
                WeatherType.CLOUDY: [0.3, 0.3, 0.0, 0.0, 0.35, 0.05, 0.0],
                WeatherType.SNOW: [0.2, 0.2, 0.0, 0.0, 0.5, 0.1, 0.0],
                WeatherType.BLIZZARD: [0.1, 0.1, 0.0, 0.0, 0.6, 0.2, 0.0],
            },
        }

        # Weather types in order for indexing
        self.weather_types = [
            WeatherType.CLEAR,
            WeatherType.CLOUDY,
            WeatherType.RAIN,
            WeatherType.HEAVY_RAIN,
            WeatherType.SNOW,
            WeatherType.BLIZZARD,
            WeatherType.FOG,
        ]

    def update(self, delta_time):
        """Update weather simulation"""
        self.time_in_current_weather += delta_time

        # Only consider transition after minimum duration
        if self.time_in_current_weather < self.min_weather_duration:
            return

        # Check for transition (1% chance per second after minimum)
        transition_chance = delta_time * 0.01
        if random.random() < transition_chance:
            self._transition_weather()
            self.time_in_current_weather = 0.0

    def _transition_weather(self):
        """Choose next weather based on Markov chain"""
        season_probs = self.transition_probabilities[self.current_season]

        if self.current_weather not in season_probs:
            # Current weather not valid for season, force to CLEAR
            self.current_weather = WeatherType.CLEAR
            return

        probabilities = season_probs[self.current_weather]

        # Weighted random choice
        rand_val = random.random()
        cumulative = 0.0

        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val < cumulative:
                self.current_weather = self.weather_types[i]
                break

    def set_season(self, season):
        """Change season, may trigger immediate weather change"""
        old_season = self.current_season
        self.current_season = season

        # Check if current weather is valid for new season
        season_probs = self.transition_probabilities[season]
        if self.current_weather not in season_probs:
            # Force transition to valid weather
            self._transition_weather()
```

**Benefits**:
- Natural weather patterns (not purely random)
- Seasonal variation (snow in winter, not summer)
- Prevents rapid oscillation (minimum duration)
- Controllable probabilities (easy to tune)

**Pattern Characteristics**:
- **Spring**: Rainy (60% chance), moderate temperatures
- **Summer**: Clear and hot (70% clear), occasional thunderstorms
- **Fall**: Cloudy and foggy, transition to cold
- **Winter**: Snow and blizzards (40% snow), cold

**When to use**:
- Open-world games with seasons
- Long play sessions (players notice patterns)
- Realistic simulation games

**Real-world example**: Animal Crossing uses Markov-like weather with seasonal variation.

### Pattern 5: Time Control UI and "Wait" Mechanic

**Problem**: Players forced to wait through boring periods (night, storms).

**Solution**: Provide time acceleration controls and "wait until" actions.

```python
class TimeControlSystem:
    def __init__(self, time_of_day_system, weather_system):
        self.time_system = time_of_day_system
        self.weather_system = weather_system

        self.available_speeds = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]  # 0=pause
        self.current_speed_index = 2  # Start at 1.0x

        self.waiting = False
        self.wait_target_hour = None
        self.wait_callback = None

    def increase_speed(self):
        """Increase time scale (up to 10x)"""
        if self.current_speed_index < len(self.available_speeds) - 1:
            self.current_speed_index += 1
            self.time_system.set_time_scale(
                self.available_speeds[self.current_speed_index]
            )

    def decrease_speed(self):
        """Decrease time scale (down to pause)"""
        if self.current_speed_index > 0:
            self.current_speed_index -= 1
            self.time_system.set_time_scale(
                self.available_speeds[self.current_speed_index]
            )

    def set_normal_speed(self):
        """Reset to 1x speed"""
        self.current_speed_index = 2  # 1.0x
        self.time_system.set_time_scale(1.0)

    def wait_until_morning(self, callback=None):
        """Fast-forward to next morning (6 AM)"""
        self.wait_target_hour = 6.0
        self.waiting = True
        self.wait_callback = callback

        # Accelerate time during wait
        self.time_system.set_time_scale(60.0)  # 60x speed

    def wait_until_night(self, callback=None):
        """Fast-forward to next night (8 PM)"""
        self.wait_target_hour = 20.0
        self.waiting = True
        self.wait_callback = callback
        self.time_system.set_time_scale(60.0)

    def wait_for_hours(self, hours, callback=None):
        """Wait for specific number of hours"""
        target = (self.time_system.current_time + hours) % 24.0
        self.wait_target_hour = target
        self.waiting = True
        self.wait_callback = callback
        self.time_system.set_time_scale(60.0)

    def update(self, delta_time):
        """Check if wait target reached"""
        if not self.waiting:
            return

        current = self.time_system.current_time
        target = self.wait_target_hour

        # Check if we've passed the target hour
        # Handle wraparound (23:00 → 6:00)
        if target > current:
            if current >= target:
                self._complete_wait()
        else:
            # Wrapped around midnight
            if current >= target and current < target + 1.0:
                self._complete_wait()

    def _complete_wait(self):
        """Finish waiting, restore normal time"""
        self.waiting = False
        self.set_normal_speed()

        if self.wait_callback:
            self.wait_callback()
            self.wait_callback = None

    def can_wait(self, player):
        """Check if player can wait (safe location, not in combat)"""
        if player.in_combat:
            return False, "Cannot wait during combat"

        if player.enemies_nearby():
            return False, "Enemies nearby"

        if not player.in_safe_zone():
            return False, "Not in safe location"

        return True, ""

# Example UI integration
class TimeControlUI:
    def __init__(self, time_control):
        self.time_control = time_control

    def render_time_controls(self, ui):
        """Render time speed controls"""
        speeds = time_control.available_speeds
        current_idx = time_control.current_speed_index

        ui.label(f"Time Speed: {speeds[current_idx]}x")

        if ui.button("<<"):  # Slower
            time_control.decrease_speed()

        if ui.button("||"):  # Pause
            time_control.current_speed_index = 0
            time_control.time_system.set_time_scale(0.0)

        if ui.button(">>"):  # Faster
            time_control.increase_speed()

    def render_wait_options(self, ui, player):
        """Render wait/sleep menu"""
        can_wait, reason = time_control.can_wait(player)

        if not can_wait:
            ui.label(f"Cannot wait: {reason}", color="red")
            return

        if ui.button("Wait until morning (6 AM)"):
            time_control.wait_until_morning(
                callback=lambda: player.restore_energy(50)
            )

        if ui.button("Wait until night (8 PM)"):
            time_control.wait_until_night()

        if ui.button("Wait 1 hour"):
            time_control.wait_for_hours(1)

        if ui.button("Wait 4 hours"):
            time_control.wait_for_hours(4)
```

**Benefits**:
- Players skip boring periods (night, waiting for shop to open)
- Strategic use (wait for enemies to leave area)
- Performance optimization (faster time = fewer frames to render)
- Quality-of-life feature (respects player's time)

**Design Considerations**:
- Only allow waiting in safe locations (no combat exploit)
- Restore energy/health during wait (reward for using feature)
- Fast-forward at 60x speed (visible but quick)
- Stop if interrupted (enemy appears)

**When to use**:
- Games with day/night cycles
- Survival games (wait for weather to clear)
- RPGs (wait for shops to open)

**Real-world examples**:
- Skyrim: "Wait" menu to skip time
- Stardew Valley: Sleep to skip to next day
- Zelda BOTW: Campfire rest to skip to morning/night

### Pattern 6: Fog as Post-Process Effect (Zero Particles)

**Problem**: Fog with particles is expensive and looks bad.

**Solution**: Use distance-based post-process fog (no particles needed).

```python
class FogSystem:
    def __init__(self):
        self.fog_enabled = False
        self.fog_density = 0.0  # 0-1
        self.fog_color = (0.7, 0.7, 0.75)  # Gray-white
        self.fog_start = 10.0  # Meters
        self.fog_end = 100.0  # Meters

    def update_fog_from_weather(self, weather_type):
        """Set fog based on weather"""
        if weather_type == WeatherType.FOG:
            self.fog_enabled = True
            self.fog_density = 0.8
            self.fog_start = 5.0
            self.fog_end = 50.0
            self.fog_color = (0.7, 0.7, 0.75)

        elif weather_type == WeatherType.RAIN:
            self.fog_enabled = True
            self.fog_density = 0.3
            self.fog_start = 50.0
            self.fog_end = 500.0
            self.fog_color = (0.6, 0.6, 0.7)

        elif weather_type == WeatherType.BLIZZARD:
            self.fog_enabled = True
            self.fog_density = 0.9
            self.fog_start = 10.0
            self.fog_end = 100.0
            self.fog_color = (0.9, 0.9, 1.0)  # White

        else:
            self.fog_enabled = False

    def get_shader_parameters(self):
        """Get parameters for fog shader"""
        return {
            'fog_enabled': self.fog_enabled,
            'fog_color': self.fog_color,
            'fog_start': self.fog_start,
            'fog_end': self.fog_end,
            'fog_density': self.fog_density,
        }

# Fragment shader (GLSL) for exponential fog
"""
uniform bool fog_enabled;
uniform vec3 fog_color;
uniform float fog_start;
uniform float fog_end;
uniform float fog_density;

void main() {
    vec3 color = texture(scene_texture, uv).rgb;

    if (fog_enabled) {
        float distance = length(frag_position - camera_position);

        // Exponential squared fog (most realistic)
        float fog_factor = distance / fog_end;
        fog_factor = exp(-fog_density * fog_factor * fog_factor);
        fog_factor = clamp(fog_factor, 0.0, 1.0);

        // Blend between scene color and fog color
        color = mix(fog_color, color, fog_factor);
    }

    frag_color = vec4(color, 1.0);
}
"""
```

**Benefits**:
- Zero particles (massive performance win)
- Better visual quality (smooth falloff)
- Easy to control (density, color, distance)
- Works with any weather

**When to use**:
- Fog weather type
- Distance-based visibility reduction
- Atmospheric depth cues

**Real-world example**: Nearly all modern games use post-process fog, not particle-based.


## Common Pitfalls

### Pitfall 1: Performance Death - Unbounded Particle Growth

**Symptom**: FPS drops from 60 to single digits after a few minutes of rain.

**Root Cause**:
```python
# WRONG: No particle limit
for _ in range(1000):  # Spawn 1000 per frame!
    particles.append(RainParticle())
# After 60 frames: 60,000 particles → 1 FPS
```

**Why it happens**: Developers test for a few seconds, don't notice particles accumulating.

**Fix**: Hard particle cap + object pooling
```python
# RIGHT: Enforce maximum
MAX_PARTICLES = 5000

if len(active_particles) < MAX_PARTICLES:
    particle = particle_pool.get()  # Reuse from pool
    active_particles.append(particle)
```

**Testing**: Run game for 5+ minutes with heavy rain, monitor particle count.

### Pitfall 2: Night Too Dark - Unplayable Visibility

**Symptom**: Players complain they can't see anything at night, quit game.

**Root Cause**:
```python
# WRONG: Realistic night (pitch black)
if is_night():
    ambient_light = 0.0  # Can't see ANYTHING
```

**Why it happens**: Developers prioritize realism over playability.

**Fix**: Minimum ambient light for gameplay
```python
# RIGHT: "Moonlight" minimum
if is_night():
    ambient_light = 0.15  # Dim but playable
    # Add blue tint to simulate moonlight
    ambient_color = (0.2, 0.3, 0.5)
```

**Balance**: Night should feel atmospheric, not frustrating.

**Alternative**: Provide torch/lantern that player must manage (fuel, battery).

### Pitfall 3: Weather Too Random - No Predictability

**Symptom**: Snow in summer, instant weather changes, feels chaotic.

**Root Cause**:
```python
# WRONG: Completely random
weather = random.choice(['clear', 'rain', 'snow', 'fog'])
# Can jump from clear to blizzard instantly!
```

**Why it happens**: Randomness is easy to implement, patterns require system design.

**Fix**: Use Markov chain with seasonal constraints
```python
# RIGHT: Pattern-based transitions
next_weather = markov_chain.transition(current_weather, current_season)
# Clear → Cloudy → Rain (gradual)
# No snow in summer (seasonal rules)
```

**Result**: Weather feels natural, players can anticipate changes.

### Pitfall 4: Instant Weather Changes - Jarring Transitions

**Symptom**: Weather switches instantly (clear → downpour in 1 frame).

**Root Cause**:
```python
# WRONG: Instant switch
if should_change_weather():
    weather = new_weather  # Instant!
```

**Fix**: Smooth transition over time
```python
# RIGHT: 5-second transition
if transitioning:
    t += delta_time / transition_duration  # 0 to 1
    intensity = lerp(old_intensity, new_intensity, t)
```

**Transition duration**:
- Clear → Cloudy: 10 seconds
- Cloudy → Rain: 5 seconds
- Rain → Clear: 15 seconds (gradual clearing)

### Pitfall 5: Particle Update Cost - No Culling

**Symptom**: Particles far from camera still tank FPS.

**Root Cause**:
```python
# WRONG: Update ALL particles
for particle in all_particles:
    particle.update()  # Even if 500m away!
```

**Fix**: Distance-based culling
```python
# RIGHT: Only update visible particles
for particle in all_particles:
    if distance(particle, camera) < fog_distance:
        particle.update()
    else:
        # Cull distant particles
        particle_pool.return(particle)
```

**Savings**: 50-70% of particle update cost.

### Pitfall 6: No LOD - Same Density Everywhere

**Symptom**: Performance poor even with particle cap.

**Root Cause**: Same particle density near and far.

**Fix**: LOD-based spawn
```python
# RIGHT: Reduce density with distance
if distance < 50:
    spawn_count = 100  # Full density
elif distance < 150:
    spawn_count = 50  # 50% density
elif distance < 300:
    spawn_count = 25  # 25% density
```

**Result**: 2-3× better performance with same visual quality.

### Pitfall 7: No Time Budget - Frame Spikes

**Symptom**: Occasional frame drops (60 → 40 FPS) when spawning particles.

**Root Cause**: Particle system takes too long some frames.

**Fix**: Time budget with early exit
```python
# RIGHT: Enforce budget
start = time.perf_counter()
for particle in particles:
    particle.update()
    if (time.perf_counter() - start) > 0.002:  # 2ms budget
        break  # Continue next frame
```

**Result**: Consistent frame pacing.

### Pitfall 8: GC Pressure - Creating Objects Every Frame

**Symptom**: Micro-stuttering, frame time spikes every few seconds.

**Root Cause**: Garbage collector running constantly.

**Fix**: Object pooling
```python
# WRONG: New objects every frame
particle = RainParticle()

# RIGHT: Reuse from pool
particle = particle_pool.get_or_create()
```

**Impact**: Eliminates GC spikes.

### Pitfall 9: No Gameplay Integration - Cosmetic Only

**Symptom**: Weather looks nice but doesn't matter strategically.

**Root Cause**: Weather is just visual, no mechanical effects.

**Fix**: Weather modifies gameplay
```python
# RIGHT: Weather affects mechanics
player.movement_speed *= weather.get_movement_modifier()
enemy_detection_range *= weather.get_visibility_modifier()
footstep_audio_range *= (1 - weather.get_audio_masking())
```

**Result**: Weather becomes tactical consideration.

### Pitfall 10: No Seasonal System - Snow in Summer

**Symptom**: Immersion broken by incorrect weather for season.

**Root Cause**: Weather independent of season.

**Fix**: Seasonal probability curves
```python
# RIGHT: Season determines weather chances
if season == SUMMER:
    weather_chances = {'clear': 0.7, 'rain': 0.3, 'snow': 0.0}
elif season == WINTER:
    weather_chances = {'clear': 0.3, 'rain': 0.0, 'snow': 0.7}
```


## Real-World Examples

### Example 1: Minecraft - Simple But Effective

**Time System**:
- 20-minute day/night cycle (accelerated 72×)
- Day: 10 minutes, Night: 7 minutes, Twilight: 3 minutes
- Synchronized across multiplayer (server-authoritative)

**Weather**:
- Simple binary: Clear or Rain
- Rain reduces sky brightness by 20%
- Rain extinguishes fire, fills cauldrons
- Thunder can strike and start fires

**Performance**:
- Rain particles: ~500-1000 (very simple)
- Heavy LOD (particles only near player)
- No complex weather simulation

**Key Insight**: Simplicity works if weather serves gameplay (rain fills cauldrons, enables fishing).

### Example 2: Zelda: Breath of the Wild - Gameplay-First Weather

**Time System**:
- Accelerated time (1 minute real = 1 hour game)
- Realistic sun path and shadows
- Time doesn't pass during cutscenes/menus

**Weather**:
- Dynamic procedural weather
- **Gameplay effects**:
  - Rain makes surfaces slippery (can't climb)
  - Lightning targets metal equipment (must unequip)
  - Cold/heat require appropriate clothing
  - Updrafts form during certain weather (gliding)

**Performance**:
- Particle budget: ~3,000 rain/snow particles
- Heavy LOD (75% reduction at 100m)
- Post-process fog (no particles)

**Key Insight**: Weather creates gameplay challenges and opportunities, not just atmosphere.

### Example 3: Animal Crossing - Real-Time Clock

**Time System**:
- Real-time clock (1 second real = 1 second game)
- Synchronized to system time
- Shops open/close at specific hours

**Weather**:
- Dynamic but gentle (no extreme weather)
- Seasonal variation (snow in December, rain in June)
- Weather affects villager behavior (stay indoors during rain)
- Meteors during clear nights

**Key Insight**: Real-time creates daily routine, encourages checking in regularly.

### Example 4: Red Dead Redemption 2 - Best-in-Class Transitions

**Time System**:
- Realistic sun path with latitude consideration
- Dynamic length of day/night (longer days in summer)
- Accurate sunrise/sunset colors

**Weather**:
- Extremely smooth transitions (5-15 minutes)
- Weather fronts visible in distance (see storm approaching)
- Regional weather (snow in mountains, clear in desert)
- Weather affects NPC behavior (seek shelter, change routes)

**Performance**:
- Advanced particle LOD
- GPU-based particle simulation
- Temporal reprojection for particles

**Key Insight**: Slow, smooth transitions make weather feel natural and immersive.

### Example 5: Skyrim - Magic and Weather Interaction

**Time System**:
- Accelerated time (1 minute real = 20 minutes game, configurable)
- "Wait" menu to skip time
- Time passes during fast travel

**Weather**:
- Regional weather patterns (more snow in north)
- Weather affects spells:
  - Lightning spells more powerful during thunderstorms
  - Fire spells less effective in rain
  - Frost spells more effective in snow
- Visibility reduced in fog/snow

**Key Insight**: Weather can integrate with core mechanics (magic system).

### Example 6: Don't Starve - Seasonal Survival

**Seasons**:
- 16 days per season (64-day year)
- Spring: Rain, flooding, aggressive bees
- Summer: Drought, heat stroke, fires
- Fall: Mild, good for preparing
- Winter: Freezing, reduced food, hound attacks

**Weather as Hazard**:
- Summer heat requires cooling
- Winter cold requires warming
- Rain reduces sanity, extinguishes fires
- Lightning can strike and kill player

**Key Insight**: Seasons and weather are core survival challenges, not cosmetic.


## Cross-References

### Within Bravos/Simulation-Tactics

**Physics Simulation Patterns** → Weather and Time
- Particle physics for rain/snow
- Wind forces affecting particles

**Spatial Partitioning** → Weather and Time
- Spatial grid for particle culling
- Region-based weather systems

**Traffic and Pathfinding** → Weather and Time
- Weather affects pathfinding costs (slower in snow)
- NPC AI reacts to weather (seek shelter)

### External Skillpacks

**Yzmir/Performance-Optimization**
- Object pooling for particles
- Time budgets and profiling
- Cache-friendly particle updates

**Axiom/Game-Engine-Patterns**
- Update loop integration
- Delta-time handling
- Time scaling and pause

**Lyra/Game-Feel**
- Weather feedback (audio, visuals)
- Camera effects (rain on lens)
- Smooth transitions

**Lyra/UX-Design**
- Time control UI
- Weather indicators
- Player communication of effects


## Testing Checklist

Use this checklist to verify your weather and time system:

### Performance

- [ ] Particle count never exceeds budget (5000 max)
- [ ] FPS stays above 60 with maximum weather particles
- [ ] No frame time spikes (time budget enforced)
- [ ] No GC pressure (object pooling used)
- [ ] Distance culling removes particles beyond visibility
- [ ] LOD reduces particles by 75% at 300m

### Gameplay Integration

- [ ] Weather affects player movement speed
- [ ] Weather affects visibility range
- [ ] Weather affects audio masking
- [ ] AI reacts to weather (seeks shelter, changes behavior)
- [ ] Weather creates strategic opportunities (stealth in fog)
- [ ] Weather creates strategic challenges (slow movement in snow)

### Visual Quality

- [ ] Smooth twilight transitions (no instant darkness)
- [ ] Night is dim but playable (minimum 0.15 ambient light)
- [ ] Weather transitions gradually (5-15 seconds)
- [ ] Sun moves realistically (east to west)
- [ ] Sun color changes with elevation (orange at sunset)
- [ ] Fog uses post-process (not particles)

### Time System

- [ ] Time of day progresses smoothly
- [ ] Day/night cycle is noticeable to player
- [ ] Time can be accelerated (2x, 5x, 10x)
- [ ] Time can be paused (strategic pause)
- [ ] "Wait" mechanic skips boring periods
- [ ] Time synchronizes in multiplayer (if applicable)

### Weather System

- [ ] Weather has seasonal variation (no snow in summer)
- [ ] Weather transitions follow patterns (Clear → Cloudy → Rain)
- [ ] Weather durations feel right (not too rapid)
- [ ] Weather types are distinct (clear visual difference)
- [ ] Weather probabilities are tuned (not too much rain)
- [ ] Extreme weather is rare (blizzards uncommon)

### Edge Cases

- [ ] Midnight is playable (not pitch black)
- [ ] Heavy rain/blizzard doesn't crash game
- [ ] Time wraps correctly at 24 hours
- [ ] Seasons transition smoothly
- [ ] Weather is consistent across multiplayer clients
- [ ] Time acceleration doesn't break physics

### User Experience

- [ ] Weather changes are noticeable but not jarring
- [ ] Time control UI is accessible and clear
- [ ] "Wait" option is available when appropriate
- [ ] Weather effects are communicated to player
- [ ] Night is atmospheric but not frustrating
- [ ] Players can plan around weather


**End of Skill**

This skill should enable you to build production-quality weather and time systems that:
1. Maintain 60 FPS with thousands of particles
2. Integrate weather into gameplay meaningfully
3. Balance realism with playability
4. Provide smooth, natural transitions
5. Give players control over time pacing
6. Create seasonal variety and patterns

Apply these patterns, avoid these pitfalls, and test thoroughly using the checklist above.
