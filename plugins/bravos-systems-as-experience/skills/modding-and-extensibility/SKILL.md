# Modding and Extensibility: Designing Games for Player Creativity

## Overview

The most successful games often aren't the ones with the most content—they're the ones that empower players to create their own content. Counter-Strike started as a Half-Life mod. DOTA emerged from a Warcraft 3 custom map. Minecraft's modding community has created experiences the original developers never imagined. When you design for extensibility, you're not just building a game—you're building a platform for infinite creativity.

This skill teaches you how to design games that can be extended, modded, and transformed by players. You'll learn from games like Skyrim (Creation Kit ecosystem), Minecraft (Java mods), Factorio (stable mod API), and Roblox (user-generated games). We'll cover plugin architectures, API design, content creation tools, security, performance, and community infrastructure.

**What You'll Learn:**
- Designing extensible architectures from day one
- Creating stable, well-documented mod APIs
- Building content creation tools for players
- Managing mod conflicts, dependencies, and versions
- Securing mod systems against exploits
- Balancing performance with flexibility
- Supporting thriving modding communities

**Prerequisites:**
- Understanding of game architecture and design patterns
- Programming experience (examples in C++/C#/Lua/Python)
- Familiarity with modded games as a player

---

## RED Phase: The Unmoddable Game

### Test Scenario

**Goal:** Build a tower defense game that supports player-created content (custom towers, enemies, maps).

**Baseline Implementation:** Traditional game architecture with no extensibility considerations.

### RED Failures Documented

Let's build the game **without** extensibility in mind and document every failure:

```cpp
// RED Failure #1: Monolithic, hardcoded tower system
class TowerDefenseGame {
private:
    // All game logic tightly coupled
    Player player;
    std::vector<Tower*> towers;
    std::vector<Enemy*> enemies;
    Map currentMap;

public:
    void Update(float dt) {
        // Hardcoded game logic - can't extend without source
        for (auto tower : towers) {
            if (tower->GetType() == TowerType::BASIC) {
                tower->Shoot(FindNearestEnemy(tower));
            } else if (tower->GetType() == TowerType::SNIPER) {
                tower->ShootSlow(FindFarthestEnemy(tower));
            } else if (tower->GetType() == TowerType::SPLASH) {
                tower->ShootArea(FindEnemyCluster(tower));
            }
            // Adding new tower type requires modifying source code!
        }
    }
};
```

**Failure #1: Hardcoded Systems**
- Tower types are enum-based, can't add new types without recompiling
- Behavior logic is embedded in game loop
- No separation between engine and content
- **Impact:** Players need source code access to add content

```cpp
// RED Failure #2: Opaque data formats
void SaveGame(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    // Binary format with no documentation
    file.write(reinterpret_cast<char*>(&player), sizeof(Player));
    file.write(reinterpret_cast<char*>(&towers), sizeof(towers));
    // Completely opaque to modders - can't read or modify saves
}
```

**Failure #2: Closed Data Formats**
- Binary format with no schema documentation
- Pointers serialized directly (crashes on load)
- No version information
- **Impact:** Mods can't persist data, saves break with mods installed

```cpp
// RED Failure #3: No API or extension points
class Tower {
private:
    float damage;
    float range;
    float fireRate;
    // All members private - no way to access from outside

    void UpdateInternal(float dt) {
        // Complex logic with no hooks
        cooldown -= dt;
        if (cooldown <= 0 && HasTarget()) {
            Fire();
            cooldown = fireRate;
        }
    }

public:
    // Minimal interface, no extensibility
    void Update(float dt) { UpdateInternal(dt); }
};
```

**Failure #3: No API/Extension Points**
- No public interface for modders
- No event system or callbacks
- Can't override behavior
- **Impact:** Modders must reverse-engineer and binary-patch

```cpp
// RED Failure #4: Fragile versioning
// v1.0
struct TowerData {
    float damage;
    float range;
};

// v1.1 - breaks all existing mods!
struct TowerData {
    float damage;
    float range;
    float critChance;  // New field breaks binary compatibility
    int ammoType;      // Offset changes break everything
};
```

**Failure #4: No API Stability**
- Data structures change between versions
- No deprecation policy
- Binary breaking changes in patches
- **Impact:** Every game update breaks all mods

```cpp
// RED Failure #5: No content sharing infrastructure
void LoadCustomTower(const std::string& filename) {
    // Just loads local files - no validation, no metadata
    TowerData* data = new TowerData();
    std::ifstream file(filename);
    file >> *data;  // No error checking, crashes on bad data
    towers.push_back(new Tower(data));
}
```

**Failure #5: No Sharing/Discovery**
- No metadata (author, version, dependencies)
- No validation or sanitization
- No central repository or rating system
- **Impact:** Players manually download files, high risk of malware

```cpp
// RED Failure #6: Naive mod loading
void LoadAllMods() {
    for (auto& modFile : GetModFiles()) {
        LoadMod(modFile);  // Load order random!
        // No dependency resolution
        // No conflict detection
        // Duplicate IDs overwrite silently
    }
}
```

**Failure #6: No Dependency Management**
- Random load order causes inconsistent behavior
- No way to specify "mod A requires mod B"
- ID conflicts cause silent overwrites
- **Impact:** Mod combinations cause unpredictable crashes

```cpp
// RED Failure #7: Unrestricted mod execution
void ExecuteModScript(const std::string& script) {
    // Lua script with full system access!
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);  // Opens ALL libraries including io, os
    luaL_dostring(L, script.c_str());
    // Mod can delete files, open network connections, anything!
}
```

**Failure #7: Security Vulnerabilities**
- No sandboxing - mods have full system access
- Can delete player data, install malware
- No permission system
- **Impact:** Players risk security installing any mod

```cpp
// RED Failure #8: No performance constraints
void LoadCustomEnemy(const std::string& script) {
    // No validation of computational cost
    while (true) {
        // Mod creates infinite loop
        SpawnParticle();  // Or spawns billions of particles
    }
    // Game freezes, no protection
}
```

**Failure #8: No Performance Limits**
- Mods can create infinite loops
- No memory limits
- No particle/entity caps
- **Impact:** Poorly optimized mods crash game for everyone

```cpp
// RED Failure #9: No creation tools provided
// Players must:
// 1. Reverse engineer binary formats
// 2. Use hex editors to modify data
// 3. Decompile code to understand systems
// 4. Share knowledge on forums

// Example "modding guide" from community:
/*
To create a custom tower:
1. Open game.exe in hex editor
2. Find tower data at offset 0x004A3F20
3. Modify bytes 12-16 for damage value
4. Pray it doesn't crash
*/
```

**Failure #9: No Official Tools**
- No level editor
- No documentation
- No example mods
- **Impact:** Only dedicated reverse-engineers can mod

```cpp
// RED Failure #10: Unclear legal status
// README.txt:
// "© 2024 GameCo. All rights reserved."
//
// Is modding allowed? Can I share mods?
// Can I monetize mods? What about mod assets?
// Nobody knows - no explicit policy.
```

**Failure #10: Legal Ambiguity**
- No Terms of Service regarding mods
- Unclear content ownership
- No monetization policy
- **Impact:** Fear of legal action stifles community

### RED Baseline Metrics

**Extensibility Score: 0/10**

- ❌ No documented API (0%)
- ❌ No mod tools provided (0%)
- ❌ No content sharing infrastructure (0%)
- ❌ No security (dangerous to run mods)
- ❌ No performance protection
- ❌ No dependency management
- ❌ No versioning stability
- ❌ No legal clarity
- ❌ 100% of updates break mods
- ❌ Requires reverse engineering to mod

**Community Outcome:**
- Small community of hardcore modders
- Frequent crashes and security scares
- Mods break every patch
- Most players avoid mods entirely

---

## GREEN Phase: Building for Extensibility

### Architecture: Design for Mods from Day One

The foundation of moddable games is **separation of engine from content**. Your game should be a platform that loads content, not a monolith.

```cpp
// GREEN: Plugin architecture with clean separation
class IGamePlugin {
public:
    virtual ~IGamePlugin() = default;

    // Stable API contract
    virtual const char* GetName() const = 0;
    virtual const char* GetVersion() const = 0;
    virtual const char* GetAuthor() const = 0;

    // Lifecycle hooks
    virtual bool OnLoad() = 0;
    virtual void OnUnload() = 0;
    virtual void OnUpdate(float deltaTime) = 0;

    // Game interaction through interfaces only
    virtual void RegisterContent(IContentRegistry* registry) = 0;
};

// Engine provides stable interfaces to plugins
class IContentRegistry {
public:
    virtual ~IContentRegistry() = default;

    // Register new content types
    virtual void RegisterTowerType(const TowerDefinition& def) = 0;
    virtual void RegisterEnemyType(const EnemyDefinition& def) = 0;
    virtual void RegisterMapTemplate(const MapDefinition& def) = 0;

    // Query existing content
    virtual const TowerDefinition* GetTowerType(const std::string& id) = 0;
    virtual std::vector<std::string> GetAllTowerIDs() const = 0;
};

// Game engine loads plugins dynamically
class ModdableGameEngine {
private:
    std::vector<std::unique_ptr<IGamePlugin>> plugins;
    std::unique_ptr<IContentRegistry> contentRegistry;

public:
    bool LoadPlugin(const std::string& pluginPath) {
        // Load shared library (.dll/.so)
        void* handle = dlopen(pluginPath.c_str(), RTLD_LAZY);
        if (!handle) {
            LogError("Failed to load plugin: %s", dlerror());
            return false;
        }

        // Get plugin factory function
        typedef IGamePlugin* (*CreatePluginFunc)();
        CreatePluginFunc createPlugin =
            (CreatePluginFunc)dlsym(handle, "CreatePlugin");

        if (!createPlugin) {
            LogError("Plugin missing CreatePlugin function");
            dlclose(handle);
            return false;
        }

        // Create plugin instance
        IGamePlugin* plugin = createPlugin();
        if (!plugin->OnLoad()) {
            delete plugin;
            dlclose(handle);
            return false;
        }

        // Let plugin register content
        plugin->RegisterContent(contentRegistry.get());
        plugins.push_back(std::unique_ptr<IGamePlugin>(plugin));

        LogInfo("Loaded plugin: %s v%s by %s",
                plugin->GetName(), plugin->GetVersion(), plugin->GetAuthor());
        return true;
    }

    void Update(float deltaTime) {
        // Update all plugins
        for (auto& plugin : plugins) {
            plugin->OnUpdate(deltaTime);
        }
    }
};
```

**Key Principles:**
- **Interface-based:** Plugins interact only through stable interfaces
- **Dynamic loading:** Mods are DLLs/SOs loaded at runtime
- **Lifecycle management:** Clear initialization and cleanup
- **Version isolation:** Engine and plugin versions tracked separately

### Example Plugin Implementation

```cpp
// CustomTowersPlugin.cpp - Example mod
class CustomTowersPlugin : public IGamePlugin {
private:
    std::string name = "Laser Towers Mod";
    std::string version = "1.2.0";
    std::string author = "ModderPro";

public:
    const char* GetName() const override { return name.c_str(); }
    const char* GetVersion() const override { return version.c_str(); }
    const char* GetAuthor() const override { return author.c_str(); }

    bool OnLoad() override {
        LogInfo("Loading Laser Towers Mod...");
        return true;
    }

    void OnUnload() override {
        LogInfo("Unloading Laser Towers Mod...");
    }

    void OnUpdate(float deltaTime) override {
        // Plugin-specific per-frame logic
    }

    void RegisterContent(IContentRegistry* registry) override {
        // Define new tower type
        TowerDefinition laserTower;
        laserTower.id = "laser_tower";
        laserTower.displayName = "Laser Tower";
        laserTower.description = "Continuous beam weapon";
        laserTower.cost = 500;
        laserTower.range = 300.0f;
        laserTower.damage = 25.0f;
        laserTower.fireRate = 60.0f;  // 60 ticks per second
        laserTower.modelPath = "mods/lasertowers/laser_tower.model";
        laserTower.texturePath = "mods/lasertowers/laser_tower.png";

        // Custom behavior through behavior factory
        laserTower.behaviorFactory = [](Tower* tower) {
            return std::make_unique<LaserTowerBehavior>(tower);
        };

        registry->RegisterTowerType(laserTower);

        // Can register multiple content pieces
        TowerDefinition railgunTower;
        // ... configure railgun
        registry->RegisterTowerType(railgunTower);
    }
};

// Plugin export (C linkage for cross-compiler compatibility)
extern "C" {
    IGamePlugin* CreatePlugin() {
        return new CustomTowersPlugin();
    }
}
```

### Data-Driven Content System

Many mods don't need code—just data. Support data-only mods for accessibility.

```cpp
// JSON-based content definition
class DataDrivenContentLoader {
public:
    void LoadContentPack(const std::string& manifestPath) {
        json manifest = ParseJSON(manifestPath);

        // Load all tower definitions from data
        for (const auto& towerDef : manifest["towers"]) {
            TowerDefinition def;
            def.id = towerDef["id"];
            def.displayName = towerDef["name"];
            def.description = towerDef["description"];
            def.cost = towerDef["cost"];
            def.damage = towerDef["damage"];
            def.range = towerDef["range"];
            def.fireRate = towerDef["fireRate"];
            def.modelPath = ResolveModPath(towerDef["model"]);
            def.texturePath = ResolveModPath(towerDef["texture"]);

            // Behavior from predefined templates
            std::string behaviorType = towerDef["behavior"];
            def.behaviorFactory = GetBehaviorFactory(behaviorType);

            contentRegistry->RegisterTowerType(def);
        }
    }
};
```

**Example mod manifest (no code required):**

```json
{
  "modInfo": {
    "id": "elemental_towers",
    "name": "Elemental Towers Pack",
    "version": "2.0.1",
    "author": "ElementalGamer",
    "description": "Adds fire, ice, and lightning towers",
    "dependencies": [
      {"id": "base_game", "version": ">=1.5.0"}
    ]
  },
  "towers": [
    {
      "id": "fire_tower",
      "name": "Fire Tower",
      "description": "Burns enemies over time",
      "cost": 300,
      "damage": 15,
      "range": 200,
      "fireRate": 2.0,
      "behavior": "damage_over_time",
      "model": "models/fire_tower.obj",
      "texture": "textures/fire_tower.png",
      "effects": {
        "onHit": "fire_particle",
        "sound": "fire_whoosh"
      },
      "statusEffect": {
        "type": "burn",
        "damagePerSecond": 5,
        "duration": 3.0
      }
    },
    {
      "id": "ice_tower",
      "name": "Ice Tower",
      "description": "Slows enemy movement",
      "cost": 350,
      "damage": 10,
      "range": 250,
      "fireRate": 1.5,
      "behavior": "status_applier",
      "model": "models/ice_tower.obj",
      "texture": "textures/ice_tower.png",
      "statusEffect": {
        "type": "slow",
        "speedMultiplier": 0.5,
        "duration": 4.0
      }
    }
  ],
  "enemies": [
    {
      "id": "fire_elemental",
      "name": "Fire Elemental",
      "health": 500,
      "speed": 80,
      "reward": 50,
      "resistances": {
        "fire": 0.9,
        "ice": 1.5
      }
    }
  ]
}
```

### Event System: Hooking into Game Logic

Allow mods to react to game events without modifying core code.

```cpp
// Event system for mod hooks
enum class GameEventType {
    TOWER_PLACED,
    TOWER_SOLD,
    ENEMY_SPAWNED,
    ENEMY_KILLED,
    WAVE_STARTED,
    WAVE_COMPLETED,
    GAME_OVER,
    PLAYER_DAMAGED
};

class GameEvent {
public:
    GameEventType type;
    std::map<std::string, std::any> data;
    bool cancelled = false;  // Mods can cancel events
};

class EventSystem {
private:
    using EventCallback = std::function<void(GameEvent&)>;
    std::map<GameEventType, std::vector<EventCallback>> listeners;

public:
    // Mods register listeners
    void Subscribe(GameEventType type, EventCallback callback) {
        listeners[type].push_back(callback);
    }

    // Game fires events
    void FireEvent(GameEvent& event) {
        auto it = listeners.find(event.type);
        if (it != listeners.end()) {
            for (auto& callback : it->second) {
                callback(event);
                if (event.cancelled) break;
            }
        }
    }
};

// Example: Mod that gives bonus rewards
class BonusRewardsMod : public IGamePlugin {
public:
    void RegisterContent(IContentRegistry* registry) override {
        // Subscribe to enemy death events
        GetEventSystem()->Subscribe(
            GameEventType::ENEMY_KILLED,
            [](GameEvent& event) {
                Enemy* enemy = std::any_cast<Enemy*>(event.data["enemy"]);
                Player* player = std::any_cast<Player*>(event.data["player"]);

                // 20% chance to double rewards
                if (Random::Float() < 0.2f) {
                    int baseReward = std::any_cast<int>(event.data["reward"]);
                    event.data["reward"] = baseReward * 2;
                    ShowFloatingText("DOUBLE REWARD!", enemy->GetPosition());
                }
            }
        );
    }
};

// Example: Difficulty scaling mod
class DifficultyScalingMod : public IGamePlugin {
private:
    int waveCount = 0;

public:
    void RegisterContent(IContentRegistry* registry) override {
        GetEventSystem()->Subscribe(
            GameEventType::WAVE_STARTED,
            [this](GameEvent& event) {
                waveCount++;
            }
        );

        GetEventSystem()->Subscribe(
            GameEventType::ENEMY_SPAWNED,
            [this](GameEvent& event) {
                Enemy* enemy = std::any_cast<Enemy*>(event.data["enemy"]);
                // Scale health with wave count
                float multiplier = 1.0f + (waveCount * 0.1f);
                enemy->SetMaxHealth(enemy->GetMaxHealth() * multiplier);
            }
        );
    }
};
```

### Scriptable Behaviors with Lua

Expose safe, sandboxed scripting for complex mod logic.

```cpp
// Lua scripting system with sandboxing
class LuaScriptingEngine {
private:
    lua_State* L;

    // Sandbox: restricted functions only
    static int SafePrint(lua_State* L) {
        const char* msg = lua_tostring(L, 1);
        GameLog::Info("[Lua] %s", msg);
        return 0;
    }

    static int SpawnParticle(lua_State* L) {
        const char* particleType = lua_tostring(L, 1);
        float x = lua_tonumber(L, 2);
        float y = lua_tonumber(L, 3);
        ParticleSystem::Spawn(particleType, Vector2(x, y));
        return 0;
    }

    static int GetEnemy(lua_State* L) {
        int enemyId = lua_tointeger(L, 1);
        Enemy* enemy = EntityManager::GetEnemy(enemyId);
        lua_pushlightuserdata(L, enemy);
        return 1;
    }

public:
    void Initialize() {
        L = luaL_newstate();

        // DON'T open all libs (security risk!)
        // luaL_openlibs(L);  // NO! This includes io, os, debug

        // Open only safe libraries
        lua_pushcfunction(L, luaopen_base);
        lua_call(L, 0, 0);
        lua_pushcfunction(L, luaopen_table);
        lua_call(L, 0, 0);
        lua_pushcfunction(L, luaopen_string);
        lua_call(L, 0, 0);
        lua_pushcfunction(L, luaopen_math);
        lua_call(L, 0, 0);

        // Remove dangerous functions
        lua_pushnil(L);
        lua_setglobal(L, "dofile");
        lua_pushnil(L);
        lua_setglobal(L, "loadfile");
        lua_pushnil(L);
        lua_setglobal(L, "require");  // Could load arbitrary code

        // Register safe game API
        lua_register(L, "print", SafePrint);
        lua_register(L, "SpawnParticle", SpawnParticle);
        lua_register(L, "GetEnemy", GetEnemy);

        // Add more safe functions as needed...
    }

    bool ExecuteScript(const std::string& script, float timeout) {
        // Set execution timeout to prevent infinite loops
        lua_sethook(L, ExecutionTimeoutHook, LUA_MASKCOUNT, 100000);

        int result = luaL_dostring(L, script.c_str());
        if (result != LUA_OK) {
            const char* error = lua_tostring(L, -1);
            LogError("Lua script error: %s", error);
            return false;
        }
        return true;
    }
};
```

**Example Lua mod script:**

```lua
-- laser_tower_behavior.lua
-- Custom behavior for laser tower

function OnTowerUpdate(tower, deltaTime)
    -- Get tower's target
    local target = tower:GetTarget()
    if not target then return end

    -- Continuous beam damage
    local damage = tower:GetDamage() * deltaTime
    target:TakeDamage(damage)

    -- Visual effect: beam from tower to target
    local startPos = tower:GetPosition()
    local endPos = target:GetPosition()
    SpawnBeamEffect("laser_beam", startPos, endPos)

    -- Sound (throttled to avoid spam)
    if tower:GetTimeSinceLastSound() > 0.5 then
        PlaySound("laser_hum")
        tower:ResetSoundTimer()
    end
end

function OnTowerUpgraded(tower, newLevel)
    -- Change beam color based on level
    if newLevel == 2 then
        tower:SetBeamColor(0, 255, 0)  -- Green
    elseif newLevel == 3 then
        tower:SetBeamColor(255, 0, 0)  -- Red
    end

    print("Laser tower upgraded to level " .. newLevel)
end

function OnEnemyKilled(tower, enemy)
    -- 10% chance to create chain lightning effect
    if math.random() < 0.1 then
        local nearbyEnemies = GetEnemiesInRadius(enemy:GetPosition(), 100)
        for _, nearbyEnemy in ipairs(nearbyEnemies) do
            nearbyEnemy:TakeDamage(tower:GetDamage() * 0.5)
            SpawnParticle("electric_arc", enemy:GetPosition(), nearbyEnemy:GetPosition())
        end
    end
end
```

### Stable API Design

A well-designed API is the foundation of a healthy modding ecosystem.

```cpp
// API versioning system
namespace GameAPI {
    constexpr int MAJOR_VERSION = 2;  // Breaking changes
    constexpr int MINOR_VERSION = 3;  // New features (backwards compatible)
    constexpr int PATCH_VERSION = 1;  // Bug fixes

    struct Version {
        int major, minor, patch;

        bool IsCompatible(const Version& required) const {
            // Major version must match exactly
            if (major != required.major) return false;
            // Minor version must be >= required
            if (minor < required.minor) return false;
            return true;
        }
    };
}

// Deprecation policy
class IGameAPI_V2 {
public:
    // New API (preferred)
    virtual Entity* SpawnEntityAtPosition(
        const std::string& entityType,
        const Vector3& position,
        const Quaternion& rotation
    ) = 0;

    // Deprecated API (still works, marked for removal)
    [[deprecated("Use SpawnEntityAtPosition instead. Will be removed in v3.0")]]
    virtual Entity* SpawnEntity(
        const std::string& entityType,
        float x, float y, float z
    ) {
        // Forward to new implementation
        return SpawnEntityAtPosition(entityType, Vector3(x, y, z), Quaternion::Identity());
    }
};

// Extension API: New features without breaking compatibility
class IGameAPI_V2_3 : public IGameAPI_V2 {
public:
    // Added in v2.3, older mods don't need to implement
    virtual bool SupportsFeature(const std::string& feature) const {
        return false;  // Default: feature not supported
    }

    virtual void SetEntityGlow(Entity* entity, const Color& color) {
        // Optional feature, no-op if not overridden
    }
};
```

### Mod Metadata and Manifests

Every mod needs metadata for dependency resolution and discovery.

```cpp
struct ModManifest {
    std::string id;              // Unique identifier
    std::string name;            // Display name
    std::string version;         // Semantic version (1.2.3)
    std::string author;
    std::string description;
    std::string homepage;        // Documentation/source

    // Dependencies
    struct Dependency {
        std::string modId;
        std::string versionConstraint;  // ">=1.0.0", "^2.0", etc.
        bool optional;
    };
    std::vector<Dependency> dependencies;

    // Conflicts
    std::vector<std::string> conflicts;  // Incompatible mods

    // API version required
    GameAPI::Version apiVersion;

    // Load order hints
    enum class LoadPhase {
        EARLY,      // Load before most mods
        NORMAL,     // Default
        LATE        // Load after most mods
    };
    LoadPhase loadPhase = LoadPhase::NORMAL;

    // Content provided (for dependency resolution)
    std::vector<std::string> provides;  // "laser_weapons", "particle_api", etc.
};

// Parse from JSON
ModManifest ParseManifest(const std::string& modPath) {
    json data = ParseJSON(modPath + "/manifest.json");
    ModManifest manifest;

    manifest.id = data["id"];
    manifest.name = data["name"];
    manifest.version = data["version"];
    manifest.author = data.value("author", "Unknown");
    manifest.description = data.value("description", "");
    manifest.homepage = data.value("homepage", "");

    // Parse dependencies
    if (data.contains("dependencies")) {
        for (const auto& dep : data["dependencies"]) {
            ModManifest::Dependency dependency;
            dependency.modId = dep["id"];
            dependency.versionConstraint = dep.value("version", "*");
            dependency.optional = dep.value("optional", false);
            manifest.dependencies.push_back(dependency);
        }
    }

    // Parse API version
    if (data.contains("apiVersion")) {
        auto apiVer = data["apiVersion"];
        manifest.apiVersion.major = apiVer.value("major", 1);
        manifest.apiVersion.minor = apiVer.value("minor", 0);
        manifest.apiVersion.patch = apiVer.value("patch", 0);
    }

    return manifest;
}
```

**Example mod manifest.json:**

```json
{
  "id": "advanced_towers",
  "name": "Advanced Towers Pack",
  "version": "3.1.4",
  "author": "ModMaster",
  "description": "Adds 15 new advanced towers with unique mechanics",
  "homepage": "https://github.com/modmaster/advanced-towers",
  "license": "MIT",

  "apiVersion": {
    "major": 2,
    "minor": 3,
    "patch": 0
  },

  "dependencies": [
    {
      "id": "base_game",
      "version": ">=1.5.0"
    },
    {
      "id": "particle_effects_enhanced",
      "version": "^2.0.0",
      "optional": true
    }
  ],

  "conflicts": [
    "old_towers_mod"
  ],

  "loadPhase": "NORMAL",

  "provides": [
    "tesla_coil_api",
    "energy_system"
  ]
}
```

### Dependency Resolution and Load Order

Mods must load in the correct order to handle dependencies.

```cpp
class ModLoader {
private:
    std::map<std::string, ModManifest> availableMods;
    std::vector<std::string> loadOrder;  // Computed order

    // Topological sort for dependency resolution
    bool ComputeLoadOrder() {
        loadOrder.clear();
        std::set<std::string> loaded;
        std::set<std::string> visiting;

        for (const auto& [modId, manifest] : availableMods) {
            if (!VisitMod(modId, loaded, visiting)) {
                return false;  // Circular dependency
            }
        }

        return true;
    }

    bool VisitMod(
        const std::string& modId,
        std::set<std::string>& loaded,
        std::set<std::string>& visiting
    ) {
        if (loaded.count(modId)) return true;

        if (visiting.count(modId)) {
            LogError("Circular dependency detected involving mod: %s", modId.c_str());
            return false;
        }

        visiting.insert(modId);
        const ModManifest& manifest = availableMods[modId];

        // Visit dependencies first
        for (const auto& dep : manifest.dependencies) {
            if (dep.optional) continue;

            // Check if dependency exists
            if (!availableMods.count(dep.modId)) {
                LogError("Mod '%s' requires missing dependency: %s",
                         modId.c_str(), dep.modId.c_str());
                return false;
            }

            // Check version compatibility
            const ModManifest& depManifest = availableMods[dep.modId];
            if (!CheckVersionConstraint(depManifest.version, dep.versionConstraint)) {
                LogError("Mod '%s' requires %s version %s, but %s is installed",
                         modId.c_str(), dep.modId.c_str(),
                         dep.versionConstraint.c_str(), depManifest.version.c_str());
                return false;
            }

            // Recursively load dependency
            if (!VisitMod(dep.modId, loaded, visiting)) {
                return false;
            }
        }

        visiting.erase(modId);
        loaded.insert(modId);
        loadOrder.push_back(modId);

        return true;
    }

    bool CheckVersionConstraint(const std::string& version, const std::string& constraint) {
        // Simple semantic versioning check
        if (constraint == "*") return true;

        SemanticVersion ver = ParseSemanticVersion(version);
        SemanticVersion req = ParseSemanticVersion(constraint);

        if (constraint.starts_with(">=")) {
            return ver >= req;
        } else if (constraint.starts_with("^")) {
            // Compatible changes (same major version)
            return ver.major == req.major && ver >= req;
        } else if (constraint.starts_with("~")) {
            // Patch changes only
            return ver.major == req.major && ver.minor == req.minor && ver.patch >= req.patch;
        } else {
            // Exact version
            return ver == req;
        }
    }

public:
    bool LoadAllMods(const std::string& modsDirectory) {
        // Scan for mods
        for (const auto& modPath : GetSubdirectories(modsDirectory)) {
            try {
                ModManifest manifest = ParseManifest(modPath);
                availableMods[manifest.id] = manifest;
            } catch (const std::exception& e) {
                LogWarning("Failed to load manifest from %s: %s",
                          modPath.c_str(), e.what());
            }
        }

        // Check for conflicts
        for (const auto& [modId, manifest] : availableMods) {
            for (const auto& conflictId : manifest.conflicts) {
                if (availableMods.count(conflictId)) {
                    LogError("Mod conflict: '%s' and '%s' cannot be loaded together",
                            modId.c_str(), conflictId.c_str());
                    return false;
                }
            }
        }

        // Compute load order
        if (!ComputeLoadOrder()) {
            LogError("Failed to resolve mod dependencies");
            return false;
        }

        // Load mods in order
        for (const std::string& modId : loadOrder) {
            const ModManifest& manifest = availableMods[modId];
            LogInfo("Loading mod: %s v%s", manifest.name.c_str(), manifest.version.c_str());

            if (!LoadPlugin(manifest)) {
                LogError("Failed to load mod: %s", modId.c_str());
                return false;
            }
        }

        LogInfo("Successfully loaded %d mods", loadOrder.size());
        return true;
    }
};
```

### Content Creation Tools

Provide tools so modders don't have to reverse-engineer your formats.

```python
# Level Editor: Drag-and-drop map creation
class LevelEditor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tower Defense Level Editor")

        # Canvas for map
        self.canvas = tk.Canvas(self.window, width=800, height=600, bg="white")
        self.canvas.pack()

        # Tool palette
        self.tools = tk.Frame(self.window)
        self.tools.pack()

        tk.Button(self.tools, text="Path", command=lambda: self.set_tool("path")).pack(side=tk.LEFT)
        tk.Button(self.tools, text="Spawn", command=lambda: self.set_tool("spawn")).pack(side=tk.LEFT)
        tk.Button(self.tools, text="Exit", command=lambda: self.set_tool("exit")).pack(side=tk.LEFT)
        tk.Button(self.tools, text="Save", command=self.save_map).pack(side=tk.LEFT)
        tk.Button(self.tools, text="Load", command=self.load_map).pack(side=tk.LEFT)

        # Map data
        self.map_data = {
            "width": 800,
            "height": 600,
            "path": [],
            "spawnPoint": None,
            "exitPoint": None,
            "buildableAreas": []
        }

        self.current_tool = "path"
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def set_tool(self, tool):
        self.current_tool = tool

    def on_click(self, event):
        x, y = event.x, event.y

        if self.current_tool == "path":
            self.map_data["path"].append({"x": x, "y": y})
            self.draw_path_point(x, y)
        elif self.current_tool == "spawn":
            self.map_data["spawnPoint"] = {"x": x, "y": y}
            self.draw_spawn_point(x, y)
        elif self.current_tool == "exit":
            self.map_data["exitPoint"] = {"x": x, "y": y}
            self.draw_exit_point(x, y)

    def save_map(self):
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.map_data, f, indent=2)
            messagebox.showinfo("Success", "Map saved!")

    def load_map(self):
        filename = tk.filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            with open(filename, 'r') as f:
                self.map_data = json.load(f)
            self.redraw_map()
            messagebox.showinfo("Success", "Map loaded!")
```

```cpp
// Asset pipeline: Convert common formats to game format
class AssetPipelineTool {
public:
    // Convert OBJ to game's optimized format
    bool ConvertModel(const std::string& objPath, const std::string& outputPath) {
        OBJLoader loader;
        Model model = loader.Load(objPath);

        // Optimize: merge vertices, compute normals, generate LODs
        model.MergeVertices();
        model.ComputeNormals();
        model.GenerateLODs({0.75f, 0.5f, 0.25f});

        // Export to game format
        GameModelExporter exporter;
        return exporter.Export(model, outputPath);
    }

    // Convert texture with mipmap generation
    bool ConvertTexture(const std::string& imagePath, const std::string& outputPath) {
        Image image = Image::Load(imagePath);

        // Generate mipmaps
        std::vector<Image> mipmaps;
        mipmaps.push_back(image);
        for (int i = 0; i < 4; i++) {
            Image mip = mipmaps.back().Resize(
                mipmaps.back().width / 2,
                mipmaps.back().height / 2
            );
            mipmaps.push_back(mip);
        }

        // Compress (DXT5 for RGBA, DXT1 for RGB)
        TextureFormat format = image.HasAlpha() ? TextureFormat::DXT5 : TextureFormat::DXT1;

        GameTextureExporter exporter;
        return exporter.Export(mipmaps, format, outputPath);
    }

    // Batch convert entire mod directory
    bool ProcessModAssets(const std::string& modPath) {
        bool success = true;

        // Find all OBJ files
        for (const auto& objFile : FindFiles(modPath, "*.obj")) {
            std::string outputFile = ReplaceExtension(objFile, ".gmodel");
            if (!ConvertModel(objFile, outputFile)) {
                LogError("Failed to convert model: %s", objFile.c_str());
                success = false;
            }
        }

        // Find all image files
        for (const auto& imageFile : FindFiles(modPath, "*.png", "*.jpg", "*.tga")) {
            std::string outputFile = ReplaceExtension(imageFile, ".gtex");
            if (!ConvertTexture(imageFile, outputFile)) {
                LogError("Failed to convert texture: %s", imageFile.c_str());
                success = false;
            }
        }

        return success;
    }
};
```

### Security: Sandboxing and Permissions

Protect players from malicious mods without breaking functionality.

```cpp
// Permission system for mods
enum class ModPermission {
    READ_GAME_STATE,      // Read entities, player data, etc.
    MODIFY_ENTITIES,      // Change existing entities
    SPAWN_ENTITIES,       // Create new entities
    ACCESS_FILESYSTEM,    // Read/write files (limited to mod directory)
    NETWORK_ACCESS,       // HTTP requests for online features
    EXECUTE_COMMANDS,     // Run console commands
    MODIFY_UI,            // Add UI elements
    PLAY_AUDIO,           // Play sounds/music
};

class ModSandbox {
private:
    std::set<ModPermission> grantedPermissions;
    std::string modId;
    std::filesystem::path modDataPath;

public:
    ModSandbox(const std::string& modId, const ModManifest& manifest)
        : modId(modId) {
        // Parse requested permissions from manifest
        // Player can review and approve

        // Restrict filesystem access to mod's own directory
        modDataPath = GetModDataDirectory(modId);
        std::filesystem::create_directories(modDataPath);
    }

    // Guarded API calls
    Entity* SpawnEntity(const std::string& type, const Vector3& pos) {
        if (!HasPermission(ModPermission::SPAWN_ENTITIES)) {
            LogError("Mod '%s' attempted to spawn entity without permission", modId.c_str());
            return nullptr;
        }

        // Enforce resource limits
        if (GetModEntityCount(modId) >= MAX_ENTITIES_PER_MOD) {
            LogError("Mod '%s' hit entity limit", modId.c_str());
            return nullptr;
        }

        return EntityManager::SpawnEntity(type, pos);
    }

    std::string ReadFile(const std::string& relativePath) {
        if (!HasPermission(ModPermission::ACCESS_FILESYSTEM)) {
            LogError("Mod '%s' attempted file access without permission", modId.c_str());
            return "";
        }

        // Restrict to mod's data directory
        std::filesystem::path fullPath = modDataPath / relativePath;
        fullPath = std::filesystem::canonical(fullPath);  // Resolve ../ etc.

        // Prevent path traversal attacks
        if (!fullPath.string().starts_with(modDataPath.string())) {
            LogError("Mod '%s' attempted to access file outside its directory: %s",
                     modId.c_str(), relativePath.c_str());
            return "";
        }

        return ReadFileContents(fullPath.string());
    }

    bool HasPermission(ModPermission permission) const {
        return grantedPermissions.count(permission) > 0;
    }
};

// Resource limits per mod
struct ModResourceLimits {
    int maxEntities = 1000;
    int maxParticles = 5000;
    size_t maxMemoryBytes = 256 * 1024 * 1024;  // 256 MB
    float maxCPUTimePerFrame = 5.0f;  // milliseconds
    int maxNetworkRequestsPerSecond = 10;
};

class ModResourceMonitor {
private:
    std::map<std::string, ModResourceLimits> limits;
    std::map<std::string, ModResourceUsage> usage;

public:
    bool CheckCPUTime(const std::string& modId, float deltaTime) {
        usage[modId].cpuTimeThisFrame += deltaTime;

        if (usage[modId].cpuTimeThisFrame > limits[modId].maxCPUTimePerFrame) {
            LogWarning("Mod '%s' exceeded CPU time limit (%.2fms / %.2fms)",
                      modId.c_str(), usage[modId].cpuTimeThisFrame,
                      limits[modId].maxCPUTimePerFrame);
            return false;
        }
        return true;
    }

    void ResetFrameCounters() {
        for (auto& [modId, modUsage] : usage) {
            modUsage.cpuTimeThisFrame = 0;
        }
    }
};
```

**Permission UI for players:**

```
====================================
Mod: Advanced Weapons Pack v2.1
Author: WeaponMaster
====================================

This mod requests the following permissions:

✓ Read game state (view entities and player data)
✓ Spawn entities (create weapons and projectiles)
✓ Modify UI (add weapon selector)
✓ Play audio (weapon sound effects)
⚠ Network access (download daily weapon challenges)

[Grant All] [Customize] [Deny]

Note: You can change permissions later in Settings > Mods
```

### Performance: Optimization and Profiling

Give modders tools to optimize and validate performance.

```cpp
// Profiling API for mod developers
class ModProfiler {
private:
    struct ProfileScope {
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        float duration;
    };

    std::vector<ProfileScope> scopes;

public:
    class ScopedTimer {
    private:
        ModProfiler* profiler;
        std::string name;
        std::chrono::high_resolution_clock::time_point start;

    public:
        ScopedTimer(ModProfiler* prof, const std::string& name)
            : profiler(prof), name(name) {
            start = std::chrono::high_resolution_clock::now();
        }

        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            profiler->RecordScope(name, duration);
        }
    };

    ScopedTimer Profile(const std::string& name) {
        return ScopedTimer(this, name);
    }

    void RecordScope(const std::string& name, float duration) {
        scopes.push_back({name, {}, duration});
    }

    void GenerateReport() {
        std::sort(scopes.begin(), scopes.end(),
                 [](const ProfileScope& a, const ProfileScope& b) {
                     return a.duration > b.duration;
                 });

        LogInfo("=== Mod Performance Report ===");
        for (const auto& scope : scopes) {
            LogInfo("  %s: %.2fms", scope.name.c_str(), scope.duration);
        }
    }
};

// Usage in mod:
void CustomTower::Update(float dt) {
    auto timer = GetProfiler()->Profile("CustomTower::Update");

    // Your update logic...
    UpdateTargeting(dt);
    UpdateWeapon(dt);
    UpdateEffects(dt);

}  // Automatically records timing

// Validation tool for mod developers
class ModValidator {
public:
    struct ValidationResult {
        bool passed;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    ValidationResult ValidateMod(const std::string& modPath) {
        ValidationResult result;
        result.passed = true;

        // Check manifest
        if (!std::filesystem::exists(modPath + "/manifest.json")) {
            result.errors.push_back("Missing manifest.json");
            result.passed = false;
        }

        // Check file sizes
        for (const auto& file : GetAllFiles(modPath)) {
            size_t size = std::filesystem::file_size(file);
            if (size > 100 * 1024 * 1024) {  // 100 MB
                result.warnings.push_back(
                    "Large file: " + file.filename().string() +
                    " (" + FormatBytes(size) + ")"
                );
            }
        }

        // Check texture resolutions
        for (const auto& texture : FindTextures(modPath)) {
            Image img = Image::Load(texture);
            if (img.width > 4096 || img.height > 4096) {
                result.warnings.push_back(
                    "High-res texture: " + texture.filename().string() +
                    " (" + std::to_string(img.width) + "x" + std::to_string(img.height) + ")"
                );
            }
            if (!IsPowerOfTwo(img.width) || !IsPowerOfTwo(img.height)) {
                result.errors.push_back(
                    "Texture dimensions must be power of 2: " + texture.filename().string()
                );
                result.passed = false;
            }
        }

        // Check model poly counts
        for (const auto& model : FindModels(modPath)) {
            ModelInfo info = AnalyzeModel(model);
            if (info.triangleCount > 50000) {
                result.warnings.push_back(
                    "High poly model: " + model.filename().string() +
                    " (" + std::to_string(info.triangleCount) + " tris)"
                );
            }
        }

        // Validate scripts
        for (const auto& script : FindScripts(modPath)) {
            LuaSyntaxChecker checker;
            if (!checker.ValidateSyntax(script)) {
                result.errors.push_back(
                    "Lua syntax error in " + script.filename().string() +
                    ": " + checker.GetError()
                );
                result.passed = false;
            }
        }

        return result;
    }
};
```

### Community Infrastructure: Sharing and Discovery

Build systems for players to share and discover mods.

```cpp
// Steam Workshop integration example
class WorkshopIntegration {
private:
    ISteamUGC* steamUGC;

public:
    struct WorkshopItem {
        PublishedFileId_t fileId;
        std::string title;
        std::string description;
        std::string previewImageUrl;
        uint64 fileSize;
        uint32 timeCreated;
        uint32 timeUpdated;
        uint32 subscriptions;
        uint32 favorites;
        float score;  // Upvote ratio
    };

    // Upload mod to Workshop
    bool PublishMod(const std::string& modPath, const WorkshopItem& metadata) {
        // Create Workshop item
        SteamAPICall_t handle = steamUGC->CreateItem(
            GetAppID(),
            k_EWorkshopFileTypeCommunity
        );

        // Set metadata
        UGCUpdateHandle_t updateHandle = steamUGC->StartItemUpdate(
            GetAppID(),
            metadata.fileId
        );

        steamUGC->SetItemTitle(updateHandle, metadata.title.c_str());
        steamUGC->SetItemDescription(updateHandle, metadata.description.c_str());
        steamUGC->SetItemPreview(updateHandle, metadata.previewImageUrl.c_str());
        steamUGC->SetItemContent(updateHandle, modPath.c_str());
        steamUGC->SetItemVisibility(updateHandle, k_ERemoteStoragePublishedFileVisibilityPublic);

        // Submit update
        SteamAPICall_t submitHandle = steamUGC->SubmitItemUpdate(
            updateHandle,
            "Initial release"
        );

        return true;
    }

    // Subscribe to mod (automatic download/update)
    bool SubscribeToMod(PublishedFileId_t fileId) {
        steamUGC->SubscribeItem(fileId);
        return true;
    }

    // Query popular mods
    std::vector<WorkshopItem> GetPopularMods(int count) {
        UGCQueryHandle_t queryHandle = steamUGC->CreateQueryAllUGCRequest(
            k_EUGCQuery_RankedByTrend,
            k_EUGCMatchingUGCType_Items,
            GetAppID(),
            GetAppID(),
            1  // Page
        );

        steamUGC->SetReturnLongDescription(queryHandle, true);
        steamUGC->SetReturnMetadata(queryHandle, true);
        steamUGC->SetReturnChildren(queryHandle, true);

        SteamAPICall_t handle = steamUGC->SendQueryUGCRequest(queryHandle);

        // Wait for results...
        std::vector<WorkshopItem> items;
        // Parse results into items vector
        return items;
    }

    // In-game mod browser UI
    void ShowModBrowser() {
        ImGui::Begin("Workshop Mods");

        // Search bar
        static char searchQuery[256] = "";
        ImGui::InputText("Search", searchQuery, sizeof(searchQuery));

        // Filter buttons
        if (ImGui::Button("Most Popular")) {
            currentMods = GetPopularMods(50);
        }
        ImGui::SameLine();
        if (ImGui::Button("Highest Rated")) {
            currentMods = GetHighestRatedMods(50);
        }
        ImGui::SameLine();
        if (ImGui::Button("Most Recent")) {
            currentMods = GetRecentMods(50);
        }

        // Mod list
        ImGui::BeginChild("ModList");
        for (const auto& mod : currentMods) {
            ImGui::PushID(mod.fileId);

            // Preview image
            ImGui::Image(GetModPreview(mod.fileId), ImVec2(128, 128));
            ImGui::SameLine();

            // Metadata
            ImGui::BeginGroup();
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", mod.title.c_str());
            ImGui::Text("Subscriptions: %u", mod.subscriptions);
            ImGui::Text("Rating: %.0f%%", mod.score * 100);

            if (IsSubscribed(mod.fileId)) {
                if (ImGui::Button("Unsubscribe")) {
                    UnsubscribeFromMod(mod.fileId);
                }
            } else {
                if (ImGui::Button("Subscribe")) {
                    SubscribeToMod(mod.fileId);
                }
            }
            ImGui::EndGroup();

            ImGui::Separator();
            ImGui::PopID();
        }
        ImGui::EndChild();

        ImGui::End();
    }
};
```

**Alternative: Custom mod repository**

```cpp
// REST API for custom mod hosting
class ModRepository {
public:
    struct ModListing {
        std::string id;
        std::string name;
        std::string author;
        std::string version;
        std::string description;
        std::vector<std::string> tags;
        std::string downloadUrl;
        int downloads;
        float rating;
        std::string thumbnailUrl;
    };

    // Fetch mod list from server
    std::vector<ModListing> FetchModList(const std::string& category = "") {
        std::string url = "https://api.mygame.com/mods";
        if (!category.empty()) {
            url += "?category=" + URLEncode(category);
        }

        std::string response = HTTPGet(url);
        json data = json::parse(response);

        std::vector<ModListing> mods;
        for (const auto& item : data["mods"]) {
            ModListing mod;
            mod.id = item["id"];
            mod.name = item["name"];
            mod.author = item["author"];
            mod.version = item["version"];
            mod.description = item["description"];
            mod.tags = item["tags"].get<std::vector<std::string>>();
            mod.downloadUrl = item["downloadUrl"];
            mod.downloads = item["downloads"];
            mod.rating = item["rating"];
            mod.thumbnailUrl = item["thumbnailUrl"];
            mods.push_back(mod);
        }

        return mods;
    }

    // Download and install mod
    bool DownloadAndInstallMod(const ModListing& mod) {
        std::string tempFile = DownloadFile(mod.downloadUrl);
        if (tempFile.empty()) {
            LogError("Failed to download mod: %s", mod.name.c_str());
            return false;
        }

        // Verify checksum
        std::string checksum = ComputeSHA256(tempFile);
        if (checksum != mod.checksum) {
            LogError("Mod checksum mismatch! Possible corruption or tampering.");
            std::filesystem::remove(tempFile);
            return false;
        }

        // Extract to mods directory
        std::string modPath = GetModsDirectory() + "/" + mod.id;
        if (!ExtractZip(tempFile, modPath)) {
            LogError("Failed to extract mod");
            return false;
        }

        // Validate mod structure
        ModValidator validator;
        auto result = validator.ValidateMod(modPath);
        if (!result.passed) {
            LogError("Mod validation failed");
            std::filesystem::remove_all(modPath);
            return false;
        }

        LogInfo("Successfully installed mod: %s v%s", mod.name.c_str(), mod.version.c_str());
        return true;
    }
};
```

### Legal and Licensing

Clear policies prevent legal issues and encourage mod creation.

```markdown
# Modding Terms of Service

## What You Can Do

✅ **Create and share mods** for personal and community use
✅ **Modify game content** including textures, models, sounds, and scripts
✅ **Distribute mods** for free through Workshop, Nexus, or other platforms
✅ **Create derivative content** inspired by the game
✅ **Monetize mods** through donations (Patreon, Ko-fi, etc.)
✅ **Use game assets** in screenshots, videos, and promotional material for your mods

## What You Can't Do

❌ **Sell mods** directly (no paid downloads/paywalls)
❌ **Include copyrighted content** from other games/media without permission
❌ **Redistribute base game files** (only distribute your additions/modifications)
❌ **Create cheats** for multiplayer/competitive modes
❌ **Remove DRM** or crack the game
❌ **Violate the law** (harassment, illegal content, malware, etc.)

## Ownership

- **You own your mods**: Original content you create is yours
- **We own the game**: Base game code/assets remain our IP
- **License**: Mods are considered derivative works, licensed to you under this ToS
- **Revenue**: We don't take a cut of donations/Patreon for your mods

## Liability

- **No warranty**: Mods are provided "as-is" without support
- **Responsibility**: Mod authors responsible for their content
- **Game updates**: We're not liable for breaking mods with updates
- **Data loss**: Back up your saves when using mods

## Policy Changes

We may update this policy with 90 days notice. Major changes will be announced.

Last updated: 2024-01-15
```

**Example mod license header:**

```cpp
/*
 * Advanced Towers Pack
 * Copyright (c) 2024 ModMaster
 *
 * Licensed under MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Uses assets from "MyTowerDefenseGame" which are (c) GameStudio Inc.
 * Used under GameStudio's Modding Terms of Service.
 */
```

---

## REFACTOR Phase: Pressure Testing

### Scenario 1: Skyrim-Style Deep Modding

**Goal:** Support extensive game modification like Skyrim's Creation Kit.

**Requirements:**
- Modify quests, dialogue, world spaces
- Create new NPCs with AI behaviors
- Change game balance and progression
- 100+ mods installed simultaneously
- Total conversion mods possible

**Implementation:**

```cpp
// Skyrim-style data-driven everything
class CreationKitEngine {
private:
    // All game content as data records
    std::map<std::string, Record*> records;

public:
    // Record types: NPCs, Items, Quests, Spells, etc.
    struct Record {
        std::string editorID;  // Unique identifier
        std::string formID;    // Persistent ID across game versions
        RecordType type;
        std::map<std::string, std::any> fields;
    };

    // Load plugin file (.esp/.esm)
    bool LoadPlugin(const std::string& pluginPath) {
        BinaryReader reader(pluginPath);

        while (!reader.AtEnd()) {
            Record* record = ReadRecord(reader);

            // Check if record already exists (from master or another plugin)
            if (records.count(record->editorID)) {
                // Override: merge changes
                MergeRecordChanges(records[record->editorID], record);
            } else {
                // New record: add to database
                records[record->editorID] = record;
            }
        }

        return true;
    }

    void MergeRecordChanges(Record* existing, Record* override) {
        // Merge changed fields only
        for (const auto& [fieldName, fieldValue] : override->fields) {
            existing->fields[fieldName] = fieldValue;
        }
    }

    // Script system for complex behaviors
    struct ScriptAttachment {
        std::string scriptName;
        std::map<std::string, std::any> properties;
    };

    void AttachScriptToRecord(const std::string& recordID, const ScriptAttachment& script) {
        Record* record = records[recordID];
        if (!record) return;

        std::vector<ScriptAttachment>& scripts =
            std::any_cast<std::vector<ScriptAttachment>&>(
                record->fields["scripts"]
            );
        scripts.push_back(script);
    }
};

// Example: Quest mod
/*
Quest: "The Lost Artifact"
- Add new NPC "Scholar Aldric"
- Add dialogue options
- Add quest objectives
- Add new dungeon location
- Add unique reward item
*/

// scholar_aldric.esp (plugin file defines new record)
{
  "recordType": "NPC",
  "editorID": "ScholarAldric",
  "name": "Scholar Aldric",
  "race": "Imperial",
  "level": 10,
  "inventory": [
    {"item": "Gold", "count": 50},
    {"item": "Book_Ancient", "count": 1}
  ],
  "dialogue": [
    {
      "topic": "GREETING",
      "text": "I seek a brave adventurer. Interested in ancient artifacts?",
      "conditions": [{"quest": "LostArtifact", "stage": 0}]
    }
  ],
  "scripts": [
    {
      "name": "QuestGiverScript",
      "properties": {
        "questID": "LostArtifact"
      }
    }
  ]
}
```

**Result:** ✅ Passes - Deep modification supported through data record system

### Scenario 2: Minecraft Java Mods (Deep System Hooks)

**Goal:** Allow mods to completely transform game mechanics like Minecraft Forge.

**Requirements:**
- Hook into core game loop
- Replace vanilla systems entirely
- Add new dimensions, biomes, mobs
- Mods interact with each other's content

**Implementation:**

```java
// Forge-style event bus system
public class ModEventBus {
    private Map<Class<?>, List<EventHandler>> handlers = new HashMap<>();

    public <T> void subscribe(Class<T> eventType, Consumer<T> handler) {
        handlers.computeIfAbsent(eventType, k -> new ArrayList<>())
                .add(new EventHandler(handler));
    }

    public <T> T post(T event) {
        List<EventHandler> eventHandlers = handlers.get(event.getClass());
        if (eventHandlers != null) {
            for (EventHandler handler : eventHandlers) {
                handler.handle(event);
                if (event instanceof Cancelable && ((Cancelable) event).isCanceled()) {
                    break;
                }
            }
        }
        return event;
    }
}

// Game events that mods can hook
public class EntitySpawnEvent extends Event implements Cancelable {
    private Entity entity;
    private World world;
    private BlockPos pos;
    private boolean canceled = false;

    // Mods can cancel spawn or modify entity
    public void setCanceled(boolean cancel) { this.canceled = cancel; }
    public boolean isCanceled() { return canceled; }
    public Entity getEntity() { return entity; }
    public void setEntity(Entity entity) { this.entity = entity; }
}

// Example mod: Custom dimension
@Mod("dimensional_mod")
public class DimensionalMod {

    @SubscribeEvent
    public void onEntitySpawn(EntitySpawnEvent event) {
        // Custom spawn logic in our dimension
        if (event.getWorld().getDimension().getType() == ModDimensions.VOID_DIMENSION) {
            // Replace zombie with void zombie
            if (event.getEntity() instanceof ZombieEntity) {
                VoidZombieEntity voidZombie = new VoidZombieEntity(event.getWorld());
                voidZombie.setPosition(event.getPos());
                event.setEntity(voidZombie);
            }
        }
    }

    @SubscribeEvent
    public void onWorldTick(WorldTickEvent event) {
        // Custom world logic
        if (event.world.getDimension().getType() == ModDimensions.VOID_DIMENSION) {
            // Void dimension effects
            for (PlayerEntity player : event.world.getPlayers()) {
                if (player.getY() < 0) {
                    // Teleport to spawn
                    player.setPosition(0, 64, 0);
                }
            }
        }
    }
}

// Registry system for cross-mod compatibility
public class ModRegistry {
    private static Map<ResourceLocation, Block> blocks = new HashMap<>();
    private static Map<ResourceLocation, Item> items = new HashMap<>();

    public static void registerBlock(ResourceLocation id, Block block) {
        blocks.put(id, block);
    }

    public static Block getBlock(ResourceLocation id) {
        return blocks.get(id);
    }

    // Other mods can reference by ID
    // ResourceLocation("dimensional_mod:void_stone")
}
```

**Result:** ✅ Passes - Deep hooks and cross-mod systems work

### Scenario 3: Warcraft 3 Custom Games (Spawned New Genre)

**Goal:** Support such extensive modding that entirely new game genres emerge (like DOTA).

**Requirements:**
- Full map editor with triggers/scripting
- Create custom game modes
- Complete AI control
- Custom UI layouts
- Multiplayer support for custom games

**Implementation:**

```lua
-- Warcraft 3 JASS-style trigger system
-- dota.j - Simplified DOTA map script

function InitDOTA()
    -- Custom game mode setup
    SetGameMode("AllPick")
    DisableDefaultVictory()

    -- Create custom bases
    CreateBase(TEAM_RADIANT, Location(0, 0))
    CreateBase(TEAM_DIRE, Location(5000, 5000))

    -- Spawn creeps every 30 seconds
    CreateTimer(30.0, true, function()
        SpawnCreepWave(TEAM_RADIANT)
        SpawnCreepWave(TEAM_DIRE)
    end)

    -- Custom victory condition
    RegisterEventHandler(EVENT_UNIT_DIES, function(dyingUnit)
        if IsAncient(dyingUnit) then
            local winningTeam = GetOpposingTeam(GetOwningTeam(dyingUnit))
            EndGame(winningTeam)
        end
    end)
end

function CreateBase(team, location)
    -- Ancient (main building)
    local ancient = CreateUnit("Ancient", team, location)
    ancient.isInvulnerable = false
    ancient.maxHealth = 5000

    -- Towers
    for i = 1, 11 do
        local towerPos = GetTowerPosition(team, i)
        local tower = CreateUnit("Tower", team, towerPos)
        tower.damage = 100
        tower.range = 700
    end

    -- Barracks
    local barracks = CreateUnit("Barracks", team, location + Vector(100, 0))

    return { ancient = ancient, towers = towers, barracks = barracks }
end

function SpawnCreepWave(team)
    local spawnPos = GetCreepSpawnPosition(team)

    -- Spawn lane creeps
    for i = 1, 3 do  -- 3 lanes
        for j = 1, 4 do  -- 4 creeps per lane
            local creep = CreateUnit("LaneCreep", team, spawnPos[i])
            -- AI: Follow lane path
            IssueOrder(creep, "AttackMoveTo", GetLaneEndpoint(team, i))
        end
    end
end

-- Hero selection system
function OnPlayerPicksHero(player, heroType)
    local hero = CreateUnit(heroType, player.team, GetFountainPosition(player.team))
    hero.owner = player
    hero.level = 1
    hero.gold = 625

    -- Custom hero abilities
    AddAbility(hero, GetHeroAbility(heroType, 1))
    AddAbility(hero, GetHeroAbility(heroType, 2))
    AddAbility(hero, GetHeroAbility(heroType, 3))
    AddAbility(hero, GetHeroAbility(heroType, 4))  -- Ultimate

    -- Custom items
    EnableShop(hero, "ItemShop")
    EnableShop(hero, "SecretShop")
end

-- Custom ability example
function RegisterAbility_Blink(hero)
    local ability = {
        name = "Blink",
        cooldown = 15.0,
        range = 1000,
        manaCost = 75
    }

    ability.OnCast = function(caster, targetPoint)
        -- Teleport to target location
        local newPos = ClampToRange(targetPoint, caster.position, ability.range)
        caster.position = newPos

        -- Visual effect
        CreateEffect("blink_start", caster.position)
        CreateEffect("blink_end", newPos)

        -- Sound
        PlaySound("blink.wav", newPos)
    end

    return ability
end
```

**Map editor supports:**
- Trigger editor (visual scripting)
- Terrain editor
- Object editor (modify units/items/abilities)
- AI editor
- Import custom models/textures

**Result:** ✅ Passes - Full custom game creation possible

### Scenario 4: Factorio Stable Mod API

**Goal:** Provide API so stable that mods rarely break across game updates.

**Requirements:**
- Semantic versioning with deprecation cycle
- Compatibility layers for old API versions
- Extensive documentation with examples
- Breaking changes announced months in advance

**Implementation:**

```lua
-- Factorio Lua API (highly stable across versions)
-- data.lua - Mod defines new content

data:extend({
  {
    type = "assembling-machine",
    name = "advanced-assembler",
    icon = "__advanced-manufacturing__/graphics/assembler.png",
    flags = {"placeable-player", "player-creation"},
    minable = {mining_time = 1, result = "advanced-assembler"},
    max_health = 300,
    crafting_categories = {"crafting", "advanced-crafting"},
    crafting_speed = 2.0,
    energy_usage = "250kW",
    energy_source = {
      type = "electric",
      usage_priority = "secondary-input"
    },
    collision_box = {{-1.2, -1.2}, {1.2, 1.2}},
    selection_box = {{-1.5, -1.5}, {1.5, 1.5}},
    animation = {
      filename = "__advanced-manufacturing__/graphics/assembler-anim.png",
      width = 108,
      height = 110,
      frame_count = 32,
      line_length = 8,
      animation_speed = 0.5
    }
  }
})

-- control.lua - Runtime scripting
script.on_event(defines.events.on_player_created, function(event)
  local player = game.players[event.player_index]
  player.print("Advanced Manufacturing mod loaded!")

  -- Give starter items
  player.insert{name="advanced-assembler", count=1}
end)

-- API versioning
if script.active_mods["base"] >= "1.0" then
  -- Use new API
  local surface = game.surfaces[1]
  surface.create_entity{
    name = "advanced-assembler",
    position = {0, 0},
    force = "player"
  }
else
  -- Use old API (compatibility)
  game.create_entity{
    name = "advanced-assembler",
    position = {x=0, y=0},
    force = game.forces.player
  }
end
```

**API Stability Contract:**

```
Factorio Modding API Stability Promise

1. MAJOR VERSION (1.x.x → 2.x.x)
   - May contain breaking changes
   - Announced 6+ months in advance
   - Migration guide provided
   - Compatibility layer for 1 major version

2. MINOR VERSION (x.1.x → x.2.x)
   - No breaking changes
   - New features added
   - Old features may be deprecated (with warnings)
   - Deprecated features work for 2+ minor versions

3. PATCH VERSION (x.x.1 → x.x.2)
   - Bug fixes only
   - Zero breaking changes

Example:
- v1.5: Add new API function "get_recipe_data()"
- v1.6: Deprecate old function "get_recipe()", still works with warning
- v1.7: Both functions work
- v1.8: Both functions work
- v2.0: Remove old "get_recipe()", migration guide provided
```

**Result:** ✅ Passes - Stable API with clear versioning

### Scenario 5: Roblox (Game Within a Game)

**Goal:** Let players create entirely new games within your game.

**Requirements:**
- Visual scripting for non-programmers
- Full game creation tools
- Multiplayer networking handled automatically
- Monetization for creators
- Built-in distribution platform

**Implementation:**

```lua
-- Roblox-style game creation
-- Script attached to game object

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")

-- Game setup
local function SetupObby()
    -- Create spawn point
    local spawn = Instance.new("SpawnLocation")
    spawn.Position = Vector3.new(0, 5, 0)
    spawn.Parent = workspace

    -- Create checkpoints
    for i = 1, 10 do
        local checkpoint = Instance.new("Part")
        checkpoint.Position = Vector3.new(i * 10, 5, 0)
        checkpoint.BrickColor = BrickColor.new("Bright green")
        checkpoint.Touched:Connect(function(hit)
            local player = Players:GetPlayerFromCharacter(hit.Parent)
            if player then
                player.RespawnLocation = checkpoint
                print(player.Name .. " reached checkpoint " .. i)
            end
        end)
        checkpoint.Parent = workspace
    end

    -- Create finish line
    local finish = Instance.new("Part")
    finish.Position = Vector3.new(110, 5, 0)
    finish.BrickColor = BrickColor.new("Gold")
    finish.Touched:Connect(function(hit)
        local player = Players:GetPlayerFromCharacter(hit.Parent)
        if player then
            print(player.Name .. " finished!")
            -- Award badges, currency, etc.
            AwardPlayer(player, 100)
        end
    end)
    finish.Parent = workspace
end

-- Monetization: Game passes
local MarketplaceService = game:GetService("MarketplaceService")
local GAME_PASS_ID = 123456

local function CheckGamePass(player)
    local hasPass = false
    local success, message = pcall(function()
        hasPass = MarketplaceService:UserOwnsGamePassAsync(player.UserId, GAME_PASS_ID)
    end)
    return hasPass
end

Players.PlayerAdded:Connect(function(player)
    if CheckGamePass(player) then
        -- Give VIP benefits
        player.WalkSpeed = 20  -- Faster movement
        player.JumpPower = 60  -- Higher jumps
    end
end)

-- Visual scripting output (block-based)
--[[
When [Player] touches [Part]
  > Set [Player]'s [RespawnLocation] to [Part]
  > Print "[Player] reached checkpoint"
  > Play [Sound] "checkpoint_sound"
  > Add [10] to [Player]'s [Points]
]]

SetupObby()
```

**Creation Tools UI:**

```
================================
Roblox Studio (Game Editor)
================================

[Toolbox: Assets]
- Terrain
- Parts (Cube, Sphere, Cylinder, etc.)
- Models (Trees, Buildings, Vehicles)
- Scripts (Pre-made behaviors)

[Properties Panel]
Part: "Checkpoint"
- Position: (50, 5, 0)
- Size: (10, 1, 10)
- Color: Bright Green
- Transparency: 0.5
- CanCollide: false

[Script Editor]
function onTouch(hit)
    -- Your code here
end
script.Parent.Touched:Connect(onTouch)

[Test Mode]
▶ Play Solo
▶ Play Multiplayer (4 players)
▶ Publish to Platform

================================
```

**Result:** ✅ Passes - Full game creation within game

### Scenario 6: Portal 2 Puzzle Maker (Constrained Creativity)

**Goal:** Let players create content with simplified, accessible tools.

**Requirements:**
- Intuitive drag-and-drop editor
- Constrained toolset (only "legal" combinations)
- Automatic validation (puzzles must be solvable)
- One-click sharing

**Implementation:**

```cpp
// Puzzle Maker: Constrained editor with validation
class PuzzleMaker {
private:
    enum class TileType {
        EMPTY,
        FLOOR,
        WALL,
        PORTALABLE_WALL,
        BUTTON,
        DOOR,
        LASER_EMITTER,
        LASER_RECEIVER,
        CUBE_DROPPER,
        EXIT
    };

    struct Tile {
        TileType type;
        Vector3 position;
        Quaternion rotation;
        std::map<std::string, std::any> properties;
    };

    std::vector<Tile> tiles;
    Vector3 playerStartPos;

public:
    // Constraint: Only allow valid tile placements
    bool CanPlaceTile(const Tile& tile, const Vector3& position) {
        // Floors must have walls/floor below
        if (tile.type == TileType::FLOOR) {
            Tile* below = GetTileAt(position + Vector3(0, -1, 0));
            if (!below || below->type == TileType::EMPTY) {
                return false;  // Floating floor not allowed
            }
        }

        // Portalable surfaces must be flat and 2x2 minimum
        if (tile.type == TileType::PORTALABLE_WALL) {
            if (!IsValidPortalSurface(position)) {
                return false;
            }
        }

        // Buttons must be on floor
        if (tile.type == TileType::BUTTON) {
            Tile* below = GetTileAt(position + Vector3(0, -1, 0));
            if (!below || below->type != TileType::FLOOR) {
                return false;
            }
        }

        return true;
    }

    // Validation: Puzzle must be completable
    struct ValidationResult {
        bool isValid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    ValidationResult ValidatePuzzle() {
        ValidationResult result;
        result.isValid = true;

        // Must have start and exit
        if (playerStartPos == Vector3::Zero()) {
            result.errors.push_back("Puzzle has no player start point");
            result.isValid = false;
        }

        if (!HasTileOfType(TileType::EXIT)) {
            result.errors.push_back("Puzzle has no exit");
            result.isValid = false;
        }

        // Exit must be reachable
        if (!IsExitReachable()) {
            result.errors.push_back("Exit is not reachable from start");
            result.isValid = false;
        }

        // Doors must have buttons
        for (const Tile& tile : tiles) {
            if (tile.type == TileType::DOOR) {
                std::string buttonId = std::any_cast<std::string>(tile.properties["buttonId"]);
                if (!HasButton(buttonId)) {
                    result.errors.push_back("Door has no associated button");
                    result.isValid = false;
                }
            }
        }

        // Warnings (not invalid, but suspicious)
        if (CountTilesOfType(TileType::CUBE_DROPPER) > 5) {
            result.warnings.push_back("Many cube droppers - are they all needed?");
        }

        return result;
    }

    // AI solver: Check if puzzle is solvable
    bool IsExitReachable() {
        // Run simplified AI solver
        PuzzleSolver solver;
        solver.SetPuzzle(this);
        solver.SetStartPosition(playerStartPos);

        // Try to reach exit using available tools
        return solver.CanReachExit(60.0f);  // 60 second timeout
    }

    // Export to shareable format
    std::string ExportPuzzle() {
        json data;
        data["version"] = 1;
        data["author"] = GetPlayerName();
        data["title"] = puzzleTitle;
        data["description"] = puzzleDescription;
        data["difficulty"] = estimatedDifficulty;  // 1-5 stars
        data["playerStart"] = {playerStartPos.x, playerStartPos.y, playerStartPos.z};

        for (const Tile& tile : tiles) {
            json tileData;
            tileData["type"] = static_cast<int>(tile.type);
            tileData["pos"] = {tile.position.x, tile.position.y, tile.position.z};
            tileData["rot"] = {tile.rotation.x, tile.rotation.y, tile.rotation.z, tile.rotation.w};
            data["tiles"].push_back(tileData);
        }

        return data.dump();
    }

    // One-click publish to Steam Workshop
    void PublishToWorkshop() {
        auto validation = ValidatePuzzle();
        if (!validation.isValid) {
            ShowDialog("Cannot publish puzzle - validation errors:\n" +
                      Join(validation.errors, "\n"));
            return;
        }

        std::string puzzleData = ExportPuzzle();
        std::string thumbnailPath = GenerateThumbnail();

        WorkshopItem item;
        item.title = puzzleTitle;
        item.description = puzzleDescription;
        item.previewImagePath = thumbnailPath;

        WorkshopIntegration::PublishItem(puzzleData, item);
        ShowDialog("Puzzle published successfully!");
    }
};
```

**Puzzle Maker UI:**

```
====================================
Portal 2 Puzzle Maker
====================================

[Palette]
Basic:
  [ ] Floor Panel
  [ ] Wall Panel
  [ ] Angled Panel
  [ ] Portalable Surface

Test Elements:
  [ ] Weighted Cube
  [ ] Companion Cube
  [ ] Button
  [ ] Door

Hazards:
  [ ] Turret
  [ ] Laser Field
  [ ] Goo

[Properties]
Selected: Button
- Connects to: Door_01
- Hold time: 1.0s
- Weight required: 50 units

[Validation]
✅ Puzzle has start and exit
✅ Exit is reachable
⚠ Warning: 3 cubes provided but only 2 needed

[Actions]
📝 Save
✔ Test Puzzle
🌐 Publish to Workshop
====================================
```

**Result:** ✅ Passes - Accessible creation with validation

---

## Decision Framework: Should You Support Modding?

### When to Support Mods

✅ **Your game benefits from mods if:**

1. **Replayability is key** - Mods provide infinite fresh content
   - Skyrim: 60,000+ mods, people still play after 13 years
   - Minecraft: Mods completely transform the experience

2. **Community creativity adds value** - Players can make things you never imagined
   - Warcraft 3: DOTA spawned MOBA genre
   - Half-Life: Counter-Strike became bigger than the base game

3. **You have limited content budget** - Community fills content gaps
   - Indie games: Mods extend small games significantly
   - Stardew Valley: Mods add hundreds of hours of content

4. **Game has strong core systems** - Good mechanics deserve more content
   - Factorio: Core factory building is so good, mods just add more
   - Kerbal Space Program: Physics sandbox perfect for modding

5. **Multiplayer benefits from variety** - Custom modes keep players engaged
   - Garry's Mod: Entirely community-driven game modes
   - CS:GO: Community maps and modes

6. **You want a long tail** - Mods keep games relevant for years
   - Games with mods have 2-5x longer player retention

### When NOT to Support Mods

❌ **Mods may hurt your game if:**

1. **Tight narrative experience** - Mods dilute story impact
   - The Last of Us: Story works because it's curated
   - God of War: Specific vision would be compromised

2. **Competitive balance is critical** - Mods create unfair advantages
   - Competitive shooters: Even cosmetic mods can give info advantage
   - Fighting games: Frame data mods change gameplay

3. **You're selling DLC** - Free mods compete with paid content
   - Risk: Players may prefer free community content
   - Mitigation: Make official content clearly superior

4. **Technical barriers** - Your engine isn't modder-friendly
   - Custom engine with no documentation
   - Heavily obfuscated code
   - Closed platforms (consoles without mod support)

5. **Legal/IP concerns** - Brand protection matters
   - Licensed IP: Can't allow mods that violate license
   - Family-friendly brand: Can't risk adult content mods

6. **Resource constraints** - Modding requires ongoing support
   - API maintenance
   - Community management
   - Legal oversight

### Hybrid Approaches

**Limited modding** works for some games:

1. **Cosmetic only** - Skins, textures, sounds
   - League of Legends: Custom skins (unsupported)
   - Overwatch: Workshop for custom modes only

2. **Custom maps only** - No gameplay changes
   - Call of Duty: Map editor but no mod support
   - Halo: Forge mode for maps, no scripting

3. **Sandbox mode** - Mods don't affect main game
   - Portal 2: Puzzle maker separate from campaign
   - Minecraft: Realms (no mods) vs. Java (mods everywhere)

4. **Curated mods** - Only approved mods allowed
   - Bethesda Creation Club: Paid, curated mods
   - Roblox: All content reviewed before publishing

---

## Implementation Checklist

When building moddable systems, check these boxes:

### Architecture
- [ ] Engine and content clearly separated
- [ ] Plugin system with stable interfaces
- [ ] Event hooks for common game moments
- [ ] Data-driven content (JSON/XML/YAML configs)
- [ ] Hot-reloading for rapid iteration

### API Design
- [ ] Semantic versioning (MAJOR.MINOR.PATCH)
- [ ] Deprecation cycle (warn before breaking changes)
- [ ] Comprehensive documentation with examples
- [ ] Stable C API or scripting language bindings
- [ ] Extension points don't require engine recompile

### Content Creation
- [ ] Official tools provided (editor, converter, validator)
- [ ] Example mods with source code
- [ ] Asset pipeline documented
- [ ] Testing tools for mod developers
- [ ] Performance profiling available

### Mod Management
- [ ] Metadata in manifest (name, version, author, dependencies)
- [ ] Dependency resolution and load order
- [ ] Conflict detection
- [ ] Enable/disable mods without restarting game
- [ ] Mod compatibility warnings

### Security
- [ ] Sandboxed scripting (no filesystem/network by default)
- [ ] Permission system for privileged operations
- [ ] Code signing for trusted mods
- [ ] Malware scanning integration
- [ ] Resource limits (CPU, memory, entities)

### Performance
- [ ] Profiling API for modders
- [ ] Validation tool checks performance
- [ ] Hard limits on resource usage
- [ ] Optimization guidelines documented
- [ ] LOD and culling systems respect modded content

### Distribution
- [ ] Steam Workshop / Mod.io / Nexus integration
- [ ] In-game mod browser
- [ ] One-click install/update
- [ ] Rating and review system
- [ ] Search and filtering by tags/category

### Legal
- [ ] Modding Terms of Service published
- [ ] Content ownership clarified
- [ ] Monetization policy defined
- [ ] DMCA process for copyright violations
- [ ] Age-appropriate content guidelines

### Community
- [ ] Official forums or Discord for modders
- [ ] Modding documentation wiki
- [ ] Regular communication about API changes
- [ ] Featured mods highlighted
- [ ] Mod author rewards/recognition program

---

## Common Pitfalls and Solutions

### Pitfall 1: Breaking Mods Every Update

**Problem:** Game patches constantly break all mods.

**Symptoms:**
- Player complaints after every update
- Modders give up maintaining mods
- Community fragments across game versions

**Solution:**

```cpp
// Version-stable ABI (Application Binary Interface)
class IGameAPI_V1 {
public:
    // Virtual destructor MUST be first
    virtual ~IGameAPI_V1() = default;

    // Never reorder functions!
    virtual void FunctionA() = 0;
    virtual void FunctionB() = 0;
    // If adding new function, add at END
    virtual void FunctionC() = 0;  // Added in 1.1

    // Don't change signatures - add new overload instead
    virtual Entity* SpawnEntity_V1(const char* type, float x, float y) = 0;
    virtual Entity* SpawnEntity_V2(const char* type, Vector3 pos, Quaternion rot) = 0;
};

// Maintain old API versions
class GameAPI_V1 : public IGameAPI_V1 {
    // Keep implementing old version even when you have V2
};

class IGameAPI_V2 : public IGameAPI_V1 {
    // V2 extends V1, doesn't replace it
};
```

### Pitfall 2: Security Vulnerabilities

**Problem:** Mods install malware or steal player data.

**Symptoms:**
- Players report stolen accounts
- Antivirus flags game as malware
- Negative press coverage

**Solution:**

```cpp
// Multi-layer security
class SecureModLoader {
    // 1. Static analysis before loading
    bool StaticAnalysis(const std::string& modPath) {
        // Check for suspicious API calls
        std::string code = ReadAllFiles(modPath);
        if (Contains(code, "DeleteFile") || Contains(code, "HttpRequest")) {
            LogWarning("Mod contains potentially dangerous calls");
            // Require user confirmation
            return AskUserPermission(modPath, "filesystem");
        }
        return true;
    }

    // 2. Sandboxed execution
    void LoadMod(const std::string& modPath) {
        ModSandbox sandbox(modPath);
        sandbox.RestrictFilesystem(GetModDataDirectory(modPath));
        sandbox.RestrictNetwork(false);  // No network by default
        sandbox.SetMemoryLimit(256 * 1024 * 1024);  // 256 MB
        sandbox.LoadPlugin(modPath);
    }

    // 3. Runtime monitoring
    void MonitorMod(const std::string& modId) {
        if (GetModCPUUsage(modId) > 100.0f) {  // 100ms per frame
            LogError("Mod %s using excessive CPU", modId.c_str());
            SuspendMod(modId);
        }
    }

    // 4. Code signing for trusted mods
    bool VerifySignature(const std::string& modPath) {
        std::string signature = ReadFile(modPath + "/signature.sig");
        std::string publicKey = GetPublisherPublicKey();
        return CryptoVerify(signature, publicKey, GetModHash(modPath));
    }
};
```

### Pitfall 3: Mod Conflicts and Crashes

**Problem:** Mods override each other, causing crashes.

**Symptoms:**
- Game crashes with certain mod combinations
- Load order matters but isn't clear
- Silent failures (features just don't work)

**Solution:**

```cpp
// Conflict detection and resolution
class ModConflictResolver {
    struct Conflict {
        std::string modA;
        std::string modB;
        std::string resource;
        ConflictType type;
    };

    enum class ConflictType {
        OVERWRITE,      // Both define same resource
        INCOMPATIBLE,   // Mods explicitly incompatible
        DEPENDENCY,     // Circular dependency
        API_VERSION     // Require different API versions
    };

    std::vector<Conflict> DetectConflicts(const std::vector<ModManifest>& mods) {
        std::vector<Conflict> conflicts;
        std::map<std::string, std::string> resourceOwners;

        for (const auto& mod : mods) {
            // Check explicit incompatibilities
            for (const auto& conflict : mod.conflicts) {
                if (IsModLoaded(conflict)) {
                    conflicts.push_back({
                        mod.id, conflict, "", ConflictType::INCOMPATIBLE
                    });
                }
            }

            // Check resource overwrites
            for (const auto& resource : mod.provides) {
                if (resourceOwners.count(resource)) {
                    // Resource already provided by another mod
                    std::string owner = resourceOwners[resource];

                    // Check if there's a priority or patch system
                    if (CanPatch(owner, mod.id, resource)) {
                        // Patchable - allow overwrite
                        LogInfo("Mod %s patches %s from %s",
                               mod.id.c_str(), resource.c_str(), owner.c_str());
                    } else {
                        conflicts.push_back({
                            owner, mod.id, resource, ConflictType::OVERWRITE
                        });
                    }
                }
                resourceOwners[resource] = mod.id;
            }
        }

        return conflicts;
    }

    void ShowConflictUI(const std::vector<Conflict>& conflicts) {
        if (conflicts.empty()) return;

        ImGui::Begin("Mod Conflicts Detected");
        ImGui::TextColored(ImVec4(1, 0, 0, 1),
                          "%d conflicts found", (int)conflicts.size());

        for (const auto& conflict : conflicts) {
            ImGui::Separator();
            ImGui::Text("Conflict: %s vs %s",
                       conflict.modA.c_str(), conflict.modB.c_str());

            if (conflict.type == ConflictType::OVERWRITE) {
                ImGui::Text("Both mods provide: %s", conflict.resource.c_str());
                ImGui::Text("Choose which mod to use:");

                if (ImGui::Button(("Use " + conflict.modA).c_str())) {
                    SetModPriority(conflict.modA, PRIORITY_HIGH);
                }
                ImGui::SameLine();
                if (ImGui::Button(("Use " + conflict.modB).c_str())) {
                    SetModPriority(conflict.modB, PRIORITY_HIGH);
                }
            } else if (conflict.type == ConflictType::INCOMPATIBLE) {
                ImGui::Text("These mods cannot be used together");
                if (ImGui::Button(("Disable " + conflict.modB).c_str())) {
                    DisableMod(conflict.modB);
                }
            }
        }

        ImGui::End();
    }
};
```

### Pitfall 4: Poor Documentation

**Problem:** Modders can't figure out how to use your API.

**Symptoms:**
- Forum full of "how do I..." questions
- Same questions asked repeatedly
- Mods use hacky workarounds instead of proper API

**Solution:**

```cpp
// Self-documenting API with examples in comments

/**
 * Spawns an entity in the game world.
 *
 * @param entityType The type of entity to spawn (e.g., "enemy_zombie", "pickup_health")
 * @param position World position to spawn at
 * @param rotation Initial rotation (use Quaternion::Identity() for default)
 * @return Pointer to spawned entity, or nullptr if spawn failed
 *
 * @example
 * ```cpp
 * // Spawn a zombie at position (100, 0, 50)
 * Entity* zombie = SpawnEntity("enemy_zombie", Vector3(100, 0, 50), Quaternion::Identity());
 * if (zombie) {
 *     zombie->SetHealth(150);
 *     zombie->SetAggressive(true);
 * }
 * ```
 *
 * @note Entity types must be registered before spawning. See RegisterEntityType().
 * @warning Spawning too many entities (>1000) may impact performance.
 * @see Entity::SetHealth(), RegisterEntityType()
 */
Entity* SpawnEntity(const std::string& entityType, const Vector3& position, const Quaternion& rotation);

// Generate HTML documentation from comments
// Use Doxygen, Sphinx, or custom tool
```

**Create comprehensive wiki:**

```markdown
# Modding API Reference

## Getting Started

### Your First Mod

1. Create folder: `mods/my_first_mod/`
2. Create `manifest.json`:
   ```json
   {
     "id": "my_first_mod",
     "name": "My First Mod",
     "version": "1.0.0",
     "author": "YourName"
   }
   ```
3. Create `plugin.cpp`:
   ```cpp
   #include "GameAPI.h"

   class MyFirstMod : public IGamePlugin {
   public:
       bool OnLoad() override {
           LogInfo("My first mod loaded!");
           return true;
       }
   };

   extern "C" {
       IGamePlugin* CreatePlugin() {
           return new MyFirstMod();
       }
   }
   ```
4. Compile: `make`
5. Launch game - your mod should load!

## Tutorials

- [Tutorial 1: Adding a Custom Tower](tutorials/custom-tower.md)
- [Tutorial 2: Creating New Enemies](tutorials/custom-enemy.md)
- [Tutorial 3: Building a Custom Map](tutorials/custom-map.md)

## API Reference

- [Entity System](api/entities.md)
- [Event System](api/events.md)
- [Resource Management](api/resources.md)

## Examples

See `examples/` folder for complete example mods with source code.
```

### Pitfall 5: Mod Tools Are Afterthought

**Problem:** Developer tools exist but are never released to modders.

**Symptoms:**
- Modders reverse-engineer file formats
- Community creates unofficial tools
- Mods are low quality due to poor tooling

**Solution:**

```cpp
// Design public tools from day one
// Use the SAME tools internally that modders will use

class UnifiedToolchain {
public:
    // Asset compiler used by developers AND modders
    static bool CompileAssets(const std::string& sourceDir, const std::string& outputDir) {
        AssetCompiler compiler;

        // Public, documented formats
        for (const auto& model : FindFiles(sourceDir, "*.fbx", "*.obj")) {
            compiler.CompileModel(model, outputDir);
        }

        for (const auto& texture : FindFiles(sourceDir, "*.png", "*.tga")) {
            compiler.CompileTexture(texture, outputDir);
        }

        // Validate all assets
        ValidationResult result = compiler.Validate();
        if (!result.passed) {
            PrintErrors(result.errors);
            return false;
        }

        return true;
    }

    // Map editor used by designers AND modders
    static void LaunchMapEditor(const std::string& mapPath = "") {
        MapEditor editor;

        if (!mapPath.empty()) {
            editor.LoadMap(mapPath);
        }

        editor.Run();  // Same editor, no special "dev mode"
    }
};

// If YOUR developers struggle with the tools, modders will too
// Make tools good enough that you WANT to use them
```

---

## Summary

**Designing for extensibility transforms players into creators.** The most successful moddable games—Skyrim, Minecraft, Factorio, Warcraft 3—didn't just allow mods as an afterthought. They designed for modding from day one, providing stable APIs, powerful tools, and thriving ecosystems.

**Key principles:**

1. **Separate engine from content** - Plugin architecture with clear interfaces
2. **Stable API contracts** - Semantic versioning, deprecation cycles, compatibility layers
3. **Provide tools** - Use the same tools internally that you give to modders
4. **Secure the system** - Sandboxing, permissions, resource limits
5. **Support the community** - Documentation, examples, featured mods
6. **Clear legal policies** - ToS, licensing, monetization rules

**When done right, mods:**
- Extend game lifespan by years
- Create content you never imagined
- Build passionate communities
- Generate free marketing
- Inspire entirely new genres

**The investment pays off:** Games with healthy modding ecosystems have 2-5x longer player retention and generate word-of-mouth marketing that no ad budget can match.

Your game isn't done when you ship it—it's just the beginning of what players will create.
