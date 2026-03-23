# Expert Routing Guide

Advanced guidance for routing decisions, including tips, checklists, self-checks, and edge cases.

---

## Expert Routing Tips

### Tip 1: Listen for hidden requirements

Users often describe WHAT they want without understanding WHICH simulation type they need.

**Examples**:
- "I want intelligent enemies" → Could be ai-and-agent-simulation OR traffic-and-pathfinding OR both
- "I need realistic physics" → Could be physics-simulation-patterns OR just kinematic movement
- "I want a living world" → Could be ecosystem-simulation OR ai-and-agent-simulation OR weather-and-time

**Fix**: Ask clarifying questions:
- "Do enemies need to navigate complex terrain?" (pathfinding)
- "Do enemies need to make tactical decisions?" (AI)
- "Does 'living world' mean wildlife, weather, or both?" (ecosystem vs weather)

### Tip 2: Recognize anti-patterns

Some phrases indicate the user is heading toward common mistakes:

**Red flags**:
- "I want to simulate EVERYTHING" → Over-engineering, route to simulation-vs-faking
- "It needs to be perfectly realistic" → Perfectionism trap, route to simulation-vs-faking
- "I'll optimize later" → True, but ensure they know when "later" is (after profiling)
- "I changed one parameter and it exploded" → Chaos, route to debugging-simulation-chaos
- "It works on my machine but desyncs in multiplayer" → Determinism bug, route to debugging-simulation-chaos

### Tip 3: Recognize interdependencies

Some skill combinations have ordering requirements:

**Dependencies**:
- ai-and-agent-simulation depends on traffic-and-pathfinding (if agents need to navigate)
- crowd-simulation depends on traffic-and-pathfinding (for underlying navigation)
- ecosystem-simulation depends on ai-and-agent-simulation (for animal behaviors)
- performance-optimization-for-sims depends on having working simulation first

**Rule**: Foundation skills (simulation-vs-faking, core implementations) before dependent skills (optimization, debugging)

### Tip 4: Scale determines routing

The number of entities changes which skills are needed:

**Scale breakpoints**:
- **< 10 entities**: Basic implementation, no special optimization
- **10-100 entities**: May need performance-optimization-for-sims
- **100-1000 entities**: Definitely need performance-optimization-for-sims, spatial partitioning, LOD
- **1000+ entities**: Need aggressive optimization, time-slicing, hybrid LOD

**Example**: "I need 10 NPCs" vs "I need 10,000 NPCs" route to same implementation skill, but latter ALSO routes to performance-optimization-for-sims.

### Tip 5: Genre provides context

Game genre suggests which skills are commonly needed:

**Genre routing patterns**:
- **RTS/Strategy**: ai-and-agent-simulation + traffic-and-pathfinding + performance-optimization
- **Survival**: ecosystem-simulation + ai-and-agent-simulation + weather-and-time
- **City Builder**: traffic-and-pathfinding + economic-simulation + simulation-vs-faking
- **Racing**: physics-simulation-patterns + performance-optimization
- **MMO**: economic-simulation + debugging-simulation-chaos (determinism)
- **Open World**: traffic-and-pathfinding + crowd-simulation + weather-and-time
- **Battle Royale**: simulation-vs-faking (aggressive LOD) + debugging-simulation-chaos (determinism)

Don't over-assume based on genre, but use it as a starting hypothesis.

---

## Implementation Checklist

When routing to multiple skills, use this checklist to ensure proper workflow:

### Phase 1: Planning (Always First)
- [ ] Route to simulation-vs-faking
- [ ] Identify all applicable simulation domains
- [ ] Determine implementation order based on dependencies
- [ ] Validate that simulation is actually needed

### Phase 2: Implementation (Core Systems)
- [ ] Implement foundation skills first (pathfinding before AI, etc.)
- [ ] Test each system independently before integration
- [ ] Ensure determinism if multiplayer is planned
- [ ] Validate against "good enough" threshold from simulation-vs-faking

### Phase 3: Integration (Combining Systems)
- [ ] Integrate systems in dependency order
- [ ] Test combined systems at target scale
- [ ] Profile to identify bottlenecks (if any)

### Phase 4: Optimization (Only If Needed)
- [ ] Profile to measure performance
- [ ] Route to performance-optimization-for-sims only if bottleneck exists
- [ ] Re-test after optimization
- [ ] Validate gameplay still feels correct

### Phase 5: Debugging (Only If Broken)
- [ ] Route to debugging-simulation-chaos if bugs occur
- [ ] Use systematic debugging process
- [ ] Fix root cause, not symptoms
- [ ] Add prevention measures

---

## Meta-Skill Self-Check

After using this meta-skill, verify your routing with these questions:

**Routing accuracy**:
- [ ] Did I start with simulation-vs-faking?
- [ ] Did I identify ALL applicable simulation domains?
- [ ] Did I avoid routing to performance-optimization-for-sims prematurely?
- [ ] Did I only route to debugging-simulation-chaos for actual bugs?

**Workflow correctness**:
- [ ] Am I implementing foundation skills before dependent skills?
- [ ] Have I considered interdependencies between skills?
- [ ] Is the implementation order logical?

**Efficiency**:
- [ ] Am I using the minimum skills needed?
- [ ] Have I avoided over-engineering?
- [ ] Am I respecting the "good enough" threshold?

**Completeness**:
- [ ] Have I considered multiplayer determinism (if applicable)?
- [ ] Have I planned for scale (if thousands of entities)?
- [ ] Have I validated gameplay implications?

---

## Advanced Routing: Edge Cases

### Edge Case 1: "I don't know what kind of simulation I need"

**Symptom**: User describes game but unclear which simulation domains apply

**Process**:
1. Route to simulation-vs-faking anyway (helps clarify requirements)
2. Ask probing questions about specific systems:
   - "Do you have moving agents?" (pathfinding/AI)
   - "Is there combat?" (physics/AI)
   - "Is there economy/trading?" (economic)
   - "Is there wildlife?" (ecosystem)
3. Route to identified domains

**Example**: "I'm making a survival game" → Ask about hunting (ecosystem), crafting (economy), weather (weather-and-time), etc.

### Edge Case 2: "My simulation needs to be deterministic"

**Symptom**: Multiplayer, replay system, or deterministic requirement

**Process**:
1. Read debugging-simulation-chaos for **determinism requirements** (NOT for debugging - for learning constraints)
2. Then route to implementation skill(s)
3. Implement with determinism constraints from start (cheaper than refactoring)

**Why**: Determinism requirements affect implementation decisions. This is preventive learning, not reactive debugging. Better to know constraints early than refactor later.

### Edge Case 3: "I need simulation but performance is already a concern"

**Symptom**: Performance budget known to be tight from start

**Process**:
1. Route to simulation-vs-faking (aggressive use of faking/LOD)
2. Route to implementation skill(s)
3. Route to performance-optimization-for-sims for architectural guidance
4. Implement with performance in mind from start

**Why**: If performance is constrained, design for performance from the beginning. Don't implement naive version first.

### Edge Case 4: "I'm refactoring existing simulation"

**Symptom**: Working simulation exists but needs improvement

**Process**:
1. If broken: debugging-simulation-chaos first
2. If slow: performance-optimization-for-sims
3. If wrong architecture: simulation-vs-faking to reconsider design, then relevant implementation skill

**Why**: Refactoring is different from greenfield. Identify the problem (bug, performance, design) before routing.

### Edge Case 5: "I need simulation for tool/editor, not game"

**Symptom**: Simulation is for preview/visualization, not runtime gameplay

**Process**:
1. Route to simulation-vs-faking (tools have different constraints than games)
2. Route to implementation skill(s)
3. Optimize for accuracy over performance (tools can be slower)

**Why**: Tool simulations prioritize accuracy and debuggability over frame rate.
