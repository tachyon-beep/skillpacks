# Codebase Confidence Building

Systematically internalize code you're responsible for. Go from "I sort of know this" to "I own this."

## Core Principle

**Confidence comes from mental models, not memorization.** You don't need to remember every line - you need to understand how pieces fit together, why decisions were made, and where dragons live. Build maps, not encyclopedias.

## When to Use This

- Taking ownership of unfamiliar code
- "I'm responsible for this but only ~70% confident"
- New to a team/codebase
- Inherited code from departed colleague
- Need to maintain code you didn't write

**Don't use for**: One-time debugging (use [complex-debugging.md](complex-debugging.md)), architecture analysis for documentation (use `axiom-system-archaeologist`), understanding before refactoring (this skill first, THEN [systematic-refactoring.md](systematic-refactoring.md)).

---

## The Confidence Building Process

```
┌─────────────────┐
│ 1. ORIENT       │ ← Get the lay of the land
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. TRACE        │ ← Follow the data
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. INTERROGATE  │ ← Ask why, find assumptions
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. EXPERIMENT   │ ← Verify understanding
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. DOCUMENT     │ ← Capture for future you
└─────────────────┘
```

---

## Phase 1: Orient

**Get the big picture before diving in.**

### 30-Minute Overview

| Activity | Time | Goal |
|----------|------|------|
| Read README | 5 min | Stated purpose and setup |
| List top-level dirs | 5 min | Structural organization |
| Read entry points | 10 min | Where execution starts |
| Scan config files | 5 min | External dependencies, feature flags |
| Check CI/CD | 5 min | How it's tested and deployed |

### Key Questions

| Question | Why It Matters |
|----------|----------------|
| What does this system DO? | Core purpose guides everything |
| Who are the users? | Internal? External? Both? |
| What are the inputs/outputs? | System boundaries |
| What does it depend on? | External services, databases |
| What depends on it? | Downstream consumers |

### Structural Map

Draw a rough map:

```
┌─────────────────────────────────────────┐
│               SYSTEM                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │  API    │→ │ Service │→ │  Data   │  │
│  │  Layer  │  │  Layer  │  │  Layer  │  │
│  └─────────┘  └─────────┘  └─────────┘  │
│       ↑                          ↓      │
│   [Users]                   [Database]  │
└─────────────────────────────────────────┘
```

Doesn't need to be accurate - you'll refine it.

---

## Phase 2: Trace

**Follow data through the system.**

### Pick Critical Paths

Identify 3-5 most important operations:
- Most frequent user action
- Most critical business operation
- Most complex flow
- Most failure-prone area

### Trace Technique

For each critical path:

1. **Start at entry** - API endpoint, event handler, CLI command
2. **Follow the data** - What transforms happen?
3. **Mark decision points** - Where does flow branch?
4. **Note side effects** - What gets written/sent?
5. **Find the exit** - Response, result, completion

```python
# Example trace notes
# POST /api/orders
#   → OrderController.create()
#     → validates input (OrderSchema)
#     → OrderService.create_order()
#       → checks inventory (InventoryService)  # EXTERNAL CALL
#       → calculates pricing (PricingEngine)   # COMPLEX LOGIC
#       → persists (OrderRepository)           # DB WRITE
#       → emits event (EventBus)               # SIDE EFFECT
#     → returns OrderResponse
```

### Claude-Specific Advantage

You can trace FAST:
- Read entire call chains in seconds
- Search all usages of a function
- Find all implementations of an interface

```bash
# Find all callers
grep -r "create_order" --include="*.py"

# Find all implementations
grep -r "class.*OrderService" --include="*.py"

# Find all event handlers
grep -r "on_order_created\|OrderCreated" --include="*.py"
```

---

## Phase 3: Interrogate

**Ask why, find hidden assumptions.**

### The Why Questions

For each significant piece of code:

| Question | What It Reveals |
|----------|-----------------|
| Why this approach? | Design rationale |
| Why not the obvious way? | Hidden constraints |
| What would break this? | Fragility points |
| When was this written? | Context and age |
| Who wrote this? | Who to ask |

### Finding Hidden Assumptions

Look for:

```python
# Magic numbers - what do they mean?
TIMEOUT = 30  # Why 30? What happens at 29? 31?

# Implicit ordering - what if order changes?
process_a()
process_b()  # Does B depend on A?

# Error swallowing - what's being hidden?
try:
    risky_operation()
except Exception:
    pass  # Why ignore ALL errors?

# Comments that smell
# TODO: This is temporary  # How temporary? 3 years old?
# Don't change this!  # Why not? What happens?
# This shouldn't happen  # But what if it does?
```

### Git Archaeology

History reveals intent:

```bash
# Who touched this and when?
git log --oneline -20 -- path/to/file.py

# What did the original commit say?
git log --follow -p -- path/to/file.py | head -100

# When did this behavior start?
git log -S "suspicious_pattern" --oneline

# What was the PR/issue?
git log --grep="ticket-123" --oneline
```

### Ask Humans (If Available)

Questions for teammates:
- "What's the history of this module?"
- "Are there any gotchas I should know?"
- "What's the scariest part of this code?"
- "What would you change if you could?"

---

## Phase 4: Experiment

**Verify your mental model.**

### Safe Experiments

| Experiment | What It Proves |
|------------|----------------|
| Run tests | What's actually tested |
| Break something, see what fails | Dependencies and coverage |
| Add logging, trace a request | Actual flow matches mental model |
| Change a value, observe result | You understand the effect |

### Prediction Testing

Before running code:
1. **Predict** what will happen
2. **Run** the code
3. **Compare** prediction to reality
4. **Update** mental model if wrong

```python
# Prediction: This will return 404 if user not found
response = client.get("/users/nonexistent")
# Actual: Returns 500 with stack trace
# Update: Error handling is incomplete, user lookup crashes on miss
```

### Test the Tests

```bash
# Run test suite
pytest tests/

# Check what's NOT tested
pytest --cov=src --cov-report=term-missing

# Break something, verify test catches it
# If test doesn't fail, you found a coverage gap
```

---

## Phase 5: Document

**Capture understanding for future you.**

### Personal Notes

Keep notes as you learn:

```markdown
# Order System Notes

## Architecture
- Controllers → Services → Repositories
- All async, uses event bus for side effects

## Critical Paths
1. Order creation: See trace in `traces/order-create.md`
2. Payment processing: Third-party integration, fragile

## Gotchas
- InventoryService caches for 5 min - can cause overselling
- PricingEngine has undocumented discount rules in `pricing_rules.json`
- Never call OrderService.delete() - soft delete only

## Questions to Investigate
- [ ] Why does checkout have retry logic but payment doesn't?
- [ ] What happens when event bus is down?

## Key People
- Alice: Original author, knows history
- Bob: Runs production, knows failure modes
```

### Update Existing Docs

If you find docs are wrong or missing:
- Fix them as you learn
- Add "last verified" dates
- Note assumptions explicitly

### Create Missing Docs

If key documentation doesn't exist:

```markdown
# [Module] Overview

## Purpose
One paragraph on what this does and why.

## Architecture
Brief structural description with diagram.

## Key Components
- Component A: Does X
- Component B: Does Y

## Common Operations
How to do typical tasks.

## Gotchas
What will surprise people.

## Dependencies
What this needs to work.
```

---

## Confidence Levels

### Self-Assessment

| Level | Description | Can You... |
|-------|-------------|------------|
| **20%** | Aware it exists | Find the code |
| **40%** | Surface understanding | Explain what it does |
| **60%** | Working knowledge | Make simple changes |
| **80%** | Solid understanding | Debug issues, add features |
| **95%** | Deep expertise | Redesign, mentor others |

### Target: 80% for Code You Own

You don't need 95% on everything. Target:
- **95%**: Core components you modify often
- **80%**: Areas you're responsible for
- **60%**: Adjacent systems
- **40%**: External dependencies

### Diminishing Returns

```
Time invested:   |████████████████████|
Confidence:      20%  40%  60%  80%  95%
Marginal effort: Low  Low  Med  High Very High
```

**Stop at 80% for most code.** The last 15% takes as much time as the first 80%.

---

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| [technical-debt-triage.md](technical-debt-triage.md) | Confidence reveals debt |
| [code-review-methodology.md](code-review-methodology.md) | Reviews deepen understanding |
| [complex-debugging.md](complex-debugging.md) | Debugging builds confidence |
| [systematic-refactoring.md](systematic-refactoring.md) | Refactoring requires confidence |
| `axiom-system-archaeologist` | For formal architecture analysis |

---

## Red Flags

| Thought | Reality | Action |
|---------|---------|--------|
| "I'll learn it when I need to" | Reactive learning is slower | Invest time upfront |
| "I mostly understand it" | Vague confidence is false confidence | Test your understanding |
| "The code is self-documenting" | It never is | Take notes anyway |
| "I'll remember this" | You won't | Write it down |
| "I need to understand everything" | Diminishing returns | Target 80%, move on |
| "I can just ask Alice" | Alice might leave | Document what Alice knows |

---

## Quick Reference

### Confidence Checklist

- [ ] **Orient**: Do I have a structural map?
- [ ] **Trace**: Have I followed critical paths?
- [ ] **Interrogate**: Do I know why, not just what?
- [ ] **Experiment**: Have I verified my mental model?
- [ ] **Document**: Will future-me thank me?

### Quick Wins

1. **Read tests first** - They document expected behavior
2. **Find the entry point** - Follow from there
3. **Git blame strategically** - History reveals intent
4. **Break it on purpose** - See what catches it
5. **Draw a diagram** - Forces clarity

### Time Investment Guide

| Codebase Size | Initial Investment | Ongoing |
|---------------|-------------------|---------|
| Small (<10K LOC) | 1-2 days | As needed |
| Medium (10-50K) | 1 week | 1 day/week |
| Large (50K+) | 2-4 weeks | Continuous |
