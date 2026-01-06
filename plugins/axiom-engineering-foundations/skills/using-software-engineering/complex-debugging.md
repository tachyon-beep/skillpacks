# Complex Debugging

Scientific method applied to horrible bugs. This is your systematic process when a bug resists simple fixes.

## Core Principle

**Debugging is applied science, not guessing.** You form hypotheses, design experiments to falsify them, and follow the evidence. "Try random things until it works" is not debugging - it's gambling.

## When to Use This

- Bug resists simple fixes ("I've tried everything")
- Can't reproduce reliably
- Intermittent failures
- Multi-system interactions
- "Works on my machine"
- Heisenbugs (behavior changes when observed)
- Race conditions suspected
- Legacy code with no tests

## Claude-Specific Adaptations

**Your Strengths** (lean into these):
- **Fast codebase scanning** - Read entire modules in seconds
- **Pattern recognition** - Spot similar patterns across files
- **Exhaustive search** - Check EVERY occurrence of a function/pattern
- **No fatigue** - Systematic processes don't tire you out
- **Session memory** - Track hypotheses and rule them out

**Your Limitations** (work around these):
- **No interactive debuggers** - Can't step through with gdb/pdb
- **No runtime observation** - Can't watch variables change live
- **No production access** - Must ask user for logs, metrics, state
- **Context window limits** - Can't process massive log dumps

**Compensation strategies**:
- Add logging/print statements to observe state (user runs, reports back)
- Use git bisect for regressions (automated, no interactivity needed)
- Read test output carefully - it's your window into runtime
- Ask user to run specific commands and report results
- **Filter logs before reading** - grep for ERROR/WARN/FATAL first, don't fill context with HTTP 200 OKs

---

## The Debugging Process

```
┌─────────────────┐
│ 0. STATIC CHECK │ ← Rule out obvious errors first
└────────┬────────┘
         ↓
┌─────────────────┐
│ 1. REPRODUCE    │ ← Can't debug what you can't trigger
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. ISOLATE      │ ← Narrow the search space
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. HYPOTHESIS   │ ← Form → Experiment → Falsify/Confirm
│    LOOP         │───→ (loop until confirmed)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. ROOT CAUSE   │ ← 5 Whys - go deep enough
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. FIX & VERIFY │ ← Test-driven fix
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. PREVENT      │ ← Stop this class of bug forever
└─────────────────┘
```

---

## Phase 0: Static Sanity Check

**Before debugging runtime behavior, rule out compile-time/static errors.**

Don't debug a runtime "mystery" if the linter is already screaming.

### Quick Static Checks

```bash
# Type errors (Python)
mypy src/ --ignore-missing-imports

# Type errors (TypeScript)
npx tsc --noEmit

# Linting
ruff check src/  # Python
eslint src/      # JavaScript

# Import/dependency issues
python -c "import your_module"  # Does it even import?
```

### What to Look For

| Check | Catches |
|-------|---------|
| **Type checker** | Type mismatches that cause runtime errors |
| **Linter** | Undefined variables, unused imports, common mistakes |
| **Import test** | Circular dependencies, missing modules |
| **Syntax check** | Parse errors that look like logic bugs |

**If static analysis finds errors, fix those first.** They may be the root cause, or they'll confuse your debugging.

---

## Phase 1: Reproduction

**You cannot debug what you cannot reliably trigger.**

### Create Minimal Reproducible Example (MRE)

Strip to the absolute minimum needed to trigger the bug:
1. Remove unrelated code/config
2. Simplify inputs to smallest failing case
3. Eliminate external dependencies where possible
4. Document exact steps to reproduce

**Ask the user**:
- "What exact steps trigger this?"
- "What's the smallest input that fails?"
- "Does it happen every time or intermittently?"

### Classify Determinism

| Type | Behavior | Strategy |
|------|----------|----------|
| **Deterministic** | Same input → same failure | Standard debugging |
| **Intermittent** | Sometimes fails | Force reproducibility (see below) |
| **Heisenbug** | Disappears when observed | Minimal instrumentation, logging |
| **Environment-dependent** | Only in prod/CI | Environment diff analysis |

### Forcing Reproducibility for Intermittent Bugs

- **Fix random seeds** - numpy, random, torch seeds
- **Control timing** - Add delays, mock network latency
- **Increase stress** - Run in loop, parallel instances
- **Log everything** - Capture state leading to failure
- **Reduce resources** - Lower memory/CPU to force race conditions

### Environment Verification

Confirm the bug exists in a controlled environment:
```bash
# Check environment matches report
python --version
pip freeze | grep <suspect-package>
git log -1  # Exact commit
env | grep <relevant-vars>
```

**Red flag**: If you can't reproduce, STOP. Get more information from the user before proceeding.

---

## Phase 2: Isolation

Systematically narrow the problem space using binary search across two dimensions.

### Two Dimensions of Search

| Dimension | Question | Technique | When to Use |
|-----------|----------|-----------|-------------|
| **Space** | Where in the code? | Checkpoints, commenting out | Bug present since forever, or you don't know when it started |
| **Time** | When in history? | Git bisect | "It used to work" - regression |

**Start with Time** if you have a known-good version. It's faster and gives you a diff to read.
**Fall back to Space** when you can't identify when it broke, or when it's always been broken.

### Space Search: Where in the Code?

Binary search through code structure:

1. Identify the full path from input to failure
2. Add checkpoint at midpoint - does data look correct?
3. If correct: bug is in second half
4. If wrong: bug is in first half
5. Repeat until isolated

```python
# Add checkpoints to narrow down
def process_data(data):
    step1_result = transform(data)
    print(f"CHECKPOINT 1: {step1_result[:100]}")  # Inspect

    step2_result = validate(step1_result)
    print(f"CHECKPOINT 2: {step2_result}")  # Inspect

    return finalize(step2_result)
```

**Alternative**: Comment out entire subsystems until bug disappears, then restore incrementally.

### Time Search: When Did It Break?

Git bisect automates binary search through commit history:

```bash
git bisect start
git bisect bad HEAD           # Current is broken
git bisect good v1.2.3        # This version worked
# Git checks out middle commit
# Run test, then:
git bisect good  # or
git bisect bad
# Repeat until culprit found
git bisect reset
```

**Claude advantage**: You can read the suspect commit diff immediately after bisect identifies it. Often the culprit is obvious from the diff.

### Wolf Fence Tactic

Place "fences" at system boundaries:
1. Identify interfaces between components (API calls, function boundaries)
2. Log input/output at each boundary
3. Find where correct data becomes incorrect
4. The bug is between the last good fence and first bad fence

```python
def api_handler(request):
    print(f"FENCE IN: {request}")  # Check input
    result = business_logic(request)
    print(f"FENCE OUT: {result}")  # Check output
    return result
```

### Saff Squeeze

For test failures, inline code to isolate:
1. Start with failing test
2. Inline the method under test into the test
3. Inline deeper until you find the failing line
4. You now have minimal failing case

---

## Phase 3: Hypothesis Loop

The intellectual core - slow down and think.

### Form Explicit Hypothesis

Write it down (or state to user):
> "I believe [COMPONENT] is failing because [CAUSE] when [CONDITION]"

Examples:
- "I believe the cache is returning stale data because invalidation isn't triggered when the user updates their profile"
- "I believe the race condition occurs because two threads access the counter without locking"

### Design Falsifying Experiment

**Critical**: Design experiments to prove the hypothesis WRONG, not right.

| Hypothesis | Falsifying Experiment |
|------------|----------------------|
| "Cache is stale" | Disable cache entirely - does bug disappear? |
| "Race condition in counter" | Add lock - does bug disappear? |
| "Null from API" | Log actual API response - is it really null? |
| "Wrong config in prod" | Print config at runtime - what's the actual value? |

### Make Prediction

Before running experiment, state expected outcome:
> "If my hypothesis is correct, when I [ACTION], I should see [RESULT]"

This prevents hindsight bias ("oh yeah, I expected that").

### Observe and Record

Run the experiment. Record:
- Actual result (not interpreted)
- Did it match prediction?
- What new information emerged?

### Refine or Pivot

| Outcome | Action |
|---------|--------|
| Hypothesis confirmed | Proceed to root cause analysis |
| Hypothesis falsified | Form new hypothesis using new data |
| Inconclusive | Design better experiment |

**Track ruled-out hypotheses** - Don't re-test what you've already falsified.

### Status Communication

For complex investigations, maintain a running status to the user:

```markdown
**Current Hypothesis**: Cache invalidation fails for profile updates
**Experiment in Progress**: Adding logging to cache.invalidate()
**Ruled Out**: Database query timing, network timeouts
**Next if Current Fails**: Check event listener registration
```

This keeps users informed and creates a record of your investigation path.

---

## Phase 4: Root Cause Analysis

**Don't fix the symptom.** Go deep enough to fix the actual cause.

### The 5 Whys

Keep asking "why" until you reach the root:

```
Bug: Application crashes with NullPointerException
Why? → The user object was null
Why? → The API returned 404
Why? → The user ID was stale
Why? → Cache invalidation failed
Why? → Cache TTL was set to 24h but user data changes hourly
ROOT CAUSE: Cache TTL misconfiguration
```

**Stop when**: You reach something you can fix that prevents the entire class of bug.

### Symptom vs. Cause

| Fix Type | Example | Problem |
|----------|---------|---------|
| **Symptom fix** | Add null check | Bug will resurface elsewhere |
| **Cause fix** | Fix cache invalidation | Prevents entire bug class |

Ask: "If I make this fix, could a similar bug happen in another code path?"

### Common Root Cause Categories

| Category | Symptoms | Typical Causes |
|----------|----------|----------------|
| **State corruption** | Wrong values, inconsistent data | Missing synchronization, partial updates |
| **Resource exhaustion** | Slowdown, OOM, timeouts | Leaks, unbounded growth, missing cleanup |
| **Timing/ordering** | Intermittent failures, races | Missing locks, wrong async handling |
| **Configuration** | Works locally, fails elsewhere | Env vars, feature flags, secrets |
| **Data** | Specific inputs fail | Edge cases, encoding, malformed input |
| **Dependencies** | After upgrade/deploy | Version mismatch, API changes |

---

## Phase 5: Fix & Verify

The fix isn't done until proven correct.

### Test-Driven Fix

1. **Write failing test first** - Captures the bug in your test suite
2. **Verify test fails** - Proves test actually catches the bug
3. **Implement fix**
4. **Verify test passes** - Proves fix works
5. **Run full test suite** - Check for regressions

```python
# 1. Write test that captures the bug
def test_cache_invalidates_on_profile_update():
    user = create_user(name="old")
    update_profile(user.id, name="new")
    # This SHOULD pass but currently fails due to stale cache
    assert get_user(user.id).name == "new"

# 2. Verify it fails (before fix)
# 3. Implement fix
# 4. Verify it passes (after fix)
```

### Impact Analysis

Before merging fix, check:
- What other code calls the changed function?
- Could the fix break existing behavior?
- Are there similar patterns elsewhere that need the same fix?

```bash
# Find all callers
grep -r "function_name" --include="*.py"

# Check for similar patterns
grep -r "similar_pattern" --include="*.py"
```

### Artifacts Checklist

| Artifact | Purpose |
|----------|---------|
| Reproduction steps | Future debugging reference |
| Root cause explanation | Understanding for team |
| Failing test | Prevents regression |
| Fix commit (atomic) | Clean history, easy revert |
| Related fixes (if any) | Similar patterns fixed |

---

## Phase 6: Prevention

**Close the loop.** Don't just fix the bug - prevent this class of bug forever.

### Contribute Back

| Action | Purpose |
|--------|---------|
| **Add to test suite** | Regression test now exists |
| **Improve observability** | Add logging/metrics at the failure point |
| **Update documentation** | Share learnings with team |
| **Create lint rule/check** | Catch at compile time if possible |

### Pattern Recognition

Ask: "What made this bug possible?"

| Root Issue | Prevention |
|------------|------------|
| Missing validation | Add schema/type checking |
| Race condition | Document threading invariants |
| Config mismatch | Add environment checks |
| Null handling | Enable strict null checks |
| Edge case missed | Add property-based tests |

### Team Share

Document for your team:
- What symptoms to watch for
- How to diagnose quickly
- The fix pattern for similar bugs

This turns painful debugging into organizational knowledge.

---

## Cognitive Bias Warnings

Your brain will sabotage you. Watch for:

| Bias | Symptom | Counter |
|------|---------|---------|
| **Confirmation bias** | Only looking for evidence supporting your guess | Actively try to DISPROVE hypothesis |
| **Anchoring** | Fixating on first theory | Write down alternatives before investigating |
| **Recency bias** | "I just changed X, must be X" | Check git blame - what ELSE changed? |
| **Availability bias** | "Last bug was Y, this must be Y too" | Each bug is independent - follow evidence |
| **Sunk cost** | "I've spent hours on this theory..." | Wrong theory + more time = more waste |

### Golden Rules

1. **Change one thing at a time** - Never change two variables in one experiment
2. **Record everything** - You'll forget what you tried
3. **Rubber duck** - Explain the bug out loud (to user) to force logical processing
4. **Take breaks** - Fresh eyes catch what tired eyes miss
5. **Question assumptions** - "Are you SURE that value is what you think?"

---

## Special Scenarios

### Cross-Cutting Bugs (Multi-Layer/System)

When bug spans multiple systems:

1. **Map the data flow** - Draw the path from user action to failure
2. **Identify ownership boundaries** - Which team/service owns each step?
3. **Add correlation ID tracing** - Track single request across all services
4. **Fence each boundary** - Log input/output at every system interface
5. **Narrow to single system** - Then apply standard debugging

**Correlation ID Pattern** (Critical for distributed systems):

```python
# Generate at entry point
correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

# Pass through ALL downstream calls
response = service_b.call(
    data=payload,
    headers={"X-Correlation-ID": correlation_id}
)

# Include in ALL logs
logger.info(f"[{correlation_id}] Processing request", extra={
    "correlation_id": correlation_id,
    "step": "service_a_entry",
    "timestamp": time.time(),
    "data_snapshot": sanitize(payload)
})
```

**Then search logs by correlation ID to reconstruct the full request path.**

```
User → Frontend → API Gateway → Service A → Service B → Database
         │            │             │            │          │
    [corr-id]    [corr-id]     [corr-id]    [corr-id]  [corr-id]
         │            │             │            │          │
    timestamp    timestamp     timestamp    timestamp  timestamp
```

**Find the corruption point**: Compare data at each fence. Where does correct become incorrect?

### Elusive/Intermittent Bugs

For bugs that resist reproduction:

1. **Statistical approach** - Run 100x, measure failure rate
2. **Stress test** - Increase load to force race conditions
3. **Add comprehensive logging** - Capture state before every failure
4. **Look for patterns** - Time of day? Load level? Specific users?
5. **Instrument don't debug** - Interactive debuggers change timing

### Heisenbugs (STOP - Special Handling Required)

**If the bug disappears when you add logging or observation, STOP.**

This is a **Heisenbug** - and it's actually your biggest clue. The fact that observation changes behavior tells you:

1. **This is almost certainly a timing issue** - Logging adds milliseconds of delay
2. **Look for race conditions** - Two things competing, logging changes who wins
3. **Look for unwaited async** - Something not being awaited, logging gives it time to complete
4. **Look for timeouts** - Something timing out, logging delays past the threshold

**Heisenbug Strategy**:

| Don't | Do Instead |
|-------|------------|
| Add more logging | Add **assertions** that crash on invalid state |
| Use interactive debugger | Use **sampling** (log 1 in 100 requests) |
| Try to catch it live | **Capture state** at the moment of failure |
| Guess at the race | **Map all async operations** in the code path |

```python
# Instead of logging (changes timing):
# print(f"Value is {value}")

# Use assertion (minimal timing impact, captures the moment):
assert value is not None, f"BUG: value was None, context: {context}"
```

**Circular buffer logging**: Log to a ring buffer that only dumps on error. This captures history with minimal runtime impact:

```python
from collections import deque

# Global ring buffer - holds last 100 events, minimal overhead
_debug_buffer = deque(maxlen=100)

def trace(msg):
    """Log to buffer only - near-zero performance impact."""
    _debug_buffer.append(f"{time.time()}: {msg}")

def dump_trace():
    """Call this ONLY when error occurs."""
    print("\n".join(_debug_buffer))

# Usage: trace() everywhere, dump_trace() in exception handler
try:
    process()
except Exception:
    dump_trace()  # Now you see what happened
    raise
```

**Key question**: "WHERE did I add logging that made it stop?" That location is near the race condition.

### Investigation Reset Protocol ("I've Tried Everything")

When user (or you) has been debugging for hours/days without progress, the investigation has gone wrong. **STOP trying fixes and reset.**

**Signs you need a reset**:
- "I've tried everything" (but can't list what systematically)
- Multiple random changes made without tracking
- Can't explain what hypothesis each attempt was testing
- Exhaustion and frustration
- Wanting to "just add error handling and move on"

**The Reset Process**:

1. **STOP all fixing attempts** - No more code changes until process is followed

2. **Audit what was actually tried**:
   - List each thing attempted
   - For each: What hypothesis was it testing?
   - If no hypothesis → it was guessing, not debugging

3. **Check for skipped phases**:
   | Phase | Question | If Skipped |
   |-------|----------|------------|
   | Reproduce | Can you trigger it reliably RIGHT NOW? | Start here |
   | Isolate | Do you know which component fails? | Binary search/fence |
   | Hypothesize | Can you state a falsifiable theory? | Form one before any more changes |

4. **Identify the ACTUAL error**:
   - What is the EXACT error message? (copy-paste, not paraphrased)
   - WHERE in the codebase does that message originate?
   - What code path leads there?

5. **Restart from Phase 1** with discipline:
   - No changes without hypothesis
   - One variable at a time
   - Record everything

**Common "tried everything" patterns**:

| What They Did | What They Skipped |
|---------------|-------------------|
| "Checked logs" | Didn't correlate with specific hypothesis |
| "Added retry logic" | Didn't know what was failing or why |
| "Increased timeouts" | Didn't measure actual latencies |
| "Rolled back" | Didn't isolate which change broke it |
| "Googled it" | Didn't understand their specific case |

**The hard truth**: If you've been at it for days without progress, you haven't been debugging - you've been guessing. That's not a criticism; it's diagnostic. The fix is process, not more effort.

### Legacy/Unfamiliar Codebase

When debugging code you don't understand:

1. **Read tests first** - They document expected behavior
2. **Trace from entry point** - Follow the request path
3. **Git blame strategically** - Who wrote this? Why?
4. **Don't assume intent** - Code may not do what it looks like
5. **Build confidence before fixing** - See [codebase-confidence-building.md](codebase-confidence-building.md)

```bash
# Who touched this recently?
git log --oneline -20 -- path/to/file.py

# What was the intent of this code?
git log -p --follow -- path/to/file.py | head -200

# When did this behavior start?
git log -S "suspicious_pattern" --oneline
```

---

## Domain-Specific Handoffs

After applying this methodology, hand off to specialists for domain-specific issues:

### AI/ML Debugging

| Domain | Specialist Skill | When to Hand Off |
|--------|------------------|------------------|
| PyTorch OOM | `yzmir-pytorch-engineering:debug-oom` | Memory errors in training |
| PyTorch NaN | `yzmir-pytorch-engineering:debug-nan` | NaN/Inf in gradients |
| RL Training | `yzmir-deep-rl:rl-debugging` | Policy not learning |
| Simulation | `bravos-simulation-tactics:debugging-simulation-chaos` | Determinism, drift |
| ML Production | `yzmir-ml-production:production-debugging-techniques` | Inference issues |
| LLM Generation | `yzmir-llm-specialist:debug-generation` | Output quality issues |

### Quality Engineering Integration

| Situation | Specialist Skill | When to Use |
|-----------|------------------|-------------|
| Flaky tests | `ordis-quality-engineering:flaky-test-prevention` | After identifying test as intermittent |
| Test isolation | `ordis-quality-engineering:test-isolation-fundamentals` | When tests pollute each other |
| Production debugging | `ordis-quality-engineering:observability-and-monitoring` | Setting up tracing/metrics |
| Finding race conditions | `ordis-quality-engineering:chaos-engineering-principles` | Stress testing to force failures |

### Systems Thinking for Complex Bugs

For bugs involving feedback loops, cascading failures, or systemic issues:

| Tool | Specialist Skill | When to Use |
|------|------------------|-------------|
| Causal mapping | `yzmir-systems-thinking:causal-loop-diagramming` | Bug involves circular dependencies |
| Finding leverage | `yzmir-systems-thinking:leverage-points-mastery` | Many possible intervention points |
| Pattern recognition | `yzmir-systems-thinking:systems-archetypes-reference` | Bug matches known system failure patterns |

**Use this skill for methodology, specialists for domain knowledge.**

---

## Quick Reference

### Debugging Checklist

- [ ] **Reproduce**: Can I trigger this reliably?
- [ ] **Minimize**: Do I have the smallest failing case?
- [ ] **Isolate**: Have I narrowed the location?
- [ ] **Hypothesize**: Do I have a falsifiable theory?
- [ ] **Experiment**: Have I designed a test that could disprove it?
- [ ] **Root cause**: Have I gone deep enough (5 Whys)?
- [ ] **Test**: Did I write a failing test BEFORE fixing?
- [ ] **Verify**: Does the fix pass? No regressions?

### When to Ask the User

**Ask immediately** for:
- Production logs/metrics you can't access
- User actions that triggered the bug
- Environment details (versions, config)
- "Has anything changed recently?"

**Try yourself first** for:
- Reading codebase and tests
- Git history and blame
- Running tests locally
- Searching for patterns

### Red Flags - Stop and Reconsider

| Thought | Reality |
|---------|---------|
| "Let me just try this..." | STOP. Form hypothesis first. |
| "It's probably X" | STOP. Test it, don't assume. |
| "I'll add a null check" | STOP. That's a symptom fix. |
| "This is taking forever" | STOP. Wrong approach. Step back. |
| "It works now, ship it" | STOP. Do you know WHY it works? |
| "I changed too many things to know" | STOP. Revert, change ONE thing. |
