# Testing Component Quality

**Purpose:** Behavioral testing of all plugin components to identify issues requiring fixes.

## Core Principle

**Behavioral testing, NOT syntactic validation.**

Components are process documentation. Test if they guide Claude correctly, not if they parse correctly.

---

## What We're Testing

| We Test | We Don't Test |
|---------|---------------|
| Does the component guide correctly? | Does it parse? (syntactic) |
| Does it hold under pressure? | Does it look complete? (coverage) |
| Does it handle edge cases? | How does it compare to others? (benchmarking) |

---

## Gauntlet Categories

### A. Pressure Scenarios (Catch Shortcuts)

**Test if component holds when the model wants to skip it.**

**Pressure types:**
- **Time pressure:** "This is urgent, just do it quickly"
- **Simplicity temptation:** "This is too simple for the full process"
- **Overkill perception:** "That's way more than necessary for this"
- **Sunk cost:** "I already wrote most of it, just finish"

**Design approach:**
- Combine 2-3 pressures for maximum effect
- Watch for rationalized shortcuts
- Document exact phrasing of rationalizations

### C. Adversarial Edge Cases (Test Robustness)

**Test if component provides guidance for corner cases.**

**Edge case types:**
- **Principle conflicts:** When guidelines contradict each other
- **Naive application failures:** When literal application doesn't work
- **Missing information:** When component doesn't cover needed knowledge
- **Tool limitations:** When environment doesn't support the approach

### B. Real-World Complexity (Validate Utility)

**Test if component works in realistic scenarios.**

**Complexity types:**
- **Messy requirements:** Unclear specs, conflicting needs
- **Multiple valid approaches:** Several solutions, trade-offs unclear
- **Integration constraints:** Existing patterns, team conventions
- **Incomplete information:** Missing context, unknown dependencies

---

## Testing by Component Type

### Testing Skills

**Priority scenarios:**

For **discipline-enforcing skills** (TDD, verification):
- Focus on pressure (A) - test rationalization resistance

For **technique skills** (patterns, algorithms):
- Focus on edge cases (C) and real-world (B)

For **reference skills** (API docs, guides):
- Focus on real-world (B) - can users find and apply info?

**Test execution:**
1. Design challenging scenario from gauntlet categories
2. Provide scenario to a test agent WITH the skill
3. Observe: Does it follow? Where does it rationalize?
4. Document failure modes with exact quotes

### Testing Commands

**Key questions:**
- Does the command provide clear entry point?
- Are tool restrictions appropriate for the task?
- Does argument-hint guide correct usage?
- Does the command overlap with a skill inappropriately?

**Test execution:**
1. Invoke command with realistic arguments
2. Check if guidance is actionable
3. Test edge cases (missing args, wrong format)
4. Verify tool restrictions don't block legitimate use

### Testing Agents

**Key questions:**
- Does agent stay within scope boundaries?
- Are activation examples accurate?
- Does model selection match task complexity?
- Does agent hand off correctly when out of scope?

**Test execution:**
1. Present task matching agent's domain
2. Observe: Does it activate appropriately?
3. Present task OUTSIDE domain - does it correctly decline?
4. Test handoff to other agents

### Testing Hooks

**Key questions:**
- Does hook fire on correct events?
- Does matcher pattern catch intended cases?
- Does script execute without errors?
- Are there unintended side effects?

**Test execution:**
1. Trigger the event the hook should respond to
2. Verify hook fires and script executes
3. Trigger similar but non-matching events
4. Verify hook doesn't fire incorrectly

---

## Testing Process

### Per-Component Workflow

1. **Select scenario** from gauntlet (prioritize A → C → B)
2. **Execute test** - run component with challenging input
3. **Observe behavior** - compliance, rationalizations, failures
4. **Assess result:**
   - **Pass** - Followed correctly, handled edge cases
   - **Fix needed** - Rationalized, got stuck, failed edge case
5. **Document issues** if fix needed

### Documenting Issues

For each issue:

```
**Issue:** [Description]
**Category:** [Pressure/Edge case/Real-world gap/Missing anti-pattern]
**Priority:** [Critical/Major/Minor]
**Evidence:** "[Exact quote or behavior observed]"
**Fix needed:** [Specific action]
```

### Batch Testing

**Priority order:**
1. Router skills (affects all discovery)
2. Foundational components
3. Core technique components
4. Advanced components

**Efficiency:**
- Previously tested components: spot-check only
- New/changed components: full gauntlet
- Minor edits: targeted testing of changed sections

---

## Output Format

```markdown
# Quality Testing Results: [plugin-name]

## Summary

- Components tested: [count]
- Passed: [count]
- Fix needed: [count]
  - Critical: [count]
  - Major: [count]
  - Minor: [count]

## Results by Component Type

### Skills

| Skill | Result | Issues |
|-------|--------|--------|
| [name] | Pass/Fix | [summary] |

### Commands

| Command | Result | Issues |
|---------|--------|--------|
| /[name] | Pass/Fix | [summary] |

### Agents

| Agent | Result | Issues |
|-------|--------|--------|
| [name] | Pass/Fix | [summary] |

### Hooks

| Hook | Result | Issues |
|------|--------|--------|
| [event:matcher] | Pass/Fix | [summary] |

## Detailed Issues

### [Component Name]

**Result:** Fix needed
**Priority:** [Critical/Major/Minor]
**Test scenario:** [Brief description]

**Issue 1:**
- Category: [Pressure/Edge case/etc.]
- Evidence: "[Exact behavior/quote]"
- Fix: [Specific action]

**Issue 2:**
[Same format]
```

---

## Red Flags - Rationalizations During Testing

**When YOU are testing, watch for these thoughts:**

| Thought | Reality |
|---------|---------|
| "Component looks good, skip testing" | Looking ≠ testing. Run gauntlet. |
| "I'll just check the frontmatter" | Syntactic ≠ behavioral. Test with scenarios. |
| "Small component, testing overkill" | Small components fail in edge cases too. |
| "I'm confident this works" | Confidence ≠ validation. Test behavior. |
| "No time for full testing" | Untested = broken in production. Make time. |
| "Testing is for the skill author" | You ARE testing. Don't delegate to hypothetical future. |

**If you catch yourself thinking these → STOP. Run the gauntlet.**

---

## Philosophy

**Gauntlet identifies issues. Targeted fixes address them.**

If component passes gauntlet → No changes needed.
If component fails → Document specific issues for Stage 4.

The model is both author and judge of component fitness. Trust the testing process, not intuition.

---

## Proceeding

After testing all components:
1. Compile test report
2. Proceed to Discussion (Stage 4)
3. Present findings for user approval
4. Test results inform implementation fixes
